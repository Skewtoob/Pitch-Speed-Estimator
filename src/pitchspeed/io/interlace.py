"""Interlace / field-order detection and the field-sample view of a video.

`detect_scan(reader)` decides from *content* (metadata is only a hint)
whether the video is interlaced and in which field order, and returns a
`ScanInfo`.  `SampleSource(reader, scan)` then presents the video as a time
ordered sequence of samples: fields (59.94/s for NTSC) for interlaced video,
frames otherwise.

Field conventions (DESIGN.md section 2, types.py):
  * sample index s = 2*i + k for frame i, k = temporal order inside the
    frame; parity p = k for TFF, 1 - k for BFF.
  * the field image is ``frame[p::2]`` at its native rows (H/2), never
    resized; field row r is frame row 2*r + p.
  * t_s = t_i + k * frame_dt / 2 (constant exposure offsets cancel).

DETECTION (io_ux.md section 2).  Evidence is gathered in windows spread over
the file (at most `max_frames` decoded frames in total):

1. *Comb test* (whole-image, works for any sizeable motion).  T = even rows,
   B = odd rows, T interpolated onto B's rows, 3x3 blur.  On a motion mask
   (same-parity change > 6 sigma) the mean absolute differences
   a = D(T_k, B_k), b = D(B_k, T_k+1), c = D(T_k, B_k+1) are compared.  All
   three are cross-parity so vertical-detail bias cancels.  Progressive:
   a < 0.4 min(b, c) (measured 0.02-0.12); TFF: b < c (measured b/c 0.35-0.45); BFF: b > c (2.2-2.8);
   min(b, c) << a: progressive frames whose fields are split across woven
   frames ("field-shifted"), which are re-paired.
2. *Blob-midpoint test* (small fast objects, static camera).  The comb test
   saturates once an object moves more than its own size per field (a lone
   ball), so windows where it is not decisive also run this test: compact
   foreground blobs are linked between same-parity fields of frames k and
   k+1 (two field periods apart); their midpoint must coincide with the
   opposite-parity blob of frame k (TFF) or frame k+1 (BFF), or lie halfway
   between both (progressive).  Score q = d_tff / (d_tff + d_bff): TFF ~0,
   BFF ~1, progressive ~0.5.
3. Optional cross-check with ffmpeg's ``idet`` filter (bundled ffmpeg,
   bounded in time).  Disagreement lowers the confidence one level; idet
   alone never overrides the content tests (it is wrong on the
   "static camera, lone ball" case), it only breaks a complete tie.

Confidence: 'high' = a strong, consistent verdict; 'medium' = decisive but
weaker, or tests disagree mildly, or idet disagrees with a strong verdict;
'low' = conflicting or no evidence (then `source` is 'default' and the
verdict comes from priors: frame rate > 40 -> progressive; SD 25/29.97 fps
-> interlaced, field order from stream metadata, else DV -> BFF, other -> TFF).
The per-pitch check (`residual_alternation`, `field_order_check`) is the
final word on field order for each pitch.

DIGITIZATION VARIANTS
  * 50/59.94p made by repeating each frame ("fake 60p"): detected by the
    alternation of frame-to-frame change; ScanInfo is progressive at half
    the rate with ``evidence['frame_step'] = 2`` and ``evidence['frame_phase']``
    (the first frame of each repeated pair).  SampleSource then uses one
    frame per pair.  A phase change mid-file (capture glitch) is not handled;
    it is reported in the evidence.
  * Genuine 59.94p bob output, blend-deinterlaced and single-rate
    deinterlaced video: progressive (blend leaves the ball as two
    half-contrast ghosts one field apart; the detector must expect them).
  * Field-shifted progressive: ``evidence['repair']`` is 'bottom_from_next'
    or 'top_from_next'; SampleSource weaves the matching fields of frames
    i and i+1 into sample i.
  * Capture duplicates (a frame repeating its predecessor) are found from
    content in the detection windows and while iterating samples, and are
    recorded in ``reader.info.duplicate_frames``.
  * Vertically rescaled interlaced video and telecine are not detected
    (they usually show as 'uncertain'/'progressive').

ScanInfo.evidence keys read back by SampleSource: 'frame_step',
'frame_phase', 'repair' (all optional).  Everything else is diagnostics.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
from typing import Iterator, Optional, Sequence

import cv2
import numpy as np

from pitchspeed.types import Sample, ScanInfo

log = logging.getLogger(__name__)

OVERRIDES = ("tff", "bff", "progressive")

# --- comb test ---------------------------------------------------------------
COMB_BLUR = 3
COMB_KSIG = 6.0
COMB_MIN_PX = 24
# --- blob-midpoint test ------------------------------------------------------
BLOB_KSIG = 4.0
BLOB_MIN_AREA = 3          # field pixels (scaled with the frame width)
BLOB_MAX_AREA = 3000
BLOB_MIN_DISP = 4.0        # px between same-parity fields (2 field periods)
BLOB_MAX_DISP = 150.0
# --- duplicates / frame doubling ---------------------------------------------
DUP_BLOCK = 8              # block size for the duplicate test
DUP_MAX_LEVEL = 2.0        # max block-mean |dY| (gray levels) of a repeated frame
DOUBLING_RATIO = 0.3       # change(within pair) / change(across pairs) below this -> fake 60p
# --- windows -----------------------------------------------------------------
MAX_WINDOWS = 15
MIN_WINDOW = 30
MAX_WINDOW = 300

_FIELD_ORDER_META = {1: "progressive", 2: "tff", 3: "bff", 4: "bff", 5: "tff"}
_DECIDED = ("tff", "bff", "progressive", "shifted")


# ======================================================================= utils
def split_fields(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(top, bottom) fields of a frame with the same row count (odd last row dropped)."""
    h = gray.shape[0] - gray.shape[0] % 2
    return np.ascontiguousarray(gray[0:h:2]), np.ascontiguousarray(gray[1:h:2])


def _analysis_frame(gray: np.ndarray) -> np.ndarray:
    """Even row count; columns halved for wide frames (rows untouched: fields!)."""
    g = gray[: gray.shape[0] - gray.shape[0] % 2]
    if g.shape[1] > 1000:
        g = cv2.resize(g, (g.shape[1] // 2, g.shape[0]), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(g)


def change_score(a: np.ndarray, b: np.ndarray, block: int = DUP_BLOCK) -> float:
    """Largest block-mean |a - b| (gray levels): ~0 for a repeated frame.

    Block means make the test insensitive to noise yet sensitive to a small
    moving object (a ball changes its block by tens of levels), unlike a
    whole-frame mean.
    """
    d = cv2.absdiff(a, b)
    h, w = d.shape[:2]
    small = cv2.resize(d, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    return float(small.max())


def is_duplicate(a: np.ndarray, b: np.ndarray) -> bool:
    """True if frame b repeats frame a (capture duplicate / frame doubling)."""
    return change_score(a, b) < DUP_MAX_LEVEL


# ================================================================== comb test
def _comb_fields(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = gray.astype(np.float32)
    t, b = f[0::2], f[1::2]
    n = min(len(t), len(b))
    t, b = cv2.blur(t[:n], (COMB_BLUR, COMB_BLUR)), cv2.blur(b[:n], (COMB_BLUR, COMB_BLUR))
    ti = 0.5 * (t[:-1] + t[1:])          # top field interpolated onto bottom rows 2r+1
    return ti, b[:-1]


def comb_measurements(frames: Sequence[tuple[int, np.ndarray]]) -> np.ndarray:
    """Rows (a, b, c, n_px) for each consecutive frame pair with enough motion."""
    rows = []
    prev_i, prev = None, None
    for i, g in frames:
        cur = _comb_fields(g)
        if prev is not None and i == prev_i + 1:
            (t0, b0), (t1, b1) = prev, cur
            d_b = np.abs(b1 - b0)
            d_t = np.abs(t1 - t0)
            sub = np.concatenate([d_b[::4, ::4].ravel(), d_t[::4, ::4].ravel()])
            sigma = 1.4826 * float(np.median(sub)) / np.sqrt(2.0) + 0.1
            mask = (d_b > COMB_KSIG * sigma) | (d_t > COMB_KSIG * sigma)
            n = int(mask.sum())
            if n >= COMB_MIN_PX:
                rows.append((float(np.abs(t0 - b0)[mask].mean()), float(np.abs(b0 - t1)[mask].mean()),
                             float(np.abs(t0 - b1)[mask].mean()), n))
        prev_i, prev = i, cur
    return np.array(rows, dtype=np.float64).reshape(-1, 4)


def classify_comb(rows: np.ndarray) -> dict:
    """Verdict of the comb test: 'progressive' | 'tff' | 'bff' | 'shifted' | 'uncertain' | 'none'."""
    n = len(rows)
    if n < 8:
        return {"verdict": "none", "strong": False, "n": n}
    a, b, c = rows[:, 0] + 1e-6, rows[:, 1] + 1e-6, rows[:, 2] + 1e-6
    lo = np.minimum(b, c)
    pa, bc, sh = float(np.median(a / lo)), float(np.median(b / c)), float(np.median(lo / a))
    pv, tv = float(np.mean(a < 0.45 * lo)), float(np.mean(b < 0.8 * c))
    bv, sv = float(np.mean(c < 0.8 * b)), float(np.mean(lo < 0.6 * a))
    out = {"n": n, "a_over_minbc": round(pa, 3), "b_over_c": round(bc, 3), "minbc_over_a": round(sh, 3),
           "votes": {"progressive": round(pv, 3), "tff": round(tv, 3), "bff": round(bv, 3), "shifted": round(sv, 3)}}
    big = n >= 20
    # Real progressive content measures a/min(b,c) = 0.02-0.12 (blend, bob and
    # single-rate deinterlaced included); compression noise around a lone
    # ball in interlaced MPEG-2 gave ~0.5, hence the strict thresholds.
    if pa < 0.4 and pv >= 0.6:
        out.update(verdict="progressive", strong=big and pa < 0.25 and pv >= 0.8)
    elif sh < 0.6 and sv >= 0.6:
        out.update(verdict="shifted", strong=big and sh < 0.35 and sv >= 0.8,
                   repair="bottom_from_next" if float(np.median(c / b)) < 1.0 else "top_from_next")
    elif bc < 0.8 and tv >= 0.6:
        out.update(verdict="tff", strong=big and bc < 0.65 and tv >= 0.8)
    elif bc > 1.25 and bv >= 0.6:
        out.update(verdict="bff", strong=big and bc > 1.55 and bv >= 0.8)
    else:
        out.update(verdict="uncertain", strong=False)
    out["combing"] = bool(pa > 0.8)       # the two fields differ in time, order unclear
    return out


# ========================================================== blob-midpoint test
def _field_blobs(stack: np.ndarray, scale: float) -> list[list[tuple[float, float]]]:
    """Foreground blob centroids per field of a (n, rows, cols) same-parity stack."""
    step = max(1, len(stack) // 15)
    bg = np.median(stack[::step], axis=0).astype(np.uint8)
    min_a, max_a = BLOB_MIN_AREA * scale * scale, BLOB_MAX_AREA * scale * scale
    kernel = np.ones((2, 2), np.uint8)
    out = []
    for f in stack:
        d = cv2.blur(cv2.absdiff(f, bg).astype(np.float32), (3, 3))
        sigma = 1.4826 * float(np.median(d[::4, ::4])) + 0.1
        m = cv2.morphologyEx((d > BLOB_KSIG * sigma).astype(np.uint8), cv2.MORPH_OPEN, kernel)
        n, _, stats, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
        out.append([(float(cen[j][0]), float(cen[j][1])) for j in range(1, n)
                    if min_a <= stats[j, cv2.CC_STAT_AREA] <= max_a])
    return out


def _nearest(pts: list[tuple[float, float]], q: tuple[float, float]):
    if not pts:
        return None, np.inf, np.inf
    d = np.hypot(np.array([p[0] for p in pts]) - q[0], np.array([p[1] for p in pts]) - q[1])
    o = np.argsort(d)
    return pts[o[0]], float(d[o[0]]), float(d[o[1]]) if len(o) > 1 else np.inf


def blob_measurements(frames: Sequence[tuple[int, np.ndarray]]) -> np.ndarray:
    """Scores q = d_tff / (d_tff + d_bff) for linked moving blobs (see module doc)."""
    if len(frames) < 5:
        return np.zeros(0)
    idx = [i for i, _ in frames]
    stack = np.stack([g for _, g in frames])
    scale = stack.shape[2] / 720.0
    blobs = {}
    for p in (0, 1):
        per = _field_blobs(np.ascontiguousarray(stack[:, p::2]), scale)
        for i, bl in zip(idx, per):
            blobs[(i, p)] = [(x, 2.0 * y + p) for x, y in bl]      # frame-row coordinates
    qs = []
    min_d, max_d = BLOB_MIN_DISP * scale, BLOB_MAX_DISP * scale
    for i in idx:
        if (i + 1, 0) not in blobs:
            continue
        for p in (0, 1):
            o = 1 - p
            for x0, y0 in blobs[(i, p)]:
                p2, d1, d2 = _nearest(blobs[(i + 1, p)], (x0, y0))
                if p2 is None or not (min_d <= d1 <= max_d) or d2 < 1.5 * d1:
                    continue
                mid = (0.5 * (x0 + p2[0]), 0.5 * (y0 + p2[1]))
                _, dk, _ = _nearest(blobs[(i, o)], mid)
                _, dk1, _ = _nearest(blobs[(i + 1, o)], mid)
                lim = 1.5 * d1 + 3.0
                if min(dk, dk1) > lim:
                    continue
                dk, dk1 = min(dk, lim), min(dk1, lim)
                # top pair (p=0): TFF -> bottom of frame i matches; BFF -> bottom of i+1.
                # bottom pair (p=1): TFF -> top of frame i+1; BFF -> top of frame i.
                d_tff, d_bff = (dk, dk1) if p == 0 else (dk1, dk)
                qs.append(d_tff / (d_tff + d_bff + 1e-6))
    return np.array(qs, dtype=np.float64)


def classify_blob(q: np.ndarray) -> dict:
    n = len(q)
    if n < 5:
        return {"verdict": "none", "strong": False, "n": n}
    tv, bv = float(np.mean(q < 0.25)), float(np.mean(q > 0.75))
    pv = float(np.mean(np.abs(q - 0.5) < 0.15))
    out = {"n": n, "median_q": round(float(np.median(q)), 3),
           "votes": {"tff": round(tv, 3), "bff": round(bv, 3), "progressive": round(pv, 3)}}
    best = max((tv, "tff"), (bv, "bff"), (pv, "progressive"))
    if best[0] > 0.6:
        out.update(verdict=best[1], strong=n >= 10 and best[0] >= 0.8)
    else:
        out.update(verdict="uncertain", strong=False)
    return out


# ================================================================ combination
def _vote_strength(res: dict) -> float:
    return float(res.get("votes", {}).get(res.get("verdict"), 0.0))


def combine_tests(comb: dict, blob: dict) -> tuple[str, str, list[str]]:
    """(verdict, confidence, notes) from the two content tests."""
    vc, vb = comb.get("verdict"), blob.get("verdict")
    sc, sb = bool(comb.get("strong")), bool(blob.get("strong"))
    notes: list[str] = []
    c_ok, b_ok = vc in _DECIDED, vb in _DECIDED
    if c_ok and b_ok:
        # Field-shifted progressive looks like weak BFF to the blob test (q ~ 0.75).
        if vc == vb or (vc == "shifted" and vb == "bff" and not sb):
            return vc, "high" if (sc or sb) else "medium", notes
        notes.append(f"comb test says {vc}, blob test says {vb}")
        if sc and not sb:
            return vc, "medium", notes
        if sb and not sc:
            return vb, "medium", notes
        return (vc if _vote_strength(comb) >= _vote_strength(blob) else vb), "low", notes
    if c_ok:
        return vc, "high" if sc else "medium", notes
    if b_ok:
        return vb, "high" if (sb and blob.get("n", 0) >= 20) else "medium", notes
    return "uncertain", "low", notes


def _lower(conf: str) -> str:
    return {"high": "medium", "medium": "low"}.get(conf, "low")


# ============================================================ frame doubling
def doubling_measurements(frames: Sequence[tuple[int, np.ndarray]]) -> list[tuple[int, float]]:
    """(i, change from frame i to i+1) on coarse images, for the fake-60p test."""
    out = []
    prev_i, prev = None, None
    for i, g in frames:
        small = cv2.resize(g, (max(1, g.shape[1] // 8), max(1, g.shape[0] // 8)), interpolation=cv2.INTER_AREA)
        small = small.astype(np.float32)
        if prev is not None and i == prev_i + 1:
            out.append((prev_i, float(np.mean(np.abs(small - prev)))))
        prev_i, prev = i, small
    return out


def classify_doubling(changes: list[tuple[int, float]]) -> dict:
    """Detect repeated-frame pairs: change alternates small / large with a fixed phase."""
    if len(changes) < 10:
        return {"doubled": False, "n": len(changes)}
    idx = np.array([i for i, _ in changes])
    m = np.array([v for _, v in changes])
    moving = m > max(0.05, 0.2 * float(np.percentile(m, 90)))
    res = {"n": len(changes)}
    meds = {}
    for ph in (0, 1):
        within = m[(idx % 2 == ph)]            # transition i -> i+1 with i % 2 == ph: inside a pair
        across = m[(idx % 2 != ph) & moving]
        if len(within) < 4 or len(across) < 4:
            continue
        meds[ph] = float(np.median(within)) / (float(np.median(across)) + 1e-6)
    if not meds:
        res["doubled"] = False
        return res
    ph = min(meds, key=meds.get)
    res.update(ratio=round(meds[ph], 3), phase=int(ph), doubled=bool(meds[ph] < DOUBLING_RATIO))
    return res


# ====================================================================== idet
_IDET_RE = re.compile(r"Multi frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*Progressive:\s*(\d+)"
                      r"\s*Undetermined:\s*(\d+)")


def run_idet(path: str, start_s: float = 0.0, n_frames: int = 300, timeout_s: float = 30.0) -> Optional[dict]:
    """ffmpeg ``idet`` on n_frames from start_s (bundled ffmpeg); None if unavailable."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:                 # noqa: BLE001 - optional cross-check
        log.info("idet cross-check skipped: no ffmpeg (%s)", e)
        return None
    cmd = [exe, "-hide_banner", "-nostdin", "-nostats"]
    if start_s > 0:
        cmd += ["-ss", f"{start_s:.3f}"]
    cmd += ["-i", os.fspath(path), "-map", "0:v:0", "-an", "-sn", "-dn", "-frames:v", str(int(n_frames)),
            "-vf", "idet", "-f", "null", "-"]
    flags = 0x08000000 if os.name == "nt" else 0          # CREATE_NO_WINDOW
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout_s, creationflags=flags)
    except subprocess.TimeoutExpired:
        return {"verdict": "none", "error": f"timeout after {timeout_s:.0f} s"}
    except OSError as e:
        return {"verdict": "none", "error": str(e)}
    text = p.stderr.decode("utf-8", "replace")
    found = _IDET_RE.findall(text)
    if not found:
        return {"verdict": "none", "error": f"no idet summary (exit {p.returncode})"}
    tff, bff, prog, und = (int(x) for x in found[-1])
    out = {"tff": tff, "bff": bff, "progressive": prog, "undetermined": und, "start_s": round(start_s, 3)}
    counts = sorted([(tff, "tff"), (bff, "bff"), (prog, "progressive")], reverse=True)
    if counts[0][0] >= 10 and counts[0][0] >= 2 * counts[1][0]:
        out["verdict"] = counts[0][1]
    else:
        out["verdict"] = "uncertain"
    return out


# ================================================================ detect_scan
def _windows(n_frames: int, max_frames: int) -> list[tuple[int, int]]:
    """Frame windows [a, b) spread over the file, at most max_frames frames in total."""
    if n_frames <= 0:
        return []
    if n_frames <= max_frames:
        k = max(1, -(-n_frames // MAX_WINDOW))
        edges = np.linspace(0, n_frames, k + 1).round().astype(int)
        return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    w = int(np.clip(max_frames // MAX_WINDOWS, MIN_WINDOW, MAX_WINDOW))
    k = max(1, max_frames // w)
    starts = np.linspace(0, n_frames - w, k).round().astype(int)
    return [(int(a), int(a) + w) for a in sorted(set(starts.tolist()))]


def _metadata_hint(path: str) -> dict:
    """Field-order hints from stream metadata (never trusted on their own)."""
    out: dict = {}
    try:
        import av
        with av.open(path) as c:
            s = c.streams.video[0]
            fo = int(getattr(s.codec_context, "field_order", 0) or 0)
            out["stream_field_order"] = _FIELD_ORDER_META.get(fo, "unknown")
            for fr in c.decode(s):
                out["frame_interlaced_flag"] = bool(fr.interlaced_frame)
                break
    except Exception as e:                 # noqa: BLE001 - hint only
        out["error"] = str(e)
    return out


def _prior(info, hint: dict, comb: dict) -> tuple[str, str]:
    """Fallback verdict when the content is not decisive: (verdict, reason)."""
    rate = info.frame_rate
    if rate > 40:
        return "progressive", f"{rate:.2f} frames/s cannot be field-interlaced frames"
    meta = hint.get("stream_field_order")
    sd_interlaced_raster = info.height in (480, 486, 576, 1080) and 20 <= rate <= 31
    if comb.get("combing") or (sd_interlaced_raster and meta != "progressive"):
        if meta in ("tff", "bff"):
            return meta, "field order from stream metadata"
        if info.codec == "dvvideo":
            return "bff", "DV is bottom field first"
        return "tff", "assumed top field first (no metadata)"
    return "progressive", "no sign of interlacing"


def _build_scan(verdict: str, confidence: str, source: str, rate: float, evidence: dict) -> ScanInfo:
    if verdict in ("tff", "bff"):
        return ScanInfo("interlaced", verdict, confidence, source, 2.0 * rate, evidence)
    step = int(evidence.get("frame_step", 1) or 1)
    return ScanInfo("progressive", None, confidence, source, rate / step, evidence)


def _normalize_override(override: Optional[str]) -> Optional[str]:
    if override is None:
        return None
    o = str(override).strip().lower()
    if o in ("", "auto"):
        return None
    o = {"top": "tff", "bottom": "bff", "p": "progressive", "prog": "progressive"}.get(o, o)
    if o not in OVERRIDES:
        raise ValueError(f"field order override must be one of {OVERRIDES} (or 'auto'), got {override!r}")
    return o


def detect_scan(reader, override: Optional[str] = None, max_frames: int = 900, *,
                use_idet: bool = True, idet_timeout_s: float = 30.0,
                verify_override: bool = True) -> ScanInfo:
    """Detect progressive / interlaced and field order from content (module doc).

    override: 'tff' | 'bff' | 'progressive' (None/'auto' = detect).  With an
    override the content is still analysed (unless verify_override=False)
    and a disagreement is reported in ``evidence['override_conflict']``.
    Duplicate frames seen in the windows are recorded in the reader's info.
    """
    ov = _normalize_override(override)
    info = reader.info
    rate = float(info.frame_rate)
    if ov is not None and not verify_override:
        return _build_scan(ov, "high", "override", rate, {"override": ov})

    evidence: dict = {"max_frames": int(max_frames), "windows": []}
    all_rows, all_q, all_changes, dups = [], [], [], []
    best_window, best_motion = None, -1
    for a, b in _windows(reader.n_frames, max_frames):
        frames = [(i, _analysis_frame(g)) for i, g in reader.iter_frames(a, b, "gray")]
        for (i0, g0), (i1, g1) in zip(frames[:-1], frames[1:]):
            if i1 == i0 + 1 and is_duplicate(g0, g1):
                dups.append(i1)
        rows = comb_measurements(frames)
        comb = classify_comb(rows)
        q = blob_measurements(frames) if not comb["strong"] else np.zeros(0)
        blob = classify_blob(q)
        verdict, conf, _ = combine_tests(comb, blob)
        evidence["windows"].append({"start": a, "stop": b, "decoded": len(frames), "verdict": verdict,
                                    "confidence": conf, "comb": comb, "blob": blob})
        all_rows.append(rows)
        all_q.append(q)
        if rate > 40:
            all_changes += doubling_measurements(frames)
        if len(rows) > best_motion:
            best_window, best_motion = (a, b), len(rows)

    comb = classify_comb(np.concatenate(all_rows) if all_rows else np.zeros((0, 4)))
    blob = classify_blob(np.concatenate(all_q) if all_q else np.zeros(0))
    verdict, conf, notes = combine_tests(comb, blob)
    evidence.update(comb=comb, blob=blob)
    source = "content"

    decided = [w["verdict"] for w in evidence["windows"] if w["verdict"] in _DECIDED]
    if verdict in _DECIDED and decided:
        dissent = sum(v != verdict for v in decided)
        if dissent:
            notes.append(f"{dissent} of {len(decided)} windows disagree (mixed content, e.g. a camcorder "
                         f"slow-shutter segment)")
            if dissent >= 0.25 * len(decided):
                conf = _lower(conf)

    if rate > 40:
        dbl = classify_doubling(all_changes)
        evidence["doubling"] = dbl
        if dbl.get("doubled"):
            verdict = "progressive"
            evidence.update(frame_step=2, frame_phase=int(dbl["phase"]))
            notes.append(f"every frame is repeated (fake {rate:.2f}p): using one frame per pair, "
                         f"{rate / 2:.3f} samples/s")
    if evidence.get("frame_step") == 2:
        # The second frame of each pair is an expected copy, not a capture glitch.
        dups = [i for i in dups if (i - evidence["frame_phase"]) % 2 == 0]
    if dups:
        reader.note_duplicate_frames(dups)
    evidence["duplicates_in_windows"] = len(dups)
    if verdict == "shifted":
        evidence["repair"] = comb.get("repair")
        notes.append(f"progressive frames with fields split across woven frames: re-paired "
                     f"({evidence['repair']})")

    if use_idet and reader.n_frames >= 10:
        start = best_window[0] if best_window else 0
        n_idet = min(int(max_frames), MAX_WINDOW, reader.n_frames - start)
        idet = run_idet(reader.path, float(reader.frame_times[start]) + reader.video_start_s
                        - reader.container_start_s, n_idet, idet_timeout_s)
        evidence["idet"] = idet
        iv = (idet or {}).get("verdict")
        if iv in ("tff", "bff", "progressive"):
            mine = "progressive" if verdict == "shifted" else verdict
            if verdict in _DECIDED and iv != mine:
                notes.append(f"ffmpeg idet says {iv}")
                conf = _lower(conf)
            elif verdict not in _DECIDED:
                verdict, conf = iv, "low"
                notes.append("content tests inconclusive; using ffmpeg idet's verdict")

    if verdict not in _DECIDED:
        hint = _metadata_hint(reader.path)
        evidence["metadata"] = hint
        verdict, why = _prior(info, hint, comb)
        conf, source = "low", "default"
        notes.append(f"no decisive motion evidence; {why}")
    if verdict == "shifted":
        verdict = "progressive"
    evidence["notes"] = notes
    scan = _build_scan(verdict, conf, source, rate, evidence)
    log.info("scan: %s %s (confidence %s, %s)%s", scan.mode, scan.field_order or "", conf, source,
             "; " + "; ".join(notes) if notes else "")
    if ov is None:
        return scan
    content = {"mode": scan.mode, "field_order": scan.field_order, "confidence": scan.confidence,
               "source": scan.source}
    ov_mode = "progressive" if ov == "progressive" else "interlaced"
    conflict = scan.source == "content" and scan.confidence != "low" and (
        ov_mode != scan.mode or (ov != "progressive" and ov != scan.field_order))
    if conflict:
        log.warning("field order override %r disagrees with the content (%s %s, confidence %s)",
                    ov, scan.mode, scan.field_order or "", scan.confidence)
    ov_ev = {"override": ov, "content_verdict": content, "override_conflict": bool(conflict),
             "content_evidence": evidence}
    if ov == "progressive" and evidence.get("frame_step"):
        ov_ev.update(frame_step=evidence["frame_step"], frame_phase=evidence["frame_phase"])
    return _build_scan(ov, "high", "override", rate, ov_ev)


# =============================================================== SampleSource
class SampleSource:
    """The video as time-ordered samples (fields or frames); see module doc.

    Attributes: reader, scan, n_samples, sample_dt, sample_shape (rows, cols),
    frame_dt, interlaced, frame_step, frame_phase, repair.
    Iterating also detects repeated frames and records them in
    ``reader.info.duplicate_frames`` (set detect_duplicates=False to skip).
    """

    def __init__(self, reader, scan: ScanInfo, detect_duplicates: bool = True):
        self.reader = reader
        self.scan = scan
        self.detect_duplicates = detect_duplicates
        self._all_times: Optional[np.ndarray] = None
        info = reader.info
        n = int(reader.n_frames)
        self.frame_dt = float(reader.frame_dt)
        self._t = np.asarray(reader.frame_times, dtype=np.float64)
        self._rows = info.height - info.height % 2
        ev = scan.evidence or {}
        self.interlaced = scan.interlaced
        self.repair: Optional[str] = None
        self.frame_step, self.frame_phase = 1, 0
        if self.interlaced:
            if scan.field_order not in ("tff", "bff"):
                raise ValueError(f"interlaced ScanInfo needs field_order 'tff' or 'bff', got {scan.field_order!r}")
            self._first_parity = 0 if scan.field_order == "tff" else 1
            self.n_samples = 2 * n
            self.sample_dt = self.frame_dt / 2.0
            self.sample_shape = (self._rows // 2, info.width)
        else:
            self._first_parity = 0
            self.repair = ev.get("repair") or None
            if self.repair not in (None, "bottom_from_next", "top_from_next"):
                raise ValueError(f"unknown repair mode {self.repair!r}")
            if self.repair:
                self.n_samples = max(0, n - 1)
                self.sample_shape = (info.height, info.width)
            else:
                self.frame_step = max(1, int(ev.get("frame_step", 1) or 1))
                self.frame_phase = int(ev.get("frame_phase", 0) or 0) % self.frame_step
                self.n_samples = max(0, -(-(n - self.frame_phase) // self.frame_step))
                self.sample_shape = (info.height, info.width)
            self.sample_dt = self.frame_dt * self.frame_step

    # ------------------------------------------------------------ indexing
    def _check(self, s: int) -> int:
        s = int(s)
        if not 0 <= s < self.n_samples:
            raise IndexError(f"sample {s} out of range [0, {self.n_samples})")
        return s

    def locate(self, s: int) -> tuple[int, int, int]:
        """(frame, k, parity) of sample s."""
        s = self._check(s)
        if self.interlaced:
            k = s & 1
            return s >> 1, k, k ^ self._first_parity
        return self.frame_phase + s * self.frame_step, 0, 0

    def sample_of(self, frame: int, k: int = 0) -> int:
        """Sample index showing (frame, k).  `frame == n_frames` gives the
        exclusive end.  For progressive sources k is ignored; with
        frame_step 2 the repeated copy maps to its pair's sample."""
        frame = int(frame)
        if self.interlaced:
            if k not in (0, 1):
                raise ValueError(f"k must be 0 or 1, got {k!r}")
            return 2 * frame + int(k)
        return max(0, (frame - self.frame_phase) // self.frame_step)

    def parity(self, k: int) -> int:
        """Field parity (0 top, 1 bottom) of temporal position k."""
        return (int(k) ^ self._first_parity) if self.interlaced else 0

    def sample_time(self, s: int) -> float:
        i, k, _ = self.locate(s)
        return float(self._t[i] + k * self.frame_dt / 2.0)

    def sample_times(self, s: Optional[np.ndarray] = None) -> np.ndarray:
        """Times of samples `s` (default: all, cached), vectorized."""
        if s is None:
            if self._all_times is None:
                self._all_times = self.sample_times(np.arange(self.n_samples))
                self._all_times.setflags(write=False)
            return self._all_times
        s = np.asarray(s, dtype=np.int64)
        if np.any((s < 0) | (s >= self.n_samples)):
            raise IndexError("sample index out of range")
        if self.interlaced:
            return self._t[s >> 1] + (s & 1) * (self.frame_dt / 2.0)
        return self._t[self.frame_phase + s * self.frame_step]

    def sample_at_time(self, t: float) -> int:
        """Nearest sample to time t (seconds)."""
        if self.n_samples == 0:
            raise IndexError("no samples")
        times = self.sample_times()
        j = int(np.clip(np.searchsorted(times, t), 1, len(times) - 1)) if len(times) > 1 else 0
        if len(times) > 1 and abs(times[j - 1] - t) <= abs(times[j] - t):
            j -= 1
        return j

    # ------------------------------------------------------------- images
    def _make(self, s: int, i: int, k: int, p: int, gray, bgr, nxt=None) -> Sample:
        t = float(self._t[i] + k * self.frame_dt / 2.0)
        if self.interlaced:
            img = np.ascontiguousarray(gray[p:self._rows:2])
            col = None if bgr is None else np.ascontiguousarray(bgr[p:self._rows:2])
        elif self.repair:
            g2, b2 = nxt
            img = _weave(gray, g2, self.repair)
            col = None if bgr is None else _weave(bgr, b2, self.repair)
        else:
            img, col = gray, bgr
        return Sample(s=s, frame=i, k=k, parity=p, t=t, image=img, bgr=col)

    def get_sample(self, s: int, color: bool = False) -> Sample:
        i, k, p = self.locate(s)
        gray, bgr = self._frame(i, color)
        nxt = self._frame(i + 1, color) if self.repair else None
        return self._make(s, i, k, p, gray, bgr, nxt)

    def _frame(self, i: int, color: bool):
        if color:
            return self.reader.get_frame(i, "both")
        return self.reader.get_frame(i, "gray"), None

    def iter_samples(self, s0: int = 0, s1: Optional[int] = None, color: bool = False) -> Iterator[Sample]:
        """Samples s0 <= s < s1 in time order (sequential decode).

        A frame that fails to decode is logged by the reader and its
        samples are absent (the gap in `s` is visible); nothing is
        substituted.
        """
        s0 = max(0, int(s0))
        s1 = self.n_samples if s1 is None else min(self.n_samples, int(s1))
        if s0 >= s1:
            return
        f0 = self.locate(s0)[0]
        f1 = self.locate(s1 - 1)[0] + 1 + (1 if self.repair else 0)
        fmt = "both" if color else "gray"
        prev_i, prev_g, prev_b = None, None, None
        dups: list[int] = []
        try:
            for i, img in self.reader.iter_frames(f0, f1, fmt):
                gray, bgr = img if color else (img, None)
                consecutive = prev_i is not None and i == prev_i + 1
                expected_copy = self.frame_step == 2 and (i - self.frame_phase) % 2 == 1
                if (self.detect_duplicates and consecutive and not expected_copy
                        and is_duplicate(prev_g, gray)):
                    dups.append(i)
                if self.interlaced:
                    for k in (0, 1):
                        s = 2 * i + k
                        if s0 <= s < s1:
                            yield self._make(s, i, k, k ^ self._first_parity, gray, bgr)
                elif self.repair:
                    if consecutive and s0 <= prev_i < s1:
                        yield self._make(prev_i, prev_i, 0, 0, prev_g, prev_b, (gray, bgr))
                elif (i - self.frame_phase) % self.frame_step == 0:
                    s = (i - self.frame_phase) // self.frame_step
                    if s0 <= s < s1:
                        yield self._make(s, i, 0, 0, gray, bgr)
                prev_i, prev_g, prev_b = i, gray, bgr
        finally:
            if dups:
                self.reader.note_duplicate_frames(dups)

    def __repr__(self) -> str:
        kind = f"fields {self.scan.field_order}" if self.interlaced else "frames"
        return (f"SampleSource({kind}, {self.n_samples} samples @ {1 / self.sample_dt:.3f}/s, "
                f"shape {self.sample_shape})")


def _weave(a: np.ndarray, b: np.ndarray, repair: str) -> np.ndarray:
    """Progressive frame from frame a and the next frame b (field-shifted video)."""
    out = a.copy()
    if repair == "bottom_from_next":
        out[1::2] = b[1::2]
    else:
        out[0::2] = b[0::2]
    return out


# ========================================================= per-pitch checks
def _poly_residuals(t: np.ndarray, v: np.ndarray, deg: int, use: np.ndarray) -> np.ndarray:
    coef = np.polyfit(t[use], v[use], deg)
    return v - np.polyval(coef, t)


def residual_alternation(t, x, y=None, degree: Optional[int] = None) -> float:
    """Lag-1 autocorrelation of the residuals of a smooth fit to a track.

    Fits a polynomial in t (degree 2 for < 10 points, else 3) to x (and y),
    rejects > 4-MAD outliers once, and returns
        r = mean(e_i . e_(i+1)) / mean(|e_i|^2)
    over consecutive samples (time step < 1.5x the median step).  Correct
    field order: r ~ 0 (slightly negative from the fit).  Wrong field order
    (positions alternate by +-v*dt/2): r -> -1.  NaN with < 6 points.

    Caveat (measured on synthetic clips): sub-pixel centroid biases that
    depend on the field parity also alternate, so with the right order and
    very clean tracks r can be ~ -0.5 at 0.6 px rms residual.  Judge the
    order by `field_order_check` (residual rms under both orders: 0.6 px vs
    18 px there) and use r as supporting evidence.
    """
    t = np.asarray(t, dtype=np.float64)
    cols = [np.asarray(x, dtype=np.float64)] + ([] if y is None else [np.asarray(y, dtype=np.float64)])
    n = len(t)
    if n < 6 or any(len(c) != n for c in cols):
        return float("nan")
    o = np.argsort(t)
    t = t[o]
    cols = [c[o] for c in cols]
    tc = (t - t.mean()) / (np.ptp(t) or 1.0)
    deg = degree if degree is not None else (2 if n < 10 else 3)
    deg = min(deg, n - 3)
    use = np.ones(n, dtype=bool)
    res = [_poly_residuals(tc, c, deg, use) for c in cols]
    mag = np.sqrt(sum(r ** 2 for r in res))
    mad = 1.4826 * float(np.median(mag)) + 1e-9
    use = mag <= 4.0 * mad + 1e-9
    if use.sum() >= deg + 3 and not use.all():
        res = [_poly_residuals(tc, c, deg, use) for c in cols]
    dt = np.diff(t)
    step = float(np.median(dt)) if len(dt) else 0.0
    pair = (dt < 1.5 * step) & use[:-1] & use[1:]
    if not pair.any():
        return float("nan")
    num = float(np.mean(sum(r[:-1][pair] * r[1:][pair] for r in res)))
    den = float(np.mean(sum(r[use] ** 2 for r in res)))
    if den <= 0:
        return 0.0
    return float(np.clip(num / den, -1.0, 1.0))


def flipped_times(t, k, frame_dt: float) -> np.ndarray:
    """Sample times under the opposite field order: t + (1 - 2k) * frame_dt / 2."""
    return np.asarray(t, dtype=np.float64) + (1 - 2 * np.asarray(k)) * (frame_dt / 2.0)


def _fit_rms(t: np.ndarray, x: np.ndarray, y: Optional[np.ndarray], deg: int) -> float:
    o = np.argsort(t)
    tc = (t[o] - t.mean()) / (np.ptp(t) or 1.0)
    cols = [x[o]] + ([] if y is None else [y[o]])
    use = np.ones(len(t), dtype=bool)
    return float(np.sqrt(np.mean(sum(_poly_residuals(tc, c, deg, use) ** 2 for c in cols))))


def field_order_check(t, x, y, k, frame_dt: float, degree: Optional[int] = None) -> dict:
    """Compare a track's smoothness under the assumed and the flipped field order.

    t, x, y, k: per-point sample time, position (frame px; y may be None)
    and temporal field index k (0/1).  Returns rms and alternation for both
    orders and 'preferred': 'given' | 'flipped' | 'unclear'.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    yy = None if y is None else np.asarray(y, dtype=np.float64)
    n = len(t)
    out = {"n": n, "preferred": "unclear"}
    if n < 6:
        return out
    deg = degree if degree is not None else (2 if n < 10 else 3)
    deg = min(deg, n - 3)
    tf = flipped_times(t, k, frame_dt)
    rg, rf = _fit_rms(t, x, yy, deg), _fit_rms(tf, x, yy, deg)
    out.update(rms_given=rg, rms_flipped=rf, alt_given=residual_alternation(t, x, yy, deg),
               alt_flipped=residual_alternation(tf, x, yy, deg))
    if rf < 0.5 * rg:
        out["preferred"] = "flipped"
    elif rg < 0.5 * rf:
        out["preferred"] = "given"
    return out
