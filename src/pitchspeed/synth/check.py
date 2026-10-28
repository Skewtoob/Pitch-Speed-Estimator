"""check_clip(): decode a synthetic clip with PyAV and verify it against its GT.

Checks
* decoded frame count == GT n_out_frames;
* container metadata (codec, declared field order, SAR) -- reported, not
  judged (metadata is unreliable on purpose, see synth.md pitfall 8);
* if the clip was made with video.debug_stamp: the per-field index barcode
  of every decoded frame against the frame map (field order, frame
  alignment, drops/duplicates), i.e. that field k of the file was exposed
  at t_k = k * 1001/60000 s and sits in rows of the GT parity;
* optionally (`rerender=N`): re-render N frames from the GT config and
  report PSNR decoded-vs-render and the best-matching frame offset.

Never decodes with OpenCV (wrong frames for interlaced video in this build).
"""
from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np

from pitchspeed.synth.gt import load_gt, video_path_from_gt
from pitchspeed.synth.stamp import read_stamp


def _expected_stamps(entry: dict, video: dict) -> Optional[list]:
    """[(parity or -1 for full frame, k)] readable in an out frame, or None if unreadable (blend)."""
    if video.get("postprocess") == "blend":
        return None
    if video.get("interlaced_file"):
        return list(zip(entry["parity"], entry["field_index"]))
    return [(-1, entry["field_index"][0])]


_FIELD_ORDER = {0: "unknown", 1: "progressive", 2: "tt", 3: "bb", 4: "tb", 5: "bt"}


def _field_order_name(x) -> str:
    try:
        return _FIELD_ORDER.get(int(x), str(x))
    except (TypeError, ValueError):
        return str(x or "")


def _decode_info(path: str):
    import av
    c = av.open(path)
    try:
        s = c.streams.video[0]
        cc = s.codec_context
        meta = dict(codec=cc.name, width=cc.width, height=cc.height,
                    sar=str(s.sample_aspect_ratio or cc.sample_aspect_ratio or ""),
                    field_order=_field_order_name(getattr(cc, "field_order", None)),
                    container=c.format.name, average_rate=str(s.average_rate))
    finally:
        c.close()
    return meta


def iter_gray_frames(path: str):
    """Yield (index, gray uint8 frame, PyAV frame flags) in presentation order (PyAV)."""
    import av
    c = av.open(path)
    try:
        for i, f in enumerate(c.decode(video=0)):
            flags = dict(interlaced=bool(getattr(f, "interlaced_frame", False)),
                         tff=bool(getattr(f, "top_field_first", False)))
            yield i, f.to_ndarray(format="gray"), flags
    finally:
        c.close()


def iter_bgr_frames(path: str):
    import av
    c = av.open(path)
    try:
        for i, f in enumerate(c.decode(video=0)):
            yield i, f.to_ndarray(format="bgr24")
    finally:
        c.close()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return math.inf if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def check_clip(path: str, rerender: int = 0, max_report: int = 10) -> dict:
    """Verify a clip (video path or its .gt.json path) against its GT; returns a report dict.

    report['ok'] is True when the frame count matches and (if present) every
    readable barcode matches the frame map.
    """
    p = str(path)
    gt = load_gt(p)
    video_path = video_path_from_gt(p, gt) if p.endswith(".gt.json") else p
    if not os.path.exists(video_path):
        video_path = os.path.join(os.path.dirname(os.path.abspath(p)), gt["video"]["file"])
    V = gt["video"]
    fmap = gt["frame_map"]
    stamped = bool(gt["config"]["video"].get("debug_stamp"))
    rep = dict(video=video_path, metadata=_decode_info(video_path), n_expected=len(fmap))
    n = 0
    mismatches, n_checked, unreadable = [], 0, 0
    first_flags = None
    for i, gray, flags in iter_gray_frames(video_path):
        n = i + 1
        if first_flags is None:
            first_flags = flags
        if not stamped or i >= len(fmap):
            continue
        exp = _expected_stamps(fmap[i], V)
        if exp is None:
            unreadable += 1
            continue
        for par, k in exp:
            got = read_stamp(gray[par::2], 1) if par >= 0 else read_stamp(gray, 2)
            n_checked += 1
            if got != k:
                mismatches.append(dict(out_frame=i, parity=par, expected=k, got=got))
    rep.update(n_decoded=n, frame_count_ok=n == len(fmap), first_frame_flags=first_flags)
    if stamped:
        rep.update(stamps_checked=n_checked, stamp_mismatches=len(mismatches), stamp_unreadable_frames=unreadable,
                   mismatch_examples=mismatches[:max_report])
    else:
        rep["stamps_checked"] = 0
        rep["note"] = "clip has no debug stamp; field timing not verified (generate with video.debug_stamp=true)"
    if rerender > 0:
        rep["rerender"] = _rerender_check(gt, video_path, rerender)
    rep["ok"] = bool(rep["frame_count_ok"] and (not stamped or not mismatches))
    return rep


def _rerender_check(gt: dict, video_path: str, n_check: int) -> dict:
    from pitchspeed.synth.config import resolve_config
    from pitchspeed.synth.render import Renderer
    cfg = resolve_config(gt["config"])
    fmap = gt["frame_map"]
    ref = gt["calibration"]["ref_exposure_times"]
    r = Renderer(cfg, ref)
    targets = sorted(set(np.linspace(2, len(fmap) - 3, n_check).astype(int).tolist())) if len(fmap) > 5 else []
    want = {}
    for j in targets:
        want[j] = fmap[j]["src_frame"]
    lo = min(targets) - 2 if targets else 0
    hi = max(targets) + 2 if targets else -1
    dec = {i: f for i, f in iter_bgr_frames(video_path) if lo <= i <= hi} if targets else {}
    rows = []
    cache = {}

    def rendered(nsrc: int) -> np.ndarray:
        if nsrc not in cache:
            cache[nsrc] = r.render_frame(nsrc, collect=False).image
        return cache[nsrc]

    for j in targets:
        if j not in dec:
            continue
        ns = want[j]
        scores = {}
        for off in (-1, 0, 1):
            if 0 <= ns + off < gt["video"]["n_src_frames"]:
                scores[off] = psnr(dec[j], rendered(ns + off))
        best = max(scores, key=scores.get)
        rows.append(dict(out_frame=j, src_frame=ns, psnr_db=scores.get(0), best_offset=best))
    ps = [x["psnr_db"] for x in rows if x["psnr_db"] is not None and math.isfinite(x["psnr_db"])]
    note = None
    if gt["video"].get("postprocess") in ("yadif", "yadif_bob"):
        note = "yadif is applied by ffmpeg, the in-process render is the interlaced source: low PSNR expected"
    identical = bool(rows) and all(x["psnr_db"] == math.inf for x in rows)
    return dict(frames=rows, mean_psnr_db=float(np.mean(ps)) if ps else None, identical=identical,
                aligned=all(x["best_offset"] == 0 for x in rows), note=note)


def format_check(rep: dict) -> str:
    lines = [f"clip: {rep['video']}",
             f"  metadata: {rep['metadata']}",
             f"  frames: decoded {rep['n_decoded']}, expected {rep['n_expected']} -> "
             f"{'OK' if rep['frame_count_ok'] else 'MISMATCH'}"]
    if rep.get("stamps_checked"):
        lines.append(f"  field barcodes: {rep['stamps_checked']} checked, {rep['stamp_mismatches']} mismatches"
                     + (f", {rep['stamp_unreadable_frames']} frames unreadable" if rep.get("stamp_unreadable_frames") else ""))
        for m in rep.get("mismatch_examples", []):
            lines.append(f"    out frame {m['out_frame']} parity {m['parity']}: expected k={m['expected']} got {m['got']}")
    elif rep.get("note"):
        lines.append("  " + rep["note"])
    rr = rep.get("rerender")
    if rr:
        mp = rr["mean_psnr_db"]
        ps = "identical (lossless)" if rr.get("identical") else ("n/a" if mp is None else f"{mp:.1f} dB")
        lines.append(f"  re-render: {len(rr['frames'])} frames, mean PSNR {ps}, aligned={rr['aligned']}"
                     + (f" ({rr['note']})" if rr.get("note") else ""))
    lines.append("  RESULT: " + ("OK" if rep["ok"] else "PROBLEMS FOUND"))
    return "\n".join(lines)
