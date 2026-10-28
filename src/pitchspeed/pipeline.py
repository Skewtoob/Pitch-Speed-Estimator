"""End-to-end analysis: video + calibration -> pitches with speeds.

    reader -> scan detection -> SampleSource -> PitchFinder (per calibration
    segment) -> speed estimates -> AnalysisResult -> results.json / .csv
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from pitchspeed import __version__
from pitchspeed.calib.model import Calibration, CalibrationFile
from pitchspeed.types import AnalysisResult, EventTime, Pitch, PitchTrack, ScanInfo

log = logging.getLogger("pitchspeed")

OUT_SUFFIX = "_pitchspeed"


def default_out_dir(video_path: str) -> str:
    p = Path(video_path)
    return str(p.with_name(p.stem + OUT_SUFFIX))


def out_paths(out_dir: str) -> dict:
    d = Path(out_dir)
    return {
        "calibration": str(d / "calibration.json"),
        "results_json": str(d / "results.json"),
        "results_csv": str(d / "results.csv"),
        "annotated": str(d / "annotated.mp4"),
        "transforms": str(d / "transforms.npz"),
        "index": str(d / "index.npz"),
        "log": str(d / "log.txt"),
        "debug": str(d / "debug"),
    }


@dataclass
class AnalyzeOptions:
    field_order: Optional[str] = None       # None/'auto' | 'tff' | 'bff' | 'progressive'
    par: Optional[float] = None             # override stored-pixel aspect ratio
    start_s: Optional[float] = None
    end_s: Optional[float] = None
    debug: bool = False
    finder: dict = field(default_factory=dict)   # extra FinderOptions fields


class Progress:
    """Minimal console progress reporter (works in a Windows console)."""

    def __init__(self, total: int, label: str, enabled: bool = True, every_s: float = 1.0):
        self.total = max(1, total)
        self.label = label
        self.enabled = enabled
        self.every_s = every_s
        self._t0 = time.monotonic()
        self._last = 0.0

    def update(self, done: int) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self._last < self.every_s and done < self.total:
            return
        self._last = now
        el = now - self._t0
        frac = min(1.0, done / self.total)
        eta = (el / frac - el) if frac > 0 else 0.0
        print(f"\r  {self.label}: {100 * frac:5.1f}%  ({done}/{self.total})  "
              f"elapsed {el:5.0f}s  eta {eta:5.0f}s", end="", flush=True)

    def done(self) -> None:
        if self.enabled:
            self.update(self.total)
            print()


def _manual_flagged(ev: Optional[EventTime]) -> bool:
    return ev is not None and ev.method == "manual"


def estimate_for(calib: Calibration, scan: ScanInfo, sample_dt: float, pitch: Pitch):
    from pitchspeed.estimate.speed import estimate_speed
    manual = _manual_flagged(pitch.release) or _manual_flagged(pitch.catch) or pitch.status == "manual"
    return estimate_speed(calib, scan, sample_dt, pitch.release, pitch.catch,
                          track=pitch.track, manual=manual)


def make_recompute(calib_file: CalibrationFile, scan: ScanInfo, source) -> Callable[[Pitch], Pitch]:
    """Callback for the review UI: re-estimate one pitch after an edit."""

    def recompute(pitch: Pitch) -> Pitch:
        frame = pitch.frame_release if pitch.frame_release >= 0 else source.locate(pitch.release.s or 0)[0]
        calib = calib_file.for_frame(frame) or calib_file.segment_calibs()[0]
        if pitch.release is not None and pitch.release.s is not None:
            pitch.frame_release = source.locate(pitch.release.s)[0]
        if pitch.catch is not None and pitch.catch.s is not None:
            pitch.frame_catch = source.locate(pitch.catch.s)[0]
        pitch.speed = estimate_for(calib, scan, source.sample_dt, pitch)
        return pitch

    return recompute


def renumber(pitches: list[Pitch]) -> list[Pitch]:
    """Chronological numbering of non-rejected pitches; rejected ones get 0."""
    pitches.sort(key=lambda p: p.release.t)
    n = 0
    for p in pitches:
        if p.status == "rejected":
            p.number = 0
        else:
            n += 1
            p.number = n
    return pitches


def _pitch_from_track(tr: PitchTrack, source, calib: Calibration, scan: ScanInfo) -> Pitch:
    rel = tr.release
    cat = tr.catch
    p = Pitch(number=0, status="auto", release=rel, catch=cat,
              speed=None, track=tr, confidence=tr.confidence, flags=list(tr.flags))
    s_rel = rel.s if rel.s is not None else source.sample_at_time(rel.t)
    s_cat = cat.s if cat.s is not None else source.sample_at_time(cat.t)
    p.frame_release = source.locate(s_rel)[0]
    p.frame_catch = source.locate(s_cat)[0]
    p.speed = estimate_for(calib, scan, source.sample_dt, p)
    return p


DOUBTFUL_CONFIDENCE = 0.3


def _timing_rms(t: np.ndarray, x: np.ndarray, y: np.ndarray, deg: int = 3) -> float:
    """RMS (px) of a smooth polynomial fit of x(t), y(t)."""
    tc = t - t.mean()
    r = []
    for v in (x, y):
        c = np.polyfit(tc, v, deg)
        r.append(v - np.polyval(c, tc))
    return float(np.sqrt(np.mean(np.concatenate(r) ** 2)))


def track_timing_votes(tracks: list[PitchTrack], source, frame_dt: float) -> dict:
    """Which field timing makes each ball track smooth?

    For interlaced input every track point carries the time of its field.
    Re-timing the same points as 'flipped' (the other field order) or as
    'progressive' (both fields of a frame at the same instant) and refitting a
    smooth curve shows which hypothesis the video really follows: the wrong
    ones make the track zig-zag by about one field of motion.
    """
    votes = {"given": 0, "flipped": 0, "progressive": 0, "unclear": 0}
    for tr in tracks:
        pts = [p for p in tr.points if p.inlier and not p.manual]
        if len(pts) < 12:
            continue
        t = np.array([p.t for p in pts])
        x = np.array([p.x for p in pts])
        y = np.array([p.y for p in pts])
        k = np.array([source.locate(p.s)[1] for p in pts], dtype=float)
        hyp = {"given": t, "flipped": t + (1.0 - 2.0 * k) * frame_dt / 2.0, "progressive": t - k * frame_dt / 2.0}
        rms = {h: _timing_rms(tt, x, y, 3 if len(pts) >= 16 else 2) for h, tt in hyp.items()}
        best = min(rms, key=rms.get)
        others = min(v for h, v in rms.items() if h != best)
        if others > 3.0 * max(rms[best], 0.2):
            votes[best] += 1
        else:
            votes["unclear"] += 1
        tr.diagnostics["timing_rms"] = rms
    return votes


def resolve_field_order(scan: ScanInfo, tracks: list[PitchTrack], source=None, frame_dt: float = 0.0) -> str:
    """Use the ball tracks to confirm or contradict the scan verdict.

    Returns 'confirmed' (scan.confidence raised to 'high'), 'flip' (tracks say
    the field order is reversed), 'progressive' (tracks say the fields are
    not separated in time) or 'unknown'.  User overrides are never second-guessed.
    With `source` given, the three timing hypotheses are tested directly;
    otherwise the finder's own given-vs-flipped check is used.
    """
    if not scan.interlaced or scan.source == "override":
        return "unknown"
    if source is not None:
        v = track_timing_votes(tracks, source, frame_dt)
        given, flipped, prog = v["given"], v["flipped"], v["progressive"]
    else:
        given = flipped = prog = 0
        for tr in tracks:
            fo = (tr.diagnostics or {}).get("field_order") or {}
            n = fo.get("n", 0)
            rg, rf = fo.get("rms_given"), fo.get("rms_flipped")
            if n < 12 or rg is None or rf is None:
                continue
            if fo.get("conflict") and rg > 3.0 * rf:
                flipped += 1
            elif fo.get("preferred") == "given" and rf > 3.0 * max(rg, 0.2):
                given += 1
    scan.evidence["tracks"] = {"prefer_given": given, "prefer_flipped": flipped, "prefer_progressive": prog}
    if prog >= 1 and prog > given and prog >= flipped:
        return "progressive"
    if flipped >= 1 and flipped > given:
        return "flip"
    if given >= 1 and flipped == 0 and prog == 0:
        scan.confidence = "high"
        scan.evidence["confirmed_by"] = "ball tracks"
        return "confirmed"
    return "unknown"


def analyze(video_path: str, calib_file: CalibrationFile, out_dir: Optional[str] = None,
            opts: Optional[AnalyzeOptions] = None, show_progress: bool = True, _flipped: bool = False):
    """Run the full analysis.  Returns (AnalysisResult, SampleSource)."""
    from pitchspeed.io.reader import VideoReader
    from pitchspeed.io.interlace import SampleSource, detect_scan
    from pitchspeed.vision.finder import FinderOptions, PitchFinder
    from pitchspeed.render.report import write_csv, write_json

    opts = opts or AnalyzeOptions()
    out_dir = out_dir or default_out_dir(video_path)
    os.makedirs(out_dir, exist_ok=True)
    paths = out_paths(out_dir)
    warnings: list[str] = []

    reader = VideoReader(video_path, sar_override=opts.par, index_cache=paths["index"])
    info = reader.info
    fo = None if opts.field_order in (None, "auto") else opts.field_order
    scan = detect_scan(reader, override=fo)
    source = SampleSource(reader, scan)
    if scan.mode != "interlaced":
        warnings.append("video is not interlaced (or was deinterlaced): timing uses whole frames "
                        "(1/30 s), roughly doubling timing error")
    if scan.confidence == "low":
        warnings.append(f"interlace/field-order detection is uncertain ({scan.mode}, {scan.field_order}); "
                        f"use --field-order to override if results look wrong")
    if info.timing_mode != "cfr":
        warnings.append("variable frame timing detected; timestamps taken from the file")
    if info.dropped_frames:
        warnings.append(f"{len(info.dropped_frames)} dropped-frame gaps in the capture")

    warnings += calib_file.validate()
    # 8:9 vs 10:11 (the two SD conventions) differ by ~2%, which barely matters here.
    if abs(calib_file.settings.sar - info.sar) > 0.05 * info.sar:
        warnings.append(f"calibration pixel aspect {calib_file.settings.sar:.4f} differs from the video's "
                        f"{info.sar:.4f}; using the calibration's value for geometry")

    fopts = FinderOptions.from_dict(opts.finder)
    if opts.debug:
        fopts.debug_dir = paths["debug"]

    s_lo = 0 if opts.start_s is None else source.sample_at_time(opts.start_s)
    s_hi = source.n_samples if opts.end_s is None else min(source.n_samples, source.sample_at_time(opts.end_s) + 1)

    pitches: list[Pitch] = []
    rejected: list[PitchTrack] = []
    missed: list[dict] = []
    seg_results = []
    tf_s, tf_m, tf_q, tf_h = [], [], [], []
    for seg_i, calib in enumerate(calib_file.segment_calibs()):
        seg = calib.segment
        s0 = max(s_lo, source.sample_of(seg.start_frame, 0))
        s1 = s_hi if seg.end_frame is None else min(s_hi, source.sample_of(seg.end_frame, 0))
        if s1 <= s0:
            continue
        ref = reader.get_frame(calib.ref_frame, "gray")
        finder = PitchFinder(calib, scan, source.sample_dt, source.sample_shape, ref, fopts)
        prog = Progress(s1 - s0, f"segment {seg_i + 1}: tracking", enabled=show_progress)
        for j, sample in enumerate(source.iter_samples(s0, s1)):
            finder.process(sample)
            prog.update(j + 1)
        prog.done()
        fr = finder.finish()
        tf_s.append(np.asarray(fr.sample_index))
        tf_m.append(np.asarray(fr.transforms))
        tf_q.append(np.asarray(fr.transform_quality))
        h = fr.diagnostics.get("transforms_3x3") if isinstance(fr.diagnostics, dict) else None
        tf_h.append(np.asarray(h) if h is not None and len(h) == len(fr.sample_index)
                    else _affine_to_3x3(np.asarray(fr.transforms)))
        seg_results.append((calib, fr))

    # Field order: the detector can be unsure on real footage, but a well-tracked
    # ball settles it (wrong order makes the track zig-zag by a field's motion).
    verdict = resolve_field_order(scan, [tr for _, fr in seg_results for tr in fr.tracks],
                                  source, 1.0 / info.frame_rate)
    if verdict in ("flip", "progressive") and not _flipped:
        if verdict == "flip":
            new_order = "tff" if scan.field_order == "bff" else "bff"
            msg = (f"field order corrected to {new_order.upper()} from the ball tracks "
                   f"(the detector said {scan.field_order.upper()})")
        else:
            new_order = "progressive"
            msg = ("the ball tracks show the two fields of each frame were captured at the same "
                   "instant (deinterlaced or progressive video): using whole frames")
        log.warning("%s; re-running", msg)
        if show_progress:
            print(f"  note: {msg}; re-running")
        reader.close()
        o2 = AnalyzeOptions(**{**opts.__dict__, "field_order": new_order})
        result, source2 = analyze(video_path, calib_file, out_dir, o2, show_progress, _flipped=True)
        result.scan.source = "tracks"
        result.scan.confidence = "medium"
        result.params["field_order_option"] = "auto (corrected from tracks)"
        result.warnings.insert(0, msg)
        write_json(result, paths["results_json"])
        write_csv(result, paths["results_csv"])
        return result, source2
    if verdict == "confirmed":
        warnings = [w for w in warnings if not w.startswith("interlace/field-order detection is uncertain")]

    for calib, fr in seg_results:
        for tr in fr.tracks:
            if tr.release is None or tr.catch is None:
                tr.status = "rejected"
                tr.reject_code = tr.reject_code or "NO_EVENTS"
                rejected.append(tr)
                continue
            p = _pitch_from_track(tr, source, calib, scan)
            if tr.confidence < DOUBTFUL_CONFIDENCE:
                # Low-confidence track: often not a pitch at all (bird, throw back, arm,
                # clutter).  List it for review instead of giving it a pitch number; the
                # user can accept it there.
                tr.status = "rejected"
                tr.reject_code = "DOUBTFUL"
                rejected.append(tr)
                if p.speed.v_release_mph is not None:
                    what = (f"if it is a pitch: about {p.speed.v_release_mph:.0f} mph "
                            f"+/- {p.speed.sigma_mph:.0f}")
                else:
                    what = p.speed.reason or "speed not measurable"
                missed.append({"s_start": tr.points[0].s, "s_end": tr.points[-1].s,
                               "t_start": tr.points[0].t, "t_end": tr.points[-1].t,
                               "reason": f"unconfirmed moving object (confidence {tr.confidence:.2f}); {what}"})
                continue
            pitches.append(p)
        rejected.extend(fr.rejected)
        missed.extend(fr.possible_missed)

    missed.sort(key=lambda m: m.get("t_start", 0.0))
    renumber(pitches)
    if tf_s:
        np.savez_compressed(paths["transforms"], samples=np.concatenate(tf_s),
                            transforms=np.concatenate(tf_m), quality=np.concatenate(tf_q),
                            transforms_3x3=np.concatenate(tf_h))

    params = {
        "field_order_option": opts.field_order or "auto",
        "par_override": opts.par,
        "start_s": opts.start_s,
        "end_s": opts.end_s,
        "sample_dt": source.sample_dt,
        "finder": opts.finder,
    }
    result = AnalysisResult(video=info, scan=scan, calibration=calib_file.to_dict(), params=params,
                            pitches=pitches, rejected_tracks=rejected, possible_missed=missed,
                            warnings=warnings, version=__version__)
    write_json(result, paths["results_json"])
    write_csv(result, paths["results_csv"])
    return result, source


def _affine_to_3x3(A: np.ndarray) -> np.ndarray:
    if A.size == 0:
        return np.zeros((0, 3, 3))
    H = np.zeros((len(A), 3, 3))
    H[:, :2, :] = A
    H[:, 2, 2] = 1.0
    return H


def load_transforms(out_dir: str):
    p = out_paths(out_dir)["transforms"]
    if not os.path.exists(p):
        return None, None
    d = np.load(p)
    return d["transforms"], d["samples"]
