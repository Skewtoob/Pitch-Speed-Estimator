"""End-to-end analysis: video + calibration -> pitches with speeds.

    reader -> scan detection -> SampleSource -> PitchFinder (per calibration
    segment) -> speed estimates -> AnalysisResult -> results.json / .csv

USER WORK IS NEVER THROWN AWAY.  Before `analyze` replaces a results.json
that holds review work (edited / manual / rejected pitches, notes, edit
history, checked windows) it copies it to results.backup-<time>.json and,
when the new analysis uses the same sample space and calibration frames,
carries the reviewed pitches over into the new result (`carry_over_user_work`).
results.json is always written before results.csv; a locked CSV (open in
Excel) is reported, never fatal (`save_results`).
"""
from __future__ import annotations

import copy
import datetime as _dt
import json
import logging
import os
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from pitchspeed import __version__
from pitchspeed.calib.model import Calibration, CalibrationFile, is_manual
from pitchspeed.types import AnalysisResult, EventTime, Pitch, PitchTrack, ScanInfo, to_jsonable

log = logging.getLogger("pitchspeed")

OUT_SUFFIX = "_pitchspeed"


def default_out_dir(video_path: str) -> str:
    p = Path(video_path)
    return str(p.with_name(p.stem + OUT_SUFFIX))


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def recorded_video_names(out_dir: str) -> dict:
    """File names (basenames) the calibration / results in `out_dir` were made for."""
    p = out_paths(out_dir)
    out = {}
    cal = _read_json(p["calibration"]) if os.path.exists(p["calibration"]) else None
    if cal and isinstance(cal.get("video"), dict) and cal["video"].get("name"):
        out["calibration"] = os.path.basename(str(cal["video"]["name"]))
    res = _read_json(p["results_json"]) if os.path.exists(p["results_json"]) else None
    if res and isinstance(res.get("video"), dict) and res["video"].get("path"):
        out["results"] = os.path.basename(str(res["video"]["path"]))
    return out


def _same_name(a: str, b: str) -> bool:
    return os.path.normcase(a) == os.path.normcase(b)


def resolve_out_dir(video_path: str) -> tuple[str, Optional[str]]:
    """Default output folder for `video_path`, plus a note when it is not `default_out_dir`.

    <stem>_pitchspeed is shared by files that differ only in the extension
    (game1.avi and its export game1.mpg).  If that folder already belongs to
    such a sibling - its results / calibration name another file that exists
    next to this one - this video gets <stem>_<ext>_pitchspeed instead, so it
    neither reuses the other file's clicks nor overwrites its reviewed results.
    """
    d = default_out_dir(video_path)
    me = os.path.basename(video_path)
    names = recorded_video_names(d)
    if "results" in names and _same_name(names["results"], me):
        return d, None
    folder = os.path.dirname(os.path.abspath(video_path))
    p = Path(video_path)
    others = [n for n in names.values()
              if not _same_name(n, me) and _same_name(Path(n).stem, p.stem)        # game1.avi vs game1.mpg
              and os.path.isfile(os.path.join(folder, n))]
    if not others:
        return d, None
    alt = str(p.with_name(f"{p.stem}_{p.suffix.lstrip('.') or 'video'}{OUT_SUFFIX}"))
    return alt, (f"{os.path.basename(d)} belongs to {others[0]}, so {me} uses its own folder "
                 f"{os.path.basename(alt)}")


def video_fingerprint(video_path: str, info) -> dict:
    """What calibration.json records about the video it was clicked on."""
    try:
        size = os.path.getsize(video_path)
    except OSError:
        size = 0
    return {"name": os.path.basename(video_path), "size": int(size), "n_frames": int(info.n_frames),
            "width": int(info.width), "height": int(info.height)}


FILE_CHANGED_WARNING = "the video file has changed since it was calibrated (different file size)"


def calibration_video_problems(cf: CalibrationFile, video_path: str, info) -> tuple[list[str], list[str]]:
    """(errors, warnings) from comparing a calibration with the video it is applied to.

    errors: the clicks cannot belong to this video (different image size).
    warnings: probably another file or another version of it (name, frame
    count, file size differ) - the clicks, the reference frame and the
    segment frame numbers may not fit."""
    errors: list[str] = []
    warns: list[str] = []
    fp = cf.video if isinstance(cf.video, dict) else {}
    w, h = int(info.width), int(info.height)
    fp_dims = (int(fp["width"]), int(fp["height"])) if fp.get("width") and fp.get("height") else None
    isz = tuple(int(v) for v in cf.image_size) if cf.image_size else None
    if fp_dims is not None and fp_dims != (w, h):
        errors.append(f"the calibration was clicked on a {fp_dims[0]}x{fp_dims[1]} video, but this video "
                      f"is {w}x{h}: the clicks cannot be used for it")
    elif isz is not None and isz != (w, h):
        warns.append(f"the calibration is for {isz[0]}x{isz[1]} images, this video is {w}x{h}: "
                     f"check the clicks")
    name = fp.get("name")
    me = os.path.basename(video_path)
    if name and not _same_name(str(name), me):
        warns.append(f"the calibration was made for '{name}', not for '{me}'")
    nf = fp.get("n_frames")
    if nf and int(nf) != int(info.n_frames):
        warns.append(f"the calibration's video had {int(nf)} frames, this one has {int(info.n_frames)}: if it is "
                     f"a trimmed or re-encoded copy, the clicked reference frame and segment start/end frames "
                     f"no longer fit")
    size = fp.get("size")
    if size and not warns and not errors:
        try:
            now = os.path.getsize(video_path)
        except OSError:
            now = None
        if now is not None and int(size) != int(now):
            warns.append(FILE_CHANGED_WARNING)
    return errors, warns


# ----------------------------------------------------- user work / saving
USER_STATUSES = ("edited", "manual", "rejected")
BACKUP_PREFIX = "results.backup-"
CSV_WARNING_PREFIX = "results.csv was NOT updated"


def is_user_pitch(p: Pitch) -> bool:
    """A pitch the user touched in review (status, edit history or notes)."""
    return p.status in USER_STATUSES or bool(p.edit_history) or bool((p.notes or "").strip())


def user_work_summary(result: AnalysisResult) -> dict:
    """Counts of review work in `result` (all zero for an untouched analysis)."""
    ps = [p for p in result.pitches if is_user_pitch(p)]
    return {
        "pitches": len(ps),
        "manual": sum(1 for p in ps if p.status == "manual"),
        "edited": sum(1 for p in ps if p.status == "edited"),
        "rejected": sum(1 for p in ps if p.status == "rejected"),
        "windows": sum(1 for m in (result.possible_missed or []) if isinstance(m, dict) and m.get("review")),
        "tracks": sum(1 for t in (result.rejected_tracks or [])
                      if (t.diagnostics or {}).get("review")
                      and not str(t.diagnostics["review"]).startswith("replaced")),   # set by carry-over
    }


def has_user_work(result: AnalysisResult) -> bool:
    s = user_work_summary(result)
    return bool(s["pitches"] or s["windows"] or s["tracks"])


def describe_user_work(s: dict) -> str:
    parts = []
    if s.get("pitches"):
        kinds = [f"{s[k]} {k}" for k in ("manual", "edited", "rejected") if s.get(k)]
        other = s["pitches"] - sum(s.get(k, 0) for k in ("manual", "edited", "rejected"))
        if other:
            kinds.append(f"{other} accepted/annotated")
        parts.append(f"{s['pitches']} reviewed pitch{'es' if s['pitches'] != 1 else ''} ({', '.join(kinds)})")
    if s.get("windows"):
        parts.append(f"{s['windows']} checked 'possible missed' window{'s' if s['windows'] != 1 else ''}")
    if s.get("tracks"):
        parts.append(f"{s['tracks']} checked rejected track{'s' if s['tracks'] != 1 else ''}")
    return ", ".join(parts) or "no review edits"


def backup_results(path: str) -> str:
    """Copy results.json to results.backup-YYYYmmdd-HHMMSS.json (never overwrites); returns the copy."""
    folder = os.path.dirname(os.path.abspath(path))
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    dst = os.path.join(folder, f"{BACKUP_PREFIX}{stamp}.json")
    i = 1
    while os.path.exists(dst):
        dst = os.path.join(folder, f"{BACKUP_PREFIX}{stamp}-{i}.json")
        i += 1
    shutil.copy2(path, dst)
    return dst


def save_results(result: AnalysisResult, out_dir: str, csv_retry_s: Optional[float] = None) -> Optional[str]:
    """Write results.json, then results.csv.

    results.json is the safe copy: if it cannot be written the error is
    raised.  A CSV that cannot be replaced (typically open in Excel) is not
    fatal: the problem is returned as a message (also added to
    result.warnings in memory - never into the saved results.json - so the
    console table shows it).  csv_retry_s: how long to retry a locked CSV
    (None = the report module's default; 0 = one attempt, for autosaves)."""
    from pitchspeed.render.report import write_csv, write_json
    p = out_paths(out_dir)
    result.warnings = [w for w in (result.warnings or []) if not str(w).startswith(CSV_WARNING_PREFIX)]
    write_json(result, p["results_json"])
    try:
        if csv_retry_s is None:
            write_csv(result, p["results_csv"])
        else:
            write_csv(result, p["results_csv"], retry_s=csv_retry_s)
    except OSError as e:
        msg = (f"{CSV_WARNING_PREFIX}: {e} - it still holds older results. results.json is up to date. "
               f"Close results.csv (e.g. in Excel) and run `pitchspeed review` (Q saves it again).")
        log.warning("%s", msg)
        result.warnings.append(msg)
        return msg
    return None


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


F_DUPLICATE_FRAME = "duplicate_frame_suspected"
F_DROPPED_FRAME = "dropped_frame_suspected"
GLITCH_FLAGS = (F_DUPLICATE_FRAME, F_DROPPED_FRAME)


def frame_glitch_flags(frame_release: int, frame_catch: int, duplicate_frames=(), dropped_frames=()) -> list[str]:
    """Capture glitches that can distort a pitch's flight time.

    duplicate_frames: frame i repeats frame i-1 (VideoInfo.duplicate_frames);
    dropped_frames: frame i has a timestamp gap before it (VideoInfo.dropped_frames).
    A glitch from one frame before the release to one after the catch counts
    (the true release / catch instant may lie in it)."""
    try:
        a, b = int(frame_release), int(frame_catch)
    except (TypeError, ValueError):
        return []
    if a < 0 or b < 0:
        return []
    lo, hi = min(a, b), max(a, b)
    out = []
    if any(lo - 1 <= int(f) <= hi + 1 for f in (duplicate_frames or ())):
        out.append(F_DUPLICATE_FRAME)
    if any(lo <= int(f) <= hi + 1 for f in (dropped_frames or ())):
        out.append(F_DROPPED_FRAME)
    return out


def undecodable_frames(reader) -> list[int]:
    """Frames the reader could not decode (VideoReader.decode_errors), sorted."""
    out: set[int] = set()
    for e in getattr(reader, "decode_errors", None) or ():
        if not isinstance(e, dict):
            continue
        if e.get("frame") is not None:
            out.add(int(e["frame"]))
        fr = e.get("frames")
        if isinstance(fr, (list, tuple)) and len(fr) == 2:
            out.update(range(int(fr[0]), int(fr[1]) + 1))
    return sorted(out)


def _glitch_lists(*infos, undecodable=()) -> tuple[list[int], list[int]]:
    """Union of duplicate / dropped frame lists of the given VideoInfo objects (None skipped);
    undecodable frames (never seen by the tracker) count as dropped."""
    dup, drop = set(), set(int(f) for f in (undecodable or ()))
    for info in infos:
        if info is None:
            continue
        dup.update(int(f) for f in (getattr(info, "duplicate_frames", None) or ()))
        drop.update(int(f) for f in (getattr(info, "dropped_frames", None) or ()))
    return sorted(dup), sorted(drop)


def set_glitch_flags(pitch: Pitch, duplicate_frames=(), dropped_frames=()) -> list[str]:
    """(Re)set the duplicate/dropped-frame flags of a pitch and its track for its current frame range."""
    new = frame_glitch_flags(pitch.frame_release, pitch.frame_catch, duplicate_frames, dropped_frames)
    holders = [pitch.flags]
    if pitch.track is not None:
        if pitch.track.flags is None:
            pitch.track.flags = []
        holders.append(pitch.track.flags)
    for fl in holders:
        fl[:] = [f for f in fl if f not in GLITCH_FLAGS] + new
    return new


def estimate_for(calib: Calibration, scan: ScanInfo, sample_dt: float, pitch: Pitch):
    from pitchspeed.estimate.speed import estimate_speed
    manual = _manual_flagged(pitch.release) or _manual_flagged(pitch.catch) or pitch.status == "manual"
    track = pitch.track
    if track is None:
        glitch = [f for f in (pitch.flags or []) if f in GLITCH_FLAGS]
        if glitch:
            # Track-less (hand-marked) pitch: a point-less stand-in carries the
            # capture-glitch flags to the estimator (same result as track=None otherwise).
            track = PitchTrack(track_id=-1, points=[], flags=glitch)
    return estimate_speed(calib, scan, sample_dt, pitch.release, pitch.catch,
                          track=track, manual=manual)


def _event_sample(source, ev: Optional[EventTime]) -> Optional[int]:
    if ev is None:
        return None
    if ev.s is not None:
        return int(ev.s)
    if ev.t is None or not np.isfinite(ev.t):
        return None
    return int(source.sample_at_time(float(ev.t)))


def make_recompute(calib_file: CalibrationFile, scan: ScanInfo, source, video=None,
                   undecodable=()) -> Callable[[Pitch], Pitch]:
    """Callback for the review UI: re-estimate one pitch after an edit.

    Frame numbers and the duplicate/dropped-frame flags follow the (possibly
    edited) release/catch; `video` (the result's VideoInfo) and `undecodable`
    (result.params['undecodable_frames']) add what the analysis recorded to
    what the source has seen."""

    def recompute(pitch: Pitch) -> Pitch:
        s_rel, s_cat = _event_sample(source, pitch.release), _event_sample(source, pitch.catch)
        if s_rel is not None:
            pitch.frame_release = int(source.locate(s_rel)[0])
        if s_cat is not None:
            pitch.frame_catch = int(source.locate(s_cat)[0])
        frame = pitch.frame_release if pitch.frame_release >= 0 else 0
        calib = calib_file.for_frame(frame) or calib_file.segment_calibs()[0]
        reader = getattr(source, "reader", None)
        dup, drop = _glitch_lists(video, getattr(reader, "info", None),
                                  undecodable=list(undecodable or ()) + undecodable_frames(reader))
        set_glitch_flags(pitch, dup, drop)
        _drop_stale_event_flags(pitch)
        pitch.speed = estimate_for(calib, scan, source.sample_dt, pitch)
        return pitch

    return recompute


# Flags that describe how the AUTOMATIC event was found; meaningless once the
# user has marked that event by hand.
_RELEASE_TRACK_FLAGS = {"release_unconfirmed", "release_uncertain", "release_not_observed",
                        "release_detection_late", "release_gap", "windup_trimmed",
                        "release_far_extrapolated", "ball_free_before_release_point"}
_CATCH_TRACK_FLAGS = {"catch_extrapolated", "catch_not_observed", "track_end_gap_trimmed", "contact"}


def _drop_stale_event_flags(pitch: Pitch) -> None:
    drop = set()
    if pitch.release is not None and pitch.release.method == "manual":
        drop |= _RELEASE_TRACK_FLAGS
    if pitch.catch is not None and pitch.catch.method == "manual":
        drop |= _CATCH_TRACK_FLAGS
    if drop:
        pitch.flags = [f for f in pitch.flags if f not in drop]


def recompute_all(result: AnalysisResult, calib_file: CalibrationFile, source) -> int:
    """Re-estimate every pitch of `result` with `calib_file` (e.g. after the
    settings changed), keeping all review edits; updates result.calibration.
    Returns the number of pitches recomputed."""
    from pitchspeed.types import SpeedEstimate
    rc = make_recompute(calib_file, result.scan, source, video=result.video,
                        undecodable=(result.params or {}).get("undecodable_frames"))
    n = 0
    for p in result.pitches:
        try:
            rc(p)
            n += 1
        except Exception as e:                                   # noqa: BLE001 - reported, never hidden
            log.warning("pitch %s: speed could not be recomputed: %s", p.number, e)
            p.speed = SpeedEstimate(v_release_mph=None, reason=f"speed could not be recomputed: {e}")
    _refresh_doubtful_windows(result, calib_file, source)
    result.calibration = calib_file.to_dict()
    renumber(result.pitches)
    return n


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


def _pitch_from_track(tr: PitchTrack, source, calib: Calibration, scan: ScanInfo, info=None,
                      undecodable=()) -> Pitch:
    rel = tr.release
    cat = tr.catch
    p = Pitch(number=0, status="auto", release=rel, catch=cat,
              speed=None, track=tr, confidence=tr.confidence, flags=list(tr.flags))
    s_rel = rel.s if rel.s is not None else source.sample_at_time(rel.t)
    s_cat = cat.s if cat.s is not None else source.sample_at_time(cat.t)
    p.frame_release = source.locate(s_rel)[0]
    p.frame_catch = source.locate(s_cat)[0]
    dup, drop = _glitch_lists(info if info is not None else getattr(getattr(source, "reader", None), "info", None),
                              undecodable=undecodable)
    set_glitch_flags(p, dup, drop)
    p.speed = estimate_for(calib, scan, source.sample_dt, p)
    return p


def _doubtful_window(tr: PitchTrack, p: Pitch) -> dict:
    """'possible missed' entry for a low-confidence track (listed for review, not numbered)."""
    if p.speed is not None and p.speed.v_release_mph is not None:
        what = f"if it is a pitch: about {p.speed.v_release_mph:.0f} mph +/- {p.speed.sigma_mph:.0f}"
    else:
        what = (p.speed.reason if p.speed is not None else None) or "speed not measurable"
    return {"s_start": tr.points[0].s, "s_end": tr.points[-1].s,
            "t_start": tr.points[0].t, "t_end": tr.points[-1].t,
            "reason": f"unconfirmed moving object (confidence {tr.confidence:.2f}); {what}"}


def _refresh_doubtful_windows(result: AnalysisResult, calib_file: CalibrationFile, source) -> None:
    """Redo the 'if it is a pitch: about N mph' hints with the current calibration."""
    by_range = {}
    for tr in result.rejected_tracks or []:
        if tr.reject_code == "DOUBTFUL" and tr.points and tr.release is not None and tr.catch is not None:
            by_range[(tr.points[0].s, tr.points[-1].s)] = tr
    for m in result.possible_missed or []:
        tr = by_range.get((m.get("s_start"), m.get("s_end")))
        if tr is None or not str(m.get("reason", "")).startswith("unconfirmed moving object"):
            continue
        try:
            frame = int(source.locate(_event_sample(source, tr.release))[0])
            calib = calib_file.for_frame(frame) or calib_file.segment_calibs()[0]
            p = _pitch_from_track(copy.deepcopy(tr), source, calib, result.scan, info=result.video,
                                  undecodable=(result.params or {}).get("undecodable_frames") or ())
            m["reason"] = _doubtful_window(tr, p)["reason"]
        except Exception as e:                                   # noqa: BLE001
            log.info("could not refresh the hint for a doubtful track: %s", e)


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
        pair = (tr.diagnostics or {}).get("field_order_pair") or {}
        if min(int((k == 0).sum()), int((k == 1).sum())) < 4:
            # (nearly) one field parity only: the hypotheses differ by a constant
            # time shift and cannot be told apart.  With the wrong field order the
            # tracker tends to build one such track per parity; the finder's pair
            # check catches that case.
            votes["flipped" if pair.get("conflict") else "unclear"] += 1
            continue
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


def _video_warnings(video_path: str, reader, scan: ScanInfo, source, calib_file: CalibrationFile) -> list[str]:
    """Warnings about the video and the calibration (shared by analyze and start_marking)."""
    info = reader.info
    warnings: list[str] = []
    cal_errors, cal_warns = calibration_video_problems(calib_file, video_path, info)
    warnings += [f"calibration does not fit this video: {w}" for w in cal_errors + cal_warns]
    if scan.mode != "interlaced":
        step = 1.0 / scan.sample_rate if scan.sample_rate else source.sample_dt
        if step > 0.025:
            warnings.append(f"video is not interlaced (or was deinterlaced): timing uses whole frames "
                            f"(1/{1.0 / step:.0f} s), roughly doubling timing error compared with "
                            f"interlaced tape")
        else:
            warnings.append(f"video is progressive at {1.0 / step:.2f} frames/s (no fields to split)")
    if scan.confidence == "low":
        warnings.append(f"interlace/field-order detection is uncertain ({scan.mode}, {scan.field_order}); "
                        f"use --field-order to override if results look wrong")
    for note in getattr(reader, "notes", []) or []:
        warnings.append(f"video file: {note}")
    if info.timing_mode != "cfr":
        warnings.append("variable frame timing detected; timestamps taken from the file")
    if info.dropped_frames:
        warnings.append(f"{len(info.dropped_frames)} dropped-frame gaps in the capture; pitches that span "
                        f"one are flagged '{F_DROPPED_FRAME}' and get a wider +/-")

    warnings += calib_file.validate()
    # 8:9 vs 10:11 (the two SD conventions) differ by ~2%, which barely matters here.
    if not is_manual(calib_file) and abs(calib_file.settings.sar - info.sar) > 0.05 * info.sar:
        warnings.append(f"calibration pixel aspect {calib_file.settings.sar:.4f} differs from the video's "
                        f"{info.sar:.4f}; using the calibration's value for geometry")

    return warnings


def analyze(video_path: str, calib_file: CalibrationFile, out_dir: Optional[str] = None,
            opts: Optional[AnalyzeOptions] = None, show_progress: bool = True, _flipped: bool = False):
    """Run the full analysis.  Returns (AnalysisResult, SampleSource).

    Writes results.json / results.csv / transforms.npz in out_dir.  An
    existing results.json with review work is backed up first and its
    reviewed pitches are carried over when that is safe (see module doc)."""
    from pitchspeed.io.reader import VideoReader
    from pitchspeed.io.interlace import SampleSource, detect_scan
    from pitchspeed.vision.finder import FinderOptions, PitchFinder

    opts = opts or AnalyzeOptions()
    if is_manual(calib_file):
        raise ValueError("this video is set up for marking the pitches by hand (no clicks on the rubber and "
                         "plate), so there is nothing to search automatically: use `pitchspeed mark`, or "
                         "`pitchspeed calibrate` to click the points for the automatic search")
    out_dir = out_dir or default_out_dir(video_path)
    os.makedirs(out_dir, exist_ok=True)
    paths = out_paths(out_dir)
    warnings: list[str] = []
    # Before anything in out_dir is replaced: back up review work (results + transforms).
    prev = None if _flipped else PreviousResults.load(paths)

    reader = VideoReader(video_path, sar_override=opts.par, index_cache=paths["index"])
    info = reader.info
    fo = None if opts.field_order in (None, "auto") else opts.field_order
    scan = detect_scan(reader, override=fo)
    source = SampleSource(reader, scan)
    warnings += _video_warnings(video_path, reader, scan, source, calib_file)

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
        if calib.view == "manual":                 # marked by hand: nothing to search here
            continue
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
        _finish_analysis(result, calib_file, source2, out_dir, prev, show_progress)
        return result, source2
    if verdict == "confirmed":
        warnings = [w for w in warnings if not w.startswith("interlace/field-order detection is uncertain")]

    bad_frames = undecodable_frames(reader)          # skipped while tracking: treated like a timing gap
    for calib, fr in seg_results:
        for tr in fr.tracks:
            if tr.release is None or tr.catch is None:
                tr.status = "rejected"
                tr.reject_code = tr.reject_code or "NO_EVENTS"
                rejected.append(tr)
                continue
            p = _pitch_from_track(tr, source, calib, scan, info, undecodable=bad_frames)
            if tr.confidence < DOUBTFUL_CONFIDENCE:
                # Low-confidence track: often not a pitch at all (bird, throw back, arm,
                # clutter).  List it for review instead of giving it a pitch number; the
                # user can accept it there.
                tr.status = "rejected"
                tr.reject_code = "DOUBTFUL"
                rejected.append(tr)
                missed.append(_doubtful_window(tr, p))
                continue
            pitches.append(p)
        rejected.extend(fr.rejected)
        missed.extend(fr.possible_missed)

    pitches = _demote_throwbacks(pitches, rejected, missed)
    missed.sort(key=lambda m: m.get("t_start", 0.0))
    if info.duplicate_frames:      # filled while the samples were read (SampleSource.iter_samples)
        warnings.append(f"{len(info.duplicate_frames)} repeated (duplicate) frame(s) in the capture; pitches "
                        f"that span one are flagged '{F_DUPLICATE_FRAME}' and get a wider +/-")
    if bad_frames:
        warnings.append(f"{len(bad_frames)} frame(s) could not be decoded (first: {bad_frames[:5]}); pitches "
                        f"that span one are flagged '{F_DROPPED_FRAME}' and get a wider +/-")
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
        "undecodable_frames": bad_frames,
    }
    result = AnalysisResult(video=info, scan=scan, calibration=calib_file.to_dict(), params=params,
                            pitches=pitches, rejected_tracks=rejected, possible_missed=missed,
                            warnings=warnings, version=__version__)
    if not _flipped:
        _finish_analysis(result, calib_file, source, out_dir, prev, show_progress)
    return result, source


def start_marking(video_path: str, calib_file: CalibrationFile, out_dir: Optional[str] = None,
                  opts: Optional[AnalyzeOptions] = None):
    """Open a video for marking its pitches by hand (no automatic search).

    Detects the interlacing and field order (the marks are field-exact, so
    the field timing matters), and writes results.json / results.csv with no
    pitches yet, plus the warnings about the video.  Returns
    (AnalysisResult, SampleSource).  An existing results.json is backed up
    first; a transforms.npz from an earlier automatic analysis is moved aside
    (it describes camera motion the hand-marked session does not use)."""
    from pitchspeed.io.reader import VideoReader
    from pitchspeed.io.interlace import SampleSource, detect_scan

    opts = opts or AnalyzeOptions()
    out_dir = out_dir or default_out_dir(video_path)
    os.makedirs(out_dir, exist_ok=True)
    paths = out_paths(out_dir)
    notes: list[str] = []
    if os.path.exists(paths["results_json"]):
        bak = backup_results(paths["results_json"])
        notes.append(f"the previous results were backed up as {os.path.basename(bak)}")
    if os.path.exists(paths["transforms"]):
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        aside = os.path.join(out_dir, f"transforms.backup-{stamp}.npz")
        os.replace(paths["transforms"], aside)
        notes.append(f"the camera-motion file of an earlier automatic analysis was moved to "
                     f"{os.path.basename(aside)}")
    reader = VideoReader(video_path, sar_override=opts.par, index_cache=paths["index"])
    fo = None if opts.field_order in (None, "auto") else opts.field_order
    scan = detect_scan(reader, override=fo)
    source = SampleSource(reader, scan)
    warnings = notes + [
        (w.split("; use --field-order")[0] + "; in the marking window press K if motion jumps back and forth "
         "when you step field by field") if w.startswith("interlace/field-order detection is uncertain") else w
        for w in _video_warnings(video_path, reader, scan, source, calib_file)]
    params = {
        "mode": "marked_by_hand",
        "field_order_option": opts.field_order or "auto",
        "par_override": opts.par,
        "start_s": None,
        "end_s": None,
        "sample_dt": source.sample_dt,
        "undecodable_frames": undecodable_frames(reader),
    }
    result = AnalysisResult(video=reader.info, scan=scan, calibration=calib_file.to_dict(), params=params,
                            pitches=[], warnings=warnings, version=__version__)
    save_results(result, out_dir)
    return result, source


def is_marking_session(result: AnalysisResult) -> bool:
    """True for results made by `start_marking` (pitches marked by hand, no automatic search)."""
    return (result.params or {}).get("mode") == "marked_by_hand"


THROWBACK_WINDOW_S = 4.0      # catcher's return throw arrives this soon after a catch
THROWBACK_MAX_PROGRESS = 0.3  # ... and is only seen near the pitcher's end of the path


def _demote_throwbacks(pitches: list[Pitch], rejected: list, missed: list) -> list[Pitch]:
    """Move likely catcher throw-backs from the pitch list to the review list.

    After every pitch the catcher throws the ball back, typically 1.5-4 s after
    the catch; near the pitcher that lob can look like the start of a pitch.
    A track is demoted only if it has no measurable speed, starts within
    THROWBACK_WINDOW_S of the previous pitch's catch, and never gets far along
    the release->catch path.  Real pitches come much further apart.
    """
    kept: list[Pitch] = []
    prev_catch = None
    for p in sorted(pitches, key=lambda q: q.release.t):
        d = (p.track.diagnostics if p.track else {}) or {}
        prog_last = d.get("progress_last")
        if (prev_catch is not None and p.speed.v_release_mph is None and p.status == "auto"
                and 0.0 <= p.release.t - prev_catch <= THROWBACK_WINDOW_S
                and prog_last is not None and prog_last < THROWBACK_MAX_PROGRESS):
            tr = p.track
            tr.status, tr.reject_code = "rejected", "THROWBACK_TIMING"
            rejected.append(tr)
            missed.append({"s_start": tr.points[0].s, "s_end": tr.points[-1].s,
                           "t_start": tr.points[0].t, "t_end": tr.points[-1].t,
                           "reason": f"probably the catcher's throw back to the pitcher "
                                     f"({p.release.t - prev_catch:.1f} s after the previous catch)"})
            continue
        kept.append(p)
        prev_catch = p.catch.t
    return _demote_prepitch_junk(kept, rejected, missed)


PREPITCH_WINDOW_S = 1.5   # junk this soon before a real pitch's release (wind-up, glove)


def _demote_prepitch_junk(pitches: list[Pitch], rejected: list, missed: list) -> list[Pitch]:
    """Move no-speed tracks that end just before a stronger pitch to the review list.

    The wind-up or the pitcher's glove can produce a short track at the release
    end right before the real pitch.  Demoted only if it has no measurable
    speed, never gets far along the path, and the next pitch (which has a
    speed or a higher confidence) is released within PREPITCH_WINDOW_S.
    """
    ps = sorted(pitches, key=lambda q: q.release.t)
    kept: list[Pitch] = []
    for i, p in enumerate(ps):
        nxt = ps[i + 1] if i + 1 < len(ps) else None
        d = (p.track.diagnostics if p.track else {}) or {}
        prog_last = d.get("progress_last")
        if (nxt is not None and p.status == "auto" and p.speed.v_release_mph is None
                and prog_last is not None and prog_last < THROWBACK_MAX_PROGRESS
                and 0.0 < nxt.release.t - p.release.t <= PREPITCH_WINDOW_S
                and (nxt.speed.v_release_mph is not None or nxt.confidence > p.confidence)):
            tr = p.track
            tr.status, tr.reject_code = "rejected", "PREPITCH"
            rejected.append(tr)
            missed.append({"s_start": tr.points[0].s, "s_end": tr.points[-1].s,
                           "t_start": tr.points[0].t, "t_end": tr.points[-1].t,
                           "reason": "probably the wind-up or glove just before the next pitch"})
            continue
        kept.append(p)
    return kept


def _affine_to_3x3(A: np.ndarray) -> np.ndarray:
    if A.size == 0:
        return np.zeros((0, 3, 3))
    H = np.zeros((len(A), 3, 3))
    H[:, :2, :] = A
    H[:, 2, 2] = 1.0
    return H


def load_transforms(out_dir: str):
    """(T, samples) from transforms.npz, or (None, None).

    T is (n, 3, 3) - the homographies the finder mapped positions with
    (sample frame coords -> reference coords) - when the file has
    'transforms_3x3', else the (n, 2, 3) affine set of older files.
    Consumers must accept both shapes (divide by w for 3x3)."""
    p = out_paths(out_dir)["transforms"]
    if not os.path.exists(p):
        return None, None
    try:
        with np.load(p) as d:
            samples = np.asarray(d["samples"])
            if "transforms_3x3" in d.files:
                H = np.asarray(d["transforms_3x3"], dtype=np.float64)
                if H.ndim == 3 and H.shape[1:] == (3, 3) and len(H) == len(samples):
                    return H, samples
                log.warning("%s: transforms_3x3 has shape %s for %d samples; using the 2x3 set",
                            p, H.shape, len(samples))
            return np.asarray(d["transforms"], dtype=np.float64), samples
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as e:
        log.warning("cannot read %s (%s); camera-motion correction is not applied to drawn points", p, e)
        return None, None


def source_from_result(video_path: str, result: AnalysisResult, par: Optional[float] = None,
                       index_cache: Optional[str] = None):
    """(reader, SampleSource) in exactly the sample space the results were made in.

    Uses the ScanInfo saved in results.json (mode, field order, frame step /
    phase, field repair) - no re-detection, so every saved sample index
    (release.s, track points, windows) keeps its meaning."""
    from pitchspeed.io.interlace import SampleSource
    from pitchspeed.io.reader import VideoReader
    if par is None:
        par = (result.params or {}).get("par_override")
    reader = VideoReader(video_path, sar_override=par, index_cache=index_cache)
    try:
        source = SampleSource(reader, result.scan)
    except Exception:
        reader.close()
        raise
    return reader, source


def scan_space(scan: ScanInfo) -> tuple:
    """What fixes the meaning of a sample index s."""
    if scan.mode == "interlaced":
        return ("interlaced", scan.field_order)
    ev = scan.evidence or {}
    return ("frames", int(ev.get("frame_step", 1) or 1), int(ev.get("frame_phase", 0) or 0),
            ev.get("repair") or None)


def _describe_space(sp: tuple) -> str:
    if sp[0] == "interlaced":
        return f"fields {str(sp[1]).upper()}"
    return "whole frames" + (f" (one of every {sp[1]})" if sp[1] != 1 else "") + (f", {sp[3]}" if sp[3] else "")


def _norm_json(x):
    return json.loads(json.dumps(to_jsonable(x), default=str))


SETTING_LABELS = {"distance_ft": "distance (ft)", "level": "level", "sport": "sport",
                  "extension_ft": "extension (ft)", "pitcher_throws": "throws", "altitude_ft": "altitude (ft)",
                  "temp_f": "temperature (F)", "sar": "pixel aspect", "release_height_ft": "release height (ft)",
                  "catch_behind_ft": "catch behind plate (ft)"}


def settings_changes(old: dict, new: dict) -> list[str]:
    """['distance (ft) 60.5 -> 46.0', ...] for the settings that differ."""
    old, new = _norm_json(old or {}), _norm_json(new or {})
    out = []
    for k in list(new) + [k for k in old if k not in new]:
        if old.get(k) != new.get(k):
            def f(v):
                return "not set" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))
            out.append(f"{SETTING_LABELS.get(k, k)} {f(old.get(k))} -> {f(new.get(k))}")
    return out


def calibration_differences(cf: CalibrationFile, result: AnalysisResult) -> tuple[list[str], bool]:
    """(settings changes, clicks/segments changed) between calibration.json and the
    calibration the results were made with (result.calibration)."""
    if not result.calibration:
        return [], False
    try:            # through the dataclasses, so fields added later compare as their defaults
        old = _norm_json(CalibrationFile.from_dict(copy.deepcopy(result.calibration)).to_dict())
    except Exception:                                            # noqa: BLE001 - compare as stored
        old = _norm_json(result.calibration)
    new = _norm_json(cf.to_dict())
    geom = (old.get("segments") != new.get("segments")
            or list(old.get("image_size") or []) != list(new.get("image_size") or []))
    return settings_changes(old.get("settings") or {}, new.get("settings") or {}), geom


@dataclass
class PreviousResults:
    """The results.json an analysis is about to replace (read before anything is written)."""
    result: Optional[AnalysisResult] = None
    backup: Optional[str] = None            # where it was copied (only when it held review work / was unreadable)
    unreadable: Optional[str] = None        # why it could not be read
    summary: dict = field(default_factory=dict)

    @classmethod
    def load(cls, paths: dict) -> Optional["PreviousResults"]:
        from pitchspeed.render.report import load_json
        path = paths["results_json"]
        if not os.path.exists(path):
            return None
        prev = cls()
        try:
            prev.result = load_json(path)
        except (OSError, ValueError) as e:
            prev.unreadable = str(e)
        if prev.result is not None:
            prev.summary = user_work_summary(prev.result)
            if not has_user_work(prev.result):
                return prev
        prev.backup = backup_results(path)
        tf = paths["transforms"]
        if os.path.exists(tf):          # the tracks in the backup belong to these transforms
            try:
                shutil.copy2(tf, os.path.splitext(prev.backup)[0] + ".transforms.npz")
            except OSError as e:
                log.warning("could not back up %s: %s", tf, e)
        return prev


@dataclass
class CarryReport:
    reason: Optional[str] = None        # why nothing was carried over
    pitches: int = 0
    rejected: int = 0
    windows: int = 0
    tracks: int = 0

    def text(self) -> str:
        parts = []
        if self.pitches:
            parts.append(f"{self.pitches} reviewed pitch{'es' if self.pitches != 1 else ''}")
        if self.rejected:
            parts.append(f"{self.rejected} rejection{'s' if self.rejected != 1 else ''}")
        if self.windows:
            parts.append(f"{self.windows} checked window{'s' if self.windows != 1 else ''}")
        if self.tracks:
            parts.append(f"{self.tracks} checked track{'s' if self.tracks != 1 else ''}")
        return ", ".join(parts) or "nothing"


def _flight(p: Pitch) -> tuple[float, float]:
    a = float(p.release.t) if p.release is not None and p.release.t is not None else float("nan")
    b = float(p.catch.t) if p.catch is not None and p.catch.t is not None else a
    return (min(a, b), max(a, b))


def _overlaps(a: tuple[float, float], b: tuple[float, float], frac: float = 0.5) -> bool:
    if not all(np.isfinite(v) for v in (*a, *b)):
        return False
    ov = min(a[1], b[1]) - max(a[0], b[0])
    return ov >= frac * min(a[1] - a[0], b[1] - b[0]) - 1e-6


def same_pitch(a: Pitch, b: Pitch) -> bool:
    """Do two pitches (from two analyses of one video) describe the same throw?"""
    fa, fb = _flight(a), _flight(b)
    return _overlaps(fa, fb) or (np.isfinite(fa[0]) and np.isfinite(fb[0]) and abs(fa[0] - fb[0]) <= 0.15)


def _carry_incompatibility(old: AnalysisResult, new: AnalysisResult, calib_file: CalibrationFile) -> Optional[str]:
    if int(old.video.n_frames) != int(new.video.n_frames):
        return f"the video now has {new.video.n_frames} frames instead of {old.video.n_frames}"
    a, b = scan_space(old.scan), scan_space(new.scan)
    if a != b:
        return (f"the analysis now uses {_describe_space(b)} instead of {_describe_space(a)}, so the "
                f"marked fields mean something else")

    def frames(segs):
        return [(int(s.get("start_frame", 0) or 0), s.get("end_frame"), int(s.get("ref_frame", 0) or 0))
                for s in segs]
    old_segs = (old.calibration or {}).get("segments") or []
    new_segs = _norm_json(calib_file.to_dict()).get("segments") or []
    if frames(old_segs) != frames(new_segs):
        return "the calibration segments or reference frames changed, so the ball positions no longer line up"
    return None


def carry_over_user_work(old: AnalysisResult, new: AnalysisResult, calib_file: CalibrationFile,
                         source) -> CarryReport:
    """Put the review work of `old` into the fresh analysis `new` (in place).

    Only done when both use the same video, sample space and calibration
    frames (else `reason` says why, and nothing is changed):
    * a rejected pitch -> the new automatic pitch(es) at the same time are
      rejected too (notes + history kept);
    * an accepted / annotated automatic pitch -> notes + history move to the
      new pitch at the same time (kept as it was if none was found);
    * an edited or manual pitch replaces the new automatic pitch(es) at the
      same time (those tracks are listed as rejected, code
      REPLACED_BY_REVIEW - nothing is dropped silently);
    * 'checked' marks on possible-missed windows and rejected tracks move
      to the new ones covering the same time.
    Carried pitches get an edit_history entry 'carried_over' and their speed
    is recomputed with the current calibration."""
    from pitchspeed.types import SpeedEstimate
    rep = CarryReport()
    why = _carry_incompatibility(old, new, calib_file)
    if why is not None:
        rep.reason = why
        return rep
    recompute = make_recompute(calib_file, new.scan, source, video=new.video,
                               undecodable=(new.params or {}).get("undecodable_frames"))
    stamp = _dt.datetime.now().isoformat(timespec="seconds")
    entry = {"action": "carried_over", "at": stamp,
             "note": "kept from the previous results when the video was analyzed again"}
    used: set[int] = set()
    added: list[Pitch] = []
    for u in sorted((p for p in old.pitches if is_user_pitch(p)), key=lambda p: _flight(p)[0]):
        matches = [p for p in new.pitches if id(p) not in used and p.status == "auto" and same_pitch(u, p)]
        used.update(id(p) for p in matches)
        hist = list(u.edit_history) + [dict(entry)]
        if matches and u.status in ("rejected", "auto"):
            for m in matches:
                if u.status == "rejected":
                    m.status = "rejected"
                m.notes = u.notes or m.notes
                m.edit_history = list(hist)
            if u.status == "rejected":
                rep.rejected += 1
            else:
                rep.pitches += 1
            continue
        keep = copy.deepcopy(u)
        keep.edit_history = hist
        try:
            recompute(keep)
        except Exception as e:                                   # noqa: BLE001 - shown, never hidden
            keep.speed = SpeedEstimate(v_release_mph=None, reason=f"speed could not be recomputed: {e}")
        for m in matches:
            new.pitches.remove(m)
            if m.track is not None:
                m.track.status, m.track.reject_code = "rejected", "REPLACED_BY_REVIEW"
                m.track.diagnostics = dict(m.track.diagnostics or {})
                m.track.diagnostics["review"] = "replaced_by_reviewed_pitch"
                new.rejected_tracks.append(m.track)
        added.append(keep)
        if keep.status == "rejected":
            rep.rejected += 1
        else:
            rep.pitches += 1
    new.pitches.extend(added)

    def span(d) -> tuple[float, float]:
        try:
            return (float(d.get("t_start")), float(d.get("t_end")))
        except (TypeError, ValueError):
            return (float("nan"), float("nan"))

    def best(a: tuple[float, float], cands: list, span_of) -> Optional[int]:
        """Index of the candidate overlapping `a` the most (None if none overlaps enough)."""
        scored = [(min(a[1], b[1]) - max(a[0], b[0]), i) for i, b in ((i, span_of(c)) for i, c in enumerate(cands))
                  if _overlaps(a, b)]
        return max(scored)[1] if scored else None

    free_w = [m for m in (new.possible_missed or []) if isinstance(m, dict) and not m.get("review")]
    for m_old in old.possible_missed or []:
        if isinstance(m_old, dict) and m_old.get("review"):
            j = best(span(m_old), free_w, span)
            if j is not None:
                free_w.pop(j)["review"] = m_old["review"]
                rep.windows += 1
    free_t = [t for t in (new.rejected_tracks or []) if t.points and not (t.diagnostics or {}).get("review")]
    for t_old in old.rejected_tracks or []:
        rv = (t_old.diagnostics or {}).get("review")
        if rv and t_old.points and not str(rv).startswith("replaced"):
            j = best((t_old.t_first, t_old.t_last), free_t, lambda t: (t.t_first, t.t_last))
            if j is not None:
                t_new = free_t.pop(j)
                t_new.diagnostics = dict(t_new.diagnostics or {})
                t_new.diagnostics["review"] = rv
                rep.tracks += 1
    renumber(new.pitches)
    return rep


def _finish_analysis(result: AnalysisResult, calib_file: CalibrationFile, source, out_dir: str,
                     prev: Optional[PreviousResults], show_progress: bool) -> None:
    """Carry review work over from the replaced results, then save."""
    notes = []
    if prev is not None and prev.unreadable:
        notes.append(f"the previous results.json could not be read ({prev.unreadable}); a copy was kept as "
                     f"{os.path.basename(prev.backup)}")
    elif prev is not None and prev.backup:
        rep = carry_over_user_work(prev.result, result, calib_file, source)
        txt = (f"the previous results had review work ({describe_user_work(prev.summary)}); they were backed "
               f"up as {os.path.basename(prev.backup)}")
        if rep.reason:
            txt += (f". That work was NOT carried over because {rep.reason}: redo it in review, or rename "
                    f"the backup to results.json to go back to the old results")
        else:
            txt += f"; carried over into these results: {rep.text()}"
        notes.append(txt)
    for n in reversed(notes):
        result.warnings.insert(0, n)
        log.info("%s", n)
        if show_progress:
            print(f"  note: {n}")
    save_results(result, out_dir)
