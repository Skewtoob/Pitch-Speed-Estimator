"""One short video per pitch.

    render_clips(result, source, out_dir, ...) -> list of overlay clips

writes into ``<out_dir>/clips/``:

* ``P03_81mph.mp4`` - the pitch with the overlay (radar-gun card, ball path;
  the same drawing as annotated.mp4 without the running list), from
  PRE_S before the release to POST_S after the catch, with its sound;
* ``originals/P03.<ext>`` - the same stretch of the ORIGINAL file, copied
  without re-encoding (ffmpeg stream copy: same picture quality, both fields,
  same container as the tape capture), for editing.

UNTOUCHED COPIES.  A stream copy can only start at a key frame, and seeking
in MPEG program streams can land after the requested point, so each copy is
CHECKED: it must hold video packets covering at least the whole overlay
clip (its span is measured by demuxing, no decoding).  The first attempt
starts LEADS_S[0] (1 s) early; if the copy came out short (key frames far
apart) it is cut again starting earlier (3 s, then 10 s).  A copy that still
does not cover the pitch, or holds no picture, is deleted and reported - never
kept as if it were fine.  If the container refuses the copied streams the
copy is made as .mkv.

NAMES.  ``P<number>_<speed>``: ``P03_81mph``, ``P02_66mph_rough`` (label
'poor'), ``P05_no_speed`` (not measurable); numbers are zero-padded to the
widest one so the files sort in pitch order.  Rejected pitches get no clip.

RE-RENDERING.  ``clips/clips.json`` lists every file this module wrote that
is still there; it is updated after each new file, so an interrupted run
loses track of nothing.  After a run, a listed file that was not written
again is deleted (e.g. a pitch whose speed, and so its name, changed) -
except:
  * files of a kind not made this time (``originals=False`` keeps the
    existing originals);
  * the previous file of a pitch whose new file failed this time (kept and
    reported, so a failed render never leaves the pitch with nothing);
  * a file that cannot be deleted (open in a player): it stays listed and is
    deleted next time.
Nothing else in the folder is ever touched, so the user's own files there are
safe.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from typing import Any, Callable, Optional

from pitchspeed.render.overlay import (FFMPEG_TIMEOUT_S, POST_S, PRE_S, _NO_WINDOW, _ffmpeg_exe, _source_path,
                                       _to_sample_fn, alternative_path, build_marks, can_write, render_annotated)
from pitchspeed.render.text import has_number, round_half_up
from pitchspeed.types import AnalysisResult, SpeedEstimate

log = logging.getLogger(__name__)

__all__ = ["render_clips", "clip_name", "cut_original", "video_span", "CLIPS_DIR", "ORIGINALS_DIR", "MANIFEST"]

CLIPS_DIR = "clips"
ORIGINALS_DIR = "originals"
MANIFEST = "clips.json"
LEADS_S = (1.0, 3.0, 10.0)   # untouched copies start this much earlier (tried in turn, see module doc)
COVER_TOL_S = 0.1            # a copy may end this much short of the overlay clip's end
_NUM = re.compile(r"^(?:originals/)?P(\d+)")


def clip_name(number: int, sp: Optional[SpeedEstimate], width: int = 2) -> str:
    """'P03_81mph', 'P02_66mph_rough' or 'P05_no_speed' (no extension)."""
    head = f"P{int(number):0{max(1, width)}d}"
    if not has_number(sp):
        return f"{head}_no_speed"
    v = round_half_up(sp.v_release_mph)
    return f"{head}_{v}mph" + ("_rough" if sp.label == "poor" else "")


# ------------------------------------------------------------- manifest
def _read_manifest(path: str) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return []
    files = d.get("files", []) if isinstance(d, dict) else []
    out = []
    for f in files:                      # only plain relative names inside the clips folder
        if isinstance(f, str) and f and not os.path.isabs(f) and ":" not in f:
            f = f.replace("\\", "/")
            if ".." not in f.split("/") and f not in out:
                out.append(f)
    return out


def _write_manifest(path: str, files: list[str]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"files": files, "note": "files written by pitchspeed; making the clips again replaces them"},
                  f, indent=2)
    os.replace(tmp, path)


# ------------------------------------------------------- untouched copy
def video_span(path: str) -> Optional[tuple[float, int]]:
    """(seconds covered by the video packets, number of video packets) of a file - demuxed, not
    decoded; None if it cannot be read or has no video stream."""
    import av
    try:
        with av.open(os.fspath(path)) as c:
            if not c.streams.video:
                return None
            v = c.streams.video[0]
            rate = float(v.average_rate) if v.average_rate else 0.0
            ts = []
            for pk in c.demux(v):
                t = pk.pts if pk.pts is not None else pk.dts
                if t is not None and pk.size:
                    ts.append(float(t * pk.time_base))
    except Exception:                                    # noqa: BLE001 - unreadable = not a usable copy
        return None
    if not ts:
        return 0.0, 0
    return max(ts) - min(ts) + (1.0 / rate if rate > 0 else 0.0), len(ts)


def _copy(exe: str, src: str, ss: float, duration: float, target: str) -> tuple[Optional[str], str]:
    """One ffmpeg stream copy into `target` (another free name if it is locked)."""
    if not can_write(target):
        target = alternative_path(target)
    tmp = os.path.splitext(target)[0] + ".part" + os.path.splitext(target)[1]
    cmd = [exe, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{ss:.6f}", "-i", src,
           "-t", f"{max(0.1, duration):.6f}", "-map", "0:v:0", "-map", "0:a?", "-c", "copy",
           "-avoid_negative_ts", "make_zero", tmp]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT_S, creationflags=_NO_WINDOW)
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"ffmpeg could not run ({e})"
    ok = p.returncode == 0 and os.path.isfile(tmp) and os.path.getsize(tmp) > 0
    if not ok:
        err = p.stderr.decode("utf-8", "replace").strip()[-300:]
        _remove(tmp)
        return None, f"ffmpeg exit code {p.returncode}" + (f": {err}" if err else "")
    try:
        os.replace(tmp, target)
    except OSError as e:
        _remove(tmp)
        return None, f"could not write {os.path.basename(target)}: {e}"
    return target, ""


def _remove(path: Optional[str]) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def cut_original(src: str, t_start: float, duration: float, out_path: str,
                 ffmpeg_exe: Optional[str] = None) -> tuple[Optional[str], str]:
    """Copy at least [t_start, t_start + duration] (seconds from the first video frame) of `src`
    without re-encoding, checked as described in the module doc.  Returns (path, status) or
    (None, reason)."""
    from pitchspeed.io.writer import source_audio_offset
    exe = ffmpeg_exe or _ffmpeg_exe()
    if not exe:
        return None, "skipped: no ffmpeg executable"
    try:
        _, offset = source_audio_offset(src)          # first video frame - container start
    except Exception as e:                            # noqa: BLE001
        return None, f"skipped: cannot read {os.path.basename(src)}: {e}"
    want = max(0.0, offset + float(t_start))          # file time where the overlay clip starts
    root, ext = os.path.splitext(out_path)
    targets = [out_path] if ext.lower() == ".mkv" else [out_path, root + ".mkv"]
    problem = "nothing was copied"
    for target in targets:
        refused = False
        for lead in LEADS_S:
            lead = min(lead, want)
            path, err = _copy(exe, src, want - lead, float(duration) + lead, target)
            if path is None:
                problem, refused = err, True          # this container did not work: try the next one
                break
            span = video_span(path)
            if span is not None and span[1] > 0 and span[0] >= float(duration) - COVER_TOL_S:
                return path, "copied" if path == out_path else f"copied as {os.path.basename(path)}"
            _remove(path)                             # never keep a copy that misses the pitch
            problem = ("the copy held no picture" if span is None or span[1] == 0 else
                       "the copy started after the pitch (the file's key frames are far apart)")
            if lead >= want:                          # cannot start any earlier
                break
        if not refused:
            break
    return None, f"failed: {problem}"


# --------------------------------------------------------------- clips
def _kind(rel: str) -> str:
    return "orig" if rel.startswith(ORIGINALS_DIR + "/") else "clip"


def _number(rel: str) -> Optional[int]:
    m = _NUM.match(rel)
    return int(m.group(1)) if m else None


def render_clips(result: AnalysisResult, source, out_dir: str, transforms=None, transform_samples=None,
                 size=(960, 720), audio: bool = True, progress=None, *, style: str = "radar",
                 originals: bool = True, writer_factory: Optional[Callable[..., Any]] = None,
                 info: Optional[dict] = None) -> list[str]:
    """Write one overlay clip (and one untouched original) per pitch; returns the overlay clips.

    info (optional dict) gets: 'clips', 'originals' (paths), 'notes' (problems), 'removed'
    (out-of-date files deleted), 'kept' (old files kept, see module doc), 'style', 'folder'.
    Raises ValueError when there is no pitch to cut."""
    sample_dt = float(source.sample_dt)
    n = int(source.n_samples)
    marks = build_marks(result, _to_sample_fn(source))
    if not marks:
        raise ValueError("no accepted pitches, so there are no clips to write")
    folder = os.path.join(out_dir, CLIPS_DIR)
    orig_folder = os.path.join(folder, ORIGINALS_DIR)
    os.makedirs(folder, exist_ok=True)
    manifest = os.path.join(folder, MANIFEST)
    previous = _read_manifest(manifest)
    listed = list(previous)                     # previous + everything written so far (kept on disk)
    pre, post = int(round(PRE_S / sample_dt)), int(round(POST_S / sample_dt))
    width = max(2, len(str(max(m.number for m in marks))))
    jobs = []
    for m in marks:
        s0, s1 = max(0, m.s_release - pre), min(n, m.s_catch + post + 1)
        if s1 > s0:
            jobs.append((m, s0, s1))
    total = sum(s1 - s0 for _, s0, s1 in jobs)
    src_path = _source_path(source, result) if originals else None
    written: list[str] = []
    failed: set = set()                         # (kind, pitch number) whose new file failed
    clips: list[str] = []
    origs: list[str] = []
    notes: list[str] = []
    used_style = style
    done = 0

    def record(path: str) -> None:
        rel = os.path.relpath(path, folder).replace("\\", "/")
        written.append(rel)
        if rel not in listed:
            listed.append(rel)
            try:
                _write_manifest(manifest, listed)
            except OSError as e:
                log.warning("clips: %s could not be updated: %s", MANIFEST, e)

    def sub_progress(d, _t=None):
        if progress is None:
            return
        if hasattr(progress, "update"):
            progress.update(done + int(d))
        else:
            progress(done + int(d), total)

    if originals and src_path is None:
        notes.append("the original video file was not found, so no untouched copies were made")
    for m, s0, s1 in jobs:
        name = clip_name(m.number, m.speed, width)
        clip_info: dict = {}
        try:
            out = render_annotated(result, source, os.path.join(folder, name + ".mp4"), mode="highlights",
                                   transforms=transforms, transform_samples=transform_samples, size=size,
                                   audio=audio, progress=sub_progress, writer_factory=writer_factory,
                                   info=clip_info, style=style, windows=[(s0, s1)], show_list=False)
        except (OSError, RuntimeError, ValueError) as e:
            notes.append(f"pitch {m.number}: the clip could not be written ({e})")
            out = None
        if out:
            clips.append(out)
            record(out)
            used_style = clip_info.get("style", used_style)
            if clip_info.get("note"):
                notes.append(f"pitch {m.number}: {clip_info['note']}")
        else:
            failed.add(("clip", m.number))
        if src_path is not None:
            os.makedirs(orig_folder, exist_ok=True)
            ext = os.path.splitext(src_path)[1] or ".mkv"
            t0 = float(source.sample_time(s0))
            dur = float(source.sample_time(s1 - 1)) - t0 + sample_dt
            path, status = cut_original(src_path, t0, dur, os.path.join(orig_folder, f"P{m.number:0{width}d}{ext}"))
            if path is not None:
                origs.append(path)
                record(path)
                if status != "copied":
                    notes.append(f"pitch {m.number}: untouched copy {status}")
            else:
                failed.add(("orig", m.number))
                notes.append(f"pitch {m.number}: untouched copy {status}")
        done += s1 - s0
        sub_progress(0)

    made = {"clip"} | ({"orig"} if src_path is not None else set())
    removed, kept = [], []
    for rel in previous:
        if rel in written:
            continue
        p = os.path.join(folder, *rel.split("/"))
        if not os.path.isfile(p):
            continue                                       # already gone: drop it from the list
        if _kind(rel) not in made or (_kind(rel), _number(rel)) in failed:
            kept.append(rel)                               # not remade this time: keep it
            if _kind(rel) in made:
                notes.append(f"the previous {rel} was kept because the new one could not be made")
            continue
        try:
            os.remove(p)
            removed.append(rel)
        except OSError as e:
            kept.append(rel)                               # e.g. open in a player: deleted next time
            notes.append(f"the out-of-date {rel} could not be removed ({e}); it is deleted the next time")
    try:
        _write_manifest(manifest, written + [k for k in kept if k not in written])
    except OSError as e:
        notes.append(f"{MANIFEST} could not be written ({e}); old clips may not be cleaned up next time")
    for t in notes:
        log.warning("clips: %s", t)
    if info is not None:
        info.update({"clips": clips, "originals": origs, "notes": notes, "removed": removed, "kept": kept,
                     "style": used_style, "folder": folder})
    return clips
