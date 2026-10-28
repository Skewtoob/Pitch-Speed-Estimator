"""One short video per pitch.

    render_clips(result, source, out_dir, ...) -> info dict

writes into ``<out_dir>/clips/``:

* ``P03_81mph.mp4`` - the pitch with the overlay (radar-gun card, ball path;
  the same drawing as annotated.mp4 without the running list), from
  PRE_S before the release to POST_S after the catch, with its sound;
* ``originals/P03.<ext>`` - the same stretch of the ORIGINAL file, copied
  without re-encoding (ffmpeg stream copy: same picture quality, both fields,
  same container as the tape capture), for editing.  A stream copy can only
  start at a key frame and seeking in MPEG files is not exact, so the copy
  starts SEEK_MARGIN_S (1 s) before the overlay clip, plus up to one key-frame
  interval (~0.5 s for MPEG-2 / H.264; DV has a key frame on every frame).
  If the container cannot take the copied streams the copy is written as .mkv.

NAMES.  ``P<number>_<speed>``: ``P03_81mph``, ``P02_66mph_rough`` (label
'poor'), ``P05_no_speed`` (not measurable); numbers are zero-padded to the
widest one so the files sort in pitch order.  Rejected pitches get no clip.

RE-RENDERING.  ``clips/clips.json`` lists the files written last time.  New
files are written first; then the files of the previous list that are not
written again (e.g. a pitch whose speed changed name after review edits) are
deleted.  Nothing else in the folder is ever touched, so the user's own
files there are safe.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any, Callable, Optional

from pitchspeed.render.overlay import (FFMPEG_TIMEOUT_S, POST_S, PRE_S, _NO_WINDOW, _ffmpeg_exe, _source_path,
                                       _to_sample_fn, alternative_path, build_marks, can_write, render_annotated)
from pitchspeed.render.text import has_number, round_half_up
from pitchspeed.types import AnalysisResult, SpeedEstimate

log = logging.getLogger(__name__)

__all__ = ["render_clips", "clip_name", "CLIPS_DIR", "ORIGINALS_DIR", "MANIFEST"]

CLIPS_DIR = "clips"
SEEK_MARGIN_S = 1.0          # originals start this much earlier (key frames / imprecise MPEG seeking)
ORIGINALS_DIR = "originals"
MANIFEST = "clips.json"


def clip_name(number: int, sp: Optional[SpeedEstimate], width: int = 2) -> str:
    """'P03_81mph', 'P02_66mph_rough' or 'P05_no_speed' (no extension)."""
    head = f"P{int(number):0{max(1, width)}d}"
    if not has_number(sp):
        return f"{head}_no_speed"
    v = round_half_up(sp.v_release_mph)
    return f"{head}_{v}mph" + ("_rough" if sp.label == "poor" else "")


def _read_manifest(path: str) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return []
    files = d.get("files", []) if isinstance(d, dict) else []
    out = []
    for f in files:                      # only plain relative names inside the clips folder
        if isinstance(f, str) and f and not os.path.isabs(f) and ".." not in f.replace("\\", "/").split("/"):
            out.append(f.replace("\\", "/"))
    return out


def _write_manifest(path: str, files: list[str]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"files": files, "note": "files written by pitchspeed; re-rendering replaces them"}, f, indent=2)
    os.replace(tmp, path)


def cut_original(src: str, t_start: float, duration: float, out_path: str,
                 ffmpeg_exe: Optional[str] = None) -> tuple[Optional[str], str]:
    """Copy [t_start, t_start + duration] (seconds from the first video frame) of `src` without
    re-encoding.  Returns (path written or None, status text)."""
    from pitchspeed.io.writer import source_audio_offset
    exe = ffmpeg_exe or _ffmpeg_exe()
    if not exe:
        return None, "skipped: no ffmpeg executable"
    try:
        _, offset = source_audio_offset(src)      # first video frame - container start
    except Exception as e:                        # noqa: BLE001
        return None, f"skipped: cannot read {os.path.basename(src)}: {e}"
    # Seek SEEK_MARGIN_S early: a stream copy starts at a key frame, and seeking in MPEG program
    # streams can land a few frames after the target, which would cut into the pitch.
    lead = min(SEEK_MARGIN_S, max(0.0, offset + float(t_start)))
    ss = max(0.0, offset + float(t_start)) - lead
    duration = float(duration) + lead
    root, ext = os.path.splitext(out_path)
    err = ""
    for target in (out_path, root + ".mkv") if ext.lower() != ".mkv" else (out_path,):
        if not can_write(target):
            target = alternative_path(target)
        tmp = os.path.splitext(target)[0] + ".part" + os.path.splitext(target)[1]
        cmd = [exe, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{ss:.6f}", "-i", src,
               "-t", f"{max(0.1, float(duration)):.6f}", "-map", "0:v:0", "-map", "0:a?", "-c", "copy",
               "-avoid_negative_ts", "make_zero", tmp]
        try:
            p = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT_S, creationflags=_NO_WINDOW)
        except (OSError, subprocess.SubprocessError) as e:
            err = str(e)
            continue
        if p.returncode == 0 and os.path.isfile(tmp) and os.path.getsize(tmp) > 0:
            try:
                os.replace(tmp, target)
            except OSError as e:
                err = f"could not replace {os.path.basename(target)}: {e}"
                continue
            return target, "copied" if target == out_path else f"copied as {os.path.basename(target)}"
        err = p.stderr.decode("utf-8", "replace").strip()[-400:]
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return None, f"failed: {err}"


def render_clips(result: AnalysisResult, source, out_dir: str, transforms=None, transform_samples=None,
                 size=(960, 720), audio: bool = True, progress=None, *, style: str = "radar",
                 originals: bool = True, writer_factory: Optional[Callable[..., Any]] = None,
                 info: Optional[dict] = None) -> list[str]:
    """Write one overlay clip (and one untouched original) per pitch; returns the overlay clips.

    info (optional dict) gets: 'clips', 'originals' (paths), 'notes' (problems, one per pitch at
    most), 'removed' (stale files of the previous render), 'style'.
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
    clips: list[str] = []
    origs: list[str] = []
    notes: list[str] = []
    used_style = style
    done = 0

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
            written.append(os.path.relpath(out, folder).replace("\\", "/"))
            used_style = clip_info.get("style", used_style)
            if clip_info.get("note"):
                notes.append(f"pitch {m.number}: {clip_info['note']}")
        if src_path is not None:
            os.makedirs(orig_folder, exist_ok=True)
            ext = os.path.splitext(src_path)[1] or ".mkv"
            t0 = float(source.sample_time(s0))
            dur = float(source.sample_time(s1 - 1)) - t0 + sample_dt
            path, status = cut_original(src_path, t0, dur, os.path.join(orig_folder, f"P{m.number:0{width}d}{ext}"))
            if path is not None:
                origs.append(path)
                written.append(os.path.relpath(path, folder).replace("\\", "/"))
                if status != "copied":
                    notes.append(f"pitch {m.number}: original {status}")
            else:
                notes.append(f"pitch {m.number}: untouched copy {status}")
        done += s1 - s0
        sub_progress(0)
    removed = []
    keep = set(written)
    for rel in previous:
        if rel in keep:
            continue
        p = os.path.join(folder, *rel.split("/"))
        if os.path.isfile(p):
            try:
                os.remove(p)
                removed.append(rel)
            except OSError as e:
                notes.append(f"an old clip could not be removed ({rel}: {e}); it is out of date")
    try:
        _write_manifest(manifest, written)
    except OSError as e:
        notes.append(f"{MANIFEST} could not be written ({e}); old clips may not be cleaned up next time")
    for t in notes:
        log.warning("clips: %s", t)
    if info is not None:
        info.update({"clips": clips, "originals": origs, "notes": notes, "removed": removed,
                     "style": used_style, "folder": folder})
    return clips
