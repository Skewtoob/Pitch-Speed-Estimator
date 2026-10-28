"""Review window: check, fix, reject or add pitches by eye.

``run_review_ui(result, source, calib_file, recompute, save, transforms,
transform_samples)`` shows each pitch (and each "possible missed pitch"
window) field by field with its release (R) and catch (C) marks, the ball
track and a bottom strip with a timeline and the line

    Pitch 3/12 | auto | 00:01:23.457 | fields 4986-5013 | 78+/-3 mph (avg 74) | TOF | good | flags

All logic lives in the pure state machine `pitchspeed.review.state`
(keys, statuses, numbering, undo, autosave rules: see its docstring);
this module draws, maps mouse clicks to frame pixel coordinates, plays
back (Space, 1/8x..1x, looping around the pitch) and prefetches the fields
of the current pitch (release - 60 .. catch + 60) into RAM a few fields
per loop pass (OpenCV keeps only the last key pressed between polls, so
the loop must never block for long).

Every edit calls ``recompute(pitch)`` and then ``save(result)``; Q/Esc
saves once more and returns the (same, edited) AnalysisResult.  Closing
the window with its X button also saves and returns.

Extra keyword arguments (not in DESIGN.md's contract, all optional):
``retrack(pitch) -> PitchTrack`` for the E key (default: built from
``pitchspeed.vision.finder.track_between`` with the calibration's
reference frame), ``ui_scale``, ``window`` and ``max_iterations`` (tests).
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import cv2
import numpy as np

from pitchspeed.calib.model import CalibrationFile
from pitchspeed.review.state import ReviewState, StabilizeMap
from pitchspeed.types import AnalysisResult, Pitch, PitchTrack
from pitchspeed.ui import canvas as cv
from pitchspeed.ui.frames import FrameStore
from pitchspeed.ui.keys import normalize_key
from pitchspeed.ui.loupe import make_loupe, paste_loupe

log = logging.getLogger("pitchspeed")

WINDOW_NAME = "pitchspeed - review"
STRIP_H = 100
BAR_H = 16
POLL_MS = 30
PREFETCH_PER_POLL = 6        # samples decoded ahead per loop pass (keeps the key poll responsive)
PICK_RADIUS_DISPLAY_PX = 14.0
LEVEL_COLORS = {"info": cv.WHITE, "ok": cv.GREEN, "warn": cv.ORANGE, "error": cv.RED}
STATUS_COLORS = {"auto": cv.GREEN, "edited": cv.YELLOW, "manual": cv.CYAN, "rejected": cv.RED}


# --------------------------------------------------------------- retrack --
def make_default_retrack(source, calib_file: CalibrationFile, scan) -> Callable[[Pitch], PitchTrack]:
    """E key: re-run the tracker between the pitch's R and C (vision.finder.track_between)."""
    refs: dict[int, np.ndarray] = {}

    def ref_frame(i: int) -> np.ndarray:
        if i not in refs:
            reader = getattr(source, "reader", None)
            if reader is None:
                raise RuntimeError("the calibration reference frame cannot be read from this source")
            refs[i] = reader.get_frame(int(i), "gray")
        return refs[i]

    def retrack(pitch: Pitch) -> PitchTrack:
        from pitchspeed.vision.finder import track_between
        if pitch.release is None or pitch.catch is None:
            raise ValueError("set R and C first")
        r = pitch.release.s if pitch.release.s is not None else source.sample_at_time(pitch.release.t)
        c = pitch.catch.s if pitch.catch.s is not None else source.sample_at_time(pitch.catch.t)
        frame = source.locate(int(r))[0]
        calib = calib_file.for_frame(frame) or calib_file.segment_calibs()[0]
        manual = [p for p in (pitch.track.points if pitch.track else []) if p.manual]
        return track_between(source, calib, scan, int(r), int(c), ref_frame(calib.ref_frame),
                             manual_points=manual)

    return retrack


# ---------------------------------------------------------------- render --
def render(state: ReviewState, store: FrameStore, canvas: cv.Canvas,
           mouse_uv: Optional[tuple[int, int]] = None, busy: Optional[str] = None) -> np.ndarray:
    src = store.source
    interlaced = bool(getattr(src, "interlaced", False))
    frame = int(src.locate(state.s)[0])
    if state.view_mode == "frame" and interlaced:
        img = canvas.render_frame(store.frame_image(frame))
    else:
        img = canvas.render_sample(store.sample(state.s), is_field=interlaced)
    vw, vh = canvas.video_size
    if state.show_track:
        _draw_track(img, state, canvas)
    _draw_event_badge(img, state, vw)
    if state.loupe and mouse_uv is not None and canvas.in_video(*mouse_uv):
        pts = []
        for x, y, manual, inlier, here in state.overlay_points():
            if here:
                pts.append((canvas.to_display([(x, y)])[0], cv.YELLOW if manual else cv.GREEN))
        lp = make_loupe(img[:vh], mouse_uv, points=pts, label="8x")
        paste_loupe(img, lp, mouse_uv, video_h=vh)
    lines = [state.top_text()]
    if state.save_error:
        lines.append((f"AUTOSAVE FAILED: {state.save_error}", cv.RED))
    cv.draw_lines(img, lines, org=(8, 20), scale=0.45)
    _draw_strip(img, state, canvas)
    if state.help:
        cv.draw_help(img, "Review keys (H closes)", state.help_rows(),
                     footer="Edits are saved automatically. Rejected pitches are kept (status 'rejected').")
    if busy:
        cv.draw_lines(img, [(busy, cv.YELLOW)], org=(vw // 2 - 120, vh // 2), scale=0.6)
    return img


def _draw_track(img: np.ndarray, state: ReviewState, canvas: cv.Canvas) -> None:
    pts = state.overlay_points()
    if not pts:
        return
    good = [canvas.to_display([(x, y)])[0] for x, y, m, inl, _ in pts if inl]
    if len(good) >= 2:
        poly = np.round(np.array(good) * 16).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [poly], False, (0, 120, 0), 1, cv2.LINE_AA, 4)
    for x, y, manual, inlier, here in pts:
        uv = canvas.to_display([(x, y)])[0]
        if not inlier:
            cv.draw_marker(img, uv, cv.GRAY, size=4, style="cross")
            continue
        color = cv.YELLOW if manual else cv.GREEN
        if here:
            cv.draw_marker(img, uv, color, size=9, style="circle", thickness=1)
        else:
            cv.draw_marker(img, uv, color, size=4, style="dot")


def _draw_event_badge(img: np.ndarray, state: ReviewState, vw: int) -> None:
    mk = state.markers()
    tags = []
    if mk.get("draft") is not None and state.s == mk["draft"]:
        tags.append(("R (new pitch)", cv.YELLOW))
    if mk.get("R") is not None and state.s == mk["R"]:
        tags.append(("R = release", cv.GREEN))
    if mk.get("C") is not None and state.s == mk["C"]:
        tags.append(("C = catch", cv.RED))
    x = vw - 10
    for text, color in tags:
        w = cv.text_size(text, 0.7, 2)[0]
        x -= w
        cv.draw_text(img, text, (x, 60), 0.7, color, thickness=2)
        x -= 16


def _bar_geometry(canvas: cv.Canvas) -> tuple[int, int, int, int]:
    vw, vh = canvas.video_size
    return 10, vh + 6, vw - 10, vh + 6 + BAR_H


def bar_sample(state: ReviewState, canvas: cv.Canvas, u: float) -> int:
    """Sample under display x position u of the timeline bar."""
    x0, _, x1, _ = _bar_geometry(canvas)
    w0, w1 = state.window()
    f = (u - x0) / max(1.0, (x1 - x0))
    return int(round(w0 + min(1.0, max(0.0, f)) * (w1 - w0)))


def _draw_strip(img: np.ndarray, state: ReviewState, canvas: cv.Canvas) -> None:
    vw, vh = canvas.video_size
    x0, y0, x1, y1 = _bar_geometry(canvas)
    mk = state.markers()
    w0, w1 = mk["w0"], mk["w1"]
    cv2.rectangle(img, (x0, y0), (x1, y1), (70, 70, 70), -1)

    def xs(s):
        return int(round(x0 + (s - w0) / max(1, (w1 - w0)) * (x1 - x0)))

    span = mk.get("span")
    if span is not None:
        cv2.rectangle(img, (xs(span[0]), y0 + 3), (xs(span[1]), y1 - 3), (0, 140, 255), -1)
    if mk["R"] is not None and mk["C"] is not None:
        cv2.rectangle(img, (xs(mk["R"]), y0 + 5), (xs(mk["C"]), y1 - 5), (90, 90, 90), -1)
    for s, manual, inlier in mk["points"]:
        if w0 <= s <= w1:
            color = cv.YELLOW if manual else (cv.GREEN if inlier else cv.GRAY)
            cv2.circle(img, (xs(s), (y0 + y1) // 2), 2, color, -1)
    for key, color in (("R", cv.GREEN), ("C", cv.RED), ("draft", cv.YELLOW)):
        s = mk.get(key)
        if s is not None and w0 <= s <= w1:
            thick = 3 if (key == mk["selected"]) else 2
            cv2.line(img, (xs(s), y0 - 3), (xs(s), y1 + 3), color, thick)
            cv.draw_text(img, "R" if key == "draft" else key, (xs(s) - 4, y0 - 5), 0.4, color)
    cv2.line(img, (xs(state.s), y0 - 2), (xs(state.s), y1 + 2), cv.WHITE, 1)
    y = y1 + 18
    p = state.current_pitch()
    color = STATUS_COLORS.get(p.status, cv.WHITE) if p is not None else cv.WHITE
    for part in cv.wrap_text(state.info_text(), vw - 16, 0.45)[:2]:
        cv.draw_text(img, part, (8, y), 0.45, color)
        y += 18
    cv.draw_text(img, cv.wrap_text(state.hint_text(), vw - 16, 0.42)[0], (8, y), 0.42, cv.GRAY)
    y += 18
    if state.message is not None:
        cv.draw_text(img, cv.wrap_text(state.message.text, vw - 16, 0.42)[0], (8, y), 0.42,
                     LEVEL_COLORS.get(state.message.level, cv.WHITE))


# ------------------------------------------------------------------- run --
def run_review_ui(result: AnalysisResult, source, calib_file: CalibrationFile,
                  recompute: Callable[[Pitch], Pitch], save: Callable[[AnalysisResult], None],
                  transforms=None, transform_samples=None, *,
                  retrack: Optional[Callable[[Pitch], PitchTrack]] = None,
                  ui_scale: Optional[float] = None, window=None,
                  max_iterations: Optional[int] = None) -> AnalysisResult:
    """Interactive review (see module docstring).  Returns the edited result."""
    from pitchspeed.ui.gui import Window, require_gui

    if window is None:
        require_gui()
    info = result.video
    width = int(getattr(info, "width", 0) or source.sample_shape[1])
    height = int(getattr(info, "height", 0) or source.sample_shape[0] * (2 if source.interlaced else 1))
    sar = float(getattr(info, "sar", 1.0) or 1.0)
    store = FrameStore(source)
    stab = StabilizeMap(transforms, transform_samples)
    if retrack is None and calib_file is not None and calib_file.segments:
        retrack = make_default_retrack(source, calib_file, result.scan)
    state = ReviewState(result, source, recompute, save, retrack=retrack, stab=stab)
    canvas = cv.Canvas((width, height), sar, cv.video_size_for(cv.resolve_ui_scale(ui_scale)), strip_h=STRIP_H)
    state.pick_radius = PICK_RADIUS_DISPLAY_PX / canvas.px_per_row
    win = window if window is not None else Window(WINDOW_NAME)
    mouse_uv: Optional[tuple[int, int]] = None
    next_tick = time.monotonic()
    it = 0
    try:
        while not state.done:
            rng = state.prefetch_range()
            store.set_prefetch(*(rng if rng is not None else (None, None)))
            more = store.prefetch_step(PREFETCH_PER_POLL)
            if state.pending is not None:
                win.show(render(state, store, canvas, mouse_uv, busy="working, please wait ..."))
                win.wait_key(1)
                state.run_pending()
                continue
            win.show(render(state, store, canvas, mouse_uv))
            delay = 1 if more else POLL_MS
            if state.playing:
                delay = int(max(1, min(POLL_MS, (next_tick - time.monotonic()) * 1000)))
            code = win.wait_key(delay)
            if not win.is_open():
                state.finish()
                break
            name = normalize_key(code)
            if name is not None:
                state.on_key(name)
            for ev in win.mouse_events():
                mouse_uv = (ev.x, ev.y)
                _feed_mouse(state, canvas, ev.kind, ev.x, ev.y)
            if state.playing and time.monotonic() >= next_tick:
                state.tick()
                next_tick = time.monotonic() + float(source.sample_dt) / state.play_speed
            elif not state.playing:
                next_tick = time.monotonic()
            it += 1
            if max_iterations is not None and it >= max_iterations:
                state.finish()
                break
    finally:
        win.close()
    for e in store.errors[:5]:
        log.warning("review: %s", e)
    return result


def _feed_mouse(state: ReviewState, canvas: cv.Canvas, kind: str, u: int, v: int) -> None:
    if kind not in ("ldown", "rdown"):
        return
    x0, y0, x1, y1 = _bar_geometry(canvas)
    if kind == "ldown" and x0 - 4 <= u <= x1 + 4 and y0 - 8 <= v <= y1 + 4:
        state.goto(bar_sample(state, canvas, u))
        return
    if not canvas.in_video(u, v):
        return
    x, y = canvas.to_frame([(u, v)])[0]
    if kind == "ldown":
        state.on_click(float(x), float(y))
    else:
        state.on_right_click(float(x), float(y))
