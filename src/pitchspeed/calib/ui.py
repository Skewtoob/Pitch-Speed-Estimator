"""Calibration window: click the field landmarks once per camera setup.

``run_calibration_ui(source, video, view, initial, settings, start_frame)``
opens a window showing the video (default: the bobbed field, i.e. what the
detector sees; F = full frame), with a trackbar for coarse scrubbing and
an info strip under the picture that names the next point and says
exactly what to click.  All logic (points, undo, segments, masks,
validation) lives in the pure state machine `pitchspeed.calib.state`;
this module only draws, converts mouse clicks from display to frame pixel
coordinates (`ui.canvas.Canvas.to_frame`) and feeds keys / mouse events.

Returns the CalibrationFile (settings + segments, ``video`` fingerprint
left for the caller to fill) or None if the user cancelled (Esc twice, or
closed the window and chose not to save).  Raises
`pitchspeed.ui.gui.GuiUnavailable` with a fix-it message when no window
can be opened (no DISPLAY on Linux, opencv-python-headless).

CLOSING THE WINDOW (X button) never throws clicked points away silently.
OpenCV cannot keep the window from closing, so the question is asked in
the console the program runs in (where every other question is asked):

* nothing changed since the window opened: nothing to save (None);
* all required points placed: the check that Enter does runs, its
  warnings are printed, then "Save these points? [Y/n/e]" - y saves
  (warnings recorded in the notes, as with Enter), n discards, e re-opens
  the window to keep editing.  The default is to save (closing is the
  usual way to finish on Windows), except after a first Esc (the user
  was quitting without saving): then the default is n;
* required points missing (cannot be saved yet): "e" re-opens the window
  (default), "n" quits without saving.
* no console to ask (stdin closed): save if the points can be saved and
  Esc was not pressed, else nothing is saved; either way it is printed.

Extra keyword arguments (not in DESIGN.md's contract, all optional):
``ui_scale`` (window size factor; default $PITCHSPEED_UI_SCALE or 1),
``window`` (an object with the `ui.gui.Window` interface, for tests; it is
re-opened through its ``reopen()`` method if it has one), ``ask``
(``prompt -> answer or None``, the console question, for tests) and
``max_iterations`` (stop the event loop after so many polls, for tests).
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

from pitchspeed.calib.model import CalibrationFile, Settings
from pitchspeed.calib.state import CalibState, estimate_shift
from pitchspeed.types import VideoInfo
from pitchspeed.ui import canvas as cv
from pitchspeed.ui.fmt import format_time
from pitchspeed.ui.frames import FrameStore
from pitchspeed.ui.keys import normalize_key
from pitchspeed.ui.loupe import make_loupe, paste_loupe

log = logging.getLogger("pitchspeed")

WINDOW_NAME = "pitchspeed - calibrate"
STRIP_H = 92
POLL_MS = 30
PICK_RADIUS_DISPLAY_PX = 14.0

LEVEL_COLORS = {"info": cv.WHITE, "ok": cv.GREEN, "warn": cv.ORANGE, "error": cv.RED}
MARKER_COLORS = {"selected": cv.YELLOW, "normal": cv.GREEN, "other_frame": cv.CYAN}


# ------------------------------------------------------------------ render --
def render(state: CalibState, store: FrameStore, canvas: cv.Canvas,
           mouse_uv: Optional[tuple[int, int]] = None) -> np.ndarray:
    """The full window image for the current state."""
    src = store.source
    interlaced = bool(getattr(src, "interlaced", False))
    if state.view_mode == "frame" and interlaced:
        img = canvas.render_frame(store.frame_image(state.frame))
    else:
        img = canvas.render_sample(store.sample(state.s), is_field=interlaced)
    vw, vh = canvas.video_size
    _draw_overlays(img, state, canvas)
    if state.loupe:
        center = _loupe_center(state, canvas, mouse_uv)
        if center is not None:
            pts = [(canvas.to_display([p])[0], MARKER_COLORS[k]) for _, p, k in state.markers()]
            lp = make_loupe(img[:vh], center, points=pts, label="8x")
            paste_loupe(img, lp, center, video_h=vh)
    _draw_top_line(img, state)
    _draw_strip(img, state, canvas)
    if state.confirm:
        _draw_confirm(img, state.confirm, vw, vh)
    if state.help:
        cv.draw_help(img, "Calibration keys (H closes)", state.help_rows(),
                     footer="Clicks are stored in frame pixels of the stored video; the display is square-pixel.")
    return img


def _loupe_center(state: CalibState, canvas: cv.Canvas, mouse_uv) -> Optional[tuple[float, float]]:
    if getattr(state, "last_action", "").startswith("nudge") and state.selected in state.seg.points:
        return tuple(canvas.to_display([state.seg.points[state.selected]])[0])
    if mouse_uv is not None and canvas.in_video(*mouse_uv):
        return float(mouse_uv[0]), float(mouse_uv[1])
    return None


def _draw_overlays(img: np.ndarray, state: CalibState, canvas: cv.Canvas) -> None:
    for m in state.seg.masks:
        p0, p1 = canvas.to_display([(m[0], m[1]), (m[2], m[3])])
        cv.draw_rect(img, p0, p1, cv.MAGENTA, label="mask")
    if state.drag is not None:
        d = state.drag
        p0, p1 = canvas.to_display([(d[0], d[1]), (d[2], d[3])])
        cv.draw_rect(img, p0, p1, cv.YELLOW)
    for name, p, kind in state.markers():
        uv = canvas.to_display([p])[0]
        label = name + ("*" if kind == "other_frame" else "")
        cv.draw_marker(img, uv, MARKER_COLORS[kind], label=label,
                       style="circle" if kind == "selected" else "cross", size=8)
        if kind == "selected":
            cv.draw_marker(img, uv, MARKER_COLORS[kind], size=8)


def _draw_top_line(img: np.ndarray, state: CalibState) -> None:
    tl = state.tl
    frame, k, parity = tl.locate(state.s)
    if getattr(tl, "interlaced", False):
        where = ("full frame" if state.view_mode == "frame"
                 else f"field {k + 1}/2 ({'top' if parity == 0 else 'bottom'})")
    else:
        where = "frame"
    nseg = len(state.segments)
    seg = f" | segment {state.seg_index + 1}/{nseg} (from frame {state.seg.start_frame})" if nseg > 1 else ""
    mode = " | MASK MODE" if state.mask_mode else ""
    text = (f"{state.view.upper()} view | frame {frame}/{state.n_frames - 1} {where} | "
            f"{format_time(tl.sample_time(state.s))}{seg}{mode} | H = help")
    cv.draw_lines(img, [text], org=(8, 20), scale=0.45)


def _draw_strip(img: np.ndarray, state: CalibState, canvas: cv.Canvas) -> None:
    vw, vh = canvas.video_size
    scale = 0.45
    y = vh + 18
    for part in cv.wrap_text(state.target_text(), vw - 16, scale)[:2]:
        cv.draw_text(img, part, (8, y), scale, cv.YELLOW)
        y += 18
    hint = state.hint_text()
    if hint:
        cv.draw_text(img, cv.wrap_text(hint, vw - 16, 0.42)[0], (8, y), 0.42, cv.GRAY)
        y += 18
    x = 8
    for text, level in state.status_items():
        cv.draw_text(img, text, (x, y), 0.42, LEVEL_COLORS.get(level, cv.WHITE))
        x += cv.text_size(text, 0.42)[0] + 12
        if x > vw - 40:
            break
    y += 18
    if state.message is not None and y < img.shape[0]:
        cv.draw_text(img, cv.wrap_text(state.message.text, vw - 16, 0.42)[0], (8, y), 0.42,
                     LEVEL_COLORS.get(state.message.level, cv.WHITE))


def _draw_confirm(img: np.ndarray, warns: list[str], vw: int, vh: int) -> None:
    lines = [("Please check before saving:", cv.YELLOW)] + [("- " + w, cv.ORANGE) for w in warns]
    lines.append(("Enter = save anyway   |   any other key = keep editing", cv.WHITE))
    cv.draw_lines(img, lines, org=(24, 70), scale=0.45, max_width=vw - 60)


# -------------------------------------------------------------------- run --
def _resolve_sar(video: VideoInfo, settings: Optional[Settings]) -> float:
    sar = float(video.sar or 1.0)
    if settings is not None and settings.sar and abs(float(settings.sar) - sar) > 1e-3:
        log.warning("calibration settings pixel aspect %.4f differs from the video's %.4f; "
                    "showing the video with %.4f", settings.sar, sar, sar)
    return sar


def console_ask(prompt: str) -> Optional[str]:
    """One answer typed in the console, or None when there is no console to ask."""
    try:
        return input(prompt)
    except (EOFError, OSError, RuntimeError):
        return None


def _answer(ask: Callable[[str], Optional[str]], prompt: str, choices: str, default: str) -> Optional[str]:
    """First letter of a valid answer ('' = default); None if nobody can answer."""
    for _ in range(5):
        a = ask(prompt)
        if a is None:
            return None
        a = a.strip().lower()[:1] or default
        if a in choices:
            return a
        print(f"  please type one of: {', '.join(choices)}")
    return default


def after_close(state: CalibState, ask: Callable[[str], Optional[str]] = console_ask,
                allow_edit: bool = True) -> str:
    """What to do after the user closed the window: 'saved' | 'discard' | 'edit' (see module doc).

    allow_edit=False (the window could not be re-opened): only save / discard."""
    if not state.unsaved_changes():
        print("Calibration window closed: nothing was changed, nothing to save.")
        return "discard"
    print("\nThe calibration window was closed before saving.")
    print(f"  Points clicked: {state.placed_summary()}")
    try:
        warns = state.check()
    except ValueError as e:
        print(f"  These points cannot be saved yet: {e}")
        if not allow_edit:
            print("Nothing saved.")
            return "discard"
        a = _answer(ask, "  Type e (or just Enter) to go back to the window and finish, "
                         "or n to quit WITHOUT saving: ", "en", "e")
        if a == "e":
            return "edit"
        print("Nothing saved." if a == "n" else "No answer possible (no console): nothing saved.")
        return "discard"
    if warns:
        print("  Please check before saving:")
        for w in warns:
            print(f"   - {w}")
    default = "n" if state.cancel_armed else "y"
    opts = ("[Y/n" if default == "y" else "[y/N") + ("/e]" if allow_edit else "]")
    back = ", e = back to the window" if allow_edit else ""
    a = _answer(ask, f"  Save these points? y = save, n = throw them away{back} {opts}: ",
                "yne" if allow_edit else "yn", default)
    if a is None:
        if default == "n":
            print("No answer possible (no console) and Esc had been pressed: nothing saved.")
            return "discard"
        print("No answer possible (no console): saving the points (all required points are placed).")
        a = "y"
    if a == "e":
        return "edit"
    if a == "n":
        print("Nothing saved.")
        return "discard"
    state.save_now()
    return "saved"


def _reopen(window, factory):
    if window is not None:
        fn = getattr(window, "reopen", None)
        return fn() if callable(fn) else None
    return factory()


def run_calibration_ui(source, video: VideoInfo, view: str, initial: Optional[CalibrationFile] = None,
                       settings: Optional[Settings] = None, start_frame: int = 0, *,
                       ui_scale: Optional[float] = None, window=None,
                       ask: Optional[Callable[[str], Optional[str]]] = None,
                       max_iterations: Optional[int] = None) -> Optional[CalibrationFile]:
    """Interactive calibration (see module docstring).  Returns None if cancelled."""
    from pitchspeed.ui.gui import Window, require_gui

    if window is None:
        require_gui()
    sar = _resolve_sar(video, settings)
    store = FrameStore(source)

    def measure(a: int, b: int):
        return estimate_shift(store.frame_gray(a), store.frame_gray(b))

    state = CalibState(source, view, (int(video.width), int(video.height)), int(video.n_frames),
                       settings=settings, initial=initial, start_frame=start_frame, sar=sar,
                       measure_shift=measure)
    canvas = cv.Canvas((int(video.width), int(video.height)), sar,
                       cv.video_size_for(cv.resolve_ui_scale(ui_scale)), strip_h=STRIP_H)
    state.pick_radius = PICK_RADIUS_DISPLAY_PX / canvas.px_per_row
    win = window if window is not None else Window(WINDOW_NAME)
    ask = ask or console_ask
    it = 0
    while True:
        closed, it = _event_loop(state, store, canvas, win, video, it, max_iterations)
        if not closed:
            break
        what = after_close(state, ask)
        if what != "edit":
            break
        try:
            win = _reopen(window, lambda: Window(WINDOW_NAME))
        except Exception as e:  # noqa: BLE001 - never lose the clicks because a window failed
            print(f"  The window could not be re-opened ({e}).")
            win = None
        if win is None:
            after_close(state, ask, allow_edit=False)
            break
        state.cancel_armed = False
        state.confirm = None
        state._say("window re-opened: Enter = check and save, Esc twice = quit without saving", "warn")
    if state.cancelled or not state.done:
        return None
    return state.result


def _event_loop(state: CalibState, store: FrameStore, canvas: cv.Canvas, win, video: VideoInfo, it: int,
                max_iterations: Optional[int]) -> tuple[bool, int]:
    """Poll the window until saved / cancelled / closed; returns (closed by the user, iterations)."""
    mouse_uv: Optional[tuple[int, int]] = None
    closed = False
    try:
        win.add_trackbar("frame", max(1, int(video.n_frames) - 1), state.frame)
        while not state.done:
            win.show(render(state, store, canvas, mouse_uv))
            code = win.wait_key(POLL_MS)
            if not win.is_open():
                closed = True
                break
            name = normalize_key(code)
            if name is not None:
                state.on_key(name)
            for ev in win.mouse_events():
                mouse_uv = (ev.x, ev.y)
                _feed_mouse(state, canvas, ev.kind, ev.x, ev.y)
            tb = win.trackbar_value()
            if tb is not None:
                state.on_trackbar(tb)
            win.set_trackbar(state.frame)
            it += 1
            if max_iterations is not None and it >= max_iterations:
                break
    finally:
        win.close()
    return closed, it


def _feed_mouse(state: CalibState, canvas: cv.Canvas, kind: str, u: int, v: int) -> None:
    if kind not in ("ldown", "lup", "rdown", "move"):
        return
    if kind in ("ldown", "rdown") and not canvas.in_video(u, v):
        return
    x, y = canvas.to_frame([(u, v)])[0]
    if kind == "move" and not state.mask_mode:
        return
    state.on_mouse(kind, float(x), float(y))
