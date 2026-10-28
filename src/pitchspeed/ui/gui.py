"""Thin, safe wrapper around OpenCV's highgui windows.

Two hazards are handled here so the UIs do not have to:

* On Linux the Qt backend *aborts the whole process* (no Python exception)
  when ``cv2.namedWindow`` cannot reach a display.  `gui_problem()` checks
  ``DISPLAY`` before any window is created and `require_gui()` raises a
  `GuiUnavailable` with a fix-it message instead.
* ``opencv-python-headless`` has no highgui at all; that is detected from
  ``cv2.getBuildInformation()`` (``GUI: NONE``) and reported the same way.
* On Linux, if PyAV (``av``) is imported BEFORE ``cv2``, ``cv2.namedWindow``
  hangs forever (measured with av 19.0.2 + opencv-python 5.0.0.93): both
  wheels bundle libxcb-shm / -shape / -xfixes with the *same* hashed
  sonames, so Qt ends up using PyAV's copies, which are bound to PyAV's
  private libxcb.  `gui_problem()` detects that state from
  /proc/self/maps and refuses with an explanation instead of hanging.
  The cure is to ``import cv2`` before anything imports ``av``.

`Window` queues mouse events (callback -> list), debounces the trackbar
(150 ms, and ignores the echo of its own ``setTrackbarPos``) and tells
when the user closed the window with its X button.  Windows are created
with ``WINDOW_AUTOSIZE | WINDOW_GUI_NORMAL``: the image is shown 1:1 (the
UIs draw their own scaled canvas, see ui/canvas.py) and Qt's toolbar,
whose zoom would confuse users, is hidden.  Measured with Qt under Xvfb:
mouse coordinates are image pixel coordinates and the trackbar sits below
the image.
"""
from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

TRACKBAR_DEBOUNCE_S = 0.15


class GuiUnavailable(RuntimeError):
    """No usable display / highgui; the message says how to fix it."""


def _cv2():
    import cv2
    return cv2


def highgui_backend() -> Optional[str]:
    """GUI backend from the OpenCV build information ('QT5', 'WIN32UI', 'NONE', ...)."""
    try:
        info = _cv2().getBuildInformation()
    except Exception:  # noqa: BLE001
        return None
    for line in info.splitlines():
        s = line.strip()
        if s.startswith("GUI:"):
            return s.split(":", 1)[1].strip() or None
    return None


_SHARED_XCB = re.compile(r"^libxcb-(shm|shape|xfixes)-[0-9a-f]{8}\.so")


def xcb_conflict(maps: Optional[str] = None) -> Optional[str]:
    """Directory of foreign libxcb extension copies that would hang Qt (Linux), else None.

    maps: text of /proc/self/maps (read when None).
    """
    if maps is None:
        try:
            with open("/proc/self/maps", "r", encoding="utf-8", errors="replace") as f:
                maps = f.read()
        except OSError:
            return None
    for line in maps.splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        path = parts[-1]
        d, base = os.path.split(path)
        if _SHARED_XCB.match(base) and os.path.basename(d) != "opencv_python.libs":
            return d
    return None


def gui_problem(env: Optional[dict] = None, platform: Optional[str] = None,
                backend: Optional[str] = "auto", maps: Optional[str] = "auto") -> Optional[str]:
    """Why no window can be opened, or None if it looks possible."""
    env = os.environ if env is None else env
    platform = sys.platform if platform is None else platform
    if backend == "auto":
        backend = highgui_backend()
    if backend is not None and backend.upper() == "NONE":
        return ("the installed OpenCV has no GUI support (probably opencv-python-headless).\n"
                "Fix:  pip uninstall -y opencv-python-headless opencv-python\n"
                "      pip install opencv-python")
    if platform.startswith("win") or platform == "darwin":
        return None
    if platform.startswith("linux"):
        foreign = xcb_conflict(None if maps == "auto" else maps)
        if foreign:
            return ("PyAV was loaded before OpenCV in this process, so OpenCV's Qt window would hang "
                    f"(both bundle libxcb extensions with the same names; loaded from {foreign}).\n"
                    "Fix (program code): import cv2 before anything imports av.")
    if not env.get("DISPLAY"):
        if env.get("WAYLAND_DISPLAY"):
            return ("only a Wayland display was found; OpenCV's Qt windows need X11 "
                    "(XWayland).  Make sure DISPLAY is set.")
        return ("no display: the DISPLAY environment variable is not set.  The calibration and "
                "review windows need a desktop session (for tests: xvfb-run -a ...).")
    return None


def require_gui() -> None:
    """Raise GuiUnavailable (with a fix-it message) if no window can be opened."""
    problem = gui_problem()
    if problem:
        raise GuiUnavailable(problem)


@dataclass
class MouseEvent:
    kind: str          # 'ldown' | 'lup' | 'rdown' | 'rup' | 'move' | 'wheel'
    x: int             # window image coordinates (display pixels)
    y: int
    flags: int = 0


def mouse_kind(cv_event: int) -> Optional[str]:
    cv2 = _cv2()
    table = {
        cv2.EVENT_LBUTTONDOWN: "ldown", cv2.EVENT_LBUTTONUP: "lup",
        cv2.EVENT_RBUTTONDOWN: "rdown", cv2.EVENT_RBUTTONUP: "rup",
        cv2.EVENT_MOUSEMOVE: "move", cv2.EVENT_MOUSEWHEEL: "wheel",
    }
    return table.get(cv_event)


class Window:
    """One highgui window (context manager).  Call `require_gui()` first."""

    def __init__(self, name: str, mouse: bool = True):
        require_gui()
        cv2 = _cv2()
        self.name = name
        self._events: list[MouseEvent] = []
        self._tb_name: Optional[str] = None
        self._tb_pending: Optional[tuple[int, float]] = None
        self._tb_echo: Optional[int] = None
        self._closed = False
        self._shown = False
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE | getattr(cv2, "WINDOW_GUI_NORMAL", 0))
        if mouse:
            cv2.setMouseCallback(name, self._on_mouse)

    # ----------------------------------------------------------- context
    def __enter__(self) -> "Window":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------- input
    def _on_mouse(self, event, x, y, flags, _param=None) -> None:
        kind = mouse_kind(event)
        if kind is None:
            return
        if kind == "move" and self._events and self._events[-1].kind == "move":
            self._events[-1] = MouseEvent(kind, int(x), int(y), int(flags))
        else:
            self._events.append(MouseEvent(kind, int(x), int(y), int(flags)))

    def mouse_events(self) -> list[MouseEvent]:
        ev, self._events = self._events, []
        return ev

    def wait_key(self, delay_ms: int) -> int:
        """Raw ``waitKeyEx`` code (-1: none).  Also pumps the GUI events."""
        if self._closed:
            return -1
        return int(_cv2().waitKeyEx(max(1, int(delay_ms))))

    # ---------------------------------------------------------- trackbar
    def add_trackbar(self, label: str, maximum: int, value: int = 0) -> None:
        cv2 = _cv2()
        self._tb_name = label
        cv2.createTrackbar(label, self.name, int(value), max(1, int(maximum)), self._on_trackbar)
        self._tb_echo = None

    def _on_trackbar(self, value: int) -> None:
        if self._tb_echo is not None and value == self._tb_echo:
            self._tb_echo = None
            return
        self._tb_pending = (int(value), time.monotonic())

    def set_trackbar(self, value: int) -> None:
        if self._tb_name is None or self._closed:
            return
        cv2 = _cv2()
        try:
            if cv2.getTrackbarPos(self._tb_name, self.name) == int(value):
                return
            self._tb_echo = int(value)
            cv2.setTrackbarPos(self._tb_name, self.name, int(value))
        except cv2.error:
            pass

    def trackbar_value(self, now: Optional[float] = None) -> Optional[int]:
        """The trackbar position once it has been still for 150 ms (else None)."""
        if self._tb_pending is None:
            return None
        v, t = self._tb_pending
        now = time.monotonic() if now is None else now
        if now - t < TRACKBAR_DEBOUNCE_S:
            return None
        self._tb_pending = None
        return v

    # ------------------------------------------------------------ output
    def show(self, img: np.ndarray) -> None:
        if self._closed:
            return
        _cv2().imshow(self.name, img)
        self._shown = True

    def set_title(self, title: str) -> None:
        try:
            _cv2().setWindowTitle(self.name, title)
        except Exception:  # noqa: BLE001
            pass

    def is_open(self) -> bool:
        """False once the user closed the window (X button) or close() ran."""
        if self._closed:
            return False
        if not self._shown:
            return True
        cv2 = _cv2()
        try:
            visible = cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            visible = -1
        if visible < 1:
            self._closed = True
        return not self._closed

    def close(self) -> None:
        if self._closed and not self._shown:
            return
        cv2 = _cv2()
        self._closed = True
        try:
            cv2.destroyWindow(self.name)
            for _ in range(3):
                cv2.waitKey(1)
        except cv2.error:
            pass
