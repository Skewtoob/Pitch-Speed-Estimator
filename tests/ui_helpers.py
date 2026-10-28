"""Test helpers for the UI area: fake sources, scripted windows, fake results.

FakeSource implements the SampleSource contract (DESIGN.md 3.1) on
synthetic frames: a smooth background with a grid, the frame number
burnt in, and a bright 3x3 "dot" at a known frame-pixel position, so
coordinate mapping can be checked on real pixels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

from pitchspeed.types import (AnalysisResult, EventTime, Pitch, PitchTrack, Sample, ScanInfo,
                              SpeedEstimate, TrackPoint, VideoInfo)

NTSC_FRAME_DT = 1001.0 / 30000.0


_BACKGROUNDS: dict = {}


def synthetic_frame(i: int, width: int = 720, height: int = 480, dot=(400.0, 201.0)) -> np.ndarray:
    """Gray frame i: gradient + grid + a 3x3 white dot at `dot` (frame px) + frame number bars."""
    key = (width, height, tuple(dot))
    if key not in _BACKGROUNDS:
        _BACKGROUNDS[key] = _background(width, height, dot)
    img = _BACKGROUNDS[key].copy()
    # frame number as a row of bits (top-left), robust to bobbing
    for b in range(12):
        if (i >> b) & 1:
            img[4:12, 8 + 10 * b:16 + 10 * b] = 250
    return img


def _background(width: int, height: int, dot) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    img = (40 + 0.15 * xx + 0.1 * yy).astype(np.float32)
    img[(yy % 40) == 0] += 40
    img[:, ::60] += 40
    img = np.clip(img, 0, 255).astype(np.uint8)
    x0, y0 = int(round(dot[0])), int(round(dot[1]))
    img[max(0, y0 - 1):y0 + 2, max(0, x0 - 1):x0 + 2] = 255
    return img


class FakeReader:
    def __init__(self, src: "FakeSource"):
        self.src = src
        self.info = src.info
        self.frame_dt = src.frame_dt
        self.n_frames = src.n_frames

    def get_frame(self, i: int, fmt: str = "gray"):
        g = self.src.frame(i)
        if fmt == "gray":
            return g
        bgr = np.repeat(g[:, :, None], 3, axis=2)
        return (g, bgr) if fmt == "both" else bgr


class FakeSource:
    """SampleSource look-alike (fields if field_order is 'tff'/'bff', else frames)."""

    def __init__(self, n_frames: int = 300, width: int = 720, height: int = 480,
                 field_order: Optional[str] = "bff", sar: float = 8.0 / 9.0, dot=(400.0, 201.0),
                 fail_frames=(), pan=(0.0, 0.0)):
        self.n_frames = int(n_frames)
        self.width, self.height = int(width), int(height)
        self.interlaced = field_order is not None
        self.field_order = field_order
        self.frame_dt = NTSC_FRAME_DT
        self.sample_dt = self.frame_dt / 2.0 if self.interlaced else self.frame_dt
        self.n_samples = 2 * self.n_frames if self.interlaced else self.n_frames
        self.sample_shape = (self.height // 2, self.width) if self.interlaced else (self.height, self.width)
        self.scan = ScanInfo(mode="interlaced" if self.interlaced else "progressive",
                             field_order=field_order, confidence="high",
                             sample_rate=1.0 / self.sample_dt)
        self.info = make_video_info(n_frames, width, height, sar)
        self.dot = dot
        self.fail_frames = set(fail_frames)
        self.pan = (float(pan[0]), float(pan[1]))       # camera pan, frame px per frame
        self.get_calls = 0
        self.iter_calls = 0
        self.reader = FakeReader(self)
        self._first_parity = 0 if field_order in (None, "tff") else 1

    def set_field_order(self, order: str) -> None:
        if not self.interlaced:
            raise ValueError("progressive video has no field order")
        self.field_order = self.scan.field_order = order
        self._first_parity = 0 if order == "tff" else 1

    # index arithmetic
    def locate(self, s: int) -> tuple[int, int, int]:
        s = int(s)
        if not 0 <= s < self.n_samples:
            raise IndexError(s)
        if self.interlaced:
            k = s & 1
            return s >> 1, k, k ^ self._first_parity
        return s, 0, 0

    def sample_of(self, frame: int, k: int = 0) -> int:
        return 2 * int(frame) + int(k) if self.interlaced else int(frame)

    def sample_time(self, s: int) -> float:
        f, k, _ = self.locate(s)
        return f * self.frame_dt + k * self.frame_dt / 2.0

    def sample_at_time(self, t: float) -> int:
        return int(min(max(round(t / self.sample_dt), 0), self.n_samples - 1))

    # images
    def frame(self, i: int) -> np.ndarray:
        if i in self.fail_frames:
            raise RuntimeError(f"frame {i} could not be decoded (fake)")
        img = synthetic_frame(i, self.width, self.height, self.dot)
        if self.pan != (0.0, 0.0):
            import cv2
            M = np.float32([[1, 0, i * self.pan[0]], [0, 1, i * self.pan[1]]])
            img = cv2.warpAffine(img, M, (self.width, self.height), borderMode=cv2.BORDER_REFLECT)
        return img

    def _make(self, s: int, color: bool) -> Sample:
        f, k, p = self.locate(s)
        g = self.frame(f)
        img = g[p::2].copy() if self.interlaced else g
        bgr = np.repeat(img[:, :, None], 3, axis=2) if color else None
        return Sample(s=s, frame=f, k=k, parity=p, t=self.sample_time(s), image=img, bgr=bgr)

    def get_sample(self, s: int, color: bool = False) -> Sample:
        self.get_calls += 1
        return self._make(s, color)

    def iter_samples(self, s0: int = 0, s1: Optional[int] = None, color: bool = False) -> Iterator[Sample]:
        self.iter_calls += 1
        s1 = self.n_samples if s1 is None else min(self.n_samples, s1)
        for s in range(max(0, s0), s1):
            try:
                yield self._make(s, color)
            except RuntimeError:
                continue


def make_video_info(n_frames: int = 300, width: int = 720, height: int = 480, sar: float = 8.0 / 9.0) -> VideoInfo:
    return VideoInfo(path="fake.avi", width=width, height=height, n_frames=n_frames,
                     frame_rate=30000.0 / 1001.0, frame_rate_str="30000/1001", sar=sar,
                     duration_s=n_frames * NTSC_FRAME_DT)


# ----------------------------------------------------------- scripted window
@dataclass
class ScriptedWindow:
    """Stand-in for ui.gui.Window driven by a script of events.

    script: list of items, consumed one per poll:
      int            raw key code returned by wait_key
      ('key', code)  same
      ('mouse', kind, u, v)   mouse event (display coords), then wait_key -> -1
      ('trackbar', value)     trackbar moved (reported immediately)
      ('close',)     user closed the window
      None           an idle poll
    """
    script: list
    shown: list = field(default_factory=list)
    trackbar: Optional[int] = None
    trackbar_max: Optional[int] = None
    closed: bool = False
    _mouse: list = field(default_factory=list)
    _tb: Optional[int] = None
    keep_images: bool = False

    def add_trackbar(self, label, maximum, value=0):
        self.trackbar_max = maximum
        self.trackbar = value

    def set_trackbar(self, value):
        self.trackbar = value

    def trackbar_value(self, now=None):
        v, self._tb = self._tb, None
        return v

    def show(self, img):
        self.last_image = img
        if self.keep_images:
            self.shown.append(img.copy())
        else:
            self.shown.append(img.shape)

    def wait_key(self, delay_ms):
        if not self.script:
            return -1
        item = self.script.pop(0)
        if item is None:
            return -1
        if isinstance(item, int):
            return item
        if item[0] == "key":
            return item[1]
        if item[0] == "mouse":
            from pitchspeed.ui.gui import MouseEvent
            self._mouse.append(MouseEvent(item[1], int(item[2]), int(item[3])))
            return -1
        if item[0] == "trackbar":
            self._tb = int(item[1])
            return -1
        if item[0] == "close":
            self.closed = True
            return -1
        raise ValueError(item)

    def mouse_events(self):
        ev, self._mouse = self._mouse, []
        return ev

    def is_open(self):
        return not self.closed

    def close(self):
        self.closed = True

    def reopen(self):
        """A new window after the user closed this one (the script continues)."""
        self.closed = False
        self.reopened = getattr(self, "reopened", 0) + 1
        return self


# Qt (Linux) codes measured under Xvfb, used to script key presses
QT = {"left": 65361, "up": 65362, "right": 65363, "down": 65364, "home": 65360, "end": 65367,
      "pgup": 65365, "pgdn": 65366, "del": 65535, "backspace": 8, "enter": 13, "tab": 9, "esc": 27,
      "space": 32}


def key(name: str) -> int:
    """Raw Qt code for a key name (single characters map to their code)."""
    return QT[name] if name in QT else ord(name)


# ---------------------------------------------------------------- results
def speed(v: Optional[float] = 75.0, sigma: float = 2.0, method: str = "tof", label: str = "good") -> SpeedEstimate:
    if v is None:
        return SpeedEstimate(v_release_mph=None, reason="not measurable")
    return SpeedEstimate(v_release_mph=v, sigma_mph=sigma, ci90_mph=(v - 1.645 * sigma, v + 1.645 * sigma),
                         v_avg_mph=v - 3, method=method, label=label)


def make_pitch(src: FakeSource, s_rel: int, s_cat: int, number: int = 0, status: str = "auto",
               n_points: Optional[int] = None) -> Pitch:
    t_r, t_c = src.sample_time(s_rel), src.sample_time(s_cat)
    pts = []
    n_points = (s_cat - s_rel + 1) if n_points is None else n_points
    for j in range(n_points):
        s = s_rel + j
        x = 600 - 15 * j
        y = 150 + 2 * j
        pts.append(TrackPoint(s=s, t=src.sample_time(s), x=float(x), y=float(y), score=8.0))
    tr = PitchTrack(track_id=number, points=pts,
                    release=EventTime(t_r, 0.004, "first_free_field", s_rel),
                    catch=EventTime(t_c, 0.004, "extrapolated", s_cat), confidence=0.8)
    return Pitch(number=number, status=status, release=EventTime(t_r, 0.004, "first_free_field", s_rel),
                 catch=EventTime(t_c, 0.004, "extrapolated", s_cat), speed=speed(), track=tr,
                 frame_release=src.locate(s_rel)[0], frame_catch=src.locate(s_cat)[0], confidence=0.8)


def make_result(src: FakeSource, windows=((100, 125), (300, 328), (500, 526)), missed=((420, 440),),
                rejected=0) -> AnalysisResult:
    pitches = [make_pitch(src, a, b, number=i + 1) for i, (a, b) in enumerate(windows)]
    pm = [{"s_start": a, "s_end": b, "t_start": src.sample_time(a), "t_end": src.sample_time(b),
           "reason": "strong motion without a track"} for a, b in missed]
    rej = []
    for j in range(rejected):
        a = 200 + 40 * j
        tr = make_pitch(src, a, a + 20).track
        tr.status, tr.reject_code = "rejected", "TOO_SLOW"
        rej.append(tr)
    return AnalysisResult(video=src.info, scan=src.scan, calibration={}, params={}, pitches=pitches,
                          possible_missed=pm, rejected_tracks=rej)


class Recorder:
    """recompute/save callbacks that record their calls."""

    def __init__(self, v: float = 80.0, fail_recompute: bool = False, fail_save: bool = False):
        self.recomputed: list = []
        self.saved = 0
        self.v = v
        self.fail_recompute = fail_recompute
        self.fail_save = fail_save

    def recompute(self, p: Pitch) -> Pitch:
        self.recomputed.append(p)
        if self.fail_recompute:
            raise RuntimeError("boom")
        if p.release is not None and p.catch is not None:
            T = p.catch.t - p.release.t
            v = 60.5 * 15 / 22 / T if T > 0 else None
            p.speed = speed(round(v, 2) if v else None)
        return p

    def save(self, result: AnalysisResult) -> None:
        if self.fail_save:
            raise OSError("disk full")
        self.saved += 1


# ------------------------------------------------------- Xvfb GUI driver --
# Run in a SUBPROCESS under xvfb-run (Qt aborts the whole process when it
# cannot reach a display, so the pytest process never opens a window).
# A background thread injects key / mouse events with XTEST while the UI
# loop runs in the main thread; the result is printed as one JSON line.

def _find_window(d, name: str, timeout_s: float = 15.0):
    import time
    root = d.screen().root

    def find(w):
        try:
            if w.get_wm_name() == name:
                return w
        except Exception:  # noqa: BLE001
            pass
        try:
            children = w.query_tree().children
        except Exception:  # noqa: BLE001
            return None
        for c in children:
            r = find(c)
            if r is not None:
                return r
        return None

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        w = find(root)
        if w is not None:
            return w
        time.sleep(0.1)
    raise RuntimeError(f"window {name!r} did not appear")


class XInjector:
    """XTEST key / mouse injection into one window (python-xlib)."""

    def __init__(self, window_name: str, delay_s: float = 0.15):
        from Xlib import X, XK, display
        from Xlib.ext import xtest
        self.X, self.XK, self.xtest = X, XK, xtest
        self.d = display.Display()
        self.delay_s = delay_s
        self.win = _find_window(self.d, window_name)
        root = self.d.screen().root
        tr = self.win.translate_coords(root, 0, 0)
        self.x0, self.y0 = -tr.x, -tr.y
        import time
        time.sleep(0.5)                                  # let the first frame be shown
        self.win.set_input_focus(X.RevertToParent, X.CurrentTime)
        self.move(5, 5)

    def _sleep(self):
        import time
        time.sleep(self.delay_s)

    def key(self, keysym: str, shift: bool = False) -> None:
        X = self.X
        kc = self.d.keysym_to_keycode(self.XK.string_to_keysym(keysym))
        if kc == 0:
            raise ValueError(f"no keycode for {keysym!r}")
        sh = self.d.keysym_to_keycode(self.XK.string_to_keysym("Shift_L"))
        if shift:
            self.xtest.fake_input(self.d, X.KeyPress, sh)
        self.xtest.fake_input(self.d, X.KeyPress, kc)
        self.xtest.fake_input(self.d, X.KeyRelease, kc)
        if shift:
            self.xtest.fake_input(self.d, X.KeyRelease, sh)
        self.d.sync()
        self._sleep()

    def move(self, u: float, v: float) -> None:
        self.xtest.fake_input(self.d, self.X.MotionNotify, x=self.x0 + int(round(u)), y=self.y0 + int(round(v)))
        self.d.sync()
        self._sleep()

    def close_window(self) -> None:
        """What the X button does: send WM_DELETE_WINDOW to the window."""
        from Xlib import protocol
        X = self.X
        ev = protocol.event.ClientMessage(window=self.win, client_type=self.d.intern_atom("WM_PROTOCOLS"),
                                          data=(32, [self.d.intern_atom("WM_DELETE_WINDOW"), X.CurrentTime,
                                                     0, 0, 0]))
        self.win.send_event(ev, event_mask=X.NoEventMask)
        self.d.sync()
        self._sleep()

    def click(self, u: float, v: float, button: int = 1) -> None:
        self.move(u, v)
        self.xtest.fake_input(self.d, self.X.ButtonPress, button)
        self.xtest.fake_input(self.d, self.X.ButtonRelease, button)
        self.d.sync()
        self._sleep()


def _inject(name: str, actions, errors: list) -> None:
    try:
        inj = XInjector(name)
        for a in actions:
            if a[0] == "close":
                inj.close_window()
            elif a[0] == "refind":                 # the window was re-opened: a new X window
                import time
                time.sleep(1.0)
                inj = XInjector(name)
            elif a[0] == "key":
                inj.key(*a[1:])
            elif a[0] == "click":
                inj.click(*a[1:])
            elif a[0] == "move":
                inj.move(*a[1:])
            elif a[0] == "sleep":
                import time
                time.sleep(a[1])
    except Exception as e:  # noqa: BLE001
        errors.append(repr(e))


def gui_driver_main(kind: str) -> None:
    """Entry point of the GUI smoke-test subprocess (see tests/test_ui_gui.py)."""
    import json
    import threading

    from pitchspeed.ui.canvas import Canvas

    canvas = Canvas((720, 480), 8.0 / 9.0, (720, 540))

    def disp(x, y):
        return tuple(canvas.to_display([(x, y)])[0])

    errors: list = []
    out: dict = {}
    if kind == "calib":
        from pitchspeed.calib.ui import WINDOW_NAME, run_calibration_ui
        src = FakeSource(n_frames=60)
        actions = [("click", *disp(400, 201)), ("key", "Right"), ("key", "d"), ("click", *disp(100, 300)),
                   ("key", "i"), ("key", "z"), ("move", 300, 300), ("key", "h"), ("key", "h"),
                   ("key", "Return"), ("key", "Return")]
        th = threading.Thread(target=_inject, args=(WINDOW_NAME, actions, errors), daemon=True)
        th.start()
        cf = run_calibration_ui(src, src.info, "side")
        out = {"result": None if cf is None else cf.to_dict()}
    elif kind == "calib_close":
        # X button with all points placed: the console question answers e (re-open the window),
        # then the X button again and y (save).
        from pitchspeed.calib.ui import WINDOW_NAME, run_calibration_ui
        src = FakeSource(n_frames=60)
        answers = ["e", "y"]
        prompts: list = []

        def ask(prompt):
            prompts.append(prompt)
            return answers.pop(0) if answers else None
        actions = [("click", *disp(400, 201)), ("click", *disp(100, 300)), ("close",), ("refind",),
                   ("key", "i"), ("close",)]
        th = threading.Thread(target=_inject, args=(WINDOW_NAME, actions, errors), daemon=True)
        th.start()
        cf = run_calibration_ui(src, src.info, "side", ask=ask)
        out = {"result": None if cf is None else cf.to_dict(), "prompts": len(prompts)}
    elif kind == "calib_cancel":
        from pitchspeed.calib.ui import WINDOW_NAME, run_calibration_ui
        src = FakeSource(n_frames=60)
        actions = [("click", *disp(400, 201)), ("key", "Escape"), ("key", "Escape")]
        th = threading.Thread(target=_inject, args=(WINDOW_NAME, actions, errors), daemon=True)
        th.start()
        out = {"result": run_calibration_ui(src, src.info, "side")}
    elif kind == "review":
        from pitchspeed.review.ui import WINDOW_NAME, run_review_ui
        src = FakeSource(n_frames=400)
        res = make_result(src)
        rec = Recorder()
        actions = [("key", "Right"), ("key", "Right"), ("key", "r"), ("click", *disp(500, 180)),
                   ("key", "Next"), ("key", "x"), ("key", "space"), ("sleep", 0.4), ("key", "space"),
                   ("key", "Prior"), ("key", "h"), ("key", "h"), ("key", "q")]
        th = threading.Thread(target=_inject, args=(WINDOW_NAME, actions, errors), daemon=True)
        th.start()
        run_review_ui(res, src, None, rec.recompute, rec.save)
        out = {"pitches": [(p.number, p.status, p.release.s) for p in res.pitches], "saved": rec.saved,
               "manual_points": sum(1 for q in res.pitches[0].track.points if q.manual)}
    elif kind == "keys":
        from pitchspeed.ui.keys import show_key_codes
        names = ["Left", "Up", "Right", "Down", "Prior", "Next", "Home", "End", "Delete", "BackSpace",
                 "Return", "Tab", "a", "z", "comma", "period", "space"]
        actions = [("key", n) for n in names] + [("key", "Escape"), ("key", "Escape")]
        th = threading.Thread(target=_inject, args=("pitchspeed keys", actions, errors), daemon=True)
        th.start()
        seen = show_key_codes(timeout_s=30, out=lambda s: None)
        out = {"seen": seen}
    else:
        raise SystemExit(f"unknown kind {kind!r}")
    th.join(timeout=5)
    out["errors"] = errors
    print("RESULT " + json.dumps(out))
