"""Calibration window logic as a pure state machine (no GUI, no decoding).

`CalibState` receives key names (`ui.keys.normalize_key`) and mouse events
in FRAME PIXEL COORDINATES (stored pixels; the shell converts display
clicks with `ui.canvas.Canvas.to_frame`) and keeps the clicked points.
`calib/ui.py` is the thin cv2 shell around it.

POINTS.  Names, requirements and help texts come from calib/model.py
(REQUIRED_POINTS / OPTIONAL_POINTS / POINT_HELP).  The HUD always names the
next point and says exactly what to click.  Required points first, then
optional ones; N skips an optional point, Tab/Shift+Tab pick any point
(to re-click it), I/J/K/L nudge the selected point by 0.25 px, right-click
or Del deletes, U/Backspace undoes (unbounded).

FRAMES AND THE REFERENCE FRAME.  The user may change frames between
points: the release click needs a frame where the pitcher is releasing,
the catch click one where the ball is in the glove.  Every point records
the frame it was clicked on, and all points of a segment are stored as
coordinates of ONE reference frame, ``Segment.ref_frame`` = the frame of
the *rubber* click (side view; plate if no rubber) or the *release* click
(behind view; catch if no release).  That is exact for a tripod.  When
points come from different frames, Enter measures the camera shift
between those frames (injected `measure_shift`, e.g. phase correlation)
and warns if it exceeds 1 px; without a measurement it warns that the
camera must not have moved.  Points are never moved automatically.  The
per-point frames are written to ``CalibrationFile.notes`` as a line
``point_frames: [{...}, ...]`` (one dict per segment) because
``Segment`` has no field for them (foundation request filed), and read
back from there when an existing calibration is edited.  The frame
warnings shown at save time are recorded there too ("warning at save: ...").

SEGMENTS.  G starts a new calibration segment at the current frame (after
a zoom/pan); points clicked later than that frame belong to it.  G on the
first frame of a segment removes that boundary again.  The active segment
is always the one containing the current frame.

MASKS.  M, then drag a rectangle (e.g. over the camcorder date burn-in);
right-click inside a mask deletes it.

ENTER validates (CalibrationFile.validate + the frame checks above).  A
fatal problem (missing required point) is shown and editing continues;
warnings are listed and a second Enter saves anyway.  Esc twice cancels.

NAVIGATION (same convention as the review window): Left/A and Right/D
step one field (one frame in full-frame view), Up/W and Down/S one second
back/forward, PgUp/[ and PgDn/] ten seconds, Home/End the ends of the
video, plus the trackbar.  "Down = later" everywhere, like the
previous/next-edit keys of video editors.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from pitchspeed.calib.model import (OPTIONAL_POINTS, POINT_HELP, REQUIRED_POINTS, CalibrationFile,
                                    Segment, Settings)
from pitchspeed.ui.keys import key_label, resolve

NUDGE_PX = 0.25
SHIFT_WARN_PX = 1.0
POINT_FRAMES_PREFIX = "point_frames:"
WARNING_PREFIX = "warning at save:"

BINDINGS: dict[str, list[str]] = {
    "prev_field": ["left", "a"],
    "next_field": ["right", "d"],
    "back_1s": ["up", "w"],
    "fwd_1s": ["down", "s"],
    "back_10s": ["pgup", "["],
    "fwd_10s": ["pgdn", "]"],
    "start": ["home"],
    "end": ["end"],
    "nudge_up": ["i"],
    "nudge_down": ["k"],
    "nudge_left": ["j"],
    "nudge_right": ["l"],
    "next_point": ["tab"],
    "prev_point": ["backtab"],
    "skip": ["n"],
    "delete": ["del", "x"],
    "undo": ["u", "backspace"],
    "mask": ["m"],
    "segment": ["g"],
    "toggle_view": ["f"],
    "loupe": ["z"],
    "help": ["h", "?", "f1"],
    "save": ["enter"],
    "cancel": ["esc"],
}

HELP_TEXT: dict[str, str] = {
    "prev_field": "back 1 field (1 frame in full-frame view)",
    "next_field": "forward 1 field",
    "back_1s": "back 1 second",
    "fwd_1s": "forward 1 second",
    "back_10s": "back 10 seconds",
    "fwd_10s": "forward 10 seconds",
    "start": "first frame",
    "end": "last frame",
    "nudge_up": "nudge selected point up 0.25 px",
    "nudge_down": "nudge selected point down 0.25 px",
    "nudge_left": "nudge selected point left 0.25 px",
    "nudge_right": "nudge selected point right 0.25 px",
    "next_point": "choose the next point (to click or re-click)",
    "prev_point": "choose the previous point",
    "skip": "skip this optional point (e.g. not visible)",
    "delete": "delete the selected point",
    "undo": "undo (unlimited)",
    "mask": "mask: drag a box over the date/time burn-in",
    "segment": "new calibration segment from this frame (camera zoomed/moved); again = remove",
    "toggle_view": "bobbed field / full frame view",
    "loupe": "8x loupe with crosshair",
    "help": "this help",
    "save": "check and save (shows warnings; Enter again saves)",
    "cancel": "Esc twice: quit WITHOUT saving",
}

#: How to find the right frame for points that depend on the moment.
EXTRA_HINT: dict[str, str] = {
    "release": "Step (Left/Right = 1 field) to where the ball is just leaving the hand, then click the BALL.",
    "catch": "Step to the field where the ball is in the glove, then click the BALL/glove.",
    "rubber": "Any frame works on a tripod. F = full frame (sharper for still landmarks), Z = loupe.",
    "plate": "Click the middle of the white plate (front-to-back middle). Z = loupe.",
    "vertical_top": "Optional: lets the program know which way is truly 'up' if the camera is tilted.",
}


@dataclass
class SegmentDraft:
    """Editable form of a calibration Segment."""
    start_frame: int = 0
    points: dict = field(default_factory=dict)          # name -> [x, y] (frame px)
    point_frames: dict = field(default_factory=dict)    # name -> frame index the click was made on
    skipped: set = field(default_factory=set)
    masks: list = field(default_factory=list)           # [x0, y0, x1, y1]


@dataclass
class Message:
    text: str
    level: str = "info"         # 'info' | 'ok' | 'warn' | 'error'


class CalibState:
    """Pure calibration state machine (see module docstring).

    timeline: object with ``n_samples``, ``sample_dt``, ``interlaced``,
        ``locate(s) -> (frame, k, parity)``, ``sample_of(frame, k)`` and
        ``sample_time(s)`` (a SampleSource; only its index arithmetic is used).
    image_size: (width, height) of a frame in stored pixels.
    measure_shift: optional ``f(frame_a, frame_b) -> (dx, dy) | None``, the
        camera motion between two frames in frame pixels (None: unknown).
    """

    def __init__(self, timeline, view: str, image_size: tuple[int, int], n_frames: int,
                 settings: Optional[Settings] = None, initial: Optional[CalibrationFile] = None,
                 start_frame: int = 0, sar: float = 1.0,
                 measure_shift: Optional[Callable[[int, int], Optional[tuple]]] = None):
        if view not in REQUIRED_POINTS:
            raise ValueError(f"view must be one of {sorted(REQUIRED_POINTS)}, not {view!r}")
        self.tl = timeline
        self.view = view
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.n_frames = int(n_frames)
        self.sar = float(sar or 1.0)
        self.measure_shift = measure_shift
        self.required = list(REQUIRED_POINTS[view])
        self.order = self.required + [n for n in OPTIONAL_POINTS[view] if n not in self.required]
        base = settings or (initial.settings if initial is not None else None) or Settings()
        self.settings = dataclasses.replace(base, sar=self.sar)
        self.video_meta = dict(initial.video) if initial is not None else {}
        self.user_notes = _strip_point_frames(initial.notes) if initial is not None else ""
        self.segments: list[SegmentDraft] = []
        self.message: Optional[Message] = None
        self.load_messages: list[str] = []
        if initial is not None and initial.segments:
            self._load(initial)
        if not self.segments:
            self.segments = [SegmentDraft(start_frame=0)]
        sf = int(start_frame)
        if sf == 0 and initial is not None and initial.segments:
            sf = int(initial.segments[0].ref_frame)
        self.s = self._clamp_s(self.tl.sample_of(self._clamp_frame(sf), 0))
        self.view_mode = "field" if getattr(self.tl, "interlaced", False) else "frame"
        self.loupe = False
        self.help = False
        self.mask_mode = False
        self.drag: Optional[tuple[float, float, float, float]] = None   # x0, y0, x1, y1
        self.cancel_armed = False
        self.confirm: Optional[list[str]] = None
        self.done = False
        self.cancelled = False
        self.result: Optional[CalibrationFile] = None
        self.pick_radius = 12.0          # frame rows (set by the shell from the display scale)
        self.history: list[tuple] = []
        self.last_action = ""
        self.target: Optional[str] = None
        self.selected: Optional[str] = None
        self._shift_cache: dict[tuple[int, int], Optional[tuple]] = {}
        self._reset_target()
        if self.load_messages:
            self.message = Message("; ".join(self.load_messages), "warn")

    # ------------------------------------------------------------ loading
    def _load(self, cf: CalibrationFile) -> None:
        frames = _parse_point_frames(cf.notes)
        if tuple(cf.image_size) != self.image_size:
            self.load_messages.append(f"existing calibration is for {tuple(cf.image_size)} images, this video "
                                      f"is {self.image_size}: check every point")
        valid = set(self.order)
        dropped = []
        for i, seg in enumerate(sorted(cf.segments, key=lambda g: g.start_frame)):
            pf = frames[i] if i < len(frames) else {}
            d = SegmentDraft(start_frame=int(seg.start_frame), masks=[list(map(float, m)) for m in seg.masks])
            for name, p in (seg.points or {}).items():
                if p is None:
                    continue
                if name not in valid:
                    dropped.append(name)
                    continue
                d.points[name] = [float(p[0]), float(p[1])]
                d.point_frames[name] = int(pf.get(name, seg.ref_frame))
            if seg.view != self.view:
                self.load_messages.append(f"segment {i + 1} was a {seg.view} view; now {self.view}")
            self.segments.append(d)
        if dropped:
            self.load_messages.append(f"points not used in a {self.view} view were dropped: {sorted(set(dropped))}")

    # ------------------------------------------------------------ helpers
    @property
    def frame(self) -> int:
        return int(self.tl.locate(self.s)[0])

    @property
    def field_k(self) -> int:
        return int(self.tl.locate(self.s)[1])

    @property
    def parity(self) -> int:
        return int(self.tl.locate(self.s)[2])

    def _clamp_s(self, s: int) -> int:
        return int(min(max(int(s), 0), max(0, int(self.tl.n_samples) - 1)))

    def _clamp_frame(self, f: int) -> int:
        return int(min(max(int(f), 0), max(0, self.n_frames - 1)))

    @property
    def seg_index(self) -> int:
        f = self.frame
        idx = 0
        for i, g in enumerate(self.segments):
            if g.start_frame <= f:
                idx = i
        return idx

    @property
    def seg(self) -> SegmentDraft:
        return self.segments[self.seg_index]

    def is_required(self, name: str) -> bool:
        return name in self.required

    def _snapshot(self) -> None:
        self.history.append((copy.deepcopy(self.segments), self.target, self.selected))

    def _say(self, text: str, level: str = "info") -> None:
        self.message = Message(text, level)

    def _next_open(self, after: Optional[str]) -> Optional[str]:
        seg = self.seg
        start = self.order.index(after) + 1 if after in self.order else 0
        n = len(self.order)
        for j in range(n):
            name = self.order[(start + j) % n]
            if name not in seg.points and name not in seg.skipped:
                return name
        return None

    def _reset_target(self) -> None:
        self.target = self._next_open(None)
        if self.selected not in self.seg.points:
            self.selected = None

    # --------------------------------------------------------------- input
    def on_key(self, name: Optional[str]) -> bool:
        """Handle a normalized key name; returns True if it was used."""
        action = resolve(name, BINDINGS)
        if action is None:
            if name is not None:
                self._clear_modal()
            return False
        self.on_action(action)
        return True

    def _clear_modal(self) -> None:
        if self.cancel_armed:
            self.cancel_armed = False
            self._say("quit aborted - keep clicking")
        if self.confirm is not None:
            self.confirm = None
            self._say("save aborted - keep editing")

    def on_action(self, action: str) -> None:
        """Run one action from BINDINGS (messages persist until replaced)."""
        if self.cancel_armed and action != "cancel":
            self.cancel_armed = False
            self._say("quit aborted - keep clicking")
        if self.confirm is not None and action not in ("save", "cancel"):
            self.confirm = None
            self._say("save aborted - keep editing")
        seg_before, msg_before = self.seg_index, self.message
        getattr(self, "_do_" + action)()
        self.last_action = action
        if self.seg_index != seg_before:
            self._reset_target()
            if self.message is msg_before:
                self._say(f"now in calibration segment {self.seg_index + 1} (from frame {self.seg.start_frame})")

    # navigation
    def _step(self, n_samples: int) -> None:
        self.s = self._clamp_s(self.s + n_samples)

    def _samples_per(self, seconds: float) -> int:
        return max(1, int(round(seconds / float(self.tl.sample_dt))))

    def _field_step(self) -> int:
        return 2 if (self.view_mode == "frame" and getattr(self.tl, "interlaced", False)) else 1

    def _do_prev_field(self) -> None:
        self._step(-self._field_step())

    def _do_next_field(self) -> None:
        self._step(self._field_step())

    def _do_back_1s(self) -> None:
        self._step(-self._samples_per(1.0))

    def _do_fwd_1s(self) -> None:
        self._step(self._samples_per(1.0))

    def _do_back_10s(self) -> None:
        self._step(-self._samples_per(10.0))

    def _do_fwd_10s(self) -> None:
        self._step(self._samples_per(10.0))

    def _do_start(self) -> None:
        self.s = 0

    def _do_end(self) -> None:
        self.s = self._clamp_s(int(self.tl.n_samples) - 1)

    def on_trackbar(self, frame: int) -> None:
        before = self.seg_index
        self.s = self._clamp_s(self.tl.sample_of(self._clamp_frame(frame), 0))
        if self.seg_index != before:
            self._reset_target()

    def goto_frame(self, frame: int) -> None:
        self.on_trackbar(frame)

    # points
    def _nudge(self, dx: float, dy: float) -> None:
        name = self.selected
        if name is None or name not in self.seg.points:
            self._say("no point selected to nudge (click one, or Tab to it)", "warn")
            return
        self._snapshot()
        p = self.seg.points[name]
        self.seg.points[name] = [round(p[0] + dx, 4), round(p[1] + dy, 4)]
        self._say(f"{name} at ({self.seg.points[name][0]:.2f}, {self.seg.points[name][1]:.2f})")

    def _do_nudge_up(self) -> None:
        self._nudge(0.0, -NUDGE_PX)

    def _do_nudge_down(self) -> None:
        self._nudge(0.0, NUDGE_PX)

    def _do_nudge_left(self) -> None:
        self._nudge(-NUDGE_PX, 0.0)

    def _do_nudge_right(self) -> None:
        self._nudge(NUDGE_PX, 0.0)

    def _cycle(self, step: int) -> None:
        cur = self.target if self.target is not None else self.selected
        i = self.order.index(cur) if cur in self.order else (-1 if step > 0 else 0)
        self.target = self.order[(i + step) % len(self.order)]
        self.selected = self.target if self.target in self.seg.points else None
        if self.target in self.seg.skipped:
            self._say(f"{self.target} was skipped; click to place it anyway")

    def _do_next_point(self) -> None:
        self._cycle(1)

    def _do_prev_point(self) -> None:
        self._cycle(-1)

    def _do_skip(self) -> None:
        name = self.target
        if name is None:
            self._say("all points are placed; nothing to skip")
            return
        if self.is_required(name):
            self._say(f"{name} is required and cannot be skipped", "error")
            return
        if name in self.seg.points:
            self._say(f"{name} is already placed (Del or right-click deletes it)", "warn")
            self.target = self._next_open(name)
            return
        self._snapshot()
        self.seg.skipped.add(name)
        self.target = self._next_open(name)
        self._say(f"skipped {name}")

    def _do_delete(self) -> None:
        name = self.selected
        if name is None or name not in self.seg.points:
            self._say("no point selected (right-click a point to delete it)", "warn")
            return
        self._delete_point(name)

    def _delete_point(self, name: str) -> None:
        self._snapshot()
        self.seg.points.pop(name, None)
        self.seg.point_frames.pop(name, None)
        self.target = name
        self.selected = None
        self._say(f"deleted {name}; click to place it again")

    def _do_undo(self) -> None:
        if not self.history:
            self._say("nothing to undo")
            return
        self.segments, self.target, self.selected = self.history.pop()
        self.drag = None
        self._say("undone")

    def place(self, x: float, y: float) -> None:
        """Place the current target point at frame pixel (x, y) on the current frame."""
        name = self.target
        if name is None:
            self._say("all points are placed. Tab picks a point to re-click; Enter checks and saves.")
            return
        w, h = self.image_size
        if not (-0.5 <= x <= w - 0.5 and -0.5 <= y <= h - 0.5):
            self._say("click inside the picture", "warn")
            return
        self._snapshot()
        seg = self.seg
        seg.points[name] = [round(float(x), 4), round(float(y), 4)]
        seg.point_frames[name] = self.frame
        seg.skipped.discard(name)
        self.selected = name
        self.target = self._next_open(name)
        nxt = f"; next: {self.target}" if self.target else "; all points placed - Enter checks and saves"
        self._say(f"{name} placed at ({x:.1f}, {y:.1f}) on frame {self.frame}{nxt}", "ok")

    def nearest_point(self, x: float, y: float) -> Optional[str]:
        """Name of the placed point nearest to (x, y) within pick_radius (square px)."""
        best, best_d = None, float("inf")
        for name, p in self.seg.points.items():
            d = math.hypot((p[0] - x) * self.sar, p[1] - y)
            if d < best_d:
                best, best_d = name, d
        return best if best_d <= self.pick_radius else None

    def on_mouse(self, kind: str, x: float, y: float) -> None:
        """Mouse event in frame pixel coordinates ('ldown' | 'lup' | 'rdown' | 'move')."""
        if kind in ("ldown", "rdown"):
            self.cancel_armed = False
            if self.confirm is not None:
                self.confirm = None
                self._say("save aborted - keep editing")
        if self.mask_mode:
            self._mask_mouse(kind, x, y)
            return
        if kind in ("ldown", "rdown"):
            self.last_action = "click"
        if kind == "ldown":
            self.message = None
            self.place(x, y)
        elif kind == "rdown":
            self.message = None
            name = self.nearest_point(x, y)
            if name is not None:
                self._delete_point(name)
                return
            for i, m in enumerate(self.seg.masks):
                if m[0] <= x <= m[2] and m[1] <= y <= m[3]:
                    self._snapshot()
                    self.seg.masks.pop(i)
                    self._say("mask deleted")
                    return
            self._say("nothing to delete here (right-click ON a point or inside a mask)")

    # masks
    def _do_mask(self) -> None:
        self.mask_mode = not self.mask_mode
        self.drag = None
        self._say("MASK: drag a box over the date/time burn-in (M or Esc = stop)" if self.mask_mode
                  else "mask mode off")

    def _mask_mouse(self, kind: str, x: float, y: float) -> None:
        if kind == "ldown":
            self.drag = (x, y, x, y)
        elif kind == "move" and self.drag is not None:
            self.drag = (self.drag[0], self.drag[1], x, y)
        elif kind == "lup" and self.drag is not None:
            x0, y0 = min(self.drag[0], x), min(self.drag[1], y)
            x1, y1 = max(self.drag[0], x), max(self.drag[1], y)
            self.drag = None
            if (x1 - x0) * self.sar < 3 or (y1 - y0) < 3:
                self._say("mask too small; drag a box", "warn")
                return
            w, h = self.image_size
            box = [max(-0.5, round(x0, 2)), max(-0.5, round(y0, 2)), min(w - 0.5, round(x1, 2)),
                   min(h - 0.5, round(y1, 2))]
            self._snapshot()
            self.seg.masks.append(box)
            self.mask_mode = False
            self._say(f"mask added ({len(self.seg.masks)} in this segment); right-click inside one deletes it", "ok")
        elif kind == "rdown":
            for i, m in enumerate(self.seg.masks):
                if m[0] <= x <= m[2] and m[1] <= y <= m[3]:
                    self._snapshot()
                    self.seg.masks.pop(i)
                    self._say("mask deleted")
                    return

    # segments
    def _do_segment(self) -> None:
        f = self.frame
        for i, g in enumerate(self.segments):
            if i > 0 and g.start_frame == f:
                self._remove_boundary(i)
                return
        if f == self.seg.start_frame:
            self._say("the first segment starts here; step to a later frame to start a new segment", "warn")
            return
        self._snapshot()
        i = self.seg_index
        old = self.segments[i]
        new = SegmentDraft(start_frame=f, masks=copy.deepcopy(old.masks))
        for name in list(old.points):
            if old.point_frames.get(name, old.start_frame) >= f:
                new.points[name] = old.points.pop(name)
                new.point_frames[name] = old.point_frames.pop(name)
        self.segments.insert(i + 1, new)
        self._reset_target()
        self._say(f"new calibration segment {i + 2} from frame {f}: click its points (G here again removes it)", "ok")

    def _remove_boundary(self, i: int) -> None:
        self._snapshot()
        prev, cur = self.segments[i - 1], self.segments[i]
        lost = [n for n in cur.points if n in prev.points]
        for n in cur.points:
            if n not in prev.points:
                prev.points[n] = cur.points[n]
                prev.point_frames[n] = cur.point_frames.get(n, cur.start_frame)
        self.segments.pop(i)
        self._reset_target()
        msg = f"segment boundary at frame {cur.start_frame} removed"
        if lost:
            msg += f"; its duplicate points {lost} were dropped (U undoes)"
        self._say(msg, "warn" if lost else "info")

    # views
    def _do_toggle_view(self) -> None:
        if not getattr(self.tl, "interlaced", False):
            self.view_mode = "frame"
            self._say("progressive video: field and frame view are the same")
            return
        self.view_mode = "frame" if self.view_mode == "field" else "field"
        self._say("full frame (both fields woven; moving objects show combing)" if self.view_mode == "frame"
                  else "single field, stretched to full height (what the detector sees)")

    def _do_loupe(self) -> None:
        self.loupe = not self.loupe

    def _do_help(self) -> None:
        self.help = not self.help

    # save / cancel
    def _do_cancel(self) -> None:
        if self.help:
            self.help = False
            return
        if self.mask_mode:
            self.mask_mode = False
            self.drag = None
            self._say("mask mode off")
            return
        if self.confirm is not None:
            self.confirm = None
            self._say("save aborted - keep editing")
            return
        if self.cancel_armed:
            self.cancelled = True
            self.done = True
            return
        self.cancel_armed = True
        self._say("press Esc AGAIN to quit WITHOUT saving (any other key continues)", "warn")

    def _do_save(self) -> None:
        if self.confirm is not None:
            self._finish()
            return
        try:
            cf = self.build()
            warns = cf.validate()
        except ValueError as e:
            self._say(f"cannot save yet: {e}", "error")
            return
        warns = warns + self.frame_warnings()
        if warns:
            self.confirm = warns
            self._say("Enter = save anyway | any other key = keep editing", "warn")
            return
        self._finish()

    def _finish(self) -> None:
        cf = self.build()
        warns = self.frame_warnings()
        if warns:   # keep a record of what the user was warned about
            cf.notes = cf.notes + "\n" + "\n".join(WARNING_PREFIX + " " + w for w in warns)
        self.result = cf
        self.done = True
        self.confirm = None

    # ------------------------------------------------------------- output
    def ref_frame_of(self, seg: SegmentDraft) -> int:
        keys = ("rubber", "plate") if self.view == "side" else ("release", "catch")
        for k in keys:
            if k in seg.point_frames:
                return int(seg.point_frames[k])
        if seg.point_frames:
            return int(min(seg.point_frames.values()))
        return int(seg.start_frame)

    def build(self) -> CalibrationFile:
        """The CalibrationFile for the current points (not validated)."""
        segs = []
        for i, g in enumerate(self.segments):
            end = self.segments[i + 1].start_frame if i + 1 < len(self.segments) else None
            segs.append(Segment(start_frame=int(g.start_frame), end_frame=end, ref_frame=self.ref_frame_of(g),
                                view=self.view,
                                points={n: [float(p[0]), float(p[1])] for n, p in g.points.items()},
                                masks=[list(map(float, m)) for m in g.masks]))
        notes = self.user_notes
        frames_line = POINT_FRAMES_PREFIX + " " + json.dumps(
            [{n: int(f) for n, f in sorted(g.point_frames.items())} for g in self.segments])
        notes = (notes + "\n" if notes else "") + frames_line
        return CalibrationFile(settings=dataclasses.replace(self.settings), segments=segs,
                               video=dict(self.video_meta), image_size=self.image_size, notes=notes)

    def _shift(self, a: int, b: int) -> Optional[tuple]:
        if self.measure_shift is None:
            return None
        key = (a, b)
        if key not in self._shift_cache:
            try:
                self._shift_cache[key] = self.measure_shift(a, b)
            except Exception:  # noqa: BLE001 - a failed measurement is just "unknown"
                self._shift_cache[key] = None
        return self._shift_cache[key]

    def frame_warnings(self) -> list[str]:
        """Warnings about points clicked on other frames than the segment's reference frame."""
        out = []
        for i, g in enumerate(self.segments):
            ref = self.ref_frame_of(g)
            others = {n: f for n, f in g.point_frames.items() if f != ref and n in g.points}
            if not others:
                continue
            unknown = []
            for name, f in sorted(others.items(), key=lambda kv: kv[1]):
                sh = self._shift(ref, f)
                if sh is None:
                    unknown.append(f"{name}@{f}")
                    continue
                d = math.hypot(sh[0] * self.sar, sh[1])
                if d > SHIFT_WARN_PX:
                    out.append(f"segment {i + 1}: the camera moved about {d:.1f} px between the reference "
                               f"frame {ref} and frame {f} where '{name}' was clicked; that point may be off "
                               f"by as much (re-click it on frame {ref} if possible)")
            if unknown:
                out.append(f"segment {i + 1}: points clicked on other frames than the reference frame {ref} "
                           f"({', '.join(unknown)}): fine on a tripod, wrong if the camera moved")
        return out

    # ----------------------------------------------------------- display
    def target_text(self) -> str:
        name = self.target
        if name is None:
            return "All points placed. Tab = pick a point to re-click | Enter = check and save"
        req = "required" if self.is_required(name) else "optional, N = skip"
        return f"NEXT: {name.upper()} ({req}) - click {POINT_HELP.get(name, name)}"

    def hint_text(self) -> Optional[str]:
        return EXTRA_HINT.get(self.target) if self.target else None

    def status_items(self) -> list[tuple[str, str]]:
        """(text, level) per point for the HUD status line."""
        seg = self.seg
        out = []
        for n in self.order:
            if n in seg.points:
                mark = "*" if seg.point_frames.get(n) not in (None, self.frame) else ""
                out.append((f"{n} OK{mark}", "ok"))
            elif n in seg.skipped:
                out.append((f"{n} skip", "info"))
            elif self.is_required(n):
                out.append((f"{n} --", "error" if n != self.target else "warn"))
            else:
                out.append((f"{n} (opt)", "info"))
        return out

    def markers(self) -> list[tuple[str, tuple[float, float], str]]:
        """(name, (x, y) frame px, kind) for drawing; kind in 'selected'|'normal'|'other_frame'."""
        out = []
        for n, p in self.seg.points.items():
            if n == self.selected:
                kind = "selected"
            elif self.seg.point_frames.get(n, self.frame) != self.frame:
                kind = "other_frame"
            else:
                kind = "normal"
            out.append((n, (float(p[0]), float(p[1])), kind))
        return out

    def help_rows(self) -> list[tuple[str, str]]:
        rows = [("", "Calibration: click the points the top line asks for (in frame pixels).")]
        rows += [(key_label(BINDINGS[a]), HELP_TEXT[a]) for a in BINDINGS]
        rows += [("L-click", "place the point named on the top line"),
                 ("R-click", "delete the nearest point (or the mask under the cursor)"),
                 ("trackbar", "scrub through the video (frames)"),
                 ("", "Points marked * were clicked on another frame (fine on a tripod).")]
        return rows


# ---------------------------------------------------------------- notes I/O
def _parse_point_frames(notes: str) -> list[dict]:
    for line in (notes or "").splitlines():
        if line.strip().startswith(POINT_FRAMES_PREFIX):
            try:
                data = json.loads(line.strip()[len(POINT_FRAMES_PREFIX):])
                return [{str(k): int(v) for k, v in d.items()} for d in data]
            except (ValueError, TypeError, AttributeError):
                return []
    return []


def _strip_point_frames(notes: str) -> str:
    """User notes without the lines this UI writes (they are regenerated on save)."""
    lines = [ln for ln in (notes or "").splitlines()
             if not ln.strip().startswith((POINT_FRAMES_PREFIX, WARNING_PREFIX))]
    return "\n".join(lines).strip()


def point_frames_from_notes(cf: CalibrationFile) -> list[dict]:
    """Per-segment {point name: frame clicked on} recorded by the calibration UI."""
    return _parse_point_frames(cf.notes)


def estimate_shift(a, b, max_shift: float = 60.0) -> Optional[tuple[float, float]]:
    """Global translation (dx, dy) of gray frame b relative to a (frame px), or None if unclear.

    b(x) ~ a(x - d).  Coarse estimate by phase correlation, then a few
    Lucas-Kanade (translation-only) iterations for sub-pixel accuracy:
    OpenCV's own sub-pixel phase-correlation peak is biased by up to ~2 px
    near half-pixel shifts on smooth images (measured).  NOTE:
    cv2.phaseCorrelate with a window modifies its input arrays in place
    (OpenCV 5.0, measured), so it only ever gets copies here.  A weak
    correlation peak, an ill-conditioned gradient matrix or a refinement
    that runs away returns None ("unknown").
    """
    import cv2
    import numpy as np
    fa = np.asarray(a, dtype=np.float32)
    fb = np.asarray(b, dtype=np.float32)
    if fa.shape != fb.shape or fa.ndim != 2 or min(fa.shape) < 64:
        return None
    fa = cv2.GaussianBlur(fa, (0, 0), 1.0)
    fb = cv2.GaussianBlur(fb, (0, 0), 1.0)
    win = cv2.createHanningWindow((fa.shape[1], fa.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(fa.copy(), fb.copy(), win)
    if not (np.isfinite(dx) and np.isfinite(dy)) or resp < 0.05 or math.hypot(dx, dy) > max_shift:
        return None
    res = _lk_translation(fa, fb, (dx, dy))
    if res is None:
        return None
    d, ncc = res
    # the coarse peak can be ~3 px off on smooth images; LK fixes that, but a far jump or a poor
    # match after alignment means the frames do not show the same scene: unknown
    if math.hypot(d[0] - dx, d[1] - dy) > 6.0 or ncc < 0.5:
        return None
    return float(d[0]), float(d[1])


def _lk_translation(a, b, d0, iters: int = 20, margin: int = 24):
    """Refine translation d (b(x) ~ a(x - d)) by Gauss-Newton on the gradients of a.

    Returns (d, ncc) with ncc the normalized correlation of a and the aligned b, or None.
    """
    import cv2
    import numpy as np
    h, w = a.shape
    sl = (slice(margin, h - margin), slice(margin, w - margin))
    gx = (cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3) / 8.0)[sl]
    gy = (cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3) / 8.0)[sl]
    H = np.array([[np.sum(gx * gx, dtype=np.float64), np.sum(gx * gy, dtype=np.float64)],
                  [np.sum(gx * gy, dtype=np.float64), np.sum(gy * gy, dtype=np.float64)]])
    if not np.all(np.isfinite(H)) or np.linalg.cond(H) > 1e4 or H[0, 0] <= 0:
        return None
    a0 = a[sl]
    d = np.array(d0, dtype=np.float64)
    for _ in range(iters):
        M = np.float32([[1, 0, -d[0]], [0, 1, -d[1]]])          # warped(x) = b(x + d)
        bw = cv2.warpAffine(b, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[sl]
        e = bw - a0
        g = np.array([np.sum(gx * e, dtype=np.float64), np.sum(gy * e, dtype=np.float64)])
        delta = np.linalg.solve(H, g)
        d -= delta
        if not np.all(np.isfinite(d)):
            return None
        if math.hypot(delta[0], delta[1]) < 0.002:
            break
    M = np.float32([[1, 0, -d[0]], [0, 1, -d[1]]])
    bw = cv2.warpAffine(b, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[sl]
    za, zb = a0 - a0.mean(), bw - bw.mean()
    den = float(np.sqrt(np.sum(za * za, dtype=np.float64) * np.sum(zb * zb, dtype=np.float64)))
    ncc = float(np.sum(za * zb, dtype=np.float64)) / den if den > 0 else 0.0
    return d, ncc
