"""Review window logic as a pure state machine (no GUI, no decoding).

`ReviewState` holds the pitches of an AnalysisResult plus the things worth
inspecting by eye, the current position (sample = field), the selected
R/C marker, an unbounded undo stack and the playback state.  It receives
normalized key names (ui/keys.py) and mouse clicks in FRAME PIXEL
coordinates of the current sample; `review/ui.py` is the thin cv2 shell.

ITEMS (PgUp/[ and PgDn/] step through them in time order):
  * every pitch, including rejected ones (shown as rejected; Y restores);
  * every ``possible_missed`` window (X marks it "checked: no pitch",
    N/R at the release field starts a manual pitch there; adding that
    pitch records ``review = 'pitch_added'`` on the window);
  * optionally (O toggles) the finder's rejected tracks with their reason
    code; Y promotes one that has release/catch times to a pitch.
Nothing is ever deleted: rejecting keeps the pitch with status
'rejected', dismissing a window records ``review = 'no_pitch'`` in its
dict, and the undo stack can revert everything.  The window's ``review``
key is what results.csv and the annotated video use to show it as checked.

EDITS AND STATUSES.  'auto' becomes 'edited' on the first edit (R/C
change, ball point added/removed, re-track); 'manual' pitches stay
'manual'; 'rejected' pitches stay rejected until accepted (Y), which
restores the status they had before.  Each edit is appended to
``Pitch.edit_history`` (action, details, local time), then
``recompute(pitch)`` re-estimates the speed (if it raises, the pitch keeps
NO speed and the reason says why) and ``save(result)`` autosaves (errors
are shown, the session continues, the next edit retries).

MANUAL MARKS (DESIGN.md / estimate/events.py semantics, shown in the HUD):
  R = the FIRST field in which the ball is out of the hand,
  C = the FIRST field in which the ball is in the glove.
A mark becomes an EventTime through ``make_event(s, kind)``, by default
`pitchspeed.estimate.events.manual_event` (t = field time - dt/2,
sigma = dt/sqrt(12), method 'manual', s = the marked field).  A C mark is
always the glove: the speed estimate then ends the flight at the catch
point even if the automatic track ended at a possible bat contact
(end_kind 'contact'; the info line says where the flight ends and C on such
a pitch warns, U undoes).  A ball the batter hit has no catch to mark.

FULL-FRAME VIEW (F, interlaced video): one picture shows two fields, so
Left/Right step one frame (2 fields) and marks (R, C, N, nudges, ball-point
clicks) are refused with "press F": a mark in the woven picture would get
whichever of the two fields happens to be current, a hidden 16.7 ms error.

NEW MANUAL PITCH: N (or R where no pitch is selected) marks the release
at the current field; step to the catch and press C.  Esc cancels the
half-made pitch.

NUMBERING.  After every change pitches are sorted by release time and
renumbered with the same rule as `pitchspeed.pipeline.renumber`:
accepted / edited / manual / auto pitches get 1..n chronologically,
rejected pitches get number 0 (the HUD shows "rejected, was #k" from the
edit history).  So numbers can shift when a pitch is added or rejected;
the table and the annotated video are always made from the final state.

BALL POINTS.  Left-click adds a manual ball point at the current field
(replacing an automatic point there, which is kept but excluded with
``inlier=False``); right-click near a point at the current field deletes a
manual point or toggles an automatic one in/out.  Points are stored in
stabilized coordinates (the calibration reference frame) using the
per-sample transforms the finder used (`StabilizeMap`: 3x3 homographies,
or 2x3 affine transforms from older files), like the automatic track.

KEYS: see BINDINGS / HELP_TEXT.  Navigation follows the calibration
window: Left/A Right/D one field (one frame in the full-frame view),
Up/W Down/S ten fields back/forward, J/L one second,
PgUp/[ PgDn/] previous/next item, Home/End release/catch.

MARK MODE (``mode='mark'``: `pitchspeed mark`, pitches marked by hand on a
long tape).  The same keys and rules, plus:
  * the timeline bar always covers the whole video (a click jumps there) and
    shows every marked pitch as a tick;
  * the "current" pitch is the one within PITCH_MARGIN fields of the shown
    field (or none): R/C/B/X/Y act on it, so a key pressed far from every
    pitch can never move a mark seconds away; N starts a new pitch anywhere;
  * Space plays forward without looping, at 1/8x .. 8x (+ / -; above 1x
    several fields are skipped per picture), and stops at the end.
"""
from __future__ import annotations

import copy
import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from pitchspeed.types import AnalysisResult, EventTime, Pitch, PitchTrack, SpeedEstimate, TrackPoint
from pitchspeed.ui.fmt import format_time
from pitchspeed.ui.keys import key_label, resolve

SPEEDS = (0.125, 0.25, 0.5, 1.0)
SPEED_LABELS = ("1/8x", "1/4x", "1/2x", "1x")
MARK_SPEEDS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
MARK_SPEED_LABELS = ("1/8x", "1/4x", "1/2x", "1x", "2x", "4x", "8x")
MODES = ("review", "mark")
MARK_START = ("MARK PITCHES: Space plays/pauses (+ faster, - slower), J/L = 1 second, click the bar to jump. "
              "At each pitch: pause, step with A/D, N at the release, then C at the glove (B if hit)")
PITCH_MARGIN = 60          # fields shown before the release / after the catch (io_ux.md 4)
WINDOW_MARGIN = 30         # fields around a possible-missed window or rejected track
POINT_TOLERANCE = 3        # ball points may be added this many fields outside R..C

MANUAL_HINT = ("R = FIRST field with the ball out of the hand | C = FIRST field with the ball in the glove"
               " | B = FIRST field with the ball on/off the bat (if the batter hit it)")

BINDINGS: dict[str, list[str]] = {
    "prev_field": ["left", "a"],
    "next_field": ["right", "d"],
    "back_10": ["up", "w"],
    "fwd_10": ["down", "s"],
    "back_1s": ["j"],
    "fwd_1s": ["l"],
    "prev_item": ["pgup", "["],
    "next_item": ["pgdn", "]"],
    "goto_release": ["home"],
    "goto_catch": ["end"],
    "set_release": ["r"],
    "set_catch": ["c"],
    "set_contact": ["b"],
    "toggle_marker": ["tab"],
    "nudge_back": [",", "<"],
    "nudge_fwd": [".", ">"],
    "accept": ["y", "enter"],
    "reject": ["x", "del"],
    "new_manual": ["n"],
    "retrack": ["e"],
    "play": ["space"],
    "faster": ["+", "="],
    "slower": ["-", "_"],
    "track": ["t"],
    "loupe": ["z"],
    "toggle_view": ["f"],
    "flip_order": ["k"],
    "others": ["o"],
    "help": ["h", "?", "f1"],
    "undo": ["u", "backspace"],
    "escape": ["esc"],
    "quit": ["q"],
}

HELP_TEXT: dict[str, str] = {
    "prev_field": "back 1 field (1 frame in the full-frame view)",
    "next_field": "forward 1 field (1 frame in the full-frame view)",
    "back_10": "back 10 fields",
    "fwd_10": "forward 10 fields",
    "back_1s": "back 1 second",
    "fwd_1s": "forward 1 second",
    "prev_item": "previous pitch / window to check",
    "next_item": "next pitch / window to check",
    "goto_release": "go to the release (R)",
    "goto_catch": "go to the catch (C)",
    "set_release": "release = this field (FIRST field with the ball out of the hand)",
    "set_catch": "catch = this field (FIRST field with the ball in the glove; if the batter hit the ball, use B)",
    "set_contact": "bat contact = this field (FIRST field with the ball touching or just off the bat); "
                   "use instead of C when the batter hit the ball",
    "toggle_marker": "select the R or the C marker for nudging",
    "nudge_back": "move the selected marker 1 field earlier",
    "nudge_fwd": "move the selected marker 1 field later",
    "accept": "accept this pitch (restores a rejected one) and go to the next",
    "reject": "reject (kept as 'rejected') / mark a window 'no pitch'",
    "new_manual": "new manual pitch: release here, then C at the catch (or B at the bat contact)",
    "retrack": "re-run the ball tracker between R and C",
    "play": "play / pause (loops around the pitch)",
    "faster": "faster playback (up to 1x)",
    "slower": "slower playback (down to 1/8x)",
    "track": "show / hide the ball track",
    "loupe": "8x loupe at the mouse",
    "toggle_view": "bobbed field / full frame view (marks only in the field view)",
    "flip_order": "marking by hand: switch the field order (if motion jumps back and forth when you "
                  "step field by field)",
    "others": "also list the tracks the finder rejected",
    "help": "this help",
    "undo": "undo (unlimited)",
    "escape": "close help / cancel a half-made pitch / else save and quit",
    "quit": "save and quit",
}

_KEEP = object()


# ------------------------------------------------------------- helpers ----
def now_text() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


METHOD_NAMES = {"tof": "TOF", "side_fit": "SIDE", "tof+side_fit": "TOF+SIDE", "manual_tof": "MANUAL TOF",
                "none": "-"}


def speed_text(sp: Optional[SpeedEstimate]) -> str:
    """'78+/-3 mph (avg 74)', '~71+/-6 mph' for poor, 'no speed: <reason>'."""
    if sp is None or sp.v_release_mph is None or not math.isfinite(sp.v_release_mph):
        reason = getattr(sp, "reason", None) if sp is not None else None
        return "no speed" + (f": {reason}" if reason else "")
    pre = "~" if sp.label == "poor" else ""
    t = f"{pre}{sp.v_release_mph:.0f}"
    if sp.sigma_mph is not None and math.isfinite(sp.sigma_mph):
        t += f"+/-{sp.sigma_mph:.1f}" if sp.sigma_mph < 2 else f"+/-{sp.sigma_mph:.0f}"
    t += " mph"
    if sp.v_avg_mph is not None and math.isfinite(sp.v_avg_mph):
        t += f" (avg {sp.v_avg_mph:.0f})"
    return t


def renumber(pitches: list[Pitch]) -> list[Pitch]:
    """Sort by release time; number non-rejected pitches 1..n, rejected ones 0.

    Same rule as pitchspeed.pipeline.renumber (kept here so review does
    not depend on the pipeline module).
    """
    pitches.sort(key=lambda p: (p.release.t if p.release is not None else math.inf))
    n = 0
    for p in pitches:
        if p.status == "rejected":
            p.number = 0
        else:
            n += 1
            p.number = n
    return pitches


def default_make_event(timeline) -> Callable[[int, str], EventTime]:
    """EventTime factory for a marked field using estimate.events.manual_event."""
    try:
        from pitchspeed.estimate.events import manual_event
    except ImportError:  # pragma: no cover - estimate area missing
        manual_event = None

    def make(s: int, kind: str) -> EventTime:
        t = float(timeline.sample_time(int(s)))
        dt = float(timeline.sample_dt)
        if manual_event is not None:
            return manual_event(t, dt, kind, s=int(s))
        return EventTime(t=t - 0.5 * dt, sigma=dt / math.sqrt(12.0), method="manual", s=int(s))

    return make


def _as_3x3(transforms) -> np.ndarray:
    """(n, 3, 3) float array from (n, 3, 3) homographies or (n, 2, 3) affine transforms."""
    T = np.asarray(transforms, dtype=np.float64)
    shape = T.shape[-2:] if T.ndim >= 2 else ()
    if T.ndim in (2, 3) and shape == (3, 3):
        return T.reshape(-1, 3, 3).copy()
    if (T.ndim in (2, 3) and shape == (2, 3)) or (T.ndim == 2 and T.shape[1] == 6) or T.size == 0:
        A = T.reshape(-1, 2, 3)
        H = np.zeros((len(A), 3, 3))
        H[:, :2, :] = A
        H[:, 2, 2] = 1.0
        return H
    raise ValueError(f"transforms must have shape (n, 2, 3) or (n, 3, 3), not {T.shape}")


class StabilizeMap:
    """Per-sample transforms (sample frame coords -> reference frame coords).

    Accepts the (n, 3, 3) homographies that `pipeline.load_transforms`
    returns when transforms.npz has 'transforms_3x3' (the transforms the
    finder mapped its track points with) or the (n, 2, 3) affine set of
    older files.  Points are mapped projectively, v = H [x, y, 1] ->
    (v0 / v2, v1 / v2), so manual ball points are stored, and automatic
    ones drawn, in exactly the finder's coordinates.

    Missing samples use the nearest transform within `max_gap` samples,
    else the identity (no stabilization information); so does a sample
    whose matrix is degenerate (not invertible, or maps the point to
    infinity).
    """

    def __init__(self, transforms=None, samples=None, max_gap: int = 3):
        self.T = None if transforms is None else _as_3x3(transforms)
        if self.T is not None and samples is None:
            samples = np.arange(len(self.T))
        self.S = None if samples is None else np.asarray(samples, dtype=np.int64).reshape(-1)
        if self.T is not None and len(self.S) != len(self.T):
            raise ValueError("transforms and transform_samples differ in length")
        if self.S is not None and len(self.S) and np.any(np.diff(self.S) < 0):
            order = np.argsort(self.S, kind="stable")
            self.S, self.T = self.S[order], self.T[order]
        self.max_gap = int(max_gap)
        self._inv: dict[int, Optional[np.ndarray]] = {}

    def _index(self, s: int) -> Optional[int]:
        if self.T is None or self.S is None or len(self.S) == 0:
            return None
        j = int(np.searchsorted(self.S, s))
        best = None
        for c in (j - 1, j):
            if 0 <= c < len(self.S):
                d = abs(int(self.S[c]) - int(s))
                if d <= self.max_gap and (best is None or d < best[0]):
                    best = (d, c)
        return None if best is None else best[1]

    def matrix(self, s: int) -> Optional[np.ndarray]:
        """The 3x3 matrix used for sample s (None: identity)."""
        i = self._index(s)
        return None if i is None else self.T[i]

    def _inverse(self, i: int) -> Optional[np.ndarray]:
        if i not in self._inv:
            try:
                Mi = np.linalg.inv(self.T[i])
                self._inv[i] = Mi if np.all(np.isfinite(Mi)) else None
            except np.linalg.LinAlgError:
                self._inv[i] = None
        return self._inv[i]

    @staticmethod
    def _apply(M: Optional[np.ndarray], x: float, y: float) -> tuple[float, float]:
        if M is None:
            return float(x), float(y)
        v = M @ np.array([float(x), float(y), 1.0])
        w = float(v[2])
        if not (math.isfinite(w) and abs(w) > 1e-12):
            return float(x), float(y)
        return float(v[0] / w), float(v[1] / w)

    def to_ref(self, s: int, x: float, y: float) -> tuple[float, float]:
        i = self._index(s)
        return self._apply(None if i is None else self.T[i], x, y)

    def from_ref(self, s: int, x: float, y: float) -> tuple[float, float]:
        i = self._index(s)
        return self._apply(None if i is None else self._inverse(i), x, y)


@dataclass
class _Entry:
    uid: int
    pitch: Pitch


@dataclass(frozen=True)
class ItemKey:
    kind: str       # 'pitch' | 'missed' | 'rtrack'
    ref: int        # pitch uid, or index into possible_missed / rejected_tracks


@dataclass
class Item:
    key: ItemKey
    s0: int
    s1: int


@dataclass
class Message:
    text: str
    level: str = "info"


@dataclass
class _Undo:
    ops: list                    # list of tuples, applied in reverse
    item: Optional[ItemKey]
    s: int
    label: str
    draft: Optional[dict] = field(default=None)


# ------------------------------------------------------------------ state --
class ReviewState:
    """Pure review state machine (see module docstring).

    timeline: ``n_samples``, ``sample_dt``, ``locate(s)``, ``sample_time(s)``
        (and optionally ``sample_at_time(t)``) - a SampleSource.
    recompute(pitch) -> pitch, save(result) -> None: the lead's callbacks.
    retrack(pitch) -> PitchTrack: optional constrained re-tracking.
    make_event(s, kind) -> EventTime: optional, default manual_event.
    stab: StabilizeMap for ball points (default: identity).
    """

    def __init__(self, result: AnalysisResult, timeline, recompute: Callable[[Pitch], Pitch],
                 save: Callable[[AnalysisResult], None], retrack: Optional[Callable[[Pitch], PitchTrack]] = None,
                 make_event: Optional[Callable[[int, str], EventTime]] = None,
                 stab: Optional[StabilizeMap] = None, mode: str = "review"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.speeds = MARK_SPEEDS if mode == "mark" else SPEEDS
        self.speed_labels = MARK_SPEED_LABELS if mode == "mark" else SPEED_LABELS
        self.result = result
        self.tl = timeline
        self.recompute_cb = recompute
        self.save_cb = save
        self.retrack_cb = retrack
        self.make_event = make_event or default_make_event(timeline)
        self.stab = stab or StabilizeMap()
        self._next_uid = 0
        self._error_said = False
        self.entries: list[_Entry] = []
        for p in result.pitches:
            self._add_entry(p)
        self.entries.sort(key=lambda e: self._t_release(e.pitch))
        self.n_samples = int(timeline.n_samples)
        self.s = 0
        self.cur: Optional[ItemKey] = None
        self.marker = "R"
        self.draft: Optional[dict] = None
        self.playing = False
        self.speed_i = self.speeds.index(1.0) if mode == "mark" else 1
        self.show_track = True
        self.show_others = False
        self.loupe = False
        self.help = False
        self.view_mode = "field" if getattr(timeline, "interlaced", False) else "frame"
        self.undo_stack: list[_Undo] = []
        self.message: Optional[Message] = None
        self.save_error: Optional[str] = None
        self.quit_armed = False
        self.done = False
        self.pending: Optional[tuple] = None
        self.pick_radius = 10.0          # frame rows; set by the shell from the display scale
        self.last_action = ""
        self.frames_changed = False      # set when the pictures of the samples changed (K)
        items = self.items()
        if self.mode == "mark":
            sc = getattr(result, "scan", None)
            if getattr(timeline, "interlaced", False) and sc is not None and sc.confidence == "low":
                self._say("FIELD ORDER UNCERTAIN: step through some motion with D; if it jumps back and forth, "
                          "press K to switch it. Then: Space plays/pauses; paused, N at a release, C at the catch",
                          "warn")
            else:
                self._say(MARK_START, "ok")
        elif items:
            self._goto_item(items[0].key)
        else:
            self._say("no pitches found: step through the video and press N at a release to add one", "warn")

    # ----------------------------------------------------------- basics
    def _add_entry(self, p: Pitch) -> _Entry:
        e = _Entry(self._next_uid, p)
        self._next_uid += 1
        self.entries.append(e)
        return e

    def _say(self, text: str, level: str = "info") -> None:
        """Set the HUD message; within one action an error is not overwritten by later news."""
        if self._error_said and level != "error":
            return
        self.message = Message(text, level)
        if level == "error":
            self._error_said = True

    def _clamp(self, s: int) -> int:
        return int(min(max(int(s), 0), max(0, self.n_samples - 1)))

    def ev_s(self, ev: Optional[EventTime]) -> Optional[int]:
        """Sample index of an event (its `s`, else the sample nearest its time)."""
        if ev is None:
            return None
        if ev.s is not None:
            return int(ev.s)
        if not math.isfinite(ev.t):
            return None
        if hasattr(self.tl, "sample_at_time"):
            return int(self.tl.sample_at_time(ev.t))
        return self._clamp(int(round(ev.t / float(self.tl.sample_dt))))

    @staticmethod
    def _t_release(p: Pitch) -> float:
        return p.release.t if p.release is not None and math.isfinite(p.release.t) else math.inf

    def entry(self, uid: int) -> Optional[_Entry]:
        for e in self.entries:
            if e.uid == uid:
                return e
        return None

    @property
    def pitches(self) -> list[Pitch]:
        return [e.pitch for e in self.entries]

    # ------------------------------------------------------------ items
    def items(self) -> list[Item]:
        out = []
        for e in self.entries:
            r, c = self.ev_s(e.pitch.release), self.ev_s(e.pitch.catch)
            r = r if r is not None else (c if c is not None else 0)
            c = c if c is not None else r
            out.append(Item(ItemKey("pitch", e.uid), r, c))
        for i, m in enumerate(self.result.possible_missed or []):
            a, b = self._window_of(m)
            out.append(Item(ItemKey("missed", i), a, b))
        if self.show_others:
            for i, tr in enumerate(self.result.rejected_tracks or []):
                if tr is None:
                    continue
                ss = [p.s for p in tr.points] or [self.ev_s(tr.release) or 0]
                out.append(Item(ItemKey("rtrack", i), int(min(ss)), int(max(ss))))
        kind_order = {"pitch": 0, "missed": 1, "rtrack": 2}
        out.sort(key=lambda it: (it.s0, kind_order[it.key.kind], it.key.ref))
        return out

    def _window_of(self, m: dict) -> tuple[int, int]:
        a = m.get("s_start")
        b = m.get("s_end")
        if a is None and m.get("t_start") is not None:
            a = self.ev_s(EventTime(float(m["t_start"]), 0.0, "window"))
        if b is None and m.get("t_end") is not None:
            b = self.ev_s(EventTime(float(m["t_end"]), 0.0, "window"))
        a = self._clamp(a if a is not None else 0)
        b = self._clamp(b if b is not None else a)
        return a, max(a, b)

    def item_index(self) -> Optional[int]:
        items = self.items()
        self._sync_cur(items)
        for i, it in enumerate(items):
            if it.key == self.cur:
                return i
        return None

    @property
    def marking(self) -> bool:
        return self.mode == "mark"

    def _pitch_key_near(self, items: list[Item]) -> Optional[ItemKey]:
        """Mark mode: the item within PITCH_MARGIN fields of the shown field (the nearest; a pitch
        wins a tie), else None.  Possible-missed windows and listed rejected tracks count too, so
        X / Y work on them where they are shown."""
        rank = {"pitch": 0, "missed": 1, "rtrack": 2}
        best = None
        for it in items:
            if it.key.kind == "missed" and not self._navigable(it):
                continue                      # already handled: nothing to do with it here
            d = 0 if it.s0 <= self.s <= it.s1 else min(abs(self.s - it.s0), abs(self.s - it.s1))
            if d <= PITCH_MARGIN and (best is None or (d, rank[it.key.kind]) < best[0]):
                best = ((d, rank[it.key.kind]), it.key)
        return best[1] if best is not None else None

    def _sync_cur(self, items: Optional[list[Item]] = None) -> None:
        """Mark mode: the current item always follows the shown field (see module doc)."""
        if self.marking:
            self.cur = self._pitch_key_near(self.items() if items is None else items)

    def current_item(self) -> Optional[Item]:
        items = self.items()
        self._sync_cur(items)
        for it in items:
            if it.key == self.cur:
                return it
        return None

    def current_entry(self) -> Optional[_Entry]:
        self._sync_cur()
        if self.cur is not None and self.cur.kind == "pitch":
            return self.entry(self.cur.ref)
        return None

    def current_pitch(self) -> Optional[Pitch]:
        e = self.current_entry()
        return e.pitch if e is not None else None

    def _goto_item(self, key: Optional[ItemKey], where: str = "start") -> None:
        # Find the item by key (not via current_item(): in mark mode that follows the shown
        # field, which only moves to the item below).
        it = next((x for x in self.items() if x.key == key), None) if key is not None else None
        self.cur = key
        if it is None:
            return
        self.s = self._clamp(it.s1 if where == "end" else it.s0)
        self.marker = "R"
        self.playing = False
        self._say_arrival(it)

    def _say_arrival(self, it: Item) -> None:
        """Replace any old message with what this item is and what to do with it."""
        if it.key.kind == "pitch":
            e = self.entry(it.key.ref)
            if e is not None:
                p = e.pitch
                what = "rejected pitch (Y restores it)" if p.status == "rejected" else f"pitch {p.number}"
                self._say(f"{what}: {speed_text(p.speed)} - Y accepts, R/C (or B) fix the marks, X rejects")
        elif it.key.kind == "missed":
            m = (self.result.possible_missed or [])[it.key.ref]
            why = m.get("reason", "") if isinstance(m, dict) else ""
            self._say(f"possible missed pitch{(' (' + why[:60] + ')') if why else ''}: if there is a pitch, "
                      f"press N at its release then C (or B); if not, press X")
        else:
            self._say("rejected track: Y makes it a pitch, X confirms the rejection")

    def window(self) -> tuple[int, int]:
        """Sample range [w0, w1] worth showing / looping for the current item or draft
        (mark mode: always the whole video)."""
        if self.marking:
            return 0, max(0, self.n_samples - 1)
        it = self.current_item()
        if self.draft is not None:
            a = self.draft["s_release"]
            return self._clamp(a - PITCH_MARGIN), self._clamp(max(self.s, a) + PITCH_MARGIN)
        if it is None:
            return self._clamp(self.s - PITCH_MARGIN), self._clamp(self.s + PITCH_MARGIN)
        a, b = it.s0, it.s1
        margin = PITCH_MARGIN if it.key.kind == "pitch" else WINDOW_MARGIN
        p = self.current_pitch()
        if p is not None and p.track is not None and p.track.points:
            a = min(a, min(pt.s for pt in p.track.points))
            b = max(b, max(pt.s for pt in p.track.points))
        return self._clamp(a - margin), self._clamp(b + margin)

    def prefetch_range(self) -> Optional[tuple[int, int]]:
        """Samples worth decoding ahead: the item window, or a fixed span after a draft's release."""
        if self.draft is not None:
            a = self.draft["s_release"]
            return self._clamp(a - PITCH_MARGIN), self._clamp(a + 4 * PITCH_MARGIN) + 1
        it = self.current_item()
        if it is None:
            return None
        if self.marking:
            return self._clamp(it.s0 - PITCH_MARGIN), self._clamp(it.s1 + PITCH_MARGIN) + 1
        w0, w1 = self.window()
        return w0, w1 + 1

    def _nearest_item_key(self) -> Optional[ItemKey]:
        items = self.items()
        if not items:
            return None
        best = min(items, key=lambda it: abs(it.s0 - self.s))
        return best.key

    # ------------------------------------------------------------ input
    def on_key(self, name: Optional[str]) -> bool:
        action = resolve(name, BINDINGS)
        if action is None:
            return False
        self.on_action(action)
        return True

    def on_action(self, action: str) -> None:
        """Run one action from BINDINGS (messages persist until replaced)."""
        self._error_said = False
        if self.quit_armed and action != "quit":
            self.quit_armed = False
            self._say("quit aborted")
        getattr(self, "_do_" + action)()
        self.last_action = action

    # navigation
    def _step(self, n: int) -> None:
        self.s = self._clamp(self.s + n)
        self.playing = False

    def goto(self, s: int) -> None:
        """Jump to sample s (e.g. a click on the timeline bar)."""
        self.s = self._clamp(s)
        self.playing = False

    def woven(self) -> bool:
        """True in the full-frame view of interlaced video (one picture = two fields)."""
        return self.view_mode == "frame" and bool(getattr(self.tl, "interlaced", False))

    def _field_step(self) -> int:
        """Samples per Left/Right press: 2 (one frame) in the full-frame view, else 1."""
        return 2 if self.woven() else 1

    def _paused_for_mark(self, key: str) -> bool:
        """A mark made while the video plays lands on whatever field is on screen (at 2x-8x
        several fields are skipped), so its +/- would be too small: pause instead and ask the
        user to step to the exact field.  True = paused, nothing marked."""
        if not self.playing:
            return False
        self.playing = False
        self._say(f"paused - step with A/D (one field) to the exact field, then press {key} again", "warn")
        return True

    def _single_field_needed(self, what: str = "mark") -> bool:
        """Refuse a mark in the full-frame view (True = refused, message shown).

        A woven frame shows two fields 1/60 s apart, so the user cannot see
        which of them a mark or ball point would get: a hidden +-1 field
        error (16.7 ms, ~3.7 % of a 90 mph flight) that the marks' sigma
        (one field) does not cover.  Marks are only made in the field view.
        """
        if not self.woven():
            return False
        self._say(f"full-frame view shows two fields at once: press F for the single-field view to {what}",
                  "warn")
        return True

    def _do_prev_field(self) -> None:
        self._step(-self._field_step())

    def _do_next_field(self) -> None:
        self._step(self._field_step())

    def _do_back_10(self) -> None:
        self._step(-10)

    def _do_fwd_10(self) -> None:
        self._step(10)

    def _one_second(self) -> int:
        return max(1, int(round(1.0 / float(self.tl.sample_dt))))

    def _do_back_1s(self) -> None:
        self._step(-self._one_second())

    def _do_fwd_1s(self) -> None:
        self._step(self._one_second())

    def _do_prev_item(self) -> None:
        self._move_item(-1)

    def _do_next_item(self) -> None:
        self._move_item(1)

    def _navigable(self, it: Item) -> bool:
        """PgUp/PgDn stop here?  Windows already handled ('no pitch' / pitch added) are
        skipped unless O (show everything) is on; they stay in the results."""
        if it.key.kind != "missed" or self.show_others:
            return True
        m = (self.result.possible_missed or [])[it.key.ref]
        return not (isinstance(m, dict) and m.get("review") in ("pitch_added", "no_pitch"))

    def _move_item(self, step: int) -> None:
        items = self.items()
        if not items:
            self._say("no pitches marked yet: press N at a release to add one" if self.marking else
                      "nothing to review: press N at a release to add a pitch")
            return
        if self.marking:
            # relative to the shown field: PgDn = the next item starting after it, PgUp = the
            # last one starting before it (from inside a pitch: back to its release)
            cands = [it for it in items if (it.s0 > self.s if step > 0 else it.s0 < self.s)
                     and self._navigable(it)]
            if not cands:
                self._say_end(step)
                return
            self._goto_item((cands[0] if step > 0 else cands[-1]).key)
            return
        i = self.item_index()
        if i is None:
            # not on an item: go to the first one after / before the current field
            cands = [k for k, it in enumerate(items) if (it.s0 > self.s if step > 0 else it.s0 < self.s)
                     and self._navigable(it)]
            if not cands:
                self._say_end(step)
                return
            j = cands[0] if step > 0 else cands[-1]
        else:
            j = i + step
            while 0 <= j < len(items) and not self._navigable(items[j]):
                j += step
            if not 0 <= j < len(items):
                self._say_end(step)
                return
        self._goto_item(items[j].key)

    def _move_then_say(self, done: str) -> None:
        """Go to the next item; show what was just done followed by where we are now."""
        self._move_item(1)
        now = self.message.text if self.message else ""
        level = self.message.level if self.message else "ok"
        self._say(f"{done} | {now}" if done and now and now != done else (done or now), level)

    def _say_end(self, step: int) -> None:
        if self.marking:
            self._say("no marked pitch after this point - Space plays on to find more; Q saves and quits"
                      if step > 0 else "no marked pitch before this point")
            return
        if step > 0:
            self._say("that was the last one - everything is checked. Press Q to save and quit "
                      "(PgUp goes back)", "ok")
        else:
            self._say("this is the first item (PgDn goes forward)")

    def _do_goto_release(self) -> None:
        if self.draft is not None:
            self.s = self._clamp(self.draft["s_release"])
            return
        it = self.current_item()
        if it is not None:
            self.s = self._clamp(it.s0)
            self.marker = "R"
        self.playing = False

    def _do_goto_catch(self) -> None:
        it = self.current_item()
        if it is not None and self.draft is None:
            self.s = self._clamp(it.s1)
            self.marker = "C"
        self.playing = False

    # playback
    def _do_play(self) -> None:
        self.playing = not self.playing
        if self.playing and self.marking:
            if self.s >= self.n_samples - 1:
                self.playing = False
                self._say("end of the video (click the bar or press PgUp to go back)")
            return
        if self.playing:
            w0, w1 = self.window()
            if not w0 <= self.s <= w1:
                self.s = w0

    def _do_faster(self) -> None:
        self.speed_i = min(len(self.speeds) - 1, self.speed_i + 1)
        self._say(f"playback {self.speed_labels[self.speed_i]}")

    def _do_slower(self) -> None:
        self.speed_i = max(0, self.speed_i - 1)
        self._say(f"playback {self.speed_labels[self.speed_i]}")

    @property
    def play_speed(self) -> float:
        return self.speeds[self.speed_i]

    @property
    def play_step(self) -> int:
        """Fields advanced per shown picture: 1 up to 1x, else the speed (2x shows every 2nd field)."""
        return max(1, int(round(self.play_speed))) if self.play_speed > 1.0 else 1

    def tick(self) -> None:
        """Advance playback: review mode loops one field at a time inside window();
        mark mode runs forward by play_step fields and stops at the end."""
        if not self.playing:
            return
        if self.marking:
            last = max(0, self.n_samples - 1)
            self.s = min(last, self.s + self.play_step)
            if self.s >= last:
                self.playing = False
                self._say("end of the video (Q saves and quits)")
            return
        w0, w1 = self.window()
        self.s = w0 if self.s >= w1 or self.s < w0 else self.s + 1

    # toggles
    def _do_track(self) -> None:
        self.show_track = not self.show_track

    def _do_loupe(self) -> None:
        self.loupe = not self.loupe

    def _do_help(self) -> None:
        self.help = not self.help

    def _do_toggle_view(self) -> None:
        if not getattr(self.tl, "interlaced", False):
            self._say("progressive video: field and frame view are the same")
            return
        self.view_mode = "frame" if self.view_mode == "field" else "field"
        if self.view_mode == "frame":
            self._say("full frame (two fields woven): Left/Right step one frame; marks need the field view "
                      "(F again)")
        else:
            self._say("single field: Left/Right step one field")

    def _do_others(self) -> None:
        n = len(self.result.rejected_tracks or [])
        self.show_others = not self.show_others
        if not self.show_others and self.cur is not None and self.cur.kind == "rtrack":
            self.cur = self._nearest_item_key()
        self._say(f"listing {n} rejected track(s) too" if self.show_others else "rejected tracks hidden")

    # ------------------------------------------------------------ edits
    def _push(self, ops: list, label: str, draft_before=_KEEP) -> None:
        """Record an undo step; `draft_before` is the draft to restore (default: the current one)."""
        d = self.draft if draft_before is _KEEP else draft_before
        self.undo_stack.append(_Undo(ops=ops, item=self.cur, s=self.s, label=label, draft=copy.deepcopy(d)))

    def _edit(self, e: _Entry, action: str, mutate: Callable[[Pitch], None], recompute: bool = True,
              **info) -> None:
        before = copy.deepcopy(e.pitch)
        mutate(e.pitch)
        if e.pitch.status == "auto":
            e.pitch.status = "edited"
        e.pitch.edit_history.append({"action": action, "at": now_text(), **info})
        self._push([("replace", e.uid, before)], action)
        if recompute:
            self._recompute(e)
        self._commit()

    def _recompute(self, e: _Entry) -> None:
        try:
            new = self.recompute_cb(e.pitch)
            if new is not None:
                e.pitch = new
        except Exception as ex:  # noqa: BLE001 - shown to the user, never hidden
            e.pitch.speed = SpeedEstimate(v_release_mph=None, reason=f"speed could not be recomputed: {ex}")
            self._say(f"speed could not be recomputed: {ex}", "error")

    def _commit(self) -> None:
        """Sort + renumber, write the list back into the result and autosave."""
        self.entries.sort(key=lambda e: self._t_release(e.pitch))
        self.result.pitches[:] = [e.pitch for e in self.entries]
        renumber(self.result.pitches)
        self._save()
        # save() may re-sort result.pitches (pipeline.renumber); follow its order
        order = {id(p): i for i, p in enumerate(self.result.pitches)}
        if all(id(e.pitch) in order for e in self.entries):
            self.entries.sort(key=lambda e: order[id(e.pitch)])

    def _save(self) -> None:
        try:
            self.save_cb(self.result)
            if self.save_error:
                self._say("autosave works again", "ok")
            self.save_error = None
        except Exception as ex:  # noqa: BLE001
            self.save_error = str(ex)
            self._say(f"AUTOSAVE FAILED: {ex} (will retry on the next edit and on quit)", "error")

    def _need_pitch(self) -> Optional[_Entry]:
        e = self.current_entry()
        if e is None:
            self._say("no pitch selected here: press N at the release field to add one", "warn")
        return e

    def _set_event(self, kind: str, s: int, point: Optional[str] = None) -> None:
        e = self._need_pitch()
        if e is None:
            return
        p = e.pitch
        r, c = self.ev_s(p.release), self.ev_s(p.catch)
        s = self._clamp(s)
        if kind == "release" and c is not None and s >= c:
            self._say(f"release must be before the catch (field {c})", "error")
            return
        if kind == "catch" and r is not None and s <= r:
            self._say(f"catch must be after the release (field {r})", "error")
            return
        old = r if kind == "release" else c
        old_ev = p.release if kind == "release" else p.catch
        if old == s and old_ev.method == "manual" and getattr(old_ev, "point", None) == point:
            self._say(f"{'bat contact' if point == 'contact' else kind} is already at field {s}")
            return
        ev = self.make_event(s, kind)
        ev.point = point if kind == "catch" else None
        frame = int(self.tl.locate(s)[0])

        def mutate(pp: Pitch) -> None:
            if kind == "release":
                pp.release, pp.frame_release = ev, frame
            else:
                pp.catch, pp.frame_catch = ev, frame

        self._edit(e, "set_" + kind, mutate, from_s=old, to_s=s)
        self.marker = "R" if kind == "release" else "C"
        p = self.current_pitch()
        if point == "contact":
            self._say(f"bat contact set to field {s}: {speed_text(p.speed if p else None)}", "ok")
            return
        if kind == "catch" and p is not None and _track_end(p) == "contact":
            # A C mark is the glove (estimate.events): the speed now uses the glove distance,
            # about 6 % more than the bat-contact distance the automatic track ended at.
            self._say(f"catch set to field {s} = ball IN THE GLOVE: {speed_text(p.speed)}. The automatic "
                      f"track ended at a possible bat contact; if the batter hit the ball, press B here "
                      f"instead (or U to undo)", "warn")
            return
        self._say(f"{kind} set to field {s}: {speed_text(p.speed if p else None)}", "ok")

    def _do_set_release(self) -> None:
        if self._paused_for_mark("R") or self._single_field_needed("mark the release"):
            return
        if self.draft is not None:
            before = copy.deepcopy(self.draft)
            self._push([], "draft_release", draft_before=before)
            self.draft["s_release"] = self.s
            self._say(f"manual pitch: release at field {self.s}; now C at the FIRST field with the ball in the glove")
            return
        if self.current_entry() is None:
            self._start_draft()
            return
        self._set_event("release", self.s)

    def _do_set_contact(self) -> None:
        """B: the flight ended at the bat (a hit ball has no catch)."""
        if self._paused_for_mark("B") or self._single_field_needed("mark the bat contact"):
            return
        if self.draft is not None:
            self._finish_draft(point="contact")
            return
        if self.current_entry() is None:
            self._say("press N (or R) at the release field first, then B at the bat contact", "warn")
            return
        self._set_event("catch", self.s, point="contact")

    def _do_set_catch(self) -> None:
        if self._paused_for_mark("C") or self._single_field_needed("mark the catch"):
            return
        if self.draft is not None:
            self._finish_draft()
            return
        if self.current_entry() is None:
            self._say("press N (or R) at the release field first, then C at the catch", "warn")
            return
        self._set_event("catch", self.s)

    def _do_toggle_marker(self) -> None:
        self.marker = "C" if self.marker == "R" else "R"
        self._say(f"{self.marker} marker selected (, and . move it)")

    def _nudge(self, step: int) -> None:
        if self._single_field_needed("move a mark"):
            return
        if self.draft is not None:
            before = copy.deepcopy(self.draft)
            self._push([], "draft_nudge", draft_before=before)
            self.draft["s_release"] = self._clamp(self.draft["s_release"] + step)
            self.s = self.draft["s_release"]
            return
        p = self.current_pitch()
        if p is None:
            self._say("no pitch selected", "warn")
            return
        kind = "release" if self.marker == "R" else "catch"
        cur = self.ev_s(p.release if kind == "release" else p.catch)
        if cur is None:
            return
        target = self._clamp(cur + step)
        before_s = self.s
        self._set_event(kind, target)
        moved = self.ev_s(self.current_pitch().release if kind == "release" else self.current_pitch().catch)
        self.s = moved if moved is not None else before_s

    def _do_nudge_back(self) -> None:
        self._nudge(-1)

    def _do_nudge_fwd(self) -> None:
        self._nudge(1)

    def _drafting(self) -> bool:
        """Mark mode: while a new pitch is half made, keys that act on other items are refused."""
        if self.marking and self.draft is not None:
            self._say("a new pitch is half made: press C at the glove (B at the bat), or Esc to cancel it",
                      "warn")
            return True
        return False

    def _do_accept(self) -> None:
        if self._drafting():
            return
        it = self.current_item()
        if it is None:
            self._say("nothing selected to accept")
            return
        if it.key.kind == "missed":
            self._say("possible missed pitch: press N at its release to add it, or X if there is no pitch")
            return
        if it.key.kind == "rtrack":
            self._promote(it.key.ref)
            return
        e = self.entry(it.key.ref)
        before = copy.deepcopy(e.pitch)
        p = e.pitch
        if p.status == "rejected":
            prev = _status_before_reject(p)
            p.status = prev
            p.edit_history.append({"action": "unreject", "at": now_text(), "status": prev})
            msg = f"pitch restored ({prev})"
        else:
            p.edit_history.append({"action": "accept", "at": now_text()})
            msg = "accepted"
        self._push([("replace", e.uid, before)], "accept")
        self._commit()
        self._move_then_say(f"{msg}: pitch {p.number}")

    def _do_reject(self) -> None:
        if self._drafting():
            return
        it = self.current_item()
        if it is None:
            self._say("nothing selected to reject")
            return
        if it.key.kind == "missed":
            m = self.result.possible_missed[it.key.ref]
            if m.get("review") == "pitch_added":
                self._say("a pitch was added here by hand: to remove it, select that pitch (PgUp/PgDn) and "
                          "press X", "warn")
                return
            before = copy.deepcopy(m)
            if m.get("review") == "no_pitch":
                m.pop("review", None)
                self._say("window marked 'not checked' again")
            else:
                m["review"] = "no_pitch"
                self._say("window marked: checked, no pitch")
            self._push([("missed", it.key.ref, before)], "dismiss")
            self._save()
            self._move_then_say(self.message.text if self.message else "")
            return
        if it.key.kind == "rtrack":
            tr = self.result.rejected_tracks[it.key.ref]
            if str(tr.diagnostics.get("review", "")).startswith("promoted"):
                self._say("this track was made into a pitch: reject that pitch instead", "warn")
                return
            before = copy.deepcopy(tr.diagnostics)
            tr.diagnostics["review"] = "confirmed_not_a_pitch"
            self._push([("rtrack", it.key.ref, before)], "confirm_reject")
            self._save()
            self._say("rejected track confirmed: not a pitch")
            self._move_item(1)
            return
        e = self.entry(it.key.ref)
        p = e.pitch
        if p.status == "rejected":
            self._say("already rejected (Y restores it)")
            return
        before = copy.deepcopy(p)
        number, status = p.number, p.status
        p.status = "rejected"
        p.edit_history.append({"action": "reject", "at": now_text(), "number": number, "status": status})
        self._push([("replace", e.uid, before)], "reject")
        self._commit()
        self._say(f"pitch {number} rejected (kept in the results as 'rejected'; Y restores)", "warn")
        self._move_item(1)

    # manual pitches
    def _do_new_manual(self) -> None:
        if self._paused_for_mark("N") or self._single_field_needed("mark the release of a new pitch"):
            return
        self._start_draft()

    def _start_draft(self) -> None:
        before = copy.deepcopy(self.draft)
        self._push([], "draft_start", draft_before=before)
        self.draft = {"s_release": self.s, "item": self.cur}
        self.marker = "R"
        self._say(f"manual pitch: release at field {self.s}. Step to the FIRST field with the ball in the "
                  f"glove and press C, or B if the batter hit it (Esc cancels)", "ok")

    def _finish_draft(self, point: Optional[str] = None) -> None:
        d = self.draft
        sr, sc = int(d["s_release"]), self.s
        if sc <= sr:
            self._say(f"the catch must be after the release (field {sr}); step forward", "error")
            return
        ops = []
        track = None
        origin = d.get("item")
        if origin is not None and origin.kind == "rtrack":
            tr = self.result.rejected_tracks[origin.ref]
            ops.append(("rtrack", origin.ref, copy.deepcopy(tr.diagnostics)))
            tr.diagnostics["review"] = "promoted_to_manual_pitch"
            track = _track_copy(tr, sr - 2, sc + 2)
        end_ev = self.make_event(sc, "catch")
        end_ev.point = point
        p = Pitch(number=0, status="manual", release=self.make_event(sr, "release"),
                  catch=end_ev,
                  speed=SpeedEstimate(v_release_mph=None, reason="not computed yet"),
                  frame_release=int(self.tl.locate(sr)[0]), frame_catch=int(self.tl.locate(sc)[0]),
                  track=track, confidence=0.0,
                  edit_history=[{"action": "create_manual", "at": now_text(), "s_release": sr, "s_catch": sc,
                                 "from": origin.kind if origin is not None else "free"}])
        ops += self._mark_windows_resolved(sr, sc, origin)
        e = self._add_entry(p)
        ops.append(("add", e.uid))
        self._push(ops, "add_manual", draft_before=d)
        self.draft = None
        self._recompute(e)
        self._commit()
        self.cur = ItemKey("pitch", e.uid)
        self.marker = "C"
        more = " | Space plays on to the next pitch" if self.marking else ""
        self._say(f"manual pitch {e.pitch.number} added: {speed_text(e.pitch.speed)}{more}", "ok")

    def _mark_windows_resolved(self, sr: int, sc: int, origin: Optional[ItemKey]) -> list:
        """Record review = 'pitch_added' on the possible-missed windows a new pitch accounts for.

        The window the manual pitch was started from (if the pitch lies
        within WINDOW_MARGIN of it) and every other window that overlaps the
        flight sr..sc - also one marked 'no_pitch' before (the new pitch is
        the newer decision).  Returns the undo ops.  results.csv and the
        annotated video read this key (a checked window is not shown as
        'check in review' any more).
        """
        ops = []
        for i, m in enumerate(self.result.possible_missed or []):
            if not isinstance(m, dict) or m.get("review") == "pitch_added":
                continue
            a, b = self._window_of(m)
            from_here = origin is not None and origin.kind == "missed" and origin.ref == i
            pad = WINDOW_MARGIN if from_here else 0
            if a - pad <= sc and sr <= b + pad:
                ops.append(("missed", i, copy.deepcopy(m)))
                m["review"] = "pitch_added"
        return ops

    def _promote(self, idx: int) -> None:
        tr = self.result.rejected_tracks[idx]
        if str(tr.diagnostics.get("review", "")).startswith("promoted"):
            self._say("this track was already made into a pitch", "warn")
            return
        r, c = self.ev_s(tr.release), self.ev_s(tr.catch)
        if r is None or c is None or c <= r:
            self._say("this track has no usable release/catch: press N at its release, then C at the catch",
                      "warn")
            return
        ops = [("rtrack", idx, copy.deepcopy(tr.diagnostics))]
        tr.diagnostics["review"] = "promoted_to_pitch"
        ops += self._mark_windows_resolved(r, c, None)
        p = Pitch(number=0, status="edited", release=copy.deepcopy(tr.release), catch=copy.deepcopy(tr.catch),
                  speed=SpeedEstimate(v_release_mph=None, reason="not computed yet"),
                  frame_release=int(self.tl.locate(r)[0]), frame_catch=int(self.tl.locate(c)[0]),
                  track=_track_copy(tr, -math.inf, math.inf), confidence=tr.confidence, flags=list(tr.flags),
                  edit_history=[{"action": "promote_rejected_track", "at": now_text(),
                                 "reject_code": tr.reject_code}])
        e = self._add_entry(p)
        ops.append(("add", e.uid))
        self._push(ops, "promote")
        self._recompute(e)
        self._commit()
        self.cur = ItemKey("pitch", e.uid)
        self._say(f"rejected track ({tr.reject_code}) added as pitch {e.pitch.number}: "
                  f"{speed_text(e.pitch.speed)}", "ok")

    # ball points
    def on_click(self, x: float, y: float) -> None:
        """Left click at frame pixel (x, y) of the current sample: manual ball point."""
        self._error_said = False
        if self._single_field_needed("click ball points"):
            return
        e = self._need_pitch()
        if e is None:
            return
        r, c = self.ev_s(e.pitch.release), self.ev_s(e.pitch.catch)
        if r is not None and c is not None and not (r - POINT_TOLERANCE <= self.s <= c + POINT_TOLERANCE):
            self._say(f"field {self.s} is outside this pitch (R {r} .. C {c})", "warn")
            return
        s = self.s
        xr, yr = self.stab.to_ref(s, x, y)
        t = float(self.tl.sample_time(s))

        def mutate(p: Pitch) -> None:
            if p.track is None:
                p.track = PitchTrack(track_id=-1, points=[], release=p.release, catch=p.catch,
                                     flags=["manual_points"])
            pts = []
            for pt in p.track.points:
                if pt.s == s and pt.manual:
                    continue                      # replaced below
                if pt.s == s and not pt.manual:
                    pt.inlier = False             # kept, but excluded
                pts.append(pt)
            pts.append(TrackPoint(s=s, t=t, x=round(xr, 3), y=round(yr, 3), score=0.0, sigma=1.0,
                                  inlier=True, manual=True))
            pts.sort(key=lambda q: (q.s, q.manual))
            p.track.points = pts

        self._edit(e, "add_point", mutate, s=s, x=round(xr, 3), y=round(yr, 3))
        self._say(f"ball point set at field {s}: {speed_text(self.current_pitch().speed)}", "ok")

    def on_right_click(self, x: float, y: float) -> None:
        """Right click near a ball point of the current field: delete (manual) / toggle (automatic)."""
        self._error_said = False
        if self._single_field_needed("change ball points"):
            return
        e = self._need_pitch()
        if e is None:
            return
        if e.pitch.track is None:
            self._say("this pitch has no ball points", "warn")
            return
        s = self.s
        best, best_d = None, math.inf
        sar = float(getattr(getattr(self.result, "video", None), "sar", 1.0) or 1.0)
        for i, pt in enumerate(e.pitch.track.points):
            if pt.s != s:
                continue
            px, py = self.stab.from_ref(s, pt.x, pt.y)
            d = math.hypot((px - x) * sar, py - y)
            if d < best_d:
                best, best_d = i, d
        if best is None or best_d > self.pick_radius:
            self._say("no ball point of this field near the click", "warn")
            return
        pt = e.pitch.track.points[best]
        if pt.manual:
            def mutate(p: Pitch) -> None:
                p.track.points.pop(best)
            self._edit(e, "remove_point", mutate, s=s)
            self._say(f"manual point at field {s} removed")
        else:
            new_state = not pt.inlier

            def mutate(p: Pitch) -> None:
                p.track.points[best].inlier = new_state
            self._edit(e, "include_point" if new_state else "exclude_point", mutate, s=s)
            self._say(f"automatic point at field {s} {'included' if new_state else 'excluded'}")

    # re-tracking
    def _do_retrack(self) -> None:
        if self._drafting():
            return
        e = self._need_pitch()
        if e is None:
            return
        if self.retrack_cb is None:
            self._say("re-tracking is not available in this session", "warn")
            return
        self.pending = ("retrack", e.uid)
        self._say("re-tracking between R and C ...")

    def run_pending(self) -> None:
        """Run a slow action requested by the last key (the shell shows 'working' first)."""
        if self.pending is None:
            return
        self._error_said = False
        what, uid = self.pending
        self.pending = None
        if what == "flip_order":
            self._flip_order()
            return
        e = self.entry(uid)
        if what != "retrack" or e is None:
            return
        try:
            track = self.retrack_cb(e.pitch)
        except Exception as ex:  # noqa: BLE001
            self._say(f"re-tracking failed: {ex}", "error")
            return
        if track is None:
            self._say("re-tracking found nothing; marks unchanged", "warn")
            return
        n = len([p for p in track.points if p.inlier])

        def mutate(p: Pitch) -> None:
            p.track = track

        self._edit(e, "retrack", mutate, n_points=n)
        self._say(f"re-tracked: {n} ball points; {speed_text(self.current_pitch().speed)}", "ok")

    # field order (marking by hand)
    def _do_flip_order(self) -> None:
        if not getattr(self.tl, "interlaced", False):
            self._say("this video has whole frames only (not interlaced): there is no field order to switch")
            return
        if not self.marking:
            self._say("the field order of analyzed results comes from the analysis: analyze again with "
                      "--field-order tff or bff to change it", "warn")
            return
        def by_hand(p: Pitch) -> bool:
            status = _status_before_reject(p) if p.status == "rejected" else p.status
            return status == "manual" and all(pt.manual for pt in (p.track.points if p.track else []))
        auto = [e.pitch.number for e in self.entries if not by_hand(e.pitch)]
        auto += [i for i, tr in enumerate(self.result.rejected_tracks or []) if tr is not None]
        if auto:
            self._say("these results contain automatically found pitches: change the field order by analyzing "
                      "again with --field-order tff or bff", "warn")
            return
        if not hasattr(self.tl, "set_field_order"):
            self._say("the field order cannot be switched for this video source", "error")
            return
        self.pending = ("flip_order", None)
        self._say("switching the field order ...")

    def _flip_order(self) -> None:
        """Switch TFF <-> BFF: sample s now shows the other field of its frame, so every mark,
        clicked ball point and the current position move to s ^ 1 (same picture, time +-1/2
        field); speeds are recomputed.  Undo history before the switch is dropped."""
        sc = self.result.scan
        old = (sc.field_order if sc is not None else None) or "tff"
        new = "bff" if old == "tff" else "tff"
        try:
            self.tl.set_field_order(new)
        except (ValueError, AttributeError) as ex:
            self._say(f"the field order could not be switched: {ex}", "error")
            return
        if sc is not None:
            sc.field_order, sc.source, sc.confidence = new, "user", "high"
            sc.evidence = {**(sc.evidence or {}), "switched_by_user_from": old}
        for e in self.entries:
            p = e.pitch
            for kind in ("release", "catch"):
                ev = getattr(p, kind)
                if ev is not None and ev.s is not None:
                    ne = self.make_event(self._clamp(int(ev.s) ^ 1), kind)
                    ne.point = getattr(ev, "point", None)
                    setattr(p, kind, ne)
            for pt in (p.track.points if p.track is not None else []):
                pt.s = self._clamp(int(pt.s) ^ 1)
                pt.t = float(self.tl.sample_time(pt.s))
            p.edit_history.append({"action": "field_order_switched", "from": old, "to": new, "at": now_text()})
            self._recompute(e)
        if self.draft is not None:
            self.draft["s_release"] = self._clamp(int(self.draft["s_release"]) ^ 1)
        self.s = self._clamp(self.s ^ 1)
        self.undo_stack.clear()
        self.result.warnings = [w for w in (self.result.warnings or [])
                                if not str(w).startswith("interlace/field-order detection is uncertain")]
        self.result.warnings.append(f"field order switched by hand from {old.upper()} to {new.upper()}")
        self.frames_changed = True
        self._commit()
        self._say(f"field order switched to {new.upper()}: step through some motion again to check "
                  f"(K switches back). Marks kept on the same pictures; undo history cleared", "ok")

    # undo / quit
    def _do_undo(self) -> None:
        if not self.undo_stack:
            self._say("nothing to undo")
            return
        u = self.undo_stack.pop()
        touched_pitches = False
        for op in reversed(u.ops):
            kind = op[0]
            if kind == "replace":
                e = self.entry(op[1])
                if e is not None:
                    e.pitch = op[2]
                touched_pitches = True
            elif kind == "add":
                self.entries = [e for e in self.entries if e.uid != op[1]]
                touched_pitches = True
            elif kind == "missed":
                m = self.result.possible_missed[op[1]]
                m.clear()
                m.update(op[2])
            elif kind == "rtrack":
                self.result.rejected_tracks[op[1]].diagnostics = op[2]
        self.draft = u.draft
        if u.ops:
            if touched_pitches:
                self._commit()
            else:
                self._save()
        keys = {it.key for it in self.items()}
        self.cur = u.item if u.item in keys else self._nearest_item_key()
        self.s = self._clamp(u.s)
        self._say(f"undone: {u.label.replace('_', ' ')}")

    def _do_escape(self) -> None:
        if self.help:
            self.help = False
            return
        if self.draft is not None:
            self._push([], "cancel_draft")
            self.draft = None
            self._say("manual pitch cancelled (U brings it back)")
            return
        self._do_quit()

    def _do_quit(self) -> None:
        if self.help:
            self.help = False
            return
        if self.draft is not None and not self.quit_armed:
            self.quit_armed = True
            self._say("a manual pitch is half made (no catch yet): Q again quits and drops it, C sets the catch",
                      "warn")
            return
        self.finish()

    def finish(self) -> None:
        """Final save; the session ends."""
        self.draft = None
        self.playing = False
        self._commit()
        self.done = True

    # ---------------------------------------------------------- display
    def counts(self) -> tuple[int, int]:
        """(number of the current pitch or 0, number of non-rejected pitches)."""
        n = sum(1 for e in self.entries if e.pitch.status != "rejected")
        p = self.current_pitch()
        return (p.number if p is not None else 0), n

    def info_text(self) -> str:
        """The bottom info line ("Pitch 3/12 | auto | 00:01:23.457 | fields 4986-5013 | 78+/-3 mph ...")."""
        it = self.current_item()
        if self.draft is not None:
            sr = self.draft["s_release"]
            return (f"NEW MANUAL PITCH | release field {sr} ({format_time(self.tl.sample_time(sr))}) | "
                    f"now at field {self.s} (+{self.s - sr}) | press C at the catch (B if hit), Esc cancels")
        if it is None and self.marking:
            n = sum(1 for e in self.entries if e.pitch.status != "rejected")
            return (f"no pitch here | {format_time(self.tl.sample_time(self.s))} of {self._total_text()} | "
                    f"{n} pitch{'es' if n != 1 else ''} marked | N = new pitch at its release")
        if it is None:
            return f"no item selected | field {self.s} | PgUp/PgDn: pitches | N: add a pitch at its release"
        if it.key.kind == "missed":
            m = self.result.possible_missed[it.key.ref]
            missed = [x for x in self.items() if x.key.kind == "missed"]
            i = [x.key for x in missed].index(it.key) + 1
            state = {"no_pitch": "checked: no pitch", "pitch_added": "checked: pitch added"}.get(
                m.get("review"), "not checked")
            return (f"Possible missed pitch {i}/{len(missed)} | {format_time(self.tl.sample_time(it.s0))} | "
                    f"fields {it.s0}-{it.s1} | {m.get('reason', '')} | {state} | N at release adds it, X = no pitch")
        if it.key.kind == "rtrack":
            tr = self.result.rejected_tracks[it.key.ref]
            rv = tr.diagnostics.get("review", "not checked")
            return (f"Rejected track #{it.key.ref + 1} | {tr.reject_code or '?'} | "
                    f"{format_time(self.tl.sample_time(it.s0))} | fields {it.s0}-{it.s1} | "
                    f"{len(tr.points)} points | {rv} | Y = make it a pitch, X = confirm")
        p = self.entry(it.key.ref).pitch
        num, n = self.counts()
        if p.status == "rejected":
            was = _number_before_reject(p)
            head = f"Pitch -/{n} (rejected" + (f", was #{was})" if was else ")")
        else:
            head = f"Pitch {num}/{n} | {p.status}"
        sp = p.speed
        flags = list(dict.fromkeys(list(p.flags) + list(getattr(sp, "flags", []) or [])))
        parts = [head, format_time(p.release.t if p.release else None), f"fields {it.s0}-{it.s1}",
                 speed_text(sp), METHOD_NAMES.get(getattr(sp, "method", "none"), str(getattr(sp, "method", "")))]
        parts.append(getattr(sp, "label", "none") or "none")
        end = end_text(p)
        if end:
            parts.append(end)
        if flags:
            parts.append(",".join(flags))
        if _reviewed(p):
            parts.append("reviewed")
        return " | ".join(parts)

    def top_text(self) -> str:
        frame, k, parity = self.tl.locate(self.s)
        items = self.items()
        i = self.item_index()
        pos = f"item {i + 1}/{len(items)}" if i is not None else f"{len(items)} items"
        where = f"field {self.s} (frame {frame} {'top' if parity == 0 else 'bottom'})" \
            if getattr(self.tl, "interlaced", False) else f"frame {frame}"
        if self.view_mode == "frame" and getattr(self.tl, "interlaced", False):
            where += " FULL FRAME"
        rel = ""
        p = self.current_pitch()
        if p is not None and self.draft is None:
            r, c = self.ev_s(p.release), self.ev_s(p.catch)
            if r is not None:
                rel = f" | R{self.s - r:+d}"
            if c is not None:
                rel += f" C{self.s - c:+d}"
        play = f" | PLAY {self.speed_labels[self.speed_i]}" if self.playing else ""
        if self.marking:
            n = sum(1 for e in self.entries if e.pitch.status != "rejected")
            return (f"MARK | {n} marked | {where} | {format_time(self.tl.sample_time(self.s))} / "
                    f"{self._total_text()}{rel}{play} | H = help")
        return f"REVIEW | {pos} | {where} | {format_time(self.tl.sample_time(self.s))}{rel}{play} | H = help"

    def _total_text(self) -> str:
        return format_time(self.tl.sample_time(max(0, self.n_samples - 1)))

    def markers(self) -> dict:
        """Timeline data for the bottom strip."""
        w0, w1 = self.window()
        p = self.current_pitch()
        out = {"w0": w0, "w1": w1, "s": self.s, "R": None, "C": None, "points": [], "selected": self.marker,
               "draft": self.draft["s_release"] if self.draft else None,
               "pitches": [(it.s0, it.s1, self.entry(it.key.ref).pitch.status) for it in self.items()
                           if it.key.kind == "pitch" and self.entry(it.key.ref) is not None]
               if self.marking else []}
        it = self.current_item()
        if p is not None:
            out["R"], out["C"] = self.ev_s(p.release), self.ev_s(p.catch)
            if p.track is not None:
                out["points"] = [(pt.s, pt.manual, pt.inlier) for pt in p.track.points]
        elif it is not None:
            out["span"] = (it.s0, it.s1)
        return out

    def shows(self, s: Optional[int]) -> bool:
        """True if sample s is in the current picture (its field, or its frame in the full-frame view)."""
        if s is None:
            return False
        if s == self.s:
            return True
        if not self.woven():
            return False
        try:
            return int(self.tl.locate(int(s))[0]) == int(self.tl.locate(self.s)[0])
        except (IndexError, ValueError):
            return False

    def overlay_points(self) -> list[tuple[float, float, bool, bool, bool]]:
        """Track points of the current pitch in frame px of the CURRENT sample.

        (x, y, manual, inlier, in_current_picture) for each point (the
        current field; both fields of the frame in the full-frame view).
        """
        p = self.current_pitch()
        if p is None and self.cur is not None and self.cur.kind == "rtrack":
            tr = self.result.rejected_tracks[self.cur.ref]
        else:
            tr = p.track if p is not None else None
        if tr is None:
            return []
        out = []
        for pt in tr.points:
            x, y = self.stab.from_ref(self.s, pt.x, pt.y)
            out.append((x, y, bool(pt.manual), bool(pt.inlier), self.shows(pt.s)))
        return out

    def hint_text(self) -> str:
        p = self.current_pitch()
        if self.marking and self.draft is None and p is None:
            return "Space play | + - speed | J/L 1 s | W/S 10 fields | A/D 1 field | N new pitch | PgUp/PgDn marked pitches"
        if self.draft is not None or (p is not None and p.status == "manual") or self.current_entry() is None:
            return MANUAL_HINT
        return "R/C set marks here | , . nudge | Y accept | X reject | E re-track | Space play"

    def help_rows(self) -> list[tuple[str, str]]:
        rows = [("", MARK_START)] if self.marking else []
        rows += [("", MANUAL_HINT)]
        rows += [(key_label(BINDINGS[a]), HELP_TEXT[a]) for a in BINDINGS]
        rows += [("L-click", "add/move a manual ball point at this field"),
                 ("R-click", "remove a manual point / exclude-include an automatic one"),
                 ("", "Every edit recomputes the speed and autosaves results.json.")]
        return rows


# ------------------------------------------------------------ end point
def _track_end(p: Pitch) -> str:
    """How the pitch's ball track ended according to the finder ('catch' if unknown)."""
    return (getattr(p.track, "end_kind", None) if p.track is not None else None) or "catch"


def effective_end(p: Pitch) -> str:
    """Where the speed estimate ends the flight: 'contact' or 'catch' (the glove).

    Same rule as estimate.speed: a hand-marked catch (C) is always the
    glove; otherwise the finder's end kind decides ('contact' = a possible
    bat contact; 'lost' / 'occluded' tracks still end at the glove).
    """
    if p.catch is not None and p.catch.method == "manual":
        return "contact" if getattr(p.catch, "point", None) == "contact" else "catch"
    return "contact" if _track_end(p) == "contact" else "catch"


def end_text(p: Pitch) -> str:
    """Info-line note on the end point, '' for the usual automatic glove catch."""
    if effective_end(p) == "contact":
        if p.catch is not None and p.catch.method == "manual":
            return "ends at BAT CONTACT (B marked)"
        return "ends at a possible BAT CONTACT (C at the glove if it was caught)"
    if _track_end(p) == "contact":
        return "ends at the glove (C marked)"
    return ""


# ---------------------------------------------------------- history helpers
def _status_before_reject(p: Pitch) -> str:
    for h in reversed(p.edit_history):
        if h.get("action") == "reject":
            st = h.get("status")
            if st in ("auto", "edited", "manual"):
                return st
    return "edited"


def _number_before_reject(p: Pitch) -> Optional[int]:
    for h in reversed(p.edit_history):
        if h.get("action") == "reject":
            return h.get("number") or None
    return None


def _reviewed(p: Pitch) -> bool:
    return any(h.get("action") in ("accept", "unreject") for h in p.edit_history)


def _track_copy(tr: PitchTrack, s_lo: float, s_hi: float) -> PitchTrack:
    out = copy.deepcopy(tr)
    out.points = [pt for pt in out.points if s_lo <= pt.s <= s_hi]
    out.status, out.reject_code = "pitch", None
    out.diagnostics = dict(out.diagnostics)
    out.diagnostics["from_rejected"] = tr.reject_code
    return out
