"""Annotated output video: every sample with a HUD, pitch tags, ball trails
and speed cards (docs/design-notes/io_ux.md sections 3 and 6).

    render_annotated(result, source, out_path, mode='full'|'highlights', ...)

OUTPUT TIMING.  One output frame per sample: interlaced video becomes
59.94p (each field bob-deinterlaced, parity-aware, by
`pitchspeed.io.display.sample_to_display`), progressive video keeps its
sample rate.  Output frame j of a segment shows source time
t0 + j / fps.  If the source has a gap (a frame that failed to decode, a
dropped-frame PTS gap, VFR jitter), filler frames saying "no picture" are
inserted, so the picture stays in step with the source time and with the
passed-through audio; nothing is silently skipped.

WHAT IS DRAWN (all text plain ASCII - OpenCV's fonts have no other glyphs -
with a dark outline / panel so it stays legible on bright video):
* top-left HUD: source time hh:mm:ss.sss, frame number, field number;
* during each pitch's flight (release <= s < catch): a "P3" tag next to the
  ball and the trail (last 8 samples solid, older ones smaller and pale;
  green = auto, yellow = edited/manual, manual points as squares);
* from the catch: a card "Pitch 3: 78 mph" / "+/-3 mph (90%: 74-82) |
  release speed | TOF", held 3 s or until the next pitch's release.  Label
  'poor' -> "~78 mph?" in orange; label 'none' -> "Pitch 3: speed n/a" with
  a short reason in grey;
* top-right: running list "P1 76  P2 79  P3 78" (whole mph) of the pitches
  caught so far ('poor' as "65?" in orange - a small '~' looks like a minus
  sign -, 'none' as "n/a" in grey; only the most recent that fit);
* rejected pitches are not drawn (they stay in results.csv / .json);
  'possible missed pitch' windows get an orange hint under the HUD (full
  mode; highlights only cover accepted pitches) unless the user already
  checked the window in review.

TRACK COORDINATES.  Track points are in calibration-reference-frame
coordinates.  For the displayed sample s they are mapped to that sample's
own frame coordinates with the inverse of transforms[s] (sample ->
reference, either (n, 2, 3) affine or (n, 3, 3) homographies - the latter
are what the finder mapped the ball with, applied projectively; the
nearest available sample's transform if s itself has none; identity when
no transforms are given), then to display pixels with the display affine
A from `sample_to_display`.

LOCKED OUTPUT.  If the target (annotated.mp4) cannot be written because
another program has it open (a video player on Windows), the video is
written as annotated-1.mp4 (next free number) instead and `info['note']`
says so; nothing is lost after a long render.

MODES.
* 'full': the whole video, audio passed through by VideoWriter
  (audio_src = the source file, audio_start_s = time of the first sample).
* 'highlights': only [release - 2 s, catch + 3 s] around each pitch,
  windows merged when they overlap or touch.  Audio is handled PER SEGMENT:
  the video is encoded once (video only), then one ffmpeg pass cuts the
  source audio for every segment (atrim, sample accurate, padded with
  silence where the source has none) and concatenates the pieces, so each
  segment's audio starts with its first picture.  If that pass fails the
  video is kept without audio and the reason is logged and returned in
  `info['audio_status']`.
"""
from __future__ import annotations

import logging
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Optional, Sequence

import cv2
import numpy as np

from pitchspeed.render.text import (ascii_text, ci90_text, format_time, has_number, method_name,
                                    round_half_up, shorten, sigma_text, speed_reason, speed_short)
from pitchspeed.types import AnalysisResult, EventTime, Pitch, SpeedEstimate

log = logging.getLogger(__name__)

__all__ = ["render_annotated", "PitchMark", "FrameState", "Timeline", "TransformLookup",
           "build_marks", "highlight_windows", "reference_to_display", "apply_affine", "card_lines",
           "running_items", "audio_filter_graph", "output_fps", "missed_windows",
           "mux_segment_audio", "can_write", "alternative_path", "move_into_place"]

HOLD_S = 3.0                 # card hold after the catch
PRE_S = 2.0                  # highlights: seconds before release
POST_S = 3.0                 # highlights: seconds after catch
TRAIL_SOLID = 8              # trail: the last N samples drawn solid
MAX_FILL_S = 30.0            # never insert more filler than this per gap
FFMPEG_TIMEOUT_S = 3600.0
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0       # CREATE_NO_WINDOW on Windows

# colours (BGR)
WHITE = (255, 255, 255)
GREEN = (80, 230, 80)
YELLOW = (40, 225, 255)
ORANGE = (0, 150, 255)
GREY = (185, 185, 185)
RED = (80, 80, 240)
BLACK = (0, 0, 0)


# ================================================================== marks
@dataclass
class PitchMark:
    """What the overlay needs to know about one (non-rejected) pitch."""
    number: int
    status: str
    s_release: int
    s_catch: int
    speed: Optional[SpeedEstimate]
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))   # (N, 4): s, x, y, manual

    @property
    def edited(self) -> bool:
        return self.status in ("edited", "manual")

    @property
    def color(self) -> tuple:
        return YELLOW if self.edited else GREEN


def _event_sample(ev: Optional[EventTime], to_sample: Callable[[float], int]) -> Optional[int]:
    if ev is None:
        return None
    if ev.s is not None:
        return int(ev.s)
    if ev.t is None or not math.isfinite(float(ev.t)):
        return None
    return int(to_sample(float(ev.t)))


def _track_points(p: Pitch) -> np.ndarray:
    if p.track is None or not p.track.points:
        return np.zeros((0, 4))
    rows = [(tp.s, tp.x, tp.y, 1.0 if tp.manual else 0.0) for tp in p.track.points
            if tp.inlier and tp.x is not None and tp.y is not None
            and math.isfinite(tp.x) and math.isfinite(tp.y)]
    if not rows:
        return np.zeros((0, 4))
    a = np.asarray(rows, dtype=np.float64)
    return a[np.argsort(a[:, 0], kind="stable")]


def build_marks(result: AnalysisResult, to_sample: Callable[[float], int]) -> list[PitchMark]:
    """Marks for every drawable pitch, sorted by release.

    Rejected pitches are skipped (not drawn).  A pitch without a usable
    release or catch time cannot be placed on the timeline; it is skipped
    with a warning in the log (it is still in the CSV / JSON)."""
    marks = []
    for p in result.pitches:
        if p.status == "rejected":
            continue
        s_rel, s_cat = _event_sample(p.release, to_sample), _event_sample(p.catch, to_sample)
        if s_rel is None or s_cat is None:
            log.warning("pitch %s has no release/catch time; not drawn in the video", p.number)
            continue
        if s_cat < s_rel:
            s_rel, s_cat = s_cat, s_rel
        marks.append(PitchMark(number=p.number, status=p.status, s_release=s_rel, s_catch=s_cat,
                               speed=p.speed, points=_track_points(p)))
    marks.sort(key=lambda m: (m.s_release, m.number))
    return marks


# =============================================================== timeline
@dataclass
class FrameState:
    """Which overlay elements are visible at one sample (indices into marks)."""
    in_flight: list[int]          # tag + live trail
    card: Optional[int]           # speed card (at most one)
    done: list[int]               # caught so far (running list), chronological


class Timeline:
    """Visibility rules, precomputed for fast per-sample queries.

    * in flight:  s_release <= s < s_catch
    * card:       s_catch <= s < min(s_catch + hold, first release after the
                  catch); if several cards qualify the latest catch wins
    * running list: every pitch with s_catch <= s
    """

    def __init__(self, marks: Sequence[PitchMark], hold_samples: int):
        self.marks = list(marks)
        self.hold = max(0, int(hold_samples))
        self.rel = np.array([m.s_release for m in self.marks], dtype=np.int64)
        self.cat = np.array([m.s_catch for m in self.marks], dtype=np.int64)
        rel_sorted = np.sort(self.rel)
        ends = []
        for c in self.cat:
            j = np.searchsorted(rel_sorted, c, side="right")       # first release after the catch
            nxt = rel_sorted[j] if j < len(rel_sorted) else np.iinfo(np.int64).max
            ends.append(min(c + self.hold, nxt))
        self.card_end = np.array(ends, dtype=np.int64)
        self._by_catch = np.argsort(self.cat, kind="stable")
        self._cat_sorted = self.cat[self._by_catch]

    def state(self, s: int) -> FrameState:
        if not self.marks:
            return FrameState([], None, [])
        flight = np.nonzero((self.rel <= s) & (s < self.cat))[0]
        cards = np.nonzero((self.cat <= s) & (s < self.card_end))[0]
        card = int(cards[np.argmax(self.cat[cards])]) if len(cards) else None
        done = self._by_catch[:int(np.searchsorted(self._cat_sorted, s, side="right"))]
        return FrameState([int(i) for i in flight], card, [int(i) for i in done])


def highlight_windows(marks: Sequence[PitchMark], n_samples: int, pre_samples: int,
                      post_samples: int) -> list[tuple[int, int]]:
    """[s0, s1) sample windows around each pitch, clipped to the video and
    merged when they overlap or touch."""
    raw = sorted((max(0, m.s_release - pre_samples), min(n_samples, m.s_catch + post_samples + 1))
                 for m in marks)
    out: list[list[int]] = []
    for s0, s1 in raw:
        if s1 <= s0:
            continue
        if out and s0 <= out[-1][1]:
            out[-1][1] = max(out[-1][1], s1)
        else:
            out.append([s0, s1])
    return [(a, b) for a, b in out]


def missed_windows(result: AnalysisResult) -> list[tuple[int, int]]:
    """[s_start, s_end] (inclusive) of the unchecked 'possible missed pitch' windows that carry
    sample indices (a window the user checked in review - 'review' key set - gets no hint)."""
    out = []
    for d in result.possible_missed or []:
        if not isinstance(d, dict) or d.get("review"):
            continue
        try:
            out.append((int(d["s_start"]), int(d["s_end"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


# ============================================================= transforms
class TransformLookup:
    """transforms[i] (sample -> reference; (n, 2, 3) affine or (n, 3, 3) homographies) of sample
    transform_samples[i]; `get(s)` returns the transform of s or of the nearest sample that has one
    (2x3 or 3x3 like the input; the identity of that shape for a broken entry)."""

    def __init__(self, transforms: Optional[np.ndarray], transform_samples: Optional[np.ndarray]):
        self.T = None
        self.s = None
        self.identity = _IDENTITY
        if transforms is None:
            return
        T = np.asarray(transforms, dtype=np.float64)
        if T.ndim != 3 or T.shape[1:] not in ((2, 3), (3, 3)) or len(T) == 0:
            log.warning("ignoring transforms with shape %s (expected (n, 2, 3) or (n, 3, 3))", T.shape)
            return
        s = np.arange(len(T)) if transform_samples is None else np.asarray(transform_samples, dtype=np.int64)
        if len(s) != len(T):
            log.warning("transforms (%d) and transform_samples (%d) differ in length; ignoring them",
                        len(T), len(s))
            return
        order = np.argsort(s, kind="stable")
        self.T, self.s = T[order], s[order]
        self.identity = np.eye(3) if T.shape[1] == 3 else _IDENTITY

    @property
    def projective(self) -> bool:
        return self.T is not None and self.T.shape[1] == 3

    def get(self, s: int) -> np.ndarray:
        if self.T is None:
            return _IDENTITY.copy()
        j = int(np.searchsorted(self.s, s))
        if j >= len(self.s) or (j > 0 and abs(self.s[j - 1] - s) <= abs(self.s[j] - s)):
            j -= 1
        M = self.T[max(0, j)]
        if not np.all(np.isfinite(M)):
            return self.identity.copy()
        det = np.linalg.det(M) if M.shape == (3, 3) else np.linalg.det(M[:, :2])
        if not math.isfinite(det) or abs(det) < 1e-9:
            return self.identity.copy()
        return M


_IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _to3(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    if M.shape == (3, 3):
        return M
    return np.vstack([M, [0.0, 0.0, 1.0]])


def reference_to_display(T_sample_to_ref: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Map reference-frame coords -> display coords: A o inverse(T).

    T 2x3 (affine) -> 2x3 result; T 3x3 (homography) -> 3x3 result (apply with
    `apply_affine`, which divides by w)."""
    M = _to3(A) @ np.linalg.inv(_to3(T_sample_to_ref))
    return M if np.asarray(T_sample_to_ref).shape == (3, 3) else M[:2]


def apply_affine(M: np.ndarray, pts) -> np.ndarray:
    """Apply a 2x3 affine or a 3x3 homography (with the divide by w) to (n, 2) points.
    Points that map to infinity (w <= 0) come back as NaN."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    M = np.asarray(M, dtype=np.float64)
    if M.shape == (3, 3):
        q = p @ M[:, :2].T + M[:, 2]
        w = q[:, 2:3]
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(w > 1e-12, q[:, :2] / np.where(w > 1e-12, w, 1.0), np.nan)
        return out
    return p @ M[:, :2].T + M[:, 2]


# ================================================================== texts
def card_lines(m: PitchMark) -> tuple[str, str, tuple]:
    """(line 1, line 2, colour) of the speed card shown after the catch."""
    sp = m.speed
    extra = f" | {m.status}" if m.edited else ""
    if not has_number(sp):
        reason = speed_reason(sp) or "not measurable"
        return f"Pitch {m.number}: speed n/a", shorten(ascii_text(reason), 64) + extra, GREY
    v = round_half_up(sp.v_release_mph)
    head = f"Pitch {m.number}: ~{v} mph?" if sp.label == "poor" else f"Pitch {m.number}: {v} mph"
    sub = f"+/-{sigma_text(sp.sigma_mph)} mph (90%: {ci90_text(sp)}) | release speed | " \
          f"{method_name(sp.method)}{extra}"
    return head, ascii_text(sub), (ORANGE if sp.label == "poor" else WHITE)


def running_items(marks: Sequence[PitchMark], done: Sequence[int]) -> list[tuple[str, tuple]]:
    """[('P1 76', colour), ...] for the running list, in pitch order."""
    out = []
    for i in sorted(done, key=lambda i: marks[i].number):
        m = marks[i]
        txt = speed_short(m.speed)
        poor = txt.startswith("~")
        if poor:            # small '~' reads like a minus sign: '65?' in orange instead
            txt = txt[1:]
        col = GREY if txt == "n/a" else (ORANGE if poor else WHITE)
        out.append((f"P{m.number} {txt}", col))
    return out


# ================================================================ painter
class Painter:
    """cv2 drawing of the overlay elements, scaled to the output size."""

    def __init__(self, size: tuple[int, int]):
        self.w, self.h = int(size[0]), int(size[1])
        self.k = self.h / 720.0
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bold = cv2.FONT_HERSHEY_DUPLEX
        self.m = max(4, int(round(10 * self.k)))           # margin

    # -------------------------------------------------------- primitives
    def _size(self, text: str, font, scale: float, thick: int) -> tuple[int, int, int]:
        (tw, th), base = cv2.getTextSize(text, font, scale, thick)
        return tw, th, base

    def text(self, img, text: str, org, scale: float, color, thick: int = 1, font=None) -> None:
        # Outline = the same text stamped in black at small offsets.  (A thicker
        # black copy underneath does not line up with OpenCV 5's text renderer,
        # where thickness changes the glyph advance.)
        font = self.font if font is None else font
        x, y = int(round(org[0])), int(round(org[1]))
        d = max(1, int(round(1.5 * self.k)))
        for dx, dy in ((-d, 0), (d, 0), (0, -d), (0, d), (-d, -d), (d, d), (-d, d), (d, -d)):
            cv2.putText(img, text, (x + dx, y + dy), font, scale, BLACK, thick, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)

    def shade(self, img, x0, y0, x1, y1, alpha: float = 0.6) -> None:
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(img.shape[1], int(x1)), min(img.shape[0], int(y1))
        if x1 > x0 and y1 > y0:
            roi = img[y0:y1, x0:x1]
            roi[:] = (roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

    # ----------------------------------------------------------- elements
    def hud(self, img, t: float, frame: Optional[int], s: Optional[int], interlaced: bool,
            note: str = "") -> None:
        parts = [format_time(t)]
        if frame is not None:
            parts.append(f"frame {frame}")
        if interlaced and s is not None:
            parts.append(f"field {s}")
        if note:
            parts.append(note)
        txt = "  ".join(parts)
        sc, th = 0.6 * self.k, max(1, int(round(self.k)))
        tw, tht, base = self._size(txt, self.font, sc, th)
        self.shade(img, 0, 0, tw + 2 * self.m, tht + base + 2 * self.m, 0.55)
        self.text(img, txt, (self.m, self.m + tht), sc, WHITE, th)

    def running_list(self, img, items: Sequence[tuple[str, tuple]]) -> None:
        if not items:
            return
        sc, th = 0.65 * self.k, max(1, int(round(self.k)))
        gap = int(round(16 * self.k))
        max_w = 0.45 * self.w
        chosen, width = [], 0
        for txt, col in reversed(items):
            tw = self._size(txt, self.font, sc, th)[0]
            if chosen and width + gap + tw > max_w:
                break
            chosen.insert(0, (txt, col, tw))
            width += tw + (gap if len(chosen) > 1 else 0)
        if len(chosen) < len(items):
            ell = "..."
            ew = self._size(ell, self.font, sc, th)[0]
            chosen.insert(0, (ell, GREY, ew))
            width += ew + gap
        _, tht, base = self._size("P0 00", self.font, sc, th)
        x = self.w - self.m - width
        self.shade(img, x - self.m, 0, self.w, tht + base + 2 * self.m, 0.55)
        for txt, col, tw in chosen:
            self.text(img, txt, (x, self.m + tht), sc, col, th)
            x += tw + gap

    def trail(self, img, pts_disp: np.ndarray, s_pts: np.ndarray, manual: np.ndarray, s_now: int,
              color, all_faded: bool) -> None:
        r = max(2, int(round(3.5 * self.k)))
        faded = tuple(int(0.5 * c + 0.5 * 255) for c in color)      # pale (dark vanishes on grass)
        for (u, v), sp, man in zip(pts_disp, s_pts, manual):
            solid = (not all_faded) and sp > s_now - TRAIL_SOLID
            col, rad = (color, r) if solid else (faded, max(1, r - 1))
            c = (int(round(u)), int(round(v)))
            if man:
                cv2.rectangle(img, (c[0] - rad - 1, c[1] - rad - 1), (c[0] + rad + 1, c[1] + rad + 1),
                              BLACK, 1, cv2.LINE_AA)
                cv2.rectangle(img, (c[0] - rad, c[1] - rad), (c[0] + rad, c[1] + rad), col, -1, cv2.LINE_AA)
            else:
                cv2.circle(img, c, rad + 1, BLACK, 1, cv2.LINE_AA)
                cv2.circle(img, c, rad, col, -1, cv2.LINE_AA)

    def tag(self, img, text: str, head: Optional[tuple[float, float]], color, slot: int = 0) -> None:
        sc, th = 0.8 * self.k, max(1, int(round(2 * self.k)))
        tw, tht, _ = self._size(text, self.bold, sc, th)
        if head is None:
            x, y = self.m, int(70 * self.k) + slot * (tht + self.m)
        else:
            x, y = head[0] + 12 * self.k, head[1] - 12 * self.k
        x = min(max(self.m, x), self.w - tw - self.m)
        y = min(max(tht + self.m + int(40 * self.k), y), self.h - self.m)
        self.text(img, text, (x, y), sc, color, th, self.bold)

    def card(self, img, line1: str, line2: str, color) -> None:
        s1, t1 = 1.15 * self.k, max(1, int(round(2 * self.k)))
        s2, t2 = 0.62 * self.k, max(1, int(round(self.k)))
        w1, h1, b1 = self._size(line1, self.bold, s1, t1)
        w2, h2, b2 = self._size(line2, self.font, s2, t2)
        if w2 > self.w - 4 * self.m:                         # never run off the frame
            s2 *= (self.w - 4 * self.m) / w2
            w2, h2, b2 = self._size(line2, self.font, s2, t2)
        pad = self.m
        bw = max(w1, w2) + 2 * pad
        bh = h1 + b1 + h2 + b2 + 3 * pad
        x0 = (self.w - bw) // 2
        y1 = self.h - int(round(28 * self.k))
        y0 = y1 - bh
        self.shade(img, x0, y0, x0 + bw, y1, 0.65)
        cv2.rectangle(img, (x0, y0), (x0 + bw, y1), color, max(1, int(round(self.k))), cv2.LINE_AA)
        self.text(img, line1, ((self.w - w1) / 2, y0 + pad + h1), s1, color, t1, self.bold)
        self.text(img, line2, ((self.w - w2) / 2, y0 + 2 * pad + h1 + b1 + h2), s2, color, t2)

    def hint(self, img, text: str) -> None:
        """Small orange note under the HUD."""
        sc, th = 0.55 * self.k, max(1, int(round(self.k)))
        self.text(img, text, (self.m, int(round(62 * self.k))), sc, ORANGE, th)

    def gap_frame(self, t: float) -> np.ndarray:
        img = np.zeros((self.h, self.w, 3), np.uint8)
        msg = "no picture: frame missing in the source video"
        sc, th = 0.8 * self.k, max(1, int(round(2 * self.k)))
        tw = self._size(msg, self.font, sc, th)[0]
        self.text(img, msg, ((self.w - tw) / 2, self.h / 2), sc, RED, th)
        return img


# ================================================================= frames
class _FrameDrawer:
    """Draws one output frame for one sample."""

    def __init__(self, marks: list[PitchMark], timeline: Timeline, lookup: TransformLookup,
                 painter: Painter, interlaced: bool, missed: Sequence[tuple[int, int]] = ()):
        self.marks, self.timeline, self.lookup = marks, timeline, lookup
        self.p, self.interlaced = painter, interlaced
        self.missed = list(missed)

    def draw(self, img: np.ndarray, A: np.ndarray, s: int, t: float, frame: Optional[int]) -> np.ndarray:
        st = self.timeline.state(s)
        M = None
        live = list(st.in_flight) + ([st.card] if st.card is not None and st.card not in st.in_flight else [])
        for slot, i in enumerate(live):
            m = self.marks[i]
            pts = m.points[m.points[:, 0] <= s] if len(m.points) else m.points
            head = None
            if len(pts):
                if M is None:
                    M = reference_to_display(self.lookup.get(s), A)
                disp = apply_affine(M, pts[:, 1:3])
                ok = np.all(np.isfinite(disp), axis=1)
                if np.any(ok):
                    self.p.trail(img, disp[ok], pts[ok, 0], pts[ok, 3] > 0, s, m.color,
                                 all_faded=i not in st.in_flight)
                    head = (float(disp[ok][-1, 0]), float(disp[ok][-1, 1]))
            if i in st.in_flight:
                self.p.tag(img, f"P{m.number}", head, m.color, slot)
        if st.card is not None:
            l1, l2, col = card_lines(self.marks[st.card])
            self.p.card(img, l1, l2, col)
        self.p.running_list(img, running_items(self.marks, st.done[-40:]))     # only the last ones fit
        self.p.hud(img, t, frame, s, self.interlaced)
        if any(a <= s <= b for a, b in self.missed):
            self.p.hint(img, "possible missed pitch here? (not measured - check in review)")
        return img


# ================================================================ helpers
def output_fps(source) -> Fraction:
    """One output frame per sample: 2 x frame rate for fields, frame rate /
    frame_step for frames.  Falls back to 1 / sample_dt."""
    from_dt = Fraction(1.0 / float(source.sample_dt)).limit_denominator(1001000)
    info = getattr(getattr(source, "reader", None), "info", None)
    try:
        base = Fraction(str(info.frame_rate_str)) if info is not None else None
    except (ValueError, ZeroDivisionError, AttributeError):
        base = None
    if base is None or base <= 0:
        return from_dt
    fps = base * 2 if getattr(source, "interlaced", False) else base / max(1, int(getattr(source, "frame_step", 1)))
    if abs(float(fps) * float(source.sample_dt) - 1.0) > 1e-3:
        log.warning("frame rate %s does not match the sample spacing; using 1/sample_dt", fps)
        return from_dt
    return fps


def _source_path(source, result: AnalysisResult) -> Optional[str]:
    info = getattr(getattr(source, "reader", None), "info", None)
    p = getattr(info, "path", None) or getattr(getattr(source, "reader", None), "path", None)
    if p and os.path.isfile(p):
        return os.fspath(p)
    if result.video.path and os.path.isfile(result.video.path):
        return result.video.path
    return None


def _to_sample_fn(source) -> Callable[[float], int]:
    n = int(source.n_samples)
    if hasattr(source, "sample_at_time"):
        return lambda t: int(source.sample_at_time(t))
    return lambda t: int(min(max(0, round(t / float(source.sample_dt))), max(0, n - 1)))


def _sar(source, result: AnalysisResult) -> float:
    info = getattr(getattr(source, "reader", None), "info", None)
    sar = getattr(info, "sar", None) or result.video.sar or 1.0
    return float(sar)


def _report_progress(progress, done: int, total: int) -> None:
    if progress is None:
        return
    if hasattr(progress, "update"):
        progress.update(done)
    else:
        progress(done, total)


# ================================================================== audio
def audio_filter_graph(segments: Sequence[tuple[float, float]], offset: float, seek: float) -> str:
    """ffmpeg filter graph cutting one audio piece per segment and joining them.

    segments: (source time of the segment's first picture, duration) in our
    timeline (seconds from the first video frame).  offset: first video
    frame PTS - container start of the source (ffmpeg rebases inputs to the
    container start); seek: the input -ss applied to the audio input.
    Each piece is trimmed sample-accurately, shifted to start at 0, padded
    with silence where the source has no audio, and cut to exactly its
    video duration, so segment k's audio starts with segment k's picture."""
    n = len(segments)
    parts = [f"[1:a:0]asplit={n}" + "".join(f"[i{k}]" for k in range(n))]
    for k, (t0, dur) in enumerate(segments):
        a = offset + float(t0) - seek
        parts.append(f"[i{k}]atrim=start={a:.6f}:end={a + dur:.6f},asetpts=PTS-{a:.6f}/TB,"
                     f"aresample=async=1:first_pts=0,apad=whole_dur={dur:.6f},atrim=end={dur:.6f}[a{k}]")
    parts.append("".join(f"[a{k}]" for k in range(n)) + f"concat=n={n}:v=0:a=1[aout]")
    return ";".join(parts)


def _ffmpeg_exe() -> Optional[str]:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:                                   # noqa: BLE001
        log.info("no ffmpeg for audio muxing: %s", e)
        return None


def mux_segment_audio(video_path: str, audio_src: str, segments: Sequence[tuple[float, float]],
                      out_path: str, ffmpeg_exe: Optional[str] = None) -> str:
    """Mux per-segment source audio into a highlights video (video stream copied).

    Returns 'muxed' or 'skipped: <reason>'.  On 'muxed' out_path holds the
    result; otherwise out_path is untouched (the caller keeps the video)."""
    from pitchspeed.io.writer import source_audio_offset
    exe = ffmpeg_exe or _ffmpeg_exe()
    if not exe:
        return "skipped: no ffmpeg executable"
    if not segments:
        return "skipped: no segments"
    try:
        has_audio, offset = source_audio_offset(audio_src)
    except Exception as e:                                   # noqa: BLE001
        return f"skipped: cannot read {audio_src}: {e}"
    if not has_audio:
        return "skipped: source has no audio"
    seek = max(0.0, offset + float(segments[0][0]) - 1.0)
    graph = audio_filter_graph(segments, offset, seek)
    tmp_out = os.path.splitext(out_path)[0] + ".audio-tmp" + (os.path.splitext(out_path)[1] or ".mp4")
    script = None
    base = [exe, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-i", video_path,
            "-ss", f"{seek:.6f}", "-i", audio_src]
    tail = ["-map", "0:v:0", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart", tmp_out]
    try:
        if len(graph) < 8000:
            attempts = [base + ["-filter_complex", graph] + tail]
        else:                                                # Windows command lines are limited to 32k
            fd, script = tempfile.mkstemp(suffix=".txt", prefix="pitchspeed-audio-",
                                          dir=os.path.dirname(os.path.abspath(out_path)))
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(graph)
            attempts = [base + ["-/filter_complex", script] + tail,
                        base + ["-filter_complex_script", script] + tail]
        err = ""
        for cmd in attempts:
            try:
                p = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT_S,
                                   creationflags=_NO_WINDOW)
            except (OSError, subprocess.SubprocessError) as e:
                err = str(e)
                continue
            if p.returncode == 0 and os.path.isfile(tmp_out) and os.path.getsize(tmp_out) > 0:
                os.replace(tmp_out, out_path)
                return "muxed"
            err = p.stderr.decode("utf-8", "replace").strip()[-800:]
        return f"skipped: ffmpeg audio pass failed: {err}"
    finally:
        for f in (script, tmp_out):
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass


# ================================================================= render
class _WindowContext:
    """Renders sample windows [s0, s1) to the writer, keeping output frame j
    of a window at source time t0 + j / fps (filler frames for gaps)."""

    def __init__(self, source, drawer: _FrameDrawer, painter: Painter, writer, sar: float,
                 size: tuple[int, int], out_dt: float, interlaced: bool, progress, total: int):
        self.source, self.drawer, self.painter, self.writer = source, drawer, painter, writer
        self.sar, self.size, self.out_dt, self.interlaced = sar, size, out_dt, interlaced
        self.progress, self.total, self.done = progress, total, 0
        self.max_fill = int(MAX_FILL_S / out_dt)

    def _fill(self, t0: float, written: int, count: int) -> int:
        for j in range(count):
            tj = t0 + (written + j) * self.out_dt
            g = self.painter.gap_frame(tj)
            self.painter.hud(g, tj, None, None, False, "(gap)")
            self.writer.write(g)
        return count

    def render(self, s0: int, s1: int) -> dict:
        from pitchspeed.io.display import sample_to_display
        t0 = float(self.source.sample_time(s0))
        written = fillers = 0
        for sample in self.source.iter_samples(s0, s1, color=True):
            missing = round_half_up((float(sample.t) - t0) / self.out_dt) - written
            if missing > self.max_fill:
                log.warning("source timing gap of %.1f s before %s; only %.0f s of filler inserted",
                            missing * self.out_dt, format_time(sample.t), MAX_FILL_S)
            if missing > 0:
                n_fill = self._fill(t0, written, min(missing, self.max_fill))
                written += n_fill
                fillers += n_fill
            img, A = sample_to_display(sample, self.sar, self.size, is_field=self.interlaced)
            self.writer.write(self.drawer.draw(img, A, sample.s, float(sample.t), sample.frame))
            written += 1
            self.done += 1
            if self.done % 30 == 0:
                _report_progress(self.progress, self.done, self.total)
        return {"s_start": s0, "s_end": s1, "t_start": t0, "frames": written, "fillers": fillers}


def _default_writer(path, width, height, fps, audio_src=None, audio_start_s=0.0, audio_duration_s=None):
    from pitchspeed.io.writer import VideoWriter
    return VideoWriter(path, width, height, fps=fps, audio_src=audio_src, audio_start_s=audio_start_s,
                       audio_duration_s=audio_duration_s)


def render_annotated(result: AnalysisResult, source, out_path: str, mode: str = "full",
                     transforms: Optional[np.ndarray] = None, transform_samples: Optional[np.ndarray] = None,
                     size=(960, 720), audio: bool = True, progress=None, *,
                     writer_factory: Optional[Callable[..., Any]] = None,
                     info: Optional[dict] = None) -> str:
    """Write the annotated video; returns the path actually written.

    mode: 'full' (whole video) or 'highlights' (around each pitch).
    transforms / transform_samples: per-sample (n, 2, 3) or (n, 3, 3)
        (sample -> reference) stabilization transforms and their sample
        indices (pipeline.load_transforms); None = static camera.
    size: (width, height) of the output, square pixels.
    audio: pass the source audio through (see module doc for highlights).
    progress: callable(done, total) or an object with .update(done)
        (e.g. pipeline.Progress), called with samples processed.
    writer_factory: test hook, called like io.writer.VideoWriter.
    info: optional dict, filled with what happened (backend, audio_status,
        frames_written, filler_frames, segments, fps, out_path, and 'note'
        when the video had to be written under another name).
    The path may end in '.avi' if only the MJPG fallback writer worked, or
    be '<name>-1.mp4' when out_path is locked by another program.
    Raises ValueError for a bad mode or when highlights mode has no pitch.
    """
    if mode not in ("full", "highlights"):
        raise ValueError(f"mode must be 'full' or 'highlights', got {mode!r}")
    out_path = os.fspath(out_path)
    notes: list[str] = []
    if not can_write(out_path):
        requested, out_path = out_path, alternative_path(out_path)
        notes.append(f"{os.path.basename(requested)} is open in another program (a video player?) or "
                     f"read-only, so the new video was written as {os.path.basename(out_path)}")
        log.warning("%s", notes[-1])
    w, h = int(size[0]), int(size[1])
    n = int(source.n_samples)
    sample_dt = float(source.sample_dt)
    fps = output_fps(source)
    out_dt = 1.0 / float(fps)
    marks = build_marks(result, _to_sample_fn(source))
    timeline = Timeline(marks, round_half_up(HOLD_S / sample_dt))
    if mode == "full":
        windows = [(0, n)] if n > 0 else []
    else:
        windows = highlight_windows(marks, n, int(math.ceil(PRE_S / sample_dt)), int(math.ceil(POST_S / sample_dt)))
    if not windows:
        raise ValueError("nothing to render: " + ("the video has no samples" if mode == "full" else
                         "no accepted pitches for a highlights video (use mode='full')"))
    audio_src = _source_path(source, result) if audio else None
    factory = writer_factory or _default_writer
    painter = Painter((w, h))
    drawer = _FrameDrawer(marks, timeline, TransformLookup(transforms, transform_samples), painter,
                          bool(getattr(source, "interlaced", False)), missed_windows(result))
    sar = _sar(source, result)
    highlights_audio = mode == "highlights" and audio_src is not None
    video_target = (os.path.splitext(out_path)[0] + ".highlights-tmp.mp4") if highlights_audio else out_path
    t_first = float(source.sample_time(windows[0][0]))
    writer = factory(video_target, w, h, fps=str(fps) if fps.denominator != 1 else str(fps.numerator),
                     audio_src=audio_src if mode == "full" else None,
                     audio_start_s=t_first if mode == "full" else 0.0, audio_duration_s=None)
    stats = {"mode": mode, "fps": str(fps), "segments": [], "frames_written": 0, "filler_frames": 0,
             "samples_rendered": 0, "marks": len(marks)}
    total = sum(b - a for a, b in windows)
    ctx = _WindowContext(source, drawer, painter, writer, sar, (w, h), out_dt,
                         bool(getattr(source, "interlaced", False)), progress, total)
    salvaged = None
    try:
        for s0, s1 in windows:
            seg = ctx.render(s0, s1)
            stats["segments"].append(seg)
            stats["frames_written"] += seg["frames"]
            stats["filler_frames"] += seg["fillers"]
        stats["samples_rendered"] = ctx.done
        try:
            backend = writer.close()
        except OSError as e:          # e.g. the target got locked while rendering: keep the frames
            salvaged = _salvage_written_video(writer, video_target, out_path)
            if salvaged is None:
                raise
            backend = getattr(writer, "backend", None) or "unknown"
            notes.append(f"could not write {os.path.basename(out_path)} ({e}); the new video (without audio) "
                         f"was kept as {os.path.basename(salvaged)}")
            log.warning("%s", notes[-1])
    except BaseException:
        if hasattr(writer, "abort"):
            writer.abort()
        for tmp in {video_target if highlights_audio else None, _writer_temp(writer, out_path)}:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        raise
    _report_progress(progress, total, total)
    written_path = salvaged or getattr(writer, "out_path", video_target) or video_target
    stats["backend"] = backend
    if salvaged is not None:
        final, audio_status = salvaged, "skipped: the output file could not be written"
    elif highlights_audio:
        final, audio_status = _finish_highlights(written_path, out_path, audio_src, stats["segments"], out_dt)
        if os.path.abspath(final) != os.path.abspath(out_path) and not final.endswith(".avi"):
            notes.append(f"{os.path.basename(out_path)} could not be replaced (open in another program?); "
                         f"the new video was kept as {os.path.basename(final)}")
    else:
        final = written_path
        audio_status = ("off (audio=False)" if not audio else
                        "skipped: source video file not found" if audio_src is None else
                        getattr(writer, "audio_status", "unknown"))
    stats["audio_status"], stats["out_path"] = audio_status, final
    if notes:
        stats["note"] = "; ".join(notes)
    if audio and audio_status != "muxed":
        log.info("annotated video has no audio (%s)", audio_status)
    if info is not None:
        info.update(stats)
    return final


def _finish_highlights(video_path: str, out_path: str, audio_src: str, segments: list[dict],
                       out_dt: float) -> tuple[str, str]:
    """Add per-segment audio to the highlights video; keep it silent on failure."""
    if os.path.splitext(video_path)[1].lower() == ".avi":     # MJPG fallback: no audio support
        final = move_into_place(video_path, os.path.splitext(out_path)[0] + ".avi")
        return final, "skipped: no audio in the MJPG/AVI fallback"
    segs = [(d["t_start"], d["frames"] * out_dt) for d in segments if d["frames"] > 0]
    try:
        status = mux_segment_audio(video_path, audio_src, segs, out_path)
    except OSError as e:                     # e.g. out_path locked: keep the silent video under another name
        status = f"skipped: {e}"
    if status == "muxed":
        try:
            os.remove(video_path)
        except OSError:
            pass
        return out_path, status
    return move_into_place(video_path, out_path), status


# ============================================================ locked files
def can_write(path: str) -> bool:
    """False when `path` exists but cannot be opened for writing (open in a
    video player on Windows, or read-only).  Nothing is written to it."""
    if not os.path.exists(path):
        return True
    try:
        with open(path, "ab"):
            pass
        return True
    except PermissionError:
        return False
    except OSError:
        return True


def alternative_path(path: str) -> str:
    """'<root>-1<ext>', '<root>-2<ext>', ... - the first that does not exist yet."""
    root, ext = os.path.splitext(path)
    i = 1
    while os.path.exists(f"{root}-{i}{ext}") or os.path.exists(f"{root}-{i}.avi"):
        i += 1
    return f"{root}-{i}{ext}"


def move_into_place(src: str, dst: str) -> str:
    """os.replace(src, dst); if dst cannot be replaced (locked), keep src as
    alternative_path(dst) instead.  Returns the path the file ended up at."""
    try:
        os.replace(src, dst)
        return dst
    except OSError as e:
        if not os.path.exists(src):
            raise
        alt = alternative_path(dst)
        os.replace(src, alt)
        log.warning("cannot replace %s (%s); kept the new file as %s", dst, e, alt)
        return alt


def _writer_temp(writer, out_path: str) -> Optional[str]:
    """The writer's intermediate video file when it is not the final one (None otherwise)."""
    p = getattr(getattr(writer, "_b", None), "path", None)
    if p and os.path.abspath(p) != os.path.abspath(out_path):
        return p
    return None


def _salvage_written_video(writer, video_target: str, out_path: str) -> Optional[str]:
    """After writer.close() failed while finishing: move the complete intermediate
    video (if any) to a free name and return that name."""
    for cand in (_writer_temp(writer, out_path), video_target):
        if (cand and os.path.abspath(cand) != os.path.abspath(out_path) and os.path.isfile(cand)
                and os.path.getsize(cand) > 0):
            try:            # keep the container's extension (.avi from the MJPG fallback)
                alt = alternative_path(os.path.splitext(out_path)[0] + (os.path.splitext(cand)[1] or ".mp4"))
                os.replace(cand, alt)
                return alt
            except OSError:
                return None
    return None
