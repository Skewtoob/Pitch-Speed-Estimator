"""Display canvas for the interactive windows: layout, coordinates, drawing.

LAYOUT.  The window shows one image of ``video_size`` (default 720x540: a
4:3 square-pixel picture; ``ui_scale`` multiplies it) with an optional
info strip of ``strip_h`` pixels underneath.  The video picture is made by
`pitchspeed.io.display.sample_to_display`, which bobs a field to full
height with the parity offset and resizes to square pixels, returning the
2x3 affine A: frame pixel coords -> display coords.  The same A holds for
both fields of a frame and for the woven full frame.

COORDINATES.  UI state machines work in *frame pixel coordinates* of the
stored (non-square) pixels, like everything else (types.py).  A mouse
click at display (u, v) is converted with `Canvas.to_frame` (the inverse
of A); markers are drawn at `Canvas.to_display(x, y)`.  Because the bob
includes the field's parity offset, a click on a bobbed field maps to the
frame row where that picture content really is.

TEXT.  OpenCV's Hershey fonts only know ASCII; every string goes through
`ascii_safe` ('±' -> '+/-' etc.) and is drawn with a dark outline so it
reads on any background.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from pitchspeed.types import Sample

log = logging.getLogger("pitchspeed")

DEFAULT_VIDEO_SIZE = (720, 540)
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (60, 220, 60)
RED = (60, 60, 255)
YELLOW = (40, 230, 255)
ORANGE = (0, 160, 255)
CYAN = (255, 230, 60)
GRAY = (160, 160, 160)
MAGENTA = (255, 80, 255)

_ASCII_MAP = {"±": "+/-", "·": "|", "–": "-", "—": "-", "→": "->",
              "←": "<-", "↑": "^", "↓": "v", "×": "x", "…": "...",
              "‘": "'", "’": "'", "“": '"', "”": '"', "°": " deg",
              "≤": "<=", "≥": ">="}


# ------------------------------------------------------------------ scale --
def resolve_ui_scale(scale: Optional[float] = None) -> float:
    """UI scale factor: explicit value, else $PITCHSPEED_UI_SCALE, else 1.0 (clamped 0.5..3)."""
    if scale is None:
        try:
            scale = float(os.environ.get("PITCHSPEED_UI_SCALE", "1") or 1)
        except ValueError:
            scale = 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return float(min(3.0, max(0.5, scale)))


def video_size_for(scale: float = 1.0, base=DEFAULT_VIDEO_SIZE) -> tuple[int, int]:
    """Canvas video size for a UI scale (even numbers)."""
    w = int(round(base[0] * scale / 2.0)) * 2
    h = int(round(base[1] * scale / 2.0)) * 2
    return max(2, w), max(2, h)


# ------------------------------------------------------------ coordinates --
def affine_inverse(A: np.ndarray) -> np.ndarray:
    """Inverse of a 2x3 affine map."""
    M = np.vstack([np.asarray(A, dtype=np.float64), [0.0, 0.0, 1.0]])
    return np.linalg.inv(M)[:2]


def apply_affine(A: np.ndarray, pts) -> np.ndarray:
    """(N, 2) points through a 2x3 affine map."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    A = np.asarray(A, dtype=np.float64)
    return p @ A[:, :2].T + A[:, 2]


def _display(sample: Sample, sar: float, size, is_field: Optional[bool]):
    from pitchspeed.io.display import sample_to_display
    if is_field is None:
        return sample_to_display(sample, sar, size)
    try:
        return sample_to_display(sample, sar, size, is_field=is_field)
    except TypeError:          # contract version without the keyword
        return sample_to_display(sample, sar, size)


@dataclass
class Canvas:
    """Window layout + frame <-> display mapping.

    frame_size: (width, height) of a full video frame in stored pixels.
    sar: stored-pixel aspect ratio (x_square = x * sar).
    """
    frame_size: tuple[int, int]
    sar: float = 1.0
    video_size: tuple[int, int] = DEFAULT_VIDEO_SIZE
    strip_h: int = 0
    A: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        w, h = int(self.frame_size[0]), int(self.frame_size[1])
        vw, vh = int(self.video_size[0]), int(self.video_size[1])
        s = min(vw / (w * self.sar), vh / h)
        sx, sy = self.sar * s, s
        ox, oy = (vw - w * sx) / 2.0, (vh - h * sy) / 2.0
        # same convention as io.display.display_transform (pixel centres)
        self.A = np.array([[sx, 0.0, ox + 0.5 * sx - 0.5],
                           [0.0, sy, oy + 0.5 * sy - 0.5]], dtype=np.float64)

    # ------------------------------------------------------------ sizes
    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of the whole window image."""
        return int(self.video_size[0]), int(self.video_size[1]) + int(self.strip_h)

    @property
    def px_per_row(self) -> float:
        """Display pixels per frame row (vertical scale)."""
        return float(self.A[1, 1])

    def blank(self) -> np.ndarray:
        w, h = self.size
        return np.zeros((h, w, 3), np.uint8)

    # ------------------------------------------------------ coordinates
    def to_display(self, pts) -> np.ndarray:
        """Frame pixel coords (N, 2) -> display coords (N, 2)."""
        return apply_affine(self.A, pts)

    def to_frame(self, pts) -> np.ndarray:
        """Display coords (N, 2) (e.g. a mouse click) -> frame pixel coords (N, 2)."""
        return apply_affine(affine_inverse(self.A), pts)

    def in_video(self, u: float, v: float) -> bool:
        """True if display point (u, v) lies on the picture (not bars or strip)."""
        x, y = self.to_frame([(u, v)])[0]
        w, h = self.frame_size
        return -0.5 <= x <= w - 0.5 and -0.5 <= y <= h - 0.5

    # ---------------------------------------------------------- images
    def render_sample(self, sample: Sample, is_field: Optional[bool] = None) -> np.ndarray:
        """Window image with the (bobbed) sample in the video area."""
        img, A = _display(sample, self.sar, self.video_size, is_field)
        self._check_A(A)
        return self._compose(img)

    def render_frame(self, frame: np.ndarray) -> np.ndarray:
        """Window image with a full (woven) frame in the video area."""
        smp = Sample(s=0, frame=0, k=0, parity=0, t=0.0,
                     image=frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                     bgr=frame if frame.ndim == 3 else None)
        img, A = _display(smp, self.sar, self.video_size, False)
        self._check_A(A)
        return self._compose(img)

    def _check_A(self, A: np.ndarray) -> None:
        if not np.allclose(np.asarray(A, dtype=np.float64), self.A, atol=1e-6):
            # trust the display module: it produced the picture (tests pin both to one convention)
            log.debug("display transform differs from the canvas layout: %s vs %s", A, self.A)
            self.A = np.asarray(A, dtype=np.float64).copy()

    def _compose(self, video: np.ndarray) -> np.ndarray:
        if video.ndim == 2:
            video = cv2.cvtColor(video, cv2.COLOR_GRAY2BGR)
        if self.strip_h <= 0:
            return np.ascontiguousarray(video)
        out = self.blank()
        out[:video.shape[0], :video.shape[1]] = video
        return out


def weave_fields(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Full frame from its two fields (top = even rows, bottom = odd rows)."""
    h = top.shape[0] + bottom.shape[0]
    out = np.empty((h,) + top.shape[1:], dtype=top.dtype)
    out[0::2] = top
    out[1::2] = bottom
    return out


# ------------------------------------------------------------------- text --
def ascii_safe(text: str) -> str:
    """Replace characters the Hershey fonts cannot draw."""
    out = []
    for ch in str(text):
        if ch in _ASCII_MAP:
            out.append(_ASCII_MAP[ch])
        elif 32 <= ord(ch) < 127:
            out.append(ch)
        else:
            out.append("?")
    return "".join(out)


def text_size(text: str, scale: float = 0.5, thickness: int = 1) -> tuple[int, int]:
    (w, h), base = cv2.getTextSize(ascii_safe(text), FONT, scale, thickness)
    return w, h + base


def draw_text(img: np.ndarray, text: str, org, scale: float = 0.5, color=WHITE,
              thickness: int = 1, outline=BLACK) -> None:
    """Text with a dark outline; `org` is the baseline-left point.

    The outline is the same text drawn at the 8 neighbouring offsets: a
    thicker stroke would not do, because Hershey glyph advances grow with
    the thickness and the outline would drift to the right.
    """
    t = ascii_safe(text)
    x, y = int(round(org[0])), int(round(org[1]))
    if outline is not None:
        for dx, dy in _OUTLINE_OFFSETS:
            cv2.putText(img, t, (x + dx, y + dy), FONT, scale, outline, thickness, cv2.LINE_AA)
    cv2.putText(img, t, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


_OUTLINE_OFFSETS = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))


def wrap_text(text: str, max_width: int, scale: float = 0.5, thickness: int = 1) -> list[str]:
    """Greedy word wrap to `max_width` display pixels."""
    words = ascii_safe(text).split()
    lines, cur = [], ""
    for w in words:
        trial = w if not cur else cur + " " + w
        if text_size(trial, scale, thickness)[0] <= max_width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def shade(img: np.ndarray, rect, alpha: float = 0.55, color=BLACK) -> None:
    """Darken a rectangle (x0, y0, x1, y1) in place (translucent panel)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in rect)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi.astype(np.float32) * (1 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)


def draw_lines(img: np.ndarray, lines: Sequence, org=(8, 20), scale: float = 0.5,
               line_h: Optional[int] = None, panel: bool = True, max_width: Optional[int] = None) -> int:
    """Draw text lines (str or (str, color)) top-down; returns the y below the last line."""
    line_h = line_h or int(round(26 * scale + 8))
    items = []
    max_width = max_width or (img.shape[1] - org[0] - 8)
    for ln in lines:
        text, color = (ln, WHITE) if isinstance(ln, str) else (ln[0], ln[1])
        for part in wrap_text(text, max_width, scale):
            items.append((part, color))
    if not items:
        return org[1]
    if panel:
        wmax = max(text_size(t, scale)[0] for t, _ in items)
        shade(img, (org[0] - 5, org[1] - line_h + 6, org[0] + wmax + 6, org[1] + line_h * (len(items) - 1) + 8))
    y = org[1]
    for t, c in items:
        draw_text(img, t, (org[0], y), scale, c)
        y += line_h
    return y


# ---------------------------------------------------------------- markers --
def draw_marker(img: np.ndarray, uv, color=GREEN, label: Optional[str] = None, size: int = 9,
                style: str = "cross", thickness: int = 1, label_color=None) -> None:
    """Marker at display point uv (sub-pixel accurate), optional label."""
    u, v = float(uv[0]), float(uv[1])
    sh = 4
    f = 1 << sh
    cu, cv_ = int(round(u * f)), int(round(v * f))
    s = size * f
    if style == "cross":
        gap = 3 * f
        for (dx0, dy0, dx1, dy1) in ((-s, 0, -gap, 0), (gap, 0, s, 0), (0, -s, 0, -gap), (0, gap, 0, s)):
            cv2.line(img, (cu + dx0, cv_ + dy0), (cu + dx1, cv_ + dy1), BLACK, thickness + 2, cv2.LINE_AA, sh)
            cv2.line(img, (cu + dx0, cv_ + dy0), (cu + dx1, cv_ + dy1), color, thickness, cv2.LINE_AA, sh)
    elif style == "circle":
        cv2.circle(img, (cu, cv_), s, BLACK, thickness + 2, cv2.LINE_AA, sh)
        cv2.circle(img, (cu, cv_), s, color, thickness, cv2.LINE_AA, sh)
    elif style == "dot":
        cv2.circle(img, (cu, cv_), max(f, s // 3), color, -1, cv2.LINE_AA, sh)
    if label:
        draw_text(img, label, (u + size + 3, v - size + 2), 0.45, label_color or color)


def draw_rect(img: np.ndarray, p0, p1, color=MAGENTA, thickness: int = 1, label: Optional[str] = None) -> None:
    x0, y0 = int(round(min(p0[0], p1[0]))), int(round(min(p0[1], p1[1])))
    x1, y1 = int(round(max(p0[0], p1[0]))), int(round(max(p0[1], p1[1])))
    cv2.rectangle(img, (x0, y0), (x1, y1), BLACK, thickness + 2)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
    if label:
        draw_text(img, label, (x0 + 3, y0 + 14), 0.42, color)


# ------------------------------------------------------------------- help --
def draw_help(img: np.ndarray, title: str, rows: Iterable[tuple[str, str]],
              footer: Optional[str] = None, scale: float = 0.45) -> None:
    """Full-window help overlay: a title and (keys, description) rows in two columns."""
    h, w = img.shape[:2]
    shade(img, (0, 0, w, h), alpha=0.78)
    rows = list(rows)
    line_h = int(round(26 * scale + 4))
    draw_text(img, title, (16, 28), 0.6, YELLOW)
    top = 52
    per_col = max(1, (h - top - 30) // line_h)
    lines = _help_lines(rows, w - 24, scale)
    ncols = 1
    if len(lines) > per_col:
        ncols = 2
        lines = _help_lines(rows, (w - 24) // 2, scale)
        per_col = min(per_col, -(-len(lines) // 2))
    col_w = (w - 24) // ncols
    key_w = _help_key_w(rows, col_w, scale)
    for i, (kind, k, d) in enumerate(lines):
        c, r = divmod(i, per_col)
        if c >= ncols:
            break
        x, y = 16 + c * col_w, top + r * line_h
        if kind == "section":
            draw_text(img, d, (x, y), scale, CYAN)
            continue
        if k:
            draw_text(img, k, (x, y), scale, YELLOW)
        draw_text(img, d, (x + key_w, y), scale)
    if footer:
        draw_text(img, footer, (16, h - 12), scale, GRAY)


def _help_key_w(rows, col_w: int, scale: float) -> int:
    return min(int(col_w * 0.36), max([text_size(k, scale)[0] for k, _ in rows if k] + [10]) + 12)


def _help_lines(rows, col_w: int, scale: float) -> list[tuple[str, str, str]]:
    """Flatten help rows into display lines: ('section'|'row', keys, text)."""
    key_w = _help_key_w(rows, col_w, scale)
    out = []
    for k, d in rows:
        if not k:
            out.extend(("section", "", part) for part in wrap_text(d, col_w - 10, scale))
            continue
        for j, part in enumerate(wrap_text(d, col_w - key_w - 10, scale)):
            out.append(("row", k if j == 0 else "", part))
    return out
