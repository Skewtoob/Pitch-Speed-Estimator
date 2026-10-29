"""Overlay graphics of the 'radar' style: the radar-gun speed card, the
running list and pitch labels (Inter font, drawn with Pillow), and the ball
path (drawn with OpenCV).

SPEED.  Cards and labels are drawn once with Pillow (2x supersampled, then
reduced) and kept as premultiplied sprites; a video frame only alpha-blends
them (numpy, a small rectangle).  The path is drawn per frame with OpenCV,
only inside its bounding box.

HONESTY (the same rules as the console table and the CSV, see render/text.py):
* the big number is the release speed in whole mph;
* label 'poor' -> an amber '~' before the digits, 'EST' in the header and
  'rough' after the range;
* label 'none' -> '--' and 'not measurable', never a number;
* the bottom line always carries the uncertainty: +-1 sigma (to 0.5 mph)
  and the 90% range;
* the path is drawn only between the first and the last position where the
  ball was seen, never extended beyond them; every dot is a position the
  tracker measured.

DESIGN UNITS.  Sizes below are pixels of a 720-line output; everything is
scaled by k = output height / 720.

If Pillow or the bundled fonts cannot be loaded, `available()` returns the
reason and the overlay falls back to the plain OpenCV style.
"""
from __future__ import annotations

import io
import logging
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from pitchspeed.render.text import ci90_text, finite, has_number, round_half_up, sigma_text
from pitchspeed.types import SpeedEstimate

log = logging.getLogger(__name__)

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
FONT_FILES = {"bold": "Inter-Bold.otf", "semibold": "Inter-SemiBold.otf"}
SS = 2                                   # supersampling of the Pillow drawings

# --- colours: RGBA for Pillow, BGR for OpenCV
BEZEL = (28, 30, 34, 240)
BEZEL_EDGE = (90, 94, 104, 255)
SCREEN = (14, 4, 4, 255)
LED_ON = (255, 46, 34, 255)
LED_OFF = (60, 12, 10, 255)               # unlit segments, faintly visible like on a real LED
LED_LABEL = (255, 90, 70, 255)
HEADER = (200, 204, 210, 255)
AMBER = (255, 170, 40, 255)
LINE_GOOD = (220, 220, 220, 255)
LINE_POOR = (255, 196, 90, 255)
LINE_NONE = (200, 200, 200, 255)
LIST_BG = (20, 22, 26, 215)
LIST_LABEL = (150, 156, 168, 255)
LIST_VALUE = {"good": (240, 240, 240, 255), "poor": (255, 176, 64, 255), "none": (150, 150, 150, 255)}

PATH_BGR = (255, 200, 64)                 # cyan path line and glow
DOT_BGR = (255, 255, 255)                 # measured positions: white dots ...
DOT_EDGE_BGR = (200, 120, 30)             # ... with a blue rim

# --- radar card geometry (design px)
CARD_W, CARD_H = 270, 150                 # for two digits; each further digit adds DIGIT_STEP
DIGIT_W, DIGIT_H, DIGIT_T, DIGIT_STEP = 48, 66, 9, 62
DIGIT_Y = 40

_font_data: dict = {}
_fonts: dict = {}
_available: Optional[tuple] = None


# ================================================================== fonts
def _font(kind: str, px: float):
    """Inter font `kind` ('bold' | 'semibold') at `px` pixels (cached).

    The file is read into memory first: FreeType can fail on Windows paths
    with non-ASCII characters, a bytes buffer always works."""
    from PIL import ImageFont
    size = max(1, int(round(px)))
    key = (kind, size)
    f = _fonts.get(key)
    if f is None:
        data = _font_data.get(kind)
        if data is None:
            with open(os.path.join(FONT_DIR, FONT_FILES[kind]), "rb") as fh:
                data = fh.read()
            _font_data[kind] = data
        f = ImageFont.truetype(io.BytesIO(data), size)
        _fonts[key] = f
    return f


def available() -> tuple:
    """(True, '') if Pillow and the bundled fonts work, else (False, reason)."""
    global _available
    if _available is None:
        try:
            from PIL import Image, ImageDraw
            im = Image.new("RGBA", (16, 16))
            for kind in FONT_FILES:
                ImageDraw.Draw(im).text((0, 0), "8", font=_font(kind, 12), fill=(255, 255, 255, 255))
            _available = (True, "")
        except Exception as e:                               # noqa: BLE001 - any failure -> plain style
            _available = (False, f"{type(e).__name__}: {e}")
    return _available


# ================================================================ sprites
@dataclass
class Sprite:
    """An overlay picture: premultiplied BGR colour (0..255) and alpha (0..1), float32."""
    premul: np.ndarray                    # (h, w, 3)
    alpha: np.ndarray                     # (h, w, 1)

    @property
    def w(self) -> int:
        return int(self.alpha.shape[1])

    @property
    def h(self) -> int:
        return int(self.alpha.shape[0])


def _to_sprite(im, size: tuple) -> Sprite:
    """Pillow RGBA picture (drawn at SS x) -> Sprite of `size` (w, h)."""
    from PIL import Image
    w, h = max(1, int(size[0])), max(1, int(size[1]))
    pm = im.convert("RGBa").resize((w, h), Image.LANCZOS)    # premultiplied, so edges do not darken
    a = np.asarray(pm, dtype=np.float32) / 255.0
    alpha = np.clip(a[:, :, 3:4], 0.0, 1.0)
    premul = np.minimum(np.clip(a[:, :, 2::-1], 0.0, None), alpha)  # RGB -> BGR; no overshoot
    return Sprite(np.ascontiguousarray(premul * 255.0), np.ascontiguousarray(alpha))


def blend(img: np.ndarray, sp: Sprite, x: int, y: int, opacity: float = 1.0) -> None:
    """Alpha-blend `sp` onto the BGR uint8 image with its top-left corner at (x, y), in place.
    Parts outside the image are cut off."""
    if opacity <= 0.0:
        return
    H, W = img.shape[:2]
    x, y = int(x), int(y)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + sp.w), min(H, y + sp.h)
    if x1 <= x0 or y1 <= y0:
        return
    op = float(min(1.0, opacity))
    a = sp.alpha[y0 - y:y1 - y, x0 - x:x1 - x] * op
    c = sp.premul[y0 - y:y1 - y, x0 - x:x1 - x] * op
    roi = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = np.clip(roi * (1.0 - a) + c + 0.5, 0, 255).astype(np.uint8)


class _Pen:
    """Pillow drawing in design units, at scale f = k * SS."""

    def __init__(self, w: float, h: float, k: float):
        from PIL import Image, ImageDraw
        self.k, self.f = k, k * SS
        self.size = (max(1, int(round(w * self.f))), max(1, int(round(h * self.f))))
        self.out = (max(1, int(round(w * k))), max(1, int(round(h * k))))
        self.im = Image.new("RGBA", self.size, (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)

    def S(self, *v) -> list:
        return [float(x) * self.f for x in v]

    def font(self, kind: str, px: float):
        return _font(kind, px * self.f)

    def width(self, text: str, kind: str, px: float, spacing: float = 0.0) -> float:
        """Text width in design px (spacing: extra gap after each letter, in em)."""
        f = self.font(kind, px)
        if not text:
            return 0.0
        w = sum(self.d.textlength(ch, font=f) for ch in text) if spacing else self.d.textlength(text, font=f)
        return w / self.f + spacing * px * (len(text) - 1)

    def text(self, x: float, y: float, text: str, kind: str, px: float, fill, spacing: float = 0.0,
             stroke: float = 0.0, stroke_fill=None) -> float:
        """Draw `text` with its top-left at (x, y); returns the x where it ends (design px)."""
        f = self.font(kind, px)
        sw = int(round(stroke * self.f))
        if not spacing:
            self.d.text(self.S(x, y), text, font=f, fill=fill, stroke_width=sw, stroke_fill=stroke_fill)
            return x + self.d.textlength(text, font=f) / self.f
        for i, ch in enumerate(text):
            self.d.text(self.S(x, y), ch, font=f, fill=fill, stroke_width=sw, stroke_fill=stroke_fill)
            x += self.d.textlength(ch, font=f) / self.f + (spacing * px if i < len(text) - 1 else 0.0)
        return x

    def sprite(self) -> Sprite:
        return _to_sprite(self.im, self.out)


# ============================================================ radar card
_SEGMENTS = {"0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg", "5": "acdfg",
             "6": "acdefg", "7": "abc", "8": "abcdefg", "9": "abcdfg", "-": "g"}


def _segment_polys(x: float, y: float, w: float, h: float, t: float) -> dict:
    """Seven-segment digit polygons (design px) for a digit box at (x, y), w x h, stroke t."""
    g = t * 0.55
    m = y + h / 2
    return {
        "a": [(x + g, y), (x + w - g, y), (x + w - t, y + t), (x + t, y + t)],
        "d": [(x + t, y + h - t), (x + w - t, y + h - t), (x + w - g, y + h), (x + g, y + h)],
        "g": [(x + t, m - t / 2), (x + w - t, m - t / 2), (x + w - g * 0.4, m),
              (x + w - t, m + t / 2), (x + t, m + t / 2), (x + g * 0.4, m)],
        "f": [(x, y + g), (x + t, y + t), (x + t, m - t / 2), (x, m - g * 0.3)],
        "b": [(x + w, y + g), (x + w, m - g * 0.3), (x + w - t, m - t / 2), (x + w - t, y + t)],
        "e": [(x, m + g * 0.3), (x + t, m + t / 2), (x + t, y + h - t), (x, y + h - g)],
        "c": [(x + w, m + g * 0.3), (x + w, y + h - g), (x + w - t, y + h - t), (x + w - t, m + t / 2)],
    }


def card_digits(sp: Optional[SpeedEstimate]) -> str:
    """The LED digits: whole mph (at least two places, a blank in front of a single digit)
    or '--' when there is no number."""
    if not has_number(sp):
        return "--"
    return str(round_half_up(sp.v_release_mph)).rjust(2)


def card_state(sp: Optional[SpeedEstimate]) -> str:
    """'good' (labels good / fair), 'poor' (rough number) or 'none' (no number)."""
    if not has_number(sp):
        return "none"
    return "poor" if sp.label == "poor" else "good"


def card_line(sp: Optional[SpeedEstimate]) -> str:
    """Bottom line of the card: '±2  90%: 78–84', '±4.5  90%: 58–74  rough' or 'not measurable'."""
    if not has_number(sp):
        return "not measurable"
    parts = []
    if finite(sp.sigma_mph):
        parts.append(f"±{sigma_text(sp.sigma_mph)}")
    if ci90_text(sp) != "-":
        lo, hi = sp.ci90_mph
        parts.append(f"90%: {round_half_up(lo)}–{round_half_up(hi)}")
    if sp.label == "poor":
        parts.append("rough")
    return "  ".join(parts) if parts else "no error estimate"


def card_size(sp: Optional[SpeedEstimate]) -> tuple:
    """(w, h) of the card in design px."""
    return CARD_W + DIGIT_STEP * max(0, len(card_digits(sp)) - 2), CARD_H


def radar_card(number: int, sp: Optional[SpeedEstimate], k: float) -> Sprite:
    """The radar-gun speed card for one pitch: black bezel, glowing red LED digits."""
    from PIL import Image, ImageDraw, ImageFilter
    state, digits, line = card_state(sp), card_digits(sp), card_line(sp)
    W, H = card_size(sp)
    p = _Pen(W, H, k)
    d = p.d
    d.rounded_rectangle(p.S(0, 0, W - 1, H - 1), radius=18 * p.f, fill=BEZEL, outline=BEZEL_EDGE,
                        width=max(1, int(round(3 * p.f))))
    d.rounded_rectangle(p.S(14, 30, W - 14, 116), radius=8 * p.f, fill=SCREEN)
    head_end = p.text(18, 9, f"PITCH {number}", "bold", 14, HEADER, spacing=0.18)
    peak_w = p.width("PEAK", "bold", 14, spacing=0.2)
    p.text(W - 22 - peak_w, 9, "PEAK", "bold", 14, LED_LABEL, spacing=0.2)
    if state == "poor":
        est_x = max(120.0, head_end + 12)
        if est_x + p.width("EST", "bold", 14, spacing=0.2) < W - 30 - peak_w:
            p.text(est_x, 9, "EST", "bold", 14, AMBER, spacing=0.2)

    x0 = 52 if state == "poor" else 34
    glow = Image.new("RGBA", p.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)

    def draw_digits(draw, off_fill):
        for i, ch in enumerate(digits):
            lit = _SEGMENTS.get(ch, "")
            polys = _segment_polys(x0 + DIGIT_STEP * i, DIGIT_Y, DIGIT_W, DIGIT_H, DIGIT_T)
            for seg, poly in polys.items():
                fill = LED_ON if seg in lit else off_fill
                if fill is not None:
                    draw.polygon([(px * p.f, py * p.f) for px, py in poly], fill=fill)

    draw_digits(d, LED_OFF)
    draw_digits(gd, None)
    p.im.alpha_composite(glow.filter(ImageFilter.GaussianBlur(7 * p.f)))
    draw_digits(d, LED_OFF)                                  # crisp digits on top of the glow
    if state == "poor":
        p.text(18, 46, "~", "bold", 44, AMBER)
    p.text(x0 + DIGIT_STEP * len(digits) + 12, 78, "MPH", "bold", 22, LED_LABEL)

    col = {"good": LINE_GOOD, "poor": LINE_POOR, "none": LINE_NONE}[state]
    px = 15.0
    while px > 8 and p.width(line, "semibold", px) > W - 36:     # never run past the bezel
        px -= 0.5
    p.text(18, 122, line, "semibold", px, col)
    return p.sprite()


# ======================================================= broadcast box (A)
TV_BG = (16, 30, 62, 235)          # navy panel
TV_BAND = (200, 30, 40, 255)       # red top band
TV_TEXT = (255, 255, 255, 255)
TV_SUB = (190, 205, 230, 255)
BROADCAST_H = 112


def broadcast_size(sp: Optional[SpeedEstimate]) -> tuple:
    return 190 + 34 * max(0, len(card_digits(sp).strip()) - 2), BROADCAST_H


def broadcast_card(number: int, sp: Optional[SpeedEstimate], k: float) -> Sprite:
    """TV pitch box: red band with the pitch number, big white speed, MPH, the +/- line."""
    state, line = card_state(sp), card_line(sp)
    digits = card_digits(sp).strip()
    W, H = broadcast_size(sp)
    p = _Pen(W, H, k)
    d = p.d
    d.rounded_rectangle(p.S(0, 0, W - 1, H - 1), radius=8 * p.f, fill=TV_BG)
    d.rounded_rectangle(p.S(0, 0, W - 1, 24), radius=8 * p.f, fill=TV_BAND)
    d.rectangle(p.S(0, 14, W - 1, 24), fill=TV_BAND)
    p.text(10, 4, f"PITCH {number}", "bold", 14, TV_TEXT, spacing=0.12)
    if state == "poor":
        ew = p.width("ESTIMATE", "bold", 12, spacing=0.1)
        p.text(W - 10 - ew, 5, "ESTIMATE", "bold", 12, AMBER, spacing=0.1)
    big = ("~" if state == "poor" else "") + digits
    x_end = p.text(12, 22, big, "bold", 62, AMBER if state == "poor" else TV_TEXT)
    p.text(x_end + 8, 54, "MPH", "bold", 20, TV_SUB)
    col = {"good": TV_SUB, "poor": LINE_POOR, "none": LINE_NONE}[state]
    px = 14.0
    while px > 8 and p.width(line, "semibold", px) > W - 20:
        px -= 0.5
    p.text(12, 92, line, "semibold", px, col)
    return p.sprite()


# ========================================================= sports bar (C)
BAR_BG = (12, 12, 14, 225)
BAR_ACCENT = (230, 120, 20, 255)   # orange block with the pitch number
BAR_H = 52


def bar_size(sp: Optional[SpeedEstimate]) -> tuple:
    return 420, BAR_H


def bar_strip(number: int, sp: Optional[SpeedEstimate], k: float) -> Sprite:
    """Lower-third strip: orange block 'P3', then '78 MPH' and the +/- line."""
    state, line = card_state(sp), card_line(sp)
    digits = card_digits(sp).strip()
    W, H = bar_size(sp)
    p = _Pen(W, H, k)
    d = p.d
    d.rectangle(p.S(0, 0, W - 1, H - 1), fill=BAR_BG)
    d.rectangle(p.S(0, 0, 64, H - 1), fill=BAR_ACCENT)
    d.rectangle(p.S(0, H - 4, W - 1, H - 1), fill=BAR_ACCENT)
    tw = p.width(f"P{number}", "bold", 24)
    p.text((64 - tw) / 2, 10, f"P{number}", "bold", 24, TV_TEXT)
    big = ("~" if state == "poor" else "") + digits
    x = p.text(78, 4, big, "bold", 38, AMBER if state == "poor" else TV_TEXT)
    x = p.text(x + 6, 20, "MPH", "bold", 16, TV_SUB)
    col = {"good": TV_SUB, "poor": LINE_POOR, "none": LINE_NONE}[state]
    px = 14.0
    while px > 8 and p.width(line, "semibold", px) > W - x - 30:
        px -= 0.5
    p.text(x + 18, 19, line, "semibold", px, col)
    return p.sprite()


# ============================================================ small labels
def tag_label(text: str, k: float) -> Sprite:
    """White bold label with a dark outline (the 'P3' next to the ball in flight)."""
    px, stroke = 18.0, 2.5
    from PIL import Image, ImageDraw
    f = _font("bold", px * k * SS)
    sw = int(round(stroke * k * SS))
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    x0, y0, x1, y1 = probe.textbbox((0, 0), text, font=f, stroke_width=sw)
    pad = 2 * SS
    im = Image.new("RGBA", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((pad - x0, pad - y0), text, font=f, fill=(255, 255, 255, 255),
                            stroke_width=sw, stroke_fill=(0, 0, 0, 220))
    return _to_sprite(im, (int(round(im.size[0] / SS)), int(round(im.size[1] / SS))))


def value_kind(value: str) -> str:
    """Colour class of a running-list value from text.speed_short: '78', '~78?', 'n/a'."""
    if value == "n/a":
        return "none"
    return "poor" if value.startswith("~") else "good"


LIST_H, LIST_PAD, LIST_GAP = 30.0, 10.0, 14.0
ELLIPSIS = "…"


def _list_item_width(p: _Pen, label: str, value: str) -> float:
    return p.width(label, "semibold", 12) + 4 + p.width(value, "bold", 18)


def running_strip(items: Sequence[tuple], k: float, max_w: float) -> Optional[Sprite]:
    """The running list 'P1 76  P2 ~65?  P3 n/a' as a dark strip.

    items: (label, value) in pitch order; the most recent ones that fit into
    max_w output px are shown, with a leading '…' when older ones are left out."""
    if not items:
        return None
    probe = _Pen(1, 1, k)
    limit = max_w / max(k, 1e-6) - 2 * LIST_PAD
    chosen, width = [], 0.0
    for label, value in reversed(list(items)):
        w = _list_item_width(probe, label, value)
        extra = w + (LIST_GAP if chosen else 0.0)
        if chosen and width + extra > limit:
            break
        chosen.insert(0, (label, value, w))
        width += extra
    truncated = len(chosen) < len(items)
    ell_w = probe.width(ELLIPSIS, "bold", 18)
    if truncated:
        width += ell_w + LIST_GAP
    W = width + 2 * LIST_PAD
    p = _Pen(W, LIST_H, k)
    p.d.rounded_rectangle(p.S(0, 0, W - 1, LIST_H - 1), radius=8 * p.f, fill=LIST_BG, outline=BEZEL_EDGE,
                          width=max(1, int(round(1.5 * p.f))))
    x = LIST_PAD
    if truncated:
        p.text(x, 3, ELLIPSIS, "bold", 18, LIST_LABEL)
        x += ell_w + LIST_GAP
    for label, value, w in chosen:
        lx = p.text(x, 8.5, label, "semibold", 12, LIST_LABEL)
        p.text(lx + 4, 3, value, "bold", 18, LIST_VALUE[value_kind(value)])
        x += w + LIST_GAP
    return p.sprite()


class SpriteCache:
    """Small least-recently-used cache of sprites."""

    def __init__(self, size: int = 8):
        self.size = size
        self._d: OrderedDict = OrderedDict()

    def get(self, key, make):
        sp = self._d.get(key)
        if sp is None:
            sp = make()
            self._d[key] = sp
            while len(self._d) > self.size:
                self._d.popitem(last=False)
        else:
            self._d.move_to_end(key)
        return sp


# =================================================================== path
def thin_dots(pts: np.ndarray, min_dist: float) -> np.ndarray:
    """Keep the first point and then each point at least `min_dist` px from the last kept one,
    so dots never pile up into a blob (e.g. a ball coming straight at the camera)."""
    if len(pts) < 2 or min_dist <= 0:
        return pts
    keep = [0]
    last = pts[0]
    for i in range(1, len(pts)):
        if math.hypot(pts[i, 0] - last[0], pts[i, 1] - last[1]) >= min_dist:
            keep.append(i)
            last = pts[i]
    return pts[keep]


def path_style(k: float) -> dict:
    """Line width, glow and dot radius (output px) for scale k."""
    return {"line": max(1.0, 3.0 * k), "glow": max(1.0, 5.0 * k), "dot": max(1.5, 4.0 * k),
            "rim": max(1.0, 1.0 * k)}


def _runs(seen: np.ndarray) -> list[tuple[bool, int, int]]:
    """[(seen, i0, i1)] maximal runs of equal flags, each run's last index shared with the next run's
    first, so the pieces join up."""
    n = len(seen)
    cuts = [0] + [i for i in range(1, n) if seen[i] != seen[i - 1]] + [n]
    return [(bool(seen[a]), a, min(b, n - 1)) for a, b in zip(cuts[:-1], cuts[1:]) if min(b, n - 1) > a]


def _dashes(pts: np.ndarray, dash: float, gap: float) -> list[np.ndarray]:
    """Cut the polyline `pts` (n, 2) into dashes of length `dash` separated by `gap` (px)."""
    seg = np.hypot(*np.diff(pts, axis=0).T)
    L = np.concatenate([[0.0], np.cumsum(seg)])
    out, a = [], 0.0
    while a < L[-1] and len(out) < 2000:
        b = min(a + dash, L[-1])
        inner = (L > a) & (L < b)
        xs = np.concatenate([[np.interp(a, L, pts[:, 0])], pts[inner, 0], [np.interp(b, L, pts[:, 0])]])
        ys = np.concatenate([[np.interp(a, L, pts[:, 1])], pts[inner, 1], [np.interp(b, L, pts[:, 1])]])
        out.append(np.column_stack([xs, ys]))
        a = b + gap
    return out


def draw_path(img: np.ndarray, line: np.ndarray, dots: np.ndarray, k: float, opacity: float = 1.0,
              color=PATH_BGR, seen: Optional[np.ndarray] = None) -> None:
    """Glowing line through `line` (n, 2 display px) and a white dot at each of `dots`, in place.

    seen: per line point, False where the ball was not seen (a gap in the track); those
    stretches are drawn as a thin dashed line without glow.  A thin dark edge under the
    line keeps it visible on a bright sky, the glow on dark grass.  opacity < 1 fades
    everything together."""
    line = np.asarray(line, dtype=np.float64).reshape(-1, 2)
    dots = np.asarray(dots, dtype=np.float64).reshape(-1, 2)
    if (len(line) < 2 and not len(dots)) or opacity <= 0.0:
        return
    if seen is None or len(seen) != len(line):
        seen = np.ones(len(line), bool)
    st = path_style(k)
    pad = int(math.ceil(3.0 * st["glow"] + st["dot"] + st["rim"] + 3))
    pts = np.vstack([line, dots])
    H, W = img.shape[:2]
    x0 = max(0, int(math.floor(pts[:, 0].min())) - pad)
    y0 = max(0, int(math.floor(pts[:, 1].min())) - pad)
    x1 = min(W, int(math.ceil(pts[:, 0].max())) + pad + 1)
    y1 = min(H, int(math.ceil(pts[:, 1].max())) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    off = np.array([x0, y0], dtype=np.float64)
    if len(line) >= 2:
        def fix(p):
            return np.round((p - off) * 16.0).astype(np.int32).reshape(-1, 1, 2)
        solid, dashed = [], []
        for ok, a, b in _runs(np.asarray(seen, bool)):
            (solid if ok else dashed).append(line[a:b + 1])
        core = np.zeros(roi.shape[:2], np.uint8)
        edge = np.zeros(roi.shape[:2], np.uint8)
        thin = np.zeros(roi.shape[:2], np.uint8)
        if solid:
            q = [fix(p) for p in solid]
            cv2.polylines(edge, q, False, 255, max(1, int(round(st["line"] + 2.0 * st["rim"]))),
                          cv2.LINE_AA, shift=4)
            cv2.polylines(core, q, False, 255, max(1, int(round(st["line"]))), cv2.LINE_AA, shift=4)
        if dashed:
            q = [fix(d) for p in dashed for d in _dashes(p, 7.0 * k, 6.0 * k)]
            if q:
                cv2.polylines(edge, q, False, 255, max(1, int(round(st["line"]))), cv2.LINE_AA, shift=4)
                cv2.polylines(thin, q, False, 255, max(1, int(round(0.6 * st["line"]))), cv2.LINE_AA,
                              shift=4)
        if solid:                                            # glow only where the ball was seen
            g = np.clip(cv2.GaussianBlur(core.astype(np.float32) * (1.0 / 255.0), (0, 0), st["glow"]) * 1.6,
                        0.0, 1.0)
            a_glow = (1.0 - (1.0 - g) ** 2) * 0.8
        else:
            a_glow = np.zeros(roi.shape[:2], np.float32)
        a_core = np.maximum(core, thin).astype(np.float32) * (1.0 / 255.0)
        a_edge = edge.astype(np.float32) * (0.45 / 255.0)
        # dark edge, then the glow, then the core, composited in one pass:
        # out = img (1-e)(1-g)(1-c) + colour (g (1-c) + c)
        keep = (1.0 - a_edge) * (1.0 - a_glow) * (1.0 - a_core)
        add = a_glow * (1.0 - a_core) + a_core
        work = roi.astype(np.float32) * keep[:, :, None] + add[:, :, None] * np.asarray(color, np.float32)
        out = np.clip(work + 0.5, 0, 255).astype(np.uint8)
    else:
        out = roi.copy()
    r16, e16 = int(round(st["dot"] * 16)), int(round((st["dot"] + st["rim"]) * 16))
    for u, v in dots:
        c16 = (int(round((u - x0) * 16)), int(round((v - y0) * 16)))
        cv2.circle(out, c16, e16, DOT_EDGE_BGR, -1, cv2.LINE_AA, shift=4)
        cv2.circle(out, c16, r16, DOT_BGR, -1, cv2.LINE_AA, shift=4)
    if opacity < 1.0:
        out = np.clip(roi.astype(np.float32) * (1.0 - opacity) + out.astype(np.float32) * opacity + 0.5,
                      0, 255).astype(np.uint8)
    roi[:] = out
