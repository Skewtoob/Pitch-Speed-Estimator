"""8x loupe with a crosshair, for precise clicking.

The loupe magnifies a 31x31 pixel patch of the *display* image (already
square-pixel and bobbed) with nearest-neighbour scaling, so individual
pixels stay visible.  Sub-pixel positions (nudged points, the cursor) are
drawn at their exact place: display coordinate u maps to loupe coordinate
``(u - j0 + 0.5) * zoom - 0.5`` (pixel-centre convention) where j0 is the
first display pixel of the patch.
"""
from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np

from pitchspeed.ui.canvas import BLACK, WHITE, YELLOW, draw_marker, draw_text

LOUPE_HALF = 15          # 31 x 31 display pixels
LOUPE_ZOOM = 8


def loupe_origin(center_uv, half: int = LOUPE_HALF) -> tuple[int, int]:
    """First display pixel (j0, i0) of the patch around `center_uv`."""
    return int(round(center_uv[0])) - half, int(round(center_uv[1])) - half


def to_loupe(uv, center_uv, half: int = LOUPE_HALF, zoom: int = LOUPE_ZOOM) -> tuple[float, float]:
    """Display point -> loupe image coordinates."""
    j0, i0 = loupe_origin(center_uv, half)
    return (uv[0] - j0 + 0.5) * zoom - 0.5, (uv[1] - i0 + 0.5) * zoom - 0.5


def make_loupe(img: np.ndarray, center_uv, half: int = LOUPE_HALF, zoom: int = LOUPE_ZOOM,
               points: Iterable = (), label: Optional[str] = None) -> np.ndarray:
    """Magnified patch of `img` around `center_uv` with a crosshair at the exact centre.

    points: iterable of (uv, color) markers to draw inside the loupe.
    """
    n = 2 * half + 1
    j0, i0 = loupe_origin(center_uv, half)
    pad = n + 2
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    y0, x0 = i0 + pad, j0 + pad
    y0 = int(np.clip(y0, 0, padded.shape[0] - n))
    x0 = int(np.clip(x0, 0, padded.shape[1] - n))
    patch = padded[y0:y0 + n, x0:x0 + n]
    big = cv2.resize(patch, (n * zoom, n * zoom), interpolation=cv2.INTER_NEAREST)
    for uv, color in points:
        lu, lv = to_loupe(uv, center_uv, half, zoom)
        if -zoom <= lu <= n * zoom + zoom and -zoom <= lv <= n * zoom + zoom:
            draw_marker(big, (lu, lv), color, size=zoom * 2, style="circle")
    cu, cv_ = to_loupe(center_uv, center_uv, half, zoom)
    _crosshair(big, cu, cv_, gap=zoom)
    cv2.rectangle(big, (0, 0), (big.shape[1] - 1, big.shape[0] - 1), WHITE, 1)
    if label:
        draw_text(big, label, (6, big.shape[0] - 8), 0.42, YELLOW)
    return big


def _crosshair(img: np.ndarray, u: float, v: float, gap: int) -> None:
    h, w = img.shape[:2]
    sh, f = 4, 16
    cu, cv_ = int(round(u * f)), int(round(v * f))
    g = gap * f
    segs = [((0, cv_), (cu - g, cv_)), ((cu + g, cv_), (w * f, cv_)),
            ((cu, 0), (cu, cv_ - g)), ((cu, cv_ + g), (cu, h * f))]
    for a, b in segs:
        cv2.line(img, a, b, BLACK, 3, cv2.LINE_AA, sh)
        cv2.line(img, a, b, (0, 255, 255), 1, cv2.LINE_AA, sh)


def paste_loupe(canvas: np.ndarray, loupe: np.ndarray, avoid_uv, video_h: Optional[int] = None,
                margin: int = 8) -> tuple[int, int]:
    """Paste the loupe into a corner of the video area that does not hold `avoid_uv`.

    Preference: top-right (the HUD text is top-left), bottom-right,
    bottom-left, top-left.  Returns the loupe's top-left corner.
    """
    h, w = canvas.shape[:2]
    lh, lw = loupe.shape[:2]
    video_h = min(h, video_h or h)
    if lw + 2 * margin > w or lh + 2 * margin > video_h:
        return 0, 0
    top, bottom = margin + 44, video_h - lh - margin
    top = min(top, bottom)
    left, right = margin, w - lw - margin
    u, v = float(avoid_uv[0]), float(avoid_uv[1])
    pad = 12
    for x, y in ((right, top), (right, bottom), (left, bottom), (left, top)):
        if not (x - pad <= u <= x + lw + pad and y - pad <= v <= y + lh + pad):
            break
    canvas[y:y + lh, x:x + lw] = loupe
    return x, y
