"""Display images of samples: parity-aware bob + square-pixel scaling.

A field sample has H/2 rows and field row r is frame row 2r + p.  Showing
it with a naive resize to full height puts both parities on the same rows,
so the picture jumps one frame line up and down between consecutive fields.
Here every image is warped (cv2.warpAffine) with the exact map

    frame pixel (x, y)  ->  display (u, v) = A @ (x, y, 1)

composed with r -> 2r + p for fields, so a static scene shows no vertical
jump at all (verified in tests: 0.00 px vs 1 frame line for a resize).

A maps *frame pixel coordinates* (stored pixels, see types.py) to display
pixels, with pixel-centre conventions: the frame extent [-0.5, W-0.5] x
[-0.5, H-0.5] maps onto the content rectangle.  The content keeps square
pixels (x_square = x * sar) and is centred in `out_size` with black bars
when the aspects differ.  Use `map_points` / `display_to_frame` to convert
overlay or mouse coordinates.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from pitchspeed.types import Sample

FIELD_ASPECT_THRESHOLD = 2.4   # display aspect above which an image is taken to be a field


def display_transform(width: int, height: int, sar: float, out_size=(960, 720)) -> np.ndarray:
    """2x3 affine A: frame pixel coords of a width x height frame -> display coords."""
    ow, oh = int(out_size[0]), int(out_size[1])
    if width <= 0 or height <= 0 or sar <= 0 or ow <= 0 or oh <= 0:
        raise ValueError(f"bad geometry: {width}x{height} sar {sar} -> {out_size}")
    s = min(ow / (width * sar), oh / height)
    sx, sy = sar * s, s
    ox = (ow - width * sx) / 2.0
    oy = (oh - height * sy) / 2.0
    return np.array([[sx, 0.0, ox + 0.5 * sx - 0.5],
                     [0.0, sy, oy + 0.5 * sy - 0.5]], dtype=np.float64)


def content_rect(A: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    """Display rectangle (x0, y0, x1, y1), exclusive, covered by the frame."""
    u0, v0 = map_points(A, [(-0.5, -0.5)])[0] + 0.5
    u1, v1 = map_points(A, [(width - 0.5, height - 0.5)])[0] + 0.5
    return int(round(u0)), int(round(v0)), int(round(u1)), int(round(v1))


def map_points(A: np.ndarray, pts) -> np.ndarray:
    """Apply a 2x3 affine to points: (N, 2) -> (N, 2) float."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return p @ np.asarray(A, dtype=np.float64)[:, :2].T + np.asarray(A, dtype=np.float64)[:, 2]


def invert_affine(A: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(np.asarray(A, dtype=np.float64))


def display_to_frame(A: np.ndarray, pts) -> np.ndarray:
    """Display coords (e.g. a mouse click) -> frame pixel coords."""
    return map_points(invert_affine(A), pts)


def frame_to_field(y, parity: int):
    """Frame row -> field row of the given parity (inverse of r -> 2r + p)."""
    return (np.asarray(y, dtype=np.float64) - parity) / 2.0


def field_to_frame(r, parity: int):
    return 2.0 * np.asarray(r, dtype=np.float64) + parity


def _warp(img: np.ndarray, M: np.ndarray, out_size, rect) -> np.ndarray:
    out = cv2.warpAffine(img, M.astype(np.float64), (int(out_size[0]), int(out_size[1])),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    x0, y0, x1, y1 = rect
    if y0 > 0:
        out[:y0] = 0
    if y1 < out.shape[0]:
        out[y1:] = 0
    if x0 > 0:
        out[:, :x0] = 0
    if x1 < out.shape[1]:
        out[:, x1:] = 0
    return out


def _to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def frame_to_display(frame: np.ndarray, sar: float, out_size=(960, 720)) -> tuple[np.ndarray, np.ndarray]:
    """Full frame (gray or BGR) -> (bgr display image, A)."""
    h, w = frame.shape[:2]
    A = display_transform(w, h, sar, out_size)
    return _to_bgr(_warp(np.ascontiguousarray(frame), A, out_size, content_rect(A, w, h))), A


def sample_to_display(sample: Sample, sar: float, out_size=(960, 720),
                      is_field: Optional[bool] = None) -> tuple[np.ndarray, np.ndarray]:
    """Parity-aware bob of a field to full height + square-pixel resize.

    Returns (bgr, A) with A the 2x3 affine frame pixel coords -> display
    coords (identical for both fields of a frame).  Uses `sample.bgr` when
    present, else the gray image.

    is_field: whether the sample is a field.  `Sample` does not say, so by
    default an image whose square-pixel aspect (cols * sar / rows) exceeds
    2.4 is taken to be a field (a 4:3 field is 2.67, a 16:9 field 3.56; real
    progressive frames are <= 1.78).  Pass it explicitly when known
    (`SampleSource.interlaced`).
    """
    img = sample.bgr if sample.bgr is not None else sample.image
    rows, cols = img.shape[:2]
    if is_field is None:
        is_field = cols * sar / rows > FIELD_ASPECT_THRESHOLD
    if not is_field:
        return frame_to_display(img, sar, out_size)
    frame_h = 2 * rows
    A = display_transform(cols, frame_h, sar, out_size)
    p = int(sample.parity)
    M = np.array([[A[0, 0], 0.0, A[0, 2]],
                  [0.0, 2.0 * A[1, 1], A[1, 1] * p + A[1, 2]]], dtype=np.float64)
    out = _warp(np.ascontiguousarray(img), M, out_size, content_rect(A, cols, frame_h))
    return _to_bgr(out), A
