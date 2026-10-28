"""Exact motion-blurred ball coverage on a field (or frame) pixel grid.

The exposure is split into short sub-intervals; over each one the ball's
image is an ellipse (semi-axes fx*r/Z, fy*r/Z) moving linearly between the
projected end points.  At every fine sub-sample point we compute the exact
fraction of the sub-interval during which the point lies inside the moving
ellipse (a quadratic solve), so there is no temporal point sampling (no
"beaded" streaks).  An optional Gaussian optical PSF is applied on the fine
grid BEFORE averaging over the pixel aperture (applying it after sampling
leaves ~0.4 px centroid quantization on tiny balls).

The vertical pixel aperture of an interlaced field is 2 frame rows (CCD
field integration), which keeps a 1-px ball visible in both fields.

Verified in the prototype (synth.md section 8): static-ball energy std
<= 1.4 %, centroid error 0.02-0.03 px RMS (PSF sigma 0.7), moving-streak
energy std <= 0.2 %.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import cv2
import numpy as np

# One segment: (u_a, v_a, u_b, v_b, semi_axis_x, semi_axis_y, weight) in FRAME pixel coords;
# weight = fraction of the exposure time covered by the segment.
Segment = tuple


def ball_patch(segs: Sequence[Segment], W: int, Hf: int, row_off: float, row_step: float,
               aperture: float, psf_sigma: float = 0.0) -> Optional[tuple[int, int, np.ndarray]]:
    """Coverage of the moving ball on an image grid.

    The image has `Hf` rows; image row r is frame row  r * row_step + row_off
    (field of parity p: row_step=2, row_off=p; progressive: 1, 0).
    `aperture` (frame rows) must equal `row_step` so the fine grid tiles the
    plane.  Returns (r0, c0, cov) with cov[h, w] = time-averaged coverage
    (0..1) of image pixels [r0:r0+h, c0:c0+w], or None if off-image.
    """
    segs = np.asarray(segs, dtype=float)
    if segs.size == 0:
        return None
    pad = 1.5 + 3 * psf_sigma
    umin = (np.minimum(segs[:, 0], segs[:, 2]) - segs[:, 4]).min() - pad
    umax = (np.maximum(segs[:, 0], segs[:, 2]) + segs[:, 4]).max() + pad
    vmin = (np.minimum(segs[:, 1], segs[:, 3]) - segs[:, 5]).min() - aperture / 2 - pad
    vmax = (np.maximum(segs[:, 1], segs[:, 3]) + segs[:, 5]).max() + aperture / 2 + pad
    if not (np.isfinite(umin) and np.isfinite(umax) and np.isfinite(vmin) and np.isfinite(vmax)):
        return None
    c0 = max(int(math.floor(umin)), 0)
    c1 = min(int(math.ceil(umax)), W - 1)
    r0 = max(int(math.floor((vmin - row_off) / row_step)), 0)
    r1 = min(int(math.ceil((vmax - row_off) / row_step)), Hf - 1)
    if c1 < c0 or r1 < r0:
        return None
    rmin = max(min(segs[:, 4].min(), segs[:, 5].min()), 0.2)
    nh = int(min(24, max(4, math.ceil(5.0 / rmin))))
    nv = int(max(2, round(nh * aperture)))
    oh = (np.arange(nh) + 0.5) / nh - 0.5
    ov = ((np.arange(nv) + 0.5) / nv - 0.5) * aperture
    cols = np.arange(c0, c1 + 1)
    rows = np.arange(r0, r1 + 1)
    U = cols[None, :, None, None] + oh[None, None, None, :]
    V = rows[:, None, None, None] * row_step + row_off + ov[None, None, :, None]
    cov = np.zeros((len(rows), len(cols), nv, nh))
    for ua, va, ub, vb, ax, ay, w in segs:
        wx = (U - ua) / ax
        wy = (V - va) / ay
        dx = (ub - ua) / ax
        dy = (vb - va) / ay
        dd = dx * dx + dy * dy
        ww = wx * wx + wy * wy
        if dd < 1e-12:
            cov += w * (ww <= 1)
            continue
        wd = wx * dx + wy * dy
        disc = wd * wd - dd * (ww - 1)
        sq = np.sqrt(np.maximum(disc, 0))
        s1 = np.clip((wd - sq) / dd, 0, 1)
        s2 = np.clip((wd + sq) / dd, 0, 1)
        cov += w * np.where(disc > 0, s2 - s1, 0)
    if psf_sigma > 0:
        R_, C_ = len(rows), len(cols)
        fine = cov.transpose(0, 2, 1, 3).reshape(R_ * nv, C_ * nh).astype(np.float32)
        fine = cv2.GaussianBlur(fine, (0, 0), sigmaX=psf_sigma * nh, sigmaY=psf_sigma * nv / aperture,
                                borderType=cv2.BORDER_CONSTANT)
        cov = fine.reshape(R_, nv, C_, nh).transpose(0, 2, 1, 3)
    cov = cov.mean(axis=(2, 3))
    return r0, c0, cov
