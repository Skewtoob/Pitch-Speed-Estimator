"""Out-of-focus backstop chain-link fence between the camera and the field.

The fence is a vertical plane `dist_ft` in front of the camera (normal =
horizontal direction of the optical axis) carrying two families of straight
wires (a diamond mesh: `mesh_in` between parallel wires, `wire_in` thick,
at +/-`angle_deg` from horizontal).  Its image is an occupancy (alpha) map:
the box-filtered wire coverage per canvas pixel, convolved with the thin-lens
defocus disk

    blur diameter [px] = fx * A * (1/d - 1/s),   A = f / N  (aperture),

f = fx * pixel pitch (from the sensor width), d = fence distance, s = focus
distance.  The renderer composites  out = in * (1 - a) + wire * a  after the
actors and the ball (the fence is the nearest object).  Under the modelled
camera motion (rotation + zoom) the fence image moves exactly like the
background, so it lives in a 4th channel of the background canvas.

Limitations: compositing happens in gamma-encoded space (not linear light);
no fence posts / rails, no auto-focus hunting onto the fence, no parallax.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

MM_PER_FT = 304.8


def defocus_blur_px(fx_px: float, width_px: int, sensor_width_mm: float, f_number: float,
                    dist_ft: float, focus_ft: float) -> float:
    """Defocus blur-disk diameter (output px) of an object at dist_ft when focused at focus_ft."""
    pitch_mm = sensor_width_mm / float(width_px)
    f_mm = fx_px * pitch_mm
    aperture_mm = f_mm / max(f_number, 0.5)
    return float(fx_px * aperture_mm / MM_PER_FT * abs(1.0 / dist_ft - 1.0 / max(focus_ft, dist_ft + 1e-6)))


def _strip_coverage(d: np.ndarray, width: float, footprint: np.ndarray) -> np.ndarray:
    """1-D box-filter coverage of a strip [-w/2, w/2] by pixels of size `footprint` centred at d."""
    lo = np.maximum(d - footprint / 2, -width / 2)
    hi = np.minimum(d + footprint / 2, width / 2)
    return np.clip((hi - lo) / footprint, 0.0, 1.0)


def disk_kernel(diameter_px: float) -> np.ndarray:
    """Normalized anti-aliased disk (pillbox) kernel."""
    r = max(diameter_px / 2.0, 0.5)
    n = int(math.ceil(r)) + 1
    ss = 4
    g = (np.arange(-n, n + 1)[:, None] + ((np.arange(ss) + 0.5) / ss - 0.5)[None, :]).ravel()
    X, Y = np.meshgrid(g, g)
    inside = (X * X + Y * Y <= r * r).astype(np.float32)
    k = inside.reshape(2 * n + 1, ss, 2 * n + 1, ss).mean(axis=(1, 3))
    return (k / k.sum()).astype(np.float32)


def fence_alpha(rays: np.ndarray, C: np.ndarray, R0: np.ndarray, fx_canvas: float, fcfg: dict,
                blur_canvas_px: float) -> np.ndarray:
    """Alpha (0..1, float32) of the fence for canvas rays.

    rays: (H, W, 3) world ray directions of canvas pixels (not normalized).
    fx_canvas: focal length in canvas pixels (fx * supersampling).
    """
    axis = np.array([R0[2, 0], R0[2, 1], 0.0])
    nrm = axis / max(np.linalg.norm(axis), 1e-9)
    e1 = np.array([-nrm[1], nrm[0], 0.0])                       # horizontal in-plane axis
    denom = rays @ nrm
    t = np.where(denom > 1e-6, float(fcfg["dist_ft"]) / np.where(denom > 1e-6, denom, 1.0), np.inf)
    hit = np.isfinite(t)
    X = C[None, None, :] + np.where(hit, t, 0.0)[..., None] * rays
    a = (X - C) @ e1
    b = X[..., 2]
    rng = np.where(hit, t * np.linalg.norm(rays, axis=-1), 1.0)
    footprint = rng / fx_canvas                                  # ft per canvas px at the fence
    mesh = float(fcfg["mesh_in"]) / 12.0
    wire = float(fcfg["wire_in"]) / 12.0
    th = math.radians(float(fcfg.get("angle_deg", 45.0)))
    cov = np.zeros(a.shape, np.float64)
    for sgn in (1.0, -1.0):
        s = -sgn * a * math.sin(th) + b * math.cos(th)          # coordinate across one wire family
        d = s - mesh * np.round(s / mesh)
        c = _strip_coverage(d, wire, footprint)
        cov = cov + c - cov * c
    cov = np.where(hit, cov, 0.0).astype(np.float32)
    if blur_canvas_px > 1.0:
        cov = cv2.filter2D(cov, -1, disk_kernel(blur_canvas_px), borderType=cv2.BORDER_REFLECT)
    return np.clip(cov * float(fcfg.get("opacity", 1.0)), 0.0, 1.0).astype(np.float32)
