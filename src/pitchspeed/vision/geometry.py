"""Geometry shared by the vision stages: coordinate maps, corridor, scales.

Coordinate systems (see `pitchspeed.types`):

* *sample grid*: (col, row) pixel indices of a sample image.  For a field
  of parity p, row r is frame row y = 2 r + p; for a progressive frame
  y = r.  `row_scale` is 2 for fields, 1 for frames.
* *frame px*: stored-pixel frame coordinates (x, y) of one sample.
* *reference px*: frame px of the calibration reference frame.  Every
  candidate / track position is expressed here ("stabilized").
* *square px*: (x * sar, y), for distances and angles.

Transforms map sample frame px -> reference px.  Internally they are 3x3
matrices (a similarity, or a homography when the camera rotated enough for
perspective to matter); `apply` / `invert` accept 2x3 affine or 3x3
projective matrices.  `local_affine` gives the 2x3 first-order
approximation used for the FinderResult.transforms contract.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from pitchspeed import constants as C
from pitchspeed.calib.model import Calibration

MPH = C.MPH_TO_FTPS


# ------------------------------------------------------------ transforms ----
def to3(A: np.ndarray) -> np.ndarray:
    """2x3 affine (or 3x3) -> 3x3 homogeneous (copy)."""
    A = np.asarray(A, dtype=np.float64)
    if A.shape == (3, 3):
        return A.copy()
    M = np.eye(3)
    M[:2] = A[:2]
    return M


def is_affine(A: np.ndarray, tol: float = 1e-12) -> bool:
    A = np.asarray(A)
    return A.shape == (2, 3) or (abs(A[2, 0]) < tol and abs(A[2, 1]) < tol and abs(A[2, 2] - 1.0) < tol)


def apply(A: np.ndarray, pts) -> np.ndarray:
    """Apply a 2x3 affine or 3x3 projective transform to (N, 2) points."""
    P = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    A = np.asarray(A, dtype=np.float64)
    if A.shape == (2, 3) or is_affine(A):
        return P @ A[:2, :2].T + A[:2, 2]
    q = P @ A[:, :2].T + A[:, 2]
    return q[:, :2] / q[:, 2:3]


def invert(A: np.ndarray) -> np.ndarray:
    """Inverse transform, same shape as the input (2x3 -> 2x3, 3x3 -> 3x3)."""
    A = np.asarray(A, dtype=np.float64)
    Ai = np.linalg.inv(to3(A))
    if A.shape == (2, 3):
        return Ai[:2]
    return Ai / Ai[2, 2]


def local_affine(A: np.ndarray, at) -> np.ndarray:
    """2x3 first-order approximation of transform A around point `at` (source coords)."""
    A = np.asarray(A, dtype=np.float64)
    if A.shape == (2, 3) or is_affine(A):
        return to3(A)[:2]
    c = np.asarray(at, dtype=np.float64).reshape(2)
    h = 1.0
    f0 = apply(A, c[None])[0]
    jx = (apply(A, (c + [h, 0])[None])[0] - apply(A, (c - [h, 0])[None])[0]) / (2 * h)
    jy = (apply(A, (c + [0, h])[None])[0] - apply(A, (c - [0, h])[None])[0]) / (2 * h)
    J = np.column_stack([jx, jy])
    out = np.zeros((2, 3))
    out[:, :2] = J
    out[:, 2] = f0 - J @ c
    return out


def grid_to_frame(row_scale: int, parity: int) -> np.ndarray:
    """3x3: sample grid (col, row) -> sample frame px (x, y = row_scale*row + parity)."""
    return np.array([[1.0, 0, 0], [0, row_scale, parity], [0, 0, 1]])


def frame_to_grid(row_scale: int, parity: int) -> np.ndarray:
    """3x3: sample frame px -> sample grid."""
    return np.array([[1.0, 0, 0], [0, 1.0 / row_scale, -parity / row_scale], [0, 0, 1]])


def translate(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0, dx], [0, 1.0, dy], [0, 0, 1]])


def sq_matrix(sar: float) -> np.ndarray:
    """3x3: frame px -> square px."""
    return np.diag([sar, 1.0, 1.0])


def similarity_params(A: np.ndarray, sar: float = 1.0) -> dict:
    """Decompose a frame-px similarity into (tx, ty, angle_deg, scale) in square px."""
    S = sq_matrix(sar) @ to3(A) @ np.linalg.inv(sq_matrix(sar))
    a, b = S[0, 0], S[1, 0]
    return {"tx": float(S[0, 2]), "ty": float(S[1, 2]),
            "angle_deg": math.degrees(math.atan2(b, a)), "scale": float(math.hypot(a, b))}


def similarity_from_params(tx: float, ty: float, angle_deg: float, scale: float, sar: float = 1.0) -> np.ndarray:
    """Inverse of `similarity_params` (returns 2x3 in frame px)."""
    c = scale * math.cos(math.radians(angle_deg))
    s = scale * math.sin(math.radians(angle_deg))
    S = np.array([[c, -s, tx], [s, c, ty], [0, 0, 1.0]])
    return (np.linalg.inv(sq_matrix(sar)) @ S @ sq_matrix(sar))[:2]


def half_transform(A: np.ndarray, sar: float = 1.0) -> np.ndarray:
    """The similarity 'half way' between identity and A (for small motions)."""
    p = similarity_params(A, sar)
    # half rotation/scale about the same fixed point: compose R^(1/2) and solve for t
    ang, sc = p["angle_deg"] / 2.0, math.sqrt(max(p["scale"], 1e-9))
    H = similarity_from_params(0.0, 0.0, ang, sc, sar)
    # H(H(x)) + t2 must equal A(x): choose t_h with H(t_h) + t_h = t_A  -> (H_lin + I) t_h = t_A
    L = to3(H)[:2, :2]
    t_h = np.linalg.solve(L + np.eye(2), np.asarray(A)[:2, 2])
    H[:, 2] = t_h
    return H


# --------------------------------------------------------------- geometry ----
@dataclass
class ViewGeometry:
    """Per-segment geometry derived from a Calibration (all in reference px)."""
    view: str
    sar: float
    sample_dt: float
    release: np.ndarray                  # (2,) reference px
    catch: np.ndarray
    corridor: np.ndarray                 # (N, 2) reference px polygon
    bbox: tuple                          # (x0, y0, x1, y1) of the corridor, reference px
    axis_sq: np.ndarray                  # unit vector release -> catch (square px)
    length_sq: float                     # |catch - release| (square px)
    scale_rel: float                     # square px per ft near release (estimate)
    scale_catch: float                   # square px per ft near catch (estimate)
    scale_known: bool                    # False when the behind-view scale is a guess
    d_rel: float                         # ball diameter (px) near release
    d_catch: float                       # ball diameter (px) near catch
    v_min_px: float                      # min plausible image speed (square px / sample), side view
    v_max_px: float                      # max plausible image speed (square px / sample)
    distance_ft: float
    t_window: tuple                      # plausible flight-time window (s)
    length_ft: float = 0.0               # release -> catch distance along the pitch axis (ft)
    extra: dict = field(default_factory=dict)

    # -------- helpers
    def to_sq(self, xy) -> np.ndarray:
        q = np.atleast_2d(np.asarray(xy, dtype=np.float64)).copy()
        q[:, 0] *= self.sar
        return q

    def progress(self, xy) -> np.ndarray:
        """0 at release, 1 at catch (projection in square px)."""
        q = self.to_sq(xy) - self.to_sq(self.release)[0]
        return q @ self.axis_sq / self.length_sq

    def lateral(self, xy) -> np.ndarray:
        """Signed distance (square px) from the release->catch line."""
        q = self.to_sq(xy) - self.to_sq(self.release)[0]
        n = np.array([-self.axis_sq[1], self.axis_sq[0]])
        return q @ n

    def ball_diameter(self, prog) -> np.ndarray:
        """Expected ball diameter (px) at progress `prog` (linear in 1/size for behind view)."""
        p = np.clip(np.asarray(prog, dtype=np.float64), 0.0, 1.2)
        if self.view == "side":
            return np.full_like(p, self.d_rel)
        # apparent size ~ 1/depth; depth roughly linear in progress
        inv = (1 - p) / self.d_rel + p / self.d_catch
        return 1.0 / np.maximum(inv, 1e-6)

    def scale_at(self, prog) -> np.ndarray:
        p = np.clip(np.asarray(prog, dtype=np.float64), 0.0, 1.2)
        if self.view == "side":
            return np.full_like(p, self.scale_rel)
        inv = (1 - p) / self.scale_rel + p / self.scale_catch
        return 1.0 / np.maximum(inv, 1e-9)

    def in_corridor(self, xy, margin: float = 0.0) -> np.ndarray:
        """Point-in-corridor test (reference px), with an outward margin (px)."""
        P = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        key = round(float(margin), 2)
        cache = self.extra.setdefault("_corr_raster", {})
        if key not in cache:
            x0, y0, x1, y1 = self.bbox
            pad = int(math.ceil(margin)) + 2
            ox, oy = int(math.floor(x0)) - pad, int(math.floor(y0)) - pad
            w = int(math.ceil(x1)) - ox + pad + 1
            h = int(math.ceil(y1)) - oy + pad + 1
            m = np.zeros((h, w), np.uint8)
            poly = np.round((self.corridor - [ox, oy]) * 16).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(m, [poly], 255, lineType=cv2.LINE_8, shift=4)
            if margin > 0:
                r = int(math.ceil(margin))
                m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
            cache[key] = (ox, oy, m)
        ox, oy, m = cache[key]
        j = np.round(P[:, 0] - ox).astype(np.int64)
        i = np.round(P[:, 1] - oy).astype(np.int64)
        ok = (i >= 0) & (i < m.shape[0]) & (j >= 0) & (j < m.shape[1])
        out = np.zeros(len(P), dtype=bool)
        out[ok] = m[i[ok], j[ok]] > 0
        return out

    def speed_mph_from_px(self, v_sq_per_sample: float, prog: float = 0.5) -> float:
        """Rough ball speed from image speed along the axis (side view)."""
        s = float(self.scale_at(prog))
        return v_sq_per_sample / s / self.sample_dt / MPH


BEHIND_RELEASE_SPREAD_FT = 2.5     # lateral reach of the corridor at the release end (behind view)
BEHIND_BACK_FRAC = 0.22            # corridor extent behind the release point, fraction of |Pc - Prel|


def behind_corridor(calib: Calibration, sr: float, sc: float) -> np.ndarray:
    """Behind-view search corridor (reference px polygon).

    `Calibration.corridor_polygon()` is a narrow trapezoid around the
    straight release->catch line (half width max(1.5 ft, 6 % L) at the
    release end, starting 8 % L behind the release click).  On the synthetic
    behind-view suites 15 of 84 pitches spent 20-50 % of their flight outside
    it, for two geometric reasons:

    * the release point varies from pitch to pitch (+-2 ft laterally and
      +-1 ft in height around the clicked one: arm slot, stride, both
      pitchers of a game share one calibration);
    * early in flight the image motion is mostly perspective expansion away
      from the focus of expansion, which for a camera lower than the
      release point is *upward*, i.e. backwards along the release->catch
      line (down to progress -0.19 measured).

    This corridor reaches `BEHIND_RELEASE_SPREAD_FT` (scaled by the
    release-end px/ft) to each side of the release point and
    `BEHIND_BACK_FRAC` * L behind it, and keeps the catch end of the
    calibration's polygon.  The ball (and so the relevant clutter) stays the
    same; only where we look widens.
    """
    Rp = calib.to_sq(calib.release_point())
    Cp = calib.to_sq(calib.catch_point())
    e = Cp - Rp
    L = float(np.linalg.norm(e))
    e /= L
    n = np.array([-e[1], e[0]])
    w_r = max(BEHIND_RELEASE_SPREAD_FT * sr, 0.06 * L, 10.0)
    w_c = max(3.0 * sc, 0.35 * L, 20.0)
    back = max(BEHIND_BACK_FRAC * L, 1.5 * sr)
    a0 = Rp - back * e
    r0 = Rp                      # full width from the release point on (the region behind it is a cap)
    a1 = Cp + 0.15 * L * e
    w_back = 0.75 * w_r
    poly = [a0 + w_back * n, r0 + w_r * n, a1 + w_c * n, a1 - w_c * n, r0 - w_r * n, a0 - w_back * n]
    return np.array([calib.from_sq(p) for p in poly])


def view_geometry(calib: Calibration, sample_dt: float) -> ViewGeometry:
    """Build the per-segment geometry (scales, ball size, gates) from a calibration."""
    view = calib.view
    sar = calib.sar
    rel = np.asarray(calib.release_point(), dtype=np.float64)
    cat = np.asarray(calib.catch_point(), dtype=np.float64)
    corr = np.asarray(calib.corridor_polygon(), dtype=np.float64)
    x0, y0 = corr.min(axis=0)
    x1, y1 = corr.max(axis=0)
    e = calib.to_sq(cat) - calib.to_sq(rel)
    L = float(np.linalg.norm(e))
    axis = e / L
    D = calib.distance_ft
    diam_ft = calib.ball_diameter_ft()
    extra: dict = {}
    if view == "side":
        k = calib.px_per_ft_axis()
        s_rel = s_c = float(k)
        known = True
    else:
        sr = calib.px_per_ft_at_release()
        sc = calib.px_per_ft_at_catch()
        known = sr is not None and sc is not None
        # Without rubber/plate width clicks the scale is a guess: assume the
        # image release->catch distance spans a typical behind-view geometry
        # (ball ~ 1.2 % of L at release, ~ 4 % at catch).  Only used for
        # kernel sizes and gates, never for speeds.
        if sr is None:
            sr = 0.012 * L / diam_ft
            extra["scale_rel_guess"] = True
        if sc is None:
            sc = max(0.04 * L / diam_ft, 1.5 * sr)
            extra["scale_catch_guess"] = True
        # the glove is behind the plate: slightly closer to the camera
        s_rel, s_c = float(sr), float(sc)
        corr = behind_corridor(calib, s_rel, s_c)
        x0, y0 = corr.min(axis=0)
        x1, y1 = corr.max(axis=0)
    d_rel = max(1.2, diam_ft * s_rel)
    d_c = max(1.2, diam_ft * s_c)
    dt = sample_dt
    v_lo = 30.0 * MPH * s_rel * dt
    v_hi = 105.0 * MPH * max(s_rel, s_c) * dt
    t_lo = max(0.15, (D - 7.0) / (110.0 * MPH))
    t_hi = (D - 3.0) / (30.0 * MPH)
    return ViewGeometry(view=view, sar=sar, sample_dt=dt, release=rel, catch=cat, corridor=corr,
                        bbox=(float(x0), float(y0), float(x1), float(y1)), axis_sq=axis, length_sq=L,
                        scale_rel=s_rel, scale_catch=s_c, scale_known=known, d_rel=float(d_rel),
                        d_catch=float(d_c), v_min_px=float(v_lo), v_max_px=float(v_hi), distance_ft=D,
                        t_window=(float(t_lo), float(t_hi)),
                        length_ft=float(calib.release_y_ft - calib.catch_y_ft), extra=extra)


def polygon_mask(poly: np.ndarray, shape: tuple, row_scale: int = 1, parity: float = 0.0,
                 dilate_px: int = 0) -> np.ndarray:
    """Rasterize a reference-px polygon into a (rows, cols) uint8 mask of a grid.

    Grid row r corresponds to y = row_scale * r + parity.
    """
    P = np.asarray(poly, dtype=np.float64).copy()
    P[:, 1] = (P[:, 1] - parity) / row_scale
    m = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(m, [np.round(P * 16).astype(np.int32).reshape(-1, 1, 2)], 255, lineType=cv2.LINE_8, shift=4)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * max(1, dilate_px // row_scale) + 1))
        m = cv2.dilate(m, k)
    return m
