"""E2: side-view release speed from a fit to the ball's track (math.md §3).

Pipeline for one track:

1. *Plumb projection.*  Each stabilized ball position b (square px) is moved
   along the image vertical (calib.up_dir_sq()) onto the image line P->R
   (P = plate click, R = rubber click); mu = fraction along P->R.  This
   removes the ball's height above the pitch line.
2. *1-D projective map.*  World fraction lam (0 at the plate click, 1 at the
   rubber click) -> mu = lam / (rho (1 - lam) + lam), rho = Z_plate / Z_rubber
   fitted from the ball's own motion (rho = 1 is the affine case).
3. *Motion model.*  Y(t) = y_r - ln(1 + k v0 (t - t_r)) / k, y_r = D - ext.
4. *Fit* theta = (v0, t_r, ln rho) with least_squares(soft_l1, f_scale 2 px),
   multi-start from a Theil-Sen line, one outlier-rejection pass.
5. *Uncertainty* sigma^2 = cov + click perturbations (each click coordinate
   moved by 2 px, re-fit) + vertical-direction perturbation + sideways-motion
   term + extension term + model term (0.5 %); the drag-k perturbation is
   reported separately as the part shared with E1.
6. *Checks*: obliquity (E2 not used when |phi| > 45 deg), partial track,
   field order (fit under both orders; residual alternation), poor fit.

Sideways motion (the dominant error; `lateral_budget`).  The obliquity phi
(angle between the pitch line and the image plane) and the camera depth
Z_mid need the focal length f, which a side view does not give.  With the
principal point at the image centre, tan(phi) = f (1 - rho) / (u_R - rho u_P)
(u = position along the line axis relative to the image centre).  The
budget is averaged over a log-normal prior on f, reweighted by two cues:
the ball's apparent height (verticals are not foreshortened, the line is,
so observed/expected height ~ 1/cos(phi)) and, if clicked, the rubber ends.
This refines math.md §3's single (|tan phi|/L + 0.4/Z) * sigma_dx term,
which under-covered in simulation (see tests/test_estimate_validation.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import theilslopes

from pitchspeed import constants as C
from pitchspeed.calib.model import Calibration
from pitchspeed.physics import DragModel, time_to_travel
from pitchspeed.types import EventTime, PitchTrack, ScanInfo, SpeedComponent

from pitchspeed.estimate import common as K

# ------------------------------------------------------------- parameters
SIGMA_CLICK_PX = 2.0          # 1-sigma of a calibration click (stored px), math.md §3
SIGMA_UP_DEG = 1.5            # ASSUMED camera-roll sigma when no vertical reference was clicked
F_SCALE_PX = 2.0              # soft_l1 scale
OUTLIER_K = 3.5               # outlier threshold in robust sigmas ...
OUTLIER_MIN_PX = 3.0          # ... but never below this many px
RHO_BOUNDS = (0.05, 20.0)
V_BOUNDS_FTPS = (20.0, 250.0)
# Sideways motion (refines math.md §3, see `lateral_budget`): with x_r, x_t
# the ball's lateral offsets (ft, toward the camera) at release and at the
# plate, the relative speed error is  e = a x_r + b x_t,
#   a = tan(phi)/L + c/Z_mid,  b = -tan(phi)/L + c/Z_mid,  c ~ 0.5
# (obliquity acting on the sideways travel + magnification of the mean
# offset; c = 0.5 from theory, 0.44-0.55 fitted in simulation).
LATERAL_PERSPECTIVE_COEF = 0.5
RELEASE_SIDE_FT = 1.8         # |release x| of a typical pitcher (ASSUMED; MLB ~1.5-2.5 ft)
SIGMA_XR_UNKNOWN_FT = 1.8     # release side sigma when handedness / camera side unknown
SIGMA_XR_KNOWN_FT = 0.5       # ... when known (the mean is then corrected)
SIGMA_XT_FT = 0.8             # lateral spread of the ball at the plate / glove
# Plumb lines are image columns (or one clicked vertical), but a camera tilted
# down by theta sees verticals converge: speed error ~ +h_ball tan(theta) / Z_mid.
TILT_PRIOR_MEAN_DEG = 3.0     # ASSUMED: side views are usually filmed from slightly above
TILT_PRIOR_SIGMA_DEG = 4.0
# Focal-length prior (multiple of the image width in square px).  Camcorders
# of the era: widest HFOV ~45-55 deg -> f ~ 0.95-1.2 W; side views are often
# zoomed in somewhat.  ASSUMED, not measured.
F_PRIOR_MEDIAN_W = 1.3
F_PRIOR_LOG_SIGMA = 0.35
F_MIN_W = 0.95                # widest plausible lens (HFOV ~55 deg): bounds phi from below
MAX_OBLIQUITY_DEG = 45.0
POOR_FIT_RMS_PX = 4.0
PARTIAL_COVERAGE = 0.6        # observed fraction of the release->catch path below this -> partial_track
ALT_ORDER_RATIO = 0.6         # swapped-field-order rms below this fraction of the scan's -> swap
ALTERNATION_FLAG = -0.6       # lag-1 residual autocorrelation below this (with rms > 1.5 px) -> flag


# ------------------------------------------------------------- geometry
@dataclass
class AxisGeometry:
    """Plate->rubber image line and the vertical direction (square px)."""
    P: np.ndarray
    R: np.ndarray
    up: np.ndarray
    y_P: float                    # world y of the plate click (ft)
    y_R: float                    # world y of the rubber click (ft)

    @property
    def length_px(self) -> float:
        return float(np.linalg.norm(self.R - self.P))

    @property
    def dy(self) -> float:
        return self.y_R - self.y_P

    def mu(self, xy_sq: np.ndarray) -> np.ndarray:
        """Plumb projection: fraction along P->R of each point (N, 2)."""
        d = self.R - self.P
        u = self.up
        b = np.atleast_2d(xy_sq) - self.P
        den = d[0] * u[1] - d[1] * u[0]
        if abs(den) < 1e-9:
            raise K.NotMeasurable("the vertical direction is parallel to the plate->rubber line")
        return (b[:, 0] * u[1] - b[:, 1] * u[0]) / den

    def lam_of_y(self, y):
        return (np.asarray(y, dtype=float) - self.y_P) / self.dy


def axis_geometry(calib: Calibration) -> AxisGeometry:
    if not (calib.has("plate") and calib.has("rubber")):
        raise K.NotMeasurable("side-view fit needs the plate and rubber clicks")
    P, R = calib.axis_sq()
    up = np.asarray(calib.up_dir_sq(), dtype=float)
    if np.linalg.norm(R - P) < 20:
        raise K.NotMeasurable("plate and rubber clicks are too close together")
    return AxisGeometry(P=np.asarray(P, float), R=np.asarray(R, float), up=up / np.linalg.norm(up),
                        y_P=calib.plate_center_y_ft, y_R=calib.rubber_center_y_ft)


def _rotate(v: np.ndarray, ang: float) -> np.ndarray:
    c, s = math.cos(ang), math.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


# ------------------------------------------------------------ motion model
def model_y(t: np.ndarray, v0: float, t_r: float, y_r: float, k: float) -> np.ndarray:
    """Along-axis world position Y(t) (ft) under quadratic drag."""
    tau = np.asarray(t, dtype=float) - t_r
    if k <= 0:
        return y_r - v0 * tau
    q = np.maximum(1.0 + k * v0 * tau, 0.05)
    return y_r - np.log(q) / k


def model_mu(t, theta, y_r: float, k: float, geom: AxisGeometry) -> np.ndarray:
    v0, t_r, ln_rho = theta
    lam = geom.lam_of_y(model_y(t, v0, t_r, y_r, k))
    rho = math.exp(ln_rho)
    return lam / np.maximum(rho * (1.0 - lam) + lam, 1e-9)


def _model_and_jac(t, theta, y_r, k, geom):
    """mu(t) and d mu / d(v0, t_r, ln rho)."""
    v0, t_r, ln_rho = theta
    rho = math.exp(ln_rho)
    tau = np.asarray(t, dtype=float) - t_r
    if k > 0:
        q_raw = 1.0 + k * v0 * tau
        q = np.maximum(q_raw, 0.05)
        Y = y_r - np.log(q) / k
        live = q_raw > 0.05
        dY_dv = np.where(live, -tau / q, 0.0)
        dY_dtr = np.where(live, v0 / q, 0.0)
    else:
        Y = y_r - v0 * tau
        dY_dv = -tau
        dY_dtr = np.full_like(tau, v0)
    lam = (Y - geom.y_P) / geom.dy
    den = np.maximum(rho * (1.0 - lam) + lam, 1e-9)
    mu = lam / den
    dmu_dlam = rho / den ** 2
    dmu_dlnrho = -rho * lam * (1.0 - lam) / den ** 2
    J = np.stack([dmu_dlam * dY_dv / geom.dy, dmu_dlam * dY_dtr / geom.dy, dmu_dlnrho], axis=1)
    return mu, J


@dataclass
class FitResult:
    theta: np.ndarray             # (v0 ft/s, t_r s (relative to t0), ln rho)
    cov: np.ndarray               # 3x3 covariance of theta
    rms_px: float
    resid_px: np.ndarray          # residuals of the used points (px)
    used: np.ndarray              # bool mask over the input points
    cost: float
    n_used: int
    at_bound: bool

    @property
    def v0(self) -> float:
        return float(self.theta[0])

    @property
    def rho(self) -> float:
        return float(math.exp(self.theta[2]))


def _robust_init(t: np.ndarray, mu: np.ndarray, geom: AxisGeometry, y_r: float) -> tuple[float, float]:
    """(v0, t_r) from a Theil-Sen line through the affine positions."""
    Y = geom.y_P + mu * geom.dy
    if len(t) >= 3:
        slope, icpt, _, _ = theilslopes(Y, t)
    else:
        slope, icpt = np.polyfit(t, Y, 1)
    v = float(np.clip(-slope, V_BOUNDS_FTPS[0] + 1, V_BOUNDS_FTPS[1] - 1))
    if slope < 0:
        t_r = float((y_r - icpt) / slope)
    else:
        t_r = float(t[0] - (y_r - Y[0]) / v)
    return v, t_r


def _solve(t, mu, wt, geom, y_r, k, x0, lb, ub, loss="soft_l1"):
    L = geom.length_px

    def fun(x):
        return (model_mu(t, x, y_r, k, geom) - mu) * L * wt

    def jac(x):
        _, J = _model_and_jac(t, x, y_r, k, geom)
        return J * (L * wt)[:, None]

    x0 = np.clip(np.asarray(x0, float), lb + 1e-9, ub - 1e-9)
    return least_squares(fun, x0, jac=jac, bounds=(lb, ub), loss=loss, f_scale=F_SCALE_PX,
                         x_scale=np.array([20.0, 0.02, 0.2]), max_nfev=200)


def fit_track(t: np.ndarray, mu: np.ndarray, weights: np.ndarray, geom: AxisGeometry, y_r: float,
              k: float, x0: Optional[np.ndarray] = None, used: Optional[np.ndarray] = None,
              reject_outliers: bool = True) -> FitResult:
    """Robust fit of (v0, t_r, ln rho) to plumb-projected positions.

    `t` should be relative to a reference time near the track (conditioning).
    With `x0` (warm start) a single start is used and, with `used`, the
    point set is kept fixed (perturbation re-fits).
    """
    t = np.asarray(t, float)
    mu = np.asarray(mu, float)
    w = np.asarray(weights, float)
    n = len(t)
    lb = np.array([V_BOUNDS_FTPS[0], t.min() - 1.0, math.log(RHO_BOUNDS[0])])
    ub = np.array([V_BOUNDS_FTPS[1], t.min() + 0.15, math.log(RHO_BOUNDS[1])])
    mask = np.ones(n, bool) if used is None else np.asarray(used, bool).copy()
    if x0 is None:
        v_i, tr_i = _robust_init(t[mask], mu[mask], geom, y_r)
        starts = [np.array([v_i, tr_i, lr]) for lr in (0.0, math.log(0.7), math.log(1.45))]
    else:
        starts = [np.asarray(x0, float)]
    best = None
    for x_start in starts:
        sol = _solve(t[mask], mu[mask], w[mask], geom, y_r, k, x_start, lb, ub)
        if best is None or sol.cost < best.cost:
            best = sol
    sol = best
    if reject_outliers and used is None:
        r_all = (model_mu(t, sol.x, y_r, k, geom) - mu) * geom.length_px
        scale = 1.4826 * float(np.median(np.abs(r_all[mask] - np.median(r_all[mask])))) + 1e-9
        thr = max(OUTLIER_K * scale, OUTLIER_MIN_PX)
        new_mask = np.abs(r_all) <= thr
        if new_mask.sum() >= K.MIN_SIDE_POINTS and not np.array_equal(new_mask, mask):
            mask = new_mask
            sol = _solve(t[mask], mu[mask], w[mask], geom, y_r, k, sol.x, lb, ub)
    x = sol.x
    mu_m, J = _model_and_jac(t[mask], x, y_r, k, geom)
    r = (mu_m - mu[mask]) * geom.length_px
    Jw = J * (geom.length_px * w[mask])[:, None]
    rw = r * w[mask]
    dof = max(int(mask.sum()) - 3, 1)
    s2 = max(float(np.sum(rw ** 2)) / dof, 0.3 ** 2)
    try:
        cov = np.linalg.pinv(Jw.T @ Jw) * s2
    except np.linalg.LinAlgError:
        cov = np.full((3, 3), np.nan)
    at_bound = bool(np.any(np.isclose(x, lb, rtol=0, atol=1e-6)) or np.any(np.isclose(x, ub, rtol=0, atol=1e-6)))
    return FitResult(theta=x, cov=cov, rms_px=float(np.sqrt(np.mean(r ** 2))), resid_px=r, used=mask,
                     cost=float(sol.cost), n_used=int(mask.sum()), at_bound=at_bound)


def residual_alternation(t: np.ndarray, r: np.ndarray, sample_dt: float) -> float:
    """Lag-1 autocorrelation of residuals of consecutive samples (NaN if < 6 pairs)."""
    t = np.asarray(t, float)
    r = np.asarray(r, float)
    if len(t) < 7:
        return float("nan")
    dt = np.diff(t)
    pair = np.abs(dt - sample_dt) < 0.25 * sample_dt
    if pair.sum() < 6:
        return float("nan")
    den = float(np.mean(r ** 2))
    if den <= 0:
        return float("nan")
    return float(np.mean(r[:-1][pair] * r[1:][pair]) / den)


# ------------------------------------------------------------- obliquity
def _line_coords(geom: AxisGeometry, centre: np.ndarray) -> tuple[float, float]:
    """(u_P, u_R): positions of P and R along the image line axis relative to the centre."""
    e = (geom.R - geom.P) / geom.length_px
    return float((geom.P - centre) @ e), float((geom.R - centre) @ e)


def tan_phi_and_depth(geom: AxisGeometry, rho: float, f: float, centre: np.ndarray) -> tuple[float, float]:
    """(signed tan phi, camera depth of the line middle in ft) for focal f (square px).

    Pinhole camera with the principal point at `centre`: depths along the
    line are linear in lam, Z_P = rho Z_R, which gives
    tan phi = f (1 - rho) / (u_R - rho u_P) and Z_R = f dy cos(phi) / (u_R - rho u_P).
    tan phi > 0 when the rubber end is farther away (rho < 1).
    """
    u_P, u_R = _line_coords(geom, centre)
    den = u_R - rho * u_P
    if den <= 1e-6:
        return float("inf"), float("nan")
    tan_phi = f * (1.0 - rho) / den
    cos_phi = 1.0 / math.sqrt(1.0 + tan_phi ** 2)
    z_R = f * geom.dy * cos_phi / den
    return tan_phi, z_R * (1.0 + rho) / 2.0


@dataclass
class LateralBudget:
    """Sideways-travel error of E2, marginalized over the unknown focal length."""
    rel_bias: float              # expected relative error of the apparent speed (0 if handedness unknown)
    rel_sigma: float             # 1-sigma relative error after removing rel_bias
    tan_phi: float               # posterior-mean signed tan phi
    sigma_tan_phi: float
    z_mid_ft: float              # posterior-mean depth of the line middle
    phi_min_deg: float           # obliquity for the widest plausible lens (lower bound)
    source: str                  # 'focal_prior' | 'focal_prior+rubber_ends'

    @property
    def phi_deg(self) -> float:
        return math.degrees(math.atan(abs(self.tan_phi))) if math.isfinite(self.tan_phi) else 90.0


def lateral_budget(calib: Calibration, geom: AxisGeometry, rho: float, sigma_ln_rho: float,
                   L: float, xr_mean: float, xr_sigma: float, h_ft: float,
                   height_ratio: Optional[tuple[float, float]] = None) -> LateralBudget:
    """Sideways-motion and plumb-tilt error of E2 (see the constants above).

    For a focal length f: tan_phi(f), Z_mid(f) from the fitted rho, and
        e = a x_r + b x_t + h tan(theta) / Z_mid
    with x_r ~ N(xr_mean, xr_sigma), x_t ~ N(0, SIGMA_XT_FT), theta from the
    tilt prior.  Mean and variance of e are averaged over a log-normal prior
    on f (grid quadrature), reweighted by the likelihood of the observed
    apparent-height ratio (`height_ratio` = (kappa, rel sigma), kappa ~
    1/cos(phi), see `apparent_height_ratio`) and of the rubber-end clicks if
    present.  The mean (rel_bias) is removed from the estimate by the caller.
    """
    w_sq = float(calib.image_size[0]) * calib.sar
    centre = np.array([w_sq / 2.0, float(calib.image_size[1]) / 2.0])
    s_ln = sigma_ln_rho if math.isfinite(sigma_ln_rho) else 0.5
    z = np.linspace(-3.0, 3.0, 25)
    wts = np.exp(-0.5 * z ** 2)
    fs = F_PRIOR_MEDIAN_W * w_sq * np.exp(F_PRIOR_LOG_SIGMA * z)
    tans, zmids, s_rho = [], [], []
    for f in fs:
        t_f, z_f = tan_phi_and_depth(geom, rho, f, centre)
        t_hi, _ = tan_phi_and_depth(geom, rho * math.exp(s_ln), f, centre)
        t_lo, _ = tan_phi_and_depth(geom, rho * math.exp(-s_ln), f, centre)
        tans.append(t_f)
        zmids.append(z_f)
        s_rho.append(0.5 * abs(t_hi - t_lo) if math.isfinite(t_hi) and math.isfinite(t_lo) else abs(t_f))
    tans, zmids, s_rho = np.array(tans), np.array(zmids), np.array(s_rho)
    t_min, _ = tan_phi_and_depth(geom, rho, F_MIN_W * w_sq, centre)
    phi_min = math.degrees(math.atan(abs(t_min))) if math.isfinite(t_min) else 90.0
    if not np.all(np.isfinite(tans)) or not np.all(np.isfinite(zmids)):
        return LateralBudget(0.0, float("inf"), float("inf"), float("inf"), float("nan"), phi_min, "focal_prior")

    source = "focal_prior"
    if height_ratio is not None and all(math.isfinite(x) for x in height_ratio):
        kap, s_rel = height_ratio
        k_pred = np.sqrt(1.0 + tans ** 2)
        like = np.exp(-0.5 * ((kap - k_pred) / (k_pred * s_rel)) ** 2)
        if np.sum(wts * like) > 1e-12:
            wts = wts * like
            source += "+ball_height"
    if calib.has("rubber_left") and calib.has("rubber_right"):
        # Rubber ends are 2 ft apart, perpendicular to the pitch line: their
        # separation along the line axis / (2 ft * px-per-ft at R) = |tan phi|.
        e = (geom.R - geom.P) / geom.length_px
        a = calib.to_sq(calib.point("rubber_right")) - calib.to_sq(calib.point("rubber_left"))
        scale_R = rho * geom.length_px / geom.dy          # d mu / d lam = rho at the rubber
        t_m = abs(float(a @ e)) / (C.RUBBER_WIDTH_FT * scale_R)
        s_m = SIGMA_CLICK_PX * math.sqrt(2.0) * calib.sar / (C.RUBBER_WIDTH_FT * scale_R)
        like = np.exp(-0.5 * (np.abs(tans) - t_m) ** 2 / (s_m ** 2 + s_rho ** 2))
        if np.sum(wts * like) > 1e-12:
            wts = wts * like
            source += "+rubber_ends"
    wts = wts / np.sum(wts)

    c = LATERAL_PERSPECTIVE_COEF
    a_r = tans / L + c / zmids
    b_t = -tans / L + c / zmids
    th0, sth = math.tan(math.radians(TILT_PRIOR_MEAN_DEG)), math.radians(TILT_PRIOR_SIGMA_DEG)
    g_tilt = max(h_ft, 0.0) / zmids
    bias = a_r * xr_mean + g_tilt * th0
    var = (a_r ** 2 * xr_sigma ** 2 + b_t ** 2 * SIGMA_XT_FT ** 2 + (g_tilt * sth) ** 2
           + (s_rho / L) ** 2 * (xr_sigma ** 2 + xr_mean ** 2 + SIGMA_XT_FT ** 2))
    mean_bias = float(np.sum(wts * bias))
    total_var = float(np.sum(wts * (var + (bias - mean_bias) ** 2)))
    t_mean = float(np.sum(wts * tans))
    t_sig = float(math.sqrt(np.sum(wts * ((tans - t_mean) ** 2 + s_rho ** 2))))
    return LateralBudget(rel_bias=mean_bias, rel_sigma=math.sqrt(total_var), tan_phi=t_mean,
                         sigma_tan_phi=t_sig, z_mid_ft=float(np.sum(wts * zmids)), phi_min_deg=phi_min,
                         source=source)


def mound_height_ft(calib: Calibration) -> float:
    """Height of the rubber click above the plate (ASSUMED: 10 in at 60.5 ft, 6 in youth, flat softball)."""
    if calib.settings.sport == "softball":
        return 0.0
    return 10.0 / 12.0 if calib.distance_ft >= 58.0 else 0.5


def apparent_height_ratio(calib: Calibration, geom: AxisGeometry, xy_sq: np.ndarray, rho: float,
                          T: float) -> Optional[tuple[float, float]]:
    """(kappa, relative sigma): observed / expected ball height above the pitch line.

    Heights are measured in units of the local along-line scale, which is
    foreshortened by cos(phi) while verticals are not, so kappa ~ 1/cos(phi).
    The expected height is the release->catch chord (assumed heights) plus a
    gravity sag with an effective g of 0.8 +- 0.4 g (Magnus unknown).
    Returns None when the cue is unusable.
    """
    xy = np.atleast_2d(xy_sq)
    if len(xy) < 4:
        return None
    d = geom.R - geom.P
    b = xy - geom.P
    den = d[0] * geom.up[1] - d[1] * geom.up[0]
    s_px = (d[0] * b[:, 1] - d[1] * b[:, 0]) / den               # plumb offset above the line (px)
    mu = geom.mu(xy)
    lam = np.asarray(Calibration.lam_of_mu(mu, rho), float)
    g_px_ft = rho / np.maximum(rho * (1.0 - lam) + lam, 1e-6) ** 2 * geom.length_px / geom.dy
    h_app = s_px / g_px_ft
    y = geom.y_P + lam * geom.dy
    y_r, y_c = calib.release_y_ft, calib.catch_y_ft
    z_r, z_c = calib.release_height_ft, float(calib.settings.catch_height_ft)
    frac = np.clip((y - y_c) / (y_r - y_c), 0.0, 1.0)
    sag = 0.8 * C.G_FTPS2 * T * T / 8.0
    h_exp = z_c + (z_r - z_c) * frac + 4.0 * sag * frac * (1.0 - frac) - lam * mound_height_ft(calib)
    if np.median(h_exp) < 1.0 or np.median(h_app) <= 0.5:
        return None
    kappa = float(np.sum(h_app * h_exp) / np.sum(h_exp ** 2))
    s_zr = 0.3 if calib.settings.release_height_ft is not None else 0.6
    s_h = math.sqrt((s_zr / 2) ** 2 + (0.7 / 2) ** 2 + (2.0 / 3.0 * 0.5 * sag) ** 2)
    return kappa, s_h / float(np.mean(h_exp))


def camera_side(geom: AxisGeometry) -> str:
    """'first_base' if the rubber appears to the right of the plate, else 'third_base'.

    (A camera on the +x / first-base side looking at the pitch line sees
    +y, the pitcher, on its right.)
    """
    return "first_base" if geom.R[0] > geom.P[0] else "third_base"


def release_side(calib: Calibration, geom: AxisGeometry) -> tuple[float, float, bool]:
    """(mean, sigma) of the release-point lateral offset toward the camera (ft), known?"""
    throws = (calib.settings.pitcher_throws or "").upper()[:1]
    if throws not in ("R", "L"):
        return 0.0, SIGMA_XR_UNKNOWN_FT, False
    # A RHP releases on the third-base side (x < 0); 'toward the camera' is
    # +x for a first-base-side camera.
    x_world = -RELEASE_SIDE_FT if throws == "R" else RELEASE_SIDE_FT
    toward = x_world if camera_side(geom) == "first_base" else -x_world
    return toward, SIGMA_XR_KNOWN_FT, True


def mean_height_ft(geom: AxisGeometry, xy_sq: np.ndarray) -> float:
    """Mean height of the ball above the P->R line along the plumb direction (ft, affine scale)."""
    d = geom.R - geom.P
    b = np.atleast_2d(xy_sq) - geom.P
    den = d[0] * geom.up[1] - d[1] * geom.up[0]
    s = (d[0] * b[:, 1] - d[1] * b[:, 0]) / den
    return float(np.mean(s) * geom.dy / geom.length_px)


# ---------------------------------------------------------- the estimator
@dataclass
class _Points:
    t: np.ndarray
    s: np.ndarray
    xy_sq: np.ndarray
    w: np.ndarray
    n_total: int
    n_dropped_events: int = 0


def _collect_points(track: PitchTrack, calib: Calibration, sample_dt: float,
                    release: Optional[EventTime], catch: Optional[EventTime]) -> _Points:
    pts = [p for p in track.points if p.inlier and K.finite(p.x) and K.finite(p.y) and K.finite(p.t)]
    n_total = len(pts)
    lo = release.t - 0.25 * sample_dt if release is not None and K.finite(release.t) else -math.inf
    hi = catch.t + 0.25 * sample_dt if catch is not None and K.finite(catch.t) else math.inf
    kept = [p for p in pts if lo <= p.t <= hi]
    kept.sort(key=lambda p: p.t)
    t = np.array([p.t for p in kept], float)
    s = np.array([p.s for p in kept], int)
    xy = np.array([[p.x * calib.sar, p.y] for p in kept], float).reshape(-1, 2)
    sig = np.array([max(float(p.sigma) if K.finite(p.sigma) else 1.0, 0.3) for p in kept], float)
    w = (np.median(sig) / sig) if len(sig) else sig
    return _Points(t=t, s=s, xy_sq=xy, w=w, n_total=n_total, n_dropped_events=n_total - len(kept))


def swapped_times(t: np.ndarray, s: np.ndarray, sample_dt: float) -> np.ndarray:
    """Sample times under the opposite field order (sample s = 2*frame + k)."""
    k = np.mod(s, 2)
    return t + (1.0 - 2.0 * k) * sample_dt


def side_fit_estimate(calib: Calibration, scan: Optional[ScanInfo], sample_dt: float,
                      track: PitchTrack, *, release: Optional[EventTime] = None,
                      catch: Optional[EventTime] = None, drag: Optional[DragModel] = None,
                      sigma_click_px: float = SIGMA_CLICK_PX) -> SpeedComponent:
    """E2 side-view fit.  Raises `NotMeasurable` if no fit is possible.

    A fit that is possible but should not be trusted (oblique view, poor
    fit, implausible) is returned with details['usable'] = False and a reason.
    `release` / `catch` only bound which track points are used.
    """
    if calib.view != "side":
        raise K.NotMeasurable("side-view fit needs a side-view calibration")
    if track is None:
        raise K.NotMeasurable("no ball track")
    geom = axis_geometry(calib)
    pts = _collect_points(track, calib, sample_dt, release, catch)
    if len(pts.t) < K.MIN_SIDE_POINTS:
        raise K.NotMeasurable(f"only {len(pts.t)} usable track points (need {K.MIN_SIDE_POINTS})")

    d0 = drag or K.drag_for(calib)
    k = d0.k
    y_r = calib.release_y_ft
    t0 = float(pts.t[0])
    mu = geom.mu(pts.xy_sq)
    flags: list[str] = []
    notes: list[str] = []

    # --- main fit, and the opposite field order as a consistency check
    tt, w_used, xy_used = pts.t - t0, pts.w, pts.xy_sq
    fit = fit_track(tt, mu, w_used, geom, y_r, k)
    order_used = "scan"
    rms_alt = float("nan")
    if scan is not None and scan.interlaced:
        t_alt = swapped_times(pts.t, pts.s, sample_dt) - t0
        o = np.argsort(t_alt, kind="stable")
        fit_alt = fit_track(t_alt[o], mu[o], pts.w[o], geom, y_r, k)
        rms_alt = fit_alt.rms_px
        if fit.rms_px > 1.0 and fit_alt.rms_px < ALT_ORDER_RATIO * fit.rms_px:
            flags.append(K.F_FIELD_ORDER)
            notes.append(f"track fits much better with the opposite field order "
                         f"({fit_alt.rms_px:.1f} vs {fit.rms_px:.1f} px rms); used that order for E2")
            order_used = "swapped"
            tt, mu, w_used, xy_used = t_alt[o], mu[o], pts.w[o], pts.xy_sq[o]
            rms_alt = fit.rms_px                       # now the 'other' order is the scan's
            fit = fit_alt
    elif scan is not None:
        flags.append(K.F_FRAMES_ONLY)

    t_used = tt[fit.used]
    alt = residual_alternation(t_used, fit.resid_px, sample_dt)
    if math.isfinite(alt) and alt < ALTERNATION_FLAG and fit.rms_px > 1.5 and K.F_FIELD_ORDER not in flags:
        flags.append(K.F_FIELD_ORDER)
        notes.append(f"fit residuals alternate field to field (r={alt:.2f}): field order or "
                     f"field parity problem")

    v_h = fit.v0
    rho = fit.rho
    sig_cov = math.sqrt(max(float(fit.cov[0, 0]), 0.0)) if np.isfinite(fit.cov[0, 0]) else float("nan")
    sig_lnrho = math.sqrt(max(float(fit.cov[2, 2]), 0.0)) if np.isfinite(fit.cov[2, 2]) else float("nan")

    # --- perturbation re-fits (same point set, warm start)
    def refit(g: AxisGeometry, k_p: float) -> float:
        mu_p = mu if g is geom else g.mu(xy_used)
        return fit_track(tt, mu_p, w_used, g, y_r, k_p, x0=fit.theta, used=fit.used,
                         reject_outliers=False).v0

    click_terms = []
    for which, axis in (("P", 0), ("P", 1), ("R", 0), ("R", 1)):
        Pp, Rp = geom.P.copy(), geom.R.copy()
        (Pp if which == "P" else Rp)[axis] += sigma_click_px * (calib.sar if axis == 0 else 1.0)
        click_terms.append(refit(AxisGeometry(Pp, Rp, geom.up, geom.y_P, geom.y_R), k) - v_h)
    if calib.has("vertical_top") and calib.has("vertical_bottom"):
        vlen = float(np.linalg.norm(calib.to_sq(calib.point("vertical_top")) -
                                    calib.to_sq(calib.point("vertical_bottom"))))
        sig_up = math.atan2(sigma_click_px * math.sqrt(2.0), max(vlen, 1.0))
    else:
        sig_up = math.radians(SIGMA_UP_DEG)
    dv_up = refit(AxisGeometry(geom.P, geom.R, _rotate(geom.up, sig_up), geom.y_P, geom.y_R), k) - v_h
    d = drag or K.drag_for(calib, K.mph(v_h))
    dv_k = refit(geom, k * (1.0 + d.rel_sigma)) - v_h

    # --- obliquity and the sideways-travel term
    y_end, _, _ = K.end_point(calib, "catch")
    L = y_r - y_end
    xr_mean, xr_sig, side_known = release_side(calib, geom)
    h_ft = mean_height_ft(geom, xy_used[fit.used])
    hr = apparent_height_ratio(calib, geom, xy_used[fit.used], rho, time_to_travel(v_h, L, k))
    lat = lateral_budget(calib, geom, rho, sig_lnrho, L, xr_mean, xr_sig, h_ft, hr)
    rel_lat = lat.rel_sigma
    lat_corr = 1.0 / max(1.0 + lat.rel_bias, 0.5)      # apparent = true * (1 + bias)

    v_h_corr = v_h * lat_corr
    T = time_to_travel(v_h_corr, L, k)
    sf = K.slope_factor(calib, L, T)
    v0 = v_h_corr * sf
    scale = lat_corr * sf                          # applies to all the horizontal-speed deltas

    rel_ext = k * float(calib.settings.extension_sigma_ft)    # e^{k dExt}: speed at the true release point
    var_ind = ((sig_cov * scale) ** 2 + sum((dv * scale) ** 2 for dv in click_terms) + (dv_up * scale) ** 2
               + (v0 * rel_lat) ** 2 + (v0 * rel_ext) ** 2 + (v0 * K.SIGMA_MODEL_SIDE) ** 2)
    sigma_ind = math.sqrt(var_ind) if math.isfinite(var_ind) else float("inf")

    # --- coverage of the release->catch path by the observations
    y_first = float(model_y(t_used[:1], fit.theta[0], fit.theta[1], y_r, k)[0])
    y_last = float(model_y(t_used[-1:], fit.theta[0], fit.theta[1], y_r, k)[0])
    coverage = (y_first - y_last) / L
    if coverage < PARTIAL_COVERAGE:
        flags.append(K.F_PARTIAL_TRACK)
        notes.append(f"track covers only {100 * coverage:.0f}% of the release->catch path")

    usable, reason = True, None
    if not math.isfinite(sigma_ind):
        usable = False
        flags.append(K.F_POOR_FIT)
        reason = "side-view fit uncertainty could not be computed; not used"
    elif not math.isfinite(rel_lat) or max(lat.phi_min_deg, lat.phi_deg) > MAX_OBLIQUITY_DEG:
        usable = False
        flags.append(K.F_OBLIQUE)
        reason = (f"camera too oblique to the pitch line (phi ~ {max(lat.phi_min_deg, lat.phi_deg):.0f} deg, "
                  f"rho={rho:.2f}); side-view fit not used")
    elif fit.at_bound:
        usable = False
        flags.append(K.F_OBLIQUE if abs(fit.theta[2]) > 2.9 else K.F_POOR_FIT)
        reason = "side-view fit ran into a parameter bound; not used"
    elif (rho < 0.5 or rho > 2.0) and fit.rms_px > 2.0:
        usable = False
        flags.append(K.F_OBLIQUE)
        reason = f"rho={rho:.2f} far from 1 with a poor fit ({fit.rms_px:.1f} px rms); side-view fit not used"
    elif fit.rms_px > POOR_FIT_RMS_PX:
        usable = False
        flags.append(K.F_POOR_FIT)
        reason = f"side-view fit residuals too large ({fit.rms_px:.1f} px rms); not used"
    elif not (K.V_MIN_MPH <= K.mph(v0) <= K.V_MAX_MPH):
        usable = False
        flags.append(K.F_IMPLAUSIBLE)
        reason = f"side-view fit gives {K.mph(v0):.0f} mph, implausible; not used"

    t_r_abs = float(fit.theta[1]) + t0
    details = {
        "usable": usable,
        "reason": reason,
        "flags": sorted(set(flags)),
        "notes": notes,
        "n_points": int(pts.n_total),
        "n_outside_events": int(pts.n_dropped_events),
        "n_used": int(fit.n_used),
        "rms_px": fit.rms_px,
        "rms_px_other_field_order": rms_alt,
        "field_order_used": order_used,
        "alternation": alt,
        "rho": rho,
        "sigma_ln_rho": sig_lnrho,
        "at_bound": fit.at_bound,
        "phi_deg": lat.phi_deg,
        "phi_min_deg": lat.phi_min_deg,
        "tan_phi": lat.tan_phi,
        "sigma_tan_phi": lat.sigma_tan_phi,
        "phi_source": lat.source,
        "z_mid_ft": lat.z_mid_ft,
        "rel_bias_lateral": lat.rel_bias,
        "ball_height_ft": h_ft,
        "height_ratio": None if hr is None else hr[0],
        "camera_side": camera_side(geom),
        "lateral_correction": lat_corr,
        "lateral_known": side_known,
        "t_release_fit": t_r_abs,
        "coverage": coverage,
        "L_ft": L,
        "k_per_ft": k,
        "k_rel_sigma": d.rel_sigma,
        "slope_factor": sf,
        "v_horizontal_mph": K.mph(v_h),
        "v_plate_mph": K.mph(K.plate_speed(v0, y_r, k)),
        "sigma_cov_mph": K.mph(sig_cov * scale),
        "sigma_click_mph": K.mph(math.sqrt(sum(dv ** 2 for dv in click_terms)) * scale),
        "sigma_up_mph": K.mph(abs(dv_up) * scale),
        "rel_sigma_lateral": rel_lat,
        "rel_sigma_extension": rel_ext,
        "rel_sigma_model": K.SIGMA_MODEL_SIDE,
        "rel_sigma_k": abs(dv_k * scale) / v0,
        "dv_k_mph": K.mph(dv_k * scale),
    }
    return SpeedComponent(method="side_fit", v_release_mph=K.mph(v0), sigma_ind_mph=K.mph(sigma_ind),
                          sigma_shared_mph=abs(K.mph(dv_k * scale)), v_avg_mph=K.mph(L / T), details=details)
