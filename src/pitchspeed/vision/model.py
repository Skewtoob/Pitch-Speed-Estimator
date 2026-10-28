"""Image-trajectory models for tracking (gating, gap filling, extrapolation).

Global model (cv.md section 4), with normalized time tau = (t - t0) / ts:

    x(tau) = (a0 + a1 tau + a2 tau^2) / (1 + c1 tau + c2 tau^2)
    y(tau) = (b0 + b1 tau + b2 tau^2) / (1 + c1 tau + c2 tau^2)

It is exact for a pinhole camera and constant 3-D acceleration, and a good
approximation under drag.  'poly' is the same with c = 0.  These models are
used for gating, gap filling and event-time extrapolation only; speeds are
computed by `pitchspeed.estimate`.

Fitting: linearized least squares (multiply out the denominator) inside
RANSAC, then `scipy.optimize.least_squares(loss='soft_l1')`.  Model
selection for side views by BIC (rational vs poly); a rational fit whose
denominator comes close to zero over the (extended) data span is rejected.

`TrajModel.to_dict()` is what goes into `PitchTrack.model`; evaluate it with
`eval_model(model_dict, t)`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import least_squares


@dataclass
class TrajModel:
    kind: str                         # 'rational' | 'poly' | 'linear'
    t0: float
    ts: float
    a: np.ndarray                     # x numerator (3,)
    b: np.ndarray                     # y numerator (3,)
    c: np.ndarray                     # denominator (2,) (zeros for poly)
    t_min: float
    t_max: float
    inliers: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    rms: float = float("nan")         # px (combined x/y, inliers)
    rms_x: float = float("nan")
    rms_y: float = float("nan")
    cov: Optional[np.ndarray] = None  # parameter covariance (8x8 or 6x6)
    n: int = 0
    bic: float = float("nan")

    # ------------------------------------------------------------ evaluate
    def _tau(self, t):
        return (np.asarray(t, dtype=np.float64) - self.t0) / self.ts

    def denom(self, t) -> np.ndarray:
        u = self._tau(t)
        return 1.0 + self.c[0] * u + self.c[1] * u * u

    def eval(self, t) -> np.ndarray:
        """(N, 2) positions at times t."""
        u = self._tau(t)
        d = 1.0 + self.c[0] * u + self.c[1] * u * u
        x = (self.a[0] + self.a[1] * u + self.a[2] * u * u) / d
        y = (self.b[0] + self.b[1] * u + self.b[2] * u * u) / d
        return np.stack([np.atleast_1d(x), np.atleast_1d(y)], axis=-1)

    def vel(self, t, h: float = 1e-3) -> np.ndarray:
        """(N, 2) velocity (px / s) at times t (central difference)."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        return (self.eval(t + h) - self.eval(t - h)) / (2 * h)

    def params(self) -> np.ndarray:
        if self.kind == "rational":
            return np.concatenate([self.a, self.b, self.c])
        return np.concatenate([self.a, self.b])

    def pos_sigma(self, t) -> np.ndarray:
        """1-sigma position uncertainty (px, combined) of eval(t) from the covariance."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        if self.cov is None:
            return np.full(len(t), 1.0)
        p0 = self.params()
        J = np.zeros((len(t), 2, len(p0)))
        base = self.eval(t)
        for i in range(len(p0)):
            dp = np.zeros_like(p0)
            dp[i] = 1e-6 * max(1.0, abs(p0[i]))
            m2 = _with_params(self, p0 + dp)
            J[:, :, i] = (m2.eval(t) - base) / dp[i]
        var = np.einsum("nij,jk,nik->n", J, self.cov, J)
        return np.sqrt(np.maximum(var, 0.0))

    def to_dict(self) -> dict:
        return {"kind": self.kind, "t0": float(self.t0), "ts": float(self.ts),
                "a": [float(v) for v in self.a], "b": [float(v) for v in self.b],
                "c": [float(v) for v in self.c], "t_min": float(self.t_min), "t_max": float(self.t_max),
                "rms_px": float(self.rms), "rms_x": float(self.rms_x), "rms_y": float(self.rms_y),
                "n_inliers": int(self.inliers.sum()) if len(self.inliers) else int(self.n),
                "coords": "reference frame px (stored pixels); t in seconds",
                "formula": "x = (a0 + a1 u + a2 u^2) / (1 + c1 u + c2 u^2), u = (t - t0) / ts; y likewise with b"}

    def pole_time(self) -> Optional[float]:
        """Time (s) where the denominator vanishes nearest after the data (behind view: ball at camera)."""
        c1, c2 = self.c
        roots = []
        if abs(c2) > 1e-12:
            disc = c1 * c1 - 4 * c2
            if disc >= 0:
                roots = [(-c1 + s * math.sqrt(disc)) / (2 * c2) for s in (1, -1)]
        elif abs(c1) > 1e-12:
            roots = [-1.0 / c1]
        ts = [self.t0 + r * self.ts for r in roots]
        ts = [t for t in ts if t > self.t_max]
        return min(ts) if ts else None


def _with_params(m: TrajModel, p: np.ndarray) -> TrajModel:
    if m.kind == "rational":
        a, b, c = p[0:3], p[3:6], p[6:8]
    else:
        a, b, c = p[0:3], p[3:6], np.zeros(2)
    return TrajModel(kind=m.kind, t0=m.t0, ts=m.ts, a=a, b=b, c=c, t_min=m.t_min, t_max=m.t_max)


def model_from_dict(d: dict) -> TrajModel:
    return TrajModel(kind=d["kind"], t0=d["t0"], ts=d["ts"], a=np.asarray(d["a"], float),
                     b=np.asarray(d["b"], float), c=np.asarray(d["c"], float),
                     t_min=d.get("t_min", d["t0"] - d["ts"]), t_max=d.get("t_max", d["t0"] + d["ts"]),
                     rms=d.get("rms_px", float("nan")))


def eval_model(d: dict, t) -> np.ndarray:
    """Evaluate a `PitchTrack.model` dict at times t -> (N, 2) reference px."""
    return model_from_dict(d).eval(t)


# ------------------------------------------------------------------ fitting ----
def _norm(t: np.ndarray) -> tuple[float, float]:
    lo, hi = float(np.min(t)), float(np.max(t))
    t0 = 0.5 * (lo + hi)
    ts = max(0.5 * (hi - lo), 1e-3)
    return t0, ts


def _lin_fit(u, x, y, wx, wy, kind):
    """Weighted linear LS of the (linearized) model; returns params or None."""
    n = len(u)
    one = np.ones(n)
    if kind == "rational":
        Ax = np.column_stack([one, u, u * u, 0 * u, 0 * u, 0 * u, -u * x, -u * u * x])
        Ay = np.column_stack([0 * u, 0 * u, 0 * u, one, u, u * u, -u * y, -u * u * y])
    elif kind == "poly":
        Ax = np.column_stack([one, u, u * u, 0 * u, 0 * u, 0 * u])
        Ay = np.column_stack([0 * u, 0 * u, 0 * u, one, u, u * u])
    else:  # linear
        Ax = np.column_stack([one, u, 0 * u, 0 * u, 0 * u, 0 * u])
        Ay = np.column_stack([0 * u, 0 * u, 0 * u, one, u, 0 * u])
    A = np.vstack([Ax * wx[:, None], Ay * wy[:, None]])
    rhs = np.concatenate([x * wx, y * wy])
    try:
        p, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if kind == "linear":
        p = p.copy()
        p[2] = p[5] = 0.0
    return p


def _make(kind, p, t0, ts, t_min, t_max) -> TrajModel:
    if kind == "rational":
        return TrajModel(kind, t0, ts, p[0:3].copy(), p[3:6].copy(), p[6:8].copy(), t_min, t_max)
    return TrajModel(kind, t0, ts, p[0:3].copy(), p[3:6].copy(), np.zeros(2), t_min, t_max)


def _denom_ok(m: TrajModel, lo_ext: float = 0.2, hi_ext: float = 0.1, min_d: float = 0.05) -> bool:
    """The denominator (~ depth ratio) must stay positive over the data span.

    Behind views legitimately approach the pole (ball nearing the camera),
    so only a small extrapolation margin past the data end is required.
    """
    if m.kind != "rational":
        return True
    u = np.linspace(-1.0 - lo_ext, 1.0 + hi_ext, 41)
    d = 1.0 + m.c[0] * u + m.c[1] * u * u
    return bool(np.all(d > min_d))


def _residuals(m: TrajModel, t, x, y) -> tuple[np.ndarray, np.ndarray]:
    P = m.eval(t)
    return x - P[:, 0], y - P[:, 1]


def fit_model(t, x, y, sigma=None, kind: str = "auto", view: str = "side",
              tol_x: float = 1.5, tol_y: float = 2.0, ransac_iters: int = 200,
              rng: Optional[np.random.Generator] = None, fixed: Optional[np.ndarray] = None) -> Optional[TrajModel]:
    """Robust fit of an image trajectory.

    t, x, y: observations (reference px); sigma: per-point 1-sigma (px).
    kind: 'rational' | 'poly' | 'linear' | 'auto' (side: BIC; behind:
    rational when >= 8 points).  tol_x / tol_y: inlier tolerances (px).
    fixed: bool mask of points that must stay inliers (manual points).
    Returns None with fewer than 3 points.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(t)
    if n < 3:
        return None
    sig = np.ones(n) if sigma is None else np.maximum(np.asarray(sigma, dtype=np.float64), 0.2)
    fixed = np.zeros(n, bool) if fixed is None else np.asarray(fixed, bool)
    rng = rng or np.random.default_rng(12345)
    t0, ts = _norm(t)
    u = (t - t0) / ts
    if kind == "auto":
        if n < 4:
            kinds = ["linear"]
        elif n < 8:
            kinds = ["poly"]
        elif view == "behind":
            kinds = ["rational", "poly"]
        else:
            kinds = ["poly", "rational"]
    else:
        kinds = [kind]
    best_models = []
    for kd in kinds:
        m = _robust_fit(kd, t, u, x, y, sig, t0, ts, tol_x, tol_y, ransac_iters, rng, fixed)
        if m is not None:
            best_models.append(m)
    if not best_models:
        return None
    if len(best_models) == 1:
        return best_models[0]
    if view == "behind":
        # prefer the rational model unless it is clearly worse
        r = [m for m in best_models if m.kind == "rational"]
        p = [m for m in best_models if m.kind == "poly"]
        if r and p and r[0].inliers.sum() + 1 >= p[0].inliers.sum():
            return r[0]
        return max(best_models, key=lambda m: (m.inliers.sum(), -m.rms))
    # side view: BIC on a common inlier set
    common = np.logical_and.reduce([m.inliers for m in best_models])
    if common.sum() < 6:
        return max(best_models, key=lambda m: (m.inliers.sum(), -m.rms))
    scores = []
    for m in best_models:
        rx, ry = _residuals(m, t[common], x[common], y[common])
        rss = float(np.sum((rx / sig[common]) ** 2 + (ry / sig[common]) ** 2)) + 1e-9
        k = 8 if m.kind == "rational" else 6
        n_eq = 2 * int(common.sum())
        bic = n_eq * math.log(rss / n_eq) + k * math.log(n_eq)
        m.bic = bic
        scores.append(bic - 0.5 * m.inliers.sum())   # small preference for more inliers
    return best_models[int(np.argmin(scores))]


def _robust_fit(kind, t, u, x, y, sig, t0, ts, tol_x, tol_y, iters, rng, fixed) -> Optional[TrajModel]:
    n = len(t)
    need = {"rational": 5, "poly": 3, "linear": 2}[kind]
    if n < need + (1 if kind == "rational" else 0):
        return None
    w = 1.0 / sig
    best_inl, best_cost = None, np.inf
    idx_all = np.arange(n)
    n_iter = iters if n > need else 1
    it = -1
    while it + 1 < n_iter:
        it += 1
        if n > need:
            sel = rng.choice(n, need, replace=False)
            if fixed.any() and it % 3 == 0:
                fx = idx_all[fixed]
                sel = np.unique(np.concatenate([sel, fx[:need]]))
        else:
            sel = idx_all
        p = _lin_fit(u[sel], x[sel], y[sel], w[sel], w[sel], kind)
        if p is None:
            continue
        m = _make(kind, p, t0, ts, t[0], t[-1])
        if not _denom_ok(m, 0.1, 0.05):
            continue
        rx, ry = _residuals(m, t, x, y)
        inl = (np.abs(rx) <= tol_x) & (np.abs(ry) <= tol_y)
        inl |= fixed
        # MSAC cost
        cost = np.sum(np.minimum((rx / tol_x) ** 2 + (ry / tol_y) ** 2, 1.0 + 1.0))
        cost -= 0.5 * inl.sum()
        if best_inl is None or cost < best_cost:
            best_inl, best_cost = inl, cost
            # adaptive termination (99 % confidence of one all-inlier sample)
            wfr = min(0.999, max(1e-3, inl.mean()))
            denom = math.log(max(1e-12, 1.0 - wfr ** need))
            n_need = int(math.ceil(math.log(0.01) / denom)) if denom < 0 else 1
            n_iter = min(n_iter, max(12, n_need))
    if best_inl is None or best_inl.sum() < need:
        return None
    m = None
    inl = best_inl
    for _round in range(3):
        m = _refine(kind, t, u, x, y, sig, inl, t0, ts)
        if m is None:
            return None
        rx, ry = _residuals(m, t, x, y)
        new = ((np.abs(rx) <= tol_x) & (np.abs(ry) <= tol_y)) | fixed
        if new.sum() < need:
            break
        if np.array_equal(new, inl):
            break
        inl = new
    rx, ry = _residuals(m, t, x, y)
    m.inliers = inl
    m.n = n
    m.rms_x = float(np.sqrt(np.mean(rx[inl] ** 2)))
    m.rms_y = float(np.sqrt(np.mean(ry[inl] ** 2)))
    m.rms = float(np.sqrt(np.mean(rx[inl] ** 2 + ry[inl] ** 2)))
    m.t_min, m.t_max = float(t[inl].min()), float(t[inl].max())
    if kind == "rational" and not _denom_ok(m):
        return None
    return m


def _refine(kind, t, u, x, y, sig, inl, t0, ts) -> Optional[TrajModel]:
    w = 1.0 / sig[inl]
    p0 = _lin_fit(u[inl], x[inl], y[inl], w, w, kind)
    if p0 is None:
        return None
    ui, xi, yi = u[inl], x[inl], y[inl]
    if kind == "rational":
        def res(p):
            d = 1.0 + p[6] * ui + p[7] * ui * ui
            d = np.where(np.abs(d) < 1e-3, 1e-3, d)
            fx = (p[0] + p[1] * ui + p[2] * ui * ui) / d
            fy = (p[3] + p[4] * ui + p[5] * ui * ui) / d
            return np.concatenate([(xi - fx) * w, (yi - fy) * w])
        try:
            sol = least_squares(res, p0, loss="soft_l1", f_scale=1.0, max_nfev=200)
            p = sol.x
            J = sol.jac
        except Exception:
            return None
    else:
        p = p0
        one = np.ones(len(ui))
        cols = [one, ui, ui * ui] if kind == "poly" else [one, ui, 0 * ui]
        Ax = np.column_stack(cols + [0 * ui] * 3)
        Ay = np.column_stack([0 * ui] * 3 + cols)
        J = -np.vstack([Ax * w[:, None], Ay * w[:, None]])
    m = _make(kind, p, t0, ts, float(t[inl].min()), float(t[inl].max()))
    # covariance: s^2 (J^T J)^-1 with s from the residuals (>= the given sigmas)
    rx, ry = _residuals(m, t[inl], x[inl], y[inl])
    dof = max(1, 2 * int(inl.sum()) - len(p))
    s2 = max(1.0, float(np.sum((rx * w) ** 2 + (ry * w) ** 2)) / dof)
    try:
        JtJ = J.T @ J
        if kind == "linear":
            keep = [0, 1, 3, 4]
            C = np.zeros((6, 6))
            C[np.ix_(keep, keep)] = np.linalg.pinv(JtJ[np.ix_(keep, keep)]) * s2
            m.cov = C
        else:
            m.cov = np.linalg.pinv(JtJ) * s2
    except np.linalg.LinAlgError:
        m.cov = None
    return m


# ------------------------------------------------------------- local predict ----
def local_predict(t, x, y, t_new: float, max_pts: int = 8, sigma: float = 1.0) -> tuple[np.ndarray, float, np.ndarray]:
    """Predict the position at t_new from the nearest <= max_pts points (quadratic if >= 4).

    Returns (position (2,), 1-sigma prediction error (px), velocity (2,) px/s).
    """
    t = np.asarray(t, dtype=np.float64)
    o = np.argsort(np.abs(t - t_new))[:max_pts]
    o = np.sort(o)
    tt, xx, yy = t[o], np.asarray(x, float)[o], np.asarray(y, float)[o]
    n = len(tt)
    if n == 1:
        return np.array([xx[0], yy[0]]), 5.0, np.zeros(2)
    t0 = tt.mean()
    ts = max(np.ptp(tt), 1e-3)
    u = (tt - t0) / ts
    un = (t_new - t0) / ts
    deg = 2 if n >= 4 else 1
    V = np.vander(u, deg + 1)
    vn = np.vander(np.array([un]), deg + 1)[0]
    try:
        cx, *_ = np.linalg.lstsq(V, xx, rcond=None)
        cy, *_ = np.linalg.lstsq(V, yy, rcond=None)
        cov = np.linalg.pinv(V.T @ V)
    except np.linalg.LinAlgError:
        return np.array([xx[-1], yy[-1]]), 5.0, np.zeros(2)
    px, py = float(vn @ cx), float(vn @ cy)
    dof = max(1, n - deg - 1)
    rx = xx - V @ cx
    ry = yy - V @ cy
    s2 = max(sigma ** 2, float(np.sum(rx ** 2 + ry ** 2)) / (2 * dof)) if n > deg + 1 else sigma ** 2
    var = float(vn @ cov @ vn) * s2 * 2
    dv = np.polyder(np.poly1d(cx)), np.polyder(np.poly1d(cy))
    vel = np.array([dv[0](un), dv[1](un)]) / ts
    return np.array([px, py]), math.sqrt(max(var, 0.0)), vel
