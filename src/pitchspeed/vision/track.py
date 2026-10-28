"""Seed-extend-robust-fit ball tracking on stabilized candidates.

* Seeds: triplets (a, b, c) of candidates in samples (k-g1, k, k+g2),
  g1, g2 in 1..3, g1 + g2 <= 4, with b strong and not attached, both
  velocities inside the view's gate and mutually consistent.
* Extension: from a seed in both directions, predict the next position
  with a local quadratic (nearest <= 8 points) and accept the best
  candidate (weak / attached allowed) inside a radius that grows with
  speed, gap and prediction uncertainty.  Stops after `miss_limit`
  consecutive misses (more inside a body blob or with a good model), or
  when the prediction leaves the corridor / progress [-0.15, 1.15].
* Robust global fit (`model.fit_model`), pass-2 weak re-detection inside
  the model gate on the cached z maps, refit.
* Field-order consistency: residual RMS under the given vs the flipped
  field timing (t' = t + (1 - 2k) frame_dt / 2) plus the lag-1 residual
  alternation; `field_order_conflict` when the flip is strongly preferred.

All positions are reference-frame px (stabilized frame rows).
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pitchspeed.vision.detect import DetFrame
from pitchspeed.vision.geometry import ViewGeometry
from pitchspeed.vision.model import TrajModel, fit_model, local_predict
from pitchspeed.vision.options import FinderOptions


# ------------------------------------------------------------------ store ----
class CandStore:
    """Ring buffer of recent DetFrames (candidates + z maps), keyed by sample."""

    def __init__(self, max_len: int = 360):
        self.max_len = max_len
        self.frames: "OrderedDict[int, DetFrame]" = OrderedDict()
        self._used: dict = {}                  # s -> bool array aligned with frames[s].cands

    def add(self, df: DetFrame) -> None:
        self.frames[df.s] = df
        self._used[df.s] = np.zeros(len(df.cands), bool)
        while len(self.frames) > self.max_len:
            old, _ = self.frames.popitem(last=False)
            self._used.pop(old, None)

    def used_mask(self, s: int) -> np.ndarray:
        u = self._used.get(s)
        return u if u is not None else np.zeros(0, bool)

    def _index(self, s: int, uid: int) -> int:
        df = self.frames.get(s)
        if df is None or len(df.cands) == 0 or uid < 0:
            return -1
        i = int(uid - df.cands["uid"][0])
        return i if 0 <= i < len(df.cands) and df.cands["uid"][i] == uid else -1

    def is_used(self, s: int, uid: int) -> bool:
        i = self._index(s, uid)
        return bool(self._used[s][i]) if i >= 0 else False

    def mark(self, s: int, uid: int, value: bool = True) -> None:
        i = self._index(s, uid)
        if i >= 0:
            self._used[s][i] = value

    def claim(self, tr: "Track", value: bool = True) -> None:
        for p in tr.pts.values():
            if p.inlier or not value:
                self.mark(p.s, p.uid, value)

    def get(self, s: int) -> Optional[DetFrame]:
        return self.frames.get(s)

    @property
    def s_min(self) -> Optional[int]:
        return next(iter(self.frames)) if self.frames else None

    @property
    def s_max(self) -> Optional[int]:
        return next(reversed(self.frames)) if self.frames else None


# ------------------------------------------------------------------ track ----
@dataclass
class Pt:
    s: int
    t: float
    k: int
    x: float
    y: float
    z: float
    attached: bool
    sigma: float
    uid: int = -1
    length: float = 0.0
    src: str = "det"              # 'det' | 'pass2' | 'manual'
    inlier: bool = True
    sign: int = 1                 # contrast sign (+1 brighter than background)


@dataclass
class Track:
    tid: int
    pts: dict = field(default_factory=dict)       # s -> Pt
    model: Optional[TrajModel] = None
    seed_score: float = 0.0
    open_fwd: bool = False
    info: dict = field(default_factory=dict)

    def sorted(self) -> list:
        return [self.pts[s] for s in sorted(self.pts)]

    def arrays(self, inliers_only: bool = False) -> dict:
        P = [p for p in self.sorted() if (p.inlier or not inliers_only)]
        return {
            "s": np.array([p.s for p in P], np.int64), "t": np.array([p.t for p in P]),
            "k": np.array([p.k for p in P], np.int64), "x": np.array([p.x for p in P]),
            "y": np.array([p.y for p in P]), "z": np.array([p.z for p in P]),
            "att": np.array([p.attached for p in P], bool), "sigma": np.array([p.sigma for p in P]),
            "src": [p.src for p in P],
        }

    @property
    def s_first(self) -> int:
        return min(self.pts)

    @property
    def s_last(self) -> int:
        return max(self.pts)

    @property
    def n_inliers(self) -> int:
        return sum(1 for p in self.pts.values() if p.inlier)

    def uids(self) -> set:
        return {p.uid for p in self.pts.values() if p.uid >= 0}


# ------------------------------------------------------------------ gates ----
def _vec_sq(geom: ViewGeometry, dx, dy) -> np.ndarray:
    return np.stack([np.asarray(dx, float) * geom.sar, np.asarray(dy, float)], axis=-1)


def velocity_gate(geom: ViewGeometry, v_sq: np.ndarray, opts: FinderOptions, seed: bool = True) -> np.ndarray:
    """Plausible image velocity (square px / sample) for a pitched (or thrown-back) ball."""
    v_sq = np.atleast_2d(v_sq)
    sp = np.linalg.norm(v_sq, axis=1)
    f = opts.gate_speed_factor
    if geom.view == "side":
        s_rel = geom.scale_rel
        v_lo = opts.min_speed_mph * 22.0 / 15.0 * s_rel * geom.sample_dt / f
        v_hi = opts.max_speed_mph * 22.0 / 15.0 * s_rel * geom.sample_dt * f
        cosang = np.abs(v_sq @ geom.axis_sq) / np.maximum(sp, 1e-9)
        ok = (sp >= v_lo) & (sp <= v_hi) & (cosang >= math.cos(math.radians(opts.seed_max_angle_deg)))
        return ok
    # behind view: bounded speed, clear motion for seeds, progress along +-axis
    dprog = (v_sq @ geom.axis_sq) / geom.length_sq
    v_hi = 0.5 * geom.length_sq
    ok = sp <= v_hi
    if seed:
        ok &= (sp >= 0.6) & (np.abs(dprog) >= 0.002)
    return ok


def find_seeds(store: CandStore, s_lo: int, s_hi: int, geom: ViewGeometry, opts: FinderOptions) -> list:
    """Seed triplets with middle sample in [s_lo, s_hi]; sorted by score (desc)."""
    seeds = []
    for sb in range(s_lo, s_hi + 1):
        fb = store.get(sb)
        if fb is None or len(fb.cands) == 0:
            continue
        B = fb.cands
        strong = np.nonzero((B["z"] >= opts.z_strong) & (~B["attached"]))[0]
        if len(strong) == 0:
            continue
        before, after = [], []
        for g in (1, 2, 3):
            fa = store.get(sb - g)
            if fa is not None and len(fa.cands):
                before.append((g, fa))
            fc = store.get(sb + g)
            if fc is not None and len(fc.cands):
                after.append((g, fc))
        if not before or not after:
            continue
        for bi in strong:
            b = B[bi]
            if store.used_mask(sb)[bi]:
                continue
            A_list = []
            for g1, fa in before:
                a = fa.cands
                v1 = _vec_sq(geom, b["x"] - a["x"], b["y"] - a["y"]) / g1
                ok = velocity_gate(geom, v1, opts) & ~store.used_mask(sb - g1) & (a["sign"] == b["sign"])
                for ai in np.nonzero(ok)[0]:
                    A_list.append((g1, a[ai], v1[ai]))
            if not A_list:
                continue
            C_list = []
            for g2, fc in after:
                c = fc.cands
                v2 = _vec_sq(geom, c["x"] - b["x"], c["y"] - b["y"]) / g2
                ok = velocity_gate(geom, v2, opts) & ~store.used_mask(sb + g2) & (c["sign"] == b["sign"])
                for ci in np.nonzero(ok)[0]:
                    C_list.append((g2, c[ci], v2[ci]))
            for g1, a, v1 in A_list:
                n1 = math.hypot(v1[0], v1[1])
                for g2, c, v2 in C_list:
                    if g1 + g2 > 4:
                        continue
                    tol = max(1.5, 0.25 * n1) * (g1 + g2) / 2.0
                    if math.hypot(v2[0] - v1[0], v2[1] - v1[1]) > tol:
                        continue
                    if v1[0] * v2[0] + v1[1] * v2[1] <= 0:
                        continue
                    zsum = float(a["z"] + b["z"] + c["z"])
                    if zsum < 3.0 * opts.z_strong:
                        continue
                    score = zsum - 0.5 * (g1 + g2 - 2)
                    seeds.append((score, a, b, c))
    seeds.sort(key=lambda r: -r[0])
    return seeds


def _pt_from(c, src: str = "det") -> Pt:
    return Pt(s=int(c["s"]), t=float(c["t"]), k=int(c["k"]), x=float(c["x"]), y=float(c["y"]),
              z=float(c["z"]), attached=bool(c["attached"]), sigma=float(c["sigma"]), uid=int(c["uid"]),
              length=float(c["length"]), src=src, sign=int(c["sign"]) or 1)


def track_sign(tr: "Track") -> int:
    """Majority contrast sign of the track's detections."""
    v = sum(p.sign * max(p.z, 0.0) for p in tr.pts.values() if p.src != "manual")
    return 1 if v >= 0 else -1


def _in_blob(df: DetFrame, p: np.ndarray, pad: float = 4.0) -> bool:
    for (x0, y0, x1, y1) in df.blobs:
        if x0 - pad <= p[0] <= x1 + pad and y0 - pad <= p[1] <= y1 + pad:
            return True
    return False


# -------------------------------------------------------------- extension ----
def extend(tr: Track, store: CandStore, direction: int, geom: ViewGeometry, opts: FinderOptions,
           s_data_end: int, s_data_start: int, sample_dt: float, s_min_allowed: Optional[int] = None,
           s_max_allowed: Optional[int] = None) -> str:
    """Grow the track one sample at a time.  Returns 'stopped' | 'data_end'."""
    miss = 0
    P = tr.sorted()
    sign = track_sign(tr)
    s = (P[-1].s + 1) if direction > 0 else (P[0].s - 1)
    while True:
        if direction > 0 and s > s_data_end:
            return "data_end"
        if direction < 0 and s < s_data_start:
            return "stopped"
        if s_min_allowed is not None and s < s_min_allowed:
            return "stopped"
        if s_max_allowed is not None and s > s_max_allowed:
            return "stopped"
        df = store.get(s)
        P = tr.sorted()
        ts_ = np.array([p.t for p in P])
        xs = np.array([p.x for p in P])
        ys = np.array([p.y for p in P])
        t_s = df.t if df is not None else (P[-1].t + (s - P[-1].s) * sample_dt if direction > 0
                                           else P[0].t + (s - P[0].s) * sample_dt)
        pred, sig_p, vel = local_predict(ts_, xs, ys, t_s, max_pts=8)
        prog = float(geom.progress(pred)[0])
        if prog < opts.progress_back_limit or prog > 1.15 or not geom.in_corridor(pred, margin=10.0)[0]:
            return "stopped"
        near = P[-1] if direction > 0 else P[0]
        gap = abs(s - near.s)
        good_model = tr.model is not None and tr.model.rms <= 1.5 and tr.n_inliers >= 8
        if df is None or df.skipped:
            miss += 1
        else:
            v_px = float(np.hypot(vel[0] * geom.sar, vel[1])) * sample_dt
            d = float(geom.ball_diameter(prog))
            r = max(2.0, d / 2 + 1.0) + 0.15 * v_px * gap + 2.0 * sig_p
            r = min(r, 6.0 + 0.6 * v_px + d)
            C = df.cands
            hit = None
            if len(C):
                dxs = (C["x"] - pred[0]) * geom.sar
                dys = C["y"] - pred[1]
                dist = np.hypot(dxs, dys)
                ok = (dist <= r) & ~store.used_mask(s) & (C["sign"] == sign)
                if ok.any():
                    score = C["z"] - 2.0 * (dist / r) ** 2
                    score[~ok] = -np.inf
                    hit = int(np.argmax(score))
            if hit is not None:
                tr.pts[s] = _pt_from(C[hit])
                miss = 0
            else:
                miss += 1
            limit = opts.miss_limit
            if _in_blob(df, pred):
                limit = opts.miss_limit_body
            if good_model:
                limit = max(limit, opts.merge_gap)
            if miss > limit:
                return "stopped"
        if df is None and miss > opts.merge_gap:
            return "stopped"
        s += direction


# ---------------------------------------------------------------- fitting ----
def fit_track(tr: Track, geom: ViewGeometry, opts: FinderOptions) -> Optional[TrajModel]:
    A = tr.arrays()
    if len(A["t"]) < 3:
        tr.model = None
        return None
    d = float(np.median(geom.ball_diameter(geom.progress(np.stack([A["x"], A["y"]], 1)))))
    tol_x = max(1.5, 0.3 * d)
    tol_y = max(2.0, 0.3 * d)
    fixed = np.array([s == "manual" for s in A["src"]])
    m = fit_model(A["t"], A["x"], A["y"], A["sigma"], kind="auto", view=geom.view, tol_x=tol_x, tol_y=tol_y,
                  ransac_iters=opts.ransac_iters, fixed=fixed if fixed.any() else None)
    tr.model = m
    if m is not None:
        for p, inl in zip(tr.sorted(), m.inliers):
            p.inlier = bool(inl)
    return m


def build_track(tid: int, seed, store: CandStore, geom: ViewGeometry, opts: FinderOptions,
                sample_dt: float, s_data_start: int, s_data_end: int,
                s_min_allowed: Optional[int] = None, s_max_allowed: Optional[int] = None) -> Track:
    score, a, b, c = seed
    tr = Track(tid=tid, seed_score=score)
    for cand in (a, b, c):
        tr.pts[int(cand["s"])] = _pt_from(cand)
    st_b = extend(tr, store, -1, geom, opts, s_data_end, s_data_start, sample_dt, s_min_allowed, s_max_allowed)
    st_f = extend(tr, store, +1, geom, opts, s_data_end, s_data_start, sample_dt, s_min_allowed, s_max_allowed)
    tr.open_fwd = st_f == "data_end"
    if len(tr.pts) >= 5 or tr.open_fwd:
        fit_track(tr, geom, opts)
    return tr


def continue_track(tr: Track, store: CandStore, geom: ViewGeometry, opts: FinderOptions, sample_dt: float,
                   s_data_start: int, s_data_end: int) -> None:
    """Extend an open track forward with newly available samples."""
    st_f = extend(tr, store, +1, geom, opts, s_data_end, s_data_start, sample_dt)
    tr.open_fwd = st_f == "data_end"
    fit_track(tr, geom, opts)


# ----------------------------------------------------------------- pass 2 ----
def _zmap_peak(df: DetFrame, pred: np.ndarray, r: float, sign: int, row_scale: int, sar: float):
    """Best |z| (with the expected sign) within radius r of pred in df's z map.

    Returns (z, ref_xy) or None.
    """
    if df.zmap is None:
        return None
    g = df.ref_to_grid(np.atleast_2d(pred), row_scale)[0]
    h, w = df.zmap.shape
    rx, ry = r / sar, max(1.0, r / row_scale)
    j0, j1 = int(max(1, math.floor(g[0] - rx))), int(min(w - 1, math.ceil(g[0] + rx) + 1))
    i0, i1 = int(max(1, math.floor(g[1] - ry))), int(min(h - 1, math.ceil(g[1] + ry) + 1))
    if j1 <= j0 or i1 <= i0:
        return None
    win = df.zmap[i0:i1, j0:j1].astype(np.float32) / 100.0
    if sign != 0:
        win = win * sign
    else:
        win = np.abs(win)
    ii, jj = np.mgrid[i0:i1, j0:j1]
    inside = ((jj - g[0]) * sar) ** 2 + ((ii - g[1]) * row_scale) ** 2 <= r * r
    win = np.where(inside, win, -np.inf)
    k = int(np.argmax(win))
    zi, zj = np.unravel_index(k, win.shape)
    z = float(win[zi, zj])
    if not np.isfinite(z):
        return None
    ci, cj = zi + i0, zj + j0
    full = df.zmap.astype(np.float32) / 100.0 * (sign if sign else 1)
    # must be a local maximum of the full map (not the slope of something outside the gate)
    nb = full[max(0, ci - 1):ci + 2, max(0, cj - 1):cj + 2]
    if z < float(nb.max()) - 1e-6:
        return None
    # parabolic sub-pixel refinement
    def sub(a, b, c):
        den = a - 2 * b + c
        return 0.0 if abs(den) < 1e-9 else float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    dj = sub(full[ci, cj - 1], full[ci, cj], full[ci, cj + 1]) if 0 < cj < w - 1 else 0.0
    di = sub(full[ci - 1, cj], full[ci, cj], full[ci + 1, cj]) if 0 < ci < h - 1 else 0.0
    ref = df.grid_to_ref([cj + dj], [ci + di], row_scale)[0]
    return z, ref


def pass2(tr: Track, store: CandStore, geom: ViewGeometry, opts: FinderOptions, row_scale: int,
          clutter_factor=None, s_min_allowed: Optional[int] = None, s_max_allowed: Optional[int] = None) -> int:
    """Weak re-detection (z >= z_pass2) inside the model gate; returns points added."""
    m = tr.model
    if m is None or tr.n_inliers < 6:
        return 0
    A = tr.arrays(inliers_only=True)
    sign = track_sign(tr)
    s_first, s_last = int(A["s"][0]), int(A["s"][-1])
    todo = []
    # gaps inside
    have = set(int(s) for s in A["s"])
    todo += [s for s in range(s_first, s_last) if s not in have]
    todo_back = list(range(s_first - 1, s_first - 1 - opts.pass2_back, -1))
    todo_fwd = list(range(s_last + 1, s_last + 1 + opts.pass2_fwd))
    added = 0

    def try_s(s) -> bool:
        nonlocal added
        if s_min_allowed is not None and s < s_min_allowed:
            return False
        if s_max_allowed is not None and s > s_max_allowed:
            return False
        df = store.get(s)
        if df is None or df.zmap is None or df.skipped:
            return False
        pred = m.eval([df.t])[0]
        prog = float(geom.progress(pred)[0])
        if prog < max(-0.05, opts.progress_back_limit) or prog > 1.2:
            return False
        sp = float(m.pos_sigma([df.t])[0])
        if sp > 6.0:
            return False
        d = float(geom.ball_diameter(prog))
        r = max(1.5, d / 2 + 0.5) + 2.0 * min(sp, 3.0)
        res = _zmap_peak(df, pred, r, sign, row_scale, geom.sar)
        if res is None:
            return False
        z, ref = res
        if clutter_factor is not None:
            z *= float(clutter_factor(np.array([ref[0]]), np.array([ref[1]]))[0])
        if z < opts.z_pass2:
            return False
        # an existing candidate at that place keeps its attributes
        best = None
        if len(df.cands):
            dd = np.hypot((df.cands["x"] - ref[0]) * geom.sar, df.cands["y"] - ref[1])
            j = int(np.argmin(dd))
            if dd[j] <= 1.5 and int(df.cands["sign"][j]) == sign:
                best = df.cands[j]
        if best is not None and not store.is_used(s, int(best["uid"])):
            p = _pt_from(best, "pass2")
        else:
            att = _in_blob(df, ref, pad=0.0)
            p = Pt(s=s, t=df.t, k=df.k, x=float(ref[0]), y=float(ref[1]), z=z, attached=att,
                   sigma=max(0.6, 3.0 / max(z, 1e-3)), src="pass2", sign=sign)
        tr.pts[s] = p
        added += 1
        return True

    for s in todo:
        try_s(s)
    misses = 0
    for s in todo_back:
        if try_s(s):
            misses = 0
        else:
            misses += 1
            if misses > 2:
                break
    misses = 0
    for s in todo_fwd:
        if try_s(s):
            misses = 0
        else:
            misses += 1
            if misses > 1:
                break
    return added


# ------------------------------------------------------------------ merge ----
def try_merge(a: Track, b: Track, geom: ViewGeometry, opts: FinderOptions) -> Optional[Track]:
    """Merge two fragments of one flight if a joint fit explains both."""
    if a.model is None or b.model is None:
        return None
    lo = max(a.s_first, b.s_first)
    hi = min(a.s_last, b.s_last)
    gap = lo - hi
    if gap > opts.merge_gap:
        return None
    Aa, Ab = a.arrays(True), b.arrays(True)
    if len(Aa["t"]) < 3 or len(Ab["t"]) < 3:
        return None
    # quick reject: the models must roughly agree where the tracks meet
    t_meet = 0.5 * (min(Aa["t"][-1], Ab["t"][-1]) + max(Aa["t"][0], Ab["t"][0]))
    qa, qb = a.model.eval([t_meet])[0], b.model.eval([t_meet])[0]
    if np.hypot((qa[0] - qb[0]) * geom.sar, qa[1] - qb[1]) > 10.0 + 2.0 * max(0, gap):
        return None
    # each model predicts the other's nearest points
    pa = a.model.eval(Ab["t"])
    pb = b.model.eval(Aa["t"])
    da = np.hypot((pa[:, 0] - Ab["x"]) * geom.sar, pa[:, 1] - Ab["y"])
    db = np.hypot((pb[:, 0] - Aa["x"]) * geom.sar, pb[:, 1] - Aa["y"])
    lim = opts.merge_dist_px + 0.2 * max(0, gap)
    if min(np.median(np.sort(da)[:3]), np.median(np.sort(db)[:3])) > lim * 3:
        return None
    m = Track(tid=min(a.tid, b.tid), seed_score=max(a.seed_score, b.seed_score))
    for p in a.sorted() + b.sorted():
        q = m.pts.get(p.s)
        if q is None or p.z > q.z:
            m.pts[p.s] = Pt(**{**p.__dict__})
    fit_track(m, geom, opts)
    if m.model is None:
        return None
    n_in = m.n_inliers
    if n_in < 0.8 * (a.n_inliers + b.n_inliers) - 1 or m.model.rms > opts.max_rms_px:
        return None
    m.open_fwd = a.open_fwd or b.open_fwd
    return m


# ------------------------------------------------------------ field order ----
def _poly_rms(t, x, y, deg):
    tc = (t - t.mean()) / (np.ptp(t) or 1.0)
    ex = x - np.polyval(np.polyfit(tc, x, deg), tc)
    ey = y - np.polyval(np.polyfit(tc, y, deg), tc)
    return float(np.sqrt(np.mean(ex ** 2 + ey ** 2))), ex, ey


def _alternation(ex, ey) -> float:
    if len(ex) < 4:
        return float("nan")
    num = float(np.mean(ex[:-1] * ex[1:] + ey[:-1] * ey[1:]))
    den = float(np.mean(ex ** 2 + ey ** 2))
    return float(np.clip(num / den, -1, 1)) if den > 0 else 0.0


def field_order_check(t, x, y, k, frame_dt: float) -> dict:
    """Is the track smoother with the opposite field order?

    Fits a cubic (quadratic for < 10 points) under the given times and under
    t' = t + (1 - 2k) frame_dt / 2.  'conflict' when the flipped RMS is less
    than half the given RMS and the given RMS is > 0.5 px.
    """
    t = np.asarray(t, float)
    n = len(t)
    out = {"n": n, "preferred": "unclear", "conflict": False}
    if n < 8:
        return out
    x, y, k = np.asarray(x, float), np.asarray(y, float), np.asarray(k)
    o = np.argsort(t)
    t, x, y, k = t[o], x[o], y[o], k[o]
    deg = 2 if n < 10 else 3
    tf = t + (1 - 2 * k) * (frame_dt / 2.0)
    of = np.argsort(tf)
    rg, exg, eyg = _poly_rms(t, x, y, deg)
    rf, exf, eyf = _poly_rms(tf[of], x[of], y[of], deg)
    out.update(rms_given=rg, rms_flipped=rf, alt_given=_alternation(exg, eyg), alt_flipped=_alternation(exf, eyf))
    if rf < 0.5 * rg and rg > 0.5:
        out["preferred"] = "flipped"
        out["conflict"] = True
    elif rg < 0.5 * rf:
        out["preferred"] = "given"
    return out
