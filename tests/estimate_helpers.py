"""Simulation helpers for validating pitchspeed.estimate (ported from the math study).

* `simulate_pitches`: batched 3-D RK4 flights with drag, gravity and Magnus
  (random extension, release side/height, Cd, spin, target).
* `Cam`: pinhole camera with autozoom (square pixels).
* scenario builders that turn a flight into what the vision module would
  hand to the estimator: a side- or behind-view `Calibration` with noisy
  clicks, stabilized `TrackPoint`s with pixel noise at 59.94 Hz with a
  random phase, and release / catch `EventTime`s with the event-time
  noise models of math.md §2.
* `run_*` functions returning error statistics (bias, RMS, coverage).

World frame (constants.py): origin at the plate apex, +y toward the
pitcher, +x toward first base, +z up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pitchspeed import constants as C
from pitchspeed.calib.model import Calibration, Segment, Settings
from pitchspeed.estimate import estimate_speed, manual_event
from pitchspeed.types import EventTime, PitchTrack, ScanInfo, TrackPoint

FIELD_DT = C.NTSC_FIELD_DT
FRAME_DT = C.NTSC_FRAME_DT
G = C.G_FTPS2
FT = 0.3048
M_KG, DIAM_M = 0.145, 0.0737
AREA = math.pi * (DIAM_M / 2) ** 2
R_FT = DIAM_M / 2 / FT


def k_of_cd(cd, rho_air=1.20):
    return rho_air * np.asarray(cd) * AREA / (2 * M_KG) * FT


def _cl(S):
    return np.where(S < 0.1, 1.5 * S, 0.09 + 0.6 * S)


# ------------------------------------------------------------------ flight
@dataclass
class Flight:
    t: np.ndarray              # dense times from release (s)
    P: np.ndarray              # (n, 3) positions (ft)
    v0: float                  # true 3-D release speed (ft/s)
    ext: float                 # true extension (ft)
    catch_behind: float        # true catch point behind the apex (ft)
    cd: float
    D: float
    x_release: float

    @property
    def T(self) -> float:
        return float(self.t[-1])

    def pos(self, tau):
        tau = np.atleast_1d(tau)
        return np.stack([np.interp(tau, self.t, self.P[:, j]) for j in range(3)], 1)


def _integrate(p, v, w_hat, w_mag, k, kl, y_stop, dt=1e-3, t_max=3.0):
    """Batched RK4 until every ball crosses its y_stop.  Returns list of (t, P)."""
    n = len(p)

    def acc(p, v):
        sp = np.linalg.norm(v, axis=1, keepdims=True)
        a = -k[:, None] * sp * v
        a[:, 2] -= G
        S = R_FT * w_mag[:, None] / sp
        a += kl[:, None] * _cl(S) * sp * np.cross(w_hat, v)
        return a

    Ps, Ts = [p.copy()], [0.0]
    t = 0.0
    done = np.zeros(n, bool)
    while t < t_max and not done.all():
        k1v = acc(p, v); k1p = v
        k2v = acc(p + dt / 2 * k1p, v + dt / 2 * k1v); k2p = v + dt / 2 * k1v
        k3v = acc(p + dt / 2 * k2p, v + dt / 2 * k2v); k3p = v + dt / 2 * k2v
        k4v = acc(p + dt * k3p, v + dt * k3v); k4p = v + dt * k3v
        p = p + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        t += dt
        Ps.append(p.copy()); Ts.append(t)
        done |= p[:, 1] <= y_stop
    Ps = np.array(Ps)
    Ts = np.array(Ts)
    out = []
    for i in range(n):
        y = Ps[:, i, 1]
        j = int(np.argmax(y <= y_stop[i]))
        f = (y[j - 1] - y_stop[i]) / (y[j - 1] - y[j])
        tt = np.r_[Ts[:j], Ts[j - 1] + f * dt]
        PP = np.vstack([Ps[:j, i], Ps[j - 1, i] + f * (Ps[j, i] - Ps[j - 1, i])])
        out.append((tt, PP))
    return out


PITCH_TYPES = {
    # name: (mph, D, release height mean, spin rpm, spin axes (x, y, z) choices)
    "90FB": (90.0, 60.5, 6.0, 2200, [(-1, 0, 0), (-1, 0, 0.3)]),
    "75FB": (75.0, 60.5, 5.8, 1900, [(-1, 0, 0), (-1, 0, 0.3)]),
    "75CB": (75.0, 60.5, 5.8, 2400, [(1, 0, 0), (1, 0, 0.5)]),
    "60FB": (60.0, 60.5, 5.5, 1500, [(-1, 0, 0)]),
    "45FB46": (45.0, 46.0, 4.5, 1100, [(-1, 0, 0), (1, 0, 0)]),
    "55FB46": (55.0, 46.0, 4.8, 1300, [(-1, 0, 0), (-1, 0, 0.3)]),
}


def simulate_pitches(kind: str, n: int, rng: np.random.Generator, *, ext_sigma: float = 1.0,
                     catch_sigma: float = 0.75, cd_sigma: Optional[float] = None,
                     hand: Optional[str] = None) -> list[Flight]:
    """n random pitches of one type.  Extension / catch point / Cd are drawn
    from the same priors the estimator assumes (honest coverage test)."""
    mph, D, h_mean, spin, axes = PITCH_TYPES[kind]
    ext_mean = C.default_extension_ft(D)
    if cd_sigma is None:
        cd_sigma = 0.05 if mph >= 65 else 0.07
    ext = np.clip(ext_mean + rng.normal(0, ext_sigma, n), 1.5, 8.5)
    cb = np.clip(C.CATCH_BEHIND_FT + rng.normal(0, catch_sigma, n), -0.5, 4.0)
    cd = np.clip(rng.normal(0.33, cd_sigma, n), 0.2, 0.55)
    if hand is None:
        sgn = rng.choice([-1.0, 1.0], n)
    else:
        sgn = np.full(n, -1.0 if hand == "R" else 1.0)     # RHP releases at x < 0
    xr = sgn * rng.uniform(1.0, 2.5, n)
    zr = h_mean + rng.normal(0, 0.4, n)
    v0 = mph * C.MPH_TO_FTPS * (1 + rng.normal(0, 0.03, n))
    tx = rng.normal(0, 0.7, n)
    tz = 2.5 + rng.normal(0, 0.6, n)
    axis = np.array([axes[i] for i in rng.integers(len(axes), size=n)], float)
    w_hat = axis / np.linalg.norm(axis, axis=1, keepdims=True)
    w_mag = np.full(n, spin * 2 * math.pi / 60)
    k = k_of_cd(cd)
    kl = 1.20 * AREA / (2 * M_KG) * FT * np.ones(n)
    p0 = np.stack([xr, D - ext, zr], 1)
    y_stop = -cb
    ang_v = np.arctan2(tz - zr, p0[:, 1] - y_stop)
    ang_h = np.arctan2(tx - xr, p0[:, 1] - y_stop)
    for _ in range(4):                        # aim at (tx, tz) on the catch plane
        d = np.stack([np.sin(ang_h), -np.ones(n), np.tan(ang_v)], 1)
        v = d / np.linalg.norm(d, axis=1, keepdims=True) * v0[:, None]
        res = _integrate(p0, v, w_hat, w_mag, k, kl, y_stop, dt=2e-3)
        end = np.array([r[1][-1] for r in res])
        L = p0[:, 1] - y_stop
        ang_v += np.arctan((tz - end[:, 2]) / L)
        ang_h += np.arctan((tx - end[:, 0]) / L)
    d = np.stack([np.sin(ang_h), -np.ones(n), np.tan(ang_v)], 1)
    v = d / np.linalg.norm(d, axis=1, keepdims=True) * v0[:, None]
    res = _integrate(p0, v, w_hat, w_mag, k, kl, y_stop, dt=1e-3)
    return [Flight(t=tt, P=PP, v0=float(v0[i]), ext=float(ext[i]), catch_behind=float(cb[i]),
                   cd=float(cd[i]), D=D, x_release=float(xr[i])) for i, (tt, PP) in enumerate(res)]


# ------------------------------------------------------------------ camera
class Cam:
    """Pinhole camera, square pixels; image (W, H) with the principal point at the centre."""

    def __init__(self, C_, target, roll_deg=0.0, f=1000.0, W=720.0, H=480.0):
        self.C = np.asarray(C_, float)
        fwd = np.asarray(target, float) - self.C
        fwd /= np.linalg.norm(fwd)
        up = np.array([0, 0, 1.0])
        right = np.cross(fwd, up); right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        r = math.radians(roll_deg)
        self.R = np.stack([math.cos(r) * right + math.sin(r) * down,
                           -math.sin(r) * right + math.cos(r) * down, fwd])
        self.f, self.W, self.H = f, W, H
        self.cx, self.cy = W / 2, H / 2

    def cam(self, P):
        return (np.atleast_2d(P) - self.C) @ self.R.T

    def proj(self, P):
        q = self.cam(P)
        return np.stack([self.cx + self.f * q[:, 0] / q[:, 2], self.cy + self.f * q[:, 1] / q[:, 2]], 1)

    def depth(self, P):
        return self.cam(P)[:, 2]

    def autozoom(self, pts, margin=0.10):
        q = self.cam(pts)
        x, y = q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]
        self.f = min((1 - 2 * margin) * self.W / 2 / np.max(np.abs(x)),
                     (1 - 2 * margin) * self.H / 2 / np.max(np.abs(y)))
        return self.f

    @property
    def tan_phi(self) -> float:
        fwd = self.R[2]
        s = abs(fwd[1]) / np.linalg.norm(fwd[:2])
        return s / math.sqrt(max(1 - s * s, 1e-9))


SIDE_SCENES = {   # camera position (x off line, y along line, z) for D = 60.5 (scaled for youth)
    "perp90": (90.0, 30.0, 5.0),
    "perp150": (150.0, 30.0, 6.0),
    "elev15ft": (90.0, 10.0, 15.0),
    "mound-side": (60.0, 85.0, 5.0),
    "far-oblique": (120.0, -40.0, 6.0),
    "deep-oblique": (40.0, -60.0, 6.0),
    "perp90-3B": (-90.0, 30.0, 5.0),
}


def side_camera(scene: str, D: float, sar: float = 1.0, roll_deg: float = 0.0) -> Cam:
    pos = np.array(SIDE_SCENES[scene], float) * (D / 60.5 if D < 60 else 1.0)
    cam = Cam(pos, (0, D / 2, 3.0), roll_deg, W=720.0 * sar, H=480.0)
    cam.autozoom(np.array([[0, -4, 0], [0, D + 2, 0], [0, -4, 7], [0, D + 2, 7], [2.5, D - 4, 6],
                           [-2.5, D - 4, 6], [-2, -2, 0], [2, -2, 0]], float))
    return cam


def behind_camera(dist: float, height: float, D: float) -> Cam:
    cam = Cam((0.5, -dist, height), (0, D * 0.55, 3.0))
    cam.autozoom(np.array([[-4, D - 6, 0], [4, D - 6, 7], [-4, D - 6, 7], [4, D - 6, 0], [0, -1.5, 2.0],
                           [0, -1.5, 3.5], [-1, 0, 0], [1, 0, 0]], float), margin=0.06)
    return cam


def _to_stored(xy_sq: np.ndarray, sar: float) -> np.ndarray:
    xy = np.array(xy_sq, float, copy=True).reshape(-1, 2)
    xy[:, 0] /= sar
    return xy


def make_settings(D: float, **kw) -> Settings:
    level = {60.5: "hs", 46.0: "ll"}.get(D)
    return Settings(distance_ft=D, level=level, **kw)


def side_calibration(cam: Cam, D: float, rng, sar: float = 1.0, click_sigma: float = 2.0,
                     settings: Optional[Settings] = None, rubber_ends: bool = False,
                     vertical: bool = False) -> Calibration:
    st = settings or make_settings(D)
    st.sar = sar
    zR = 10 / 12 if D > 55 else 0.5
    world = {"plate": (0, C.PLATE_CENTER_Y_FT, 0.0), "rubber": (0, D + C.RUBBER_CENTER_OFFSET_FT, zR)}
    if rubber_ends:
        world["rubber_left"] = (-1.0, D + 0.25, zR)
        world["rubber_right"] = (1.0, D + 0.25, zR)
    if vertical:
        world["vertical_top"] = (25.0, D / 2, 12.0)
        world["vertical_bottom"] = (25.0, D / 2, 0.0)
    pts = {}
    for name, P in world.items():          # click noise is in stored pixels
        xy = _to_stored(cam.proj(np.array([P], float)), sar)[0] + rng.normal(0, click_sigma, 2)
        pts[name] = xy.tolist()
    if rubber_ends:   # 'left' / 'right' as seen in the image
        if pts["rubber_left"][0] > pts["rubber_right"][0]:
            pts["rubber_left"], pts["rubber_right"] = pts["rubber_right"], pts["rubber_left"]
    return Calibration(st, Segment(view="side", points=pts), (720, 480))


def behind_calibration(cam: Cam, D: float, rng, settings: Optional[Settings] = None) -> Calibration:
    st = settings or make_settings(D)
    ext = C.default_extension_ft(D)
    rel = cam.proj(np.array([[-1.5, D - ext, 5.5]]))[0] + rng.normal(0, 2, 2)
    cat = cam.proj(np.array([[0, -1.5, 2.5]]))[0] + rng.normal(0, 2, 2)
    return Calibration(st, Segment(view="behind", points={"release": rel.tolist(), "catch": cat.tolist()}),
                       (720, 480))


# ---------------------------------------------------------- vision outputs
@dataclass
class Shot:
    """One simulated pitch as the estimator sees it, plus the truth."""
    flight: Flight
    t_release: float           # true release time in video time
    track: Optional[PitchTrack]
    release: Optional[EventTime]
    catch: Optional[EventTime]
    truth_mph: float


def sample_times(t_rel: float, T: float, dt: float, t_from: float, t_to: float):
    """Indices s and times s*dt of samples with t_rel + t_from < t < t_rel + T + t_to."""
    s0 = math.floor((t_rel + t_from) / dt) + 1
    s1 = math.ceil((t_rel + T + t_to) / dt) - 1
    s = np.arange(s0, s1 + 1)
    return s, s * dt


def make_track(fl: Flight, cam: Cam, t_rel: float, rng, *, dt=FIELD_DT, sar=1.0, sigma_px=1.0,
               visible_frac=1.0, outlier_frac=0.0, parity_bias_px=0.0, swap_order=False) -> PitchTrack:
    t_from = max(2 * dt, (1 - visible_frac) * fl.T)
    s, ts = sample_times(t_rel, fl.T, dt, t_from, -dt)
    true_t = ts.copy()
    if swap_order:                 # images exposed in the opposite order inside each frame
        true_t = ts + (1 - 2 * np.mod(s, 2)) * dt
    tau = np.clip(true_t - t_rel, 0, fl.T)
    xy = cam.proj(fl.pos(tau)) + rng.normal(0, sigma_px, (len(s), 2))
    xy[:, 1] += parity_bias_px * (1 - 2 * np.mod(s, 2))
    n_out = int(round(outlier_frac * len(s)))
    if n_out:
        idx = rng.choice(len(s), n_out, replace=False)
        xy[idx] += rng.normal(0, 15, (n_out, 2))
    xy = _to_stored(xy, sar)
    pts = [TrackPoint(s=int(si), t=float(ti), x=float(x), y=float(y), score=8.0, sigma=sigma_px)
           for si, ti, (x, y) in zip(s, ts, xy)]
    return PitchTrack(track_id=0, points=pts)


def side_release_event(fl: Flight, cam: Cam, calib: Calibration, t_rel: float, rng, sar: float,
                       model_sigma: float = 0.4 * FIELD_DT) -> EventTime:
    """Side-view 'extrapolated' release as the vision module computes it (cv.md §5):
    the time the ball's image crosses the calibration release point along the
    release->catch progress coordinate (extrapolated linearly before the
    true release), plus model noise.  Its error therefore includes the
    sideways-offset / perspective effects of an image-space extrapolation.
    sigma per cv.md: sqrt(sigma_model^2 + (1 ft / speed)^2)."""
    tau = np.linspace(0.0, min(fl.T, 0.25), 200)
    xy = _to_stored(cam.proj(fl.pos(tau)), sar)
    prog = calib.progress(xy)
    if prog[0] >= 0:                    # ball already past the point at release: extrapolate back
        slope = (prog[1] - prog[0]) / (tau[1] - tau[0])
        t_cross = -prog[0] / slope
    else:
        t_cross = float(np.interp(0.0, prog, tau))
    return EventTime(t_rel + t_cross + rng.normal(0, model_sigma),
                     math.hypot(model_sigma, 1.0 / fl.v0), "extrapolated")


def release_first_field(t_rel: float, dt: float, rng, latency_probs=(1.0,)) -> EventTime:
    """'first_free_field' release: first sample after release (+ missed fields),
    minus the latency prior (its mean), sigma = its spread."""
    m = rng.choice(len(latency_probs), p=np.asarray(latency_probs) / np.sum(latency_probs))
    s_first = math.floor(t_rel / dt) + 1 + m
    lp = np.asarray(latency_probs, float) / np.sum(latency_probs)
    mean_m = float(np.sum(np.arange(len(lp)) * lp))
    var_m = float(np.sum(np.arange(len(lp)) ** 2 * lp)) - mean_m ** 2
    t = s_first * dt - (0.5 + mean_m) * dt
    return EventTime(t=t, sigma=dt * math.sqrt(1 / 12 + var_m), method="first_free_field", s=int(s_first))


def catch_extrapolated(t_catch: float, rng, bias=0.001, sigma=0.003, reported=0.0035) -> EventTime:
    return EventTime(t=t_catch + bias + rng.normal(0, sigma), sigma=reported, method="extrapolated")


# Simulated non-expert marking: each mark is one field early or late with this
# probability (the estimator's ASSUMED identification term, estimate.events).
MANUAL_SLIP_PROB = 0.25


def _slip(rng, p: float) -> int:
    if rng is None or p <= 0 or rng.uniform() >= p:
        return 0
    return 1 if rng.uniform() < 0.5 else -1


def manual_pair(t_rel: float, t_catch: float, dt: float, rng=None, slip_prob: float = 0.0):
    """User marks: first sample after release, first sample at/after the catch,
    each one field off with probability `slip_prob` (needs `rng`)."""
    s_r = math.floor(t_rel / dt) + 1 + _slip(rng, slip_prob)
    s_c = math.ceil(t_catch / dt) + _slip(rng, slip_prob)
    return (manual_event(s_r * dt, dt, "release", s=s_r), manual_event(s_c * dt, dt, "catch", s=s_c))


# --------------------------------------------------------------- statistics
@dataclass
class Stats:
    n: int = 0
    errs: list = field(default_factory=list)      # % errors of reported numbers
    rel_sig: list = field(default_factory=list)   # reported relative sigma (%)
    cover: list = field(default_factory=list)
    labels: dict = field(default_factory=dict)
    methods: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)
    n_none: int = 0

    def add(self, est, truth_mph):
        self.n += 1
        self.labels[est.label] = self.labels.get(est.label, 0) + 1
        self.methods[est.method] = self.methods.get(est.method, 0) + 1
        for f in est.flags:
            self.flags[f] = self.flags.get(f, 0) + 1
        if est.v_release_mph is None:
            self.n_none += 1
            return
        e = 100 * (est.v_release_mph / truth_mph - 1)
        self.errs.append(e)
        self.rel_sig.append(100 * est.sigma_mph / est.v_release_mph)
        lo, hi = est.ci90_mph
        self.cover.append(lo <= truth_mph <= hi)

    @property
    def bias(self):
        return float(np.mean(self.errs)) if self.errs else float("nan")

    @property
    def rms(self):
        return float(np.sqrt(np.mean(np.square(self.errs)))) if self.errs else float("nan")

    @property
    def coverage(self):
        return float(np.mean(self.cover)) if self.cover else float("nan")

    @property
    def mean_sigma(self):
        return float(np.mean(self.rel_sig)) if self.rel_sig else float("nan")

    def line(self, name: str) -> str:
        return (f"{name:28s} n={self.n:4d} reported={len(self.errs):4d} bias {self.bias:+5.2f}% "
                f"RMS {self.rms:4.2f}% mean sigma {self.mean_sigma:4.2f}% cover90 {self.coverage:.3f} "
                f"labels {self.labels} methods {self.methods}")


INTERLACED = ScanInfo("interlaced", "bff", "high", "content", 1 / FIELD_DT)
PROGRESSIVE = ScanInfo("progressive", None, "high", "content", 1 / FRAME_DT)


def run_side(scene: str, kind: str, n: int, seed: int = 0, *, sar: float = 10 / 11, sigma_px=1.0,
             visible_frac=1.0, outlier_frac=0.0, manual=False, hand_known=False, rubber_ends=False,
             parity_bias_px=0.0, swap_order=False, click_sigma=2.0, roll_deg=0.0,
             slip_prob=MANUAL_SLIP_PROB) -> Stats:
    rng = np.random.default_rng(seed)
    hand = rng.choice(["R", "L"]) if hand_known else None
    flights = simulate_pitches(kind, n, rng, hand=hand)
    D = flights[0].D
    cam = side_camera(scene, D, sar, roll_deg)
    st = Stats()
    for fl in flights:
        settings = make_settings(D, pitcher_throws=hand)
        calib = side_calibration(cam, D, rng, sar, click_sigma, settings, rubber_ends=rubber_ends)
        t_rel = 1.0 + rng.uniform(0, FIELD_DT)
        track = make_track(fl, cam, t_rel, rng, sar=sar, sigma_px=sigma_px, visible_frac=visible_frac,
                           outlier_frac=outlier_frac, parity_bias_px=parity_bias_px, swap_order=swap_order)
        t_c = t_rel + fl.T
        if manual:
            rel, cat = manual_pair(t_rel, t_c, FIELD_DT, rng, slip_prob)
        else:
            rel = side_release_event(fl, cam, calib, t_rel, rng, sar)
            cat = catch_extrapolated(t_c, rng)
        est = estimate_speed(calib, INTERLACED, FIELD_DT, rel, cat, track=track, manual=manual)
        st.add(est, fl.v0 * C.FTPS_TO_MPH)
    return st


BEHIND_SCENARIOS = {
    # name: (release latency probs, user-entered extension sigma or None, frames only)
    "A": ((1.0,), None, False),
    "B": ((0.6, 0.3, 0.1), None, False),
    "C": ((1.0,), 0.5, False),
    "D": ((1.0,), None, True),
    "manual": (None, None, False),
}


def run_behind(scenario: str, kind: str, n: int, seed: int = 0, slip_prob: float = MANUAL_SLIP_PROB) -> Stats:
    rng = np.random.default_rng(seed)
    flights = simulate_pitches(kind, n, rng)
    D = flights[0].D
    lat, ext_sig, frames = BEHIND_SCENARIOS[scenario]
    dt = FRAME_DT if frames else FIELD_DT
    scan = PROGRESSIVE if frames else INTERLACED
    cam = behind_camera(20.0, 6.0, D)
    st = Stats()
    for fl in flights:
        if ext_sig is not None:
            settings = make_settings(D, extension_ft=float(fl.ext + rng.normal(0, ext_sig)),
                                     extension_sigma_ft=ext_sig)
        else:
            settings = make_settings(D)
        calib = behind_calibration(cam, D, rng, settings)
        t_rel = 1.0 + rng.uniform(0, dt)
        t_c = t_rel + fl.T
        if scenario == "manual":
            rel, cat = manual_pair(t_rel, t_c, dt, rng, slip_prob)
        else:
            rel = release_first_field(t_rel, dt, rng, lat)
            cat = catch_extrapolated(t_c, rng, sigma=0.005 if frames else 0.003,
                                     reported=0.0055 if frames else 0.0035)
        est = estimate_speed(calib, scan, dt, rel, cat, track=None, manual=scenario == "manual")
        st.add(est, fl.v0 * C.FTPS_TO_MPH)
    return st
