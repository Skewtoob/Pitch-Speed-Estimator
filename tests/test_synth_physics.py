"""Generator physics (3-D flight, aiming, hand arc) and the ball coverage renderer."""
import math

import numpy as np
import pytest

from pitchspeed.physics import v0_from_tof
from pitchspeed.synth.ball import ball_patch
from pitchspeed.synth.camera import Camera
from pitchspeed.synth.config import MPH, level_of, par_of, resolve_config
from pitchspeed.synth.events import make_events
from pitchspeed.synth.physics import Trajectory, aim, drag_k


def test_drag_constant():
    # rho 0.0749 lb/ft^3, r 1.45 in, m 5.125 oz -> k ~ 0.00182/ft at Cd 0.34 (synth.md section 4)
    assert drag_k(0.34) == pytest.approx(0.00182, rel=0.01)


def test_trajectory_matches_1d_drag_solution():
    k, v0 = 0.002, 130.0
    tr = Trajectory([0, 60, 6], [0, -v0, 0], k, accel=(0, 0, 32.174), t0=1.0, tmax=1.0)   # gravity cancelled
    for t in (0.1, 0.37, 0.6):
        p, v = tr.state(1.0 + t)
        assert p[1] == pytest.approx(60 - math.log1p(k * v0 * t) / k, abs=1e-7)
        assert -v[1] == pytest.approx(v0 / (1 + k * v0 * t), abs=1e-6)
        assert abs(p[2] - 6) < 1e-9


def test_aim_hits_target():
    v = aim([1.0, 55.0, 6.0], 85 * MPH, [-0.4, 2.5], 17 / 12, 0.0019, [5.0, 0, 8.0], [0, 0, 0])
    assert np.linalg.norm(v) == pytest.approx(85 * MPH)
    tr = Trajectory([1.0, 55.0, 6.0], v, 0.0019, [5.0, 0, 8.0], stop_y=0.0)
    p, _ = tr.state(tr.time_at_y(17 / 12))
    assert abs(p[0] + 0.4) < 2e-3 and abs(p[2] - 2.5) < 2e-3


@pytest.mark.parametrize("level", ["hs", "ll"])
def test_e1_inversion_with_true_inputs(level):
    """E1 with exact T and k: 3-D path length gives |err| < 0.5 mph, y-distance underestimates."""
    cfg = resolve_config({"level": level, "duration_s": 400, "pitches": {"count": 25, "first_t": 3.0}})
    lev = level_of(cfg)
    ev = make_events(cfg, lev, Camera(cfg["camera"], lev, 720, 480, par_of(cfg)))
    assert len(ev.pitches) == 25
    lo, hi = cfg["pitches"]["speed_mph"]
    for p in ev.pitches:
        v_true = np.linalg.norm(p.release_vel) / MPH
        assert lo - 1e-6 <= v_true <= hi + 1e-6
        T = p.t_catch - p.t_release
        v_path = v0_from_tof(p.path_len_ft, T, p.k_per_ft) / MPH
        v_y = v0_from_tof(p.release_pos[1] - p.catch_pos[1], T, p.k_per_ft) / MPH
        assert abs(v_path - v_true) < 0.5
        assert -2.0 < v_y - v_true < 0.05
        loss = 1 - np.linalg.norm(p.catch_vel) / np.linalg.norm(p.release_vel)
        assert 0.02 < loss < 0.15


def test_hand_arc_is_continuous_at_release():
    cfg = resolve_config({"seed": 2, "duration_s": 12})
    lev = level_of(cfg)
    p = make_events(cfg, lev, Camera(cfg["camera"], lev, 720, 480, par_of(cfg))).pitches[0]
    eps = 1e-6
    h0 = p.hand.at(p.t_release - eps)[0]
    assert np.allclose(h0, p.release_pos, atol=1e-3)
    vh = (p.hand.at(p.t_release - eps)[0] - p.hand.at(p.t_release - 2 * eps)[0]) / eps
    assert np.allclose(vh, p.release_vel, rtol=1e-3, atol=0.05)
    # hand starts at rest 60 ms earlier
    v_start = (p.hand.at(p.t_hand + 2e-4)[0] - p.hand.at(p.t_hand + 1e-4)[0]) / 1e-4
    assert np.linalg.norm(v_start) < 0.02 * np.linalg.norm(p.release_vel)
    assert p.ball_pos(p.t_catch + 1e-3) is None and p.ball_pos(p.t_hand - 1e-3) is None


def test_ball_patch_flux_and_centroid_static():
    rng = np.random.default_rng(0)
    for r in (0.6, 1.5, 3.0):
        fl, ce = [], []
        for _ in range(60):
            u, v = 50 + rng.random(), 50 + rng.random()
            par = int(rng.integers(0, 2))
            ax, ay = r * 1.1, r
            r0, c0, cov = ball_patch([(u, v, u, v, ax, ay, 1.0)], 200, 100, par, 2, 2.0, 0.7)
            fl.append(cov.sum() * 2.0 / (math.pi * ax * ay))
            rows = (np.arange(cov.shape[0]) + r0) * 2 + par
            cols = np.arange(cov.shape[1]) + c0
            cu = (cov.sum(0) * cols).sum() / cov.sum()
            cv = (cov.sum(1) * rows).sum() / cov.sum()
            ce.append(math.hypot(cu - u, cv - v))
        assert abs(np.mean(fl) - 1) < 0.02 and np.std(fl) < 0.02
        assert np.sqrt(np.mean(np.square(ce))) < 0.06


def test_ball_patch_moving_streak_energy_and_time_split():
    # a streak split into 6 sub-segments has the same coverage as one segment
    one = ball_patch([(30.0, 40.3, 52.0, 44.1, 1.2, 1.1, 1.0)], 100, 60, 1, 2, 2.0, 0.0)
    segs = []
    for i in range(6):
        a, b = i / 6, (i + 1) / 6
        segs.append((30 + 22 * a, 40.3 + 3.8 * a, 30 + 22 * b, 40.3 + 3.8 * b, 1.2, 1.1, 1 / 6))
    six = ball_patch(segs, 100, 60, 1, 2, 2.0, 0.0)
    assert one[0] == six[0] and one[1] == six[1]
    assert np.allclose(one[2], six[2], atol=1e-9)
    assert six[2].sum() * 2.0 == pytest.approx(math.pi * 1.2 * 1.1, rel=0.01)
    assert ball_patch([(-50, -50, -40, -40, 1, 1, 1.0)], 100, 60, 0, 2, 2.0) is None
