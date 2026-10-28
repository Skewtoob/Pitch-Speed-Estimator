"""Unit tests: gating, seeding/extension, field-order check, segmentation codes and events."""
import math

import numpy as np
import pytest

from pitchspeed.constants import MPH_TO_FTPS
from pitchspeed.vision import geometry as G
from pitchspeed.vision.detect import CAND_DTYPE, DetFrame
from pitchspeed.vision.options import FinderOptions
from pitchspeed.vision.segment import Segmenter
from pitchspeed.vision.track import (CandStore, Pt, Track, build_track, field_order_check, find_seeds, fit_track,
                                     velocity_gate)
from vision_helpers import side_calibration

DT = 1001 / 60000


@pytest.fixture()
def geom():
    return G.view_geometry(side_calibration().segment_calibs()[0], DT)


def _line(geom, speed_mph=80.0, reverse=False, angle_deg=0.0, t0=1.0, n_extra=0):
    """Reference-px positions of a ball moving release -> catch at the given speed."""
    P0, P1 = geom.release, geom.catch
    t0 = round(t0 / DT) * DT                                   # sample aligned
    v_sq = speed_mph * MPH_TO_FTPS * geom.scale_rel          # square px / s
    L = geom.length_sq
    T = L / v_sq
    n = int(T / DT) + 1 + n_extra
    t = t0 + np.arange(n) * DT
    u = (t - t0) / T
    if reverse:
        u = 1 - u
    d = (P1 - P0) * [geom.sar, 1.0]
    a = math.radians(angle_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    q = (P0 * [geom.sar, 1.0])[None] + (u[:, None] * (rot @ d)[None]) + np.c_[np.zeros(n), 10 * u * u]
    return t, q[:, 0] / geom.sar, q[:, 1]


def _store(samples):
    """samples: {s: [(t, x, y, z, attached, sign), ...]} -> CandStore."""
    st = CandStore(1000)
    uid = 0
    for s in sorted(samples):
        rows = samples[s]
        arr = np.zeros(len(rows), CAND_DTYPE)
        for i, (t, x, y, z, att, sign) in enumerate(rows):
            arr[i] = (uid, s, t, s & 1, x, y, z, z, sign, 3.0, 0.0, att, max(0.4, 3 / z), 0.0, 0)
            uid += 1
        t_s = rows[0][0] if rows else s * DT
        st.add(DetFrame(s=s, t=t_s, k=s & 1, parity=s & 1, A=np.eye(3), quality=1.0, cands=arr))
    return st


def _scene(geom, rng, clutter=4, **kw):
    t, x, y = _line(geom, **kw)
    s0 = int(round(t[0] / DT))
    samples = {}
    for i in range(-20, len(t) + 20):
        s = s0 + i
        rows = []
        ts = s * DT
        if 0 <= i < len(t):
            rows.append((ts, x[i] + rng.normal(0, 0.3), y[i] + rng.normal(0, 0.3), 12.0, False, 1))
        for _ in range(clutter):
            x0, y0, x1, y1 = geom.bbox
            rows.append((ts, rng.uniform(x0, x1), rng.uniform(y0, y1), rng.uniform(3.5, 6.0), False,
                         int(rng.choice([-1, 1]))))
        samples[s] = rows
    return samples, (t, x, y)


def test_velocity_gate_side(geom):
    o = FinderOptions()
    v85 = 85 * MPH_TO_FTPS * geom.scale_rel * DT
    ax = geom.axis_sq
    perp = np.array([-ax[1], ax[0]])
    assert velocity_gate(geom, ax * v85, o)[0]
    assert velocity_gate(geom, -ax * v85, o)[0]                 # throwbacks are seeded too (rejected later)
    assert not velocity_gate(geom, perp * v85, o)[0]
    assert not velocity_gate(geom, ax * v85 / 10, o)[0]         # 8.5 mph
    assert not velocity_gate(geom, ax * v85 * 3, o)[0]          # 255 mph


def test_seed_and_extend_recovers_track_in_clutter(geom):
    rng = np.random.default_rng(0)
    samples, (t, x, y) = _scene(geom, rng, clutter=6)
    st = _store(samples)
    o = FinderOptions()
    seeds = find_seeds(st, st.s_min + 1, st.s_max - 1, geom, o)
    assert seeds
    tr = build_track(1, seeds[0], st, geom, o, DT, st.s_min, st.s_max)
    A = tr.arrays(inliers_only=True)
    assert len(A["t"]) >= len(t) - 2
    # every inlier is on the true line
    for ti, xi, yi in zip(A["t"], A["x"], A["y"]):
        j = int(np.argmin(np.abs(t - ti)))
        assert abs(t[j] - ti) < 1e-6 and math.hypot(xi - x[j], yi - y[j]) < 2.0
    assert tr.model is not None and tr.model.rms < 1.0


def test_extension_keeps_contrast_sign(geom):
    rng = np.random.default_rng(1)
    samples, (t, x, y) = _scene(geom, rng, clutter=0)
    s0 = min(samples)
    # a strong opposite-sign ghost right on the predicted path one sample before the start
    first = min(s for s, rows in samples.items() if rows)
    s_ghost = first + 20 - 1
    samples[s_ghost] = [(s_ghost * DT, x[0] - (x[1] - x[0]), y[0], 30.0, False, -1)]
    st = _store(samples)
    o = FinderOptions()
    seeds = find_seeds(st, st.s_min + 1, st.s_max - 1, geom, o)
    tr = build_track(1, seeds[0], st, geom, o, DT, st.s_min, st.s_max)
    assert s_ghost not in tr.pts


def test_field_order_check_detects_swapped_fields():
    t = np.arange(30) * DT
    k = np.arange(30) % 2
    x = 600 - 1100 * t
    y = 230 + 300 * t * t
    ok = field_order_check(t, x, y, k, 2 * DT)
    assert ok["preferred"] in ("given", "unclear") and not ok["conflict"]
    # positions observed with the other field order: each pair of fields swapped in time
    t_true = t + (1 - 2 * k) * DT
    bad = field_order_check(t, 600 - 1100 * t_true, 230 + 300 * t_true ** 2, k, 2 * DT)
    assert bad["conflict"] and bad["preferred"] == "flipped"
    assert bad["alt_given"] < -0.5


def _track_from(t, x, y, z=12.0, att=False):
    tr = Track(tid=7, seed_score=40.0)
    for ti, xi, yi in zip(t, x, y):
        s = int(round(ti / DT))
        tr.pts[s] = Pt(s=s, t=float(ti), k=s & 1, x=float(xi), y=float(yi), z=z, attached=att, sigma=0.5)
    return tr


def _segmenter(geom, t_lo, t_hi):
    s0, s1 = int(t_lo / DT) - 30, int(t_hi / DT) + 30
    st = _store({s: [] for s in range(s0, s1)})
    for s, df in st.frames.items():
        df.t = s * DT
    return Segmenter(geom, FinderOptions(), DT, 2 * DT, True, st, quality_of=lambda a, b: 0.8)


def test_classify_good_pitch_events(geom):
    t, x, y = _line(geom, speed_mph=80.0, t0=2.0)
    tr = _track_from(t, x, y)
    fit_track(tr, geom, FinderOptions())
    seg = _segmenter(geom, t[0], t[-1])
    pt = seg.classify(tr)
    assert pt.status == "pitch", pt.reject_code
    assert pt.release.method == "extrapolated" and abs(pt.release.t - t[0]) < 0.004
    assert pt.catch.method == "last_moving_field" and abs(pt.catch.t - (t[-1] + DT / 2)) < 1e-9
    assert 70 < pt.diagnostics["speed_mph_rough"] < 90
    assert pt.confidence > 0.6
    assert pt.model["kind"] in ("poly", "rational")


@pytest.mark.parametrize("kw,code", [
    (dict(reverse=True), "THROWBACK"),
    (dict(angle_deg=50.0), "OFF_AXIS"),
    (dict(speed_mph=180.0), "SPEED_IMPLAUSIBLE"),
])
def test_classify_reject_codes(geom, kw, code):
    t, x, y = _line(geom, t0=2.0, **kw)
    tr = _track_from(t, x, y)
    fit_track(tr, geom, FinderOptions())
    pt = _segmenter(geom, t[0], t[-1]).classify(tr)
    assert pt.status == "rejected" and pt.reject_code == code


def test_classify_too_short_and_no_progress(geom):
    t, x, y = _line(geom, t0=2.0)
    tr = _track_from(t[:4], x[:4], y[:4])
    fit_track(tr, geom, FinderOptions())
    assert _segmenter(geom, t[0], t[-1]).classify(tr).reject_code == "TOO_SHORT"
    tr = _track_from(t[:8], x[:8], y[:8])                      # ~25 % of the way
    fit_track(tr, geom, FinderOptions())
    assert _segmenter(geom, t[0], t[-1]).classify(tr).reject_code == "NO_PROGRESS"


def test_occluded_end_is_extrapolated_with_honest_sigma(geom):
    t, x, y = _line(geom, t0=2.0)
    n = int(0.7 * len(t))
    tr = _track_from(t[:n], x[:n], y[:n])
    fit_track(tr, geom, FinderOptions())
    pt = _segmenter(geom, t[0], t[-1]).classify(tr)
    assert pt.status == "pitch"
    assert pt.catch.method == "extrapolated" and "catch_extrapolated" in pt.flags
    t_true = t[-1] + (t[1] - t[0]) * 0.0
    assert abs(pt.catch.t - t_true) < 3 * pt.catch.sigma + 0.01
