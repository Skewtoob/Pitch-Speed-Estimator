"""Behind-view event times (segment.py): hidden catch, release latency, honest sigmas.

A pinhole camera 30 ft behind the plate films a drag + gravity pitch; the
track is the projected ball positions (+ noise), with the end hidden for a
given number of fields (the catcher), so the true event times are known.
"""
import math

import numpy as np
import pytest

from pitchspeed.calib.model import CalibrationFile, Segment, Settings
from pitchspeed.estimate.speed import estimate_speed
from pitchspeed.vision import geometry as G
from pitchspeed.vision.detect import CAND_DTYPE, DetFrame
from pitchspeed.vision.options import FinderOptions
from pitchspeed.vision.segment import Segmenter, truncated_normal
from pitchspeed.vision.track import CandStore, Pt, Track, fit_track

DT = 1001 / 60000
SAR = 10 / 11
W, H = 720, 480


def _camera(back=30.0, height=7.0, f=1500.0):
    C = np.array([0.0, -back, height])
    target = np.array([0.0, 25.0, 3.0])
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return C, R, f


def _project(cam, X):
    C, R, f = cam
    Xc = (np.atleast_2d(X) - C) @ R.T
    u = f * Xc[:, 0] / Xc[:, 2] / SAR + (W - 1) / 2      # stored (non-square) pixels
    v = f * Xc[:, 1] / Xc[:, 2] + (H - 1) / 2
    return np.stack([u, v], 1)


def _pitch(speed_mph=65.0, t_rel=2.0, ext=5.5, D=60.5, x_r=-1.5, z_r=5.6, target=(0.3, 2.4), catch_y=-1.5):
    """Positions (world ft) every field from release to catch; drag 0.0018/ft, gravity, mild lift."""
    k, g, lift = 0.0018, 32.17, 8.0
    p = np.array([x_r, D - ext, z_r])
    v0 = speed_mph * 5280 / 3600
    # aim (iteratively) so the ball reaches the target at catch_y
    aim = np.array([target[0], catch_y, target[1]]) - p
    vel = v0 * aim / np.linalg.norm(aim)
    for _ in range(4):
        P, T = _integrate(p, vel, k, g, lift, catch_y)
        miss = np.array([target[0], catch_y, target[1]]) - P[-1]
        vel = vel + np.array([miss[0], 0.0, miss[2]]) / max(T, 1e-3)
        vel *= v0 / np.linalg.norm(vel)
    P, T = _integrate(p, vel, k, g, lift, catch_y)
    return P, T


def _integrate(p, vel, k, g, lift, catch_y, h=1e-4):
    p, vel = p.copy(), vel.copy()
    out, t = [p.copy()], 0.0
    next_t = DT
    while p[1] > catch_y:
        a = -k * np.linalg.norm(vel) * vel + np.array([0.0, 0.0, -g + lift])
        vel = vel + a * h
        p = p + vel * h
        t += h
        if t >= next_t:
            out.append(p.copy())
            next_t += DT
    return np.array(out), t


def _calibration(cam, D=60.5, release=(-1.5, 55.0, 5.6), catch=(0.0, -1.5, 2.2)):
    pts = {"release": _project(cam, np.array(release))[0].tolist(),
           "catch": _project(cam, np.array(catch))[0].tolist()}
    for name, a, b in (("rubber", [-1.0, D + 0.25, 0.83], [1.0, D + 0.25, 0.83]),
                       ("plate", [-17 / 24, 17 / 12, 0.0], [17 / 24, 17 / 12, 0.0])):
        pa, pb = _project(cam, np.array(a))[0], _project(cam, np.array(b))[0]
        pts[f"{name}_left"], pts[f"{name}_right"] = pa.tolist(), pb.tolist()
    st = Settings(distance_ft=D, level="hs", sar=SAR)
    seg = Segment(start_frame=0, ref_frame=0, view="behind", points=pts)
    return CalibrationFile(settings=st, segments=[seg], image_size=(W, H)).segment_calibs()[0]


def _setup(hidden=5, latency=1, noise=0.4, seed=0, speed_mph=65.0, target=(0.3, 2.4), scale_clicks=True,
           cue=None, pre=80, skipped=None):
    """cue: None (no motion data) or the time offset (s, vs the true release) of a burst of
    delivery motion near the release point in DetFrame.e_rel.  pre: samples of data before the
    first detection (the processed range starts there); skipped: (s0, s1) samples marked
    skipped ('no_registration')."""
    rng = np.random.default_rng(seed)
    cam = _camera()
    calib = _calibration(cam)
    if not scale_clicks:
        for n in ("rubber_left", "rubber_right", "plate_left", "plate_right"):
            calib.segment.points.pop(n)
    geom = G.view_geometry(calib, DT)
    t_rel = 120 * DT + 0.37 * DT                       # true release between two fields
    P, T = _pitch(speed_mph=speed_mph, target=target)
    t_cat = t_rel + T
    uv = _project(cam, P)
    s_first = int(math.ceil(t_rel / DT)) + latency     # first field after release + latency
    tr = Track(tid=3, seed_score=200.0)
    n_vis = 0
    for s in range(s_first, 10_000):
        t = s * DT
        if t >= t_cat:
            break
        i = (t - t_rel) / DT
        j = int(math.floor(i))
        if j + 1 >= len(uv):
            break
        q = uv[j] + (i - j) * (uv[j + 1] - uv[j])
        n_vis += 1
        tr.pts[s] = Pt(s=s, t=t, k=s & 1, x=float(q[0] + rng.normal(0, noise)), y=float(q[1] + rng.normal(0, noise)),
                       z=20.0, attached=False, sigma=0.5)
    keep = sorted(tr.pts)[:max(0, len(tr.pts) - hidden)]
    tr.pts = {s: tr.pts[s] for s in keep}
    st = CandStore(2000)
    for s in range(keep[0] - pre, keep[-1] + 60):
        e = float("nan")
        if cue is not None:
            e = 0.03 + 0.002 * math.sin(1.7 * s) + 0.12 * math.exp(-0.5 * ((s * DT - t_rel - cue) / (1.5 * DT)) ** 2)
        skip = "no_registration" if skipped is not None and skipped[0] <= s <= skipped[1] else ""
        st.add(DetFrame(s=s, t=s * DT, k=s & 1, parity=s & 1, A=np.eye(3), quality=1.0,
                        cands=np.zeros(0, CAND_DTYPE), e_rel=float("nan") if skip else e, skipped=skip))
    fit_track(tr, geom, FinderOptions())
    seg = Segmenter(geom, FinderOptions(), DT, 2 * DT, True, st, quality_of=lambda a, b: 0.8, view="behind")
    seg.s_range = (keep[0] - pre, keep[-1] + 59)
    return seg, tr, t_rel, t_cat


def test_truncated_normal_moments():
    m, s = truncated_normal(0.0, 1.0, 0.0, math.inf)
    assert abs(m - math.sqrt(2 / math.pi)) < 1e-6 and abs(s - math.sqrt(1 - 2 / math.pi)) < 1e-6
    assert truncated_normal(5.0, 1.0, -math.inf, math.inf) == pytest.approx((5.0, 1.0))
    assert truncated_normal(0.0, 1.0, 20.0, 30.0) == (20.0, 0.0)          # mass at the lower bound
    m, s = truncated_normal(0.0, 1.0, -30.0, -20.0)
    assert math.isnan(m) and math.isnan(s)


@pytest.mark.parametrize("hidden", [3, 5, 7])
def test_hidden_catch_is_extrapolated_with_honest_sigma(hidden):
    seg, tr, t_rel, t_cat = _setup(hidden=hidden)
    pt = seg.classify(tr)
    assert pt.status == "pitch", pt.reject_code
    assert pt.catch.method == "extrapolated" and "catch_extrapolated" in pt.flags
    assert "catch_not_observed" not in pt.flags
    # informative (a few fields), and honest
    assert pt.catch.sigma < 0.05
    assert abs(pt.catch.t - t_cat) < 2.5 * pt.catch.sigma + 0.004


def test_catch_seen_is_last_moving_field():
    seg, tr, t_rel, t_cat = _setup(hidden=0)
    pt = seg.classify(tr)
    assert pt.catch.method == "last_moving_field"
    assert abs(pt.catch.t - t_cat) < 2.5 * pt.catch.sigma
    assert pt.catch.sigma < 0.03


def test_long_hidden_flight_is_not_observed():
    # the catcher hides half of the flight: no fake catch time
    seg, tr, t_rel, t_cat = _setup(hidden=14)
    pt = seg.classify(tr)
    assert pt.catch is not None
    assert "catch_not_observed" in pt.flags
    assert pt.catch.sigma > 0.08
    assert abs(pt.catch.t - t_cat) < 2.0 * pt.catch.sigma


def test_hidden_catch_without_scale_clicks_is_wider():
    a = _setup(hidden=6)[0:2]
    b = _setup(hidden=6, scale_clicks=False)[0:2]
    pa, pb = a[0].classify(a[1]), b[0].classify(b[1])
    assert pb.catch.sigma >= pa.catch.sigma * 0.9          # a guessed scale is never more precise


@pytest.mark.parametrize("latency", [0, 1, 2])
def test_release_first_free_field_covers_truth(latency):
    seg, tr, t_rel, t_cat = _setup(hidden=4, latency=latency)
    pt = seg.classify(tr)
    assert pt.release.method == "first_free_field"
    assert abs(pt.release.t - t_rel) < 2.0 * pt.release.sigma
    assert "release_methods_disagree" not in pt.flags


def test_no_clean_detection_gives_uninformative_release():
    # every detection weak (z below z_strong): the release is not observed, and the
    # image model's backward extrapolation must not be passed off as a release time
    seg, tr, t_rel, t_cat = _setup(hidden=3)
    for p in tr.pts.values():
        p.z = 4.0
    pt = seg.classify(tr)
    assert pt.release is not None and "release_not_observed" in pt.flags
    assert pt.release.method == "first_free_field"
    assert pt.release.sigma > 0.1
    assert abs(pt.release.t - t_rel) < 2.0 * pt.release.sigma


def test_behind_estimate_is_poor_not_none_for_short_hidden_end():
    # end-to-end with E1: a few hidden fields at the end still give an honest number
    from pitchspeed.estimate.speed import estimate_speed
    seg, tr, t_rel, t_cat = _setup(hidden=5, latency=1)
    pt = seg.classify(tr)
    sp = estimate_speed(seg_calib(), None, DT, pt.release, pt.catch, track=pt)
    assert sp.label in ("poor", "fair"), sp.reason
    assert sp.v_release_mph is not None and 0.04 < sp.sigma_mph / sp.v_release_mph < 0.10
    assert abs(sp.v_release_mph - 65.0) < 2.0 * sp.sigma_mph       # _pitch() throws 65 mph out of the hand


def seg_calib():
    return _calibration(_camera())


def test_delivery_cue_agreeing_changes_nothing():
    a = _setup(hidden=4, latency=1)
    b = _setup(hidden=4, latency=1, cue=0.5 * DT)
    pa, pb = a[0].classify(a[1]), b[0].classify(b[1])
    assert "release_detection_late" not in pb.flags
    assert pb.release.t == pytest.approx(pa.release.t) and pb.release.sigma == pytest.approx(pa.release.sigma)
    assert pb.diagnostics["release_cue_t"] == pytest.approx(b[2], abs=1.5 * DT)


def test_late_first_detection_is_widened_by_the_delivery_cue():
    # the detector missed the first 12 fields of the flight; the arm burst shows the release
    seg, tr, t_rel, t_cat = _setup(hidden=4, latency=12, cue=0.5 * DT)
    pt = seg.classify(tr)
    assert "release_detection_late" in pt.flags and "release_uncertain" in pt.flags
    assert abs(pt.release.t - t_rel) < 1.645 * pt.release.sigma
    lo, hi = pt.diagnostics["release_interval"]
    assert lo <= t_rel <= hi + 2 * DT
    # without the cue the same track gives a confidently wrong release
    seg2, tr2, _, _ = _setup(hidden=4, latency=12)
    p2 = seg2.classify(tr2)
    assert abs(p2.release.t - t_rel) > 3 * p2.release.sigma


# ---- release just after the start of the data (clip / --start / segment start, skipped stretch)
def _late_start_case(**kw):
    # the detector also missed the first 8 fields of the flight (common: the ball barely moves
    # in the image early on), and there is no data from before the first detection
    return _setup(hidden=4, latency=8, **kw)


def test_release_at_the_data_start_is_not_observed():
    seg, tr, t_rel, t_cat = _late_start_case(pre=3)
    pt = seg.classify(tr)
    assert "release_not_observed" in pt.flags
    assert pt.diagnostics["release_not_observed_reason"] == "release_before_data"
    assert pt.diagnostics["release_pre_data_coverage"] < 0.6
    lo, hi = pt.diagnostics["release_interval"]
    assert lo <= t_rel <= hi
    # the speed is not given with a tight, wrong interval
    sp = estimate_speed(seg_calib(), None, DT, pt.release, pt.catch, track=pt)
    assert sp.v_release_mph is None or abs(sp.v_release_mph - 65.0) < 1.645 * sp.sigma_mph


def test_release_after_a_skipped_stretch_is_not_observed():
    seg, tr, t_rel, t_cat = _late_start_case()
    s_clean = min(tr.pts)
    seg2, tr2, _, _ = _late_start_case(skipped=(s_clean - 60, s_clean - 1))
    pt = seg2.classify(tr2)
    assert "release_not_observed" in pt.flags
    assert pt.diagnostics["release_not_observed_reason"] == "release_before_data"


def test_release_with_data_before_is_unchanged():
    seg, tr, t_rel, t_cat = _late_start_case()
    pt = seg.classify(tr)
    assert pt.diagnostics["release_pre_data_coverage"] == pytest.approx(1.0)
    assert pt.diagnostics.get("release_not_observed_reason") != "release_before_data"
    # a few missing samples (registration hiccup) are not "no data"
    s_clean = min(tr.pts)
    seg2, tr2, _, _ = _late_start_case(skipped=(s_clean - 8, s_clean - 1))
    p2 = seg2.classify(tr2)
    assert p2.diagnostics.get("release_not_observed_reason") != "release_before_data"
    assert p2.release.t == pytest.approx(pt.release.t) and p2.release.sigma == pytest.approx(pt.release.sigma)


def test_fields_lost_in_alternation_are_not_missing_data():
    # registration failing on every other field (eval2/s31337_002): half the fields still look
    seg, tr, t_rel, t_cat = _late_start_case()
    pt = seg.classify(tr)
    s_clean = min(tr.pts)
    seg2, tr2, _, _ = _late_start_case()
    for s in range(s_clean - 60, s_clean):
        if s % 2 == 0 or s % 7 == 0:
            seg2.store.get(s).skipped = "no_registration"
    p2 = seg2.classify(tr2)
    assert p2.diagnostics["release_pre_data_coverage"] < 0.6 and p2.diagnostics["release_pre_data_gap"] < 8
    assert p2.diagnostics.get("release_not_observed_reason") != "release_before_data"
    assert p2.release.t == pytest.approx(pt.release.t)


def test_finder_usable_mask_covers_unprocessed_samples():
    from types import SimpleNamespace
    from pitchspeed.vision.finder import PitchFinder
    f = SimpleNamespace(_usable={s: (s % 4 != 0) for s in range(100, 200)})
    m = PitchFinder._usable_mask(f, 96, 104)
    assert m == [False, False, False, False, False, True, True, True, False]   # 96..99: never processed
    assert PitchFinder._usable_mask(f, 10, 5) == []
