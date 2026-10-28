"""E1 (time of flight), manual events, labels and small helpers."""
import math

import numpy as np
import pytest

from pitchspeed import constants as C
from pitchspeed import physics
from pitchspeed.calib.model import Calibration, Segment, Settings
from pitchspeed.estimate import NotMeasurable, label_for, manual_event, tof_estimate
from pitchspeed.estimate import common as K
from pitchspeed.types import EventTime, ScanInfo

DT = C.NTSC_FIELD_DT
SCAN = ScanInfo("interlaced", "bff", "high", "content", 1 / DT)


def behind(D=60.5, **kw) -> Calibration:
    st = Settings(distance_ft=D, **kw)
    return Calibration(st, Segment(view="behind", points={"release": [300, 200], "catch": [360, 330]}))


def events_for(cal: Calibration, mph: float, t0=2.0, s_r=DT / math.sqrt(12), s_c=0.003,
               m_r="first_free_field", m_c="extrapolated"):
    k = physics.DragModel.default(cal.settings.sport).k
    L = cal.release_y_ft - cal.catch_y_ft
    T = physics.time_to_travel(mph * C.MPH_TO_FTPS, L, k)
    return EventTime(t0, s_r, m_r), EventTime(t0 + T, s_c, m_c), L, T


@pytest.mark.parametrize("mph", [45.0, 70.0, 95.0])
def test_inverts_the_drag_model_and_applies_slope_correction(mph):
    cal = behind()
    r, c, L, T = events_for(cal, mph)
    comp = tof_estimate(cal, SCAN, DT, r, c)
    d = comp.details
    assert d["v_horizontal_mph"] == pytest.approx(mph, rel=1e-9)
    sf = physics.slope_correction(L, T, cal.release_height_ft, cal.settings.catch_height_ft)
    assert comp.v_release_mph == pytest.approx(mph * sf, rel=1e-9)
    assert comp.v_avg_mph == pytest.approx(L / T * C.FTPS_TO_MPH)
    k = physics.DragModel.default().k
    assert d["v_plate_mph"] == pytest.approx(comp.v_release_mph * math.exp(-k * (cal.release_y_ft - 17 / 12)))
    assert d["L_ft"] == pytest.approx(60.5 - 5.5 + 1.5)


def test_youth_distance_and_extension_defaults():
    cal = behind(D=46.0, level="ll")
    r, c, L, T = events_for(cal, 50.0)
    comp = tof_estimate(cal, SCAN, DT, r, c)
    assert comp.details["L_ft"] == pytest.approx(46.0 - 4.0 + 1.5)
    assert comp.details["k_rel_sigma"] == pytest.approx(0.25)          # slow pitch -> 25 % on k


@pytest.mark.parametrize("mph,D,expected", [(90, 60.5, 2.8), (75, 60.5, 2.7), (60, 60.5, 2.8), (45, 46.0, 3.3)])
def test_sigma_reproduces_math_md_budget_A(mph, D, expected):
    """math.md §4 scenario A: release in the first field, catch extrapolated (3 ms), default ext."""
    cal = behind(D=D)
    r, c, L, T = events_for(cal, mph, s_r=4.8e-3, s_c=3e-3)
    comp = tof_estimate(cal, SCAN, DT, r, c)
    total = math.hypot(comp.sigma_ind_mph, comp.sigma_shared_mph) / comp.v_release_mph * 100
    assert total == pytest.approx(expected, abs=0.15)


def test_k_sensitivity_is_positive_and_about_kL_over_2():
    cal = behind()
    r, c, L, T = events_for(cal, 85)
    comp = tof_estimate(cal, SCAN, DT, r, c)
    k = physics.DragModel.default().k
    assert comp.details["dv_k_mph"] > 0
    assert comp.details["rel_sigma_k"] == pytest.approx(0.15 * k * L / 2, rel=0.1)


def test_event_sigma_floor_and_flags():
    cal = behind()
    r, c, _, _ = events_for(cal, 80, s_r=1e-4, s_c=1e-5)
    comp = tof_estimate(cal, SCAN, DT, r, c)
    assert comp.details["sigma_release_s"] == pytest.approx(DT / math.sqrt(12))   # whole-field event
    assert comp.details["sigma_catch_s"] == pytest.approx(0.001)                  # sub-field event
    assert K.F_EVENT_SIGMA_FLOORED in comp.details["flags"]
    assert K.F_RELEASE_UNCONFIRMED in comp.details["flags"]
    # a missing / zero sigma is not trusted: half a field
    r0, c0, _, _ = events_for(cal, 80, s_r=0.0, s_c=float("nan"))
    d0 = tof_estimate(cal, SCAN, DT, r0, c0).details
    assert d0["sigma_release_s"] == pytest.approx(DT / 2) and d0["sigma_catch_s"] == pytest.approx(DT / 2)


def test_invalid_events_raise():
    cal = behind()
    r, c, _, _ = events_for(cal, 80)
    with pytest.raises(NotMeasurable):
        tof_estimate(cal, SCAN, DT, c, r)                   # reversed
    with pytest.raises(NotMeasurable):
        tof_estimate(cal, SCAN, DT, r, None)
    with pytest.raises(NotMeasurable):
        tof_estimate(cal, SCAN, DT, r, EventTime(r.t + 5 * DT, 0.003, "extrapolated"))   # < 8 fields
    with pytest.raises(NotMeasurable):
        tof_estimate(cal, SCAN, DT, r, EventTime(float("nan"), 0.003, "extrapolated"))


def test_implausible_speed_is_not_usable():
    cal = behind()
    r = EventTime(1.0, 0.005, "manual")
    comp = tof_estimate(cal, SCAN, DT, r, EventTime(1.3, 0.005, "manual"), manual=True)   # ~130 mph
    assert comp.details["usable"] is False
    assert K.F_IMPLAUSIBLE in comp.details["flags"]
    assert "outside" in comp.details["reason"]


def test_contact_end_uses_contact_point():
    cal = behind()
    r, c, _, _ = events_for(cal, 80)
    comp = tof_estimate(cal, SCAN, DT, r, c, end_kind="contact")
    assert comp.details["L_ft"] == pytest.approx(cal.release_y_ft - K.CONTACT_Y_FT)
    assert K.F_CONTACT in comp.details["flags"]


def test_lost_track_flags_catch_extrapolated():
    cal = behind()
    r, c, _, _ = events_for(cal, 80)
    assert K.F_CATCH_EXTRAPOLATED in tof_estimate(cal, SCAN, DT, r, c, end_kind="lost").details["flags"]
    assert K.F_CATCH_EXTRAPOLATED not in tof_estimate(cal, SCAN, DT, r, c).details["flags"]


def test_field_order_uncertainty_and_dropped_frames_inflate_timing():
    cal = behind()
    r, c, _, _ = events_for(cal, 80)
    base = tof_estimate(cal, SCAN, DT, r, c)
    low = ScanInfo("interlaced", "bff", "low", "content", 1 / DT)
    lo = tof_estimate(cal, low, DT, r, c)
    assert K.F_FIELD_ORDER in lo.details["flags"]
    assert lo.details["sigma_T_s"] ** 2 == pytest.approx(base.details["sigma_T_s"] ** 2 + DT ** 2)
    # a user override is trusted
    ov = ScanInfo("interlaced", "bff", "low", "override", 1 / DT)
    assert K.F_FIELD_ORDER not in tof_estimate(cal, ov, DT, r, c).details["flags"]
    dr = tof_estimate(cal, SCAN, DT, r, c, track_flags=[K.F_DROPPED_FRAME])
    assert K.F_DROPPED_FRAME in dr.details["flags"]
    assert dr.sigma_ind_mph > base.sigma_ind_mph


def test_progressive_flags_frames_only():
    cal = behind()
    r, c, _, _ = events_for(cal, 80)
    prog = ScanInfo("progressive", None, "high", "content", 29.97)
    comp = tof_estimate(cal, prog, C.NTSC_FRAME_DT, r, c)
    assert K.F_FRAMES_ONLY in comp.details["flags"]
    assert comp.details["sigma_release_s"] == pytest.approx(C.NTSC_FRAME_DT / math.sqrt(12))   # floor


def test_user_extension_reduces_sigma():
    r, c, _, _ = events_for(behind(), 80)
    a = tof_estimate(behind(), SCAN, DT, r, c)
    b = tof_estimate(behind(extension_ft=5.5, extension_sigma_ft=0.3), SCAN, DT, r, c)
    assert b.sigma_ind_mph < 0.8 * a.sigma_ind_mph          # catch point (0.75 ft) still remains


def test_softball_drag_sigma():
    cal = behind(D=43.0, sport="softball", level="sb43")
    r, c, _, _ = events_for(cal, 60)
    assert tof_estimate(cal, SCAN, DT, r, c).details["k_rel_sigma"] == pytest.approx(0.30)


def side_calib(**pts):
    points = {"plate": [100.0, 300.0], "rubber": [620.0, 290.0]}
    points.update(pts)
    return Calibration(Settings(distance_ft=60.5), Segment(view="side", points=points))


def test_side_view_extrapolated_release_refers_to_the_release_point():
    cal = side_calib()
    r, c, L, T = events_for(cal, 85, m_r="extrapolated")
    comp = tof_estimate(cal, SCAN, DT, r, c)
    d = comp.details
    assert d["track_dependent"] is True
    assert d["release_point"] == "default_release_point"
    assert d["release_y_ft"] == pytest.approx(cal.release_y_ft)       # no rho known: affine
    # catch point + unknown-rho term enter the distance sigma; extension enters through drag only
    k = physics.DragModel.default().k
    kl = k * L
    s_pos = math.hypot(0.75, 1.5)
    assert d["rel_sigma_distance"] == pytest.approx(s_pos / L * kl * math.exp(kl) / math.expm1(kl))
    # with E2's rho (and no obliquity) only the catch point remains
    d1 = tof_estimate(cal, SCAN, DT, r, c, side={"rho": 1.0, "tan_phi": 0.0, "sigma_tan_phi": 0.0}).details
    assert d1["rel_sigma_distance"] == pytest.approx(0.75 / L * kl * math.exp(kl) / math.expm1(kl))
    d2 = tof_estimate(cal, SCAN, DT, r, c, side={"rho": 0.8, "tan_phi": 0.3, "sigma_tan_phi": 0.05}).details
    assert d2["release_y_ft"] < cal.release_y_ft and d2["rel_sigma_distance"] > d1["rel_sigma_distance"]
    assert d["rel_sigma_extension_drag"] == pytest.approx(k * 1.0)
    # a manual release in side view is an observed instant: extension sigma applies
    rm, cm, _, _ = events_for(cal, 85, m_r="manual", m_c="manual")
    dm = tof_estimate(cal, SCAN, DT, rm, cm, manual=True).details
    assert dm["track_dependent"] is False and dm["rel_sigma_distance"] > 1.5 * d1["rel_sigma_distance"]


def test_side_view_clicked_release_point_is_mapped_to_world_y():
    P, R = np.array([100.0, 300.0]), np.array([620.0, 290.0])
    lam = Calibration(Settings(), Segment()).lam_of_y(55.0)
    q = P + lam * (R - P) + np.array([0.0, -40.0])              # straight above y = 55 ft
    cal = side_calib(release=q.tolist())
    from pitchspeed.estimate.tof import side_release_point
    y, sig, src = side_release_point(cal, {"rho": 1.0})
    assert src == "clicked_release_point" and y == pytest.approx(55.0, abs=1e-6)
    assert sig == pytest.approx(0.3)


# ------------------------------------------------------------------ helpers
def test_manual_event_semantics():
    ev = manual_event(10.0, DT, "release", s=600)
    assert ev.t == pytest.approx(10.0 - DT / 2)
    assert ev.sigma == pytest.approx(DT / math.sqrt(12))
    assert ev.method == "manual" and ev.s == 600
    with pytest.raises(ValueError):
        manual_event(1.0, DT, "glove")
    with pytest.raises(ValueError):
        manual_event(1.0, 0.0, "catch")


def test_manual_events_are_unbiased_for_random_phase():
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(4000):
        t_true = rng.uniform(0, 10)
        s = math.floor(t_true / DT) + 1          # first field with the ball out of the hand
        errs.append(manual_event(s * DT, DT, "release").t - t_true)
    assert abs(np.mean(errs)) < 2e-4
    assert np.std(errs) == pytest.approx(DT / math.sqrt(12), rel=0.05)


@pytest.mark.parametrize("rel,label", [(0.01, "good"), (0.025, "good"), (0.03, "fair"), (0.05, "fair"),
                                       (0.07, "poor"), (0.10, "poor"), (0.11, "none"),
                                       (float("nan"), "none"), (None, "none")])
def test_labels(rel, label):
    assert label_for(rel) == label


def test_ci90():
    lo, hi = K.ci90(80.0, 2.0)
    assert hi - 80.0 == pytest.approx(80.0 - lo) == pytest.approx(1.645 * 2.0, rel=1e-3)
