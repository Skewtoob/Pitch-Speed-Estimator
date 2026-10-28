"""combine() rules and the estimate_speed() orchestration."""
import json
import math

import numpy as np
import pytest

from pitchspeed import constants as C
from pitchspeed import physics
from pitchspeed.calib.model import Calibration, Segment, Settings
from pitchspeed.estimate import combine, estimate_speed, manual_event
from pitchspeed.estimate import common as K
from pitchspeed.types import EventTime, PitchTrack, ScanInfo, SpeedComponent, speed_from_dict, to_jsonable

import estimate_helpers as H

DT = C.NTSC_FIELD_DT
SCAN = ScanInfo("interlaced", "bff", "high", "content", 1 / DT)


def comp(method, v, s_ind, dv_k=0.5, **details):
    d = {"usable": True, "flags": [], "dv_k_mph": dv_k}
    d.update(details)
    return SpeedComponent(method=method, v_release_mph=v, sigma_ind_mph=s_ind, sigma_shared_mph=abs(dv_k),
                          details=d)


# ------------------------------------------------------------------ combine
def test_inverse_variance_with_shared_k():
    c = combine([comp("tof", 80.0, 2.0, 0.6), comp("side_fit", 82.0, 1.0, 0.4, rms_px=1.0)])
    w1, w2 = 1 / 4, 1 / 1
    assert c.method == "tof+side_fit"
    assert c.v_release_mph == pytest.approx((w1 * 80 + w2 * 82) / (w1 + w2))
    assert c.sigma_ind_mph == pytest.approx(1 / math.sqrt(w1 + w2))
    assert c.sigma_shared_mph == pytest.approx((w1 * 0.6 + w2 * 0.4) / (w1 + w2))
    assert c.sigma_mph == pytest.approx(math.hypot(c.sigma_ind_mph, c.sigma_shared_mph))
    assert c.weights["side_fit"] == pytest.approx(0.8)
    assert K.F_DISAGREE not in c.flags


def test_disagreement_prefers_a_healthy_side_fit():
    c = combine([comp("tof", 80.0, 1.0), comp("side_fit", 86.0, 1.0, rms_px=0.9)])
    assert K.F_DISAGREE in c.flags
    assert c.method == "side_fit" and c.v_release_mph == pytest.approx(86.0)


def test_disagreement_with_unhealthy_side_fit_widens_the_interval():
    c = combine([comp("tof", 80.0, 1.0), comp("side_fit", 86.0, 1.0, rms_px=3.0)])
    assert K.F_DISAGREE in c.flags and c.method == "tof+side_fit"
    assert c.v_release_mph == pytest.approx(83.0)
    lo, hi = c.v_release_mph - 1.645 * c.sigma_mph, c.v_release_mph + 1.645 * c.sigma_mph
    assert lo < 80.5 and hi > 85.5


def test_track_dependent_tof_is_never_combined():
    dep = comp("tof", 80.0, 1.5, track_dependent=True)
    side = comp("side_fit", 81.0, 1.0, rms_px=1.0)
    c = combine([dep, side])
    assert c.method == "side_fit" and c.used == ["side_fit"]
    c2 = combine([comp("tof", 80.0, 0.8, track_dependent=True), comp("side_fit", 81.0, 2.0, rms_px=1.0)])
    assert c2.method == "side_fit"                     # even when the dependent E1 claims more precision
    c3 = combine([dep, comp("side_fit", 81.0, 1.0, usable=False)])
    assert c3.method == "tof"                          # E2 unusable: the dependent E1 is the fallback


def test_unusable_components_are_ignored():
    c = combine([comp("tof", 80.0, 1.0, usable=False), comp("side_fit", 70.0, float("nan"))])
    assert c.v_release_mph is None and c.method == "none"
    assert combine([]).v_release_mph is None


# ------------------------------------------------------------ estimate_speed
def behind(**kw):
    return Calibration(Settings(distance_ft=60.5, **kw),
                       Segment(view="behind", points={"release": [300, 200], "catch": [360, 330]}))


def flight_time(cal, mph):
    k = physics.DragModel.default().k
    return physics.time_to_travel(mph * C.MPH_TO_FTPS, cal.release_y_ft - cal.catch_y_ft, k)


def test_behind_view_report_is_complete():
    cal = behind()
    T = flight_time(cal, 80)
    r = EventTime(5.0, DT / math.sqrt(12), "first_free_field", s=300)
    c = EventTime(5.0 + T, 0.0035, "extrapolated", s=327)
    e = estimate_speed(cal, SCAN, DT, r, c)
    assert e.method == "tof" and e.label == "fair"
    assert e.v_release_mph == pytest.approx(80.0, rel=2e-3)
    lo, hi = e.ci90_mph
    assert hi - e.v_release_mph == pytest.approx(e.v_release_mph - lo) == pytest.approx(1.645 * e.sigma_mph, rel=1e-3)
    assert e.v_plate_mph < e.v_avg_mph < e.v_release_mph
    assert K.F_RELEASE_UNCONFIRMED in e.flags
    for key in ("distance_ft", "extension_ft", "extension_assumed", "catch_behind_ft", "cd",
                "air_density_kg_m3", "drag_k_per_ft", "sample_dt_s"):
        assert key in e.assumptions
    assert e.assumptions["extension_assumed"] is True
    assert len(e.components) == 1


def test_missing_events_give_no_number_with_a_reason():
    e = estimate_speed(behind(), SCAN, DT, None, None)
    assert e.v_release_mph is None and e.label == "none" and e.method == "none"
    assert "release" in e.reason


def test_manual_behind_view():
    cal = behind()
    T = flight_time(cal, 70)
    s_r = 600
    s_c = math.ceil((s_r * DT - DT / 2 + T) / DT)
    r = manual_event(s_r * DT, DT, "release", s=s_r)
    c = manual_event(s_c * DT, DT, "catch", s=s_c)
    e = estimate_speed(cal, SCAN, DT, r, c, manual=True)
    assert e.method == "manual_tof"
    assert K.F_MANUAL in e.flags and K.F_RELEASE_UNCONFIRMED not in e.flags
    assert e.v_release_mph == pytest.approx(70.0, rel=0.03)


def test_huge_uncertainty_gives_label_none_without_a_number():
    cal = behind()
    T = flight_time(cal, 80)
    e = estimate_speed(cal, SCAN, DT, EventTime(1.0, 0.06, "first_free_field"),
                       EventTime(1.0 + T, 0.06, "extrapolated"))
    assert e.label == "none" and e.v_release_mph is None and e.ci90_mph is None
    assert "uncertainty" in e.reason
    assert e.components and e.components[0].v_release_mph == pytest.approx(80.0, rel=0.01)   # raw kept


def test_track_flags_are_propagated():
    cal = behind()
    T = flight_time(cal, 80)
    tr = PitchTrack(0, [], flags=[K.F_DROPPED_FRAME, "some_vision_only_flag"])
    e = estimate_speed(cal, SCAN, DT, EventTime(1.0, 0.005, "first_free_field"),
                       EventTime(1.0 + T, 0.0035, "extrapolated"), track=tr)
    assert K.F_DROPPED_FRAME in e.flags and "some_vision_only_flag" not in e.flags


def side_shot(scene="perp90", kind="90FB", seed=0, sigma_px=1.0, manual=False):
    rng = np.random.default_rng(seed)
    fl = H.simulate_pitches(kind, 1, rng)[0]
    cam = H.side_camera(scene, fl.D, 10 / 11)
    cal = H.side_calibration(cam, fl.D, rng, 10 / 11, 2.0)
    t_rel = 3.0 + rng.uniform(0, DT)
    tr = H.make_track(fl, cam, t_rel, rng, sar=10 / 11, sigma_px=sigma_px)
    if manual:
        r, c = H.manual_pair(t_rel, t_rel + fl.T, DT)
    else:
        r = EventTime(t_rel + ((fl.D - fl.ext) - cal.release_y_ft) / fl.v0, 0.009, "extrapolated")
        c = H.catch_extrapolated(t_rel + fl.T, rng)
    return cal, tr, r, c, fl.v0 * C.FTPS_TO_MPH


def test_side_view_uses_the_fit_and_keeps_tof_as_cross_check():
    cal, tr, r, c, truth = side_shot()
    e = estimate_speed(cal, SCAN, DT, r, c, track=tr)
    assert e.method == "side_fit" and e.label == "good"
    assert {x.method for x in e.components} == {"tof", "side_fit"}
    assert abs(e.v_release_mph / truth - 1) < 0.05
    assert "cross-check" in (e.reason or "")


def test_side_view_with_manual_events_combines_both():
    cal, tr, r, c, truth = side_shot(manual=True, seed=1)
    e = estimate_speed(cal, SCAN, DT, r, c, track=tr, manual=True)
    assert e.method == "tof+side_fit"
    assert K.F_MANUAL in e.flags
    side = next(x for x in e.components if x.method == "side_fit")
    assert e.sigma_mph < math.hypot(side.sigma_ind_mph, side.sigma_shared_mph)


def test_side_view_without_track_falls_back_to_tof():
    cal, tr, r, c, truth = side_shot()
    e = estimate_speed(cal, SCAN, DT, r, c, track=None)
    assert e.method == "tof"
    assert "no ball track" in e.reason


def test_result_survives_json_round_trip():
    cal, tr, r, c, truth = side_shot()
    e = estimate_speed(cal, SCAN, DT, r, c, track=tr)
    text = json.dumps(to_jsonable(e), allow_nan=False)
    back = speed_from_dict(json.loads(text))
    assert back.v_release_mph == pytest.approx(e.v_release_mph)
    assert back.ci90_mph == pytest.approx(e.ci90_mph)
    assert [x.method for x in back.components] == [x.method for x in e.components]
    assert back.flags == e.flags and back.label == e.label
