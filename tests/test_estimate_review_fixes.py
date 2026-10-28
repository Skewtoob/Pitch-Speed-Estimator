"""Regression tests for the speed-math review findings.

* a hand-marked catch ends the flight at the glove, never at an automatic
  'contact' end point;
* after a partial manual edit the flags describe the events actually used;
* impossible settings (extension typed in inches, ...) never crash and never
  give a number;
* hand marks carry a human identification term.
"""
import copy
import math

import numpy as np
import pytest

from pitchspeed import constants as C
from pitchspeed import physics
from pitchspeed.calib.model import (Calibration, CalibrationFile, Segment, Settings, extension_limit_ft,
                                    settings_errors)
from pitchspeed.estimate import estimate_speed, manual_event, side_fit_estimate, tof_estimate
from pitchspeed.estimate import common as K
from pitchspeed.estimate import sidefit as SF
from pitchspeed.estimate import speed as SPEED
from pitchspeed.estimate.events import MANUAL_SLIP_PROB, manual_sigma
from pitchspeed.types import EventTime, PitchTrack, ScanInfo

import estimate_helpers as H

DT = C.NTSC_FIELD_DT
SCAN = ScanInfo("interlaced", "bff", "high", "content", 1 / DT)


def behind(D=60.5, **kw):
    return Calibration(Settings(distance_ft=D, **kw),
                       Segment(view="behind", points={"release": [300, 200], "catch": [360, 330]}))


def glove_flight(cal, mph):
    """Flight time release -> glove for a release speed `mph` (horizontal, 1-D drag)."""
    k = physics.DragModel.default().k
    return physics.time_to_travel(mph * C.MPH_TO_FTPS, cal.release_y_ft - cal.catch_y_ft, k)


def marks(t_rel, t_catch, dt=DT):
    """The user marks exactly what the HUD asks: first field out of the hand / in the glove."""
    s_r = math.floor(t_rel / dt) + 1
    s_c = math.ceil(t_catch / dt)
    return manual_event(s_r * dt, dt, "release", s=s_r), manual_event(s_c * dt, dt, "catch", s=s_c)


def horizontal(e):
    return next(c for c in e.components if c.method == "tof").details["v_horizontal_mph"]


# ------------------------------------------------- manual catch = the glove
def test_manual_catch_ends_at_the_glove_even_if_the_track_ended_in_contact():
    cal = behind()
    t_rel = 4.0 + 0.3 * DT
    T = glove_flight(cal, 72.5)
    r, c = marks(t_rel, t_rel + T)
    contact_track = PitchTrack(0, [], end_kind="contact", flags=["contact"])
    e = estimate_speed(cal, SCAN, DT, r, c, track=contact_track, manual=True)
    tof = next(x for x in e.components if x.method == "tof")
    assert tof.details["end_point"] == "catch" and e.assumptions["end_point"] == "catch"
    assert tof.details["L_ft"] == pytest.approx(60.5 - 5.5 + 1.5)
    assert K.F_CONTACT not in e.flags
    assert abs(horizontal(e) / 72.5 - 1) < 0.02          # was ~6-7 % low with the 'contact' end point
    assert "glove" in (e.reason or "")
    # same marks without a track: identical number
    e0 = estimate_speed(cal, SCAN, DT, r, c, manual=True)
    assert e0.v_release_mph == pytest.approx(e.v_release_mph)


def test_automatic_contact_end_is_still_used_without_a_manual_catch():
    cal = behind()
    r = EventTime(4.0, 0.005, "first_free_field", s=240)
    c = EventTime(4.0 + 0.40, 0.004, "extrapolated", s=264)
    e = estimate_speed(cal, SCAN, DT, r, c, track=PitchTrack(0, [], end_kind="contact"))
    tof = next(x for x in e.components if x.method == "tof")
    assert tof.details["end_point"] == "contact" and K.F_CONTACT in e.flags
    # a manual release alone does not change the automatic end point
    rm = manual_event(4.0 + DT, DT, "release", s=241)
    e2 = estimate_speed(cal, SCAN, DT, rm, c, track=PitchTrack(0, [], end_kind="contact"))
    assert next(x for x in e2.components if x.method == "tof").details["end_point"] == "contact"


def test_tof_estimate_never_ends_a_manual_catch_at_contact():
    cal = behind()
    r, c = marks(3.0, 3.0 + glove_flight(cal, 80))
    comp = tof_estimate(cal, SCAN, DT, r, c, end_kind="contact", manual=True)
    assert comp.details["end_point"] == "catch" and K.F_CONTACT not in comp.details["flags"]


# ------------------------------------------------- flags after partial edits
STALE_TRACK = [K.F_CATCH_EXTRAPOLATED, K.F_FIELD_ORDER, K.F_RELEASE_UNCONFIRMED]


def test_catch_only_edit_keeps_release_unconfirmed_and_drops_catch_extrapolated():
    cal = behind()
    T = glove_flight(cal, 84.0)
    r = EventTime(2.0, 0.025, "first_free_field", s=120)
    _, c = marks(2.0, 2.0 + T)
    tr = PitchTrack(0, [], end_kind="lost", flags=list(STALE_TRACK))
    e = estimate_speed(cal, SCAN, DT, r, c, track=tr, manual=True)
    assert K.F_RELEASE_UNCONFIRMED in e.flags            # the release is still the automatic guess
    assert K.F_CATCH_EXTRAPOLATED not in e.flags         # the catch is the user's mark now
    assert K.F_FIELD_ORDER in e.flags and K.F_MANUAL in e.flags
    assert e.method == "tof"                             # not 'manual_tof': only one event was marked
    assert "only the catch was marked by hand" in (e.reason or "")


def test_release_only_edit_drops_release_unconfirmed_keeps_catch_flags():
    cal = behind()
    T = glove_flight(cal, 84.0)
    r, _ = marks(2.0, 2.0 + T)
    c = EventTime(2.0 + T, 0.004, "extrapolated", s=140)
    tr = PitchTrack(0, [], end_kind="lost", flags=list(STALE_TRACK))
    e = estimate_speed(cal, SCAN, DT, r, c, track=tr, manual=True)
    assert K.F_RELEASE_UNCONFIRMED not in e.flags and K.F_CATCH_EXTRAPOLATED in e.flags
    assert e.method == "tof" and K.F_MANUAL in e.flags
    assert "only the release was marked by hand" in (e.reason or "")


def test_both_marked_is_manual_tof_without_stale_event_flags():
    cal = behind()
    r, c = marks(2.0, 2.0 + glove_flight(cal, 84.0))
    tr = PitchTrack(0, [], end_kind="lost", flags=list(STALE_TRACK))
    e = estimate_speed(cal, SCAN, DT, r, c, track=tr, manual=True)
    assert e.method == "manual_tof"
    assert K.F_RELEASE_UNCONFIRMED not in e.flags and K.F_CATCH_EXTRAPOLATED not in e.flags
    assert K.F_FIELD_ORDER in e.flags                    # not about the events: kept


def test_tof_flags_release_unconfirmed_whenever_the_release_is_automatic():
    cal = behind()
    _, c = marks(2.0, 2.0 + glove_flight(cal, 70.0))
    r = EventTime(2.0, 0.02, "first_free_field")
    comp = tof_estimate(cal, SCAN, DT, r, c, manual=True)
    assert K.F_RELEASE_UNCONFIRMED in comp.details["flags"]


# ------------------------------------------------------ impossible settings
def test_settings_errors_explain_inches_typed_as_feet():
    errs = settings_errors(Settings(distance_ft=46.0, extension_ft=48.0))
    assert len(errs) == 1
    assert "48" in errs[0] and "FEET" in errs[0] and "inches" in errs[0] and "4.0 ft" in errs[0]
    assert settings_errors(Settings(distance_ft=46.0, extension_ft=4.0)) == []
    assert settings_errors(Settings(distance_ft=60.5, extension_ft=7.5)) == []
    assert settings_errors(Settings(distance_ft=60.5)) == []
    assert extension_limit_ft(60.5) == pytest.approx(10.0) and extension_limit_ft(35.0) == pytest.approx(8.75)
    for bad in (-1.0, 30.0, float("nan"), float("inf")):
        assert settings_errors(Settings(distance_ft=60.5, extension_ft=bad)), bad
    assert settings_errors(Settings(distance_ft=605.0))
    assert settings_errors(Settings(distance_ft=float("nan")))
    assert settings_errors(Settings(altitude_ft=float("nan")))


def test_validate_lists_impossible_settings_without_raising():
    cf = CalibrationFile(settings=Settings(distance_ft=46.0, extension_ft=48.0),
                         segments=[Segment(view="behind", points={"release": [300, 200], "catch": [360, 330]})])
    warns = cf.validate()                                # must not raise: the clicks must stay saveable
    assert any("IMPOSSIBLE SETTING" in w and "inches" in w for w in warns)
    ok = CalibrationFile(settings=Settings(distance_ft=46.0, extension_ft=4.0), segments=cf.segments)
    assert not any("IMPOSSIBLE" in w for w in ok.validate())
    odd = CalibrationFile(settings=Settings(distance_ft=60.5, extension_ft=9.0), segments=cf.segments)
    assert any("unusual" in w for w in odd.validate())


@pytest.mark.parametrize("ext", [30.0, 48.0, 60.0, 66.0, -2.0, float("nan")])
def test_behind_view_impossible_extension_gives_no_number(ext):
    cal = behind(extension_ft=ext)
    r = EventTime(2.0, 0.005, "first_free_field")
    c = EventTime(2.45, 0.004, "extrapolated")
    e = estimate_speed(cal, SCAN, DT, r, c)
    assert e.v_release_mph is None and e.label == "none" and e.sigma_mph is None
    assert "extension" in e.reason and K.F_INVALID_SETTINGS in e.flags


@pytest.mark.parametrize("ext", [44.0, 48.0, 60.0])
def test_side_view_impossible_extension_never_crashes(ext):
    rng = np.random.default_rng(3)
    fl = H.simulate_pitches("45FB46", 1, rng)[0]
    cam = H.side_camera("perp90", fl.D, 10 / 11)
    cal = H.side_calibration(cam, fl.D, rng, 10 / 11, 2.0)
    tr = H.make_track(fl, cam, 3.0, rng, sar=10 / 11)
    r, c = H.manual_pair(3.0, 3.0 + fl.T, DT)
    good = estimate_speed(cal, SCAN, DT, r, c, track=tr)
    assert good.v_release_mph is not None
    bad = copy.deepcopy(cal)
    bad.settings.extension_ft = ext
    e = estimate_speed(bad, SCAN, DT, r, c, track=tr)
    assert e.v_release_mph is None and e.label == "none" and "extension" in e.reason
    # the E2 piece on its own refuses a release point at/behind the plate instead of overflowing
    if fl.D - ext + C.CATCH_BEHIND_FT <= 5.0:
        with pytest.raises(K.NotMeasurable, match="distance"):
            side_fit_estimate(bad, SCAN, DT, tr, release=r, catch=c)


def test_lateral_budget_survives_an_undetermined_rho():
    rng = np.random.default_rng(4)
    cam = H.side_camera("perp90", 60.5, 10 / 11)
    cal = H.side_calibration(cam, 60.5, rng, 10 / 11, 2.0)
    geom = SF.axis_geometry(cal)
    for s_ln in (1e3, 1e6, float("inf")):              # exp(s_ln) used to overflow
        lat = SF.lateral_budget(cal, geom, 1.0, s_ln, 55.0, 0.0, 1.8, 4.0)
        assert not math.isfinite(lat.rel_sigma) or lat.rel_sigma > 0.05


@pytest.mark.parametrize("dt", [0.0, -DT, float("nan"), None])
def test_bad_sample_interval_gives_no_number(dt):
    e = estimate_speed(behind(), SCAN, dt, EventTime(1.0, 0.005, "first_free_field"),
                       EventTime(1.45, 0.004, "extrapolated"))
    assert e.v_release_mph is None and e.label == "none" and "sample interval" in e.reason


def test_bad_event_numbers_give_no_number():
    cal = behind()
    nan_t = estimate_speed(cal, SCAN, DT, EventTime(float("nan"), 0.005, "first_free_field"),
                           EventTime(1.45, 0.004, "extrapolated"))
    assert nan_t.v_release_mph is None and "not a number" in nan_t.reason
    inf_sig = estimate_speed(cal, SCAN, DT, EventTime(1.0, float("inf"), "first_free_field"),
                             EventTime(1.45, 0.004, "extrapolated"))
    assert inf_sig.v_release_mph is None and inf_sig.label == "none"
    no_cal = estimate_speed(None, SCAN, DT, EventTime(1.0, 0.005, "manual"), EventTime(1.45, 0.005, "manual"))
    assert no_cal.v_release_mph is None and "calibration" in no_cal.reason


def test_an_estimator_failure_becomes_a_reason_not_a_crash(monkeypatch):
    def boom(*a, **k):
        raise OverflowError("math range error")
    cal = behind()
    r, c = marks(2.0, 2.0 + glove_flight(cal, 80.0))
    monkeypatch.setattr(SPEED, "tof_estimate", boom)
    e = estimate_speed(cal, SCAN, DT, r, c)
    assert e.v_release_mph is None and e.label == "none" and "OverflowError" in e.reason
    monkeypatch.setattr(SPEED, "combine", boom)          # even a failure outside the estimators
    e = estimate_speed(cal, SCAN, DT, r, c)
    assert e.v_release_mph is None and e.label == "none" and "failed" in e.reason


def test_side_fit_failure_falls_back_to_time_of_flight(monkeypatch):
    rng = np.random.default_rng(5)
    fl = H.simulate_pitches("75FB", 1, rng)[0]
    cam = H.side_camera("perp90", fl.D, 10 / 11)
    cal = H.side_calibration(cam, fl.D, rng, 10 / 11, 2.0)
    tr = H.make_track(fl, cam, 3.0, rng, sar=10 / 11)
    r, c = H.manual_pair(3.0, 3.0 + fl.T, DT)

    def boom(*a, **k):
        raise FloatingPointError("bad track")
    monkeypatch.setattr(SPEED, "side_fit_estimate", boom)
    e = estimate_speed(cal, SCAN, DT, r, c, track=tr)
    assert e.method == "manual_tof" and e.v_release_mph is not None
    assert "side-view fit failed (FloatingPointError" in e.reason


# ---------------------------------------------------- manual-mark sigma
def test_manual_sigma_has_an_identification_term():
    assert MANUAL_SLIP_PROB == pytest.approx(0.25)
    assert manual_sigma(DT) == pytest.approx(DT * math.sqrt(1 / 12 + 0.25))
    # a manual_event mark (quantization sigma) is used with the identification term added
    s, floored = K.event_sigma(manual_event(2.0, DT, "release"), DT)
    assert s == pytest.approx(manual_sigma(DT)) and not floored
    # hand marks built elsewhere (e.g. re-tracking with sigma dt/2) get it too, silently
    s, floored = K.event_sigma(EventTime(1.0, DT / 2, "manual"), DT)
    assert s == pytest.approx(math.hypot(DT / 2, DT / 2)) and not floored
    s, floored = K.event_sigma(EventTime(1.0, None, "manual"), DT)
    assert s == pytest.approx(math.hypot(DT / 2, DT / 2)) and not floored
    r, c = marks(2.0, 2.45)
    comp = tof_estimate(behind(), SCAN, DT, r, c, manual=True)
    assert comp.details["sigma_release_s"] == pytest.approx(manual_sigma(DT))
    assert comp.details["sigma_T_s"] == pytest.approx(math.sqrt(2) * manual_sigma(DT))


def test_manual_timing_term_covers_one_slipped_field():
    """A one-field slip of one mark stays inside the 90 % interval at 90 mph."""
    cal = behind()
    t_rel = 2.0 + 0.5 * DT
    T = glove_flight(cal, 90.0)
    r, c = marks(t_rel, t_rel + T)
    e = estimate_speed(cal, SCAN, DT, r, c)
    late = manual_event((r.s + 1) * DT, DT, "release", s=r.s + 1)
    e_late = estimate_speed(cal, SCAN, DT, late, c)
    lo, hi = e_late.ci90_mph
    assert lo <= e.v_release_mph <= hi
    assert e.label in ("good", "fair")


def test_manual_behind_view_coverage_with_simulated_slips():
    st = H.run_behind("manual", "90FB", 300, seed=41)
    print(st.line("behind manual 90FB (slips)"))
    assert st.n_none == 0 and abs(st.bias) < 0.8
    assert 0.86 <= st.coverage <= 0.97, st.line("coverage")
    exact = H.run_behind("manual", "90FB", 300, seed=41, slip_prob=0.0)
    print(exact.line("behind manual 90FB (exact)"))
    assert 0.88 <= exact.coverage <= 0.995, exact.line("coverage")        # conservative, not absurd
