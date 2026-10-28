"""E2 side-view fit: geometry, exact recovery, robustness, checks."""
import math

import numpy as np
import pytest

from pitchspeed import constants as C
from pitchspeed import physics
from pitchspeed.calib.model import Calibration, Segment, Settings
from pitchspeed.estimate import NotMeasurable, side_fit_estimate
from pitchspeed.estimate import common as K
from pitchspeed.estimate import sidefit as SF
from pitchspeed.types import EventTime, PitchTrack, ScanInfo, TrackPoint

import estimate_helpers as H

DT = C.NTSC_FIELD_DT
SCAN = ScanInfo("interlaced", "bff", "high", "content", 1 / DT)
P_SQ = np.array([90.0, 330.0])
R_SQ = np.array([600.0, 310.0])


def calib_sq(sar=1.0, D=60.5, **st):
    pts = {"plate": [P_SQ[0] / sar, P_SQ[1]], "rubber": [R_SQ[0] / sar, R_SQ[1]]}
    return Calibration(Settings(distance_ft=D, sar=sar, **st), Segment(view="side", points=pts), (720, 480))


def model_track(cal: Calibration, v0_mph=80.0, rho=1.0, t_r=1.003, n_skip=2, h_px=30.0, up=(0.0, -1.0),
                noise=0.0, rng=None, extra=None) -> PitchTrack:
    """Image points that follow the 1-D model exactly (plus optional noise)."""
    k = physics.DragModel.default().k
    v0 = v0_mph * C.MPH_TO_FTPS
    y_r = cal.release_y_ft
    T = physics.time_to_travel(v0, y_r - cal.catch_y_ft, k)
    s = np.arange(math.floor(t_r / DT) + 1 + n_skip, math.ceil((t_r + T) / DT) - 1)
    t = s * DT
    Y = y_r - np.log1p(k * v0 * (t - t_r)) / k
    mu = Calibration.mu_of_lam(cal.lam_of_y(Y), rho)
    u = np.asarray(up, float) / np.linalg.norm(up)
    xy = P_SQ + mu[:, None] * (R_SQ - P_SQ) + h_px * u
    if noise:
        xy = xy + rng.normal(0, noise, xy.shape)
    pts = [TrackPoint(s=int(si), t=float(ti), x=float(x / cal.sar), y=float(y)) for si, ti, (x, y) in zip(s, t, xy)]
    if extra:
        pts += extra
    return PitchTrack(track_id=1, points=pts)


def test_plumb_projection_is_exact():
    cal = calib_sq()
    g = SF.axis_geometry(cal)
    mu = np.array([-0.05, 0.0, 0.3, 0.97, 1.1])
    for up in ([0.0, -1.0], [0.08, -1.0]):
        u = np.array(up) / np.linalg.norm(up)
        g2 = SF.AxisGeometry(g.P, g.R, u, g.y_P, g.y_R)
        pts = g.P + mu[:, None] * (g.R - g.P) + np.array([[10.0], [55.0], [0.0], [-20.0], [80.0]]) * u
        assert np.allclose(g2.mu(pts), mu, atol=1e-12)


@pytest.mark.parametrize("rho", [0.7, 1.0, 1.6])
@pytest.mark.parametrize("mph", [45.0, 85.0])
def test_fit_recovers_the_exact_model(rho, mph):
    cal = calib_sq()
    tr = model_track(cal, v0_mph=mph, rho=rho)
    comp = side_fit_estimate(cal, SCAN, DT, tr)
    d = comp.details
    assert d["v_horizontal_mph"] == pytest.approx(mph, rel=2e-4)
    assert d["rho"] == pytest.approx(rho, rel=2e-3)
    assert d["t_release_fit"] == pytest.approx(1.003, abs=5e-4)
    assert d["rms_px"] < 1e-3
    assert d["field_order_used"] == "scan"
    assert comp.v_release_mph > mph * 0.98       # slope + small tilt-prior correction only


def test_non_square_pixels_give_the_same_answer():
    a = side_fit_estimate(calib_sq(sar=1.0), SCAN, DT, model_track(calib_sq(sar=1.0)))
    b = side_fit_estimate(calib_sq(sar=10 / 11), SCAN, DT, model_track(calib_sq(sar=10 / 11)))
    assert a.details["v_horizontal_mph"] == pytest.approx(b.details["v_horizontal_mph"], rel=1e-6)
    # (the focal prior scales with the image width in square px, so the small
    # tilt-prior correction differs slightly)
    assert a.v_release_mph == pytest.approx(b.v_release_mph, rel=1e-3)
    # sigma differs only via the click perturbation (2 stored px is 2*sar square px in x)
    assert b.sigma_ind_mph == pytest.approx(a.sigma_ind_mph, rel=0.1)


def test_tilted_vertical_reference_is_used():
    up = (0.1, -1.0)                                  # camera rolled ~5.7 deg
    cal = calib_sq()
    bottom = np.array([290.0, 430.0])                 # a post somewhere in the picture
    u = np.array(up) / np.linalg.norm(up)
    cal.segment.points["vertical_bottom"] = bottom.tolist()
    cal.segment.points["vertical_top"] = (bottom + 150 * u).tolist()
    comp = side_fit_estimate(cal, SCAN, DT, model_track(cal, up=up, h_px=60.0))
    assert comp.details["v_horizontal_mph"] == pytest.approx(80.0, rel=2e-4)
    plain = calib_sq()
    off = side_fit_estimate(plain, SCAN, DT, model_track(plain, up=up, h_px=60.0))
    assert abs(off.details["v_horizontal_mph"] - 80.0) > 0.05      # columns as vertical: biased


def test_wrong_field_order_is_detected_and_corrected():
    rng = np.random.default_rng(3)
    fl = H.simulate_pitches("90FB", 1, rng)[0]
    cam = H.side_camera("perp90", fl.D)
    cal = H.side_calibration(cam, fl.D, rng, 1.0, 0.0)
    good = side_fit_estimate(cal, SCAN, DT, H.make_track(fl, cam, 1.0, rng, sigma_px=0.5))
    bad = side_fit_estimate(cal, SCAN, DT, H.make_track(fl, cam, 1.0, rng, sigma_px=0.5, swap_order=True))
    assert K.F_FIELD_ORDER not in good.details["flags"]
    assert K.F_FIELD_ORDER in bad.details["flags"] and bad.details["field_order_used"] == "swapped"
    assert bad.v_release_mph == pytest.approx(good.v_release_mph, rel=0.01)
    assert bad.details["rms_px_other_field_order"] > 5 * bad.details["rms_px"]


def test_progressive_scan_skips_the_field_order_test():
    prog = ScanInfo("progressive", None, "high", "content", 29.97)
    cal = calib_sq()
    comp = side_fit_estimate(cal, prog, DT, model_track(cal))
    assert math.isnan(comp.details["rms_px_other_field_order"])
    assert K.F_FRAMES_ONLY in comp.details["flags"]


def test_too_few_points_or_wrong_view_raise():
    cal = calib_sq()
    tr = model_track(cal)
    tr.points = tr.points[:7]
    with pytest.raises(NotMeasurable):
        side_fit_estimate(cal, SCAN, DT, tr)
    with pytest.raises(NotMeasurable):                 # all points marked outliers by the tracker
        side_fit_estimate(cal, SCAN, DT, PitchTrack(0, [TrackPoint(1, 0.1, 1, 1, inlier=False)] * 20))
    beh = Calibration(Settings(), Segment(view="behind", points={"release": [1, 1], "catch": [50, 50]}))
    with pytest.raises(NotMeasurable):
        side_fit_estimate(beh, SCAN, DT, model_track(cal))


def test_points_outside_the_events_are_ignored():
    cal = calib_sq()
    base = model_track(cal)
    last = base.points[-1]
    glove = [TrackPoint(s=last.s + i, t=last.t + i * DT, x=last.x, y=last.y) for i in range(1, 8)]
    tr = model_track(cal, extra=glove)
    catch = EventTime(last.t + 0.5 * DT, 0.003, "extrapolated")
    comp = side_fit_estimate(cal, SCAN, DT, tr, catch=catch)
    assert comp.details["n_outside_events"] == 7
    assert comp.details["v_horizontal_mph"] == pytest.approx(80.0, rel=2e-4)


def test_gross_outliers_are_rejected():
    rng = np.random.default_rng(1)
    cal = calib_sq()
    tr = model_track(cal, noise=0.7, rng=rng)
    for i in (3, 9, 15, 21):
        tr.points[i].x += 25.0          # along the line (a vertical offset is removed by the plumb projection)
    comp = side_fit_estimate(cal, SCAN, DT, tr)
    assert comp.details["n_used"] == len(tr.points) - 4
    assert comp.details["v_horizontal_mph"] == pytest.approx(80.0, rel=0.01)


def test_partial_track_is_flagged():
    cal = calib_sq()
    tr = model_track(cal)
    tr.points = tr.points[len(tr.points) // 2:]
    comp = side_fit_estimate(cal, SCAN, DT, tr)
    assert K.F_PARTIAL_TRACK in comp.details["flags"]
    assert comp.details["coverage"] < 0.6


def test_very_oblique_view_is_rejected():
    rng = np.random.default_rng(2)
    fl = H.simulate_pitches("75FB", 1, rng)[0]
    cam = H.side_camera("deep-oblique", fl.D)
    cal = H.side_calibration(cam, fl.D, rng, 1.0, 2.0)
    comp = side_fit_estimate(cal, SCAN, DT, H.make_track(fl, cam, 1.0, rng))
    assert comp.details["usable"] is False
    assert K.F_OBLIQUE in comp.details["flags"]
    assert comp.details["phi_deg"] > 50
    assert "oblique" in comp.details["reason"]


def test_obliquity_geometry_with_the_true_focal_length():
    for scene in ("elev15ft", "far-oblique", "mound-side"):
        cam = H.side_camera(scene, 60.5)
        cal = H.side_calibration(cam, 60.5, np.random.default_rng(0), 1.0, 0.0)
        g = SF.axis_geometry(cal)
        zP = cam.depth(np.array([[0, C.PLATE_CENTER_Y_FT, 0.0]]))[0]
        zR = cam.depth(np.array([[0, 60.75, 10 / 12]]))[0]
        t, zmid = SF.tan_phi_and_depth(g, zP / zR, cam.f, np.array([cam.cx, cam.cy]))
        assert abs(t) == pytest.approx(cam.tan_phi, rel=0.08)
        assert np.sign(t) == np.sign(1 - zP / zR)
        assert zmid == pytest.approx((zP + zR) / 2, rel=0.08)


def test_apparent_height_ratio_tracks_obliquity():
    rng = np.random.default_rng(4)
    fl = H.simulate_pitches("75FB", 1, rng)[0]
    for scene, expect in (("perp90", 1.0), ("deep-oblique", 2.47)):
        cam = H.side_camera(scene, fl.D)
        cal = H.side_calibration(cam, fl.D, rng, 1.0, 0.0)
        comp = side_fit_estimate(cal, SCAN, DT, H.make_track(fl, cam, 1.0, rng, sigma_px=0.3))
        assert comp.details["height_ratio"] == pytest.approx(expect, rel=0.2)


def test_camera_side_and_release_side():
    g = SF.axis_geometry(calib_sq())                          # rubber right of plate
    assert SF.camera_side(g) == "first_base"
    assert SF.release_side(calib_sq(pitcher_throws="R"), g) == (-SF.RELEASE_SIDE_FT, SF.SIGMA_XR_KNOWN_FT, True)
    assert SF.release_side(calib_sq(pitcher_throws="L"), g)[0] == SF.RELEASE_SIDE_FT
    assert SF.release_side(calib_sq(), g) == (0.0, SF.SIGMA_XR_UNKNOWN_FT, False)
    flipped = SF.AxisGeometry(g.R, g.P, g.up, g.y_P, g.y_R)
    assert SF.camera_side(flipped) == "third_base"


def test_uncertainty_terms_are_reported():
    rng = np.random.default_rng(5)
    cal = calib_sq()
    comp = side_fit_estimate(cal, SCAN, DT, model_track(cal, noise=1.0, rng=rng))
    d = comp.details
    parts = [d["sigma_cov_mph"], d["sigma_click_mph"], d["sigma_up_mph"],
             comp.v_release_mph * d["rel_sigma_lateral"], comp.v_release_mph * d["rel_sigma_extension"],
             comp.v_release_mph * d["rel_sigma_model"]]
    assert comp.sigma_ind_mph == pytest.approx(math.sqrt(sum(p * p for p in parts)), rel=1e-6)
    assert comp.sigma_shared_mph == pytest.approx(abs(d["dv_k_mph"]))
    assert d["dv_k_mph"] > 0                               # same direction as E1
    assert 0.001 < d["rel_sigma_lateral"] < 0.05
