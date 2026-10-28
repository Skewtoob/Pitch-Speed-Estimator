"""Unit tests: trajectory model fitting, prediction and geometry helpers."""
import math

import numpy as np

from pitchspeed.vision import geometry as G
from pitchspeed.vision.model import eval_model, fit_model, local_predict, model_from_dict
from vision_helpers import side_calibration

DT = 1001 / 60000


def _behind_traj(t):
    Z = 70 - 120 * t
    return 360 + 900 * (1 + 0.5 * t) / Z, 240 + 900 * (-5 + 2 * t + 16 * t * t) / Z


def test_rational_fit_recovers_projective_track_with_outliers():
    rng = np.random.default_rng(0)
    t = np.arange(30) * DT
    u, v = _behind_traj(t)
    x, y = u + rng.normal(0, 0.3, 30), v + rng.normal(0, 0.3, 30)
    x[5] += 20
    y[12] -= 15
    m = fit_model(t, x, y, view="behind")
    assert m.kind == "rational"
    assert not m.inliers[5] and not m.inliers[12] and m.inliers.sum() >= 26
    assert m.rms < 0.6
    te = t[-1] + 2 * DT
    assert np.hypot(*(m.eval([te])[0] - np.array(_behind_traj(te)))) < 2.0
    assert abs(m.pole_time() - 70 / 120) < 0.01      # ball reaches the camera plane


def test_side_view_prefers_simple_model_and_dict_roundtrip():
    rng = np.random.default_rng(1)
    t = np.arange(30) * DT
    x = 600 - 1100 * t + 40 * t * t + rng.normal(0, 0.4, 30)
    y = 230 + 20 * t + 300 * t * t + rng.normal(0, 0.3, 30)
    m = fit_model(t, x, y, view="side")
    assert m.inliers.all() and m.rms < 0.8
    d = m.to_dict()
    np.testing.assert_allclose(eval_model(d, t), m.eval(t), atol=1e-9)
    assert model_from_dict(d).kind == m.kind


def test_pos_sigma_grows_with_extrapolation():
    rng = np.random.default_rng(2)
    t = np.arange(20) * DT
    m = fit_model(t, 100 + 900 * t + rng.normal(0, 0.5, 20), 50 + rng.normal(0, 0.5, 20), view="side")
    s = m.pos_sigma([t[10], t[-1], t[-1] + 10 * DT])
    assert s[0] < s[1] < s[2]


def test_fit_model_small_inputs():
    assert fit_model([0, 1], [0, 1], [0, 1]) is None
    m = fit_model(np.arange(3) * DT, [0.0, 10.0, 20.0], [5.0, 5.0, 5.0])
    assert m is not None and m.kind == "linear"
    np.testing.assert_allclose(m.eval([3 * DT])[0], [30.0, 5.0], atol=1e-6)


def test_fixed_points_stay_inliers():
    t = np.arange(12) * DT
    x = 10 + 600 * t
    y = 20 + 0 * t
    x2 = x.copy()
    x2[0] += 8.0                          # a "manual" point the user insists on
    fixed = np.zeros(12, bool)
    fixed[0] = True
    m = fit_model(t, x2, y, kind="poly", fixed=fixed)
    assert m.inliers[0]


def test_local_predict_linear_motion():
    t = np.arange(6) * DT
    pos, sig, vel = local_predict(t, 100 + 1200 * t, 50 + 60 * t, 6 * DT)
    np.testing.assert_allclose(pos, [100 + 1200 * 6 * DT, 50 + 60 * 6 * DT], atol=1e-6)
    np.testing.assert_allclose(vel, [1200, 60], rtol=1e-6)
    assert sig >= 0


def test_half_transform_squares_to_transform():
    A = G.similarity_from_params(1.3, -0.7, 0.4, 1.002, sar=10 / 11)
    H = G.half_transform(A, sar=10 / 11)
    np.testing.assert_allclose(G.to3(H) @ G.to3(H), G.to3(A), atol=1e-9)
    p = G.similarity_params(A, sar=10 / 11)
    assert abs(p["angle_deg"] - 0.4) < 1e-9 and abs(p["scale"] - 1.002) < 1e-9


def test_view_geometry_side():
    cal = side_calibration().segment_calibs()[0]
    g = G.view_geometry(cal, DT)
    assert abs(g.progress(cal.release_point())[0]) < 1e-9
    assert abs(g.progress(cal.catch_point())[0] - 1.0) < 1e-9
    assert abs(g.scale_rel - cal.px_per_ft_axis()) < 1e-9
    assert g.in_corridor(0.5 * (cal.release_point() + cal.catch_point()))[0]
    assert not g.in_corridor([[360.0, 20.0]])[0]
    assert 1.2 <= g.d_rel < 4
    # 85 mph at this scale is inside the image-speed gate
    v85 = 85 * 22 / 15 * g.scale_rel * DT
    assert g.v_min_px < v85 < g.v_max_px


def test_polygon_mask_field_rows():
    poly = np.array([[10.0, 20.0], [50.0, 20.0], [50.0, 40.0], [10.0, 40.0]])
    m = G.polygon_mask(poly, (40, 80), row_scale=2, parity=0)
    rows = np.nonzero(m.any(axis=1))[0]
    assert rows.min() == 10 and rows.max() == 20
