"""Unit tests: detection primitives (kernels, NMS, median, gain, noise) and the Detector."""
import math

import numpy as np
import pytest

from pitchspeed.vision import geometry as G
from pitchspeed.vision.detect import (Detector, block_noise, kernel_bank, line_kernel, median_stack, nms_peaks,
                                      robust_gain)
from pitchspeed.vision.options import FinderOptions
from vision_helpers import side_calibration


def _moments(k):
    w = np.maximum(k, 0)
    ii, jj = np.mgrid[:k.shape[0], :k.shape[1]]
    cy, cx = (w * ii).sum() / w.sum(), (w * jj).sum() / w.sum()
    return cx, cy, (w * (jj - cx) ** 2).sum() / w.sum(), (w * (ii - cy) ** 2).sum() / w.sum()


def test_line_kernel_zero_mean_unit_norm_symmetric():
    for L, a in ((0.0, 0.0), (10.0, 0.3), (20.0, math.pi / 2)):
        k = line_kernel(L, a, 1.0, sar=10 / 11, row_scale=2)
        assert abs(float(k.sum())) < 1e-4
        assert abs(float((k * k).sum()) - 1.0) < 1e-4
        np.testing.assert_allclose(k, k[::-1, ::-1], atol=1e-6)   # point symmetric -> correlation = convolution


def test_line_kernel_elongated_along_angle():
    k = line_kernel(16.0, 0.0, 1.0, sar=1.0, row_scale=1)
    _, _, vx, vy = _moments(k)
    assert vx > 5 * vy
    kv = line_kernel(16.0, math.pi / 2, 1.0, sar=1.0, row_scale=1)
    _, _, vx2, vy2 = _moments(kv)
    assert vy2 > 5 * vx2


def test_kernel_bank_side_and_behind():
    cal = side_calibration().segment_calibs()[0]
    geom = G.view_geometry(cal, 1001 / 60000)
    bank = kernel_bank(geom, 2)
    assert len(bank) == 8
    assert bank[0].length == 0.0
    assert all(3.0 <= b.length <= 30.0 for b in bank[1:])


def test_nms_peaks_suppresses_neighbours_and_orders():
    z = np.zeros((40, 80), np.float32)
    z[10, 10] = 9.0
    z[11, 13] = 8.0          # inside the 5x11 window of the first peak -> suppressed
    z[30, 60] = 6.0
    z[30, 40] = 3.0          # below threshold
    r, c, v = nms_peaks(z, (5, 11), thr=3.5, max_n=10)
    assert list(zip(r.tolist(), c.tolist())) == [(10, 10), (30, 60)]
    assert v[0] > v[1]
    r, c, v = nms_peaks(z, (5, 11), thr=3.5, max_n=1)
    assert len(r) == 1


def test_nms_plateau_gives_one_peak_per_window():
    z = np.zeros((20, 60), np.float32)
    z[10, 10:40] = 5.0       # a flat ridge
    r, c, v = nms_peaks(z, (5, 11), thr=3.5, max_n=30)
    assert 2 <= len(r) <= 3
    assert np.all(np.diff(np.sort(c)) >= 11)


@pytest.mark.parametrize("n", [3, 4, 7, 8])
def test_median_stack_matches_numpy(n):
    rng = np.random.default_rng(n)
    imgs = [rng.integers(0, 255, (17, 23), dtype=np.uint8) for _ in range(n)]
    np.testing.assert_allclose(median_stack(imgs), np.median(np.stack(imgs).astype(np.float64), axis=0), atol=1e-4)


def test_robust_gain_recovers_agc_with_outliers():
    rng = np.random.default_rng(0)
    bg = rng.uniform(20, 220, (60, 200)).astype(np.float32)
    cur = 1.12 * bg - 7.0 + rng.normal(0, 1.0, bg.shape)
    cur[10:30, 50:90] = 250.0         # a moving player
    a, b = robust_gain(cur.astype(np.float32), bg)
    assert abs(a - 1.12) < 0.02 and abs(b + 7.0) < 2.0


def test_block_noise_estimates_sigma_and_floor():
    rng = np.random.default_rng(1)
    R = rng.normal(0, 3.0, (48, 256)).astype(np.float32)
    sig, g = block_noise(R, (8, 32))
    assert abs(g - 3.0) < 0.3
    assert abs(float(np.median(sig)) - 3.0) < 0.5
    sig0, g0 = block_noise(np.zeros((48, 256), np.float32), (8, 32), floor=0.5)
    assert g0 >= 0.5 and float(sig0.min()) >= 0.5


def _field_sequence(n=30, ball=(400.0, 270.0), amp=60.0, sign=1, blob=False, seed=0):
    """Static textured fields (identity transforms) with a ball in field 15 only."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(80, 120, (240, 720)).astype(np.float32)
    out = []
    for s in range(n):
        p = s & 1
        img = base + rng.normal(0, 2.0, base.shape)
        if s == 15:
            yy, xx = np.mgrid[0:240, 0:720]
            fy = (ball[1] - p) / 2.0
            img += sign * amp * np.exp(-((xx - ball[0]) ** 2 / (2 * 1.2 ** 2) + (yy - fy) ** 2 / (2 * 0.6 ** 2)))
            if blob:   # an arm-sized moving region touching the ball
                img[int(fy) - 3:int(fy) + 4, int(ball[0]) + 2:int(ball[0]) + 26] += sign * 50
        out.append((s, s * 1001 / 60000, s & 1, p, np.clip(img, 0, 255).astype(np.uint8)))
    return out


def _run_detector(seq, **kw):
    cal = side_calibration().segment_calibs()[0]
    geom = G.view_geometry(cal, 1001 / 60000)
    det = Detector(geom, True, (240, 720), FinderOptions(**kw))
    frames = []
    for s, t, k, p, img in seq:
        frames += det.push(s, t, k, p, img, np.eye(3)[:2], 1.0)
    frames += det.flush()
    return {f.s: f for f in frames}


@pytest.mark.parametrize("sign", [1, -1])
def test_detector_finds_ball_both_signs_subpixel(sign):
    ball = (400.3, 271.0)
    out = _run_detector(_field_sequence(ball=ball, sign=sign))
    c = out[15].cands
    assert len(c) >= 1
    best = c[np.argmin(np.hypot(c["x"] - ball[0], c["y"] - ball[1]))]
    assert abs(best["x"] - ball[0]) < 0.5 and abs(best["y"] - ball[1]) < 0.8
    assert best["sign"] == sign and best["z"] > 8 and not best["attached"]
    # no strong detections in fields without the ball
    assert all((f.cands["z"] < 8).all() for s, f in out.items() if s != 15 and len(f.cands))


def test_detector_attached_flag_for_ball_touching_body():
    ball = (400.0, 271.0)
    out = _run_detector(_field_sequence(ball=ball, blob=True))
    c = out[15].cands
    near = c[np.hypot(c["x"] - ball[0], c["y"] - ball[1]) < 5]   # centroid is pulled toward the body
    assert len(near) and near["attached"].all()
    assert out[15].blobs                     # the big blob is reported in reference px


def test_detector_mask_drops_candidates():
    ball = (400.0, 271.0)
    seq = _field_sequence(ball=ball)
    cal = side_calibration().segment_calibs()[0]
    geom = G.view_geometry(cal, 1001 / 60000)
    det = Detector(geom, True, (240, 720), FinderOptions(), ref_masks=[(380, 250, 420, 290)])
    frames = []
    for s, t, k, p, img in seq:
        frames += det.push(s, t, k, p, img, np.eye(3)[:2], 1.0)
    frames += det.flush()
    f15 = [f for f in frames if f.s == 15][0]
    assert not np.any(np.hypot(f15.cands["x"] - ball[0], f15.cands["y"] - ball[1]) < 5)


def test_detector_skips_unregistered_sample():
    seq = _field_sequence()
    cal = side_calibration().segment_calibs()[0]
    geom = G.view_geometry(cal, 1001 / 60000)
    det = Detector(geom, True, (240, 720), FinderOptions())
    frames = []
    for s, t, k, p, img in seq:
        frames += det.push(s, t, k, p, img, np.eye(3)[:2], 0.0 if s == 15 else 1.0)
    frames += det.flush()
    f15 = [f for f in frames if f.s == 15][0]
    assert f15.skipped == "no_registration" and len(f15.cands) == 0
