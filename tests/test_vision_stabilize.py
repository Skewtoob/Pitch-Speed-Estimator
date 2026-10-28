"""Unit tests: global motion compensation (Stabilizer)."""
import math

import cv2
import numpy as np

from pitchspeed.vision import geometry as G
from pitchspeed.vision.stabilize import Stabilizer
from vision_helpers import textured_background

SAR = 10 / 11
PTS = np.array([[120.0, 100.0], [600.0, 120.0], [360.0, 240.0], [150.0, 400.0], [620.0, 380.0]])


def _render(bg, A_s_to_ref, noise=1.0, rng=None, overlay=None):
    """Frame whose pixel x shows bg at A(x) (A: frame px -> background/reference px)."""
    A = G.to3(A_s_to_ref)
    if G.is_affine(A):
        img = cv2.warpAffine(bg, A[:2], (720, 480), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REFLECT)
    else:
        img = cv2.warpPerspective(bg, A, (720, 480), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_REFLECT)
    if rng is not None:
        img = img + rng.normal(0, noise, img.shape)
    if overlay is not None:
        x0, y0, x1, y1 = overlay
        img[y0:y1, x0:x1] = 30
        for xx in range(x0 + 3, x1 - 3, 9):           # crisp "digits"
            img[y0 + 4:y1 - 4, xx:xx + 4] = 240
    return np.clip(img, 0, 255).astype(np.uint8)


def _err(A_est, A_true):
    return float(np.max(np.hypot(*(G.apply(A_est, PTS) - G.apply(A_true, PTS)).T)))


def test_similarity_shake_registered_to_subpixel():
    rng = np.random.default_rng(0)
    bg = textured_background(seed=3)
    ref = _render(bg, np.eye(3), rng=rng)
    st = Stabilizer(ref, SAR, interlaced=True)
    errs = []
    for s in range(24):
        A = G.similarity_from_params(3.0 * math.sin(s / 4), 2.0 * math.cos(s / 5), 0.2 * math.sin(s / 7),
                                     1.0 + 0.002 * math.sin(s / 3), SAR)
        frame = _render(bg, A, rng=rng)
        p = s & 1
        r = st.register(frame[p::2], p)
        assert r.ok and r.quality > 0.3
        errs.append(_err(r.A, A))
    assert np.median(errs) < 0.15 and max(errs) < 0.4


def test_pan_with_perspective_uses_homography():
    rng = np.random.default_rng(1)
    bg = textured_background(seed=4)
    f = 500.0
    K = np.array([[f / SAR, 0, 359.5], [0, f, 239.5], [0, 0, 1]])   # square pixels in x * sar
    ref = _render(bg, np.eye(3), rng=rng)
    st = Stabilizer(ref, SAR, interlaced=True)
    errs, used_h = [], 0
    for s in range(40):
        yaw = math.radians(5.0) * s / 39.0                  # slow 5 degree pan
        R = cv2.Rodrigues(np.array([0.0, yaw, 0.0]))[0]
        H = K @ R @ np.linalg.inv(K)                         # sample frame -> reference
        H /= H[2, 2]
        frame = _render(bg, H, rng=rng)
        p = s & 1
        r = st.register(frame[p::2], p)
        assert r.ok
        used_h += int(r.homography)
        errs.append(float(np.max(np.hypot(*(G.apply(r.A, PTS[2:3]) - G.apply(H, PTS[2:3])).T))))
    assert used_h > 5
    assert max(errs[-10:]) < 0.6          # centre of the image, after a 4-5 degree pan


def test_static_overlay_is_flagged_and_ignored():
    rng = np.random.default_rng(2)
    bg = textured_background(seed=5)
    box = (470, 400, 640, 450)
    ref = _render(bg, np.eye(3), rng=rng, overlay=box)
    st = Stabilizer(ref, SAR, interlaced=True)
    errs = []
    for s in range(30):
        A = G.similarity_from_params(4.0 * math.sin(s / 3.0), 3.0 * math.cos(s / 4.0), 0.0, 1.0, SAR)
        frame = _render(bg, A, rng=rng, overlay=box)
        p = s & 1
        r = st.register(frame[p::2], p)
        errs.append(_err(r.A, A))
    boxes = st.overlay_boxes()
    assert boxes, "date-stamp-like overlay not flagged"
    b = boxes[0]
    assert b[0] <= box[2] and b[2] >= box[0] and b[1] <= box[3] and b[3] >= box[1]
    assert np.median(errs[10:]) < 0.3


def test_blank_sample_holds_previous_transform():
    rng = np.random.default_rng(3)
    bg = textured_background(seed=6)
    ref = _render(bg, np.eye(3), rng=rng)
    st = Stabilizer(ref, SAR, interlaced=True)
    r0 = st.register(ref[0::2], 0)
    assert r0.ok
    r1 = st.register(np.full((240, 720), 128, np.uint8), 1)
    assert not r1.ok and r1.quality == 0.0 and r1.method == "hold"
    np.testing.assert_allclose(r1.A, r0.A)


def test_progressive_frames():
    rng = np.random.default_rng(4)
    bg = textured_background(seed=7)
    ref = _render(bg, np.eye(3), rng=rng)
    st = Stabilizer(ref, SAR, interlaced=False)
    A = G.similarity_from_params(5.0, -3.0, 0.3, 1.0, SAR)
    r = st.register(_render(bg, A, rng=rng), 0)
    assert r.ok and _err(r.A, A) < 0.3
