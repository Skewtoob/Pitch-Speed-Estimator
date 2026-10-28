"""Tests for pitchspeed.io.display: parity-aware bob, square pixels, point mapping."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from pitchspeed.io.display import (content_rect, display_to_frame, display_transform, frame_to_display,
                                   map_points, sample_to_display)
from pitchspeed.types import Sample

H, W, SAR = 480, 720, 8 / 9


def _band_frame(yc: float) -> np.ndarray:
    y = np.arange(H, dtype=np.float64)[:, None]
    return np.clip(40 + 180 * np.exp(-0.5 * ((y - yc) / 6.0) ** 2) * np.ones((1, W)), 0, 255).astype(np.uint8)


def _row_centroid(img: np.ndarray) -> float:
    prof = img[:, 300:600].astype(np.float64)
    prof = (prof.mean(axis=(1, 2)) if prof.ndim == 3 else prof.mean(axis=1)) - 40
    prof[prof < 0] = 0
    return float((np.arange(len(prof)) * prof).sum() / prof.sum())


def _field(frame, p, k=None):
    return Sample(s=0, frame=0, k=p if k is None else k, parity=p, t=0.0, image=np.ascontiguousarray(frame[p::2]))


def test_bob_has_no_vertical_jump_between_fields():
    yc = 241.3
    frame = _band_frame(yc)
    cents, naive = [], []
    for p in (0, 1):
        bgr, A = sample_to_display(_field(frame, p), SAR, (960, 720))
        assert bgr.shape == (720, 960, 3) and bgr.dtype == np.uint8
        cents.append(_row_centroid(bgr))
        naive.append(_row_centroid(cv2.resize(frame[p::2], (960, 720), interpolation=cv2.INTER_LINEAR)))
    expected = map_points(A, [(100.0, yc)])[0, 1]
    assert abs(cents[1] - cents[0]) < 0.02              # display px; 1 frame line = 1.5 display px
    assert abs(cents[0] - expected) < 0.05
    assert abs(abs(naive[1] - naive[0]) - 1.5) < 0.1   # a plain resize jumps one frame line


def test_display_transform_square_pixels_and_letterbox():
    A = display_transform(W, H, SAR, (960, 720))
    assert A[0, 0] / A[1, 1] == pytest.approx(SAR)      # square-pixel display
    assert content_rect(A, W, H) == (0, 0, 960, 720)
    # extent [-0.5, W-0.5] maps onto the full display width
    np.testing.assert_allclose(map_points(A, [(-0.5, -0.5), (W - 0.5, H - 0.5)]), [(-0.5, -0.5), (959.5, 719.5)])
    B = display_transform(W, H, 32 / 27, (960, 720))     # 16:9 picture in a 4:3 box -> letterbox
    x0, y0, x1, y1 = content_rect(B, W, H)
    assert (x0, x1) == (0, 960) and y0 > 0 and y1 < 720
    assert (x1 - x0) / (y1 - y0) == pytest.approx(16 / 9, rel=0.01)
    bgr, _ = frame_to_display(np.full((H, W), 200, np.uint8), 32 / 27, (960, 720))
    assert bgr[:y0 - 1].max() == 0 and bgr[y0 + 2:y1 - 2].min() == 200


def test_points_roundtrip_and_field_rows():
    A = display_transform(W, H, SAR, (960, 720))
    pts = np.array([[10.0, 20.0], [700.5, 471.25]])
    np.testing.assert_allclose(display_to_frame(A, map_points(A, pts)), pts, atol=1e-9)
    # a bright dot on frame row 101 (bottom field row 50) lands at A(x, 101)
    frame = np.zeros((H, W), np.uint8)
    frame[101, 400] = 255
    bgr, A1 = sample_to_display(_field(frame, 1), SAR, (960, 720))
    v = np.unravel_index(np.argmax(bgr[:, :, 0]), bgr.shape[:2])
    u_exp, v_exp = map_points(A1, [(400, 101)])[0]
    assert abs(v[0] - v_exp) <= 1.0 and abs(v[1] - u_exp) <= 1.0


def test_progressive_and_color_samples():
    frame = _band_frame(100.0)
    s = Sample(s=0, frame=0, k=0, parity=0, t=0.0, image=frame)
    bgr, A = sample_to_display(s, SAR, (640, 480))
    assert bgr.shape == (480, 640, 3)
    assert abs(_row_centroid(bgr) - map_points(A, [(0, 100.0)])[0, 1]) < 0.05
    col = cv2.cvtColor(frame[0::2], cv2.COLOR_GRAY2BGR)
    col[:, :, 2] = 255
    sc = Sample(s=0, frame=0, k=0, parity=0, t=0.0, image=np.ascontiguousarray(frame[0::2]), bgr=col)
    out, _ = sample_to_display(sc, SAR, (960, 720))
    assert out[:, :, 2].min() == 255                    # the colour image was used
    # explicit is_field overrides the aspect heuristic
    out2, A2 = sample_to_display(_field(frame, 0), SAR, (960, 720), is_field=False)
    assert A2[1, 1] == pytest.approx(960 / (W * SAR))      # shown as a wide 240-row frame...
    assert content_rect(A2, W, 240)[1] > 0                 # ...letterboxed, not bobbed
    _, A3 = sample_to_display(_field(frame, 0), SAR, (960, 720))
    assert A3[1, 1] == pytest.approx(1.5)                  # heuristic: field -> full-height bob
