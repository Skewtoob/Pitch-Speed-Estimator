"""Canvas coordinate mapping (both parities, several SARs), text, loupe, frame cache."""
from __future__ import annotations

import numpy as np
import pytest

from pitchspeed.io.display import display_transform, sample_to_display
from pitchspeed.types import Sample
from pitchspeed.ui import canvas as cv
from pitchspeed.ui.frames import FrameStore
from pitchspeed.ui.loupe import LOUPE_HALF, LOUPE_ZOOM, make_loupe, paste_loupe, to_loupe

from ui_helpers import FakeSource

SARS = [8.0 / 9.0, 10.0 / 11.0, 1.0, 32.0 / 27.0]


def _dot_frame(x0: int, y0: int, w: int = 720, h: int = 480) -> np.ndarray:
    img = np.full((h, w), 30, np.uint8)
    img[y0 - 1:y0 + 2, x0 - 1:x0 + 2] = 255
    return img


def _centroid(img: np.ndarray, thr: int = 128) -> tuple[float, float]:
    g = img if img.ndim == 2 else img[..., 1]
    ys, xs = np.nonzero(g > thr)
    wts = g[ys, xs].astype(np.float64)
    return float((xs * wts).sum() / wts.sum()), float((ys * wts).sum() / wts.sum())


@pytest.mark.parametrize("sar", SARS)
@pytest.mark.parametrize("size", [(720, 540), (1080, 810), (960, 540)])
def test_canvas_A_matches_display_module(sar, size):
    c = cv.Canvas((720, 480), sar, size)
    np.testing.assert_allclose(c.A, display_transform(720, 480, sar, size), atol=1e-12)
    pts = np.array([[0, 0], [719, 479], [400.25, 201.5], [-0.5, -0.5]])
    np.testing.assert_allclose(c.to_frame(c.to_display(pts)), pts, atol=1e-9)


@pytest.mark.parametrize("sar", SARS)
@pytest.mark.parametrize("parity", [0, 1])
def test_field_click_maps_back_to_frame_pixel(sar, parity):
    """A dot shown in a bobbed field is found at its true frame position (both parities)."""
    x0, y0 = 400, 201
    frame = _dot_frame(x0, y0)
    field = np.ascontiguousarray(frame[parity::2])
    smp = Sample(s=parity, frame=0, k=parity, parity=parity, t=0.0, image=field)
    c = cv.Canvas((720, 480), sar, (720, 540))
    img = c.render_sample(smp, is_field=True)
    _, A = sample_to_display(smp, sar, (720, 540))
    np.testing.assert_allclose(A, c.A, atol=1e-12)
    u, v = _centroid(img)
    x, y = c.to_frame([(u, v)])[0]
    # rows of the dot in this field: 200/202 (top) or 201 (bottom): centroid row 201 either way
    assert abs(x - x0) < 0.3
    assert abs(y - y0) < 0.6


@pytest.mark.parametrize("sar", [8.0 / 9.0, 1.0])
def test_frame_view_maps_back(sar):
    x0, y0 = 123, 333
    c = cv.Canvas((720, 480), sar, (720, 540), strip_h=40)
    img = c.render_frame(_dot_frame(x0, y0))
    assert img.shape == (580, 720, 3)
    u, v = _centroid(img[:540])
    x, y = c.to_frame([(u, v)])[0]
    assert abs(x - x0) < 0.3 and abs(y - y0) < 0.3


def test_parities_do_not_jump():
    """Static content: both fields of a frame put it on the same display row."""
    frame = np.full((480, 720), 30, np.uint8)
    frame[236:245, 300:310] = 255          # 9 rows: centroid row 240 in both fields
    c = cv.Canvas((720, 480), 8 / 9, (720, 540))
    imgs = [c.render_sample(Sample(s=p, frame=0, k=p, parity=p, t=0.0, image=np.ascontiguousarray(frame[p::2])),
                            is_field=True) for p in (0, 1)]
    c0, c1 = _centroid(imgs[0], 200), _centroid(imgs[1], 200)
    assert abs(c0[1] - c1[1]) < 0.75     # well under one frame line (1.125 display px)


def test_letterbox_and_in_video():
    c = cv.Canvas((720, 480), 8 / 9, (960, 540))      # 4:3 picture in a 16:9 area
    assert c.in_video(480, 270)
    assert not c.in_video(20, 270)                     # left bar
    assert not c.in_video(480, 560)                    # below the picture
    u0, _ = c.to_display([(-0.5, 0)])[0]
    assert abs(u0 + 0.5 - 120) < 1e-6                  # (960 - 720) / 2 bar


def test_ui_scale(monkeypatch):
    assert cv.video_size_for(1.0) == (720, 540)
    assert cv.video_size_for(1.5) == (1080, 810)
    monkeypatch.setenv("PITCHSPEED_UI_SCALE", "1.25")
    assert cv.resolve_ui_scale(None) == 1.25
    monkeypatch.setenv("PITCHSPEED_UI_SCALE", "nonsense")
    assert cv.resolve_ui_scale(None) == 1.0
    assert cv.resolve_ui_scale(10) == 3.0
    assert cv.resolve_ui_scale(-1) == 1.0


def test_ascii_safe_and_text():
    assert cv.ascii_safe("78±3 mph · TOF") == "78+/-3 mph | TOF"
    assert cv.ascii_safe("José") == "Jos?"
    img = np.zeros((60, 300, 3), np.uint8)
    cv.draw_text(img, "Pitch 3 ± 2", (5, 30))
    assert img.any()
    lines = cv.wrap_text("one two three four five six seven eight nine ten", 80, 0.5)
    assert len(lines) > 1 and all(cv.text_size(ln, 0.5)[0] <= 80 or " " not in ln for ln in lines)


def test_outline_does_not_drift():
    """The dark outline stays around the text (no offset copy to the right)."""
    img = np.full((40, 400, 3), 128, np.uint8)
    text = "00:00:00.017 | H = help"
    cv.draw_text(img, text, (5, 25), 0.45)
    w = cv.text_size(text, 0.45)[0]
    dark = np.nonzero((img[..., 0] < 40).any(axis=0))[0]
    assert dark.max() <= 5 + w + 3


def test_help_overlay_and_lines_draw():
    img = np.zeros((640, 720, 3), np.uint8)
    rows = [("", "section"), ("Left/A", "back"), ("Enter", "a very long description " * 6)]
    cv.draw_help(img, "Keys", rows * 12, footer="footer")
    y = cv.draw_lines(img, ["line 1", ("line 2", cv.RED)], org=(8, 20))
    assert y > 20 and img.any()


# ------------------------------------------------------------------ loupe --
def test_loupe_geometry():
    img = np.zeros((540, 720, 3), np.uint8)
    img[50, 100] = (255, 255, 255)
    lp = make_loupe(img, (100.0, 50.0))
    n = (2 * LOUPE_HALF + 1) * LOUPE_ZOOM
    assert lp.shape == (n, n, 3)
    c = LOUPE_HALF * LOUPE_ZOOM
    block = lp[c + 2:c + LOUPE_ZOOM - 2, c + 2:c + LOUPE_ZOOM - 2]
    # the centre pixel is magnified into the centre block (the crosshair leaves a gap there)
    assert block.min() == 255
    assert to_loupe((100.0, 50.0), (100.0, 50.0)) == (c + LOUPE_ZOOM / 2 - 0.5,) * 2
    # a quarter-pixel offset moves 2 loupe pixels
    lu, lv = to_loupe((100.25, 50.0), (100.0, 50.0))
    assert lu - (c + LOUPE_ZOOM / 2 - 0.5) == pytest.approx(2.0)


def test_loupe_at_border_and_paste():
    img = np.full((540, 720, 3), 90, np.uint8)
    lp = make_loupe(img, (0.0, 0.0), points=[((1.0, 1.0), (0, 255, 0))])
    assert lp.shape[0] == lp.shape[1]
    canvas = np.zeros((640, 720, 3), np.uint8)
    n = lp.shape[0]
    for cursor in [(700.0, 100.0), (100.0, 100.0), (700.0, 500.0), (360.0, 270.0)]:
        x, y = paste_loupe(canvas, lp, cursor, video_h=540)
        assert not (x <= cursor[0] <= x + n and y <= cursor[1] <= y + n), cursor
        assert y + n <= 540
    x, y = paste_loupe(canvas, lp, (100.0, 400.0), video_h=540)
    assert x > 360                         # preferred: top-right, away from the HUD text


# ------------------------------------------------------------ frame store --
def test_frame_store_caches_and_weaves():
    src = FakeSource(n_frames=40, field_order="bff")
    st = FrameStore(src)
    a = st.sample(10)
    assert a.bgr is not None and st.sample(10) is a and src.get_calls == 1
    full = st.frame_image(5)
    np.testing.assert_array_equal(full[..., 0], src.frame(5))
    assert st.prefetch(0, 20) > 0
    calls = src.iter_calls
    assert st.prefetch(2, 18) == 0 and src.iter_calls == calls      # all cached: no decode
    assert st.prefetch(15, 25) == 5                                   # only the missing ones


def test_frame_store_byte_bound():
    src = FakeSource(n_frames=40)
    one = src.get_sample(0, color=True)
    per = one.image.nbytes + one.bgr.nbytes
    st = FrameStore(src, max_bytes=per * 5)
    st.prefetch(0, 30)
    assert len(st) == 5 and st.cached(29) and not st.cached(0)


def test_frame_store_placeholder_on_decode_error():
    src = FakeSource(n_frames=10, fail_frames={3})
    st = FrameStore(src)
    smp = st.sample(src.sample_of(3, 0))
    assert smp.bgr.shape == (240, 720, 3) and smp.frame == 3
    assert st.errors and "could not be decoded" in st.errors[0]
    with pytest.raises(IndexError):
        st.sample(10 ** 6)


def test_frame_store_progressive():
    src = FakeSource(n_frames=10, field_order=None)
    st = FrameStore(src)
    np.testing.assert_array_equal(st.frame_gray(4), src.frame(4))


def test_incremental_prefetch():
    src = FakeSource(n_frames=100)
    st = FrameStore(src)
    st.set_prefetch(10, 40)
    steps = 0
    while st.prefetch_step(6):
        steps += 1
        assert steps < 20
    assert all(st.cached(s) for s in range(10, 40)) and src.iter_calls == 1
    assert st.prefetch_step(6) is False
    st.set_prefetch(5, 45)                       # only the missing edges are decoded
    while st.prefetch_step(100):
        pass
    assert all(st.cached(s) for s in range(5, 45))
    st.set_prefetch(None)
    assert st.prefetch_step() is False
