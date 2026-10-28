"""The cv2 shells driven by a scripted window (no display needed): events -> results."""
from __future__ import annotations

import numpy as np
import pytest

from pitchspeed.calib.model import CalibrationFile, Segment, Settings
from pitchspeed.calib.ui import render as render_calib
from pitchspeed.calib.ui import run_calibration_ui
from pitchspeed.calib.state import CalibState
from pitchspeed.review.state import ReviewState
from pitchspeed.review.ui import bar_sample, render as render_review, run_review_ui
from pitchspeed.ui.canvas import Canvas
from pitchspeed.ui.frames import FrameStore

from ui_helpers import FakeSource, Recorder, ScriptedWindow, key, make_result

CANVAS = Canvas((720, 480), 8 / 9, (720, 540))


def disp(x, y):
    u, v = CANVAS.to_display([(x, y)])[0]
    return int(round(u)), int(round(v))


def click(x, y):
    u, v = disp(x, y)
    return [("mouse", "ldown", u, v), ("mouse", "lup", u, v)]


# ------------------------------------------------------------ calibration --
def test_calibration_clicks_land_on_frame_pixels():
    src = FakeSource(n_frames=200)
    script = [None] + click(400, 201) + [key("right")] + click(100, 300) + [key("enter"), key("enter")]
    w = ScriptedWindow(script)
    cf = run_calibration_ui(src, src.info, "side", window=w, max_iterations=50)
    assert cf is not None and w.closed
    pts = cf.segments[0].points
    # display px are 1/1.125 frame rows apart vertically: within half a display pixel
    assert pts["rubber"][0] == pytest.approx(400, abs=0.5) and pts["rubber"][1] == pytest.approx(201, abs=0.5)
    assert pts["plate"][0] == pytest.approx(100, abs=0.5) and pts["plate"][1] == pytest.approx(300, abs=0.5)
    assert w.shown and all(s == (540 + 92, 720, 3) for s in w.shown)
    assert w.trackbar_max == 199


def test_calibration_cancel_and_close():
    src = FakeSource(n_frames=50)
    assert run_calibration_ui(src, src.info, "side", window=ScriptedWindow([key("esc"), key("esc")]),
                              max_iterations=20) is None
    assert run_calibration_ui(src, src.info, "side", window=ScriptedWindow(click(400, 201) + [("close",)]),
                              max_iterations=20) is None
    # running out of iterations is not a save
    assert run_calibration_ui(src, src.info, "side", window=ScriptedWindow([]), max_iterations=3) is None


def test_calibration_trackbar_and_clicks_outside_ignored():
    src = FakeSource(n_frames=300)
    script = [("trackbar", 123), None, ("mouse", "ldown", 5, 600)] + click(400, 201) + click(100, 300) + \
             [key("enter"), key("enter")]
    w = ScriptedWindow(script)
    cf = run_calibration_ui(src, src.info, "side", window=w, max_iterations=50)
    assert cf.segments[0].ref_frame == 123 and len(cf.segments[0].points) == 2
    assert w.trackbar == 123


def test_calibration_mask_drag_through_shell():
    src = FakeSource(n_frames=50)
    u0, v0 = disp(600, 420)
    u1, v1 = disp(700, 470)
    script = [key("m"), ("mouse", "ldown", u0, v0), ("mouse", "move", u1, v1), ("mouse", "lup", u1, v1)] + \
        click(400, 201) + click(100, 300) + [key("enter"), key("enter")]
    cf = run_calibration_ui(src, src.info, "side", window=ScriptedWindow(script), max_iterations=50)
    (m,) = cf.segments[0].masks
    assert m == pytest.approx([600, 420, 700, 470], abs=0.6)


@pytest.mark.parametrize("pan,moved", [((0.0, 0.0), False), ((0.25, 0.0), True)])
def test_calibration_measures_camera_shift_between_frames(pan, moved):
    """Behind view, release and catch on frames 12 apart: a 3 px pan is measured and recorded."""
    src = FakeSource(n_frames=100, pan=pan)
    script = [("trackbar", 10), None] + click(360, 150) + [("trackbar", 22), None] + click(370, 300) + \
             [key("enter"), key("enter")]
    cf = run_calibration_ui(src, src.info, "behind", window=ScriptedWindow(script), max_iterations=30)
    assert cf is not None and cf.segments[0].ref_frame == 10
    assert ("moved about 2.7 px" in cf.notes or "moved about 2.6 px" in cf.notes) is moved, cf.notes
    if not moved:
        assert "warning at save" not in cf.notes


def test_calibration_render_all_overlays():
    src = FakeSource(n_frames=50)
    cf = CalibrationFile(settings=Settings(sar=8 / 9),
                         segments=[Segment(ref_frame=3, view="side", points={"rubber": [400, 201], "plate": [100, 300]},
                                           masks=[[600, 420, 700, 470]])], image_size=(720, 480))
    st = CalibState(src, "side", (720, 480), 50, initial=cf, sar=8 / 9)
    st.loupe = st.help = True
    st.confirm = ["a warning", "another one " * 10]
    st.mask_mode, st.drag = True, (10, 10, 50, 40)
    canvas = Canvas((720, 480), 8 / 9, (720, 540), strip_h=92)
    for mode in ("field", "frame"):
        st.view_mode = mode
        img = render_calib(st, FrameStore(src), canvas, mouse_uv=(300, 200))
        assert img.shape == (632, 720, 3)


def test_calibration_marker_drawn_where_clicked():
    """The marker of a placed point is drawn on the display pixel of the point."""
    src = FakeSource(n_frames=20, dot=(50.0, 50.0))
    st = CalibState(src, "side", (720, 480), 20, sar=8 / 9)
    st.place(400.0, 201.0)
    canvas = Canvas((720, 480), 8 / 9, (720, 540), strip_h=92)
    img = render_calib(st, FrameStore(src), canvas)
    u, v = canvas.to_display([(400.0, 201.0)])[0]
    r = 9                                   # marker radius 8; the label starts right of u + 11
    patch = img[int(v) - r:int(v) + r + 1, int(u) - r:int(u) + r + 1]
    yellow = (patch[..., 2] > 200) & (patch[..., 1] > 200) & (patch[..., 0] < 100)
    ys, xs = np.nonzero(yellow)
    assert len(xs) > 10
    assert abs(xs.mean() - r - (u - int(u))) < 0.75 and abs(ys.mean() - r - (v - int(v))) < 0.75


# ----------------------------------------------------------------- review --
def test_review_scripted_session():
    src = FakeSource(n_frames=400)
    res = make_result(src)
    rec = Recorder()
    u, v = disp(500, 180)
    script = [None, key("right"), key("right"), ord("r"), ("mouse", "ldown", u, v), ord("t"), ord("z"),
              ("mouse", "move", 300, 200), key("pgdn"), ord("x"), ord("h"), ord("h"), key("space"), None,
              key("space"), ord("n"), key("down"), ord("c"), ord("q")]
    w = ScriptedWindow(script)
    out = run_review_ui(res, src, None, rec.recompute, rec.save, window=w, max_iterations=200)
    assert out is res and w.closed
    assert [(p.number, p.status) for p in res.pitches] == [(1, "edited"), (0, "rejected"), (2, "manual"),
                                                         (3, "auto")]
    assert any(q.manual for q in res.pitches[0].track.points)
    assert rec.saved >= 5
    assert all(s == (540 + 100, 720, 3) for s in w.shown)


def test_review_window_close_saves():
    src = FakeSource(n_frames=200)
    res = make_result(src, windows=((50, 70),), missed=())
    rec = Recorder()
    run_review_ui(res, src, None, rec.recompute, rec.save, window=ScriptedWindow([key("right"), ord("r"), ("close",)]),
                  max_iterations=20)
    assert rec.saved == 2                   # autosave after the edit + final save on close
    assert res.pitches[0].release.s == 51


def test_review_retrack_shows_busy_then_runs():
    src = FakeSource(n_frames=200)
    res = make_result(src, windows=((50, 70),), missed=())
    rec = Recorder()
    called = []

    def retrack(p):
        called.append(p)
        return p.track

    w = ScriptedWindow([ord("e"), None, ord("q")], keep_images=True)
    run_review_ui(res, src, None, rec.recompute, rec.save, retrack=retrack, window=w, max_iterations=20)
    assert len(called) == 1 and res.pitches[0].status == "edited"


def test_review_timeline_click_jumps():
    src = FakeSource(n_frames=200)
    res = make_result(src, windows=((100, 120),), missed=())
    rec = Recorder()
    canvas = Canvas((720, 480), 8 / 9, (720, 540), strip_h=100)
    st = ReviewState(res, src, rec.recompute, rec.save)
    w0, w1 = st.window()
    assert bar_sample(st, canvas, 10) == w0 and bar_sample(st, canvas, 710) == w1
    w = ScriptedWindow([("mouse", "ldown", 360, 540 + 12), ord("r"), ord("q")])
    run_review_ui(res, src, None, rec.recompute, rec.save, window=w, max_iterations=20)
    assert res.pitches[0].status == "edited"
    assert abs(res.pitches[0].release.s - (w0 + w1) / 2) <= 1


def test_review_default_retrack_uses_vision(monkeypatch):
    """Without a retrack callback, E calls vision.finder.track_between with the reference frame."""
    import pitchspeed.vision.finder as finder
    from pitchspeed.review.ui import make_default_retrack
    src = FakeSource(n_frames=200)
    res = make_result(src, windows=((50, 70),), missed=())
    cf = CalibrationFile(settings=Settings(sar=8 / 9),
                         segments=[Segment(ref_frame=7, view="side", points={"rubber": [600, 150], "plate": [100, 250]})],
                         image_size=(720, 480))
    seen = {}

    def fake_track_between(source, calib, scan, s_release, s_catch, reference_frame, opts=None, manual_points=None):
        seen.update(s=(s_release, s_catch), ref=reference_frame.shape, manual=manual_points, view=calib.view)
        return res.pitches[0].track

    monkeypatch.setattr(finder, "track_between", fake_track_between)
    rt = make_default_retrack(src, cf, src.scan)
    tr = rt(res.pitches[0])
    assert tr is res.pitches[0].track
    assert seen == {"s": (50, 70), "ref": (480, 720), "manual": [], "view": "side"}


def test_review_render_all_overlays():
    src = FakeSource(n_frames=400)
    res = make_result(src, rejected=1)
    rec = Recorder()
    st = ReviewState(res, src, rec.recompute, rec.save)
    st.loupe = st.help = True
    st.save_error = "disk full"
    canvas = Canvas((720, 480), 8 / 9, (720, 540), strip_h=100)
    store = FrameStore(src)
    for mode in ("field", "frame"):
        st.view_mode = mode
        assert render_review(st, store, canvas, mouse_uv=(300, 200), busy="working").shape == (640, 720, 3)
    st.on_key("o")
    for _ in range(6):
        st.on_key("pgdn")
        render_review(st, store, canvas)
    st.on_key("n")
    render_review(st, store, canvas)


def test_review_with_undecodable_frames():
    src = FakeSource(n_frames=200, fail_frames={30, 31})
    res = make_result(src, windows=((58, 70),), missed=())
    rec = Recorder()
    w = ScriptedWindow([key("left")] * 5 + [ord("q")])
    run_review_ui(res, src, None, rec.recompute, rec.save, window=w, max_iterations=20)
    assert rec.saved == 1
