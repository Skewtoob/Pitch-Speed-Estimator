"""Review window in mark mode (`pitchspeed mark`): marking pitches by hand on a long tape."""
from __future__ import annotations

import numpy as np
import pytest

from pitchspeed.calib.model import Settings, manual_calibration
from pitchspeed.estimate.speed import estimate_speed
from pitchspeed.review.state import MARK_SPEEDS, PITCH_MARGIN, ReviewState
from pitchspeed.types import AnalysisResult
from test_ui_review_fixes import keys
from ui_helpers import FakeSource, Recorder, make_result


def _manual_recompute(src):
    """recompute() with the real estimator on a hand-marking calibration (no clicked points)."""
    calib = manual_calibration(Settings(distance_ft=60.5, level="hs")).segment_calibs()[0]

    def recompute(p):
        p.speed = estimate_speed(calib, src.scan, src.sample_dt, p.release, p.catch, track=p.track, manual=True)
        return p
    return recompute


def make_mark(n_frames=3000, windows=(), **kw):
    src = FakeSource(n_frames=n_frames)
    res = make_result(src, windows=windows, missed=())
    rec = Recorder()
    st = ReviewState(res, src, _manual_recompute(src), rec.save, mode="mark", **kw)
    return st, src, res, rec


def test_starts_with_instructions_and_whole_tape_bar():
    st, src, res, _ = make_mark()
    assert st.mode == "mark" and st.s == 0 and st.cur is None
    assert "N at the release" in st.message.text and "Space" in st.message.text
    assert st.window() == (0, st.n_samples - 1)
    assert st.top_text().startswith("MARK | 0 marked")
    assert "no pitch here" in st.info_text() and "0 pitches marked" in st.info_text()
    with pytest.raises(ValueError):
        ReviewState(res, src, lambda p: p, lambda r: None, mode="fancy")


def test_mark_a_pitch_with_n_and_c_and_then_a_hit_with_n_and_b():
    st, src, res, rec = make_mark()
    st.goto(1000)
    keys(st, "n")
    st.goto(1029)
    keys(st, "c")
    assert len(res.pitches) == 1
    p = res.pitches[0]
    assert p.status == "manual" and p.release.s == 1000 and p.catch.s == 1029
    assert p.speed.method == "manual_tof" and p.speed.label in ("good", "fair") and 70 < p.speed.v_release_mph < 95
    assert "added" in st.message.text and "Space plays on" in st.message.text and rec.saved
    st.goto(2000)
    keys(st, "n")
    st.goto(2027)
    keys(st, "b")
    q = [x for x in res.pitches if x.release.s == 2000][0]
    assert q.catch.point == "contact" and q.speed.v_release_mph is not None
    assert [x.number for x in sorted(res.pitches, key=lambda x: x.release.s)] == [1, 2]
    assert st.window() == (0, st.n_samples - 1)                      # bar still covers the whole tape
    ticks = st.markers()["pitches"]
    assert [(a, b) for a, b, _ in ticks] == [(1000, 1029), (2000, 2027)]


def test_keys_only_act_on_a_pitch_near_the_shown_field():
    st, src, res, _ = make_mark()
    st.goto(1000)
    keys(st, "n")
    st.goto(1029)
    keys(st, "c")
    st.goto(1029 + PITCH_MARGIN + 50)                                # far from the pitch
    assert st.current_pitch() is None and "no pitch here" in st.info_text()
    keys(st, "c")                                                    # must NOT move the old catch
    assert res.pitches[0].catch.s == 1029 and st.message.level == "warn" and "N" in st.message.text
    keys(st, "x")                                                    # nor reject it
    assert res.pitches[0].status == "manual"
    keys(st, "r")                                                    # R away from every pitch starts a new one
    assert st.draft is not None and st.draft["s_release"] == 1029 + PITCH_MARGIN + 50
    keys(st, "esc")
    st.goto(1015)                                                    # back near the pitch: it is current again
    assert st.current_pitch() is res.pitches[0]
    keys(st, "c")
    assert res.pitches[0].catch.s == 1015


def test_forward_playback_up_to_8x_without_looping():
    st, src, res, _ = make_mark(n_frames=100)                        # 200 fields
    assert MARK_SPEEDS[-1] == 8.0 and st.play_speed == 1.0 and st.play_step == 1
    for _ in range(5):
        keys(st, "+")
    assert st.play_speed == 8.0 and st.play_step == 8 and "8x" in st.message.text
    st.goto(150)
    keys(st, "space")
    assert st.playing
    st.tick()
    assert st.s == 158
    for _ in range(10):
        st.tick()
    assert st.s == st.n_samples - 1 and not st.playing and "end of the video" in st.message.text
    keys(st, "space")                                                # at the end: does not restart
    assert not st.playing and st.s == st.n_samples - 1
    for _ in range(10):
        keys(st, "-")
    assert st.play_speed == 0.125 and st.play_step == 1


def test_one_second_jumps_and_bar_position():
    st, src, res, _ = make_mark()
    one = int(round(1.0 / src.sample_dt))
    st.goto(500)
    keys(st, "l")
    assert st.s == 500 + one
    keys(st, "j", "j")
    assert st.s == 500 - one
    keys(st, "j", "j", "j", "j", "j", "j", "j", "j", "j")
    assert st.s == 0


def test_pgup_pgdn_step_through_marked_pitches():
    st, src, res, _ = make_mark(windows=((400, 428), (1600, 1630)))     # two pitches already marked
    keys(st, "pgdn")
    assert st.s == 400 and st.current_pitch() is not None
    keys(st, "pgdn")
    assert st.s == 1600
    keys(st, "pgdn")
    assert "last one" in st.message.text
    st2, _, _, _ = make_mark()
    keys(st2, "pgdn")
    assert "no pitches marked yet" in st2.message.text


def test_review_mode_is_unchanged():
    """The plain review window still zooms the bar on the item and loops playback."""
    src = FakeSource(n_frames=400)
    res = make_result(src, windows=((100, 125),), missed=())
    rec = Recorder()
    st = ReviewState(res, src, rec.recompute, rec.save)
    assert st.mode == "review" and st.cur is not None
    w0, w1 = st.window()
    assert w0 == 100 - PITCH_MARGIN and w1 < 400 and st.markers()["pitches"] == []
    keys(st, "space")
    st.s = w1
    st.tick()
    assert st.s == w0 and st.playing                                 # looped


def test_manual_calibration_needs_no_points_and_names_the_view():
    cf = manual_calibration(Settings(distance_ft=46.0, level="ll"))
    assert cf.validate() and all("extension" in w for w in cf.validate())
    assert cf.segments[0].view == "manual" and cf.segments[0].points == {}
    from pitchspeed.render.report import format_table
    src = FakeSource(n_frames=50)
    res = make_result(src, windows=(), missed=())
    res.calibration = cf.to_dict()
    assert "marked by hand" in format_table(res)


def test_mark_window_renders(monkeypatch):
    """The cv2 shell draws the whole-tape bar with pitch ticks (no window needed)."""
    from pitchspeed.review import ui
    from pitchspeed.ui import canvas as cv
    from pitchspeed.ui.frames import FrameStore
    st, src, res, _ = make_mark(windows=((400, 428),))
    canvas = cv.Canvas((720, 480), 8 / 9, cv.video_size_for(1.0), strip_h=ui.STRIP_H)
    img = ui.render(st, FrameStore(src), canvas)
    assert img.ndim == 3 and img.shape[0] > canvas.video_size[1]
    x0, y0, x1, y1 = ui._bar_geometry(canvas)
    xs = int(round(x0 + 400 / (st.n_samples - 1) * (x1 - x0)))
    tick = img[(y0 + y1) // 2, xs].astype(int)
    assert np.abs(tick - np.array(cv.GREEN)).sum() < 60                # an automatic pitch: green tick
    assert ui.bar_sample(st, canvas, x1) == st.n_samples - 1          # bar end = end of the tape


# ------------------------------------------------------- field order (K)
def test_k_switches_the_field_order_and_keeps_marks_on_the_same_pictures():
    st, src, res, _ = make_mark()
    st.goto(1000)
    keys(st, "n")
    st.goto(1029)
    keys(st, "c")
    p = res.pitches[0]
    img_r = src.get_sample(1000).image.copy()          # the pictures the user marked
    img_c = src.get_sample(1029).image.copy()
    v_before = p.speed.v_release_mph
    keys(st, "k")
    assert st.pending == ("flip_order", None)
    st.run_pending()
    assert src.scan.field_order == "tff" and res.scan.field_order == "tff" and res.scan.source == "user"
    assert st.frames_changed and not st.undo_stack
    assert (p.release.s, p.catch.s) == (1001, 1028)
    assert np.array_equal(src.get_sample(1001).image, img_r) and np.array_equal(src.get_sample(1028).image, img_c)
    assert p.release.t == pytest.approx(src.sample_time(1001) - src.sample_dt / 2)
    assert p.speed.v_release_mph > v_before                          # 2 fields shorter flight -> faster
    assert any("switched by hand" in w for w in res.warnings)
    assert "switched to TFF" in st.message.text
    keys(st, "k")
    st.run_pending()                                                 # and back
    assert (p.release.s, p.catch.s) == (1000, 1029) and src.scan.field_order == "bff"
    assert p.speed.v_release_mph == pytest.approx(v_before)


def test_k_is_refused_where_it_would_break_saved_tracks():
    st, src, res, _ = make_mark(windows=((400, 428),))                # an automatically found pitch
    keys(st, "k")
    assert st.pending is None and "analyzing again" in st.message.text
    src2 = FakeSource(n_frames=300)
    res2 = make_result(src2, windows=((100, 125),), missed=())
    rec = Recorder()
    st2 = ReviewState(res2, src2, rec.recompute, rec.save)            # review mode
    keys(st2, "k")
    assert st2.pending is None and "--field-order" in st2.message.text
    src3 = FakeSource(n_frames=300, field_order=None)                 # progressive
    st3 = ReviewState(make_result(src3, windows=(), missed=()), src3, rec.recompute, rec.save, mode="mark")
    keys(st3, "k")
    assert st3.pending is None and "not interlaced" in st3.message.text


def test_uncertain_field_order_is_pointed_out_at_the_start():
    src = FakeSource(n_frames=300)
    src.scan.confidence = "low"
    res = make_result(src, windows=(), missed=())
    st = ReviewState(res, src, _manual_recompute(src), Recorder().save, mode="mark")
    assert "FIELD ORDER UNCERTAIN" in st.message.text and "K" in st.message.text


def test_real_sample_source_switches_in_place():
    from pitchspeed.io.interlace import SampleSource
    from pitchspeed.types import ScanInfo
    from ui_helpers import FakeReader
    fake = FakeSource(n_frames=20)
    scan = ScanInfo(mode="interlaced", field_order="tff", confidence="low", sample_rate=1 / fake.sample_dt)
    rd = FakeReader(fake)
    rd.frame_times = np.arange(fake.n_frames) * fake.frame_dt
    ss = SampleSource(rd, scan)
    a = ss.get_sample(7)
    ss.set_field_order("bff")
    b = ss.get_sample(6)
    assert scan.field_order == "bff" and (a.frame, a.parity) == (b.frame, b.parity)
    assert np.array_equal(a.image, b.image) and ss.sample_time(7) == a.t
    with pytest.raises(ValueError):
        ss.set_field_order("xyz")
