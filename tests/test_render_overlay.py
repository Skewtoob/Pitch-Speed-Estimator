"""Tests for pitchspeed.render.overlay: timing rules, coordinate mapping,
highlight windows, and rendering through fakes (no video I/O)."""
from __future__ import annotations

import math

import numpy as np
import pytest

import render_helpers as H
from pitchspeed.io.display import display_transform, map_points
from pitchspeed.render import overlay as O
from pitchspeed.types import EventTime


def mark(n, s0, s1, sp=None, status="auto", pts=None):
    return O.PitchMark(number=n, status=status, s_release=s0, s_catch=s1,
                       speed=sp or H.speed(70.0 + n, 1.5, "good"),
                       points=np.zeros((0, 4)) if pts is None else np.asarray(pts, float))


# --------------------------------------------------------------- timeline
def test_timeline_basic_visibility():
    ms = [mark(1, 100, 130), mark(2, 1000, 1030)]
    tl = O.Timeline(ms, hold_samples=180)
    st = tl.state(99)
    assert st.in_flight == [] and st.card is None and st.done == []
    assert tl.state(100).in_flight == [0] and tl.state(129).in_flight == [0]
    st = tl.state(130)                                  # the catch: tag gone, card up, listed
    assert st.in_flight == [] and st.card == 0 and st.done == [0]
    assert tl.state(130 + 179).card == 0
    assert tl.state(130 + 180).card is None and tl.state(130 + 180).done == [0]
    assert tl.state(1000).in_flight == [1] and tl.state(5000).done == [0, 1]


def test_card_ends_at_next_release():
    ms = [mark(1, 100, 130), mark(2, 200, 230)]
    tl = O.Timeline(ms, hold_samples=180)
    assert tl.state(199).card == 0
    st = tl.state(200)
    assert st.card is None and st.in_flight == [1]
    assert tl.state(230).card == 1


def test_overlapping_pitches_do_not_lose_a_card():
    # pitch 2 is released before pitch 1 is caught (bad data): pitch 1 still gets
    # its card after its catch; when both cards qualify the latest catch wins.
    ms = [mark(1, 100, 140), mark(2, 120, 160)]
    tl = O.Timeline(ms, hold_samples=180)
    assert tl.state(130).in_flight == [0, 1]
    st = tl.state(145)
    assert st.card == 0 and st.in_flight == [1]
    assert tl.state(160).card == 1 and tl.state(160).done == [0, 1]


def test_timeline_empty():
    st = O.Timeline([], 10).state(5)
    assert st.in_flight == [] and st.card is None and st.done == []


# ------------------------------------------------------------------ marks
def test_build_marks_skips_rejected_and_uses_sample_or_time():
    r = H.rich_result()
    r.pitches[1].release = EventTime(t=1400 * H.FIELD_DT, sigma=0.01, method="x", s=None)
    r.pitches[2].catch = EventTime(t=float("nan"), sigma=0.01, method="x", s=None)
    ms = O.build_marks(r, lambda t: int(round(t / H.FIELD_DT)))
    assert [m.number for m in ms] == [1, 2, 4, 5]          # rejected + unplaceable pitch 3 skipped
    assert ms[1].s_release == 1400 and ms[0].s_release == 600 and ms[0].s_catch == 628
    assert len(ms[0].points) == 28                          # 29 points, one outlier dropped
    assert ms[2].status == "manual" and len(ms[2].points) == 0 and ms[2].color == O.YELLOW
    assert ms[0].color == O.GREEN


def test_card_lines():
    l1, l2, col = O.card_lines(mark(3, 0, 10, H.speed(78.4, 3.1, "fair")))
    assert l1 == "Pitch 3: 78 mph" and col == O.WHITE
    assert l2 == "+/-3 mph (90%: 73-83) | release speed | TOF"
    l1, l2, col = O.card_lines(mark(3, 0, 10, H.speed(78.4, 6.0, "poor")))
    assert l1 == "Pitch 3: ~78 mph?" and col == O.ORANGE and l2.startswith("+/-6 mph (90%: 69-88)")
    l1, l2, col = O.card_lines(mark(3, 0, 10, H.speed(None, None, "none",
                                                        reason="not measurable: uncertainty ±14% (> 10%)")))
    assert l1 == "Pitch 3: speed n/a" and col == O.GREY and l2 == "not measurable: uncertainty +/-14% (> 10%)"
    assert l2.isascii()
    l1, l2, _ = O.card_lines(O.PitchMark(number=3, status="auto", s_release=0, s_catch=10, speed=None))
    assert l1 == "Pitch 3: speed n/a" and l2 == "no speed estimate"
    _, l2, _ = O.card_lines(mark(4, 0, 10, H.speed(71.0, 2.9, "fair", method="manual_tof"), status="manual"))
    assert l2.endswith("| manual TOF | manual")
    long = "x" * 200
    _, l2, _ = O.card_lines(mark(4, 0, 10, H.speed(None, None, "none", reason=long)))
    assert len(l2) <= 64 and l2.endswith("...")


def test_running_items():
    ms = [mark(2, 0, 1, H.speed(65.4, 5.4, "poor")), mark(1, 0, 1, H.speed(76.5, 1.0, "good")),
          mark(3, 0, 1, H.speed(None, None, "none"))]
    items = O.running_items(ms, [0, 1, 2])
    assert items == [("P1 77", O.WHITE), ("P2 65?", O.ORANGE), ("P3 n/a", O.GREY)]


# ------------------------------------------------------- highlight windows
def test_highlight_windows_merge_and_clip():
    ms = [mark(1, 100, 130), mark(2, 2000, 2030), mark(3, 2300, 2330), mark(4, 9900, 9950)]
    w = O.highlight_windows(ms, n_samples=10000, pre_samples=120, post_samples=180)
    assert w == [(0, 311), (1880, 2511), (9780, 10000)]
    # touching windows merge too, separate ones do not
    ms = [mark(1, 200, 230), mark(2, 531, 560)]
    assert O.highlight_windows(ms, 10000, 120, 180) == [(80, 741)]
    ms = [mark(1, 200, 230), mark(2, 532, 560)]
    assert O.highlight_windows(ms, 10000, 120, 180) == [(80, 411), (412, 741)]
    assert O.highlight_windows([], 100, 1, 1) == []


# ---------------------------------------------------- coordinate mapping
def test_reference_to_display_inverts_the_sample_transform():
    A = display_transform(720, 480, 8 / 9, (960, 720))
    th = math.radians(2.0)
    T = np.array([[1.01 * math.cos(th), -math.sin(th), 7.0],       # sample -> reference
                  [math.sin(th), 1.01 * math.cos(th), -4.0]])
    frame_pts = np.array([[100.0, 200.0], [600.0, 50.0], [360.0, 479.0]])
    ref_pts = map_points(T, frame_pts)
    M = O.reference_to_display(T, A)
    np.testing.assert_allclose(O.apply_affine(M, ref_pts), map_points(A, frame_pts), atol=1e-9)
    # identity transform -> plain display map
    np.testing.assert_allclose(O.reference_to_display(np.eye(2, 3), A), A, atol=1e-12)


def test_transform_lookup():
    Ts = np.stack([np.array([[1, 0, dx], [0, 1, 0.0]]) for dx in (1.0, 2.0, 3.0)])
    lk = O.TransformLookup(Ts, np.array([10, 20, 30]))
    assert lk.get(10)[0, 2] == 1.0 and lk.get(14)[0, 2] == 1.0 and lk.get(16)[0, 2] == 2.0
    assert lk.get(0)[0, 2] == 1.0 and lk.get(99)[0, 2] == 3.0
    np.testing.assert_array_equal(O.TransformLookup(None, None).get(5), np.eye(2, 3))
    # unsorted samples are fine; implicit samples = row index
    lk = O.TransformLookup(Ts[::-1], np.array([30, 20, 10]))
    assert lk.get(29)[0, 2] == 3.0
    assert O.TransformLookup(Ts, None).get(2)[0, 2] == 3.0
    # broken input -> identity, never a crash
    bad = Ts.copy()
    bad[1] = np.nan
    np.testing.assert_array_equal(O.TransformLookup(bad, np.array([10, 20, 30])).get(20), np.eye(2, 3))
    np.testing.assert_array_equal(O.TransformLookup(Ts, np.array([1, 2])).get(1), np.eye(2, 3))
    np.testing.assert_array_equal(O.TransformLookup(np.zeros((3, 3, 3)), None).get(1), np.eye(2, 3))


def test_audio_filter_graph():
    g = O.audio_filter_graph([(10.0, 5.5), (40.0, 6.0)], offset=0.01, seek=9.01)
    assert g.startswith("[1:a:0]asplit=2[i0][i1];")
    assert "[i0]atrim=start=1.000000:end=6.500000,asetpts=PTS-1.000000/TB" in g
    assert "[i1]atrim=start=31.000000:end=37.000000" in g
    assert "apad=whole_dur=5.500000,atrim=end=5.500000[a0]" in g
    assert g.endswith("[a0][a1]concat=n=2:v=0:a=1[aout]")


def test_output_fps():
    from fractions import Fraction
    assert O.output_fps(H.FakeSource(10, interlaced=True)) == Fraction(60000, 1001)
    assert O.output_fps(H.FakeSource(10, interlaced=False)) == Fraction(30000, 1001)
    src = H.FakeSource(10, interlaced=False)
    src.reader.info.frame_rate_str = "bogus"
    assert O.output_fps(src) == Fraction(30000, 1001)


# ------------------------------------------------------- render via fakes
def _render(result, src, mode="full", **kw):
    H.FakeWriter.instances.clear()
    info = {}
    out = O.render_annotated(result, src, "unused.mp4", mode=mode, writer_factory=H.FakeWriter,
                             info=info, **kw)
    return out, H.FakeWriter.instances[-1], info


def _small_result(n_pitches=2, gap=400):
    r = H.rich_result()
    r.pitches = [H.pitch(i + 1, 60 + i * gap, 88 + i * gap, H.speed(70.0 + i, 1.5, "good"))
                 for i in range(n_pitches)]
    return r


CARD_ROI = (slice(600, 700), slice(300, 660))       # where the card sits on 960x720


def _card_visible(img) -> bool:
    roi = img[CARD_ROI].astype(int)
    return (roi.max(axis=2) > 240).sum() > 200      # white text / border


def test_render_full_frame_per_sample_and_card_timing():
    r = _small_result()
    src = H.FakeSource(600)
    out, w, info = _render(r, src)
    assert out == "unused.mp4" and w.closed and w.fps == "60000/1001"
    assert len(w.frames) == 600 and info["frames_written"] == 600 and info["filler_frames"] == 0
    assert w.audio_src is None and info["audio_status"].startswith("skipped")
    hold = round(3.0 / src.sample_dt)
    assert not _card_visible(w.frames[87])
    assert _card_visible(w.frames[88]) and _card_visible(w.frames[88 + hold - 1])
    assert not _card_visible(w.frames[88 + hold])
    assert not _card_visible(w.frames[459]) and _card_visible(w.frames[488])


def test_render_progressive_uses_frame_rate():
    r = _small_result()
    r.scan.mode = "progressive"
    src = H.FakeSource(300, interlaced=False)
    _, w, info = _render(r, src)
    assert w.fps == "30000/1001" and len(w.frames) == 300


def test_rejected_pitch_is_not_drawn():
    r = _small_result()
    r.pitches[1].status = "rejected"
    src = H.FakeSource(600)
    _, w, info = _render(r, src)
    assert info["marks"] == 1
    assert not _card_visible(w.frames[488])


def _green_near(img, u, v, r=8, color=O.GREEN) -> int:
    """Number of trail-coloured pixels within r px of display point (u, v)."""
    d = np.abs(img.astype(int) - np.array(color)[None, None, :]).sum(axis=2)
    ys, xs = np.nonzero(d < 40)
    return int(((xs - u) ** 2 + (ys - v) ** 2 <= r * r).sum())


def test_trail_is_drawn_where_the_track_is_and_follows_transforms():
    r = _small_result(1)
    tr = r.pitches[0].track
    for p in tr.points:                       # one fixed reference position for every point
        p.x, p.y, p.inlier = 400.0, 300.0, True
    src = H.FakeSource(300)
    A = display_transform(720, 480, 8 / 9, (960, 720))
    _, w, _ = _render(r, src)
    u, v = map_points(A, [(400.0, 300.0)])[0]
    u2, _ = map_points(A, [(380.0, 300.0)])[0]
    assert _green_near(w.frames[70], u, v) > 15 and _green_near(w.frames[70], u2, v) == 0
    assert _green_near(w.frames[59], u, v) == 0                     # before release: nothing
    # camera moved: sample -> reference is a +20 px x shift, so the ball sits 20 frame px left
    Ts = np.stack([np.array([[1.0, 0.0, 20.0], [0.0, 1.0, 0.0]])] * 300)
    _, w, _ = _render(r, src, transforms=Ts, transform_samples=np.arange(300))
    assert _green_near(w.frames[70], u2, v) > 15 and _green_near(w.frames[70], u, v) == 0


def test_highlights_only_windows_and_progress():
    r = _small_result(2, gap=2000)
    src = H.FakeSource(3000)
    calls = []
    _, w, info = _render(r, src, mode="highlights", progress=lambda d, t: calls.append((d, t)))
    pre, post = math.ceil(2.0 / src.sample_dt), math.ceil(3.0 / src.sample_dt)
    expect = [(max(0, 60 - pre), 88 + post + 1), (2060 - pre, 2088 + post + 1)]
    assert src.calls == expect
    assert len(w.frames) == sum(b - a for a, b in expect)
    assert [(d["s_start"], d["s_end"]) for d in info["segments"]] == expect
    assert calls[-1] == (len(w.frames), len(w.frames))
    assert all(d1 <= d2 for (d1, _), (d2, _) in zip(calls, calls[1:]))


def test_progress_object_with_update():
    class P:
        def __init__(self):
            self.seen = []

        def update(self, done):
            self.seen.append(done)
    p = P()
    _render(_small_result(1), H.FakeSource(100), progress=p)
    assert p.seen[-1] == 100


def test_gaps_are_filled_to_keep_time():
    r = _small_result(1)
    # samples 10, 11 fail to decode; from sample 50 on the source time jumps by one frame
    src = H.FakeSource(200, missing={10, 11}, time_offset={50: H.FIELD_DT * 2})
    _, w, info = _render(r, src)
    assert info["filler_frames"] == 4
    assert len(w.frames) == 198 + 4
    assert w.frames[10].mean() < 15 and w.frames[11].mean() < 15         # black "no picture" frames
    assert (np.abs(w.frames[10].astype(int) - np.array(O.RED)).sum(axis=2) < 60).sum() > 50
    assert w.frames[12][..., 0].mean() > 50                               # real picture again
    assert w.frames[50].mean() < 15 and w.frames[51].mean() < 15


def test_errors():
    with pytest.raises(ValueError, match="mode"):
        O.render_annotated(_small_result(), H.FakeSource(10), "x.mp4", mode="fast",
                           writer_factory=H.FakeWriter)
    r = _small_result()
    for p in r.pitches:
        p.status = "rejected"
    with pytest.raises(ValueError, match="no accepted pitches"):
        O.render_annotated(r, H.FakeSource(1000), "x.mp4", mode="highlights", writer_factory=H.FakeWriter)


def test_writer_aborted_on_error():
    class Boom(H.FakeWriter):
        def write(self, bgr):
            raise RuntimeError("disk full")
    H.FakeWriter.instances.clear()
    with pytest.raises(RuntimeError, match="disk full"):
        O.render_annotated(_small_result(), H.FakeSource(50), "x.mp4", writer_factory=Boom)
    assert H.FakeWriter.instances[-1].closed


def test_text_fits_frame_at_small_size():
    r = _small_result(1)
    r.pitches[0].speed = H.speed(None, None, "none", reason="y" * 300)
    _, w, _ = _render(r, H.FakeSource(200), size=(480, 360))
    assert w.frames[0].shape == (360, 480, 3)


def test_possible_missed_hint():
    r = _small_result(1)
    r.possible_missed = [{"s_start": 150, "s_end": 160, "t_start": 150 * H.FIELD_DT,
                          "t_end": 160 * H.FIELD_DT, "reason": "rejected_track:PARTIAL"},
                         {"t_start": 1.0, "t_end": 2.0, "reason": "no sample indices"}]
    assert O.missed_windows(r) == [(150, 160)]
    _, w, _ = _render(r, H.FakeSource(200))

    def orange(img):
        band = img[40:80, 0:700].astype(int)
        return int((np.abs(band - np.array(O.ORANGE)).sum(axis=2) < 60).sum())
    assert orange(w.frames[155]) > 50 and orange(w.frames[140]) == 0
