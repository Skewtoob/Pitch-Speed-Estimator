"""Side-by-side view (G): the fields around the shown one next to each other, magnified, to
pick the exact release / catch field."""
from __future__ import annotations

import numpy as np

from pitchspeed.review import ui
from pitchspeed.review.state import (STRIP_BEFORE, STRIP_N, STRIP_WHOLE, STRIP_ZOOMS, STRIP_ZOOM_START,
                                     ReviewState)
from pitchspeed.ui import canvas as cv
from pitchspeed.ui.frames import FrameStore
from test_ui_mark_mode import make_mark
from test_ui_review_fixes import keys
from ui_helpers import FakeSource, Recorder, ScriptedWindow, key, make_result

DOT = (400.0, 201.0)            # FakeSource's bright 3x3 dot (frame px)


def _canvas():
    return cv.Canvas((720, 480), 8 / 9, cv.video_size_for(1.0), strip_h=ui.STRIP_H)


def _open(st, s, pointer=None):
    st.goto(s)
    st.pointer = pointer
    keys(st, "g")
    st.pointer = None


def test_g_shows_the_fields_around_the_shown_one_and_closes_again():
    st, src, res, _ = make_mark()
    _open(st, 1000, pointer=DOT)
    assert st.strip and st.strip_samples() == list(range(1000 - STRIP_BEFORE, 1000 - STRIP_BEFORE + STRIP_N))
    assert st.strip_focus == DOT and st.strip_zoom == STRIP_ZOOMS[STRIP_ZOOM_START]
    assert "SIDE BY SIDE" in st.message.text and "at the mouse" in st.message.text
    assert "SIDE BY SIDE" in st.top_text() and "pick the picture" in st.hint_text()
    keys(st, "g")
    assert not st.strip and st.strip_samples() == []
    _open(st, 2000)                                   # no mouse on the picture: keeps the old spot
    assert st.strip_focus == DOT and st.strip_zoom == STRIP_ZOOMS[STRIP_ZOOM_START]
    assert "same spot as last time" in st.message.text and "at the mouse" not in st.message.text
    keys(st, "esc")                                   # Esc closes it, nothing else
    assert not st.strip and not st.done


def test_a_second_esc_after_closing_does_not_quit_but_q_does():
    st, *_ = make_mark()
    _open(st, 1000, pointer=DOT)
    keys(st, "esc", "esc", "esc")                     # pressed twice (or held): still open for marking
    assert not st.strip and not st.done and "Q saves and quits" in st.message.text
    keys(st, "d", "esc")                              # any other key first: Esc quits as before
    assert st.done
    st2, *_ = make_mark()
    keys(st2, "esc")                                  # never opened: unchanged
    assert st2.done


def test_without_a_spot_it_shows_whole_pictures_and_a_right_click_magnifies():
    st, src, res, _ = make_mark()
    canvas = _canvas()
    _open(st, 500)
    assert st.strip_focus is None and st.strip_zoom == STRIP_WHOLE
    assert "whole pictures" in st.message.text and "right-click" in st.message.text
    assert "SIDE BY SIDE whole" in st.top_text()
    tile = ui.strip_tiles(canvas)[2]
    x0, y0, tw, th = tile
    tl = ui.tile_to_frame(st, canvas, tile, x0, y0 + (th - 1) / 2)          # left edge, middle
    br = ui.tile_to_frame(st, canvas, tile, x0 + tw - 1, y0 + (th - 1) / 2)
    assert tl[0] < 5 and br[0] > 715                                     # the whole width is in the tile
    centre = (x0 + (tw - 1) / 2.0, y0 + (th - 1) / 2.0)
    u, v = centre[0] + 20, centre[1] - 10                                 # right-click off-centre
    want = ui.tile_to_frame(st, canvas, tile, u, v)
    ui._feed_mouse(st, canvas, "rdown", int(u), int(v))
    assert st.strip_zoom == STRIP_ZOOMS[STRIP_ZOOM_START]
    assert abs(st.strip_focus[0] - want[0]) < 6 and abs(st.strip_focus[1] - want[1]) < 6
    keys(st, *"---")                                  # 3x -> 2x -> 1.5x -> whole again
    assert st.strip_zoom == STRIP_WHOLE and "whole pictures" in st.message.text


def test_the_set_slides_only_when_the_pick_leaves_it_and_long_jumps_start_afresh():
    st, *_ = make_mark()
    _open(st, 1000)
    first = st.strip_samples()
    keys(st, "d", "d", "d", "d")                       # 1004 = last picture: no slide yet
    assert st.s == 1004 and st.strip_samples() == first
    keys(st, "d")                                     # one past: slides by one picture
    assert st.strip_samples() == [x + 1 for x in first] and st.strip_samples()[-1] == st.s
    keys(st, *"aaaaaaaa")                              # back past the start: slides back
    assert st.strip_samples()[0] == st.s == 997
    keys(st, "s")                                     # 10 fields on: starts afresh around it
    assert st.strip_samples()[STRIP_BEFORE] == st.s == 1007
    st.goto(4000)                                     # far away (bar click)
    assert st.strip_samples()[STRIP_BEFORE] == st.s
    keys(st, "f")                                     # full frame: frames around it, s 4th again
    assert st.strip_samples()[STRIP_BEFORE] == st.s
    keys(st, "s")                                     # 10 fields = 5 frames: afresh, not at the edge
    assert st.strip_samples()[STRIP_BEFORE] == st.s
    keys(st, "f")
    assert st.strip_samples()[STRIP_BEFORE] == st.s


def test_the_set_stays_inside_the_video():
    st, *_ = make_mark()
    _open(st, 1)
    assert st.strip_samples() == list(range(STRIP_N))
    st.goto(st.n_samples - 1)
    assert st.strip_samples() == list(range(st.n_samples - STRIP_N, st.n_samples))
    short = FakeSource(n_frames=3)                    # 6 fields: fewer pictures than STRIP_N
    st2 = ReviewState(make_result(short, windows=(), missed=()), short, lambda p: p, lambda r: None, mode="mark")
    _open(st2, 4)
    assert st2.strip_samples() == list(range(6))


def test_marks_made_side_by_side_land_on_the_picked_field_and_show_on_it():
    st, src, res, rec = make_mark()
    _open(st, 1000)
    st.strip_pick(1002)                               # a click on the 6th picture
    keys(st, "n")
    assert st.strip_tags(1002) == ["new R"] and st.strip_tags(1001) == []
    st.goto(1029)                                     # the set starts afresh around the catch
    st.strip_pick(1030)
    keys(st, "c")
    p = res.pitches[0]
    assert p.release.s == 1002 and p.catch.s == 1030 and p.status == "manual"
    assert st.strip_tags(1030) == ["C"]
    st.goto(1002)
    assert st.strip_tags(1002) == ["R"]
    st.strip_pick(5000)                               # not one of the pictures: ignored
    assert st.s == 1002


def test_plus_minus_zoom_while_open_and_space_closes_and_plays():
    st, *_ = make_mark()
    _open(st, 1000, pointer=DOT)
    speed = st.play_speed
    keys(st, "+")
    assert st.strip_zoom == STRIP_ZOOMS[STRIP_ZOOM_START + 1] and st.play_speed == speed
    keys(st, *"---------")
    assert st.strip_zoom == STRIP_WHOLE and st.play_speed == speed
    keys(st, "space")
    assert not st.strip and st.playing
    keys(st, "space", "g")                            # G pauses nothing: playback was stopped
    keys(st, "space")
    assert not st.strip and st.playing
    keys(st, "g")                                     # G while playing stops it and opens
    assert st.strip and not st.playing


def test_esc_closes_the_view_before_it_cancels_a_half_made_pitch():
    st, src, res, _ = make_mark()
    _open(st, 1000)
    keys(st, "n", "esc")
    assert not st.strip and st.draft is not None
    keys(st, "esc")
    assert st.draft is None


def test_full_frame_view_shows_frames_and_refuses_marks_as_before():
    src = FakeSource(n_frames=400)
    st = ReviewState(make_result(src, windows=((100, 126),), missed=()), src, Recorder().recompute,
                     lambda r: None)
    keys(st, "f")
    st.goto(200)
    keys(st, "g")
    ss = st.strip_samples()
    assert len(ss) == STRIP_N and all(b - a == 2 for a, b in zip(ss, ss[1:])) and st.s in ss
    keys(st, "r")
    assert "press F" in st.message.text


def test_review_mode_tags_and_track_points():
    src = FakeSource(n_frames=400)
    res = make_result(src, windows=((100, 126),), missed=())
    st = ReviewState(res, src, Recorder().recompute, lambda r: None)
    st.goto(101)
    keys(st, "g")
    assert st.strip_tags(100) == ["R"] and st.strip_tags(101) == []
    pts = st.strip_points(103)
    assert len(pts) == 1 and pts[0][:2] == (600 - 15 * 3, 150 + 2 * 3)


# ------------------------------------------------------------------ drawing
def test_every_tile_magnifies_the_same_spot_of_the_frame():
    """The dot at the chosen spot sits at the centre of every tile, in both field parities."""
    st, src, res, _ = make_mark()
    canvas = _canvas()
    _open(st, 1001, pointer=DOT)
    img = ui.render(st, FrameStore(src), canvas)
    assert img.shape == (canvas.size[1], canvas.size[0], 3)
    for tile, t in zip(ui.strip_tiles(canvas), st.strip_samples()):
        x0, y0, tw, th = tile
        cx, cy = x0 + (tw - 1) / 2.0, y0 + (th - 1) / 2.0
        r = 20
        win = img[int(cy) - r:int(cy) + r + 1, int(cx) - r:int(cx) + r + 1, 1].astype(float)
        ys, xs = np.nonzero(win > 230)                    # the dot, magnified (background < 200 here)
        assert len(xs) > 10, t
        # a wrong or missing bob parity puts it 1 frame row (3.4 display px at 3x) off centre
        assert abs(xs.mean() - r - (cx - int(cx))) < 1.5 and abs(ys.mean() - r - (cy - int(cy))) < 1.5, t
        far = img[y0 + 30:y0 + 40, x0 + 10:x0 + 20, 1]
        assert far.max() < 225                            # not everywhere
    # the shown field has the yellow border
    i = st.strip_samples().index(st.s)
    x0, y0, tw, th = ui.strip_tiles(canvas)[i]
    border = img[y0 + th // 2, x0 + 1].astype(int)
    assert np.abs(border - np.array(cv.YELLOW)).sum() < 60


def test_zoom_changes_the_magnification():
    st, src, res, _ = make_mark()
    canvas = _canvas()
    _open(st, 1000, pointer=DOT)
    tile = ui.strip_tiles(canvas)[0]
    store = FrameStore(src)
    x0, y0, tw, th = tile

    def dot_pixels():                                 # around the tile centre (not the white label)
        img = ui.render(st, store, canvas)
        cx, cy = x0 + tw // 2, y0 + th // 2
        return int((img[cy - 40:cy + 40, cx - 40:cx + 40, 1] > 230).sum())
    a, n_a = ui.tile_to_frame(st, canvas, tile, tile[0] + tile[2] - 1, tile[1]), dot_pixels()
    keys(st, "+")
    b, n_b = ui.tile_to_frame(st, canvas, tile, tile[0] + tile[2] - 1, tile[1]), dot_pixels()
    assert abs(b[0] - DOT[0]) < abs(a[0] - DOT[0])         # closer to the spot = more magnified
    assert n_b > 1.3 * n_a                                # and the drawn dot really is bigger


def test_clicks_pick_a_tile_and_right_clicks_move_the_spot():
    st, src, res, _ = make_mark()
    canvas = _canvas()
    _open(st, 1000, pointer=DOT)
    tiles = ui.strip_tiles(canvas)
    x0, y0, tw, th = tiles[6]
    centre = (x0 + (tw - 1) / 2.0, y0 + (th - 1) / 2.0)
    fx, fy = ui.tile_to_frame(st, canvas, tiles[6], *centre)
    assert abs(fx - DOT[0]) < 0.5 and abs(fy - DOT[1]) < 0.5     # the tile centre is the spot
    ui._feed_mouse(st, canvas, "ldown", int(centre[0]), int(centre[1]))
    assert st.s == st.strip_samples()[6] and not res.pitches                   # picked, no ball point
    ui._feed_mouse(st, canvas, "rdown", x0 + 5, y0 + 5)
    nx, ny = st.strip_focus
    assert nx < DOT[0] and ny < DOT[1]                                          # moved up-left
    before, focus = st.s, st.strip_focus
    gx, gy = tiles[0][0] + tiles[0][2] + 1, tiles[0][1] + 10                    # the gap between tiles
    ui._feed_mouse(st, canvas, "ldown", gx, gy)
    ui._feed_mouse(st, canvas, "rdown", gx, gy)
    assert st.s == before and st.strip_focus == focus and not res.pitches


def test_scripted_session_marks_with_the_mouse_spot_and_a_tile_click():
    """G with the mouse on the picture magnifies there; a tile click picks the field N marks."""
    src = FakeSource(n_frames=400)
    res = make_result(src, windows=(), missed=())
    rec = Recorder()
    canvas = _canvas()
    u, v = canvas.to_display([DOT])[0]
    tiles = ui.strip_tiles(canvas)
    x0, y0, tw, th = tiles[5]
    script = [("mouse", "move", int(u), int(v)), ord("g"), ("mouse", "ldown", x0 + tw // 2, y0 + th // 2),
              ord("n"), key("down"), ord("c"), ord("g"), ord("q")]
    w = ScriptedWindow(script, keep_images=True)
    ui.run_review_ui(res, src, None, rec.recompute, rec.save, window=w, max_iterations=60, mode="mark")
    assert len(res.pitches) == 1
    p = res.pitches[0]
    assert p.release.s == 5 and p.catch.s == 15           # opened at 0: pictures 0..7, the 6th clicked


def test_long_messages_wrap_onto_a_second_line():
    st, src, res, _ = make_mark()
    canvas = _canvas()
    img = ui.render(st, FrameStore(src), canvas)                 # the start instructions are long
    vh = canvas.video_size[1]
    second = img[vh + 82:vh + 98, 8:400, 1]
    assert second.max() > 150                                     # green text on the second line


def test_undecodable_fields_are_tried_once_not_on_every_redraw():
    src = FakeSource(n_frames=400, fail_frames={201})
    calls = []
    orig = src.get_sample

    def counting(s, color=False):
        calls.append(s)
        return orig(s, color=color)
    src.get_sample = counting
    st = ReviewState(make_result(src, windows=(), missed=()), src, lambda p: p, lambda r: None, mode="mark")
    store, canvas = FrameStore(src), _canvas()
    _open(st, 400)                                     # fields 397..404; 402/403 = frame 201
    for _ in range(3):
        ui.render(st, store, canvas)
    assert calls.count(402) == 1 and calls.count(403) == 1 and store.known(402)
    assert src.iter_calls == 1                         # one sequential read, not one per redraw
    assert store.errors


def test_the_loupe_is_not_drawn_over_the_pictures_side_by_side():
    st, src, res, _ = make_mark()
    canvas, store = _canvas(), FrameStore(src)
    keys(st, "z")
    _open(st, 1000, pointer=DOT)
    tile = ui.strip_tiles(canvas)[5]
    mouse = (tile[0] + 20, tile[1] + 20)
    np.testing.assert_array_equal(ui.render(st, store, canvas, mouse), ui.render(st, store, canvas, None))
    keys(st, "g")                                     # back in the normal view the loupe is still on
    assert st.loupe


def test_an_undecodable_field_says_so_on_its_picture():
    src = FakeSource(n_frames=400, fail_frames={201})
    st = ReviewState(make_result(src, windows=(), missed=()), src, lambda p: p, lambda r: None, mode="mark")
    canvas = _canvas()
    _open(st, 400, pointer=DOT)
    img = ui.render(st, FrameStore(src), canvas)
    red = (img[..., 2] > 200) & (img[..., 1] < 120) & (img[..., 0] < 120)
    for tile, t in zip(ui.strip_tiles(canvas), st.strip_samples()):
        x0, y0, tw, th = tile
        n = int(red[y0:y0 + th, x0:x0 + tw].sum())
        assert (n > 30) == (t in (402, 403)), (t, n)
