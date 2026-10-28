"""Regression tests for the verified review / calibration UI findings.

* manual C on a track that ended at a possible bat contact (ui #2);
* full-frame view: stepping and marking (ui #4);
* 3x3 homographies for ball points (ui #5);
* checked 'possible missed' windows recorded for the outputs (ui #6);
* closing the calibration window with clicked points (ui #8).
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from pitchspeed.review.state import ReviewState, StabilizeMap, effective_end, end_text
from pitchspeed.types import EventTime, PitchTrack, TrackPoint

from ui_helpers import FakeSource, Recorder, ScriptedWindow, key, make_pitch, make_result


def make(src=None, windows=((100, 125), (300, 328), (500, 526)), missed=((420, 440),), rejected=0,
         recompute=None, **kw):
    src = src or FakeSource(n_frames=400)
    res = make_result(src, windows=windows, missed=missed, rejected=rejected)
    rec = Recorder()
    st = ReviewState(res, src, recompute or rec.recompute, rec.save, **kw)
    return st, src, res, rec


def keys(st, *names):
    for n in names:
        st.on_key(n)


# ---------------------------------------------- ui #2: C over a 'contact' end --
def _behind_estimator(src):
    """recompute() with the real estimator on a behind-view calibration."""
    from estimate_helpers import behind_calibration, behind_camera
    from pitchspeed.estimate import estimate_speed
    calib = behind_calibration(behind_camera(20.0, 6.0, 60.5), 60.5, np.random.default_rng(3))

    def recompute(p):
        p.speed = estimate_speed(calib, src.scan, src.sample_dt, p.release, p.catch, track=p.track)
        return p
    return recompute


def test_manual_catch_on_contact_track_ends_at_the_glove():
    src = FakeSource(n_frames=400)
    rc = _behind_estimator(src)
    st, _, res, _ = make(src, windows=((100, 130),), missed=(), recompute=rc)
    p = res.pitches[0]
    p.track.end_kind = "contact"            # the finder thought the bat hit the ball
    rc(p)
    assert p.speed.assumptions["end_point"] == "contact"
    assert effective_end(p) == "contact" and "BAT CONTACT" in st.info_text()
    keys(st, "end", "c")                    # the user marks the FIRST field with the ball in the glove
    p = st.current_pitch()
    assert p.catch.method == "manual" and p.catch.s == 130
    assert p.speed.assumptions["end_point"] == "catch"        # glove distance, not the contact one
    assert effective_end(p) == "catch" and "ends at the glove (C marked)" in st.info_text()
    assert st.message.level == "warn" and "IN THE GLOVE" in st.message.text and "press U" in st.message.text
    assert p.track.end_kind == "contact"     # the finder's observation is kept (diagnostics)
    keys(st, "u")
    p = st.current_pitch()
    assert p.catch.method == "extrapolated" and p.speed.assumptions["end_point"] == "contact"


def test_end_text_quiet_for_an_ordinary_catch():
    src = FakeSource(n_frames=200)
    p = make_pitch(src, 50, 75)
    assert effective_end(p) == "catch" and end_text(p) == ""
    p.track = None
    assert effective_end(p) == "catch" and end_text(p) == ""


def test_release_mark_on_contact_track_keeps_contact_end():
    st, _, res, _ = make(windows=((100, 125),), missed=())
    res.pitches[0].track.end_kind = "contact"
    keys(st, "right", "r")
    assert effective_end(st.current_pitch()) == "contact"
    assert "BAT CONTACT" in st.info_text() and st.message.level == "ok"


# ------------------------------------------------- ui #4: full-frame view --
def test_frame_view_steps_one_frame_per_press():
    st, src, *_ = make()
    keys(st, "f")
    assert st.view_mode == "frame" and st.woven()
    f0 = src.locate(st.s)[0]
    keys(st, "right")
    assert src.locate(st.s)[0] == f0 + 1     # every press shows a different picture
    keys(st, "left", "left")
    assert src.locate(st.s)[0] == f0 - 1
    keys(st, "f")
    s = st.s
    keys(st, "right")
    assert st.s == s + 1                     # back to one field per press


@pytest.mark.parametrize("action", ["r", "c", "n", ",", ".", "click", "rclick"])
def test_frame_view_refuses_marks(action):
    st, src, res, rec = make()
    keys(st, "right", "right", "f")
    before = copy.deepcopy(res.pitches)
    n_undo = len(st.undo_stack)
    if action == "click":
        st.on_click(500.0, 180.0)
    elif action == "rclick":
        st.on_right_click(600.0, 150.0)
    else:
        keys(st, action)
    assert st.message.level == "warn" and "press F" in st.message.text
    assert len(st.undo_stack) == n_undo and st.draft is None and rec.saved == 0
    assert [(p.release.s, p.catch.s, p.status, len(p.track.points)) for p in res.pitches] == \
        [(p.release.s, p.catch.s, p.status, len(p.track.points)) for p in before]
    keys(st, "f")                            # field view: the same key works
    if action == "click":
        st.on_click(500.0, 180.0)
        assert any(q.manual for q in res.pitches[0].track.points)
    elif action == "r":
        keys(st, "r")
        assert res.pitches[0].release.s == 102 and res.pitches[0].release.method == "manual"


def test_frame_view_draft_catch_refused_until_field_view():
    st, *_ = make(windows=(), missed=())
    keys(st, "n")
    for _ in range(20):
        keys(st, "right")
    keys(st, "f", "c")
    assert st.draft is not None and "press F" in st.message.text and not st.pitches
    keys(st, "f", "c")
    assert st.draft is None and len(st.pitches) == 1 and st.pitches[0].catch.s == 20


def test_frame_view_shows_marks_and_points_of_both_fields():
    st, src, res, _ = make()                 # pitch 1: R = 100, C = 125, a point on every field
    keys(st, "right", "f")                   # field 101, then the full frame 50 (fields 100 + 101)
    assert st.s == 101 and st.shows(100) and st.shows(101) and not st.shows(102)
    here = [q for q in st.overlay_points() if q[4]]
    assert len(here) == 2
    keys(st, "f")
    assert not st.shows(100) and len([q for q in st.overlay_points() if q[4]]) == 1


def test_progressive_video_marks_in_frame_view():
    src = FakeSource(n_frames=400, field_order=None)
    st, _, res, _ = make(src, windows=((100, 125),), missed=())
    assert st.view_mode == "frame" and not st.woven()
    keys(st, "right", "r")
    assert st.s == 101 and res.pitches[0].release.s == 101


# ------------------------------------------------- ui #5: 3x3 homographies --
def _homography(n, s0=0):
    H = np.tile(np.eye(3), (n, 1, 1))
    H[:, 0, 0], H[:, 0, 1], H[:, 0, 2] = 1.02, 0.01, 4.0
    H[:, 1, 0], H[:, 1, 1], H[:, 1, 2] = -0.015, 0.99, -3.0
    H[:, 2, 0], H[:, 2, 1] = 2e-4, -1e-4                    # projective part (pan with a wide lens)
    return H, np.arange(s0, s0 + n)


def _apply(H, x, y):
    v = H @ np.array([x, y, 1.0])
    return v[0] / v[2], v[1] / v[2]


def test_stabilize_map_applies_homographies_projectively():
    H, S = _homography(50, 100)
    m = StabilizeMap(H, S)
    x, y = m.to_ref(120, 600.0, 150.0)
    ex, ey = _apply(H[20], 600.0, 150.0)
    assert (x, y) == pytest.approx((ex, ey), abs=1e-9)
    # the affine part alone would be several px off here
    assert math.hypot(x - (1.02 * 600 + 0.01 * 150 + 4.0), y - (-0.015 * 600 + 0.99 * 150 - 3.0)) > 5
    assert m.from_ref(120, x, y) == pytest.approx((600.0, 150.0), abs=1e-9)
    assert m.matrix(120).shape == (3, 3)


def test_stabilize_map_accepts_both_shapes():
    A = np.array([[[1.0, 0.0, 3.0], [0.0, 1.0, 1.0]]] * 3)
    H = np.tile(np.eye(3), (3, 1, 1))
    H[:, :2, :] = A
    for T in (A, H):
        m = StabilizeMap(T, [5, 6, 7])
        assert m.to_ref(6, 5.0, 5.0) == pytest.approx((8.0, 6.0))
        assert m.from_ref(6, 8.0, 6.0) == pytest.approx((5.0, 5.0))
    with pytest.raises(ValueError):
        StabilizeMap(np.zeros((3, 4, 4)), [1, 2, 3])


def test_stabilize_map_degenerate_matrix_is_identity():
    H = np.zeros((2, 3, 3))
    m = StabilizeMap(H, [10, 11])
    assert m.to_ref(10, 7.0, 8.0) == (7.0, 8.0) and m.from_ref(11, 7.0, 8.0) == (7.0, 8.0)


def test_review_clicks_and_drawing_use_the_homography():
    H, S = _homography(1000)
    st, _, res, _ = make(stab=StabilizeMap(H, S))
    st.on_click(300.0, 200.0)
    man = [q for q in res.pitches[0].track.points if q.manual][0]
    ex, ey = _apply(H[100], 300.0, 200.0)
    assert (man.x, man.y) == pytest.approx((ex, ey), abs=1e-3)
    here = [q for q in st.overlay_points() if q[2]][0]          # drawn where it was clicked
    assert here[0] == pytest.approx(300.0, abs=1e-2) and here[1] == pytest.approx(200.0, abs=1e-2)
    auto = res.pitches[0].track.points[5]
    st.s = auto.s
    drawn = [q for q in st.overlay_points() if not q[2] and q[4]][0]
    assert _apply(H[auto.s], drawn[0], drawn[1]) == pytest.approx((auto.x, auto.y), abs=1e-6)


def test_load_transforms_feeds_the_review_map(tmp_path):
    """pipeline.load_transforms (3x3 when stored) -> StabilizeMap: the finder's exact mapping."""
    from pitchspeed.pipeline import load_transforms, out_paths
    H, S = _homography(40, 200)
    A = H[:, :2, :] / H[:, 2:3, 2:3]                       # what a 2x3 approximation would store
    np.savez_compressed(out_paths(str(tmp_path))["transforms"], samples=S, transforms=A,
                        quality=np.ones(len(S)), transforms_3x3=H)
    T, samples = load_transforms(str(tmp_path))
    m = StabilizeMap(T, samples)
    assert m.to_ref(210, 650.0, 120.0) == pytest.approx(_apply(H[10], 650.0, 120.0), abs=1e-9)


# ----------------------------------------- ui #6: checked windows recorded --
def test_manual_pitch_from_window_marks_it_checked():
    st, src, res, _ = make(windows=((100, 125),), missed=((420, 440),))
    keys(st, "pgdn")
    assert st.cur.kind == "missed"
    keys(st, "n")
    for _ in range(24):
        keys(st, "right")
    keys(st, "c")
    m = res.possible_missed[0]
    assert m["review"] == "pitch_added"
    keys(st, "pgdn")                                    # the window comes after the new pitch
    assert st.cur.kind == "missed" and "checked: pitch added" in st.info_text()
    keys(st, "x")                                       # X must not hide the added pitch
    assert m["review"] == "pitch_added" and "select that pitch" in st.message.text
    keys(st, "u")                                       # undo the manual pitch: window unchecked again
    assert "review" not in m and len(st.pitches) == 1


def test_manual_pitch_elsewhere_marks_only_overlapping_windows():
    st, src, res, _ = make(windows=(), missed=((420, 440), (700, 720)))
    st.goto(418)
    keys(st, "n")
    st.goto(445)
    keys(st, "c")
    assert res.possible_missed[0].get("review") == "pitch_added"
    assert "review" not in res.possible_missed[1]


def test_pitch_added_overrides_an_earlier_no_pitch():
    st, src, res, _ = make(windows=(), missed=((420, 440),))
    keys(st, "x")
    assert res.possible_missed[0]["review"] == "no_pitch"
    st.goto(421)
    keys(st, "n")
    st.goto(446)
    keys(st, "c")
    assert res.possible_missed[0]["review"] == "pitch_added"
    keys(st, "u")
    assert res.possible_missed[0]["review"] == "no_pitch"


def test_promoted_track_marks_its_window_checked():
    st, src, res, _ = make(windows=(), missed=((200, 220),), rejected=1)    # rejected track 200..220
    keys(st, "o")
    items = [it.key for it in st.items()]
    st._goto_item([k for k in items if k.kind == "rtrack"][0])
    keys(st, "y")
    assert len(st.pitches) == 1 and res.possible_missed[0]["review"] == "pitch_added"
    keys(st, "u")
    assert "review" not in res.possible_missed[0]


def test_window_review_values_are_known_to_the_report():
    from pitchspeed.render.report import MISSED_REVIEW_STATUS
    st, src, res, _ = make(windows=(), missed=((420, 440), (700, 720)))
    keys(st, "x")                                        # first window: checked, no pitch
    st.goto(698)
    keys(st, "n")
    st.goto(725)
    keys(st, "c")
    vals = {m["review"] for m in res.possible_missed}
    assert vals == {"no_pitch", "pitch_added"} and vals <= set(MISSED_REVIEW_STATUS)


# --------------------------------------- ui #8: calibration window closed --
from pitchspeed.calib.ui import run_calibration_ui            # noqa: E402
from pitchspeed.ui.canvas import Canvas                       # noqa: E402

CANVAS = Canvas((720, 480), 8 / 9, (720, 540))


def click(x, y):
    u, v = (int(round(c)) for c in CANVAS.to_display([(x, y)])[0])
    return [("mouse", "ldown", u, v), ("mouse", "lup", u, v)]


class Asker:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.answers.pop(0) if self.answers else None


def _run(script, ask, view="side", **kw):
    src = FakeSource(n_frames=100, **kw)
    w = ScriptedWindow(script)
    cf = run_calibration_ui(src, src.info, view, window=w, ask=ask, max_iterations=60)
    return cf, w


def test_close_with_all_points_saves_on_enter(capsys):
    ask = Asker("")
    cf, w = _run(click(400, 201) + click(100, 300) + [("close",)], ask)
    assert cf is not None and len(ask.prompts) == 1 and "[Y/n/e]" in ask.prompts[0]
    assert cf.segments[0].points["rubber"] == pytest.approx([400, 201], abs=0.5)
    assert "closed before saving" in capsys.readouterr().out


def test_close_with_all_points_and_no_discards():
    cf, _ = _run(click(400, 201) + click(100, 300) + [("close",)], Asker("n"))
    assert cf is None


def test_close_then_keep_editing_reopens_the_window():
    script = click(400, 201) + [("close",)] + click(100, 300) + [key("enter"), key("enter")]
    ask = Asker("")                          # incomplete: Enter = back to the window
    cf, w = _run(script, ask)
    assert getattr(w, "reopened", 0) == 1 and "e (or just Enter)" in ask.prompts[0]
    assert cf is not None and set(cf.segments[0].points) == {"rubber", "plate"}


def test_close_incomplete_and_no_saves_nothing():
    cf, w = _run(click(400, 201) + [("close",)], Asker("n"))
    assert cf is None and getattr(w, "reopened", 0) == 0


def test_close_without_console():
    """No console to ask (stdin closed): a complete calibration is saved, an incomplete one cannot be."""
    cf, _ = _run(click(400, 201) + click(100, 300) + [("close",)], Asker())
    assert cf is not None
    cf, _ = _run(click(400, 201) + [("close",)], Asker())
    assert cf is None
    # Esc once = the user was quitting without saving: no silent save then
    cf, _ = _run(click(400, 201) + click(100, 300) + [key("esc"), ("close",)], Asker())
    assert cf is None


def test_close_after_esc_defaults_to_discard():
    ask = Asker("")
    cf, _ = _run(click(400, 201) + click(100, 300) + [key("esc"), ("close",)], ask)
    assert cf is None and "[y/N/e]" in ask.prompts[0]


def test_close_unchanged_does_not_ask():
    ask = Asker("y")
    cf, _ = _run([None, ("close",)], ask)
    assert cf is None and ask.prompts == []


def test_close_save_records_frame_warnings(capsys):
    """The warnings Enter would show are printed, and kept in the notes as on Enter."""
    script = [("trackbar", 10), None] + click(360, 150) + [("trackbar", 22), None] + click(370, 300) + \
             [("close",)]
    cf, _ = _run(script, Asker("y"), view="behind", pan=(0.25, 0.0))
    assert cf is not None and "warning at save" in cf.notes and "moved about" in cf.notes
    assert "Please check before saving" in capsys.readouterr().out


def test_close_answers_are_validated():
    ask = Asker("maybe", "y")
    cf, _ = _run(click(400, 201) + click(100, 300) + [("close",)], ask)
    assert cf is not None and len(ask.prompts) == 2


def test_close_when_the_window_cannot_reopen_asks_save_or_discard():
    class NoReopen(ScriptedWindow):
        reopen = None

    src = FakeSource(n_frames=100)
    ask = Asker("e", "y")
    w = NoReopen(click(400, 201) + click(100, 300) + [("close",)])
    cf = run_calibration_ui(src, src.info, "side", window=w, ask=ask, max_iterations=60)
    assert cf is not None and len(ask.prompts) == 2 and "/e]" not in ask.prompts[1]
