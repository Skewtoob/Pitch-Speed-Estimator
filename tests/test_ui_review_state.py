"""Review state machine: key sequences -> markers, statuses, numbering, undo, autosave."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from pitchspeed.review.state import (MANUAL_HINT, ReviewState, StabilizeMap, format_time, renumber,
                                     speed_text)
from pitchspeed.types import EventTime, PitchTrack, SpeedEstimate, to_jsonable

from ui_helpers import FakeSource, Recorder, make_result, speed


def make(windows=((100, 125), (300, 328), (500, 526)), missed=((420, 440),), rejected=0, **kw):
    src = FakeSource(n_frames=400)
    res = make_result(src, windows=windows, missed=missed, rejected=rejected)
    rec = Recorder()
    st = ReviewState(res, src, rec.recompute, rec.save, **kw)
    return st, src, res, rec


def keys(st, *names):
    for n in names:
        st.on_key(n)


def content(pitches):
    """Comparable content of pitches (without edit-history timestamps)."""
    out = to_jsonable(pitches)
    for p in out:
        p["edit_history"] = [{k: v for k, v in h.items() if k != "at"} for h in p["edit_history"]]
    return out


# ------------------------------------------------------------ start / nav --
def test_starts_on_first_pitch():
    st, src, res, _ = make()
    assert st.cur.kind == "pitch" and st.s == 100
    assert st.info_text() == "Pitch 1/3 | auto | 00:00:01.668 | fields 100-125 | 75+/-2 mph (avg 72) | TOF | good"
    assert st.top_text().startswith("REVIEW | item 1/4 | field 100 (frame 50 bottom)")


def test_field_and_item_navigation():
    st, *_ = make()
    keys(st, "right", "d")
    assert st.s == 102
    keys(st, "left")
    assert st.s == 101
    keys(st, "down", "s")
    assert st.s == 121
    keys(st, "w")
    assert st.s == 111
    keys(st, "end")
    assert st.s == 125 and st.marker == "C"
    keys(st, "home")
    assert st.s == 100 and st.marker == "R"
    keys(st, "pgdn")
    assert st.s == 300
    keys(st, "]")
    assert st.cur.kind == "missed" and st.s == 420
    keys(st, "pgdn", "pgdn")
    assert st.s == 500 and "last one" in st.message.text and "Q" in st.message.text
    keys(st, "pgup", "[")
    assert st.s == 300


def test_no_pitches():
    st, *_ = make(windows=(), missed=())
    assert st.cur is None and st.message.level == "warn"
    keys(st, "pgdn")
    assert "nothing to review" in st.message.text
    assert MANUAL_HINT in st.hint_text()


# ----------------------------------------------------------- set R / C --
def test_set_release_and_catch():
    st, src, res, rec = make()
    keys(st, "right", "right", "r")
    p = res.pitches[0]
    assert p.status == "edited" and p.release.method == "manual" and p.release.s == 102
    assert p.release.t == pytest.approx(src.sample_time(102) - src.sample_dt / 2)
    assert p.release.sigma == pytest.approx(src.sample_dt / math.sqrt(12))
    assert p.frame_release == 51
    assert rec.recomputed == [p] and rec.saved == 1
    assert p.edit_history[-1]["action"] == "set_release"
    assert p.edit_history[-1]["from_s"] == 100 and p.edit_history[-1]["to_s"] == 102
    keys(st, "end", "left", "c")
    assert p.catch.s == 124 and p.catch.method == "manual" and rec.saved == 2
    assert "edited" in st.info_text() and "fields 102-124" in st.info_text()


def test_release_after_catch_refused():
    st, _, res, rec = make()
    keys(st, "end", "r")
    assert res.pitches[0].release.s == 100 and st.message.level == "error" and rec.saved == 0
    keys(st, "home", "c")
    assert res.pitches[0].catch.s == 125 and st.message.level == "error"


def test_marker_nudge():
    st, _, res, _ = make()
    keys(st, ".", ".")                                # R selected: move release later
    assert res.pitches[0].release.s == 102 and st.s == 102
    keys(st, "tab", ",")                              # C marker one earlier
    assert res.pitches[0].catch.s == 124 and st.s == 124
    keys(st, "<")
    assert res.pitches[0].catch.s == 123


def test_manual_status_stays_manual():
    st, _, res, _ = make()
    res.pitches[0].status = "manual"
    keys(st, "right", "r")
    assert res.pitches[0].status == "manual"


# ------------------------------------------------------ accept / reject --
def test_reject_renumbers_and_accept_restores():
    st, _, res, rec = make()
    keys(st, "pgdn", "x")
    p2 = res.pitches[1]
    assert p2.status == "rejected" and p2.number == 0
    assert [p.number for p in res.pitches] == [1, 0, 2]
    assert st.cur.kind == "missed"                    # moved on to the next item
    keys(st, "pgup")
    assert st.info_text().startswith("Pitch -/2 (rejected, was #2)")
    keys(st, "x")
    assert "already rejected" in st.message.text
    keys(st, "pgup", "pgdn")
    keys(st, "y")
    assert p2.status == "auto" and p2.number == 2 and [p.number for p in res.pitches] == [1, 2, 3]
    assert p2.edit_history[-1]["action"] == "unreject"


def test_reject_restores_edited_status():
    st, _, res, _ = make()
    keys(st, "right", "r", "x")
    assert res.pitches[0].status == "rejected"
    keys(st, "pgup", "y")
    assert res.pitches[0].status == "edited"


def test_accept_marks_reviewed_and_advances():
    st, _, res, rec = make()
    keys(st, "enter")
    assert res.pitches[0].status == "auto" and res.pitches[0].edit_history[-1]["action"] == "accept"
    assert st.s == 300 and rec.saved == 1
    keys(st, "pgup")
    assert st.info_text().endswith("reviewed")


def test_missed_window_dismiss_toggle_and_accept_hint():
    st, _, res, rec = make()
    keys(st, "pgdn", "pgdn")
    assert st.cur.kind == "missed" and "Possible missed pitch 1/1" in st.info_text()
    keys(st, "y")
    assert "press N" in st.message.text
    keys(st, "x")                                   # mark 'no pitch' and move on
    assert res.possible_missed[0]["review"] == "no_pitch" and rec.saved == 1
    assert st.cur.kind == "pitch" and "no pitch" in st.message.text
    keys(st, "pgup")                                # a handled window is skipped by PgUp / PgDn
    assert st.cur.kind == "pitch"
    keys(st, "o", "pgdn")                           # O shows everything: the window is reachable
    assert st.cur.kind == "missed"
    keys(st, "x")                                   # X again: 'not checked' again
    assert "review" not in res.possible_missed[0]
    keys(st, "u")
    assert res.possible_missed[0]["review"] == "no_pitch"


# ------------------------------------------------------- manual pitches --
def test_new_manual_pitch_in_missed_window():
    st, src, res, rec = make()
    keys(st, "pgdn", "pgdn")                          # missed window at 420
    keys(st, "right", "right", "n")
    assert st.draft == {"s_release": 422, "item": st.cur}
    assert "NEW MANUAL PITCH" in st.info_text()
    keys(st, "s", "s", "right", "c")                  # catch at 443
    assert st.draft is None
    p = [q for q in res.pitches if q.status == "manual"][0]
    assert (p.release.s, p.catch.s) == (422, 443) and p.release.method == p.catch.method == "manual"
    assert p.catch.t - p.release.t == pytest.approx(src.sample_time(443) - src.sample_time(422))
    assert [q.number for q in res.pitches] == [1, 2, 3, 4] and p.number == 3      # chronological
    assert st.current_pitch() is p and p.speed.v_release_mph is not None
    assert p.edit_history[0]["action"] == "create_manual" and p.edit_history[0]["from"] == "missed"
    keys(st, "u")
    assert all(q.status != "manual" for q in res.pitches) and len(res.pitches) == 3
    assert st.draft is not None and st.draft["s_release"] == 422        # back to the half-made pitch


def test_r_on_window_starts_draft_and_c_needs_draft():
    st, *_ = make()
    keys(st, "pgdn", "pgdn")
    keys(st, "c")
    assert st.message.level == "warn" and st.draft is None
    keys(st, "r")
    assert st.draft["s_release"] == 420
    keys(st, "right", "r")                             # R again moves the draft release
    assert st.draft["s_release"] == 421
    keys(st, "c")
    assert st.draft is not None and st.message.level == "error"     # catch must be after release


def test_escape_cancels_draft_and_undo_restores_it():
    st, *_ = make()
    keys(st, "n")
    keys(st, "esc")
    assert st.draft is None and not st.done
    keys(st, "u")
    assert st.draft is not None
    keys(st, "q")
    assert st.quit_armed and not st.done
    keys(st, "q")
    assert st.done and st.draft is None


def test_quit_aborted_by_other_key():
    st, *_ = make()
    keys(st, "n", "q", "right")
    assert not st.quit_armed and st.draft is not None


def test_escape_quits_and_saves():
    st, _, res, rec = make()
    keys(st, "h", "esc")
    assert not st.help and not st.done
    keys(st, "esc")
    assert st.done and rec.saved == 1


# ------------------------------------------------------------ ball points --
def test_click_adds_manual_point_and_excludes_auto():
    st, src, res, rec = make()
    keys(st, "right", "right")                        # field 102: auto point exists
    st.on_click(300.0, 200.0)
    p = res.pitches[0]
    here = [q for q in p.track.points if q.s == 102]
    assert len(here) == 2
    auto = [q for q in here if not q.manual][0]
    man = [q for q in here if q.manual][0]
    assert not auto.inlier and man.inlier and (man.x, man.y) == (300.0, 200.0)
    assert p.status == "edited" and rec.saved == 1
    st.on_click(301.0, 201.0)                         # moves the manual point
    assert len([q for q in p.track.points if q.s == 102 and q.manual]) == 1
    st.pick_radius = 5.0
    st.on_right_click(301.5, 201.0)
    assert not any(q.manual for q in p.track.points)
    st.on_right_click(auto.x + 1, auto.y)             # toggle the auto point back in
    assert [q for q in p.track.points if q.s == 102][0].inlier
    st.on_right_click(50.0, 50.0)
    assert "no ball point" in st.message.text


def test_click_outside_pitch_refused_and_without_pitch():
    st, *_ = make()
    keys(st, "w", "w")                                # 20 fields before release
    st.on_click(100.0, 100.0)
    assert "outside this pitch" in st.message.text
    keys(st, "pgdn", "pgdn")                          # missed window
    st.on_click(100.0, 100.0)
    assert "press N" in st.message.text


def test_click_point_on_manual_pitch_without_track():
    st, _, res, _ = make()
    res.pitches[0].track = None
    st.on_click(200.0, 150.0)
    tr = res.pitches[0].track
    assert isinstance(tr, PitchTrack) and len(tr.points) == 1 and tr.points[0].manual


def test_click_is_stabilized():
    T = np.zeros((1000, 2, 3))
    T[:, 0, 0] = T[:, 1, 1] = 1.0
    T[:, 0, 2], T[:, 1, 2] = 5.0, -2.0                # sample -> reference: shift (+5, -2)
    st, _, res, _ = make(stab=StabilizeMap(T, np.arange(1000)))
    st.on_click(300.0, 200.0)
    man = [q for q in res.pitches[0].track.points if q.manual][0]
    assert (man.x, man.y) == (305.0, 198.0)
    pts = st.overlay_points()
    here = [q for q in pts if q[2]][0]
    assert here[0] == pytest.approx(300.0) and here[1] == pytest.approx(200.0)


# --------------------------------------------------------------- retrack --
def test_retrack_runs_callback_and_recomputes():
    calls = []

    def retrack(p):
        calls.append(p)
        tr = copy.deepcopy(p.track)
        tr.points = tr.points[:5]
        return tr

    st, _, res, rec = make(retrack=retrack)
    keys(st, "e")
    assert st.pending == ("retrack", st.cur.ref) and not calls
    st.run_pending()
    assert len(calls) == 1 and len(res.pitches[0].track.points) == 5
    assert res.pitches[0].status == "edited" and rec.saved == 1 and "re-tracked: 5" in st.message.text


def test_retrack_failure_and_unavailable():
    def bad(p):
        raise RuntimeError("no ball")
    st, _, res, rec = make(retrack=bad)
    keys(st, "e")
    st.run_pending()
    assert st.message.level == "error" and "no ball" in st.message.text and rec.saved == 0
    st2, *_ = make()
    keys(st2, "e")
    assert st2.pending is None and "not available" in st2.message.text


# ---------------------------------------------------- recompute / save errors --
def test_recompute_failure_leaves_no_speed():
    src = FakeSource(n_frames=400)
    res = make_result(src)
    rec = Recorder(fail_recompute=True)
    st = ReviewState(res, src, rec.recompute, rec.save)
    keys(st, "right", "r")
    sp = res.pitches[0].speed
    assert sp.v_release_mph is None and "could not be recomputed" in sp.reason
    assert "no speed" in st.info_text()
    assert rec.saved == 1                              # still saved (with the honest 'no speed')


def test_save_failure_is_shown_and_retried():
    src = FakeSource(n_frames=400)
    res = make_result(src)
    rec = Recorder(fail_save=True)
    st = ReviewState(res, src, rec.recompute, rec.save)
    keys(st, "right", "r")
    assert st.save_error == "disk full" and "AUTOSAVE FAILED" in st.message.text
    rec.fail_save = False
    keys(st, "right", "r")
    assert st.save_error is None and rec.saved == 1


# ------------------------------------------------------------------- undo --
def test_unbounded_undo_restores_everything():
    st, _, res, rec = make(rejected=1)
    original = content(res.pitches)
    missed0 = copy.deepcopy(res.possible_missed)
    keys(st, "right", "r", "end", "left", "c", ".")
    st.on_click(300.0, 200.0)
    keys(st, "x")                                     # reject pitch 1 -> moves on
    keys(st, "x")                                     # reject pitch 2
    keys(st, "x")                                     # dismiss missed window
    keys(st, "n", "s", "c")                           # manual pitch
    keys(st, "o")
    n_undo = len(st.undo_stack)
    assert n_undo >= 9
    for _ in range(n_undo):
        keys(st, "u")
    assert content(res.pitches) == original
    assert res.possible_missed == missed0
    assert "nothing to undo" in (keys(st, "u") or st.message.text)


# ---------------------------------------------------------- rejected tracks --
def test_rejected_tracks_listed_on_demand_and_promoted():
    st, src, res, rec = make(rejected=2)
    n_items = len(st.items())
    keys(st, "o")
    assert len(st.items()) == n_items + 2
    while st.cur.kind != "rtrack":
        keys(st, "pgdn")
    assert "Rejected track #1 | TOO_SLOW" in st.info_text()
    keys(st, "y")
    promoted = [p for p in res.pitches if p.edit_history and p.edit_history[0]["action"] == "promote_rejected_track"]
    assert len(promoted) == 1 and promoted[0].status == "edited" and promoted[0].track.status == "pitch"
    assert res.rejected_tracks[0].diagnostics["review"] == "promoted_to_pitch"
    assert res.rejected_tracks[0].status == "rejected"            # the original is kept
    keys(st, "pgdn")                                  # the promoted track is still listed
    assert st.cur == type(st.cur)("rtrack", 0)
    keys(st, "x", "y")
    assert res.rejected_tracks[0].diagnostics["review"] == "promoted_to_pitch"
    assert "already made into a pitch" in st.message.text
    keys(st, "pgdn")
    assert st.cur.ref == 1
    keys(st, "x")
    assert res.rejected_tracks[1].diagnostics["review"] == "confirmed_not_a_pitch"
    keys(st, "o")
    assert all(it.key.kind != "rtrack" for it in st.items())


def test_draft_from_rejected_track_takes_its_points():
    st, src, res, rec = make(rejected=1)
    keys(st, "o")
    while st.cur.kind != "rtrack":
        keys(st, "pgdn")
    keys(st, "n", "s", "c")
    p = [q for q in res.pitches if q.status == "manual"][0]
    assert p.track is not None and p.track.points and all(200 - 2 <= q.s <= 210 + 2 for q in p.track.points)
    assert res.rejected_tracks[0].diagnostics["review"] == "promoted_to_manual_pitch"


# --------------------------------------------------------------- playback --
def test_playback_loops_in_window_and_speed():
    st, *_ = make()
    w0, w1 = st.window()
    assert (w0, w1) == (40, 185)
    keys(st, "space")
    assert st.playing
    st.s = w1
    st.tick()
    assert st.s == w0
    st.tick()
    assert st.s == w0 + 1
    assert st.play_speed == 0.25
    keys(st, "+", "+", "+")
    assert st.play_speed == 1.0
    keys(st, "-", "-", "-", "-")
    assert st.play_speed == 0.125
    keys(st, "right")
    assert not st.playing
    st.tick()
    assert st.s == w0 + 2


def test_toggles():
    st, *_ = make()
    assert st.show_track
    keys(st, "t", "z", "f", "h")
    assert not st.show_track and st.loupe and st.view_mode == "frame" and st.help


# ---------------------------------------------------------- numbering / save --
def test_renumber_matches_pipeline_rule():
    from pitchspeed.pipeline import renumber as pipeline_renumber
    src = FakeSource(n_frames=400)
    a = make_result(src, windows=((300, 320), (100, 120), (500, 520), (200, 220))).pitches
    a[1].status = "rejected"
    b = copy.deepcopy(a)
    renumber(a)
    pipeline_renumber(b)
    assert [(p.release.s, p.number) for p in a] == [(p.release.s, p.number) for p in b]


def test_state_follows_lead_save_that_resorts():
    from pitchspeed.pipeline import renumber as pipeline_renumber
    src = FakeSource(n_frames=400)
    res = make_result(src)
    saves = []

    def save(r):
        pipeline_renumber(r.pitches)
        saves.append([p.number for p in r.pitches])

    st = ReviewState(res, src, lambda p: p, save)
    keys(st, "pgdn", "pgdn", "pgdn")                  # third pitch
    st.goto(290)
    keys(st, "r")                                     # release moved before the 2nd pitch's
    assert st.current_pitch() is res.pitches[1] and st.current_pitch().number == 2
    assert saves[-1] == [1, 2, 3]


def test_finish_saves():
    st, _, _, rec = make()
    st.finish()
    assert st.done and rec.saved == 1


# ----------------------------------------------------------------- helpers --
def test_speed_text_variants():
    assert speed_text(speed(78.4, 3.2)) == "78+/-3 mph (avg 75)"
    assert speed_text(speed(78.4, 1.4)) == "78+/-1.4 mph (avg 75)"
    poor = speed(70.0, 6.0, label="poor")
    assert speed_text(poor).startswith("~70")
    assert speed_text(SpeedEstimate(v_release_mph=None, reason="ball lost")) == "no speed: ball lost"
    assert speed_text(None) == "no speed"


def test_format_time():
    assert format_time(83.4567) == "00:01:23.457"
    assert format_time(3725.0) == "01:02:05.000"
    assert format_time(float("nan")) == "--:--:--.---"


def test_stabilize_map():
    T = np.array([[[1.0, 0.0, 3.0], [0.0, 1.0, 1.0]], [[1.0, 0.0, 10.0], [0.0, 1.0, 0.0]]])
    m = StabilizeMap(T, [100, 200], max_gap=3)
    assert m.to_ref(101, 5.0, 5.0) == (8.0, 6.0)
    assert m.from_ref(101, 8.0, 6.0) == pytest.approx((5.0, 5.0))
    assert m.to_ref(150, 5.0, 5.0) == (5.0, 5.0)        # no transform near: identity
    assert m.to_ref(198, 0.0, 0.0) == (10.0, 0.0)
    assert StabilizeMap().to_ref(5, 1.0, 2.0) == (1.0, 2.0)
    with pytest.raises(ValueError):
        StabilizeMap(T, [1, 2, 3])


def test_event_without_sample_index_uses_time():
    st, src, res, _ = make()
    res.pitches[0].release = EventTime(t=src.sample_time(110), sigma=0.004, method="first_free_field", s=None)
    assert st.ev_s(res.pitches[0].release) == 110


def test_help_rows_cover_bindings():
    st, *_ = make()
    rows = st.help_rows()
    assert rows[0][1] == MANUAL_HINT and len(rows) > 25


def test_markers_for_strip():
    st, *_ = make()
    mk = st.markers()
    assert (mk["R"], mk["C"], mk["s"]) == (100, 125, 100) and len(mk["points"]) == 26
    keys(st, "pgdn", "pgdn")
    mk = st.markers()
    assert mk["R"] is None and mk["span"] == (420, 440)
