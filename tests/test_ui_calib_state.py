"""Calibration state machine: key/mouse sequences -> points, frames, undo, segments, masks."""
from __future__ import annotations

import json

import numpy as np
import pytest

from pitchspeed.calib.model import CalibrationFile, Segment, Settings
from pitchspeed.calib.state import (NUDGE_PX, CalibState, estimate_shift, point_frames_from_notes)

from ui_helpers import FakeSource


def make(view="side", n_frames=600, field_order="bff", **kw) -> tuple[CalibState, FakeSource]:
    src = FakeSource(n_frames=n_frames, field_order=field_order)
    st = CalibState(src, view, (720, 480), n_frames, sar=8 / 9, **kw)
    return st, src


def keys(st, *names):
    for n in names:
        st.on_key(n)


def click(st, x, y):
    st.on_mouse("ldown", x, y)
    st.on_mouse("lup", x, y)


# ----------------------------------------------------------- basic flow --
def test_side_view_flow_and_save():
    st, src = make("side")
    assert st.target == "rubber" and "RUBBER" in st.target_text() and "required" in st.target_text()
    keys(st, "pgdn")                                   # 10 s later
    f1 = st.frame
    assert f1 == round(10 / src.sample_dt) // 2
    click(st, 500.0, 300.0)
    assert st.target == "plate" and st.seg.points["rubber"] == [500.0, 300.0]
    keys(st, "right", "right", "right")               # 1.5 frames later: different frame
    click(st, 100.0, 350.0)
    assert st.target == "release"                      # first optional point next
    keys(st, "enter")
    assert st.confirm and not st.done                  # warnings (extension assumed, frames differ)
    assert any("extension" in w for w in st.confirm)
    keys(st, "enter")
    assert st.done and not st.cancelled
    cf = st.result
    seg = cf.segments[0]
    assert seg.ref_frame == f1                         # side view: frame of the rubber click
    assert seg.points == {"rubber": [500.0, 300.0], "plate": [100.0, 350.0]}
    assert seg.view == "side" and seg.start_frame == 0 and seg.end_frame is None
    assert cf.image_size == (720, 480) and cf.settings.sar == pytest.approx(8 / 9)
    cf.validate()
    frames = point_frames_from_notes(cf)
    assert frames == [{"plate": st.tl.locate(st.s)[0], "rubber": f1}]


def test_behind_view_reference_is_release_frame():
    st, _ = make("behind")
    assert st.target == "release" and "Step" in st.hint_text()
    st.goto_frame(200)
    click(st, 360.0, 150.0)
    assert st.target == "catch"
    st.goto_frame(213)
    click(st, 370.0, 300.0)
    cf = st.build()
    assert cf.segments[0].ref_frame == 200
    assert st.target == "rubber_left"


def test_enter_with_missing_required_point_is_an_error():
    st, _ = make("side")
    click(st, 500.0, 300.0)
    keys(st, "enter")
    assert not st.done and st.confirm is None
    assert st.message.level == "error" and "plate" in st.message.text


def test_click_outside_picture_ignored():
    st, _ = make("side")
    st.place(-5.0, 10.0)
    assert not st.seg.points and st.message.level == "warn"


# ------------------------------------------------------- frame warnings --
def _two_frame_side(shift):
    st, _ = make("side", measure_shift=lambda a, b: shift)
    st.goto_frame(100)
    click(st, 500.0, 300.0)
    st.goto_frame(140)
    click(st, 100.0, 350.0)
    return st


def test_no_warning_when_camera_is_still():
    st = _two_frame_side((0.2, -0.1))
    assert st.frame_warnings() == []


def test_warning_when_camera_moved():
    st = _two_frame_side((3.0, 0.5))
    w = st.frame_warnings()
    assert len(w) == 1 and "moved about" in w[0] and "'plate'" in w[0] and "frame 100" in w[0]


def test_warning_when_motion_unknown():
    st = _two_frame_side(None)
    w = st.frame_warnings()
    assert len(w) == 1 and "fine on a tripod" in w[0] and "plate@140" in w[0]


def test_measure_failure_is_unknown_not_crash():
    def boom(a, b):
        raise RuntimeError("decode error")
    st, _ = make("side", measure_shift=boom)
    click(st, 500.0, 300.0)
    st.goto_frame(50)
    click(st, 100.0, 350.0)
    assert "fine on a tripod" in st.frame_warnings()[0]


def test_same_frame_points_no_warning():
    st, _ = make("side")
    click(st, 500.0, 300.0)
    click(st, 100.0, 350.0)
    assert st.frame_warnings() == []


# ------------------------------------------------------- editing keys --
def test_tab_reclick_nudge_and_undo():
    st, _ = make("side")
    click(st, 500.0, 300.0)
    click(st, 100.0, 350.0)
    assert st.selected == "plate"
    keys(st, "i", "i", "l", "k", "j", "j")
    p = st.seg.points["plate"]
    assert p == pytest.approx([100.0 - NUDGE_PX, 350.0 - NUDGE_PX])
    keys(st, "u")                                     # undo last nudge (j)
    assert st.seg.points["plate"] == pytest.approx([100.0, 350.0 - NUDGE_PX])
    keys(st, "backspace", "backspace", "backspace", "backspace", "backspace")
    assert st.seg.points["plate"] == [100.0, 350.0]
    keys(st, "u")                                     # undo the plate click
    assert "plate" not in st.seg.points and st.target == "plate"
    keys(st, "u")
    assert st.seg.points == {} and st.target == "rubber"
    keys(st, "u")
    assert "nothing to undo" in st.message.text


def test_tab_selects_point_to_reclick():
    st, _ = make("side")
    click(st, 500.0, 300.0)
    click(st, 100.0, 350.0)
    assert st.target == "release"
    keys(st, "tab")
    assert st.target == "catch" and st.selected is None      # not placed: nothing to nudge
    keys(st, "backtab", "backtab")
    assert st.target == "plate" and st.selected == "plate"
    keys(st, "backtab")
    assert st.target == "rubber" and st.selected == "rubber"
    click(st, 510.0, 305.0)                           # re-place rubber
    assert st.seg.points["rubber"] == [510.0, 305.0]
    assert st.target == "release"                     # next open point
    keys(st, "backtab", "backtab", "backtab")
    assert st.target == "vertical_bottom"


def test_skip_optional_and_refuse_required():
    st, _ = make("side")
    keys(st, "n")
    assert st.message.level == "error" and "required" in st.message.text and st.target == "rubber"
    click(st, 500.0, 300.0)
    click(st, 100.0, 350.0)
    keys(st, "n", "n")                                # skip release, catch
    assert st.seg.skipped == {"release", "catch"} and st.target == "vertical_top"
    keys(st, "n", "n")
    assert st.target is None and "All points placed" in st.target_text()
    st.place(1.0, 1.0)                                # nothing to place
    assert len(st.seg.points) == 2
    keys(st, "u")
    assert st.target == "vertical_bottom"


def test_nudge_without_selection_warns():
    st, _ = make("side")
    keys(st, "i")
    assert st.message.level == "warn"


def test_right_click_deletes_nearest_within_radius():
    st, _ = make("side")
    st.pick_radius = 10.0
    click(st, 500.0, 300.0)
    click(st, 100.0, 350.0)
    st.on_mouse("rdown", 300.0, 300.0)                # far from both
    assert len(st.seg.points) == 2
    st.on_mouse("rdown", 503.0, 304.0)
    assert "rubber" not in st.seg.points and st.target == "rubber"
    keys(st, "u")
    assert "rubber" in st.seg.points


def test_delete_key_deletes_selected():
    st, _ = make("side")
    click(st, 500.0, 300.0)
    keys(st, "del")
    assert not st.seg.points and st.target == "rubber"
    keys(st, "x")
    assert st.message.level == "warn"


# ---------------------------------------------------------------- masks --
def test_mask_drag_and_delete():
    st, _ = make("side")
    keys(st, "m")
    assert st.mask_mode
    st.on_mouse("ldown", 600.0, 420.0)
    st.on_mouse("move", 650.0, 440.0)
    assert st.drag == (600.0, 420.0, 650.0, 440.0)
    st.on_mouse("lup", 700.0, 470.0)
    assert st.seg.masks == [[600.0, 420.0, 700.0, 470.0]] and not st.mask_mode
    assert st.seg.points == {}                         # the drag did not place a point
    st.on_mouse("rdown", 650.0, 450.0)
    assert st.seg.masks == []
    keys(st, "u")
    assert len(st.seg.masks) == 1
    keys(st, "m")
    st.on_mouse("ldown", 10.0, 10.0)
    st.on_mouse("lup", 11.0, 11.0)                     # too small
    assert len(st.seg.masks) == 1 and st.mask_mode
    keys(st, "esc")
    assert not st.mask_mode and not st.cancel_armed


def test_masks_in_result():
    st, _ = make("side")
    keys(st, "m")
    st.on_mouse("ldown", 600.0, 420.0)
    st.on_mouse("lup", 715.0, 479.0)
    click(st, 500.0, 300.0)
    click(st, 100.0, 350.0)
    assert st.build().segments[0].masks == [[600.0, 420.0, 715.0, 479.0]]


# ---------------------------------------------------------- cancel / help --
def test_esc_twice_cancels():
    st, _ = make("side")
    keys(st, "esc")
    assert st.cancel_armed and not st.done and "AGAIN" in st.message.text
    keys(st, "right")                                  # any other key disarms
    assert not st.cancel_armed
    keys(st, "esc", "esc")
    assert st.done and st.cancelled and st.result is None


def test_unbound_key_disarms_cancel():
    st, _ = make("side")
    keys(st, "esc")
    assert st.on_key("q") is False
    assert not st.cancel_armed


def test_click_aborts_confirm_and_help_closes_with_esc():
    st, _ = make("side")
    click(st, 500.0, 300.0)
    click(st, 100.0, 350.0)
    keys(st, "enter")
    assert st.confirm
    keys(st, "right")
    assert st.confirm is None and not st.done
    keys(st, "h")
    assert st.help
    keys(st, "esc")
    assert not st.help and not st.cancel_armed


# ------------------------------------------------------------ navigation --
def test_navigation_steps():
    st, src = make("side", n_frames=2000)
    per_s = round(1 / src.sample_dt)
    per_10 = round(10 / src.sample_dt)
    assert (per_s, per_10) == (60, 599)
    keys(st, "right")
    assert st.s == 1
    keys(st, "d", "d")
    assert st.s == 3
    keys(st, "left", "a")
    assert st.s == 1
    keys(st, "down")
    assert st.s == 1 + per_s
    keys(st, "w")
    assert st.s == 1
    keys(st, "pgdn", "]")
    assert st.s == 1 + 2 * per_10
    keys(st, "[")
    assert st.s == 1 + per_10
    keys(st, "end")
    assert st.s == src.n_samples - 1
    keys(st, "right")
    assert st.s == src.n_samples - 1                  # clamped
    keys(st, "home")
    assert st.s == 0
    keys(st, "a")
    assert st.s == 0


def test_frame_view_steps_whole_frames():
    st, _ = make("side")
    assert st.view_mode == "field"
    keys(st, "f")
    assert st.view_mode == "frame"
    keys(st, "right")
    assert st.s == 2 and st.frame == 1
    keys(st, "f", "right")
    assert st.s == 3


def test_progressive_source():
    st, src = make("side", field_order=None)
    assert st.view_mode == "frame"
    keys(st, "f")
    assert st.view_mode == "frame" and "progressive" in st.message.text
    keys(st, "right")
    assert st.s == 1 and st.frame == 1


def test_trackbar_goes_to_frame():
    st, _ = make("side")
    st.on_trackbar(321)
    assert st.frame == 321 and st.field_k == 0
    st.on_trackbar(10 ** 6)
    assert st.frame == 599


def test_loupe_toggle():
    st, _ = make("side")
    keys(st, "z")
    assert st.loupe
    keys(st, "z")
    assert not st.loupe


# -------------------------------------------------------------- segments --
def test_new_segment_and_removal():
    st, _ = make("side", n_frames=1000)
    click(st, 500.0, 300.0)
    click(st, 100.0, 350.0)
    st.goto_frame(400)
    keys(st, "g")
    assert len(st.segments) == 2 and st.seg_index == 1 and st.seg.start_frame == 400
    assert st.target == "rubber" and st.seg.points == {}
    click(st, 520.0, 310.0)
    click(st, 120.0, 360.0)
    st.goto_frame(100)
    assert st.seg_index == 0 and st.seg.points["rubber"] == [500.0, 300.0]
    cf = st.build()
    assert [(g.start_frame, g.end_frame) for g in cf.segments] == [(0, 400), (400, None)]
    assert cf.segments[1].ref_frame == 400
    assert cf.for_frame(500).point("rubber").tolist() == [520.0, 310.0]
    cf.validate()
    st.goto_frame(400)
    keys(st, "g")                                     # remove the boundary again
    assert len(st.segments) == 1 and "dropped" in st.message.text
    keys(st, "u")
    assert len(st.segments) == 2


def test_segment_split_moves_later_points():
    st, _ = make("behind", n_frames=1000)
    st.goto_frame(100)
    click(st, 360.0, 150.0)
    st.goto_frame(500)
    click(st, 370.0, 300.0)                           # catch clicked at frame 500
    st.goto_frame(300)
    keys(st, "g")
    assert "catch" not in st.segments[0].points and "catch" in st.segments[1].points


def test_segment_at_start_refused():
    st, _ = make("side")
    keys(st, "g")
    assert len(st.segments) == 1 and st.message.level == "warn"


def test_incomplete_second_segment_blocks_save():
    st, _ = make("side", n_frames=1000)
    click(st, 500.0, 300.0)
    click(st, 100.0, 350.0)
    st.goto_frame(400)
    keys(st, "g", "enter")
    assert not st.done and st.message.level == "error"


# ----------------------------------------------------------- initial file --
def _existing() -> CalibrationFile:
    seg = Segment(start_frame=0, ref_frame=120, view="side",
                  points={"rubber": [500.0, 300.0], "plate": [100.0, 350.0], "release": [480.0, 200.0]},
                  masks=[[600.0, 420.0, 715.0, 479.0]])
    return CalibrationFile(settings=Settings(distance_ft=46.0, sar=8 / 9), segments=[seg],
                           video={"name": "x.avi"}, image_size=(720, 480),
                           notes='my note\npoint_frames: [{"rubber": 120, "plate": 120, "release": 133}]')


def test_edit_existing_calibration():
    st, _ = make("side", initial=_existing())
    assert st.frame == 120                             # starts on the reference frame
    assert st.seg.points["release"] == [480.0, 200.0]
    assert st.seg.point_frames == {"rubber": 120, "plate": 120, "release": 133}
    assert st.settings.distance_ft == 46.0
    assert st.target == "catch"
    keys(st, "enter", "enter")
    cf = st.result
    assert cf.video == {"name": "x.avi"}
    assert cf.notes.startswith("my note") and cf.notes.count("point_frames:") == 1
    again = CalibrationFile.from_dict(json.loads(json.dumps(cf.to_dict())))
    assert point_frames_from_notes(again) == [{"plate": 120, "release": 133, "rubber": 120}]
    assert again.segments[0].masks == [[600.0, 420.0, 715.0, 479.0]]


def test_settings_argument_wins_and_start_frame():
    st, _ = make("side", initial=_existing(), settings=Settings(distance_ft=60.5), start_frame=50)
    assert st.settings.distance_ft == 60.5 and st.frame == 50
    assert st.settings.sar == pytest.approx(8 / 9)


def test_view_change_drops_unusable_points():
    cf = _existing()
    cf.segments[0].points["vertical_top"] = [10.0, 10.0]
    st, _ = make("behind", initial=cf)
    assert "vertical_top" not in st.seg.points and "release" in st.seg.points
    assert st.message is not None and "dropped" in st.message.text


def test_image_size_mismatch_warned():
    cf = _existing()
    cf.image_size = (640, 480)
    st, _ = make("side", initial=cf)
    assert "check every point" in st.message.text


def test_bad_view_rejected():
    with pytest.raises(ValueError):
        make("top")


# -------------------------------------------------------- shift estimator --
def test_estimate_shift_recovers_translation():
    rng = np.random.default_rng(1)
    base = rng.normal(128, 40, (480, 720)).astype(np.float32)
    import cv2
    base = cv2.GaussianBlur(base, (0, 0), 3)
    M = np.float32([[1, 0, 2.5], [0, 1, -1.0]])
    moved = cv2.warpAffine(base, M, (720, 480), borderMode=cv2.BORDER_REFLECT)
    dx, dy = estimate_shift(base, moved)
    assert dx == pytest.approx(2.5, abs=0.2) and dy == pytest.approx(-1.0, abs=0.2)
    assert estimate_shift(base, base[:-2]) is None


def test_display_helpers():
    st, _ = make("side")
    click(st, 500.0, 300.0)
    st.goto_frame(30)
    items = dict(st.status_items())
    assert "rubber OK*" in items                       # clicked on another frame
    assert any(k == "other_frame" for _, _, k in st.markers()) is False   # rubber is selected
    keys(st, "tab")
    kinds = {n: k for n, _, k in st.markers()}
    assert kinds["rubber"] == "other_frame"
    assert len(st.help_rows()) > 20
