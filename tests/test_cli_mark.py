"""End-to-end: mark the pitches of a clip by hand (`run` option 2 / `pitchspeed mark`), then
one clip per pitch.  The review window is replaced by a script that drives the real
mark-mode state machine (N at the release field, C at the catch field)."""
from __future__ import annotations

import json
import math
import os

import pytest

pytest.importorskip("av")

from pitchspeed import cli, pipeline as P                    # noqa: E402
from pitchspeed.calib.model import CalibrationFile           # noqa: E402
from pitchspeed.render.report import load_json               # noqa: E402
from pitchspeed.review import ui                              # noqa: E402
from pitchspeed.review.state import ReviewState               # noqa: E402
from pitchspeed.synth.generate import generate_clip           # noqa: E402
from pitchspeed.synth.gt import gt_path_for, load_gt          # noqa: E402
from synth_helpers import small                               # noqa: E402
from test_cli_rerun import _inputs                            # noqa: E402

FIELD_DT = 1001 / 60000


@pytest.fixture(scope="module")
def tape(tmp_path_factory):
    d = tmp_path_factory.mktemp("mark_clip")
    video = str(d / "tape 1.mpg")
    generate_clip(small(), video)
    return video, load_gt(gt_path_for(video))


def _marking_window(gt, seen):
    """Stand-in for run_review_ui: marks every ground-truth pitch like a careful user."""
    def run(result, source, calib_file, recompute, save, mode="review", **kw):
        seen["mode"] = mode
        st = ReviewState(result, source, recompute, save, mode=mode)
        for p in gt["pitches"]:
            st.goto(math.floor(p["t_release"] / FIELD_DT) + 1)       # first field with the ball out
            st.on_key("n")
            st.goto(math.ceil(p["t_catch"] / FIELD_DT))               # first field in the glove
            st.on_key("c")
        st.finish()
        return result
    return run


def test_guided_run_mark_by_hand_then_clips(tape, monkeypatch, capsys):
    video, gt = tape
    seen = {}
    monkeypatch.setattr(ui, "run_review_ui", _marking_window(gt, seen))
    _inputs(monkeypatch, ["2", "1", "clips"])          # by hand; level 1 (60.5 ft); clips
    assert cli.main(["run", video, "--no-audio"]) == 0
    out = P.default_out_dir(video)
    txt = capsys.readouterr().out
    assert "How should the pitches be found?" in txt and "Step 2/3: mark the pitches" in txt
    cf = CalibrationFile.load(P.out_paths(out)["calibration"])
    assert [s.view for s in cf.segments] == ["manual"] and cf.segments[0].points == {}
    assert seen["mode"] == "mark"
    r = load_json(P.out_paths(out)["results_json"])
    assert P.is_marking_session(r) and len(r.pitches) == len(gt["pitches"])
    for p, g in zip(sorted(r.pitches, key=lambda q: q.release.t), gt["pitches"]):
        sp = p.speed
        assert p.status == "manual" and sp.method == "manual_tof" and sp.v_release_mph is not None
        assert abs(sp.v_release_mph - g["speed_release_mph"]) < 4 * sp.sigma_mph      # honest +/-
    folder = os.path.join(out, "clips")
    names = sorted(os.listdir(folder))
    assert "clips.json" in names and "originals" in names
    mp4 = [n for n in names if n.endswith(".mp4")]
    assert len(mp4) == len(gt["pitches"]) and all(n.startswith(("P01_", "P02_")) for n in mp4)
    assert sorted(os.listdir(os.path.join(folder, "originals"))) == ["P01.mpg", "P02.mpg"]
    assert "Clips: 2" in txt and "marked by hand" in txt


def test_second_run_continues_the_session(tape, monkeypatch, capsys):
    video, gt = tape
    out = P.default_out_dir(video)
    if not os.path.exists(P.out_paths(out)["results_json"]):
        pytest.skip("needs the guided run above")
    before = load_json(P.out_paths(out)["results_json"])
    seen = {}

    def nothing(result, source, calib_file, recompute, save, mode="review", **kw):
        seen["mode"] = mode
        return result
    monkeypatch.setattr(ui, "run_review_ui", nothing)
    _inputs(monkeypatch, ["none"])                     # no method question; nothing rendered
    assert cli.main(["run", video]) == 0
    txt = capsys.readouterr().out
    assert "marked by hand" in txt and "Continuing with the saved results: 2 pitches" in txt
    assert "How should the pitches be found?" not in txt and seen["mode"] == "mark"
    after = load_json(P.out_paths(out)["results_json"])
    assert [p.release.s for p in after.pitches] == [p.release.s for p in before.pitches]
    # review and analyze know it is a hand-marked session
    assert cli.main(["review", video]) == 0 and seen["mode"] == "mark"
    with pytest.raises(SystemExit, match="marking the pitches by hand"):
        cli.main(["analyze", video, "--render", "none"])


def test_rerendering_clips_replaces_only_its_own_files(tape, monkeypatch):
    video, gt = tape
    out = P.default_out_dir(video)
    folder = os.path.join(out, "clips")
    if not os.path.isdir(folder):
        pytest.skip("needs the guided run above")
    mine = os.path.join(folder, "my own notes.txt")
    with open(mine, "w") as f:
        f.write("keep me")
    old = json.load(open(os.path.join(folder, "clips.json")))["files"]
    r = load_json(P.out_paths(out)["results_json"])
    r.pitches[0].speed.v_release_mph = 12.0                       # the first clip gets a new name
    P.save_results(r, out)
    assert cli.main(["render", video, "--mode", "clips", "--no-audio", "--no-originals"]) == 0
    now = json.load(open(os.path.join(folder, "clips.json")))["files"]
    assert any(n.startswith("P01_12mph") for n in now)
    old_originals = [n for n in old if n.startswith("originals/")]
    assert old_originals and all(n in now and os.path.exists(os.path.join(folder, n)) for n in old_originals)
    for n in old:                                                 # --no-originals keeps the originals
        if n not in now:
            assert not os.path.exists(os.path.join(folder, n))   # the stale clip is removed
    assert os.path.exists(mine)                                   # never touches other files


def test_mark_command_with_flags_needs_no_questions(tmp_path, monkeypatch, capsys):
    video = str(tmp_path / "short.mpg")
    generate_clip(small(duration_s=4.0, pitches={"count": 1}), video)
    gt = load_gt(gt_path_for(video))
    seen = {}
    monkeypatch.setattr(ui, "run_review_ui", _marking_window(gt, seen))
    _inputs(monkeypatch, [])                                      # no console answers at all
    assert cli.main(["mark", video, "--level", "hs", "--render", "none"]) == 0
    r = load_json(P.out_paths(P.default_out_dir(video))["results_json"])
    assert len(r.pitches) == 1 and r.pitches[0].speed.v_release_mph is not None
    assert not os.path.exists(os.path.join(P.default_out_dir(video), "clips"))


def _fresh(tmp_path, name="t.mpg", **cfg):
    video = str(tmp_path / name)
    generate_clip(small(duration_s=4.0, pitches={"count": 1}, **cfg), video)
    return video, load_gt(gt_path_for(video))


def test_calibrate_after_marking_asks_the_view_and_starts_without_points(tmp_path, monkeypatch, capsys):
    from pitchspeed.calib import ui as calib_ui
    video, gt = _fresh(tmp_path)
    monkeypatch.setattr(ui, "run_review_ui", _marking_window(gt, {}))
    _inputs(monkeypatch, [])
    assert cli.main(["mark", video, "--level", "hs", "--render", "none"]) == 0
    seen = {}

    def fake_calib(source, info, view, initial=None, settings=None, **kw):
        seen.update(view=view, initial=initial, settings=settings)
        return None                                               # cancelled: nothing saved
    monkeypatch.setattr(calib_ui, "run_calibration_ui", fake_calib)
    _inputs(monkeypatch, ["1"])                                   # side
    cli.main(["calibrate", video])
    assert seen["view"] == "side" and seen["initial"] is None and seen["settings"].distance_ft == 60.5
    assert "Where was the camera?" in capsys.readouterr().out


def test_a_changed_tape_file_does_not_stop_a_hand_marked_setup(tmp_path, monkeypatch, capsys):
    video, gt = _fresh(tmp_path)
    monkeypatch.setattr(ui, "run_review_ui", _marking_window(gt, {}))
    _inputs(monkeypatch, [])
    assert cli.main(["mark", video, "--level", "hs", "--render", "none"]) == 0
    generate_clip(small(duration_s=5.0, pitches={"count": 1}), video)      # re-captured: other length
    gt2 = load_gt(gt_path_for(video))
    monkeypatch.setattr(ui, "run_review_ui", _marking_window(gt2, {}))
    _inputs(monkeypatch, [])                                      # no "use anyway?" question
    assert cli.main(["mark", video, "--render", "none"]) == 0
    txt = capsys.readouterr().out
    assert "Use this calibration anyway" not in txt and "starting a new session" in txt
    cf = CalibrationFile.load(P.out_paths(P.default_out_dir(video))["calibration"])
    r = load_json(P.out_paths(P.default_out_dir(video))["results_json"])
    assert cf.video["n_frames"] == r.video.n_frames
    assert not any("does not fit this video" in w for w in r.warnings)


def test_field_order_advice_for_a_hand_marked_session_points_to_k(tmp_path, monkeypatch):
    video, gt = _fresh(tmp_path)
    monkeypatch.setattr(ui, "run_review_ui", _marking_window(gt, {}))
    _inputs(monkeypatch, [])
    assert cli.main(["mark", video, "--level", "hs", "--render", "none"]) == 0
    r = load_json(P.out_paths(P.default_out_dir(video))["results_json"])
    stored = r.scan.field_order
    other = "bff" if stored == "tff" else "tff"
    for w in r.warnings:
        if w.startswith("interlace/field-order detection is uncertain"):
            assert "press K" in w and "--field-order" not in w
    with pytest.raises(SystemExit, match="press K"):
        cli.main(["mark", video, "--field-order", other, "--render", "none"])
