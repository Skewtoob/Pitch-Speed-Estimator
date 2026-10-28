"""End-to-end CLI on a small synthetic clip: re-running never loses review work,
settings flags are applied, review/render reuse the saved scan, locked files and
calibrations of other files are handled."""
from __future__ import annotations

import glob
import json
import os
import shutil

import pytest

pytest.importorskip("av")

from pitchspeed import cli, pipeline as P                    # noqa: E402
from pitchspeed.estimate.speed import manual_event            # noqa: E402
from pitchspeed.render.report import load_json               # noqa: E402
from pitchspeed.synth.generate import generate_clip           # noqa: E402
from pitchspeed.synth.gt import calib_path_for                # noqa: E402
from pitchspeed.types import Pitch, SpeedEstimate             # noqa: E402
from synth_helpers import small                               # noqa: E402


def _inputs(monkeypatch, answers):
    it = iter(answers)

    def fake(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError
    monkeypatch.setattr("builtins.input", fake)


@pytest.fixture(scope="module")
def analyzed(tmp_path_factory):
    """A clip + its output folder (calibration.json, results.json, transforms.npz), analyzed once."""
    d = tmp_path_factory.mktemp("cli_clip")
    video = str(d / "game 1.mpg")
    generate_clip(small(), video)
    out = P.default_out_dir(video)
    os.makedirs(out)
    shutil.copy(calib_path_for(video), P.out_paths(out)["calibration"])
    assert cli.main(["analyze", video, "--render", "none"]) == 0
    assert load_json(P.out_paths(out)["results_json"]).pitches, "the clip should give at least one pitch"
    return video


@pytest.fixture
def clip(analyzed, tmp_path):
    """A private copy of the analyzed clip and its output folder."""
    video = str(tmp_path / os.path.basename(analyzed))
    shutil.copy(analyzed, video)
    shutil.copytree(P.default_out_dir(analyzed), P.default_out_dir(video))
    return video, P.default_out_dir(video)


def _review_work(video, out):
    """What a user does in review: reject the first pitch (with a note), add a manual pitch."""
    from pitchspeed.calib.model import CalibrationFile
    r = load_json(P.out_paths(out)["results_json"])
    reader, src = P.source_from_result(video, r)
    cf = CalibrationFile.load(P.out_paths(out)["calibration"])
    p = r.pitches[0]
    p.status, p.notes = "rejected", "throw to first"
    p.edit_history.append({"action": "reject", "at": "2026-10-28T10:00:00", "status": "auto"})
    sr, sc = src.sample_at_time(5.0), src.sample_at_time(5.45)
    m = Pitch(number=0, status="manual", release=manual_event(src.sample_time(sr), src.sample_dt, "release", s=sr),
              catch=manual_event(src.sample_time(sc), src.sample_dt, "catch", s=sc),
              speed=SpeedEstimate(v_release_mph=None),
              edit_history=[{"action": "create_manual", "at": "2026-10-28T10:01:00"}])
    P.make_recompute(cf, r.scan, src, video=r.video)(m)
    r.pitches.append(m)
    P.save_results(r, out)
    reader.close()
    return r


def _state(out):
    r = load_json(P.out_paths(out)["results_json"])
    return sorted((round(p.release.t, 2), p.status, p.notes) for p in r.pitches), r


def test_run_again_keeps_the_review_by_default(clip, monkeypatch, capsys):
    video, out = clip
    _review_work(video, out)
    before, _ = _state(out)

    def no_analysis(*a, **kw):
        raise AssertionError("must not re-analyze")
    monkeypatch.setattr(P, "analyze", no_analysis)
    _inputs(monkeypatch, ["", "n", "none"])          # keep (default), no review window, no video
    assert cli.main(["run", video]) == 0
    text = capsys.readouterr().out
    assert "Saved results found" in text and "1 manual" in text and "1 rejected" in text
    assert _state(out)[0] == before
    assert not glob.glob(os.path.join(out, "results.backup-*"))


def test_analyze_again_backs_up_and_carries_the_review_over(clip, capsys):
    video, out = clip
    _review_work(video, out)
    before, _ = _state(out)
    assert cli.main(["analyze", video, "--render", "none"]) == 0
    backups = glob.glob(os.path.join(out, "results.backup-*.json"))
    assert len(backups) == 1 and glob.glob(os.path.join(out, "results.backup-*.transforms.npz"))
    assert sorted((round(p.release.t, 2), p.status, p.notes) for p in load_json(backups[0]).pitches) == before
    after, r = _state(out)
    assert [s for _, s, _ in after].count("rejected") == 1 and [s for _, s, _ in after].count("manual") == 1
    assert ("throw to first" in [n for _, _, n in after])
    assert any(h.get("action") == "carried_over" for p in r.pitches for h in p.edit_history)
    assert "backed up as results.backup-" in r.warnings[0] and "carried over" in r.warnings[0]
    assert "backed up as" in capsys.readouterr().out


def test_run_settings_flags_update_calibration_and_speeds(clip, monkeypatch, capsys):
    video, out = clip
    _review_work(video, out)
    _inputs(monkeypatch, ["", "n", "none"])
    assert cli.main(["run", video, "--level", "ll", "--extension", "3.5"]) == 0
    text = capsys.readouterr().out
    assert "Updated the saved settings" in text and "60.5 -> 46" in text
    st = json.load(open(P.out_paths(out)["calibration"], encoding="utf-8"))["settings"]
    assert st["distance_ft"] == 46.0 and st["extension_ft"] == 3.5
    after, r = _state(out)
    assert [s for _, s, _ in after].count("manual") == 1                     # edits kept
    for p in r.pitches:
        assert p.speed.assumptions.get("distance_ft") == 46.0, p
    assert r.calibration["settings"]["distance_ft"] == 46.0


def _set_scan_progressive(out):
    path = P.out_paths(out)["results_json"]
    d = json.load(open(path, encoding="utf-8"))
    d["scan"].update(mode="progressive", field_order=None, source="override", evidence={})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f)


def test_review_and_render_use_the_scan_saved_with_the_results(clip, monkeypatch):
    from pitchspeed.render import overlay
    from pitchspeed.review import ui
    video, out = clip
    _set_scan_progressive(out)              # e.g. analyzed with --field-order progressive
    seen = {}

    def fake_review(result, source, calib_file, recompute, save, **kw):
        seen["review"] = source
        return result
    monkeypatch.setattr(ui, "run_review_ui", fake_review)
    assert cli.main(["review", video]) == 0
    src = seen["review"]
    r = load_json(P.out_paths(out)["results_json"])
    assert not src.interlaced and src.n_samples == r.video.n_frames             # frames, not re-detected fields

    def fake_render(result, source, out_path, **kw):
        seen["render"] = source
        return out_path
    monkeypatch.setattr(overlay, "render_annotated", fake_render)
    assert cli.main(["render", video, "--no-audio"]) == 0
    assert not seen["render"].interlaced and seen["render"].n_samples == r.video.n_frames
    with pytest.raises(SystemExit, match="analyze again"):
        cli.main(["review", video, "--field-order", "bff"])


def test_review_autosave_survives_a_locked_csv(clip, monkeypatch, capsys):
    from pitchspeed.render import report
    from pitchspeed.review import ui
    video, out = clip
    real = os.replace

    def locked(src, dst):
        if str(dst).endswith("results.csv"):
            raise PermissionError(13, "in use", dst)
        return real(src, dst)
    monkeypatch.setattr(report.os, "replace", locked)
    monkeypatch.setattr(report, "_REPLACE_RETRY_S", 0.01)

    def fake_review(result, source, calib_file, recompute, save, **kw):
        result.pitches[0].notes = "edited in review"
        save(result)                         # the per-edit autosave must not raise
        return result
    monkeypatch.setattr(ui, "run_review_ui", fake_review)
    _inputs(monkeypatch, ["s"])
    assert cli.main(["review", video]) == 0
    assert "results.csv is open in another program" in capsys.readouterr().out
    assert load_json(P.out_paths(out)["results_json"]).pitches[0].notes == "edited in review"


def test_calibration_of_another_size_is_refused(clip):
    video, out = clip
    path = P.out_paths(out)["calibration"]
    d = json.load(open(path, encoding="utf-8"))
    d["video"]["width"], d["image_size"] = 720, [720, 480]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f)
    with pytest.raises(SystemExit, match="cannot be used"):
        cli.main(["analyze", video, "--render", "none"])


def test_run_asks_before_using_another_files_calibration(clip, monkeypatch, capsys):
    video, out = clip
    other = os.path.join(os.path.dirname(video), "export.mpg")
    shutil.copy(video, other)
    os.makedirs(P.default_out_dir(other))
    shutil.copy(P.out_paths(out)["calibration"], P.out_paths(P.default_out_dir(other))["calibration"])
    _inputs(monkeypatch, ["n", "n"])             # do not use it, do not recalibrate now
    assert cli.main(["run", other]) == 1
    text = capsys.readouterr().out
    assert "may not belong to this video" in text and "game 1.mpg" in text and "Stopped" in text
    assert not os.path.exists(P.out_paths(P.default_out_dir(other))["results_json"])
