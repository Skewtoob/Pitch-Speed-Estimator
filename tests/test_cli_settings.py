"""CLI: settings flags, sport for custom distances, locked files, render errors (no video needed)."""
from __future__ import annotations

import json
import os

import pytest

from pitchspeed import cli
from pitchspeed.calib.model import CalibrationFile, Segment, Settings
from pitchspeed.types import AnalysisResult, ScanInfo, VideoInfo


def _args(argv):
    return cli.build_parser().parse_args(argv)


def _inputs(monkeypatch, answers):
    it = iter(answers)

    def fake(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError
    monkeypatch.setattr("builtins.input", fake)


def _cf(view="side", sport="baseball", distance=60.5, level="hs"):
    seg = Segment(view=view, points={"rubber": [600.0, 300.0], "plate": [100.0, 320.0]})
    return CalibrationFile(settings=Settings(distance_ft=distance, level=level, sport=sport), segments=[seg])


# ------------------------------------------------------------------ sport
def test_custom_distance_keeps_softball(monkeypatch):
    _inputs(monkeypatch, [])
    a = _args(["calibrate", "v.avi", "--distance", "30"])
    assert cli._ask_level(a, base=Settings(sport="softball")) == (None, 30.0, "softball")
    a = _args(["calibrate", "v.avi", "--distance", "43", "--level", "SB43"])      # case-insensitive
    assert cli._ask_level(a) == ("sb43", 43.0, "softball")
    a = _args(["calibrate", "v.avi", "--distance", "50", "--sport", "softball"])
    assert cli._ask_level(a) == (None, 50.0, "softball")
    with pytest.raises(SystemExit, match="does not go with"):
        cli._ask_level(_args(["calibrate", "v.avi", "--level", "hs", "--sport", "softball"]))
    with pytest.raises(SystemExit, match="unknown level"):
        cli._ask_level(_args(["calibrate", "v.avi", "--level", "majors"]))


def test_interactive_other_distance_asks_the_sport(monkeypatch):
    n_other = str(len(cli.C.LEVELS) + 1)
    _inputs(monkeypatch, [n_other, "5", "30", "s"])          # 5 ft is refused, then 30 ft softball
    assert cli._ask_level(_args(["calibrate", "v.avi"])) == (None, 30.0, "softball")
    _inputs(monkeypatch, [n_other, "52", ""])                 # Enter = baseball
    assert cli._ask_level(_args(["calibrate", "v.avi"])) == (None, 52.0, "baseball")


def test_impossible_flag_values_are_refused():
    with pytest.raises(SystemExit, match="extension"):
        cli._settings_from_args(_args(["run", "v.avi", "--extension", "66"]), base=Settings())
    with pytest.raises(SystemExit, match="distance"):
        cli._settings_from_args(_args(["run", "v.avi", "--distance", "605"]), base=Settings())


# ------------------------------------------------- settings flags on run/analyze
def test_settings_flags_update_the_saved_calibration(tmp_path, capsys):
    path = str(tmp_path / "calibration.json")
    cf = _cf()
    cf.settings.sar = 0.9091
    cf.save(path)
    a = _args(["analyze", "v.avi", "--level", "ll", "--extension", "4.2", "--throws", "l"])
    changes = cli._apply_settings_flags(a, cf, path)
    assert any("60.5 -> 46" in c for c in changes) and any("extension" in c for c in changes)
    saved = CalibrationFile.load(path).settings
    assert (saved.distance_ft, saved.level, saved.extension_ft, saved.pitcher_throws) == (46.0, "ll", 4.2, "L")
    assert saved.sar == 0.9091                                   # never touched by the flags
    assert "Updated the saved settings" in capsys.readouterr().out
    assert cli._apply_settings_flags(_args(["analyze", "v.avi"]), cf, path) == []
    # only --sport: the sport changes, the distance stays
    cli._apply_settings_flags(_args(["run", "v.avi", "--sport", "softball"]), cf, path)
    assert CalibrationFile.load(path).settings.sport == "softball"
    assert CalibrationFile.load(path).settings.distance_ft == 46.0


def test_view_flag_cannot_silently_change_the_view(tmp_path):
    path = str(tmp_path / "calibration.json")
    cf = _cf(view="side")
    cf.save(path)
    with pytest.raises(SystemExit, match="calibrate"):
        cli._apply_settings_flags(_args(["run", "v.avi", "--view", "behind"]), cf, path)
    cli._apply_settings_flags(_args(["run", "v.avi", "--view", "side"]), cf, path)   # same view: fine


def test_analyze_accepts_settings_flags():
    a = _args(["analyze", "v.avi", "--distance", "46", "--sport", "baseball", "--temp", "50"])
    assert a.distance == 46.0 and a.temp == 50.0


# ---------------------------------------------------------- locked files
def _result(tmp_path):
    info = VideoInfo(path=str(tmp_path / "v.avi"), width=720, height=480, n_frames=10, frame_rate=29.97,
                     frame_rate_str="30000/1001")
    return AnalysisResult(video=info, scan=ScanInfo(mode="interlaced", field_order="bff"),
                          calibration={}, params={}, pitches=[])


def test_final_review_save_never_loses_the_edits(tmp_path, monkeypatch, capsys):
    from pitchspeed.render import report
    r = _result(tmp_path)
    r.warnings = ["user-visible"]
    real = os.replace

    def locked_json(src, dst):
        if os.path.basename(dst) == "results.json":
            raise PermissionError(13, "in use", dst)
        return real(src, dst)
    monkeypatch.setattr(report.os, "replace", locked_json)
    monkeypatch.setattr(report, "_REPLACE_RETRY_S", 0.01)
    cli._final_save(r, str(tmp_path))
    unsaved = [f for f in os.listdir(tmp_path) if f.startswith("results.unsaved-")]
    assert len(unsaved) == 1 and json.load(open(tmp_path / unsaved[0], encoding="utf-8"))["warnings"] == ["user-visible"]
    assert "Your review was saved as" in capsys.readouterr().out


def test_locked_csv_offers_a_retry(tmp_path, monkeypatch, capsys):
    from pitchspeed.render import report
    r = _result(tmp_path)
    real = os.replace
    state = {"locked": True}

    def locked_csv(src, dst):
        if state["locked"] and str(dst).endswith(".csv"):
            raise PermissionError(13, "in use", dst)
        return real(src, dst)
    monkeypatch.setattr(report.os, "replace", locked_csv)
    monkeypatch.setattr(report, "_REPLACE_RETRY_S", 0.01)

    def answer(prompt=""):           # the user closes Excel, then presses Enter
        state["locked"] = False
        return ""
    monkeypatch.setattr("builtins.input", answer)
    cli._final_save(r, str(tmp_path))
    out = capsys.readouterr().out
    assert "results.csv is open in another program" in out and "results.csv written." in out
    assert os.path.exists(tmp_path / "results.csv") and not cli._csv_failed(r)
    # no console (EOF): skipped with a message, nothing raised
    state["locked"] = True
    _inputs(monkeypatch, [])
    cli._final_save(r, str(tmp_path))
    assert "still shows older results" in capsys.readouterr().out


# ----------------------------------------------------------- render errors
def test_render_errors_are_reported_not_raised(tmp_path, monkeypatch, capsys):
    from pitchspeed.render import overlay
    r = _result(tmp_path)
    r.pitches = []

    def boom(*a, **kw):
        raise PermissionError(13, "The process cannot access the file", "annotated.mp4")
    monkeypatch.setattr(overlay, "render_annotated", boom)
    assert cli._render(r, object(), str(tmp_path), "full", False) is None
    out = capsys.readouterr().out
    assert "Could not write the annotated video" in out and "pitchspeed render" in out

    def note(result, source, out, info=None, **kw):
        info["note"] = "annotated.mp4 is open in another program; written as annotated-1.mp4"
        return out
    monkeypatch.setattr(overlay, "render_annotated", note)
    cli._render(r, object(), str(tmp_path), "full", False)
    assert "annotated-1.mp4" in capsys.readouterr().out


def test_main_reports_locked_files_without_traceback(monkeypatch, capsys):
    def locked(args):
        raise PermissionError("cannot write results.csv: the file is locked by another program")
    monkeypatch.setattr(cli, "cmd_probe", locked)
    assert cli.main(["probe", "v.avi"]) == 1
    assert "locked by another program" in capsys.readouterr().err


def test_render_style_option():
    ap = cli.build_parser()
    assert ap.parse_args(["render", "v.mp4"]).style == "radar"
    assert ap.parse_args(["render", "v.mp4", "--style", "classic"]).style == "classic"
    with pytest.raises(SystemExit):
        ap.parse_args(["render", "v.mp4", "--style", "fancy"])
