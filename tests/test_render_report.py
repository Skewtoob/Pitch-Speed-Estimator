"""Tests for pitchspeed.render.report and render.text (JSON, CSV, console table)."""
from __future__ import annotations

import csv
import dataclasses
import json
import math
import os

import numpy as np
import pytest

import render_helpers as H
from pitchspeed import __version__
from pitchspeed.render import report as R
from pitchspeed.render import text as T
from pitchspeed.types import (AnalysisResult, EventTime, Pitch, PitchTrack, ScanInfo, SpeedComponent,
                              SpeedEstimate, TrackPoint, VideoInfo)


# ------------------------------------------------------------------ text
@pytest.mark.parametrize("t, expected", [
    (0.0, "00:00:00.000"), (83.4567, "00:01:23.457"), (59.9996, "00:01:00.000"),
    (2 * 3600 + 1.5, "02:00:01.500"), (0.0005, "00:00:00.001"), (-1.25, "-00:00:01.250"),
    (None, T.NO_TIME), (math.nan, T.NO_TIME), (math.inf, T.NO_TIME), (np.float64(10.0), "00:00:10.000"),
])
def test_format_time(t, expected):
    assert R.format_time(t) == expected


@pytest.mark.parametrize("sigma, expected", [(1.6, "1.5"), (2.3, "2.5"), (2.0, "2"), (0.1, "0.5"),
                                             (2.74, "2.5"), (2.76, "3"), (None, "-"), (math.nan, "-")])
def test_sigma_text(sigma, expected):
    assert T.sigma_text(sigma) == expected


def test_speed_short_and_rounding():
    assert T.round_half_up(78.5) == 79 and T.round_half_up(77.49) == 77
    assert T.speed_short(H.speed(78.4, 1.6, "good")) == "78"
    assert T.speed_short(H.speed(65.5, 5.4, "poor")) == "~66?"
    assert T.speed_short(H.speed(None, None, "none")) == "n/a"
    assert T.speed_short(None) == "n/a"
    # a number with label 'none' is never shown
    sp = H.speed(70.0, 9.0, "fair")
    sp.label = "none"
    assert T.speed_short(sp) == "n/a" and T.ci90_text(sp) == "-"


def test_ascii_text():
    assert T.ascii_text("78 ± 3 – é") == "78 +/- 3 - ?"
    assert T.ascii_text("plain") == "plain"


# ------------------------------------------------------------------ JSON
def _walk_fields(obj, path="r"):
    """Yield (path, value) for every dataclass field in the object tree."""
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            yield f"{path}.{f.name}", getattr(obj, f.name)
            yield from _walk_fields(getattr(obj, f.name), f"{path}.{f.name}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk_fields(v, f"{path}[{i}]")


def test_rich_result_populates_every_field():
    """Guard for the round-trip test: the fixture leaves no dataclass field at its default."""
    r = H.rich_result()
    seen_types = set()
    for obj in [r, r.video, r.scan, r.pitches[5], r.pitches[5].speed, r.pitches[5].speed.components[0],
                r.rejected_tracks[0], r.pitches[5].track.points[0], r.pitches[5].release]:
        seen_types.add(type(obj))
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name)
            default = f.default if f.default is not dataclasses.MISSING else (
                f.default_factory() if f.default_factory is not dataclasses.MISSING else object())
            if f.name in ("schema",):
                continue
            assert v is not None and (not isinstance(v, (list, dict, str)) or v), f"{type(obj).__name__}.{f.name} empty"
            if isinstance(default, (int, float, str, bool)) and not isinstance(v, (list, dict)):
                assert v != default or f.name in ("sar_source", "end_kind", "status", "inlier",
                                                  "timing_mode", "source", "manual"), \
                    f"{type(obj).__name__}.{f.name} left at default"
    assert {AnalysisResult, VideoInfo, ScanInfo, Pitch, SpeedEstimate, SpeedComponent, PitchTrack,
            TrackPoint, EventTime} <= seen_types


def test_json_round_trip_every_field(tmp_path):
    r = H.rich_result()
    p = tmp_path / "results.json"
    R.write_json(r, str(p))
    r2 = R.load_json(str(p))
    # identical document (numpy -> lists, NaN -> null, tuples -> lists on both sides)
    assert R.result_to_dict(r2) == R.result_to_dict(r)
    # every field of every dataclass survived with the right type
    a = dict(_walk_fields(r))
    b = dict(_walk_fields(r2))
    assert set(a) == set(b)
    for k, v in a.items():
        if dataclasses.is_dataclass(v):
            assert type(b[k]) is type(v), k
    assert isinstance(r2.pitches[0].speed.ci90_mph, tuple)
    assert isinstance(r2.pitches[0].track.points[0], TrackPoint)
    assert isinstance(r2.pitches[0].speed.components[0], SpeedComponent)
    assert isinstance(r2.rejected_tracks[0], PitchTrack) and r2.rejected_tracks[0].reject_code == "TOO_SHORT"
    assert r2.pitches[4].track is None and r2.pitches[4].status == "manual"
    assert r2.pitches[0].track.diagnostics["nan_value"] is None           # NaN -> null
    assert r2.video.path == r.video.path                                  # non-ASCII kept
    assert r2.pitches[0].speed.v_release_mph == pytest.approx(78.4)
    assert r2.possible_missed == r.possible_missed
    # stable: writing the loaded result again gives the same bytes
    p2 = tmp_path / "again.json"
    R.write_json(r2, str(p2))
    assert p.read_bytes() == p2.read_bytes()


def test_json_is_strict_utf8_with_schema_and_version(tmp_path):
    r = H.rich_result()
    r.version = ""
    r.pitches[0].confidence = float("nan")
    r.params["inf"] = float("inf")
    r.params["np"] = {"a": np.float32(1.5), "b": np.arange(3), "c": {1, 2}}
    p = tmp_path / "r.json"
    R.write_json(r, str(p))
    raw = p.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    d = json.loads(raw.decode("utf-8"), parse_constant=lambda c: pytest.fail(f"non-strict token {c}"))
    assert d["schema"] == "pitchspeed.results/1"
    assert d["version"] == __version__
    assert d["pitches"][0]["confidence"] is None and d["params"]["inf"] is None
    assert d["params"]["np"] == {"a": 1.5, "b": [0, 1, 2], "c": [1, 2]}
    assert "é" in raw.decode("utf-8")                                 # ensure_ascii=False


def test_json_atomic_replace_and_failure_keeps_old_file(tmp_path, monkeypatch):
    r = H.rich_result()
    p = tmp_path / "results.json"
    R.write_json(r, str(p))
    before = p.read_bytes()
    r.pitches[0].notes = "changed"

    def boom(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr(R.os, "replace", boom)
    with pytest.raises(OSError):
        R.write_json(r, str(p))
    assert p.read_bytes() == before                                       # old file intact
    assert sorted(os.listdir(tmp_path)) == ["results.json"]                # temp file cleaned up


def test_locked_file_gives_clear_message(tmp_path, monkeypatch):
    def locked(src, dst):
        raise PermissionError(13, "Permission denied")
    monkeypatch.setattr(R.os, "replace", locked)
    monkeypatch.setattr(R, "_REPLACE_RETRY_S", 0.05)
    with pytest.raises(PermissionError, match="open in Excel"):
        R.write_csv(H.rich_result(), str(tmp_path / "results.csv"))
    assert os.listdir(tmp_path) == []


def test_non_ascii_folder(tmp_path):
    d = tmp_path / "Spiel été 野球"
    d.mkdir()
    r = H.rich_result()
    R.write_json(r, str(d / "results.json"))
    R.write_csv(r, str(d / "results.csv"))
    assert R.load_json(str(d / "results.json")).video.path == r.video.path


def test_load_json_errors(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        R.load_json(str(p))
    p.write_text(json.dumps({"hello": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a pitchspeed results file"):
        R.load_json(str(p))
    d = R.result_to_dict(H.rich_result())
    d["schema"] = "pitchspeed.results/2"
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ValueError, match="newer"):
        R.load_json(str(p))
    d["schema"] = "something.else/1"
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown schema"):
        R.load_json(str(p))
    # a BOM (e.g. after editing in Notepad) is tolerated; a missing schema means /1
    del d["schema"]
    p.write_bytes(b"\xef\xbb\xbf" + json.dumps(d).encode("utf-8"))
    assert len(R.load_json(str(p)).pitches) == 6


# ------------------------------------------------------------------- CSV
def _read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_csv_format_and_columns(tmp_path):
    p = tmp_path / "results.csv"
    R.write_csv(H.rich_result(), str(p))
    raw = p.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")                                 # BOM for Excel
    assert b"\r\n" in raw and raw.count(b"\r\n") == raw.count(b"\n")       # CRLF only
    rows = _read_csv(p)
    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        assert next(csv.reader(f)) == R.CSV_COLUMNS
    assert [r["status"] for r in rows] == ["auto", "auto", "auto", "rejected", "manual", "edited",
                                          "possible_missed"]
    assert [r["pitch"] for r in rows[:6]] == ["1", "2", "3", "", "4", "5"]


def test_csv_values(tmp_path):
    p = tmp_path / "results.csv"
    R.write_csv(H.rich_result(), str(p))
    rows = _read_csv(p)
    good, poor, none, rej, man, edi = rows[:6]
    fd = H.FIELD_DT
    assert good["time_release"] == R.format_time(600 * fd - 0.004)
    assert good["t_release_s"] == f"{600 * fd - 0.004:.3f}" and good["t_catch_s"] == f"{628 * fd + 0.002:.3f}"
    assert good["flight_s"] == f"{(628 * fd + 0.002) - (600 * fd - 0.004):.3f}"
    assert (good["frame_release"], good["frame_catch"], good["field_release"], good["field_catch"]) == \
        ("300", "314", "600", "628")
    assert good["view"] == "side" and edi["view"] == "behind"             # per calibration segment
    assert good["method"] == "tof" and man["method"] == "manual_tof" and edi["method"] == "tof+side_fit"
    assert (good["v_release_mph"], good["v_release_unc_mph"]) == ("78.4", "1.6")
    assert (good["ci90_lo"], good["ci90_hi"]) == ("75.8", "81.0")
    assert (good["v_avg_mph"], good["v_plate_mph"]) == ("74.4", "71.4")
    assert good["label"] == "good" and good["confidence"] == "0.83"
    assert good["flags"] == "catch_extrapolated"
    assert poor["flags"] == "catch_extrapolated;release_unconfirmed" and poor["label"] == "poor"
    assert poor["reason"] == "release time uncertain"
    # not measurable: no numbers at all, the reason instead
    for c in ("v_release_mph", "v_release_unc_mph", "ci90_lo", "ci90_hi", "v_avg_mph", "v_plate_mph"):
        assert none[c] == ""
    assert none["label"] == "none" and none["reason"].startswith("not measurable")
    # rejected pitch: kept, no number in the pitch column, values preserved
    assert rej["pitch"] == "" and rej["notes"] == "throw to first" and rej["v_release_mph"] == "40.0"
    assert man["notes"] == "marked by hand ± 1 field" and man["flags"] == "manual_events"


def test_csv_review_rows(tmp_path):
    p = tmp_path / "results.csv"
    r = H.rich_result()
    R.write_csv(r, str(p), rejected_tracks=True)
    rows = _read_csv(p)
    tr90, tr91, missed = rows[6:]                                          # chronological
    assert missed["status"] == "possible_missed" and "rejected_track:PARTIAL" in missed["notes"]
    assert (missed["field_release"], missed["field_catch"]) == ("5400", "5440")
    assert (missed["frame_release"], missed["frame_catch"]) == ("2700", "2720")
    assert missed["view"] == "behind" and missed["v_release_mph"] == "" and missed["pitch"] == ""
    assert "TOO_SHORT" in tr90["notes"] and "WRONG_DIRECTION" in tr91["notes"]
    assert tr90["field_release"] == "5000" and tr90["frame_release"] == "2500"
    R.write_csv(r, str(p), review_rows=False)
    assert len(_read_csv(p)) == 6
    R.write_csv(r, str(p))                                                 # default: no rejected tracks
    assert [x["status"] for x in _read_csv(p)[6:]] == ["possible_missed"]


def test_csv_progressive_frame_of_sample():
    r = H.rich_result()
    r.scan = ScanInfo(mode="progressive", field_order=None, evidence={"frame_step": 2, "frame_phase": 1})
    assert R.frame_of_sample(r, 10) == 21
    r.scan.evidence = {}
    assert R.frame_of_sample(r, 10) == 10 and R.frame_of_sample(r, None) is None


def test_csv_handles_missing_speed_and_events(tmp_path):
    r = H.rich_result()
    r.pitches[0].speed = None
    r.pitches[1].frame_release = -1
    p = tmp_path / "x.csv"
    R.write_csv(r, str(p))
    rows = _read_csv(p)
    assert rows[0]["label"] == "none" and rows[0]["method"] == "none" and rows[0]["reason"] == "no speed estimate"
    assert rows[1]["frame_release"] == ""


# ----------------------------------------------------------------- table
def test_table_contents():
    out = R.format_table(H.rich_result())
    lines = out.splitlines()
    assert "Spiel 2003" in lines[0] and "BFF" in lines[0] and "59.94 fields/s" in lines[0]
    assert "side + behind" in lines[1] and "60.5 ft" in lines[1] and "5 pitches + 1 rejected" in lines[1]
    assert "extension 5.5 ft +/-1.0 (assumed)" in lines[2] and "Cd 0.33" in lines[2]
    hdr = next(i for i, l in enumerate(lines) if l.startswith("#  time"))
    for col in ["frames", "speed", "+/-", "90% range", "avg", "method", "label", "flags"]:
        assert col in lines[hdr]
    rows = lines[hdr + 1: hdr + 7]
    assert rows[0].split()[:7] == ["1", R.format_time(600 * H.FIELD_DT - 0.004), "300-314", "78", "1.5",
                                   "76-81", "74"]
    assert "~65?" in rows[1] and "poor" in rows[1] and "release_unconfirmed" in rows[1]
    assert rows[2].split()[3] == "n/a" and " none " in rows[2]
    assert rows[3].startswith("-") and "rejected" in rows[3]
    assert "manual TOF" in rows[4] and "manual,manual_events" in rows[4] and " fair " in rows[4]
    assert "TOF+side fit" in rows[5] and "edited" in rows[5]
    # columns line up: every row's time column starts where the header's does
    col = lines[hdr].index("time")
    assert all(r[col:col + 2] == "00" for r in rows)
    sp_end = lines[hdr].index("speed") + len("speed")
    assert [r[sp_end - 1] for r in rows if "rejected" not in r] == ["8", "?", "a", "1", "0"]


def test_table_notes_review_and_footer():
    out = R.format_table(H.rich_result())
    assert "#3: not measurable: uncertainty +-14% (> 10%)" in out
    assert "#2: release time uncertain" in out
    assert "throw to first" in out
    assert "Possible missed pitches (1)" in out and "rejected_track:PARTIAL" in out
    assert "not accepted as pitches: 2 (TOO_SHORT 1, WRONG_DIRECTION 1); details in results.json" in out
    assert "Warnings:" in out and "dropped-frame" in out
    assert "RELEASE speed" in out and "radar gun" in out
    assert "~78?" in out and "n/a = not measurable" in out


def test_table_ascii_only_for_own_text():
    r = H.rich_result(path="plain.avi")
    r.pitches[4].notes = ""
    assert R.format_table(r).isascii()


def test_table_empty_and_progressive():
    r = H.rich_result()
    r.pitches, r.possible_missed, r.rejected_tracks, r.warnings = [], [], [], []
    r.scan = ScanInfo(mode="progressive", field_order=None, confidence="medium")
    out = R.format_table(r)
    assert "No pitches found." in out and "frame timing only" in out and "0 pitches" in out


def test_load_json_damaged_structure(tmp_path):
    d = R.result_to_dict(H.rich_result())
    del d["pitches"][0]["release"]
    p = tmp_path / "r.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ValueError, match="damaged or incomplete"):
        R.load_json(str(p))


def test_long_notes_are_wrapped_not_cut():
    r = H.rich_result()
    long = "field order uncertain: flight time may be off by +-2 fields; " * 5 + "END"
    r.pitches[0].speed.reason = long
    out = R.format_table(r)
    assert "END" in out and max(len(l) for l in out.splitlines() if l.startswith("  ")) <= 110


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_new_files_follow_umask(tmp_path):
    um = os.umask(0o022)
    try:
        R.write_json(H.rich_result(), str(tmp_path / "r.json"))
    finally:
        os.umask(um)
    assert (os.stat(tmp_path / "r.json").st_mode & 0o777) == 0o644
