"""Suite sampling / generation and the `pitchspeed synth` command line."""
import csv
import json
import os
from collections import Counter

import pytest

from pitchspeed.synth.cli import main as synth_main
from pitchspeed.synth.config import resolve_config
from pitchspeed.synth.suite import generate_suite, suite_configs
from pitchspeed.types import AnalysisResult, EventTime, Pitch, ScanInfo, SpeedEstimate, VideoInfo, to_jsonable


def test_suite_configs_are_stratified_and_valid():
    cfgs = suite_configs(40, seed=3)
    assert len(cfgs) == 40 and len({n for n, _ in cfgs}) == 40
    views = Counter(c["camera"]["view"] for _, c in cfgs)
    levels = Counter(c["level"] for _, c in cfgs)
    codecs = Counter(c["video"]["codec"] for _, c in cfgs)
    assert views == {"side": 20, "behind": 20} and levels == {"ll": 20, "hs": 20}
    assert codecs == {"mpeg2": 14, "dv": 12, "h264": 10, "ffv1": 4}
    orders = Counter(c["video"]["field_order"] for _, c in cfgs)
    assert orders["tff"] >= 12 and orders["bff"] >= 12
    assert all(c["video"]["field_order"] == "bff" for _, c in cfgs if c["video"]["codec"] == "dv")
    sh = sorted(1 / c["render"]["shutter_s"] for _, c in cfgs)
    assert 60 <= sh[0] < 60 * (1000 / 60) ** (1 / 40) and sh[-1] > 1000 / (1000 / 60) ** (1 / 40)   # LHS ends
    for _, c in cfgs:
        r = resolve_config(c)
        lo, hi = r["pitches"]["speed_mph"]
        assert (lo, hi) == ((40.0, 70.0) if c["level"] == "ll" else (55.0, 92.0))
        assert r["fence"]["enabled"] == (c["camera"]["view"] == "behind")
    assert suite_configs(40, seed=3) == cfgs and suite_configs(40, seed=4) != cfgs
    var = suite_configs(30, seed=3, variants=True)
    assert any(c["video"].get("postprocess") or c["video"].get("scan") for _, c in var)


def test_generate_suite_parallel_and_resume(tmp_path):
    base = {"duration_s": 3.5, "pitches": {"count": 1, "first_t": 2.0},
            "video": {"width": 360, "height": 240, "codec": "mpeg2"}}
    msgs = []
    rows = generate_suite(str(tmp_path), n=2, seed=9, jobs=2, base=base, progress=msgs.append)
    assert [r["status"] for r in rows] == ["ok", "ok"], rows
    with open(tmp_path / "index.csv", newline="", encoding="utf-8") as f:
        idx = list(csv.DictReader(f))
    assert [r["name"] for r in idx] == [r["name"] for r in rows]
    for r in idx:
        for k in ("file", "gt", "calib", "calib_exact"):
            assert os.path.exists(tmp_path / r[k]), (k, r[k])
        assert int(r["n_pitches"]) == 1 and float(r["speeds_mph"]) > 39
    assert {r["view"] for r in idx} == {"side", "behind"}
    assert len(msgs) == 2
    # resume: nothing is regenerated
    before = {r["file"]: os.path.getmtime(tmp_path / r["file"]) for r in idx}
    rows2 = generate_suite(str(tmp_path), n=2, seed=9, jobs=1, base=base)
    assert [r["status"] for r in rows2] == ["ok", "ok"]
    assert {r["file"]: os.path.getmtime(tmp_path / r["file"]) for r in rows2} == before


def test_suite_records_failures(tmp_path):
    # DV needs 720x480: forcing a small raster makes the DV clips fail -> status=error rows, others fine
    base = {"duration_s": 3.0, "pitches": {"count": 0, "first_t": 2.0}, "video": {"width": 360, "height": 240}}
    names = [n for n, c in suite_configs(4, seed=2)]
    rows = generate_suite(str(tmp_path), n=4, seed=2, jobs=1, base=base)
    assert [r["name"] for r in rows] == names
    st = {r["name"]: r["status"] for r in rows}
    assert any(s == "error" for s in st.values()) and any(s == "ok" for s in st.values())
    for r in rows:
        assert (r["status"] == "error") == ("_dv" in r["name"])
        if r["status"] == "error":
            assert "720x480" in r["error"]


def test_cli_generate_check_evaluate(tmp_path, capsys):
    out = str(tmp_path / "cli clip.mpg")
    rc = synth_main(["generate", "--out", out, "--seed", "4", "--quiet", "--set", "duration_s=4",
                     "--set", "video.width=360", "--set", "video.height=240", "--set", "video.debug_stamp=true",
                     "--set", "pitches.count=1", "--set", "pitches.first_t=2.0", "--set", "camera.view=behind"])
    assert rc == 0
    assert "pitch 1" in capsys.readouterr().out
    assert synth_main(["check", out]) == 0
    assert "RESULT: OK" in capsys.readouterr().out
    gt = json.load(open(out + ".gt.json", encoding="utf-8"))
    assert gt["config"]["camera"]["view"] == "behind" and gt["seed"] == 4
    p = gt["pitches"][0]
    res = AnalysisResult(VideoInfo(out, 360, 240, 119, 29.97, "30000/1001"), ScanInfo("interlaced", "bff"), {}, {},
                         [Pitch(1, "auto", EventTime(p["t_release"] + 0.01, 0.005, "first_free_field"),
                                EventTime(p["t_catch"], 0.005, "extrapolated"),
                                SpeedEstimate(p["speed_release_mph"] + 1.0, 1.0, (p["speed_release_mph"] - 0.6,
                                                                                  p["speed_release_mph"] + 2.6)))])
    rj = tmp_path / "results.json"
    rj.write_text(json.dumps(to_jsonable(res)), encoding="utf-8")
    assert synth_main(["evaluate", str(rj), out + ".gt.json"]) == 0
    txt = capsys.readouterr().out
    assert "recall 100%" in txt and "+1.0" in txt
    assert synth_main(["defaults"]) == 0
    assert '"codec": "mpeg2"' in capsys.readouterr().out
    assert synth_main(["generate", "--out", str(tmp_path / "x.mpg"), "--set", "video.codec=nope"]) == 1
