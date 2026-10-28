"""synth.evaluate: matching, speed / timing errors, interval coverage, suite aggregation."""
import csv
import json
import os

import pytest

from pitchspeed.synth.evaluate import (evaluate, evaluate_dir, format_dir_summary, format_summary,
                                       gt_time_mapper, match_by_time, write_pitch_csv)
from pitchspeed.types import (AnalysisResult, EventTime, Pitch, PitchTrack, ScanInfo, SpeedEstimate, TrackPoint,
                              VideoInfo, to_jsonable)


def _gt():
    def p(i, tr, tc, v, avg, plate):
        return dict(id=i, t_release=tr, t_catch=tc, flight_time=tc - tr, speed_release_mph=v,
                    avg_speed_y_mph=avg, speed_plate_front_mph=plate, extension_ft=5.0)
    return dict(
        video=dict(file="clip.mpg", codec="mpeg2", field_order="bff", scan="interlaced"),
        frame_map_identity=True, frame_map=[],
        pitches=[p(1, 3.000, 3.500, 80.0, 76.0, 75.0), p(2, 10.000, 10.600, 70.0, 66.5, 66.0),
                 p(3, 17.000, 17.700, 60.0, 57.0, 56.5), p(4, 24.000, 24.550, 75.0, 71.5, 71.0)],
        distractors=[dict(type="throwback", t_start=5.4, t_end=6.6, speed_mph=45.0, after_pitch=1),
                     dict(type="bird", t_start=12.0, t_end=16.0)],
        camera=dict(view="side", fence=dict(enabled=False)), level=dict(key="hs"),
        config=dict(render=dict(shutter_s=0.004)))


def _pitch(n, tr, tc, v, sigma=1.0, avg=None, plate=None, label="good", status="auto"):
    sp = SpeedEstimate(v_release_mph=v, sigma_mph=sigma if v is not None else None,
                       ci90_mph=(v - 1.645 * sigma, v + 1.645 * sigma) if v is not None else None,
                       v_avg_mph=avg, v_plate_mph=plate, method="tof" if v is not None else "none",
                       label=label if v is not None else "none", reason=None if v is not None else "no catch seen")
    return Pitch(number=n, status=status, release=EventTime(tr, 0.004, "first_free_field"),
                 catch=EventTime(tc, 0.004, "extrapolated"), speed=sp)


def _result(pitches, possible_missed=(), rejected=()):
    return AnalysisResult(video=VideoInfo("clip.mpg", 720, 480, 600, 29.97, "30000/1001"),
                          scan=ScanInfo("interlaced", "bff"), calibration={}, params={}, pitches=list(pitches),
                          possible_missed=list(possible_missed), rejected_tracks=list(rejected))


def _res_basic():
    return _result([
        _pitch(1, 3.010, 3.505, 81.0, 1.0, avg=76.5, plate=75.2),          # +1.0 mph, inside CI
        _pitch(2, 5.500, 6.400, 44.0, 2.0),                                # throwback -> false pitch
        _pitch(3, 10.150, 10.620, 74.0, 1.0),                              # +4.0 mph, outside CI, dt 150 ms
        _pitch(4, 17.050, 17.690, None),                                   # matched, no number
        _pitch(5, 17.200, 17.800, 61.0),                                   # second detection of pitch 3: false
        _pitch(7, 14.000, 14.500, 33.0),                                   # during the bird: false
        _pitch(6, 30.000, 30.5, 50.0, status="rejected"),                  # rejected pitches are ignored
    ], possible_missed=[dict(t_start=23.8, t_end=24.8, s_start=0, s_end=1, reason="weak track")])


def test_match_by_time_is_one_to_one_and_windowed():
    assert match_by_time([3.01, 5.5, 10.15, 17.05, 17.2], [3.0, 10.0, 17.0, 24.0]) == [(0, 0), (2, 1), (3, 2)]
    assert match_by_time([1.0], [1.3]) == []
    assert match_by_time([], [1.0]) == [] and match_by_time([1.0], []) == []
    # optimal (not greedy) assignment
    assert match_by_time([1.0, 1.3], [1.1, 0.82]) == [(1, 0), (0, 1)]   # greedy would match only one


def test_evaluate_counts_and_errors():
    m = evaluate(_res_basic(), _gt())
    s = m["summary"]
    assert (s["n_gt"], s["n_detected"], s["n_matched"], s["n_false"], s["n_missed"]) == (4, 6, 3, 3, 1)
    assert s["recall"] == pytest.approx(0.75) and s["precision"] == pytest.approx(0.5)
    assert s["n_missed_flagged"] == 1                         # pitch 4 lies in a possible-missed window
    assert s["n_with_speed"] == 2 and s["n_no_speed"] == 1
    assert s["speed"]["mean"] == pytest.approx(2.5) and s["speed"]["max_abs"] == pytest.approx(4.0)
    assert s["speed"]["rms"] == pytest.approx((0.5 * (1 + 16)) ** 0.5)
    assert s["coverage90"] == dict(n=2, inside=1, frac=0.5)
    assert s["release_ms"]["n"] == 3
    assert s["release_ms"]["max_abs"] == pytest.approx(150.0)
    assert s["catch_ms"]["mean"] == pytest.approx((5 + 20 - 10) / 3)
    assert s["avg_speed"]["n"] == 1 and s["avg_speed"]["mean"] == pytest.approx(0.5)
    assert s["plate_speed"]["mean"] == pytest.approx(0.2)
    rows = {r["gt_id"]: r for r in m["pitches"]}
    assert rows[1]["in_ci90"] is True and rows[2]["in_ci90"] is False
    assert rows[2]["z"] == pytest.approx(4.0) and rows[2]["err_pct"] == pytest.approx(100 * 4 / 70)
    assert rows[3]["v_release"] is None and rows[3]["reason"] == "no catch seen"
    assert not rows[4]["matched"] and rows[4]["flagged_possible_missed"]
    causes = {f["number"]: f["cause"] for f in m["false_pitches"]}
    assert causes == {2: "throwback", 5: "duplicate", 7: "bird"}
    assert s["false_by_cause"] == {"throwback": 1, "duplicate": 1, "bird": 1}
    txt = format_summary(m)
    for needle in ("recall 75%", "FALSE PITCH", "GT throwback", "second detection of GT pitch 3",
                   "possible-missed window", "no catch seen", "coverage"):
        assert needle in txt


def test_evaluate_accepts_dict_and_handles_empty():
    m = evaluate(to_jsonable(_res_basic()), _gt())
    assert m["summary"]["n_matched"] == 3 and m["summary"]["n_false"] == 3
    e = evaluate(_result([]), _gt())
    assert e["summary"]["recall"] == 0 and e["summary"]["precision"] is None
    assert e["summary"]["speed"]["n"] == 0
    format_summary(e)
    none = evaluate(_result([_pitch(1, 1.0, 1.5, 70.0)]), dict(_gt(), pitches=[]))
    assert none["summary"]["recall"] is None and none["summary"]["n_false"] == 1


def test_time_mapping_with_drops():
    gt = _gt()
    fdt = 1001 / 30000
    # source frame 30 dropped: from there on the video timeline runs one frame behind the truth
    fm = []
    for n in range(200):
        if n == 30:
            continue
        j = len(fm)
        fm.append(dict(out_frame=j, src_frame=n, t_video=j * fdt, exposure_times=[2 * n * 1001 / 60000], dup=False))
    gt.update(frame_map_identity=False, frame_map=fm)
    f = gt_time_mapper(gt)
    assert f(0.5) == pytest.approx(0.5)
    assert f(3.0) == pytest.approx(3.0 - fdt)
    res = _result([_pitch(1, 3.0 - fdt, 3.5 - fdt, 80.0)])
    m = evaluate(res, gt)
    assert m["pitches"][0]["dt_release_ms"] == pytest.approx(0.0, abs=1e-6)
    assert m["notes"]


def test_evaluate_dir(tmp_path):
    gt = _gt()
    gt["video"]["file"] = "a.mpg"
    (tmp_path / "a.mpg.gt.json").write_text(json.dumps(gt), encoding="utf-8")
    os.makedirs(tmp_path / "a_pitchspeed")
    (tmp_path / "a_pitchspeed" / "results.json").write_text(json.dumps(to_jsonable(_res_basic())), encoding="utf-8")
    gt2 = dict(_gt(), video=dict(gt["video"], file="b.mpg"))
    (tmp_path / "b.mpg.gt.json").write_text(json.dumps(gt2), encoding="utf-8")   # not analyzed
    with open(tmp_path / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "status", "file", "gt"])
        w.writeheader()
        w.writerow(dict(name="a", status="ok", file="a.mpg", gt="a.mpg.gt.json"))
        w.writerow(dict(name="b", status="ok", file="b.mpg", gt="b.mpg.gt.json"))
        w.writerow(dict(name="c", status="error", file="c.mpg", gt=""))
    d = evaluate_dir(str(tmp_path))
    assert d["status"] == {"ok": 1, "no_results": 1, "error": 1}
    o = d["overall"]
    assert o["n_gt"] == 4 and o["n_matched"] == 3 and o["n_false"] == 3
    assert o["speed"]["n"] == 2
    assert d["by"]["view"]["side"]["n_gt"] == 4
    txt = format_dir_summary(d)
    assert "no_results" in txt and "error" in txt
    write_pitch_csv(d, str(tmp_path / "pitches.csv"))
    rows = list(csv.DictReader(open(tmp_path / "pitches.csv", encoding="utf-8")))
    assert len(rows) == 7


def test_rejected_track_near_missed_pitch_is_reported():
    tr = PitchTrack(1, [TrackPoint(0, 24.1, 1.0, 1.0)], status="rejected", reject_code="too_short")
    m = evaluate(_result([], rejected=[tr]), _gt())
    assert {r["gt_id"]: r["rejected_track_nearby"] for r in m["pitches"]}[4] is True
