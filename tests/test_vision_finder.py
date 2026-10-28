"""PitchFinder / track_between on in-memory synthetic scenes (no video files needed)."""
import json
import os

import numpy as np
import pytest

from pitchspeed.types import TrackPoint, to_jsonable
from pitchspeed.vision import FinderOptions, FinderResult, PitchFinder, eval_model, track_between
from vision_helpers import ListSource, NTSC_FRAME_DT, make_side_scene, scan_for

DT = NTSC_FRAME_DT / 2


def _run(samples, cf, ref, fo="bff", opts=None):
    cal = cf.segment_calibs()[0]
    f = PitchFinder(cal, scan_for(fo), DT, samples[0].image.shape, ref, opts or FinderOptions())
    for smp in samples:
        f.process(smp)
    return f.finish(), f


@pytest.fixture(scope="module")
def scene():
    return make_side_scene(n_fields=170, field_order="bff")


@pytest.fixture(scope="module")
def result(scene):
    samples, cf, ref, gt = scene
    return _run(samples, cf, ref)


def test_finds_the_pitch_with_accurate_events(scene, result):
    samples, cf, ref, gt = scene
    res, f = result
    assert isinstance(res, FinderResult)
    assert len(res.tracks) == 1, [(r.reject_code, len(r.points)) for r in res.rejected]
    tr = res.tracks[0]
    assert tr.status == "pitch" and tr.end_kind == "catch"
    assert abs(tr.release.t - gt["t_release"]) < 0.004
    assert abs(tr.catch.t - gt["t_catch"]) < DT
    assert tr.release.method == "extrapolated" and tr.catch.method in ("last_moving_field", "extrapolated")
    assert 0 < tr.release.sigma < 0.03 and 0 < tr.catch.sigma < 0.02
    assert tr.release.s is not None and tr.catch.s is not None
    # stabilized positions: reference-frame px (frame rows)
    P = np.array([[p.t, p.x, p.y] for p in tr.points if p.inlier])
    err = P[:, 1:] - gt["traj"](P[:, 0])
    assert np.sqrt(np.mean(err ** 2)) < 0.3
    assert "field_order_conflict" not in tr.flags
    assert tr.diagnostics["field_order"]["preferred"] == "given"
    for key in ("t_first_obs", "t_last_obs", "s_first_obs", "s_last_obs", "attached"):
        assert key in tr.diagnostics
    # the model dict can be evaluated and stays close to the points
    q = eval_model(tr.model, P[:, 0])
    assert np.max(np.hypot(*(q - P[:, 1:]).T)) < 1.5
    json.dumps(to_jsonable(tr))                      # serializable


def test_transforms_contract(scene, result):
    samples, cf, ref, gt = scene
    res, f = result
    n = len(samples)
    assert res.transforms.shape == (n, 2, 3)
    assert res.transform_quality.shape == (n,) and res.sample_index.shape == (n,)
    assert (res.transform_quality > 0.2).all()
    # transforms map sample frame px -> reference px (checked near the corridor centre)
    c = np.array([360.0, 255.0, 1.0])
    for i in (5, 60, 120):
        A_true = gt["A_true"](int(res.sample_index[i]))
        assert np.hypot(*(res.transforms[i] @ c - A_true @ c)) < 0.3


def test_no_ball_no_pitch():
    samples, cf, ref, gt = make_side_scene(n_fields=150, ball=False, seed=5)
    res, f = _run(samples, cf, ref)
    assert res.tracks == []


def test_throwback_is_rejected_not_dropped():
    samples, cf, ref, gt = make_side_scene(n_fields=170, throwback=True, speed_mph=50.0, seed=7)
    res, f = _run(samples, cf, ref)
    assert res.tracks == []
    assert any(r.reject_code == "THROWBACK" for r in res.rejected)


def test_wrong_field_order_is_flagged():
    from pitchspeed.types import Sample
    samples, cf, ref, gt = make_side_scene(n_fields=170, field_order="tff", seed=9)
    # read the TFF content as BFF: in each frame the bottom field is taken as the first one
    wrong = []
    for i in range(len(samples) // 2):
        a, b = samples[2 * i], samples[2 * i + 1]          # a: top (exposed first), b: bottom
        wrong.append(Sample(s=2 * i, frame=i, k=0, parity=1, t=a.t, image=b.image))
        wrong.append(Sample(s=2 * i + 1, frame=i, k=1, parity=0, t=b.t, image=a.image))
    res, f = _run(wrong, cf, ref, fo="bff")
    flagged = [t for t in res.tracks + res.rejected if "field_order_conflict" in t.flags]
    assert flagged


def test_debug_outputs(tmp_path, scene):
    samples, cf, ref, gt = scene
    d = tmp_path / "dbg ü"                           # non-ASCII path
    res, f = _run(samples, cf, ref, opts=FinderOptions(debug_dir=str(d)))
    names = set(os.listdir(d))
    assert {"overview.png", "timeline.png", "tracks.jsonl", "pitch_001.png", "finder_diagnostics.json"} <= names
    lines = (d / "tracks.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert any(json.loads(l).get("status") == "pitch" for l in lines)


def test_options_from_dict_and_pipeline_style_kwargs():
    o = FinderOptions.from_dict({"z_strong": 6.0, "not_an_option": 1})
    assert o.z_strong == 6.0 and o.extra["not_an_option"] == 1
    assert FinderOptions(**{"z_weak": 3.0}).z_weak == 3.0


def test_track_between_with_manual_events(scene):
    samples, cf, ref, gt = scene
    cal = cf.segment_calibs()[0]
    src = ListSource(samples)
    s_rel = int(round(gt["t_release"] / DT))
    s_cat = int(round(gt["t_catch"] / DT))
    tr = track_between(src, cal, scan_for("bff"), s_rel, s_cat, ref)
    assert tr.status == "pitch"
    assert tr.release.method == "manual" and tr.release.s == s_rel
    assert tr.catch.method == "manual" and tr.catch.s == s_cat
    assert abs(tr.release.t - samples[s_rel].t) < 1e-12
    assert len(tr.points) > 15
    assert all(s_rel - 2 <= p.s <= s_cat + 2 for p in tr.points)
    assert "auto_release" in tr.diagnostics


def test_track_between_manual_points_only(scene):
    samples, cf, ref, gt = scene
    cal = cf.segment_calibs()[0]
    # a window with no ball: only the user's points
    src = ListSource(samples)
    pts = [TrackPoint(s=10 + i, t=samples[10 + i].t, x=500.0 - 15 * i, y=240.0, manual=True) for i in range(5)]
    tr = track_between(src, cal, scan_for("bff"), 10, 20, ref, manual_points=pts)
    assert tr.status == "pitch"
    assert sum(p.manual for p in tr.points) == 5
