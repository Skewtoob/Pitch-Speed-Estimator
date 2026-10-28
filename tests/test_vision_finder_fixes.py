"""Regression tests for the vision review fixes: re-seeding after rejected tracks, review-window
timing across clock jumps, hand-marked catch end kind, field-order flag reaching the estimator."""
import numpy as np
import pytest

from pitchspeed.estimate import common as K
from pitchspeed.types import Sample
from pitchspeed.vision import finder as F
from pitchspeed.vision import segment as SG
from pitchspeed.vision.detect import CAND_DTYPE, DetFrame
from pitchspeed.vision.finder import PitchFinder, track_between
from pitchspeed.vision.options import FinderOptions
from pitchspeed.vision.track import Pt, Track
from vision_helpers import ListSource, NTSC_FRAME_DT, behind_calibration, make_side_scene, scan_for, \
    textured_background

DT = NTSC_FRAME_DT / 2


def _behind_finder(opts=None):
    return PitchFinder(behind_calibration(), scan_for("bff"), DT, (240, 720),
                       textured_background(480, 720).astype(np.uint8), opts or FinderOptions())


def _frame(s, xs, uid0):
    c = np.zeros(len(xs), CAND_DTYPE)
    for i, x in enumerate(xs):
        c[i]["uid"], c[i]["s"], c[i]["t"], c[i]["x"], c[i]["y"], c[i]["z"] = uid0 + i, s, s * DT, x, 200.0, 20.0
    return DetFrame(s=s, t=s * DT, k=s & 1, parity=s & 1, A=np.eye(3), quality=1.0, cands=c)


def test_rejected_track_keeps_its_detections_by_default():
    f = _behind_finder()
    for s in range(100, 110):
        f.store.add(_frame(s, [300.0 + s, 500.0], uid0=10 * s))
    tr = Track(tid=7, seed_score=50.0)
    for s in range(100, 104):
        tr.pts[s] = Pt(s=s, t=s * DT, k=s & 1, x=300.0 + s, y=200.0, z=20.0, attached=False, sigma=0.5, uid=10 * s)
    f.store.claim(tr)
    f._finalize(tr)
    assert all(f.store.is_used(s, 10 * s) for s in range(100, 104)) and not f._blocked_seeds
    assert f._reseed


def test_rejected_track_releases_its_detections_but_not_as_seeds():
    f = _behind_finder(FinderOptions(reseed_release_claims=True))
    for s in range(100, 110):
        f.store.add(_frame(s, [300.0 + s, 500.0], uid0=10 * s))
    tr = Track(tid=7, seed_score=50.0)
    for s in range(100, 104):                       # 4 points: TOO_SHORT -> rejected
        tr.pts[s] = Pt(s=s, t=s * DT, k=s & 1, x=300.0 + s, y=200.0, z=20.0, attached=False, sigma=0.5, uid=10 * s)
    f.store.claim(tr)
    assert f.store.is_used(101, 1010)
    f._finalize(tr)
    assert not any(f.store.is_used(s, 10 * s) for s in range(100, 104))     # free for the ball again
    assert all(10 * s in f._blocked_seeds[s] for s in range(100, 104))      # ... to extend into
    assert f._reseed                                                         # and the window re-seeded
    # the blocked detections are forgotten once they leave the candidate cache
    f.store.frames.clear()
    for s in range(200, 205):
        f.store.add(_frame(s, [300.0], uid0=10 * s))
    f._run_tracker(final=False)
    assert not any(k < 200 for k in f._blocked_seeds)


def test_blocked_seed_is_not_rebuilt(monkeypatch):
    f = _behind_finder()
    for s in range(100, 110):
        f.store.add(_frame(s, [300.0 + s], uid0=10 * s))
    cand = lambda s: f.store.get(s).cands[0]
    seeds = [(50.0, cand(100), cand(101), cand(102)), (40.0, cand(104), cand(105), cand(106))]
    monkeypatch.setattr(F, "find_seeds", lambda *a, **k: list(seeds))
    built = []

    def fake_build(tid, seed, *a, **k):
        built.append(int(seed[2]["s"]))
        t = Track(tid=tid, seed_score=5.0)
        return t                                      # no points: discarded right away

    monkeypatch.setattr(F, "build_track", fake_build)
    f._blocked_seeds[101] = {1010}
    f._seed_range(100, 109, 100, 109, reseed=True)
    assert built == [105]


def test_stabilization_window_uses_recorded_sample_times():
    # 30 frame slots of PTS gaps before frame 300, registration failed for s 800..829
    f = _behind_finder()
    grid = np.arange(600).astype(float)
    grid[300:] += 30
    s = np.arange(1200)
    t = grid[s >> 1] * 2 * DT + (s & 1) * DT
    q = np.ones(1200)
    q[800:830] = 0.0
    f._s, f._tt, f._q = list(s), list(t), list(q)
    f._s_first, f._t_first = 0, float(t[0])
    f._tl = {"s": list(s), "t": list(t), "e_rel": [float("nan")] * 1200, "e_catch": [float("nan")] * 1200}
    out = [m for m in f._possible_missed([], []) if m["reason"] == "stabilization_failed"]
    assert len(out) == 1
    m = out[0]
    assert (m["s_start"], m["s_end"]) == (800, 829)
    assert m["t_start"] == pytest.approx(t[800]) and m["t_end"] == pytest.approx(t[829])


@pytest.fixture(scope="module")
def scene():
    return make_side_scene(n_fields=170, field_order="bff")


def test_track_between_never_inherits_an_automatic_contact_end(scene, monkeypatch):
    samples, cf, ref, gt = scene
    cal = cf.segment_calibs()[0]
    monkeypatch.setattr(SG.Segmenter, "_contact", lambda self, tr, m, last: True)   # a 'swing'
    s_rel, s_cat = int(round(gt["t_release"] / DT)), int(round(gt["t_catch"] / DT))
    tr = track_between(ListSource(samples), cal, scan_for("bff"), s_rel, s_cat, ref)
    assert tr.diagnostics["auto_end_kind"] == "contact"
    assert tr.end_kind == "catch"          # a hand-marked catch is the ball in the glove


def test_field_order_flag_name_matches_the_estimator():
    assert SG.F_FIELD_ORDER == K.F_FIELD_ORDER
    assert K.F_FIELD_ORDER in K.SPEED_RELEVANT_TRACK_FLAGS


def test_wrong_field_order_reaches_the_estimator():
    samples, cf, ref, gt = make_side_scene(n_fields=170, field_order="tff", seed=9)
    wrong = []
    for i in range(len(samples) // 2):
        a, b = samples[2 * i], samples[2 * i + 1]
        wrong.append(Sample(s=2 * i, frame=i, k=0, parity=1, t=a.t, image=b.image))
        wrong.append(Sample(s=2 * i + 1, frame=i, k=1, parity=0, t=b.t, image=a.image))
    cal = cf.segment_calibs()[0]
    f = PitchFinder(cal, scan_for("bff"), DT, wrong[0].image.shape, ref, FinderOptions())
    for smp in wrong:
        f.process(smp)
    res = f.finish()
    flagged = [t for t in res.tracks + res.rejected if "field_order_conflict" in t.flags]
    assert flagged
    assert all(K.F_FIELD_ORDER in t.flags for t in flagged)
