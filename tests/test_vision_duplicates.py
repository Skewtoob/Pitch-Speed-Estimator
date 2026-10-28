"""Duplicate resolution across candidate tracks (finder._resolve_duplicates).

Only candidates whose observations overlap (or nearly meet) in time are one
flight; the one that observed most of it wins; nothing is hidden.  The
candidates are built directly as PitchTracks along the release -> catch
image path of a behind-view calibration.
"""
import numpy as np
import pytest

from pitchspeed.types import EventTime, PitchTrack, TrackPoint
from pitchspeed.vision.finder import PitchFinder, _Record, flight_score, same_flight, same_path
from pitchspeed.vision.model import fit_model
from pitchspeed.vision.options import FinderOptions
from pitchspeed.vision.track import Track
from vision_helpers import behind_calibration, scan_for, textured_background

DT = 1001 / 60000


@pytest.fixture(scope="module")
def finder():
    cal = behind_calibration()
    return PitchFinder(cal, scan_for("bff"), DT, (240, 720), textured_background(480, 720).astype(np.uint8),
                       FinderOptions())


def _cand(f, tid, s0, s1, p0, p1, conf=0.5, z=50.0, offset=(0.0, 0.0), rel_sigma=0.025, catch=None,
          flags=()):
    """A 'pitch' candidate seen at samples s0..s1, progress p0 -> p1 (linear in time)."""
    g = f.geom
    rel_px, cat_px = np.asarray(g.release, float), np.asarray(g.catch, float)
    ss = np.arange(s0, s1 + 1)
    pp = p0 + (p1 - p0) * (ss - s0) / max(1, s1 - s0)
    xy = rel_px[None] + pp[:, None] * (cat_px - rel_px)[None] + np.asarray(offset, float)[None]
    tt = ss * DT
    pts = [TrackPoint(s=int(s), t=float(t), x=float(x), y=float(y), score=z, sigma=0.5, inlier=True)
           for s, t, (x, y) in zip(ss, tt, xy)]
    m = fit_model(tt, xy[:, 0], xy[:, 1], kind="poly", view="behind")
    t_rel = float(tt[0] - DT)
    pt = PitchTrack(track_id=tid, points=pts, status="pitch", confidence=conf, flags=list(flags),
                    release=EventTime(t=t_rel, sigma=rel_sigma, method="first_free_field", s=int(s0) - 1),
                    catch=catch or EventTime(t=float(tt[-1] + DT / 2), sigma=0.02, method="last_moving_field",
                                             s=int(s1) + 1),
                    model=m.to_dict())
    pt.diagnostics = {"progress_first": float(pp[0]), "progress_last": float(pp[-1]), "t_first_clean": float(tt[0]),
                      "n_inliers": len(pts)}
    return _Record(track=Track(tid=tid), pt=pt)


def _resolve(f, recs):
    f._records = list(recs)
    tracks, rejected = f._resolve_duplicates()
    missed = f._possible_missed(tracks, rejected)
    return tracks, rejected, missed


def test_candidates_a_second_apart_are_not_duplicates(finder):
    # eval2/s31337_029: junk near the release point (conf 0.35) 1 s before the real pitch (conf
    # 0.34); the old rule dropped the pitch as its DUPLICATE (release times < 1.5 s apart)
    junk = _cand(finder, 606, 378, 400, 0.05, 0.16, conf=0.35)
    real = _cand(finder, 499, 450, 471, 0.03, 0.34, conf=0.34)
    tracks, rejected, _ = _resolve(finder, [junk, real])
    assert {t.track_id for t in tracks} == {606, 499}
    assert not any(t.reject_code == "DUPLICATE" for t in rejected)


def test_fragment_that_saw_most_of_the_flight_wins(finder):
    # one flight: the full track (confidence 0.30) and a late fragment that only looks more
    # complete (its late 'first free field' release, confidence 0.55)
    full = _cand(finder, 166, 450, 480, 0.0, 0.7, conf=0.25)
    late = _cand(finder, 343, 462, 480, 0.4, 0.7, conf=0.55)
    tracks, rejected, missed = _resolve(finder, [late, full])
    assert [t.track_id for t in tracks] == [166]
    w = tracks[0]
    assert w.release.t == pytest.approx(449 * DT)            # its own release, not the late one
    assert w.confidence == pytest.approx(0.55) and w.diagnostics["confidence_own"] == pytest.approx(0.25)
    assert "review" in w.flags and "doubtful" not in w.flags
    lost = [t for t in rejected if t.track_id == 343][0]
    assert lost.reject_code == "DUPLICATE" and lost.diagnostics["duplicate_of"] == 166
    assert not missed                                         # inside the reported flight: nothing to list


def test_low_contrast_echo_loses_to_the_ball(finder):
    # eval2b/s31337_004: the ball (median z 96) and a parallel weak echo 15 px away (z 24) that
    # reached further along the corridor and had about the same inliers x progress
    ball = _cand(finder, 435, 150, 172, 0.03, 0.83, conf=0.99, z=96.0)
    echo = _cand(finder, 178, 153, 171, 0.03, 1.00, conf=0.83, z=24.0, offset=(-12.0, 15.0))
    assert flight_score(ball.pt) > flight_score(echo.pt)
    tracks, rejected, _ = _resolve(finder, [echo, ball])
    assert [t.track_id for t in tracks] == [435]
    assert not same_path(ball.pt, echo.pt, finder.geom, DT)


def test_same_path_fragment_seen_earlier_widens_a_late_release(finder):
    # the winner saw most of the flight but its first clean detection is late; a short fragment
    # on the same image path saw the ball 15 fields earlier: the winner's release (+-25 ms) is
    # provably late -> widened to cover both, never narrowed
    w = _cand(finder, 1, 110, 150, 0.2075, 0.7075, conf=0.5)
    early = _cand(finder, 2, 95, 112, 0.02, 0.2325, conf=0.4)
    assert same_path(w.pt, early.pt, finder.geom, DT)
    rel0 = w.pt.release
    tracks, rejected, _ = _resolve(finder, [w, early])
    assert [t.track_id for t in tracks] == [1]
    r = tracks[0].release
    assert r.sigma > rel0.sigma and r.t < rel0.t
    lo, hi = tracks[0].diagnostics["release_interval"]
    assert lo <= early.pt.release.t + 1e-9 and hi >= rel0.t - 1e-9
    assert "release_uncertain" in tracks[0].flags
    assert tracks[0].diagnostics["release_widened_by_track"] == 2


def test_echo_seen_earlier_does_not_move_the_release(finder):
    w = _cand(finder, 1, 110, 150, 0.2075, 0.7075, conf=0.5, z=80.0)
    echo = _cand(finder, 2, 95, 120, 0.02, 0.33, conf=0.4, z=20.0, offset=(0.0, 25.0))
    rel0 = w.pt.release
    tracks, _, _ = _resolve(finder, [w, echo])
    assert tracks[0].track_id == 1 and tracks[0].release.t == pytest.approx(rel0.t)


def test_loser_seen_outside_the_reported_flight_is_listed(finder):
    # a long track that ends inside the pitch's observations is the same flight (DUPLICATE),
    # but most of it was seen before the flight: listed for review, although the pitch's
    # release lies right there
    w = _cand(finder, 10, 150, 180, 0.0, 0.8, conf=0.6, z=90.0)
    long_junk = _cand(finder, 11, 90, 152, 0.20, 0.27, conf=0.31, z=60.0)
    tracks, rejected, missed = _resolve(finder, [w, long_junk])
    assert [t.track_id for t in tracks] == [10]
    assert [t.reject_code for t in rejected] == ["DUPLICATE"]
    dup = [m for m in missed if m["reason"] == "rejected_track:DUPLICATE"]
    assert len(dup) == 1
    assert dup[0]["t_start"] <= 90 * DT and dup[0]["t_end"] >= 149 * DT


def test_field_order_conflict_of_a_fragment_reaches_the_winner(finder):
    w = _cand(finder, 1, 150, 180, 0.0, 0.8, conf=0.6)
    frag = _cand(finder, 2, 152, 170, 0.05, 0.5, conf=0.4, flags=("field_order_conflict", "field_order_ambiguous"))
    frag.pt.diagnostics["field_order_pair"] = {"other_track": 3, "conflict": True, "rms_given": 12.5,
                                               "rms_flipped": 0.44}
    tracks, _, _ = _resolve(finder, [w, frag])
    assert {"field_order_conflict", "field_order_ambiguous"} <= set(tracks[0].flags)
    assert tracks[0].diagnostics["field_order_conflict_from_track"] == 2
    assert tracks[0].diagnostics["field_order_pair"]["rms_flipped"] == pytest.approx(0.44)


def test_same_flight_needs_overlapping_or_adjacent_observations(finder):
    a = _cand(finder, 1, 100, 130, 0.0, 0.7)
    assert same_flight(a.pt, _cand(finder, 2, 125, 140, 0.5, 0.8).pt, 12)
    assert same_flight(a.pt, _cand(finder, 3, 142, 150, 0.5, 0.8).pt, 12)        # 12 samples apart
    assert not same_flight(a.pt, _cand(finder, 4, 143, 150, 0.5, 0.8).pt, 12)
