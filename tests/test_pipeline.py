"""Pipeline glue: field-order resolution from tracks, numbering, CLI parsing."""
from __future__ import annotations

import pytest

from pitchspeed.cli import build_parser
from pitchspeed.pipeline import renumber, resolve_field_order
from pitchspeed.types import EventTime, Pitch, PitchTrack, ScanInfo, SpeedEstimate, TrackPoint


def _scan(conf="low", source="content", mode="interlaced"):
    return ScanInfo(mode=mode, field_order="bff" if mode == "interlaced" else None,
                    confidence=conf, source=source, sample_rate=59.94)


def _track(preferred, rg, rf, n=30, conflict=False):
    return PitchTrack(track_id=0, points=[TrackPoint(s=0, t=0.0, x=0.0, y=0.0)],
                      diagnostics={"field_order": {"n": n, "preferred": preferred, "conflict": conflict,
                                                   "rms_given": rg, "rms_flipped": rf}})


def test_tracks_confirm_field_order():
    scan = _scan()
    assert resolve_field_order(scan, [_track("given", 0.45, 14.2)]) == "confirmed"
    assert scan.confidence == "high"


def test_tracks_contradict_field_order():
    scan = _scan()
    v = resolve_field_order(scan, [_track("flipped", 14.0, 0.4, conflict=True),
                                   _track("flipped", 12.0, 0.5, conflict=True)])
    assert v == "flip"
    assert scan.confidence == "low"


def test_weak_or_short_tracks_do_not_decide():
    scan = _scan()
    assert resolve_field_order(scan, [_track("given", 0.45, 0.6), _track("given", 0.4, 14.0, n=6),
                                      _track("unclear", None, None)]) == "unknown"
    assert scan.confidence == "low"


def test_mixed_evidence_is_unknown():
    scan = _scan()
    assert resolve_field_order(scan, [_track("given", 0.4, 14.0),
                                      _track("flipped", 14.0, 0.4, conflict=True)]) == "unknown"


def test_override_and_progressive_untouched():
    s1 = _scan(source="override")
    assert resolve_field_order(s1, [_track("flipped", 14.0, 0.4, conflict=True)]) == "unknown"
    s2 = _scan(mode="progressive")
    assert resolve_field_order(s2, [_track("given", 0.4, 14.0)]) == "unknown"


def _pitch(t, status="auto"):
    ev = EventTime(t=t, sigma=0.01, method="manual")
    return Pitch(number=0, status=status, release=ev, catch=EventTime(t=t + 0.5, sigma=0.01, method="manual"),
                 speed=SpeedEstimate(v_release_mph=70.0))


def test_renumber_chronological_skips_rejected():
    ps = [_pitch(9.0), _pitch(1.0, "rejected"), _pitch(5.0, "manual"), _pitch(3.0)]
    renumber(ps)
    assert [(p.release.t, p.number) for p in ps] == [(1.0, 0), (3.0, 1), (5.0, 2), (9.0, 3)]


@pytest.mark.parametrize("argv", [
    ["analyze", "v.avi", "--render", "none", "--field-order", "tff", "--start", "3", "--end", "9.5"],
    ["calibrate", "v.avi", "--view", "side", "--level", "ll", "--extension", "4.2", "--throws", "L"],
    ["run", "v.avi", "--ui-scale", "1.5"],
    ["review", "v.avi"],
    ["render", "v.avi", "--mode", "highlights", "--no-audio"],
    ["probe", "v.avi", "--par", "0.9091"],
])
def test_cli_parses(argv):
    args = build_parser().parse_args(argv)
    assert args.func is not None


class _Src:
    """Minimal SampleSource stand-in: sample s = 2*frame + k."""
    def locate(self, s):
        return (s // 2, s % 2, s % 2)


def _sim_track(kind, n=30, frame_dt=1001 / 30000, v=600.0, a=-300.0):
    """Ball at x = v t + a t^2 / 2 (px), sampled per field, timed as interlaced fields."""
    pts = []
    for s in range(n):
        f, k = s // 2, s % 2
        t_field = f * frame_dt + k * frame_dt / 2          # time the pipeline assigns
        if kind == "interlaced":
            t_true = t_field
        elif kind == "flipped":                            # real order is the other one
            t_true = f * frame_dt + (1 - k) * frame_dt / 2
        else:                                              # progressive: both fields at frame time
            t_true = f * frame_dt
        x = v * t_true + 0.5 * a * t_true ** 2
        y = 200.0 + 40.0 * t_true
        pts.append(TrackPoint(s=s, t=t_field, x=x, y=y))
    return PitchTrack(track_id=0, points=pts)


@pytest.mark.parametrize("kind,expected", [("interlaced", "confirmed"), ("flipped", "flip"),
                                           ("progressive", "progressive")])
def test_track_timing_hypotheses(kind, expected):
    scan = _scan()
    assert resolve_field_order(scan, [_sim_track(kind)], _Src(), 1001 / 30000) == expected


def test_slow_ball_is_unclear():
    scan = _scan()
    # ~0.3 px per field: no timing hypothesis can be told apart
    assert resolve_field_order(scan, [_sim_track("interlaced", v=18.0, a=0.0)], _Src(), 1001 / 30000) == "unknown"


def _tp(t, prog_last, speed=None, status="auto"):
    tr = PitchTrack(track_id=0, points=[TrackPoint(s=int(t * 60), t=t, x=0.0, y=0.0),
                                        TrackPoint(s=int(t * 60) + 20, t=t + 0.33, x=0.0, y=0.0)],
                    diagnostics={"progress_last": prog_last})
    return Pitch(number=0, status=status, release=EventTime(t=t, sigma=0.01, method="first_free_field"),
                 catch=EventTime(t=t + 0.5, sigma=0.01, method="extrapolated"),
                 speed=SpeedEstimate(v_release_mph=speed), track=tr)


def test_throwback_after_catch_is_demoted():
    from pitchspeed.pipeline import _demote_throwbacks
    rejected, missed = [], []
    ps = [_tp(2.0, 0.8, 70.0), _tp(5.0, 0.15), _tp(14.0, 0.8, 68.0)]   # catch 2.5 -> 2.5 s later
    kept = _demote_throwbacks(ps, rejected, missed)
    assert [p.release.t for p in kept] == [2.0, 14.0]
    assert rejected[0].reject_code == "THROWBACK_TIMING" and "throw back" in missed[0]["reason"]


@pytest.mark.parametrize("pitch", [
    _tp(5.0, 0.15, speed=65.0),      # has a speed: never demoted
    _tp(5.0, 0.6),                    # got far along the path
    _tp(9.0, 0.15),                   # too long after the catch
    _tp(5.0, 0.15, status="manual"),  # user-made
])
def test_throwback_rule_keeps_real_candidates(pitch):
    from pitchspeed.pipeline import _demote_throwbacks
    kept = _demote_throwbacks([_tp(2.0, 0.8, 70.0), pitch], [], [])
    assert len(kept) == 2
