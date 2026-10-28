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
