"""Behind view: early detections separated from the flight by a gap widen the release."""
from types import SimpleNamespace

from pitchspeed.vision.segment import RELEASE_GAP_SAMPLES, Segmenter


def _pts(ss):
    return [SimpleNamespace(s=s, t=s / 60.0) for s in ss]


def _seg():
    seg = Segmenter.__new__(Segmenter)
    seg.opts = SimpleNamespace(extra={})
    return seg


def test_gap_after_first_clean_is_found():
    early = list(range(100, 107))
    flight = list(range(107 + RELEASE_GAP_SAMPLES + 8, 107 + RELEASE_GAP_SAMPLES + 8 + 25))
    P = _pts(early + flight)
    assert _seg()._continuous_flight_start(P, 0) == len(early)


def test_continuous_track_has_no_gap():
    assert _seg()._continuous_flight_start(_pts(range(100, 130)), 0) is None


def test_small_tail_after_gap_is_not_the_flight():
    P = _pts(list(range(100, 120)) + [140, 141, 142])
    assert _seg()._continuous_flight_start(P, 0) is None
