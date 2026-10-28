"""Event times marked by hand in the review UI.

Semantics of a mark (the same for both kinds, so constant exposure offsets
cancel in the flight time):

* release: the user marks the FIRST field in which the ball is out of the hand;
* catch:   the user marks the FIRST field in which the ball is in the glove
  (always the glove: a hand-marked catch ends the flight at the catch point,
  never at a bat contact; see estimate.speed).

If the right field is marked, the true event lies in (t - dt, t] with t the
marked sample's time: `manual_event` reports it as t - dt/2, with the
uniform-quantization sigma dt / sqrt(12) (what the mark itself says).

Human identification term (ASSUMED, not measured), added by the estimator
(`common.event_sigma`) to every hand-marked event, whoever built it: on
blurry interlaced SD footage the ball barely separates from the hand /
merges with the glove inside one field, and a non-expert marks the
neighbouring field now and then.  Modelled as an independent +-1-field slip
with probability MANUAL_SLIP_PROB = 0.25 per mark (variance p dt^2 -> 0.5
field), so a `manual_event` mark is used with

    sigma = dt * sqrt(1/12 + MANUAL_SLIP_PROB)  ~ 0.58 field  (9.6 ms NTSC)

The size is a judgement ("about one mark in four is a field off"): it
doubles the timing variance of a hand-marked flight (sigma_T 0.41 -> 0.82
field), which is what a single slipped mark costs (1 field ~ 3.7 % of the
flight time at 90 mph), while hand marks stay far tighter than the automatic
first-clean-field release prior (1.5 fields).  Typical behind-view manual
estimates move from ~2.8 % to ~3.9 % (still 'fair'); validated for coverage
with simulated slips in tests/test_estimate_validation.py.  Revisit when
real tapes give a measured marking repeatability.
"""
from __future__ import annotations

import math
from typing import Optional

from pitchspeed.types import EventTime

MANUAL_KINDS = ("release", "catch")
MANUAL_SLIP_PROB = 0.25          # ASSUMED probability that a mark is one field off (see module doc)


def manual_id_sigma(sample_dt: float) -> float:
    """1-sigma (s) of the human identification term of one hand mark (see module doc)."""
    return float(sample_dt) * math.sqrt(MANUAL_SLIP_PROB)


def manual_sigma(sample_dt: float) -> float:
    """1-sigma (s) the estimator uses for a `manual_event` mark: quantization + identification."""
    return math.hypot(float(sample_dt) / math.sqrt(12.0), manual_id_sigma(sample_dt))


def manual_event(sample_time_of_marked_field: float, sample_dt: float, kind: str,
                 s: Optional[int] = None) -> EventTime:
    """EventTime for a field the user marked as release or catch (see module doc)."""
    if kind not in MANUAL_KINDS:
        raise ValueError(f"kind must be one of {MANUAL_KINDS}, not {kind!r}")
    if not (sample_dt > 0 and math.isfinite(sample_dt)):
        raise ValueError("sample_dt must be a positive number")
    t = float(sample_time_of_marked_field)
    return EventTime(t=t - 0.5 * sample_dt, sigma=sample_dt / math.sqrt(12.0), method="manual", s=s)
