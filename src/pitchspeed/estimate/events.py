"""Event times marked by hand in the review UI.

Semantics of a mark (the same for both kinds, so constant exposure offsets
cancel in the flight time):

* release: the user marks the FIRST field in which the ball is out of the hand;
* catch:   the user marks the FIRST field in which the ball is in the glove.

The true event therefore lies in (t - dt, t] with t the marked sample's
time: it is reported as t - dt/2 with the uniform-quantization sigma
dt / sqrt(12).
"""
from __future__ import annotations

import math
from typing import Optional

from pitchspeed.types import EventTime

MANUAL_KINDS = ("release", "catch")


def manual_event(sample_time_of_marked_field: float, sample_dt: float, kind: str,
                 s: Optional[int] = None) -> EventTime:
    """EventTime for a field the user marked as release or catch (see module doc)."""
    if kind not in MANUAL_KINDS:
        raise ValueError(f"kind must be one of {MANUAL_KINDS}, not {kind!r}")
    if not (sample_dt > 0 and math.isfinite(sample_dt)):
        raise ValueError("sample_dt must be a positive number")
    t = float(sample_time_of_marked_field)
    return EventTime(t=t - 0.5 * sample_dt, sigma=sample_dt / math.sqrt(12.0), method="manual", s=s)
