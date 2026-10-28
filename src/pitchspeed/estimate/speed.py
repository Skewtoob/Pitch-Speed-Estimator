"""Per-pitch speed estimate: runs E1 / E2, combines them, labels the result.

    estimate_speed(calib, scan, sample_dt, release, catch, track=None, manual=False)

* E1 (`tof_estimate`) whenever a release and a catch time exist.
* E2 (`side_fit_estimate`) when the calibration is a side view and the
  track has enough usable points.
* `combine` merges them (inverse variance on the independent parts plus the
  shared drag term; disagreement flag and preference rule).
* The result reports the release speed (radar-gun "out of the hand"
  equivalent), the release->catch average and the speed at the front of the
  plate, a 1-sigma and a 90 % interval, a label (good/fair/poor/none), the
  flags and the assumptions.  Label 'none' never carries a number; the
  components keep the raw values for diagnosis.

The pieces (`tof_estimate`, `side_fit_estimate`, `combine`, `manual_event`)
are re-exported here for the review UI and the tests.
"""
from __future__ import annotations

import math
from typing import Optional

from pitchspeed.calib.model import Calibration
from pitchspeed.physics import time_to_travel
from pitchspeed.types import EventTime, PitchTrack, ScanInfo, SpeedComponent, SpeedEstimate

from pitchspeed.estimate import common as K
from pitchspeed.estimate.combine import Combined, combine
from pitchspeed.estimate.events import manual_event
from pitchspeed.estimate.sidefit import side_fit_estimate
from pitchspeed.estimate.tof import tof_estimate

__all__ = ["estimate_speed", "tof_estimate", "side_fit_estimate", "combine", "Combined",
           "manual_event", "NotMeasurable", "label_for"]

NotMeasurable = K.NotMeasurable
label_for = K.label_for


def _derived_speeds(calib: Calibration, v0_mph: float, k: float, end_kind: str) -> tuple[float, float]:
    """(average release->end speed, plate speed) in mph implied by a release speed."""
    y_r = calib.release_y_ft
    y_end, _, _ = K.end_point(calib, end_kind)
    L = y_r - y_end
    v0 = K.ftps(v0_mph)
    v_h = v0
    for _ in range(4):                       # v0 = v_h * slope_factor(T(v_h))
        T = time_to_travel(v_h, L, k)
        v_h = v0 / K.slope_factor(calib, L, T)
    T = time_to_travel(v_h, L, k)
    return K.mph(L / T), K.mph(K.plate_speed(v0, y_r, k))


def _is_manual(manual: bool, release: Optional[EventTime], catch: Optional[EventTime]) -> bool:
    return bool(manual) or any(e is not None and e.method == "manual" for e in (release, catch))


def estimate_speed(calib: Calibration, scan: Optional[ScanInfo], sample_dt: float,
                   release: Optional[EventTime], catch: Optional[EventTime],
                   track: Optional[PitchTrack] = None, manual: bool = False) -> SpeedEstimate:
    """Full speed estimate for one pitch (DESIGN.md §3.3)."""
    if not (K.finite(sample_dt) and sample_dt > 0):
        raise ValueError(f"sample_dt must be a positive number of seconds, not {sample_dt!r}")
    is_manual = _is_manual(manual, release, catch)
    end_kind = track.end_kind if track is not None else "catch"
    track_flags = [f for f in (track.flags if track is not None else []) if f in K.SPEED_RELEVANT_TRACK_FLAGS]
    drag = K.drag_for(calib)
    problems: list[str] = []
    comps: list[SpeedComponent] = []

    side = None
    if calib.view == "side":
        if track is None or not track.points:
            problems.append("side-view fit: no ball track")
        else:
            try:
                side = side_fit_estimate(calib, scan, sample_dt, track, release=release, catch=catch)
            except K.NotMeasurable as e:
                problems.append(f"side-view fit: {e}")
    tof_flags = list(track_flags)
    if side is not None and K.F_FIELD_ORDER in side.details.get("flags", []):
        tof_flags.append(K.F_FIELD_ORDER)          # the fit saw a field-order problem: E1 timing too

    if release is not None and catch is not None:
        try:
            comps.append(tof_estimate(calib, scan, sample_dt, release, catch, end_kind=end_kind,
                                      manual=is_manual, track_flags=tof_flags,
                                      side=side.details if side is not None else None))
        except K.NotMeasurable as e:
            problems.append(f"time of flight: {e}")
    else:
        problems.append("time of flight: release and/or catch time missing")
    if side is not None:
        comps.append(side)

    for c in comps:
        if not c.details.get("usable", True) and c.details.get("reason"):
            problems.append(c.details["reason"])

    comb = combine(comps)
    flags: list[str] = []
    for f in track_flags:
        K.add_flag(flags, f)
    for c in comps:
        for f in c.details.get("flags", []):
            K.add_flag(flags, f)
    for f in comb.flags:
        K.add_flag(flags, f)
    if is_manual:
        K.add_flag(flags, K.F_MANUAL)
    if scan is not None and not scan.interlaced:
        K.add_flag(flags, K.F_FRAMES_ONLY)

    assumptions = K.assumptions(calib, drag, end_kind)
    assumptions.update({
        "sample_dt_s": float(sample_dt),
        "scan_mode": scan.mode if scan is not None else None,
        "field_order": scan.field_order if scan is not None else None,
        "model_sigma_tof": K.SIGMA_MODEL_TOF,
        "model_sigma_side_fit": K.SIGMA_MODEL_SIDE,
    })
    notes = []
    for c in comps:
        notes += [n for n in c.details.get("notes", []) if n not in notes]
    notes += [n for n in comb.notes if n not in notes]

    if comb.v_release_mph is None:
        reason = "; ".join(problems) or "no estimator could be applied"
        return SpeedEstimate(v_release_mph=None, method="none", label="none", components=comps,
                             flags=flags, assumptions=assumptions, reason=reason)

    v, sigma = comb.v_release_mph, comb.sigma_mph
    rel = sigma / v if v > 0 else math.inf
    label = K.label_for(rel)
    method = comb.method
    if method == "tof":
        e1 = next(c for c in comps if c.method == "tof")
        v_avg, v_plate = e1.v_avg_mph, e1.details["v_plate_mph"]
    else:
        v_avg, v_plate = _derived_speeds(calib, v, drag.k, end_kind)
    if method == "tof" and is_manual:
        method = "manual_tof"

    if label == "none":
        reason = (f"not measurable: uncertainty +-{100 * rel:.0f}% (> {100 * K.LABEL_POOR:.0f}%)"
                  + (f"; {'; '.join(problems)}" if problems else ""))
        return SpeedEstimate(v_release_mph=None, method=method, label="none", components=comps,
                             flags=flags, assumptions=assumptions, reason=reason)

    caveats = problems + [n for n in notes if n not in problems]
    return SpeedEstimate(v_release_mph=float(v), sigma_mph=float(sigma), ci90_mph=K.ci90(v, sigma),
                         v_avg_mph=float(v_avg), v_plate_mph=float(v_plate), method=method, label=label,
                         components=comps, flags=flags, assumptions=assumptions,
                         reason="; ".join(caveats) if caveats else None)
