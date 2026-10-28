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
* Hand-marked events (EventTime.method == 'manual', estimate.events): a
  manual catch always ends the flight at the glove (end point 'catch', never
  the finder's 'contact'); track flags about how an automatic event was
  found are dropped for the event the user marked; method 'manual_tof' only
  when both events were marked.
* Never raises for bad inputs: impossible settings
  (`calib.model.settings_errors`, e.g. an extension typed in inches), a bad
  sample interval or a failure inside an estimator give label 'none' (or
  the other estimator's number) with the reason.

The pieces (`tof_estimate`, `side_fit_estimate`, `combine`, `manual_event`)
are re-exported here for the review UI and the tests.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

from pitchspeed.calib.model import Calibration, settings_errors
from pitchspeed.physics import time_to_travel
from pitchspeed.types import EventTime, PitchTrack, ScanInfo, SpeedComponent, SpeedEstimate

from pitchspeed.estimate import common as K
from pitchspeed.estimate.combine import Combined, combine
from pitchspeed.estimate.events import manual_event
from pitchspeed.estimate.sidefit import side_fit_estimate
from pitchspeed.estimate.tof import tof_estimate

__all__ = ["estimate_speed", "tof_estimate", "side_fit_estimate", "combine", "Combined",
           "manual_event", "NotMeasurable", "label_for"]

log = logging.getLogger(__name__)

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


def _dominant_term(comps: list, method: str) -> Optional[str]:
    """Plain-words name of the largest uncertainty term of the time-of-flight estimate."""
    if method not in ("tof", "manual_tof"):
        return None
    e1 = next((c for c in comps if c.method == "tof"), None)
    if e1 is None:
        return None
    d = e1.details
    s_r, s_c = d.get("sigma_release_s"), d.get("sigma_catch_s")
    terms = {"timing": d.get("rel_sigma_timing"), "distance": d.get("rel_sigma_distance")}
    if not all(K.finite(v) for v in terms.values()):
        return None
    if terms["distance"] >= terms["timing"]:
        return "the assumed release and catch points (extension / catch position)"
    if not (K.finite(s_r) and K.finite(s_c)):
        return "the flight time"
    if s_c >= s_r:
        return (f"the catch time (+-{1e3 * s_c:.0f} ms, {d.get('catch_method')}): the end of the flight "
                f"was not seen well enough")
    return (f"the release time (+-{1e3 * s_r:.0f} ms, {d.get('release_method')}): the ball was not seen "
            f"clearly right after release")


def _is_manual(manual: bool, release: Optional[EventTime], catch: Optional[EventTime]) -> bool:
    return bool(manual) or any(e is not None and e.method == "manual" for e in (release, catch))


def _safe_assumptions(calib, sample_dt, end_kind: str = "catch") -> dict:
    try:
        a = K.assumptions(calib, K.drag_for(calib), end_kind)
    except Exception:                     # assumptions are informative only
        a = {}
    a["sample_dt_s"] = float(sample_dt) if K.finite(sample_dt) else None
    return a


def _input_problems(calib, sample_dt) -> tuple[list[str], bool]:
    """(problems that rule out any estimate, whether they are settings errors)."""
    if calib is None:
        return ["no calibration for this part of the video"], False
    errs = settings_errors(calib.settings)
    if errs:
        return errs, True
    if not (K.finite(sample_dt) and sample_dt > 0):
        return [f"sample interval {sample_dt!r} is not a positive number of seconds"], False
    return [], False


def estimate_speed(calib: Calibration, scan: Optional[ScanInfo], sample_dt: float,
                   release: Optional[EventTime], catch: Optional[EventTime],
                   track: Optional[PitchTrack] = None, manual: bool = False) -> SpeedEstimate:
    """Full speed estimate for one pitch (DESIGN.md §3.3).

    Never raises for bad inputs: impossible settings (e.g. an extension
    typed in inches), missing or non-numeric event times, or a failure
    inside one estimator give label 'none' (or the other estimator's
    result) with the reason.
    """
    try:
        bad, settings_bad = _input_problems(calib, sample_dt)
        if bad:
            return SpeedEstimate(v_release_mph=None, method="none", label="none",
                                 flags=[K.F_INVALID_SETTINGS] if settings_bad else [],
                                 assumptions=_safe_assumptions(calib, sample_dt) if calib is not None else {},
                                 reason="cannot estimate the speed: " + "; ".join(bad))
        return _estimate(calib, scan, sample_dt, release, catch, track, manual)
    except Exception as e:                # last resort: one pitch must never abort a run
        log.exception("speed estimate failed")
        return SpeedEstimate(v_release_mph=None, method="none", label="none",
                             assumptions=_safe_assumptions(calib, sample_dt) if calib is not None else {},
                             reason=f"speed estimate failed ({type(e).__name__}: {e}); no number reported")


def _estimate(calib: Calibration, scan: Optional[ScanInfo], sample_dt: float,
              release: Optional[EventTime], catch: Optional[EventTime],
              track: Optional[PitchTrack], manual: bool) -> SpeedEstimate:
    rel_manual = release is not None and release.method == "manual"
    cat_manual = catch is not None and catch.method == "manual"
    is_manual = _is_manual(manual, release, catch)
    auto_end = (track.end_kind if track is not None else None) or "catch"
    # A hand-marked catch is "the first field with the ball in the glove"
    # (estimate.events): the flight ends at the glove, whatever the finder
    # thought the end of its track was (e.g. a 'contact' from a swing).
    end_kind = "catch" if cat_manual else auto_end
    # Track flags about how an automatic event was found are stale once the
    # user has marked that event.
    stale = (K.RELEASE_EVENT_TRACK_FLAGS if rel_manual else frozenset()) | \
            (K.CATCH_EVENT_TRACK_FLAGS if cat_manual else frozenset())
    track_flags = [f for f in (track.flags if track is not None else [])
                   if f in K.SPEED_RELEVANT_TRACK_FLAGS and f not in stale]
    drag = K.drag_for(calib)
    problems: list[str] = []
    comps: list[SpeedComponent] = []
    own_notes: list[str] = []
    if cat_manual and auto_end == "contact":
        own_notes.append("catch marked by hand: the flight ends at the glove (the automatic track had "
                         "ended at a possible bat contact)")
    if rel_manual != cat_manual and release is not None and catch is not None:
        done, other, auto_ev = ("catch", "release", release) if cat_manual else ("release", "catch", catch)
        own_notes.append(f"only the {done} was marked by hand; the {other} is still the automatic "
                         f"estimate ({auto_ev.method}): check it in review")

    side = None
    if calib.view == "side":
        if track is None or not track.points:
            problems.append("side-view fit: no ball track")
        else:
            try:
                side = side_fit_estimate(calib, scan, sample_dt, track, release=release, catch=catch)
            except K.NotMeasurable as e:
                problems.append(f"side-view fit: {e}")
            except Exception as e:          # bad numbers in the track / clicks: E1 may still work
                log.warning("side-view fit failed", exc_info=True)
                problems.append(f"side-view fit failed ({type(e).__name__}: {e})")
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
        except Exception as e:
            log.warning("time-of-flight estimate failed", exc_info=True)
            problems.append(f"time of flight failed ({type(e).__name__}: {e})")
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
    notes = list(own_notes)
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
    if method == "tof" and rel_manual and cat_manual:   # both events marked by hand
        method = "manual_tof"

    if label == "none":
        why = _dominant_term(comps, method)
        reason = (f"not measurable: uncertainty +-{100 * rel:.0f}% (> {100 * K.LABEL_POOR:.0f}%)"
                  + (f", mostly {why}" if why else "")
                  + (f"; {'; '.join(problems)}" if problems else ""))
        return SpeedEstimate(v_release_mph=None, method=method, label="none", components=comps,
                             flags=flags, assumptions=assumptions, reason=reason)

    caveats = problems + [n for n in notes if n not in problems]
    return SpeedEstimate(v_release_mph=float(v), sigma_mph=float(sigma), ci90_mph=K.ci90(v, sigma),
                         v_avg_mph=float(v_avg), v_plate_mph=float(v_plate), method=method, label=label,
                         components=comps, flags=flags, assumptions=assumptions,
                         reason="; ".join(caveats) if caveats else None)
