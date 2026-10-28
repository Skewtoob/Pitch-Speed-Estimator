"""Combining E1 (time of flight) and E2 (side-view fit) estimates.

Each component carries an independent 1-sigma part and a part shared
through the drag constant k (both estimators lean on the same k, in the
same direction).  The combination (math.md §5):

    v       = sum w_i v_i,   w_i ~ 1 / sigma_ind_i^2
    sig_ind = 1 / sqrt(sum sigma_ind_i^-2)
    sig_k   = |sum w_i dv_k_i|          (signed k sensitivities, fully correlated)
    sigma   = sqrt(sig_ind^2 + sig_k^2)

Rules:
* Only components with details['usable'] (default True) are combined.
* A time-of-flight estimate whose release time was extrapolated along the
  same track (details['track_dependent']) is not independent of E2: when a
  usable E2 exists it is a cross-check only (E2 is reported even if the
  dependent E1 claims a smaller sigma: that sigma rests on the vision
  module's image-space extrapolation, which the validation showed to be
  biased in oblique views).
* |v1 - v2| > 2 sqrt(s1^2 + s2^2) (independent parts) -> flag E1_E2_disagree.
  Then prefer E2 if its fit looks healthy; otherwise keep the combination
  but widen sigma_ind by half the gap so the interval covers both.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from pitchspeed.types import SpeedComponent

from pitchspeed.estimate import common as K

DISAGREE_NSIGMA = 2.0
HEALTHY_RMS_PX = 2.0
UNHEALTHY_FLAGS = frozenset({K.F_FIELD_ORDER, K.F_PARTIAL_TRACK, K.F_POOR_FIT, K.F_OBLIQUE})


@dataclass
class Combined:
    v_release_mph: Optional[float]
    sigma_mph: Optional[float]
    sigma_ind_mph: Optional[float] = None
    sigma_shared_mph: Optional[float] = None
    method: str = "none"                       # 'tof' | 'side_fit' | 'tof+side_fit' | 'none'
    used: list[str] = field(default_factory=list)
    weights: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _ok(c: SpeedComponent) -> bool:
    return (bool(c.details.get("usable", True)) and K.finite(c.v_release_mph)
            and K.finite(c.sigma_ind_mph) and c.sigma_ind_mph > 0)


def _dv_k(c: SpeedComponent) -> float:
    dv = c.details.get("dv_k_mph")
    return float(dv) if K.finite(dv) else float(c.sigma_shared_mph or 0.0)


def side_fit_healthy(c: SpeedComponent) -> bool:
    d = c.details
    rms = d.get("rms_px")
    return (K.finite(rms) and rms <= HEALTHY_RMS_PX
            and not (set(d.get("flags", [])) & UNHEALTHY_FLAGS))


def _merge(parts: list[SpeedComponent]) -> tuple[float, float, float, dict]:
    w = [1.0 / c.sigma_ind_mph ** 2 for c in parts]
    sw = sum(w)
    wn = [x / sw for x in w]
    v = sum(x * c.v_release_mph for x, c in zip(wn, parts))
    s_ind = 1.0 / math.sqrt(sw)
    s_k = abs(sum(x * _dv_k(c) for x, c in zip(wn, parts)))
    return v, s_ind, s_k, {c.method: x for x, c in zip(wn, parts)}


def combine(components: list[SpeedComponent]) -> Combined:
    """Combine usable components; see the module docstring for the rules."""
    flags: list[str] = []
    notes: list[str] = []
    ok = [c for c in components if _ok(c)]
    sides = [c for c in ok if c.method == "side_fit"]
    tofs = [c for c in ok if c.method == "tof"]
    side = min(sides, key=lambda c: c.sigma_ind_mph) if sides else None

    indep_tofs = tofs
    if side is not None:
        indep_tofs = [c for c in tofs if not c.details.get("track_dependent")]
        if len(indep_tofs) < len(tofs):
            notes.append("time of flight is a cross-check only: its release time was "
                         "extrapolated from the same track as the side-view fit")
    tof = min(indep_tofs, key=lambda c: c.sigma_ind_mph) if indep_tofs else None

    # disagreement check against every computed time-of-flight estimate
    if side is not None:
        for c in tofs:
            gap = abs(c.v_release_mph - side.v_release_mph)
            lim = DISAGREE_NSIGMA * math.hypot(c.sigma_ind_mph, side.sigma_ind_mph)
            if gap > lim:
                K.add_flag(flags, K.F_DISAGREE)
                notes.append(f"time of flight ({c.v_release_mph:.1f} mph) and side-view fit "
                             f"({side.v_release_mph:.1f} mph) disagree by {gap:.1f} mph "
                             f"(> {lim:.1f}); check clicks, extension/catch point, event times, field order")

    if side is None and tof is None:
        return Combined(None, None, flags=flags, notes=notes)

    if side is not None and tof is not None:
        if K.F_DISAGREE in flags:
            if side_fit_healthy(side):
                parts, method = [side], "side_fit"
                notes.append("disagreement: using the side-view fit (its residuals look healthy)")
            else:
                parts, method = [tof, side], "tof+side_fit"
                notes.append("disagreement: neither estimate is clearly better; "
                             "interval widened to cover both")
        else:
            parts, method = [tof, side], "tof+side_fit"
    elif side is not None:
        parts, method = [side], "side_fit"
    else:
        parts, method = [tof], "tof"

    v, s_ind, s_k, weights = _merge(parts)
    if len(parts) == 2 and K.F_DISAGREE in flags:
        s_ind = math.hypot(s_ind, 0.5 * abs(parts[0].v_release_mph - parts[1].v_release_mph))
    sigma = math.hypot(s_ind, s_k)
    return Combined(v_release_mph=v, sigma_mph=sigma, sigma_ind_mph=s_ind, sigma_shared_mph=s_k,
                    method=method, used=[c.method for c in parts], weights=weights,
                    flags=flags, notes=notes)
