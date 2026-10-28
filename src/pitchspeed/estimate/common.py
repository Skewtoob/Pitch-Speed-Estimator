"""Constants and small helpers shared by the speed estimators.

Units: feet, seconds and ft/s internally; mph only in reported numbers.
The numbers below come from `docs/design-notes/math.md` (tags there say
which are simulated, derived or assumed).
"""
from __future__ import annotations

import math
from typing import Optional

from pitchspeed import constants as C
from pitchspeed.calib.model import Calibration
from pitchspeed.physics import (BASEBALL_DIAM_M, BASEBALL_MASS_KG, CD_BASEBALL, CD_SOFTBALL,
                                SOFTBALL_DIAM_M, SOFTBALL_MASS_KG, DragModel, air_density,
                                slope_correction)
from pitchspeed.types import EventTime

# ---------------------------------------------------------------- model terms
SIGMA_MODEL_TOF = 0.0035     # E1 relative model error after the slope correction (math.md §1, SIM)
SIGMA_MODEL_SIDE = 0.005     # E2 relative model error (math.md §1/§3)
SLOW_MPH = 65.0              # below this the drag constant carries 25 % (DragModel slow=True)

# ------------------------------------------------------------ plausibility
V_MIN_MPH = 15.0             # outside [V_MIN, V_MAX] a number is not reported
V_MAX_MPH = 115.0
MIN_FLIGHT_SAMPLES = 8       # flight shorter than this many samples -> not measurable
MIN_SIDE_POINTS = 8          # E2 needs at least this many usable track points

# -------------------------------------------------- end point for bat contact
# ASSUMED: bat contact happens around the front of the plate.  y measured
# from the plate apex toward the pitcher (constants.py world frame).
CONTACT_Y_FT = C.PLATE_FRONT_Y_FT + 0.5
CONTACT_SIGMA_FT = 1.0

# ---------------------------------------------------------------- reporting
Z90 = 1.6448536269514722     # two-sided 90 % normal quantile
LABEL_GOOD = 0.025           # relative 1-sigma thresholds (DESIGN.md §1.5)
LABEL_FAIR = 0.05
LABEL_POOR = 0.10

# --------------------------------------------------------------------- flags
F_RELEASE_UNCONFIRMED = "release_unconfirmed"
F_CATCH_EXTRAPOLATED = "catch_extrapolated"
F_PARTIAL_TRACK = "partial_track"
F_FRAMES_ONLY = "frames_only"
F_FIELD_ORDER = "field_order_ambiguous"
F_DROPPED_FRAME = "dropped_frame_suspected"
F_OBLIQUE = "oblique_view"
F_DISAGREE = "E1_E2_disagree"
F_MANUAL = "manual_events"
F_CONTACT = "contact_end"
F_POOR_FIT = "poor_fit"
F_IMPLAUSIBLE = "implausible_speed"
F_EVENT_SIGMA_FLOORED = "event_sigma_floored"

# Track flags (set by the vision module) that matter for the speed and are
# copied into SpeedEstimate.flags.
SPEED_RELEVANT_TRACK_FLAGS = frozenset({
    F_RELEASE_UNCONFIRMED, F_CATCH_EXTRAPOLATED, F_PARTIAL_TRACK, F_FRAMES_ONLY, F_FIELD_ORDER,
    F_DROPPED_FRAME, "duplicate_frame_suspected", F_OBLIQUE,
})

# Variance (in sample_dt**2 units) added to the flight time when the field
# order is uncertain: with probability p the order is wrong and each event
# moves by +-1 field, i.e. T moves by 0 or +-2 fields -> var = p * 2 dt^2.
# p = 0.5 ("no idea") gives 1.0.
FIELD_ORDER_TIMING_VAR = 1.0
# A suspected dropped / duplicated frame inside the flight: one field of
# extra timing uncertainty (ASSUMED; timestamps normally cover PTS gaps).
DROPPED_FRAME_TIMING_VAR = 1.0


class NotMeasurable(ValueError):
    """Raised by an estimator that cannot produce a number; str(e) is the reason."""


def mph(v_ftps: float) -> float:
    return float(v_ftps) * C.FTPS_TO_MPH


def ftps(v_mph: float) -> float:
    return float(v_mph) * C.MPH_TO_FTPS


def finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def label_for(rel_sigma: Optional[float]) -> str:
    """Confidence label from the relative 1-sigma (DESIGN.md §1.5)."""
    if not finite(rel_sigma) or rel_sigma < 0:
        return "none"
    if rel_sigma <= LABEL_GOOD:
        return "good"
    if rel_sigma <= LABEL_FAIR:
        return "fair"
    if rel_sigma <= LABEL_POOR:
        return "poor"
    return "none"


def ci90(v: float, sigma: float) -> tuple[float, float]:
    return (float(v - Z90 * sigma), float(v + Z90 * sigma))


def drag_for(calib: Calibration, v_mph: Optional[float] = None) -> DragModel:
    """Default drag model for this calibration; `slow` from the speed if known."""
    st = calib.settings
    slow = finite(v_mph) and float(v_mph) < SLOW_MPH
    return DragModel.default(st.sport, st.altitude_ft, st.temp_f, slow=bool(slow))


def end_point(calib: Calibration, end_kind: str = "catch") -> tuple[float, float, str]:
    """(y_ft, sigma_ft, name) of the point where the flight time ends."""
    if end_kind == "contact":
        return CONTACT_Y_FT, CONTACT_SIGMA_FT, "contact"
    return calib.catch_y_ft, float(calib.settings.catch_sigma_ft), "catch"


def slope_factor(calib: Calibration, L: float, T: float) -> float:
    """Horizontal -> 3-D release speed factor (physics.slope_correction)."""
    return slope_correction(L, T, calib.release_height_ft, float(calib.settings.catch_height_ft))


def plate_speed(v0: float, release_y_ft: float, k: float) -> float:
    """Speed at the front edge of the plate (1-D drag model)."""
    return v0 * math.exp(-k * max(0.0, release_y_ft - C.PLATE_FRONT_Y_FT))


def event_sigma(ev: EventTime, sample_dt: float) -> tuple[float, bool]:
    """1-sigma of an event time with a floor; returns (sigma, floored).

    Events located to a whole sample cannot be better than the
    quantization sigma dt/sqrt(12); extrapolated (sub-field) events get a
    1 ms floor.  A missing / invalid sigma becomes dt/2.
    """
    quant = sample_dt / math.sqrt(12.0)
    floor = 0.001 if ev.method in ("extrapolated", "side_fit") else quant
    s = float(ev.sigma) if finite(ev.sigma) and ev.sigma > 0 else 0.5 * sample_dt
    if s < floor:
        return floor, True
    return s, False


def assumptions(calib: Calibration, drag: DragModel, end_kind: str = "catch") -> dict:
    """What the estimate assumed rather than measured (always reported)."""
    st = calib.settings
    rho_air = air_density(st.altitude_ft, st.temp_f)
    softball = st.sport == "softball"
    y_end, sig_end, end_name = end_point(calib, end_kind)
    return {
        "distance_ft": calib.distance_ft,
        "level": st.level,
        "sport": st.sport,
        "extension_ft": calib.extension_ft,
        "extension_sigma_ft": float(st.extension_sigma_ft),
        "extension_assumed": bool(calib.extension_is_assumed),
        "release_y_ft": calib.release_y_ft,
        "release_height_ft": calib.release_height_ft,
        "release_height_assumed": st.release_height_ft is None,
        "end_point": end_name,
        "end_y_ft": y_end,
        "end_sigma_ft": sig_end,
        "catch_behind_ft": float(st.catch_behind_ft),
        "catch_height_ft": float(st.catch_height_ft),
        "cd": CD_SOFTBALL if softball else CD_BASEBALL,
        "ball_mass_kg": SOFTBALL_MASS_KG if softball else BASEBALL_MASS_KG,
        "ball_diameter_m": SOFTBALL_DIAM_M if softball else BASEBALL_DIAM_M,
        "air_density_kg_m3": rho_air,
        "altitude_ft": float(st.altitude_ft),
        "temp_f": st.temp_f,
        "drag_k_per_ft": drag.k,
        "drag_k_rel_sigma": drag.rel_sigma,
        "pitcher_throws": st.pitcher_throws,
    }


def add_flag(flags: list[str], flag: str) -> None:
    if flag not in flags:
        flags.append(flag)
