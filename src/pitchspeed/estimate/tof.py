"""E1: release speed from the time of flight (any camera view).

    L  = (D - ext) + c                horizontal release -> end distance
    v0 = (exp(kL) - 1) / (k T)        1-D quadratic drag (physics.py)
    v  = v0 * slope_correction        horizontal -> 3-D release speed

Uncertainty (relative, added in quadrature; math.md §2):
  * timing   sigma_T / T,  sigma_T^2 = sigma_release^2 + sigma_catch^2 (+ inflation
    for an uncertain field order or a suspected dropped frame)
  * distance sigma_L / L * dlnv0/dlnL, sigma_L^2 = sigma_ext^2 + sigma_catch_point^2
  * model    0.35 %
  * drag k   reported separately as the *shared* part (E2 depends on k too).

Side view, release method 'extrapolated' (cv.md §5) -- interface contract:
EventTime.t is the time the track passed the calibration release point
(`calib.release_point()`: the clicked one, else the default from the
assumed extension), found in the image, and EventTime.sigma includes the
~1 ft uncertainty of where along the pitch line that image crossing is
(sideways offset of the ball, perspective).  L then starts at that point,
mapped to world y with E2's fitted rho (the default point is drawn with the
affine scale, which is ~1-2 ft off in oblique views) plus an obliquity term;
the extension only enters through drag (k * sigma_ext), and the
estimate is marked ``track_dependent``: it comes from the same track as E2,
so `combine` never adds the two.  E1 never uses E2's fitted release time.
"""
from __future__ import annotations

import math
from typing import Optional

from pitchspeed.calib.model import Calibration
from pitchspeed.physics import DragModel, v0_from_tof
from pitchspeed.types import EventTime, ScanInfo, SpeedComponent

from pitchspeed.estimate import common as K


def _timing_inflation(scan: Optional[ScanInfo], sample_dt: float, track_flags) -> tuple[float, list[str], list[str]]:
    """Extra flight-time variance (s^2), flags and notes from scan / track state."""
    var, flags, notes = 0.0, [], []
    if scan is not None and scan.interlaced and scan.source != "override" and scan.confidence == "low":
        var += K.FIELD_ORDER_TIMING_VAR * sample_dt ** 2
        flags.append(K.F_FIELD_ORDER)
        notes.append("field order uncertain: flight time may be off by +-2 fields")
    elif K.F_FIELD_ORDER in track_flags:
        var += K.FIELD_ORDER_TIMING_VAR * sample_dt ** 2
        flags.append(K.F_FIELD_ORDER)
        notes.append("track suggests a field-order problem: flight time may be off by +-2 fields")
    if K.F_DROPPED_FRAME in track_flags or "duplicate_frame_suspected" in track_flags:
        var += K.DROPPED_FRAME_TIMING_VAR * sample_dt ** 2
        flags.append(K.F_DROPPED_FRAME)
        notes.append("dropped/duplicated frame suspected inside the flight")
    return var, flags, notes


# An image-space (side-view 'extrapolated') release time is not trusted when
# E2's obliquity estimate exceeds this, or when even the widest plausible lens
# gives more than MAX_OBLIQUITY_MIN_DEG (validation: fine at 42 deg, biased by
# 5-15 % at 66 deg, where phi is sometimes under-estimated by ~10 deg).
MAX_OBLIQUITY_FOR_IMAGE_RELEASE_DEG = 55.0
MAX_OBLIQUITY_MIN_DEG = 45.0
SIDE_RELEASE_CLICK_SIGMA_FT = 0.3     # a clicked release point (2 px) mapped to world y
SIDE_RELEASE_NO_RHO_SIGMA_FT = 1.5    # rho unknown: the affine map can be off by ~1-2 ft near the rubber


def side_release_point(calib: Calibration, side: Optional[dict] = None) -> tuple[float, float, str]:
    """(world y, extra position sigma ft, source) of the point a side-view
    extrapolated release time refers to.

    The point is `calib.release_point()` (clicked, or the default drawn with
    the affine scale from the assumed extension).  Its world y comes from the
    plumb projection and E2's fitted rho (`side` = E2 details); without rho
    the affine map is used with a larger sigma.  E2's obliquity adds the
    image-crossing error of a ball released off the pitch line.
    """
    from pitchspeed.estimate import sidefit as SF            # local import: avoids a cycle at load
    try:
        geom = SF.axis_geometry(calib)
    except K.NotMeasurable:
        return calib.release_y_ft, float(calib.settings.extension_sigma_ft), "assumed_extension"
    clicked = calib.has("release")
    mu = float(geom.mu(calib.to_sq(calib.release_point())[None, :])[0])
    side = side or {}
    rho = side.get("rho")
    rho_ok = K.finite(rho) and rho > 0 and not side.get("at_bound", False)
    lam = float(Calibration.lam_of_mu(mu, rho if rho_ok else 1.0))
    var = (SIDE_RELEASE_CLICK_SIGMA_FT ** 2 if clicked else 0.0)
    var += 0.0 if rho_ok else SIDE_RELEASE_NO_RHO_SIGMA_FT ** 2
    t_phi, s_t = side.get("tan_phi"), side.get("sigma_tan_phi")
    if K.finite(t_phi) and K.finite(s_t):
        _, xr_sig, _ = SF.release_side(calib, geom)
        var += (t_phi ** 2 + s_t ** 2) * xr_sig ** 2
    src = "clicked_release_point" if clicked else "default_release_point"
    return calib.y_of_lam(lam), math.sqrt(var), src


def tof_estimate(calib: Calibration, scan: Optional[ScanInfo], sample_dt: float,
                 release: Optional[EventTime], catch: Optional[EventTime], *,
                 end_kind: str = "catch", manual: bool = False,
                 track_flags=(), drag: Optional[DragModel] = None,
                 side: Optional[dict] = None) -> SpeedComponent:
    """E1 time-of-flight estimate.  Raises `NotMeasurable` with the reason.

    `drag` overrides the default drag model (tests / sensitivity studies);
    `track_flags` are the vision flags of the pitch's track, if any; `side`
    is E2's details dict (side view: rho and obliquity for the release point).
    """
    if release is None or catch is None:
        raise K.NotMeasurable("time of flight needs both a release and a catch time")
    if not (K.finite(release.t) and K.finite(catch.t)):
        raise K.NotMeasurable("release or catch time is not a number")
    T = float(catch.t) - float(release.t)
    if T <= 0:
        raise K.NotMeasurable(f"catch ({catch.t:.3f} s) is not after release ({release.t:.3f} s)")
    if T < K.MIN_FLIGHT_SAMPLES * sample_dt:
        raise K.NotMeasurable(f"flight time {T * 1e3:.0f} ms is shorter than {K.MIN_FLIGHT_SAMPLES} samples")

    at_point = calib.view == "side" and release.method in ("extrapolated", "side_fit")
    if at_point:
        y_r, s_rel_pos, rel_src = side_release_point(calib, side)
    else:
        y_r, s_rel_pos, rel_src = calib.release_y_ft, float(calib.settings.extension_sigma_ft), "assumed_extension"
    y_end, sig_end, end_name = K.end_point(calib, end_kind)
    L = y_r - y_end
    if L <= 5.0:
        raise K.NotMeasurable(f"release->{end_name} distance {L:.1f} ft is not plausible; check distance/extension")

    d0 = drag or K.drag_for(calib)
    k = d0.k
    v_h = v0_from_tof(L, T, k)
    sf = K.slope_factor(calib, L, T)
    v0 = v_h * sf
    d = drag or K.drag_for(calib, K.mph(v0))       # 'slow' sigma once the speed is known

    # -- timing
    s_r, fl_r = K.event_sigma(release, sample_dt)
    s_c, fl_c = K.event_sigma(catch, sample_dt)
    var_extra, flags, notes = _timing_inflation(scan, sample_dt, set(track_flags or ()))
    s_T = math.sqrt(s_r ** 2 + s_c ** 2 + var_extra)
    rel_T = s_T / T
    # -- distance
    kl = k * L
    dlnv_dlnL = kl * math.exp(kl) / math.expm1(kl) if kl > 1e-9 else 1.0
    s_L = math.hypot(s_rel_pos, sig_end)
    rel_L = s_L / L * dlnv_dlnL
    # release at a known point, true release elsewhere: v_release = v_point * exp(k * dy)
    rel_ext_drag = k * float(calib.settings.extension_sigma_ft) if at_point else 0.0
    # -- drag (shared with E2), signed change of v0 for k -> k (1 + rel_sigma)
    dv_k = v0_from_tof(L, T, k * (1.0 + d.rel_sigma)) * sf - v0
    rel_model = K.SIGMA_MODEL_TOF
    rel_ind = math.sqrt(rel_T ** 2 + rel_L ** 2 + rel_ext_drag ** 2 + rel_model ** 2)

    if scan is not None and not scan.interlaced:
        flags.append(K.F_FRAMES_ONLY)
    if fl_r or fl_c:
        flags.append(K.F_EVENT_SIGMA_FLOORED)
    if not manual and release.method == "first_free_field":
        flags.append(K.F_RELEASE_UNCONFIRMED)
    if end_kind == "contact":
        flags.append(K.F_CONTACT)
    if catch.method == "extrapolated" and end_kind in ("lost", "occluded"):
        flags.append(K.F_CATCH_EXTRAPOLATED)
    if manual or release.method == "manual" or catch.method == "manual":
        flags.append(K.F_MANUAL)
    track_dependent = at_point

    v_plate = K.plate_speed(v0, y_r, k)
    details = {
        "usable": True,
        "reason": None,
        "flags": sorted(set(flags)),
        "notes": notes,
        "track_dependent": track_dependent,
        "L_ft": L,
        "T_s": T,
        "t_release": float(release.t),
        "t_catch": float(catch.t),
        "release_method": release.method,
        "catch_method": catch.method,
        "end_point": end_name,
        "release_point": rel_src,
        "release_y_ft": y_r,
        "sigma_release_s": s_r,
        "sigma_catch_s": s_c,
        "sigma_T_s": s_T,
        "k_per_ft": k,
        "k_rel_sigma": d.rel_sigma,
        "slope_factor": sf,
        "v_horizontal_mph": K.mph(v_h),
        "v_plate_mph": K.mph(v_plate),
        "rel_sigma_timing": rel_T,
        "rel_sigma_distance": rel_L,
        "rel_sigma_extension_drag": rel_ext_drag,
        "rel_sigma_model": rel_model,
        "rel_sigma_k": abs(dv_k) / v0,
        "dv_k_mph": K.mph(dv_k),
    }
    phi = (side or {}).get("phi_deg")
    phi_min = (side or {}).get("phi_min_deg")
    too_oblique = ((K.finite(phi) and phi > MAX_OBLIQUITY_FOR_IMAGE_RELEASE_DEG)
                   or (K.finite(phi_min) and phi_min > MAX_OBLIQUITY_MIN_DEG))
    if at_point and too_oblique:
        details["usable"] = False
        details["reason"] = (f"camera too oblique (phi ~ {phi:.0f} deg) for a release time extrapolated in "
                             f"the image; mark the release by hand or calibrate as a behind-plate view")
        details["flags"] = sorted(set(flags + [K.F_OBLIQUE]))
    elif not (K.V_MIN_MPH <= K.mph(v0) <= K.V_MAX_MPH):
        details["usable"] = False
        details["reason"] = (f"time of flight gives {K.mph(v0):.0f} mph, outside "
                             f"{K.V_MIN_MPH:.0f}-{K.V_MAX_MPH:.0f} mph: check release/catch and distance")
        details["flags"] = sorted(set(flags + [K.F_IMPLAUSIBLE]))
    return SpeedComponent(method="tof", v_release_mph=K.mph(v0), sigma_ind_mph=K.mph(v0 * rel_ind),
                          sigma_shared_mph=abs(K.mph(dv_k)), v_avg_mph=K.mph(L / T), details=details)
