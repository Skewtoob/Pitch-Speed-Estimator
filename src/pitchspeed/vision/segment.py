"""Turn a fitted track into a `PitchTrack`: reject codes, events, confidence.

Reject codes (cv.md section 5): TOO_SHORT, NO_PROGRESS, THROWBACK,
OFF_AXIS, SPEED_IMPLAUSIBLE, POOR_FIT (rms > 2 px, or < 50 % inliers between
the first and last inlier, or inliers cover < 40 % of the spanned samples), DUPLICATE (finder, across
tracks); additionally NO_EVENTS (release or catch time not determinable)
and LOW_CONFIDENCE (confidence < FinderOptions.min_confidence, default 0.1).
End kinds: 'catch' | 'contact' | 'occluded' | 'lost'.

Event-time semantics implemented (all times in seconds, video clock):

release
  side view, method 'extrapolated': the time at which the fitted image
    track crosses the release point's progress (s = 0 at
    `Calibration.release_point()`, clicked or derived from the assumed
    extension).  This is the time consistent with E1's
    L = D - ext + c.  sigma^2 = sigma_model^2 + (1 ft / v)^2, the second
    term for the unknown true release point (cv.md section 5).  Flag
    'ball_free_before_release_point' when the ball was already seen free
    before that crossing.
  behind view, method 'first_free_field': t = t_first_clean -
    latency_mean * dt, sigma^2 = (latency_sigma * dt)^2 + dt^2 / 12, where
    t_first_clean is the sample time of the first inlier detection that is
    strong (z >= z_strong), not attached to a body blob and followed by the
    rest of the track.  The image model's backward extrapolation to s = 0
    is only a diagnostic ('release_methods_delta'): in this view it cannot
    locate the release (perspective degeneracy, math.md 4b; on the
    synthetic suites it was off by 8-40 fields while claiming 1-2), so it
    neither moves nor widens the estimate.
  If the first clean detection is far into the flight (behind view), or
    there is no clean detection at all, the flag 'release_not_observed' is
    set and the release is taken uniform over
    [catch - slowest plausible flight time, first (clean) detection].
  If weak / attached detections on the flight path run continuously from
    earlier up to the first clean one (the ball merged with the hand), the
    release is taken as the middle of [t_pre, t_clean] - latency with sigma
    covering the interval (flag 'release_uncertain').
  Late first detection (behind view): the ball barely moves in the image
    for the first ~40 % of the flight (1-2 px per field) and the detector
    can miss those fields entirely; the first clean detection is then late
    by 5-30 fields (~20 % of the synthetic behind tracks).  The delivery-
    motion cue (`_delivery_cue_time`: last strong burst of foreground near
    the release click, DetFrame.e_rel) flags most of them: first clean
    detection > CUE_LATE_GAP_S after the burst -> flags
    'release_detection_late' + 'release_uncertain', release uniform over
    [burst - CUE_LEAD_S, first clean - latency].  The cue only ever widens.
    The rest (no clear burst) still read high; 'release_unconfirmed'
    (estimate) asks the user to check the release.
catch  (t_c = time the fitted track crosses the catch point, progress 1)
  side view:
  'last_moving_field': t_c <= t_last + dt (the catch point is reached in the
    field interval after the last detection): t = t_last + dt / 2,
    sigma^2 = dt^2/12 + (dt/2)^2.
  'extrapolated', track ran > 2 ft past the catch point: the last
    detections are glove / catcher motion; t = t_c, sigma from the model
    and the glove-position prior (flag 'track_past_catch_point').
  'extrapolated', track ended before the catch point: the glove may have
    been in front of the click (caught right away) or the ball was hidden,
    so the catch is taken uniform over [t_last + dt/2, t_c]: t = middle,
    sigma^2 = width^2/12 + sigma_model^2 + sigma_glove^2 + dt^2/12, with
    sigma_glove = catch_sigma_ft / v.  end_kind 'occluded' (runs into a
    body blob) or 'lost'; flag 'catch_extrapolated'.  If the ball is seen
    again in the next field short of the prediction (stopped in the glove),
    t is the model's closest approach to that point (sigma dt / sqrt(12)).
  Uninformative ('catch_not_observed'): extrapolation longer than 0.25 s
    or outside the flight-time window: t = middle of the plausible window
    given the release, sigma = window / sqrt(12).
  behind view (`Segmenter._catch_behind`): the catcher / umpire usually
    hide the last fields.  t_x = the model's crossing of the catch click's
    progress; s_glove = GLOVE_SPREAD_FT (in the image plane) x px/ft at the
    catch / the model's image speed along the axis at t_x; s_model = the
    model's position sigma at t_x / that speed, or the shift of t_x when the
    last JACKKNIFE_DROP inliers are left out, whichever is larger.
    'last_moving_field': t_x <= t_last + dt: t = t_last + dt/2,
      sigma^2 = dt^2/12 + (dt/2)^2 + min(s_glove, 3 dt)^2.
    'extrapolated' (flag 'catch_extrapolated'): N(t_x, s_glove^2 +
      s_model^2) truncated to [t_last, release + slowest flight]: t, sigma
      = its mean and sd.  Validated on GT ball positions of 169 synthetic
      behind pitches (90-97 % of catch times inside +-1.645 sigma for 1-16
      hidden fields, RMS 15-38 ms) and on ~100 real tracks of the synthetic
      suites (90-100 % for 1-10 hidden fields, 50-60 % beyond: hence the
      10-field limit below).
    'catch_not_observed' (uninformative window, as for the side view) when
      the model does not reach the catch point within CATCH_SEARCH_S, the
      extrapolated path turns back, the jackknife refit does not reach the
      catch point, the hidden flight is longer than CATCH_MAX_EXTRAP_S, or
      the sigma exceeds CATCH_MAX_SIGMA_S.
  'contact': a fast candidate track leaves the end point within
    `contact_gap` samples in another direction: end_kind 'contact', t as for
    'last_moving_field'.

Confidence (heuristic ranking, not a probability; cv.md section 6):
  f_snr * f_cov * f_fit * f_ends * f_speed * f_mc * f_cue.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from pitchspeed import constants as C
from pitchspeed.types import EventTime, PitchTrack, TrackPoint
from pitchspeed.vision.geometry import ViewGeometry
from pitchspeed.vision.model import TrajModel, fit_model
from pitchspeed.vision.options import FinderOptions
from pitchspeed.vision.track import CandStore, Track, _in_blob, field_order_check

MPH = C.MPH_TO_FTPS

# ---- behind-view catch (see Segmenter._catch_behind); overridable via FinderOptions.extra
GLOVE_SPREAD_FT = 0.75        # 1-sigma glove position vs the catch click, image plane (ASSUMED;
#                               real pitch locations scatter ~0.7-1 ft; synthetic suite ~0.5 ft)
SCALE_GUESS_FACTOR = 1.3      # px/ft at the catch is a geometry guess without plate-width clicks
CATCH_SEARCH_S = 0.4          # look for the catch-point crossing this far past the last detection
CATCH_MODEL_INFLATE = 1.0     # x the covariance-based extrapolation sigma
CATCH_MAX_EXTRAP_S = 0.17     # longer hidden flight (> 10 fields) -> 'catch_not_observed' (real tracks:
#                               90-100 % coverage up to 10 hidden fields, 50-60 % beyond)
CATCH_MAX_SIGMA_S = 0.10      # ... or an extrapolated catch sigma above this
JACKKNIFE_DROP = 2            # refit without the last N inliers to test the extrapolation
CONTACT_MAX_REMAINING_S = 3 * 1001 / 60000   # behind: contact only within 3 fields of the catch point
END_GAP_S = 0.1               # behind: trailing inliers after a longer gap ...
END_GAP_MAX_PTS = 4           # ... (at most this many) are not used for the catch
# ---- behind-view release: delivery-motion cue (see Segmenter._delivery_cue_time)
CUE_WINDOW_S = 0.65           # search this far before the first clean detection
CUE_K = 2.0                   # burst: >= median + CUE_K robust sigmas ...
CUE_FRAC = 0.6                # ... and >= this fraction of the window maximum
CUE_LATE_GAP_S = 5 * 1001 / 60000   # first clean detection > 5 fields after the burst -> late
CUE_LEAD_S = 3 * 1001 / 60000       # the burst may lag the release by this much


def truncated_normal(mu: float, sigma: float, lo: float = -math.inf, hi: float = math.inf) -> tuple[float, float]:
    """(mean, sd) of N(mu, sigma^2) truncated to [lo, hi]; (nan, nan) if [lo, hi] has no mass."""
    if not (sigma > 0 and math.isfinite(sigma)):
        return (min(max(mu, lo), hi), 0.0) if lo <= hi else (math.nan, math.nan)
    from scipy.stats import truncnorm
    a, b = (lo - mu) / sigma, (hi - mu) / sigma
    if a >= b or b < -8.0:
        return math.nan, math.nan
    if a > 8.0:                      # all the mass sits at the lower bound
        return float(lo), 0.0
    m, v = truncnorm.stats(a, b, loc=mu, scale=sigma, moments="mv")
    return float(m), float(math.sqrt(max(float(v), 0.0)))


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))


def _progress_rate(m: TrajModel, geom: ViewGeometry, t: float) -> float:
    """d(progress)/dt of the model at time t (1/s)."""
    h = 1e-3
    p = geom.progress(m.eval([t - h, t + h]))
    return float((p[1] - p[0]) / (2 * h))


def _cross_time(m: TrajModel, geom: ViewGeometry, target: float, t_lo: float, t_hi: float) -> Optional[float]:
    """First time in [t_lo, t_hi] where the model's progress crosses `target`."""
    ts = np.linspace(t_lo, t_hi, 401)
    d = m.denom(ts)
    ok = d > 0.02
    if not ok.any():
        return None
    p = geom.progress(m.eval(ts)) - target
    p[~ok] = np.nan
    sgn = np.sign(p)
    for i in range(len(ts) - 1):
        if np.isfinite(sgn[i]) and np.isfinite(sgn[i + 1]) and sgn[i] != sgn[i + 1]:
            a, b = ts[i], ts[i + 1]
            pa = p[i]
            for _ in range(40):
                mid = 0.5 * (a + b)
                pm = float(geom.progress(m.eval([mid]))[0] - target)
                if np.sign(pm) == np.sign(pa):
                    a, pa = mid, pm
                else:
                    b = mid
            return 0.5 * (a + b)
    return None


def _time_sigma(m: TrajModel, geom: ViewGeometry, t: float) -> float:
    """Timing uncertainty of a progress-crossing from the model's position covariance."""
    rate = abs(_progress_rate(m, geom, t)) * geom.length_sq       # square px / s along the axis
    if rate <= 1e-6:
        return float("inf")
    return float(m.pos_sigma([t])[0]) / rate


class Segmenter:
    """Classify finalized tracks.  Needs the candidate store for contact / stop-point checks."""

    def __init__(self, geom: ViewGeometry, opts: FinderOptions, sample_dt: float, frame_dt: float,
                 interlaced: bool, store: CandStore, quality_of=None, view: str = "side"):
        self.geom = geom
        self.opts = opts
        self.dt = sample_dt
        self.frame_dt = frame_dt
        self.interlaced = interlaced
        self.rs = 2 if interlaced else 1
        self.store = store
        self.quality_of = quality_of or (lambda s0, s1: 1.0)
        self.s_range = (None, None)          # (first, last) processed sample, set by the finder

    # -------------------------------------------------------------- main
    def classify(self, tr: Track, manual: bool = False) -> PitchTrack:
        g, o = self.geom, self.opts
        A = tr.arrays()
        inl = np.array([p.inlier for p in tr.sorted()], bool)
        pts = [TrackPoint(s=p.s, t=p.t, x=p.x, y=p.y, score=p.z, sigma=p.sigma, inlier=p.inlier,
                          manual=p.src == "manual") for p in tr.sorted()]
        pt = PitchTrack(track_id=tr.tid, points=pts)
        diag: dict = {"n_points": len(pts), "n_inliers": int(inl.sum()),
                      "n_pass2": int(sum(1 for p in tr.sorted() if p.src == "pass2")),
                      "seed_score": float(tr.seed_score),
                      "delivery_cue": bool(tr.info.get("delivery_cue", False))}
        pt.diagnostics = diag
        m = tr.model
        if m is not None:
            pt.model = m.to_dict()
            diag["model_kind"] = m.kind
            diag["rms_px"] = float(m.rms)
            if g.view == "behind":
                pole = m.pole_time()
                diag["pole_time"] = pole
        n_inl = int(inl.sum())
        if len(pts):
            first, last = pts[0], pts[-1]
            ip = [p for p in pts if p.inlier] or pts
            diag.update(t_first_obs=ip[0].t, t_last_obs=ip[-1].t, s_first_obs=ip[0].s, s_last_obs=ip[-1].s)
            diag["attached"] = [bool(a) for a in A["att"]]
        if n_inl < o.min_inliers or m is None:
            return self._reject(pt, "TOO_SHORT")
        Ai = tr.arrays(inliers_only=True)
        xy = np.stack([Ai["x"], Ai["y"]], 1)
        prog = g.progress(xy)
        diag["progress_first"] = float(prog[0])
        diag["progress_last"] = float(prog[-1])
        slope = float(np.polyfit(Ai["t"] - Ai["t"].mean(), prog, 1)[0])
        diag["progress_rate"] = slope
        # field order check
        if self.interlaced:
            fo = field_order_check(Ai["t"], Ai["x"], Ai["y"], Ai["k"], self.frame_dt)
            diag["field_order"] = fo
            if fo.get("conflict"):
                pt.flags.append("field_order_conflict")
        # direction / speed
        if slope < 0:
            return self._reject(pt, "THROWBACK")
        end_occ = self._end_occluded(tr, +1)
        start_occ = self._end_occluded(tr, -1)
        dprog = float(prog[-1] - prog[0])
        dur = float(Ai["t"][-1] - Ai["t"][0])
        diag["duration_s"] = dur
        if g.view == "side":
            no_prog = dprog < o.min_progress and not (end_occ or start_occ)
        else:
            # image progress is very non-linear in time for behind views (the ball
            # barely moves early on) and the catcher often hides the end: require
            # clear progress and a minimum duration instead of 35 % of the image path
            no_prog = dprog < o.min_progress_behind or dur < o.min_duration_behind_s
        if no_prog and not manual:
            return self._reject(pt, "NO_PROGRESS")
        if g.view == "side":
            dv = (xy[-1] - xy[0]) * [g.sar, 1.0]
            cosang = float(dv @ g.axis_sq / (np.linalg.norm(dv) + 1e-9))
            ang = math.degrees(math.acos(np.clip(cosang, -1, 1)))
            diag["angle_off_axis_deg"] = ang
            if ang > o.max_off_axis_deg and not manual:
                return self._reject(pt, "OFF_AXIS")
        else:
            lat = g.lateral(xy) / g.length_sq
            diag["lateral_first"] = float(lat[0])
            diag["lateral_last"] = float(lat[-1])
            if (abs(lat[0]) > o.off_axis_rel + max(0.0, prog[0]) * 0.3 or abs(lat[-1]) > o.off_axis_catch + 0.1) \
                    and not manual:
                return self._reject(pt, "OFF_AXIS")
        # leading / trailing outliers (extension into the hand or the glove) are not a
        # poor fit; judge the inlier share between the first and the last inlier
        s_in0, s_in1 = int(Ai["s"][0]), int(Ai["s"][-1])
        n_all = sum(1 for p in pts if s_in0 <= p.s <= s_in1)
        span = s_in1 - s_in0 + 1
        diag["inlier_fraction"] = n_inl / max(1, n_all)
        diag["coverage"] = n_inl / max(1, span)
        if (m.rms > o.max_rms_px or n_inl < 0.5 * n_all or n_inl < 0.4 * span) and not manual:
            return self._reject(pt, "POOR_FIT")
        # speed (side view): along-axis image speed of the model
        v_mph = None
        if g.view == "side":
            tt = np.linspace(Ai["t"][0], Ai["t"][-1], 9)
            rates = np.array([_progress_rate(m, g, t) for t in tt]) * g.length_sq
            v_mph = float(np.median(rates)) / g.scale_rel / MPH
            diag["speed_mph_rough"] = v_mph
            lo, hi = o.side_speed_range
            if not (lo <= v_mph <= hi) and not manual:
                return self._reject(pt, "SPEED_IMPLAUSIBLE")
        # events
        rel = self._release(tr, m, Ai, prog, diag, pt.flags)
        cat, end_kind = self._catch(tr, m, Ai, prog, diag, pt.flags, rel)
        if (rel is not None and cat is not None and g.view == "behind"
                and "release_not_observed" in pt.flags and rel.method == "first_free_field"):
            # the first clean detection is deep in the flight: all we know is
            # catch - (slowest plausible flight) <= release <= first clean detection
            lo, hi = cat.t - g.t_window[1], rel.t
            if hi > lo:
                rel = EventTime(t=0.5 * (lo + hi), sigma=(hi - lo) / math.sqrt(12.0), method="first_free_field",
                                s=self._nearest_s(0.5 * (lo + hi)))
                diag["release_interval"] = [lo, hi]
        pt.release, pt.catch, pt.end_kind = rel, cat, end_kind
        if (rel is None or cat is None) and not manual:
            return self._reject(pt, "NO_EVENTS")
        if rel is not None and cat is not None:
            T = cat.t - rel.t
            diag["flight_time"] = T
            lo, hi = g.t_window
            if (T <= 0 or not (lo <= T <= hi)) and not manual:
                pt.release, pt.catch = rel, cat
                return self._reject(pt, "SPEED_IMPLAUSIBLE")
        # confidence
        pt.confidence = self._confidence(tr, m, Ai, prog, rel, cat, v_mph, diag, end_kind, pt.flags)
        if pt.confidence < o.min_confidence and not manual:
            return self._reject(pt, "LOW_CONFIDENCE")
        if pt.confidence < 0.3:
            pt.flags.append("doubtful")
        elif pt.confidence < 0.6:
            pt.flags.append("review")
        pt.status = "pitch"
        return pt

    def _reject(self, pt: PitchTrack, code: str) -> PitchTrack:
        pt.status = "rejected"
        pt.reject_code = code
        return pt

    # ------------------------------------------------------------- ends
    def _end_occluded(self, tr: Track, direction: int) -> bool:
        """Does the model run into a body blob right after (before) the track end?"""
        m = tr.model
        if m is None:
            return False
        P = tr.sorted()
        s0 = P[-1].s if direction > 0 else P[0].s
        hits = 0
        for j in range(1, 4):
            df = self.store.get(s0 + direction * j)
            if df is None:
                continue
            pred = m.eval([df.t])[0]
            if _in_blob(df, pred, pad=3.0):
                hits += 1
        return hits >= 2

    def _first_clean(self, tr: Track) -> Optional[int]:
        P = [p for p in tr.sorted() if p.inlier]
        o = self.opts
        for i, p in enumerate(P):
            if p.attached or p.z < o.z_strong or p.src != "det":
                continue
            # the next two inliers must exist within a few samples (track continues)
            nxt = P[i + 1:i + 3]
            if len(nxt) >= 2 and nxt[-1].s - p.s <= 6:
                return i
        return None

    def _release(self, tr, m, Ai, prog, diag, flags) -> Optional[EventTime]:
        g, o, dt = self.geom, self.opts, self.dt
        P = [p for p in tr.sorted() if p.inlier]
        t_first = P[0].t
        manual = [p for p in tr.sorted() if p.src == "manual"]
        i_clean = self._first_clean(tr)
        t_clean = P[i_clean].t if i_clean is not None else None
        diag["t_first_clean"] = t_clean
        diag["s_first_clean"] = P[i_clean].s if i_clean is not None else None
        # model crossing of the release point (progress 0)
        span = max(0.6, 2.0 * (Ai["t"][-1] - Ai["t"][0]))
        t_x = _cross_time(m, g, 0.0, t_first - span, Ai["t"][-1])
        sig_x = _time_sigma(m, g, t_x) if t_x is not None else float("inf")
        diag["release_extrapolated_t"] = t_x
        diag["release_extrapolated_sigma"] = sig_x if np.isfinite(sig_x) else None
        if g.view == "side":
            if t_x is None:
                flags.append("release_not_found")
                return None
            v_px = abs(_progress_rate(m, g, t_x)) * g.length_sq          # square px / s
            v_fts = v_px / g.scale_rel if g.scale_rel > 0 else float("nan")
            sig_ext = 1.0 / v_fts if v_fts and np.isfinite(v_fts) and v_fts > 1 else 0.02
            sigma = math.sqrt(min(sig_x, 0.1) ** 2 + sig_ext ** 2)
            diag["release_sigma_components"] = {"model": min(sig_x, 0.1), "extension_1ft": sig_ext}
            if t_clean is not None and t_clean < t_x - 0.5 * dt:
                flags.append("ball_free_before_release_point")
            gap_s = t_first - t_x
            diag["release_extrapolation_s"] = gap_s
            if gap_s > 0.15:
                flags.append("release_far_extrapolated")
            return EventTime(t=float(t_x), sigma=float(sigma), method="extrapolated",
                             s=self._nearest_s(t_x))
        # behind view: first clean detection minus a latency prior
        if t_clean is None:
            # no clean free-flight detection at all: the release is somewhere before the
            # first detection (classify() turns this into the window
            # [catch - slowest flight, first detection]).  The model's backward
            # extrapolation is not used: it is degenerate in this view (math.md 4b).
            flags.append("release_not_observed")
            t_rel = t_first - o.release_latency_mean * dt
            return EventTime(t=float(t_rel), sigma=float(o.release_latency_sigma * dt), method="first_free_field",
                             s=self._nearest_s(t_rel))
        lat_m, lat_s = o.release_latency_mean * dt, o.release_latency_sigma * dt
        t_rel = t_clean - lat_m
        sigma = math.sqrt(lat_s ** 2 + dt * dt / 12.0)
        p_clean = float(g.progress(np.array([[P[i_clean].x, P[i_clean].y]]))[0])
        diag["progress_first_clean"] = p_clean
        if p_clean > 0.15:
            flags.append("release_not_observed")
        # Evidence of the ball before the first clean detection (weak, attached or
        # off-model detections near the release end that run continuously into it):
        # the ball was probably there but merged with the hand -> the release lies
        # anywhere in [t_pre, t_clean] - latency.  Widen sigma to that interval.
        t_pre = self._pre_clean_start(tr, P[i_clean])
        lo = None
        if t_pre is not None and t_clean - t_pre >= 2 * dt:
            flags.append("release_uncertain")
            lo = t_pre - lat_m
            diag["release_pre_clean_t"] = t_pre
        # Delivery-motion cue: the pitcher's arm sweeps through the release point at
        # release (hand speed peaks there), so the last strong burst of motion near the
        # release point marks the release to within a few fields.  It is used only to
        # widen, never to narrow: when the first clean detection comes clearly later
        # than that burst, the detector probably missed the first fields of the flight
        # (the ball barely moves in the image early on), and the release is taken
        # anywhere in [cue - CUE_LEAD_S, first clean detection - latency].
        t_cue = self._delivery_cue_time(P[i_clean].s, t_clean)
        diag["release_cue_t"] = t_cue
        if t_cue is not None and t_clean - t_cue > float(o.extra.get("cue_late_gap_s", CUE_LATE_GAP_S)):
            flags.append("release_detection_late")
            if "release_uncertain" not in flags:
                flags.append("release_uncertain")
            c_lo = t_cue - float(o.extra.get("cue_lead_s", CUE_LEAD_S))
            lo = c_lo if lo is None else min(lo, c_lo)
        if lo is not None:
            hi = t_clean - lat_m
            lo = min(lo, hi)
            t_rel = 0.5 * (lo + hi)
            sigma = math.sqrt(lat_s ** 2 + (hi - lo) ** 2 / 12.0 + dt * dt / 12.0)
            diag["release_interval"] = [lo, hi]
        if t_x is not None and np.isfinite(sig_x):
            # diagnostic only: the image model cannot locate the behind-view release
            # (errors of 8-40 fields with claimed sigmas of 1-2 fields on the synthetic
            # suites; perspective degeneracy, math.md 4b), so it neither moves nor widens it
            diag["release_methods_delta"] = t_x - t_rel
        return EventTime(t=float(t_rel), sigma=float(sigma), method="first_free_field", s=self._nearest_s(t_rel))

    def _delivery_cue_time(self, s_clean: int, t_clean: float) -> Optional[float]:
        """Time of the last strong burst of motion near the release point before (or at)
        the first clean detection (DetFrame.e_rel, foreground fraction around the
        release click), or None when the window has no clear burst.

        Window CUE_WINDOW_S before the first clean detection (+2 samples); a burst is
        a local maximum >= max(median + CUE_K * 1.4826 MAD, median + 0.02, CUE_FRAC *
        window max).  On the synthetic behind views it was 0.4 fields (median) after
        the true release, 83 % within +-3 fields (the renderer's arm peaks at release;
        real arms do too, but bodies add motion, hence the loose use above).
        """
        n_w = int(round(CUE_WINDOW_S / self.dt))
        ts, es = [], []
        for s in range(s_clean - n_w, s_clean + 3):
            df = self.store.get(s)
            e = getattr(df, "e_rel", float("nan")) if df is not None else float("nan")
            if df is not None and e is not None and np.isfinite(e):
                ts.append(float(df.t))
                es.append(float(e))
        if len(es) < 0.6 * (n_w + 3):
            return None
        e = np.asarray(es)
        b = float(np.median(e))
        mad = 1.4826 * float(np.median(np.abs(e - b))) + 1e-4
        thr = max(b + CUE_K * mad, b + 0.02, CUE_FRAC * float(e.max()))
        peaks = [i for i in range(len(e)) if e[i] >= thr and (i == 0 or e[i] >= e[i - 1])
                 and (i == len(e) - 1 or e[i] >= e[i + 1])]
        return ts[peaks[-1]] if peaks else None

    def _pre_clean_start(self, tr, clean) -> Optional[float]:
        """Earliest time of a near-continuous run of detections (any kind) ending at the
        first clean one, near the release end of the corridor (progress <= 0.2)."""
        g, o = self.geom, self.opts
        m = tr.model
        before = [p for p in tr.sorted() if p.s < clean.s and p.z >= o.z_pass2]
        t_pre, s_prev = None, clean.s
        for p in reversed(before):
            if s_prev - p.s > 3:
                break
            prog = float(g.progress(np.array([[p.x, p.y]]))[0])
            # in-hand positions lie behind the release point and off the flight path
            if prog > 0.2 or prog < -0.03:
                break
            if m is not None:
                q = m.eval([p.t])[0]
                d = float(np.hypot((q[0] - p.x) * g.sar, q[1] - p.y))
                if d > max(3.0, 1.5 * float(g.ball_diameter(prog))):
                    break
            t_pre, s_prev = p.t, p.s
        return t_pre

    def _nearest_s(self, t: float) -> Optional[int]:
        """Nearest sample index to time t, clamped to the samples processed so far."""
        s = self._nearest_s_raw(t)
        if s is None:
            return None
        lo, hi = self.s_range
        if lo is not None:
            s = max(int(lo), s)
        if hi is not None:
            s = min(int(hi), s)
        return int(s)

    def _nearest_s_raw(self, t: float) -> Optional[int]:
        best, bd = None, float("inf")
        for s, df in self.store.frames.items():
            d = abs(df.t - t)
            if d < bd:
                best, bd = s, d
        if best is not None and bd <= self.dt:
            return int(best)
        # outside the cached window: estimate from any frame
        if self.store.frames:
            s0, df0 = next(iter(self.store.frames.items()))
            return int(round(s0 + (t - df0.t) / self.dt))
        return None

    def _catch(self, tr, m, Ai, prog, diag, flags, rel=None):
        g, dt = self.geom, self.dt
        P = [p for p in tr.sorted() if p.inlier]
        last = P[-1]
        p_last = float(prog[-1])
        v = m.vel([last.t])[0]
        v_sq = float(np.hypot(v[0] * g.sar, v[1]))             # square px / s
        catch_sig_ft = float(self.opts.extra.get("catch_sigma_ft", C.CATCH_SIGMA_FT))
        # catch in the field interval after the last detection (dt^2/12), plus half a
        # field for a track ending one field early / late (measured on the synthetic
        # suite: RMS 8 ms vs 4.8 ms for dt/sqrt(12) alone)
        lmf = EventTime(t=last.t + dt / 2, sigma=math.sqrt(dt * dt / 12 + (dt / 2) ** 2),
                        method="last_moving_field", s=self._nearest_s(last.t + dt))
        if g.view == "behind":
            trimmed = self._trim_end_gap(m, Ai, P)
            if trimmed is not None:
                # a few inliers after a long gap at the end are not the ball reaching the
                # glove (the ball does not vanish for > END_GAP_S mid-flight and reappear):
                # judge the end of the flight without them
                m, Ai, P, n_drop = trimmed
                flags.append("track_end_gap_trimmed")
                diag["end_gap_dropped"] = n_drop
                last = P[-1]
                p_last = float(g.progress(np.array([[last.x, last.y]]))[0])
                lmf = EventTime(t=last.t + dt / 2, sigma=math.sqrt(dt * dt / 12 + (dt / 2) ** 2),
                                method="last_moving_field", s=self._nearest_s(last.t + dt))
        # contact: a fast departing track
        if self._contact(tr, m, last):
            if g.view != "behind" or self._near_plate_behind(m, P, last, p_last):
                flags.append("contact")
                return lmf, "contact"
            # behind the plate the catcher, the umpire and the swing move fast right where
            # the ball disappears: a bat can only meet the ball at the plate, so a
            # 'departing track' while the ball is still well short of it is not contact
            diag["contact_ignored"] = "ball still far from the plate"
        if g.view == "behind":
            return self._catch_behind(tr, m, Ai, P, last, p_last, diag, flags, rel, lmf)
        # crossing of the catch point (progress 1) by the model
        t_c = _cross_time(m, g, 1.0, P[0].t, last.t + 0.6)
        if t_c is None and p_last < 1.0:
            rate = _progress_rate(m, g, last.t)
            if rate > 1e-6:
                t_c = last.t + (1.0 - p_last) / rate
        diag["catch_point_crossing_t"] = t_c
        T_lo, T_hi = g.t_window
        lo, hi = last.t + dt / 2, last.t + T_hi
        if rel is not None:
            lo = max(lo, rel.t + T_lo)
            hi = max(lo + dt, rel.t + T_hi)
        # side view from here on (behind: _catch_behind)
        v_fts = max(v_sq / max(g.scale_rel, 1e-6), 1.0)
        sig_glove = catch_sig_ft / v_fts                      # glove position along the axis, in time
        max_ext = 0.25

        def prior(kind):
            flags.append("catch_not_observed")
            t_mid = 0.5 * (lo + hi)
            return EventTime(t=float(t_mid), sigma=float((hi - lo) / math.sqrt(12.0)), method="extrapolated",
                             s=self._nearest_s(t_mid)), kind

        if t_c is None:
            if p_last >= 1.0:
                return lmf, "catch"
            flags.append("catch_extrapolated")
            return prior("occluded" if self._end_occluded(tr, +1) else "lost")
        diag["catch_remaining_samples"] = (t_c - last.t) / dt
        if t_c <= last.t + dt:
            # the catch point is reached within the next field interval (or was passed)
            if t_c < last.t and p_last > 1.0 + 2.0 / max(1.0, g.length_ft):
                # > 2 ft past the catch point: the last detections are most likely
                # glove / catcher motion on the extrapolated path
                flags.append("track_past_catch_point")
                sigma = math.sqrt(min(_time_sigma(m, g, t_c), 1.0) ** 2 + sig_glove ** 2)
                return EventTime(t=float(t_c), sigma=float(sigma), method="extrapolated",
                                 s=self._nearest_s(t_c)), "catch"
            return lmf, "catch"
        # The ball should have been seen again before reaching the catch point.  Either the
        # glove was in front of the clicked point (caught right after the last detection)
        # or the ball was hidden / lost: the catch lies in [t_last, t_c] (+ glove spread).
        stop = self._stop_point(tr, m, last)
        if stop is not None:
            t_s, gpt = stop
            diag["glove_point"] = [float(gpt[0]), float(gpt[1])]
            return EventTime(t=float(t_s), sigma=dt / math.sqrt(12), method="extrapolated",
                             s=last.s), "catch"
        end_kind = "occluded" if self._end_occluded(tr, +1) else "lost"
        flags.append("catch_extrapolated")
        if t_c - last.t > max_ext or not (lo <= t_c <= hi):
            return prior(end_kind)
        a, b = last.t + dt / 2, t_c
        t_mid = 0.5 * (a + b)
        sig_m = min(_time_sigma(m, g, t_c), 1.0)
        sigma = math.sqrt((b - a) ** 2 / 12.0 + sig_m ** 2 + sig_glove ** 2 + dt * dt / 12.0)
        diag["catch_extrapolation_s"] = t_c - last.t
        if t_c - last.t > 0.15:
            flags.append("catch_far_extrapolated")
        return EventTime(t=float(t_mid), sigma=float(sigma), method="extrapolated",
                         s=self._nearest_s(t_mid)), end_kind

    def _catch_behind(self, tr, m, Ai, P, last, p_last, diag, flags, rel, lmf):
        """Behind-view catch time: the fitted perspective track's crossing of the catch click.

        The ball usually disappears behind the catcher / umpire a few fields
        before it reaches the glove.  The rational image model is a good
        short-range predictor of where the ball goes; *when* it reaches the
        glove is the time its image path crosses the catch click's progress
        (the line through the click, perpendicular to release->catch), up to
        the per-pitch glove position (the click is one catch, the glove ends
        wherever the pitch goes).  So, with t_x that crossing time:

            catch ~ N(t_x, s_glove^2 + s_model^2)  truncated to [t_last, rel + T_max]

        s_glove = glove spread (GLOVE_SPREAD_FT in the image plane, times the
        px/ft at the catch: clicked plate width, else the geometry guess
        x SCALE_GUESS_FACTOR) / the model's image speed along the axis at t_x;
        s_model = the model's position sigma at t_x (covariance) / that speed,
        x CATCH_MODEL_INFLATE (the model is only approximately right far from
        the data).  The truncation encodes that the ball was still in flight
        at the last detection.  Measured on synthetic behind views (GT field
        positions, docs in the report): unbiased, RMS 20-40 ms for 3-9 hidden
        fields.  Longer extrapolations (> CATCH_MAX_EXTRAP_S) or a resulting
        sigma > CATCH_MAX_SIGMA_S are not trusted: uninformative window,
        flag 'catch_not_observed'.  A depth-ratio method (rubber/plate widths
        + the model's denominator) was tried and is not used: the
        denominator is poorly determined (perspective degeneracy, math.md 4b),
        RMS 60 ms even with the true depth ratio.
        """
        g, o, dt = self.geom, self.opts, self.dt
        ex = o.extra
        glove_ft = float(ex.get("glove_spread_ft", GLOVE_SPREAD_FT))
        inflate = float(ex.get("catch_model_inflate", CATCH_MODEL_INFLATE))
        max_ext = float(ex.get("catch_max_extrap_s", CATCH_MAX_EXTRAP_S))
        max_sig = float(ex.get("catch_max_sigma_s", CATCH_MAX_SIGMA_S))
        T_lo, T_hi = g.t_window
        lo_w, hi_w = last.t, last.t + T_hi
        if rel is not None:
            lo_w = max(lo_w, rel.t + T_lo)
            hi_w = max(lo_w + dt, rel.t + T_hi)
        end_kind = "occluded" if self._end_occluded(tr, +1) else "lost"

        def prior(reason, kind):
            flags.append("catch_not_observed")
            diag["catch_not_observed_reason"] = reason
            a = max(last.t + dt / 2, lo_w)
            b = max(a + dt, hi_w)
            t_mid = 0.5 * (a + b)
            return EventTime(t=float(t_mid), sigma=float((b - a) / math.sqrt(12.0)), method="extrapolated",
                             s=self._nearest_s(t_mid)), kind

        t_x = _cross_time(m, g, 1.0, P[0].t, last.t + CATCH_SEARCH_S)
        diag["catch_point_crossing_t"] = t_x
        if t_x is None:
            if p_last >= o.catch_near_progress - 0.1:
                # the track reached the glove area and the model flattens out before the click
                return lmf, "catch"
            flags.append("catch_extrapolated")
            return prior("the fitted track does not reach the catch point", end_kind)
        diag["catch_remaining_samples"] = (t_x - last.t) / dt
        hidden = t_x > last.t + dt
        if hidden:
            # the extrapolated image path must keep moving toward the catch point
            tt = np.linspace(last.t, t_x, 25)
            pp = g.progress(m.eval(tt))
            if np.any(np.diff(pp) < -1e-4):
                flags.append("catch_extrapolated")
                return prior("the extrapolated track turns back before the catch point", end_kind)
        rate = abs(_progress_rate(m, g, t_x)) * g.length_sq          # square px / s along the axis
        guessed = bool(g.extra.get("scale_catch_guess"))
        glove_px = glove_ft * g.scale_catch * (SCALE_GUESS_FACTOR if guessed else 1.0)
        s_glove = glove_px / max(rate, 1e-6)
        s_model = inflate * float(m.pos_sigma([t_x])[0]) / max(rate, 1e-6)
        s_jack = self._jackknife_crossing(m, Ai, t_x, last) if hidden else 0.0
        if s_jack is None:
            flags.append("catch_extrapolated")
            return prior("the catch-point crossing is not stable when the last detections are left out",
                         end_kind)
        s_ext = max(s_model, s_jack)
        s_x = math.hypot(s_glove, s_ext)
        diag["catch_sigma_components"] = {"glove": s_glove, "model": s_model, "jackknife": s_jack}
        if hidden:
            flags.append("catch_extrapolated")
            diag["catch_extrapolation_s"] = t_x - last.t
            if t_x - last.t > max_ext:
                return prior(f"catch {1e3 * (t_x - last.t):.0f} ms after the last detection "
                             f"(> {1e3 * max_ext:.0f} ms)", end_kind)
            if s_x > max_sig:
                return prior(f"extrapolated catch too uncertain (+-{1e3 * s_x:.0f} ms)", end_kind)
            if t_x - last.t > 0.15:
                flags.append("catch_far_extrapolated")
        if not hidden:
            # the track reached the catch click's progress while still visible: the ball
            # stopped (or vanished behind the catcher) in the field after the last
            # detection, give or take the glove spread (capped: the end was seen)
            sd = math.hypot(lmf.sigma, min(s_glove, 3 * dt))
            return EventTime(t=lmf.t, sigma=float(sd), method=lmf.method, s=lmf.s), "catch"
        mean, sd = truncated_normal(t_x, s_x, lo_w, hi_w)
        if not (math.isfinite(mean) and math.isfinite(sd)):
            return prior("catch-point crossing outside the plausible flight-time window", end_kind)
        if mean < lmf.t:
            mean = lmf.t
        sd = max(sd, lmf.sigma)
        return EventTime(t=float(mean), sigma=float(sd), method="extrapolated", s=self._nearest_s(mean)), end_kind

    def _near_plate_behind(self, m, P, last, p_last) -> bool:
        """Behind view: did the ball get to the plate area (the catch-point crossing at most
        CONTACT_MAX_REMAINING_S after the last detection)?"""
        if p_last >= 1.0:
            return True
        t_x = _cross_time(m, self.geom, 1.0, P[0].t, last.t + CATCH_SEARCH_S)
        return t_x is not None and t_x - last.t <= CONTACT_MAX_REMAINING_S

    def _trim_end_gap(self, m, Ai, P):
        """(model, Ai, P, n_dropped) without the trailing inliers that follow a gap longer
        than END_GAP_S, when at most END_GAP_MAX_PTS of them follow it; None otherwise."""
        g = self.geom
        t = np.asarray(Ai["t"])
        if len(t) < 8:
            return None
        big = np.nonzero(np.diff(t) > END_GAP_S)[0]
        if not len(big):
            return None
        j = int(big[-1]) + 1                      # first index after the last long gap
        n_drop = len(t) - j
        if n_drop > END_GAP_MAX_PTS or j < 6:
            return None
        A2 = {k: (v[:j] if isinstance(v, np.ndarray) else list(v)[:j]) for k, v in Ai.items()}
        d = float(np.median(g.ball_diameter(g.progress(np.stack([A2["x"], A2["y"]], 1)))))
        m2 = fit_model(A2["t"], A2["x"], A2["y"], A2["sigma"], kind=m.kind, view=g.view,
                       tol_x=max(1.5, 0.3 * d), tol_y=max(2.0, 0.3 * d), ransac_iters=60)
        if m2 is None:
            return None
        t_keep = float(A2["t"][-1])
        P2 = [p for p in P if p.t <= t_keep + 1e-9]
        return m2, A2, P2, n_drop

    def _jackknife_crossing(self, m, Ai, t_x, last, drop: int = JACKKNIFE_DROP) -> Optional[float]:
        """|change| of the catch-point crossing when the model is refitted without the last
        `drop` inliers (extrapolation stability; the covariance misses end-point junk and
        model error).  None when the refit does not reach the catch point at all."""
        g = self.geom
        n = len(Ai["t"])
        if n < max(10, drop + 8):
            return 0.0
        k = n - drop
        d = float(np.median(g.ball_diameter(g.progress(np.stack([Ai["x"][:k], Ai["y"][:k]], 1)))))
        m2 = fit_model(Ai["t"][:k], Ai["x"][:k], Ai["y"][:k], Ai["sigma"][:k], kind=m.kind, view=g.view,
                       tol_x=max(1.5, 0.3 * d), tol_y=max(2.0, 0.3 * d), ransac_iters=60)
        if m2 is None:
            return 0.0
        t2 = _cross_time(m2, g, 1.0, float(Ai["t"][0]), last.t + CATCH_SEARCH_S)
        if t2 is None:
            return None
        return abs(float(t2) - float(t_x))

    def _stop_point(self, tr, m, last):
        """Ball seen in the next field short of the prediction (stopped in the glove)."""
        g = self.geom
        for j in (1, 2):
            df = self.store.get(last.s + j)
            if df is None:
                return None
            pred = m.eval([df.t])[0]
            p0 = np.array([last.x, last.y])
            seg = (pred - p0) * [g.sar, 1.0]
            Ls = float(np.linalg.norm(seg))
            if Ls < 2.0:
                return None
            u = seg / Ls
            C_ = df.cands
            best = None
            for c in C_:
                q = (np.array([c["x"], c["y"]]) - p0) * [g.sar, 1.0]
                along = float(q @ u)
                perp = abs(float(q @ np.array([-u[1], u[0]])))
                if -2.0 <= along <= 0.6 * Ls and perp <= 3.0 + 0.1 * Ls and c["z"] >= 3.0:
                    if best is None or c["z"] > best[0]:
                        best = (float(c["z"]), np.array([c["x"], c["y"]]))
            if best is not None:
                gpt = best[1]
                ts = np.linspace(last.t, last.t + j * self.dt, 81)
                P = m.eval(ts)
                d = np.hypot((P[:, 0] - gpt[0]) * g.sar, P[:, 1] - gpt[1])
                return float(ts[int(np.argmin(d))]), gpt
        return None

    def _contact(self, tr, m, last) -> bool:
        """A batted ball: >= 3 consistent, non-attached, fast detections leaving the end point
        within `contact_gap` samples in a clearly different direction."""
        g, o = self.geom, self.opts
        v_pitch = m.vel([last.t])[0] * [g.sar, 1.0]
        n_pitch = np.linalg.norm(v_pitch) + 1e-9
        p0 = np.array([last.x, last.y])
        v_lo = o.min_speed_mph * MPH * g.scale_catch * self.dt

        def cands(s):
            f = self.store.get(s)
            if f is None or not len(f.cands):
                return []
            c = f.cands
            return [r for r in c if not r["attached"] and r["z"] >= o.z_strong]

        for g1 in range(1, o.contact_gap + 1):
            for c1 in cands(last.s + g1):
                q1 = np.array([c1["x"], c1["y"]])
                if np.hypot((q1[0] - p0[0]) * g.sar, q1[1] - p0[1]) > 10.0 + 1.5 * v_lo * g1:
                    continue
                for c2 in cands(last.s + g1 + 1):
                    q2 = np.array([c2["x"], c2["y"]])
                    vv = (q2 - q1) * [g.sar, 1.0]
                    sp = float(np.linalg.norm(vv))
                    if sp < v_lo or float(vv @ v_pitch) / (sp * n_pitch) > 0.5:
                        continue
                    if float((q2 - p0) @ (q2 - q1)) <= 0:
                        continue
                    pred = q2 + (q2 - q1)
                    for c3 in cands(last.s + g1 + 2):
                        q3 = np.array([c3["x"], c3["y"]])
                        err = np.hypot((q3[0] - pred[0]) * g.sar, q3[1] - pred[1])
                        if err <= max(2.0, 0.25 * sp) and c3["sign"] == c2["sign"] == c1["sign"]:
                            return True
        return False

    # ------------------------------------------------------- confidence
    def _confidence(self, tr, m, Ai, prog, rel, cat, v_mph, diag, end_kind="catch", flags=()) -> float:
        g, o, dt = self.geom, self.opts, self.dt
        f_snr = _logistic(float(np.median(Ai["z"])) - 4.0)
        unobserved = "catch_not_observed" in flags
        # behind the plate the end of the flight is normally hidden by the catcher: the
        # hidden fields after the last detection are expected, not missing detections
        hidden_end = g.view == "behind" and cat is not None and cat.method == "extrapolated"
        if rel is not None and cat is not None and cat.t > rel.t and not (unobserved or hidden_end):
            n_exp = (cat.t - rel.t) / dt + 1
        elif rel is not None:
            n_exp = (Ai["t"][-1] - rel.t) / dt + 1
        else:
            n_exp = len(Ai["t"])
        f_cov = min(1.0, len(Ai["t"]) / max(1.0, 0.6 * n_exp))
        f_fit = math.exp(-max(0.0, m.rms - 1.0))
        d_rel = max(0.0, float(prog[0]))
        d_c = max(0.0, 1.0 - float(prog[-1]))
        f_rel = math.exp(-(d_rel / 0.35) ** 2)
        if g.view == "side":
            f_cat = math.exp(-(d_c / 0.35) ** 2)
        else:   # image progress is far from linear in time behind the plate
            f_cat = 1.0 if prog[-1] >= 0.75 else 0.5
        if end_kind == "occluded" or unobserved:
            # the flag tells the user; the track itself may still be a clean pitch
            f_cat = max(f_cat, 0.5)
        f_ends = f_rel * f_cat
        if v_mph is not None:
            f_speed = 1.0 if 30 <= v_mph <= 105 else math.exp(-((min(abs(v_mph - 30), abs(v_mph - 105))) / 10) ** 2)
        else:
            f_speed = 1.0
            if rel is not None and cat is not None:
                T = cat.t - rel.t
                lo, hi = g.t_window
                f_speed = 1.0 if lo <= T <= hi else 0.3
        s0, s1 = int(Ai["s"][0]), int(Ai["s"][-1])
        f_mc = float(min(1.0, max(0.0, self.quality_of(s0, s1)) / 0.15))   # median q, see finder
        f_cue = 1.0 if diag.get("delivery_cue") else 0.8
        conf = f_snr * f_cov * f_fit * f_ends * f_speed * f_mc * f_cue
        diag["confidence_factors"] = {"snr": f_snr, "coverage": f_cov, "fit": f_fit, "ends": f_ends,
                                      "speed": f_speed, "mc": f_mc, "cue": f_cue}
        return float(np.clip(conf, 0.0, 1.0))
