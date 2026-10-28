"""Turn a fitted track into a `PitchTrack`: reject codes, events, confidence.

Reject codes (cv.md section 5): TOO_SHORT, NO_PROGRESS, THROWBACK,
OFF_AXIS, SPEED_IMPLAUSIBLE, POOR_FIT (rms > 2 px, or < 60 % inliers, or
inliers cover < 40 % of the spanned samples), DUPLICATE (finder, across
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
    rest of the track.  The model's backward extrapolation to s = 0 is kept
    in diagnostics; if it is precise and disagrees by > 2 sigma the flag
    'release_methods_disagree' is set and sigma is inflated to cover both.
  If the first clean detection is far into the flight (behind view) the
    flag 'release_not_observed' is set.
catch
  'last_moving_field': the track reaches the glove zone (progress >=
    catch_near_progress) and stops: t = t_last + dt / 2 (catch somewhere in
    the next field interval), sigma = dt / sqrt(12).
  'extrapolated' (stop point): if the ball is found again in the next
    field nearer than predicted (stopped in the glove), t is the time the
    model reaches that point, clipped to [t_last, t_last + dt], sigma 3 ms.
  'extrapolated' (occluded / lost end): the model is extrapolated to the
    catch progress (s = 1); sigma combines the model uncertainty and the
    glove-position uncertainty (catch_sigma_ft / v side, 0.15 |Pc - Prel|
    in the image for behind views).  end_kind 'occluded' or 'lost'.
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
from pitchspeed.vision.model import TrajModel
from pitchspeed.vision.options import FinderOptions
from pitchspeed.vision.track import CandStore, Track, _in_blob, _zmap_peak, field_order_check

MPH = C.MPH_TO_FTPS


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
        n_all = len(pts)
        span = int(Ai["s"][-1] - Ai["s"][0] + 1)
        diag["inlier_fraction"] = n_inl / max(1, n_all)
        diag["coverage"] = n_inl / max(1, span)
        if (m.rms > o.max_rms_px or n_inl < 0.6 * n_all or n_inl < 0.4 * span) and not manual:
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
            flags.append("release_not_observed")
            if t_x is not None and np.isfinite(sig_x):
                return EventTime(t=float(t_x), sigma=float(max(sig_x, 2 * dt)), method="extrapolated",
                                 s=self._nearest_s(t_x))
            return None
        lat_m, lat_s = o.release_latency_mean * dt, o.release_latency_sigma * dt
        t_rel = t_clean - lat_m
        sigma = math.sqrt(lat_s ** 2 + dt * dt / 12.0)
        p_clean = float(g.progress(np.array([[P[i_clean].x, P[i_clean].y]]))[0])
        diag["progress_first_clean"] = p_clean
        if p_clean > 0.15:
            flags.append("release_not_observed")
        if t_x is not None and np.isfinite(sig_x) and sig_x < 2 * dt:
            delta = t_x - t_rel
            diag["release_methods_delta"] = delta
            if abs(delta) > 2 * math.sqrt(sigma ** 2 + sig_x ** 2):
                flags.append("release_methods_disagree")
                sigma = math.sqrt(sigma ** 2 + (delta / 2) ** 2)
        return EventTime(t=float(t_rel), sigma=float(sigma), method="first_free_field", s=self._nearest_s(t_rel))

    def _nearest_s(self, t: float) -> Optional[int]:
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
        g, o, dt = self.geom, self.opts, self.dt
        P = [p for p in tr.sorted() if p.inlier]
        last = P[-1]
        p_last = float(prog[-1])
        v = m.vel([last.t])[0]
        v_sq = float(np.hypot(v[0] * g.sar, v[1]))             # square px / s
        catch_sig_ft = float(self.opts.extra.get("catch_sigma_ft", C.CATCH_SIGMA_FT))
        lmf = EventTime(t=last.t + dt / 2, sigma=dt / math.sqrt(12), method="last_moving_field", s=last.s + 1)
        # contact: a fast departing track
        if self._contact(tr, m, last):
            flags.append("contact")
            return lmf, "contact"
        # crossing of the catch point (progress 1) by the model
        t_c = _cross_time(m, g, 1.0, P[0].t, last.t + 0.6)
        if t_c is None and p_last < 1.0:
            rate = _progress_rate(m, g, last.t)
            if rate > 1e-6:
                t_c = last.t + (1.0 - p_last) / rate
        diag["catch_point_crossing_t"] = t_c
        stop = self._stop_point(tr, m, last)
        if stop is not None and p_last >= 0.6:
            t_s, gpt = stop
            diag["glove_point"] = [float(gpt[0]), float(gpt[1])]
            return EventTime(t=float(t_s), sigma=dt / math.sqrt(12), method="extrapolated", s=last.s), "catch"
        T_lo, T_hi = g.t_window
        lo, hi = last.t + dt / 2, last.t + T_hi
        if rel is not None:
            lo = max(lo, rel.t + T_lo)
            hi = max(lo + dt, rel.t + T_hi)

        def prior(kind):
            flags.append("catch_not_observed")
            t_mid = 0.5 * (lo + hi)
            return EventTime(t=float(t_mid), sigma=float((hi - lo) / math.sqrt(12.0)), method="extrapolated",
                             s=self._nearest_s(t_mid)), kind

        def extrap(t_x, sig_g, kind):
            sig_m = _time_sigma(m, g, t_x)
            sigma = math.sqrt(min(sig_m, 1.0) ** 2 + min(sig_g, 1.0) ** 2 + dt * dt / 12)
            diag["catch_extrapolation_s"] = t_x - last.t
            if t_x - last.t > 0.15:
                flags.append("catch_far_extrapolated")
            return EventTime(t=float(t_x), sigma=float(sigma), method="extrapolated", s=self._nearest_s(t_x)), kind

        if g.view == "side":
            L_ft = max(1.0, g.length_ft)
            v_fts = max(v_sq / max(g.scale_rel, 1e-6), 1.0)
            if p_last > 1.0 + 2.0 / L_ft and t_c is not None and t_c < last.t:
                # the track ran > 2 ft past the catch point: the last detections are most
                # likely glove / catcher motion on the extrapolated path
                flags.append("track_past_catch_point")
                return extrap(t_c, catch_sig_ft / v_fts, "catch")
            if (1.0 - p_last) * L_ft <= 1.5:
                return lmf, "catch"
            end_kind = "occluded" if self._end_occluded(tr, +1) else "lost"
            flags.append("catch_extrapolated")
            if t_c is None or not (lo <= t_c <= hi) or t_c - last.t > 0.25:
                return prior(end_kind)
            return extrap(t_c, catch_sig_ft / v_fts, end_kind)
        # behind view: image progress is non-linear; judge in samples of remaining motion
        if t_c is None:
            if p_last >= o.catch_near_progress - 0.1:
                return lmf, "catch"
            flags.append("catch_extrapolated")
            return prior("occluded" if self._end_occluded(tr, +1) else "lost")
        n_rem = (t_c - last.t) / dt
        diag["catch_remaining_samples"] = n_rem
        if n_rem <= 1.0:
            return lmf, "catch"
        end_kind = "occluded" if self._end_occluded(tr, +1) else "lost"
        flags.append("catch_extrapolated")
        # long extrapolations are degenerate behind the plate (perspective scale / time
        # ambiguity, math.md 4b); a few samples toward the clicked glove point are fine
        if n_rem > 6.0 or not (lo <= t_c <= hi):
            return prior(end_kind)
        rate = abs(_progress_rate(m, g, t_c))
        sig_g = max(0.15 / max(rate, 1e-6), 0.35 * (t_c - last.t))
        return extrap(t_c, sig_g, end_kind)

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
        if rel is not None and cat is not None and cat.t > rel.t and not unobserved:
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
        f_mc = float(min(1.0, max(0.0, self.quality_of(s0, s1)) / 0.15))
        f_cue = 1.0 if diag.get("delivery_cue") else 0.8
        conf = f_snr * f_cov * f_fit * f_ends * f_speed * f_mc * f_cue
        diag["confidence_factors"] = {"snr": f_snr, "coverage": f_cov, "fit": f_fit, "ends": f_ends,
                                      "speed": f_speed, "mc": f_mc, "cue": f_cue}
        return float(np.clip(conf, 0.0, 1.0))
