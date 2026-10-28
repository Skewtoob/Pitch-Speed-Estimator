"""PitchFinder: streaming pitch detection over the samples of one camera setup.

    finder = PitchFinder(calib, scan, sample_dt, sample_shape, ref_frame, opts)
    for sample in source.iter_samples(s0, s1):
        finder.process(sample)
    result = finder.finish()          # FinderResult

Per sample: register to the calibration reference frame (`stabilize`),
push into the detector ring buffer (`detect`, ~10 samples latency), store
candidates + z maps in a bounded ring (`track.CandStore`).  Every
`track_every` samples the tracker extends open tracks, seeds new ones in the
fresh part of the window, and finalizes tracks that ended more than
`finalize_lag` samples ago (merge, pass-2 re-detection, refit; behind view:
trim / regrow along the global model / trim; then
`segment.Segmenter.classify` and, behind view, the plausibility guard
`_behind_past_release_check`).  Memory is bounded by the ring sizes, except
for the per-sample timeline (a few floats per sample) and the transforms.

Coordinates: candidate / track positions are reference-frame pixel
coordinates (stored pixels, *frame* rows) -- see `pitchspeed.types`.

Release latency prior (behind view, 'first_free_field'): default
1.0 +- 1.5 samples between the true release and the first clean detection.
On 10 synthetic behind-view pitches where the ball was seen right after
release the latency was -1.7 .. +3.5 fields (mean ~1.0, sd ~1.9; negative
= ball visible in the hand); when the ball stays merged with the hand the
first clean detection can come 10+ fields late, which `segment` widens
via 'release_uncertain' but cannot always detect.  Real tapes may differ;
revisit with labelled footage.  See `segment` for the exact event
semantics.

Rejected tracks keep their reject code; tracks with fewer than 6 points
that never looked like a ball flight (seed score < 40) are only counted
(`diagnostics['n_discarded_short']`) and, with `debug_dir`, written to
tracks.jsonl.  "Possible missed pitch" windows come from a delivery-energy
cue (activity near the release point followed by activity near the catch
point 0.3-1.3 s later) without an accepted track, from rejected tracks that
look like partial pitches, from DUPLICATE tracks seen mostly outside the
flight of the pitch they lost to, and from stretches where stabilization
failed.

Duplicates: only candidates whose observations overlap (or nearly meet) in
time are one flight; the one that observed most of it wins (see
`PitchFinder._resolve_duplicates`).
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from pitchspeed.calib.model import Calibration
from pitchspeed.types import EventTime, PitchTrack, Sample, ScanInfo, TrackPoint, to_jsonable
from pitchspeed.vision import geometry as G
from pitchspeed.vision.detect import DetFrame, Detector
from pitchspeed.vision.options import FinderOptions
from pitchspeed.vision.model import model_from_dict
from pitchspeed.vision.segment import F_FIELD_ORDER, F_FIELD_ORDER_CONFLICT, Segmenter
from pitchspeed.vision.stabilize import Stabilizer, StabilizerOptions
from pitchspeed.vision.track import (CandStore, Pt, Track, build_track, continue_track, find_seeds, fit_track,
                                     pass2, regrow, trim_sparse_ends, try_merge)

log = logging.getLogger("pitchspeed.vision")

__all__ = ["FinderOptions", "FinderResult", "PitchFinder", "track_between"]

PARTIAL_CODES = ("TOO_SHORT", "NO_PROGRESS", "POOR_FIT", "SPEED_IMPLAUSIBLE", "OFF_AXIS")


@dataclass
class FinderResult:
    tracks: list                       # PitchTrack, status 'pitch', with release/catch EventTimes
    rejected: list                     # PitchTrack, status 'rejected', reject_code set
    possible_missed: list              # {s_start, s_end, t_start, t_end, reason}
    transforms: np.ndarray             # (n, 2, 3) sample frame px -> reference px
    transform_quality: np.ndarray      # (n,) 0..1
    sample_index: np.ndarray           # (n,) sample index s of each row
    diagnostics: dict = field(default_factory=dict)


@dataclass
class _Record:
    track: Track
    pt: PitchTrack
    strobe: Optional[np.ndarray] = None


def _add_flags(pt, *flags) -> None:
    for f in flags:
        if f not in pt.flags:
            pt.flags.append(f)


def _inliers(pt) -> list:
    return [p for p in pt.points if p.inlier]


def flight_score(pt) -> float:
    """How much of a pitch flight a candidate observed (duplicate arbitration).

    inliers x image progress covered (first -> last inlier, at least 0.02)
    x log(1 + median detection score).  Junk near the release point (wind-up,
    glove) covers little progress; fragments cover part of the flight; the
    ball's weaker echoes (parallel low-contrast tracks next to it, common on
    the synthetic behind views) lose on contrast.  Not the confidence: it
    rewards late fragments (see _resolve_duplicates).
    """
    P = _inliers(pt)
    if not P:
        return 0.0
    d = pt.diagnostics or {}
    span = float(d.get("progress_last", 0.0)) - float(d.get("progress_first", 0.0))
    z = float(np.median([max(0.0, float(p.score)) for p in P]))
    return float(len(P) * max(span, 0.02) * math.log1p(z))


def same_flight(a, b, merge_gap: int) -> bool:
    """Inlier observation spans overlap or are at most merge_gap samples apart."""
    A, B = _inliers(a), _inliers(b)
    if not A or not B:
        return False
    gap = max(A[0].s, B[0].s) - min(A[-1].s, B[-1].s)
    return gap <= merge_gap


SAME_PATH_MIN_PX = 4.0     # same_path tolerance: max(this, expected ball diameter)


def _path_mismatch(a, b, geom, dt: float) -> Optional[float]:
    """Median over b's inliers inside a's span (+-2 samples) of |a's model - b| / tolerance."""
    A, B = _inliers(a), _inliers(b)
    if not A or not a.model:
        return None
    t0, t1 = A[0].t - 2 * dt, A[-1].t + 2 * dt
    Q = [p for p in B if t0 <= p.t <= t1]
    if len(Q) < 3:
        return None
    try:
        xy = model_from_dict(a.model).eval([p.t for p in Q])
    except Exception:                      # malformed model dict: no evidence
        return None
    obs = np.array([[p.x, p.y] for p in Q], float)
    d = np.hypot((xy[:, 0] - obs[:, 0]) * geom.sar, xy[:, 1] - obs[:, 1])
    tol = np.maximum(SAME_PATH_MIN_PX, geom.ball_diameter(geom.progress(obs)))
    return float(np.median(d / tol))


def same_path(a, b, geom, dt: float) -> bool:
    """Do two time-overlapping candidates follow the same image path (the same object)?

    Each model must predict the other's inlier points where both have data
    (>= 3 points) to within max(SAME_PATH_MIN_PX, ball diameter), median.
    Fragments built from the same ball detections give ~0.1, the ball's
    parallel echoes / other objects >= 2 (synthetic behind suites).
    """
    m = [v for v in (_path_mismatch(a, b, geom, dt), _path_mismatch(b, a, geom, dt)) if v is not None]
    return bool(m) and max(m) <= 1.0


class PitchFinder:
    """Streaming pitch finder for one calibration segment (see module doc)."""

    def __init__(self, calib: Calibration, scan: ScanInfo, sample_dt: float, sample_shape: tuple,
                 reference_frame: np.ndarray, opts: Optional[FinderOptions] = None):
        self.calib = calib
        self.scan = scan
        self.opts = opts if opts is not None else FinderOptions()
        if isinstance(self.opts, dict):
            self.opts = FinderOptions.from_dict(self.opts)
        self.dt = float(sample_dt)
        self.interlaced = bool(scan.interlaced)
        self.rs = 2 if self.interlaced else 1
        self.frame_dt = self.dt * 2 if self.interlaced else self.dt
        self.shape = tuple(int(v) for v in sample_shape)
        ref = np.asarray(reference_frame)
        if ref.ndim == 3:
            import cv2
            ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        rows = self.shape[0] * self.rs
        ref = np.ascontiguousarray(ref[:rows])
        self.ref = ref
        self.geom = G.view_geometry(calib, self.dt)
        self.masks = [tuple(float(v) for v in m) for m in (calib.segment.masks or [])]
        o = self.opts
        static = self._static_mask(ref.shape)
        so = StabilizerOptions(max_features=o.stab_max_features, ransac_thr=o.stab_ransac_thr,
                               min_inliers=o.stab_min_inliers, homography=o.stab_homography)
        self.stab = Stabilizer(ref, calib.sar, self.interlaced, static, so)
        self.debug = bool(o.debug_dir)
        self.det = Detector(self.geom, self.interlaced, self.shape, o, ref_masks=self.masks,
                            keep_rmap=self.debug)
        self.store = CandStore(max_len=o.track_window + 3 * o.finalize_lag + 60)
        self.seg = Segmenter(self.geom, o, self.dt, self.frame_dt, self.interlaced, self.store,
                             quality_of=self._min_quality, view=self.geom.view, usable_of=self._usable_mask)
        # per-sample records
        self._s: list = []
        self._tt: list = []             # sample time of each processed sample (aligned with _s)
        self._A: list = []              # 2x3 local affine (FinderResult.transforms)
        self._H: list = []              # 3x3 transforms actually used
        self._n_h = 0
        self._q: list = []
        self._qd: dict = {}
        self._method: list = []
        self._tl = {"s": [], "t": [], "n": [], "sigma": [], "e_rel": [], "e_catch": [], "skipped": []}
        self._usable: dict = {}         # s -> detection data usable (DetFrame not skipped)
        self._open: list = []
        self._records: list = []
        self._next_tid = 1
        self._seed_hi: Optional[int] = None
        self._n_since = 0
        self._n_overlay = 0
        self._finished: Optional[FinderResult] = None
        self._diag = {"n_discarded_short": 0, "n_seeds": 0, "n_tracks_built": 0, "n_merged": 0}
        # sample ranges to seed again once (after a track there was rejected / trimmed)
        self._reseed: list[tuple[int, int]] = []
        # s -> uids of the detections of rejected tracks: released for other tracks to extend
        # into, but never seed points again (the re-seed would just rebuild the same junk)
        self._blocked_seeds: dict = {}
        self._dup_windows: list = []    # (t0, t1) of DUPLICATE losers to list for review
        self._time = {"stabilize": 0.0, "detect": 0.0, "track": 0.0}
        self._t_first: Optional[float] = None
        self._s_first: Optional[int] = None
        self._s_last: Optional[int] = None
        self.s_min_allowed: Optional[int] = None
        self.s_max_allowed: Optional[int] = None

    # ------------------------------------------------------------- setup
    def _static_mask(self, shape: tuple) -> np.ndarray:
        o = self.opts
        H, W = shape
        m = np.full((H, W), 255, np.uint8)
        b = int(o.border_px)
        if b > 0:
            m[:, :b] = 0
            m[:, W - b:] = 0
        if o.bottom_rows > 0:
            m[H - int(o.bottom_rows):] = 0
        corr = G.polygon_mask(self.geom.corridor, (H, W), 1, 0.0, dilate_px=int(o.corridor_feature_margin))
        m[corr > 0] = 0
        for (x0, y0, x1, y1) in self.masks:
            m[max(0, int(y0)):max(0, int(math.ceil(y1)) + 1), max(0, int(x0)):max(0, int(math.ceil(x1)) + 1)] = 0
        return m

    def _min_quality(self, s0: int, s1: int) -> float:
        """Registration quality over a sample range: median quality times sqrt(share registered)."""
        qs = np.array([self._qd[s] for s in range(s0, s1 + 1) if s in self._qd])
        if not len(qs):
            return 0.0
        ok = qs > 0
        if not ok.any():
            return 0.0
        return float(np.median(qs[ok]) * math.sqrt(ok.mean()))

    def _usable_mask(self, s0: int, s1: int) -> list:
        """Per sample s0..s1: processed with usable detection data (not skipped)?"""
        u = self._usable
        return [u.get(s, False) for s in range(s0, s1 + 1)]

    # ----------------------------------------------------------- stream
    def process(self, sample: Sample) -> None:
        """Feed the next sample (time order).  Gaps in `s` are allowed."""
        if self._finished is not None:
            raise RuntimeError("PitchFinder.finish() was already called")
        img = sample.image
        if img.ndim == 3:
            import cv2
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.shape != self.shape:
            img = np.ascontiguousarray(img[:self.shape[0], :self.shape[1]])
        t0 = time.perf_counter()
        r = self.stab.register(img, sample.parity)
        self._time["stabilize"] += time.perf_counter() - t0
        if self._s_first is None:
            self._s_first, self._t_first = sample.s, sample.t
        self._s_last = sample.s
        self._s.append(sample.s)
        self._tt.append(float(sample.t))
        c_ref = np.array([0.5 * (self.geom.bbox[0] + self.geom.bbox[2]), 0.5 * (self.geom.bbox[1] + self.geom.bbox[3])])
        c_s = G.apply(G.invert(r.A), c_ref[None])[0]
        self._A.append(G.local_affine(r.A, c_s))
        self._H.append(G.to3(r.A))
        self._n_h += int(r.homography)
        self._q.append(r.quality)
        self._qd[sample.s] = r.quality
        self._method.append(r.method)
        self._update_overlay_mask()
        t0 = time.perf_counter()
        dfs = self.det.push(sample.s, sample.t, sample.k, sample.parity, img, r.A, r.quality)
        self._time["detect"] += time.perf_counter() - t0
        for df in dfs:
            self._on_detection(df)

    def _update_overlay_mask(self) -> None:
        n = sum(int(f.sum()) for f in self.stab._flags.values())
        if n == self._n_overlay:
            return
        self._n_overlay = n
        boxes = self.stab.overlay_boxes(pad=8.0)
        # Never let auto-masks blank out a large part of the ball's corridor.
        cx0, cy0, cx1, cy1 = self.geom.bbox
        c_area = max(1.0, (cx1 - cx0) * (cy1 - cy0))
        def _inter(b):
            return max(0.0, min(b[2], cx1) - max(b[0], cx0)) * max(0.0, min(b[3], cy1) - max(b[1], cy0))
        if sum(_inter(b) for b in boxes) > 0.15 * c_area:
            kept = [b for b in boxes if _inter(b) == 0.0]
            self._diag["overlay_boxes_dropped"] = [list(map(float, b)) for b in boxes if _inter(b) > 0.0]
            boxes = kept
        H, W = self.shape
        m = np.full((H, W), 255, np.uint8)
        for (x0, y0, x1, y1) in boxes:
            r0 = max(0, int(math.floor(y0 / self.rs)))
            r1 = min(H, int(math.ceil(y1 / self.rs)) + 1)
            m[r0:r1, max(0, int(x0)):min(W, int(math.ceil(x1)) + 1)] = 0
        self.det.raw_mask = m
        self._diag["overlay_boxes"] = [list(map(float, b)) for b in boxes]
        inter = []
        for b in boxes:
            x0, y0, x1, y1 = self.geom.bbox
            if not (b[2] < x0 or b[0] > x1 or b[3] < y0 or b[1] > y1):
                inter.append(list(map(float, b)))
        if inter:
            self._diag["overlay_in_corridor"] = inter

    def _on_detection(self, df: DetFrame) -> None:
        self.store.add(df)
        self._usable[df.s] = not df.skipped
        tl = self._tl
        tl["s"].append(df.s)
        tl["t"].append(df.t)
        tl["n"].append(len(df.cands))
        tl["sigma"].append(df.sigma)
        tl["e_rel"].append(df.e_rel)
        tl["e_catch"].append(df.e_catch)
        tl["skipped"].append(df.skipped)
        self._n_since += 1
        if self._n_since >= self.opts.track_every:
            self._n_since = 0
            self._run_tracker(final=False)

    # ---------------------------------------------------------- tracking
    def _run_tracker(self, final: bool) -> None:
        t0 = time.perf_counter()
        try:
            self._run_tracker_inner(final)
        finally:
            self._time["track"] += time.perf_counter() - t0

    def _run_tracker_inner(self, final: bool) -> None:
        st = self.store
        if not st.frames:
            return
        o = self.opts
        s_lo, s_hi = st.s_min, st.s_max
        for s_old in [k for k in self._blocked_seeds if k < s_lo]:
            del self._blocked_seeds[s_old]          # bounded memory: only the cached window
        for tr in list(self._open):
            if tr.open_fwd:
                continue_track(tr, st, self.geom, o, self.dt, s_lo, s_hi)
                st.claim(tr)
        seed_lo = s_lo + 1 if self._seed_hi is None else max(s_lo + 1, self._seed_hi - 3)
        seed_hi = s_hi if final else s_hi - 3
        if self.s_min_allowed is not None:
            seed_lo = max(seed_lo, self.s_min_allowed)
        if self.s_max_allowed is not None:
            seed_hi = min(seed_hi, self.s_max_allowed)
        if seed_hi >= seed_lo:
            self._seed_range(seed_lo, seed_hi, s_lo, s_hi)
            self._seed_hi = seed_hi
        # Seeding is greedy and streaming: a window is normally seeded once.  If a junk
        # track held some of the ball's detections at that moment and later let them go
        # (rejected / trimmed), the ball would never get its own seed.  Seed such
        # windows once more.
        self._do_reseed(s_lo, seed_hi, s_hi)
        self._finalize_open(final, s_hi)
        if final:
            # tracks rejected in this last pass get their window re-seeded too
            for _ in range(2):
                if not self._reseed:
                    break
                self._do_reseed(s_lo, seed_hi, s_hi)
                self._finalize_open(final, s_hi)

    def _do_reseed(self, s_lo: int, seed_hi: int, s_hi: int) -> None:
        pending, self._reseed = self._reseed, []
        for (a, b) in pending:
            a, b = max(a, s_lo + 1), min(b, seed_hi)
            if b >= a:
                self._diag["n_reseeds"] = self._diag.get("n_reseeds", 0) + 1
                self._seed_range(a, b, s_lo, s_hi, reseed=True)

    def _seed_range(self, seed_lo: int, seed_hi: int, s_lo: int, s_hi: int, reseed: bool = False) -> None:
        st, o = self.store, self.opts
        if True:
            seeds = find_seeds(st, seed_lo, seed_hi, self.geom, o)
            self._diag["n_seeds"] += len(seeds)
            n_built = 0
            blocked = self._blocked_seeds
            for seed in seeds:
                _, a, b, c = seed
                key = tuple((int(n["s"]), int(n["uid"])) for n in (a, b, c))
                if any(st.is_used(sk, uk) or uk in blocked.get(sk, ()) for sk, uk in key):
                    continue
                if n_built >= o.max_builds_per_run:
                    # bounded work in heavy clutter; seeds are processed strongest first
                    self._diag["n_seeds_skipped"] = self._diag.get("n_seeds_skipped", 0) + 1
                    continue
                n_built += 1
                tr = build_track(self._next_tid, seed, st, self.geom, o, self.dt, s_lo, s_hi,
                                 self.s_min_allowed, self.s_max_allowed)
                self._diag["n_tracks_built"] += 1
                if tr.n_inliers < 6 and not tr.open_fwd and tr.seed_score < 40:
                    self._discard(tr)
                    continue
                self._next_tid += 1
                if reseed:
                    tr.info["from_reseed"] = True
                st.claim(tr)
                self._open.append(tr)

    def _finalize_open(self, final: bool, s_hi: int) -> None:
        o = self.opts
        for tr in sorted(self._open, key=lambda t: t.s_first):
            if tr.info.get("merged_into") is not None or tr not in self._open:
                continue                       # absorbed by a merge in this loop
            if final or (tr.s_last < s_hi - o.finalize_lag and not tr.open_fwd):
                self._open.remove(tr)
                self._finalize(tr)

    def _discard(self, tr: Track) -> None:
        self._diag["n_discarded_short"] += 1
        if self.debug and len(self._diag.setdefault("discarded", [])) < 2000:
            A = tr.arrays()
            self._diag["discarded"].append({"s": A["s"].tolist(), "x": np.round(A["x"], 1).tolist(),
                                            "y": np.round(A["y"], 1).tolist(), "z": np.round(A["z"], 1).tolist()})

    def _finalize(self, tr: Track) -> None:
        o, st = self.opts, self.store
        # merge with other ended fragments
        for other in list(self._open):
            if other is tr or other.model is None or tr.model is None:
                continue
            if other.s_first > tr.s_last + o.merge_gap or other.s_last < tr.s_first - o.merge_gap:
                continue
            m = try_merge(tr, other, self.geom, o)
            if m is not None:
                st.claim(other, False)
                st.claim(tr, False)
                self._open.remove(other)
                other.info["merged_into"] = tr.tid
                m.tid = tr.tid
                tr.pts, tr.model, tr.open_fwd = m.pts, m.model, m.open_fwd
                st.claim(tr)
                self._diag["n_merged"] += 1
        if o.pass2 and tr.model is not None and self._worth_pass2(tr):
            n_add = pass2(tr, st, self.geom, o, self.rs, clutter_factor=self.det._clutter_factor,
                          s_min_allowed=self.s_min_allowed, s_max_allowed=self.s_max_allowed)
            if n_add:
                fit_track(tr, self.geom, o)
                st.claim(tr)
        if tr.model is not None and self.geom.view == "behind":
            # behind view: drop wind-up / glove stragglers, regrow along the global model, trim again
            st.claim(tr, False)
            n_trim = trim_sparse_ends(tr, self.geom, o)
            n_re = 0
            if tr.model is not None:
                n_re = regrow(tr, st, self.geom, o, self.rs, clutter_factor=self.det._clutter_factor,
                              s_min_allowed=self.s_min_allowed, s_max_allowed=self.s_max_allowed)
                if n_re and tr.model is not None:
                    n_trim += trim_sparse_ends(tr, self.geom, o)
            tr.info["n_regrow"], tr.info["n_trimmed"] = n_re, n_trim
            st.claim(tr)
        tr.info["delivery_cue"] = self._delivery_cue(tr)
        self.seg.s_range = (self._s_first, self._s_last)
        pt = self.seg.classify(tr)
        pt.diagnostics["delivery_cue"] = tr.info["delivery_cue"]
        self._behind_past_release_check(tr, pt)
        discard = (pt.status == "rejected" and pt.reject_code == "TOO_SHORT" and len(pt.points) < 6
                   and tr.seed_score < 40)
        if pt.status == "rejected" and not discard and o.reseed_release_claims:
            # A rejected track gives its detections back: some of them may be the ball's
            # (a junk track built first can take part of the flight) and the window is
            # re-seeded below, so the ball's own track can extend through them.  They
            # are not used as seed points again (that just rebuilds the same junk).
            # Only with FinderOptions.reseed_release_claims (see there: off by default,
            # it did more harm than good on the held-out suite).
            st.claim(tr, False)
            for other in self._open:          # detections shared with live tracks stay theirs
                st.claim(other)
            for p in tr.pts.values():
                if p.uid >= 0:
                    self._blocked_seeds.setdefault(int(p.s), set()).add(int(p.uid))
        if (pt.status == "rejected" or tr.info.get("n_trimmed")) and not tr.info.get("from_reseed"):
            self._reseed.append((tr.s_first - 3, tr.s_last + 3))
        if discard:
            st.claim(tr, False)
            self._discard(tr)
            return
        rec = _Record(track=tr, pt=pt)
        n_rej_img = sum(1 for r in self._records if r.strobe is not None and r.pt.status != "pitch")
        if self.debug and (pt.status == "pitch" or n_rej_img < self.opts.debug_max_pitch_images):
            from pitchspeed.vision import debug as D
            try:
                rec.strobe = D.strobe_image(tr, pt, st, self.geom, self.rs)
            except Exception as e:      # debug output must never break the analysis
                log.debug("strobe image failed: %s", e)
        self._records.append(rec)

    def _behind_past_release_check(self, tr: Track, pt) -> None:
        """Behind view plausibility: past the release point, and approaching the camera.

        1. A pitch must get clearly past the release point.

        The behind-view corridor reaches behind and beside the release point
        (geometry.behind_corridor), where the catcher's throw-back lob comes
        down onto the pitcher: a short track moving *toward* the release
        point with positive progress rate that would otherwise pass
        segmentation.  A pitched ball always moves on toward the catch point;
        if the track never gets past progress `min_progress_last_behind`
        (default 0.1) it is not a measurable pitch either way (the ball is
        hidden right after release: no catch, no speed) and is rejected as
        NO_PROGRESS (diagnostics['no_progress_reason']).  On the synthetic
        suites every pitch visible for >= 6 fields beyond the release region
        reached >= 0.15.
        2. A pitch comes toward the camera, so its image speed must grow as
        perspective predicts (see the comment below); otherwise THROWBACK with
        diagnostics['throwback_reason'].
        (Both belong in segment.Segmenter.classify; kept here so the corridor
        change is self-contained.)
        """
        if self.geom.view != "behind" or pt.status != "pitch" or pt.release is not None and pt.release.method == "manual":
            return
        A = tr.arrays(inliers_only=True)
        if not len(A["x"]):
            return
        g = self.geom
        prog = g.progress(np.stack([A["x"], A["y"]], 1))
        pmax = float(np.max(prog))
        pt.diagnostics["progress_max"] = pmax
        if pmax < self.opts.min_progress_last_behind:
            pt.status = "rejected"
            pt.reject_code = "NO_PROGRESS"
            pt.diagnostics["no_progress_reason"] = "never_past_release_point"
            return
        # Approach test.  A pitched ball comes toward the camera, so its image speed grows
        # with perspective: with image position ~ 1/Z and progress p linear in 1/Z between
        # the release (p = 0) and the catch (p = 1, ball q = d_catch / d_rel times larger),
        # Z_rel / Z = 1 + p (q - 1) and the image speed ~ 1/Z^2 grows by
        #   expected = ((1 + p2 (q - 1)) / (1 + p1 (q - 1)))^2   from progress p1 to p2.
        # The end of a throw-back lob coming down to the pitcher's glove also moves toward
        # the catch point in the image (progress -0.07 -> 0.3) but at a steady or falling
        # image speed.  Measured log(observed) / log(expected) on the synthetic behind
        # suites: real pitches 0.68 .. 2.7, throw-back endings -3.9 .. 0.8 (5 of 7 < 0.5).
        m = tr.model
        if m is None or len(A["t"]) < 6:
            return
        q = g.d_catch / max(g.d_rel, 1e-6)
        if not g.scale_known:
            q = min(q, 2.0)                       # guessed scale: weaker, conservative test
        p1, p2 = (float(np.clip(v, 0.0, 1.2)) for v in (prog[0], prog[-1]))
        expected = ((1 + p2 * (q - 1)) / (1 + p1 * (q - 1))) ** 2
        if expected < self.opts.approach_min_expected:
            return
        v = m.vel([float(A["t"][0]), float(A["t"][-1])]) * [g.sar, 1.0]
        sp = np.hypot(v[:, 0], v[:, 1])
        if sp[0] <= 1e-6 or sp[1] <= 1e-6:
            return
        score = math.log(sp[1] / sp[0]) / math.log(expected)
        pt.diagnostics["approach_score"] = float(score)
        if score < self.opts.approach_min_score:
            pt.status = "rejected"
            pt.reject_code = "THROWBACK"
            pt.diagnostics["throwback_reason"] = "image_speed_not_growing (not approaching the camera)"

    def _worth_pass2(self, tr: Track) -> bool:
        """Cheap pre-check: pass 2 only for tracks that could still become pitches."""
        A = tr.arrays(inliers_only=True)
        if len(A["t"]) < self.opts.min_inliers - 2:
            return False
        prog = self.geom.progress(np.stack([A["x"], A["y"]], 1))
        return bool(prog[-1] - prog[0] > 0.02)

    def _delivery_cue(self, tr: Track) -> bool:
        """Activity burst near the release point shortly before the track."""
        tl = self._tl
        if len(tl["t"]) < 30:
            return False
        t = np.asarray(tl["t"])
        e = np.asarray(tl["e_rel"], float)
        ok = np.isfinite(e)
        if ok.sum() < 30:
            return False
        base = float(np.median(e[ok]))
        mad = 1.4826 * float(np.median(np.abs(e[ok] - base)))
        thr = base + max(0.02, 4 * mad)
        t0 = tr.sorted()[0].t
        w = ok & (t >= t0 - 1.0) & (t <= t0 + 0.1)
        return bool(w.any() and np.max(e[w]) > thr)

    # ------------------------------------------------------------ finish
    def finish(self) -> FinderResult:
        if self._finished is not None:
            return self._finished
        t0 = time.perf_counter()
        flushed = self.det.flush()
        self._time["detect"] += time.perf_counter() - t0
        for df in flushed:
            self.store.add(df)
            self._usable[df.s] = not df.skipped
            tl = self._tl
            for k, v in (("s", df.s), ("t", df.t), ("n", len(df.cands)), ("sigma", df.sigma),
                         ("e_rel", df.e_rel), ("e_catch", df.e_catch), ("skipped", df.skipped)):
                tl[k].append(v)
        self._run_tracker(final=True)
        self._parity_pairs()
        tracks, rejected = self._resolve_duplicates()
        missed = self._possible_missed(tracks, rejected)
        n = len(self._s)
        transforms = np.array(self._A, dtype=np.float64).reshape(n, 2, 3) if n else np.zeros((0, 2, 3))
        diag = dict(self._diag)
        diag.update(self._summary())
        diag["transforms_3x3"] = np.array(self._H, dtype=np.float64).reshape(n, 3, 3) if n else np.zeros((0, 3, 3))
        diag["n_homography_samples"] = int(self._n_h)
        diag["timing_ms_per_sample"] = {k: 1000.0 * v / max(1, n) for k, v in self._time.items()}
        res = FinderResult(tracks=tracks, rejected=rejected, possible_missed=missed, transforms=transforms,
                           transform_quality=np.array(self._q, dtype=np.float64),
                           sample_index=np.array(self._s, dtype=np.int64), diagnostics=diag)
        if self.debug:
            from pitchspeed.vision import debug as D
            try:
                D.write_debug(self.opts.debug_dir, self, res)
            except Exception as e:
                log.warning("could not write debug output: %s", e)
                res.diagnostics["debug_error"] = str(e)
        res.diagnostics.pop("discarded", None)
        self._finished = res
        return res

    def _parity_pairs(self) -> None:
        """Field-order check across tracks.

        With the wrong field order the two fields of a flight are offset by a
        field of motion, so the tracker builds one track per parity.  Two
        overlapping tracks that are each (almost) single-parity and of
        opposite parity are tested jointly: if their union is much smoother
        under the flipped field timing, both get 'field_order_conflict' and the
        estimator's 'field_order_ambiguous' (wider flight time), and
        diagnostics['field_order_pair'] (pipeline: a 'flipped' vote).
        """
        if not self.interlaced:
            return
        from pitchspeed.vision.track import field_order_check

        def parity_share(r):
            P = [p for p in r.track.sorted() if p.inlier]
            if len(P) < 6:
                return None, 0.0
            k1 = sum(p.k for p in P) / len(P)
            return (1 if k1 >= 0.5 else 0), max(k1, 1 - k1)

        recs = [(r,) + parity_share(r) for r in self._records]
        recs = [(r, k, sh) for r, k, sh in recs if k is not None and sh >= 0.85]
        for i, (a, ka, _) in enumerate(recs):
            for b, kb, _ in recs[i + 1:]:
                if ka == kb or b.pt.t_first > a.pt.t_last or a.pt.t_first > b.pt.t_last:
                    continue
                P = [p for p in a.track.sorted() + b.track.sorted() if p.inlier]
                P.sort(key=lambda p: p.t)
                fo = field_order_check([p.t for p in P], [p.x for p in P], [p.y for p in P],
                                       [p.k for p in P], self.frame_dt)
                if fo.get("conflict"):
                    for r, other in ((a, b), (b, a)):
                        _add_flags(r.pt, F_FIELD_ORDER_CONFLICT, F_FIELD_ORDER)
                        r.pt.diagnostics["field_order_pair"] = {"other_track": other.pt.track_id, **fo}

    def _resolve_duplicates(self):
        """One pitch per flight: overlapping candidates are fragments / echoes of one flight.

        Two 'pitch' candidates are the same flight when their inlier
        observation spans overlap or are at most `merge_gap` samples apart (a
        second ball cannot be in flight then).  Candidates further apart are
        never duplicates, however close their release times (the old rule
        dropped real pitches for junk 1 s away); the pipeline sorts out
        throw-backs and doubtful tracks.  Within a flight the candidate that
        observed most of it wins (`flight_score`: inliers x image progress
        covered x contrast): not the confidence, which favours late fragments
        (their own late 'first free field' release makes them look complete).
        The losers become DUPLICATE (diagnostics 'duplicate_of'); the winner:
          * confidence = max over the flight (each fragment is evidence of the
            pitch; diagnostics 'confidence_own');
          * behind view: a fragment on the same image path (same_path) that
            saw the ball clearly free *before* the winner's release estimate
            proves that estimate late: the release is widened to cover both
            ('release_uncertain', diagnostics 'release_widened_by_track');
          * a field-order conflict found on any fragment is kept.
        A loser that was seen mostly outside the winner's flight is listed in
        possible_missed ('rejected_track:DUPLICATE'), so nothing is hidden.
        """
        o = self.opts
        recs = sorted(self._records, key=lambda r: r.pt.t_first)
        pitches = [r for r in recs if r.pt.status == "pitch"]
        order = sorted(pitches, key=lambda r: (-flight_score(r.pt), -r.pt.confidence, r.pt.t_first))
        kept: list = []
        group: dict = {}
        for r in order:
            r.pt.diagnostics["flight_score"] = flight_score(r.pt)
            w = next((k for k in kept if same_flight(k.pt, r.pt, o.merge_gap)), None)
            if w is None:
                kept.append(r)
                group[id(r)] = []
                continue
            r.pt.status = "rejected"
            r.pt.reject_code = "DUPLICATE"
            r.pt.diagnostics["duplicate_of"] = w.pt.track_id
            group[id(w)].append(r)
        self._dup_windows = []
        for w in kept:
            self._absorb_duplicates(w, group[id(w)])
        tracks = [r.pt for r in recs if r.pt.status == "pitch"]
        rejected = [r.pt for r in recs if r.pt.status != "pitch"]
        return tracks, rejected

    def _absorb_duplicates(self, w: "_Record", losers: list) -> None:
        """Carry what the DUPLICATE losers of one flight know over to its winner (see above)."""
        if not losers:
            return
        pt, dt = w.pt, self.dt
        pt.diagnostics["duplicates"] = [q.pt.track_id for q in losers]
        # confidence: the flight as a whole
        best = max(losers, key=lambda q: q.pt.confidence)
        if best.pt.confidence > pt.confidence:
            pt.diagnostics["confidence_own"] = pt.confidence
            pt.diagnostics["confidence_from_track"] = best.pt.track_id
            pt.confidence = float(best.pt.confidence)
            pt.flags = [f for f in pt.flags if f not in ("doubtful", "review")]
            if pt.confidence < 0.3:
                pt.flags.append("doubtful")
            elif pt.confidence < 0.6:
                pt.flags.append("review")
        # field order: a conflict seen on any fragment of the flight
        fo_from = [q for q in losers if F_FIELD_ORDER_CONFLICT in q.pt.flags]
        if fo_from and F_FIELD_ORDER_CONFLICT not in pt.flags:
            _add_flags(pt, F_FIELD_ORDER_CONFLICT, F_FIELD_ORDER)
            pt.diagnostics["field_order_conflict_from_track"] = fo_from[0].pt.track_id
        pair = next((q.pt.diagnostics["field_order_pair"] for q in losers
                     if q.pt.diagnostics.get("field_order_pair")), None)
        if pair is not None and not pt.diagnostics.get("field_order_pair"):
            pt.diagnostics["field_order_pair"] = dict(pair)     # the pipeline's field-order vote reads it
        # behind-view release: the ball seen free earlier on the same path
        rel = pt.release
        if self.geom.view == "behind" and rel is not None and rel.method == "first_free_field":
            early = []
            for q in losers:
                tq = q.pt.diagnostics.get("t_first_clean")
                if (tq is not None and q.pt.release is not None and tq < rel.t - 2.0 * rel.sigma
                        and same_path(pt, q.pt, self.geom, dt)):
                    early.append(q)
            if early:
                q = min(early, key=lambda q: q.pt.release.t)
                lo = min(q.pt.release.t, rel.t)
                iv = pt.diagnostics.get("release_interval")
                if iv:
                    lo = min(lo, float(iv[0]))
                hi = rel.t if not iv else max(rel.t, float(iv[1]))
                sig = math.sqrt(max(rel.sigma, q.pt.release.sigma) ** 2 + (hi - lo) ** 2 / 12.0)
                t_mid = 0.5 * (lo + hi)
                pt.diagnostics["release_before_widening"] = to_jsonable(rel)
                pt.diagnostics["release_widened_by_track"] = q.pt.track_id
                pt.diagnostics["release_interval"] = [lo, hi]
                pt.release = EventTime(t=float(t_mid), sigma=float(sig), method="first_free_field",
                                       s=self.seg._nearest_s(t_mid))
                if "release_uncertain" not in pt.flags:
                    pt.flags.append("release_uncertain")
                if pt.catch is not None:
                    pt.diagnostics["flight_time"] = pt.catch.t - pt.release.t
        # never hide a loser that saw something outside the reported flight
        if pt.release is not None and pt.catch is not None:
            f0, f1 = pt.release.t - 2 * dt, pt.catch.t + 2 * dt
            for q in losers:
                ts = [p.t for p in q.pt.points if p.inlier]
                if ts and sum(1 for t in ts if t < f0 or t > f1) >= 0.5 * len(ts):
                    self._dup_windows.append((ts[0] - 0.25, ts[-1] + 0.25))

    def _sample_at(self, t: float) -> int:
        tl = self._tl
        if not tl["t"]:
            return 0
        tt = np.asarray(tl["t"])
        i = int(np.argmin(np.abs(tt - t)))
        return int(tl["s"][i] + round((t - tt[i]) / self.dt))

    def _possible_missed(self, tracks: list, rejected: list) -> list:
        o = self.opts
        out = []
        rel_times = [p.release.t if p.release else p.t_first for p in tracks]

        def covered(t0, t1):
            return any(t0 - 0.6 <= rt <= t1 + 0.3 for rt in rel_times)

        def add(t0, t1, reason):
            if self._t_first is not None:
                t0 = max(t0, self._t_first)
            out.append({"s_start": self._sample_at(t0), "s_end": self._sample_at(t1),
                        "t_start": float(t0), "t_end": float(t1), "reason": reason})

        # 1. delivery cue without a track
        tl = self._tl
        if len(tl["t"]) > 60:
            t = np.asarray(tl["t"])
            er = np.asarray(tl["e_rel"], float)
            ec = np.asarray(tl["e_catch"], float)
            ok = np.isfinite(er) & np.isfinite(ec)
            if ok.sum() > 60:
                def thr(e):
                    b = float(np.median(e[ok]))
                    return b + max(0.03, 5 * 1.4826 * float(np.median(np.abs(e[ok] - b))))
                tr_, tc_ = thr(er), thr(ec)
                hot = ok & (er > tr_)
                i = 0
                n = len(t)
                while i < n:
                    if not hot[i]:
                        i += 1
                        continue
                    j = i
                    while j + 1 < n and (hot[j + 1] or (j + 2 < n and hot[j + 2])):
                        j += 1
                    if j - i + 1 >= 3:
                        tb = t[i]
                        w = ok & (t >= t[j] + o.cue_min_gap_s - 0.2) & (t <= t[j] + o.cue_max_gap_s)
                        if w.any() and np.max(ec[w]) > tc_ and not covered(tb, t[j] + o.cue_max_gap_s):
                            add(tb - 0.5, t[j] + o.cue_max_gap_s + 0.3, "delivery_cue_without_track")
                    i = j + 1
        # 2. rejected tracks that look like partial pitches: enough clear detections moving
        #    from the release end toward the catch end
        for p in rejected:
            if p.reject_code not in PARTIAL_CODES or covered(p.t_first, p.t_last):
                continue
            inl = [q for q in p.points if q.inlier]
            d = p.diagnostics
            if (len(inl) >= 8 and float(np.median([q.score for q in inl])) >= o.z_strong + 1.0
                    and d.get("progress_rate", 0.0) > 0
                    and (d.get("progress_last", 0.0) - d.get("progress_first", 0.0)) >= 0.15):
                add(p.t_first - 0.5, p.t_last + 0.5, f"rejected_track:{p.reject_code}")
        # 2b. DUPLICATE losers seen mostly outside the flight of the pitch they lost to
        #     (_resolve_duplicates): listed even near a reported pitch
        for (t0, t1) in getattr(self, "_dup_windows", []):
            add(t0, t1, "rejected_track:DUPLICATE")
        # 3. stabilization failures (recorded sample times and indices: the clock can
        #    jump at dropped frames / VFR, so s and t are not linear in each other)
        q = np.asarray(self._q)
        if len(q):
            bad = q <= 0
            s = np.asarray(self._s)
            tt = np.asarray(self._tt, float)
            i = 0
            while i < len(q):
                if bad[i]:
                    j = i
                    while j + 1 < len(q) and bad[j + 1]:
                        j += 1
                    if j - i + 1 >= 15:
                        out.append({"s_start": int(s[i]), "s_end": int(s[j]), "t_start": float(tt[i]),
                                    "t_end": float(tt[j]), "reason": "stabilization_failed"})
                    i = j + 1
                else:
                    i += 1
        # merge overlapping windows with the same reason
        out.sort(key=lambda d: (d["reason"], d["t_start"]))
        merged = []
        for d in out:
            if merged and merged[-1]["reason"] == d["reason"] and d["t_start"] <= merged[-1]["t_end"]:
                merged[-1]["t_end"] = max(merged[-1]["t_end"], d["t_end"])
                merged[-1]["s_end"] = max(merged[-1]["s_end"], d["s_end"])
            else:
                merged.append(dict(d))
        merged.sort(key=lambda d: d["t_start"])
        return merged

    def _summary(self) -> dict:
        q = np.asarray(self._q)
        tl = self._tl
        n_c = np.asarray(tl["n"]) if tl["n"] else np.zeros(0)
        sig = np.asarray(tl["sigma"], float) if tl["sigma"] else np.zeros(0)
        out = {
            "n_samples": len(self._s),
            "n_detected": len(tl["s"]),
            "view": self.geom.view,
            "geometry": {"scale_rel_px_per_ft": self.geom.scale_rel, "scale_catch_px_per_ft": self.geom.scale_catch,
                         "scale_known": self.geom.scale_known, "ball_px_release": self.geom.d_rel,
                         "ball_px_catch": self.geom.d_catch, "corridor_bbox": list(self.geom.bbox)},
            "stabilization": {"methods": dict(self.stab.stats),
                              "quality_median": float(np.median(q)) if len(q) else None,
                              "quality_p05": float(np.percentile(q, 5)) if len(q) else None,
                              "reference_field_offset": self.stab.field_offset},
            "candidates_per_sample": float(np.mean(n_c)) if len(n_c) else 0.0,
            "residual_sigma_median": float(np.nanmedian(sig)) if len(sig) and np.isfinite(sig).any() else None,
            "skipped_samples": int(sum(1 for s in tl["skipped"] if s)),
            "kernels": [{"length": k.length, "angle_deg": math.degrees(k.angle), "width": k.width}
                        for k in self.det.bank],
        }
        dl = self._duty_estimate()
        if dl is not None:
            out["shutter_duty_estimate"] = dl
        return out

    def _duty_estimate(self) -> Optional[float]:
        """median((L_obs - d) / |v|) over accepted-track detections (cv.md section 3)."""
        vals = []
        for r in self._records:
            if r.pt.status != "pitch" or r.track.model is None:
                continue
            for p in r.track.sorted():
                if not p.inlier or p.src != "det" or p.length <= 0:
                    continue
                v = r.track.model.vel([p.t])[0]
                sp = float(np.hypot(v[0] * self.geom.sar, v[1])) * self.dt
                if sp < 3.0:
                    continue
                d = float(self.geom.ball_diameter(self.geom.progress(np.array([[p.x, p.y]]))[0]))
                vals.append((p.length - 1.2 * d) / sp)
        if len(vals) < 5:
            return None
        return float(np.clip(np.median(vals), 0.0, 1.5))

    # helpers for track_between / review
    def records(self) -> list:
        return list(self._records)


# ------------------------------------------------------------ track_between ----
def track_between(source, calib: Calibration, scan: ScanInfo, s_release: int, s_catch: int,
                  reference_frame: np.ndarray, opts: Optional[FinderOptions] = None,
                  manual_points: Optional[list] = None) -> PitchTrack:
    """Constrained re-tracking between user-marked release and catch samples.

    Runs a PitchFinder over [s_release - margin, s_catch + margin] (margin
    covers the background offsets), seeds only inside [s_release, s_catch]
    with relaxed thresholds, and returns the track with the most inliers in
    that window (or a track made of `manual_points` only).  The returned
    track's release / catch are the user's marks (method 'manual',
    t = sample time, sigma = dt / 2); the automatic estimates inside the
    window are kept in diagnostics['auto_release'] / ['auto_catch'] (and
    the automatic end kind in ['auto_end_kind']: the returned end_kind is
    always 'catch', a hand-marked catch being the ball in the glove).
    status is 'pitch' unless no point at all could be found
    (reject_code 'NO_TRACK', events still set).
    """
    if s_catch <= s_release:
        raise ValueError("s_catch must be after s_release")
    o = replace(opts) if opts is not None else FinderOptions()
    o.debug_dir = None
    o.z_strong = min(o.z_strong, 4.0)
    o.track_every = 10 ** 9
    o.min_progress = 0.0
    dt = float(source.sample_dt)
    margin = max(max(o.bg_offsets), max(o.bg_offsets_slow) if calib.view == "behind" else 0) + 4
    s0 = max(0, int(s_release) - margin - o.pass2_back)
    s1 = min(int(source.n_samples), int(s_catch) + margin + 1)
    finder = PitchFinder(calib, scan, dt, source.sample_shape, reference_frame, o)
    finder.s_min_allowed = int(s_release) - 2
    finder.s_max_allowed = int(s_catch) + 2
    for smp in source.iter_samples(s0, s1):
        finder.process(smp)
    finder.finish()
    t_rel = float(source.sample_time(int(s_release)))
    t_cat = float(source.sample_time(int(s_catch)))
    best, best_n = None, 0
    for r in finder.records():
        n = sum(1 for p in r.track.sorted() if p.inlier and s_release - 2 <= p.s <= s_catch + 2)
        if n > best_n:
            best, best_n = r, n
    manual_points = list(manual_points or [])
    if best is not None:
        tr = best.track
        auto = best.pt
        tr.pts = {s: p for s, p in tr.pts.items() if s_release - 2 <= s <= s_catch + 2}
    else:
        tr = Track(tid=0)
        auto = None
    for mp in manual_points:
        k = int(mp.s) & 1 if finder.interlaced else 0
        tr.pts[int(mp.s)] = Pt(s=int(mp.s), t=float(mp.t), k=k, x=float(mp.x), y=float(mp.y), z=10.0,
                               attached=False, sigma=float(mp.sigma or 1.0), src="manual")
    if tr.pts:
        fit_track(tr, finder.geom, o)
    pts = [TrackPoint(s=p.s, t=p.t, x=p.x, y=p.y, score=p.z, sigma=p.sigma, inlier=p.inlier,
                      manual=p.src == "manual") for p in tr.sorted()]
    out = PitchTrack(track_id=tr.tid, points=pts)
    out.release = EventTime(t=t_rel, sigma=dt / 2, method="manual", s=int(s_release))
    out.catch = EventTime(t=t_cat, sigma=dt / 2, method="manual", s=int(s_catch))
    out.flags.append("manual_events")
    out.model = tr.model.to_dict() if tr.model is not None else {}
    out.diagnostics = {"window": [int(s_release), int(s_catch)], "n_points": len(pts),
                       "n_manual_points": len(manual_points)}
    if auto is not None:
        out.diagnostics["auto_release"] = to_jsonable(auto.release)
        out.diagnostics["auto_catch"] = to_jsonable(auto.catch)
        out.diagnostics["auto_status"] = auto.status
        out.diagnostics["auto_reject_code"] = auto.reject_code
        out.diagnostics["auto_flags"] = list(auto.flags)
        out.diagnostics["auto_end_kind"] = auto.end_kind
        # The user's catch mark is "the first field with the ball in the glove": the flight
        # ends at the glove (end_kind 'catch'), whatever the automatic track's end looked
        # like (e.g. a 'contact' from a swing) -- never inherit the automatic end kind.
        out.end_kind = "catch"
        out.confidence = auto.confidence
        if F_FIELD_ORDER_CONFLICT in auto.flags:
            _add_flags(out, F_FIELD_ORDER_CONFLICT, F_FIELD_ORDER)
    if pts:
        out.diagnostics.update(t_first_obs=pts[0].t, t_last_obs=pts[-1].t, s_first_obs=pts[0].s,
                               s_last_obs=pts[-1].s)
        out.status = "pitch"
    else:
        out.status = "rejected"
        out.reject_code = "NO_TRACK"
    return out
