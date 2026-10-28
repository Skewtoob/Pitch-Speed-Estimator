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
`finalize_lag` samples ago (merge, pass-2 re-detection, refit, then
`segment.Segmenter.classify`).  Memory is bounded by the ring sizes, except
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
look like partial pitches, and from stretches where stabilization failed.
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
from pitchspeed.vision.segment import Segmenter
from pitchspeed.vision.stabilize import Stabilizer, StabilizerOptions
from pitchspeed.vision.track import (CandStore, Pt, Track, build_track, continue_track, find_seeds, fit_track,
                                     pass2, try_merge)

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
                             quality_of=self._min_quality, view=self.geom.view)
        # per-sample records
        self._s: list = []
        self._A: list = []              # 2x3 local affine (FinderResult.transforms)
        self._H: list = []              # 3x3 transforms actually used
        self._n_h = 0
        self._q: list = []
        self._qd: dict = {}
        self._method: list = []
        self._tl = {"s": [], "t": [], "n": [], "sigma": [], "e_rel": [], "e_catch": [], "skipped": []}
        self._open: list = []
        self._records: list = []
        self._next_tid = 1
        self._seed_hi: Optional[int] = None
        self._n_since = 0
        self._n_overlay = 0
        self._finished: Optional[FinderResult] = None
        self._diag = {"n_discarded_short": 0, "n_seeds": 0, "n_tracks_built": 0, "n_merged": 0}
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
            seeds = find_seeds(st, seed_lo, seed_hi, self.geom, o)
            self._diag["n_seeds"] += len(seeds)
            n_built = 0
            for seed in seeds:
                _, a, b, c = seed
                if any(st.is_used(int(n["s"]), int(n["uid"])) for n in (a, b, c)):
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
                st.claim(tr)
                self._open.append(tr)
            self._seed_hi = seed_hi
        # finalize
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
        tr.info["delivery_cue"] = self._delivery_cue(tr)
        self.seg.s_range = (self._s_first, self._s_last)
        pt = self.seg.classify(tr)
        pt.diagnostics["delivery_cue"] = tr.info["delivery_cue"]
        if pt.status == "rejected" and pt.reject_code == "TOO_SHORT" and len(pt.points) < 6 and tr.seed_score < 40:
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
        under the flipped field timing, both get 'field_order_conflict'.
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
                        if "field_order_conflict" not in r.pt.flags:
                            r.pt.flags.append("field_order_conflict")
                        r.pt.diagnostics["field_order_pair"] = {"other_track": other.pt.track_id, **fo}

    def _resolve_duplicates(self):
        o = self.opts
        recs = sorted(self._records, key=lambda r: r.pt.t_first)
        pitches = [r for r in recs if r.pt.status == "pitch"]
        for r in sorted(pitches, key=lambda r: -r.pt.confidence):
            if r.pt.status != "pitch":
                continue
            for q in pitches:
                if q is r or q.pt.status != "pitch":
                    continue
                if q.pt.confidence > r.pt.confidence:
                    continue
                tr_, tq = r.pt.release.t if r.pt.release else r.pt.t_first, \
                    q.pt.release.t if q.pt.release else q.pt.t_first
                if abs(tr_ - tq) < o.duplicate_s:
                    q.pt.status = "rejected"
                    q.pt.reject_code = "DUPLICATE"
                    q.pt.diagnostics["duplicate_of"] = r.pt.track_id
        tracks = [r.pt for r in recs if r.pt.status == "pitch"]
        rejected = [r.pt for r in recs if r.pt.status != "pitch"]
        return tracks, rejected

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
        # 3. stabilization failures
        q = np.asarray(self._q)
        if len(q):
            bad = q <= 0
            s = np.asarray(self._s)
            i = 0
            while i < len(q):
                if bad[i]:
                    j = i
                    while j + 1 < len(q) and bad[j + 1]:
                        j += 1
                    if j - i + 1 >= 15:
                        t0 = self._t_first + (s[i] - self._s_first) * self.dt
                        t1 = self._t_first + (s[j] - self._s_first) * self.dt
                        add(t0, t1, "stabilization_failed")
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
    window are kept in diagnostics['auto_release'] / ['auto_catch'].
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
        out.end_kind = auto.end_kind
        out.confidence = auto.confidence
        for f in auto.flags:
            if f == "field_order_conflict":
                out.flags.append(f)
    if pts:
        out.diagnostics.update(t_first_obs=pts[0].t, t_last_obs=pts[-1].t, s_first_obs=pts[0].s,
                               s_last_obs=pts[-1].s)
        out.status = "pitch"
    else:
        out.status = "rejected"
        out.reject_code = "NO_TRACK"
    return out
