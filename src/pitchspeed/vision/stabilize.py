"""Global motion compensation: every sample -> calibration reference frame.

Each sample (field or frame) is registered *directly* to the same-parity
field of the calibration reference frame (no chaining, so no drift while
the camera stays near the calibrated pose):

1. Shi-Tomasi corners are detected once per parity on the reference field
   (outside the static mask: dilated corridor, borders, head-switching
   lines, user / date-stamp masks).
2. Pyramidal Lucas-Kanade tracks them into the current sample, starting
   from the previous sample's transform (handles slow pans of many px).
3. A similarity (pan, tilt, roll, zoom) is fitted in *square* pixels with
   RANSAC (`estimateAffinePartial2D`) and converted back to frame px.  When
   the camera has rotated enough that perspective matters (a pan of a few
   degrees with a wide lens bends the motion field by ~10 px at the image
   edges) and a homography explains clearly more features, the homography
   is used instead (a rotating camera maps by a homography exactly).
   Transforms are therefore 3x3 (`StabResult.A`).

Fallbacks, in order: phase correlation to re-initialise LK; registration to
a keyframe (the last good same-parity sample, chained transform); after a
run of failures (scene cut / big move) periodic AKAZE re-registration to the
reference.  If everything fails the sample keeps the previous transform
with quality 0 ('hold') and the detector ignores it.

Reference coordinate system: for interlaced video the two fields of the
reference frame were exposed 1/60 s apart and differ by the camera shake
between them.  The reference system is defined half way between them
(the frame's mean exposure time); both parities are mapped into it, so
there is no parity-dependent offset in stabilized tracks.

Static overlays (burnt-in date stamp, logos) are stable in *raw*
coordinates and would pull the estimate toward zero motion.  Every sample
the RANSAC outliers are tested for a second, zero-motion hypothesis; when a
spatially compact set of features keeps a zero raw displacement while the
(spatially wider) camera hypothesis moves, those features are flagged as
overlay after 3 votes and dropped.  Their region is exposed by
`overlay_boxes()` (raw sample frame px) so the detector can mask it.

Features are detected per tile (4 x 3 grid) so a high-contrast overlay or a
textured corner cannot take all of them.

Noisy, low-texture footage (zoomed behind-the-plate views through a blurred
backstop fence on worn tape: LK errors of 1-2 px per feature) needs three
things, all measured against the synthetic ground-truth camera:

* features whose LK patch no longer matches (a player walked in front of
  them since the reference frame) are dropped before fitting
  (`changed_patch_factor`);
* the RANSAC inlier threshold follows the measured feature noise
  (`_adaptive_thr`, 1 .. `ransac_thr_max` px) instead of a fixed 1 px;
* the quality measures the precision of the fitted transform, not the
  per-feature residual (see `_quality`).
With the old fixed threshold / rms-based quality 82-98 % of the samples of
such clips were rejected ('hold') although the registration was good to
< 1 px, and the detector skipped them.

A registration that jumps implausibly far from the previous sample's is
rejected (`_continuous`): after a pan, whole-image phase correlation or LK
can lock onto a periodic fence lattice or static foreground and the wrong
transform then propagated through the keyframe chain (synthetic clip with a
7 deg pan: 300 px error with quality 0.6).  Keyframes are only taken from
solid direct registrations (`keyframe_min_quality`).

Quality: q = clip((inlier_ratio - 0.3) / 0.4, 0, 1) * exp(-sigma_T / 0.25 px),
sigma_T = rms * sqrt(2 / n_inliers) ('legacy': exp(-rms / 1 px)).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from pitchspeed.vision import geometry as G

log = logging.getLogger("pitchspeed.vision")


# Static-overlay (date stamp) detection limits
OVERLAY_MIN_CAMERA_MOTION_PX = 0.8   # camera must visibly move to tell overlay from scene
OVERLAY_MAX_AREA_FRAC = 0.08         # a date stamp covers a few % of the frame, not a scene


@dataclass
class StabilizerOptions:
    max_features: int = 300
    feature_quality: float = 0.01
    min_distance: int = 8
    lk_win: tuple = (31, 15)          # (cols, rows) in sample grid (larger window: less LK noise on low-texture, noisy footage)
    lk_levels: int = 3
    ransac_thr: float = 1.0           # square px; minimum inlier threshold
    # Noise-adaptive inlier threshold: the threshold is raised to 2.5 x the
    # per-feature LK noise (estimated from the residuals of the first fit),
    # at most `ransac_thr_max`.  0 disables.
    ransac_thr_max: float = 3.0
    presmooth_sigma: float = 0.0      # Gaussian pre-blur (px) of images before features / LK; 0 = off
    changed_patch_factor: float = 2.5  # drop features whose LK patch error > this x the 30th percentile; 0 = off
    quality_mode: str = "precision"   # 'precision' (transform precision) | 'legacy' (feature rms)
    min_inliers: int = 30
    min_quality: float = 0.05
    keyframe_every: int = 30
    fail_run_for_reanchor: int = 6
    reanchor_every: int = 15
    homography: str = "auto"          # 'auto' (only when clearly better) | 'prefer' | 'never'
    max_jump_frac: float = 0.06       # max image motion between consecutive samples (x width), 0 = no check
    keyframe_min_quality: float = 0.15  # only registrations at least this good become keyframes


@dataclass
class StabResult:
    A: np.ndarray                     # 3x3, sample frame px -> reference px (similarity or homography)
    quality: float                    # 0..1
    n_inliers: int = 0
    rms: float = float("nan")
    method: str = "ref"               # 'ref' | 'ref_pc' | 'key' | 'akaze' | 'hold' | 'identity'
    ok: bool = True
    params: dict = field(default_factory=dict)
    homography: bool = False


def _quality(n_inl: int, n_tot: int, rms: float, mode: str = "precision") -> float:
    """Registration quality 0..1 = support (inlier share) x precision.

    'legacy': exp(-rms / 1 px) with rms the per-feature residual RMS.  That
    punishes noisy but numerous features: on a zoomed, low-contrast,
    noisy behind view (LK noise ~1.5 px per feature) a registration that is
    good to 0.2 px scored < 0.05 and the sample was dropped.
    'precision': exp(-sigma_T / 0.25 px) with sigma_T = rms * sqrt(2 / n_inl),
    the approximate 1-sigma error of the fitted transform itself
    (translation + rotation/zoom lever arm), which is what matters for
    background subtraction and stabilized track coordinates.
    """
    if n_tot <= 0 or n_inl <= 0:
        return 0.0
    ratio = n_inl / n_tot
    support = float(np.clip((ratio - 0.3) / 0.4, 0.0, 1.0))
    rms = max(float(rms), 0.0) if np.isfinite(rms) else 1.0
    if mode == "legacy":
        return float(support * math.exp(-rms / 1.0))
    sig_t = rms * math.sqrt(2.0 / n_inl)
    return float(support * math.exp(-sig_t / 0.25))


class Stabilizer:
    """Streaming registration of samples to the calibration reference frame."""

    def __init__(self, reference_frame: np.ndarray, sar: float, interlaced: bool,
                 static_mask: Optional[np.ndarray] = None, opts: Optional[StabilizerOptions] = None):
        self.opts = opts or StabilizerOptions()
        self.sar = float(sar)
        self.interlaced = bool(interlaced)
        self.row_scale = 2 if self.interlaced else 1
        ref = np.ascontiguousarray(reference_frame)
        if ref.ndim == 3:
            ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        self.ref = ref
        H, W = ref.shape
        self.H, self.W = H, W
        if static_mask is None:
            static_mask = np.full((H, W), 255, np.uint8)
        self.static_mask = static_mask.astype(np.uint8)
        parities = (0, 1) if self.interlaced else (0,)
        self.ref_fields = {}
        self.ref_pts = {}
        for p in parities:
            img = self._pre(np.ascontiguousarray(ref[p::2]) if self.interlaced else ref)
            m = np.ascontiguousarray(self.static_mask[p::2]) if self.interlaced else self.static_mask
            self.ref_fields[p] = img
            self.ref_pts[p] = self._features(img, m)
        self._flags = {p: np.zeros(len(self.ref_pts[p]), bool) for p in parities}
        self._votes = {p: np.zeros(len(self.ref_pts[p]), np.int32) for p in parities}
        # mid-exposure correction per parity (identity for progressive)
        self.M = {p: np.eye(3) for p in parities}
        self.field_offset = {"available": False}
        if self.interlaced:
            self._init_midpoint()
        self.A_prev: Optional[np.ndarray] = None
        self.keyframes: dict = {}
        self.fail_run = 0
        self.n_since_reanchor = 0
        self._akaze_ref = None
        self.stats = {"ref": 0, "ref_pc": 0, "key": 0, "akaze": 0, "hold": 0, "identity": 0}

    # --------------------------------------------------------------- setup
    def _pre(self, img: np.ndarray) -> np.ndarray:
        """Optional Gaussian pre-smoothing (sample grid; rows of a field are 2 frame rows apart)."""
        s = float(self.opts.presmooth_sigma or 0.0)
        if s <= 0:
            return img
        return cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=max(s / self.row_scale, 0.3))

    def _features(self, img: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        """Shi-Tomasi corners per tile (4 x 3 grid) for good spatial spread."""
        o = self.opts
        h, w = img.shape[:2]
        nx, ny = 4, 3
        per = max(4, o.max_features // (nx * ny))
        out = []
        for iy in range(ny):
            for ix in range(nx):
                y0, y1 = iy * h // ny, (iy + 1) * h // ny
                x0, x1 = ix * w // nx, (ix + 1) * w // nx
                sub_m = None if mask is None else np.ascontiguousarray(mask[y0:y1, x0:x1])
                if sub_m is not None and not sub_m.any():
                    continue
                pts = cv2.goodFeaturesToTrack(np.ascontiguousarray(img[y0:y1, x0:x1]), per, o.feature_quality,
                                              o.min_distance, mask=sub_m, blockSize=7)
                if pts is not None:
                    out.append(pts.reshape(-1, 2) + [x0, y0])
        if not out:
            return np.zeros((0, 1, 2), np.float32)
        return np.concatenate(out).astype(np.float32).reshape(-1, 1, 2)

    def _init_midpoint(self) -> None:
        """Measure the shake between the reference frame's two fields."""
        T, n, rms = self._lk_similarity(self.ref_fields[0], 0, self.ref_pts[0], self.ref_fields[1], 1,
                                        np.eye(3))
        if T is None or n < self.opts.min_inliers:
            log.info("reference-frame field offset not measurable; assuming none")
            return
        # T maps bottom-field frame px -> top-field frame px.
        # With x_top = T(x_bot) and H o H = T:  x_mid = H(x_bot) = H^-1(x_top).
        T = G.local_affine(T, (self.W / 2.0, self.H / 2.0))
        Hh = G.to3(G.half_transform(T, self.sar))
        self.M[1] = Hh
        self.M[0] = np.linalg.inv(Hh)
        self.field_offset = {"available": True, "n_inliers": int(n), "rms": float(rms),
                             **G.similarity_params(T, self.sar)}

    # ------------------------------------------------------------ core LK
    def _lk_track(self, src_img, src_par, src_pts, dst_img, dst_par, A_dst_to_src_guess):
        """Track src_pts (src grid) into dst.

        A_dst_to_src_guess: 3x3 guess mapping dst frame px -> src frame px.
        Returns (idx, src_f, dst_f): indices of tracked features and their
        frame-px positions in src and dst, or None.
        """
        o = self.opts
        if src_pts is None or len(src_pts) < 8:
            return None
        rs = self.row_scale
        Fu_s = G.grid_to_frame(rs, src_par)
        Fd_d = G.frame_to_grid(rs, dst_par)
        Fu_d = G.grid_to_frame(rs, dst_par)
        P = src_pts.reshape(-1, 2).astype(np.float64)
        M = Fd_d @ np.linalg.inv(G.to3(A_dst_to_src_guess)) @ Fu_s
        init = G.apply(M, P).astype(np.float32).reshape(-1, 1, 2)
        nxt, st, err = cv2.calcOpticalFlowPyrLK(
            src_img, dst_img, src_pts, init.copy(), winSize=tuple(o.lk_win), maxLevel=o.lk_levels,
            criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        if nxt is None:
            return None
        st = st.ravel().astype(bool)
        Q = nxt.reshape(-1, 2).astype(np.float64)
        h, w = dst_img.shape[:2]
        st &= (Q[:, 0] >= 0) & (Q[:, 0] <= w - 1) & (Q[:, 1] >= 0) & (Q[:, 1] <= h - 1)
        if o.changed_patch_factor and err is not None and st.sum() >= 16:
            # A feature whose patch no longer matches (a player walked in front of it,
            # it sits on a moving object) carries no camera-motion information: its
            # dissimilarity (mean |difference| after alignment) is far above the noise
            # level of the matched patches.  Measured on the synthetic behind views:
            # err > 2.5 x median -> 92-100 % of such features are > 3 px off.
            with np.errstate(invalid="ignore"):        # lost features carry NaN errors
                e = np.asarray(err, np.float64).ravel()
            fin = st & np.isfinite(e)
            if fin.sum() >= 16:
                ref_e = float(np.percentile(e[fin], 30))
                if ref_e > 0:
                    st &= fin & (e <= o.changed_patch_factor * ref_e)
        idx = np.nonzero(st)[0]
        if len(idx) < 8:
            return None
        src_f = P[idx] @ Fu_s[:2, :2].T + Fu_s[:2, 2]
        dst_f = Q[idx] @ Fu_d[:2, :2].T + Fu_d[:2, 2]
        return idx, src_f, dst_f

    def _adaptive_thr(self, resid: np.ndarray, thr0: float) -> float:
        """Inlier threshold matched to the per-feature LK noise.

        resid: |residual| (square px) of every tracked feature under a first
        fit.  For isotropic Gaussian feature noise sigma, |r| is Rayleigh
        distributed with median 1.1774 sigma; the median is taken over the
        features within 2 x ransac_thr_max (gross outliers such as a date
        stamp or a moving player excluded).  Returns 2.5 sigma (95.6 % of
        true inliers) clipped to [thr0, ransac_thr_max].
        """
        tmax = float(self.opts.ransac_thr_max or 0.0)
        if tmax <= thr0:
            return thr0
        near = resid[resid < 2.0 * tmax]
        if len(near) < 12:
            return thr0
        sigma = float(np.median(near)) / 1.1774
        return float(np.clip(2.5 * sigma, thr0, tmax))

    def _fit(self, src_f, dst_f, thr=None):
        """Similarity dst frame px -> src frame px (RANSAC in square px).

        The inlier threshold starts at `thr` (default ransac_thr) and is
        raised to the measured feature noise level (`_adaptive_thr`) when
        the features are noisy (low contrast / heavy tape noise): with a
        fixed 1 px threshold most correct features of such footage are
        "outliers" and the registration is rejected although it is precise.
        Returns (A 2x3 or None, inlier mask, rms).
        """
        o = self.opts
        if len(src_f) < 4:
            return None, np.zeros(len(src_f), bool), float("nan")
        D = G.sq_matrix(self.sar)
        src_q = src_f * [self.sar, 1.0]
        dst_q = dst_f * [self.sar, 1.0]
        thr0 = thr or o.ransac_thr
        S, inl = cv2.estimateAffinePartial2D(dst_q, src_q, method=cv2.RANSAC,
                                             ransacReprojThreshold=thr0, maxIters=2000,
                                             confidence=0.995, refineIters=10)
        if S is None or inl is None:
            return None, np.zeros(len(src_f), bool), float("nan")
        inl = inl.ravel().astype(bool)
        self._last_thr = thr0
        if o.ransac_thr_max and o.ransac_thr_max > thr0 and len(src_f) >= 12:
            r_all = np.linalg.norm(dst_q @ S[:, :2].T + S[:, 2] - src_q, axis=1)
            thr1 = self._adaptive_thr(r_all, thr0)
            if thr1 > 1.05 * thr0:
                S2, inl2 = cv2.estimateAffinePartial2D(dst_q, src_q, method=cv2.RANSAC,
                                                       ransacReprojThreshold=thr1, maxIters=2000,
                                                       confidence=0.995, refineIters=10)
                if S2 is not None and inl2 is not None and inl2.sum() >= inl.sum():
                    S, inl = S2, inl2.ravel().astype(bool)
                    self._last_thr = thr1
        pred = dst_q[inl] @ S[:, :2].T + S[:, 2]
        rms = float(np.sqrt(np.mean(np.sum((pred - src_q[inl]) ** 2, axis=1)))) if inl.any() else float("nan")
        return (np.linalg.inv(D) @ G.to3(S) @ D)[:2], inl, rms

    def _fit_h(self, src_f, dst_f, thr=None):
        """Homography dst frame px -> src frame px (RANSAC in square px); sanity-checked."""
        o = self.opts
        if len(src_f) < 12:
            return None, np.zeros(len(src_f), bool), float("nan")
        D = G.sq_matrix(self.sar)
        src_q = src_f * [self.sar, 1.0]
        dst_q = dst_f * [self.sar, 1.0]
        Hq, inl = cv2.findHomography(dst_q, src_q, cv2.RANSAC, thr or o.ransac_thr, maxIters=2000,
                                     confidence=0.995)
        if Hq is None or inl is None or abs(Hq[2, 2]) < 1e-9:
            return None, np.zeros(len(src_f), bool), float("nan")
        Hq = Hq / Hq[2, 2]
        # a camcorder rotating by a few tens of degrees at most: small projective terms, no flips
        det = float(np.linalg.det(Hq[:2, :2]))
        if not (0.5 < det < 2.0) or abs(Hq[2, 0]) > 3e-3 or abs(Hq[2, 1]) > 3e-3:
            return None, np.zeros(len(src_f), bool), float("nan")
        inl = inl.ravel().astype(bool)
        pred = G.apply(Hq, dst_q[inl])
        rms = float(np.sqrt(np.mean(np.sum((pred - src_q[inl]) ** 2, axis=1)))) if inl.any() else float("nan")
        Hf = np.linalg.inv(D) @ Hq @ D
        return Hf / Hf[2, 2], inl, rms

    def _lk_similarity(self, src_img, src_par, src_pts, dst_img, dst_par, A_dst_to_src_guess, flags=None):
        """Track + fit; `flags` (bool per src feature) excludes overlay features.

        Returns (A 3x3 or None, n_inliers, rms) and sets self._last_ntot.
        A is a similarity unless a homography explains clearly more features.
        """
        tr = self._lk_track(src_img, src_par, src_pts, dst_img, dst_par, A_dst_to_src_guess)
        if tr is None:
            return None, 0, float("nan")
        idx, src_f, dst_f = tr
        if flags is not None:
            use = ~flags[idx]
            idx, src_f, dst_f = idx[use], src_f[use], dst_f[use]
        A, inl, rms = self._fit(src_f, dst_f)
        thr_sim = getattr(self, "_last_thr", None)
        self._last_ntot = len(idx)
        if A is None:
            return None, 0, float("nan")
        if flags is not None:
            A, inl, rms = self._overlay_check(A, inl, rms, idx, src_f, dst_f, flags, src_par)
        A = G.to3(A)
        self._last_h = False
        n = len(src_f)
        mode = self.opts.homography
        self._n_calls = getattr(self, "_n_calls", 0) + 1
        # try the homography when the similarity leaves outliers -- every sample while the
        # previous one needed it, otherwise every 4th sample (cost ~2.5 ms per try)
        want = mode == "prefer" or (inl.sum() < 0.92 * n and (
            getattr(self, "_prev_h", False) or self._n_calls % 4 == 0))
        if mode != "never" and n >= 20 and want:
            keep = ~flags[idx] if flags is not None else np.ones(n, bool)
            # same inlier threshold as the similarity, so the inlier counts are comparable
            Hh, inlh, rmsh = self._fit_h(src_f[keep], dst_f[keep], thr=thr_sim)
            if Hh is not None:
                if mode == "prefer":
                    better = inlh.sum() >= max(40, 0.95 * inl.sum())
                else:
                    better = inlh.sum() >= inl.sum() + max(6, 0.08 * n)
                if better:
                    self._last_h = True
                    return Hh, int(inlh.sum()), rmsh
        return A, int(inl.sum()), rms

    def _overlay_check(self, A, inl, rms, idx, src_f, dst_f, flags, p):
        """Detect a compact set of features with zero raw motion (date stamp)."""
        raw = np.linalg.norm(dst_f - src_f, axis=1)          # raw displacement (ref vs sample)
        votes = self._votes[p]
        out = ~inl
        hyp = None
        if out.sum() >= 8:
            A2, inl2, rms2 = self._fit(src_f[out], dst_f[out])
            if A2 is not None and inl2.sum() >= 6:
                sel2 = np.nonzero(out)[0][inl2]
                hyp = (A2, sel2, rms2)
        cands = []
        if hyp is not None:
            sel1 = np.nonzero(inl)[0]
            for (Ah, sel, r) in ((A, sel1, rms), (hyp[0], hyp[1], hyp[2])):
                ident = float(np.median(raw[sel])) < 0.3
                pts = src_f[sel]
                spread = float(np.sqrt(max(np.linalg.det(np.cov(pts.T) + 1e-6 * np.eye(2)), 0.0))) if len(sel) > 2 else 0.0
                cands.append((Ah, sel, r, ident, spread))
            (A1, s1, r1, i1, sp1), (A2, s2, r2, i2, sp2) = cands
            # The camera hypothesis must move clearly (sub-pixel shake is ambiguous),
            # and the zero-motion group must be small and compact like a date stamp:
            # on a near-static tripod shot the whole scene has ~zero raw motion and
            # must never be taken for an overlay.
            moving = lambda Ah, sel: float(np.median(np.linalg.norm(
                G.apply(G.invert(Ah), src_f[sel]) - src_f[sel], axis=1))) > OVERLAY_MIN_CAMERA_MOTION_PX
            if i1 and not i2 and sp2 > sp1 and self._compact(src_f[s1]) and moving(A2, s2):
                votes[idx[s1]] += 1
                A, inl, rms = A2, np.isin(np.arange(len(src_f)), s2), r2
            elif i2 and not i1 and sp1 > sp2 and self._compact(src_f[s2]) and moving(A1, s1):
                votes[idx[s2]] += 1
        newly = (votes >= 3) & ~flags
        if newly.any():
            flags |= newly
            log.info("flagged %d static-overlay features (date stamp?) in parity %d", int(newly.sum()), p)
        return A, inl, rms

    def _compact(self, pts: np.ndarray) -> bool:
        """True if field-grid points span a small box (<= OVERLAY_MAX_AREA_FRAC of the frame)."""
        if len(pts) == 0:
            return False
        w = float(np.ptp(pts[:, 0]))
        h = float(np.ptp(pts[:, 1])) * self.row_scale
        return (w * h <= OVERLAY_MAX_AREA_FRAC * self.W * self.H and w <= 0.5 * self.W
                and h <= 0.35 * self.H)

    def overlay_boxes(self, pad: float = 10.0) -> list:
        """Bounding boxes (frame px, x0, y0, x1, y1) of flagged overlay feature clusters."""
        pts = []
        for p, fl in self._flags.items():
            if fl.any():
                P = self.ref_pts[p].reshape(-1, 2)[fl].astype(np.float64)
                P[:, 1] = P[:, 1] * self.row_scale + p
                pts.append(P)
        if not pts:
            return []
        P = np.concatenate(pts)
        # cluster by proximity (60 px)
        boxes: list = []
        for x, y in P:
            for b in boxes:
                if b[0] - 60 <= x <= b[2] + 60 and b[1] - 60 <= y <= b[3] + 60:
                    b[0], b[1], b[2], b[3] = min(b[0], x), min(b[1], y), max(b[2], x), max(b[3], y)
                    break
            else:
                boxes.append([x, y, x, y])
        out = []
        for b in boxes:
            if (b[2] - b[0]) * (b[3] - b[1]) > OVERLAY_MAX_AREA_FRAC * self.W * self.H:
                log.info("ignoring implausibly large overlay region %s", [round(v) for v in b])
                continue
            out.append((b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad))
        return out

    # ------------------------------------------------------------ public
    def register(self, image: np.ndarray, parity: int) -> StabResult:
        """Register one sample image (grid of the given parity) to the reference."""
        p = int(parity) if self.interlaced else 0
        img = self._pre(np.ascontiguousarray(image))
        pred = G.to3(self.A_prev) if self.A_prev is not None else np.eye(3)
        res = self._continuous(self._try_ref(img, p, pred, "ref"))
        if res is None:
            res = self._continuous(self._try_phase(img, p, pred))
        if res is None:
            res = self._continuous(self._try_keyframe(img, p))
        if res is None and self.fail_run >= self.opts.fail_run_for_reanchor:
            self.n_since_reanchor += 1
            if self.n_since_reanchor >= self.opts.reanchor_every or self.fail_run == self.opts.fail_run_for_reanchor:
                self.n_since_reanchor = 0
                res = self._try_akaze(img, p)
        self._prev_h = bool(res is not None and res.homography)
        if res is None:
            self.fail_run += 1
            A = self.A_prev if self.A_prev is not None else np.eye(3)
            res = StabResult(A=G.to3(A), quality=0.0, method="hold", ok=False)
        else:
            self.fail_run = 0
            self.A_prev = res.A
            # Keyframes only from solid direct registrations: a keyframe's transform is
            # inherited by every chained ('key') registration until the next refresh.
            if res.method in ("ref", "ref_pc", "akaze") and res.quality >= self.opts.keyframe_min_quality:
                self.keyframes[p] = {"img": img.copy(), "A": res.A.copy(), "pts": None, "age": 0}
            elif res.method == "key" and p in self.keyframes:
                kf = self.keyframes[p]
                kf["age"] += 1
                if kf["age"] >= self.opts.keyframe_every:      # bound the chain length
                    self.keyframes[p] = {"img": img.copy(), "A": res.A.copy(), "pts": None, "age": 0}
        self.stats[res.method] = self.stats.get(res.method, 0) + 1
        res.params = G.similarity_params(G.local_affine(res.A, (self.W / 2.0, self.H / 2.0)), self.sar)
        return res

    def _continuous(self, res: Optional[StabResult]) -> Optional[StabResult]:
        """Reject a registration that jumps implausibly far from the previous sample's.

        A camera moves continuously: between two fields (1/60 s) even a fast
        telephoto pan or strong shake moves the image by a few tens of px.  A
        result that puts the image hundreds of px away from where the
        previous sample was registered is a false match -- typically a
        periodic backstop-fence lattice or static foreground (people, date
        stamp) locking phase correlation / LK onto the wrong solution after a
        pan.  Measured at the four image corners (catches zoom / roll jumps
        too); the allowance grows with the number of failed samples since
        the last registration.  AKAZE re-anchoring after a failure run is
        exempt (it exists to re-acquire after scene cuts / big moves).
        """
        if res is None or self.A_prev is None:
            return res
        lim = float(self.opts.max_jump_frac) * self.W * (1 + self.fail_run)
        if lim <= 0:
            return res
        C = np.array([[0.0, 0.0], [self.W - 1.0, 0.0], [0.0, self.H - 1.0], [self.W - 1.0, self.H - 1.0]])
        d = (G.apply(res.A, C) - G.apply(self.A_prev, C)) * [self.sar, 1.0]
        jump = float(np.max(np.linalg.norm(d, axis=1)))
        if jump > lim:
            self.stats["rejected_jump"] = self.stats.get("rejected_jump", 0) + 1
            return None
        return res

    # --------------------------------------------------------- strategies
    def _accept(self, A_p, n_inl, rms, p, method, qmul=1.0) -> Optional[StabResult]:
        if A_p is None or n_inl < self.opts.min_inliers:
            return None
        q = _quality(n_inl, getattr(self, "_last_ntot", n_inl), rms, self.opts.quality_mode) * qmul
        if q < self.opts.min_quality:
            return None
        A = self.M[p] @ G.to3(A_p)
        return StabResult(A=A / A[2, 2], quality=q, n_inliers=int(n_inl), rms=float(rms), method=method,
                          homography=bool(getattr(self, "_last_h", False)))

    def _try_ref(self, img, p, pred_mid, method) -> Optional[StabResult]:
        # guess for sample frame -> reference *field* frame
        guess = np.linalg.inv(self.M[p]) @ pred_mid
        A_p, n, rms = self._lk_similarity(self.ref_fields[p], p, self.ref_pts[p], img, p, guess,
                                          flags=self._flags[p])
        return self._accept(A_p, n, rms, p, method)

    def _try_phase(self, img, p, pred_mid) -> Optional[StabResult]:
        ref = self.ref_fields[p].astype(np.float32)
        cur = img.astype(np.float32)
        win = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(ref, cur, win)
        if resp < 0.05:
            return None
        # cur(x) ~ ref(x - d): sample frame -> ref frame is a translation by -d (grid -> frame rows)
        guess = G.translate(-dx, -dy * self.row_scale)
        A_p, n, rms = self._lk_similarity(self.ref_fields[p], p, self.ref_pts[p], img, p, guess,
                                          flags=self._flags[p])
        return self._accept(A_p, n, rms, p, "ref_pc")

    def _try_keyframe(self, img, p) -> Optional[StabResult]:
        kf = self.keyframes.get(p)
        if kf is None:
            return None
        if kf["pts"] is None:
            m = np.full(kf["img"].shape, 255, np.uint8)
            m[:, :12] = 0
            m[:, -12:] = 0
            m[-6:] = 0
            # exclude the corridor region (mapped from the reference) to avoid the moving players
            if self.interlaced:
                ref_m = self.static_mask[p::2]
            else:
                ref_m = self.static_mask
            M = G.frame_to_grid(self.row_scale, p) @ np.linalg.inv(G.to3(kf["A"])) @ \
                G.grid_to_frame(self.row_scale, p)
            warped = cv2.warpPerspective(ref_m, M, (m.shape[1], m.shape[0]), flags=cv2.INTER_NEAREST,
                                         borderValue=255)
            m = cv2.bitwise_and(m, warped)
            kf["pts"] = self._features(kf["img"], m)
        A_k, n, rms = self._lk_similarity(kf["img"], p, kf["pts"], img, p, np.eye(3))
        if A_k is None:
            return None
        # sample -> keyframe -> reference
        A = G.to3(kf["A"]) @ G.to3(A_k)
        res = self._accept(A_k, n, rms, p, "key", qmul=0.7)
        if res is None:
            return None
        res.A = A / A[2, 2]
        return res                      # keyframe age / refresh: in register(), once accepted

    def _try_akaze(self, img, p) -> Optional[StabResult]:
        try:
            det = cv2.AKAZE_create()
        except Exception:      # pragma: no cover - very old OpenCV
            return None
        if self._akaze_ref is None:
            self._akaze_ref = {}
        if p not in self._akaze_ref:
            m = np.ascontiguousarray(self.static_mask[p::2]) if self.interlaced else self.static_mask
            self._akaze_ref[p] = det.detectAndCompute(self.ref_fields[p], m)
        kr, dr = self._akaze_ref[p]
        kc, dc = det.detectAndCompute(img, None)
        if dr is None or dc is None or len(kr) < 10 or len(kc) < 10:
            return None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        good = []
        for pair in bf.knnMatch(dc, dr, k=2):
            if len(pair) == 2 and pair[0].distance < 0.8 * pair[1].distance:
                good.append(pair[0])
        if len(good) < 15:
            return None
        rs = self.row_scale
        cur = np.array([kc[g.queryIdx].pt for g in good], np.float64)
        ref = np.array([kr[g.trainIdx].pt for g in good], np.float64)
        cur[:, 1] = cur[:, 1] * rs + p
        ref[:, 1] = ref[:, 1] * rs + p
        S, inl = cv2.estimateAffinePartial2D(cur * [self.sar, 1], ref * [self.sar, 1], method=cv2.RANSAC,
                                             ransacReprojThreshold=2.0, maxIters=3000, confidence=0.995)
        if S is None or inl is None or inl.sum() < 12:
            return None
        D = G.sq_matrix(self.sar)
        A_p = np.linalg.inv(D) @ G.to3(S) @ D
        # refine with LK from the AKAZE guess
        A_r, n, rms = self._lk_similarity(self.ref_fields[p], p, self.ref_pts[p], img, p, G.to3(A_p),
                                          flags=self._flags[p])
        res = self._accept(A_r, n, rms, p, "akaze")
        if res is not None:
            log.info("re-registered to the reference frame after %d failed samples", self.fail_run)
        return res
