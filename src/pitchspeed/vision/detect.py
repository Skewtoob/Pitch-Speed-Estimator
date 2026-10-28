"""Stabilized background model + ball candidate detection (one sample at a time).

For sample c (parity p) the background is the per-pixel median of the
same-parity neighbours c +- {4, 6, 8, 10} fields, each warped into c's own
sample grid with the stabilization transforms (the current sample itself is
never resampled, so the ball keeps its shape).  Only the corridor bounding
box (mapped into c's grid) is processed.

    D = cur - (a * B + b)          robust gain/offset fit (camcorder AGC)
    R = blur(D)                    small anisotropic pre-blur
    z = max_k |R * k_k| / (c_k * sigma_loc)   matched line kernels, both signs

`sigma_loc` is a block-median noise scale (floored at the global median);
`c_k` converts it to the noise of kernel k's response.  Each kernel's
response counts only where the kernel fits inside the ROI (so a ball next
to the image edge is still seen by the small kernels); kernels for balls of
>= `decimate_min_d` px (zoomed behind views) run on a 2x column-decimated
grid (`KernelSpec.sub`).  Peaks after 5x11
non-maximum suppression become candidates; a clutter map (in reference
coordinates) down-weights places that fire persistently.

Candidate positions are centroids of max(sign*R - 1.5 sigma, 0) around the
peak (mid-exposure position of a streak), mapped to reference px.
`attached` marks candidates touching a large |R| > 3 sigma blob (arm, glove,
body).  The per-sample z map is kept (int16) for pass-2 re-detection.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from pitchspeed.vision import geometry as G
from pitchspeed.vision.geometry import ViewGeometry
from pitchspeed.vision.options import FinderOptions

CAND_DTYPE = np.dtype([
    ("uid", np.int64), ("s", np.int64), ("t", np.float64), ("k", np.int8),
    ("x", np.float64), ("y", np.float64), ("z", np.float32), ("zraw", np.float32),
    ("sign", np.int8), ("length", np.float32), ("angle", np.float32), ("attached", np.bool_),
    ("sigma", np.float32), ("prog", np.float32), ("kidx", np.int16),
])


# ---------------------------------------------------------------- kernels ----
def line_kernel(length: float, angle: float, width: float, sar: float = 1.0,
                row_scale: int = 1) -> np.ndarray:
    """Zero-mean, unit-norm line-segment kernel on the sample grid.

    length: segment length (square px, 0 = round blob); angle: radians in
    square-pixel image space (x right, y down); width: Gaussian sigma
    (square px) across the segment.  Grid column step = sar square px,
    row step = row_scale frame rows.
    """
    L = max(0.0, float(length))
    c, s = math.cos(angle), math.sin(angle)
    hx = int(math.ceil((L / 2 * abs(c) + 3 * width) / sar))
    hy = int(math.ceil((L / 2 * abs(s) + 3 * width) / row_scale))
    hx, hy = max(hx, 1), max(hy, 1)
    ii, jj = np.mgrid[-hy:hy + 1, -hx:hx + 1].astype(np.float64)
    dx, dy = jj * sar, ii * row_scale
    t = np.clip(dx * c + dy * s, -L / 2, L / 2)
    d2 = (dx - t * c) ** 2 + (dy - t * s) ** 2
    k = np.exp(-d2 / (2 * width * width))
    k -= k.mean()
    n = np.sqrt((k * k).sum())
    if n <= 0:
        k[hy, hx] = 1.0
        n = 1.0
    return (k / n).astype(np.float32)


@dataclass
class KernelSpec:
    length: float
    angle: float
    width: float
    kernel: np.ndarray
    diam: float = 0.0          # ball diameter (px) the kernel is matched to (behind view: used to
                               # restrict it to the part of the corridor where that size is expected)
    sep: Optional[tuple] = None   # (gx, gy, mean, norm) when the kernel is a separable round blob
    sub: Optional["KernelSpec"] = None   # same kernel on a 2x column-decimated grid (large balls)

    def apply(self, R: np.ndarray) -> np.ndarray:
        """Correlate R with the kernel (separable fast path for round blobs).

        With `sub` set (a ball of >= ~7 px, zoomed behind views) the
        correlation runs on R averaged over column pairs and the response is
        interpolated back: a matched filter that wide has no content at the
        frequencies the decimation removes, and it costs 2-4x less.  The
        noise normalization (Detector._ck) measures the response of this same
        path, so z stays consistent.
        """
        if self.sub is not None:
            h, w = R.shape
            wd = max(1, (w + 1) // 2)
            Rd = cv2.resize(R, (wd, h), interpolation=cv2.INTER_AREA)
            return cv2.resize(self.sub.apply(Rd), (w, h), interpolation=cv2.INTER_LINEAR)
        if self.sep is not None:
            gx, gy, mean, norm = self.sep
            kh, kw = self.kernel.shape
            out = cv2.sepFilter2D(R, cv2.CV_32F, gx, gy, borderType=cv2.BORDER_REPLICATE)
            box = cv2.boxFilter(R, cv2.CV_32F, (kw, kh), normalize=False, borderType=cv2.BORDER_REPLICATE)
            return (out - np.float32(mean) * box) * np.float32(1.0 / norm)
        return cv2.filter2D(R, cv2.CV_32F, self.kernel, borderType=cv2.BORDER_REPLICATE)


def _blob_spec(width: float, sar: float, row_scale: int, diam: float = 0.0) -> KernelSpec:
    k = line_kernel(0.0, 0.0, width, sar, row_scale)
    hy, hx = k.shape[0] // 2, k.shape[1] // 2
    gx = np.exp(-((np.arange(-hx, hx + 1) * sar) ** 2) / (2 * width * width)).astype(np.float32)
    gy = np.exp(-((np.arange(-hy, hy + 1) * row_scale) ** 2) / (2 * width * width)).astype(np.float32)
    g = np.outer(gy, gx)
    mean = float(g.mean())
    norm = float(np.sqrt(((g - mean) ** 2).sum()))
    return KernelSpec(0.0, 0.0, width, k, diam=diam, sep=(gx, gy, mean, norm))


def _decimated(spec: KernelSpec, sar: float, row_scale: int) -> KernelSpec:
    """Attach the 2x column-decimated version of a kernel (grid column step 2 * sar)."""
    if spec.length <= 0:
        spec.sub = _blob_spec(spec.width, 2.0 * sar, row_scale, diam=spec.diam)
    else:
        spec.sub = KernelSpec(spec.length, spec.angle, spec.width,
                              line_kernel(spec.length, spec.angle, spec.width, 2.0 * sar, row_scale), diam=spec.diam)
    return spec


def kernel_bank(geom: ViewGeometry, row_scale: int, decimate_min_d: float = 0.0) -> list[KernelSpec]:
    """Kernel bank for the view (cv.md section 3, reduced for speed).

    decimate_min_d > 0 (behind view): kernels matched to a ball of at least
    that diameter (px) are evaluated on a 2x column-decimated grid.
    """
    sar = geom.sar
    bank: list[KernelSpec] = []
    if geom.view == "side":
        d = geom.d_rel
        w = max(0.8, d / 2.5)
        # typical image displacement per sample at ~70 mph
        dbar = 70.0 * G.MPH * geom.scale_rel * geom.sample_dt
        ax = math.atan2(geom.axis_sq[1], geom.axis_sq[0])
        lens = sorted({float(np.clip(d + f * dbar, 3.0, 30.0)) for f in (0.15, 0.5, 1.0)})
        bank.append(_blob_spec(w, sar, row_scale))
        for i, L in enumerate(lens):
            angs = (0.0,) if i == 0 else (0.0, -math.radians(20), math.radians(20))
            for da in angs:
                a = ax + da
                bank.append(KernelSpec(L, a, w, line_kernel(L, a, w, sar, row_scale)))
    else:
        d0, d1 = geom.d_rel, geom.d_catch
        sizes = sorted({round(float(v), 1) for v in (d0, math.sqrt(d0 * d1), d1)})
        for d in sizes:
            w = max(0.8, d / 2.5)
            bank.append(_blob_spec(w, sar, row_scale, diam=d))
        dm = math.sqrt(d0 * d1)
        w = max(0.8, dm / 2.5)
        ax = math.atan2(geom.axis_sq[1], geom.axis_sq[0])
        # short streaks along the image axis of the pitch and +-45 deg (the ball comes
        # roughly along the release->catch line; perpendicular streaks are not useful)
        for a in (ax, ax - math.pi / 4, ax + math.pi / 4):
            L = float(np.clip(dm + 4.0, 3.0, 12.0))
            bank.append(KernelSpec(L, a, w, line_kernel(L, a, w, sar, row_scale), diam=dm))
        # longer streaks along the axis near the catch (fast image motion there)
        wl = max(0.8, d1 / 2.5)
        for L in (d1 + 8.0, d1 + 16.0):
            L = float(np.clip(L, 3.0, 30.0))
            bank.append(KernelSpec(L, ax, wl, line_kernel(L, ax, wl, sar, row_scale), diam=d1))
        if decimate_min_d > 0:
            bank = [_decimated(k, sar, row_scale) if k.diam >= decimate_min_d else k for k in bank]
    return bank


# ------------------------------------------------------------- primitives ----
def nms_peaks(z: np.ndarray, size: tuple = (5, 11), thr: float = 3.5, max_n: int = 30,
              valid: Optional[np.ndarray] = None):
    """Local maxima of z over a (rows, cols) window with z >= thr.

    Returns (rows, cols, values) sorted by value, at most max_n.
    """
    kr, kc = int(size[0]), int(size[1])
    dil = cv2.dilate(z, np.ones((kr, kc), np.uint8))
    m = (z >= dil) & (z >= thr)
    if valid is not None:
        m &= valid
    rr, cc = np.nonzero(m)
    if len(rr) == 0:
        return rr, cc, np.zeros(0, np.float32)
    v = z[rr, cc]
    o = np.argsort(-v, kind="stable")
    # plateau duplicates: keep the first of peaks closer than the window
    keep = []
    taken = np.zeros(len(o), bool)
    for ii, i in enumerate(o):
        if taken[ii]:
            continue
        keep.append(i)
        if len(keep) >= max_n:
            break
        close = (np.abs(rr[o] - rr[i]) < kr) & (np.abs(cc[o] - cc[i]) < kc)
        taken |= close
    keep = np.array(keep, dtype=np.int64)
    return rr[keep], cc[keep], v[keep]


def robust_gain(cur: np.ndarray, bg: np.ndarray, step: int = 4) -> tuple[float, float]:
    """Fit cur ~ a*bg + b robustly (AGC / exposure drift).  Clamped a in [0.7, 1.4]."""
    c = cur[::step, ::step].ravel().astype(np.float64)
    b = bg[::step, ::step].ravel().astype(np.float64)
    if len(c) < 20 or np.ptp(b) < 1.0:
        return 1.0, float(np.median(c - b)) if len(c) else 0.0
    a, off = 1.0, float(np.median(c - b))
    for _ in range(2):
        r = c - (a * b + off)
        mad = 1.4826 * float(np.median(np.abs(r - np.median(r)))) + 1e-6
        use = np.abs(r) < 3.0 * mad
        if use.sum() < 20:
            break
        bb, cc = b[use], c[use]
        vb = bb.var()
        if vb < 1.0:
            off = float(np.mean(cc - bb))
            a = 1.0
            break
        a = float(np.clip(np.cov(bb, cc, bias=True)[0, 1] / vb, 0.7, 1.4))
        off = float(np.mean(cc) - a * np.mean(bb))
    return a, off


CK_MAX_FACTOR = 8.0     # max measured/white-noise kernel gain ratio (correlated noise)

def block_noise(R: np.ndarray, block: tuple = (8, 32), floor: float = 0.0) -> tuple[np.ndarray, float]:
    """Local robust noise scale: 1.4826 * block median |R|, floored at the global value,
    then a 3x3 max over neighbouring blocks.

    The max filter keeps structure that straddles a block border (edges of
    players, the horizon, compression blocks) from producing false alarms in
    the quieter neighbour block; on the synthetic suite it cut candidates per
    sample ~3x at equal ball recall compared with no dilation or a lower
    quantile.  Price: a ball right next to a large moving body has a lower z
    (it is flagged `attached` anyway).  `floor` (grey levels) guards against
    noise-free / heavily quantized input.
    """
    h, w = R.shape
    ar = np.abs(R)
    g = max(1.4826 * float(np.median(ar[::2, ::3])), floor) + 1e-6
    bh, bw = int(block[0]), int(block[1])
    nh, nw = max(1, h // bh), max(1, w // bw)
    if h < bh or w < bw:
        return np.full(R.shape, g, dtype=np.float32), g
    a = ar[:nh * bh, :nw * bw:2].reshape(nh, bh, nw, bw // 2).transpose(0, 2, 1, 3).reshape(nh, nw, -1)
    k = a.shape[2] // 2
    blk = 1.4826 * np.partition(a, k, axis=2)[:, :, k].astype(np.float32)
    blk = np.maximum(blk, g)
    blk = cv2.dilate(blk, np.ones((3, 3), np.uint8))
    sig = cv2.resize(blk, (w, h), interpolation=cv2.INTER_LINEAR)
    return sig.astype(np.float32), g


# Optimal sorting network for 8 inputs (19 compare-exchange operations).
_NET8 = [(0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7), (0, 1), (2, 3), (4, 5), (6, 7),
         (2, 4), (3, 5), (1, 4), (3, 6), (1, 2), (3, 4), (5, 6)]


def median_stack(stack) -> np.ndarray:
    """Per-pixel median of a list/stack of equally shaped images (float32 result).

    8 inputs (the usual case) use a sorting network on cv2.min/max, which is
    several times faster than np.median / np.partition.
    """
    imgs = list(stack)
    n = len(imgs)
    if n == 1:
        return imgs[0].astype(np.float32)
    if n == 8:
        v = list(imgs)
        for i, j in _NET8:
            lo = cv2.min(v[i], v[j])
            v[j] = cv2.max(v[i], v[j])
            v[i] = lo
        return 0.5 * (v[3].astype(np.float32) + v[4].astype(np.float32))
    arr = np.stack(imgs)
    if n % 2 == 1:
        return np.partition(arr, n // 2, axis=0)[n // 2].astype(np.float32)
    part = np.partition(arr, [n // 2 - 1, n // 2], axis=0)
    return 0.5 * (part[n // 2 - 1].astype(np.float32) + part[n // 2].astype(np.float32))


# ------------------------------------------------------------- data types ----
@dataclass
class DetFrame:
    """Detection output of one sample."""
    s: int
    t: float
    k: int
    parity: int
    A: np.ndarray                      # 3x3 (or 2x3) sample frame px -> reference px
    quality: float
    cands: np.ndarray                  # CAND_DTYPE records
    roi: tuple = (0, 0, 0, 0)          # (c0, r0, w, h) in the sample grid
    zmap: Optional[np.ndarray] = None  # int16, signed z * 100 over the ROI
    rmap: Optional[np.ndarray] = None  # int8, signed R / sigma * 16 (debug strobe), optional
    sigma: float = float("nan")        # global residual noise (grey levels)
    e_rel: float = float("nan")        # foreground fraction near the release point
    e_catch: float = float("nan")      # ... near the catch point
    blobs: list = field(default_factory=list)   # big blobs: (x0, y0, x1, y1) reference px
    n_peaks: int = 0
    skipped: str = ""                  # reason when the sample was not processed

    def grid_to_ref(self, cols, rows, row_scale: int) -> np.ndarray:
        x = np.asarray(cols, np.float64) + self.roi[0]
        y = (np.asarray(rows, np.float64) + self.roi[1]) * row_scale + self.parity
        return G.apply(self.A, np.stack([x, y], axis=-1))

    def ref_to_grid(self, pts, row_scale: int) -> np.ndarray:
        """Reference px -> ROI grid (col, row) floats."""
        f = G.apply(G.invert(self.A), pts)
        return np.stack([f[:, 0] - self.roi[0], (f[:, 1] - self.parity) / row_scale - self.roi[1]], axis=-1)


@dataclass
class _Buf:
    s: int
    t: float
    k: int
    parity: int
    image: np.ndarray
    A: np.ndarray
    quality: float


# --------------------------------------------------------------- detector ----
class Detector:
    """Ring-buffered background subtraction + matched-filter candidates."""

    def __init__(self, geom: ViewGeometry, interlaced: bool, sample_shape: tuple,
                 opts: FinderOptions, ref_masks: Optional[list] = None,
                 raw_mask: Optional[np.ndarray] = None, keep_rmap: bool = False):
        self.geom = geom
        self.opts = opts
        self.interlaced = bool(interlaced)
        self.rs = 2 if self.interlaced else 1
        self.shape = tuple(sample_shape)
        if self.interlaced:
            self.offsets = tuple(int(o) for o in opts.bg_offsets)
            slow = tuple(int(o) for o in opts.bg_offsets_slow)
        else:
            self.offsets = tuple(max(1, int(o) // 2) for o in opts.bg_offsets)
            slow = tuple(max(1, int(o) // 2) for o in opts.bg_offsets_slow)
        self.use_slow = geom.view == "behind" and len(slow) > 0
        self.slow_offsets = slow if self.use_slow else ()
        self.max_off = max(self.offsets + self.slow_offsets)
        self.buf: "OrderedDict[int, _Buf]" = OrderedDict()
        self.next_s: Optional[int] = None
        self.bank = kernel_bank(geom, self.rs, float(getattr(opts, "decimate_min_d", 0.0) or 0.0))
        self.kmax_h = max(k.kernel.shape[0] for k in self.bank) // 2
        self.kmax_w = max(k.kernel.shape[1] for k in self.bank) // 2
        self._kmin_h = min(k.kernel.shape[0] for k in self.bank) // 2
        self._kmin_w = min(k.kernel.shape[1] for k in self.bank) // 2
        self._ck_white = self._white_noise_gain()
        self._ck = self._ck_white.copy()
        self._ck_age = 10 ** 9
        self.ref_masks = list(ref_masks or [])
        self.raw_mask = raw_mask        # uint8 (rows, cols) sample grid, 0 = ignore (date stamp)
        self.keep_rmap = keep_rmap
        self._uid = 0
        # clutter map in reference px at reduced resolution (cells of 4 x 4 px)
        x0, y0, x1, y1 = geom.bbox
        self._cl_origin = (x0 - 16.0, y0 - 16.0)
        self._cl_cell = 4.0
        cw = int((x1 - x0 + 32) / self._cl_cell) + 2
        ch = int((y1 - y0 + 32) / self._cl_cell) + 2
        self.clutter = np.zeros((ch, cw), np.float32)
        self._clutter_decay = 1.0 - 1.0 / max(1.0, opts.clutter_tau)
        self.cue_r = float(np.clip(opts.cue_radius_ft * geom.scale_rel, 6.0, 60.0))
        self.cue_r_c = float(np.clip(opts.cue_radius_ft * geom.scale_catch, 6.0, 80.0))
        self.n_processed = 0

    # ------------------------------------------------------------ stream
    def push(self, s: int, t: float, k: int, parity: int, image: np.ndarray, A: np.ndarray,
             quality: float) -> list[DetFrame]:
        self.buf[s] = _Buf(s, t, k, parity, image, np.asarray(A, np.float64), float(quality))
        if self.next_s is None:
            self.next_s = s
        out = []
        while self.next_s is not None and self.next_s + self.max_off <= s:
            if self.next_s in self.buf:
                out.append(self._detect(self.next_s))
            self.next_s += 1
            self._trim()
        return out

    def flush(self) -> list[DetFrame]:
        out = []
        if self.next_s is None or not self.buf:
            return out
        last = max(self.buf)
        while self.next_s <= last:
            if self.next_s in self.buf:
                out.append(self._detect(self.next_s))
            self.next_s += 1
        self.buf.clear()
        return out

    def _trim(self) -> None:
        lo = (self.next_s or 0) - self.max_off - 1
        while self.buf and next(iter(self.buf)) < lo:
            self.buf.popitem(last=False)

    # ------------------------------------------------------------- ROI
    def _roi(self, b: _Buf) -> Optional[tuple]:
        Ainv = G.invert(b.A)
        P = G.apply(Ainv, self.geom.corridor)
        cols = P[:, 0]
        rows = (P[:, 1] - b.parity) / self.rs
        m = self.opts.roi_margin
        c0 = int(math.floor(cols.min())) - m - self.kmax_w
        c1 = int(math.ceil(cols.max())) + m + self.kmax_w + 1
        r0 = int(math.floor(rows.min())) - max(2, m // self.rs) - self.kmax_h
        r1 = int(math.ceil(rows.max())) + max(2, m // self.rs) + self.kmax_h + 1
        H, W = self.shape
        c0, r0 = max(0, c0), max(0, r0)
        c1, r1 = min(W, c1), min(H, r1)
        if c1 - c0 < 8 or r1 - r0 < 4:
            return None
        return c0, r0, c1 - c0, r1 - r0

    def _background(self, b: _Buf, roi: tuple, offsets: tuple) -> Optional[np.ndarray]:
        c0, r0, w, h = roi
        Fu = G.grid_to_frame(self.rs, b.parity) @ G.translate(c0, r0)
        Ac = G.to3(b.A)
        flags = (cv2.INTER_CUBIC if self.opts.interp_cubic else cv2.INTER_LINEAR) | cv2.WARP_INVERSE_MAP
        stack = []
        n_before = n_after = 0
        for o in offsets:
            for sgn in (-1, 1):
                n = self.buf.get(b.s + sgn * o)
                if n is None or n.parity != b.parity or n.quality <= 0:
                    continue
                M = G.frame_to_grid(self.rs, n.parity) @ np.linalg.inv(G.to3(n.A)) @ Ac @ Fu
                if G.is_affine(M, 1e-12):
                    stack.append(cv2.warpAffine(n.image, M[:2], (w, h), flags=flags,
                                                borderMode=cv2.BORDER_REPLICATE))
                else:
                    stack.append(cv2.warpPerspective(n.image, M / M[2, 2], (w, h), flags=flags,
                                                     borderMode=cv2.BORDER_REPLICATE))
                if sgn < 0:
                    n_before += 1
                else:
                    n_after += 1
        if len(stack) < 3:
            return None
        return median_stack(stack)

    def _progress_map(self, b: _Buf, roi: tuple) -> np.ndarray:
        """Corridor progress of every ROI pixel (first-order in the transform; cheap)."""
        c0, r0, w, h = roi
        jc, ic = 0.5 * w, 0.5 * h
        g = np.array([[jc, ic], [jc + 1, ic], [jc, ic + 1]])
        x = g[:, 0] + c0
        y = (g[:, 1] + r0) * self.rs + b.parity
        p = self.geom.progress(G.apply(b.A, np.stack([x, y], axis=1)))
        pj, pi = p[1] - p[0], p[2] - p[0]
        jj = (np.arange(w, dtype=np.float32) - jc)[None, :]
        ii = (np.arange(h, dtype=np.float32) - ic)[:, None]
        return (p[0] + pj * jj + pi * ii).astype(np.float32)

    def _sub_rect(self, mask: np.ndarray, mh: int, mw: int):
        """Bounding (r0, r1, c0, c1) of a boolean mask, expanded by a margin; None if empty."""
        rows = np.nonzero(mask.any(axis=1))[0]
        if not len(rows):
            return None
        cols = np.nonzero(mask.any(axis=0))[0]
        h, w = mask.shape
        return (max(0, rows[0] - mh), min(h, rows[-1] + mh + 1), max(0, cols[0] - mw), min(w, cols[-1] + mw + 1))

    # ---------------------------------------------------------- detection
    def _detect(self, s: int) -> DetFrame:
        b = self.buf[s]
        o = self.opts
        empty = np.zeros(0, CAND_DTYPE)
        df = DetFrame(s=s, t=b.t, k=b.k, parity=b.parity, A=b.A, quality=b.quality, cands=empty)
        if b.quality <= 0:
            df.skipped = "no_registration"
            return df
        roi = self._roi(b)
        if roi is None:
            df.skipped = "corridor_outside_image"
            return df
        c0, r0, w, h = roi
        df.roi = roi
        B = self._background(b, roi, self.offsets)
        if B is None:
            df.skipped = "no_background"
            return df
        cur = b.image[r0:r0 + h, c0:c0 + w].astype(np.float32)
        a, off = robust_gain(cur, B)
        D = cur - (a * B + off)
        behind_multi = self.geom.view == "behind"
        prog = self._progress_map(b, roi) if (self.use_slow or behind_multi) else None
        if self.use_slow:
            # long-offset background only where the ball moves slowly (release end)
            slow = prog < o.slow_progress
            rect = self._sub_rect(slow, self.kmax_h + 2, self.kmax_w + 2)
            if rect is not None:
                sr0, sr1, sc0, sc1 = rect
                Bs = self._background(b, (c0 + sc0, r0 + sr0, sc1 - sc0, sr1 - sr0), self.slow_offsets)
                if Bs is not None:
                    cs = cur[sr0:sr1, sc0:sc1]
                    a2, off2 = robust_gain(cs, Bs)
                    Ds = cs - (a2 * Bs + off2)
                    D[sr0:sr1, sc0:sc1] = np.where(slow[sr0:sr1, sc0:sc1], Ds, D[sr0:sr1, sc0:sc1])
        if o.preblur_x > 0 or o.preblur_y > 0:
            R = cv2.GaussianBlur(D, (0, 0), sigmaX=max(o.preblur_x, 0.01), sigmaY=max(o.preblur_y, 0.01))
        else:
            R = D
        if self.raw_mask is not None:
            R = R * (self.raw_mask[r0:r0 + h, c0:c0 + w] > 0)
        sig_map, sig = block_noise(R, o.noise_block, o.noise_floor)
        df.sigma = sig
        # matched filters
        resp = np.zeros((len(self.bank), h, w), np.float32)
        rects = [(0, h, 0, w, None)] * len(self.bank)
        if behind_multi and len({k.diam for k in self.bank}) > 1:
            # each kernel only where its ball size is expected (also saves most of the work)
            dmap = self.geom.ball_diameter(prog).astype(np.float32)
            rects = []
            for j, ks in enumerate(self.bank):
                good = (dmap >= ks.diam / 2.2) & (dmap <= ks.diam * 2.2) if ks.diam > 0 else np.ones_like(dmap, bool)
                kh, kw = ks.kernel.shape[0] // 2, ks.kernel.shape[1] // 2
                rect = self._sub_rect(good, kh, kw)
                rects.append(None if rect is None else rect + (good[rect[0]:rect[1], rect[2]:rect[3]],))
                if rect is None:
                    continue
                q0, q1, p0, p1 = rect
                resp[j, q0:q1, p0:p1] = ks.apply(np.ascontiguousarray(R[q0:q1, p0:p1]))
                resp[j][~good] = 0.0
        else:
            for j, ks in enumerate(self.bank):
                resp[j] = ks.apply(R)
        # Each kernel's response is valid only where the kernel fits inside the ROI (the
        # border is replicated).  Invalidating per kernel instead of one margin of the
        # *largest* kernel matters where the ROI is clipped by the image edge: with the
        # long catch-end streak kernels (half height ~16 field rows) a margin of the largest
        # kernel blanked the top ~32 frame rows -- exactly where a zoomed behind view puts
        # the release point.  The small blob kernels still see a ball there.
        for j, ks in enumerate(self.bank):
            kh, kw = ks.kernel.shape[0] // 2, ks.kernel.shape[1] // 2
            if kh > self._kmin_h:
                resp[j, :min(h, kh)] = 0.0
                resp[j, max(0, h - kh):] = 0.0
            if kw > self._kmin_w:
                resp[j, :, :min(w, kw)] = 0.0
                resp[j, :, max(0, w - kw):] = 0.0
        self._ck_age += 1
        if self._ck_age >= 4:
            self._update_ck(R, resp, rects)
        best = np.abs(resp[0]) / np.float32(self._ck[0])
        kidx = np.zeros((h, w), np.int16)
        for j in range(1, len(self.bank)):
            a_j = np.abs(resp[j]) / np.float32(self._ck[j])
            m = a_j > best
            kidx[m] = j
            np.maximum(best, a_j, out=best)
        sel = np.take_along_axis(resp, kidx[None].astype(np.intp), axis=0)[0]
        zz = best / sig_map
        sgn = np.where(sel >= 0, 1, -1).astype(np.int8)
        # invalidate the outer margin (kernels / borders)
        valid = np.zeros((h, w), bool)
        my, mx = min(self._kmin_h, h // 3), min(self._kmin_w, w // 3)
        valid[my:h - my, mx:w - mx] = True
        rr, cc, vv = nms_peaks(zz, o.nms_size, o.z_weak * 0.8, 4 * o.max_cands, valid)
        df.n_peaks = len(rr)
        zs = (zz * sgn * 100.0).clip(-32000, 32000).astype(np.int16)
        df.zmap = zs
        if self.keep_rmap:
            df.rmap = (R / sig_map * 16.0).clip(-127, 127).astype(np.int8)
        # foreground / blobs
        # foreground for body blobs / cues: global noise scale (a large body inflates the
        # local scale and would hide itself)
        fg = (np.abs(R) > 3.0 * sig).astype(np.uint8)
        nlab, lab, stats, _cent = cv2.connectedComponentsWithStats(fg, connectivity=8)
        dmax = max(self.geom.d_rel, self.geom.d_catch)
        big_area = max(o.attached_area_min, 25.0 * dmax * dmax) / self.rs
        big = np.zeros(nlab, bool)
        if nlab > 1:
            big[1:] = stats[1:, cv2.CC_STAT_AREA] >= big_area
        for li in np.nonzero(big)[0]:
            x, y, bw, bh = stats[li, :4]
            corners = df.grid_to_ref([x, x + bw, x, x + bw], [y, y, y + bh, y + bh], self.rs)
            df.blobs.append((float(corners[:, 0].min()), float(corners[:, 1].min()),
                             float(corners[:, 0].max()), float(corners[:, 1].max())))
        df.e_rel = self._cue(fg, df, self.geom.release, self.cue_r)
        df.e_catch = self._cue(fg, df, self.geom.catch, self.cue_r_c)
        df.cands = self._candidates(df, R, sig_map, zz, sgn, kidx, rr, cc, vv, lab, big)
        self._update_clutter(df.cands)
        self.n_processed += 1
        return df

    def _white_noise_gain(self) -> np.ndarray:
        """c_k for white noise through the same pre-blur (a physical lower bound).

        Protects the z normalization on noise-free or heavily quantized input,
        where the measured residual is mostly exact zeros.
        """
        o = self.opts
        n = np.random.default_rng(12345).normal(0, 1, (160, 480)).astype(np.float32)
        if o.preblur_x > 0 or o.preblur_y > 0:
            n = cv2.GaussianBlur(n, (0, 0), sigmaX=max(o.preblur_x, 0.01), sigmaY=max(o.preblur_y, 0.01))
        core = (slice(30, -30), slice(40, -40))
        mr = float(np.median(np.abs(n[core])))
        return np.array([float(np.median(np.abs(ks.apply(n)[core]))) / mr for ks in self.bank])

    def _update_ck(self, R: np.ndarray, resp: np.ndarray, rects=None) -> None:
        """Noise gain c_k of each kernel response relative to R (robust, over its valid area)."""
        ck = np.empty(len(resp))
        for j, r in enumerate(resp):
            rect = rects[j] if rects is not None else (0, R.shape[0], 0, R.shape[1], None)
            if rect is None:
                ck[j] = self._ck[j]
                continue
            q0, q1, p0, p1, good = rect
            Rs = R[q0:q1:2, p0:p1:3]
            rs_ = r[q0:q1:2, p0:p1:3]
            if good is not None:            # only where the kernel was evaluated
                g = good[::2, ::3]
                Rs, rs_ = Rs[g], rs_[g]
            # Static areas of compressed / lossless video often have R == 0 exactly
            # (unchanged blocks).  There the noise is not "small", it is absent, and
            # including those pixels drives median|R| to ~0 and c_k to absurd values
            # (seen: 4e4 instead of ~1), which suppresses every detection.
            nz = np.abs(Rs) > 1e-3
            Rs, rs_ = Rs[nz], rs_[nz]
            if rs_.size < 50:
                ck[j] = self._ck[j]
                continue
            ck[j] = float(np.median(np.abs(rs_))) / (float(np.median(np.abs(Rs))) + 1e-6)
        # Correlated (blurred / compressed) noise changes a kernel's gain relative to white
        # noise by a modest factor, never by orders of magnitude.
        ck = np.clip(ck, self._ck_white, CK_MAX_FACTOR * self._ck_white) + 1e-6
        if self._ck_age > 10 ** 8:
            self._ck = ck
        else:
            self._ck = 0.7 * self._ck + 0.3 * ck
        self._ck_age = 0

    def _cue(self, fg: np.ndarray, df: DetFrame, pt: np.ndarray, r: float) -> float:
        g = df.ref_to_grid(np.atleast_2d(pt), self.rs)[0]
        h, w = fg.shape
        rx = r / self.geom.sar
        ry = r / self.rs
        j0, j1 = int(max(0, g[0] - rx)), int(min(w, g[0] + rx + 1))
        i0, i1 = int(max(0, g[1] - ry)), int(min(h, g[1] + ry + 1))
        if j1 <= j0 or i1 <= i0:
            return float("nan")
        ii, jj = np.mgrid[i0:i1, j0:j1]
        inside = ((jj - g[0]) / rx) ** 2 + ((ii - g[1]) / ry) ** 2 <= 1.0
        if not inside.any():
            return float("nan")
        return float(fg[i0:i1, j0:j1][inside].mean())

    def _candidates(self, df, R, sig_map, zz, sgn, kidx, rr, cc, vv, lab, big) -> np.ndarray:
        o = self.opts
        if len(rr) == 0:
            return np.zeros(0, CAND_DTYPE)
        # cheap pre-filter on the raw peaks (vectorized): corridor, masks, clutter weight
        pre = df.grid_to_ref(cc, rr, self.rs)
        keep = self.geom.in_corridor(pre, margin=4.0)
        if self.ref_masks:
            for (x0, y0, x1, y1) in self.ref_masks:
                keep &= ~((pre[:, 0] >= x0 - 3) & (pre[:, 0] <= x1 + 3) & (pre[:, 1] >= y0 - 3) & (pre[:, 1] <= y1 + 3))
        if not keep.any():
            return np.zeros(0, CAND_DTYPE)
        rr, cc, vv, pre = rr[keep], cc[keep], vv[keep], pre[keep]
        cf = self._clutter_factor(pre[:, 0], pre[:, 1])
        ok = vv * cf >= o.z_weak * 0.9
        rr, cc, vv = rr[ok], cc[ok], vv[ok]
        order = np.argsort(-(vv * cf[ok]), kind="stable")[:int(1.5 * o.max_cands) + 2]
        rr, cc, vv = rr[order], cc[order], vv[order]
        h, w = R.shape
        sar = self.geom.sar
        rows = []
        for r, c, v in zip(rr.tolist(), cc.tolist(), vv.tolist()):
            sign = int(sgn[r, c]) or 1
            ki = int(kidx[r, c])
            ks = self.bank[ki]
            L = ks.length
            hx = int(math.ceil((L / 2 * abs(math.cos(ks.angle)) + 2.0) / sar)) + 1
            hy = max(1, int(math.ceil((L / 2 * abs(math.sin(ks.angle)) + 1.5) / self.rs)))
            j0, j1 = max(0, c - hx), min(w, c + hx + 1)
            i0, i1 = max(0, r - hy), min(h, r + hy + 1)
            win = sign * R[i0:i1, j0:j1] - 1.5 * sig_map[r, c]
            np.maximum(win, 0, out=win)
            tot = float(win.sum())
            if tot > 0:
                jj = np.arange(j0, j1, dtype=np.float64)[None, :]
                ii = np.arange(i0, i1, dtype=np.float64)[:, None]
                gx = float((win * jj).sum() / tot)
                gy = float((win * ii).sum() / tot)
                dxs = (jj - gx) * sar
                dys = (ii - gy) * self.rs
                cxx = float((win * dxs * dxs).sum() / tot)
                cyy = float((win * dys * dys).sum() / tot)
                cxy = float((win * dxs * dys).sum() / tot)
                tr, det = cxx + cyy, cxx * cyy - cxy * cxy
                lam = tr / 2 + math.sqrt(max(tr * tr / 4 - det, 0.0))
                length = math.sqrt(12.0 * max(lam, 0.0))
                angle = 0.5 * math.atan2(2 * cxy, cxx - cyy)
            else:
                gx, gy = float(c), float(r)
                length, angle = 0.0, 0.0
            # attached: touches a big blob
            attached = bool(big[lab[max(0, r - 1):r + 2, max(0, c - 1):c + 2]].any())
            rows.append((gx, gy, float(v), sign, length, angle, attached, ki))
        if not rows:
            return np.zeros(0, CAND_DTYPE)
        A = np.array([r_[:2] for r_ in rows])
        ref = df.grid_to_ref(A[:, 0], A[:, 1], self.rs)
        out = [(ref[i, 0], ref[i, 1]) + rows[i][2:] for i in range(len(rows))]
        # peaks whose centroids converged to the same place: keep the strongest
        out.sort(key=lambda r_: -r_[2])
        uniq = []
        for r_ in out:
            if all(math.hypot((r_[0] - u[0]) * sar, r_[1] - u[1]) > 1.5 or r_[3] != u[3] for u in uniq):
                uniq.append(r_)
        arr = np.zeros(len(uniq), CAND_DTYPE)
        for i, (x, y, v, sign, length, angle, att, ki) in enumerate(uniq):
            arr[i] = (0, df.s, df.t, df.k, x, y, v, v, sign, length, angle, att, 1.0, 0.0, ki)
        keep = self.geom.in_corridor(np.stack([arr["x"], arr["y"]], 1), margin=2.0)
        for i in range(len(arr)):
            if keep[i] and self._in_ref_masks(arr["x"][i], arr["y"][i]):
                keep[i] = False
        arr = arr[keep]
        if len(arr):
            cf = self._clutter_factor(arr["x"], arr["y"])
            arr["z"] = arr["zraw"] * cf
            arr = arr[arr["z"] >= o.z_weak]
            arr = arr[np.argsort(-arr["z"], kind="stable")][:o.max_cands]
            arr["sigma"] = np.maximum(0.4, 3.0 / np.maximum(arr["z"], 1e-3))
            arr["prog"] = self.geom.progress(np.stack([arr["x"], arr["y"]], 1))
            n = len(arr)
            arr["uid"] = np.arange(self._uid, self._uid + n)
            self._uid += n
        return arr

    def _in_ref_masks(self, x: float, y: float) -> bool:
        for (x0, y0, x1, y1) in self.ref_masks:
            if x0 - 3 <= x <= x1 + 3 and y0 - 3 <= y <= y1 + 3:
                return True
        return False

    # ------------------------------------------------------------ clutter
    def _cl_index(self, x, y):
        j = ((np.asarray(x) - self._cl_origin[0]) / self._cl_cell).astype(int)
        i = ((np.asarray(y) - self._cl_origin[1]) / self._cl_cell).astype(int)
        h, w = self.clutter.shape
        return np.clip(i, 0, h - 1), np.clip(j, 0, w - 1)

    def _clutter_factor(self, x, y) -> np.ndarray:
        i, j = self._cl_index(x, y)
        # max over the 3x3 cell neighbourhood (positions jitter by a few px)
        h, w = self.clutter.shape
        best = np.zeros(len(i), np.float32)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                best = np.maximum(best, self.clutter[np.clip(i + di, 0, h - 1), np.clip(j + dj, 0, w - 1)])
        return np.minimum(1.0, np.sqrt(self.opts.clutter_ref / np.maximum(best, 1e-6))).astype(np.float32)

    def _update_clutter(self, cands: np.ndarray) -> None:
        self.clutter *= self._clutter_decay
        if len(cands) == 0:
            return
        sel = cands[(~cands["attached"]) & (cands["zraw"] >= 3.0)]
        if len(sel) == 0:
            return
        i, j = self._cl_index(sel["x"], sel["y"])
        add = 1.0 - self._clutter_decay
        # one count per cell per sample
        cells = set(zip(i.tolist(), j.tolist()))
        for (ii, jj) in cells:
            self.clutter[ii, jj] += add

    def clutter_image(self) -> np.ndarray:
        return self.clutter.copy()
