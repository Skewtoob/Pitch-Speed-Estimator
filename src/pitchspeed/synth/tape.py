"""Per-frame camera electronics and tape chain.

Only point-wise or horizontal operations are used, so the two fields of an
interlaced frame never mix:

  BGR -> YCbCr; luma noise sigma^2 = sigma0^2 + beta * Y; streaky chroma
  noise; in-camera date/time stamp (seconds tick); luma low-pass +
  sharpening overshoot; chroma low-pass + delay; per-line timebase jitter
  (AR(1) within each field); head-switching band; dropouts.

The RNG is keyed by (seed, 7, frame) so frames can be rendered in any order
or in parallel and stay bit-identical.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.signal import lfilter

from pitchspeed.synth.config import FPS_DEN, FPS_NUM, TAPE_PROFILES, rng_for


def _hkernel(sigma: float) -> np.ndarray:
    return cv2.getGaussianKernel(int(2 * math.ceil(3 * sigma) + 1), sigma).astype(np.float32)


class TapeChain:
    """Applies noise, date stamp and tape artefacts to rendered frames."""

    def __init__(self, cfg: dict, W: int, H: int):
        self.seed = int(cfg["seed"])
        self.W, self.H = W, H
        self.noise = dict(cfg["noise"])
        self.tp = dict(TAPE_PROFILES[cfg["tape"]["profile"]])
        self.tp.update({k: v for k, v in cfg["tape"].items() if k != "profile"})
        self.ds = dict(cfg["date_stamp"])
        self._stamp_cache: dict = {}
        self.k_luma = _hkernel(self.tp["luma_sigma"]) if self.tp["luma_sigma"] > 0 else None
        self.k_sharp = _hkernel(1.5)
        self.k_chroma = _hkernel(self.tp["chroma_sigma"]) if self.tp["chroma_sigma"] > 0 else None
        self.one = np.array([[1.0]], np.float32)
        nz = self.noise
        self.identity = (nz["sigma0"] == 0 and nz["beta"] == 0 and nz["chroma_sigma"] == 0
                         and not self.ds["enabled"] and all(not v for k, v in self.tp.items()))

    # ------------------------------------------------------------ date stamp
    def stamp_mask(self, n: int) -> np.ndarray:
        """Anti-aliased text mask (float32 0..1) of the date stamp in frame n (cached per second)."""
        ds = self.ds
        sec = int(ds["start_hms"][0] * 3600 + ds["start_hms"][1] * 60 + ds["start_hms"][2] + n * FPS_DEN / FPS_NUM)
        m = self._stamp_cache.get(sec)
        if m is None:
            hh, mm, ss = sec // 3600 % 24, sec // 60 % 60, sec % 60
            txt = [ds["text_date"], "%d:%02d:%02d %s" % ((hh - 1) % 12 + 1, mm, ss, "PM" if hh >= 12 else "AM")]
            W, H = self.W, self.H
            scale = 0.85 * W / 720.0
            mask = np.zeros((H, W), np.uint8)
            for i, tx in enumerate(txt):
                org = (int(W - 250 * W / 720.0), int(H - (60 - 30 * i) * H / 480.0))
                cv2.putText(mask, tx, org, cv2.FONT_HERSHEY_SIMPLEX, scale, 255, max(1, int(round(2 * W / 720.0))),
                            cv2.LINE_AA)
            m = mask.astype(np.float32) / 255.0
            if len(self._stamp_cache) > 8:
                self._stamp_cache.clear()
            self._stamp_cache[sec] = m
        return m

    def _hblur(self, X: np.ndarray, k) -> np.ndarray:
        if k is None:
            return X
        return cv2.sepFilter2D(X, -1, k, self.one, borderType=cv2.BORDER_REPLICATE)

    # ------------------------------------------------------------ main
    def apply(self, frame: np.ndarray, n: int) -> np.ndarray:
        """frame: (H, W, 3) uint8 BGR render -> uint8 BGR after the camera/tape chain."""
        if self.identity:
            return frame
        r = rng_for(self.seed, 7, n)
        H, W = frame.shape[:2]
        ycc = cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycc)
        Cr -= 0.5                                   # OpenCV float YCrCb offset
        Cb -= 0.5
        nz = self.noise
        sig = np.sqrt(np.maximum(Y, 0.0) * np.float32(nz["beta"]) + np.float32(nz["sigma0"] ** 2))
        Y += r.standard_normal(Y.shape, dtype=np.float32) * sig
        if nz["chroma_sigma"] > 0:
            for Cc in (Cb, Cr):
                cn = r.standard_normal((H, max(W // 8, 1)), dtype=np.float32)
                Cc += cv2.resize(cn, (W, H), interpolation=cv2.INTER_LINEAR) * np.float32(nz["chroma_sigma"])
        if self.ds["enabled"]:
            mf = self.stamp_mask(n)
            inv = 1.0 - mf
            Y = Y * inv + 235.0 * mf
            Cb *= inv
            Cr *= inv
        tp = self.tp
        Y = self._hblur(Y, self.k_luma)
        if tp["sharpen"] > 0:
            Y = Y + np.float32(tp["sharpen"]) * (Y - self._hblur(Y, self.k_sharp))
        Cb = self._hblur(Cb, self.k_chroma)
        Cr = self._hblur(Cr, self.k_chroma)
        if tp["chroma_shift"]:
            Mx = np.float32([[1, 0, tp["chroma_shift"]], [0, 1, 0]])
            Cb = cv2.warpAffine(Cb, Mx, (W, H), borderMode=cv2.BORDER_REPLICATE)
            Cr = cv2.warpAffine(Cr, Mx, (W, H), borderMode=cv2.BORDER_REPLICATE)
        if tp["jitter"] > 0:                       # per-line timebase error, AR(1) within each field
            d = np.zeros(H, np.float32)
            for par in (0, 1):
                e = r.standard_normal(len(range(par, H, 2))) * tp["jitter"] * 0.3
                d[par::2] = lfilter([1.0], [1.0, -0.9], e)
            mx = np.arange(W, dtype=np.float32)[None, :] + d[:, None]
            my = np.repeat(np.arange(H, dtype=np.float32)[:, None], W, 1)
            Y = cv2.remap(Y, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if tp["headswitch"]:
            hs = int(tp["headswitch"]) * 2
            Y[-hs:] = np.roll(Y[-hs:], int(r.integers(6, 16)), axis=1) + r.normal(0, 10, (hs, W)).astype(np.float32)
        for _ in range(int(r.poisson(tp["dropouts"]))):
            row = int(r.integers(0, H))
            x0 = int(r.integers(0, max(W - 20, 1)))
            L = int(r.integers(8, 120))
            Y[row, x0:x0 + L] = 200 + 40 * r.random(min(L, W - x0))
        out = cv2.cvtColor(cv2.merge([Y, Cr + 0.5, Cb + 0.5]), cv2.COLOR_YCrCb2BGR)
        np.clip(out, 0.0, 255.0, out=out)
        out += 0.5
        return out.astype(np.uint8)
