"""Cached access to sample images for the interactive windows.

Stepping field by field must feel instant, but random access in a long
MPEG/DV file may need a seek plus decoding up to 1.5 s of video.
`FrameStore` keeps recently shown samples (colour) in an LRU cache bounded
in bytes and can `prefetch` a range with one sequential decode (the review
window prefetches each pitch's window: release - 60 to catch + 60 fields).

For an interactive loop, `set_prefetch(s0, s1)` + `prefetch_step(n)`
decode a range a few samples at a time from one persistent sequential
decoder, so the key poll stays responsive: OpenCV's Qt backend keeps only
the LAST key pressed between two ``waitKeyEx`` calls, so a loop blocked
for 0.3 s in a decode loses keystrokes (measured under Xvfb).

A sample that cannot be decoded is replaced by a dark placeholder that
says so (and the error is kept in `errors`); nothing is invented.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Optional

import cv2
import numpy as np

from pitchspeed.types import Sample
from pitchspeed.ui.canvas import weave_fields

log = logging.getLogger("pitchspeed")

DEFAULT_MAX_BYTES = 256 * 1024 * 1024


class FrameStore:
    """LRU cache of colour samples from a SampleSource (see module doc)."""

    def __init__(self, source, max_bytes: int = DEFAULT_MAX_BYTES):
        self.source = source
        self.max_bytes = int(max_bytes)
        self._cache: "OrderedDict[int, Sample]" = OrderedDict()
        self._bytes = 0
        self.errors: list[str] = []
        self._pf_range: Optional[tuple[int, int]] = None
        self._pf_iter = None
        self._pf_next = 0

    # ------------------------------------------------------------- cache
    def _put(self, smp: Sample) -> None:
        old = self._cache.pop(smp.s, None)
        if old is not None:
            self._bytes -= _nbytes(old)
        self._cache[smp.s] = smp
        self._bytes += _nbytes(smp)
        while self._bytes > self.max_bytes and len(self._cache) > 1:
            _, ev = self._cache.popitem(last=False)
            self._bytes -= _nbytes(ev)

    def cached(self, s: int) -> bool:
        return int(s) in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    # ------------------------------------------------------------ access
    def sample(self, s: int) -> Sample:
        """Sample s with its colour image (placeholder if it cannot be decoded)."""
        s = int(s)
        smp = self._cache.get(s)
        if smp is not None:
            self._cache.move_to_end(s)
            return smp
        try:
            smp = self.source.get_sample(s, color=True)
        except IndexError:
            raise
        except Exception as e:  # noqa: BLE001 - decoding problems must not kill the UI
            msg = f"sample {s}: could not be decoded ({e})"
            log.warning(msg)
            self.errors.append(msg)
            return self.placeholder(s, "could not decode this field/frame")
        if smp.bgr is None:
            smp.bgr = cv2.cvtColor(smp.image, cv2.COLOR_GRAY2BGR)
        self._put(smp)
        return smp

    def prefetch(self, s0: int, s1: int) -> int:
        """Decode samples s0 <= s < s1 sequentially into the cache; returns how many."""
        s0 = max(0, int(s0))
        s1 = min(int(self.source.n_samples), int(s1))
        missing = [s for s in range(s0, s1) if not self.cached(s)]
        if not missing:
            return 0
        s0, s1 = missing[0], missing[-1] + 1
        n = 0
        try:
            for smp in self.source.iter_samples(s0, s1, color=True):
                if smp.bgr is None:
                    smp.bgr = cv2.cvtColor(smp.image, cv2.COLOR_GRAY2BGR)
                self._put(smp)
                n += 1
        except Exception as e:  # noqa: BLE001
            msg = f"prefetch {s0}-{s1}: {e}"
            log.warning(msg)
            self.errors.append(msg)
        return n

    # ------------------------------------------------- incremental prefetch
    def set_prefetch(self, s0: Optional[int], s1: Optional[int] = None) -> None:
        """Range to fill with prefetch_step() (None: stop prefetching)."""
        rng = None if s0 is None else (max(0, int(s0)), min(int(self.source.n_samples), int(s1)))
        if rng == self._pf_range:
            return
        self._close_iter()
        self._pf_range = rng
        self._pf_next = rng[0] if rng else 0

    def _close_iter(self) -> None:
        if self._pf_iter is not None:
            try:
                self._pf_iter.close()
            except Exception:  # noqa: BLE001
                pass
        self._pf_iter = None

    def prefetch_step(self, max_samples: int = 6) -> bool:
        """Decode up to `max_samples` more samples of the prefetch range; True if work remains."""
        if self._pf_range is None:
            return False
        s1 = self._pf_range[1]
        if self._pf_iter is None:
            while self._pf_next < s1 and self.cached(self._pf_next):
                self._pf_next += 1
            if self._pf_next >= s1:
                self._pf_range = None
                return False
            self._pf_iter = iter(self.source.iter_samples(self._pf_next, s1, color=True))
        n = 0
        try:
            for smp in self._pf_iter:
                if smp.bgr is None:
                    smp.bgr = cv2.cvtColor(smp.image, cv2.COLOR_GRAY2BGR)
                if not self.cached(smp.s):
                    self._put(smp)
                self._pf_next = smp.s + 1
                n += 1
                if n >= max_samples:
                    return True
        except Exception as e:  # noqa: BLE001
            msg = f"prefetch from sample {self._pf_next}: {e}"
            log.warning(msg)
            self.errors.append(msg)
        self._close_iter()
        self._pf_range = None
        return False

    def frame_image(self, frame: int) -> np.ndarray:
        """Full BGR frame: the two fields woven (interlaced) or the sample itself."""
        src = self.source
        if not getattr(src, "interlaced", False):
            return self.sample(src.sample_of(frame, 0)).bgr
        a = self.sample(src.sample_of(frame, 0))
        b = self.sample(src.sample_of(frame, 1))
        top, bottom = (a, b) if a.parity == 0 else (b, a)
        if top.bgr.shape != bottom.bgr.shape:
            return top.bgr
        return weave_fields(top.bgr, bottom.bgr)

    def frame_gray(self, frame: int) -> np.ndarray:
        img = self.frame_image(frame)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    def placeholder(self, s: int, text: str) -> Sample:
        rows, cols = self.source.sample_shape
        img = np.full((rows, cols), 24, np.uint8)
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.putText(bgr, text, (10, rows // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        try:
            frame, k, parity = self.source.locate(s)
            t = float(self.source.sample_time(s))
        except Exception:  # noqa: BLE001
            frame, k, parity, t = -1, 0, 0, float("nan")
        return Sample(s=s, frame=frame, k=k, parity=parity, t=t, image=img, bgr=bgr)


def _nbytes(smp: Sample) -> int:
    n = smp.image.nbytes if smp.image is not None else 0
    return n + (smp.bgr.nbytes if smp.bgr is not None else 0)
