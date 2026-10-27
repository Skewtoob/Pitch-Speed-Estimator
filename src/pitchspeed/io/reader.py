"""Video decoding with an exact frame index (PyAV only; never OpenCV).

`VideoReader` opens a file, builds a presentation-timestamp (PTS) index of
every video frame, derives exact frame times, and offers frame-exact random
access plus sequential iteration.  `probe()` returns fast, metadata-only
information.

Frame index i is the i-th frame in presentation order as decoded by FFmpeg
(through PyAV).  Frame times are seconds from the first video frame
(``frame_times[0] == 0``).

Index construction (``index_mode='auto'``), cheapest reliable method first:

* ``container`` - intra-only codec (DV, MJPEG, HuffYUV, ...) in AVI / MOV /
  MP4: the container's own index (AVI idx1 / OpenDML, MOV sample table)
  lists every frame; for intra-only codecs DTS == PTS.  No data is read.
* ``demux`` - codec without frame reordering (raw .dv, FFV1 in MKV, WMV2,
  ...): read packets without decoding; every packet must carry a unique,
  increasing PTS.
* ``decode`` - everything else (MPEG-2, H.264, MPEG-4 ASP...): decode every
  frame (no pixel conversion) and record the PTS the decoder outputs.  This
  is the ground truth for what `iter_frames()` will deliver.

The container/demux indexes are validated by decoding the first and last
frame; on any mismatch the reader falls back to a full decode index.

Timing: the nominal frame duration dt is chosen from the stream's rate
fields and common broadcast rates, matched against the median PTS step
(raw .dv advertises a bogus avg_frame_rate of 60000/1; WMV/MKV have 1 ms
time bases).  If every PTS lies within 0.25*dt of a grid point n*dt the
video is constant-frame-rate ('cfr') and t_i = n_i*dt exactly (removes the
millisecond jitter); grid slots without a frame are reported as dropped
frames (PTS gap > 1.5*dt).  Otherwise timing is 'vfr' and t_i = pts_i - pts_0.

Pixel aspect ratio (SAR, x_square = x * sar), in order of precedence:

1. ``sar_override`` (source 'override').
2. The stream's SAR when it is set and plausible (source 'stream').
   A stream SAR of exactly 1:1 on a standard-definition raster
   (480/486/576/240/288 lines with 720/704/352/... columns) is treated as a
   missing tag, because 1:1 would give a 3:2 picture that SD video never has.
3. Defaults (source 'default'): SD rasters are assumed to be 4:3 display,
   sar = (4/3) * H / W, i.e. 8:9 for 720x480, 10:11 for 704x480 and
   352x240, 16:15 for 720x576.  Everything else: 1:1.
   (Strict BT.601 would be 10:11 for 720x480; the 2% difference only matters
   where x and y are mixed and does not affect time-of-flight.)

Paths: plain ``str``/``os.PathLike``; non-ASCII names work (PyAV passes UTF-8
to FFmpeg, which converts to wide characters on Windows).  If opening by
name fails, a Python file object is tried as a fallback.
"""
from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterator, Optional, Sequence, Union

import av
import numpy as np

from pitchspeed.types import VideoInfo

log = logging.getLogger(__name__)

PathLike = Union[str, "os.PathLike[str]"]

INDEX_CACHE_VERSION = 2
CFR_JITTER_TOL = 0.25        # max |pts - grid| as a fraction of dt for CFR
DROP_GAP = 1.5               # a PTS step > DROP_GAP*dt marks a dropped frame
SEEK_PREROLLS_S = (1.5, 4.0, 12.0)   # seek this far before the target, then from the start
FORWARD_DECODE_S = 2.0       # decode forward instead of seeking if target is this close ahead

_STANDARD_RATES = [Fraction(30000, 1001), Fraction(60000, 1001), Fraction(24000, 1001),
                   Fraction(25), Fraction(50), Fraction(30), Fraction(60), Fraction(24),
                   Fraction(15000, 1001), Fraction(15), Fraction(12), Fraction(10),
                   Fraction(120000, 1001), Fraction(120), Fraction(100)]
_CONTAINER_INDEX_FORMATS = ("avi", "mov", "mp4")
_SD_HEIGHTS = (240, 288, 480, 486, 576)
_SD_WIDTHS = (352, 360, 480, 528, 544, 640, 704, 720)
_FMT_ALIASES = {"gray": "gray", "grey": "gray", "y": "gray",
                "bgr": "bgr24", "bgr24": "bgr24", "rgb": "rgb24", "rgb24": "rgb24"}
_REORDER_PROP = 8            # AV_CODEC_PROP_REORDER


class FrameDecodeError(RuntimeError):
    """A frame listed in the index could not be decoded."""


# ----------------------------------------------------------------- helpers --
def _fmt(fmt: str) -> str:
    """Normalize a pixel-format name; 'both' returns (gray, bgr) tuples."""
    if fmt == "both":
        return "both"
    try:
        return _FMT_ALIASES[fmt.lower()]
    except KeyError:
        raise ValueError(f"unsupported fmt {fmt!r}; use 'gray', 'bgr', 'rgb' or 'both'") from None


def _to_array(frame: "av.VideoFrame", fmt: str):
    if fmt == "both":
        return frame.to_ndarray(format="gray"), frame.to_ndarray(format="bgr24")
    return frame.to_ndarray(format=fmt)


def _open_container(path: str):
    """Open with PyAV; fall back to a Python file object (odd paths/encodings)."""
    try:
        return av.open(path, "r")
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        raise
    except Exception as e:                  # noqa: BLE001 - retry through a file object
        if not os.path.isfile(path):
            raise
        log.info("av.open(%r) failed (%s); retrying through a file object", path, e)
        fh = open(path, "rb")
        try:
            return av.open(fh, "r")
        except Exception:
            fh.close()
            raise


def _rational(x) -> Optional[Fraction]:
    if x is None:
        return None
    try:
        f = Fraction(x)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return f if f > 0 else None


def _fraction_str(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator)


def default_sar(width: int, height: int) -> Fraction:
    """Default stored-pixel aspect ratio when the stream does not say (see module doc)."""
    if height in _SD_HEIGHTS and width in _SD_WIDTHS and (width, height) not in ((640, 480), (320, 240)):
        return (Fraction(4, 3) * height / width).limit_denominator(1000)
    return Fraction(1)


def resolve_sar(stream_sar, width: int, height: int,
                override: Optional[float] = None) -> tuple[float, str, Optional[str]]:
    """Return (sar, source, note) following the precedence in the module docstring."""
    if override is not None:
        if not override > 0:
            raise ValueError(f"sar_override must be > 0, got {override!r}")
        return float(override), "override", None
    s = _rational(stream_sar)
    dflt = default_sar(width, height)
    if s is not None:
        if s == 1 and dflt != 1:
            return float(dflt), "default", (
                f"stream says 1:1 pixels for {width}x{height}; that is almost certainly a missing tag, "
                f"assuming a 4:3 picture (sar {_fraction_str(dflt)})")
        return float(s), "stream", None
    note = None
    if dflt != 1:
        note = f"no pixel aspect in the stream; assuming a 4:3 picture (sar {_fraction_str(dflt)})"
    return float(dflt), "default", note


# ----------------------------------------------------------------- timing --
@dataclass
class TimingAnalysis:
    """Result of `analyze_timing` (pure function of the PTS list)."""
    times: np.ndarray                  # (n,) seconds, times[0] == 0
    rate: Fraction                     # nominal frames per second
    mode: str                          # 'cfr' | 'vfr'
    dropped: list[int]                 # frame i has a gap before it
    max_jitter_s: float                # max |pts - grid| (cfr) or 0
    notes: list[str] = field(default_factory=list)

    @property
    def dt(self) -> float:
        return 1.0 / float(self.rate)


def _candidate_rates(rate_candidates: Sequence) -> list[Fraction]:
    out: list[Fraction] = []
    for c in list(rate_candidates) + _STANDARD_RATES:
        r = _rational(c)
        if r is not None and 1 <= r <= 1000 and r not in out:
            out.append(r)
    return out


def _grid_fit(rel: np.ndarray, dt: float):
    """Snap relative times to slots n*dt; returns (n, resid, valid)."""
    n = np.round(rel / dt).astype(np.int64)
    resid = rel - n * dt
    ok = bool(n[0] == 0 and (len(n) < 2 or np.all(np.diff(n) >= 1)))
    return n, resid, ok and float(np.max(np.abs(resid))) < CFR_JITTER_TOL * dt


def _best_grid(rel: np.ndarray, rates: Sequence[Fraction]):
    """Rate with a valid grid, fewest empty slots, then smallest RMS residual."""
    best = None
    for r in rates:
        dt = 1.0 / float(r)
        n, resid, ok = _grid_fit(rel, dt)
        if not ok:
            continue
        key = (int(n[-1]) - (len(n) - 1), float(np.sqrt(np.mean(resid ** 2))) / dt)
        if best is None or key < best[0]:
            best = (key, r, n, resid)
    return best


def analyze_timing(pts_s: np.ndarray, rate_candidates: Sequence = ()) -> TimingAnalysis:
    """CFR detection, exact frame times and dropped frames from PTS in seconds.

    `rate_candidates` are the stream's claimed rates (most trusted first);
    common broadcast rates are always tried too.  A rate is accepted when
    every timestamp lies within 0.25*dt of a slot n*dt; among accepted rates
    the one leaving the fewest empty slots (then the smallest residual) wins.
    """
    pts_s = np.asarray(pts_s, dtype=np.float64)
    n = len(pts_s)
    cands = _candidate_rates(rate_candidates)
    fallback = cands[0] if cands else Fraction(30000, 1001)
    if n == 0:
        return TimingAnalysis(np.zeros(0), fallback, "cfr", [], 0.0, ["no frames"])
    if n == 1:
        return TimingAnalysis(np.zeros(1), fallback, "cfr", [], 0.0, ["single frame"])
    rel = pts_s - pts_s[0]
    steps = np.diff(rel)
    pos = steps[steps > 0]
    if len(pos) == 0:
        return TimingAnalysis(np.arange(n) / float(fallback), fallback, "vfr", [], 0.0,
                              ["all frames share one timestamp; times synthesized from the nominal rate"])
    notes: list[str] = []
    best = _best_grid(rel, cands)
    if best is None:
        # The rate may be slightly off every standard (e.g. a capture clock): fit it.
        mean_step = float(rel[-1] / (n - 1))
        n0, _, ok0 = _grid_fit(rel, mean_step)
        if n0[0] == 0 and np.all(np.diff(n0) >= 1) and n0[-1] > 0:
            dt_fit = float(np.dot(n0, rel) / np.dot(n0, n0))
            r_fit = Fraction(1.0 / dt_fit).limit_denominator(100000)
            best = _best_grid(rel, [r_fit])
            if best is not None:
                notes.append(f"timestamps fit no standard rate; using the fitted rate {float(r_fit):.5f} fps")
    if best is not None:
        _, rate, grid, resid = best
        dt = 1.0 / float(rate)
        dropped = [int(i) for i in np.nonzero(np.diff(grid) >= 2)[0] + 1]
        return TimingAnalysis(grid * dt, rate, "cfr", dropped, float(np.max(np.abs(resid))), notes)
    # Variable frame rate (or broken timestamps): keep the measured times.
    median_step = float(np.median(pos))
    rate = Fraction(1.0 / median_step).limit_denominator(100000)
    if np.any(steps <= 0):
        notes.append("timestamps are not strictly increasing")
    notes.append("timestamps do not lie on a constant-rate grid: timing mode 'vfr'")
    dropped = [int(i) for i in np.nonzero(steps > DROP_GAP * median_step)[0] + 1]
    return TimingAnalysis(rel.copy(), rate, "vfr", dropped, 0.0, notes)


# ------------------------------------------------------------------ index --
def _decoded_frames(container, stream, errors: Optional[list] = None) -> Iterator["av.VideoFrame"]:
    """Decode packet by packet so one corrupt packet does not end the stream."""
    for packet in container.demux(stream):
        try:
            frames = packet.decode()
        except av.error.FFmpegError as e:
            if errors is not None:
                errors.append({"pts": packet.pts, "error": str(e)})
            log.warning("decode error at packet pts=%s: %s", packet.pts, e)
            continue
        yield from frames


def _is_keyframe(frame) -> bool:
    return bool(frame.key_frame) or int(frame.pict_type) == 1


@dataclass
class _Index:
    pts: np.ndarray            # (n,) int64, stream time-base ticks, presentation order
    method: str                # 'container' | 'demux' | 'decode'
    synthesized: bool = False  # some PTS were missing and were synthesized
    decode_errors: list = field(default_factory=list)


def _index_from_container(stream) -> Optional[np.ndarray]:
    try:
        entries = stream.index_entries
    except Exception:          # noqa: BLE001 - older PyAV
        return None
    if not entries:
        return None
    ts = np.array([e.timestamp for e in entries if not e.is_discard], dtype=np.int64)
    if len(ts) < 2 or not np.all(np.diff(ts) > 0):
        return None
    if stream.frames and stream.frames != len(ts):
        return None
    return ts


def _index_from_demux(container, stream) -> Optional[np.ndarray]:
    pts = []
    for packet in container.demux(stream):
        if packet.size == 0:
            continue
        if packet.pts is None:
            return None
        pts.append(packet.pts)
    a = np.array(pts, dtype=np.int64)
    if len(a) < 1 or (len(a) > 1 and not np.all(np.diff(a) > 0)):
        return None
    return a


def _index_from_decode(container, stream, nominal_step: int) -> _Index:
    pts: list[int] = []
    synthesized = False
    errors: list = []
    for frame in _decoded_frames(container, stream, errors):
        p = frame.pts
        if p is None:
            p = (pts[-1] + nominal_step) if pts else 0
            synthesized = True
        pts.append(int(p))
    return _Index(np.array(pts, dtype=np.int64), "decode", synthesized, errors)


def _codec_reorders(stream) -> bool:
    try:
        return bool(stream.codec_context.codec.properties & _REORDER_PROP)
    except Exception:          # noqa: BLE001
        return True


def _codec_intra_only(stream) -> bool:
    try:
        return bool(stream.codec_context.codec.intra_only)
    except Exception:          # noqa: BLE001
        return False


def _cache_key(path: str) -> dict:
    st = os.stat(path)
    return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns), "version": INDEX_CACHE_VERSION}


def _load_cached_index(cache_path: str, path: str) -> Optional[_Index]:
    try:
        with np.load(cache_path, allow_pickle=False) as z:
            key = _cache_key(path)
            if (int(z["size"]) != key["size"] or int(z["mtime_ns"]) != key["mtime_ns"]
                    or int(z["version"]) != key["version"]):
                return None
            return _Index(z["pts"].astype(np.int64), str(z["method"]) + "+cache",
                          bool(z["synthesized"]))
    except (OSError, KeyError, ValueError):
        return None


def _save_cached_index(cache_path: str, path: str, idx: _Index) -> None:
    key = _cache_key(path)
    tmp = cache_path + ".tmp.npz"
    try:
        np.savez(tmp, pts=idx.pts, method=np.array(idx.method), synthesized=np.array(idx.synthesized),
                 size=np.array(key["size"]), mtime_ns=np.array(key["mtime_ns"]),
                 version=np.array(key["version"]))
        os.replace(tmp, cache_path)
    except OSError as e:
        log.warning("could not write index cache %s: %s", cache_path, e)


# ------------------------------------------------------------------- probe --
def _stream_rate_candidates(stream) -> list:
    out = []
    for name in ("base_rate", "guessed_rate", "average_rate"):
        try:
            out.append(getattr(stream, name))
        except Exception:      # noqa: BLE001
            pass
    try:
        out.append(stream.codec_context.framerate)
    except Exception:          # noqa: BLE001
        pass
    return out


def _metadata_rate(stream) -> Fraction:
    """Best metadata frame rate (raw .dv: avg_frame_rate is bogus, base_rate is right)."""
    cands = [_rational(c) for c in _stream_rate_candidates(stream)]
    cands = [c for c in cands if c is not None and 1 <= c <= 240]
    if not cands:
        return Fraction(30000, 1001)
    # base_rate / guessed_rate first: they agree with the container timing
    return cands[0]


def _basic_info(container, stream, path: str, sar_override: Optional[float]) -> tuple[VideoInfo, Optional[str]]:
    cc = stream.codec_context
    width, height = int(cc.width), int(cc.height)
    sar_stream = stream.sample_aspect_ratio or cc.sample_aspect_ratio
    sar, sar_source, sar_note = resolve_sar(sar_stream, width, height, sar_override)
    rate = _metadata_rate(stream)
    info = VideoInfo(
        path=os.fspath(path), width=width, height=height, n_frames=0,
        frame_rate=float(rate), frame_rate_str=_fraction_str(rate),
        sar=sar, sar_source=sar_source, codec=cc.name, container=container.format.name,
        has_audio=len(container.streams.audio) > 0, file_size=os.path.getsize(path))
    return info, sar_note


def probe(path: PathLike, exact: bool = False, sar_override: Optional[float] = None) -> VideoInfo:
    """Fast metadata.  ``n_frames``/``duration_s`` are the container's claim
    unless ``exact=True`` (which builds the full index like `VideoReader`).

    For intra-only codecs in AVI/MOV/MP4 the container index count is exact;
    for other files it may be off by a few frames (e.g. WMV), and
    ``timing_mode`` is not verified.  Use `VideoReader` for analysis.
    """
    path = os.fspath(path)
    if exact:
        with VideoReader(path, sar_override=sar_override) as r:
            return r.info
    container = _open_container(path)
    try:
        if not container.streams.video:
            raise ValueError(f"{path}: no video stream")
        stream = container.streams.video[0]
        info, _ = _basic_info(container, stream, path, sar_override)
        n = 0
        if _codec_intra_only(stream) and container.format.name.split(",")[0] in _CONTAINER_INDEX_FORMATS:
            ts = _index_from_container(stream)
            n = len(ts) if ts is not None else 0
        if n == 0:
            n = int(stream.frames or 0)
        dur = None
        if stream.duration is not None and stream.time_base is not None:
            dur = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            dur = container.duration / 1e6
        if n == 0 and dur is not None:
            n = int(round(dur * info.frame_rate))
        info.n_frames = n
        info.duration_s = n / info.frame_rate if n else (dur or 0.0)
        return info
    finally:
        container.close()


# ------------------------------------------------------------------ reader --
class _Cursor:
    """A decoding position inside the random-access container."""

    def __init__(self, frames: Iterator, start_after_seek: bool):
        self.frames = frames
        self.last_index = -1          # index of the most recently decoded frame
        self.key_index: Optional[int] = None   # index of the first keyframe seen
        self.after_seek = start_after_seek


class VideoReader:
    """Frame-exact video access (see module docstring).

    Attributes:
        info: VideoInfo with exact n_frames, timing mode, dropped frames.
        frame_times: (n_frames,) seconds from the first frame.
        pts: (n_frames,) int64 presentation timestamps (stream time base).
        time_base: Fraction, seconds per PTS tick.
        frame_dt: nominal frame duration (s).
        index_method: how the index was built ('container'|'demux'|'decode', '+cache').
        notes: human-readable remarks (SAR assumption, VFR, dropped frames ...).
        decode_errors: list of dicts describing packets/frames that failed.
        video_start_s: PTS of frame 0 in seconds (container timeline).
        container_start_s: the container's start time in seconds.
    """

    def __init__(self, path: PathLike, sar_override: Optional[float] = None, *,
                 index_mode: str = "auto", index_cache: Optional[PathLike] = None,
                 cache_size: int = 32, threads: bool = True):
        if index_mode not in ("auto", "decode", "demux", "container"):
            raise ValueError(f"index_mode must be auto|decode|demux|container, got {index_mode!r}")
        self.path = os.fspath(path)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(self.path)
        self._threads = threads
        self._lock = threading.RLock()
        self._cache: "OrderedDict[tuple[int, str], object]" = OrderedDict()
        self._cache_size = max(0, int(cache_size))
        self._cursor: Optional[_Cursor] = None
        self.notes: list[str] = []
        self.decode_errors: list[dict] = []

        self._container = _open_container(self.path)
        if not self._container.streams.video:
            self._container.close()
            raise ValueError(f"{self.path}: no video stream")
        self._stream = self._container.streams.video[0]
        self._configure_stream(self._stream)
        self.time_base = Fraction(self._stream.time_base)
        info, sar_note = _basic_info(self._container, self._stream, self.path, sar_override)
        if sar_note:
            self.notes.append(sar_note)
        fmt_start = self._container.start_time
        self.container_start_s = (fmt_start / 1e6) if fmt_start is not None else 0.0

        idx = self._get_index(index_mode, os.fspath(index_cache) if index_cache else None)
        self.pts = idx.pts
        self.index_method = idx.method
        self.decode_errors.extend(idx.decode_errors)
        if len(self.pts) == 0:
            self._container.close()
            raise ValueError(f"{self.path}: no decodable video frames")
        self._pts_increasing = bool(len(self.pts) < 2 or np.all(np.diff(self.pts) > 0))
        self._seekable = self._pts_increasing and not idx.synthesized
        if idx.synthesized:
            self.notes.append("some frames had no timestamp; times were synthesized and random "
                              "access decodes from the start (slow)")
        self.video_start_s = float(self.pts[0] * self.time_base)

        timing = analyze_timing(self.pts.astype(np.float64) * float(self.time_base),
                                _stream_rate_candidates(self._stream))
        self.timing = timing
        self.frame_times = timing.times
        self.frame_dt = timing.dt
        self.notes.extend(timing.notes)
        if timing.dropped:
            self.notes.append(f"{len(timing.dropped)} timestamp gap(s) (dropped frames) before frames "
                              f"{timing.dropped[:10]}{' ...' if len(timing.dropped) > 10 else ''}")
        info.n_frames = len(self.pts)
        info.frame_rate = float(timing.rate)
        info.frame_rate_str = _fraction_str(timing.rate)
        info.timing_mode = timing.mode
        info.dropped_frames = list(timing.dropped)
        info.duration_s = float(self.frame_times[-1] + self.frame_dt)
        self.info = info
        self._forward_frames = max(2, int(round(FORWARD_DECODE_S / self.frame_dt)))

    # ------------------------------------------------------------- setup
    def _configure_stream(self, stream) -> None:
        if self._threads:
            try:
                stream.thread_type = "AUTO"
            except Exception:  # noqa: BLE001
                pass

    def _new_container(self):
        c = _open_container(self.path)
        s = c.streams.video[0]
        self._configure_stream(s)
        return c, s

    def _get_index(self, mode: str, cache_path: Optional[str]) -> _Index:
        if cache_path:
            cached = _load_cached_index(cache_path, self.path)
            if cached is not None:
                return cached
        idx = self._build_index(mode)
        if cache_path:
            _save_cached_index(cache_path, self.path, idx)
        return idx

    def _build_index(self, mode: str) -> _Index:
        stream = self._stream
        fmt = self._container.format.name.split(",")[0]
        intra = _codec_intra_only(stream)
        reorders = _codec_reorders(stream)
        tried = []
        if mode in ("auto", "container") and intra and fmt in _CONTAINER_INDEX_FORMATS:
            ts = _index_from_container(stream)
            tried.append("container")
            if ts is not None and self._validate_index(ts):
                return _Index(ts, "container")
        if mode in ("auto", "demux", "container") and (mode != "auto" or not reorders):
            c, s = self._new_container()
            try:
                ts = _index_from_demux(c, s)
            finally:
                c.close()
            tried.append("demux")
            if ts is not None and self._validate_index(ts):
                return _Index(ts, "demux")
        if tried:
            log.info("%s: %s index not usable, decoding all frames", self.path, "/".join(tried))
        c, s = self._new_container()
        try:
            step = max(1, int(round(1 / (float(_metadata_rate(s)) * float(self.time_base)))))
            return _index_from_decode(c, s, step)
        finally:
            c.close()

    def _validate_index(self, ts: np.ndarray) -> bool:
        """Decode the first and last frame and check they carry the indexed PTS."""
        if len(ts) == 0:
            return False
        c, s = self._new_container()
        try:
            first = None
            for f in _decoded_frames(c, s):
                first = f.pts
                break
            if first != ts[0]:
                return False
            if len(ts) == 1:
                return True
            target = int(ts[-1])
            back = int(round(2.0 / float(self.time_base)))
            c.seek(max(int(ts[0]), target - back), stream=s, backward=True, any_frame=False)
            last = None
            for f in _decoded_frames(c, s):
                last = f.pts
            return last == target
        except av.error.FFmpegError as e:
            log.info("index validation failed: %s", e)
            return False
        finally:
            c.close()

    # ------------------------------------------------------------ access
    @property
    def n_frames(self) -> int:
        return len(self.pts)

    def frame_time(self, i: int) -> float:
        return float(self.frame_times[i])

    def index_of_pts(self, pts: Optional[int]) -> Optional[int]:
        """Frame index of an exact PTS, or None if it is not in the index."""
        if pts is None or not self._pts_increasing:
            return None
        j = int(np.searchsorted(self.pts, pts))
        if j < len(self.pts) and self.pts[j] == pts:
            return j
        return None

    def frame_at_time(self, t: float) -> int:
        """Index of the frame whose time is closest to t (seconds)."""
        j = int(np.searchsorted(self.frame_times, t))
        if j <= 0:
            return 0
        if j >= self.n_frames:
            return self.n_frames - 1
        return j if abs(self.frame_times[j] - t) < abs(t - self.frame_times[j - 1]) else j - 1

    def _check_index(self, i: int) -> int:
        i = int(i)
        if i < 0:
            i += self.n_frames
        if not 0 <= i < self.n_frames:
            raise IndexError(f"frame {i} out of range [0, {self.n_frames})")
        return i

    def _cache_get(self, key):
        v = self._cache.get(key)
        if v is not None:
            self._cache.move_to_end(key)
        return v

    def _cache_put(self, key, value) -> None:
        if self._cache_size == 0:
            return
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def get_frame(self, i: int, fmt: str = "gray"):
        """Frame i as uint8 array: gray (H, W) or bgr (H, W, 3); fmt='both' -> (gray, bgr).

        Frame-exact: seeks before the target, decodes forward and matches PTS;
        retries further back and finally decodes from the start.
        Returns a copy (callers may modify it).
        """
        i = self._check_index(i)
        f = _fmt(fmt)
        fmts = ("gray", "bgr24") if f == "both" else (f,)
        with self._lock:
            vals = [self._cache_get((i, x)) for x in fmts]
            if any(v is None for v in vals):
                frame = self._decode_frame(i)
                vals = [v if v is not None else frame.to_ndarray(format=x) for v, x in zip(vals, fmts)]
                for x, v in zip(fmts, vals):
                    self._cache_put((i, x), v)
            out = tuple(v.copy() for v in vals)
        return out if f == "both" else out[0]

    def _decode_frame(self, i: int):
        cur = self._cursor
        ahead = self._forward_frames if self._seekable else self.n_frames
        if cur is not None and cur.key_index is not None and cur.last_index < i <= cur.last_index + ahead:
            fr = self._advance(cur, i)
            if fr is not None:
                return fr
        if self._seekable:
            for preroll in SEEK_PREROLLS_S:
                start_pts = self._preroll_pts(i, preroll)
                if start_pts is None:
                    break
                fr = self._advance(self._seek_cursor(start_pts), i)
                if fr is not None:
                    return fr
        fr = self._advance(self._seek_cursor(None), i)
        if fr is not None:
            return fr
        self.decode_errors.append({"frame": i, "error": "frame could not be decoded"})
        raise FrameDecodeError(f"{self.path}: frame {i} (pts {int(self.pts[i])}) could not be decoded")

    def _preroll_pts(self, i: int, preroll_s: float) -> Optional[int]:
        """Seek target `preroll_s` before frame i, or None if that is before frame 0."""
        p = int(self.pts[i]) - int(round(preroll_s / float(self.time_base)))
        return p if p > int(self.pts[0]) else None

    def _seek_cursor(self, start_pts: Optional[int]) -> _Cursor:
        """New cursor at a keyframe before `start_pts` (None: reopen at the start)."""
        if start_pts is None:
            self._container.close()
            self._container, self._stream = self._new_container()
        else:
            self._container.seek(start_pts, stream=self._stream, backward=True, any_frame=False)
        frames = self._indexed(self._decoded(self._container, self._stream))
        self._cursor = _Cursor(frames, start_after_seek=start_pts is not None)
        return self._cursor

    def _decoded(self, c, s) -> Iterator["av.VideoFrame"]:
        return _decoded_frames(c, s, self.decode_errors)

    def _indexed(self, frames: Iterator["av.VideoFrame"]) -> Iterator[tuple[int, "av.VideoFrame"]]:
        """Attach frame indices: by PTS lookup, or by counting when PTS are unusable."""
        if self._seekable:
            for fr in frames:
                j = self.index_of_pts(fr.pts)
                if j is not None:
                    yield j, fr
        else:
            yield from enumerate(frames)

    def _advance(self, cur: _Cursor, target: int):
        """Decode forward on `cur` until frame `target`; None if it cannot be reached cleanly.

        After a seek, frames are trusted only from the first keyframe on
        (open-GOP leading B-frames reference data from before the seek point).
        """
        for j, fr in cur.frames:
            if cur.after_seek and cur.last_index < 0 and j > target:
                return None                      # seek landed after the target
            cur.last_index = j
            if cur.key_index is None and (not cur.after_seek or _is_keyframe(fr)):
                cur.key_index = j
            if j >= target:
                if j == target and cur.key_index is not None:
                    return fr
                return None
        self._cursor = None
        return None

    def iter_frames(self, start: int = 0, stop: Optional[int] = None,
                    fmt: str = "gray") -> Iterator[tuple[int, object]]:
        """Yield (i, image) for start <= i < stop, in order.

        fmt: 'gray' (H, W) uint8, 'bgr' (H, W, 3), 'rgb', or 'both' -> (gray, bgr).
        Uses its own decoder, so it can run alongside `get_frame()`.
        A frame that fails to decode is logged, recorded in `decode_errors`
        and skipped (the index jump is visible to the caller); it is never
        replaced by a neighbour.
        """
        f = _fmt(fmt)
        n = self.n_frames
        start = max(0, int(start))
        stop = n if stop is None else min(n, int(stop))
        if start >= stop:
            return
        c, s = self._new_container()
        try:
            want = start
            for j, fr in self._iter_from(c, s, start):
                if j < want:
                    continue
                if j >= stop:
                    break
                if j > want:
                    self._report_missing(want, j)
                yield j, _to_array(fr, f)
                want = j + 1
            if want < stop:
                self._report_missing(want, stop)
        finally:
            c.close()

    def _report_missing(self, a: int, b: int) -> None:
        msg = f"frames {a}..{b - 1} could not be decoded and were skipped"
        log.warning("%s: %s", self.path, msg)
        self.decode_errors.append({"frames": [a, b - 1], "error": "not decoded during iteration"})

    def _iter_from(self, c, s, start: int) -> Iterator[tuple[int, "av.VideoFrame"]]:
        """(index, frame) pairs, clean from `start` on, starting at or before `start`."""
        if self._seekable and start > 0:
            for preroll in SEEK_PREROLLS_S:
                start_pts = self._preroll_pts(start, preroll)
                if start_pts is None:
                    break
                c.seek(start_pts, stream=s, backward=True, any_frame=False)
                frames = self._indexed(self._decoded(c, s))
                first = self._clean_start(frames, start)
                if first is not False:
                    if first is not None:
                        yield first
                    yield from frames
                    return
            c.close()
            c, s = self._new_container()       # from the very start
            try:
                yield from self._indexed(self._decoded(c, s))
            finally:
                c.close()
            return
        yield from self._indexed(self._decoded(c, s))

    @staticmethod
    def _clean_start(frames: Iterator, start: int):
        """Consume frames up to the first clean one at index >= start.

        Returns that (j, frame) pair, None if the stream ended first, or
        False if no keyframe was seen at or before `start` (retry earlier).
        """
        key_seen = False
        for j, fr in frames:
            if not key_seen:
                if j > start:
                    return False
                key_seen = _is_keyframe(fr)
                if not key_seen:
                    continue
            if j >= start:
                return j, fr
        return None if key_seen else False

    # ----------------------------------------------------------- lifecycle
    def close(self) -> None:
        with self._lock:
            self._cursor = None
            self._cache.clear()
            if self._container is not None:
                self._container.close()
                self._container = None

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        i = self.info
        return (f"VideoReader({os.path.basename(self.path)!r}, {i.width}x{i.height}, {i.n_frames} frames, "
                f"{i.frame_rate_str} fps {i.timing_mode}, sar {i.sar:.4f} ({i.sar_source}), "
                f"index={self.index_method})")

