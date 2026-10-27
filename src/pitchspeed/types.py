"""Shared data structures passed between modules.

COORDINATES.  All image positions are *frame pixel coordinates* of the
stored (non-square) pixels: x to the right, y down, integer values at pixel
centres.  For a field sample of parity p, field row r is frame row
y = 2*r + p.  Positions attached to candidates / tracks are additionally
*stabilized*: they are expressed in the coordinates of the calibration
reference frame (Calibration.ref_frame), so they can be compared directly
with the clicked calibration points.  Square-pixel coordinates (for
Euclidean distances / angles) are (x * sar, y).

TIME.  Seconds from the first decoded video frame.  For interlaced video a
sample is one field: t = t_frame + k * frame_dt / 2, k = temporal order of
the field inside its frame (0 or 1).  The exposure-midpoint offset is a
constant and cancels in every speed estimate.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------- video ----
@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    n_frames: int
    frame_rate: float              # base frames/s, e.g. 29.97003
    frame_rate_str: str            # exact rational as text, e.g. "30000/1001"
    sar: float = 1.0               # stored-pixel aspect: x_square = x * sar
    sar_source: str = "default"    # 'stream' | 'default' | 'override'
    codec: str = ""
    container: str = ""
    duration_s: float = 0.0
    timing_mode: str = "cfr"       # 'cfr' | 'vfr'
    has_audio: bool = False
    dropped_frames: list[int] = field(default_factory=list)    # frame i has a PTS gap before it
    duplicate_frames: list[int] = field(default_factory=list)  # frame i ~ identical to i-1
    file_size: int = 0


@dataclass
class ScanInfo:
    mode: str                      # 'interlaced' | 'progressive'
    field_order: Optional[str]     # 'tff' | 'bff' | None (progressive)
    confidence: str = "low"        # 'high' | 'medium' | 'low'
    source: str = "content"        # 'content' | 'override' | 'default'
    sample_rate: float = 0.0       # samples (fields or frames) per second
    evidence: dict = field(default_factory=dict)

    @property
    def interlaced(self) -> bool:
        return self.mode == "interlaced"


@dataclass
class Sample:
    """One field (interlaced) or one frame (progressive)."""
    s: int                         # global sample index, 0-based, time ordered
    frame: int                     # source frame index
    k: int                         # temporal order inside the frame (0/1); 0 if progressive
    parity: int                    # 0 = top (even rows), 1 = bottom; 0 if progressive
    t: float                       # seconds
    image: np.ndarray              # uint8 gray, shape (H/2, W) for a field, (H, W) for a frame
    bgr: Optional[np.ndarray] = None   # optional colour version (same shape + 3)


# ------------------------------------------------------------- tracking ----
@dataclass
class Candidate:
    s: int
    t: float
    x: float                       # stabilized frame coords
    y: float
    score: float                   # detection score (z_eff, ~ SNR)
    length: float = 0.0            # streak length (square px)
    angle: float = 0.0             # streak angle (radians, square-pixel space)
    attached: bool = False         # touches a large moving blob (hand/glove/body)
    sigma: float = 1.0             # 1-sigma position error (px)


@dataclass
class TrackPoint:
    s: int
    t: float
    x: float                       # stabilized frame coords
    y: float
    score: float = 0.0
    sigma: float = 1.0
    inlier: bool = True
    manual: bool = False


@dataclass
class EventTime:
    t: float                       # seconds
    sigma: float                   # 1-sigma (seconds)
    method: str                    # e.g. 'first_free_field', 'extrapolated', 'side_fit', 'manual'
    s: Optional[int] = None        # nearest sample index (for UI / reports)


@dataclass
class PitchTrack:
    track_id: int
    points: list[TrackPoint]
    release: Optional[EventTime] = None
    catch: Optional[EventTime] = None
    end_kind: str = "catch"        # 'catch' | 'contact' | 'lost' | 'occluded'
    status: str = "pitch"          # 'pitch' | 'rejected'
    reject_code: Optional[str] = None
    confidence: float = 0.0        # 0..1 heuristic
    flags: list[str] = field(default_factory=list)
    model: dict = field(default_factory=dict)   # fitted curve params for drawing/extrapolation
    diagnostics: dict = field(default_factory=dict)

    @property
    def t_first(self) -> float:
        return self.points[0].t if self.points else math.nan

    @property
    def t_last(self) -> float:
        return self.points[-1].t if self.points else math.nan


# ---------------------------------------------------------------- speed ----
@dataclass
class SpeedComponent:
    method: str                    # 'tof' | 'side_fit'
    v_release_mph: float
    sigma_ind_mph: float           # independent part of 1-sigma
    sigma_shared_mph: float = 0.0  # part shared between methods (drag k)
    v_avg_mph: Optional[float] = None
    details: dict = field(default_factory=dict)


@dataclass
class SpeedEstimate:
    v_release_mph: Optional[float]           # None => not measurable
    sigma_mph: Optional[float] = None        # 1-sigma
    ci90_mph: Optional[tuple] = None         # (lo, hi)
    v_avg_mph: Optional[float] = None        # release->catch average
    v_plate_mph: Optional[float] = None      # at the front edge of the plate
    method: str = "none"                     # 'tof' | 'side_fit' | 'tof+side_fit' | 'manual_tof' | 'none'
    label: str = "none"                      # 'good' | 'fair' | 'poor' | 'none'
    components: list[SpeedComponent] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    assumptions: dict = field(default_factory=dict)
    reason: Optional[str] = None             # why not measurable / caveats


# -------------------------------------------------------------- results ----
@dataclass
class Pitch:
    number: int                    # 1-based, chronological
    status: str                    # 'auto' | 'edited' | 'manual' | 'rejected'
    release: EventTime
    catch: EventTime
    speed: SpeedEstimate
    frame_release: int = -1
    frame_catch: int = -1
    track: Optional[PitchTrack] = None
    confidence: float = 0.0
    flags: list[str] = field(default_factory=list)
    notes: str = ""
    edit_history: list[dict] = field(default_factory=list)


@dataclass
class AnalysisResult:
    video: VideoInfo
    scan: ScanInfo
    calibration: dict
    params: dict
    pitches: list[Pitch]
    rejected_tracks: list[PitchTrack] = field(default_factory=list)
    possible_missed: list[dict] = field(default_factory=list)   # {t_start, t_end, s_start, s_end, reason}
    warnings: list[str] = field(default_factory=list)
    schema: str = "pitchspeed.results/1"
    version: str = ""


# -------------------------------------------------------- serialization ----
def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses / numpy values into JSON-friendly data."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = {}
        for f in dataclasses.fields(obj):
            if f.name in ("image", "bgr"):
                continue
            out[f.name] = to_jsonable(getattr(obj, f.name))
        return out
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def _from(cls, d):
    if d is None:
        return None
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in d:
            kwargs[f.name] = d[f.name]
    return cls(**kwargs)


def track_from_dict(d: Optional[dict]) -> Optional[PitchTrack]:
    if d is None:
        return None
    d = dict(d)
    d["points"] = [_from(TrackPoint, p) for p in d.get("points", [])]
    d["release"] = _from(EventTime, d.get("release"))
    d["catch"] = _from(EventTime, d.get("catch"))
    return _from(PitchTrack, d)


def speed_from_dict(d: dict) -> SpeedEstimate:
    d = dict(d)
    d["components"] = [_from(SpeedComponent, c) for c in d.get("components", [])]
    if d.get("ci90_mph") is not None:
        d["ci90_mph"] = tuple(d["ci90_mph"])
    return _from(SpeedEstimate, d)


def pitch_from_dict(d: dict) -> Pitch:
    d = dict(d)
    d["release"] = _from(EventTime, d["release"])
    d["catch"] = _from(EventTime, d["catch"])
    d["speed"] = speed_from_dict(d["speed"])
    d["track"] = track_from_dict(d.get("track"))
    return _from(Pitch, d)


def result_from_dict(d: dict) -> AnalysisResult:
    d = dict(d)
    d["video"] = _from(VideoInfo, d["video"])
    d["scan"] = _from(ScanInfo, d["scan"])
    d["pitches"] = [pitch_from_dict(p) for p in d.get("pitches", [])]
    d["rejected_tracks"] = [track_from_dict(t) for t in d.get("rejected_tracks", [])]
    return _from(AnalysisResult, d)
