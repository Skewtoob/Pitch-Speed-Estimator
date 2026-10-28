"""Builders and fakes for tests/test_render_*.py.

* `rich_result()`  an AnalysisResult with every field populated (good, fair,
  poor, none, rejected, manual, edited pitches; rejected tracks; possible
  missed windows; numpy values; non-ASCII text).
* `FakeSource`     a SampleSource stand-in yielding synthetic Samples.
* `FakeWriter`     collects frames instead of encoding them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pitchspeed.constants import NTSC_FIELD_DT, NTSC_FRAME_DT
from pitchspeed.types import (AnalysisResult, EventTime, Pitch, PitchTrack, Sample, ScanInfo,
                              SpeedComponent, SpeedEstimate, TrackPoint, VideoInfo)

FIELD_DT = NTSC_FIELD_DT


def assumptions() -> dict:
    return {"distance_ft": 60.5, "level": "hs", "sport": "baseball", "extension_ft": 5.5,
            "extension_sigma_ft": 1.0, "extension_assumed": True, "catch_behind_ft": 1.5,
            "cd": 0.33, "air_density_kg_m3": 1.204, "drag_k_per_ft": 0.00178, "temp_f": None}


def speed(v: Optional[float], sigma: Optional[float], label: str, method: str = "tof",
          reason: Optional[str] = None, flags=()) -> SpeedEstimate:
    if v is None:
        return SpeedEstimate(v_release_mph=None, method=method, label="none", flags=list(flags),
                             assumptions=assumptions(), reason=reason)
    comp = SpeedComponent(method="tof", v_release_mph=v, sigma_ind_mph=sigma * 0.8,
                          sigma_shared_mph=sigma * 0.6, v_avg_mph=v - 4.0,
                          details={"T_s": 0.47, "L_ft": 55.3, "v_plate_mph": v - 7.0,
                                   "flags": list(flags), "budget": {"t": 0.01, "ext": 0.02}})
    return SpeedEstimate(v_release_mph=v, sigma_mph=sigma, ci90_mph=(v - 1.645 * sigma, v + 1.645 * sigma),
                         v_avg_mph=v - 4.0, v_plate_mph=v - 7.0, method=method, label=label,
                         components=[comp], flags=list(flags), assumptions=assumptions(), reason=reason)


def track(tid: int, s0: int, s1: int, x0=100.0, x1=600.0, y=240.0, status="pitch",
          code=None, manual_at=()) -> PitchTrack:
    pts = []
    for s in range(s0, s1 + 1):
        f = (s - s0) / max(1, s1 - s0)
        pts.append(TrackPoint(s=s, t=s * FIELD_DT, x=x0 + f * (x1 - x0), y=y + 10 * f * f,
                              score=12.5, sigma=0.8, inlier=(s != s0 + 3), manual=s in manual_at))
    rel = EventTime(t=s0 * FIELD_DT - 0.004, sigma=0.006, method="first_free_field", s=s0)
    cat = EventTime(t=s1 * FIELD_DT + 0.002, sigma=0.004, method="extrapolated", s=s1)
    return PitchTrack(track_id=tid, points=pts, release=rel, catch=cat, end_kind="catch",
                      status=status, reject_code=code, confidence=0.8,
                      flags=["catch_extrapolated"] if status == "pitch" else ["low_contrast"],
                      model={"vx": np.float64(10.5), "coef": np.array([1.0, 2.0])},
                      diagnostics={"t_first_obs": s0 * FIELD_DT, "s_first_obs": np.int64(s0),
                                   "nan_value": float("nan")})


def pitch(number: int, s0: int, s1: int, sp: SpeedEstimate, status="auto", notes="",
          with_track=True) -> Pitch:
    tr = track(number, s0, s1) if with_track else None
    rel = EventTime(t=s0 * FIELD_DT - 0.004, sigma=0.006,
                    method="manual" if status == "manual" else "first_free_field", s=s0)
    cat = EventTime(t=s1 * FIELD_DT + 0.002, sigma=0.004,
                    method="manual" if status == "manual" else "extrapolated", s=s1,
                    point="contact" if status == "manual" else None)   # manual pitch: a hit ball (B)
    hist = [{"action": "set_release", "s": s0, "t": rel.t}] if status in ("edited", "manual") else []
    return Pitch(number=number, status=status, release=rel, catch=cat, speed=sp,
                 frame_release=s0 // 2, frame_catch=s1 // 2, track=tr, confidence=0.83,
                 flags=list(tr.flags) if tr else [], notes=notes, edit_history=hist)


def rich_result(path: str = "C:\\Videos\\Spiel 2003 \u00e9t\u00e9.avi") -> AnalysisResult:
    """Pitches (chronological): 1 good, 2 poor, 3 none, rejected, 4 manual, 5 edited fair."""
    ps = [
        pitch(1, 600, 628, speed(78.4, 1.6, "good", flags=["catch_extrapolated"])),
        pitch(2, 1400, 1430, speed(65.2, 5.4, "poor", reason="release time uncertain",
                                   flags=["release_unconfirmed"])),
        pitch(3, 2200, 2228, speed(None, None, "none", method="none",
                                   reason="not measurable: uncertainty +-14% (> 10%)")),
        pitch(0, 3000, 3030, speed(40.0, 2.0, "good"), status="rejected", notes="throw to first"),
        pitch(4, 3800, 3828, speed(71.0, 2.9, "fair", method="manual_tof", flags=["manual_events"]),
              status="manual", notes="marked by hand \u00b1 1 field", with_track=False),
        pitch(5, 4600, 4629, speed(80.25, 2.6, "fair", method="tof+side_fit", reason="E1/E2 agree",
                                   flags=["oblique_view"]),
              status="edited", notes="release moved 1 field"),
    ]
    info = VideoInfo(path=path, width=720, height=480, n_frames=3000, frame_rate=30000 / 1001,
                     frame_rate_str="30000/1001", sar=8 / 9, sar_source="default", codec="dvvideo",
                     container="avi", duration_s=3000 * NTSC_FRAME_DT, timing_mode="cfr",
                     has_audio=True, dropped_frames=[1234], duplicate_frames=[17, 18], file_size=123456789)
    scan = ScanInfo(mode="interlaced", field_order="bff", confidence="high", source="content",
                    sample_rate=60000 / 1001, evidence={"ratios": np.array([2.3, 2.4]), "votes": (5, 1)})
    calib = {"schema": "pitchspeed.calibration/1",
             "settings": {"distance_ft": 60.5, "level": "hs", "sport": "baseball"},
             "segments": [{"start_frame": 0, "end_frame": 2000, "ref_frame": 10, "view": "side",
                           "points": {"rubber": [600.0, 300.0], "plate": [100.0, 320.0]}, "masks": []},
                          {"start_frame": 2000, "end_frame": None, "ref_frame": 2010, "view": "behind",
                           "points": {"release": [360.0, 150.0], "catch": [362.0, 400.0]}, "masks": []}],
             "video": {"name": "x.avi"}, "image_size": [720, 480], "notes": ""}
    rejected = [track(90, 5000, 5006, status="rejected", code="TOO_SHORT"),
                track(91, 5200, 5240, status="rejected", code="WRONG_DIRECTION")]
    missed = [{"s_start": 5400, "s_end": 5440, "t_start": 5400 * FIELD_DT, "t_end": 5440 * FIELD_DT,
               "reason": "rejected_track:PARTIAL"}]
    return AnalysisResult(video=info, scan=scan, calibration=calib,
                          params={"sample_dt": FIELD_DT, "finder": {}, "start_s": None},
                          pitches=ps, rejected_tracks=rejected, possible_missed=missed,
                          warnings=["1 dropped-frame gaps in the capture"], version="0.1.0")


# ------------------------------------------------------------------ fakes
@dataclass
class _Info:
    path: str = ""
    frame_rate: float = 30000 / 1001
    frame_rate_str: str = "30000/1001"
    sar: float = 8 / 9
    has_audio: bool = False
    width: int = 720
    height: int = 480


@dataclass
class _Reader:
    info: _Info = field(default_factory=_Info)


class FakeSource:
    """Synthetic SampleSource: interlaced (fields) or progressive (frames).

    Sample s shows a grey ramp with the sample number encoded in pixel
    [0, 0] (s % 256) so tests can check which sample produced a frame.
    `missing` lists sample indices that iter_samples skips (decode gaps).
    """

    def __init__(self, n_samples: int, interlaced: bool = True, width: int = 720, height: int = 480,
                 missing=(), path: str = "", has_audio: bool = False, time_offset: dict | None = None):
        self.interlaced = interlaced
        self.n_samples = n_samples
        self.frame_dt = NTSC_FRAME_DT
        self.sample_dt = self.frame_dt / 2 if interlaced else self.frame_dt
        rows = height // 2 if interlaced else height
        self.sample_shape = (rows, width)
        self.reader = _Reader(_Info(path=path, has_audio=has_audio, width=width, height=height))
        self.missing = set(missing)
        self.time_offset = dict(time_offset or {})      # s -> extra seconds (timing gaps)
        self.calls: list[tuple[int, int]] = []

    def locate(self, s: int):
        if self.interlaced:
            return s // 2, s % 2, (s % 2) ^ 1          # BFF
        return s, 0, 0

    def sample_time(self, s: int) -> float:
        extra = sum(v for k, v in self.time_offset.items() if s >= k)
        return s * self.sample_dt + extra

    def sample_times(self, s=None):
        s = np.arange(self.n_samples) if s is None else np.asarray(s)
        return np.array([self.sample_time(int(i)) for i in s])

    def sample_at_time(self, t: float) -> int:
        return int(np.clip(round(t / self.sample_dt), 0, self.n_samples - 1))

    def get_sample(self, s: int, color: bool = False) -> Sample:
        fr, k, p = self.locate(s)
        img = np.tile(np.linspace(30, 200, self.sample_shape[1]).astype(np.uint8), (self.sample_shape[0], 1))
        img[0, 0] = s % 256
        bgr = np.dstack([img, img, img]) if color else None
        return Sample(s=s, frame=fr, k=k, parity=p, t=self.sample_time(s), image=img, bgr=bgr)

    def iter_samples(self, s0: int = 0, s1: Optional[int] = None, color: bool = False):
        s1 = self.n_samples if s1 is None else min(s1, self.n_samples)
        self.calls.append((s0, s1))
        for s in range(max(0, s0), s1):
            if s not in self.missing:
                yield self.get_sample(s, color)


class FakeWriter:
    """Records the constructor arguments and every frame written."""

    instances: list["FakeWriter"] = []

    def __init__(self, path, width, height, fps="60000/1001", audio_src=None, audio_start_s=0.0,
                 audio_duration_s=None, crf=18, **kw):
        self.path, self.width, self.height, self.fps = path, width, height, fps
        self.audio_src, self.audio_start_s, self.audio_duration_s = audio_src, audio_start_s, audio_duration_s
        self.frames: list[np.ndarray] = []
        self.closed = False
        self.backend = "fake"
        self.out_path = path
        self.audio_status = "not requested" if audio_src is None else "muxed"
        FakeWriter.instances.append(self)

    def write(self, bgr: np.ndarray) -> None:
        assert bgr.shape == (self.height, self.width, 3) and bgr.dtype == np.uint8
        self.frames.append(bgr.copy())

    def close(self) -> str:
        self.closed = True
        return self.backend

    def abort(self) -> None:
        self.closed = True


# ------------------------------------------------------- real media helpers
BEEP_PHASE_S = 0.25          # beeps start 0.25 s after every whole second of the audio
BEEP_LEN_S = 0.03


def make_test_clip(folder: str, seconds: float = 9.0, name: str = "clip.mpg") -> str:
    """Interlaced (TFF) 720x480 MPEG-2 program stream with MP2 audio beeps.

    The 59.94p testsrc2 counter is woven into fields, so field s of the clip
    shows counter value s.  MPEG-PS starts its video ~10 ms after the audio
    (like real DVD/MPEG captures), which exercises the A/V offset handling."""
    import os
    import subprocess

    import imageio_ffmpeg
    out = os.path.join(folder, name)
    beep = (f"aevalsrc='if(between(mod(t\\,1)\\,{BEEP_PHASE_S}\\,{BEEP_PHASE_S + BEEP_LEN_S})"
            f"\\,sin(2*PI*1000*t)\\,0)':s=48000")
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", "testsrc2=size=720x480:rate=60000/1001", "-f", "lavfi", "-i", beep,
           "-t", f"{seconds}", "-vf", "interlace=scan=tff:lowpass=0,setsar=8/9",
           "-c:v", "mpeg2video", "-flags", "+ilme+ildct", "-top", "1", "-q:v", "4",
           "-c:a", "mp2", "-b:a", "192k", out]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    return out


def audio_onsets(path: str) -> list[float]:
    """Beep onset times (s) relative to the first video frame of `path`."""
    import av
    with av.open(path) as c:
        v0 = None
        for f in c.decode(video=0):
            v0 = float(f.pts * f.time_base)
            break
    chunks, t0, sr = [], None, None
    with av.open(path) as c:
        for f in c.decode(audio=0):
            a = f.to_ndarray()
            a = a[0] if f.format.is_planar else a.reshape(-1, len(f.layout.channels))[:, 0]
            a = a.astype(np.float64)
            if f.format.name.startswith("s16"):
                a /= 32768.0
            if t0 is None:
                t0, sr = float(f.pts * f.time_base), f.sample_rate
            chunks.append(a)
    x = np.abs(np.concatenate(chunks))
    idx = np.nonzero(x > 0.3 * x.max())[0]
    starts = [idx[0]] + [j for i, j in zip(idx[:-1], idx[1:]) if j - i > sr * 0.1]
    return [t0 + i / sr - v0 for i in starts]
