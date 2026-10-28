"""Encode rendered BGR frames with an ffmpeg subprocess (raw frames on stdin).

The ffmpeg binary is found in this order: $PITCHSPEED_FFMPEG, the binary
bundled with imageio-ffmpeg (a package dependency, also on Windows), then
`ffmpeg` on PATH.

Filter chain for interlaced output (all verified in the prototype, synth.md
sections 5 and 8):  setfield=tff|bff, scale=interl=1 (interlace-aware 4:2:0
chroma), setsar, format.  MPEG-2: +ilme+ildct, -top, alternate scan.  x264:
interlaced=1:tff=1|bff=1 (without the explicit flag x264 marks BFF as TFF).
DV: dvvideo yuv411p (720x480 only).  FFV1: lossless bgr0 (bit-exact round
trip).  The *metadata* field order can be set to lie on purpose
(video.metadata_field_order).  Postprocess 'yadif' / 'yadif_bob' runs
ffmpeg's yadif=0 / yadif=1 on the interlaced render (a capture tool that
deinterlaced the tape); 'yadif_bob' doubles the frame rate (out frame m =
field m).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from typing import Optional

DEFAULT_EXT = {"mpeg2": ".mpg", "dv": ".avi", "h264": ".mp4", "ffv1": ".mkv", "mpeg4": ".avi"}


class EncoderError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    """Path of the ffmpeg executable to use."""
    env = os.environ.get("PITCHSPEED_FFMPEG")
    if env:
        return env
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:                       # noqa: BLE001 - fall back to PATH
        pass
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise EncoderError("ffmpeg not found: install imageio-ffmpeg or put ffmpeg on PATH "
                       "(or set PITCHSPEED_FFMPEG)")


@lru_cache(maxsize=1)
def ffmpeg_version() -> str:
    try:
        out = subprocess.run([ffmpeg_exe(), "-version"], capture_output=True, text=True, timeout=30).stdout
        return out.splitlines()[0].strip() if out else "unknown"
    except Exception:                       # noqa: BLE001
        return "unknown"


def output_rate(scan: str, postprocess: Optional[str]) -> str:
    """Frame rate (rational string) of the encoded file."""
    if scan == "progressive60" or postprocess == "yadif_bob":
        return "60000/1001"
    return "30000/1001"


def input_rate(scan: str) -> str:
    """Rate of the raw frames piped to ffmpeg."""
    return "60000/1001" if scan == "progressive60" else "30000/1001"


def is_interlaced_output(video: dict) -> bool:
    """True when the encoded file holds interlaced frames (fields not mixed or dropped)."""
    return video["scan"] == "interlaced" and video["postprocess"] is None


def ffmpeg_cmd(video: dict, out_path: str, true_tff: bool) -> list:
    """ffmpeg command line encoding raw bgr24 frames from stdin to out_path."""
    V = video
    br = int(V["bitrate_kbps"])
    codec = V["codec"]
    interlaced_out = is_interlaced_output(V)
    mfo = V.get("metadata_field_order", "auto")
    meta_tff = true_tff if mfo == "auto" else (mfo == "tff")
    vf = []
    if V.get("ffmpeg_vf"):
        vf.append(str(V["ffmpeg_vf"]))
    post = V.get("postprocess")
    if post in ("yadif", "yadif_bob"):
        vf += ["setfield=" + ("tff" if true_tff else "bff"), "yadif=%d" % (1 if post == "yadif_bob" else 0)]
    if interlaced_out:
        vf += ["setfield=" + ("tff" if meta_tff else "bff"), "scale=interl=1"]
    par = V["par"]
    vf.append("setsar=%d/%d" % (int(par[0]), int(par[1])))
    vf.append({"dv": "format=yuv411p", "ffv1": "format=bgr0"}.get(codec, "format=yuv420p"))
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", "%dx%d" % (int(V["width"]), int(V["height"])), "-r", input_rate(V["scan"]), "-i", "-",
           "-vf", ",".join(vf), "-an"]
    il = ["-flags", "+ilme+ildct"] if interlaced_out else []
    if codec == "mpeg2":
        cmd += ["-c:v", "mpeg2video", "-b:v", "%dk" % br, "-maxrate", "%dk" % int(br * 1.6),
                "-bufsize", "%dk" % int(br * 1.2), "-g", "15", "-bf", "2"] + il
        if interlaced_out:
            cmd += ["-top", "1" if meta_tff else "0", "-alternate_scan", "1"]
    elif codec == "h264":
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-b:v", "%dk" % br, "-threads", "1"] + il
        if interlaced_out:
            cmd += ["-x264-params", "interlaced=1:" + ("tff=1" if meta_tff else "bff=1")]
    elif codec == "mpeg4":
        cmd += ["-c:v", "mpeg4", "-vtag", "XVID", "-b:v", "%dk" % br] + il
    elif codec == "dv":
        cmd += ["-c:v", "dvvideo"]
    elif codec == "ffv1":
        cmd += ["-c:v", "ffv1", "-level", "3"]
    else:
        raise ValueError(f"unknown codec {codec!r}")
    if interlaced_out:
        cmd += ["-field_order", "tt" if meta_tff else "bb"]
    return cmd + [str(out_path)]


class Encoder:
    """One ffmpeg process fed with raw BGR frames."""

    def __init__(self, video: dict, out_path: str, true_tff: bool):
        self.out_path = str(out_path)
        self.cmd = ffmpeg_cmd(video, self.out_path, true_tff)
        self._err = tempfile.TemporaryFile()
        creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)     # no console flash on Windows
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=self._err, creationflags=creation)
        self.n_written = 0

    def _stderr(self) -> str:
        try:
            self._err.seek(0)
            return self._err.read().decode("utf-8", "replace").strip()
        except Exception:                   # noqa: BLE001
            return ""

    def write(self, frame_bytes: bytes) -> None:
        try:
            self.proc.stdin.write(frame_bytes)
        except (BrokenPipeError, OSError) as e:
            self.proc.wait()
            raise EncoderError(f"ffmpeg stopped while encoding {self.out_path}: {self._stderr() or e}") from e
        self.n_written += 1

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        rc = self.proc.wait()
        msg = self._stderr()
        self._err.close()
        if rc != 0:
            raise EncoderError(f"ffmpeg failed (exit {rc}) for {self.out_path}: {msg}\ncommand: {' '.join(self.cmd)}")

    def abort(self) -> None:
        try:
            self.proc.kill()
            self.proc.wait()
        except Exception:                   # noqa: BLE001
            pass
        try:
            self._err.close()
        except Exception:                   # noqa: BLE001
            pass
