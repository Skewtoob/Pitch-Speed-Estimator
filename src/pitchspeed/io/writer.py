"""Annotated-video writer: BGR frames -> H.264 MP4, optionally with the source audio.

Video backends, tried in this order (automatic, each switch is logged):

1. ``'ffmpeg-libx264'`` - raw BGR frames piped into imageio-ffmpeg's bundled
   ffmpeg: libx264, yuv420p (mandatory: bgr24 input would otherwise give
   4:4:4, which many players refuse), High profile, -movflags +faststart.
2. ``'pyav-libx264'`` - the same encoder through PyAV (its wheels ship
   libx264).
3. ``'opencv-mp4v'`` - cv2.VideoWriter MPEG-4 Part 2 in MP4.
4. ``'opencv-mjpg'`` - MJPG in AVI; the output suffix becomes ``.avi``
   (see `VideoWriter.out_path`).

OpenCV's 'avc1'/'H264' writers are never used (missing/unshippable H.264
encoders on Windows and Linux wheels).  A backend that fails while starting
(within the first REPLAY_FRAMES frames) is replaced by the next one and the
buffered frames are replayed; a failure later raises RuntimeError with the
ffmpeg error output.

AUDIO.  With ``audio_src`` the video is first encoded to a temporary file
next to the output, then ``close()`` muxes it with the source audio in one
ffmpeg pass (video stream-copied, audio AAC-LC 160 kb/s):

    ffmpeg -i VIDEO_TMP -ss SHIFT -t DUR -i SRC -map 0:v:0 -map 1:a:0
           -c:v copy -af aresample=async=1:first_pts=0 -c:a aac ...

SHIFT = (first video frame PTS - container start) + audio_start_s: the
ffmpeg CLI rebases every input to its container start, and our time 0 is
the first video frame (MPEG-PS video starts ~10 ms after the container, a
naive offset was off by 523 ms in io_ux.md section 0).  DUR =
audio_duration_s, or the length of the written video when None (so no
``-shortest``, which broke the pipe in io_ux.md).  ``aresample
first_pts=0`` pads with silence when the source audio starts later than
SHIFT.  If the source has no audio, or muxing fails, the video is kept
without audio and the reason is logged and stored in `audio_status`.
"""
from __future__ import annotations

import collections
import functools
import logging
import os
import subprocess
import threading
from fractions import Fraction
from typing import Optional, Union

import numpy as np

log = logging.getLogger(__name__)

BACKENDS = ("ffmpeg-libx264", "pyav-libx264", "opencv-mp4v", "opencv-mjpg")
REPLAY_FRAMES = 8
FFMPEG_CLOSE_TIMEOUT_S = 600.0
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0      # CREATE_NO_WINDOW on Windows


class WriterError(RuntimeError):
    """The video could not be written."""


# ------------------------------------------------------------------- helpers
def _default_ffmpeg() -> Optional[str]:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:                       # noqa: BLE001
        log.info("imageio-ffmpeg's ffmpeg is not available: %s", e)
        return None


@functools.lru_cache(maxsize=8)
def ffmpeg_has_encoder(exe: str, name: str) -> bool:
    """True if `exe -encoders` lists encoder `name`."""
    try:
        p = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, timeout=30,
                           creationflags=_NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return False
    return any(len(line.split()) > 1 and line.split()[1] == name
               for line in p.stdout.decode("utf-8", "replace").splitlines())


def _fraction(fps: Union[str, float, int, Fraction]) -> Fraction:
    try:
        f = Fraction(fps) if not isinstance(fps, float) else Fraction(fps).limit_denominator(1001000)
    except (ValueError, ZeroDivisionError) as e:
        raise ValueError(f"bad fps {fps!r}") from e
    if f <= 0:
        raise ValueError(f"fps must be > 0, got {fps!r}")
    return f


def _fraction_str(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}"


def _tmp_path(path: str, tag: str) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}.{tag}-tmp{ext or '.mp4'}"


def _remove(path: Optional[str]) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError as e:
            log.warning("could not remove %s: %s", path, e)


def source_audio_offset(path: str) -> tuple[bool, float]:
    """(has_audio, first video frame time - container start) of a source file, seconds."""
    import av
    with av.open(os.fspath(path)) as c:
        has_audio = len(c.streams.audio) > 0
        cstart = (c.start_time / 1e6) if c.start_time is not None else 0.0
        if not c.streams.video:
            return has_audio, 0.0
        s = c.streams.video[0]
        vstart = None
        for fr in c.decode(s):
            if fr.pts is not None:
                vstart = float(fr.pts * fr.time_base)
            break
        if vstart is None:
            vstart = float(s.start_time * s.time_base) if s.start_time is not None else cstart
    return has_audio, max(0.0, vstart - cstart)


# ------------------------------------------------------------------ backends
class _FFmpegPipe:
    name = "ffmpeg-libx264"

    def __init__(self, exe: str, path: str, w: int, h: int, fps: Fraction, crf: int, faststart: bool):
        if not ffmpeg_has_encoder(exe, "libx264"):
            raise WriterError(f"{exe} has no libx264 encoder")
        cmd = [exe, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", _fraction_str(fps), "-i", "-",
               "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(int(crf)),
               "-pix_fmt", "yuv420p", "-profile:v", "high"]
        if faststart:
            cmd += ["-movflags", "+faststart"]
        cmd += [path]
        self.path = path
        self._tail: collections.deque = collections.deque(maxlen=40)
        try:
            self._p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.PIPE, creationflags=_NO_WINDOW)
        except OSError as e:
            raise WriterError(f"cannot start ffmpeg: {e}") from e
        self._t = threading.Thread(target=self._drain, daemon=True)
        self._t.start()

    def _drain(self) -> None:
        for line in iter(self._p.stderr.readline, b""):
            self._tail.append(line.decode("utf-8", "replace").rstrip())

    def errors(self) -> str:
        self._t.join(timeout=2.0)
        return "\n".join(self._tail) or f"exit code {self._p.poll()}"

    def write(self, bgr: np.ndarray) -> None:
        try:
            self._p.stdin.write(bgr.tobytes())
        except (BrokenPipeError, OSError, ValueError) as e:
            try:
                self._p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._p.kill()
            raise WriterError(f"ffmpeg stopped accepting frames ({type(e).__name__}): {self.errors()}") from e

    def close(self) -> None:
        try:
            self._p.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            rc = self._p.wait(timeout=FFMPEG_CLOSE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self._p.kill()
            raise WriterError("ffmpeg did not finish in time") from None
        if rc != 0:
            raise WriterError(f"ffmpeg failed (exit {rc}): {self.errors()}")

    def abort(self) -> None:
        try:
            self._p.kill()
            self._p.wait(timeout=10)
        except Exception:                          # noqa: BLE001
            pass


class _PyAVWriter:
    name = "pyav-libx264"

    def __init__(self, path: str, w: int, h: int, fps: Fraction, crf: int, faststart: bool):
        import av
        self._av = av
        self.path = path
        opts = {"movflags": "+faststart"} if faststart else {}
        try:
            self._c = av.open(path, "w", options=opts)
            self._s = self._c.add_stream("libx264", rate=fps)
            self._s.width, self._s.height, self._s.pix_fmt = w, h, "yuv420p"
            self._s.options = {"crf": str(int(crf)), "preset": "veryfast", "profile": "high"}
        except Exception as e:                     # noqa: BLE001
            raise WriterError(f"PyAV libx264 unavailable: {e}") from e

    def write(self, bgr: np.ndarray) -> None:
        try:
            frame = self._av.VideoFrame.from_ndarray(bgr, format="bgr24")
            for pkt in self._s.encode(frame):
                self._c.mux(pkt)
        except Exception as e:                     # noqa: BLE001
            raise WriterError(f"PyAV encoding failed: {e}") from e

    def close(self) -> None:
        try:
            for pkt in self._s.encode():
                self._c.mux(pkt)
            self._c.close()
        except Exception as e:                     # noqa: BLE001
            self.abort()
            raise WriterError(f"PyAV could not finish the file: {e}") from e

    def abort(self) -> None:
        try:
            self._c.close()
        except Exception:                          # noqa: BLE001
            pass


class _OpenCVWriter:
    """cv2.VideoWriter; cv2 cannot open non-ASCII paths on Windows, so such
    outputs are written to an ASCII temp name and moved."""

    def __init__(self, path: str, w: int, h: int, fps: Fraction, fourcc: str):
        import cv2
        self._cv2 = cv2
        self.name = "opencv-mp4v" if fourcc == "mp4v" else "opencv-mjpg"
        self.path = path
        self._work = path
        if not path.isascii():
            import tempfile
            fd, self._work = tempfile.mkstemp(suffix=os.path.splitext(path)[1], prefix="pitchspeed-")
            os.close(fd)
            if not self._work.isascii():
                raise WriterError("OpenCV cannot write to a non-ASCII path")
        self._w = cv2.VideoWriter(self._work, cv2.VideoWriter_fourcc(*fourcc), float(fps), (w, h))
        if not self._w.isOpened():
            raise WriterError(f"OpenCV VideoWriter({fourcc}) could not open {self._work}")

    def write(self, bgr: np.ndarray) -> None:
        self._w.write(bgr)

    def close(self) -> None:
        self._w.release()
        if self._work != self.path:
            try:
                os.replace(self._work, self.path)
            except OSError as e:
                raise WriterError(f"cannot move {self._work} to {self.path}: {e}") from e

    def abort(self) -> None:
        self._w.release()
        if self._work != self.path:
            _remove(self._work)


# -------------------------------------------------------------------- writer
class VideoWriter:
    """Write BGR uint8 frames (height, width, 3) to an MP4 (see module doc).

    Args:
        path: output file (.mp4 recommended).
        width, height: frame size; odd sizes are padded by one pixel (yuv420p).
        fps: output rate, e.g. "60000/1001" (one frame per field) or 29.97.
        audio_src: source video whose first audio stream is copied in.
        audio_start_s: source time (seconds from the first video frame, our
            timeline) that plays at output time 0.
        audio_duration_s: audio length; None = length of the written video.
        crf: x264 quality (lower = better/larger).
        backend: 'auto' or one of BACKENDS to start with (later ones remain
            as fallbacks); ffmpeg_exe: override the ffmpeg executable.
    Attributes after close(): backend, out_path, audio_status, frames_written.
    """

    def __init__(self, path, width: int, height: int, fps="60000/1001", audio_src=None,
                 audio_start_s: float = 0.0, audio_duration_s: Optional[float] = None, crf: int = 18,
                 *, backend: str = "auto", ffmpeg_exe: Optional[str] = None):
        self.path = os.fspath(path)
        self.width, self.height = int(width), int(height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"bad frame size {width}x{height}")
        self._enc_w, self._enc_h = self.width + self.width % 2, self.height + self.height % 2
        self.fps = _fraction(fps)
        self.crf = int(crf)
        self.audio_src = os.fspath(audio_src) if audio_src is not None else None
        self.audio_start_s = float(audio_start_s)
        self.audio_duration_s = None if audio_duration_s is None else float(audio_duration_s)
        if self.audio_start_s < 0:
            raise ValueError("audio_start_s must be >= 0")
        self._ffmpeg = ffmpeg_exe if ffmpeg_exe is not None else _default_ffmpeg()
        if backend != "auto" and backend not in BACKENDS:
            raise ValueError(f"backend must be 'auto' or one of {BACKENDS}")
        self._queue = list(BACKENDS[BACKENDS.index(backend):] if backend != "auto" else BACKENDS)
        self.frames_written = 0
        self.backend: Optional[str] = None
        self.out_path = self.path
        self.audio_status = "not requested" if self.audio_src is None else "pending"
        self.failures: list[str] = []
        self._replay: list[np.ndarray] = []
        self._closed = False
        out_dir = os.path.dirname(os.path.abspath(self.path))
        if not os.path.isdir(out_dir):
            raise FileNotFoundError(f"output folder does not exist: {out_dir}")
        self._b = None
        self._start_next()

    # ------------------------------------------------------------ backends
    def _video_target(self, name: str) -> str:
        if name == "opencv-mjpg":
            return os.path.splitext(self.path)[0] + ".avi"
        return _tmp_path(self.path, "video") if self.audio_src is not None else self.path

    def _open(self, name: str):
        target = self._video_target(name)
        faststart = self.audio_src is None
        if name == "ffmpeg-libx264":
            if not self._ffmpeg:
                raise WriterError("no ffmpeg executable")
            return _FFmpegPipe(self._ffmpeg, target, self._enc_w, self._enc_h, self.fps, self.crf, faststart)
        if name == "pyav-libx264":
            return _PyAVWriter(target, self._enc_w, self._enc_h, self.fps, self.crf, faststart)
        return _OpenCVWriter(target, self._enc_w, self._enc_h, self.fps, "mp4v" if name == "opencv-mp4v" else "MJPG")

    def _start_next(self) -> None:
        while self._queue:
            name = self._queue.pop(0)
            try:
                self._b = self._open(name)
            except Exception as e:                 # noqa: BLE001 - try the next backend
                self._note_failure(name, e)
                continue
            if self.backend is not None:
                log.warning("video writer: falling back to %s", name)
            self.backend = name
            return
        raise WriterError(f"no video writer backend works for {self.path}: " + "; ".join(self.failures))

    def _note_failure(self, name: str, e: Exception) -> None:
        msg = f"{name}: {e}"
        self.failures.append(msg)
        log.warning("video writer backend failed - %s", msg)

    def _fail_over(self, e: Exception) -> None:
        """The current backend failed; switch and replay if no frame is lost."""
        self._note_failure(self.backend or "?", e)
        if self._b is not None:
            self._b.abort()
            _remove(self._b.path)
        if self.frames_written > len(self._replay):
            self._closed = True                    # unusable now; frames were lost
            raise WriterError(f"writing {self.path} failed after {self.frames_written} frames: {e}") from e
        self._start_next()
        for f in self._replay:
            try:
                self._b.write(f)
            except WriterError as e2:
                self._fail_over(e2)
                return

    # ---------------------------------------------------------------- API
    def _prepare(self, bgr: np.ndarray) -> np.ndarray:
        a = np.asarray(bgr)
        if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
            raise ValueError(f"expected uint8 BGR (H, W, 3), got {a.dtype} {a.shape}")
        if a.shape[:2] != (self.height, self.width):
            raise ValueError(f"frame size {a.shape[1]}x{a.shape[0]} != writer size {self.width}x{self.height}")
        if (self._enc_w, self._enc_h) != (self.width, self.height):
            a = np.pad(a, ((0, self._enc_h - self.height), (0, self._enc_w - self.width), (0, 0)), mode="edge")
        return np.ascontiguousarray(a)

    def write(self, bgr: np.ndarray) -> None:
        if self._closed:
            raise ValueError("writer is closed")
        a = self._prepare(bgr)
        if len(self._replay) < REPLAY_FRAMES:
            self._replay.append(a.copy())
        self.frames_written += 1
        try:
            self._b.write(a)
        except WriterError as e:
            self._fail_over(e)

    def close(self) -> str:
        """Finish the file; returns the backend used (+ '+audio' when audio was muxed)."""
        if self._closed:
            return self._result()
        self._closed = True
        while True:
            try:
                self._b.close()
                break
            except WriterError as e:
                self._fail_over(e)                 # replays if every frame is still buffered
        video_path = self._b.path
        self._replay = []
        if self.audio_src is None:
            self.out_path = video_path
        elif self.backend == "opencv-mjpg":
            self.out_path = video_path
            self.audio_status = "skipped: no audio in the MJPG/AVI fallback"
            log.warning("annotated video written without audio (%s)", self.audio_status)
        else:
            self._mux_audio(video_path)
        return self._result()

    def _result(self) -> str:
        return (self.backend or "none") + ("+audio" if self.audio_status == "muxed" else "")

    def abort(self) -> None:
        """Stop and delete partial output (used on exceptions)."""
        if self._closed:
            return
        self._closed = True
        if self._b is not None:
            self._b.abort()
            _remove(self._b.path)

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()

    # --------------------------------------------------------------- audio
    def _mux_audio(self, video_path: str) -> None:
        reason = None
        try:
            has_audio, offset = source_audio_offset(self.audio_src)
        except Exception as e:                     # noqa: BLE001
            has_audio, offset, reason = False, 0.0, f"cannot read {self.audio_src}: {e}"
        if reason is None and not has_audio:
            reason = "source has no audio"
        if reason is None and not self._ffmpeg:
            reason = "no ffmpeg executable for muxing"
        if reason is None:
            dur = self.audio_duration_s if self.audio_duration_s is not None else self.frames_written / float(self.fps)
            shift = offset + self.audio_start_s
            cmd = [self._ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
                   "-i", video_path, "-ss", f"{shift:.6f}", "-t", f"{dur:.6f}", "-i", self.audio_src,
                   "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                   "-af", "aresample=async=1:first_pts=0", "-c:a", "aac", "-b:a", "160k",
                   "-movflags", "+faststart", self.path]
            try:
                p = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_CLOSE_TIMEOUT_S,
                                   creationflags=_NO_WINDOW)
                if p.returncode == 0 and os.path.isfile(self.path) and os.path.getsize(self.path) > 0:
                    _remove(video_path)
                    self.out_path = self.path
                    self.audio_status = "muxed"
                    return
                reason = "ffmpeg mux failed: " + p.stderr.decode("utf-8", "replace").strip()[-800:]
            except (OSError, subprocess.SubprocessError) as e:
                reason = f"ffmpeg mux failed: {e}"
        self.audio_status = f"skipped: {reason}"
        log.warning("annotated video written without audio (%s)", reason)
        if os.path.abspath(self.path) != os.path.abspath(video_path):
            os.replace(video_path, self.path)      # keep the video, without audio
        self.out_path = self.path
