"""Synthetic interlaced / progressive test clips with a known ball position.

Used by tests/test_io_*.py.  Clips are rendered field by field (59.94 Hz)
with numpy and encoded with the ffmpeg bundled with imageio-ffmpeg, so the
tests are self-contained.  Ground truth: the ball centre moves along row
BALL_Y with x(t) = x0 + v_px_per_field * (t - t0) / FIELD_DT; the rendered
(motion-blurred) centroid of a field exposed from t to t + shutter is
x(t + shutter / 2).
"""
from __future__ import annotations

import os
import subprocess
import wave
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

W, H = 720, 480
FPS = "30000/1001"
FRAME_DT = 1001.0 / 30000.0
FIELD_DT = FRAME_DT / 2.0
BALL_Y = 241.0
BALL_R = 4.0


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


@dataclass
class Scene:
    """A static textured background, a slowly moving 'pitcher' block and a ball."""
    seed: int = 1
    noise: float = 3.0
    block: bool = True              # large slow object (drives the comb test)
    ball_x0: float = 60.0
    ball_t0: float = 0.25           # s; ball moves from this time on
    ball_v: float = 18.0            # px per field
    ball_level: float = 235.0
    width: int = W
    height: int = H
    bg: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(self.seed)
        g = rng.normal(0, 1, (self.height // 8 + 2, self.width // 8 + 2))
        g = np.kron(g, np.ones((8, 8)))[: self.height, : self.width]
        g = gaussian_filter(g, 4)
        self.bg = (100 + 25 * g / g.std()).astype(np.float32)
        self._rng = np.random.default_rng(self.seed + 1000)

    def ball_x(self, t: float) -> Optional[float]:
        if t < self.ball_t0:
            return None
        x = self.ball_x0 + self.ball_v * (t - self.ball_t0) / FIELD_DT
        return x if x < self.width - 40 else None

    def render(self, t: float, shutter: float = FIELD_DT) -> np.ndarray:
        img = self.bg.copy()
        if self.block:
            bx = int(200 + 120 * np.sin(2 * np.pi * 0.4 * t))
            img[300:460, bx:bx + 60] = 170.0      # below the ball's rows
        xs = [self.ball_x(t + shutter * (j + 0.5) / 6) for j in range(6)]
        xs = [x for x in xs if x is not None]
        if xs:
            x_lo, x_hi = int(min(xs) - 8), int(max(xs) + 9)
            y_lo, y_hi = int(BALL_Y - 8), int(BALL_Y + 9)
            yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi].astype(np.float32)
            acc = np.zeros_like(yy)
            for x in xs:
                acc += np.clip(BALL_R - np.hypot(xx - x, yy - BALL_Y) + 0.5, 0, 1)
            a = acc / 6.0
            patch = img[y_lo:y_hi, x_lo:x_hi]
            img[y_lo:y_hi, x_lo:x_hi] = patch * (1 - a) + self.ball_level * a
        return img

    def finish(self, img: np.ndarray) -> np.ndarray:
        if self.noise > 0:
            img = img + self._rng.normal(0, self.noise, img.shape).astype(np.float32)
        return np.clip(img, 0, 255).astype(np.uint8)


def weave_frames(scene: Scene, n_frames: int, order: str) -> list[np.ndarray]:
    """Frames for order 'tff' | 'bff' | 'prog' (progressive, 1/30 s shutter)."""
    out = []
    for i in range(n_frames):
        t0 = i * FRAME_DT
        if order == "prog":
            f = scene.render(t0, FRAME_DT)
        else:
            first, second = scene.render(t0), scene.render(t0 + FIELD_DT)
            f = np.empty_like(first)
            p = 0 if order == "tff" else 1
            f[p::2], f[1 - p::2] = first[p::2], second[1 - p::2]
        out.append(scene.finish(f))
    return out


def write_beep_wav(path: str, duration_s: float, beep_frames: Sequence[int], sr: int = 48000) -> list[float]:
    """Mono WAV with 20 ms 2 kHz beeps starting at the given video frames' times."""
    a = np.zeros(int(duration_s * sr))
    starts = []
    for k in beep_frames:
        s = int(round(k * FRAME_DT * sr))
        n = int(0.02 * sr)
        a[s:s + n] = 0.5 * np.sin(2 * np.pi * 2000 * np.arange(n) / sr)
        starts.append(k * FRAME_DT)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((a * 32767).astype("<i2").tobytes())
    return starts


# codec presets: (extension, video args, audio args)
PRESETS = {
    "dv_avi": (".avi", ["-c:v", "dvvideo", "-pix_fmt", "yuv411p"], ["-c:a", "pcm_s16le"]),
    "dv_raw": (".dv", ["-c:v", "dvvideo", "-pix_fmt", "yuv411p", "-f", "dv"], None),
    "mpeg2_ps": (".mpg", ["-c:v", "mpeg2video", "-b:v", "6M", "-pix_fmt", "yuv420p",
                          "-flags", "+ildct+ilme", "-f", "dvd"], ["-c:a", "mp2", "-b:a", "192k"]),
    "h264_mp4": (".mp4", ["-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-flags", "+ildct+ilme"],
                 ["-c:a", "aac"]),
    "h264_prog": (".mp4", ["-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p"], ["-c:a", "aac"]),
    "wmv": (".wmv", ["-c:v", "wmv2", "-b:v", "4M"], ["-c:a", "wmav2"]),
    "ffv1_mkv": (".mkv", ["-c:v", "ffv1", "-pix_fmt", "yuv420p"], ["-c:a", "pcm_s16le"]),
}


def encode(frames: Sequence[np.ndarray], out_path: str, preset: str, order: Optional[str] = None,
           rate: str = FPS, audio_wav: Optional[str] = None, extra: Sequence[str] = (),
           vf: Optional[str] = None) -> str:
    """Encode gray frames with a preset; returns the output path."""
    _, vargs, aargs = PRESETS[preset]
    vargs = list(vargs)
    if order in ("tff", "bff"):
        if preset == "mpeg2_ps":
            vargs += ["-top", "1" if order == "tff" else "0"]
        elif preset == "h264_mp4":
            vargs += ["-x264-params", f"{order}=1"]
    h, w = frames[0].shape
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray",
           "-s", f"{w}x{h}", "-r", rate, "-i", "-"]
    if audio_wav and aargs is not None:
        cmd += ["-i", audio_wav]
    if vf:
        cmd += ["-vf", vf]
    cmd += vargs + list(extra)
    if audio_wav and aargs is not None:
        cmd += aargs
    cmd += [out_path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for f in frames:
            p.stdin.write(np.ascontiguousarray(f).tobytes())
        p.stdin.close()
    except BrokenPipeError:
        pass
    err = p.stderr.read().decode("utf-8", "replace")
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}: {err}")
    return out_path


def make_clip(dir_path: str, name: str, preset: str, order: str, n_frames: int = 60,
              scene: Optional[Scene] = None, audio_beeps: Optional[Sequence[int]] = None, **kw) -> dict:
    """Render + encode; returns {'path', 'frames', 'scene', 'order', 'beeps'}."""
    scene = scene or Scene()
    frames = weave_frames(scene, n_frames, order)
    ext = PRESETS[preset][0]
    path = os.path.join(dir_path, name + ext)
    wav, beeps = None, None
    if audio_beeps is not None:
        wav = os.path.join(dir_path, name + "_beep.wav")
        beeps = write_beep_wav(wav, n_frames * FRAME_DT, audio_beeps)
    encode(frames, path, preset, order=order, audio_wav=wav, **kw)
    return {"path": path, "frames": frames, "scene": scene, "order": order, "beeps": beeps}


def ball_centroid(img: np.ndarray, row_coords: np.ndarray, background: np.ndarray,
                  x_guess: Optional[float] = None, thresh: float = 15.0) -> Optional[tuple[float, float]]:
    """Ball centroid (x, frame row) in an image whose rows sit at frame rows `row_coords`.

    `background` is the scene background at the same rows (Scene.bg[row_coords]);
    weights are the brightness above it minus `thresh`, near row BALL_Y.
    """
    band = np.abs(row_coords - BALL_Y) <= 7
    d = img[band].astype(np.float32) - background[band].astype(np.float32)
    d = np.clip(d - thresh, 0, None)
    if x_guess is not None:
        lo, hi = int(max(0, x_guess - 40)), int(min(img.shape[1], x_guess + 40))
        d[:, :lo] = 0
        d[:, hi:] = 0
    if d.sum() <= 0:
        return None
    ys, xs = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    return float((xs * d).sum() / d.sum()), float((row_coords[band][ys] * d).sum() / d.sum())


def beep_time(path: str) -> float:
    """Time (s, stream timeline) of the first beep onset in a file's first audio stream."""
    import av
    with av.open(path) as c:
        a = c.streams.audio[0]
        t0, buf = None, []
        for fr in c.decode(a):
            x = fr.to_ndarray().astype(np.float64)
            x = x.mean(axis=0) if x.ndim == 2 and x.shape[0] < 8 else x.ravel()
            if t0 is None:
                t0 = float(fr.pts * fr.time_base)
            buf.append(x)
        x = np.concatenate(buf)
        idx = int(np.argmax(np.abs(x) > 0.25 * np.abs(x).max()))
        return t0 + idx / a.rate


def video_frame_time(path: str, k: int) -> float:
    import av
    with av.open(path) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i == k:
                return float(fr.pts * fr.time_base)
    raise IndexError(k)

# --------------------------------------------------------------- clip catalog
N_FRAMES = 60


def _hard_scene(seed: int) -> Scene:
    """Static camera, nothing but a lone ball moving, heavy noise (io_ux.md 0.6)."""
    return Scene(seed=seed, noise=7.0, block=False, ball_v=10.0)


def _build(d: str, name: str) -> dict:
    """Create catalog clip `name` in directory d."""
    simple = {
        "dv_avi_bff": ("dv_avi", "bff"), "dv_raw_bff": ("dv_raw", "bff"), "mpeg2_tff": ("mpeg2_ps", "tff"),
        "h264_tff": ("h264_mp4", "tff"), "h264_bff_untagged": ("h264_prog", "bff"),
        "h264_prog": ("h264_prog", "prog"), "wmv_tff": ("wmv", "tff"), "ffv1_bff": ("ffv1_mkv", "bff"),
        "ffv1_tff": ("ffv1_mkv", "tff"),
    }
    if name in simple:
        preset, order = simple[name]
        return make_clip(d, name, preset, order, N_FRAMES)
    if name == "mpeg2_tff_audio":
        return make_clip(d, name, "mpeg2_ps", "tff", 90, audio_beeps=[30])
    if name == "dv_bff_audio":
        return make_clip(d, name, "dv_avi", "bff", 90, audio_beeps=[30])
    if name == "hard_tff_mpeg2":
        return make_clip(d, name, "mpeg2_ps", "tff", N_FRAMES, scene=_hard_scene(7), extra=["-b:v", "3M"])
    if name == "hard_bff_dv":
        return make_clip(d, name, "dv_avi", "bff", N_FRAMES, scene=_hard_scene(8))
    scene = Scene(seed=3)
    if name in ("dropped", "vfr", "broken_pts", "dup"):
        frames = weave_frames(scene, N_FRAMES, "bff")
        vf = {"dropped": "select='not(eq(n\\,30))'",
              "vfr": "settb=1/1000,setpts='(N+0.4*mod(N\\,2))*1001/30'",
              "broken_pts": "setpts='(N+0.4*mod(N\\,2))/(30000/1001)/TB'"}.get(name)
        if name == "dup":                       # frame 41 repeats frame 40 (capture duplicate)
            frames = frames[:41] + [frames[40]] + frames[42:]
            path = encode(frames, os.path.join(d, name + ".avi"), "dv_avi")
        else:
            path = encode(frames, os.path.join(d, name + ".mkv"), "ffv1_mkv", vf=vf,
                          extra=["-fps_mode", "passthrough"])
        return {"path": path, "frames": frames, "scene": scene, "order": "bff"}
    if name == "woven60_tff":                   # TFF-woven content labelled 59.94 frames/s
        frames = weave_frames(scene, N_FRAMES, "tff")
        path = encode(frames, os.path.join(d, name + ".mkv"), "ffv1_mkv", rate="60000/1001")
        return {"path": path, "frames": frames, "scene": scene, "order": "tff"}
    prog = weave_frames(scene, N_FRAMES + 1, "prog")
    if name in ("fake60_p0", "fake60_p1"):
        dbl = [f for f in prog for _ in (0, 1)]
        if name == "fake60_p1":
            dbl = dbl[1:]
        path = encode(dbl, os.path.join(d, name + ".mp4"), "h264_prog", rate="60000/1001")
        return {"path": path, "frames": dbl, "scene": scene, "order": "prog"}
    if name == "shifted":                       # frame k = top of P_k + bottom of P_(k-1)
        frames = []
        for k in range(1, N_FRAMES + 1):
            f = prog[k].copy()
            f[1::2] = prog[k - 1][1::2]
            frames.append(f)
        path = encode(frames, os.path.join(d, name + ".mkv"), "ffv1_mkv")
        return {"path": path, "frames": frames, "scene": scene, "order": "shifted", "progressive": prog[1:]}
    raise KeyError(name)


class ClipFactory:
    """Builds catalog clips on demand into one directory and caches them."""

    def __init__(self, directory: str):
        self.dir = directory
        self._cache: dict = {}

    def get(self, name: str) -> dict:
        if name not in self._cache:
            self._cache[name] = _build(self.dir, name)
        return self._cache[name]

    def path(self, name: str) -> str:
        return self.get(name)["path"]


_FACTORY: Optional[ClipFactory] = None


def clip_factory(tmp_path_factory) -> ClipFactory:
    """Process-wide factory (shared by all io test modules)."""
    global _FACTORY
    if _FACTORY is None:
        _FACTORY = ClipFactory(str(tmp_path_factory.mktemp("io_clips")))
    return _FACTORY
