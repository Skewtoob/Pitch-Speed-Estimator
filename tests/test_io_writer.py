"""Tests for pitchspeed.io.writer: backends, fallbacks, audio passthrough and sync."""
from __future__ import annotations

import os
from fractions import Fraction
import stat
import sys

import av
import numpy as np
import pytest

import io_helpers as H
from pitchspeed.io.reader import VideoReader
from pitchspeed.io.writer import BACKENDS, VideoWriter, WriterError, source_audio_offset


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    return H.clip_factory(tmp_path_factory)


def _frames(n=24, w=320, h=240):
    out = []
    for i in range(n):
        f = np.zeros((h, w, 3), np.uint8)
        f[:, :, 1] = 60
        f[h // 3:h // 3 + 20, (i * 9) % (w - 20):(i * 9) % (w - 20) + 20] = (255, 255, 255)
        out.append(f)
    return out


def _probe(path):
    with av.open(path) as c:
        v = c.streams.video[0]
        n = sum(1 for _ in c.decode(v))
        return {"n": n, "codec": v.codec_context.name, "pix_fmt": v.codec_context.pix_fmt,
                "audio": len(c.streams.audio), "w": v.codec_context.width, "h": v.codec_context.height}


@pytest.mark.parametrize("backend", BACKENDS)
def test_each_backend_writes_decodable_video(tmp_path, backend):
    frames = _frames()
    w = VideoWriter(str(tmp_path / "out.mp4"), 320, 240, fps="60000/1001", backend=backend)
    for f in frames:
        w.write(f)
    used = w.close()
    assert used == backend and os.path.isfile(w.out_path)
    info = _probe(w.out_path)
    assert info["n"] == len(frames) and (info["w"], info["h"]) == (320, 240)
    if backend in ("ffmpeg-libx264", "pyav-libx264"):
        assert info["codec"] == "h264" and info["pix_fmt"] == "yuv420p"
    if backend == "opencv-mjpg":
        assert w.out_path.endswith(".avi")
    assert not [p for p in os.listdir(tmp_path) if "-tmp" in p]


def test_ffmpeg_output_rate_and_profile(tmp_path):
    p = str(tmp_path / "rate.mp4")
    with VideoWriter(p, 320, 240) as w:
        for f in _frames(30):
            w.write(f)
    with av.open(p) as c:
        v = c.streams.video[0]
        assert Fraction(v.average_rate) == Fraction(60000, 1001)
        assert "High" in (v.codec_context.profile or "")


def test_odd_size_is_padded(tmp_path):
    w = VideoWriter(str(tmp_path / "odd.mp4"), 321, 241)
    for f in _frames(5, 321, 241):
        w.write(f)
    w.close()
    info = _probe(w.out_path)
    assert info["n"] == 5 and (info["w"], info["h"]) == (322, 242)
    with pytest.raises(ValueError):
        w2 = VideoWriter(str(tmp_path / "bad.mp4"), 320, 240)
        w2.write(np.zeros((240, 321, 3), np.uint8))


def test_missing_ffmpeg_falls_back_to_pyav(tmp_path):
    w = VideoWriter(str(tmp_path / "fb.mp4"), 320, 240, ffmpeg_exe=os.path.join("no", "such", "ffmpeg"))
    for f in _frames(10):
        w.write(f)
    assert w.close() == "pyav-libx264"
    assert w.failures and _probe(w.out_path)["n"] == 10


def _fake_ffmpeg(tmp_path, body: str) -> str:
    """A shell script posing as ffmpeg that lists libx264 but then fails."""
    p = tmp_path / "fake_ffmpeg.sh"
    p.write_text("#!/bin/sh\nif [ \"$2\" = \"-encoders\" ]; then echo ' V....D libx264  H.264'; exit 0; fi\n"
                 + body + "\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


@pytest.mark.skipif(sys.platform == "win32", reason="uses a POSIX shell script as a fake ffmpeg")
def test_broken_pipe_at_start_falls_back_and_replays(tmp_path):
    fake = _fake_ffmpeg(tmp_path, "echo 'Unknown encoder libx264 (fake)' >&2; exit 1")
    w = VideoWriter(str(tmp_path / "bp.mp4"), 320, 240, ffmpeg_exe=fake)
    for f in _frames(20):
        w.write(f)
    assert w.close() == "pyav-libx264"
    assert any("fake" in m for m in w.failures)          # ffmpeg's own message is kept
    assert _probe(w.out_path)["n"] == 20                 # buffered frames were replayed


@pytest.mark.skipif(sys.platform == "win32", reason="uses a POSIX shell script as a fake ffmpeg")
def test_failure_after_start_raises_with_message(tmp_path):
    # accepts ~20 frames (230 KB each), then dies
    fake = _fake_ffmpeg(tmp_path, "head -c 4600000 > /dev/null; echo 'No space left on device (fake)' >&2; exit 1")
    w = VideoWriter(str(tmp_path / "late.mp4"), 320, 240, ffmpeg_exe=fake)
    with pytest.raises(WriterError, match="No space left"):
        for f in _frames(200):
            w.write(f)
        w.close()


def test_context_manager_abort_removes_partial_output(tmp_path):
    p = tmp_path / "abort.mp4"
    with pytest.raises(RuntimeError):
        with VideoWriter(str(p), 320, 240) as w:
            for f in _frames(3):
                w.write(f)
            raise RuntimeError("render failed")
    assert not p.exists()


def test_non_ascii_output_path(tmp_path):
    d = tmp_path / "Ergebnisse 日本"
    d.mkdir()
    for backend in ("ffmpeg-libx264", "pyav-libx264", "opencv-mp4v"):
        p = d / f"vidéo {backend}.mp4"
        w = VideoWriter(str(p), 320, 240, backend=backend)
        for f in _frames(4):
            w.write(f)
        assert w.close() == backend
        assert _probe(str(p))["n"] == 4


# ------------------------------------------------------------------ audio
@pytest.mark.parametrize("name", ["mpeg2_tff_audio", "dv_bff_audio"])
@pytest.mark.parametrize("start_frame", [0, 20])
def test_audio_passthrough_is_in_sync(clips, tmp_path, name, start_frame):
    """Beep at source frame 30 must coincide with output frame 30 - start."""
    src = clips.path(name)
    out = str(tmp_path / f"a_{start_frame}.mp4")
    with VideoReader(src) as r:
        w = VideoWriter(out, 720, 480, fps="30000/1001", audio_src=src,
                        audio_start_s=float(r.frame_times[start_frame]))
        for _, g in r.iter_frames(start_frame, None, "bgr"):
            w.write(g)
        assert w.close() == "ffmpeg-libx264+audio" and w.audio_status == "muxed"
        src_err = H.beep_time(src) - r.video_start_s - r.frame_times[30]
    out_err = H.beep_time(out) - H.video_frame_time(out, 30 - start_frame)
    assert abs(src_err) < 1e-3
    assert abs(out_err - src_err) < 2e-3, (out_err, src_err)
    with av.open(out) as c:
        a, v = c.streams.audio[0], c.streams.video[0]
        vdur, adur = float(v.duration * v.time_base), float(a.duration * a.time_base)
        assert a.codec_context.name == "aac"
        assert abs(adur - vdur) < 0.05                          # bounded to the video length, no -shortest


def test_mpeg_ps_offset_is_nontrivial(clips):
    has_audio, off = source_audio_offset(clips.path("mpeg2_tff_audio"))
    assert has_audio and 0.005 < off < 0.02                    # video starts ~10 ms after the container


def test_audio_excerpt_duration(clips, tmp_path):
    src = clips.path("dv_bff_audio")
    out = str(tmp_path / "excerpt.mp4")
    w = VideoWriter(out, 320, 240, fps="60000/1001", audio_src=src, audio_start_s=0.5, audio_duration_s=1.25)
    for f in _frames(30):
        w.write(f)
    w.close()
    with av.open(out) as c:
        a = c.streams.audio[0]
        assert abs(float(a.duration * a.time_base) - 1.25) < 0.05


def test_source_without_audio_keeps_video(clips, tmp_path):
    out = str(tmp_path / "noaudio.mp4")
    w = VideoWriter(out, 320, 240, audio_src=clips.path("dv_avi_bff"))
    for f in _frames(6):
        w.write(f)
    assert w.close() == "ffmpeg-libx264"
    assert w.audio_status.startswith("skipped") and "no audio" in w.audio_status
    info = _probe(out)
    assert info["n"] == 6 and info["audio"] == 0
    assert not [p for p in os.listdir(tmp_path) if "-tmp" in p]


def test_field_rate_render_pipeline_with_audio(clips, tmp_path):
    """reader -> detect_scan -> SampleSource -> bob display -> 59.94p MP4 with audio:
    one output frame per field, beep (source frame 30) at output frame 60."""
    from pitchspeed.io.display import sample_to_display
    from pitchspeed.io.interlace import SampleSource, detect_scan
    src_path = clips.path("dv_bff_audio")
    out = str(tmp_path / "annotated.mp4")
    with VideoReader(src_path) as r:
        scan = detect_scan(r, use_idet=False)
        src = SampleSource(r, scan)
        assert src.interlaced and scan.field_order == "bff"
        with VideoWriter(out, 960, 720, fps="60000/1001", audio_src=src_path) as w:
            for smp in src.iter_samples(color=True):
                bgr, _ = sample_to_display(smp, r.info.sar, (960, 720), is_field=True)
                w.write(bgr)
        assert w.backend == "ffmpeg-libx264" and w.audio_status == "muxed"
        n_frames = r.n_frames
    assert _probe(out)["n"] == 2 * n_frames
    err = H.beep_time(out) - H.video_frame_time(out, 60)
    assert abs(err) < 2e-3, err
