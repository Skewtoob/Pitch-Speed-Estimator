"""End-to-end rendering through the real io stack (VideoReader, SampleSource,
display, VideoWriter + ffmpeg) on a generated interlaced MPEG-2 PS clip.

Checks: one 59.94p output frame per field, H.264 960x720, audio present and
in sync to < 2 ms (full mode: passthrough by VideoWriter; highlights: the
per-segment audio pass), the speed card visible after the catch, a
non-ASCII output folder, and progressive (frame-mode) output at 29.97."""
from __future__ import annotations

import os

import numpy as np
import pytest

import render_helpers as H

av = pytest.importorskip("av")
pytest.importorskip("imageio_ffmpeg")

from pitchspeed.io.interlace import SampleSource            # noqa: E402
from pitchspeed.io.reader import VideoReader                # noqa: E402
from pitchspeed.render.overlay import render_annotated     # noqa: E402
from pitchspeed.types import ScanInfo                       # noqa: E402

FPS = 60000 / 1001
SYNC_TOL_S = 0.002


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("render_clip")
    path = H.make_test_clip(str(d), seconds=9.0)
    return path, H.audio_onsets(path)


def _result():
    r = H.rich_result()
    fd = H.FIELD_DT
    r.pitches = [H.pitch(1, int(2.5 / fd), int(3.0 / fd), H.speed(78.4, 1.6, "good")),
                 H.pitch(2, int(8.2 / fd), int(8.6 / fd), H.speed(66.0, 5.0, "poor"))]
    r.rejected_tracks, r.possible_missed = [], []
    return r


def _source(path, mode="interlaced"):
    reader = VideoReader(path)
    scan = (ScanInfo(mode="interlaced", field_order="tff", confidence="high", sample_rate=FPS)
            if mode == "interlaced" else ScanInfo(mode="progressive", field_order=None, sample_rate=FPS / 2))
    return reader, SampleSource(reader, scan)


def _probe(path):
    with av.open(path) as c:
        v = c.streams.video[0]
        n = sum(1 for _ in c.decode(video=0))
        return {"rate": v.average_rate, "w": v.codec_context.width, "h": v.codec_context.height,
                "frames": n, "codec": v.codec_context.name, "audio": len(c.streams.audio)}


def _frame(path, i):
    with av.open(path) as c:
        for j, f in enumerate(c.decode(video=0)):
            if j == i:
                return f.to_ndarray(format="bgr24")
    raise IndexError(i)


def _expected_onsets(src_onsets, segments):
    out, off = [], 0.0
    for d in segments:
        dur = d["frames"] / FPS
        out += [off + t - d["t_start"] for t in src_onsets
                if d["t_start"] <= t and t + H.BEEP_LEN_S < d["t_start"] + dur]
        off += dur
    return out


def _panel_brightness(img) -> float:
    """Mean brightness where the card's dark panel sits (static test-pattern bars there)."""
    return float(img[612:688, 340:620].mean())


def test_full_render_real_io(clip, tmp_path):
    path, src_onsets = clip
    reader, src = _source(path)
    info = {}
    try:
        out = render_annotated(_result(), src, str(tmp_path / "annotated.mp4"), mode="full", info=info)
    finally:
        reader.close()
    assert out == str(tmp_path / "annotated.mp4") and os.path.isfile(out)
    pr = _probe(out)
    assert pr["codec"] == "h264" and (pr["w"], pr["h"]) == (960, 720)
    assert float(pr["rate"]) == pytest.approx(FPS) and pr["frames"] == src.n_samples == info["frames_written"]
    assert pr["audio"] == 1 and info["audio_status"] == "muxed" and info["filler_frames"] == 0
    got = H.audio_onsets(out)
    assert len(got) == len(src_onsets)
    np.testing.assert_allclose(got, src_onsets, atol=SYNC_TOL_S)
    s_catch = int(3.0 / H.FIELD_DT)
    before, during = _panel_brightness(_frame(out, s_catch - 5)), _panel_brightness(_frame(out, s_catch + 30))
    assert during < 0.7 * before                                     # card panel shown after the catch


def test_highlights_real_io_per_segment_audio(clip, tmp_path):
    path, src_onsets = clip
    reader, src = _source(path)
    folder = tmp_path / "Ausgabe été"
    folder.mkdir()
    info = {}
    try:
        out = render_annotated(_result(), src, str(folder / "annotated.mp4"), mode="highlights", info=info)
    finally:
        reader.close()
    assert sorted(os.listdir(folder)) == ["annotated.mp4"]            # temp files cleaned up
    segs = info["segments"]
    assert len(segs) == 2 and info["audio_status"] == "muxed"
    pr = _probe(out)
    assert pr["frames"] == sum(d["frames"] for d in segs) and pr["audio"] == 1
    exp = _expected_onsets(src_onsets, segs)
    got = H.audio_onsets(out)
    assert len(exp) >= 6 and len(got) == len(exp)
    np.testing.assert_allclose(got, exp, atol=SYNC_TOL_S)


def test_progressive_frame_mode_real_io(clip, tmp_path):
    path, _ = clip
    reader, src = _source(path, "progressive")
    info = {}
    try:
        out = render_annotated(_result(), src, str(tmp_path / "p.mp4"), mode="highlights", audio=False,
                               info=info)
    finally:
        reader.close()
    pr = _probe(out)
    assert float(pr["rate"]) == pytest.approx(FPS / 2) and pr["audio"] == 0
    assert pr["frames"] == info["frames_written"] and info["audio_status"].startswith("off")


def _ffmpeg(*args):
    import subprocess

    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args],
                   check=True, capture_output=True, timeout=120)


def test_mux_many_segments_uses_script_file(clip, tmp_path):
    from pitchspeed.render.overlay import audio_filter_graph, mux_segment_audio
    path, _ = clip
    video = str(tmp_path / "v.mp4")
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x240:rate=60000/1001", "-t", "6", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", video)
    segs = [(0.2 + 0.1 * k, 0.1) for k in range(60)]
    assert len(audio_filter_graph(segs, 0.0, 0.0)) > 8000          # too long for the command line
    out = str(tmp_path / "o.mp4")
    assert mux_segment_audio(video, path, segs, out) == "muxed"
    with av.open(out) as c:
        n = sum(f.samples for f in c.decode(audio=0))
    assert n / 48000 == pytest.approx(6.0, abs=0.025)
    assert sorted(os.listdir(tmp_path)) == ["o.mp4", "v.mp4"]


def test_highlights_source_without_audio_keeps_video(tmp_path):
    src_path = str(tmp_path / "silent.mkv")
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=720x480:rate=60000/1001", "-t", "7",
            "-vf", "interlace=scan=tff:lowpass=0", "-c:v", "mpeg2video", "-flags", "+ilme+ildct", "-top", "1",
            "-q:v", "5", "-an", src_path)
    reader, src = _source(src_path)
    r = _result()
    r.pitches = r.pitches[:1]
    info = {}
    try:
        out = render_annotated(r, src, str(tmp_path / "h.mp4"), mode="highlights", info=info)
    finally:
        reader.close()
    assert info["audio_status"] == "skipped: source has no audio"
    assert _probe(out)["frames"] == info["frames_written"] > 0
    assert sorted(os.listdir(tmp_path)) == ["h.mp4", "silent.mkv"]          # no temp files left
