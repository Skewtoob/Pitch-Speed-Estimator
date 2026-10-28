"""Tests for pitchspeed.io.reader: frame counts, timing, random access, SAR, paths."""
from __future__ import annotations

import os
import shutil
import subprocess
from fractions import Fraction

import av
import numpy as np
import pytest

import io_helpers as H
from pitchspeed.io.reader import (FrameDecodeError, VideoReader, analyze_timing, default_sar, probe,
                                  resolve_sar)

DT = 1001.0 / 30000.0

CONTAINERS = ["dv_avi_bff", "dv_raw_bff", "mpeg2_tff", "h264_tff", "h264_prog", "wmv_tff", "ffv1_bff"]


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    return H.clip_factory(tmp_path_factory)


def _decode_all(path):
    with av.open(path) as c:
        return [f.to_ndarray(format="gray") for f in c.decode(video=0)]


# ------------------------------------------------------------ counts/timing
@pytest.mark.parametrize("name", CONTAINERS)
def test_frame_count_and_cfr_timing(clips, name):
    clip = clips.get(name)
    with VideoReader(clip["path"]) as r:
        i = r.info
        assert i.n_frames == H.N_FRAMES == r.n_frames == len(r.frame_times)
        assert i.frame_rate_str == "30000/1001"          # raw .dv: avg_frame_rate is a bogus 60000/1
        assert i.timing_mode == "cfr"
        assert i.dropped_frames == []
        assert r.frame_times[0] == 0.0
        np.testing.assert_allclose(r.frame_times, np.arange(H.N_FRAMES) * DT, atol=1e-12)  # WMV/MKV ms jitter snapped
        assert i.width == 720 and i.height == 480
        assert abs(i.duration_s - H.N_FRAMES * DT) < 1e-9
        frames = dict(r.iter_frames())
    assert sorted(frames) == list(range(H.N_FRAMES))
    # decoded content matches what was encoded (lossy codecs: small error)
    err = np.mean([np.abs(frames[k].astype(float) - clip["frames"][k]).mean() for k in (0, 30, 59)])
    assert err < 6.0, err
    assert probe(clip["path"]).width == 720


def test_probe_fast_metadata(clips):
    info = probe(clips.path("dv_avi_bff"))
    assert info.n_frames == H.N_FRAMES and info.codec == "dvvideo" and info.has_audio is False
    info = probe(clips.path("dv_raw_bff"))
    assert info.n_frames == H.N_FRAMES and info.frame_rate_str == "30000/1001"


@pytest.mark.parametrize("name", CONTAINERS)
def test_random_access_is_frame_exact(clips, name):
    path = clips.path(name)
    seq = _decode_all(path)
    with VideoReader(path, cache_size=0) as r:
        rng = np.random.default_rng(3)
        idx = [int(k) for k in rng.integers(0, r.n_frames, 10)] + [0, 1, r.n_frames - 1, 5, 4, 40, 41]
        for k in idx:
            assert np.array_equal(r.get_frame(k), seq[k]), f"frame {k}"
        sub = list(r.iter_frames(33, 45))
        assert [j for j, _ in sub] == list(range(33, 45))
        assert all(np.array_equal(g, seq[j]) for j, g in sub)
        assert r.get_frame(-1).shape == (480, 720)
        bgr = r.get_frame(10, "bgr")
        assert bgr.shape == (480, 720, 3) and bgr.dtype == np.uint8
        g, b = r.get_frame(11, "both")
        assert np.array_equal(g, seq[11]) and b.shape == (480, 720, 3)
        with pytest.raises(IndexError):
            r.get_frame(r.n_frames)
        assert r.decode_errors == []


def test_frame_cache_returns_copies(clips):
    with VideoReader(clips.path("dv_avi_bff")) as r:
        a = r.get_frame(3)
        a[:] = 0
        assert r.get_frame(3).any()


def test_dropped_frame_detected(clips):
    with VideoReader(clips.path("dropped")) as r:
        assert r.n_frames == H.N_FRAMES - 1
        assert r.info.timing_mode == "cfr"
        assert r.info.dropped_frames == [30]
        assert abs(r.frame_times[30] - 31 * DT) < 1e-9 and abs(r.frame_times[29] - 29 * DT) < 1e-9


def test_variable_frame_rate_detected(clips):
    with VideoReader(clips.path("vfr")) as r:
        assert r.info.timing_mode == "vfr"
        assert r.info.frame_rate_str == "30000/1001"         # nominal, from the trimmed mean step
        expected = (np.arange(H.N_FRAMES) + 0.4 * (np.arange(H.N_FRAMES) % 2)) * DT
        np.testing.assert_allclose(r.frame_times, expected, atol=1.1e-3)   # MKV: 1 ms time base
        assert r.info.dropped_frames == []


def test_broken_timestamps_are_reported_not_trusted(clips):
    with VideoReader(clips.path("broken_pts")) as r:
        assert r.n_frames == H.N_FRAMES
        assert r.info.timing_mode == "vfr"
        assert any("not strictly increasing" in n for n in r.notes)
        np.testing.assert_allclose(r.frame_times, np.arange(H.N_FRAMES) * DT)
        seq = _decode_all(clips.path("broken_pts"))
        assert np.array_equal(r.get_frame(37), seq[37])           # counted access still exact


def test_edit_list_mp4_counts_only_presented_frames(clips, tmp_path):
    src = clips.path("h264_prog")
    cut = str(tmp_path / "cut.mp4")
    subprocess.run([H.ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", "0.7", "-i", src, "-c", "copy", cut],
                   check=True)
    seq = _decode_all(cut)
    with VideoReader(cut) as r:
        assert r.n_frames == len(seq)
        assert r.index_method == "demux"                          # discard packets skipped, no full decode
        assert np.array_equal(r.get_frame(len(seq) // 2), seq[len(seq) // 2])
    with VideoReader(cut, index_mode="decode") as rd:
        assert rd.n_frames == len(seq)


def test_non_ascii_path(clips, tmp_path):
    d = tmp_path / "Ünïcødé 日本 vidéo"
    d.mkdir()
    p = d / "clip ä ö 野球.avi"
    shutil.copy(clips.path("dv_avi_bff"), p)
    with VideoReader(str(p)) as r:
        assert r.n_frames == H.N_FRAMES
        assert r.get_frame(20).shape == (480, 720)
        assert len(list(r.iter_frames(50))) == 10
    assert probe(p).n_frames == H.N_FRAMES                        # os.PathLike accepted


def test_index_cache_roundtrip(clips, tmp_path):
    cache = str(tmp_path / "index.npz")
    path = clips.path("mpeg2_tff")
    with VideoReader(path, index_cache=cache) as r1:
        pts = r1.pts.copy()
    assert os.path.isfile(cache)
    with VideoReader(path, index_cache=cache) as r2:
        assert r2.index_method.endswith("+cache")
        assert np.array_equal(r2.pts, pts)
        assert r2.info.n_frames == H.N_FRAMES


def test_closed_reader_raises(clips):
    r = VideoReader(clips.path("dv_avi_bff"))
    r.close()
    with pytest.raises(ValueError):
        r.get_frame(0)
    with pytest.raises(ValueError):
        next(r.iter_frames())


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        VideoReader(os.path.join("no", "such", "file.avi"))


def test_note_duplicate_frames(clips):
    with VideoReader(clips.path("dv_avi_bff")) as r:
        r.note_duplicate_frames([5, 3, 5, 0, 999])
        assert r.info.duplicate_frames == [3, 5]


# ------------------------------------------------------------------- SAR
def test_sar_sources(clips):
    with VideoReader(clips.path("dv_avi_bff")) as r:
        assert r.info.sar_source == "stream" and abs(r.info.sar - 8 / 9) < 1e-9
    with VideoReader(clips.path("h264_prog")) as r:
        assert r.info.sar_source == "default" and abs(r.info.sar - 8 / 9) < 1e-9
        assert any("4:3" in n for n in r.notes)
    with VideoReader(clips.path("h264_prog"), sar_override=10 / 11) as r:
        assert r.info.sar_source == "override" and abs(r.info.sar - 10 / 11) < 1e-12


def test_default_sar_rules():
    assert default_sar(720, 480) == Fraction(8, 9)
    assert default_sar(704, 480) == Fraction(10, 11)
    assert default_sar(720, 576) == Fraction(16, 15)
    assert default_sar(640, 480) == 1 and default_sar(1920, 1080) == 1
    assert resolve_sar(Fraction(1, 1), 720, 480)[:2] == (8 / 9, "default")      # 1:1 on SD = missing tag
    assert resolve_sar(Fraction(32, 27), 720, 480)[:2] == (32 / 27, "stream")
    assert resolve_sar(None, 640, 480)[:2] == (1.0, "default")
    with pytest.raises(ValueError):
        resolve_sar(None, 720, 480, override=0)


# -------------------------------------------------------- timing analysis
def test_analyze_timing_snaps_millisecond_jitter():
    pts = np.round(np.arange(300) * DT * 1000) / 1000 + 0.043      # WMV: 1 ms time base, 43 ms start
    a = analyze_timing(pts, [Fraction(30000, 1001)])
    assert a.mode == "cfr" and a.rate == Fraction(30000, 1001) and a.dropped == []
    np.testing.assert_allclose(a.times, np.arange(300) * DT, atol=1e-12)
    assert a.max_jitter_s <= 0.0006


def test_analyze_timing_gaps_and_vfr():
    n = np.r_[np.arange(50), np.arange(52, 100)]
    a = analyze_timing(n * DT, [])
    assert a.mode == "cfr" and a.dropped == [50] and abs(a.times[50] - 52 * DT) < 1e-12
    # alternating 1.4 / 0.6 frame steps must not be mistaken for a 45 fps grid
    t = (np.arange(60) + 0.4 * (np.arange(60) % 2)) * DT
    v = analyze_timing(t, [Fraction(45)])
    assert v.mode == "vfr" and v.rate == Fraction(30000, 1001)
    np.testing.assert_allclose(v.times, t - t[0])
    assert analyze_timing(np.array([0.0]), []).times.tolist() == [0.0]


def test_frame_decode_error_type():
    assert issubclass(FrameDecodeError, RuntimeError)
