"""Index cache robustness (review finding video-io #4).

A damaged index.npz (truncated / empty after a crash, garbage, wrong arrays)
must never stop `analyze`: it is rebuilt and rewritten.  The cache key is
taken before the index is built, so a file that changes while it is read
(still being copied) is not cached under its final size/mtime.
"""
from __future__ import annotations

import glob
import os
import shutil

import numpy as np
import pytest

import io_helpers as H
from pitchspeed.io import reader as R
from pitchspeed.io.reader import VideoReader


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    return H.clip_factory(tmp_path_factory)


@pytest.fixture()
def video(clips, tmp_path):
    p = tmp_path / "tape.mpg"
    shutil.copy(clips.path("mpeg2_tff"), p)
    return str(p)


def _good_cache(video, path):
    with VideoReader(video, index_cache=path) as r:
        return r.pts.copy()


def _damage(kind: str, path: str, good: bytes) -> None:
    with open(path, "wb") as f:
        if kind == "empty":
            pass
        elif kind == "truncated":
            f.write(good[: len(good) // 2])
        elif kind == "tiny":
            f.write(good[:11])
        elif kind == "garbage":
            f.write(b"\x00not a zip file at all" * 50)
    if kind == "wrong_arrays":
        st = os.stat(path.replace("index.npz", "tape.mpg"))
        np.savez(open(path, "wb"), pts=np.zeros((3, 2), np.float32), method=np.array("demux"),
                 synthesized=np.array(False), size=np.array(st.st_size), mtime_ns=np.array(st.st_mtime_ns),
                 version=np.array(R.INDEX_CACHE_VERSION))
    if kind == "missing_key":
        np.savez(open(path, "wb"), pts=np.arange(5))


@pytest.mark.parametrize("kind", ["empty", "truncated", "tiny", "garbage", "wrong_arrays", "missing_key"])
def test_damaged_index_cache_is_rebuilt(video, tmp_path, kind):
    cache = str(tmp_path / "index.npz")
    pts = _good_cache(video, cache)
    good = open(cache, "rb").read()
    _damage(kind, cache, good)
    with VideoReader(video, index_cache=cache) as r:            # must not raise
        assert not r.index_method.endswith("+cache")
        assert np.array_equal(r.pts, pts) and r.info.n_frames == H.N_FRAMES
    # ... and the cache was rewritten, valid, under the same name
    with VideoReader(video, index_cache=cache) as r:
        assert r.index_method.endswith("+cache") and np.array_equal(r.pts, pts)
    assert glob.glob(cache + "*.tmp") == []


def test_stale_cache_is_replaced(video, tmp_path):
    cache = str(tmp_path / "index.npz")
    pts = _good_cache(video, cache)
    st = os.stat(video)
    os.utime(video, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    with VideoReader(video, index_cache=cache) as r:
        assert not r.index_method.endswith("+cache") and np.array_equal(r.pts, pts)
    with VideoReader(video, index_cache=cache) as r:
        assert r.index_method.endswith("+cache")


def test_file_changing_during_index_build_is_not_cached(video, tmp_path, monkeypatch):
    """Video still being copied: the index of the partial file must not be
    stored under the complete file's size/mtime; the user is told."""
    cache = str(tmp_path / "index.npz")
    orig = VideoReader._build_index

    def build_while_copying(self, mode):
        idx = orig(self, mode)
        with open(self.path, "ab") as f:                       # the copy finishes meanwhile
            f.write(b"\x00" * 4096)
        return idx

    monkeypatch.setattr(VideoReader, "_build_index", build_while_copying)
    with VideoReader(video, index_cache=cache) as r:
        assert any("changed while it was being read" in n for n in r.notes)
    assert not os.path.exists(cache)
    monkeypatch.setattr(VideoReader, "_build_index", orig)
    with VideoReader(video, index_cache=cache) as r:           # complete file: normal again
        assert not any("changed while" in n for n in r.notes)
    assert os.path.isfile(cache)


def test_unwritable_cache_location_is_not_fatal(video, tmp_path):
    cache = str(tmp_path / "no_such_dir" / "index.npz")
    with VideoReader(video, index_cache=cache) as r:
        assert r.info.n_frames == H.N_FRAMES
    assert not os.path.exists(cache)


def test_save_cached_index_fsyncs_before_replace(video, tmp_path, monkeypatch):
    cache = str(tmp_path / "index.npz")
    calls = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(R.os, "fsync", lambda fd: (calls.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(R.os, "replace", lambda a, b: (calls.append("replace"), real_replace(a, b))[1])
    _good_cache(video, cache)
    assert calls == ["fsync", "replace"]
    assert os.path.getsize(cache) > 0
