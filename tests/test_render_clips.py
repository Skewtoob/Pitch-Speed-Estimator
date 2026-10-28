"""render/clips.py: names, the clips.json bookkeeping (never deletes what it did not write,
never leaves a pitch with nothing after a failure) and the checked untouched copies."""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np
import pytest

pytest.importorskip("av")
pytest.importorskip("imageio_ffmpeg")

import render_helpers as H                                     # noqa: E402
import test_render_integration as T                            # noqa: E402
from pitchspeed.render import clips as CL                      # noqa: E402


@pytest.fixture(scope="module")
def tape(tmp_path_factory):
    d = tmp_path_factory.mktemp("clips_tape")
    return H.make_test_clip(str(d), seconds=9.0)


def _render(tape, out, **kw):
    reader, src = T._source(tape)
    info = {}
    try:
        CL.render_clips(T._result(), src, str(out), audio=False, info=info, **kw)
    finally:
        reader.close()
    return info


def _listed(out):
    return json.load(open(os.path.join(out, "clips", "clips.json")))["files"]


def test_clip_names():
    assert CL.clip_name(3, H.speed(80.6, 1.8, "good")) == "P03_81mph"
    assert CL.clip_name(2, H.speed(65.5, 5.0, "poor")) == "P02_66mph_rough"
    assert CL.clip_name(12, H.speed(None, None, "none"), width=3) == "P012_no_speed"


def test_a_locked_stale_clip_is_deleted_the_next_time(tape, tmp_path, monkeypatch):
    out = tmp_path / "o"
    _render(tape, out, originals=False)
    first = [n for n in _listed(out) if n.startswith("P01_")][0]
    real_remove = os.remove

    def locked(p):
        if os.path.basename(p) == first:
            raise PermissionError(13, "in use by a video player", p)
        real_remove(p)
    monkeypatch.setattr(CL, "clip_name", lambda n, sp, w=2: f"P{n:02d}_changed")
    monkeypatch.setattr(os, "remove", locked)
    info = _render(tape, out, originals=False)
    assert first in info["kept"] and any("could not be removed" in t for t in info["notes"])
    assert first in _listed(out)                                     # still listed ...
    monkeypatch.setattr(os, "remove", real_remove)
    info = _render(tape, out, originals=False)
    assert first in info["removed"] and not os.path.exists(os.path.join(out, "clips", first))   # ... gone now


def test_a_failed_new_clip_keeps_the_previous_one(tape, tmp_path, monkeypatch):
    out = tmp_path / "o"
    _render(tape, out, originals=False)
    before = _listed(out)
    real = CL.render_annotated

    def fail_p1(result, source, path, **kw):
        if os.path.basename(path).startswith("P01_"):
            raise OSError(28, "No space left on device")
        return real(result, source, path, **kw)
    monkeypatch.setattr(CL, "render_annotated", fail_p1)
    info = _render(tape, out, originals=False)
    p1 = [n for n in before if n.startswith("P01_")][0]
    assert p1 in info["kept"] and os.path.exists(os.path.join(out, "clips", p1)) and p1 in _listed(out)


def test_an_interrupted_run_keeps_track_of_its_new_files(tape, tmp_path, monkeypatch):
    out = tmp_path / "o"
    real = CL.render_annotated

    def stop_at_p2(result, source, path, **kw):
        if os.path.basename(path).startswith("P02_"):
            raise KeyboardInterrupt
        return real(result, source, path, **kw)
    monkeypatch.setattr(CL, "render_annotated", stop_at_p2)
    with pytest.raises(KeyboardInterrupt):
        _render(tape, out, originals=False)
    assert any(n.startswith("P01_") for n in _listed(out))


def _frames(path):
    import av
    with av.open(path) as c:
        return [f.to_ndarray(format="gray").astype(np.int16) for f in c.decode(video=0)]


@pytest.mark.parametrize("gop", [15, 120])
def test_untouched_copy_covers_the_pitch_or_is_not_kept(tmp_path, gop):
    """MPEG-2 with short and long key-frame intervals: a copy, when made, starts no later than the
    overlay clip and ends no earlier; one that cannot cover it is reported and deleted."""
    import imageio_ffmpeg
    src = str(tmp_path / f"g{gop}.mpg")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc2=size=320x240:rate=30000/1001", "-t", "12", "-c:v", "mpeg2video", "-g", str(gop),
                    "-bf", "2", "-q:v", "5", src], check=True, capture_output=True, timeout=120)
    S = _frames(src)
    fdt = 1001 / 30000
    for t0 in (2.0, 4.3, 6.1):
        dur = 5.0
        out = str(tmp_path / f"cut_{gop}_{t0}.mpg")
        path, status = CL.cut_original(src, t0, dur, out)
        if path is None:
            assert status.startswith("failed") and not os.path.exists(out)
            continue
        O = _frames(path)
        first = int(np.argmin([np.abs(x - O[0]).mean() for x in S]))
        last = int(np.argmin([np.abs(x - O[-1]).mean() for x in S]))
        assert first <= round(t0 / fdt) + 0 and last >= round((t0 + dur) / fdt) - 4, (gop, t0, first, last, status)


def test_ffmpeg_failure_without_message_still_says_why(tmp_path, monkeypatch, tape):
    class R:
        returncode, stderr = 139, b""
    monkeypatch.setattr(CL.subprocess, "run", lambda *a, **k: R())
    path, status = CL.cut_original(tape, 1.0, 3.0, str(tmp_path / "x.mpg"))
    assert path is None and "exit code 139" in status
