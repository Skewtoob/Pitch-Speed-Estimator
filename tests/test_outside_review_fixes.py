"""Regressions from an outside code review (2026-10-29)."""
import os

from pitchspeed.render import clips
from test_ui_mark_mode import make_mark
from test_ui_review_fixes import keys


def test_nudging_a_bat_contact_mark_keeps_it_bat_contact():
    st, src, res, rec = make_mark()
    st.goto(1000); keys(st, "n")
    st.goto(1027); keys(st, "b")
    p = res.pitches[0]
    v = p.speed.v_release_mph
    keys(st, ".")
    assert p.catch.s == 1028 and p.catch.point == "contact" and p.speed.v_release_mph < v


def test_a_bad_new_copy_never_replaces_a_good_original(tmp_path, monkeypatch):
    out = tmp_path / "P01.mp4"
    out.write_bytes(b"old good copy")

    def fake_copy(exe, src, ss, duration, target):
        tmp = str(tmp_path / "P01.part.mp4"); open(tmp, "wb").write(b"x")
        return tmp, target, ""
    monkeypatch.setattr(clips, "_copy", fake_copy)
    monkeypatch.setattr(clips, "video_span", lambda p: (0.01, 1))      # far too short
    import pitchspeed.io.writer as w
    monkeypatch.setattr(w, "source_audio_offset", lambda src: (None, 0.0))
    path, status = clips.cut_original("src.mp4", 5.0, 2.0, str(out), ffmpeg_exe="ffmpeg")
    assert path is None and out.read_bytes() == b"old good copy"
    assert not any(n.endswith(".part.mp4") for n in os.listdir(tmp_path))
