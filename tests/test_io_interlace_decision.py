"""detect_scan decision rules (review findings video-io #1 and #3).

#1  A weak blob-test 'progressive' vote (slow / noise blobs give q ~ 0.5)
    must not overrule or tie a comb-test field order, and a decisive idet
    field order decides when the content evidence is weak.  Calling
    interlaced video progressive cannot be undone downstream, while an
    interlaced verdict is still checked against the ball tracks.
#3  Above 40 frames/s a frame cannot hold an interlaced field pair: neither
    the content tests nor idet may produce an interlaced scan there (only an
    explicit override can).

The evidence dicts below are the values measured on the reviewer's 8 s cut
of an interlaced BFF DV behind-view clip (comb bff 0.633 weak, blob weak
progressive 0.637, idet 239 BFF / 0 progressive).
"""
from __future__ import annotations

import pytest

import io_helpers as H
from pitchspeed.io import interlace as I
from pitchspeed.io.interlace import combine_tests, detect_scan, idet_decisive
from pitchspeed.io.reader import VideoReader

COMB_BFF_WEAK = {"verdict": "bff", "strong": False, "n": 180, "a_over_minbc": 0.775, "b_over_c": 1.31,
                 "votes": {"progressive": 0.017, "tff": 0.06, "bff": 0.633, "shifted": 0.0}, "combing": False}
COMB_TFF_WEAK = {"verdict": "tff", "strong": False, "n": 180, "a_over_minbc": 0.7, "b_over_c": 0.78,
                 "votes": {"progressive": 0.02, "tff": 0.65, "bff": 0.05, "shifted": 0.0}, "combing": False}
COMB_TFF_STRONG = {"verdict": "tff", "strong": True, "n": 180, "a_over_minbc": 0.9, "b_over_c": 0.4,
                   "votes": {"progressive": 0.0, "tff": 0.95, "bff": 0.0, "shifted": 0.0}, "combing": True}
COMB_UNCERTAIN = {"verdict": "uncertain", "strong": False, "n": 180, "a_over_minbc": 0.7, "b_over_c": 1.0,
                  "votes": {"progressive": 0.03, "tff": 0.3, "bff": 0.3, "shifted": 0.0}, "combing": False}
BLOB_PROG_WEAK = {"verdict": "progressive", "strong": False, "n": 140, "median_q": 0.5,
                  "votes": {"tff": 0.1, "bff": 0.2, "progressive": 0.637}}
BLOB_BFF_WEAK = {"verdict": "bff", "strong": False, "n": 140, "median_q": 0.7,
                 "votes": {"tff": 0.1, "bff": 0.62, "progressive": 0.2}}
BLOB_UNCERTAIN = {"verdict": "uncertain", "strong": False, "n": 40, "median_q": 0.5,
                  "votes": {"tff": 0.3, "bff": 0.3, "progressive": 0.3}}
IDET_BFF = {"tff": 0, "bff": 239, "progressive": 0, "undetermined": 1, "start_s": 0.0, "verdict": "bff"}
IDET_BFF_LEANING = {"tff": 0, "bff": 30, "progressive": 10, "undetermined": 20, "start_s": 0.0, "verdict": "bff"}
IDET_PROG = {"tff": 0, "bff": 0, "progressive": 49, "undetermined": 11, "start_s": 0.0, "verdict": "progressive"}


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    return H.clip_factory(tmp_path_factory)


def _patched_scan(monkeypatch, path, comb, blob, idet=None, **kw):
    """detect_scan on a real clip (its rate / size / codec) with given evidence."""
    monkeypatch.setattr(I, "classify_comb", lambda rows: dict(comb))
    monkeypatch.setattr(I, "classify_blob", lambda q: dict(blob))
    monkeypatch.setattr(I, "run_idet", lambda *a, **k: None if idet is None else dict(idet))
    with VideoReader(path) as r:
        return detect_scan(r, use_idet=idet is not None, **kw)


def _notes(scan):
    return " | ".join(scan.evidence.get("notes", []))


# ------------------------------------------------------------ finding #1
def test_weak_blob_progressive_cannot_tie_a_comb_field_order():
    v, conf, notes = combine_tests(COMB_BFF_WEAK, BLOB_PROG_WEAK)      # 0.637 > 0.633 used to win
    assert (v, conf) == ("bff", "medium")
    assert "ignoring it" in notes[0]
    assert combine_tests(COMB_TFF_WEAK, BLOB_PROG_WEAK)[:2] == ("tff", "medium")
    # a STRONG blob 'progressive' is real evidence and still counts
    strong = {**BLOB_PROG_WEAK, "strong": True, "votes": {"tff": 0, "bff": 0, "progressive": 0.9}}
    assert combine_tests(COMB_BFF_WEAK, strong)[:2] == ("progressive", "medium")
    # combed fields, no comb order: the weak vote is ignored too (unchanged)
    assert combine_tests({**COMB_UNCERTAIN, "combing": True}, BLOB_PROG_WEAK)[0] == "uncertain"


def test_reviewer_cut_is_interlaced_bff(clips, monkeypatch):
    scan = _patched_scan(monkeypatch, clips.path("dv_avi_bff"), COMB_BFF_WEAK, BLOB_PROG_WEAK)
    assert (scan.mode, scan.field_order, scan.confidence, scan.source) == ("interlaced", "bff", "medium", "content")
    scan = _patched_scan(monkeypatch, clips.path("dv_avi_bff"), COMB_BFF_WEAK, BLOB_PROG_WEAK, IDET_BFF)
    assert (scan.mode, scan.field_order, scan.confidence) == ("interlaced", "bff", "medium")


def test_weak_blob_progressive_alone_is_low_and_decisive_idet_decides(clips, monkeypatch):
    p = clips.path("dv_avi_bff")                     # 720x480 29.97 DV: a normally interlaced format
    scan = _patched_scan(monkeypatch, p, COMB_UNCERTAIN, BLOB_PROG_WEAK)
    assert (scan.mode, scan.confidence) == ("progressive", "low")
    assert "only weak blob-test evidence" in _notes(scan)
    scan = _patched_scan(monkeypatch, p, COMB_UNCERTAIN, BLOB_PROG_WEAK, IDET_BFF)
    assert (scan.mode, scan.field_order, scan.confidence) == ("interlaced", "bff", "low"), _notes(scan)
    assert "decisively" in _notes(scan)
    # idet only leaning (30 vs 10): not decisive, the verdict stays (flagged low)
    scan = _patched_scan(monkeypatch, p, COMB_UNCERTAIN, BLOB_PROG_WEAK, IDET_BFF_LEANING)
    assert (scan.mode, scan.confidence) == ("progressive", "low")


def test_content_conflict_at_low_confidence_is_decided_by_decisive_idet(clips, monkeypatch):
    p = clips.path("dv_avi_bff")
    scan = _patched_scan(monkeypatch, p, COMB_TFF_WEAK, BLOB_BFF_WEAK)      # tie-break -> tff, low
    assert (scan.field_order, scan.confidence) == ("tff", "low")
    scan = _patched_scan(monkeypatch, p, COMB_TFF_WEAK, BLOB_BFF_WEAK, IDET_BFF)
    assert (scan.mode, scan.field_order, scan.confidence) == ("interlaced", "bff", "low")


def test_decisive_idet_never_overrides_strong_content(clips, monkeypatch):
    scan = _patched_scan(monkeypatch, clips.path("mpeg2_tff"), COMB_TFF_STRONG, BLOB_UNCERTAIN, IDET_BFF)
    assert (scan.field_order, scan.confidence) == ("tff", "medium")
    assert "ffmpeg idet says bff" in _notes(scan)


def test_idet_progressive_not_trusted_on_an_interlaced_format(clips, monkeypatch):
    """idet says 'progressive' on interlaced static-camera footage (lone ball)."""
    scan = _patched_scan(monkeypatch, clips.path("dv_avi_bff"), COMB_UNCERTAIN, BLOB_UNCERTAIN, IDET_PROG)
    assert (scan.mode, scan.field_order, scan.confidence, scan.source) == ("interlaced", "bff", "low", "default")
    # weak-content interlaced verdict + decisive idet 'progressive': not switched
    scan = _patched_scan(monkeypatch, clips.path("dv_avi_bff"), COMB_TFF_WEAK, BLOB_BFF_WEAK, IDET_PROG)
    assert scan.mode == "interlaced"
    # where the prior says progressive anyway (SD H.264 tagged progressive, no combing) it is used
    scan = _patched_scan(monkeypatch, clips.path("h264_prog"), COMB_UNCERTAIN, BLOB_UNCERTAIN, IDET_PROG)
    assert (scan.mode, scan.confidence) == ("progressive", "low")


def test_idet_decisive_rule():
    assert idet_decisive(IDET_BFF) == "bff"
    assert idet_decisive(IDET_BFF_LEANING) is None
    assert idet_decisive({"tff": 9, "bff": 0, "progressive": 0, "verdict": "tff"}) is None
    assert idet_decisive({"tff": 100, "bff": 10, "progressive": 0, "verdict": "tff"}) == "tff"
    assert idet_decisive({"verdict": "none", "error": "timeout"}) is None and idet_decisive(None) is None


# ------------------------------------------------------------ finding #3
def test_rate_rule_content_interlaced_at_60fps_is_progressive(clips):
    """TFF-woven content labelled 59.94 frames/s: never split into 119.88 fields/s."""
    with VideoReader(clips.path("woven60_tff")) as r:
        assert r.info.frame_rate > 40
        scan = detect_scan(r, use_idet=False)
        assert scan.evidence["comb"]["verdict"] == "tff"          # the content does look TFF
        assert (scan.mode, scan.field_order, scan.confidence, scan.source) == \
            ("progressive", None, "low", "default")
        assert scan.sample_rate == pytest.approx(60000 / 1001)
        assert "cannot be field-interlaced" in _notes(scan)
        # an explicit override still can
        ov = detect_scan(r, override="tff", use_idet=False)
        assert (ov.mode, ov.field_order, ov.source) == ("interlaced", "tff", "override")
        assert ov.sample_rate == pytest.approx(2 * 60000 / 1001)


def test_rate_rule_idet_interlaced_at_60fps_is_ignored(clips, monkeypatch):
    """Reviewer's 59.94p MPEG-2 case: content inconclusive, idet 222 BFF."""
    idet = {"tff": 0, "bff": 222, "progressive": 0, "undetermined": 78, "start_s": 0.0, "verdict": "bff"}
    scan = _patched_scan(monkeypatch, clips.path("woven60_tff"), COMB_UNCERTAIN, BLOB_UNCERTAIN, idet)
    assert (scan.mode, scan.field_order, scan.source) == ("progressive", None, "default")
    assert scan.sample_rate == pytest.approx(60000 / 1001)
    assert "ignored" in _notes(scan)
    # a weak content verdict + decisive idet does not get around the rate rule either
    scan = _patched_scan(monkeypatch, clips.path("woven60_tff"), COMB_UNCERTAIN, BLOB_PROG_WEAK, idet)
    assert scan.mode == "progressive"


def test_fake_60p_still_detected_with_rate_rule(clips):
    with VideoReader(clips.path("fake60_p0")) as r:
        scan = detect_scan(r, use_idet=True)
        assert scan.mode == "progressive" and scan.evidence["frame_step"] == 2
        assert scan.sample_rate == pytest.approx(30000 / 1001)
