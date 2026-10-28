"""Tests for pitchspeed.io.interlace: scan detection, SampleSource, per-pitch checks."""
from __future__ import annotations

import json

import numpy as np
import pytest

import io_helpers as H
from pitchspeed.io.interlace import (SampleSource, classify_comb, combine_tests, detect_scan,
                                     field_order_check, flipped_times, residual_alternation, run_idet)
from pitchspeed.io.reader import VideoReader
from pitchspeed.types import ScanInfo, result_from_dict, to_jsonable

DT = 1001.0 / 30000.0


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    return H.clip_factory(tmp_path_factory)


def _scan(path, **kw):
    r = VideoReader(path)
    return r, detect_scan(r, **kw)


# ------------------------------------------------------------- detection
@pytest.mark.parametrize("name,mode,order", [
    ("dv_avi_bff", "interlaced", "bff"),
    ("dv_raw_bff", "interlaced", "bff"),
    ("mpeg2_tff", "interlaced", "tff"),
    ("h264_tff", "interlaced", "tff"),
    ("h264_bff_untagged", "interlaced", "bff"),      # metadata says progressive, content is BFF
    ("ffv1_tff", "interlaced", "tff"),
    ("wmv_tff", "interlaced", "tff"),
    ("h264_prog", "progressive", None),
])
def test_detect_scan_from_content(clips, name, mode, order):
    r, scan = _scan(clips.path(name), use_idet=False)
    assert (scan.mode, scan.field_order) == (mode, order), scan.evidence.get("notes")
    assert scan.confidence == "high" and scan.source == "content"
    assert scan.sample_rate == pytest.approx(30000 / 1001 * (2 if mode == "interlaced" else 1))
    r.close()


@pytest.mark.parametrize("name,order", [("hard_tff_mpeg2", "tff"), ("hard_bff_dv", "bff")])
def test_detect_scan_hard_case_lone_ball(clips, name, order):
    """Static camera, only a small ball moving, noise sigma 7: the comb test
    saturates, the blob-midpoint test must decide."""
    r, scan = _scan(clips.path(name), use_idet=False)
    assert scan.mode == "interlaced" and scan.field_order == order
    assert scan.confidence in ("high", "medium")
    assert scan.evidence["blob"]["verdict"] == order
    r.close()


def test_idet_cross_check_runs_bounded(clips):
    res = run_idet(clips.path("dv_avi_bff"), 0.0, 40, timeout_s=30)
    if res is None:
        pytest.skip("bundled ffmpeg not available")
    assert res["verdict"] == "bff" and res["bff"] > res["tff"]
    r, scan = _scan(clips.path("mpeg2_tff"), use_idet=True)
    assert scan.evidence["idet"]["verdict"] == "tff" and scan.confidence == "high"
    r.close()


def test_override_is_used_and_conflict_reported(clips):
    r, scan = _scan(clips.path("dv_avi_bff"), override="tff", use_idet=False)
    assert (scan.mode, scan.field_order, scan.source, scan.confidence) == ("interlaced", "tff", "override", "high")
    assert scan.evidence["override_conflict"] is True
    assert scan.evidence["content_verdict"]["field_order"] == "bff"
    scan2 = detect_scan(r, override="progressive", verify_override=False)
    assert scan2.mode == "progressive" and scan2.sample_rate == pytest.approx(30000 / 1001)
    assert detect_scan(r, override="auto", use_idet=False).field_order == "bff"
    with pytest.raises(ValueError):
        detect_scan(r, override="sideways")
    r.close()


def test_fake_60p_frame_doubling(clips):
    for name, phase in (("fake60_p0", 0), ("fake60_p1", 1)):
        r, scan = _scan(clips.path(name), use_idet=False)
        assert scan.mode == "progressive"
        assert scan.evidence["frame_step"] == 2 and scan.evidence["frame_phase"] == phase
        assert scan.sample_rate == pytest.approx(30000 / 1001)
        src = SampleSource(r, scan)
        assert src.sample_dt == pytest.approx(DT)
        assert src.n_samples == (r.n_frames - phase + 1) // 2
        samples = list(src.iter_samples())
        assert [s.frame for s in samples] == list(range(phase, r.n_frames, 2))
        assert r.info.duplicate_frames == []          # the expected copies are not capture glitches
        r.close()


def test_field_shifted_progressive_is_repaired(clips):
    clip = clips.get("shifted")
    r, scan = _scan(clip["path"], use_idet=False)
    assert scan.mode == "progressive" and scan.evidence["repair"] == "bottom_from_next"
    src = SampleSource(r, scan)
    assert src.n_samples == r.n_frames - 1
    s = src.get_sample(20)
    # sample i = progressive frame P_(i+1) of the source (top of frame i, bottom of frame i+1)
    assert np.abs(s.image.astype(float) - clip["progressive"][20]).mean() < 3.0
    it = {x.s: x.image for x in src.iter_samples(18, 22)}
    assert sorted(it) == [18, 19, 20, 21] and np.array_equal(it[20], s.image)
    r.close()


def test_capture_duplicate_recorded(clips):
    r, scan = _scan(clips.path("dup"), use_idet=False)
    assert r.info.duplicate_frames == [41]            # found inside the detection window
    r.info.duplicate_frames = []
    src = SampleSource(r, scan)
    n = sum(1 for _ in src.iter_samples())
    assert n == 2 * H.N_FRAMES and r.info.duplicate_frames == [41]
    r.close()


def test_no_motion_falls_back_to_prior(tmp_path):
    frames = [np.full((480, 720), 120, np.uint8) for _ in range(20)]
    p = H.encode(frames, str(tmp_path / "static.avi"), "dv_avi")
    r, scan = _scan(p, use_idet=False)
    assert (scan.mode, scan.field_order, scan.confidence, scan.source) == ("interlaced", "bff", "low", "default")
    r.close()


def test_classifier_rules():
    rows = np.tile([10.0, 10.0, 25.0, 500], (30, 1))      # a ~ b < c: TFF
    assert classify_comb(rows)["verdict"] == "tff"
    rows = np.tile([10.0, 25.0, 10.0, 500], (30, 1))
    assert classify_comb(rows)["verdict"] == "bff"
    rows = np.tile([1.0, 20.0, 20.0, 500], (30, 1))
    assert classify_comb(rows)["verdict"] == "progressive"
    rows = np.tile([20.0, 40.0, 1.0, 500], (30, 1))       # c ~ 0: bottom field belongs to the next frame
    res = classify_comb(rows)
    assert res["verdict"] == "shifted" and res["repair"] == "bottom_from_next"
    assert classify_comb(rows[:3])["verdict"] == "none"
    assert combine_tests({"verdict": "uncertain"}, {"verdict": "tff", "strong": True, "n": 30})[:2] == ("tff", "high")
    assert combine_tests({"verdict": "progressive", "strong": False, "votes": {"progressive": 0.6}},
                         {"verdict": "tff", "strong": True, "n": 30, "votes": {"tff": 1.0}})[:2] == ("tff", "medium")


# ---------------------------------------------------------- SampleSource
@pytest.mark.parametrize("name", ["dv_avi_bff", "mpeg2_tff"])
def test_sample_source_indexing_and_rows(clips, name):
    clip = clips.get(name)
    r, scan = _scan(clip["path"], use_idet=False)
    src = SampleSource(r, scan)
    assert src.n_samples == 2 * r.n_frames and src.sample_dt == pytest.approx(DT / 2)
    assert src.sample_shape == (240, 720)
    first = 0 if scan.field_order == "tff" else 1
    for s in (0, 1, 2, 3, 57, 119):
        i, k, p = src.locate(s)
        assert (i, k) == (s // 2, s % 2) and p == (k ^ first)
        assert src.sample_of(i, k) == s
        assert src.sample_time(s) == pytest.approx(r.frame_times[i] + k * DT / 2, abs=1e-12)
    np.testing.assert_allclose(src.sample_times(), np.arange(120) * DT / 2, atol=1e-12)
    assert src.sample_of(r.n_frames) == src.n_samples          # exclusive end
    assert src.sample_at_time(0.5 + 0.004) == round(0.5 / (DT / 2))
    frame = r.get_frame(21)
    smp = src.get_sample(43, color=True)
    assert smp.frame == 21 and smp.k == 1 and smp.image.shape == (240, 720)
    assert np.array_equal(smp.image, frame[smp.parity::2])     # native field rows, not resized
    assert smp.bgr.shape == (240, 720, 3) and smp.image.flags["C_CONTIGUOUS"]
    samples = list(src.iter_samples(40, 47))
    assert [x.s for x in samples] == list(range(40, 47))
    assert np.array_equal(samples[3].image, smp.image)
    with pytest.raises(IndexError):
        src.locate(src.n_samples)
    r.close()


@pytest.mark.parametrize("name", ["dv_avi_bff", "mpeg2_tff", "h264_tff"])
def test_field_timing_matches_ground_truth(clips, name):
    """Ball x measured in every field vs x(t_s): ~0.4 px with the detected
    order, about one field of motion (18 px) with the wrong order."""
    clip = clips.get(name)
    sc = clip["scene"]
    r, scan = _scan(clip["path"], use_idet=False)
    wrong = ScanInfo("interlaced", "tff" if scan.field_order == "bff" else "bff", "low", "override", 59.94)
    errs = {}
    for label, s in (("right", scan), ("wrong", wrong)):
        src = SampleSource(r, s)
        e, T, X, Y, K = [], [], [], [], []
        for smp in src.iter_samples():
            truth = sc.ball_x(smp.t + H.FIELD_DT / 2)
            if truth is None or smp.t < sc.ball_t0 + 0.02 or sc.ball_x(smp.t + H.FIELD_DT * 1.5) is None:
                continue
            rows = 2 * np.arange(smp.image.shape[0]) + smp.parity
            c = H.ball_centroid(smp.image, rows, sc.bg[rows], truth)
            if c is not None:
                e.append(abs(c[0] - truth))
                T.append(smp.t), X.append(c[0]), Y.append(c[1]), K.append(smp.k)
        errs[label] = float(np.median(e))
        assert len(e) >= 25
        assert abs(np.median(Y) - H.BALL_Y) < 0.5                # field rows mapped to frame rows
        chk = field_order_check(T, X, Y, K, src.frame_dt)
        assert chk["preferred"] == ("given" if label == "right" else "flipped")
    assert errs["right"] < 1.0, errs
    assert errs["wrong"] > 12.0, errs
    r.close()


def test_progressive_sample_source(clips):
    r, scan = _scan(clips.path("h264_prog"), use_idet=False)
    src = SampleSource(r, scan)
    assert src.n_samples == r.n_frames and src.sample_shape == (480, 720)
    assert src.sample_dt == pytest.approx(DT) and src.locate(7) == (7, 0, 0)
    assert np.array_equal(src.get_sample(7).image, r.get_frame(7))
    with pytest.raises(ValueError):
        SampleSource(r, ScanInfo("interlaced", None))
    r.close()


# ------------------------------------------------------ per-pitch helpers
def test_residual_alternation_and_field_order_check():
    rng = np.random.default_rng(0)
    t = np.arange(30) * (DT / 2)
    k = np.arange(30) % 2
    x = 100 + 900 * t - 300 * t ** 2
    y = 200 + 50 * t + 400 * t ** 2
    noisy = x + rng.normal(0, 0.5, 30)
    assert abs(residual_alternation(t, noisy, y + rng.normal(0, 0.5, 30))) < 0.45
    # wrong order: positions belong to t +- dt/2 alternately
    xw = 100 + 900 * flipped_times(t, k, DT) - 300 * flipped_times(t, k, DT) ** 2
    assert residual_alternation(t, xw + rng.normal(0, 0.5, 30), y) < -0.8
    chk = field_order_check(t, xw, None, k, DT)
    assert chk["preferred"] == "flipped" and chk["rms_flipped"] < chk["rms_given"]
    assert field_order_check(t, noisy, y, k, DT)["preferred"] == "given"
    assert np.isnan(residual_alternation(t[:4], x[:4]))
    # an outlier does not create alternation
    xo = noisy.copy()
    xo[12] += 40
    assert residual_alternation(t, xo) > -0.5


def test_scan_info_survives_json_roundtrip(clips):
    """results.json stores ScanInfo; a reloaded scan must rebuild the same SampleSource."""
    r, scan = _scan(clips.path("fake60_p1"), use_idet=True)
    d = json.loads(json.dumps(to_jsonable(scan)))           # plain JSON, no numpy leftovers
    loaded = result_from_dict({"video": to_jsonable(r.info), "scan": d, "calibration": {}, "params": {},
                               "pitches": []}).scan
    a, b = SampleSource(r, scan), SampleSource(r, loaded)
    assert (a.n_samples, a.frame_step, a.frame_phase) == (b.n_samples, b.frame_step, b.frame_phase)
    assert np.array_equal(a.sample_times(), b.sample_times())
    r.close()


def test_detect_scan_non_ascii_path_with_idet(clips, tmp_path):
    import shutil
    d = tmp_path / "Ordner ü 日本"
    d.mkdir()
    p = d / "Spiel é.mpg"
    shutil.copy(clips.path("mpeg2_tff"), p)
    r, scan = _scan(str(p), use_idet=True)
    assert scan.field_order == "tff"
    if scan.evidence.get("idet") is not None:          # None only if no bundled ffmpeg
        assert scan.evidence["idet"]["verdict"] == "tff"
    r.close()
