"""End-to-end synthetic clips: files, field timing through codecs, GT / calibration consistency."""
import json
import math
import os

import numpy as np
import pytest

from pitchspeed.calib.model import CalibrationFile
from pitchspeed.synth.check import check_clip, iter_bgr_frames
from pitchspeed.synth.config import resolve_config
from pitchspeed.synth.generate import generate_clip
from pitchspeed.synth.gt import calib_path_for, cams_path_for, gt_path_for, load_gt
from pitchspeed.synth.render import Renderer
from synth_helpers import project_gt, small

FIELD = 1001 / 60000
N_FRAMES = 209                       # floor(7 s * 30000/1001)


@pytest.fixture(scope="module")
def side_clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth_side")
    path = str(d / "side_bff.mpg")
    gt = generate_clip(small(), path)
    return path, gt


@pytest.fixture(scope="module")
def behind_clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth behind ü")          # space + non-ASCII path
    path = str(d / "behind_tff.mpg")
    gt = generate_clip(small(seed=12, camera={"view": "behind"}, video={"field_order": "tff"},
                             calibration={"click_noise_px": 1.5}, distractors={"birds": 2}), path)
    return path, gt


def test_files_written(side_clip, behind_clip):
    for path, gt in (side_clip, behind_clip):
        for p in (path, gt_path_for(path), calib_path_for(path), cams_path_for(path)):
            assert os.path.exists(p), p
        assert load_gt(path)["schema"] == "pitchspeed.synth.gt/1"
        assert gt["video"]["n_out_frames"] == N_FRAMES and gt["frame_map_identity"]
        assert len(gt["pitches"]) == 2 and not gt["warnings"]
    assert os.path.exists(calib_path_for(behind_clip[0], exact=True))
    assert not os.path.exists(calib_path_for(side_clip[0], exact=True))
    assert behind_clip[1]["camera"]["fence"]["enabled"] and not side_clip[1]["camera"]["fence"]["enabled"]


@pytest.mark.parametrize("which", ["side_clip", "behind_clip"])
def test_field_barcodes_through_mpeg2(which, request):
    path, gt = request.getfixturevalue(which)
    rep = check_clip(path)
    assert rep["ok"], rep
    assert rep["n_decoded"] == N_FRAMES
    assert rep["stamps_checked"] == 2 * N_FRAMES and rep["stamp_mismatches"] == 0
    assert rep["metadata"]["field_order"] == ("tt" if gt["video"]["field_order"] == "tff" else "bb")


@pytest.mark.parametrize("which", ["side_clip", "behind_clip"])
def test_sample_times_and_parity(which, request):
    _, gt = request.getfixturevalue(which)
    tff = gt["video"]["field_order"] == "tff"
    for e in gt["frame_map"]:
        assert e["exposure_times"] == [k * 1001 / 60000 for k in e["field_index"]]
    for p in gt["pitches"]:
        for f in p["fields"]:
            k = f["field_index"]
            assert f["t"] == k * 1001 / 60000                      # t_k = k * 1001/60000 exactly
            assert f["frame"] == k // 2 and f["field_in_frame"] == k % 2 and f["out_frame"] == k // 2
            assert f["parity"] == (f["field_in_frame"] if tff else 1 - f["field_in_frame"])


@pytest.mark.parametrize("which", ["side_clip", "behind_clip"])
def test_pitch_ground_truth(which, request):
    _, gt = request.getfixturevalue(which)
    lo, hi = gt["config"]["pitches"]["speed_mph"]
    for p in gt["pitches"]:
        assert lo <= p["speed_release_mph"] <= hi
        assert p["speed_catch_mph"] < p["speed_plate_front_mph"] < p["speed_release_mph"]
        assert p["avg_speed_y_mph"] < p["speed_release_mph"]
        assert p["flight_time"] == pytest.approx(p["t_catch"] - p["t_release"])
        assert p["dist_y_ft"] == pytest.approx(p["release_pos_ft"][1] - p["catch_pos_ft"][1])
        assert p["extension_ft"] == pytest.approx(gt["level"]["distance_ft"] - p["release_pos_ft"][1])
        states = [f["state"] for f in p["fields"]]
        assert states == sorted(states, key=["in_hand", "flight", "caught"].index)   # in order
        hand = [f for f in p["fields"] if f["state"] == "in_hand"]
        flight = [f for f in p["fields"] if f["state"] == "flight"]
        assert len(hand) >= 3                                       # ball rides the hand ~60 ms
        assert all(f["t"] < p["t_release"] for f in hand)
        assert all(p["t_release"] <= f["t"] <= p["t_catch"] for f in flight)
        assert len(flight) == pytest.approx(p["flight_time"] / FIELD, abs=1.01)
        for f in p["fields"]:
            if f["vis_frac"] > 0.999 and f["state"] == "flight":
                assert f["u_vis"] == pytest.approx(f["u"], abs=0.05 + abs(f["u_end"] - f["u_start"]) * 0.01)


def test_distractors_logged(side_clip, behind_clip):
    for _, gt in (side_clip, behind_clip):
        types = [d["type"] for d in gt["distractors"]]
        assert types.count("throwback") >= 1
        pitch_windows = [(p["t_release"] - 1.0, p["t_catch"] + 0.5) for p in gt["pitches"]]
        assert types.count("bird") == gt["config"]["distractors"]["birds"]
        assert types.count("swing") == sum(p["swing"] for p in gt["pitches"])
        for d in gt["distractors"]:
            if d["type"] == "throwback":
                p = gt["pitches"][d["after_pitch"] - 1]
                assert d["t_start"] > p["t_catch"] and 0.5 < d["t_end"] - d["t_start"] < 1.5
                assert not any(a < d["t_end"] and d["t_start"] < b for a, b in pitch_windows)


@pytest.mark.parametrize("which", ["side_clip", "behind_clip"])
def test_calibration_file_matches_world_projection(which, request):
    path, gt = request.getfixturevalue(which)
    exact_path = calib_path_for(path, exact=gt["calibration"]["click_noise_px"] > 0)
    cf = CalibrationFile.load(exact_path)
    assert len(cf.segments) == 1
    seg = cf.segments[0]
    st = cf.settings
    assert seg.view == gt["config"]["camera"]["view"]
    assert seg.ref_frame == gt["calibration"]["ref_frame"]
    assert max(gt["frame_map"][seg.ref_frame]["exposure_times"]) < gt["config"]["calibration"]["clear_field_s"]
    assert st.distance_ft == 60.5 and st.level == "hs" and st.sport == "baseball"
    assert st.extension_ft is None
    assert st.sar == pytest.approx(10 / 11)
    assert tuple(cf.image_size) == (360, 240)
    assert cf.video["name"] == os.path.basename(path) and cf.video["n_frames"] == N_FRAMES
    cf.validate()                                                   # no fatal problem
    cams = np.load(cams_path_for(path))
    R, z = cams["R_ref"], float(cams["zoom_ref"])
    D, h = 60.5, gt["level"]["mound_h_ft"]
    world = {"rubber": [0, D + 0.25, h], "plate": [0, 17 / 24, 0],
             "release": np.mean([p["release_pos_ft"] for p in gt["pitches"]], axis=0),
             "catch": np.mean([p["catch_pos_ft"] for p in gt["pitches"]], axis=0)}
    for n, P in world.items():
        if n in seg.points:
            assert np.allclose(seg.points[n], project_gt(gt, R, z, P)[0], atol=1e-6), n
    for base, (a, b) in {"rubber": ([-1, D + 0.25, h], [1, D + 0.25, h]),
                         "plate": ([-8.5 / 12, 17 / 12, 0], [8.5 / 12, 17 / 12, 0])}.items():
        ua, ub = project_gt(gt, R, z, [a, b])
        left, right = (ua, ub) if ua[0] <= ub[0] else (ub, ua)
        assert np.allclose(seg.points[base + "_left"], left, atol=1e-6)
        assert np.allclose(seg.points[base + "_right"], right, atol=1e-6)
        assert seg.points[base + "_left"][0] <= seg.points[base + "_right"][0]
    # the reference camera is the mean of the two fields of the reference frame (to < 0.05 px)
    ks = gt["frame_map"][seg.ref_frame]["field_index"]
    sel = [int(np.nonzero(cams["k"] == k)[0][0]) for k in ks]
    mean_two = np.mean([project_gt(gt, cams["R"][i], float(cams["zoom"][i]), world["plate"])[0] for i in sel], axis=0)
    assert np.allclose(mean_two, seg.points["plate"], atol=0.05)
    if gt["config"]["camera"]["view"] == "side":
        assert "vertical_top" in seg.points and seg.points["vertical_top"][1] < seg.points["vertical_bottom"][1] - 0.08 * 240


def test_stabilized_gt_and_homographies(side_clip):
    path, gt = side_clip
    cams = np.load(cams_path_for(path))
    R_ref, z_ref = cams["R_ref"], float(cams["zoom_ref"])
    for p in gt["pitches"]:
        for f in p["fields"][::5]:
            ref = project_gt(gt, R_ref, z_ref, f["pos_ft"])[0]
            assert np.allclose([f["u_ref"], f["v_ref"]], ref, atol=1e-6)
            i = int(np.nonzero(cams["k"] == f["field_index"])[0][0])
            uv = project_gt(gt, cams["R"][i], float(cams["zoom"][i]), f["pos_ft"])[0]
            assert np.allclose(uv, [f["u"], f["v"]], atol=1e-6)
            q = cams["H_to_ref"][i] @ np.array([uv[0], uv[1], 1.0])
            assert np.allclose(q[:2] / q[2], ref, atol=1e-6)


def test_click_noise_and_sar_direction(behind_clip):
    path, gt = behind_clip
    noisy = CalibrationFile.load(calib_path_for(path)).segments[0].points
    exact = CalibrationFile.load(calib_path_for(path, exact=True)).segments[0].points
    assert set(noisy) == set(exact) and len(exact) == 8            # all points of a behind view
    assert exact == {k: list(v) for k, v in gt["calibration"]["points_exact"].items()}
    sar = CalibrationFile.load(calib_path_for(path)).settings.sar
    d = np.array([[(noisy[k][0] - exact[k][0]) * sar, noisy[k][1] - exact[k][1]] for k in exact])
    assert 0.3 < np.sqrt((d ** 2).mean()) < 4.0                     # sigma 1.5 square px
    # square pixels are (x * sar, y): the ball's image is round in square px
    for p in gt["pitches"]:
        for f in p["fields"]:
            assert f["r_px_x"] * sar == pytest.approx(f["r_px_y"], rel=1e-9)
    assert gt["camera"]["fy"] == pytest.approx(gt["camera"]["fx"] * sar)


def test_fence_overlay_present_in_behind_view(behind_clip):
    path, gt = behind_clip
    vis = [f for p in gt["pitches"] for f in p["fields"] if f["state"] == "flight"]
    assert max(f["fence_alpha"] for f in vis) > 0.05                # wires cross the ball
    assert all(0.0 <= f["fence_alpha"] <= 1.0 for f in vis)
    assert gt["camera"]["fence"]["blur_px"] > 0


def _decoded(path):
    return [f for _, f in iter_bgr_frames(path)]


def test_parallel_render_is_bit_identical_and_lossless(tmp_path):
    cfg = small(duration_s=2.5, video={"codec": "ffv1", "debug_stamp": False},
                pitches={"count": 0, "first_t": 2.0})
    g1 = generate_clip(cfg, str(tmp_path / "serial.mkv"), jobs=1)
    g2 = generate_clip(cfg, str(tmp_path / "parallel.mkv"), jobs=2)
    a, b = _decoded(str(tmp_path / "serial.mkv")), _decoded(str(tmp_path / "parallel.mkv"))
    assert len(a) == len(b) == g1["video"]["n_out_frames"]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    r = Renderer(resolve_config(g1["config"]), g1["calibration"]["ref_exposure_times"])
    for n in (0, 17, len(a) - 1):
        assert np.array_equal(r.render_frame(n, collect=False).image, a[n])     # FFV1 is lossless
    assert json.dumps(g1["distractors"]) == json.dumps(g2["distractors"])


def test_codecs_and_field_order_full_res(tmp_path):
    cfg = {"seed": 21, "duration_s": 3.0, "pitches": {"count": 0, "first_t": 2.0},
           "video": {"codec": "dv", "field_order": "bff", "debug_stamp": True,
                     "extra_outputs": [{"codec": "h264", "out": "c_h264.mp4", "bitrate_kbps": 1200},
                                       {"codec": "ffv1", "out": "c_ffv1.mkv"},
                                       {"codec": "mpeg2", "out": "c_lie.mpg", "metadata_field_order": "tff"}]}}
    generate_clip(cfg, str(tmp_path / "c_dv.avi"))
    for f in ("c_dv.avi", "c_h264.mp4", "c_ffv1.mkv", "c_lie.mpg"):
        rep = check_clip(str(tmp_path / f))
        assert rep["ok"] and rep["stamps_checked"] == 2 * 89, (f, rep)
    assert check_clip(str(tmp_path / "c_lie.mpg"))["metadata"]["field_order"] == "tt"   # the lie is in the file


def test_scan_variants_with_drops(tmp_path):
    base = small(duration_s=2.5, pitches={"count": 0, "first_t": 2.0},
                 capture={"drop_frames": [10], "dup_frames": [20]})
    for name, video in [("p60", {"scan": "progressive60"}), ("bob", {"postprocess": "yadif_bob", "field_order": "tff"}),
                        ("fd", {"postprocess": "field_discard"}), ("il", {})]:
        gt = generate_clip(dict(base, video=dict(base["video"], **video)), str(tmp_path / f"{name}.mpg"))
        rep = check_clip(str(tmp_path / f"{name}.mpg"))
        assert rep["ok"] and rep["stamps_checked"] > 0, (name, rep)
        assert not gt["frame_map_identity"]
