"""Config resolution, overrides and event scheduling of the synthetic generator."""
import math

import numpy as np
import pytest

from pitchspeed.synth.camera import Camera
from pitchspeed.synth.config import (DEFAULT_CONFIG, apply_overrides, level_of, n_source_frames, par_of,
                                     resolve_config)
from pitchspeed.synth.events import make_events
from pitchspeed.synth.gt import build_frame_map, choose_reference


def test_defaults_resolve():
    cfg = resolve_config({})
    assert cfg["video"]["codec"] == "mpeg2"
    assert cfg["video"]["scan"] == "interlaced"
    assert cfg["fence"]["enabled"] is False            # side view default
    assert cfg["scene"]["pole"]["enabled"] is True
    assert math.isclose(par_of(cfg), 10 / 11)
    assert DEFAULT_CONFIG["pitches"]["speed_mph"] == "auto"   # defaults not mutated


def test_level_ranges_and_fence_auto():
    ll = resolve_config({"level": "ll", "camera": {"view": "behind"}})
    hs = resolve_config({"level": "hs"})
    assert ll["pitches"]["speed_mph"] == [40.0, 70.0]
    assert hs["pitches"]["speed_mph"] == [55.0, 92.0]
    assert level_of(ll).distance_ft == 46.0 and level_of(hs).distance_ft == 60.5
    assert ll["fence"]["enabled"] is True              # behind view -> backstop fence on


@pytest.mark.parametrize("bad", [
    {"video": {"codec": "wmv"}},
    {"video": {"width": 721}},
    {"video": {"codec": "dv", "width": 360, "height": 240}},
    {"camera": {"k1": -0.05}},
    {"pitches": {"first_t": 1.0}},
    {"video": {"scan": "progressive30", "postprocess": "blend"}},
    {"level": "sb43"},
])
def test_invalid_configs_raise(bad):
    with pytest.raises((ValueError, KeyError)):
        resolve_config(bad)


def test_overrides_parse_json_values():
    cfg = apply_overrides({}, ["camera.view=behind", "render.shutter_s=0.004", "video.par=[10,11]",
                               "video.postprocess=null", "distractors.batter=false", "video.codec=h264"])
    assert cfg == {"camera": {"view": "behind"}, "render": {"shutter_s": 0.004}, "video": {
        "par": [10, 11], "postprocess": None, "codec": "h264"}, "distractors": {"batter": False}}
    with pytest.raises(ValueError):
        apply_overrides({}, ["novalue"])


def _events(cfg):
    cfg = resolve_config(cfg)
    lev = level_of(cfg)
    cam = Camera(cfg["camera"], lev, 720, 480, par_of(cfg))
    return make_events(cfg, lev, cam)


def test_events_deterministic_and_view_independent():
    a = _events({"seed": 4, "duration_s": 30, "camera": {"view": "side"}})
    b = _events({"seed": 4, "duration_s": 30, "camera": {"view": "behind"}})
    c = _events({"seed": 5, "duration_s": 30})
    ta = [p.t_release for p in a.pitches]
    assert ta == [p.t_release for p in b.pitches]
    assert np.allclose([p.release_vel for p in a.pitches], [p.release_vel for p in b.pitches])
    assert ta != [p.t_release for p in c.pitches]
    # toggling a distractor does not change the pitches
    d = _events({"seed": 4, "duration_s": 30, "distractors": {"throwback": False, "birds": 0}})
    assert ta == [p.t_release for p in d.pitches]


def test_schedule_and_truncation_warning():
    ev = _events({"seed": 1, "duration_s": 6.0,
                  "pitches": {"schedule": [{"t_release": 2.0, "speed_mph": 70.0},
                                           {"t_release": 5.9, "speed_mph": 60.0}]}})
    assert len(ev.pitches) == 1 and ev.requested == 2
    assert ev.pitches[0].t_release == 2.0
    assert abs(np.linalg.norm(ev.pitches[0].release_vel) / (22 / 15) - 70.0) < 1e-9
    assert ev.warnings and "truncated" in ev.warnings[0]


def test_frame_map_and_reference():
    cfg = resolve_config({"duration_s": 3.0, "capture": {"drop_frames": [5], "dup_frames": [7]}})
    fm = build_frame_map(cfg)
    assert n_source_frames(cfg) == 89
    assert len(fm) == 89                                # one dropped, one duplicated
    assert [e["src_frame"] for e in fm[4:9]] == [4, 6, 7, 7, 8]
    assert fm[7]["dup"] and not fm[6]["dup"]
    for e in fm:
        assert e["field_index"] == [2 * e["src_frame"], 2 * e["src_frame"] + 1]
        assert e["parity"] == [1, 0]                    # BFF default
        assert e["exposure_times"] == [k * 1001 / 60000 for k in e["field_index"]]
    ref = choose_reference(cfg, fm)
    assert max(ref["exposure_times"]) < cfg["calibration"]["clear_field_s"]
    tff = build_frame_map(resolve_config({"duration_s": 1.0, "video": {"field_order": "tff"}}))
    assert tff[0]["parity"] == [0, 1]
    bob = build_frame_map(resolve_config({"duration_s": 1.0, "video": {"postprocess": "yadif_bob"}}))
    assert [e["field_index"] for e in bob[:3]] == [[0], [1], [2]]
    assert abs(bob[3]["t_video"] - 3 * 1001 / 60000) < 1e-12
    p30 = build_frame_map(resolve_config({"duration_s": 1.0, "video": {"scan": "progressive30"}}))
    assert p30[2]["exposure_times"] == [4 * 1001 / 60000]
