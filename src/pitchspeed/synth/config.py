"""Configuration of the synthetic clip generator.

A clip is described by one nested JSON-friendly dict.  Callers pass a
*partial* dict; `resolve_config` deep-merges it over `DEFAULT_CONFIG`,
fills the "auto" values (level-dependent pitch ranges, fence on/off, ...)
and validates enumerations.  The fully resolved dict is echoed into the
ground-truth file, so every clip can be regenerated exactly from its GT.

Range-valued pitch parameters are `[lo, hi]` lists sampled uniformly per
pitch; give `[x, x]` for a fixed value, or use `pitches.schedule` for
explicit per-pitch values.

Conventions (shared with the analyzer, see constants.py / synth.md):
feet, seconds, ft/s; world origin at the plate apex, +y to the pitcher,
+x to the 1B side, +z up; field k is exposed at t_k = k * 1001/60000 s.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np

from pitchspeed import constants as C

# ------------------------------------------------------------------ timing
FPS_NUM, FPS_DEN = 30000, 1001
FRAME_DT = FPS_DEN / FPS_NUM               # 1001/30000 s
FIELD_DT = FPS_DEN / (2 * FPS_NUM)         # 1001/60000 s
MPH = C.MPH_TO_FTPS                        # ft/s per mph (22/15)
G = C.G_FTPS2
BALL_R_FT = C.BALL_DIAMETER_FT / 2.0

GENERATOR_VERSION = "pitchspeed.synth/1.0"

SCANS = ("interlaced", "progressive30", "progressive60")
POSTPROCESS = (None, "blend", "field_discard", "yadif", "yadif_bob")
CODECS = ("mpeg2", "dv", "h264", "ffv1", "mpeg4")
TAPE_PROFILES = {
    "none": dict(luma_sigma=0.0, sharpen=0.0, chroma_sigma=0.0, chroma_shift=0.0, jitter=0.0,
                 dropouts=0.0, headswitch=0),
    "dv":   dict(luma_sigma=0.35, sharpen=0.2, chroma_sigma=1.2, chroma_shift=0.0, jitter=0.0,
                 dropouts=0.0, headswitch=0),
    "hi8":  dict(luma_sigma=0.8, sharpen=0.4, chroma_sigma=3.0, chroma_shift=1.0, jitter=0.25,
                 dropouts=0.3, headswitch=4),
    "vhs":  dict(luma_sigma=1.2, sharpen=0.6, chroma_sigma=4.5, chroma_shift=2.0, jitter=0.5,
                 dropouts=1.0, headswitch=6),
}


# ------------------------------------------------------------------ levels
@dataclass(frozen=True)
class LevelGeom:
    """Scene geometry of one league level (feet)."""
    key: str                # constants.LEVELS key ('hs', 'll', ...) or 'custom'
    distance_ft: float      # rubber FRONT edge -> plate apex
    mound_h_ft: float       # mound (rubber top) height above the plate
    base_ft: float          # base-path length (scenery only)
    sport: str = "baseball"


# Mound height / base path per baseball level (scenery and release height).
_SCENE_LEVELS = {"hs": (10 / 12, 90.0), "54": (8 / 12, 80.0), "50": (8 / 12, 70.0),
                 "ll": (6 / 12, 60.0), "40": (6 / 12, 50.0)}

# Level-dependent defaults for 'auto' pitch ranges.
_PITCH_AUTO = {
    "hs": dict(speed_mph=[55.0, 92.0], extension_ft=[4.0, 6.5], release_z_ft=[4.8, 6.2]),
    "54": dict(speed_mph=[45.0, 80.0], extension_ft=[3.5, 5.5], release_z_ft=[4.2, 5.6]),
    "50": dict(speed_mph=[45.0, 78.0], extension_ft=[3.5, 5.5], release_z_ft=[4.2, 5.6]),
    "ll": dict(speed_mph=[40.0, 70.0], extension_ft=[3.0, 4.8], release_z_ft=[3.6, 5.0]),
    "40": dict(speed_mph=[35.0, 62.0], extension_ft=[2.8, 4.5], release_z_ft=[3.4, 4.6]),
}


def resolve_level(level: Any) -> LevelGeom:
    """Level key/alias (see constants.LEVELS) or a dict -> LevelGeom.

    A dict may use {distance_ft, mound_h_ft, base_ft} (or the prototype's
    {rubber, mound_h, base}).  Softball levels are not supported (the
    generator only animates overhand baseball pitchers).
    """
    if isinstance(level, dict):
        d = float(level.get("distance_ft", level.get("rubber", 60.5)))
        h = float(level.get("mound_h_ft", level.get("mound_h", 10 / 12)))
        b = float(level.get("base_ft", level.get("base", 90.0)))
        return LevelGeom(str(level.get("key", "custom")), d, h, b)
    lv = C.get_level(str(level))
    if lv.sport != "baseball" or lv.key not in _SCENE_LEVELS:
        raise ValueError(f"synthetic generator supports baseball levels {sorted(_SCENE_LEVELS)}; got {level!r}")
    h, b = _SCENE_LEVELS[lv.key]
    return LevelGeom(lv.key, lv.distance_ft, h, b, lv.sport)


def _auto_key(lev: LevelGeom) -> str:
    if lev.key in _PITCH_AUTO:
        return lev.key
    d = lev.distance_ft
    return "hs" if d >= 58 else "54" if d >= 52 else "50" if d >= 48 else "ll" if d >= 42 else "40"


# ------------------------------------------------------------------ defaults
DEFAULT_CONFIG: dict = {
    "seed": 1,
    "duration_s": 20.0,
    "level": "hs",
    "video": {
        "width": 720, "height": 480, "par": [10, 11],
        "scan": "interlaced",              # interlaced | progressive30 | progressive60
        "field_order": "bff",              # true temporal order: tff | bff
        "metadata_field_order": "auto",    # auto | tff | bff   (may lie on purpose)
        "postprocess": None,               # None | blend | field_discard | yadif | yadif_bob
        "codec": "mpeg2",                  # mpeg2 | dv | h264 | ffv1 | mpeg4
        "bitrate_kbps": 3500,
        "extra_outputs": [],               # [{codec, out, bitrate_kbps, metadata_field_order}, ...]
        "debug_stamp": False,              # per-field index barcode (top-left)
        "ffmpeg_vf": None,                 # extra ffmpeg filter prepended to the chain
    },
    "camera": {
        "view": "side",                    # side | behind | custom
        "side": "1b",                      # side view: camera on the 1b | 3b side
        "distance_ft": 70.0,               # side: from the pitch-line midpoint
        "angle_deg": 90.0,                 # side: optical axis vs pitch line (<90: toward home)
        "height_ft": 5.0,
        "back_ft": 25.0,                   # behind: distance behind the plate apex
        "lateral_ft": 0.0,                 # behind: +x offset
        "hfov_deg": None,                  # None = auto-fit rubber..plate into the frame
        "fit_margin": 0.12,
        "roll_deg": 0.0,
        "k1": 0.0,                         # lens distortion: NOT modelled, must be 0
        "C": None, "look_at": None,        # custom view
        "f_number": 2.0,                   # optics for the fence defocus model
        "sensor_width_mm": 3.6,            # 1/4" CCD
    },
    "shake": {"amp_deg": 0.12, "roll_amp_deg": 0.05, "fmin_hz": 0.3, "fmax_hz": 3.0, "n_sines": 6,
              "pans": [], "zooms": []},    # pans: [{t, dur, yaw_deg, pitch_deg}], zooms: [{t, dur, factor}]
    "pitches": {
        "count": 3, "schedule": None,      # schedule: [{t_release?, speed_mph?, cd?, ...}, ...]
        "first_t": 3.0, "gap_s": [5.0, 7.0],
        "speed_mph": "auto", "cd": [0.30, 0.40],
        "extension_ft": "auto", "release_z_ft": "auto", "release_x_ft": [-1.8, 1.8],
        "target_x_ft": [-0.7, 0.7], "target_z_ft": [1.6, 3.4],
        "magnus_x": [-10.0, 10.0], "magnus_z": [-6.0, 14.0],
        "catch_behind_ft": [0.8, 2.0], "wind_mph": [0.0, 0.0],
        "hand_time_s": 0.06,               # ball rides the hand this long before release
        "end_margin_s": 0.5,               # last catch must be this far before the clip end
    },
    "render": {"shutter_s": 1 / 250, "vertical_aperture": 2.0, "lens_sigma_px": 0.0,
               "psf_sigma_px": 0.7, "ball_value": 235, "segments": 6, "draw_balls": True},
    "noise": {"sigma0": 3.0, "beta": 0.02, "chroma_sigma": 4.0},
    "tape": {"profile": "vhs"},
    "date_stamp": {"enabled": True, "text_date": "OCT 12 2003", "start_hms": [16, 35, 2]},
    "distractors": {"pitcher": True, "catcher": True, "umpire": True, "batter": True,
                    "swing_prob": 0.5, "throwback": True, "birds": 1},
    "fence": {                             # out-of-focus backstop chain-link in front of the camera
        "enabled": "auto",                 # auto = on for behind views
        "dist_ft": 3.0,                    # fence plane distance in front of the camera
        "mesh_in": 2.0,                    # distance between parallel wires
        "wire_in": 0.148,                  # 9 gauge
        "angle_deg": 45.0,                 # wire direction vs horizontal (45 = square diamonds)
        "blur_px": None,                   # defocus blur diameter (output px); None = from optics
        "focus_ft": None,                  # None = camera -> release point distance
        "wire_value": 110,                 # wire grey level (0..255)
        "opacity": 1.0,
    },
    "scene": {"pole": {"enabled": "auto",  # vertical pole (calibration vertical_top/bottom); auto = side
                       "x_ft": None, "y_ft": None, "height_ft": 12.0, "radius_ft": 0.25,
                       "value": 140}},
    "capture": {"drop_frames": [], "dup_frames": []},
    "calibration": {"clear_field_s": 1.5,  # no people before this time; reference frame inside
                    "click_noise_px": 0.0, # sigma (square px) of simulated click errors
                    "emit_exact": None,    # also write <clip>.calib_exact.json; None = iff noise > 0
                    "points": "all"},      # all | required | [names]
}

PITCH_PARAM_KEYS = ("speed_mph", "cd", "extension_ft", "release_z_ft", "release_x_ft", "target_x_ft",
                    "target_z_ft", "magnus_x", "magnus_z", "catch_behind_ft", "wind_mph")


# ------------------------------------------------------------------ helpers
def deep_update(d: dict, u: dict) -> dict:
    """Recursively merge `u` into `d` (in place) and return `d`."""
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = copy.deepcopy(v)
    return d


def rng_for(seed: int, *keys: int) -> np.random.Generator:
    """Independent RNG stream per (seed, subsystem, ...) key."""
    return np.random.default_rng(np.random.SeedSequence([int(seed)] + [int(k) for k in keys]))


def parse_value(text: str) -> Any:
    """Parse a --set value: JSON if possible (numbers, lists, null, true), else the raw string."""
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        low = text.strip().lower()
        if low in ("none", "null"):
            return None
        if low in ("true", "false"):
            return low == "true"
        return text


def set_by_path(cfg: dict, path: str, value: Any) -> dict:
    """cfg['a']['b']['c'] = value for path 'a.b.c' (creates dicts as needed)."""
    keys = [k for k in path.split(".") if k]
    if not keys:
        raise ValueError(f"empty config path in {path!r}")
    node = cfg
    for k in keys[:-1]:
        if not isinstance(node.get(k), dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value
    return cfg


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    """Apply 'key.path=value' strings (CLI --set) to a (partial) config dict."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        set_by_path(cfg, k.strip(), parse_value(v.strip()))
    return cfg


def _check_choice(name: str, value: Any, choices: Iterable) -> None:
    if value not in tuple(choices):
        raise ValueError(f"config {name}={value!r}; expected one of {list(choices)}")


def resolve_config(config: Optional[dict] = None) -> dict:
    """Deep-merge `config` over the defaults, fill 'auto' values, validate."""
    cfg = deep_update(copy.deepcopy(DEFAULT_CONFIG), copy.deepcopy(config or {}))
    cfg["video"].pop("out", None)          # prototype key; the output path is an argument now
    lev = resolve_level(cfg["level"])
    V, cam, P = cfg["video"], cfg["camera"], cfg["pitches"]
    _check_choice("video.scan", V["scan"], SCANS)
    _check_choice("video.field_order", V["field_order"], ("tff", "bff"))
    _check_choice("video.metadata_field_order", V["metadata_field_order"], ("auto", "tff", "bff"))
    _check_choice("video.postprocess", V["postprocess"], POSTPROCESS)
    _check_choice("video.codec", V["codec"], CODECS)
    _check_choice("camera.view", cam["view"], ("side", "behind", "custom"))
    _check_choice("camera.side", cam["side"], ("1b", "3b"))
    _check_choice("tape.profile", cfg["tape"]["profile"], TAPE_PROFILES)
    if V["postprocess"] is not None and V["scan"] != "interlaced":
        raise ValueError("video.postprocess applies to interlaced renders only")
    if int(V["width"]) % 2 or int(V["height"]) % 4:
        raise ValueError("video.width must be even and video.height a multiple of 4")
    if float(cam.get("k1") or 0.0) != 0.0:
        raise ValueError("camera.k1 (lens distortion) is not modelled; leave it at 0")
    if cam["view"] == "custom" and (cam.get("C") is None or cam.get("look_at") is None):
        raise ValueError("camera.view='custom' needs camera.C and camera.look_at")
    auto = _PITCH_AUTO[_auto_key(lev)]
    for k, v in auto.items():
        if P.get(k) in (None, "auto"):
            P[k] = list(v)
    for k in PITCH_PARAM_KEYS:
        r = P[k]
        if not (isinstance(r, (list, tuple)) and len(r) == 2):
            raise ValueError(f"pitches.{k} must be [lo, hi], got {r!r}")
        P[k] = [float(r[0]), float(r[1])]
    clear = float(cfg["calibration"]["clear_field_s"])
    if P.get("schedule") is None and float(P["first_t"]) < clear + 0.5:
        raise ValueError(f"pitches.first_t ({P['first_t']}) must be >= calibration.clear_field_s + 0.5 "
                         f"({clear + 0.5}) so the reference frame shows a clear field")
    fence = cfg["fence"]
    if fence["enabled"] == "auto":
        fence["enabled"] = cam["view"] == "behind"
    pole = cfg["scene"]["pole"]
    if pole["enabled"] == "auto":
        pole["enabled"] = cam["view"] == "side"
    cfg["level_geom"] = dict(key=lev.key, distance_ft=lev.distance_ft, mound_h_ft=lev.mound_h_ft,
                             base_ft=lev.base_ft, sport=lev.sport)
    return cfg


def level_of(cfg: dict) -> LevelGeom:
    """LevelGeom of a resolved config."""
    g = cfg.get("level_geom")
    if g:
        return LevelGeom(g["key"], g["distance_ft"], g["mound_h_ft"], g["base_ft"], g.get("sport", "baseball"))
    return resolve_level(cfg["level"])


def par_of(cfg: dict) -> float:
    """True pixel aspect ratio (x_square = x * par; fy = fx * par)."""
    p = cfg["video"]["par"]
    return float(p[0]) / float(p[1]) if isinstance(p, (list, tuple)) else float(p)


def n_source_frames(cfg: dict) -> int:
    """Number of rendered (source) frames."""
    scan = cfg["video"]["scan"]
    per_s = 2 * FPS_NUM / FPS_DEN if scan == "progressive60" else FPS_NUM / FPS_DEN
    return int(math.floor(float(cfg["duration_s"]) * per_s + 1e-9))


def sample_period(cfg: dict) -> float:
    """Exposure spacing in seconds (field period, frame period for progressive30)."""
    return 2 * FIELD_DT if cfg["video"]["scan"] == "progressive30" else FIELD_DT
