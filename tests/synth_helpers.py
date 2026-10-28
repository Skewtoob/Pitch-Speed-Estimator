"""Shared helpers for the synth tests (small, fast clip configs)."""
from __future__ import annotations

import copy

import numpy as np

from pitchspeed.synth.config import deep_update

# 360x240, 7 s, 2 pitches, everything on: renders in a few seconds.
SMALL = {
    "seed": 11,
    "duration_s": 7.0,
    "video": {"width": 360, "height": 240, "codec": "mpeg2", "bitrate_kbps": 1500, "debug_stamp": True,
              "field_order": "bff"},
    "pitches": {"count": 2, "first_t": 2.0, "gap_s": [0.8, 1.0]},
    "calibration": {"clear_field_s": 1.5},
}


def small(**over) -> dict:
    """SMALL deep-merged with keyword overrides (nested dicts allowed)."""
    return deep_update(copy.deepcopy(SMALL), copy.deepcopy(over))


def project_gt(gt: dict, R: np.ndarray, zoom: float, P) -> np.ndarray:
    """Independent pinhole projection from the GT camera block: (n, 2) frame px."""
    cam = gt["camera"]
    C = np.asarray(cam["C"], float)
    Xc = (np.asarray(R, float) @ (np.atleast_2d(np.asarray(P, float)) - C).T).T
    u = cam["fx"] * zoom * Xc[:, 0] / Xc[:, 2] + cam["cx"]
    v = cam["fy"] * zoom * Xc[:, 1] / Xc[:, 2] + cam["cy"]
    return np.stack([u, v], axis=1)
