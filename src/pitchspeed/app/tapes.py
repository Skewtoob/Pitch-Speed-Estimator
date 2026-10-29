"""A tape's state, read from its results folder (<video>_pitchspeed next to the video)."""
from __future__ import annotations

import os
import statistics
from dataclasses import dataclass
from typing import Optional


@dataclass
class TapeInfo:
    video: str
    exists: bool
    status: str                     # 'Not started', 'Marked by hand', 'Analyzed', 'Missing file', ...
    pitches: int = 0
    typical: Optional[float] = None  # median speed of the pitches with a speed
    fastest: Optional[float] = None
    has_clips: bool = False
    out_dir: str = ""

    @property
    def name(self) -> str:
        return os.path.basename(self.video)


def tape_info(video: str) -> TapeInfo:
    from pitchspeed import pipeline as P
    out_dir = P.default_out_dir(video)
    if not os.path.isfile(video):
        return TapeInfo(video, False, "File not found", out_dir=out_dir)
    paths = P.out_paths(out_dir)
    if not os.path.isfile(paths["results_json"]):
        status = "Calibrated" if os.path.isfile(paths["calibration"]) else "Not started"
        return TapeInfo(video, True, status, out_dir=out_dir)
    try:
        from pitchspeed.render.report import load_json
        r = load_json(paths["results_json"])
    except Exception:                                   # noqa: BLE001 - a broken file must not stop the list
        return TapeInfo(video, True, "Results file unreadable", out_dir=out_dir)
    ok = [p for p in r.pitches if p.status != "rejected"]
    v = [p.speed.v_release_mph for p in ok if p.speed is not None and p.speed.v_release_mph is not None]
    status = "Marked by hand" if P.is_marking_session(r) else "Analyzed"
    clips = os.path.isfile(os.path.join(out_dir, "clips", "clips.json"))
    if clips:
        status += ", clips made"
    return TapeInfo(video, True, status, len(ok), statistics.median(v) if v else None,
                    max(v) if v else None, clips, out_dir)
