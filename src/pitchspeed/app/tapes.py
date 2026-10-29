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
    fastball: Optional[float] = None    # average of the fast group (see fastball_average)
    fastball_pm: Optional[float] = None  # 90 % margin of that average
    fastball_n: int = 0

    @property
    def name(self) -> str:
        return os.path.basename(self.video)


def fastball_average(speeds) -> tuple:
    """(average, 90 % margin, n) of the fastballs among `speeds` (mph).

    When the speeds fall into a slow and a fast group (the best two-group split leaves the
    group averages at least 8 mph apart, the fast group with >= 3 pitches and >= 25 % of all),
    the fast group is taken as the fastballs; otherwise all pitches.  The margin is
    1.645 x standard error of the average (the scatter of single pitches already includes
    the marking error); None with fewer than 3 pitches."""
    v = sorted(float(x) for x in speeds if x is not None)
    if not v:
        return None, None, 0
    best, cut = None, 0
    for i in range(1, len(v)):
        lo, hi = v[:i], v[i:]
        cost = sum((x - statistics.mean(lo)) ** 2 for x in lo) + sum((x - statistics.mean(hi)) ** 2 for x in hi)
        if best is None or cost < best:
            best, cut = cost, i
    fast = v
    if cut:
        lo, hi = v[:cut], v[cut:]
        if statistics.mean(hi) - statistics.mean(lo) >= 8 and len(hi) >= 3 and len(hi) >= 0.25 * len(v):
            fast = hi
    m = statistics.mean(fast)
    pm = 1.645 * statistics.stdev(fast) / len(fast) ** 0.5 if len(fast) >= 3 else None
    return m, pm, len(fast)


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
    fb, pm, n = fastball_average(v)
    return TapeInfo(video, True, status, len(ok), statistics.median(v) if v else None,
                    max(v) if v else None, clips, out_dir, fb, pm, n)
