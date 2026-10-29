"""Practice video: a computer-made tape with known pitch speeds, to mark and be scored on.

`make_practice_video` renders it with pitchspeed.synth (camera up behind the plate and off
to one side, like many home videos; the tape's two half-pictures blended into 30 frames per
second, like a typical capture), and writes the true release and catch times next to it.
`score` compares the pitches marked by hand with the truth: speed error per pitch and how many
frames early or late each release and catch mark was.

The figures are simple shapes, not people, so practice trains the timing judgment (when the
ball is out of the hand / in the glove), not reading a real pitcher's arm.
"""
from __future__ import annotations

import json
import math
import os
import statistics
from dataclasses import dataclass
from typing import Optional

PRACTICE_NAME = "practice.mp4"
SCORE_NAME = "practice_score.txt"
DEFAULT_COUNT = 8
GAP_S = 6.0


def practice_config(seed: int = 1, count: int = DEFAULT_COUNT) -> dict:
    import random
    rnd = random.Random(seed)
    first = 3.0
    sched = []
    for i in range(count):
        fast = rnd.random() < 0.7
        sched.append({"t_release": first + i * GAP_S,
                      "speed_mph": round(rnd.uniform(74, 88), 1) if fast else round(rnd.uniform(60, 72), 1)})
    return {
        "seed": int(seed), "duration_s": first + count * GAP_S + 1.0, "level": "hs",
        "video": {"scan": "interlaced", "postprocess": "blend", "codec": "h264", "bitrate_kbps": 4000},
        "camera": {"view": "behind", "lateral_ft": -9.0, "height_ft": 12.0, "back_ft": 30.0},
        "fence": {"enabled": False},
        "date_stamp": {"enabled": False},
        "distractors": {"birds": 0, "swing_prob": 0.0, "throwback": False},
        "pitches": {"schedule": sched},
    }


def answers_path(video: str) -> str:
    return video + ".gt.json"


def make_practice_video(out_dir: str, seed: int = 1, count: int = DEFAULT_COUNT, force: bool = False) -> str:
    """Path of the practice video in out_dir (made if missing or force)."""
    from pitchspeed.synth.generate import generate_clip
    os.makedirs(out_dir, exist_ok=True)
    video = os.path.join(out_dir, PRACTICE_NAME)
    if force or not (os.path.isfile(video) and os.path.isfile(answers_path(video))):
        generate_clip(practice_config(seed, count), video, write_cams=False)
    return video


@dataclass
class Row:
    number: int
    true_mph: float
    your_mph: Optional[float]
    release_frames: Optional[float]      # + = marked late
    catch_frames: Optional[float]

    @property
    def error(self) -> Optional[float]:
        return None if self.your_mph is None else self.your_mph - self.true_mph


def score(result, gt: dict) -> tuple:
    """(rows, missed true pitches, extra marked pitches) for a hand-marked result."""
    frame_dt = float(gt["video"].get("out_frame_dt") or 1001 / 30000)
    marked = [p for p in result.pitches if p.status != "rejected" and p.release is not None]
    used, rows = set(), []
    for i, tp in enumerate(gt["pitches"]):
        best = None
        for j, p in enumerate(marked):
            d = abs(p.release.t - tp["t_release"])
            if j not in used and d < 1.0 and (best is None or d < best[0]):
                best = (d, j)
        if best is None:
            rows.append(Row(i + 1, float(tp["speed_release_mph"]), None, None, None))
            continue
        used.add(best[1])
        p = marked[best[1]]
        v = p.speed.v_release_mph if p.speed is not None else None
        rel = (p.release.t - tp["t_release"]) / frame_dt
        cat = (p.catch.t - tp["t_catch"]) / frame_dt if p.catch is not None else None
        rows.append(Row(i + 1, float(tp["speed_release_mph"]), v, rel, cat))
    missed = sum(1 for r in rows if r.your_mph is None)
    return rows, missed, len(marked) - len(used)


def _lean(x: float, early: str, late: str) -> str:
    if abs(x) < 0.3:
        return "right on average"
    return f"about {abs(x):.1f} frame{'s' if abs(x) >= 1.5 else ''} {late if x > 0 else early} on average"


def format_score(rows, missed: int, extra: int) -> str:
    lines = ["Practice score (true speeds are known exactly)", "",
             f"{'Pitch':>5}  {'True':>6}  {'Yours':>6}  {'Error':>6}  {'Release':>8}  {'Catch':>7}"]
    for r in rows:
        if r.your_mph is None:
            lines.append(f"{r.number:>5}  {r.true_mph:6.1f}  {'-':>6}  {'-':>6}  {'not marked':>17}")
            continue
        f = lambda x: "-" if x is None else f"{x:+.1f}"     # noqa: E731
        lines.append(f"{r.number:>5}  {r.true_mph:6.1f}  {r.your_mph:6.1f}  {r.error:+6.1f}  {f(r.release_frames):>8}  "
                     f"{f(r.catch_frames):>7}")
    done = [r for r in rows if r.error is not None]
    lines.append("")
    lines.append("Release / Catch: how many frames after (+) or before (-) the true moment your mark was.")
    if done:
        errs = [r.error for r in done]
        rms = math.sqrt(sum(e * e for e in errs) / len(errs))
        lines += ["", f"Pitches scored: {len(done)} of {len(rows)}",
                  f"Average error: {statistics.mean(errs):+.1f} mph (+ = your speeds read high)",
                  f"Typical size of the error: {rms:.1f} mph per pitch"]
        rel = [r.release_frames for r in done if r.release_frames is not None]
        cat = [r.catch_frames for r in done if r.catch_frames is not None]
        if rel:
            lines.append(f"Your release marks: {_lean(statistics.mean(rel), 'early', 'late')}")
        if cat:
            lines.append(f"Your catch marks: {_lean(statistics.mean(cat), 'early', 'late')}")
        if rel and cat:
            lines += ["", "Late releases and early catches make speeds read high; early releases and late",
                      "catches make them read low. On 30-frames-per-second video, one frame at either end",
                      "is worth about 4-6 mph."]
    if missed:
        lines.append(f"\n{missed} pitch(es) not marked.")
    if extra:
        lines.append(f"{extra} marked pitch(es) did not match a real pitch (marked twice, or a release mark far off).")
    lines += ["", "The practice figures are simple shapes, not real players: this trains the timing",
              "judgment, not reading a real pitcher's arm. Real tapes are harder."]
    return "\n".join(lines)


def score_folder(video: str) -> str:
    """Score the hand-marked results of a practice video; writes and returns the text."""
    from pitchspeed import pipeline as P
    from pitchspeed.render.report import load_json
    out = P.default_out_dir(video)
    res = load_json(P.out_paths(out)["results_json"])
    with open(answers_path(video), encoding="utf-8") as fh:
        gt = json.load(fh)
    text = format_score(*score(res, gt))
    with open(os.path.join(out, SCORE_NAME), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text
