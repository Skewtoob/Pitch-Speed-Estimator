"""Command line of the synthetic clip generator (`pitchspeed synth ...`).

    pitchspeed synth generate --out clip.mpg [--config c.json] [--seed 3] [--set camera.view=behind ...]
    pitchspeed synth suite --n 40 --seed 1 --out-dir suite/ [--jobs 4] [--variants]
    pitchspeed synth check clip.mpg [--rerender 5]          (or clip.mpg.gt.json)
    pitchspeed synth evaluate suite/                        (results in <clip stem>_pitchspeed/)
    pitchspeed synth evaluate results.json clip.mpg.gt.json
    pitchspeed synth defaults                               (print the default config)

`--set` values are parsed as JSON when possible (numbers, lists, true/false,
null), e.g. --set render.shutter_s=0.004 --set video.par=[10,11].
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional


def _progress_printer(label: str):
    state = {"t": 0.0}

    def cb(done: int, total: int) -> None:
        now = time.monotonic()
        if now - state["t"] < 1.0 and done < total:
            return
        state["t"] = now
        print(f"\r  {label}: {done}/{total} frames", end="" if done < total else "\n", flush=True)
    return cb


def cmd_generate(a) -> int:
    from pitchspeed.synth.config import apply_overrides
    from pitchspeed.synth.generate import generate_clip
    cfg = {}
    if a.config:
        with open(a.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    if a.seed is not None:
        cfg["seed"] = a.seed
    apply_overrides(cfg, a.set)
    t0 = time.perf_counter()
    gt = generate_clip(cfg, a.out, jobs=a.jobs, progress=None if a.quiet else _progress_printer("rendering"))
    v = gt["video"]
    print(f"wrote {v['path']} ({v['n_out_frames']} frames, {v['codec']}, {v['scan']}"
          f"{', ' + v['field_order'].upper() if v['field_order'] else ''}) in {time.perf_counter() - t0:.1f} s")
    print(f"      {v['path']}.gt.json, {gt['calibration']['file']}"
          + (f", {gt['calibration']['exact_file']}" if gt["calibration"].get("exact_file") else ""))
    for p in gt["pitches"]:
        print(f"  pitch {p['id']}: release t={p['t_release']:.3f} s  {p['speed_release_mph']:.1f} mph "
              f"(avg {p['avg_speed_y_mph']:.1f}, plate {p['speed_plate_front_mph']:.1f})  flight {p['flight_time']:.3f} s")
    for w in gt["warnings"]:
        print("  warning: " + w)
    return 0


def cmd_suite(a) -> int:
    from pitchspeed.synth.config import apply_overrides
    from pitchspeed.synth.suite import generate_suite
    base = {}
    if a.config:
        with open(a.config, "r", encoding="utf-8") as f:
            base = json.load(f)
    apply_overrides(base, a.set)
    if a.duration is not None:
        base["duration_s"] = a.duration
    rows = generate_suite(a.out_dir, n=a.n, seed=a.seed, jobs=a.jobs, base=base or None, variants=a.variants,
                          resume=not a.no_resume, progress=print)
    bad = [r for r in rows if r["status"] != "ok"]
    print(f"suite: {len(rows) - len(bad)}/{len(rows)} clips ok -> {os.path.join(a.out_dir, 'index.csv')}")
    return 1 if bad else 0


def cmd_check(a) -> int:
    from pitchspeed.synth.check import check_clip, format_check
    ok = True
    for p in a.clips:
        rep = check_clip(p, rerender=a.rerender)
        print(format_check(rep))
        ok &= rep["ok"]
    return 0 if ok else 1


def cmd_evaluate(a) -> int:
    import pitchspeed.synth.evaluate as ev
    if len(a.paths) == 1 and os.path.isdir(a.paths[0]):
        d = ev.evaluate_dir(a.paths[0])
        print(ev.format_dir_summary(d))
        if a.csv:
            ev.write_pitch_csv(d, a.csv)
            print(f"per-pitch rows: {a.csv}")
        return 0
    if len(a.paths) != 2:
        print("usage: evaluate <suite_dir>  |  evaluate <results.json> <clip.gt.json>", file=sys.stderr)
        return 2
    from pitchspeed.synth.gt import load_gt
    res = ev.load_result(a.paths[0])
    m = ev.evaluate(res, load_gt(a.paths[1]))
    print(ev.format_summary(m))
    return 0


def cmd_defaults(a) -> int:
    from pitchspeed.synth.config import DEFAULT_CONFIG
    print(json.dumps(DEFAULT_CONFIG, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pitchspeed synth", description="Synthetic test videos with ground truth.")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("generate", help="render one clip (+ .gt.json, .calib.json)")
    p.add_argument("--out", required=True, help="output video file (extension picks the container)")
    p.add_argument("--config", help="JSON file with a (partial) config")
    p.add_argument("--seed", type=int)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override, e.g. camera.view=behind")
    p.add_argument("--jobs", type=int, default=1, help="render processes (worth it for clips > ~15 s)")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_generate)
    p = sub.add_parser("suite", help="stratified set of clips + index.csv")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--duration", type=float, help="clip length in seconds (default 24)")
    p.add_argument("--config", help="JSON file merged over every clip's config")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--variants", action="store_true", help="also progressive / blend / field-discard / yadif clips")
    p.add_argument("--no-resume", action="store_true", help="regenerate clips that already exist")
    p.set_defaults(func=cmd_suite)
    p = sub.add_parser("check", help="decode a clip and verify it against its GT")
    p.add_argument("clips", nargs="+", help="video file(s) or .gt.json file(s)")
    p.add_argument("--rerender", type=int, default=0, help="also re-render N frames and compare (PSNR)")
    p.set_defaults(func=cmd_check)
    p = sub.add_parser("evaluate", help="score analysis results against GT (one clip or a suite folder)")
    p.add_argument("paths", nargs="+", help="<suite_dir> or <results.json> <clip.gt.json>")
    p.add_argument("--csv", help="(suite) write one row per pitch to this CSV file")
    p.set_defaults(func=cmd_evaluate)
    p = sub.add_parser("defaults", help="print the default generator config")
    p.set_defaults(func=cmd_defaults)
    return ap


def main(argv: Optional[list] = None) -> int:
    ap = build_parser()
    a = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    if not getattr(a, "func", None):
        ap.print_help()
        return 0
    try:
        return int(a.func(a) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except (ValueError, OSError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
