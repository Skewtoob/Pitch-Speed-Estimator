"""Command-line interface: `pitchspeed <command> ...`.

Typical use (Windows: drag a video onto pitchspeed.bat, which runs `run`):

    pitchspeed run game.avi            # calibrate (first time) -> analyze -> review -> render
    pitchspeed calibrate game.avi      # click the rubber/plate (or release/catch)
    pitchspeed analyze game.avi        # find pitches, estimate speeds
    pitchspeed review game.avi         # check / fix / add pitches by hand
    pitchspeed render game.avi         # write the annotated video
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

from pitchspeed import __version__
from pitchspeed import constants as C
from pitchspeed.pipeline import AnalyzeOptions, default_out_dir, out_paths


# --------------------------------------------------------------- helpers ----
def _ask(prompt: str, default: Optional[str] = None) -> Optional[str]:
    try:
        ans = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
    except EOFError:
        return default
    return ans or default


def _ask_view(default: Optional[str]) -> str:
    if default:
        return default
    print("\nWhere was the camera?")
    print("  1) side     - from the side (dugout, fence, along a baseline); the ball crosses the screen")
    print("  2) behind   - behind home plate / the backstop, looking out at the pitcher")
    while True:
        a = (_ask("Choose 1 or 2", "1") or "1").lower()
        if a in ("1", "s", "side"):
            return "side"
        if a in ("2", "b", "behind"):
            return "behind"


def _ask_level(args) -> tuple[Optional[str], float, str]:
    """Return (level key, distance_ft, sport) from flags or a console prompt."""
    if args.distance is not None:
        sport = "softball" if (args.level and args.level.startswith("sb")) else "baseball"
        return (args.level, float(args.distance), sport)
    if args.level:
        lv = C.get_level(args.level)
        return (lv.key, lv.distance_ft, lv.sport)
    print("\nPitching distance (rubber to the back point of home plate) - check your league's rules:")
    keys = list(C.LEVELS)
    for i, k in enumerate(keys, 1):
        print(f"  {i}) {C.LEVELS[k].label}")
    print(f"  {len(keys) + 1}) other distance (type it in feet)")
    while True:
        a = _ask(f"Choose 1-{len(keys) + 1}", "1") or "1"
        try:
            i = int(a)
        except ValueError:
            continue
        if 1 <= i <= len(keys):
            lv = C.LEVELS[keys[i - 1]]
            return (lv.key, lv.distance_ft, lv.sport)
        if i == len(keys) + 1:
            d = _ask("Distance in feet (e.g. 60.5)")
            try:
                return (None, float(d), "baseball")
            except (TypeError, ValueError):
                continue


def _settings_from_args(args, base=None):
    from pitchspeed.calib.model import Settings
    st = base or Settings()
    level, dist, sport = _ask_level(args) if base is None or args.level or args.distance else (st.level, st.distance_ft, st.sport)
    st.level, st.distance_ft, st.sport = level, dist, sport
    if getattr(args, "extension", None) is not None:
        st.extension_ft = args.extension
    if getattr(args, "throws", None):
        st.pitcher_throws = args.throws.upper()
    if getattr(args, "altitude", None) is not None:
        st.altitude_ft = args.altitude
    if getattr(args, "temp", None) is not None:
        st.temp_f = args.temp
    return st


def _open_source(video: str, field_order: Optional[str] = None, par: Optional[float] = None):
    from pitchspeed.io.reader import VideoReader
    from pitchspeed.io.interlace import SampleSource, detect_scan
    reader = VideoReader(video, sar_override=par)
    fo = None if field_order in (None, "auto") else field_order
    scan = detect_scan(reader, override=fo)
    return reader, scan, SampleSource(reader, scan)


def _load_calib(args, out_dir: str):
    from pitchspeed.calib.model import CalibrationFile
    path = getattr(args, "calib", None) or out_paths(out_dir)["calibration"]
    if not os.path.exists(path):
        return None, path
    return CalibrationFile.load(path), path


def _check_video(path: str) -> None:
    if not os.path.isfile(path):
        raise SystemExit(f"error: video not found: {path}")


# -------------------------------------------------------------- commands ----
def cmd_probe(args) -> int:
    _check_video(args.video)
    reader, scan, source = _open_source(args.video, args.field_order, args.par)
    i = reader.info
    print(f"file        : {i.path}")
    print(f"container   : {i.container}   codec: {i.codec}")
    print(f"size        : {i.width}x{i.height}   pixel aspect (x_square = x * sar): {i.sar:.4f} ({i.sar_source})")
    print(f"frames      : {i.n_frames} @ {i.frame_rate_str} ({i.frame_rate:.3f} fps), {i.duration_s:.2f} s, timing {i.timing_mode}")
    print(f"audio       : {'yes' if i.has_audio else 'no'}")
    print(f"scan        : {scan.mode}" + (f", {scan.field_order.upper()}" if scan.field_order else "")
          + f"  (confidence {scan.confidence}, from {scan.source})")
    print(f"samples     : {source.n_samples} @ {1 / source.sample_dt:.3f}/s")
    if i.dropped_frames:
        print(f"dropped     : {len(i.dropped_frames)} gaps, first at frames {i.dropped_frames[:10]}")
    if i.duplicate_frames:
        print(f"duplicates  : {len(i.duplicate_frames)} frames")
    reader.close()
    return 0


def cmd_calibrate(args) -> int:
    from pitchspeed.calib.ui import run_calibration_ui
    _check_video(args.video)
    out_dir = args.out or default_out_dir(args.video)
    os.makedirs(out_dir, exist_ok=True)
    existing, path = _load_calib(args, out_dir)
    view = _ask_view(args.view or (existing.segments[0].view if existing and existing.segments else None))
    settings = _settings_from_args(args, base=existing.settings if existing else None)
    reader, scan, source = _open_source(args.video, args.field_order, args.par)
    settings.sar = reader.info.sar
    print(f"\nOpening the calibration window ({view} view). Press H in the window for help.")
    cf = run_calibration_ui(source, reader.info, view, initial=existing, settings=settings,
                            start_frame=args.frame or 0)
    reader.close()
    if cf is None:
        print("Calibration cancelled; nothing saved.")
        return 1
    cf.video = {"name": os.path.basename(args.video), "size": os.path.getsize(args.video),
                "n_frames": reader.info.n_frames, "width": reader.info.width, "height": reader.info.height}
    cf.save(path)
    print(f"Saved calibration: {path}")
    for w in cf.validate():
        print(f"  note: {w}")
    return 0


def _analyze(args, out_dir: str):
    from pitchspeed.pipeline import analyze
    from pitchspeed.render.report import format_table
    cf, path = _load_calib(args, out_dir)
    if cf is None:
        raise SystemExit(f"error: no calibration found at {path}\n"
                         f"       run:  pitchspeed calibrate \"{args.video}\"")
    opts = AnalyzeOptions(field_order=args.field_order, par=args.par, start_s=args.start,
                          end_s=args.end, debug=args.debug)
    result, source = analyze(args.video, cf, out_dir, opts)
    print()
    print(format_table(result))
    p = out_paths(out_dir)
    print(f"\nResults: {p['results_csv']}\n         {p['results_json']}")
    return result, source, cf


def _render(result, source, out_dir: str, mode: str, audio: bool) -> Optional[str]:
    from pitchspeed.pipeline import load_transforms
    from pitchspeed.render.overlay import render_annotated
    tf, ts = load_transforms(out_dir)
    out = out_paths(out_dir)["annotated"]
    print(f"\nWriting annotated video ({mode}) ...")
    out = render_annotated(result, source, out, mode=mode, transforms=tf, transform_samples=ts, audio=audio)
    print(f"Annotated video: {out}")
    return out


def cmd_analyze(args) -> int:
    _check_video(args.video)
    out_dir = args.out or default_out_dir(args.video)
    os.makedirs(out_dir, exist_ok=True)
    result, source, _ = _analyze(args, out_dir)
    if args.render != "none":
        _render(result, source, out_dir, args.render, not args.no_audio)
    return 0


def _review(args, out_dir: str, source=None):
    from pitchspeed.pipeline import load_transforms, make_recompute, renumber
    from pitchspeed.render.report import load_json, write_csv, write_json
    from pitchspeed.review.ui import run_review_ui
    p = out_paths(out_dir)
    if not os.path.exists(p["results_json"]):
        raise SystemExit(f"error: no results at {p['results_json']}; run analyze first")
    result = load_json(p["results_json"])
    cf, _ = _load_calib(args, out_dir)
    if cf is None:
        raise SystemExit("error: calibration missing")
    if source is None:
        _, _, source = _open_source(args.video, getattr(args, "field_order", None), getattr(args, "par", None))
    tf, ts = load_transforms(out_dir)

    def save(res):
        renumber(res.pitches)
        write_json(res, p["results_json"])
        write_csv(res, p["results_csv"])

    print("\nOpening the review window. Press H in the window for help.")
    result = run_review_ui(result, source, cf, make_recompute(cf, result.scan, source), save,
                           transforms=tf, transform_samples=ts)
    save(result)
    return result, source


def cmd_review(args) -> int:
    from pitchspeed.render.report import format_table
    _check_video(args.video)
    out_dir = args.out or default_out_dir(args.video)
    result, source = _review(args, out_dir)
    print(format_table(result))
    if args.render != "none":
        _render(result, source, out_dir, args.render, not args.no_audio)
    return 0


def cmd_render(args) -> int:
    from pitchspeed.render.report import load_json
    _check_video(args.video)
    out_dir = args.out or default_out_dir(args.video)
    p = out_paths(out_dir)
    if not os.path.exists(p["results_json"]):
        raise SystemExit(f"error: no results at {p['results_json']}; run analyze first")
    result = load_json(p["results_json"])
    _, _, source = _open_source(args.video, result.scan.field_order if result.scan.mode == "interlaced"
                                else "progressive", args.par)
    _render(result, source, out_dir, args.mode, not args.no_audio)
    return 0


def cmd_run(args) -> int:
    _check_video(args.video)
    out_dir = args.out or default_out_dir(args.video)
    os.makedirs(out_dir, exist_ok=True)
    cf, _ = _load_calib(args, out_dir)
    if cf is None:
        print("Step 1/4: calibration (click a few points once for this video)")
        args.frame = 0
        if cmd_calibrate(args) != 0:
            return 1
    print("\nStep 2/4: finding pitches and estimating speeds")
    result, source, _ = _analyze(args, out_dir)
    a = (_ask("\nStep 3/4: open the review window to check/fix pitches? (y/n)", "y") or "y").lower()
    if a.startswith("y"):
        result, source = _review(args, out_dir, source)
    a = (_ask("\nStep 4/4: write the annotated video? (full / highlights / none)", args.render) or "full").lower()
    if a.startswith("f") or a.startswith("h"):
        _render(result, source, out_dir, "highlights" if a.startswith("h") else "full", not args.no_audio)
    print("\nDone.")
    return 0


def cmd_evaluate(args) -> int:
    import json
    from pitchspeed.render.report import load_json
    from pitchspeed.synth.evaluate import evaluate, format_summary
    res = load_json(args.results)
    with open(args.gt, "r", encoding="utf-8") as f:
        gt = json.load(f)
    m = evaluate(res, gt)
    print(format_summary(m))
    return 0


def cmd_keys(args) -> int:
    from pitchspeed.ui.keys import show_key_codes
    show_key_codes()
    return 0


# ------------------------------------------------------------------ main ----
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pitchspeed",
                                 description="Estimate baseball pitch speeds from video.")
    ap.add_argument("--version", action="version", version=f"pitchspeed {__version__}")
    ap.add_argument("-v", "--verbose", action="store_true", help="more log output")
    sub = ap.add_subparsers(dest="cmd")

    def common(p, calib=True):
        p.add_argument("video", help="video file")
        p.add_argument("--out", help="output folder (default: <video name>_pitchspeed next to the video)")
        p.add_argument("--field-order", choices=["auto", "tff", "bff", "progressive"], default="auto",
                       help="override interlace detection (default: auto)")
        p.add_argument("--par", type=float, help="override pixel aspect ratio (x_square = x * par)")
        if calib:
            p.add_argument("--calib", help="calibration file (default: <out>/calibration.json)")

    def settings_args(p):
        p.add_argument("--view", choices=["side", "behind"], help="camera position")
        p.add_argument("--level", help=f"league preset: {', '.join(C.LEVELS)}")
        p.add_argument("--distance", type=float, help="rubber-to-plate distance in feet (overrides --level)")
        p.add_argument("--extension", type=float,
                       help="pitcher's release point in front of the rubber, feet (default: assumed by level)")
        p.add_argument("--throws", choices=["R", "L", "r", "l"], help="pitcher's throwing hand (side view)")
        p.add_argument("--altitude", type=float, help="field altitude in feet (air density)")
        p.add_argument("--temp", type=float, help="air temperature in F (air density)")

    def analyze_args(p):
        p.add_argument("--start", type=float, help="only analyze from this time (seconds)")
        p.add_argument("--end", type=float, help="only analyze up to this time (seconds)")
        p.add_argument("--debug", action="store_true", help="write debug images to <out>/debug")
        p.add_argument("--render", choices=["full", "highlights", "none"], default="full",
                       help="annotated video to write (default: full)")
        p.add_argument("--no-audio", action="store_true", help="do not copy audio into the annotated video")

    p = sub.add_parser("probe", help="show video format, interlacing and field order")
    common(p, calib=False)
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("calibrate", help="click the field landmarks for a video")
    common(p)
    settings_args(p)
    p.add_argument("--frame", type=int, help="start at this frame")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("analyze", help="find pitches and estimate speeds")
    common(p)
    analyze_args(p)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("review", help="check, fix, reject or add pitches by hand")
    common(p)
    p.add_argument("--render", choices=["full", "highlights", "none"], default="none",
                   help="re-render the annotated video afterwards")
    p.add_argument("--no-audio", action="store_true")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("render", help="write the annotated video from saved results")
    common(p, calib=False)
    p.add_argument("--mode", choices=["full", "highlights"], default="full")
    p.add_argument("--no-audio", action="store_true")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("run", help="guided: calibrate (if needed) -> analyze -> review -> render")
    common(p)
    settings_args(p)
    analyze_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("evaluate", help="(testing) compare results.json with a synthetic clip's ground truth")
    p.add_argument("results")
    p.add_argument("gt")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("keys", help="show raw key codes (troubleshooting the windows)")
    p.set_defaults(func=cmd_keys)

    sub.add_parser("synth", help="(testing) generate synthetic test videos; see `pitchspeed synth -h`")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "synth":
        from pitchspeed.synth.cli import main as synth_main
        return int(synth_main(argv[1:]) or 0)
    ap = build_parser()
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if not getattr(args, "func", None):
        ap.print_help()
        return 0
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
