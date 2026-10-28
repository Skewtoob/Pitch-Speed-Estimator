"""Command-line interface: `pitchspeed <command> ...`.

Typical use (Windows: drag a video onto pitchspeed.bat, which runs `run`):

    pitchspeed run game.avi            # calibrate (first time) -> analyze -> review -> render
    pitchspeed calibrate game.avi      # click the rubber/plate (or release/catch)
    pitchspeed analyze game.avi        # find pitches, estimate speeds
    pitchspeed review game.avi         # check / fix / add pitches by hand
    pitchspeed render game.avi         # write the annotated video (--mode clips: one per pitch)
    pitchspeed mark game.avi           # mark the pitches by hand (any camera angle) -> one clip each

Running `run` again on a video that already has results asks whether to
keep them (default; straight to review) or to analyze again.  Analyzing
again never loses review work: results.json is backed up first and the
reviewed pitches are carried over when possible (see pipeline.py).
Settings flags (--level, --distance, --extension, ...) given to `run` or
`analyze` update the saved calibration settings, and the change is printed.
`review` and `render` reuse the scan (field order / frame mode) stored in
results.json, so every saved field number keeps its meaning.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import logging
import os
import sys
import time
from typing import Optional

from pitchspeed import __version__
from pitchspeed import constants as C
from pitchspeed.pipeline import AnalyzeOptions, out_paths

SPORTS = ("baseball", "softball")
SETTINGS_FLAGS = ("level", "distance", "sport", "extension", "throws", "altitude", "temp")
try:                                   # plausible range of a typed pitching distance (feet)
    from pitchspeed.calib.model import DISTANCE_RANGE_FT
except ImportError:                    # pragma: no cover - older calib.model
    DISTANCE_RANGE_FT = (20.0, 100.0)


# --------------------------------------------------------------- helpers ----
def _ask(prompt: str, default: Optional[str] = None) -> Optional[str]:
    try:
        ans = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
    except EOFError:
        return default
    return ans or default


def _ask_enter(prompt: str) -> bool:
    """True when the user just presses Enter (retry); False for 's...' or no console input."""
    try:
        ans = input(prompt).strip().lower()
    except EOFError:
        return False
    return not ans.startswith("s")


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


def _ask_sport(default: str = "baseball") -> str:
    while True:
        a = (_ask("Baseball or softball? (b/s)", "b" if default == "baseball" else "s") or "").lower()
        if a.startswith("b"):
            return "baseball"
        if a.startswith("s"):
            return "softball"


def _level(args) -> C.Level:
    try:
        lv = C.get_level(args.level)
    except KeyError as e:
        raise SystemExit(f"error: {e.args[0]}")
    sport = getattr(args, "sport", None)
    if sport and sport != lv.sport:
        raise SystemExit(f"error: --level {lv.key} is {lv.sport}; it does not go with --sport {sport}")
    return lv


def _ask_level(args, base=None) -> tuple[Optional[str], float, str]:
    """Return (level key, distance_ft, sport) from flags or a console prompt.

    The sport (it sets the drag and the assumed release point) comes from the
    level, else --sport, else the saved calibration, else a question."""
    sport_flag = getattr(args, "sport", None)
    if args.distance is not None:
        if args.level:
            return (_level(args).key, float(args.distance), _level(args).sport)
        sport = sport_flag or (base.sport if base is not None else _ask_sport())
        return (None, float(args.distance), sport)
    if args.level:
        lv = _level(args)
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
            while True:
                d = _ask("Distance in feet (e.g. 60.5)")
                if d is None:                  # no answer (no console): back to the list
                    break
                try:
                    dist = float(d)
                except ValueError:
                    print("  type the distance as a number of feet")
                    continue
                if not DISTANCE_RANGE_FT[0] <= dist <= DISTANCE_RANGE_FT[1]:
                    print(f"  the distance must be between {DISTANCE_RANGE_FT[0]:g} and "
                          f"{DISTANCE_RANGE_FT[1]:g} ft")
                    continue
                sport = sport_flag or _ask_sport(base.sport if base is not None else "baseball")
                return (None, dist, sport)


def _check_settings(st, args) -> None:
    """Refuse impossible values the user just typed as flags (e.g. an extension in inches)."""
    try:
        from pitchspeed.calib.model import settings_errors
    except ImportError:                # pragma: no cover - older calib.model
        return
    errs = settings_errors(st)
    if errs and _settings_flags_given(args):
        raise SystemExit("error: " + "\n       ".join(errs))


def _settings_from_args(args, base=None):
    from pitchspeed.calib.model import Settings
    st = base or Settings()
    if base is None or args.level or args.distance is not None:
        st.level, st.distance_ft, st.sport = _ask_level(args, base)
    elif getattr(args, "sport", None):
        st.sport = args.sport
    if getattr(args, "extension", None) is not None:
        st.extension_ft = args.extension
    if getattr(args, "throws", None):
        st.pitcher_throws = args.throws.upper()
    if getattr(args, "altitude", None) is not None:
        st.altitude_ft = args.altitude
    if getattr(args, "temp", None) is not None:
        st.temp_f = args.temp
    _check_settings(st, args)
    return st


def _settings_flags_given(args) -> list[str]:
    return [f"--{k}" for k in SETTINGS_FLAGS if getattr(args, k, None) is not None]


def _check_view_flag(args, cf) -> None:
    v = getattr(args, "view", None)
    if v and cf.segments and any(seg.view != v for seg in cf.segments):
        raise SystemExit(f"error: the saved calibration is for the {cf.segments[0].view} view; a different "
                         f"camera view needs new clicks:\n       pitchspeed calibrate \"{args.video}\" --view {v}")


def _apply_settings_flags(args, cf, path: str) -> list[str]:
    """Settings flags given to run/analyze -> update and save calibration.json; returns the changes."""
    from pitchspeed.pipeline import settings_changes
    _check_view_flag(args, cf)
    if not _settings_flags_given(args):
        return []
    old = copy.deepcopy(cf.settings)
    new = _settings_from_args(args, base=copy.deepcopy(cf.settings))
    new.sar = old.sar
    changes = settings_changes(old, new)
    if changes:
        cf.settings = new
        cf.save(path)
        print("Updated the saved settings (calibration.json): " + "; ".join(changes))
    return changes


def _open_source(video: str, field_order: Optional[str] = None, par: Optional[float] = None):
    from pitchspeed.io.reader import VideoReader
    from pitchspeed.io.interlace import SampleSource, detect_scan
    reader = VideoReader(video, sar_override=par)
    fo = None if field_order in (None, "auto") else field_order
    scan = detect_scan(reader, override=fo)
    return reader, scan, SampleSource(reader, scan)


def _out_dir(args) -> str:
    """--out, else the default folder next to the video (resolved once per command)."""
    if getattr(args, "out", None):
        return args.out
    from pitchspeed.pipeline import resolve_out_dir
    d, note = resolve_out_dir(args.video)
    if note:
        print(f"note: {note}")
    args.out = d
    return d


def _load_calib(args, out_dir: str):
    from pitchspeed.calib.model import CalibrationFile
    path = getattr(args, "calib", None) or out_paths(out_dir)["calibration"]
    if not os.path.exists(path):
        return None, path
    return CalibrationFile.load(path), path


def _check_video(path: str) -> None:
    if not os.path.isfile(path):
        raise SystemExit(f"error: video not found: {path}")


def _video_info(video: str, out_dir: str, par: Optional[float] = None):
    from pitchspeed.io.reader import VideoReader
    reader = VideoReader(video, sar_override=par, index_cache=out_paths(out_dir)["index"])
    try:
        return reader.info
    finally:
        reader.close()


def _check_calibration(cf, video: str, info, interactive: bool) -> bool:
    """Compare the calibration's video fingerprint with `video`.  Stops on an
    impossible match; warns otherwise.  Interactive: asks before using a
    calibration that probably belongs to another file (False = declined)."""
    from pitchspeed.pipeline import FILE_CHANGED_WARNING, calibration_video_problems
    errors, warns = calibration_video_problems(cf, video, info)
    if errors:
        raise SystemExit("error: the saved calibration does not fit this video:\n"
                         + "".join(f"       - {e}\n" for e in errors)
                         + f"       calibrate it:  pitchspeed calibrate \"{video}\"\n"
                         + "       (or use --out FOLDER to keep it apart from the other video's results)")
    if not warns:
        return True
    print("\nWARNING: the saved calibration may not belong to this video:")
    for w in warns:
        print(f"  - {w}")
    serious = [w for w in warns if w != FILE_CHANGED_WARNING]
    if interactive and serious:
        return (_ask("Use this calibration anyway? (y/n)", "n") or "n").lower().startswith("y")
    if serious:
        print("  (continuing; check the clicks with `pitchspeed calibrate` if the speeds look wrong)")
    return True


def _csv_failed(result) -> bool:
    from pitchspeed.pipeline import CSV_WARNING_PREFIX
    return any(str(w).startswith(CSV_WARNING_PREFIX) for w in result.warnings or [])


def _retry_csv(result, out_dir: str) -> bool:
    """results.csv could not be replaced (open in Excel?): let the user close it and retry."""
    from pitchspeed.pipeline import save_results
    for _ in range(20):
        if not _csv_failed(result):
            print("results.csv written.")
            return True
        print("\n!! results.csv is open in another program (Excel?) and was NOT updated "
              "(results.json was saved).")
        if not _ask_enter("   Close it, then press Enter to try again (or type s and Enter to skip): "):
            print("   results.csv still shows older results; it is written again the next time you "
                  "quit `pitchspeed review`.")
            return False
        save_results(result, out_dir)
    return not _csv_failed(result)


def _load_results(out_dir: str):
    from pitchspeed.render.report import load_json
    p = out_paths(out_dir)["results_json"]
    if not os.path.exists(p):
        raise SystemExit(f"error: no results at {p}; run analyze first")
    try:
        return load_json(p)
    except (OSError, ValueError) as e:
        raise SystemExit(f"error: {e}\n       run `pitchspeed analyze` to make new results "
                         f"(a copy of the damaged file is kept)")


def _source_for_results(args, out_dir: str, result, check_field_order: bool = True):
    """The SampleSource the results were made with (saved scan, no re-detection)."""
    from pitchspeed.pipeline import source_from_result
    sc = result.scan
    stored = sc.field_order if sc.mode == "interlaced" else "progressive"
    fo = getattr(args, "field_order", None)
    if check_field_order and fo not in (None, "auto") and fo != stored:
        raise SystemExit(f"error: these results were made with field order '{stored}'. To use '{fo}', "
                         f"analyze again (your review edits are backed up):\n"
                         f"       pitchspeed analyze \"{args.video}\" --field-order {fo}")
    reader, source = source_from_result(args.video, result, par=getattr(args, "par", None),
                                        index_cache=out_paths(out_dir)["index"])
    rv = result.video
    if int(reader.info.n_frames) != int(rv.n_frames):
        n = reader.info.n_frames
        reader.close()
        raise SystemExit(f"error: results.json was made from a video with {rv.n_frames} frames "
                         f"({os.path.basename(rv.path or '?')}); this one has {n}. The saved pitch "
                         f"positions do not fit it: analyze this video again (review edits are backed up).")
    if rv.path and os.path.basename(rv.path) != os.path.basename(args.video):
        print(f"note: results.json was made from '{os.path.basename(rv.path)}' "
              f"(same number of frames as '{os.path.basename(args.video)}')")
    return source


def _calibration_for_results(args, out_dir: str, result, source):
    """The calibration to review / recompute `result` with.

    Settings changed since the analysis (e.g. --distance given to run, or
    `calibrate` re-run): all speeds are recomputed with the new settings,
    edits kept.  Clicks changed: the results' own calibration is used (the
    tracks belong to it) and the user is told to analyze again."""
    from pitchspeed.calib.model import CalibrationFile
    from pitchspeed.pipeline import calibration_differences, recompute_all, save_results
    cf_now, _ = _load_calib(args, out_dir)
    stored = CalibrationFile.from_dict(copy.deepcopy(result.calibration)) if result.calibration else None
    if cf_now is None:
        if stored is None:
            raise SystemExit("error: calibration missing")
        print("note: calibration.json is missing; using the calibration saved with the results")
        return stored
    if stored is None:
        return cf_now
    changes, clicks_changed = calibration_differences(cf_now, result)
    use = cf_now
    if clicks_changed:
        print("\nnote: the calibration was clicked again after these results were made. They are shown "
              "with the clicks they were made with; to use the new clicks, analyze again "
              "(`pitchspeed analyze`, your review edits are backed up and carried over).")
        use = stored
        if changes:
            use.settings = copy.deepcopy(cf_now.settings)
    if changes:
        print("Settings changed since these results were made: " + "; ".join(changes))
        n = recompute_all(result, use, source)
        save_results(result, out_dir)
        print(f"  the speeds of all {n} pitch(es) were recomputed with them (your review edits are kept)")
    return use


def _final_save(result, out_dir: str) -> None:
    """Save after the review window closed; never loses the edits."""
    from pitchspeed.pipeline import renumber, save_results
    from pitchspeed.render.report import write_json
    renumber(result.pitches)
    try:
        save_results(result, out_dir)
    except OSError as e:
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        alt = os.path.join(out_dir, f"results.unsaved-{stamp}.json")
        write_json(result, alt)
        print(f"\n!! results.json could not be written ({e}).\n   Your review was saved as {alt}.\n"
              f"   Close the program that has results.json open, then rename that file to results.json.")
        return
    if _csv_failed(result):
        _retry_csv(result, out_dir)


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
    from pitchspeed.pipeline import video_fingerprint
    _check_video(args.video)
    out_dir = _out_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    existing, path = _load_calib(args, out_dir)
    view = _ask_view(args.view or (existing.segments[0].view if existing and existing.segments else None))
    settings = _settings_from_args(args, base=existing.settings if existing else None)
    if view == "side" and settings.pitcher_throws is None:
        a = (_ask("Pitcher throws right or left? (R/L, Enter = not sure)", "") or "").strip().upper()
        if a in ("R", "L"):
            settings.pitcher_throws = a
    reader, scan, source = _open_source(args.video, args.field_order, args.par)
    settings.sar = reader.info.sar
    print(f"\nOpening the calibration window ({view} view). Press H in the window for help.")
    cf = run_calibration_ui(source, reader.info, view, initial=existing, settings=settings,
                            start_frame=args.frame or 0, ui_scale=getattr(args, "ui_scale", None))
    reader.close()
    if cf is None:
        print("Calibration cancelled; nothing saved.")
        return 1
    cf.video = video_fingerprint(args.video, reader.info)
    cf.save(path)
    print(f"Saved calibration: {path}")
    for w in cf.validate():
        print(f"  note: {w}")
    if existing is not None and os.path.exists(out_paths(out_dir)["results_json"]):
        print("  note: the saved results were made with the previous calibration; run `pitchspeed analyze` "
              "to use the new one (review edits are backed up and carried over)")
    return 0


def _analyze(args, out_dir: str, cf=None):
    from pitchspeed.pipeline import analyze
    from pitchspeed.render.report import format_table
    if cf is None:
        cf, path = _load_calib(args, out_dir)
        if cf is None:
            raise SystemExit(f"error: no calibration found at {path}\n"
                             f"       run:  pitchspeed calibrate \"{args.video}\"")
        _apply_settings_flags(args, cf, path)
        _check_calibration(cf, args.video, _video_info(args.video, out_dir, args.par), interactive=False)
    opts = AnalyzeOptions(field_order=args.field_order, par=args.par, start_s=args.start,
                          end_s=args.end, debug=args.debug)
    from pitchspeed.calib.model import is_manual
    if is_manual(cf):
        raise SystemExit(f"This video is set up for marking the pitches by hand, so there is nothing to search "
                         f"automatically.\n  Mark them:  pitchspeed mark \"{args.video}\"\n  Or click the rubber "
                         f"and plate for the automatic search:  pitchspeed calibrate \"{args.video}\"")
    result, source = analyze(args.video, cf, out_dir, opts)
    print()
    print(format_table(result))
    if _csv_failed(result):
        _retry_csv(result, out_dir)
    p = out_paths(out_dir)
    print(f"\nResults: {p['results_csv']}\n         {p['results_json']}")
    return result, source, cf


def _wait_until_writable(path: str) -> None:
    from pitchspeed.render.overlay import can_write
    for _ in range(20):
        if can_write(path):
            return
        print(f"\n!! {os.path.basename(path)} is open in another program (a video player?).")
        if not _ask_enter("   Close it, then press Enter (or type s and Enter to write the video under "
                          "another name): "):
            return


def _render_clips(result, source, out_dir: str, audio: bool, style: str = "radar",
                  originals: bool = True) -> Optional[str]:
    """One clip per pitch (render/clips.py); returns the clips folder or None."""
    from pitchspeed.pipeline import Progress, load_transforms
    from pitchspeed.render.clips import render_clips
    if not any(p.status != "rejected" for p in result.pitches):
        print("\nNo pitches, so no clips were written.")
        return None
    tf, ts = load_transforms(out_dir)
    print("\nWriting one clip per pitch ...")
    prog = {"p": None}

    def progress(done, total):
        if prog["p"] is None:
            prog["p"] = Progress(total, "clips")
        prog["p"].update(done)

    info: dict = {}
    try:
        clips = render_clips(result, source, out_dir, transforms=tf, transform_samples=ts, audio=audio,
                             progress=progress, style=style, originals=originals, info=info)
    except ValueError as e:
        print(f"\nNo clips were written: {e}")
        return None
    except (OSError, RuntimeError) as e:
        print(f"\nCould not write the clips: {e}\n  (the results themselves are saved; try again with:  "
              f"pitchspeed render \"{os.path.basename(result.video.path or 'VIDEO')}\" --mode clips)")
        return None
    if prog["p"] is not None:
        prog["p"].done()
    folder = info.get("folder", os.path.join(out_dir, "clips"))
    print(f"Clips: {len(clips)} in {folder}")
    if info.get("originals"):
        print(f"  untouched copies of the original video: {os.path.join(folder, 'originals')} "
              f"({len(info['originals'])})")
    for note in (info.get("notes") or [])[:12]:
        print(f"  note: {note}")
    if len(info.get("notes") or []) > 12:
        print(f"  ... and {len(info['notes']) - 12} more notes (see the log)")
    if info.get("removed"):
        print(f"  removed {len(info['removed'])} out-of-date clip(s) from the previous time")
    return folder


def _render(result, source, out_dir: str, mode: str, audio: bool, style: str = "radar",
            originals: bool = True) -> Optional[str]:
    from pitchspeed.pipeline import Progress, load_transforms
    from pitchspeed.render.overlay import render_annotated
    if mode == "clips":
        return _render_clips(result, source, out_dir, audio, style, originals)
    tf, ts = load_transforms(out_dir)
    out = out_paths(out_dir)["annotated"]
    if mode == "highlights" and not any(p.status != "rejected" for p in result.pitches):
        print("\nNo pitches to show, so no highlights video was written.")
        return None
    _wait_until_writable(out)
    print(f"\nWriting annotated video ({mode}) ...")
    prog = {"p": None}

    def progress(done, total):
        if prog["p"] is None:
            prog["p"] = Progress(total, "rendering")
        prog["p"].update(done)

    info: dict = {}
    try:
        out = render_annotated(result, source, out, mode=mode, transforms=tf, transform_samples=ts,
                               audio=audio, progress=progress, info=info, style=style)
    except ValueError as e:
        print(f"\nCould not write the annotated video: {e}")
        return None
    except (OSError, RuntimeError) as e:
        video = os.path.basename(result.video.path or "VIDEO")
        print(f"\nCould not write the annotated video: {e}\n"
              f"  If {out} is open in a video player, close it and run:  pitchspeed render \"{video}\"\n"
              f"  (the results themselves are saved)")
        return None
    if prog["p"] is not None:
        prog["p"].done()
    print(f"Annotated video: {out}")
    if info.get("note"):
        print(f"  note: {info['note']}")
    if audio and info.get("audio_status") and not str(info["audio_status"]).startswith("ok"):
        print(f"  audio: {info['audio_status']}")
    return out


def cmd_analyze(args) -> int:
    _check_video(args.video)
    out_dir = _out_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    result, source, _ = _analyze(args, out_dir)
    if args.render != "none":
        _render(result, source, out_dir, args.render, not args.no_audio)
    return 0


def _review(args, out_dir: str, source=None, result=None, cf=None, mode: Optional[str] = None):
    """Open the review window (mode 'mark' for a session marked by hand: long-tape navigation)."""
    from pitchspeed.calib.model import is_manual
    from pitchspeed.pipeline import is_marking_session, load_transforms, make_recompute, renumber, save_results
    from pitchspeed.review.ui import run_review_ui
    if result is None:
        result = _load_results(out_dir)
    if source is None:
        source = _source_for_results(args, out_dir, result)
    if cf is None:
        cf = _calibration_for_results(args, out_dir, result, source)
    tf, ts = load_transforms(out_dir)

    def save(res):
        # Autosave after every edit: results.json is the safe copy.  The CSV gets one
        # quick attempt (a locked CSV must not freeze the window); it is retried at the end.
        renumber(res.pitches)
        save_results(res, out_dir, csv_retry_s=0.0)

    if mode is None:
        mode = "mark" if (is_marking_session(result) or is_manual(cf)) else "review"
    if mode == "mark":
        print("\nOpening the video. Space plays (+ faster, - slower), J/L jump 1 second, click the bar at the\n"
              "bottom to jump anywhere. At each pitch press N at the release, then C at the glove (or B if\n"
              "the batter hit it). H shows all keys; Q saves and quits. Everything is saved as you go.")
    else:
        print("\nOpening the review window. Press H in the window for help.")
    recompute = make_recompute(cf, result.scan, source, video=result.video,
                               undecodable=(result.params or {}).get("undecodable_frames"))
    result = run_review_ui(result, source, cf, recompute, save,
                           transforms=tf, transform_samples=ts, ui_scale=getattr(args, "ui_scale", None),
                           mode=mode)
    _final_save(result, out_dir)
    return result, source


def cmd_review(args) -> int:
    from pitchspeed.render.report import format_table
    _check_video(args.video)
    out_dir = _out_dir(args)
    result, source = _review(args, out_dir)
    print(format_table(result))
    if args.render != "none":
        _render(result, source, out_dir, args.render, not args.no_audio)
    return 0


def _ask_method() -> str:
    """'auto' or 'mark': how to find the pitches of a video that has no setup yet."""
    print("\nHow should the pitches be found?")
    print("  1) automatically - you click the pitching rubber and home plate once (again if the camera")
    print("     moves or zooms). Best for a camera on a tripod.")
    print("  2) by hand - you press a key at each pitch's release and at the catch. Any camera angle;")
    print("     the camera may move and zoom. Speeds are typically within about 2-3 mph.")
    while True:
        a = (_ask("Choose 1 or 2", "1") or "1").strip().lower()
        if a in ("1", "a", "auto", "automatic"):
            return "auto"
        if a in ("2", "m", "h", "hand", "mark"):
            return "mark"


def _marking_setup(args, out_dir: str):
    """(result, source, calibration) for marking pitches by hand: the saved session (or saved
    results of any kind) when there is one for this video, else a new session.  None = stopped."""
    from pitchspeed.calib.model import manual_calibration
    from pitchspeed.pipeline import start_marking, video_fingerprint
    from pitchspeed.render.report import load_json
    p = out_paths(out_dir)
    cf, cpath = _load_calib(args, out_dir)
    if cf is not None:
        _apply_settings_flags(args, cf, cpath)
    info = _video_info(args.video, out_dir, args.par)
    if os.path.exists(p["results_json"]):
        try:
            saved = load_json(p["results_json"])
        except (OSError, ValueError) as e:
            saved = None
            print(f"\nnote: the saved results.json cannot be read ({e}); starting a new session "
                  f"(a copy of it is kept)")
        if saved is not None and int(saved.video.n_frames) == int(info.n_frames):
            n = sum(1 for q in saved.pitches if q.status != "rejected")
            print(f"\nContinuing with the saved results: {n} pitch{'es' if n != 1 else ''} so far.")
            source = _source_for_results(args, out_dir, saved)
            return saved, source, _calibration_for_results(args, out_dir, saved, source)
        if saved is not None:
            print(f"\nnote: the saved results are for a video with {saved.video.n_frames} frames, this one has "
                  f"{info.n_frames}: starting a new session (the old results are backed up)")
    if cf is None:
        print("\nSettings (marking by hand needs no clicks on the picture, only the pitching distance):")
        st = _settings_from_args(args)
        st.sar = float(info.sar)
        cf = manual_calibration(st, image_size=(int(info.width), int(info.height)),
                                video=video_fingerprint(args.video, info))
        cf.save(cpath)
        print(f"Saved the settings: {cpath}")
    elif not _check_calibration(cf, args.video, info, interactive=True):
        print("Stopped; nothing was changed.")
        return None
    print("\nOpening the video (checking the interlacing and field order) ...")
    result, source = start_marking(args.video, cf, out_dir, AnalyzeOptions(field_order=args.field_order,
                                                                           par=args.par))
    for w in result.warnings[:6]:
        print(f"  note: {w}")
    return result, source, cf


def _mark_flow(args, out_dir: str, guided: bool = False) -> int:
    from pitchspeed.render.report import format_table
    if guided:
        print("\nStep 1/3: settings")
    setup = _marking_setup(args, out_dir)
    if setup is None:
        return 1
    result, source, cf = setup
    if guided:
        print("\nStep 2/3: mark the pitches")
    result, source = _review(args, out_dir, source, result=result, cf=cf, mode="mark")
    print()
    print(format_table(result))
    what = args.render
    if guided:
        what = (_ask("\nStep 3/3: write the videos? (clips = one per pitch / full / highlights / none)",
                     "clips") or "clips").strip().lower()
        what = next((m for m in ("clips", "full", "highlights", "none") if m.startswith(what[:1] or "c")), "clips")
    if what != "none":
        _render(result, source, out_dir, what, not args.no_audio)
    if guided:
        print("\nDone.")
    return 0


def cmd_mark(args) -> int:
    _check_video(args.video)
    out_dir = _out_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    return _mark_flow(args, out_dir)


def cmd_render(args) -> int:
    _check_video(args.video)
    out_dir = _out_dir(args)
    result = _load_results(out_dir)
    source = _source_for_results(args, out_dir, result)
    _render(result, source, out_dir, args.mode, not args.no_audio, getattr(args, "style", "radar"),
            originals=not getattr(args, "no_originals", False))
    return 0


def _keep_saved_results(args, out_dir: str, cf, info):
    """`run` with results already there: keep them (default) or analyze again.

    Returns (result, source, calibration) when kept, else (None, None, cf)."""
    from pitchspeed.pipeline import describe_user_work, has_user_work, user_work_summary
    from pitchspeed.render.report import format_table, load_json
    p = out_paths(out_dir)
    try:
        saved = load_json(p["results_json"])
    except (OSError, ValueError) as e:
        print(f"\nnote: the saved results.json cannot be read ({e}); analyzing again (a copy of it is kept)")
        return None, None, cf
    edits = has_user_work(saved)
    backed = "; your review edits are backed up first and carried over where possible" if edits else ""
    if info is not None and int(info.n_frames) != int(saved.video.n_frames):
        print(f"\nnote: the saved results were made from a video with {saved.video.n_frames} frames, this one "
              f"has {info.n_frames}: analyzing again{backed}")
        return None, None, cf
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p["results_json"])))
    n = sum(1 for q in saved.pitches if q.status != "rejected")
    print(f"\nSaved results found (from {when}): {n} pitch{'es' if n != 1 else ''}"
          + (f"; your review work: {describe_user_work(user_work_summary(saved))}" if edits else ""))
    print("  1) keep them and go on to the review (default)")
    print("  2) analyze the video again" + (f" ({backed[2:]})" if edits else " (replaces them)"))
    given = [f for f, on in (("--field-order", args.field_order not in (None, "auto")),
                             ("--start", args.start is not None), ("--end", args.end is not None),
                             ("--debug", bool(args.debug))) if on]
    if given:
        print(f"  note: {', '.join(given)} only take effect if you choose 2")
    while True:
        a = (_ask("Choose 1 or 2", "1") or "1").strip().lower()
        if a in ("1", "k", "keep", "y"):
            break
        if a in ("2", "a", "analyze", "r"):
            return None, None, cf
    source = _source_for_results(args, out_dir, saved, check_field_order=False)
    use = _calibration_for_results(args, out_dir, saved, source)
    print()
    print(format_table(saved))
    if _csv_failed(saved):
        _retry_csv(saved, out_dir)
    return saved, source, use


def _is_marking_folder(cf, out_dir: str) -> bool:
    """The saved setup / results of this video are a session marked by hand."""
    from pitchspeed.calib.model import is_manual
    if is_manual(cf):
        return True
    p = out_paths(out_dir)["results_json"]
    if cf is None and os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                import json
                return (json.load(f).get("params") or {}).get("mode") == "marked_by_hand"
        except (OSError, ValueError, AttributeError):
            return False
    return False


def cmd_run(args) -> int:
    _check_video(args.video)
    out_dir = _out_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    cf, cpath = _load_calib(args, out_dir)
    if _is_marking_folder(cf, out_dir):
        print("This video's pitches are marked by hand (no automatic search).")
        return _mark_flow(args, out_dir, guided=True)
    if cf is None and not args.view and not os.path.exists(out_paths(out_dir)["results_json"]):
        if _ask_method() == "mark":
            return _mark_flow(args, out_dir, guided=True)
    info = None
    fresh = False
    if cf is not None:
        _check_view_flag(args, cf)
        info = _video_info(args.video, out_dir, args.par)
        try:
            usable = _check_calibration(cf, args.video, info, interactive=True)
        except SystemExit as e:           # clicked on a video of another size: cannot be used
            print(f"\n{e}")
            usable = False
        if not usable:
            if not (_ask("Calibrate again for this video now? (y/n)", "y") or "y").lower().startswith("y"):
                print("Stopped; nothing was changed.")
                return 1
            cf = None
    if cf is None:
        print("Step 1/4: calibration (click a few points once for this video)")
        args.frame = 0
        if cmd_calibrate(args) != 0:
            return 1
        cf, cpath = _load_calib(args, out_dir)
        fresh = True
    else:
        _apply_settings_flags(args, cf, cpath)
    result = source = None
    if os.path.exists(out_paths(out_dir)["results_json"]):
        if fresh:
            print("\nnote: new calibration, so the video is analyzed again (any review edits in the old "
                  "results are backed up first and carried over where possible)")
        else:
            result, source, cf = _keep_saved_results(args, out_dir, cf, info)
    if result is None:
        print("\nStep 2/4: finding pitches and estimating speeds")
        result, source, cf = _analyze(args, out_dir, cf=cf)
    a = (_ask("\nStep 3/4: open the review window to check/fix pitches? (y/n)", "y") or "y").lower()
    if a.startswith("y"):
        result, source = _review(args, out_dir, source, result=result, cf=cf)
    a = (_ask("\nStep 4/4: write the annotated video? (full / highlights / clips = one per pitch / none)",
              args.render) or "full").lower()
    if a[:1] in ("f", "h", "c"):
        _render(result, source, out_dir, {"f": "full", "h": "highlights", "c": "clips"}[a[:1]], not args.no_audio)
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
        p.add_argument("--ui-scale", type=float, help="enlarge (e.g. 1.5) or shrink the click/review windows")

    def settings_args(p, view=True):
        if view:
            p.add_argument("--view", choices=["side", "behind"], help="camera position")
        p.add_argument("--level", help=f"league preset: {', '.join(C.LEVELS)}")
        p.add_argument("--distance", type=float, help="rubber-to-plate distance in feet (overrides --level)")
        p.add_argument("--sport", choices=list(SPORTS),
                       help="baseball or softball, for a --distance without --level (a level implies it)")
        p.add_argument("--extension", type=float,
                       help="pitcher's release point in front of the rubber, feet (default: assumed by level)")
        p.add_argument("--throws", choices=["R", "L", "r", "l"], help="pitcher's throwing hand (side view)")
        p.add_argument("--altitude", type=float, help="field altitude in feet (air density)")
        p.add_argument("--temp", type=float, help="air temperature in F (air density)")

    def analyze_args(p):
        p.add_argument("--start", type=float, help="only analyze from this time (seconds)")
        p.add_argument("--end", type=float, help="only analyze up to this time (seconds)")
        p.add_argument("--debug", action="store_true", help="write debug images to <out>/debug")
        p.add_argument("--render", choices=["full", "highlights", "clips", "none"], default="full",
                       help="annotated video to write (default: full; clips = one per pitch)")
        p.add_argument("--no-audio", action="store_true", help="do not copy audio into the annotated video")

    p = sub.add_parser("probe", help="show video format, interlacing and field order")
    common(p, calib=False)
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("calibrate", help="click the field landmarks for a video")
    common(p)
    settings_args(p)
    p.add_argument("--frame", type=int, help="start at this frame")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("analyze", help="find pitches and estimate speeds (settings flags update the "
                                       "saved calibration)")
    common(p)
    settings_args(p)
    analyze_args(p)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("review", help="check, fix, reject or add pitches by hand")
    common(p)
    p.add_argument("--render", choices=["full", "highlights", "clips", "none"], default="none",
                   help="re-render the annotated video afterwards (clips = one per pitch)")
    p.add_argument("--no-audio", action="store_true")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("render", help="write the annotated video from saved results")
    common(p, calib=False)
    p.add_argument("--mode", choices=["full", "highlights", "clips"], default="full",
                   help="full video, only the pitches, or one clip per pitch (in <out>/clips)")
    p.add_argument("--style", choices=["radar", "classic"], default="radar",
                   help="radar: radar-gun speed display and ball path (default); classic: plain text")
    p.add_argument("--no-originals", action="store_true",
                   help="clips mode: no untouched copies of the original video")
    p.add_argument("--no-audio", action="store_true")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("mark", help="mark the pitches by hand (any camera angle, no clicks on the picture), "
                                    "then write one clip per pitch")
    common(p)
    settings_args(p, view=False)
    p.add_argument("--render", choices=["clips", "full", "highlights", "none"], default="clips",
                   help="what to write afterwards (default: clips = one per pitch)")
    p.add_argument("--no-audio", action="store_true")
    p.set_defaults(func=cmd_mark)

    p = sub.add_parser("run", help="guided: calibrate (if needed) -> analyze -> review -> render; or mark "
                                   "the pitches by hand")
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
    for stream in (sys.stdout, sys.stderr):   # file names/notes may not fit a Windows console code page
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = build_parser()
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if not getattr(args, "func", None):
        ap.print_help()
        return 0
    from pitchspeed.ui.gui import GuiUnavailable
    try:
        return int(args.func(args) or 0)
    except GuiUnavailable as e:
        print(f"\nerror: cannot open a window: {e}", file=sys.stderr)
        return 2
    except PermissionError as e:          # a file is locked by another program (Excel, a video player)
        print(f"\nerror: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
