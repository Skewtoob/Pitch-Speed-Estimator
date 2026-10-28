"""Result files and the console summary.

    write_json(result, path)   results.json, schema 'pitchspeed.results/1' (atomic)
    load_json(path)            -> AnalysisResult (inverse of write_json)
    write_csv(result, path)    results.csv, one row per pitch (+ review rows)
    format_table(result)       readable console table
    format_time(t)             'hh:mm:ss.sss'

ATOMIC WRITES.  Every file is written to a temporary file in the same
folder, flushed + fsync'ed, then moved over the target with `os.replace`,
so a crash never leaves a half-written results file.  On Windows the
replace is retried for ~2 s (virus scanners / indexers hold new files
briefly); a target that stays locked - typically results.csv open in Excel -
raises PermissionError with a message saying so.

CSV FORMAT (decided here, see docs/design-notes/io_ux.md section 6):
* UTF-8 **with BOM** ('utf-8-sig') and CRLF line ends: Excel on Windows then
  opens it correctly, including non-ASCII file names / notes.  Python's
  csv module reads it with encoding='utf-8-sig'; pandas strips the BOM.
* Decimal point '.', comma separator, a header row.  Empty cell = no value.
* Speeds in mph (ci90_lo / ci90_hi = the 90% interval) to 1 decimal; times
  in seconds to 3 decimals (1 ms; a field is 16.7 ms); confidence to 2.
* Rows, in this order:
    1. every Pitch, chronological by release time, INCLUDING rejected ones
       (status 'rejected', empty pitch number, speed values kept so nothing
       is lost - filter on `status` before averaging);
    2. 'possible_missed' review rows (windows the finder thinks may hold a
       pitch it could not measure), chronological, speed columns empty;
       a window the user already checked in review gets status
       'checked_no_pitch' / 'checked_pitch_added' instead;
    3. only with write_csv(..., rejected_tracks=True): one 'rejected_track'
       row per moving object the finder did not accept (reject code in
       `notes`).  Off by default because real video produces a few per
       second of mostly noise; they are always kept in results.json and
       counted in the console summary, so nothing is dropped silently.
* `field_release` / `field_catch` are sample indices: the field number for
  interlaced video (2 per frame), the frame-sample number for progressive.
* `flags` is a ';'-separated list (pitch flags + speed flags).  `reason`
  holds the speed estimator's explanation (why 'n/a', or caveats);
  `notes` holds the user's / review notes.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
import textwrap
import time
from collections import Counter
from typing import Any, Iterable, Optional

from pitchspeed.render.text import (ci90_text, finite, format_time, has_number, method_name,
                                    pitch_flags, sigma_text, speed_reason, speed_short,
                                    whole_mph)
from pitchspeed.types import AnalysisResult, Pitch, PitchTrack, result_from_dict, to_jsonable

__all__ = ["write_json", "load_json", "write_csv", "format_table", "format_time", "missed_review",
           "CSV_COLUMNS", "csv_rows", "SCHEMA",
           "result_to_dict", "view_for_frame", "frame_of_sample"]

SCHEMA = "pitchspeed.results/1"
_SCHEMA_FAMILY = "pitchspeed.results/"
_REPLACE_RETRY_S = 2.0

CSV_COLUMNS = [
    "pitch", "status", "time_release", "t_release_s", "t_catch_s", "flight_s",
    "frame_release", "frame_catch", "field_release", "field_catch", "view", "method",
    "v_release_mph", "v_release_unc_mph", "ci90_lo", "ci90_hi", "v_avg_mph", "v_plate_mph",
    "label", "confidence", "flags", "reason", "notes",
]


# ================================================================ atomic I/O
def _replace(tmp: str, path: str, retry_s: Optional[float] = None) -> None:
    deadline = time.monotonic() + (_REPLACE_RETRY_S if retry_s is None else max(0.0, float(retry_s)))
    while True:
        try:
            os.replace(tmp, path)
            return
        except PermissionError as e:
            if time.monotonic() > deadline:
                raise PermissionError(
                    f"cannot write {path}: the file is locked by another program "
                    f"(is it open in Excel or a viewer?). Close it and try again.") from e
            time.sleep(0.1)


def _file_mode(path: str) -> int:
    """Permissions for a new file: the existing file's, else 0o666 minus the umask
    (mkstemp would leave 0o600)."""
    try:
        return os.stat(path).st_mode & 0o777
    except OSError:
        um = os.umask(0)
        os.umask(um)
        return 0o666 & ~um


def _atomic_write_bytes(path: str, data: bytes, retry_s: Optional[float] = None) -> None:
    path = os.fspath(path)
    folder = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=folder)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if os.name != "nt":
            os.chmod(tmp, _file_mode(path))
        _replace(tmp, path, retry_s)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


# ====================================================================== JSON
def _json_default(obj: Any) -> Any:
    """Types `to_jsonable` leaves alone but json cannot encode."""
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    if hasattr(obj, "item") and callable(obj.item):          # numpy scalar types
        return obj.item()
    raise TypeError(f"results.json: cannot store a value of type {type(obj).__name__}")


def _clean_floats(obj: Any) -> Any:
    """NaN/inf anywhere (also inside numpy scalars) -> None, so the file is strict JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_floats(v) for v in obj]
    return obj


def result_to_dict(result: AnalysisResult) -> dict:
    """The JSON document for `result` (schema + version always filled in)."""
    from pitchspeed import __version__
    d = _clean_floats(to_jsonable(result))
    d["schema"] = d.get("schema") or SCHEMA
    d["version"] = d.get("version") or __version__
    return d


def write_json(result: AnalysisResult, path: str) -> None:
    """Write results.json atomically (temp file + os.replace), UTF-8, strict JSON."""
    text = json.dumps(result_to_dict(result), indent=1, ensure_ascii=False, allow_nan=False,
                      default=_json_default)
    _atomic_write_bytes(path, (text + "\n").encode("utf-8"))


def load_json(path: str) -> AnalysisResult:
    """Read a results.json written by `write_json` (any 'pitchspeed.results/1' file)."""
    path = os.fspath(path)
    with open(path, "r", encoding="utf-8-sig") as f:
        try:
            d = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} is not valid JSON ({e}); it may be damaged") from e
    if not isinstance(d, dict) or "pitches" not in d or "video" not in d:
        raise ValueError(f"{path} is not a pitchspeed results file")
    schema = d.get("schema") or SCHEMA
    if not str(schema).startswith(_SCHEMA_FAMILY):
        raise ValueError(f"{path}: unknown schema {schema!r} (expected {SCHEMA!r})")
    if schema != SCHEMA:
        raise ValueError(f"{path}: results schema {schema!r} is newer than this program "
                         f"understands ({SCHEMA!r}); update pitchspeed")
    try:
        return result_from_dict(d)
    except (KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"{path}: damaged or incomplete results file ({type(e).__name__}: {e})") from e


# ======================================================================= CSV
def _num(x: Any, nd: int) -> str:
    return f"{float(x):.{nd}f}" if finite(x) else ""


def _int(x: Any) -> str:
    try:
        return str(int(x)) if x is not None and int(x) >= 0 else ""
    except (TypeError, ValueError):
        return ""


def view_for_frame(calibration: Optional[dict], frame: Optional[int]) -> str:
    """Camera view ('side' / 'behind') of the calibration segment containing `frame`."""
    segs = (calibration or {}).get("segments") or []
    if not segs:
        return ""
    if frame is not None and frame >= 0:
        for seg in segs:
            start = seg.get("start_frame", 0) or 0
            end = seg.get("end_frame")
            if frame >= start and (end is None or frame < end):
                return str(seg.get("view", ""))
    return str(segs[0].get("view", ""))


def _pitch_row(p: Pitch, calibration: dict) -> dict:
    sp = p.speed
    rel, cat = p.release, p.catch
    t_rel = rel.t if rel is not None else None
    t_cat = cat.t if cat is not None else None
    flight = (t_cat - t_rel) if finite(t_rel) and finite(t_cat) else None
    show = has_number(sp)
    lo = hi = None
    if show and sp.ci90_mph is not None and len(sp.ci90_mph) == 2:
        lo, hi = sp.ci90_mph
    return {
        "pitch": str(p.number) if p.status != "rejected" and p.number > 0 else "",
        "status": p.status,
        "time_release": format_time(t_rel) if finite(t_rel) else "",
        "t_release_s": _num(t_rel, 3),
        "t_catch_s": _num(t_cat, 3),
        "flight_s": _num(flight, 3),
        "frame_release": _int(p.frame_release),
        "frame_catch": _int(p.frame_catch),
        "field_release": _int(rel.s if rel is not None else None),
        "field_catch": _int(cat.s if cat is not None else None),
        "view": view_for_frame(calibration, p.frame_release),
        "method": (sp.method if sp is not None else "none") or "none",
        "v_release_mph": _num(sp.v_release_mph, 1) if show else "",
        "v_release_unc_mph": _num(sp.sigma_mph, 1) if show else "",
        "ci90_lo": _num(lo, 1),
        "ci90_hi": _num(hi, 1),
        "v_avg_mph": _num(sp.v_avg_mph, 1) if show else "",
        "v_plate_mph": _num(sp.v_plate_mph, 1) if show else "",
        "label": (sp.label if sp is not None else "none") or "none",
        "confidence": _num(p.confidence, 2),
        "flags": ";".join(pitch_flags(p)),
        "reason": speed_reason(sp),
        "notes": p.notes or "",
    }


def _empty_row(**kw) -> dict:
    row = {c: "" for c in CSV_COLUMNS}
    row.update(kw)
    return row


def frame_of_sample(result: AnalysisResult, s: Optional[int]) -> Optional[int]:
    """Source frame showing sample s (fields: s = 2*frame + k; frames: frame_step/phase)."""
    if s is None:
        return None
    try:
        s = int(s)
    except (TypeError, ValueError):
        return None
    if s < 0:
        return None
    if result.scan.mode == "interlaced":
        return s // 2
    ev = result.scan.evidence or {}
    step = max(1, int(ev.get("frame_step", 1) or 1))
    return int(ev.get("frame_phase", 0) or 0) % step + s * step


MISSED_REVIEW_STATUS = {"no_pitch": ("checked_no_pitch", "checked in review: no pitch"),
                        "pitch_added": ("checked_pitch_added", "checked in review: a pitch was added by hand")}


def missed_review(d: dict) -> Optional[tuple[str, str]]:
    """(csv status, note) for a possible-missed window the user checked in review, else None."""
    rv = d.get("review") if isinstance(d, dict) else None
    if not rv:
        return None
    return MISSED_REVIEW_STATUS.get(str(rv), (f"checked_{rv}", f"checked in review: {rv}"))


def _missed_row(d: dict, result: AnalysisResult) -> dict:
    f0, f1 = frame_of_sample(result, d.get("s_start")), frame_of_sample(result, d.get("s_end"))
    checked = missed_review(d)
    if checked is not None:
        status, note = checked
        notes = f"{note} (was: possible missed pitch: {d.get('reason', 'unknown')})"
    else:
        status, notes = "possible_missed", f"possible missed pitch: {d.get('reason', 'unknown')}; check in review"
    return _empty_row(status=status,
                      time_release=format_time(d.get("t_start")) if finite(d.get("t_start")) else "",
                      t_release_s=_num(d.get("t_start"), 3), t_catch_s=_num(d.get("t_end"), 3),
                      frame_release=_int(f0), frame_catch=_int(f1),
                      field_release=_int(d.get("s_start")), field_catch=_int(d.get("s_end")),
                      view=view_for_frame(result.calibration, f0), notes=notes)


def _track_row(tr: PitchTrack, result: AnalysisResult) -> dict:
    s0 = tr.points[0].s if tr.points else None
    s1 = tr.points[-1].s if tr.points else None
    f0, f1 = frame_of_sample(result, s0), frame_of_sample(result, s1)
    flags = ";".join(str(f) for f in (tr.flags or []))
    return _empty_row(status="rejected_track",
                      time_release=format_time(tr.t_first) if finite(tr.t_first) else "",
                      t_release_s=_num(tr.t_first, 3), t_catch_s=_num(tr.t_last, 3),
                      frame_release=_int(f0), frame_catch=_int(f1),
                      field_release=_int(s0), field_catch=_int(s1),
                      view=view_for_frame(result.calibration, f0),
                      confidence=_num(tr.confidence, 2), flags=flags,
                      notes=f"track {tr.track_id} not accepted as a pitch: {tr.reject_code or 'rejected'}")


def _sort_t(x: Optional[float]) -> float:
    return float(x) if finite(x) else math.inf


def csv_rows(result: AnalysisResult, review_rows: bool = True, rejected_tracks: bool = False) -> list[dict]:
    """The CSV rows (dicts keyed by CSV_COLUMNS) in file order (see module doc)."""
    cal = result.calibration or {}
    pitches = sorted(result.pitches, key=lambda p: (_sort_t(p.release.t if p.release else None), p.number))
    rows = [_pitch_row(p, cal) for p in pitches]
    review = []
    if review_rows:
        review += [(_sort_t(d.get("t_start")), _missed_row(d, result)) for d in result.possible_missed or []]
    if rejected_tracks:
        review += [(_sort_t(tr.t_first), _track_row(tr, result)) for tr in result.rejected_tracks or []]
    rows += [row for _, row in sorted(review, key=lambda x: x[0])]
    return rows


def write_csv(result: AnalysisResult, path: str, *, review_rows: bool = True,
              rejected_tracks: bool = False, retry_s: Optional[float] = None) -> None:
    """Write results.csv atomically (UTF-8 with BOM, CRLF; see module doc).

    retry_s: how long to keep retrying a locked target (None = ~2 s; 0 = one
    attempt, for autosaves that must not freeze a window).  A target that stays
    locked raises PermissionError."""
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator="\r\n", extrasaction="ignore")
    w.writeheader()
    for row in csv_rows(result, review_rows=review_rows, rejected_tracks=rejected_tracks):
        w.writerow(row)
    _atomic_write_bytes(path, buf.getvalue().encode("utf-8-sig"), retry_s)


# ===================================================================== table
TABLE_HEADER = ["#", "time", "frames", "speed", "+/-", "90% range", "avg", "method", "label", "flags"]
_RIGHT = {"#", "speed", "+/-", "avg"}


def _frames_text(p: Pitch) -> str:
    a, b = p.frame_release, p.frame_catch
    if a is not None and a >= 0 and b is not None and b >= 0:
        return f"{a}-{b}"
    return "-"


def _table_row(p: Pitch) -> list[str]:
    sp = p.speed
    status_flags = [p.status] if p.status in ("edited", "manual") else []
    flags = ",".join(status_flags + pitch_flags(p)) or "-"
    t = p.release.t if p.release is not None else None
    if p.status == "rejected":
        return ["-", format_time(t), _frames_text(p), "", "", "", "", "", "rejected", flags]
    show = has_number(sp)
    return [
        str(p.number), format_time(t), _frames_text(p), speed_short(sp),
        sigma_text(sp.sigma_mph) if show else "-", ci90_text(sp),
        whole_mph(sp.v_avg_mph) if show else "-",
        method_name(sp.method if sp is not None else None),
        (sp.label if sp is not None else "none") or "none", flags,
    ]


def _layout(rows: list[list[str]]) -> list[str]:
    widths = [max(len(r[i]) for r in rows) for i in range(len(TABLE_HEADER))]
    out = []
    for r in rows:
        cells = []
        for i, (name, cell) in enumerate(zip(TABLE_HEADER, r)):
            last = i == len(TABLE_HEADER) - 1
            if last:
                cells.append(cell)
            elif name in _RIGHT:
                cells.append(cell.rjust(widths[i]))
            else:
                cells.append(cell.ljust(widths[i]))
        out.append("  ".join(cells).rstrip())
    return out


def _first_assumptions(result: AnalysisResult) -> dict:
    for p in result.pitches:
        if p.speed is not None and p.speed.assumptions:
            return p.speed.assumptions
    return {}


def _fmt_ft(x: Any) -> Optional[str]:
    return f"{float(x):.1f} ft" if finite(x) else None


def _header_lines(result: AnalysisResult) -> list[str]:
    v, sc = result.video, result.scan
    name = os.path.basename(v.path) if v.path else "(unknown video)"
    if sc.mode == "interlaced":
        rate = f"{v.frame_rate:.2f} fps interlaced, {(sc.field_order or '?').upper()} -> " \
               f"{2 * v.frame_rate:.2f} fields/s"
    else:
        rate = f"{v.frame_rate:.2f} fps, {sc.mode} (frame timing only)"
    lines = [f"pitchspeed results: {name}  ({v.width}x{v.height}, {rate}, "
             f"field-order confidence {sc.confidence})"]
    segs = (result.calibration or {}).get("segments") or []
    views = []
    for s in segs:
        if s.get("view") and s.get("view") not in views:
            views.append(s.get("view"))
    st = (result.calibration or {}).get("settings") or {}
    a = _first_assumptions(result)
    dist = a.get("distance_ft", st.get("distance_ft"))
    lvl = a.get("level", st.get("level"))
    n_ok = sum(1 for p in result.pitches if p.status != "rejected")
    n_rej = len(result.pitches) - n_ok
    view_txt = " + ".join(views) or "?"
    dist_txt = f"{float(dist):.1f} ft rubber->plate" if finite(dist) else "distance ?"
    lines.append(f"View: {view_txt} | {dist_txt}" + (f" (level {lvl})" if lvl else "")
                 + f" | {n_ok} pitch{'es' if n_ok != 1 else ''}"
                 + (f" + {n_rej} rejected" if n_rej else ""))
    if a:
        parts = []
        ext = _fmt_ft(a.get("extension_ft"))
        if ext:
            sig = a.get("extension_sigma_ft")
            parts.append(f"extension {ext}" + (f" +/-{float(sig):.1f}" if finite(sig) else "")
                         + (" (assumed)" if a.get("extension_assumed") else ""))
        if finite(a.get("catch_behind_ft")):
            parts.append(f"catch {float(a['catch_behind_ft']):.1f} ft behind plate")
        if finite(a.get("cd")):
            parts.append(f"Cd {float(a['cd']):.2f}")
        if finite(a.get("air_density_kg_m3")):
            parts.append(f"air {float(a['air_density_kg_m3']):.2f} kg/m3")
        if parts:
            lines.append("Assumed (not measured): " + ", ".join(parts))
    return lines


def _notes_lines(pitches: Iterable[Pitch]) -> list[str]:
    out = []
    for p in pitches:
        who = f"#{p.number}" if p.status != "rejected" else f"rejected @{format_time(p.release.t)}"
        msgs = []
        r = speed_reason(p.speed)
        if r and (p.status != "rejected"):
            msgs.append(r)
        if p.notes:
            msgs.append(p.notes)
        for m in msgs:
            out += textwrap.wrap(f"{who}: {' '.join(m.split())}", width=110, initial_indent="  ",
                                 subsequent_indent="      ", break_on_hyphens=False)
    return out


def _review_lines(result: AnalysisResult) -> list[str]:
    out = []
    every = sorted(result.possible_missed or [], key=lambda d: _sort_t(d.get("t_start")))
    missed = [d for d in every if missed_review(d) is None]
    n_checked = len(every) - len(missed)
    if n_checked:
        out.append(f"Possible missed pitches already checked in review: {n_checked}")
    if missed:
        out.append(f"Possible missed pitches ({len(missed)}) - check them in review:")
        for d in missed[:20]:
            out.append(f"  {format_time(d.get('t_start'))} - {format_time(d.get('t_end'))}  "
                       f"{d.get('reason', '')}")
        if len(missed) > 20:
            out.append(f"  ... and {len(missed) - 20} more (results.csv / results.json)")
    if result.rejected_tracks:
        codes = Counter(tr.reject_code or "rejected" for tr in result.rejected_tracks)
        out.append(f"Moving objects not accepted as pitches: {len(result.rejected_tracks)} ("
                   + ", ".join(f"{c} {n}" for c, n in codes.most_common()) + "); details in results.json")
    return out


FOOTER = [
    "speed = RELEASE speed (out of the hand; what a radar gun reads), mph.",
    "+/- = 1-sigma uncertainty; 90% range = the speed is within it 9 times out of 10",
    "      (if the assumptions above hold).  avg = average speed release->catch.",
    "~78? = rough estimate (label 'poor', uncertainty 5-10%).  n/a = not measurable",
    "      (reason listed under Notes).  label: good <= 2.5%, fair <= 5%, poor <= 10%.",
]


def format_table(result: AnalysisResult) -> str:
    """A readable console summary: header (video / view / assumptions), one
    line per pitch, notes, review items, warnings and a legend."""
    lines = _header_lines(result)
    pitches = sorted(result.pitches, key=lambda p: (_sort_t(p.release.t if p.release else None), p.number))
    lines.append("")
    if pitches:
        lines += _layout([TABLE_HEADER] + [_table_row(p) for p in pitches])
    else:
        lines.append("No pitches found.")
    notes = _notes_lines(pitches)
    if notes:
        lines += ["", "Notes:"] + notes
    review = _review_lines(result)
    if review:
        lines += [""] + review
    if result.warnings:
        lines += ["", "Warnings:"] + [f"  - {w}" for w in result.warnings]
    lines += [""] + FOOTER
    return "\n".join(lines)
