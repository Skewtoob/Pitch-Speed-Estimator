"""Score an AnalysisResult against a synthetic clip's ground truth.

    m = evaluate(result, gt)          # one clip
    print(format_summary(m))
    d = evaluate_dir("suite/")        # every clip of a suite (index.csv or *.gt.json)
    print(format_dir_summary(d))

Matching: detected pitches (status != 'rejected') are matched one-to-one to
GT pitches by release time, |dt| < window (0.25 s), minimizing the total
|dt| (linear assignment).  GT times are converted to the video timeline
first (identical unless the clip has simulated capture drops/duplicates).

Speed errors are estimate - truth (mph): release speed vs the GT 3-D speed
at release, average speed vs the GT release->catch average along y, plate
speed vs the GT speed at the front of the plate.  Coverage = fraction of
GT release speeds inside the pitch's reported 90% interval.  z = error /
reported sigma (should be ~N(0,1) if the uncertainty is honest).

Nothing is dropped silently: unmatched detections are listed as false
pitches with a cause ('duplicate' = falls on an already matched GT pitch,
else the GT distractor active at that time, e.g. 'throwback', 'bird',
'swing', 'pan', or 'unexplained'), missed GT
pitches are listed with whether a 'possible missed' window covers them,
matched pitches without a number are counted separately with their reason.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
from typing import Callable, Optional

import numpy as np

from pitchspeed.types import AnalysisResult, result_from_dict

MATCH_WINDOW_S = 0.25


# ------------------------------------------------------------------ helpers
def _stats(vals) -> dict:
    v = np.array([x for x in vals if x is not None and math.isfinite(x)], dtype=float)
    if v.size == 0:
        return dict(n=0, mean=None, rms=None, mean_abs=None, median_abs=None, max_abs=None, std=None)
    return dict(n=int(v.size), mean=float(v.mean()), rms=float(np.sqrt((v ** 2).mean())),
                mean_abs=float(np.abs(v).mean()), median_abs=float(np.median(np.abs(v))),
                max_abs=float(np.abs(v).max()), std=float(v.std(ddof=1)) if v.size > 1 else 0.0)


def gt_time_mapper(gt: dict) -> Callable[[float], float]:
    """GT (true exposure) time -> video timeline time (out frame j at j * out_frame_dt)."""
    if gt.get("frame_map_identity", True) or not gt.get("frame_map"):
        return lambda t: float(t)
    xs, ys = [], []
    for e in gt["frame_map"]:
        if e.get("dup"):
            continue
        xs.append(e["exposure_times"][0])
        ys.append(e["t_video"])
    xs, ys = np.asarray(xs), np.asarray(ys)
    if len(xs) < 2:
        return lambda t: float(t)

    def f(t: float) -> float:
        if t <= xs[0]:
            return float(ys[0] + (t - xs[0]))
        if t >= xs[-1]:
            return float(ys[-1] + (t - xs[-1]))
        return float(np.interp(t, xs, ys))
    return f


def match_by_time(t_det: list, t_gt: list, window: float = MATCH_WINDOW_S) -> list:
    """One-to-one matching minimizing total |dt| subject to |dt| < window: [(i_det, j_gt), ...]."""
    if not t_det or not t_gt:
        return []
    from scipy.optimize import linear_sum_assignment
    a, b = np.asarray(t_det, float), np.asarray(t_gt, float)
    cost = np.abs(a[:, None] - b[None, :])
    big = 1e6
    c = np.where(cost < window, cost, big)
    ri, ci = linear_sum_assignment(c)
    return sorted([(int(i), int(j)) for i, j in zip(ri, ci) if c[i, j] < big], key=lambda x: x[1])


def _nearest_distractor(t: float, gt: dict, margin: float = 0.5) -> Optional[str]:
    best, bd = None, math.inf
    for d in gt.get("distractors", []):
        a, b = d["t_start"] - margin, d["t_end"] + margin
        if a <= t <= b:
            dist = 0.0 if d["t_start"] <= t <= d["t_end"] else min(abs(t - d["t_start"]), abs(t - d["t_end"]))
            if dist < bd:
                best, bd = d["type"], dist
    return best


def _covered(t0: float, t1: float, windows: list) -> bool:
    for w in windows or []:
        a, b = w.get("t_start"), w.get("t_end")
        if a is not None and b is not None and a <= t1 and b >= t0:
            return True
    return False


def _get(obj, name, default=None):
    return getattr(obj, name, default) if obj is not None else default


# ------------------------------------------------------------------ one clip
def evaluate(result: AnalysisResult, gt: dict, window: float = MATCH_WINDOW_S) -> dict:
    """Metrics of one analysis result vs the clip's GT (see module docstring)."""
    if isinstance(result, dict):
        result = result_from_dict(result)
    to_video = gt_time_mapper(gt)
    gps = gt.get("pitches", [])
    g_rel = [to_video(p["t_release"]) for p in gps]
    g_cat = [to_video(p["t_catch"]) for p in gps]
    det = [p for p in result.pitches if p.status != "rejected"]
    d_rel = [p.release.t if p.release is not None else math.nan for p in det]
    pairs = match_by_time([t for t in d_rel], g_rel, window)
    by_gt = {j: i for i, j in pairs}
    matched_det = {i for i, _ in pairs}
    rows = []
    for j, gp in enumerate(gps):
        row = dict(gt_id=gp["id"], t_release_gt=g_rel[j], t_catch_gt=g_cat[j],
                   v_release_gt=gp["speed_release_mph"], v_avg_gt=gp["avg_speed_y_mph"],
                   v_plate_gt=gp["speed_plate_front_mph"], flight_time_gt=gp["flight_time"],
                   extension_gt=gp.get("extension_ft"), matched=j in by_gt)
        if j not in by_gt:
            row["flagged_possible_missed"] = _covered(g_rel[j] - 0.3, g_cat[j] + 0.3, result.possible_missed)
            row["rejected_track_nearby"] = any(
                t.points and (g_rel[j] - 0.3 <= t.t_first <= g_cat[j] + 0.3) for t in result.rejected_tracks)
            rows.append(row)
            continue
        p = det[by_gt[j]]
        sp = p.speed
        v = _get(sp, "v_release_mph")
        row.update(number=p.number, status=p.status, label=_get(sp, "label", "none"), method=_get(sp, "method"),
                   flags=list(_get(sp, "flags", []) or []) + list(p.flags or []), reason=_get(sp, "reason"),
                   t_release=p.release.t, dt_release_ms=1000.0 * (p.release.t - g_rel[j]),
                   release_method=p.release.method, release_sigma_ms=1000.0 * p.release.sigma,
                   t_catch=_get(p.catch, "t"),
                   dt_catch_ms=1000.0 * (p.catch.t - g_cat[j]) if p.catch is not None else None,
                   catch_method=_get(p.catch, "method"),
                   flight_time=(p.catch.t - p.release.t) if p.catch is not None else None,
                   v_release=v, sigma=_get(sp, "sigma_mph"), ci90=_get(sp, "ci90_mph"),
                   v_avg=_get(sp, "v_avg_mph"), v_plate=_get(sp, "v_plate_mph"))
        if v is not None:
            err = v - gp["speed_release_mph"]
            row.update(err=err, err_pct=100.0 * err / gp["speed_release_mph"])
            s = row["sigma"]
            row["z"] = err / s if s else None
            ci = row["ci90"]
            row["in_ci90"] = bool(ci[0] <= gp["speed_release_mph"] <= ci[1]) if ci else None
        row["err_avg"] = row["v_avg"] - gp["avg_speed_y_mph"] if row["v_avg"] is not None else None
        row["err_plate"] = row["v_plate"] - gp["speed_plate_front_mph"] if row["v_plate"] is not None else None
        rows.append(row)
    false = []
    for i, p in enumerate(det):
        if i in matched_det:
            continue
        near_gt = [gps[j]["id"] for j in range(len(gps)) if g_rel[j] - window <= d_rel[i] <= g_cat[j] + window]
        dis = _nearest_distractor(d_rel[i], gt)
        cause = "duplicate" if near_gt else (dis or "unexplained")
        false.append(dict(number=p.number, status=p.status, t_release=d_rel[i],
                          v_release=_get(p.speed, "v_release_mph"), label=_get(p.speed, "label", "none"),
                          cause=cause, near_gt_pitch=near_gt[0] if near_gt else None, nearest_distractor=dis))
    m = [r for r in rows if r["matched"]]
    with_speed = [r for r in m if r.get("v_release") is not None]
    cov = [r["in_ci90"] for r in with_speed if r.get("in_ci90") is not None]
    labels = {}
    for r in m:
        labels.setdefault(r["label"], []).append(r.get("err"))
    n_gt, n_det = len(gps), len(det)
    summary = dict(
        n_gt=n_gt, n_detected=n_det, n_matched=len(m), n_missed=n_gt - len(m), n_false=len(false),
        recall=len(m) / n_gt if n_gt else None, precision=len(m) / n_det if n_det else None,
        n_missed_flagged=sum(1 for r in rows if not r["matched"] and r.get("flagged_possible_missed")),
        n_with_speed=len(with_speed), n_no_speed=len(m) - len(with_speed),
        speed=_stats([r["err"] for r in with_speed]),
        speed_pct=_stats([r["err_pct"] for r in with_speed]),
        avg_speed=_stats([r["err_avg"] for r in m]),
        plate_speed=_stats([r["err_plate"] for r in m]),
        coverage90=dict(n=len(cov), inside=int(sum(cov)), frac=(sum(cov) / len(cov)) if cov else None),
        z=_stats([r["z"] for r in with_speed if r.get("z") is not None]),
        release_ms=_stats([r["dt_release_ms"] for r in m]),
        catch_ms=_stats([r["dt_catch_ms"] for r in m if r.get("dt_catch_ms") is not None]),
        by_label={k: dict(n=len(v), speed=_stats(v)) for k, v in sorted(labels.items())},
        false_by_cause=_count([f["cause"] for f in false]),
    )
    notes = []
    if not gt.get("frame_map_identity", True):
        notes.append("clip has capture drops/duplicates: GT times mapped to the video timeline")
    return dict(clip=gt.get("video", {}).get("file"), window_s=window, summary=summary, pitches=rows,
                false_pitches=false, notes=notes)


def _count(items) -> dict:
    out = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


# ------------------------------------------------------------------ formatting
def _fmt(x, spec: str = ".1f", none: str = "-") -> str:
    return none if x is None or (isinstance(x, float) and not math.isfinite(x)) else format(x, spec)


def _stat_line(name: str, s: dict, unit: str) -> str:
    if not s or not s.get("n"):
        return f"  {name}: n=0"
    return (f"  {name}: n={s['n']}  bias {s['mean']:+.2f} {unit}  RMS {s['rms']:.2f}  "
            f"mean|e| {s['mean_abs']:.2f}  max|e| {s['max_abs']:.2f}")


def format_summary(m: dict) -> str:
    """Human-readable report of evaluate() output."""
    s = m["summary"]
    out = [f"Evaluation of {m.get('clip') or 'clip'} (match window {m['window_s']:.2f} s)",
           f"  GT pitches {s['n_gt']}, detected {s['n_detected']}, matched {s['n_matched']}  "
           f"recall {_fmt(s['recall'], '.0%')}  precision {_fmt(s['precision'], '.0%')}  "
           f"false {s['n_false']}  missed {s['n_missed']} ({s['n_missed_flagged']} flagged as possible missed)",
           "",
           "  GT#  det#  t_rel(GT)   v_GT   v_est  err    sigma  90% CI         in?  label  dt_rel  dt_catch",
           "  ---  ----  ---------  -----  -----  -----  -----  -------------  ---  -----  ------  --------"]
    for r in m["pitches"]:
        if not r["matched"]:
            tag = "possible-missed window" if r.get("flagged_possible_missed") else "NOT FOUND"
            out.append(f"  {r['gt_id']:3d}  {'-':>4}  {r['t_release_gt']:9.3f}  {r['v_release_gt']:5.1f}  "
                       f"{tag}")
            continue
        ci = r.get("ci90")
        cis = f"{ci[0]:5.1f}-{ci[1]:5.1f}" if ci else "-"
        inn = {True: "yes", False: "NO", None: "-"}[r.get("in_ci90")]
        out.append(f"  {r['gt_id']:3d}  {r['number']:4d}  {r['t_release_gt']:9.3f}  {r['v_release_gt']:5.1f}  "
                   f"{_fmt(r.get('v_release')):>5}  {_fmt(r.get('err'), '+.1f'):>5}  {_fmt(r.get('sigma')):>5}  "
                   f"{cis:>13}  {inn:>3}  {r['label']:>5}  {_fmt(r['dt_release_ms'], '+.0f'):>5}ms  "
                   f"{_fmt(r.get('dt_catch_ms'), '+.0f'):>6}ms"
                   + (f"  ({r['reason']})" if r.get("v_release") is None and r.get("reason") else ""))
    for f in m["false_pitches"]:
        why = {"duplicate": f"second detection of GT pitch {f['near_gt_pitch']}",
               "unexplained": "nothing in the GT"}.get(f["cause"], f"GT {f['cause']}")
        out.append(f"  {'-':>3}  {f['number']:4d}  {f['t_release']:9.3f}  FALSE PITCH  v={_fmt(f['v_release'])}  "
                   f"label={f['label']}  cause: {why}")
    out.append("")
    out.append(_stat_line("release speed error", s["speed"], "mph"))
    out.append(_stat_line("release speed error %", s["speed_pct"], "%"))
    out.append(_stat_line("average speed error", s["avg_speed"], "mph"))
    out.append(_stat_line("plate speed error", s["plate_speed"], "mph"))
    c = s["coverage90"]
    out.append(f"  90% interval coverage: {c['inside']}/{c['n']} = {_fmt(c['frac'], '.0%')}   "
               f"z RMS {_fmt(s['z'].get('rms'), '.2f')} (1.0 = honest sigma)")
    out.append(_stat_line("release time error", s["release_ms"], "ms"))
    out.append(_stat_line("catch time error", s["catch_ms"], "ms"))
    if s["n_no_speed"]:
        out.append(f"  matched pitches without a speed number: {s['n_no_speed']}")
    if s["by_label"]:
        out.append("  by label: " + ", ".join(
            f"{k}: n={v['n']} RMS {_fmt(v['speed'].get('rms'), '.2f')}" for k, v in s["by_label"].items()))
    for n in m.get("notes", []):
        out.append("  note: " + n)
    return "\n".join(out)


# ------------------------------------------------------------------ suites
def default_results_path(video_path: str) -> str:
    """Where `pitchspeed analyze` writes results for a clip (<stem>_pitchspeed/results.json)."""
    try:
        from pitchspeed.pipeline import default_out_dir
        d = default_out_dir(video_path)
    except Exception:                         # noqa: BLE001 - keep evaluation usable stand-alone
        root, _ = os.path.splitext(video_path)
        d = root + "_pitchspeed"
    return os.path.join(d, "results.json")


def load_result(path: str) -> AnalysisResult:
    """results.json -> AnalysisResult (render.report.load_json if available, else types.result_from_dict)."""
    try:
        from pitchspeed.render.report import load_json
        return load_json(path)
    except Exception:                         # noqa: BLE001 - module missing or stricter than needed
        with open(path, "r", encoding="utf-8") as f:
            return result_from_dict(json.load(f))


def _clip_list(suite_dir: str) -> list:
    idx = os.path.join(suite_dir, "index.csv")
    clips = []
    if os.path.exists(idx):
        with open(idx, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status", "ok") != "ok":
                    clips.append(dict(row, gt_path=None))
                    continue
                clips.append(dict(row, gt_path=os.path.join(suite_dir, row["gt"])))
    else:
        for g in sorted(glob.glob(os.path.join(suite_dir, "*.gt.json"))):
            clips.append(dict(name=os.path.basename(g)[:-len(".gt.json")], gt_path=g, status="ok"))
    return clips


def _strata(gt: dict) -> dict:
    cfg = gt["config"]
    sh = float(cfg["render"]["shutter_s"])
    return dict(view=gt["camera"]["view"], level=gt["level"]["key"], codec=gt["video"]["codec"],
                field_order=gt["video"]["field_order"] or gt["video"]["scan"],
                shutter="<=1/500" if sh <= 1 / 499 else ("<=1/200" if sh <= 1 / 199 else ">1/200"),
                fence="fence" if gt["camera"].get("fence", {}).get("enabled") else "no fence")


def evaluate_dir(suite_dir: str, results_for: Optional[Callable[[str], str]] = None,
                 window: float = MATCH_WINDOW_S) -> dict:
    """Evaluate every clip of a suite folder; clips without results are reported, not skipped."""
    results_for = results_for or default_results_path
    clips_out, pooled = [], []
    for c in _clip_list(suite_dir):
        entry = dict(name=c.get("name"), status=c.get("status", "ok"))
        if c["gt_path"] is None or not os.path.exists(c["gt_path"]):
            entry["status"] = "no_gt" if entry["status"] == "ok" else entry["status"]
            clips_out.append(entry)
            continue
        with open(c["gt_path"], "r", encoding="utf-8") as f:
            gt = json.load(f)
        video = os.path.join(os.path.dirname(c["gt_path"]), gt["video"]["file"])
        rpath = results_for(video)
        entry.update(strata=_strata(gt), results=rpath, n_gt=len(gt["pitches"]))
        if not os.path.exists(rpath):
            entry["status"] = "no_results"
            clips_out.append(entry)
            continue
        try:
            m = evaluate(load_result(rpath), gt, window)
        except Exception as e:                # noqa: BLE001 - report, never drop
            entry.update(status="eval_error", error=f"{type(e).__name__}: {e}")
            clips_out.append(entry)
            continue
        entry.update(metrics=m)
        for r in m["pitches"]:
            pooled.append(dict(r, clip=entry["name"], **entry["strata"]))
        for fp in m["false_pitches"]:
            pooled.append(dict(fp, clip=entry["name"], false_pitch=True, **entry["strata"]))
        clips_out.append(entry)
    overall = _pooled_summary(pooled)
    by = {}
    for key in ("view", "level", "codec", "field_order", "shutter", "fence"):
        groups = {}
        for r in pooled:
            groups.setdefault(r[key], []).append(r)
        by[key] = {g: _pooled_summary(rs) for g, rs in sorted(groups.items())}
    status = _count([c["status"] for c in clips_out])
    return dict(suite_dir=suite_dir, clips=clips_out, status=status, overall=overall, by=by, pitch_rows=pooled)


def _pooled_summary(rows: list) -> dict:
    gt_rows = [r for r in rows if not r.get("false_pitch")]
    m = [r for r in gt_rows if r.get("matched")]
    ws = [r for r in m if r.get("v_release") is not None]
    cov = [r["in_ci90"] for r in ws if r.get("in_ci90") is not None]
    n_false = sum(1 for r in rows if r.get("false_pitch"))
    return dict(n_gt=len(gt_rows), n_matched=len(m), n_false=n_false,
                recall=len(m) / len(gt_rows) if gt_rows else None,
                precision=len(m) / (len(m) + n_false) if (m or n_false) else None,
                n_with_speed=len(ws), speed=_stats([r["err"] for r in ws]),
                speed_pct=_stats([r["err_pct"] for r in ws]),
                coverage90=dict(n=len(cov), frac=(sum(cov) / len(cov)) if cov else None),
                z=_stats([r.get("z") for r in ws]),
                release_ms=_stats([r.get("dt_release_ms") for r in m]),
                catch_ms=_stats([r.get("dt_catch_ms") for r in m]))


def format_dir_summary(d: dict) -> str:
    o = d["overall"]
    out = [f"Suite evaluation: {d['suite_dir']}",
           "  clips: " + ", ".join(f"{k}={v}" for k, v in sorted(d["status"].items())),
           f"  pitches: GT {o['n_gt']}, matched {o['n_matched']} (recall {_fmt(o['recall'], '.0%')}), "
           f"false {o['n_false']} (precision {_fmt(o['precision'], '.0%')}), with speed {o['n_with_speed']}",
           _stat_line("release speed error", o["speed"], "mph"),
           f"  90% coverage {_fmt(o['coverage90']['frac'], '.0%')} (n={o['coverage90']['n']}), "
           f"z RMS {_fmt(o['z'].get('rms'), '.2f')}",
           _stat_line("release time error", o["release_ms"], "ms"),
           _stat_line("catch time error", o["catch_ms"], "ms")]
    for key, groups in d["by"].items():
        out.append(f"  by {key}:")
        for g, s in groups.items():
            out.append(f"    {str(g):>10}: GT {s['n_gt']:3d} recall {_fmt(s['recall'], '.0%'):>4} "
                       f"false {s['n_false']:2d}  err bias {_fmt(s['speed']['mean'], '+.2f'):>6} "
                       f"RMS {_fmt(s['speed']['rms'], '.2f'):>5}  cov90 {_fmt(s['coverage90']['frac'], '.0%'):>4}")
    bad = [c for c in d["clips"] if c["status"] != "ok"]
    for c in bad:
        out.append(f"  clip {c.get('name')}: {c['status']}" + (f" ({c['error']})" if c.get("error") else ""))
    return "\n".join(out)


def write_pitch_csv(d: dict, path: str) -> None:
    """One row per GT pitch / false pitch of evaluate_dir() output."""
    cols = ["clip", "view", "level", "codec", "field_order", "shutter", "fence", "gt_id", "number", "matched",
            "false_pitch", "t_release_gt", "t_release", "dt_release_ms", "dt_catch_ms", "v_release_gt",
            "v_release", "err", "err_pct", "sigma", "z", "in_ci90", "label", "method", "reason", "cause"]
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in d["pitch_rows"]:
            w.writerow({k: r.get(k) for k in cols})
    os.replace(tmp, path)
