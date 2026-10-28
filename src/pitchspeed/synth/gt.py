"""Ground truth of a synthetic clip: frame map, landmarks, calibration file, GT JSON.

Files written next to a clip `<clip>` (the full file name, e.g. `a.mpg`):

  <clip>.gt.json           ground truth (schema GT_SCHEMA, synth.md section 6)
  <clip>.calib.json        pitchspeed.calib.model.CalibrationFile, one segment,
                           clicks = exact projections (+ optional click noise)
  <clip>.calib_exact.json  the noise-free calibration (when click noise > 0)
  <clip>.cams.npz          per exposure: k, t, R (world->camera), zoom, H_to_ref (3x3, exposure px ->
                           reference-frame px, exact for the rotation+zoom camera motion); and the
                           reference camera R_ref, zoom_ref, K_ref, t_ref; camera centre C

Calibration point conventions (constants.py / calib/model.py docstring):
'rubber' = rubber MIDDLE (0, D + 0.25, mound h); 'plate' = plate MIDDLE
(0, 17/24, 0); 'release' / 'catch' = projections of the clip's MEAN release /
catch positions; 'rubber_left/right' = (-/+1, D + 0.25, h) and
'plate_left/right' = the front corners (-/+8.5/12, 17/12, 0), named by their
IMAGE x (smaller u is 'left'); 'vertical_top/bottom' = two points on the
scene's vertical pole (side views).  Points are projected with the camera of
the reference frame (the camera at the frame's mean exposure time).
Settings.sar = true PAR (x_square = x * sar; the renderer uses fy = fx * PAR).
Settings.extension_ft = None on purpose: the analyzer must assume it; the GT
keeps the true release position.
"""
from __future__ import annotations

import json
import math
import os
import platform
from typing import Optional

import numpy as np

from pitchspeed import constants as C
from pitchspeed.calib.model import CalibrationFile, Segment, Settings
from pitchspeed.synth.config import (FIELD_DT, FPS_DEN, FPS_NUM, GENERATOR_VERSION, MPH, field_time, level_of,
                                     n_source_frames, par_of, rng_for)
from pitchspeed.synth.scene import pole_geometry

GT_SCHEMA = "pitchspeed.synth.gt/1"

CONVENTIONS = dict(
    units="feet, seconds, ft/s; mph only for speeds named *_mph",
    world="origin = plate apex (back point) on the ground; +y to the pitcher; +x to the 1B side; +z up",
    camera="OpenCV pinhole: X_cam = R (X - C); u = fx x/z + cx, v = fy y/z + cy; fy = fx * PAR; "
           "cx = (W-1)/2, cy = (H-1)/2",
    pixels="frame pixel coords of stored (non-square) pixels; origin top-left; integer = pixel centre; "
           "v = frame row; field row r of parity p is frame row 2r + p; square px = (u * PAR, v)",
    time="t = 0 is the exposure midpoint of the temporally first field of source frame 0; "
         "field k is exposed at t_k = k * 1001/60000 s (progressive30: frame n at 2n*1001/60000)",
    video_time="analyzer time of out frame j is j * out_frame_dt; equal to GT time when "
               "frame_map_identity is true (else map through frame_map)",
    u_v="u, v: ball centre at mid-exposure; u_vis, v_vis: exposure-time-weighted centroid of the "
        "visible part of the streak (use for fields straddling release/catch); u_blob, v_blob: "
        "centroid of the rendered coverage weighted by its visibility (not hidden by actors, times the "
        "backstop-fence transmission 1 - alpha) = what an ideal detector sees; occluded_frac: share of the "
        "coverage hidden by actors; fence_alpha: coverage-weighted mean fence opacity over the ball; u_ref, v_ref: mid-exposure centre "
        "projected with the calibration reference frame's camera (stabilized coordinates)",
)


# ------------------------------------------------------------------ paths
def gt_path_for(video_path: str) -> str:
    return str(video_path) + ".gt.json"


def calib_path_for(video_path: str, exact: bool = False) -> str:
    return str(video_path) + (".calib_exact.json" if exact else ".calib.json")


def cams_path_for(video_path: str) -> str:
    return str(video_path) + ".cams.npz"


def video_path_from_gt(gt_path: str, gt: Optional[dict] = None) -> str:
    """Clip path for a GT file (the GT sits next to the clip)."""
    p = str(gt_path)
    if p.endswith(".gt.json"):
        return p[: -len(".gt.json")]
    if gt is not None:
        return os.path.join(os.path.dirname(p), gt["video"]["file"])
    raise ValueError(f"not a GT file name: {gt_path}")


# ------------------------------------------------------------------ frame map
def out_frame_dt(cfg: dict) -> float:
    V = cfg["video"]
    if V["scan"] == "progressive60" or V["postprocess"] == "yadif_bob":
        return FIELD_DT
    return 2 * FIELD_DT


def _exposures(cfg: dict, n: int) -> list:
    """[(k, parity)] of the out frame(s) made from source frame n (one list per out frame)."""
    V = cfg["video"]
    scan, post, tff = V["scan"], V["postprocess"], V["field_order"] == "tff"
    if scan == "progressive30":
        return [[(2 * n, -1)]]
    if scan == "progressive60":
        return [[(n, -1)]]
    p0, p1 = (0, 1) if tff else (1, 0)
    if post is None:
        return [[(2 * n, p0), (2 * n + 1, p1)]]
    if post == "blend":
        return [[(2 * n, -1), (2 * n + 1, -1)]]
    if post in ("field_discard", "yadif"):
        return [[(2 * n, -1)]]
    if post == "yadif_bob":
        return [[(2 * n, -1)], [(2 * n + 1, -1)]]
    raise ValueError(f"unknown postprocess {post!r}")


def build_frame_map(cfg: dict) -> list:
    """Out frame -> source frame and exposure times (applies capture drops / duplicates)."""
    cap = cfg["capture"]
    drops, dups = set(int(x) for x in cap.get("drop_frames") or []), set(int(x) for x in cap.get("dup_frames") or [])
    dt = out_frame_dt(cfg)
    fmap = []
    for n in range(n_source_frames(cfg)):
        if n in drops:
            continue
        for rep in range(2 if n in dups else 1):
            for exp in _exposures(cfg, n):
                j = len(fmap)
                fmap.append(dict(out_frame=j, src_frame=n, t_video=field_time(j * round(dt / FIELD_DT)),
                                 field_index=[k for k, _ in exp], parity=[p for _, p in exp],
                                 exposure_times=[field_time(k) for k, _ in exp], dup=rep > 0))
    return fmap


def frame_map_is_identity(fmap: list, cfg: dict) -> bool:
    return all(abs(e["t_video"] - e["exposure_times"][0]) < 1e-9 for e in fmap)


def choose_reference(cfg: dict, fmap: list) -> dict:
    """Frame map entry of the calibration reference frame (inside the clear-field period)."""
    clear = float(cfg["calibration"]["clear_field_s"])
    target = 0.5 * clear
    cands = [e for e in fmap if not e["dup"] and max(e["exposure_times"]) < clear - 0.05]
    if not cands:
        cands = [e for e in fmap if not e["dup"]] or fmap
    return min(cands, key=lambda e: abs(e["exposure_times"][0] - target))


# ------------------------------------------------------------------ landmarks
def landmarks_world(cfg: dict) -> dict:
    lev = level_of(cfg)
    D, h = lev.distance_ft, lev.mound_h_ft
    ry = D + C.RUBBER_CENTER_OFFSET_FT
    hw = C.PLATE_WIDTH_FT / 2
    return {
        "rubber_center": [0.0, ry, h], "rubber_end_3b": [-1.0, ry, h], "rubber_end_1b": [1.0, ry, h],
        "rubber_front_center": [0.0, D, h], "rubber_front_3b": [-1.0, D, h], "rubber_front_1b": [1.0, D, h],
        "plate_center": [0.0, C.PLATE_CENTER_Y_FT, 0.0], "plate_back_point": [0.0, 0.0, 0.0],
        "plate_front_center": [0.0, C.PLATE_FRONT_Y_FT, 0.0],
        "plate_front_3b": [-hw, C.PLATE_FRONT_Y_FT, 0.0], "plate_front_1b": [hw, C.PLATE_FRONT_Y_FT, 0.0],
        "plate_side_3b": [-hw, hw, 0.0], "plate_side_1b": [hw, hw, 0.0],
    }


def _in_image(uv, W: int, H: int, margin: float = 0.0) -> bool:
    return bool(-0.5 + margin <= uv[0] <= W - 0.5 - margin and -0.5 + margin <= uv[1] <= H - 0.5 - margin)


def _pole_points(cfg: dict, renderer) -> Optional[tuple]:
    """(top, bottom) world points on the pole whose projections are inside the image, or None."""
    pole = pole_geometry(cfg, renderer.lev)
    if pole is None:
        return None
    zs = np.linspace(0.0, pole["height"], 241)
    P = np.stack([np.full_like(zs, pole["x"]), np.full_like(zs, pole["y"]), zs], axis=1)
    uv = renderer.project_ref(P)
    ok = [i for i in range(len(zs)) if _in_image(uv[i], renderer.W, renderer.H, margin=4.0)]
    if len(ok) < 2:
        return None
    lo, hi = ok[0], ok[-1]
    lo = min(lo + 2, hi)                         # a little above the base
    hi = max(hi - 2, lo)
    if abs(uv[hi][1] - uv[lo][1]) < 0.08 * renderer.H:           # too short to define a direction
        return None
    return P[hi], P[lo]


def calib_world_points(cfg: dict, renderer) -> dict:
    """World positions of every calibration point (before left/right naming by image x)."""
    lm = landmarks_world(cfg)
    ev = renderer.events
    lev = renderer.lev
    if ev.pitches:
        rel = np.mean([p.release_pos for p in ev.pitches], axis=0)
        cat = np.mean([p.catch_pos for p in ev.pitches], axis=0)
    else:                                        # no pitches: nominal positions
        P = cfg["pitches"]
        rel = np.array([0.0, lev.distance_ft - np.mean(P["extension_ft"]), lev.mound_h_ft + np.mean(P["release_z_ft"])])
        cat = np.array([0.0, -np.mean(P["catch_behind_ft"]), np.mean(P["target_z_ft"]) - 0.3])
    pts = {"rubber": lm["rubber_center"], "plate": lm["plate_center"],
           "release": rel.tolist(), "catch": cat.tolist()}
    pairs = {"rubber": (lm["rubber_end_3b"], lm["rubber_end_1b"]),
             "plate": (lm["plate_front_3b"], lm["plate_front_1b"])}
    for name, (a, b) in pairs.items():
        ua = renderer.project_ref([a])[0][0]
        ub = renderer.project_ref([b])[0][0]
        left, right = (a, b) if ua <= ub else (b, a)
        pts[name + "_left"], pts[name + "_right"] = left, right
    pp = _pole_points(cfg, renderer)
    if pp is not None:
        pts["vertical_top"], pts["vertical_bottom"] = pp[0].tolist(), pp[1].tolist()
    return {k: [float(x) for x in v] for k, v in pts.items()}


def calib_view(cfg: dict, renderer) -> str:
    v = cfg["camera"]["view"]
    if v in ("side", "behind"):
        return v
    axis = renderer.cam.R0[2]                    # optical axis (world)
    horiz = math.hypot(axis[0], axis[1])
    return "behind" if horiz > 0 and abs(axis[1]) / horiz > math.cos(math.radians(45)) else "side"


def make_calibration(cfg: dict, renderer, ref_entry: dict, video_fp: dict) -> tuple:
    """(exact CalibrationFile, clicked CalibrationFile, info dict for the GT)."""
    W, H = renderer.W, renderer.H
    ccfg = cfg["calibration"]
    world = calib_world_points(cfg, renderer)
    names = list(world)
    sel = ccfg.get("points", "all")
    view = calib_view(cfg, renderer)
    if sel == "required":
        from pitchspeed.calib.model import REQUIRED_POINTS
        names = [n for n in names if n in REQUIRED_POINTS[view]]
    elif isinstance(sel, (list, tuple)):
        names = [n for n in names if n in sel]
    exact, omitted = {}, []
    for n in names:
        uv = renderer.project_ref([world[n]])[0]
        if renderer.ref_depth([world[n]])[0] > 0.5 and _in_image(uv, W, H):
            exact[n] = [float(uv[0]), float(uv[1])]
        else:
            omitted.append(n)
    sigma = float(ccfg.get("click_noise_px") or 0.0)
    par = par_of(cfg)
    rng = rng_for(cfg["seed"], 9)
    clicked = {}
    for n in sorted(exact):                      # sorted: noise independent of dict order
        dx, dy = rng.normal(0.0, 1.0, 2) * sigma if sigma > 0 else (0.0, 0.0)
        x = exact[n][0] + dx / par               # sigma is isotropic in SQUARE pixels
        y = exact[n][1] + dy
        clicked[n] = [float(min(max(x, -0.5), W - 0.5)), float(min(max(y, -0.5), H - 0.5))]
    lev = renderer.lev
    st = Settings(distance_ft=lev.distance_ft, level=lev.key if lev.key != "custom" else None,
                  sport="baseball", extension_ft=None, sar=par)
    notes = (f"synthetic clip (pitchspeed.synth, seed {cfg['seed']}); points projected from the world model"
             + (f" + click noise sigma {sigma:g} square px" if sigma > 0 else " (exact)"))

    def cf(points: dict, note: str) -> CalibrationFile:
        seg = Segment(start_frame=0, end_frame=None, ref_frame=int(ref_entry["out_frame"]), view=view,
                      points={k: list(points[k]) for k in sorted(points)})
        return CalibrationFile(settings=st, segments=[seg], video=dict(video_fp), image_size=(W, H), notes=note)

    info = dict(view=view, ref_frame=int(ref_entry["out_frame"]), ref_src_frame=int(ref_entry["src_frame"]),
                ref_exposure_times=list(ref_entry["exposure_times"]), click_noise_px=sigma,
                points_world_ft=world, points_exact=exact, points_clicked=clicked, omitted=omitted)
    return cf(exact, notes.split(";")[0] + "; exact projections"), cf(clicked, notes), info


# ------------------------------------------------------------------ pitches
def _f(x) -> float:
    return float(x)


def pitch_gt(p, records: list, cfg: dict) -> dict:
    lev = level_of(cfg)
    fdt = 2 * FIELD_DT
    fl = [dict(r) for r in records if r.get("pitch_id") == p.id]
    for r in fl:
        r.pop("pitch_id", None)
    return dict(
        id=p.id, t_release=_f(p.t_release), t_catch=_f(p.t_catch), t_plate_front=_f(p.t_plate),
        t_hand=_f(p.t_hand), flight_time=_f(p.t_catch - p.t_release),
        release_frame_float=_f(p.t_release / fdt), catch_frame_float=_f(p.t_catch / fdt),
        release_field_float=_f(p.t_release / FIELD_DT), catch_field_float=_f(p.t_catch / FIELD_DT),
        dist_y_ft=_f(p.release_pos[1] - p.catch_pos[1]), path_len_ft=_f(p.path_len_ft),
        release_pos_ft=[_f(x) for x in p.release_pos], release_vel_ftps=[_f(x) for x in p.release_vel],
        catch_pos_ft=[_f(x) for x in p.catch_pos], catch_vel_ftps=[_f(x) for x in p.catch_vel],
        extension_ft=_f(lev.distance_ft - p.release_pos[1]),
        release_height_ft=_f(p.release_pos[2]), catch_height_ft=_f(p.catch_pos[2]),
        catch_behind_ft=_f(p.catch_behind_ft),
        speed_release_mph=_f(np.linalg.norm(p.release_vel) / MPH),
        speed_release_horiz_mph=_f(math.hypot(p.release_vel[0], p.release_vel[1]) / MPH),
        speed_plate_front_mph=_f(np.linalg.norm(p.plate_vel) / MPH),
        speed_catch_mph=_f(np.linalg.norm(p.catch_vel) / MPH),
        avg_speed_y_mph=_f((p.release_pos[1] - p.catch_pos[1]) / (p.t_catch - p.t_release) / MPH),
        avg_speed_path_mph=_f(p.path_len_ft / (p.t_catch - p.t_release) / MPH),
        physics=dict(k_per_ft=_f(p.k_per_ft), cd=_f(p.cd), magnus_ftps2=[_f(x) for x in p.magnus],
                     wind_ftps=[_f(x) for x in p.wind], drag="a = -k|v-w|(v-w) + a_M - g z"),
        swing=bool(p.swing), target_xz_ft=[_f(x) for x in p.target], fields=fl)


def distractors_gt(cfg: dict, renderer) -> list:
    ev = renderer.events
    out = []
    for th in ev.throws:
        out.append(dict(type="throwback", t_start=_f(th.t_start), t_end=_f(th.t_end), speed_mph=_f(th.speed_mph),
                        after_pitch=th.after_pitch, p0_ft=[_f(x) for x in th.p0], p1_ft=[_f(x) for x in th.p1]))
    if cfg["distractors"]["batter"]:
        for p in ev.pitches:
            if p.swing:
                a, b = p.swing_window
                out.append(dict(type="swing", pitch_id=p.id, t_start=_f(a), t_end=_f(b)))
    for b in ev.birds:
        out.append(dict(type="bird", t_start=_f(b.t0), t_end=_f(b.t0 + b.dur), p0_ft=[_f(x) for x in b.p0],
                        v_ftps=[_f(x) for x in b.v]))
    for p in cfg["shake"].get("pans") or []:
        out.append(dict(type="pan", t_start=_f(p["t"]), t_end=_f(p["t"] + p["dur"]),
                        yaw_deg=_f(p.get("yaw_deg", 0.0)), pitch_deg=_f(p.get("pitch_deg", 0.0))))
    for z in cfg["shake"].get("zooms") or []:
        out.append(dict(type="zoom", t_start=_f(z["t"]), t_end=_f(z["t"] + z["dur"]), factor=_f(z["factor"])))
    if cfg["calibration"]["clear_field_s"] > 0:
        out.append(dict(type="people_appear", t_start=_f(cfg["calibration"]["clear_field_s"]),
                        t_end=_f(cfg["calibration"]["clear_field_s"])))
    return sorted(out, key=lambda d: d["t_start"])


def video_fingerprint(path: str, n_frames: int, W: int, H: int) -> dict:
    size = os.path.getsize(path) if os.path.exists(path) else 0
    return dict(name=os.path.basename(path), size=int(size), n_frames=int(n_frames), width=int(W), height=int(H))


def build_gt(cfg: dict, renderer, records: list, fmap: list, video: dict, calib_info: dict,
             landmarks_px: dict, warnings: list, timing: dict, ffmpeg_ver: str) -> dict:
    cam = renderer.cam
    V = cfg["video"]
    lev = renderer.lev
    import cv2
    return dict(
        schema=GT_SCHEMA,
        generator=dict(version=GENERATOR_VERSION, ffmpeg=ffmpeg_ver, python=platform.python_version(),
                       numpy=np.__version__, opencv=cv2.__version__),
        seed=int(cfg["seed"]), config=cfg, conventions=CONVENTIONS,
        video=dict(video, width=renderer.W, height=renderer.H, scan=V["scan"],
                   field_order=V["field_order"] if V["scan"] == "interlaced" else None,
                   metadata_field_order=V["metadata_field_order"], postprocess=V["postprocess"],
                   codec=V["codec"], bitrate_kbps=V["bitrate_kbps"],
                   fps="60000/1001" if out_frame_dt(cfg) < 0.02 else "30000/1001",
                   out_frame_dt=out_frame_dt(cfg), field_dt=FIELD_DT, par=list(V["par"]), sar=par_of(cfg),
                   n_src_frames=n_source_frames(cfg), n_out_frames=len(fmap),
                   interlaced_file=V["scan"] == "interlaced" and V["postprocess"] is None),
        level=dict(key=lev.key, distance_ft=lev.distance_ft, mound_h_ft=lev.mound_h_ft, base_ft=lev.base_ft),
        camera=dict(view=cfg["camera"]["view"], C=cam.C.tolist(), R0=cam.R0.tolist(), fx=cam.fx0, fy=cam.fy0,
                    cx=cam.cx, cy=cam.cy, hfov_deg=cam.hfov_deg(), k1=0.0, par=cam.par,
                    fence=dict(enabled=bool(renderer.canvas.has_fence), blur_px=renderer.canvas.fence_blur_px)),
        landmarks_world_ft=landmarks_world(cfg),
        landmarks_px=landmarks_px,
        calibration=calib_info,
        pitches=[pitch_gt(p, records, cfg) for p in renderer.events.pitches],
        pitches_requested=renderer.events.requested,
        distractors=distractors_gt(cfg, renderer),
        frame_map_identity=frame_map_is_identity(fmap, cfg),
        frame_map=fmap,
        warnings=list(warnings),
        timing=timing,
    )


def landmarks_px(cfg: dict, renderer, ref_entry: dict) -> dict:
    W, H = renderer.W, renderer.H
    pts, vis = {}, {}
    for name, P in landmarks_world(cfg).items():
        uv = renderer.project_ref([P])[0]
        pts[name] = [float(uv[0]), float(uv[1])]
        vis[name] = bool(renderer.ref_depth([P])[0] > 0.5 and _in_image(uv, W, H))
    return dict(ref_frame=int(ref_entry["out_frame"]), points=pts, in_image=vis)


# ------------------------------------------------------------------ I/O
def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _clean(o):
    """Replace NaN/inf by None so the JSON is standard."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def write_json_atomic(obj, path: str, indent: Optional[int] = 1) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_clean(json.loads(json.dumps(obj, default=_json_default))), f, indent=indent,
                  allow_nan=False)
    os.replace(tmp, path)


def load_gt(path: str) -> dict:
    """Load a GT file (or the GT of a clip, given the clip path)."""
    p = str(path)
    if not p.endswith(".json"):
        p = gt_path_for(p)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cams(path: str, cams: list, renderer) -> None:
    """Per-exposure camera states + homography to the reference frame."""
    if not cams:
        np.savez_compressed(path, k=np.zeros(0, int))
        return
    k = np.array([c[0] for c in cams], dtype=np.int64)
    t = np.array([c[1] for c in cams])
    R = np.array([c[2] for c in cams])
    z = np.array([c[3] for c in cams])
    Rref, zref = renderer.ref_cam
    cam = renderer.cam
    Kref = cam.K(zref)
    Hs = np.array([cam.homography(R[i], z[i], Rref, zref) for i in range(len(k))])
    tmp = str(path) + ".tmp.npz"
    np.savez_compressed(tmp, k=k, t=t, R=R, zoom=z, H_to_ref=Hs, R_ref=Rref, zoom_ref=zref, K_ref=Kref, C=cam.C,
                        t_ref=float(np.mean(renderer.ref_times)))
    os.replace(tmp, path)


__all__ = ["GT_SCHEMA", "gt_path_for", "calib_path_for", "cams_path_for", "load_gt", "build_frame_map",
           "choose_reference", "make_calibration", "build_gt", "landmarks_world", "out_frame_dt",
           "FPS_NUM", "FPS_DEN"]
