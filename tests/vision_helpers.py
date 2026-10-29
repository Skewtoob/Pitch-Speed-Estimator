"""Test helpers for the vision area.

* `iter_fields(path, field_order)`: PyAV decode + field split (no dependency
  on pitchspeed.io).
* `proto_calibration(gt_path)`: a CalibrationFile for a *prototype*-format
  synthetic clip ($PITCHSPEED_SCRATCH/gen), built by projecting world points with
  the GT camera of the reference frame.
* `proto_gt_tracks(gt_path, ref_frame)`: per-pitch GT ball positions mapped
  into reference-frame coordinates with the exact rotation homography.
* `synth_gt_tracks(gt)`: the same for clips written by pitchspeed.synth
  (fields carry u_ref / v_ref already).
* `run_finder(...)`: drive a PitchFinder over a clip.
"""
from __future__ import annotations

import json
import os
from fractions import Fraction
from typing import Iterator, Optional

import numpy as np

from pitchspeed.calib.model import CalibrationFile, Segment, Settings
from pitchspeed.constants import NTSC_FRAME_DT
from pitchspeed.types import Sample, ScanInfo

SCRATCH = os.environ.get("PITCHSPEED_SCRATCH", os.path.join(os.path.dirname(__file__), "_media"))
GEN_DIR = os.path.join(SCRATCH, "gen")
SYNTH_DIR = os.path.join(SCRATCH, "impl-synth")
VISION_SCRATCH = os.path.join(SCRATCH, "impl-vision")


# ------------------------------------------------------------- decoding ----
def iter_frames_gray(path: str, start: int = 0, stop: Optional[int] = None) -> Iterator[tuple[int, np.ndarray]]:
    import av
    with av.open(path) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        i = 0
        for fr in c.decode(st):
            if stop is not None and i >= stop:
                break
            if i >= start:
                yield i, fr.to_ndarray(format="gray")
            i += 1


def iter_fields(path: str, field_order: str = "bff", start_frame: int = 0, stop_frame: Optional[int] = None,
                frame_dt: float = NTSC_FRAME_DT) -> Iterator[Sample]:
    """Fields of an interlaced clip as Samples (t = frame * frame_dt + k * frame_dt / 2)."""
    first = 0 if field_order == "tff" else 1
    for i, g in iter_frames_gray(path, start_frame, stop_frame):
        rows = g.shape[0] - g.shape[0] % 2
        for k in (0, 1):
            p = k ^ first
            yield Sample(s=2 * i + k, frame=i, k=k, parity=p, t=i * frame_dt + k * frame_dt / 2.0,
                         image=np.ascontiguousarray(g[p:rows:2]))


def get_frame(path: str, index: int) -> np.ndarray:
    for i, g in iter_frames_gray(path, index, index + 1):
        return g
    raise IndexError(index)


def scan_for(field_order: str) -> ScanInfo:
    return ScanInfo(mode="interlaced", field_order=field_order, confidence="high", source="override",
                    sample_rate=1.0 / (NTSC_FRAME_DT / 2))


# ----------------------------------------------------- prototype clips ----
def _proto_cams(gt_path: str):
    z = np.load(gt_path.replace(".gt.json", ".cams.npz"))
    return z["k"], z["t"], z["R"], z["zoom"]


def _mean_rotation(Rs: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.mean(Rs, axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def proto_camera(gt: dict, gt_path: str, ref_frame: int):
    cam = gt["camera"]
    k, _t, Rs, zoom = _proto_cams(gt_path)
    idx = [int(np.nonzero(k == 2 * ref_frame + j)[0][0]) for j in (0, 1)]
    R_ref = _mean_rotation(Rs[idx])
    z_ref = float(np.mean(zoom[idx]))
    K = np.array([[cam["fx"] * z_ref, 0, cam["cx"]], [0, cam["fy"] * z_ref, cam["cy"]], [0, 0, 1.0]])
    return np.asarray(cam["C"], float), R_ref, K, (k, Rs, zoom)


def _project(C, R, K, X) -> np.ndarray:
    Xc = (np.atleast_2d(X) - C) @ R.T
    uv = Xc @ K.T
    return uv[:, :2] / uv[:, 2:3]


def proto_calibration(gt_path: str, ref_frame: int = 10, level: str = "hs") -> CalibrationFile:
    """CalibrationFile for a prototype-format clip (exact clicks)."""
    gt = json.load(open(gt_path, encoding="utf-8"))
    C, R, K, _ = proto_camera(gt, gt_path, ref_frame)
    lw = gt["landmarks_world_ft"]
    D = float(lw["rubber_front_center"][1])
    h = float(lw["rubber_front_center"][2])
    par = Fraction(*gt["video"]["par"])
    rel = np.mean([p["release_pos_ft"] for p in gt["pitches"]], axis=0)
    cat = np.mean([p["catch_pos_ft"] for p in gt["pitches"]], axis=0)
    world = {"rubber": [0.0, D + 0.25, h], "plate": [0.0, 17.0 / 24.0, 0.0],
             "release": list(rel), "catch": list(cat)}
    view = gt["config"]["camera"]["view"]
    pts = {n: _project(C, R, K, np.array(v))[0].tolist() for n, v in world.items()}
    ends = {"r": [[-1.0, D + 0.25, h], [1.0, D + 0.25, h]],
            "p": [[-8.5 / 12, 17.0 / 12, 0.0], [8.5 / 12, 17.0 / 12, 0.0]]}
    if view == "behind":
        for key, name in (("r", "rubber"), ("p", "plate")):
            a, b = (_project(C, R, K, np.array(v))[0] for v in ends[key])
            lft, rgt = (a, b) if a[0] <= b[0] else (b, a)
            pts[f"{name}_left"] = lft.tolist()
            pts[f"{name}_right"] = rgt.tolist()
    st = Settings(distance_ft=D, level=level, sar=float(par))
    seg = Segment(start_frame=0, end_frame=None, ref_frame=ref_frame, view=view, points=pts)
    return CalibrationFile(settings=st, segments=[seg], image_size=(720, 480),
                           notes="test calibration from prototype GT")


def proto_gt_tracks(gt_path: str, ref_frame: int = 10) -> list[dict]:
    """Per pitch: arrays s, t, x_ref, y_ref, state, occluded, plus t_release / t_catch."""
    gt = json.load(open(gt_path, encoding="utf-8"))
    C, R_ref, K_ref, (k, Rs, zoom) = proto_camera(gt, gt_path, ref_frame)
    cam = gt["camera"]
    out = []
    for p in gt["pitches"]:
        rows = []
        for f in p["fields"]:
            if f.get("u_vis") is None:
                continue
            j = int(np.nonzero(k == f["field_index"])[0][0])
            Kk = np.array([[cam["fx"] * zoom[j], 0, cam["cx"]], [0, cam["fy"] * zoom[j], cam["cy"]], [0, 0, 1.0]])
            Hm = K_ref @ R_ref @ Rs[j].T @ np.linalg.inv(Kk)
            q = Hm @ np.array([f["u_vis"], f["v_vis"], 1.0])
            rows.append((f["field_index"], f["t"], q[0] / q[2], q[1] / q[2], f["state"], f["occluded"]))
        out.append({"id": p["id"], "t_release": p["t_release"], "t_catch": p["t_catch"],
                    "speed_mph": p["speed_release_mph"],
                    "s": np.array([r[0] for r in rows]), "t": np.array([r[1] for r in rows]),
                    "x": np.array([r[2] for r in rows]), "y": np.array([r[3] for r in rows]),
                    "state": [r[4] for r in rows], "occluded": np.array([r[5] for r in rows])})
    return out


# ---------------------------------------------------------- synth clips ----
def synth_gt_tracks(gt: dict) -> list[dict]:
    out = []
    for p in gt["pitches"]:
        fs = [f for f in p["fields"] if f.get("u_ref") is not None]
        out.append({"id": p["id"], "t_release": p["t_release"], "t_catch": p["t_catch"],
                    "speed_mph": p["speed_release_mph"],
                    "s": np.array([f["field_index"] for f in fs]), "t": np.array([f["t"] for f in fs]),
                    "x": np.array([f["u_ref"] for f in fs]), "y": np.array([f["v_ref"] for f in fs]),
                    "state": [f["state"] for f in fs],
                    "occluded": np.array([bool(f.get("occluded")) for f in fs])})
    return out


# ------------------------------------------------------------ run finder ----
def run_finder(video: str, calib_file: CalibrationFile, field_order: str, opts=None,
               start_frame: int = 0, stop_frame: Optional[int] = None):
    from pitchspeed.vision.finder import FinderOptions, PitchFinder
    calib = calib_file.segment_calibs()[0]
    ref = get_frame(video, calib.ref_frame)
    scan = scan_for(field_order)
    sample_dt = NTSC_FRAME_DT / 2
    shape = (ref.shape[0] // 2, ref.shape[1])
    f = PitchFinder(calib, scan, sample_dt, shape, ref, opts or FinderOptions())
    for smp in iter_fields(video, field_order, start_frame, stop_frame):
        f.process(smp)
    return f.finish(), f


def match_tracks(result, gt_tracks: list[dict], tol_s: float = 0.25) -> dict:
    """Match accepted tracks to GT pitches by time overlap; report errors."""
    rows = []
    used = set()
    for g in gt_tracks:
        best, bo = None, 0.0
        for i, tr in enumerate(result.tracks):
            if i in used or not tr.points:
                continue
            lo = max(tr.t_first, g["t_release"] - tol_s)
            hi = min(tr.t_last, g["t_catch"] + tol_s)
            ov = hi - lo
            if ov > bo:
                best, bo = i, ov
        if best is None:
            rows.append({"id": g["id"], "found": False})
            continue
        used.add(best)
        tr = result.tracks[best]
        gs = {int(s): (x, y) for s, x, y in zip(g["s"], g["x"], g["y"])}
        err = []
        for p in tr.points:
            if p.inlier and not p.manual and p.s in gs:
                gx, gy = gs[p.s]
                err.append((p.x - gx, p.y - gy))
        err = np.array(err) if err else np.zeros((0, 2))
        rows.append({"id": g["id"], "found": True, "track": best, "n_pts": len(tr.points),
                     "n_matched": len(err),
                     "rms_x": float(np.sqrt(np.mean(err[:, 0] ** 2))) if len(err) else float("nan"),
                     "rms_y": float(np.sqrt(np.mean(err[:, 1] ** 2))) if len(err) else float("nan"),
                     "bias_x": float(np.mean(err[:, 0])) if len(err) else float("nan"),
                     "bias_y": float(np.mean(err[:, 1])) if len(err) else float("nan"),
                     "rel_err_ms": 1000 * (tr.release.t - g["t_release"]) if tr.release else float("nan"),
                     "catch_err_ms": 1000 * (tr.catch.t - g["t_catch"]) if tr.catch else float("nan"),
                     "rel_sigma_ms": 1000 * tr.release.sigma if tr.release else float("nan"),
                     "catch_sigma_ms": 1000 * tr.catch.sigma if tr.catch else float("nan"),
                     "rel_method": tr.release.method if tr.release else None,
                     "catch_method": tr.catch.method if tr.catch else None,
                     "confidence": tr.confidence, "flags": list(tr.flags)})
    fp = [i for i in range(len(result.tracks)) if i not in used]
    return {"pitches": rows, "false_positives": fp,
            "recall": sum(r["found"] for r in rows) / max(1, len(rows))}


# ------------------------------------------------------- in-memory scenes ----
def textured_background(h: int = 480, w: int = 720, seed: int = 0) -> np.ndarray:
    """Smooth random texture with corners (good for feature tracking), float32 0..255."""
    import cv2
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), np.float32)
    for sigma, amp in ((24, 40.0), (6, 25.0), (2, 12.0)):
        n = rng.normal(0, 1, (h, w)).astype(np.float32)
        img += amp * cv2.GaussianBlur(n, (0, 0), sigma) * (sigma ** 0.5)
    for _ in range(60):                      # rectangles = strong corners
        x0, y0 = int(rng.integers(0, w - 40)), int(rng.integers(0, h - 30))
        cv2.rectangle(img, (x0, y0), (x0 + int(rng.integers(8, 40)), y0 + int(rng.integers(6, 30))),
                      float(rng.uniform(-40, 40)), -1)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    return np.clip(110 + img, 5, 250).astype(np.float32)


def side_calibration(distance_ft: float = 60.5, sar: float = 10 / 11) -> CalibrationFile:
    st = Settings(distance_ft=distance_ft, level="hs", sar=sar)
    seg = Segment(start_frame=0, ref_frame=0, view="side",
                  points={"rubber": [650.0, 272.0], "plate": [100.0, 282.0]})
    return CalibrationFile(settings=st, segments=[seg], image_size=(720, 480))





def make_side_scene(n_fields: int = 200, t_release: float = 1.5, speed_mph: float = 80.0, noise: float = 2.0,
                    shake_px: float = 1.5, ball_amp: float = 90.0, ball: bool = True, field_order: str = "bff",
                    seed: int = 1, throwback: bool = False):
    """Interlaced side-view scene with a textured, shaking background and one pitch.

    Returns (samples, calib_file, gt) with gt = {t_release, t_catch, traj(t) -> (x, y) reference px,
    A_true(s) -> 2x3 sample frame -> reference transform}.
    """
    import cv2
    from pitchspeed.constants import MPH_TO_FTPS
    rng = np.random.default_rng(seed)
    bg = textured_background(seed=seed)
    cf = side_calibration()
    cal = cf.segment_calibs()[0]
    P0 = cal.release_point()
    P1 = cal.catch_point()
    L_ft = cal.release_y_ft - cal.catch_y_ft
    T = L_ft / (speed_mph * MPH_TO_FTPS)
    t_catch = t_release + T
    dt = NTSC_FRAME_DT / 2
    first = 0 if field_order == "tff" else 1

    def traj(t):
        u = (np.asarray(t, float) - t_release) / T
        if throwback:
            u = 1.0 - u
        x = P0[0] + u * (P1[0] - P0[0])
        y = P0[1] + u * (P1[1] - P0[1]) + 12.0 * u * u
        return np.stack([np.atleast_1d(x), np.atleast_1d(y)], -1)

    def shake(t):
        return (shake_px * np.sin(2 * np.pi * 0.7 * t), 0.7 * shake_px * np.sin(2 * np.pi * 1.3 * t + 1.0),
                0.05 * np.sin(2 * np.pi * 0.4 * t))

    def A_true(t):
        # sample frame -> reference: rotate by -ang about the centre, then shift back
        dx, dy, ang = shake(t)
        c, s_ = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        cx, cy = 360.0, 240.0
        R = np.array([[c, -s_ * (1.0 / cal.sar)], [s_ * cal.sar, c]])   # rotation in square px, frame coords
        A = np.zeros((2, 3))
        A[:, :2] = R
        A[:, 2] = np.array([cx, cy]) - R @ np.array([cx + dx, cy + dy])
        return A

    def to3(A):
        return np.vstack([A, [0, 0, 1]])

    # reference system = frame 0 at its mean exposure time (as the Stabilizer defines it)
    A_mid = to3(0.5 * (A_true(0.0) + A_true(dt)))
    A_mid_inv = np.linalg.inv(A_mid)

    def A_ref(t):
        return (A_mid_inv @ to3(A_true(t)))[:2]

    samples = []
    ref_img = None
    n_frames = (n_fields + 1) // 2
    for i in range(n_frames):
        for k in (0, 1):
            s = 2 * i + k
            if s >= n_fields:
                break
            t = i * NTSC_FRAME_DT + k * dt
            A = A_true(t)
            img = cv2.warpAffine(bg, A, (720, 480), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REFLECT)
            if ball and t_release - 0.001 <= t <= t_catch:
                q = traj(t)[0]
                Ainv = np.linalg.inv(to3(A_ref(t)))[:2]
                bx, by = Ainv[:, :2] @ q + Ainv[:, 2]
                yy, xx = np.mgrid[0:480, 0:720]
                img += ball_amp * np.exp(-(((xx - bx) * cal.sar) ** 2 + (yy - by) ** 2) / (2 * 1.1 ** 2))
            img = img + rng.normal(0, noise, img.shape)
            frame = np.clip(img, 0, 255).astype(np.uint8)
            if s == 0:
                ref_img = frame
            p = k ^ first
            samples.append(Sample(s=s, frame=i, k=k, parity=p, t=t, image=np.ascontiguousarray(frame[p::2])))
    # reference frame = frame 0 with both fields (shake at the frame's mean time)
    full0 = []
    for k in (0, 1):
        t = k * dt
        A = A_true(t)
        img = cv2.warpAffine(bg, A, (720, 480), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REFLECT)
        full0.append(np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8))
    ref = np.empty((480, 720), np.uint8)
    ref[first::2] = full0[0][first::2]
    ref[1 - first::2] = full0[1][1 - first::2]
    gt = {"t_release": t_release, "t_catch": t_catch, "traj": traj, "A_true": lambda s: A_ref(
        samples[s].t if s < len(samples) else s * dt), "shake": shake, "T": T}
    return samples, cf, ref, gt


class ListSource:
    """Minimal SampleSource stand-in over an in-memory list of Samples."""

    def __init__(self, samples: list):
        self.samples = samples
        self.n_samples = len(samples)
        self.sample_dt = NTSC_FRAME_DT / 2
        self.sample_shape = samples[0].image.shape

    def sample_time(self, s: int) -> float:
        return self.samples[s].t

    def iter_samples(self, s0: int = 0, s1: Optional[int] = None, color: bool = False):
        s1 = self.n_samples if s1 is None else min(s1, self.n_samples)
        for s in range(max(0, s0), s1):
            yield self.samples[s]


# ------------------------------------------------------- behind-view helpers ----
def behind_calibration(back: float = 30.0, height: float = 7.0, f: float = 1500.0, D: float = 60.5,
                       sar: float = 10 / 11, w: int = 720, h: int = 480):
    """Calibration (segment) of a pinhole camera `back` ft behind the plate (clicks projected exactly)."""
    C = np.array([0.0, -back, height])
    z = np.array([0.0, 25.0, 3.0]) - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(z, x), z])

    def proj(X):
        Xc = (np.asarray(X, float) - C) @ R.T
        return [float(f * Xc[0] / Xc[2] / sar + (w - 1) / 2), float(f * Xc[1] / Xc[2] + (h - 1) / 2)]

    pts = {"release": proj([-1.5, 55.0, 5.6]), "catch": proj([0.0, -1.5, 2.2]),
           "rubber_left": proj([-1.0, D + 0.25, 0.83]), "rubber_right": proj([1.0, D + 0.25, 0.83]),
           "plate_left": proj([-17 / 24, 17 / 12, 0.0]), "plate_right": proj([17 / 24, 17 / 12, 0.0])}
    st = Settings(distance_ft=D, level="hs", sar=sar)
    seg = Segment(start_frame=0, ref_frame=0, view="behind", points=pts)
    return CalibrationFile(settings=st, segments=[seg], image_size=(w, h)).segment_calibs()[0]
