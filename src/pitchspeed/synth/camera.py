"""Pinhole camera, auto-framing and closed-form camera motion.

OpenCV convention: X_cam = R (X_world - C); u = fx x/z + cx, v = fy y/z + cy
with NON-square pixels fy = fx * PAR (PAR 10/11 for NTSC DV-style SD), so a
square-pixel position is (u * PAR, v), matching Settings.sar = PAR.
cx = (W-1)/2, cy = (H-1)/2: integer pixel coordinates are pixel centres.

Camera motion is rotation + zoom only (an exact homography of the
background canvas): a sum of sines with a ~1/f spectrum plus minimum-jerk
pans and zooms, all continuous functions of t, so every field is rendered
at its own exposure time.  Translational shake / parallax is not modelled.
"""
from __future__ import annotations

import math

import numpy as np

from pitchspeed.synth.config import LevelGeom, rng_for


def look_at(C, T, roll_deg: float = 0.0) -> np.ndarray:
    """World->camera rotation for a camera at C looking at T (z up in the world)."""
    f = np.asarray(T, dtype=float) - np.asarray(C, dtype=float)
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])
    a = math.radians(roll_deg)
    Rz = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    return Rz @ R


def rot_small(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Camera-frame rotation (radians): yaw about cam y, pitch about cam x, roll about cam z."""
    cy, sy, cp, sp, cr, sr = (math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch),
                              math.cos(roll), math.sin(roll))
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


class Camera:
    """Base (un-shaken) camera of a clip."""

    def __init__(self, cfg: dict, lev: LevelGeom, W: int, H: int, par: float):
        c = cfg
        rel_y = lev.distance_ft - 5.0
        if c["view"] == "side":
            mid = np.array([0.0, rel_y / 2.0, 4.0])
            s = 1.0 if c["side"] == "1b" else -1.0
            a = math.radians(c["angle_deg"])     # optical axis vs pitch line; <90 => shifted toward home
            C = mid + float(c["distance_ft"]) * np.array([s * math.sin(a), -math.cos(a), 0.0])
            C[2] = float(c["height_ft"])
            T = mid
        elif c["view"] == "behind":
            C = np.array([float(c["lateral_ft"]), -float(c["back_ft"]), float(c["height_ft"])])
            T = np.array([0.0, rel_y * 0.55, 3.5])
        else:
            C = np.asarray(c["C"], dtype=float)
            T = np.asarray(c["look_at"], dtype=float)
        self.view = c["view"]
        self.C = C
        self.R0 = look_at(C, T, float(c.get("roll_deg", 0.0)))
        self.W, self.H, self.par = int(W), int(H), float(par)
        self.cx, self.cy = (W - 1) / 2.0, (H - 1) / 2.0
        if c.get("hfov_deg") is None:
            # auto-fit rubber (ground and +7 ft) and the plate area (+ people heights)
            pts = np.array([[0, lev.distance_ft, 0], [0, lev.distance_ft, 7], [0, -3, 0], [0, -3, 6],
                            [0, -6, 0]], dtype=float)
            pc = (self.R0 @ (pts - C).T).T
            xn = max(float(np.abs(pc[:, 0] / pc[:, 2]).max()), 1e-6)
            yn = max(float(np.abs(pc[:, 1] / pc[:, 2]).max()), 1e-6)
            m = float(c["fit_margin"])
            fx = min((W / 2) * (1 - m) / xn, (H / 2) * (1 - m) / (yn * par))
        else:
            fx = (W / 2) / math.tan(math.radians(float(c["hfov_deg"])) / 2)
        self.fx0 = float(fx)
        self.fy0 = float(fx * par)
        self.K0 = self.K(1.0)

    def K(self, zoom: float) -> np.ndarray:
        return np.array([[self.fx0 * zoom, 0.0, self.cx], [0.0, self.fy0 * zoom, self.cy], [0.0, 0.0, 1.0]])

    def hfov_deg(self) -> float:
        return 2 * math.degrees(math.atan((self.W / 2) / self.fx0))

    def project(self, Rt: np.ndarray, zoom: float, P) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World points (n,3) -> (u, v, depth) arrays in frame pixel coordinates."""
        Xc = (Rt @ (np.atleast_2d(np.asarray(P, dtype=float)) - self.C).T).T
        Z = Xc[:, 2]
        u = self.fx0 * zoom * Xc[:, 0] / Z + self.cx
        v = self.fy0 * zoom * Xc[:, 1] / Z + self.cy
        return u, v, Z

    def homography(self, Rt_from: np.ndarray, zoom_from: float, Rt_to: np.ndarray, zoom_to: float) -> np.ndarray:
        """3x3 map of frame px seen by camera 'from' to frame px of camera 'to' (pure rotation + zoom)."""
        Hm = self.K(zoom_to) @ Rt_to @ Rt_from.T @ np.linalg.inv(self.K(zoom_from))
        return Hm / Hm[2, 2]


class Shake:
    """Closed-form continuous camera motion: sum of sines + minimum-jerk pans/zooms."""

    def __init__(self, cfg: dict, seed: int):
        r = rng_for(seed, 3)
        n = int(cfg["n_sines"])
        self.f = np.exp(r.uniform(math.log(cfg["fmin_hz"]), math.log(cfg["fmax_hz"]), (3, n)))
        self.ph = r.uniform(0, 2 * math.pi, (3, n))
        amp = np.array([cfg["amp_deg"], cfg["amp_deg"], cfg["roll_amp_deg"]], dtype=float)[:, None]
        w = 1.0 / self.f
        w /= np.sqrt((w ** 2).sum(1, keepdims=True))        # ~1/f spectrum, RMS = amp
        self.a = np.radians(amp) * w * math.sqrt(2)
        self.pans = list(cfg.get("pans") or [])
        self.zooms = list(cfg.get("zooms") or [])

    @staticmethod
    def mj(x: float) -> float:
        """Minimum-jerk ramp 0 -> 1 on x in [0, 1]."""
        x = min(max(x, 0.0), 1.0)
        return x ** 3 * (10 - 15 * x + 6 * x * x)

    def angles(self, t: float) -> np.ndarray:
        ang = (self.a * np.sin(2 * math.pi * self.f * t + self.ph)).sum(1)
        for p in self.pans:
            s = self.mj((t - float(p["t"])) / float(p["dur"]))
            ang[0] += math.radians(float(p.get("yaw_deg", 0.0))) * s
            ang[1] += math.radians(float(p.get("pitch_deg", 0.0))) * s
        return ang

    def zoom(self, t: float) -> float:
        z = 1.0
        for q in self.zooms:
            z *= float(q["factor"]) ** self.mj((t - float(q["t"])) / float(q["dur"]))
        return z

    def at(self, t: float) -> tuple[np.ndarray, float]:
        """(camera-frame rotation R_s, zoom) at time t; full rotation is R_s @ R0."""
        return rot_small(*self.angles(t)), self.zoom(t)


def canvas_margin_px(cam: Camera, shake: Shake, duration_s: float, step_s: float = 1 / 30.0,
                     extra: float = 24.0, lo: float = 64.0, hi: float = 480.0) -> int:
    """Margin (output px) the background canvas needs to cover all shake/pan/zoom states."""
    corners = np.array([[0, 0, 1], [cam.W - 1, 0, 1], [0, cam.H - 1, 1], [cam.W - 1, cam.H - 1, 1]], float).T
    worst = 0.0
    K0 = cam.K0
    for t in np.arange(0.0, duration_s + step_s, step_s):
        Rs, z = shake.at(float(t))
        Hinv = K0 @ Rs.T @ np.linalg.inv(cam.K(z))      # output px -> base-camera px
        q = Hinv @ corners
        q = q[:2] / q[2]
        over = max(-q[0].min(), q[0].max() - (cam.W - 1), -q[1].min(), q[1].max() - (cam.H - 1), 0.0)
        worst = max(worst, float(over))
    return int(min(max(worst + extra, lo), hi))
