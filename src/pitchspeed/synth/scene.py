"""Static background canvas and the moving actors (capsules).

Canvas: the base camera's view ray-cast once at 2x supersampling with a
margin for shake/pans/zoom-out: sky, an outfield fence cylinder with trees,
the ground as a height field (mound cap; rubber top at mound height) with
procedural turf, dirt, chalk lines, batter's boxes, plate and rubber, an
optional vertical pole (calibration vertical reference) and, in a 4th
channel, the alpha of the out-of-focus backstop fence.  Pre-blurred by
sigma = 0.5 output px (anti-aliasing).

Actors: people, glove, bat and birds as 3-D capsules (p0, p1, radius, BGR),
projected per field to stadium polygons and depth-sorted by the renderer.
People are hidden before calibration.clear_field_s so the calibration
reference frame shows a clear field.  The scenery is cartoonish on
purpose; detectors must not be tuned to it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from pitchspeed.synth.camera import Camera
from pitchspeed.synth.config import LevelGeom, rng_for
from pitchspeed.synth.events import Events, smooth
from pitchspeed.synth.fence import defocus_blur_px, fence_alpha


# ------------------------------------------------------------------ background
def field_shade(X: np.ndarray, Y: np.ndarray, lev: LevelGeom, tex: np.ndarray) -> np.ndarray:
    """Procedural ground albedo (BGR float 0..255) at world (X, Y)."""
    grass = np.array([60, 125, 70.0])
    grass2 = np.array([52, 112, 62.0])
    dirt = np.array([95, 135, 170.0])
    chalk = np.array([235, 238, 240.0])
    a = (X + Y) / math.sqrt(2)                 # along the 1B line
    b = (Y - X) / math.sqrt(2)                 # along the 3B line
    B = lev.base_ft
    mc = lev.distance_ft - 1.5
    stripe = (np.floor((Y + 3 * X) / 12.0) % 2).astype(np.float32)[..., None]
    col = grass * (1 - stripe) + grass2 * stripe
    r_mound = np.hypot(X, Y - mc)
    r_home = np.hypot(X, Y)
    infield_grass = (a > 3) & (a < B - 3) & (b > 3) & (b < B - 3)
    arc = np.hypot(X, Y - lev.distance_ft) < B * 95 / 90
    base_path = ((np.abs(b) < 3) & (a > 0) & (a < B + 3)) | ((np.abs(a) < 3) & (b > 0) & (b < B + 3))
    dirtmask = (r_mound < 9) | (r_home < 13) | (arc & ~infield_grass & (a > -3) & (b > -3)) | base_path
    col = np.where(dirtmask[..., None], dirt, col)
    line = ((np.abs(b) < 0.125) & (a > 1.5)) | ((np.abs(a) < 0.125) & (b > 1.5))
    for sx in (-1, 1):                          # batter's boxes 4x6 ft, 6 in from the plate
        x0, x1 = sx * (8.5 / 12 + 0.5), sx * (8.5 / 12 + 4.5)
        xl, xh = min(x0, x1), max(x0, x1)
        y0, y1 = 0.708 - 3, 0.708 + 3
        inb = (X > xl - .125) & (X < xh + .125) & (Y > y0 - .125) & (Y < y1 + .125)
        inn = (X > xl + .125) & (X < xh - .125) & (Y > y0 + .125) & (Y < y1 - .125)
        line |= inb & ~inn
    col = np.where(line[..., None], chalk, col)
    ax = np.abs(X)
    plate = (Y >= 0) & (Y <= 17 / 12) & (ax <= 8.5 / 12) & ((Y >= 8.5 / 12) | (ax <= Y))
    col = np.where(plate[..., None], np.array([245, 245, 245.0]), col)
    ts = tex.shape[0]
    iu = (np.floor(X * 8) % ts).astype(np.int32)
    iv = (np.floor(Y * 8) % ts).astype(np.int32)
    return col * (0.85 + 0.3 * tex[iv, iu])[..., None]


def mound_height(X: np.ndarray, Y: np.ndarray, lev: LevelGeom) -> np.ndarray:
    r = np.hypot(X, Y - (lev.distance_ft - 1.5))
    return lev.mound_h_ft * np.clip((9.0 - r) / 6.5, 0, 1)


def pole_geometry(cfg: dict, lev: LevelGeom) -> Optional[dict]:
    """World position of the vertical pole, or None when disabled."""
    pc = cfg["scene"]["pole"]
    if not pc["enabled"]:
        return None
    cam = cfg["camera"]
    s = 1.0 if cam.get("side", "1b") == "1b" else -1.0
    if cam["view"] == "side":
        x0, y0 = -s * 30.0, 0.35 * lev.distance_ft      # beyond the pitch line, far side
    else:
        x0, y0 = 28.0, 0.55 * lev.distance_ft
    x = float(pc["x_ft"]) if pc.get("x_ft") is not None else x0
    y = float(pc["y_ft"]) if pc.get("y_ft") is not None else y0
    return dict(x=x, y=y, height=float(pc["height_ft"]), radius=float(pc["radius_ft"]),
                value=float(pc["value"]))


@dataclass
class Canvas:
    img: np.ndarray            # (Hc, Wc, 3 or 4) float32; channel 3 = fence alpha if present
    ss: int                    # supersampling
    margin: int                # output px
    has_fence: bool
    fence_blur_px: float       # defocus blur diameter (output px), 0 if no fence
    C2O: np.ndarray            # canvas px -> base-camera output px (3x3)


def render_canvas(cam: Camera, lev: LevelGeom, cfg: dict, margin: int, ss: int = 2) -> Canvas:
    """Ray-cast the static background for the base camera."""
    W, H = cam.W, cam.H
    Wc, Hc = (W + 2 * margin) * ss, (H + 2 * margin) * ss
    uc, vc = np.meshgrid(np.arange(Wc, dtype=np.float32), np.arange(Hc, dtype=np.float32))
    uo = (uc - (ss - 1) / 2) / ss - margin
    vo = (vc - (ss - 1) / 2) / ss - margin
    del uc, vc
    xn = (uo - cam.cx) / cam.fx0
    yn = (vo - cam.cy) / cam.fy0
    D = np.stack([xn, yn, np.ones_like(xn)], -1) @ cam.R0.astype(np.float32)     # world ray dirs
    del uo, vo, xn, yn
    C = cam.C
    r = rng_for(cfg["seed"], 1)
    tex = cv2.GaussianBlur(r.random((256, 256)).astype(np.float32), (0, 0), 1.0)
    tex = (tex - tex.min()) / np.ptp(tex)
    img = np.empty((Hc, Wc, 3), np.float32)
    elev = D[..., 2] / np.linalg.norm(D, axis=-1)
    img[:] = np.array([235, 200, 160.0], np.float32) * (1 - elev[..., None] * 0.4)
    # outer fence cylinder (centre (0,120), radius 210) with trees above it
    cc = np.array([0.0, 120.0])
    Rf = 210.0
    px, py = C[0] - cc[0], C[1] - cc[1]
    dx, dy = D[..., 0], D[..., 1]
    A = dx * dx + dy * dy
    Bq = 2 * (px * dx + py * dy)
    Cq = px * px + py * py - Rf * Rf
    tcyl = (-Bq + np.sqrt(np.maximum(Bq * Bq - 4 * A * Cq, 0))) / (2 * A + 1e-12)
    zc = C[2] + tcyl * D[..., 2]
    ang = np.arctan2(dy, dx)
    trees = (zc >= 8) & (zc < 45 + 12 * np.sin(ang * 23) + 6 * np.sin(ang * 61))
    img[trees] = np.array([45, 80, 50.0])
    img[(zc >= 0) & (zc < 8)] = np.array([50, 70, 40.0])
    # ground with the mound height field (fixed-point re-intersection)
    down = D[..., 2] < -1e-6
    dz = np.where(down, D[..., 2], -1.0)
    tg = np.where(down, -C[2] / dz, np.inf)
    for _ in range(4):
        h = mound_height(C[0] + tg * D[..., 0], C[1] + tg * D[..., 1], lev)
        tg = np.where(down, (h - C[2]) / dz, np.inf)
    hit = down & (tg < tcyl)
    Xg = np.where(hit, C[0] + tg * D[..., 0], 0.0)
    Yg = np.where(hit, C[1] + tg * D[..., 1], 0.0)
    gcol = field_shade(Xg, Yg, lev, tex)
    img = np.where(hit[..., None], gcol, img).astype(np.float32)
    del gcol, Xg, Yg
    # rubber top (z = mound height), 24 x 6 in, front edge at y = distance_ft
    zr = lev.mound_h_ft + 0.01
    tr = (zr - C[2]) / np.where(np.abs(D[..., 2]) > 1e-9, D[..., 2], 1e-9)
    Xr = C[0] + tr * D[..., 0]
    Yr = C[1] + tr * D[..., 1]
    rub = ((tr > 0) & (np.abs(Xr) <= 1.0) & (Yr >= lev.distance_ft) & (Yr <= lev.distance_ft + 0.5)
           & (tr <= tg + 0.05))
    img[rub] = np.array([250, 250, 250.0])
    # vertical pole (in front of whatever is behind it; actors are drawn later)
    pole = pole_geometry(cfg, lev)
    if pole is not None:
        ox, oy = C[0] - pole["x"], C[1] - pole["y"]
        Bp = 2 * (ox * dx + oy * dy)
        Cp = ox * ox + oy * oy - pole["radius"] ** 2
        disc = Bp * Bp - 4 * A * Cp
        tp = (-Bp - np.sqrt(np.maximum(disc, 0))) / (2 * A + 1e-12)
        zp = C[2] + tp * D[..., 2]
        on = (disc > 0) & (tp > 0) & (zp >= 0) & (zp <= pole["height"]) & (tp < np.minimum(tg, tcyl))
        # simple cylindrical shading: brighter where the surface faces +x
        hx = (C[0] + tp * dx - pole["x"]) / pole["radius"]
        shade = 0.75 + 0.25 * np.clip(hx, -1, 1)
        img[on] = (pole["value"] * shade[on])[:, None] * np.ones(3, np.float32)
    img = np.clip(img, 0, 255).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), sigmaX=0.5 * ss, sigmaY=0.5 * ss)
    fcfg = cfg["fence"]
    blur = 0.0
    if fcfg["enabled"]:
        if fcfg.get("blur_px") is not None:
            blur = float(fcfg["blur_px"])
        else:
            focus = fcfg.get("focus_ft")
            if focus is None:        # focused on the pitcher
                focus = float(np.linalg.norm(np.array([0.0, lev.distance_ft - 5.0, 5.0]) - C))
            blur = defocus_blur_px(cam.fx0, W, float(cfg["camera"]["sensor_width_mm"]),
                                   float(cfg["camera"]["f_number"]), float(fcfg["dist_ft"]), float(focus))
        alpha = fence_alpha(D, C, cam.R0, cam.fx0 * ss, fcfg, blur * ss)
        img = np.concatenate([img, alpha[..., None]], axis=-1)
    o = -(ss - 1) / (2 * ss) - margin
    C2O = np.array([[1 / ss, 0, o], [0, 1 / ss, o], [0, 0, 1.0]])
    return Canvas(img=np.ascontiguousarray(img), ss=ss, margin=margin, has_fence=bool(fcfg["enabled"]),
                  fence_blur_px=blur, C2O=C2O)


# ------------------------------------------------------------------ actors
UNIFORM = (225, 225, 225)
PANTS = (60, 60, 70)
CAP = (90, 40, 30)
GLOVE = (30, 50, 90)


class Scene:
    """Actors as capsules at time t."""

    def __init__(self, cfg: dict, lev: LevelGeom, events: Events):
        self.cfg, self.lev, self.ev = cfg, lev, events
        self.clear_until = float(cfg["calibration"]["clear_field_s"])

    def current_pitch(self, t: float):
        best = None
        for p in self.ev.pitches:
            if t >= p.t_release - 2.5:
                best = p
        return best

    def capsules(self, t: float) -> list:
        """List of (p0, p1, radius_ft, bgr) in world feet."""
        lev, D = self.lev, self.cfg["distractors"]
        caps = []
        people = t >= self.clear_until
        p = self.current_pitch(t)
        h = lev.mound_h_ft
        if D["pitcher"] and people:
            if p is not None:
                prog = smooth((t - p.t_release + 0.45) / 0.45)
                xr = p.release_pos[0]
                body_y = lev.distance_ft + 0.6 + (p.release_pos[1] + 1.2 - lev.distance_ft - 0.6) * prog
                hand, shoulder = p.hand.at(t)
            else:
                xr, body_y = 0.0, lev.distance_ft + 0.6
                shoulder = np.array([xr, body_y, h + 4.6])
                hand = shoulder + [0.4, -0.6, -1.4]
            hip = np.array([xr, body_y + 0.3, h + 3.0])
            sh = np.asarray(shoulder, dtype=float)
            caps += [(np.array([xr - 0.4, lev.distance_ft + 0.3, h]), hip, 0.32, PANTS),
                     (np.array([xr + 0.4, body_y - 1.5 * (body_y < lev.distance_ft), h]), hip, 0.32, PANTS),
                     (hip, sh, 0.62, UNIFORM), (sh + [0, 0, 0.55], sh + [0, 0, 0.75], 0.42, CAP),
                     (sh, np.asarray(hand, dtype=float), 0.18, UNIFORM),
                     (sh + [-0.6, 0, 0], sh + [-1.0, -0.9, -0.8], 0.35, GLOVE)]
        if D["catcher"] and people:
            gl = p.glove_pos(t) if p is not None else np.array([0.0, -1.5, 2.2])
            cb = np.array([0.0, -3.2, 0.0])
            caps += [(cb + [0, 0, 0.6], cb + [0, 0, 2.6], 0.75, (70, 60, 60)),
                     (cb + [0, 0.2, 3.1], cb + [0, 0.2, 3.3], 0.45, (40, 40, 40)),
                     (cb + [0.6, 0.3, 2.4], gl, 0.2, (70, 60, 60)),
                     (gl, gl + [0, -0.05, 0], 0.42, (30, 60, 110))]
        if D["umpire"] and people:
            ub = np.array([0.4, -5.8, 0.0])
            caps += [(ub + [0, 0, 0.5], ub + [0, 0, 3.6], 0.85, (60, 40, 30)),
                     (ub + [0, 0, 4.1], ub + [0, 0, 4.3], 0.45, (40, 40, 40))]
        if D["batter"] and people and p is not None:
            bb = np.array([-3.0, 0.8, 0.0])
            sh = bb + [0, 0, 4.6]
            caps += [(bb + [0, 0, 0.4], bb + [0, 0, 3.0], 0.55, (200, 200, 200)),
                     (bb + [0, 0, 3.0], sh, 0.65, (140, 60, 40)),
                     (sh + [0, 0, 0.6], sh + [0, 0, 0.9], 0.45, (140, 60, 40))]
            hands = bb + [0.5, -0.5, 4.0]
            ang0 = math.radians(150)            # bat pointing back over the shoulder
            s = 0.0
            if p.swing:
                s = float(smooth((t - (p.t_plate - 0.10)) / 0.16))
            ang = ang0 - math.radians(260) * s
            lift = 0.45 * (1 - (p.swing and s > 0.3))
            tip = hands + 2.8 * np.array([math.cos(ang) * 0.9, math.sin(ang) * 0.9, lift])
            caps += [(hands, tip, 0.11, (60, 120, 170))]
        for b in self.ev.birds:
            pos = b.pos(t)
            if pos is not None:
                w = 0.9 * abs(math.sin(2 * math.pi * 6 * t))
                caps += [(pos - [w, 0, 0], pos + [w, 0, 0], 0.18, (40, 40, 40))]
        return caps


def capsule_polys(cam: Camera, Rt: np.ndarray, zoom: float, caps: list, row_off: float, row_step: float,
                  ncap: int = 8) -> list:
    """Project capsules to stadium polygons in field-image coords: [(depth, poly(N,2), bgr), ...]."""
    if not caps:
        return []
    P0 = np.array([c[0] for c in caps], dtype=float)
    P1 = np.array([c[1] for c in caps], dtype=float)
    rad = np.array([c[2] for c in caps], dtype=float)
    X0 = (P0 - cam.C) @ Rt.T
    X1 = (P1 - cam.C) @ Rt.T
    ok = (X0[:, 2] >= 0.5) & (X1[:, 2] >= 0.5)
    z0 = np.where(ok, X0[:, 2], 1.0)
    z1 = np.where(ok, X1[:, 2], 1.0)
    a = X0[:, :2] / z0[:, None]
    b = X1[:, :2] / z1[:, None]
    ra, rb = rad / z0, rad / z1
    d = b - a
    L = np.linalg.norm(d, axis=1)
    d = np.where((L > 1e-9)[:, None], d / np.maximum(L, 1e-12)[:, None], np.array([1.0, 0.0]))
    n = np.stack([-d[:, 1], d[:, 0]], axis=1)
    th = np.linspace(0, math.pi, ncap)
    ct, st = np.cos(th)[None, :, None], np.sin(th)[None, :, None]
    capb = b[:, None, :] + rb[:, None, None] * (n[:, None, :] * ct + d[:, None, :] * st)
    capa = a[:, None, :] + ra[:, None, None] * (-n[:, None, :] * ct - d[:, None, :] * st)
    pts = np.concatenate([capb, capa], axis=1)
    u = cam.fx0 * zoom * pts[..., 0] + cam.cx
    v = (cam.fy0 * zoom * pts[..., 1] + cam.cy - row_off) / row_step
    poly = np.stack([u, v], axis=-1)
    depth = 0.5 * (z0 + z1)
    return [(float(depth[i]), poly[i], caps[i][3]) for i in range(len(caps)) if ok[i]]
