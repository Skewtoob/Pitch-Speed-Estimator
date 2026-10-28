"""Per-field rendering and frame assembly.

Each FIELD is rendered at its own exposure midpoint t_k = k * 1001/60000 s
(never "render a frame and split it"): camera shake/zoom, actors and the ball
are all evaluated at t_k.  The background is the pre-rendered canvas warped
by the exact rotation+zoom homography directly onto the field's row grid
(field row r = frame row 2r + p).  Actors are depth-sorted capsules; the
ball is the exact motion-blurred coverage of `ball.ball_patch`, composited
in linear light (gamma 2.2); the out-of-focus backstop fence (if any) is
the nearest layer.  Fields are interleaved (`frame[p::2] = field`) and the
frame goes through the camera/tape chain (`tape.TapeChain`, horizontal-only
operations) and the optional postprocess.

Per-field ball ground truth (pitches only) is collected as plain dicts, see
`Renderer.render_field` and docs/design-notes/synth.md section 6.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from pitchspeed.synth.ball import ball_patch
from pitchspeed.synth.camera import MARGIN_MAX_PX, Camera, Shake, canvas_margin_needed
from pitchspeed.synth.config import BALL_R_FT, field_time, level_of, par_of
from pitchspeed.synth.events import make_events
from pitchspeed.synth.scene import Scene, capsule_polys, render_canvas
from pitchspeed.synth.stamp import stamp as draw_stamp
from pitchspeed.synth.tape import TapeChain

GAMMA = 2.2


@dataclass
class BallPath:
    """A ball that is visible during [t0, t1] (pitch: in hand + flight; or a throwback)."""
    kind: str                                   # 'pitch' | 'throwback'
    id: int
    t0: float
    t1: float
    pos_many: Callable[[np.ndarray], np.ndarray]   # (n,) times -> (n, 3) world ft (any t)
    breaks: list = field(default_factory=list)     # extra segment boundaries (release)
    t_release: Optional[float] = None

    def state(self, t: float) -> str:
        if self.kind != "pitch":
            return "flight"
        if t < self.t_release:
            return "in_hand"
        return "flight" if t <= self.t1 else "caught"


@dataclass
class FrameResult:
    """One rendered source frame (after the tape chain and the in-process postprocess)."""
    n: int
    image: np.ndarray                 # (H, W, 3) uint8 BGR
    times: list                       # exposure midpoints of the fields/frames it contains
    field_index: list                 # global field index k of each
    parity: list                      # frame-row parity of each (-1: full progressive frame)
    cams: list                        # [(k, t, Rt(3x3), zoom)]
    records: list                     # per-field ball GT dicts (pitches only)


def _pitch_path(p) -> BallPath:
    def pos_many(ts: np.ndarray, p=p) -> np.ndarray:
        ts = np.asarray(ts, dtype=float)
        out = np.empty((len(ts), 3))
        hand = ts < p.t_release
        for i in np.nonzero(hand)[0]:
            out[i] = p.hand.at(float(ts[i]))[0]
        if (~hand).any():
            out[~hand] = p.traj.states(ts[~hand])[0]
        return out
    return BallPath("pitch", p.id, p.t_hand, p.t_catch, pos_many, [p.t_release], p.t_release)


def _throw_path(i: int, th) -> BallPath:
    def pos_many(ts: np.ndarray, th=th) -> np.ndarray:
        return th.traj.states(np.asarray(ts, dtype=float))[0]
    return BallPath("throwback", i, th.t_start, th.t_end, pos_many)


class Renderer:
    """Renders fields/frames of one clip from a RESOLVED config (see config.resolve_config)."""

    def __init__(self, cfg: dict, ref_times: Optional[list] = None):
        t_start = time.perf_counter()
        self.cfg = cfg
        self.lev = level_of(cfg)
        V = cfg["video"]
        self.W, self.H = int(V["width"]), int(V["height"])
        self.par = par_of(cfg)
        self.scan = V["scan"]
        self.post = V["postprocess"]
        self.tff = V["field_order"] == "tff"
        self.cam = Camera(cfg["camera"], self.lev, self.W, self.H, self.par)
        self.shake = Shake(cfg["shake"], cfg["seed"])
        self.events = make_events(cfg, self.lev, self.cam)
        self.scene = Scene(cfg, self.lev, self.events)
        need = canvas_margin_needed(self.cam, self.shake, float(cfg["duration_s"]))
        self.warnings = list(self.cam.warnings)
        if need + 24.0 > MARGIN_MAX_PX:
            self.warnings.append(f"camera pans/zooms move the view {need:.0f} px beyond the background canvas "
                                 f"(max {MARGIN_MAX_PX:.0f}): image edges are smeared there")
        margin = int(min(max(need + 24.0, 64.0), MARGIN_MAX_PX))
        self.canvas = render_canvas(self.cam, self.lev, cfg, margin)
        self.t_canvas = time.perf_counter() - t_start
        self.K0inv = np.linalg.inv(self.cam.K0)
        R = cfg["render"]
        self.shutter = float(R["shutter_s"])
        self.nseg = int(R["segments"])
        self.psf = float(R["psf_sigma_px"])
        self.lens_sigma = float(R.get("lens_sigma_px") or 0.0)
        self.ball_value = float(R["ball_value"])
        self.ball_lin = (self.ball_value / 255.0) ** GAMMA
        self.draw_balls = bool(R.get("draw_balls", True))
        fz = cfg["fence"]
        self.fence_value = float(fz.get("wire_value", 110))
        self._wire_img: Optional[np.ndarray] = None
        self.tape = TapeChain(cfg, self.W, self.H)
        self.paths = [_pitch_path(p) for p in self.events.pitches]
        self.paths += [_throw_path(i, th) for i, th in enumerate(self.events.throws)]
        self.set_reference(ref_times or [0.0])

    # ------------------------------------------------------------ camera
    def camera_at(self, t: float) -> tuple[np.ndarray, float]:
        """(full world->camera rotation Rt, zoom) at time t."""
        Rs, zoom = self.shake.at(t)
        return Rs @ self.cam.R0, zoom

    def set_reference(self, times: list) -> None:
        """Camera of the calibration reference frame: the camera at the MEAN exposure time of the
        frame (for an interlaced frame the two fields are +/- half a field of shake away from it)."""
        self.ref_times = [float(t) for t in times]
        self.ref_cam = self.camera_at(float(np.mean(self.ref_times)))

    def project_ref(self, P) -> np.ndarray:
        """World points (n,3) -> (n,2) frame px seen by the reference camera."""
        Rt, z = self.ref_cam
        u, v, _ = self.cam.project(Rt, z, np.atleast_2d(np.asarray(P, dtype=float)))
        return np.stack([u, v], axis=1)

    def ref_depth(self, P) -> np.ndarray:
        """Camera depth (ft) of world points in the reference camera (<= 0: behind the camera)."""
        Rt, z = self.ref_cam
        return self.cam.project(Rt, z, np.atleast_2d(np.asarray(P, dtype=float)))[2]

    # ------------------------------------------------------------ field
    def _ball_segments(self, bp: BallPath, Rt, zoom, ta: float, tb: float) -> list:
        ts = set(np.linspace(ta, tb, self.nseg + 1).tolist())
        ts.update(e for e in bp.breaks + [bp.t0, bp.t1] if ta < e < tb)
        ts = np.array(sorted(ts))
        mids = 0.5 * (ts[:-1] + ts[1:])
        keep = (mids >= bp.t0) & (mids <= bp.t1)
        if not keep.any():
            return []
        a, b = ts[:-1][keep], ts[1:][keep]
        P = bp.pos_many(np.concatenate([a, b]))
        u, v, z = self.cam.project(Rt, zoom, P)
        n = len(a)
        segs = []
        fx, fy = self.cam.fx0 * zoom, self.cam.fy0 * zoom
        for i in range(n):
            za, zb = z[i], z[n + i]
            if za < 1.0 or zb < 1.0:
                continue
            zm = 0.5 * (za + zb)
            segs.append((u[i], v[i], u[n + i], v[n + i], fx * BALL_R_FT / zm, fy * BALL_R_FT / zm,
                         (b[i] - a[i]) / self.shutter))
        return segs

    def _ball_record(self, bp: BallPath, t: float, Rt, zoom, ta: float, tb: float, segs: list,
                     info: dict) -> dict:
        ts = np.array([t, max(ta, bp.t0), min(tb, bp.t1)])
        P = bp.pos_many(ts)
        u, v, z = self.cam.project(Rt, zoom, P)
        ref = self.project_ref(P[:1])[0]
        rec = dict(info)
        rec.update(pitch_id=bp.id, t=t, state=bp.state(t), u=float(u[0]), v=float(v[0]),
                   u_start=float(u[1]), v_start=float(v[1]), u_end=float(u[2]), v_end=float(v[2]),
                   u_ref=float(ref[0]), v_ref=float(ref[1]),
                   r_px_x=float(self.cam.fx0 * zoom * BALL_R_FT / z[0]),
                   r_px_y=float(self.cam.fy0 * zoom * BALL_R_FT / z[0]), depth_ft=float(z[0]),
                   in_frame=bool(0 <= u[0] <= self.W - 1 and 0 <= v[0] <= self.H - 1),
                   pos_ft=[float(x) for x in P[0]])
        if segs:
            sg = np.asarray(segs)
            wsum = float(sg[:, 6].sum())
            rec.update(vis_frac=wsum, u_vis=float((sg[:, 6] * (sg[:, 0] + sg[:, 2]) / 2).sum() / wsum),
                       v_vis=float((sg[:, 6] * (sg[:, 1] + sg[:, 3]) / 2).sum() / wsum))
        else:
            rec.update(vis_frac=0.0, u_vis=None, v_vis=None)
        return rec

    def render_field(self, t: float, row_off: int, row_step: int, aperture: float, Hout: int,
                     info: Optional[dict] = None, records: Optional[list] = None
                     ) -> tuple[np.ndarray, np.ndarray, float]:
        """Render the image exposed around t onto rows  r*row_step + row_off  (Hout rows).

        Returns (uint8 BGR image, Rt, zoom).  Appends one GT dict per visible
        pitch ball to `records`.
        """
        cam, cv = self.cam, self.canvas
        Rs, zoom = self.shake.at(t)
        Rt = Rs @ cam.R0
        Srow = np.array([[1, 0, 0], [0, 1 / row_step, -row_off / row_step], [0, 0, 1.0]])
        M = Srow @ cam.K(zoom) @ Rs @ self.K0inv @ cv.C2O
        warped = cv2.warpPerspective(cv.img, M, (self.W, Hout), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
        alpha = warped[..., 3] if cv.has_fence else None
        img8 = np.clip(warped[..., :3] + 0.5, 0, 255).astype(np.uint8)
        polys = capsule_polys(cam, Rt, zoom, self.scene.capsules(t), row_off, row_step)
        polys.sort(key=lambda c: -c[0])
        ta, tb = t - self.shutter / 2, t + self.shutter / 2
        balls = []
        if self.draw_balls or records is not None:
            for bp in self.paths:
                if tb < bp.t0 or ta > bp.t1:
                    continue
                segs = self._ball_segments(bp, Rt, zoom, ta, tb)
                rec = None
                if records is not None and bp.kind == "pitch":
                    rec = self._ball_record(bp, t, Rt, zoom, ta, tb, segs, info or {})
                if segs or rec is not None:
                    depth = float(np.mean([cam.fx0 * zoom * BALL_R_FT / s[4] for s in segs])) if segs else math.inf
                    balls.append((depth, segs, rec))
        z_near = min([b[0] for b in balls], default=-1.0)
        for z, poly, col in polys:
            if z > z_near:
                cv2.fillPoly(img8, [np.round(poly * 16).astype(np.int32)], col, lineType=cv2.LINE_AA, shift=4)
        if balls:
            img = img8.astype(np.float32)
            for depth, segs, rec in balls:
                self._composite_ball(img, segs, rec, depth, polys, alpha, row_off, row_step, aperture, Hout)
                if rec is not None:
                    records.append(rec)
            img8 = np.clip(img + 0.5, 0, 255).astype(np.uint8)
        for z, poly, col in polys:
            if z <= z_near:
                cv2.fillPoly(img8, [np.round(poly * 16).astype(np.int32)], col, lineType=cv2.LINE_AA, shift=4)
        if alpha is not None:                      # nearest layer: out = in (1 - a) + wire a
            a = np.ascontiguousarray(alpha)
            if self._wire_img is None or self._wire_img.shape != img8.shape:
                self._wire_img = np.full(img8.shape, int(round(self.fence_value)), np.uint8)
            img8 = cv2.blendLinear(img8, self._wire_img, 1.0 - a, a)
        if self.lens_sigma > 0:
            s = self.lens_sigma
            out = cv2.GaussianBlur(img8.astype(np.float32), (0, 0), sigmaX=s, sigmaY=max(s / row_step, 0.01))
            img8 = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        return img8, Rt, zoom

    def _composite_ball(self, img: np.ndarray, segs: list, rec: Optional[dict], depth: float, polys: list,
                        alpha: Optional[np.ndarray], row_off: int, row_step: int, aperture: float,
                        Hout: int) -> None:
        bp = ball_patch(segs, self.W, Hout, row_off, row_step, aperture, self.psf) if segs else None
        if bp is None:
            if rec is not None:
                rec.update(peak_cov=0.0, occluded_frac=1.0 if rec["vis_frac"] > 0 else 0.0,
                           occluded=rec["vis_frac"] > 0, fence_alpha=0.0, u_blob=None, v_blob=None,
                           bg_luma=None, contrast=None)
            return
        r0, c0, cov = bp
        h, w = cov.shape
        roi = img[r0:r0 + h, c0:c0 + w]
        if rec is not None:
            occ = np.zeros((h, w), np.uint8)
            for z, poly, _ in polys:
                if z <= depth:
                    q = poly - np.array([c0, r0])
                    cv2.fillPoly(occ, [np.round(q * 16).astype(np.int32)], 255, lineType=cv2.LINE_AA, shift=4)
            occ_f = occ.astype(np.float64) / 255.0
            csum = float(cov.sum())
            unocc = cov * (1.0 - occ_f)                  # not hidden by actors
            fa_roi = alpha[r0:r0 + h, c0:c0 + w] if alpha is not None else 0.0
            vis = unocc * (1.0 - fa_roi)                 # ... and seen through the fence
            vsum = float(vis.sum())
            luma = roi @ np.array([0.114, 0.587, 0.299], np.float32)
            rows = np.arange(r0, r0 + h) * row_step + row_off
            cols = np.arange(c0, c0 + w)
            fa = float((cov * fa_roi).sum() / csum) if (alpha is not None and csum > 0) else 0.0
            rec.update(peak_cov=float(cov.max()),
                       occluded_frac=float(1.0 - unocc.sum() / csum) if csum > 0 else 1.0,
                       fence_alpha=fa,
                       u_blob=float((vis.sum(0) * cols).sum() / vsum) if vsum > 1e-9 else None,
                       v_blob=float((vis.sum(1) * rows).sum() / vsum) if vsum > 1e-9 else None,
                       bg_luma=float((cov * luma).sum() / csum) if csum > 0 else None)
            rec["occluded"] = bool(rec["occluded_frac"] >= 0.5)
            rec["contrast"] = None if rec["bg_luma"] is None else self.ball_value - rec["bg_luma"]
        if self.draw_balls:
            lin = (roi / 255.0) ** GAMMA
            c3 = cov[..., None].astype(np.float32)
            lin = lin * (1 - c3) + self.ball_lin * c3
            roi[:] = 255.0 * lin ** (1 / GAMMA)

    # ------------------------------------------------------------ frame
    def render_frame(self, n: int, collect: bool = True) -> FrameResult:
        """Render source frame n (all its fields), apply the tape chain and in-process postprocess."""
        H, W = self.H, self.W
        recs: Optional[list] = [] if collect else None
        stamp_on = bool(self.cfg["video"].get("debug_stamp"))
        times, ks, pars, cams = [], [], [], []
        if self.scan == "interlaced":
            frame = np.empty((H, W, 3), np.uint8)
            for i in (0, 1):
                k = 2 * n + i
                t = field_time(k)
                p = i if self.tff else 1 - i
                img, Rt, zoom = self.render_field(t, p, 2, 2.0, H // 2,
                                                  dict(field_index=k, frame=n, field_in_frame=i, parity=p), recs)
                if stamp_on:
                    draw_stamp(img, k, 1)
                frame[p::2] = img
                times.append(t), ks.append(k), pars.append(p), cams.append((k, t, Rt, zoom))
        else:
            k = 2 * n if self.scan == "progressive30" else n
            t = field_time(k)
            frame, Rt, zoom = self.render_field(t, 0, 1, 1.0, H,
                                                dict(field_index=k, frame=n, field_in_frame=0, parity=-1), recs)
            if stamp_on:
                draw_stamp(frame, k, 2)
            times.append(t), ks.append(k), pars.append(-1), cams.append((k, t, Rt, zoom))
        out = self.tape.apply(frame, n)
        if self.post == "blend":                  # mixes the two fields ("blend" deinterlacer)
            out = cv2.filter2D(out, -1, np.array([[0.25], [0.5], [0.25]], np.float32),
                               borderType=cv2.BORDER_REPLICATE)
        elif self.post == "field_discard":        # keep the first field, line-double (exact rows 2r+p)
            p0 = pars[0]
            M = np.float32([[1, 0, 0], [0, 0.5, -0.5 * p0]])
            out = cv2.warpAffine(np.ascontiguousarray(out[p0::2]), M, (W, H),
                                 flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
            times, ks, pars = times[:1], ks[:1], [-1]
            if recs is not None:
                recs[:] = [r for r in recs if r["field_in_frame"] == 0]
        return FrameResult(n, out, times, ks, pars, cams, recs or [])
