"""--debug outputs of the PitchFinder (drawn with OpenCV; no matplotlib).

Files written to `FinderOptions.debug_dir`:

  overview.png        reference frame, corridor, clicked points, masks, clutter map
  timeline.png        per-sample stabilization quality, residual sigma, candidate
                      counts and release / catch activity, with accepted (green),
                      rejected (red) and possible-missed (orange) windows
  pitch_NNN.png       accepted tracks in chronological order: stroboscopic max of
                      sign*R/sigma with numbered detections, model curve, release
                      and catch markers, and the along-track residuals
  rejected_TTT_CODE.png  the same for rejected tracks (capped)
  tracks.jsonl        one JSON object per track (accepted, rejected, discarded short)
"""
from __future__ import annotations

import json
import math
import os
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from pitchspeed.types import to_jsonable
from pitchspeed.vision import geometry as G

if TYPE_CHECKING:  # pragma: no cover
    from pitchspeed.vision.finder import FinderResult, PitchFinder

GREEN, RED, ORANGE, YELLOW, CYAN, WHITE, BLUE, MAGENTA = (
    (60, 200, 60), (60, 60, 230), (0, 150, 255), (0, 230, 230), (230, 200, 0), (255, 255, 255), (230, 120, 40),
    (220, 60, 220))
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, color=WHITE, scale=0.45, thick=1):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def _imwrite(path: str, img: np.ndarray) -> None:
    """cv2.imwrite that works with non-ASCII paths on Windows."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise IOError(f"could not encode {path}")
    with open(path, "wb") as f:
        f.write(buf.tobytes())


# ------------------------------------------------------------------ strobe ----
def strobe_image(tr, pt, store, geom, row_scale: int, pad: int = 30, scale: int = 2) -> Optional[np.ndarray]:
    """Stroboscopic max of sign*R/sigma over the track's samples (reference coords)."""
    P = tr.sorted()
    if not P:
        return None
    xs = np.array([p.x for p in P])
    ys = np.array([p.y for p in P])
    x0, x1 = int(xs.min()) - pad, int(xs.max()) + pad
    y0, y1 = int(ys.min()) - pad, int(ys.max()) + pad
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0 or w * h > 1200 * 600:
        return None
    acc = np.full((h, w), -10.0, np.float32)
    zs = []
    for s in range(P[0].s - 3, P[-1].s + 4):
        df = store.get(s)
        if df is None or df.rmap is None:
            continue
        c0, r0, rw, rh = df.roi
        # crop px (ref) -> sample frame -> ROI grid
        M = G.translate(-c0, -r0) @ G.frame_to_grid(row_scale, df.parity) @ np.linalg.inv(G.to3(df.A)) @ \
            G.translate(x0, y0)
        r = df.rmap.astype(np.float32) / 16.0
        warped = cv2.warpPerspective(r, M / M[2, 2], (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=-10.0)
        zs.append(warped)
    if zs:
        st = np.stack(zs)
        sign = 1.0 if sum(p.sign * p.z for p in P) >= 0 else -1.0
        acc = np.max(st * sign, axis=0)
    img = np.clip(40 + acc * 18, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    def to_img(x, y):
        return int(round((x - x0) * scale)), int(round((y - y0) * scale))

    if tr.model is not None:
        ts = np.linspace(P[0].t - 0.1, P[-1].t + 0.05, 200)
        ok = tr.model.denom(ts) > 0.05
        Q = tr.model.eval(ts[ok])
        pts = np.array([to_img(x, y) for x, y in Q], np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, YELLOW, 1, cv2.LINE_AA)
    for i, p in enumerate(P):
        col = (GREEN if p.src == "det" else CYAN if p.src == "pass2" else MAGENTA) if p.inlier else RED
        c = to_img(p.x, p.y)
        cv2.circle(img, c, 3, col, 1, cv2.LINE_AA)
        if i % 3 == 0:
            _text(img, str(i), (c[0] + 4, c[1] - 4), col, 0.35)
    for ev, col, name in ((pt.release, GREEN, "R"), (pt.catch, BLUE, "C")):
        if ev is not None and tr.model is not None and tr.model.denom([ev.t])[0] > 0.05:
            q = tr.model.eval([ev.t])[0]
            c = to_img(q[0], q[1])
            cv2.drawMarker(img, c, col, cv2.MARKER_TILTED_CROSS, 14, 2)
            _text(img, f"{name} {ev.method}", (c[0] + 6, c[1] + 14), col, 0.4)
    # residual strip
    strip = np.zeros((90, img.shape[1], 3), np.uint8)
    if tr.model is not None and len(P) > 1:
        t = np.array([p.t for p in P])
        Q = tr.model.eval(t)
        res = np.hypot((np.array([p.x for p in P]) - Q[:, 0]) * geom.sar, np.array([p.y for p in P]) - Q[:, 1])
        tt = (t - t.min()) / max(np.ptp(t), 1e-6)
        cv2.line(strip, (0, 80), (strip.shape[1] - 1, 80), (90, 90, 90), 1)
        for p, u, r in zip(P, tt, res):
            xx = int(5 + u * (strip.shape[1] - 10))
            yy = int(80 - min(r, 4.0) * 18)
            cv2.circle(strip, (xx, yy), 2, GREEN if p.inlier else RED, -1)
        _text(strip, "residual (px, 0..4)", (5, 14), WHITE, 0.4)
    head = np.zeros((26, img.shape[1], 3), np.uint8)
    lab = f"track {pt.track_id} {pt.status}{' ' + pt.reject_code if pt.reject_code else ''} conf {pt.confidence:.2f}"
    _text(head, lab, (5, 18), WHITE, 0.45)
    return np.vstack([head, img, strip])


# --------------------------------------------------------------- overview ----
def overview_image(finder: "PitchFinder") -> np.ndarray:
    ref = finder.ref
    img = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR) if ref.ndim == 2 else ref.copy()
    g = finder.geom
    # clutter map overlay
    cl = finder.det.clutter_image()
    if cl.size and cl.max() > 0:
        ox, oy = finder.det._cl_origin
        cell = finder.det._cl_cell
        heat = np.clip(cl / max(cl.max(), 1e-6) * 255, 0, 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        H, W = img.shape[:2]
        M = np.array([[cell, 0, ox], [0, cell, oy]], np.float64)
        warped = cv2.warpAffine(heat, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0))
        msk = cv2.warpAffine((cl > 0.002).astype(np.uint8) * 255, M, (W, H), flags=cv2.INTER_NEAREST)
        img[msk > 0] = (0.5 * img[msk > 0] + 0.5 * warped[msk > 0]).astype(np.uint8)
    cv2.polylines(img, [np.round(g.corridor).astype(np.int32).reshape(-1, 1, 2)], True, CYAN, 1, cv2.LINE_AA)
    for name, p in finder.calib.segment.points.items():
        if p is None:
            continue
        c = (int(round(p[0])), int(round(p[1])))
        cv2.drawMarker(img, c, YELLOW, cv2.MARKER_CROSS, 10, 1)
        _text(img, name, (c[0] + 5, c[1] - 5), YELLOW, 0.4)
    for nm, p, col in (("release*", g.release, GREEN), ("catch*", g.catch, BLUE)):
        c = (int(round(p[0])), int(round(p[1])))
        cv2.circle(img, c, 5, col, 1, cv2.LINE_AA)
    for (x0, y0, x1, y1) in finder.masks:
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), RED, 1)
    for b in finder._diag.get("overlay_boxes", []):
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), MAGENTA, 1)
        _text(img, "overlay (auto mask)", (int(b[0]), int(b[1]) - 4), MAGENTA, 0.4)
    _text(img, f"view {g.view}  ball {g.d_rel:.1f}-{g.d_catch:.1f} px  scale {g.scale_rel:.1f} px/ft"
               f"{'' if g.scale_known else ' (guess)'}", (8, 18))
    return img


# --------------------------------------------------------------- timeline ----
def timeline_image(finder: "PitchFinder", res: "FinderResult", width: int = 1600) -> np.ndarray:
    tl = finder._tl
    rows = [("stab quality", np.asarray(finder._q, float), np.asarray(finder._s), (0.0, 1.0)),
            ("residual sigma", np.asarray(tl["sigma"], float), np.asarray(tl["s"]), None),
            ("candidates", np.asarray(tl["n"], float), np.asarray(tl["s"]), None),
            ("E release", np.asarray(tl["e_rel"], float), np.asarray(tl["s"]), None),
            ("E catch", np.asarray(tl["e_catch"], float), np.asarray(tl["s"]), None)]
    rh, top, left = 90, 30, 110
    H = top + rh * len(rows) + 30
    img = np.full((H, width, 3), 24, np.uint8)
    if not len(finder._s):
        return img
    s_lo, s_hi = int(np.min(finder._s)), int(np.max(finder._s))
    span = max(1, s_hi - s_lo)
    pw = width - left - 10

    def X(s):
        return int(left + (np.asarray(s) - s_lo) / span * pw)

    # windows
    def band(s0, s1, col):
        cv2.rectangle(img, (X(s0), top), (max(X(s1), X(s0) + 1), H - 30), col, -1)

    for d in res.possible_missed:
        band(d["s_start"], d["s_end"], (0, 70, 120))
    for p in res.rejected:
        if p.points:
            band(p.points[0].s, p.points[-1].s, (40, 40, 110))
    for p in res.tracks:
        if p.points:
            band(p.points[0].s, p.points[-1].s, (40, 110, 40))
    for i, (name, v, s, rng) in enumerate(rows):
        y0 = top + i * rh
        cv2.rectangle(img, (left, y0 + 4), (width - 10, y0 + rh - 4), (70, 70, 70), 1)
        _text(img, name, (5, y0 + rh // 2), WHITE, 0.4)
        if len(v) == 0:
            continue
        ok = np.isfinite(v)
        if not ok.any():
            continue
        lo, hi = rng if rng else (float(np.nanmin(v[ok])), float(np.nanpercentile(v[ok], 99.5)) + 1e-9)
        if hi <= lo:
            hi = lo + 1
        cols = X(s[ok])
        vals = np.clip((v[ok] - lo) / (hi - lo), 0, 1)
        # max-pool per pixel column
        colmax = np.full(width, -1.0)
        np.maximum.at(colmax, cols, vals)
        for c in np.nonzero(colmax >= 0)[0]:
            yy = int(y0 + rh - 6 - colmax[c] * (rh - 12))
            cv2.line(img, (int(c), y0 + rh - 6), (int(c), yy), (200, 200, 200), 1)
        _text(img, f"{hi:.3g}", (left - 45, y0 + 14), (160, 160, 160), 0.35)
    _text(img, f"samples {s_lo}..{s_hi}   green: pitch   red: rejected   orange: possible missed", (left, 20))
    for p in res.tracks:
        if p.release is not None and p.release.s is not None:
            cv2.line(img, (X(p.release.s), top), (X(p.release.s), H - 30), GREEN, 1)
    return img


# ------------------------------------------------------------------ write ----
def write_debug(out_dir: str, finder: "PitchFinder", res: "FinderResult") -> None:
    os.makedirs(out_dir, exist_ok=True)
    _imwrite(os.path.join(out_dir, "overview.png"), overview_image(finder))
    _imwrite(os.path.join(out_dir, "timeline.png"), timeline_image(finder, res))
    recs = sorted(finder.records(), key=lambda r: r.pt.t_first)
    n_acc = n_rej = 0
    cap = finder.opts.debug_max_pitch_images
    for r in recs:
        if r.strobe is None:
            continue
        if r.pt.status == "pitch":
            n_acc += 1
            if n_acc <= cap:
                _imwrite(os.path.join(out_dir, f"pitch_{n_acc:03d}.png"), r.strobe)
        else:
            n_rej += 1
            if n_rej <= cap:
                _imwrite(os.path.join(out_dir, f"rejected_{r.pt.track_id:04d}_{r.pt.reject_code}.png"), r.strobe)
    path = os.path.join(out_dir, "tracks.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            d = to_jsonable(r.pt)
            d["kind"] = "track"
            f.write(json.dumps(d) + "\n")
        for d in finder._diag.get("discarded", []):
            f.write(json.dumps({"kind": "discarded_short", **d}) + "\n")
    with open(os.path.join(out_dir, "finder_diagnostics.json"), "w", encoding="utf-8") as f:
        diag = {k: v for k, v in res.diagnostics.items() if k != "discarded"}
        json.dump(to_jsonable(diag), f, indent=2)
