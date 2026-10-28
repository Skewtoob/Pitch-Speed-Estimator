"""Unit tests: behind-view detection / tracking robustness.

Covers the fixes for the behind-view recall study (zoomed camera behind the
plate, often through a blurred backstop fence, noisy tape):

* stabilization on noisy, low-texture footage (noise-adaptive inlier
  threshold, transform-precision quality, occluded-feature filter) and the
  temporal-continuity guard against false (aliased / static-foreground)
  registrations;
* the physically sized behind-view corridor;
* per-kernel validity of the matched-filter responses (ball next to the
  image edge);
* joint-fit merging of the slow-early / fast-late fragments of one flight
  and model-driven regrowth of a track along its rational image model;
* the "pitch must get past the release point" guard.
"""
import math

import cv2
import numpy as np
import pytest

from pitchspeed.calib.model import CalibrationFile, Segment, Settings
from pitchspeed.types import EventTime, PitchTrack, ScanInfo
from pitchspeed.vision import geometry as G
from pitchspeed.vision.detect import CAND_DTYPE, DetFrame, Detector
from pitchspeed.vision.finder import PitchFinder
from pitchspeed.vision.options import FinderOptions
from pitchspeed.vision.stabilize import Stabilizer, StabilizerOptions, StabResult, _quality
from pitchspeed.vision.track import (CandStore, Pt, Track, _models_agree, fit_track, regrow, trim_sparse_ends,
                                     try_merge)
from vision_helpers import textured_background

SAR = 10 / 11
DT = 1001 / 60000
PTS = np.array([[120.0, 100.0], [600.0, 120.0], [360.0, 240.0], [150.0, 400.0], [620.0, 380.0]])


# ---------------------------------------------------------------- helpers ----
def behind_calibration(release, catch, rubber_half=15.0, plate_half=25.0) -> CalibrationFile:
    """Behind-view calibration with rubber / plate width clicks (scale known)."""
    rx, ry = release
    cx, cy = catch
    pts = {"release": [rx, ry], "catch": [cx, cy], "rubber": [rx, ry + 30], "plate": [cx, cy + 20],
           "rubber_left": [rx - rubber_half, ry + 30], "rubber_right": [rx + rubber_half, ry + 30],
           "plate_left": [cx - plate_half, cy + 20], "plate_right": [cx + plate_half, cy + 20]}
    st = Settings(distance_ft=60.5, level="hs", sar=SAR)
    seg = Segment(start_frame=0, ref_frame=0, view="behind", points=pts)
    return CalibrationFile(settings=st, segments=[seg], image_size=(720, 480))


def _traj(t):
    """Pinhole image of a ball approaching the camera: rises first, then dives (cf. test_vision_model)."""
    Z = 70 - 120 * t
    return 360 + 900 * (1 + 0.5 * t) / Z, 240 + 900 * (-5 + 2 * t + 16 * t * t) / Z


def _traj_geom():
    x0, y0 = _traj(0.0)
    x1, y1 = _traj(0.5)
    return G.view_geometry(behind_calibration((x0, y0), (x1, y1)).segment_calibs()[0], DT)


def _frag(tid, s_range, rng, sigma=0.3):
    tr = Track(tid=tid)
    for s in s_range:
        x, y = _traj(s * DT)
        tr.pts[s] = Pt(s=s, t=s * DT, k=s & 1, x=x + rng.normal(0, sigma), y=y + rng.normal(0, sigma), z=10.0,
                       attached=False, sigma=0.5, uid=s)
    return tr


def _render(bg, A, noise, rng, occluder=None):
    img = cv2.warpAffine(bg, G.to3(A)[:2], (720, 480), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_REFLECT)
    img = img + rng.normal(0, noise, img.shape)
    if occluder is not None:
        x0, y0, x1, y1 = occluder
        img[y0:y1, x0:x1] = 40 + rng.normal(0, noise, (y1 - y0, x1 - x0))
    return np.clip(img, 0, 255).astype(np.uint8)


def _register_run(opts, amp, noise, occluder=None, n=40, seed=0):
    """Share of registered samples and their max error (px) on a shaking, low-texture scene."""
    rng = np.random.default_rng(seed)
    bg = (110 + amp * (textured_background(seed=11) - 110)).astype(np.float32)
    st = Stabilizer(_render(bg, np.eye(3), noise, rng), SAR, True, None, opts)
    ok, errs = 0, []
    for s in range(n):
        A = G.similarity_from_params(4 * math.sin(s / 3), 3 * math.cos(s / 4), 0.1 * math.sin(s / 5), 1.0, SAR)
        fr = _render(bg, A, noise, rng, occluder if s > 5 else None)
        r = st.register(fr[(s & 1)::2], s & 1)
        if r.quality > 0:
            ok += 1
            errs.append(float(np.max(np.hypot(*(G.apply(r.A, PTS) - G.apply(A, PTS)).T))))
    return ok / n, np.array(errs)


LEGACY = dict(ransac_thr_max=0, quality_mode="legacy", lk_win=(21, 11), changed_patch_factor=0,
              max_jump_frac=0, keyframe_min_quality=0)


# ---------------------------------------------------------- stabilization ----
def test_noisy_low_texture_footage_is_registered():
    # weak texture (30 % of the usual contrast) + heavy tape noise: LK features are noisy
    # (~1-2 px) but numerous; the registration itself is good to ~1 px
    share, errs = _register_run(StabilizerOptions(), 0.3, 6.0)
    assert share >= 0.95
    assert np.median(errs) < 1.0 and errs.max() < 2.0
    # the former fixed 1 px inlier threshold + per-feature-rms quality rejected all of it
    share_legacy, _ = _register_run(StabilizerOptions(**LEGACY), 0.3, 6.0)
    assert share_legacy < 0.2


def test_occluder_appearing_after_reference_does_not_break_registration():
    # a player-sized region covers ~1/3 of the frame from sample 6 on (absent in the reference)
    share, errs = _register_run(StabilizerOptions(), 0.3, 4.0, occluder=(0, 150, 330, 480))
    assert share >= 0.95 and np.median(errs) < 1.0


def test_quality_reflects_transform_precision():
    # 100 features with 1.5 px noise each: the transform is good to ~0.2 px
    assert _quality(100, 150, 1.5) > 0.3
    assert _quality(100, 150, 1.5) > 1.8 * _quality(100, 150, 1.5, mode="legacy")
    assert _quality(0, 150, 0.5) == 0.0
    assert _quality(30, 100, 0.5) < 0.05                  # weak support stays weak


def test_implausible_jump_between_samples_is_rejected():
    ref = textured_background(seed=3).astype(np.uint8)
    st = Stabilizer(ref, SAR, True)
    st.A_prev = np.eye(3)
    near = StabResult(A=G.to3(G.similarity_from_params(8.0, -5.0, 0.1, 1.0, SAR)), quality=0.5)
    far = StabResult(A=G.to3(G.similarity_from_params(300.0, 0.0, 0.0, 1.0, SAR)), quality=0.5)
    assert st._continuous(near) is near
    assert st._continuous(far) is None
    st.fail_run = 10                                       # long failure run: larger allowance
    assert st._continuous(far) is far


# ---------------------------------------------------------------- geometry ----
def test_behind_corridor_covers_release_spread_and_backward_drift():
    cal = behind_calibration((360.0, 120.0), (420.0, 400.0)).segment_calibs()[0]
    g = G.view_geometry(cal, DT)
    e = g.axis_sq
    n = np.array([-e[1], e[0]])
    rel_sq = g.to_sq(g.release)[0]
    # 2 ft to the side of the release point, and 0.18 L behind it
    side = rel_sq + 2.0 * g.scale_rel * n
    back = rel_sq - 0.18 * g.length_sq * e
    for q in (side, back):
        assert g.in_corridor([[q[0] / SAR, q[1]]])[0]
    # ... but not arbitrarily large: far beside the release point is outside
    far = rel_sq + 6.0 * g.scale_rel * n
    assert not g.in_corridor([[far[0] / SAR, far[1]]])[0]
    old = np.asarray(cal.corridor_polygon(), np.float32)
    assert cv2.contourArea(g.corridor.astype(np.float32)) < 1.4 * cv2.contourArea(old)


# ---------------------------------------------------------------- detection ----
def test_ball_next_to_image_top_is_detected_in_behind_view():
    # zoomed behind view with the release point 16 px below the image top: the long
    # catch-end streak kernels (half height ~19 field rows) must not blank that region
    cal = behind_calibration((360.0, 16.0), (420.0, 400.0), rubber_half=30.0, plate_half=60.0)
    geom = G.view_geometry(cal.segment_calibs()[0], DT)
    rng = np.random.default_rng(0)
    base = rng.uniform(90, 110, (240, 720)).astype(np.float32)
    ball = (362.0, 18.0)
    det = Detector(geom, True, (240, 720), FinderOptions())
    assert det.kmax_h * 2 > ball[1]                       # the old margin would have covered the ball
    frames = []
    for s in range(50):
        p = s & 1
        img = base + rng.normal(0, 2.0, base.shape)
        if s == 25:
            yy, xx = np.mgrid[0:240, 0:720]
            d2 = ((xx - ball[0]) * SAR) ** 2 + ((2 * yy + p) - ball[1]) ** 2
            img = np.where(d2 <= (geom.d_rel / 2) ** 2, 230.0, img)
        frames += det.push(s, s * DT, p, p, np.clip(img, 0, 255).astype(np.uint8), np.eye(3)[:2], 1.0)
    frames += det.flush()
    c = [f for f in frames if f.s == 25][0].cands
    assert len(c)
    d = np.hypot((c["x"] - ball[0]) * SAR, c["y"] - ball[1])
    assert d.min() < 1.5 and c["z"][int(np.argmin(d))] > 8


# ---------------------------------------------------------------- tracking ----
def test_slow_and_fast_fragments_of_one_behind_flight_are_merged():
    geom = _traj_geom()
    o = FinderOptions()
    rng = np.random.default_rng(0)
    a = _frag(1, range(0, 10), rng)          # slow early part (image path rising)
    b = _frag(2, range(16, 25), rng)         # later part, image speed growing
    fit_track(a, geom, o)
    fit_track(b, geom, o)
    # the fragment models disagree where they meet (sharp bend), so only the joint fit can tell
    assert not _models_agree(a, b, a.arrays(True), b.arrays(True), 6, geom, o)
    m = try_merge(a, b, geom, o)
    assert m is not None and m.n_inliers == 19 and m.model.rms < 1.0


def test_unrelated_fragment_is_not_merged():
    geom = _traj_geom()
    o = FinderOptions()
    rng = np.random.default_rng(1)
    a = _frag(1, range(0, 10), rng)
    b = Track(tid=2)                          # something moving the other way, nearby in time
    for i, s in enumerate(range(14, 24)):
        b.pts[s] = Pt(s=s, t=s * DT, k=s & 1, x=440.0 - 4 * i, y=230.0 - 3 * i, z=10.0, attached=False, sigma=0.5)
    fit_track(a, geom, o)
    fit_track(b, geom, o)
    assert try_merge(a, b, geom, o) is None


def _traj_store(rng, s_range, clutter=3):
    st = CandStore(1000)
    uid = 0
    for s in s_range:
        x, y = _traj(s * DT)
        rows = [(x + rng.normal(0, 0.3), y + rng.normal(0, 0.3), 9.0)]
        for _ in range(clutter):
            rows.append((rng.uniform(300, 520), rng.uniform(120, 260), rng.uniform(3.5, 6.0)))
        arr = np.zeros(len(rows), CAND_DTYPE)
        for i, (cx, cy, z) in enumerate(rows):
            arr[i] = (uid, s, s * DT, s & 1, cx, cy, z, z, 1, 3.0, 0.0, False, 0.4, 0.0, 0)
            uid += 1
        st.add(DetFrame(s=s, t=s * DT, k=s & 1, parity=s & 1, A=np.eye(3), quality=1.0, cands=arr,
                        roi=(0, 0, 720, 240), zmap=np.zeros((240, 720), np.int16)))
    return st


def test_regrow_recovers_the_early_flight_along_the_model():
    geom = _traj_geom()
    o = FinderOptions()
    rng = np.random.default_rng(2)
    st = _traj_store(rng, range(0, 30))
    tr = Track(tid=1)
    for s in range(14, 30):                   # the tracker only got the later part
        c = st.get(s).cands[0]
        tr.pts[s] = Pt(s=s, t=float(c["t"]), k=s & 1, x=float(c["x"]), y=float(c["y"]), z=9.0, attached=False,
                       sigma=0.4, uid=int(c["uid"]))
    st.claim(tr)
    fit_track(tr, geom, o)
    n = regrow(tr, st, geom, o, 2)
    assert n >= 10
    inl = [p.s for p in tr.sorted() if p.inlier]
    assert min(inl) <= 3 and len(inl) >= 26
    # nothing from the clutter was taken
    for p in tr.sorted():
        if p.inlier:
            x, y = _traj(p.t)
            assert math.hypot(p.x - x, p.y - y) < 2.0


def test_regrow_relinks_interior_outliers_to_the_ball():
    # the local extension took a stationary clutter spot for some samples inside the flight
    geom = _traj_geom()
    o = FinderOptions()
    rng = np.random.default_rng(4)
    st = _traj_store(rng, range(0, 30))
    tr = Track(tid=1)
    for s in range(0, 30):
        c = st.get(s).cands[0]                              # the ball
        x, y = float(c["x"]), float(c["y"])
        if s % 3 == 1 and s < 15:                           # clutter instead of the ball
            x, y = 380.0, 230.0
        tr.pts[s] = Pt(s=s, t=s * DT, k=s & 1, x=x, y=y, z=9.0, attached=False, sigma=0.4)
    fit_track(tr, geom, o)
    assert sum(1 for p in tr.sorted() if not p.inlier) >= 4
    regrow(tr, st, geom, o, 2)
    bad = [p.s for p in tr.sorted() if math.hypot(p.x - _traj(p.t)[0], p.y - _traj(p.t)[1]) > 2.0]
    assert not bad and tr.n_inliers >= 28


def test_trim_drops_windup_stragglers():
    # two isolated detections 11-13 samples before the rest of the track that the model absorbs
    # as inliers (the ball held in the hand near where the flight starts)
    geom = _traj_geom()
    o = FinderOptions()
    rng = np.random.default_rng(5)
    tr = _frag(1, range(14, 30), rng)
    for s in (1, 3):
        x, y = _traj(s * DT)
        tr.pts[s] = Pt(s=s, t=s * DT, k=s & 1, x=x + 0.2, y=y - 0.2, z=4.0, attached=False, sigma=0.8)
    fit_track(tr, geom, o)
    assert tr.pts[1].inlier and tr.pts[3].inlier
    n = trim_sparse_ends(tr, geom, o)
    assert n == 2 and min(tr.pts) == 14 and tr.n_inliers == 16
    # a dense track is left alone
    assert trim_sparse_ends(_frag(2, range(0, 30), rng), geom, o) == 0


def test_regrow_is_behind_view_only():
    geom = _traj_geom()
    geom.view = "side"
    o = FinderOptions()
    rng = np.random.default_rng(3)
    st = _traj_store(rng, range(0, 30))
    tr = _frag(1, range(14, 30), rng)
    fit_track(tr, geom, o)
    assert regrow(tr, st, geom, o, 2) == 0


# ------------------------------------------------------------------ finder ----
def test_track_that_never_gets_past_the_release_point_is_rejected():
    cal = behind_calibration((360.0, 120.0), (420.0, 400.0)).segment_calibs()[0]
    scan = ScanInfo(mode="interlaced", field_order="bff", confidence="high", source="override",
                    sample_rate=1.0 / DT)
    f = PitchFinder(cal, scan, DT, (240, 720), np.full((480, 720), 100, np.uint8), FinderOptions())
    g = f.geom
    e = g.axis_sq

    qd = g.d_catch / g.d_rel

    def track(p0, p1, perspective=False):
        tr = Track(tid=1)
        for i, s in enumerate(range(100, 116)):
            w = i / 15
            if perspective:      # ball approaching the camera at constant speed: Z linear in time
                z0, z1 = 1 + p0 * (qd - 1), 1 + p1 * (qd - 1)          # Z_rel / Z at both ends
                zr = 1.0 / (1.0 / z0 + w * (1.0 / z1 - 1.0 / z0))
                u = (zr - 1) / (qd - 1)
            else:
                u = p0 + (p1 - p0) * w
            q = g.to_sq(g.release)[0] + u * g.length_sq * e
            tr.pts[s] = Pt(s=s, t=s * DT, k=s & 1, x=q[0] / SAR, y=q[1], z=10.0, attached=False, sigma=0.5)
        fit_track(tr, g, f.opts)
        return tr, PitchTrack(track_id=1, points=[], status="pitch",
                              release=EventTime(t=100 * DT, sigma=0.02, method="first_free_field"))

    # a throw-back lob coming down onto the pitcher: moves toward the release point
    tr, pt = track(-0.18, -0.03)
    f._behind_past_release_check(tr, pt)
    assert pt.status == "rejected" and pt.reject_code == "NO_PROGRESS"
    assert pt.diagnostics["no_progress_reason"] == "never_past_release_point"
    # a real pitch gets on toward the catch point, its image speed growing with perspective
    tr, pt = track(0.0, 0.45, perspective=True)
    f._behind_past_release_check(tr, pt)
    assert pt.status == "pitch" and pt.diagnostics["approach_score"] > 0.8
    # the same image path at constant image speed is not something approaching the camera
    tr, pt = track(0.0, 0.45)
    f._behind_past_release_check(tr, pt)
    assert pt.status == "rejected" and pt.reject_code == "THROWBACK"
    assert "not approaching" in pt.diagnostics["throwback_reason"]
