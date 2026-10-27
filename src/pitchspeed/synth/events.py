"""Pitch schedule and distractor events (independent of rendering).

Every pitch draws a fixed block of uniforms from its own sequential RNG
stream, so toggling a distractor never changes later pitches, and the same
seed gives the same pitch list for side and behind views.

Ball-in-hand: for `hand_time_s` (60 ms) before release the ball rides the
hand on a 2 ft-radius arc whose tangential speed ramps from 0 to v0; position
and velocity are continuous at release, so the first sighting of the ball is
NOT the release.  Glove: set 0.7 s early, gives 0.3 ft over 60 ms after the
catch; the ball is hidden from t_catch on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pitchspeed.synth.camera import Camera
from pitchspeed.synth.config import FIELD_DT, MPH, LevelGeom, rng_for
from pitchspeed.synth.physics import Trajectory, aim, drag_k, throw_velocity

N_PITCH_UNIFORMS = 16
GLOVE_REST = np.array([0.0, -1.5, 2.2])


def smooth(x):
    """Smoothstep 0 -> 1 on [0, 1] (scalar or array)."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _uni(r01: float, rng: list) -> float:
    return float(rng[0] + (rng[1] - rng[0]) * r01)


@dataclass
class HandArc:
    """Throwing hand path ending tangentially at the release point with the ball's velocity."""
    t_release: float
    release_pos: np.ndarray
    release_vel: np.ndarray
    duration: float = 0.06
    radius: float = 2.0

    def __post_init__(self):
        v0 = np.asarray(self.release_vel, dtype=float)
        self.vh = v0 / np.linalg.norm(v0)
        nn = np.array([0.0, 0.0, 1.0]) - self.vh[2] * self.vh
        self.nn = nn / np.linalg.norm(nn)
        self.center = np.asarray(self.release_pos, dtype=float) - self.radius * self.nn
        self.omega = float(np.linalg.norm(v0)) / self.radius

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """(hand position, arc centre ~ shoulder) at time t."""
        T = self.duration
        tau = float(np.clip(t - self.t_release, -T, T))
        if tau < 0:
            phi = self.omega * (tau + tau * tau / (2 * T))
        else:
            phi = self.omega * (tau - tau * tau / (2 * T))
        return self.center + self.radius * (self.nn * math.cos(phi) + self.vh * math.sin(phi)), self.center


@dataclass
class PitchEvent:
    id: int
    t_release: float
    t_catch: float
    t_plate: float
    release_pos: np.ndarray
    release_vel: np.ndarray
    catch_pos: np.ndarray
    catch_vel: np.ndarray
    plate_vel: np.ndarray
    k_per_ft: float
    cd: float
    magnus: np.ndarray
    wind: np.ndarray
    target: list
    swing: bool
    catch_behind_ft: float
    traj: Trajectory
    hand: HandArc
    path_len_ft: float = 0.0

    @property
    def t_hand(self) -> float:
        """Time the ball appears in the hand (start of the in-hand arc)."""
        return self.t_release - self.hand.duration

    def ball_pos(self, t: float) -> Optional[np.ndarray]:
        if self.t_hand <= t < self.t_release:
            return self.hand.at(t)[0]
        if self.t_release <= t <= self.t_catch:
            return self.traj.state(t)[0]
        return None

    def ball_label(self, t: float) -> str:
        return "in_hand" if t < self.t_release else "flight"

    def glove_pos(self, t: float) -> np.ndarray:
        pc, tc = self.catch_pos, self.t_catch
        base = pc + np.array([0.3, 0.0, -0.2]) * (1 - smooth((t - (tc - 0.7)) / 0.5))
        give = (self.catch_vel / np.linalg.norm(self.catch_vel)) * 0.3 * smooth((t - tc) / 0.06)
        ret = smooth((t - tc - 0.8) / 0.5)
        return base + give * (1 - ret) + ret * (GLOVE_REST - base)

    @property
    def swing_window(self) -> tuple[float, float]:
        return self.t_plate - 0.10, self.t_plate + 0.06

    def summary(self) -> dict:
        return dict(id=self.id, t_release=self.t_release, t_catch=self.t_catch)


@dataclass
class ThrowEvent:
    """Catcher's throwback to the pitcher (a ball in flight that is NOT a pitch)."""
    t_start: float
    t_end: float
    traj: Trajectory
    speed_mph: float
    after_pitch: int
    p0: np.ndarray
    p1: np.ndarray

    def ball_pos(self, t: float) -> Optional[np.ndarray]:
        return self.traj.state(t)[0] if self.t_start <= t <= self.t_end else None


@dataclass
class BirdEvent:
    t0: float
    dur: float
    p0: np.ndarray
    v: np.ndarray

    def pos(self, t: float) -> Optional[np.ndarray]:
        if self.t0 <= t <= self.t0 + self.dur:
            return self.p0 + self.v * (t - self.t0)
        return None


@dataclass
class Events:
    pitches: list
    throws: list
    birds: list
    warnings: list = field(default_factory=list)
    requested: int = 0


def _pitch_params(P: dict, u: np.ndarray, override: dict) -> dict:
    """Map a block of uniforms to pitch parameters (overrides win)."""
    keys = ("speed_mph", "cd", "release_x_ft", "extension_ft", "release_z_ft", "magnus_x", "magnus_z",
            "wind_mph", "target_x_ft", "target_z_ft", "catch_behind_ft")
    out = {k: _uni(u[i], P[k]) for i, k in enumerate(keys)}
    for k in keys:
        if k in override and override[k] is not None:
            out[k] = float(override[k])
    return out


def _make_pitch(pid: int, t: float, prm: dict, swing: bool, lev: LevelGeom, hand_time: float) -> PitchEvent:
    k = drag_k(prm["cd"])
    rel = np.array([prm["release_x_ft"], lev.distance_ft - prm["extension_ft"], lev.mound_h_ft + prm["release_z_ft"]])
    acc = np.array([prm["magnus_x"], 0.0, prm["magnus_z"]])
    wind = np.array([0.0, -prm["wind_mph"] * MPH, 0.0])
    tgt = [prm["target_x_ft"], prm["target_z_ft"]]
    y_plate = 17.0 / 12.0
    v0 = aim(rel, prm["speed_mph"] * MPH, tgt, y_plate, k, acc, wind, t)
    ycatch = -prm["catch_behind_ft"]
    tr = Trajectory(rel, v0, k, acc, wind, t, stop_y=ycatch)
    tc = tr.time_at_y(ycatch)
    tp = tr.time_at_y(y_plate)
    pc, vc = tr.state(tc)
    _, vp = tr.state(tp)
    hand = HandArc(t, rel, v0, duration=hand_time)
    return PitchEvent(id=pid, t_release=t, t_catch=tc, t_plate=tp, release_pos=rel, release_vel=v0,
                      catch_pos=pc, catch_vel=vc, plate_vel=vp, k_per_ft=k, cd=prm["cd"], magnus=acc,
                      wind=wind, target=tgt, swing=swing, catch_behind_ft=prm["catch_behind_ft"],
                      traj=tr, hand=hand, path_len_ft=tr.path_length(t, tc))


def _make_throwback(p: PitchEvent, u: np.ndarray, lev: LevelGeom) -> ThrowEvent:
    ts = p.t_catch + 1.6 + 0.8 * u[0]
    p0 = np.array([0.3, -2.8, 4.8])
    p1 = np.array([0.0, lev.distance_ft - 1.0, lev.mound_h_ft + 5.0])
    sp = (40.0 + 15.0 * u[1]) * MPH
    vt = throw_velocity(p0, p1, sp)
    trt = Trajectory(p0, vt, 0.0018, np.zeros(3), np.zeros(3), ts, tmax=2.5)
    te = trt.time_at_y(p1[1]) if trt.S[:, 1].max() > p1[1] else ts + 1.2
    return ThrowEvent(t_start=ts, t_end=te, traj=trt, speed_mph=sp / MPH, after_pitch=p.id, p0=p0, p1=p1)


def _make_birds(cfg: dict, cam: Camera) -> list:
    """Birds crossing the base camera's view (placed by back-projection so they are visible)."""
    r = rng_for(cfg["seed"], 4)
    birds = []
    dur = 4.0
    T = float(cfg["duration_s"])
    for _ in range(int(cfg["distractors"]["birds"])):
        t_mid = float(r.uniform(0.5, max(1.0, T - 0.5)))
        u = float(r.uniform(0.15, 0.85)) * (cam.W - 1)
        v = float(r.uniform(0.05, 0.45)) * (cam.H - 1)
        depth = float(r.uniform(25.0, 120.0))
        d_cam = np.array([(u - cam.cx) / cam.fx0, (v - cam.cy) / cam.fy0, 1.0])
        P_mid = cam.C + cam.R0.T @ (d_cam * depth)
        right = cam.R0[0].copy()
        right[2] = 0.0
        right /= max(np.linalg.norm(right), 1e-9)
        speed = float(r.uniform(15.0, 40.0)) * (1 if r.random() < 0.5 else -1)
        vel = right * speed + np.array([0.0, 0.0, float(r.uniform(-2.0, 2.0))])
        birds.append(BirdEvent(t0=t_mid - dur / 2, dur=dur, p0=P_mid - vel * dur / 2, v=vel))
    return birds


def make_events(cfg: dict, lev: LevelGeom, cam: Camera) -> Events:
    """Pitches (+ throwbacks) and birds for a resolved config."""
    P = cfg["pitches"]
    D = cfg["distractors"]
    r = rng_for(cfg["seed"], 2)
    duration = float(cfg["duration_s"])
    end_limit = duration - float(P.get("end_margin_s", 0.5))
    hand_time = float(P.get("hand_time_s", 0.06))
    schedule = P.get("schedule")
    n_req = len(schedule) if schedule is not None else int(P["count"])
    ev = Events(pitches=[], throws=[], birds=_make_birds(cfg, cam), requested=n_req)
    t = float(P["first_t"])
    first = True
    for i in range(n_req):
        u = r.random(N_PITCH_UNIFORMS)
        ov = dict(schedule[i]) if schedule is not None else {}
        if ov.get("t_release") is not None:
            t = float(ov["t_release"])
        else:
            if not first:
                t = ev.pitches[-1].t_catch + _uni(u[13], P["gap_s"])
            t += FIELD_DT * u[14]                     # random sub-field phase
        first = False
        prm = _pitch_params(P, u, ov)
        swing = bool(ov["swing"]) if ov.get("swing") is not None else bool(u[11] < D["swing_prob"])
        p = _make_pitch(len(ev.pitches) + 1, t, prm, swing, lev, hand_time)
        if p.t_catch > end_limit:
            ev.warnings.append(f"pitch schedule truncated: pitch {i + 1} of {n_req} would end at "
                               f"t={p.t_catch:.2f} s, past duration_s - end_margin_s = {end_limit:.2f} s")
            break
        ev.pitches.append(p)
        if D["throwback"]:
            th = _make_throwback(p, u[[12, 15]], lev)
            if th.t_end < duration:
                ev.throws.append(th)
    return ev
