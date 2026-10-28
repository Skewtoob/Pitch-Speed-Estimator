"""Ball flight physics of the generator (deliberately richer than the analyzer's).

    a = -k |v - w| (v - w) + a_M + (0, 0, -g)

with quadratic drag k = rho Cd A / (2 m) (1/ft), a constant Magnus-like
acceleration a_M and wind w.  Integration: classic RK4 at a fixed step
(0.25 ms) with a cubic-Hermite dense output, so position and velocity are
O(dt^4)-accurate at any time.  This module is independent of
pitchspeed.physics on purpose: the analyzer's 1-D model error gets measured
against it instead of being assumed away.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from pitchspeed.synth.config import BALL_R_FT, G

AIR_DENSITY_LB_FT3 = 0.0749        # ~1.20 kg/m^3 (20 C, sea level)
BALL_MASS_LB = 5.125 / 16.0


def drag_k(cd: float, rho_lb_ft3: float = AIR_DENSITY_LB_FT3, r_ft: float = BALL_R_FT,
           mass_lb: float = BALL_MASS_LB) -> float:
    """Quadratic drag constant k (1/ft) of a baseball."""
    return 0.5 * rho_lb_ft3 * cd * math.pi * r_ft ** 2 / mass_lb


class Trajectory:
    """Integrated 3-D ball trajectory starting at time t0."""

    def __init__(self, p0, v0, k: float, accel=(0.0, 0.0, 0.0), wind=(0.0, 0.0, 0.0),
                 t0: float = 0.0, dt: float = 1 / 4000.0, tmax: float = 3.0,
                 stop_y: Optional[float] = None, stop_z: Optional[float] = None):
        self.t0, self.dt = float(t0), float(dt)
        ax, ay, az = (float(a) for a in accel)
        wx, wy, wz = (float(w) for w in wind)
        az -= G
        k = float(k)

        def f(s):
            px, py, pz, vx, vy, vz = s
            rx, ry, rz = vx - wx, vy - wy, vz - wz
            sp = k * math.sqrt(rx * rx + ry * ry + rz * rz)
            return (vx, vy, vz, ax - sp * rx, ay - sp * ry, az - sp * rz)

        s = tuple(float(x) for x in list(p0) + list(v0))
        S = [s]
        h = self.dt
        for _ in range(int(tmax / h)):
            k1 = f(s)
            k2 = f(tuple(a + 0.5 * h * b for a, b in zip(s, k1)))
            k3 = f(tuple(a + 0.5 * h * b for a, b in zip(s, k2)))
            k4 = f(tuple(a + h * b for a, b in zip(s, k3)))
            s = tuple(a + h / 6.0 * (b1 + 2 * b2 + 2 * b3 + b4) for a, b1, b2, b3, b4 in zip(s, k1, k2, k3, k4))
            S.append(s)
            if stop_y is not None and s[1] < stop_y - 2.0:
                break
            if stop_z is not None and s[2] < stop_z:
                break
        self.S = np.array(S)

    @property
    def t_end(self) -> float:
        return self.t0 + (len(self.S) - 1) * self.dt

    def state(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """(position, velocity) at time t (cubic Hermite on the RK4 states)."""
        p, v = self.states(np.array([t], dtype=float))
        return p[0], v[0]

    def position(self, t: float) -> np.ndarray:
        return self.state(t)[0]

    def states(self, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized dense output: positions and velocities, shape (n, 3) each."""
        x = (np.asarray(ts, dtype=float) - self.t0) / self.dt
        i = np.clip(np.floor(x).astype(np.int64), 0, len(self.S) - 2)
        u = (x - i)[:, None]
        h = self.dt
        s0, s1 = self.S[i], self.S[i + 1]
        p0, v0, p1, v1 = s0[:, :3], s0[:, 3:], s1[:, :3], s1[:, 3:]
        u2, u3 = u * u, u * u * u
        p = (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * h * v0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * h * v1
        v = ((6 * u2 - 6 * u) / h * p0 + (3 * u2 - 4 * u + 1) * v0 + (-6 * u2 + 6 * u) / h * p1
             + (3 * u2 - 2 * u) * v1)
        return p, v

    def time_at(self, axis: int, value: float, increasing: bool = False) -> float:
        """First time coordinate `axis` crosses `value` (downward, or upward if `increasing`)."""
        c = self.S[:, axis]
        hit = np.nonzero(c >= value if increasing else c <= value)[0]
        if len(hit) == 0:
            raise ValueError(f"trajectory never reaches {'xyz'[axis]}={value:.3f} ft "
                             f"(range {c.min():.3f}..{c.max():.3f})")
        i = max(int(hit[0]) - 1, 0)
        lo, hi = self.t0 + i * self.dt, self.t0 + (i + 1) * self.dt
        for _ in range(60):
            m = 0.5 * (lo + hi)
            q = self.state(m)[0][axis]
            if (q < value) if increasing else (q > value):
                lo = m
            else:
                hi = m
        return 0.5 * (lo + hi)

    def time_at_y(self, y: float) -> float:
        """First time the ball's y coordinate drops to `y` (a pitch travels toward -y)."""
        return self.time_at(1, y, increasing=False)

    def path_length(self, ta: float, tb: float, n: int = 4000) -> float:
        P, _ = self.states(np.linspace(ta, tb, n + 1))
        return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def aim(p0, speed: float, target_xz, y_target: float, k: float, accel, wind, t0: float = 0.0) -> np.ndarray:
    """Launch velocity of magnitude `speed` so the ball crosses plane y=y_target at (x, z).

    2-D Newton shooting on (yaw, pitch) with finite-difference Jacobian.
    """
    p0 = np.asarray(p0, dtype=float)
    d = np.array([target_xz[0], y_target, target_xz[1]]) - p0
    yaw = math.atan2(d[0], -d[1])
    pitch = math.atan2(d[2], math.hypot(d[0], d[1]))

    def shoot(yw: float, pt: float):
        v = speed * np.array([math.sin(yw) * math.cos(pt), -math.cos(yw) * math.cos(pt), math.sin(pt)])
        tr = Trajectory(p0, v, k, accel, wind, t0, dt=1 / 2000.0, tmax=3.0, stop_y=y_target)
        p, _ = tr.state(tr.time_at_y(y_target))
        return np.array([p[0], p[2]]), v

    v = None
    for _ in range(8):
        e, v = shoot(yaw, pitch)
        err = e - np.asarray(target_xz, dtype=float)
        if np.abs(err).max() < 1e-4:
            break
        J = np.zeros((2, 2))
        hstep = 1e-4
        J[:, 0] = (shoot(yaw + hstep, pitch)[0] - e) / hstep
        J[:, 1] = (shoot(yaw, pitch + hstep)[0] - e) / hstep
        dy, dp = np.linalg.solve(J, -err)
        yaw += dy
        pitch += dp
    return v


def throw_velocity(p0, p1, speed: float) -> np.ndarray:
    """Gravity-only ballistic aim (catcher throwback), low-arc solution."""
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    d = p1 - p0
    dh = math.hypot(d[0], d[1])
    dz = d[2]
    v2 = speed ** 2
    disc = v2 ** 2 - G * (G * dh ** 2 + 2 * dz * v2)
    th = math.atan((v2 - math.sqrt(max(disc, 0.0))) / (G * dh))
    u = np.array([d[0], d[1], 0.0]) / dh
    return speed * (math.cos(th) * u + np.array([0.0, 0.0, math.sin(th)]))
