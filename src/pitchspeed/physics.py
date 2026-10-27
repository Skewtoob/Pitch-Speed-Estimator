"""1-D ball flight model along the pitch axis with quadratic drag.

    dv/dt = -k v^2        (k in 1/ft)
    v(t)  = v0 / (1 + k v0 t)
    s(t)  = ln(1 + k v0 t) / k           distance travelled after time t
    v(s)  = v0 exp(-k s)
    T(L)  = (exp(k L) - 1) / (k v0)      time to travel L

Gravity and Magnus forces are (nearly) perpendicular to the axis, so they
only enter at second order; `slope_correction` converts the horizontal
release speed into the 3-D release speed.

Default drag: rho = 1.20 kg/m^3, Cd = 0.33, baseball m = 0.145 kg,
d = 73.7 mm  ->  k = 0.00178 / ft.  Cd for a pitched baseball is about
0.30-0.45 (seam dependent), so k carries ~15% (>= 65 mph) to 25% (slower)
relative uncertainty.  See docs/design-notes/math.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from pitchspeed.constants import FT_PER_M, G_FTPS2

BASEBALL_MASS_KG = 0.145
BASEBALL_DIAM_M = 0.0737
SOFTBALL_MASS_KG = 0.184
SOFTBALL_DIAM_M = 0.0970
CD_BASEBALL = 0.33
CD_SOFTBALL = 0.40   # ASSUMED; less well established than baseball


def air_density(altitude_ft: float = 0.0, temp_f: float | None = None) -> float:
    """Approximate air density (kg/m^3).  Default 20 C at sea level -> ~1.20."""
    t_c = 20.0 if temp_f is None else (temp_f - 32.0) * 5.0 / 9.0
    alt_m = altitude_ft / FT_PER_M
    return 1.225 * (288.15 / (273.15 + t_c)) * math.exp(-alt_m / 8500.0)


def drag_k_per_ft(rho: float, cd: float, mass_kg: float, diam_m: float) -> float:
    area = math.pi * (diam_m / 2.0) ** 2
    k_per_m = rho * cd * area / (2.0 * mass_kg)
    return k_per_m / FT_PER_M


@dataclass(frozen=True)
class DragModel:
    k: float              # 1/ft
    rel_sigma: float      # relative 1-sigma uncertainty of k

    @staticmethod
    def default(sport: str = "baseball", altitude_ft: float = 0.0,
                temp_f: float | None = None, slow: bool = False) -> "DragModel":
        rho = air_density(altitude_ft, temp_f)
        if sport == "softball":
            return DragModel(drag_k_per_ft(rho, CD_SOFTBALL, SOFTBALL_MASS_KG, SOFTBALL_DIAM_M), 0.30)
        return DragModel(drag_k_per_ft(rho, CD_BASEBALL, BASEBALL_MASS_KG, BASEBALL_DIAM_M),
                         0.25 if slow else 0.15)


def dist_travelled(v0: float, t: float, k: float) -> float:
    if k <= 0:
        return v0 * t
    return math.log1p(k * v0 * t) / k


def speed_at_time(v0: float, t: float, k: float) -> float:
    return v0 / (1.0 + k * v0 * t)


def speed_after_distance(v0: float, s: float, k: float) -> float:
    return v0 * math.exp(-k * s)


def time_to_travel(v0: float, L: float, k: float) -> float:
    if k <= 0:
        return L / v0
    return math.expm1(k * L) / (k * v0)


def v0_from_tof(L: float, T: float, k: float) -> float:
    """Initial (release) speed that covers distance L in time T under drag k."""
    if T <= 0:
        raise ValueError("flight time must be positive")
    if k <= 0:
        return L / T
    return math.expm1(k * L) / (k * T)


def slope_correction(L: float, T: float, z_release: float, z_catch: float) -> float:
    """Factor converting horizontal release speed to 3-D release speed.

    tan(theta) = (z_catch - z_release + g T^2 / 2) / L   (Magnus ignored)
    Returns sqrt(1 + tan^2 theta)  (>= 1).
    """
    tan_t = (z_catch - z_release + 0.5 * G_FTPS2 * T * T) / L
    return math.sqrt(1.0 + tan_t * tan_t)
