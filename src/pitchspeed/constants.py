"""Field dimensions, league presets and default assumptions.

Units: feet, seconds.  World frame (shared by the analyzer and the synthetic
generator): origin at the back point (apex) of home plate on the ground,
+y toward the pitcher, +x toward the first-base side, +z up.

Sources and confidence for these numbers: docs/design-notes/research.md.
Values marked ASSUMED are defaults the user can override; they are not
measurements.
"""
from __future__ import annotations

from dataclasses import dataclass

# --- home plate / rubber geometry (Official Baseball Rules 2.02, 2.04) -----
PLATE_DEPTH_FT = 17.0 / 12.0          # apex (back point) to front edge
PLATE_FRONT_Y_FT = PLATE_DEPTH_FT     # y of the plate's front edge
PLATE_CENTER_Y_FT = PLATE_DEPTH_FT / 2.0   # middle of the white plate, front-to-back
PLATE_WIDTH_FT = 17.0 / 12.0
RUBBER_WIDTH_FT = 2.0                 # 24 in
RUBBER_DEPTH_FT = 0.5                 # 6 in
# The league distance is measured from the FRONT edge of the rubber.
# The calibration click asks for the MIDDLE of the rubber (easier to see),
# which is RUBBER_DEPTH_FT / 2 behind the front edge.
RUBBER_CENTER_OFFSET_FT = RUBBER_DEPTH_FT / 2.0

BALL_DIAMETER_FT = 2.90 / 12.0        # baseball, ~2.86-2.94 in
SOFTBALL_DIAMETER_FT = 3.82 / 12.0    # 12 in circumference


@dataclass(frozen=True)
class Level:
    key: str
    label: str
    distance_ft: float        # rubber front edge -> plate apex
    sport: str                # 'baseball' | 'softball'


# Hand-marked catch at the catcher's SQUEEZE instead of the ball reaching the glove: the
# squeeze comes about this much later.  Measured once: 16 radar-timed pitches marked this way
# read ~4 mph low at 30 frames/s (about 0.75 frame); the sigma covers how rough that is.
CATCH_CUES = ("arrival", "squeeze")
SQUEEZE_DELAY_S = 0.025
SQUEEZE_DELAY_SIGMA_S = 0.012

LEVELS: dict[str, Level] = {
    lv.key: lv
    for lv in [
        Level("hs", "High school / college / MLB / 13+ (60 ft 6 in)", 60.5, "baseball"),
        Level("54", "PONY Pony / some 13U (54 ft)", 54.0, "baseball"),
        Level("50", "Little League Intermediate / PONY Bronco (50 ft)", 50.0, "baseball"),
        Level("ll", "Little League Majors and below / PONY Mustang (46 ft)", 46.0, "baseball"),
        Level("40", "PONY Pinto (40 ft)", 40.0, "baseball"),
        Level("sb43", "Fastpitch softball, high school / college (43 ft)", 43.0, "softball"),
        Level("sb40", "Fastpitch softball, 12U (40 ft)", 40.0, "softball"),
        Level("sb35", "Fastpitch softball, 10U (35 ft)", 35.0, "softball"),
    ]
}
LEVEL_ALIASES = {"mlb": "hs", "college": "hs", "60.5": "hs", "46": "ll", "43": "sb43"}


def get_level(key: str) -> Level:
    k = key.strip().lower()
    k = LEVEL_ALIASES.get(k, k)
    if k not in LEVELS:
        raise KeyError(f"unknown level {key!r}; choose one of {sorted(LEVELS)} or give --distance")
    return LEVELS[k]


# --- ASSUMED defaults for things the video usually cannot show -------------
def default_extension_ft(distance_ft: float, sport: str = "baseball") -> float:
    """Release point distance in front of the rubber (ASSUMED; sigma 1 ft).

    MLB averages ~6.4 ft (Statcast); amateurs are shorter.  No reliable
    youth averages were found, so these are deliberately rough.
    """
    if sport == "softball":
        return 4.0
    if distance_ft >= 58.0:
        return 5.5
    if distance_ft >= 48.0:
        return 4.5
    return 4.0


EXTENSION_SIGMA_FT = 1.0
CATCH_BEHIND_FT = 1.5          # glove position behind the plate apex (ASSUMED)
CATCH_SIGMA_FT = 0.75
CATCH_HEIGHT_FT = 2.5


def default_release_height_ft(distance_ft: float, sport: str = "baseball") -> float:
    if sport == "softball":
        return 2.0
    return 5.5 if distance_ft >= 58.0 else 4.5


# Physical constants
G_FTPS2 = 32.174
MPH_TO_FTPS = 22.0 / 15.0
FTPS_TO_MPH = 15.0 / 22.0
FT_PER_M = 1.0 / 0.3048

NTSC_FIELD_DT = 1001.0 / 60000.0
NTSC_FRAME_DT = 1001.0 / 30000.0
