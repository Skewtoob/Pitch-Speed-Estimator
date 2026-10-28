"""Calibration: clicked landmarks + per-video settings.

A calibration file holds shared settings (level/distance, extension, ...)
and one or more *segments*.  A segment covers a frame range filmed from one
camera setup and holds the clicked points (frame pixel coordinates of the
segment's reference frame).  Modules work with a resolved `Calibration`
(one segment + the shared settings), obtained with
`CalibrationFile.for_frame(i)` or `CalibrationFile.segment_calibs()`.

A segment with view 'manual' has no points: its pitches are marked by hand
(see REQUIRED_POINTS).

Point names (all in frame pixel coords of `ref_frame`, stored pixels):
  rubber          middle of the pitching rubber (on the ground)      side: required
  plate           middle of home plate (front-to-back middle)        side: required
  release         ball in the pitcher's hand at release              behind: required, side: optional
  catch           ball in the catcher's glove                        behind: required, side: optional
  rubber_left/right   the two ends of the rubber (24 in apart)       optional (behind: gives scale)
  plate_left/right    the two front corners of the plate (17 in)     optional (behind: gives scale)
  vertical_top/bottom two points on something vertical (pole, post) optional (side: plumb direction)
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from pitchspeed import constants as C

SCHEMA = "pitchspeed.calibration/1"

# View 'manual': pitches are marked by hand (release and catch fields), so no
# points are clicked; the speed then comes from the time of flight alone, which
# needs only the settings (distance, extension, heights) - any camera angle,
# even a moving or zooming camera.  The automatic finder cannot run on it.
REQUIRED_POINTS = {
    "side": ["rubber", "plate"],
    "behind": ["release", "catch"],
    "manual": [],
}
OPTIONAL_POINTS = {
    "side": ["release", "catch", "vertical_top", "vertical_bottom"],
    "behind": ["rubber_left", "rubber_right", "plate_left", "plate_right", "rubber", "plate"],
    "manual": [],
}
AUTOMATIC_VIEWS = ("side", "behind")
POINT_HELP = {
    "rubber": "MIDDLE of the pitching rubber (white slab on the mound)",
    "plate": "MIDDLE of home plate",
    "release": "the ball in the pitcher's hand at the moment of release",
    "catch": "the catcher's glove where the ball is caught",
    "rubber_left": "LEFT end of the pitching rubber",
    "rubber_right": "RIGHT end of the pitching rubber",
    "plate_left": "LEFT front corner of home plate (pitcher's side edge)",
    "plate_right": "RIGHT front corner of home plate (pitcher's side edge)",
    "vertical_top": "TOP of something truly vertical (pole, post, backstop upright)",
    "vertical_bottom": "BOTTOM of the same vertical object",
}


# ------------------------------------------------------- settings sanity
# Longest plausible release extension: MLB's longest are ~8 ft (Statcast);
# amateurs and youth are shorter.  Never more than a quarter of the pitching
# distance (youth / softball).  Values beyond this are typing mistakes
# (inches typed as feet, the distance typed as the extension, ...).
EXTENSION_MAX_FT = 10.0
EXTENSION_MAX_FRACTION = 0.25
DISTANCE_RANGE_FT = (20.0, 100.0)     # leagues use 35-60.5 ft


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def extension_limit_ft(distance_ft: float) -> float:
    """Largest extension accepted for this pitching distance (ft)."""
    return min(EXTENSION_MAX_FT, EXTENSION_MAX_FRACTION * float(distance_ft))


def settings_errors(st: "Settings") -> list[str]:
    """Settings values that make a speed estimate impossible, in plain words.

    The speed estimator reports no number while any of these is present
    (estimate.speed), and `Calibration.validate` lists them as warnings that
    start with 'IMPOSSIBLE SETTING' (they are not raised: the calibration
    window cannot edit these values and must still be able to save the
    user's clicks).  Empty list = fine.
    """
    errs: list[str] = []

    def bad(name, v, lo=None, hi=None, unit="ft", hint=""):
        if not _finite(v):
            errs.append(f"{name} is not a number ({v!r})")
            return True
        v = float(v)
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            rng = (f"{lo:g} to {hi:g} {unit}" if lo is not None and hi is not None
                   else (f"at least {lo:g} {unit}" if lo is not None else f"at most {hi:g} {unit}"))
            errs.append(f"{name} {v:g} {unit} is impossible (must be {rng}){hint}")
            return True
        return False

    d_ok = not bad("rubber-to-plate distance", st.distance_ft, *DISTANCE_RANGE_FT,
                   hint="; give --distance in FEET (e.g. 60.5, 46)")
    if st.extension_ft is not None:
        lim = extension_limit_ft(st.distance_ft) if d_ok else EXTENSION_MAX_FT
        e = st.extension_ft
        hint = "; the extension is in FEET (typical 4-7 ft, youth 3-5 ft)"
        if _finite(e) and float(e) >= 12.0:
            hint += f" - did you enter inches? {float(e):g} in = {float(e) / 12.0:.1f} ft"
        hint += "; re-run 'pitchspeed calibrate' with --extension in feet"
        bad("pitcher extension", e, 0.0, lim, hint=hint)
    bad("extension uncertainty", st.extension_sigma_ft, 0.0, None)
    bad("catch point behind the plate", st.catch_behind_ft, -3.0, 8.0)
    bad("catch point uncertainty", st.catch_sigma_ft, 0.0, None)
    bad("catch height", st.catch_height_ft, 0.0, 8.0)
    if st.release_height_ft is not None:
        bad("release height", st.release_height_ft, 0.5, 9.0)
    bad("altitude", st.altitude_ft, -1500.0, 15000.0)
    if st.temp_f is not None:
        bad("air temperature", st.temp_f, -40.0, 130.0, unit="F")
    return errs


@dataclass
class Settings:
    """Shared, per-video settings (user-editable)."""
    distance_ft: float = 60.5            # rubber front edge -> plate apex
    level: Optional[str] = "hs"
    sport: str = "baseball"
    extension_ft: Optional[float] = None    # None -> constants.default_extension_ft
    extension_sigma_ft: float = C.EXTENSION_SIGMA_FT
    catch_behind_ft: float = C.CATCH_BEHIND_FT
    catch_sigma_ft: float = C.CATCH_SIGMA_FT
    release_height_ft: Optional[float] = None
    catch_height_ft: float = C.CATCH_HEIGHT_FT
    pitcher_throws: Optional[str] = None     # 'R' | 'L' | None
    altitude_ft: float = 0.0
    temp_f: Optional[float] = None
    sar: float = 1.0                         # stored pixel aspect ratio used for the clicks


@dataclass
class Segment:
    start_frame: int = 0
    end_frame: Optional[int] = None          # exclusive; None = to end of video
    ref_frame: int = 0
    view: str = "side"                       # 'side' | 'behind' | 'manual' (marked by hand, no points)
    points: dict = field(default_factory=dict)       # name -> [x, y]
    masks: list = field(default_factory=list)        # [x0, y0, x1, y1] regions to ignore

    def contains(self, frame: int) -> bool:
        return frame >= self.start_frame and (self.end_frame is None or frame < self.end_frame)


@dataclass
class CalibrationFile:
    settings: Settings = field(default_factory=Settings)
    segments: list = field(default_factory=list)     # list[Segment]
    video: dict = field(default_factory=dict)        # fingerprint {name, size, n_frames, width, height}
    image_size: tuple = (720, 480)                   # (width, height) stored pixels
    notes: str = ""
    schema: str = SCHEMA

    # ---------------------------------------------------------------- I/O
    def to_dict(self) -> dict:
        d = asdict(self)
        d["image_size"] = list(self.image_size)
        return d

    @staticmethod
    def from_dict(d: dict) -> "CalibrationFile":
        if d.get("schema") not in (SCHEMA, None):
            raise ValueError(f"unsupported calibration schema {d.get('schema')!r}")
        st = Settings(**{k: v for k, v in d.get("settings", {}).items()
                         if k in Settings.__dataclass_fields__})
        segs = [Segment(**{k: v for k, v in s.items() if k in Segment.__dataclass_fields__})
                for s in d.get("segments", [])]
        return CalibrationFile(settings=st, segments=segs, video=d.get("video", {}),
                               image_size=tuple(d.get("image_size", (720, 480))),
                               notes=d.get("notes", ""))

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        os.replace(tmp, path)

    @staticmethod
    def load(path: str) -> "CalibrationFile":
        with open(path, "r", encoding="utf-8") as f:
            return CalibrationFile.from_dict(json.load(f))

    # ------------------------------------------------------------ resolve
    def segment_calibs(self) -> list["Calibration"]:
        return [Calibration(self.settings, seg, tuple(self.image_size)) for seg in self.segments]

    def for_frame(self, frame: int) -> Optional["Calibration"]:
        for seg in self.segments:
            if seg.contains(frame):
                return Calibration(self.settings, seg, tuple(self.image_size))
        return None

    def validate(self) -> list[str]:
        """Raise ValueError for fatal problems; return a list of warnings."""
        if not self.segments:
            raise ValueError("calibration has no segments (no points were clicked)")
        warns = []
        for i, c in enumerate(self.segment_calibs()):
            for w in c.validate():
                warns.append(f"segment {i}: {w}")
        return warns


class Calibration:
    """One segment + shared settings, with geometry helpers.

    Square-pixel space: (x * sar, y).  All public helpers take and return
    frame pixel coordinates (stored pixels) unless the name says `_sq`.
    """

    def __init__(self, settings: Settings, segment: Segment, image_size=(720, 480)):
        self.settings = settings
        self.segment = segment
        self.image_size = tuple(image_size)

    # ---- basic properties
    @property
    def view(self) -> str:
        return self.segment.view

    @property
    def sar(self) -> float:
        return float(self.settings.sar or 1.0)

    @property
    def ref_frame(self) -> int:
        return self.segment.ref_frame

    @property
    def distance_ft(self) -> float:
        return float(self.settings.distance_ft)

    @property
    def extension_ft(self) -> float:
        e = self.settings.extension_ft
        return float(e) if e is not None else C.default_extension_ft(self.distance_ft, self.settings.sport)

    @property
    def extension_is_assumed(self) -> bool:
        return self.settings.extension_ft is None

    @property
    def release_height_ft(self) -> float:
        h = self.settings.release_height_ft
        return float(h) if h is not None else C.default_release_height_ft(self.distance_ft, self.settings.sport)

    @property
    def release_y_ft(self) -> float:
        """World y (distance in front of the plate apex) of the release point."""
        return self.distance_ft - self.extension_ft

    @property
    def catch_y_ft(self) -> float:
        return -float(self.settings.catch_behind_ft)

    @property
    def rubber_center_y_ft(self) -> float:
        return self.distance_ft + C.RUBBER_CENTER_OFFSET_FT

    @property
    def plate_center_y_ft(self) -> float:
        return C.PLATE_CENTER_Y_FT

    def point(self, name: str) -> Optional[np.ndarray]:
        p = self.segment.points.get(name)
        return None if p is None else np.asarray(p, dtype=float)

    def has(self, name: str) -> bool:
        return name in self.segment.points and self.segment.points[name] is not None

    def to_sq(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return np.array([p[..., 0] * self.sar, p[..., 1]]).T if p.ndim > 1 else np.array([p[0] * self.sar, p[1]])

    def from_sq(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return np.array([q[..., 0] / self.sar, q[..., 1]]).T if q.ndim > 1 else np.array([q[0] / self.sar, q[1]])

    # ---- validation
    def missing_points(self) -> list[str]:
        return [n for n in REQUIRED_POINTS[self.view] if not self.has(n)]

    def validate(self) -> list[str]:
        if self.view not in REQUIRED_POINTS:
            raise ValueError(f"unknown view {self.view!r}")
        miss = self.missing_points()
        if miss:
            raise ValueError(f"missing required calibration points for {self.view} view: {miss}")
        warns = []
        w, h = self.image_size
        for n, p in self.segment.points.items():
            if p is None:
                continue
            if not (-0.5 <= p[0] <= w - 0.5 and -0.5 <= p[1] <= h - 0.5):
                warns.append(f"point {n} {p} is outside the image")
        if self.view == "manual":
            pass                                 # nothing clicked: only the settings matter
        elif self.view == "side":
            d = np.linalg.norm(self.to_sq(self.point("rubber")) - self.to_sq(self.point("plate")))
            if d < 60:
                warns.append(f"rubber and plate are only {d:.0f} px apart; side-view accuracy will be poor")
            ang = math.degrees(math.atan2(abs(self.point("plate")[1] - self.point("rubber")[1]),
                                          abs(self.point("plate")[0] - self.point("rubber")[0]) * self.sar))
            if ang > 35:
                warns.append(f"rubber->plate line is {ang:.0f} deg from horizontal; is this really a side view?")
        else:
            d = np.linalg.norm(self.to_sq(self.point("release")) - self.to_sq(self.point("catch")))
            if d < 30:
                warns.append(f"release and catch points are only {d:.0f} px apart")
        # Impossible settings are listed, not raised (see settings_errors): no
        # speed is reported while they are present.
        errs = settings_errors(self.settings)
        for e in errs:
            warns.append(f"IMPOSSIBLE SETTING: {e}. No speeds will be reported until this is fixed.")
        if self.extension_is_assumed and not errs:
            warns.append(f"pitcher extension not given; assuming {self.extension_ft:.1f} +/- "
                         f"{self.settings.extension_sigma_ft:.1f} ft")
        elif not self.extension_is_assumed and not errs:
            e = float(self.settings.extension_ft)
            if e < 2.0 or e > 8.0:
                warns.append(f"pitcher extension {e:g} ft is unusual (typical 4-7 ft, youth 3-5 ft); "
                             f"check it is in feet")
        return warns

    # ---- side-view geometry ------------------------------------------------
    def axis_sq(self) -> tuple[np.ndarray, np.ndarray]:
        """(P, R) plate and rubber points in square px (side view)."""
        return self.to_sq(self.point("plate")), self.to_sq(self.point("rubber"))

    def px_per_ft_axis(self) -> float:
        """Average scale along the pitch axis (square px per ft), affine approximation."""
        if self.has("rubber") and self.has("plate"):
            P, R = self.axis_sq()
            return float(np.linalg.norm(R - P) / (self.rubber_center_y_ft - self.plate_center_y_ft))
        return float("nan")

    def lam_of_y(self, y_ft: float) -> float:
        """Normalized world position along the plate->rubber line (0 at plate click, 1 at rubber click)."""
        return (y_ft - self.plate_center_y_ft) / (self.rubber_center_y_ft - self.plate_center_y_ft)

    def y_of_lam(self, lam: float) -> float:
        return self.plate_center_y_ft + lam * (self.rubber_center_y_ft - self.plate_center_y_ft)

    @staticmethod
    def mu_of_lam(lam, rho: float = 1.0):
        """Image fraction along P->R for world fraction lam (1-D projective map).

        rho = Z_plate / Z_rubber (camera depth ratio); rho == 1 is affine.
        """
        lam = np.asarray(lam, dtype=float)
        return lam / (rho * (1.0 - lam) + lam)

    @staticmethod
    def lam_of_mu(mu, rho: float = 1.0):
        mu = np.asarray(mu, dtype=float)
        return rho * mu / (1.0 - mu + rho * mu)

    def ground_point(self, y_ft: float, rho: float = 1.0) -> np.ndarray:
        """Image point (frame px) of the ground point at world y on the pitch line (side view)."""
        P, R = self.axis_sq()
        mu = float(self.mu_of_lam(self.lam_of_y(y_ft), rho))
        return self.from_sq(P + mu * (R - P))

    def up_dir_sq(self, at=None) -> np.ndarray:
        """Unit vector (square px) pointing 'up' (3-D vertical) in the image.

        Uses the vertical_top/bottom clicks if present (assumed parallel
        everywhere), else straight up (camera assumed level).
        """
        if self.has("vertical_top") and self.has("vertical_bottom"):
            v = self.to_sq(self.point("vertical_top")) - self.to_sq(self.point("vertical_bottom"))
            n = np.linalg.norm(v)
            if n > 1e-6:
                return v / n
        return np.array([0.0, -1.0])

    def release_point(self) -> np.ndarray:
        """Clicked release point, or (side view) a default from the assumed extension/height."""
        if self.has("release"):
            return self.point("release")
        if self.view != "side":
            raise ValueError(f"{self.view} view needs a clicked release point")
        g = self.to_sq(self.ground_point(self.release_y_ft))
        up = self.up_dir_sq()
        return self.from_sq(g + up * self.release_height_ft * self.px_per_ft_axis())

    def catch_point(self) -> np.ndarray:
        if self.has("catch"):
            return self.point("catch")
        if self.view != "side":
            raise ValueError(f"{self.view} view needs a clicked catch point")
        g = self.to_sq(self.ground_point(self.catch_y_ft))
        up = self.up_dir_sq()
        return self.from_sq(g + up * self.settings.catch_height_ft * self.px_per_ft_axis())

    # ---- scale / ball size -------------------------------------------------
    def px_per_ft_at_release(self) -> Optional[float]:
        if self.view == "side":
            return self.px_per_ft_axis()
        if self.has("rubber_left") and self.has("rubber_right"):
            return float(np.linalg.norm(self.to_sq(self.point("rubber_left")) -
                                        self.to_sq(self.point("rubber_right"))) / C.RUBBER_WIDTH_FT)
        return None

    def px_per_ft_at_catch(self) -> Optional[float]:
        if self.view == "side":
            return self.px_per_ft_axis()
        if self.has("plate_left") and self.has("plate_right"):
            return float(np.linalg.norm(self.to_sq(self.point("plate_left")) -
                                        self.to_sq(self.point("plate_right"))) / C.PLATE_WIDTH_FT)
        return None

    def ball_diameter_ft(self) -> float:
        return C.SOFTBALL_DIAMETER_FT if self.settings.sport == "softball" else C.BALL_DIAMETER_FT

    # ---- detection corridor ------------------------------------------------
    def corridor_polygon(self) -> np.ndarray:
        """Region (frame px, Nx2) the ball can occupy between release and catch."""
        Rp = self.to_sq(self.release_point())
        Cp = self.to_sq(self.catch_point())
        e = Cp - Rp
        L = float(np.linalg.norm(e))
        if L < 1e-6:
            raise ValueError("release and catch points coincide")
        e /= L
        n = np.array([-e[1], e[0]])
        if self.view == "side":
            k = self.px_per_ft_axis()
            up = self.up_dir_sq()
            # along-axis extent: 1.5 ft behind release .. 3 ft past catch
            a0 = Rp - 1.5 * k * e
            a1 = Cp + 3.0 * k * e
            # vertical extent relative to the release->catch line
            hi_r, lo_r = 3.0 * k, 5.5 * k
            hi_c, lo_c = 3.0 * k, 3.0 * k
            poly = [a0 + up * hi_r, a1 + up * hi_c, a1 - up * lo_c, a0 - up * lo_r]
        else:
            sr = self.px_per_ft_at_release()
            sc = self.px_per_ft_at_catch()
            w_r = max(1.5 * sr if sr else 0.0, 0.06 * L, 10.0)
            w_c = max(3.0 * sc if sc else 0.0, 0.35 * L, 20.0)
            a0 = Rp - 0.08 * L * e
            a1 = Cp + 0.15 * L * e
            poly = [a0 + w_r * n, a1 + w_c * n, a1 - w_c * n, a0 - w_r * n]
        return np.array([self.from_sq(p) for p in poly])

    def progress(self, xy) -> np.ndarray:
        """Fraction along release->catch (0 at release, 1 at catch) by projection (square px)."""
        Rp = self.to_sq(self.release_point())
        Cp = self.to_sq(self.catch_point())
        e = Cp - Rp
        q = self.to_sq(np.atleast_2d(np.asarray(xy, dtype=float)))
        return (q - Rp) @ e / float(e @ e)

    def in_masks(self, x: float, y: float) -> bool:
        for (x0, y0, x1, y1) in self.segment.masks:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return True
        return False


def manual_calibration(settings: Settings, image_size=(720, 480), video: Optional[dict] = None) -> CalibrationFile:
    """A calibration for marking pitches by hand: the settings and one point-less 'manual' segment."""
    return CalibrationFile(settings=settings, segments=[Segment(view="manual")], video=dict(video or {}),
                           image_size=tuple(image_size))


def is_manual(cf: Optional[CalibrationFile]) -> bool:
    """True if every segment of `cf` is a 'manual' (marked by hand) segment."""
    return cf is not None and bool(cf.segments) and all(seg.view == "manual" for seg in cf.segments)
