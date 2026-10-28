"""Text formatting shared by the console table, the CSV and the video overlay.

One place decides how a speed is written, so the table, the CSV and the
video always agree:

* main number = release speed ("out of the hand", radar-gun equivalent),
  whole mph on screen, one decimal in the CSV;
* 1-sigma shown to the nearest 0.5 mph (docs/design-notes/math.md section 5);
* label 'poor'  -> "~78?" (rough number, shown but marked);
* label 'none'  -> "n/a" (no number at all; the reason is given instead).

Everything that ends up in the video is plain ASCII because OpenCV's
Hershey fonts cannot draw anything else (see `ascii_text`).
"""
from __future__ import annotations

import math
from typing import Optional

from pitchspeed.types import Pitch, SpeedEstimate

NO_TIME = "--:--:--.---"

METHOD_NAMES = {
    "tof": "TOF",
    "side_fit": "side fit",
    "tof+side_fit": "TOF+side fit",
    "manual_tof": "manual TOF",
    "none": "none",
}

_UNICODE_TO_ASCII = {
    "±": "+/-", "–": "-", "—": "-", "−": "-", "≈": "~",
    "°": " deg", "·": "|", "…": "...", "‘": "'", "’": "'",
    "“": '"', "”": '"', "σ": "sigma", "ρ": "rho", "φ": "phi",
    "→": "->", "≤": "<=", "≥": ">=", "×": "x",
}


def finite(x) -> bool:
    """True for a real, finite number (None / NaN / inf -> False)."""
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def round_half_up(x: float) -> int:
    """Conventional rounding (78.5 -> 79), unlike Python's round-half-even."""
    return int(math.floor(float(x) + 0.5))


def format_time(t: Optional[float]) -> str:
    """Seconds -> 'hh:mm:ss.sss' (milliseconds; no drop-frame ambiguity).

    Rounds to the nearest millisecond with carry (59.9996 s -> 00:01:00.000).
    None / NaN / inf -> '--:--:--.---'.  Negative times get a leading '-'.
    """
    if not finite(t):
        return NO_TIME
    t = float(t)
    sign = "-" if t < 0 else ""
    ms = round_half_up(abs(t) * 1000.0)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{sign}{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def method_name(method: Optional[str]) -> str:
    return METHOD_NAMES.get(method or "none", method or "none")


def has_number(sp: Optional[SpeedEstimate]) -> bool:
    """A speed is shown only when the estimator produced one and the label allows it."""
    return sp is not None and sp.label != "none" and finite(sp.v_release_mph)


def speed_short(sp: Optional[SpeedEstimate]) -> str:
    """'78', '~78?' (poor) or 'n/a' (not measurable) - whole mph."""
    if not has_number(sp):
        return "n/a"
    v = round_half_up(sp.v_release_mph)
    return f"~{v}?" if sp.label == "poor" else f"{v}"


def sigma_text(sigma: Optional[float]) -> str:
    """1-sigma to the nearest 0.5 mph ('2.5', '3'); never shows 0."""
    if not finite(sigma):
        return "-"
    half = max(1, round_half_up(2.0 * float(sigma)))
    return f"{half // 2}" if half % 2 == 0 else f"{half / 2:.1f}"


def ci90_text(sp: Optional[SpeedEstimate]) -> str:
    """'74-82' (whole mph) or '-'."""
    if not has_number(sp) or sp.ci90_mph is None or len(sp.ci90_mph) != 2:
        return "-"
    lo, hi = sp.ci90_mph
    if not (finite(lo) and finite(hi)):
        return "-"
    return f"{round_half_up(lo)}-{round_half_up(hi)}"


def whole_mph(v: Optional[float]) -> str:
    return f"{round_half_up(v)}" if finite(v) else "-"


def pitch_flags(p: Pitch) -> list[str]:
    """Union of the pitch's own flags and its speed flags, order kept, no duplicates."""
    out: list[str] = []
    for f in list(p.flags or []) + list((p.speed.flags if p.speed is not None else None) or []):
        if f and f not in out:
            out.append(str(f))
    return out


def speed_reason(sp: Optional[SpeedEstimate]) -> str:
    if sp is None:
        return "no speed estimate"
    return sp.reason or ""


def ascii_text(s: str) -> str:
    """Make `s` drawable with cv2.putText (ASCII only): map a few common
    symbols, replace anything else with '?'."""
    s = "".join(_UNICODE_TO_ASCII.get(ch, ch) for ch in str(s))
    return s.encode("ascii", "replace").decode("ascii")


def shorten(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: max(0, n - 3)].rstrip() + "..."
