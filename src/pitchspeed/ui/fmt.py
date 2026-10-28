"""Small text formatters shared by the UI state machines (no OpenCV import)."""
from __future__ import annotations

import math
from typing import Optional


def format_time(t: Optional[float]) -> str:
    """Seconds -> 'hh:mm:ss.sss' (milliseconds avoid drop-frame ;ff ambiguity)."""
    if t is None or not math.isfinite(t):
        return "--:--:--.---"
    ms = int(round(t * 1000.0))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
