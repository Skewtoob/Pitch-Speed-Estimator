"""Ball detection, stabilization, tracking and pitch segmentation.

Public API (docs/DESIGN.md section 3.2): `FinderOptions`, `FinderResult`,
`PitchFinder`, `track_between` (from `pitchspeed.vision.finder`), plus
`eval_model` to evaluate a `PitchTrack.model` dict.
"""
from pitchspeed.vision.finder import FinderOptions, FinderResult, PitchFinder, track_between
from pitchspeed.vision.model import eval_model

__all__ = ["FinderOptions", "FinderResult", "PitchFinder", "track_between", "eval_model"]
