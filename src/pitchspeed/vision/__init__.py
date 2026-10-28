"""Ball detection, stabilization, tracking and pitch segmentation.

Public API (docs/DESIGN.md section 3.2), from `pitchspeed.vision.finder`:
`FinderOptions`, `FinderResult`, `PitchFinder`, `track_between`; plus
`eval_model(PitchTrack.model, t)` to evaluate a track's fitted curve.

Modules: geometry (coordinate maps, corridor, scales), stabilize (motion
compensation to the calibration frame), detect (background model and
candidates), model (trajectory fits), track (seed / extend / pass 2 /
merge / field-order check), segment (reject codes, event times,
confidence), finder (streaming orchestration, duplicates, possible missed
pitches), debug (--debug images and tracks.jsonl), options.

Where this implementation departs from docs/design-notes/cv.md, and why
(measured on synthetic clips only):

* Stabilization registers every sample directly to the same-parity field
  of the reference frame (no keyframe chain unless that fails), so there is
  no drift; the reference system is the reference frame's mean exposure
  time (half way between its fields) so both parities agree.
* Transforms are 3x3: a similarity, or a homography when it explains
  clearly more features.  A pan of a few degrees with a wide lens bends the
  motion field by ~10 px at the image edges and broke similarity-only
  registration for the rest of a clip.  `FinderResult.transforms` keeps the
  (n, 2, 3) contract as the local affine approximation at the corridor
  centre; the exact 3x3 transforms are in diagnostics['transforms_3x3'] and
  are what positions are mapped with.
* Burnt-in date stamps are found automatically (features with zero raw
  motion while the camera moves) and masked for registration and detection.
* Detection: noise floor 0.5 grey levels (noise-free / quantized input);
  behind-view kernels are restricted to where their size matches the
  expected ball size (a large kernel's tail otherwise makes ghosts next to
  a small ball); clutter is counted from non-attached peaks with a 300
  sample time constant (counting all z >= 3 pixels suppressed the release
  area, where the pitcher moves every pitch); 'attached' uses the
  ROI-global noise scale.
* Tracking enforces the contrast sign within a track (slow behind-view
  balls leave opposite-sign ghosts in the background median), and pass 2
  accepts only local maxima of the z map.
* Event times and their semantics are documented in `segment`; catch times
  are uniform over [last detection, catch-point crossing] when the track
  ends early, and uninformative ('catch_not_observed') when a behind-view
  track would need a long extrapolation (degenerate, math.md 4b).
* Extra reject codes NO_EVENTS and LOW_CONFIDENCE (still reported).
* Field order: besides the per-track check, two overlapping single-parity
  tracks are tested jointly (with the wrong order the tracker builds one
  track per parity).
"""
from pitchspeed.vision.finder import FinderOptions, FinderResult, PitchFinder, track_between
from pitchspeed.vision.model import eval_model

__all__ = ["FinderOptions", "FinderResult", "PitchFinder", "track_between", "eval_model"]
