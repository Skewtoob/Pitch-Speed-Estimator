"""All tunable vision parameters in one flat dataclass (`FinderOptions`).

Flat on purpose: the pipeline builds it with `FinderOptions(**dict)` from
CLI / config overrides.  Every threshold here is a starting point taken
from docs/design-notes/cv.md or from experiments on synthetic clips; none
has been tuned on real tapes yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional


@dataclass
class FinderOptions:
    # ---- output / debugging
    debug_dir: Optional[str] = None       # write overview.png, timeline.png, pitch_NNN.png, tracks.jsonl
    debug_max_pitch_images: int = 60

    # ---- stabilization
    stab_max_features: int = 300
    stab_ransac_thr: float = 1.0          # square px
    stab_min_inliers: int = 30
    border_px: int = 12                   # feature-free left/right border (frame px)
    bottom_rows: int = 12                 # feature-free bottom lines (VHS head switching)
    corridor_feature_margin: int = 12     # dilate the corridor for the feature mask (px)
    auto_stamp_mask: bool = True          # find a burnt-in date stamp automatically
    stamp_warmup: int = 120               # samples used to find the date stamp

    # ---- background / detection
    bg_offsets: tuple = (4, 6, 8, 10)     # same-parity sample offsets (fields); halved for frames
    bg_offsets_slow: tuple = (8, 12, 16, 20)   # behind view, release end (slow image motion)
    slow_progress: float = 0.35           # behind view: use slow offsets below this progress
    interp_cubic: bool = True
    preblur_x: float = 0.8                # residual pre-blur (px, sample grid)
    preblur_y: float = 0.4
    roi_margin: int = 6                   # extra px around the corridor bbox
    noise_block: tuple = (8, 32)          # (rows, cols) blocks of the local noise scale
    noise_floor: float = 0.5              # grey levels; minimum residual noise scale (quantization)
    z_weak: float = 3.5                   # stored candidates (track extension)
    z_strong: float = 5.0                 # seeds
    z_pass2: float = 2.5                  # pass-2 re-detection inside model gates
    max_cands: int = 30                   # per sample
    nms_size: tuple = (5, 11)             # (rows, cols) sample grid
    clutter_tau: float = 300.0            # samples
    clutter_ref: float = 0.02
    attached_area_min: float = 150.0      # frame px; or 25 d^2, whichever larger
    cue_radius_ft: float = 3.0

    # ---- tracking
    track_window: int = 240               # samples of candidate history used per tracking run
    track_every: int = 60                 # run the tracker every N detected samples
    finalize_lag: int = 24                # a track is final once it ended this many samples ago
    seed_max_angle_deg: float = 50.0      # seeds: velocity within this angle of +-axis (side view)
    gate_speed_factor: float = 1.5
    min_speed_mph: float = 30.0
    max_speed_mph: float = 105.0
    miss_limit: int = 4
    miss_limit_body: int = 10
    merge_dist_px: float = 3.0
    merge_gap: int = 12
    ransac_iters: int = 200
    pass2: bool = True
    pass2_back: int = 14                  # samples to search before the first point
    pass2_fwd: int = 8

    # ---- segmentation / rejection
    min_inliers: int = 6
    min_progress: float = 0.35           # side view: NO_PROGRESS below this progress span
    min_progress_behind: float = 0.05     # behind view: progress span ...
    min_duration_behind_s: float = 0.12   # ... and duration
    progress_back_limit: float = -0.08    # extension stops this far behind the release point
    max_off_axis_deg: float = 35.0
    off_axis_rel: float = 0.25            # behind: miss release by > this * |Pc - Prel|
    off_axis_catch: float = 0.30
    side_speed_range: tuple = (25.0, 110.0)
    max_rms_px: float = 2.0
    duplicate_s: float = 1.5
    min_confidence: float = 0.1           # below: rejected as LOW_CONFIDENCE (still reported)
    contact_gap: int = 3

    # ---- events
    # Behind view 'first_free_field' release: latency (in samples) between the
    # true release and the first clean detection.  Prior measured on the
    # synthetic clips (see finder module docstring); revisit on real footage.
    release_latency_mean: float = 1.0
    release_latency_sigma: float = 1.0
    catch_near_progress: float = 0.85     # track reaching this progress ends at the glove

    # ---- cues
    cue_min_gap_s: float = 0.3
    cue_max_gap_s: float = 1.3
    cue_z: float = 4.0

    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "FinderOptions":
        """Build from a dict, ignoring unknown keys (they go to `extra`)."""
        d = dict(d or {})
        names = {f.name for f in fields(cls)}
        known = {k: v for k, v in d.items() if k in names}
        unknown = {k: v for k, v in d.items() if k not in names}
        o = cls(**known)
        o.extra.update(unknown)
        return o
