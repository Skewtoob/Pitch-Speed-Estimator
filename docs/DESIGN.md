# pitchspeed — design

Goal: open a (typically early-2000s camcorder, low-quality, interlaced SD)
baseball video, find every pitch, estimate each pitch's speed with an
honest uncertainty, and make it obvious which pitch each number belongs to
(table with pitch number, timestamp, frame range; annotated output video).

Background studies (with sources and simulations) live in
`docs/design-notes/`:

| note | what it covers |
|---|---|
| `research.md` | prior art, physics constants, league distances, camcorder/tape facts, sources |
| `cv.md` | ball detection, stabilization, tracking, pitch segmentation |
| `math.md` | speed estimators E1/E2, error budget, reporting rules |
| `io_ux.md` | decoding, interlace + field order, output video, UIs, packaging |
| `synth.md` | synthetic test-video generator |

Where this file and a note disagree, **this file wins**.

## 1. Key decisions

1. **Decode with PyAV, never with OpenCV.**  OpenCV 5.0 wheels return
   silently wrong frames for interlaced-flagged video (verified in the
   sandbox on DV/MPEG-2/H.264/Xvid test files).  OpenCV is used only for
   image processing and the GUI.
2. **Work in fields.**  Interlaced video is split into its two fields
   (59.94 samples/s for NTSC), detected from content (not metadata),
   with a per-pitch consistency check.  Progressive/deinterlaced input runs
   in frame mode and is flagged `frames_only`.
3. **Calibration by clicking** (once per camera setup, saved to JSON):
   * side view: middle of the rubber + middle of home plate (required);
     release / catch / a vertical reference (optional).
   * behind view: ball at release + glove at catch (required); rubber ends
     and plate front corners (optional, give scale for ball size).
4. **Speed estimators** (details in `math.md`):
   * **E1 time of flight** (all views):
     `L = (D - ext) + c`, `v0 = (exp(kL) - 1) / (k T)`, then the slope
     correction.  Release/catch times from the track with sub-field
     refinement where possible.  Behind-view release cannot be extrapolated:
     use the first field where the ball is clearly free of the hand, with a
     latency prior, and let the user confirm in review.
   * **E2 side-view fit**: plumb-project ball positions onto the
     plate->rubber line, 1-D projective map with `rho` fitted from the
     ball's own motion, drag model `Y(t) = y_r - ln(1 + k v0 (t - t_r))/k`,
     robust least squares; full uncertainty formula (covariance + click
     perturbation + k + lateral term + model term).  Reject E2 for very
     oblique views.
   * **No E3** (behind-view perspective / ball-size methods are
     degenerate; `math.md` §4).
   * Combine by inverse variance on the independent parts, add the shared
     k term; flag `E1_E2_disagree`.
5. **Report** release speed (radar-gun "out of hand" equivalent) as the
   main number, plus average and plate speed, 1-sigma and a 90% interval.
   Labels: good (sigma <= 2.5%), fair (<= 5%), poor (<= 10%, shown with
   "~"), none (> 10% or unusable -> no number, give the reason).
   Always list assumptions (extension, catch point, Cd, air density) and
   flags.
6. **Never silently drop anything.** Rejected tracks keep a reason code;
   "possible missed pitch" windows are listed for review.
7. **Manual fallback** is first class: the review UI lets the user mark
   release and catch fields for any pitch (auto-detection will fail on
   some real tapes).

## 2. Conventions (all modules)

* Units: feet, seconds, ft/s internally; mph only for reporting.
* World frame (`constants.py`): origin at plate apex, +y to pitcher, +z up.
  League distance D is rubber front edge -> plate apex.  Rubber click =
  rubber middle (y = D + 0.25 ft); plate click = plate middle
  (y = 17/24 ft).
* Image coordinates: frame pixel coordinates of stored pixels (see
  `types.py` docstring).  Field row r of parity p is frame row 2r + p.
  Square-pixel coords are (x * sar, y).
* Stabilized coordinates: candidates and track points are mapped into the
  coordinate system of the calibration reference frame, using per-sample
  transforms (sample frame coords -> reference frame coords).  The finder
  uses similarities or homographies; `transforms.npz` stores both the exact
  3x3 matrices (`transforms_3x3`) and a 2x3 local-affine approximation.
  `pipeline.load_transforms` returns the 3x3 set when present and every
  consumer (overlay, review) accepts (n,3,3) or (n,2,3).
* Time: seconds from the first video frame; field time =
  frame time + k * frame_dt / 2.

## 3. Package layout and module contracts

```
src/pitchspeed/
  constants.py physics.py types.py         foundation (lead)          DO NOT EDIT unless lead
  calib/model.py                           foundation (lead)          DO NOT EDIT unless lead
  io/reader.py io/interlace.py io/writer.py io/display.py            agent IO
  vision/stabilize.py vision/detect.py vision/track.py vision/finder.py   agent VISION
  estimate/speed.py (+ helpers)                                       agent ESTIMATE
  render/overlay.py render/report.py                                  agent RENDER
  ui/keys.py ui/canvas.py calib/ui.py review/state.py review/ui.py   agent UI
  synth/*.py                                                          agent SYNTH
  pipeline.py cli.py                                                  lead (integration)
tests/test_<area>_*.py                                                each agent its own
```

Shared files are owned by the lead.  If you need a change in a foundation
file, do not edit it: work around it locally and list the request in your
final report.  You may add new files inside your own area.

### 3.1 io (agent IO)

```python
# io/reader.py
def probe(path: str) -> VideoInfo                     # fast metadata
class VideoReader:                                     # context manager
    def __init__(self, path: str, sar_override: float | None = None): ...
    info: VideoInfo                                    # exact n_frames, timing_mode, dropped frames
    frame_times: np.ndarray                            # (n_frames,) seconds, t[0] == 0
    def iter_frames(self, start=0, stop=None, fmt="gray") -> Iterator[tuple[int, np.ndarray]]
    def get_frame(self, i: int, fmt="gray") -> np.ndarray     # frame-exact random access
    def close(self) -> None
# io/interlace.py
def detect_scan(reader: VideoReader, override: str | None = None, max_frames: int = 900) -> ScanInfo
class SampleSource:
    def __init__(self, reader: VideoReader, scan: ScanInfo): ...
    n_samples: int; sample_dt: float; sample_shape: tuple[int, int]   # (rows, cols) of a sample image
    def sample_time(self, s: int) -> float
    def locate(self, s: int) -> tuple[int, int, int]                   # (frame, k, parity)
    def sample_of(self, frame: int, k: int = 0) -> int
    def iter_samples(self, s0=0, s1=None, color=False) -> Iterator[Sample]
    def get_sample(self, s: int, color=False) -> Sample
def residual_alternation(t, x, y) -> float   # lag-1 autocorrelation helper for the field-order check
# io/display.py
def sample_to_display(sample: Sample, sar: float, out_size=(960, 720)) -> tuple[np.ndarray, np.ndarray]
    # parity-aware bob of a field to full height + square-pixel resize; returns (bgr, A)
    # with A the 2x3 affine mapping frame pixel coords -> display coords.
# io/writer.py
class VideoWriter:
    def __init__(self, path, width, height, fps="60000/1001", audio_src=None,
                 audio_start_s=0.0, audio_duration_s=None, crf=18): ...
    def write(self, bgr: np.ndarray) -> None
    def close(self) -> str                             # backend actually used
```

### 3.2 vision (agent VISION)

```python
@dataclass
class FinderOptions: ...                               # thresholds, debug_dir, etc.
@dataclass
class FinderResult:
    tracks: list[PitchTrack]                           # status == 'pitch', with release/catch EventTimes
    rejected: list[PitchTrack]                         # status == 'rejected', reject_code set
    possible_missed: list[dict]                        # {s_start, s_end, t_start, t_end, reason}
    transforms: np.ndarray                             # (n, 2, 3) local affine; exact 3x3 in diagnostics['transforms_3x3']
    transform_quality: np.ndarray                      # (n,) 0..1
    sample_index: np.ndarray                           # (n,) the sample index s of each row above
    diagnostics: dict
class PitchFinder:
    def __init__(self, calib: Calibration, scan: ScanInfo, sample_dt: float,
                 sample_shape: tuple[int, int], reference_frame: np.ndarray,  # gray full frame (calib.ref_frame)
                 opts: FinderOptions | None = None): ...
    def process(self, sample: Sample) -> None          # streaming, time order
    def finish(self) -> FinderResult
def track_between(source: SampleSource, calib: Calibration, scan: ScanInfo, s_release: int, s_catch: int,
                  reference_frame: np.ndarray, opts: FinderOptions | None = None,
                  manual_points: list[TrackPoint] | None = None) -> PitchTrack
    # constrained re-tracking for review (manual release/catch) - returns a track, maybe with few points
```

Event-time semantics on `PitchTrack`:
* `release.t` = best estimate of the instant the ball left the hand.
  `release.method` in {'first_free_field', 'extrapolated', 'manual'}; for
  'first_free_field' the time is the field time of the first clean
  detection minus a latency prior (see `math.md` §2), sigma accordingly.
* `catch.t` = instant the ball reached the glove (or contact for
  `end_kind == 'contact'`), method in {'extrapolated', 'last_moving_field',
  'manual'}.
* `diagnostics` keeps raw observations: `t_first_obs`, `t_last_obs`,
  `s_first_obs`, `s_last_obs`, attached flags, the fitted model.

### 3.3 estimate (agent ESTIMATE)

```python
def estimate_speed(calib: Calibration, scan: ScanInfo, sample_dt: float,
                   release: EventTime | None, catch: EventTime | None,
                   track: PitchTrack | None = None, manual: bool = False) -> SpeedEstimate
```
* Always tries E1 when release and catch exist; E2 when `calib.view ==
  'side'` and the track has enough inlier points; combines.
* Fills `SpeedEstimate` completely (flags, label, assumptions, reason).
* Also exposes the pieces (`tof_estimate`, `side_fit_estimate`, `combine`)
  for tests and for the review UI.

### 3.4 render (agent RENDER)

```python
# render/report.py
def write_json(result: AnalysisResult, path: str) -> None     # atomic write
def load_json(path: str) -> AnalysisResult
def write_csv(result: AnalysisResult, path: str) -> None
def format_table(result: AnalysisResult) -> str               # console summary
def format_time(t: float) -> str                              # hh:mm:ss.sss
# render/overlay.py
def render_annotated(result: AnalysisResult, source: SampleSource, out_path: str,
                     mode: str = "full",                      # 'full' | 'highlights'
                     transforms: np.ndarray | None = None, transform_samples: np.ndarray | None = None,
                     size=(960, 720), audio: bool = True, progress=None) -> str
```

### 3.5 UI (agent UI)

```python
# calib/ui.py
def run_calibration_ui(source: SampleSource, video: VideoInfo, view: str,
                       initial: CalibrationFile | None = None,
                       settings: Settings | None = None, start_frame: int = 0) -> CalibrationFile | None
# review/ui.py
def run_review_ui(result: AnalysisResult, source: SampleSource, calib_file: CalibrationFile,
                  recompute: Callable[[Pitch], Pitch], save: Callable[[AnalysisResult], None],
                  transforms=None, transform_samples=None) -> AnalysisResult
```
Pure state machines (`review/state.py`, calibration state inside
`calib/ui.py` or `calib/state.py`) are unit-tested; the cv2 shells are thin.

### 3.6 synth (agent SYNTH)

Port `docs/design-notes/synth.md`'s prototype into `pitchspeed/synth/`.
Writes `<clip>` + `<clip>.gt.json` + `<clip>.calib.json` (a real
`CalibrationFile` with exact or noise-perturbed clicks).  Also
`synth/evaluate.py: evaluate(result: AnalysisResult, gt: dict) -> dict`
(match pitches by release time, report recall/precision, speed errors,
interval coverage) and `synth/cli.py: main(argv)` used by `pitchspeed synth`.

### 3.7 pipeline + CLI (lead)

`pitchspeed probe | calibrate | analyze | review | render | run | synth | evaluate | keys`.
Output folder `<video stem>_pitchspeed/`: `calibration.json`,
`results.json`, `results.csv`, `annotated.mp4`, `transforms.npz`, `log.txt`.

## 4. Dev environment

* venv: `/home/user/venv-ps` (system site-packages + `pip install -e .`;
  has opencv-python with Qt highgui, PyAV, SciPy, imageio-ffmpeg).
  Run tests with `/home/user/venv-ps/bin/python -m pytest`.
* GUI tests: `xvfb-run -a /home/user/venv-ps/bin/python -m pytest -m gui`.
  Never open a cv2 window without a display: Qt aborts the process.
* Prototype synthetic clips with ground truth (prototype format) exist in
  the session scratchpad `gen/` directory (`ll_side_bff.mkv`,
  `ll_behind_tff.mkv`, ... + `.gt.json`).


## 5. Decisions made during integration, evaluation and review

These came out of end-to-end runs on held-out synthetic suites and an
adversarial code review (findings verified independently before fixing).

**Field order / scan.**  Content tests can be unsure on real footage.  After
tracking, `pipeline.resolve_field_order` re-times every well-tracked ball
under three hypotheses (detected order, flipped order, progressive) and
refits a smooth curve; a decisive winner confirms the scan (confidence
'high'), or triggers one automatic re-run with the corrected order /
progressive mode.  Tracks seen in only one field parity cannot vote (the
hypotheses differ by a constant shift); the finder's pair check covers the
"one track per parity" symptom of a wrong order.  Above 40 frames/s video
is never split into fields.

**What becomes a numbered pitch.**  A track becomes a pitch only if its
confidence is >= `DOUBTFUL_CONFIDENCE` (0.3); weaker tracks, likely catcher
throw-backs (no speed, starts <= 4 s after the previous catch, stays near
the pitcher's end of the path) and pre-pitch junk (no speed, <= 1.5 s before
a stronger pitch) go to the review list ("possible missed pitch") with the
reason.  Duplicates are only same-flight fragments; the fragment that saw
most of the flight wins and any loser outside the winner's flight is listed
for review.  Nothing is dropped silently.

**Honest event times (behind view).**  The release is the first clean
free-flight detection minus a latency prior.  It is widened (never narrowed)
when: earlier faint detections run into it; the delivery-motion cue comes
clearly earlier; early detections are separated from the continuous flight
by a gap (`release_gap`: the early group may be the ball still in the hand);
or the data starts too close before it.  A hidden catch is the crossing of
the fitted trajectory with the clicked glove point, with a sigma calibrated
on ground truth, and "not observed" when more than ~10 fields are hidden.

**Manual marks.**  R = first field with the ball out of the hand, C = first
field with the ball in the glove.  The event is placed half a field earlier
with quantization sigma dt/sqrt(12), plus an assumed human-judgement term
(one-field slip with p = 0.25), about 0.58 field in total.  A manual C
ends the flight at the glove (never an automatic 'contact'); a manual B
(bat contact, `EventTime.point == 'contact'`) ends it at the assumed contact
point (front of the plate + 0.5 ft, +/-1 ft).  Stale
flags of the automatic event are removed after a manual mark.

**Capture glitches.**  Duplicated, dropped or undecodable frames inside a
pitch's frame range set `duplicate_frame_suspected` / `dropped_frame_suspected`
on that pitch, which widens its timing uncertainty.

**Detection noise normalization.**  Each matched kernel's gain is measured
only on pixels where the residual is non-zero (static blocks of compressed
or lossless video are exactly zero and would otherwise collapse the
estimate), and clipped to [1, 8] x its white-noise gain.

**User data safety.**  Re-running `run` on a video with results asks first
(default: keep them and go to review).  `analyze` over results with review
work writes `results.backup-<timestamp>.json` first and carries manual /
edited / rejected pitches over when the sample timeline is unchanged.
results.json is always written before results.csv; a CSV locked by Excel
is a warning, never a lost result.  The calibration's video fingerprint is
checked; files with the same stem but different extensions get separate
output folders.
