# Accuracy on synthetic test videos

**Read this first.** All numbers below come from *computer-generated*
videos with exactly known pitch speeds. No real tape has been tested yet.
Real footage brings things the simulator does not model, so expect worse
results on real tapes until the tool has been checked and tuned on them.

## Test videos

The generator (`pitchspeed synth`) renders each clip field by field at
59.94 fields/s, using a pinhole camera and full 3-D ball physics: drag,
gravity, spin-like force, wind, and random extension and release point. It
then degrades the video like an early-2000s camcorder tape:

- 720×480 interlaced, top- or bottom-field-first, sometimes with the wrong
  field order in the file's metadata
- shutter from 1/60 to 1/1000 s (motion-blur streaks)
- sensor and tape noise, chroma blur, dropouts, a burned-in date stamp
- handheld shake and pans
- encoding as DV, MPEG-2, H.264 or lossless; some clips deinterlaced
  (blend, field-discard, bob) or progressive 30p/60p

**Behind-the-plate views** are shot through a blurred backstop fence, and
the umpire and catcher often hide the end of the flight.

**Each clip** has three pitches (40–92 mph, at 46 ft youth or 60.5 ft
distance) plus distractors: the catcher's throw back to the pitcher, a
batter swinging, birds, and people walking in.

**Calibration clicks** have about 1 px of random click error added.

The final numbers come from a suite that was generated after all tuning and
never looked at during development:

- 30 clips, 90 pitches, 45 side view and 45 behind the plate
- seed 271828
- includes deinterlaced and progressive variants

## Results (final held-out suite)

| | Side view | Behind the plate |
|---|---|---|
| Pitches found | 45 / 45 | 44 / 45 |
| False "pitches" (all shown without a speed) | 1 | 2 |
| Found pitches that got a speed | 45 / 45 | 28 / 44 |
| Release-speed error, RMS | **0.66 mph (1.0 %)** | **3.3 mph (6.0 %)** |
| Largest error | 1.5 mph (2.2 %) | 8.0 mph (14.8 %) |
| True speed inside the reported 90 % range | 45 / 45 | 27 / 28 |
| Labels | 36 good, 9 fair | 28 poor (shown as `~NN?`), 16 n/a |

"Pitches found" counts a detection within 0.25 s of the true release.
Every reported speed was also checked against whichever true pitch its
flight overlaps, so a mis-timed detection still counts as a speed error.
No reported speed was more than 3 standard deviations off.

## What this means

**Side view works well** in simulation. The speed comes from the ball's
measured motion across the frame, using the rubber-to-plate distance as the
ruler. It hardly depends on assumptions. The ± ranges are a little
conservative: the true speed was always inside the 90 % range.

**Behind the plate is much harder**, and the tool says so.

- From behind, the ball moves almost straight toward the camera, so only
  the flight time from release to catch can be measured.
- That needs two assumptions:
  - how far in front of the rubber the pitcher releases the ball (the
    "extension", ±1 ft);
  - where the glove is.
- The catcher and umpire often hide the catch.
- About a third of the pitches therefore get "not measurable" with a
  reason, rather than a guess.
- The rest are labelled "poor" (±5–10 %).

Entering the pitcher's real extension (`--extension`) and marking release
and catch by hand in review make behind-view numbers better.

## What is *not* covered by these tests

- Real camcorder optics and real tape damage: lens distortion, autofocus
  hunting, auto-exposure pumping, rolling shutter, heavy dropouts.
- Real people, with realistic motion and clothing (white uniforms against a
  white ball), and real backgrounds.
- Shadows, bounced pitches, batted balls.
- Running on Windows: only the code paths were written for it; it has not
  been run on a Windows PC.
- Comparison against a radar gun.

## How the numbers improved during development

The same kind of synthetic suite at the first end-to-end run, compared with
the final held-out suite:

| | Behind: found | Behind: false pitches | Side: found |
|---|---|---|---|
| First end-to-end run (24 clips) | 50 % | 8 | 92 % |
| Final held-out suite (30 clips) | 98 % | 2 (none with a speed) | 100 % |

The fixes along the way:

- a date-stamp auto-mask that blanked tripod shots;
- interlace misdetection;
- slow-shutter streaks;
- the catcher hiding the catch;
- a noise-normalization bug on static, compressed backgrounds;
- wind-up detections making a release look early;
- throw-backs counted as pitches;
- duplicate handling that could drop a real pitch.

The bugs listed here, and several more found by an independent code review,
have regression tests in `tests/`.

## Reproduce

```
pitchspeed synth suite --n 30 --seed 271828 --variants --out-dir suite/
# then, for every clip: pitchspeed analyze CLIP --calib CLIP.calib.json --render none
pitchspeed synth evaluate suite/
```
