# Synthetic Test-Video Generator: Spec v0.1 (prototyped and checked)

The prototype generator and its tests were throwaway scripts; the generator that shipped is `src/pitchspeed/synth/`.

## 1. Conventions (shared by the generator and the analyzer)

- **Units:** feet, seconds, ft/s, with mph only in reports (1 mph = 22/15 ft/s).
- **World frame:** right-handed. The origin is the back point (apex) of home plate on the plate surface. +y points to the pitcher, +x to the 1B side (the catcher's right), +z up.
- **Landmarks** (R = rubber distance, h = mound height; MLB/HS: R = 60.5, h = 10 in; LL Majors: R = 46, h = 6 in; both configurable):
  - `plate_back_point` (0, 0, 0)
  - `plate_side_L/R` (∓8.5/12, 8.5/12, 0)
  - `plate_front_L/R` (∓8.5/12, 17/12, 0)
  - `plate_front_center` (0, 17/12, 0)
  - `rubber_front_center` (0, R, h)
  - `rubber_front_L/R` (∓1, R, h)
  - L means the 3B side (−x).
- **Camera:** OpenCV pinhole. X_c = R(X − C). u = fx·x/z + cx, v = fy·y/z + cy, with an optional k1 distortion term.
  - Non-square pixels: **fy = fx·PAR**, with PAR 10/11 by default.
  - cx = (W−1)/2 and cy = (H−1)/2.
- **Pixels:** output-frame coordinates, origin top-left, integer values at pixel centres. In field parity p, field row r is frame row **v = 2r + p**.
- **Time:**
  - Global field index k = 2n + i, where i ∈ {0, 1} is temporal order inside frame n.
  - **t_k = k·1001/60000 s**, the exposure midpoint. t = 0 is the first field of frame 0.
  - Parity: p = i for TFF and p = 1−i for BFF.
  - 30p: t = 2n·1001/60000. 60p: t = n·1001/60000.
  - Container timestamps are never used.

## 2. Interface

```
gt = generate_clip(config: dict, out_dir) -> dict      # deep-merged over defaults; fully resolved config echoed in GT
python -m pitchsynth generate --config c.json --seed 3 --set camera.distance_ft=90 --out-dir out/
python -m pitchsynth suite --n 40 --seed 1 --out-dir suite/   # stratified sampler over the ranges below + index.csv
python -m pitchsynth check out/clip.gt.json                    # decode, verify frame count/field-order barcode
```

Each run writes:
- `<out>` (the video)
- `<out>.gt.json`
- `<out>.cams.npz`: per-field k, t, R(t), zoom
- optional `extra_outputs`: the **same render** fanned out to several ffmpeg encoders, e.g. an FFV1 lossless master plus MPEG-2 and DV.

## 3. Config schema (defaults; suite ranges in brackets)

```
seed:int  duration_s:60  level:"mlb"|"ll"|{rubber_ft,mound_h_ft,base_ft}
video:{width:720,height:480,par:[10,11],scan:"interlaced"|"progressive30"|"progressive60",
       field_order:"tff"|"bff", metadata_field_order:"auto"|"tff"|"bff"   # can lie on purpose
       postprocess:null|"blend"|"field_discard"|"yadif"|"yadif_bob",
       codec:"mpeg2"|"h264"|"mpeg4"|"dv"|"ffv1", bitrate_kbps:[800,6000], out, extra_outputs:[…], debug_stamp:false}
camera:{view:"side"|"behind"|"custom", side:"1b"|"3b", distance_ft:[40,150], angle_deg:[60,90],
        height_ft:[3,15], back_ft:[10,40], lateral_ft:[-6,6], hfov_deg:null(auto-fit)|[5,60],
        fit_margin:0.12, roll_deg, k1:[−0.1,0], C/look_at (custom)}
shake:{amp_deg:[0,0.5], roll_amp_deg, fmin_hz:0.3, fmax_hz:[2,6], n_sines:6,
       pans:[{t,dur,yaw_deg,pitch_deg}] (≤±20°), zooms:[{t,dur,factor}]}
pitches:{count | schedule:[{t_release,…overrides}], first_t, gap_s:[5,20], speed_mph:[40,95],
         cd:[0.30,0.40], extension_ft:[3,7.5], release_z_ft:[4,6.5], release_x_ft:[-2.5,2.5],
         target_x_ft:[-1,1], target_z_ft:[1.3,3.8], magnus_x/z (ft/s²):[-15,15]/[-8,16],
         catch_behind_ft:[0.5,2.5], wind_mph:[-10,10]}
render:{shutter_s:[1/60,1/1000], psf_sigma_px:[0.5,1.0], ball_value:[150,245], segments:6}
noise:{sigma0:[1,6], beta, chroma_sigma}   tape:{profile:"none"|"dv"|"hi8"|"vhs", …overrides}
agc:{amp, tau}  date_stamp:{enabled,text_date,start_hms,pos}  calibration:{clear_field_s:2}
distractors:{pitcher,catcher,umpire,batter,swing_prob,throwback,birds,pickoffs}
capture:{drop_frames:[],dup_frames:[],start_offset_frames}
```

## 4. Physics and events (independent of the analyzer's code)

- **Equation of motion:** a = −k|v−w|(v−w) + a_M + (0, 0, −32.174).
  - k = ρC_dA/(2m), with ρ = 0.0749 lb/ft³, r = 1.45 in, m = 5.125 oz. This gives k ≈ 0.00184/ft at C_d = 0.34.
  - a_M is a constant Magnus-like acceleration; w is wind.
- **Integration:** RK4 at 0.25 ms with a cubic-Hermite dense output, so position and velocity are exact at any t.
- **Aim:** a 2-D Newton shooting step sets the launch direction so the ball crosses y = 17/12 at the target (x, z).
- **t_release:** continuous, including a random sub-field phase.
- **t_catch:** the first crossing of y = −catch_behind, found by bisection on the interpolant.
- **Ball in hand:** for the last 60 ms the ball rides the hand on a 2 ft-radius arc whose tangential speed ramps from 0 to v0. Position and velocity match at release, so the hand streak overlaps the ball and **the first sighting of the ball is not the release**.
- **Glove:** sets 0.7 s early, gives 0.3 ft over 60 ms, then hides the ball.
- **Distractors:**
  - Throwback: gravity-solved, 40–55 mph.
  - Swing: bat sweeps 260° around t_plate − 0.1 s.
  - Birds: about 30 ft/s.
  - Pans and zooms.
  - All are logged in GT.
- **The generator's physics model is deliberately richer than the analyzer's** (3-D, gravity, Magnus, wind, C_d per pitch). This way the analyzer's model error gets measured instead of being assumed away.

## 5. Rendering algorithm

1. **Background canvas (once, about 4.7 s; cache by camera hash).**
   - Ray-cast the base camera at 2× supersampling, with a margin for shake, pans and zoom-out.
   - Ground is a height field (mound cap, rubber top at h) found by 4-step fixed-point ray intersection.
   - Procedural turf stripes, dirt, chalk lines, batter's boxes, plate and rubber; a fence cylinder, trees and sky.
   - Pre-blur σ = 0.5 output px.
2. **Per field k** (t_k, parity p): evaluate shake R_s(t_k) and zoom(t_k). Both are closed-form continuous functions (a sum of sines with a ~1/f spectrum, plus minimum-jerk pans and zooms).
   - Background: `warpPerspective(canvas, S_p·K(t)·R_s·K0⁻¹·C2O)` rendered **directly onto the 240-line field grid**, where S_p maps v to (v−p)/2. This is exact for rotation plus zoom.
3. **Actors:** 3-D capsules are projected to stadium polygons in field coordinates and drawn with `fillPoly(LINE_AA, shift=4)` on uint8. (OpenCV ignores anti-aliasing on float32 images; I checked.) They are depth-sorted into a far set, then the ball, then a near set.
4. **Ball:**
   - Split the exposure [t − S/2, t + S/2] into 6 sub-intervals, plus breaks at the release and catch events.
   - Treat each sub-interval as a linear image segment with ellipse semi-axes fx·r/Z and fy·r/Z.
   - At each fine sub-sample (n_h = max(4, ⌈5/r_px⌉) across, 2n_h down over the **2-row field aperture**), compute the exact fraction of time inside the swept ellipse with a quadratic solve. There is no temporal point sampling.
   - Apply the Gaussian PSF on the fine grid **before** box-averaging.
   - Composite in linear light (γ 2.2).
5. **Interleave:** `frame[p::2] = field`.
6. **Per-frame camera and tape chain.** Only pointwise or horizontal operations are allowed, so the two fields never mix.
   - Convert to YCbCr.
   - Luma noise with σ² = σ0² + βY, and streaky chroma noise.
   - Date/time stamp (in-camera, so it goes in before the tape stage; the seconds tick).
   - Luma low-pass plus sharpening overshoot; chroma low-pass plus delay.
   - Per-line timebase jitter (AR(1) within each field), a head-switching band and dropouts.
   - The RNG is keyed by (seed, subsystem, frame).
7. **Postprocess variants:**
   - blend = vertical [1, 2, 1]/4
   - field_discard + resize
   - ffmpeg `yadif=0` or `yadif=1`
8. **Encode** through an ffmpeg pipe:
   - Filter chain: `setfield, scale=interl=1, setsar=10/11, format`.
   - MPEG-2: `+ilme+ildct -top`.
   - x264: `-x264-params interlaced=1:tff=1|bff=1 -threads 1`.
   - DV: `dvvideo yuv411p`.
   - Capture drops and duplicates are applied at write time and recorded in `frame_map`.

## 6. Ground-truth JSON

Top level:
- `schema_version`, `generator` {version, git hash, ffmpeg version}, `seed`, the resolved `config`, `conventions`
- `video` {path, W, H, scan, true field order, metadata field order, postprocess, fps "30000/1001", par, n_out_frames}
- `camera` {C, R0, fx, fy, cx, cy, k1, hfov}
- `landmarks_world_ft`
- `landmarks_px` {calib_frame, [u, v] per field, visible}
- `distractors[]`, `frame_map[]` {out_frame, src_frame, exposure_times[]}, `timing`

**Per pitch:**
- Times: t_release, t_catch, t_plate_front, flight_time; release and catch as frame-unit floats.
- Distances: dist_y_ft, path_len_ft.
- Release position and velocity, catch position.
- Speeds: release, plate_front, catch, avg_y, avg_path.
- Physics {k, cd, magnus, wind}, swing.
- `fields[]`: field_index, frame, field_in_frame, parity, t, state (in_hand | flight), and:
  - **u, v at mid-exposure**, plus exposure start and end positions;
  - **u_vis, v_vis, vis_frac** (the exposure-weighted visible centroid);
  - r_px_x, r_px_y, depth;
  - in_frame, occluded fraction (todo: the prototype only tests the centre point);
  - peak_cov, local background luma.

## 7. Pitfalls

1. **Field timing.** Render each field at its own t_k and evaluate shake, actors and the ball there. Never render a frame and split it.
2. **Ball position vs. blob centroid.** In fields that straddle the release, the catch or an occlusion, the mid-exposure position is not the blob centroid. At 1/60 s I measured up to 2.6 px. Tests must score against u_vis/v_vis.
3. **Sub-field phase.** Randomize it, or timing-quantization error is hidden.
4. **No vertical filters on interlaced frames.** 4:2:0 conversion must be interlace-aware (`scale=interl=1`).
5. **Tiny balls.**
   - Temporal point sampling produces beaded streaks.
   - A 1-row aperture makes the ball vanish in alternate fields.
   - A PSF applied after sampling leaves 0.4 px centroid quantization.
6. **Estimator bias.** My first test harness averaged channels before linearizing and used a 2 % coverage threshold. That alone produced ±0.5 px centroid errors that alternated with parity on 1/60 s streaks (peak coverage is only about 0.1). Detectors face the same trap.
7. **Timestamps.** MPEG-PS reported `start_time = 0.533 s`. Use frame indices plus `frame_map`.
8. **Metadata is unreliable.**
   - MPEG-2 reports SAR 8:9 for content rendered at 10:11.
   - x264 flagged BFF content as `tt` until `bff=1` was passed explicitly.
   - The AVI files (DV and Xvid) report field order `unknown`.
   - The generator therefore supports deliberate metadata lies.
9. **Resize half-line offset.** `cv2.resize` of a field maps row r to 2r + 0.5 instead of 2r + p, which gives a ±0.5 px vertical "bob" bias.
10. **Behind view.**
    - Field order is nearly invisible in the ball track: a smooth fit gives 0.38 px residual with the wrong order vs 0.07 px with the right one. Yet the wrong order shifts timing by ±1 field, so detect field order from whole-image motion.
    - The umpire hid the plate in my render. Use the `clear_field_s` option and per-landmark visibility.
11. **Shake scope.** Only rotation and zoom are modelled (exact homography). Pans are limited to about ±20° by the planar canvas. Translational parallax is not modelled.
12. **Determinism.** Use per-subsystem `SeedSequence`. The same seed gave identical pitch lists for side and behind views. Use a frame-keyed RNG so rendering can be parallelized.
13. **Unknown extension.** The lead's E1 subtracts an assumed extension. Extension varies over 3–7.5 ft, about ±1.5 ft or ±2.7 % of D, which is roughly ±2.3 mph at 85 mph. GT records the release position so this bias can be measured.

## 8. What I checked in the prototype

- **Lossless round trip (FFV1 BGR):** decoded frames were bit-identical to the in-process render, with an identity frame mapping.
- **Centroid vs. GT on full-exposure flight fields** (TFF and BFF):

  | View | Shutter | u RMS | v RMS | Max |
  |---|---|---|---|---|
  | Side | 1/60, 1/250, 1/1000 | ≤ 0.014 px | ≤ 0.010 px | ≤ 0.04 px |
  | Behind | 1/60 – 1/500 | 0.02–0.04 px | 0.05–0.17 px | 0.64 px |

  The behind-view residual is not diagnosed. Partial glove occlusion and low background contrast are suspects, not confirmed.
- **Wrong field order, side view:** the smooth-fit residual goes from 0.1 px to 18 px.
- **Per-field index barcode** through MPEG-2/PS, H.264/MP4, Xvid/AVI and DV/AVI:
  - Identity mapping and correct BFF ordering in every frame except one, which a simulated tape dropout crossed.
  - `yadif=1` doubles the frame count, and output frame m is field m. `yadif=0`: frame n is field 2n.
  - 30p and 60p with a dropped and a duplicated frame: `frame_map` matched the barcodes exactly.
- **Codec quality:** PSNR vs. the master was about 33–34 dB for MPEG-2 (3 Mb/s), H.264 (1.2 Mb/s) and Xvid, and 39.8 dB for DV.
- **Flux, static ball** (r = 0.4–7 px at random sub-pixel positions):
  - Energy std ≤ 1.4 %; the worst range is −3 % / +7 % at r = 0.4 px.
  - Centroid error 0.02–0.03 px RMS with PSF σ = 0.7.
  - Peak-coverage variation is 16 % at r = 0.4 px and 9 % at 1.5 px. This comes from pixel sampling, and a real camera has it too.
- **Flux, moving streak:** energy std ≤ 0.2 %.
- **Physics** (40 pitches, MLB distances, C_d 0.30–0.40):
  - Release-to-catch speed loss was 4.8–10.6 %.
  - E1 inversion v0 = (e^{kD} − 1)/(kT) with exact T and k:

    | Distance used for D | Error |
    |---|---|
    | y-distance | −1.4 to 0 mph (worst on slow, arcing pitches) |
    | 3-D path length | ±0.4 mph |
    | Assuming C_d = 0.35 when the true value is 0.30–0.40 | −1.35 to +0.55 mph |

  - Timing sensitivity at 90 mph over 55 ft: one field of error is 3.6 mph, and 2 ms is 0.43 mph. **Sub-field timing is essential.**
- **Signal level (side view at about 70 ft, auto-fit):** r ≈ 1.0 px and peak coverage 0.10–0.26 at 1/120 s. That makes the ball faint, but not hidden by the configured noise.
- **Behind view (25 ft back):** r = 1.4–5 px, with 75–85 % of flight fields unoccluded.
- **Runtime:** a 60 s clip (1798 frames, 6 pitches, all effects, MPEG-2) took 60–61 s wall on a single thread here.
  - That is about 27 ms per frame, of which about 20 ms is the numpy noise/tape stage.
  - The frame-keyed RNG allows chunked multiprocessing. I expect a 2–3× speedup, but have not measured it.

## 9. Not modelled or not verified

- Lens distortion: the parameter exists but is unused.
- Translational shake and parallax.
- The out-of-focus backstop chain-link overlay. It is important for behind-backstop footage and planned as a blurred multiplicative layer.
- AGC pumping, focus hunting, shadows, bounced pitches, batted balls, rolling shutter.
  - I believe camcorders of that era used CCDs (global shutter), but I have not verified that.
- The blend and field_discard outputs, and DV with TFF content, were implemented but not checked.
- The scenery is cartoonish, so detectors must not be tuned to it. The user's real clip decides.