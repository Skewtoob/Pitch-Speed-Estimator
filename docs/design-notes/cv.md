# Spec: ball detection, tracking, pitch segmentation (SD interlaced)

Prototype scripts are in `/tmp/claude-design-detection/` (synth.py, detect.py, tbd.py, fieldorder.py). This file is also saved at `/tmp/claude-design-detection/spec.md` (1,993 words). All numbers below come from synthetic clips. They are sanity checks, not claims about how real tapes will behave.

## 0. Experiments

Setup: 2 px ball, 12 px/field, noise σ=4, ±3 px shake. FA means false alarms per field in a 720×68 corridor band. z is the median matched-filter score at the ball.

| condition | z at ball | detected / FA per field at z≥5 |
|---|---|---|
| uncompressed | 6.1 | 80% / 0.02 |
| MPEG-2 3 Mb/s | 4.8 | 45% / 0.4 |
| H.264 CRF 28 | 1.7 (streak erased) | 12% / 14 |
| 3 px ball, MPEG-2 | 8.8 | 100% / 0.15 |
| 3 px ball, 0.7 px/field (behind-view early flight) | 19 | 100% / 0.1 |

- **False alarms:** bilinear field warps gave 52 FA/field and bicubic gave 34. A local block noise scale then cut that to 0.3, but recall dropped from 97% to 84%.
- **Matched filter:** it detected about as well as thresholding plus connected components. Its centroid error was 0.8 px RMS instead of 2.5–3.5 px; both figures include about 0.4 px of bias from my renderer.
- **Shift-and-add over 8 fields:** z rose from 6.1 to 10 (uncompressed) and from 4.8 to 7.5 (MPEG-2). It adds nothing once the codec has erased the streak.
- **Shutter:** a dim ball was detected 86% of the time at shutter duty 0.3, and 16% at duty 1.
- **Field order:** the vote was correct in both directions (55:4 and 2:57).

**What this means:** a wide side view gives a 1.6 px ball, which is at the detection limit. Re-encoded files may contain no detectable ball at all. The tool must report that and fall back to manual marking. We should ask the user for the original DV-AVI or MPEG-2 capture.

## 1. Decode, fields, timing

- **Decode:** `ffmpeg -i in -map 0:v:0 -fps_mode passthrough -f rawvideo -pix_fmt gray -`. No deinterlacer, luma only.
- **Timing:** read the PTS of each frame. Field time is t = PTS + i·Δt, where Δt = 1001/60000 s and i is the field's temporal index within its frame.
- **Coordinates:** pixel centres sit on integers. Field row r of parity p is frame row y = 2r+p. Every later stage uses frame pixels of the stabilized calibration frame.
- **Interlace test:** the comb ratio is the median, over moving pixels, of |F(y)−½F(y±1)|, divided by the same measure at y±2. A ratio above 1.3 means interlaced.
  - 59.94p input: treat each frame as a field.
  - 29.97p blend-deinterlaced input: the ball appears twice at half contrast. Merge the two copies to their midpoint.
  - Anything else: run in frame mode and say so.
- **Field order:**
  - Bob each field to frame rows. On moving pixels compute:
    - e_TFF = |B0−½(T0+T1)| + |T1−½(B0+B1)|
    - e_BFF = |T0−½(B0+B1)| + |B1−½(T0+T1)|
  - Take a majority vote over at least 50 frames. The prior is BFF for DV. A CLI override is available.
  - Self-check: if the residuals along a fitted track alternate in sign, flip the field order and rerun.
- **Resolution:** detection runs on native 720×240 fields. Motion compensation runs on bobbed fields at half size (360×240).
- **Capture faults:** a duplicate frame is one whose compensated difference is below 0.3σ while the scene is moving. A dropped frame is a PTS gap above 1.5/fps. Flag any pitch that spans either.

## 2. Global motion compensation

- **Estimate:** a similarity transform (pan, tilt, zoom, roll) from each field to a keyframe, using:
  - `goodFeaturesToTrack`: 400 corners, quality 0.01, minDist 8
  - `calcOpticalFlowPyrLK`: 21×21 window, 3 levels
  - `estimateAffinePartial2D`: RANSAC, 1 px threshold
- **Feature exclusion mask:** the dilated corridor, body blobs, the timestamp mask, 12 px borders left and right, and the bottom 12 lines (VHS head-switching noise).
- **Keyframes:** refresh every 30 fields, or sooner if overlap drops below 70%. G_k = G_key∘S_k.
- **Quality:** q_mc = clip((inlier−0.3)/0.4, 0, 1)·exp(−rms/1 px).
  - With fewer than 30 inliers, fall back to `phaseCorrelate`.
  - If that also fails, mark the field MC_FAIL and down-weight its candidates.
- **Scene cut:** re-register to the calibration frame with AKAZE + RANSAC. If that fails, mark the segment UNCALIBRATED.
- **Timestamp text:** pixels with high gradient that are stable in raw coordinates but not after stabilization while the camera moves. Add any box the user draws in `calibrate`.

## 3. Candidate detection (field k)

```
B = median(warp_cubic(F[k+o]→k), o ∈ ±4,±6,±8,±10)  # same parity only; ±2 dropped (slow ball self-cancels)
R = blur(F[k]) − (a·blur(B)+b)                       # a,b: robust gain fit (AGC); blur σx=0.8, σy=0.4 rows
```

- **Same-parity backgrounds** avoid half-line interpolation error.
- **Behind view, release half:** if the ball moves less than d/4 px/field there, use offsets ±{8..20} for that half of the corridor.
- **Scale, side view:** κ = L_px/D_ft (px/ft). Ball diameter d = 0.242κ. Displacement δ(v) = κ·1.4667·v_mph·Δt px/field.
  - Example: κ = 6.6 gives d = 1.6 px, and δ = 6.5 and 16.2 px/field at 40 and 100 mph.
- **Scale, behind view:** d comes from the pixel widths of the rubber (24 in) and plate (17 in).
- **Kernels:** zero-mean line segments with a Gaussian profile.
  - Width max(0.8, d/2.5), and at least 0.6 rows.
  - Lengths L ∈ {0, d+0.15δ̄, d+0.5δ̄, d+δ̄}, clipped to 3–30.
  - Angles: side view uses the axis and axis ±20°. Behind view uses 0/45/90/135° with short lengths.
  - After the first confident tracks, estimate shutter duty as median((L_obs−d)/|v|). Pass 2 then uses the matched length.
- **Score:** z = (R∗k)/σ_loc.
  - σ_loc = 1.4826·median|R∗k| over 8×32 blocks, floored at the global median.
  - Take the maximum over kernels and over both signs, since the ball may be dark against sky.
  - Clutter map: C ← (1−1/120)C + [z≥3]/120, then z_eff = z·min(1, √(0.02/C)).
- **Candidates:** non-maximum suppression over 5 rows × 11 px. Weak if z_eff ≥ 3.5, strong if z_eff ≥ 5. Keep at most 30 per field.
  - Position: the centroid of max(sign·R−1.5σ, 0) in an oriented (L+4)×3 window. This is the mid-exposure position.
  - Shape: L_obs = √(12λ_max), plus the angle, from second moments.
  - `attached` flag: set when the candidate touches a |R|>3σ blob larger than max(150, 25d²), meaning arm, glove or body.
  - Position error σ_pos = max(0.4, 3/z) px. Map positions to the reference frame with G_k.
- **Corridor (reference coordinates):**
  - Side view: from P_r−1.5κû to P_c+3κû. Vertically from 1 ft below ground up to 8 ft at the rubber end and 6 ft at the plate end.
  - Behind view: a quadrilateral from P_rel to P_c. Half-width is 1.5 ft·scale at the release end, and the larger of 0.35|P_c−P_rel| and 3 ft·scale at the catch end.
- **Calibration clicks:** `calibrate` also asks for a **release point** (the hand at release) and a **catch point** (the glove).
  - Side view: optional. Defaults are 5.5 ft extension and 5.5 ft height for release, and 2 ft behind the plate at 2.5 ft height for the catch.
  - Behind view: **required**, because the catcher and umpire often hide the plate.
- **Progress coordinate:** s = (p−P_rel)·ê/|P_c−P_rel|, running from 0 at release to 1 at the catch.

## 4. Tracking: seed, extend, robust global fit

I chose this over DP/Viterbi. It is easier to debug and its cost is bounded. Accepting weak candidates inside tight gates gets close to track-before-detect. If real footage needs more, the upgrade path is shift-and-add stacking (§0) or a second-order Viterbi.

**Gates (px/field):**
- Side view: δ(30 mph)/ρ ≤ |v| ≤ δ(105 mph)·ρ with ρ = 1.5, and the velocity within 35° of û.
- Behind view: Δs ≥ −0.01 per field, |v| ≤ 0.5|P_c−P_rel|, and flight time T between (D−7)/154 and (D−3)/44 s. At D = 60.5 ft that is 21–78 fields.

```
for b strong ∧ ¬attached in C[k]; g1,g2∈1..3, g1+g2≤4; a∈C[k−g1]; c∈C[k+g2]:
    v1=(b−a)/g1; v2=(c−b)/g2
    if gate(v1) ∧ gate(v2) ∧ |v2−v1| ≤ max(1.5,.25|v1|)(g1+g2)/2: seeds += (a,b,c, Σz)
for seed in seeds by score desc, skipping used nodes:
    for dir in ±1:
        miss = 0
        while miss ≤ (10 if p̂ in body blob else 4):
            p̂,Σ = predict(T)              # quadratic on last ≤8 pts; global model once ≥6
            r = max(2, d/2+1) + 0.15|v|·gap + 2√trΣ
            c* = argmax_{|c−p̂|<r} z_eff(c) − 2(|c−p̂|/r)²   # weak/attached allowed
            c* ? (add, miss=0) : miss++
            stop if s(p̂) ∉ [−0.15,1.15] or p̂ leaves corridor
    fit_global(T); mark used
merge tracks agreeing within 3 px over an overlap or a gap ≤12 fields
```

**Global model:** x = (a0+a1τ+a2τ²)/(1+c1τ+c2τ²) and y = (b0+b1τ+b2τ²)/(1+c1τ+c2τ²).
- It is exact for a pinhole camera with constant 3D acceleration, and approximate under drag. It is used only for gating, filling gaps and extrapolation, not to compute speed.
- Fit: linearize by multiplying out the denominator, then RANSAC (5 points, 200 iterations, inliers within max(1.5, 0.3d) px in x and 2 px in y), then `least_squares(soft_l1)`.
- Side view: compare against a plain polynomial (c = 0) using BIC.
- Behind view: export the pole −1/c1 (the time the ball would reach the camera plane) for the 3D study.

**Pass 2, pitch windows (±1 s) only:** cache the response maps, re-detect at z ≥ 2.5 inside the model gates, and refit.

## 5. Pitch segmentation

**Reject codes:**
- TOO_SHORT: fewer than 6 inliers.
- NO_PROGRESS: Δs < 0.35, unless one end of the track is occluded.
- THROWBACK: ds/dt < 0. Kept as context.
- OFF_AXIS (pickoffs, warm-ups):
  - side view: more than 35° off the axis;
  - behind view: the track misses P_rel by more than 0.25, or P_c by more than 0.3, of |P_c−P_rel|.
- SPEED_IMPLAUSIBLE:
  - side view: outside 25–110 mph;
  - behind view: T outside its window.
- POOR_FIT: RMS above 2 px. Birds usually fail this test or the speed test.
- DUPLICATE: within 1.5 s of a better track.

**CONTACT:** a track ends near the plate and a fast track leaves within 3 fields. The end event is then bat contact, not a catch, and the speed module is told so.

**Delivery cue (works without seeing the ball):** E_rel(t) is the foreground fraction within 3 ft of P_rel. A burst there, followed by activity at P_c 0.3–1.3 s later, with no accepted track, is flagged POSSIBLE_MISSED_PITCH for `review`. The glove pop in the audio helps find events but not timing: the A/V offset is unknown, and sound lags about 1 ms per foot.

**Event times:** export the raw observations (t_first, t_last, `attached` flags) and the model estimates, labelled separately. In side view, extrapolating with an assumed extension is equivalent to E2, so the speed study must not count it twice.
- **Side-view release:** t_rel is the model time at s_rel, and must be earlier than t_first. σ² = σ_model² + (κ·1 ft/ṡ)², which is about 0.5 field at 90 mph.
- **Catch:** take the last moving field.
  - If its streak is truncated (L_obs < d+duty·|v|−2): t_c = t_open + (L_obs−d)/|v|.
  - Otherwise extrapolate to the stationary ball or glove, or to P_c ± 1 ft.
  - σ is about 0.3 field when the glove is visible; otherwise the error is uniform over the gap.
- **Behind-view release (the hard case):** the ball separates from the hand late and moves only 1–3 px/field.
  - Backward extrapolation to P_rel gives σ_t of about 2–5 fields.
  - The peak of arm motion energy is unvalidated, perhaps ±1–2 fields.
  - Report both, and inflate σ when they disagree.
  - An error of 2–4 fields on a 25-field flight is 8–16% speed error. This will probably dominate the behind-view time-of-flight (E1) error.

## 6. Confidence

These are heuristic rankings, not probabilities, until they are tuned on labelled real clips.

- **Candidate:** p = logistic((z_eff−4.5)/0.7)·exp(−max(0, L_obs−L_pred−3)/5).
- **Track:** conf = f_snr·f_cov·f_fit·f_ends·f_speed·f_mc·f_cue, where:
  - f_snr = logistic(median z − 4)
  - f_cov = min(1, N_inl/(0.6·N_expected))
  - f_fit = exp(−max(0, rms−1))
  - f_ends = exp(−(d_rel/r_rel)² − (d_c/r_c)²)
  - f_speed = a plausibility factor
  - f_mc = the minimum q_mc over the track
  - f_cue = 1 if a delivery cue is present, otherwise 0.8
- **Labels:** 0.6 and above is OK, 0.3–0.6 is REVIEW, below 0.3 is DOUBTFUL. Nothing is dropped silently.

## 7. Diagnostics (`--debug`)

- `overview.png`: corridor, clicks, masks and clutter map.
- `timeline.png`: per-field q_mc, σ, candidate counts and E_rel, with accepted, rejected and possible-missed pitches marked.
- `pitch_NNN.png`: stroboscopic maximum of sign·R with numbered detections, the model curve, release and catch markers, and s(t) residuals.
- `tracks.jsonl` (with reason codes), `candidates.npz` (for re-tuning without re-decoding), and `interlace_report.json`.
- Optional quarter-speed MP4 showing the candidates next to the |R|/σ map.

## 8. Expected failure modes

- The ball is sub-pixel, or the codec has erased it.
- Low contrast: white uniforms, bright sky, sunlit dirt.
- Fast zoom or pan, or autofocus locking onto the backstop fence (blur, parallax, moiré).
- The ball merges with the arm or glove at release, or is hidden by the umpire, batter or catcher.
- Wrong field order, blend-deinterlacing, dropped or duplicated frames, variable frame rate.
- The camera moved without recalibration, or a date overlay sits inside the corridor.
- Distractors: bullpen balls, bats, throws along the pitch axis.

## 9. Cost (10 minutes = 36,000 fields)

Measured on the unoptimized prototype, one 2.1 GHz Xeon core:
- motion compensation: 4–5 ms/field
- warps plus median: 8–11 ms/field
- 13-kernel bank: 22–34 ms/field (68-row band)

That is about 20–30 min single-threaded. The planned optimizations are cropping to the corridor bounding box, about 6 kernels, recomputing σ every 4th field, and 4 processes with 40-field chunk overlap. With those, the target is about 3–6 min, plus 1–2 min for pass 2 and about 1 min to render the MP4. A 21-field ring buffer keeps memory under 200 MB. None of this has been measured on Windows.

## 10. Unknown without real footage

- shutter duty
- ball size and contrast after the codec
- amount of shake and zoom
- how often the ball is occluded
- whether there is a date overlay
- how far the behind-view camera sits off the pitch line (this sets how much the ball moves early in flight)
- whether the arm-energy cue works

Every threshold here is a starting point. Re-tune them from the `--debug` statistics (z, duty, FA) on the first real clip.