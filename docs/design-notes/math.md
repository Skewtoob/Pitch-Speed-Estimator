# Speed-estimation math and error budget (design spec)

**Tags:** **[SIM]** means I simulated it in this study. **[DER]** means it follows from algebra. **[ASSUMED]** means it is an assumption or something I recall, not checked here. All sim code is in `/tmp/claude-0/-home-user-Pitch-Speed-Estimator/ba0610af-cbc9-58f8-8e1d-080fc4558d52/scratchpad/speedmath/` (`physics.py`, `camera.py`, `sim1*.py`–`sim5_catch.py`, `budget.py`, `diag_side.py`; raw outputs in `*.txt`). Nothing was written to the repo.

**Sim setup.** Full 3-D RK4 flight with drag, Magnus and gravity. Each pitch type has 40 random trajectories:
- Cd ~ N(0.33, 0.05)
- release side ±1–2.5 ft
- extension σ 0.7 ft, release height σ 0.4 ft
- random target and spin

Cameras are pinhole at 720×480, sampled at 59.94 fields/s with random field phase.

## 1. Flight model

Units are feet and seconds, along the pitch axis: dv/dt = −k·v².
- v(t) = v0/(1+k·v0·t)
- s(t) = ln(1+k·v0·t)/k
- v(s) = v0·e^(−ks)
- T(L) = (e^(kL) − 1)/(k·v0)

k = ρ_air·Cd·A/(2m).

**Default constants [ASSUMED]:**
- m = 0.145 kg; diameter 73.7 mm, so A = 4.27e-3 m²
- ρ_air = 1.20 kg/m³ (sea level, about 20 °C). Optional adjustment: ρ = 1.225·(288.15/(273.15+T°C))·exp(−elev_m/8500).
- Cd = 0.33, giving **k = 0.00177 /ft (0.00581 /m)**
- σ_k = 15% for v ≥ 65 mph and 25% below. Cd at youth speeds is less well established; I assume 0.30–0.50 is plausible.

**Speed lost over 54 ft by Cd [DER]:**

| Cd | 0.25 | 0.30 | 0.33 | 0.40 | 0.45 |
|---|---|---|---|---|---|
| Loss | 7.0% | 8.3% | 9.1% | 11.0% | 12.2% |

The default gives about an 8% loss from 50 ft to the front of the plate. That matches my memory of PITCHf/x-style start/end speeds, but I have not checked it.

**Gravity and Magnus [SIM sim1, sim1b].** Gravity is perpendicular to the axis and Magnus is perpendicular to the velocity, so both only affect the along-axis motion at second order. The real issue is that the 1-D model returns the *horizontal* release speed. Compared with the true 3-D release speed:
- 75–90 mph fastball, curve and slider: −0.3% to +0.15%
- 45 mph fastball: −0.5%
- 45 mph lob with an 8.5° release angle: −1.1%

Fix: multiply by √(1+tan²θ), with tanθ = (z_c − z_r + g·T²/2)/L. Defaults are z_r = 5.5 ft (4.5 ft youth) and z_c = 2.5 ft. After this correction the residual is ≤ 0.35% in every case. Use σ_model = 0.35% for E1 and 0.5% for E2.

## 2. E1: time of flight (any view)

**Formulas:**
- L = D_level − ext + c (horizontal distance, release to catch)
- v_avg = L/T
- **v0 = (e^(kL) − 1)/(k·T)**, which is v_avg × 1.04–1.05
- v_plate = v0·e^(−k·(D_level − ext − 17/12))

**Inputs:**
- D_level: 60.5, 54, 50 or 46 ft, measured from the rubber's front edge to the back tip of the plate (user-selected).
- ext default [ASSUMED]: 5.5 ft at 60.5 ft, 4.5 ft at 50/54 ft, 4.0 ft at 46 ft; σ = 1.0 ft. The user can override it.
- c, the catch point behind the back tip [ASSUMED]: 1.5 ± 0.75 ft.

**Sensitivities [DER]:**
- ∂ln v0/∂ln T = −1
- ∂ln v0/∂ln L ≈ 1 + kL/2 ≈ 1.05
- A k error of ±20% gives about ∓1% in v0.
- One field (16.68 ms) is 3.7%, 3.1%, 2.4% and 2.4% of T at 90, 75, 60 mph (60.5 ft) and 45 mph (46 ft).

**1σ % error of v0 [SIM sim2, Monte Carlo, n = 2e5].** Per-event timing models: frame quantization; field quantization; field quantization plus N(0, ½ field) jitter; sub-field N(0, 3 ms).

| case | T (ms) | frame-q | field-q | field+jitter | sub-field 3 ms | σ_L 1.25 ft | σ_L 0.4 ft | k 15% | **TOTAL default** (jitter, σ_L 1.25) | **TOTAL best** (3 ms, σ_L 0.4) |
|---|---|---|---|---|---|---|---|---|---|---|
| 90 mph @ 60.5 | 446 | 3.06 | 1.53 | 3.07 | 0.95 | 2.34 | 0.75 | 0.76 | 3.95% (3.6 mph) | 1.47% (1.3 mph) |
| 75 mph @ 60.5 | 540 | 2.53 | 1.26 | 2.53 | 0.79 | 2.33 | 0.74 | 0.76 | 3.54% (2.7) | 1.37% (1.0) |
| 60 mph @ 60.5 | 682 | 2.00 | 1.00 | 2.00 | 0.62 | 2.31 | 0.74 | 0.77 | 3.17% (1.9) | 1.28% (0.8) |
| 45 mph @ 46 | 685 | 1.99 | 1.00 | 1.99 | 0.62 | 2.98 | 0.95 | 0.59 | 3.65% (1.6) | 1.33% (0.6) |

Once timing is at field level, the **unknown extension and catch point dominate E1** (2.3–3.0%).

**Sub-field event timing:**
- **Catch [SIM sim5]:** fit a quadratic to u(t) and v(t) over the last 6 detections. Solve for closest approach to the glove point (where the ball sits in the first post-catch field, σ 3 px). RMS error is 2.2–4.1 ms, with bias ≤ +3 ms (worst with the camera 10 ft back). Plain "last detection + ½ field" gives 4.8 ms.
- **Release, behind view [SIM sim4]:** extrapolation cannot recover it, because the ball's image moves only about 1–2 px per field near release. Each undetected field biases v0 high by the per-field percentages above. With 0–3 missed fields, naive TOF was +3 to +9% biased (RMS 4–10%). The release must be taken as the first field where the ball is clearly free of the hand: detected automatically, then confirmed by the user in review.
- **Side view:** take events from the fitted E2 track: s(t_r) = y_r, and s(t_c) = the glove's plumb position.
- **Streak truncation** (measure the shutter duty cycle from mid-flight streak length vs inter-field displacement): proposed, not simulated. I assume camcorders may use shutters faster than 1/60 s in daylight.

## 3. E2: side view

**Mapping procedure.** P is the click on the plate back tip (Y = 0). R is the click on the rubber's front-edge centre (Y = D; it sits on the mound, about 10 in high, which does not matter here). Stabilize each field to the calibration frame, then:

1. **Plumb projection, not orthogonal projection.** In homogeneous coordinates, X = (b × V) × (P × R), where b is the ball and V is the vertical vanishing point. V defaults to the image column direction (0,1,0); optionally it comes from two clicked vertical lines. Then μ = (X−P)·(R−P)/|R−P|². This step removes the ball's height above the ground line.
2. **Projective 1-D map [DER].** With λ = Y/D: μ = λ/(ρ(1−λ) + λ), where ρ = Z_P/Z_R is the ratio of camera depths to plate and rubber. ρ = 1 is the affine case.
3. **Motion model.** Y(t) = y_r − ln(1 + k·v0·(t − t_r))/k, with y_r = D − ext_nominal. Extension only enters through e^(kΔ), about 0.18% per foot.
4. **Fit** θ = (v0, t_r, ln ρ) with `scipy.optimize.least_squares`:
   - residual = (μ_obs − μ_model)·|R−P|, in pixels
   - `loss='soft_l1'`, `f_scale=2`
   - initialize from a linear fit
   - bound ρ to [0.05, 20]
5. Apply the gravity correction from §1.

**Uncertainty:**

σ² = σ_cov² + Σ_j(Δv0 from refitting with click coordinate j moved by σ_click)² + (Δv0 from refitting with k·(1+σ_k))² + (v0·σ_lat)² + (0.005·v0)²

where σ_cov² comes from (JᵀJ)⁻¹·s² and:
- σ_lat = (|tan φ|/L + 0.4/Z_mid)·σ_Δx
- σ_Δx = 1.8 ft when the release side is unknown. It is 0.6 ft if handedness and camera side are entered; in that case also correct the mean.
- φ is the angle between the pitch line and the image plane. Proposed estimate (not simulated): tan φ ≈ rubber's apparent length ÷ (2 ft × along-line px/ft at R), which needs two extra clicks on the rubber ends.

Covariance alone under-covers badly: 90% intervals covered only 0.01–0.89 of cases. With the full formula, coverage is **0.93–1.00**.

**Results [SIM sim3, sim3b]:** 1 px centroid noise, 2 px click noise, random release side, Cd σ 0.05. Values are % RMS error of v0; "cov" is 90%-interval coverage.

| camera | ρ | φ | affine bias / RMS | projective self-cal RMS: 90 / 60 / 45 mph |
|---|---|---|---|---|
| perpendicular, 90 ft | 1.00 | 0° | +0.1 / 1.1 | 1.2 / 1.4 / 1.6 (cov 0.93–0.94) |
| perpendicular, 150 ft, zoomed | 1.00 | 0° | ~0 / 1.0 | 1.0 / 1.1 / 1.2 |
| elevated 15 ft, 90 ft | 0.87 | 13° | +2.0 / 2.5 | 1.8 / 2.0 / 2.3 (cov 0.93–0.96) |
| 60 ft off line, beyond the mound | 1.67 | 42° | −11 / 12 | 2.1 / 2.5 / – (cov 0.98–1.00) |
| 120 ft off line, 40 ft behind plate | 0.80 | 30° | – | 2.6 / 2.8 / 3.3 (cov 0.94–0.96) |
| 40 ft off line, 60 ft behind plate | 0.56 | 66° | 6–8 RMS | 9–12 → reject; treat as behind view |

**Findings:**
- **Self-calibrating ρ from the ball's own motion works.** RMS is within 0.0–0.5 percentage points of using the true ρ, even at ρ = 1.67. Affine is only acceptable when ln ρ ≈ 0: it is already +2% biased at ρ = 0.87. Always fit ρ and report it.
- **The main error is the ball's sideways travel combined with obliquity.** Noise-free RMS is 0.2–0.7% with no sideways travel, and 1.0–9.5% with a realistic release side. Use E2 only when |φ| ≤ about 45°.
- **Pixel noise does not limit accuracy.** Going from 1 px to 2.5 px changed RMS by ≤ 0.2 points. So fields vs frames matter little for E2; I derived this from the noise result rather than simulating it separately.
- **Partial track:** with only the last 50% of the flight visible, RMS is 2.3–4.2% and coverage 0.87–0.97.
- **Camera roll:** 3° of roll with image columns used as vertical changed results by ≤ 0.4 points.
- **Field of view [SIM geometry, ASSUMED camcorder widest HFOV about 43–54°]:** to get rubber and plate in one frame, the camera must be roughly 70–90 ft from the pitch line. A first-base dugout about 75 ft out, and a camera 60 ft from the line near the plate, both needed a lens wider than f = 700 px. If the camera pans, it needs per-field registration to the calibration frame; without that, E2 is invalid.
- **Field order (not simulated):** fit E2 under both field orders and keep the lower cost. The wrong order leaves residuals that alternate in sign (strongly negative lag-1 autocorrelation). [ASSUMED] DV is bottom-field-first.

## 4. Behind-plate view [SIM sim4, sim4_check, sim5]

Simulated cameras sat 10–40 ft behind the plate and 4–15 ft high, zoomed to include the pitcher and glove. That gives f ≈ 340–2000 px and a ball of 1.3–15 px.

**(a) Time of flight — use this.** When release was detected in its first field, RMS error was 2.3–2.9% (with extension σ 0.7 ft in the sim). Predicted budget (`budget.py`):

| scenario | 90 @ 60.5 | 75 | 60 | 45 @ 46 |
|---|---|---|---|---|
| A: release in 1st field, catch extrapolated, default ext | 2.8% (2.5 mph) | 2.7% (2.0) | 2.8% (1.7) | 3.3% (1.5) |
| B: A + release latency σ ½ field | 3.4% (3.0) | 3.1% (2.3) | 3.0% (1.8) | 3.5% (1.6) |
| C: A + user-entered ext (σ 0.5 ft) | 2.3% (2.0) | 2.1% (1.6) | 2.3% (1.4) | 2.5% (1.1) |
| D: frames only (29.97 fps) | 3.5% (3.1) | 3.2% (2.4) | 3.1% (1.9) | 3.5% (1.6) |

**(b) Perspective speed from a calibrated camera — cannot work.** With near-constant speed along the axis and unknown sideways/vertical motion (Magnus is unknown), scaling the whole trajectory about the camera centre and shifting the release time gives an identical image track [DER]. Results:
- Zero noise, true camera and true release depth: RMS 57–78%.
- Release time bounded near the first detection, 1 px noise: 15–35%.
- With catch-time and release-height priors: 8–15%, which is worse than plain TOF.

The rubber (24 in) and plate (17 in) widths are useful only for plausibility checks (expected ball size, search corridor).

**(c) Apparent ball size (1/w linear in t) — do not use.** With a 2.5 px camcorder/compression blur, the estimate was biased by −14% to −83%. Debug output only.

**Recommendation:** the behind view uses E1 only, with an honest ±3–4%.

## 5. Combining estimates and reporting

**Combining:**
- Split each estimate's variance into an independent part and the shared k term.
- v = Σ wᵢvᵢ with wᵢ ∝ 1/σᵢ,ind²; σ² = 1/Σσᵢ,ind⁻² + σ_k².
- If |v_E1 − v_E2| > 2√(σ1,ind² + σ2,ind²), flag `E1_E2_disagree`. Likely causes: clicks, ext/c, event times, field order, dropped frames. Prefer E2 if its residuals look healthy.

**What to report per pitch:** release speed (primary, "radar-gun equivalent"), average speed L/T, and plate speed, each with 1σ and a 90% interval (±1.645σ). [ASSUMED] Modern guns and trackers report peak/release speed; older "slow" guns read later in the flight. Example row:

`#7 00:03:12.45 frames 5760–5787 | 71 ± 2 mph (90%: 68–74) | avg 68 | plate 65 | E2 side, ρ=0.88`

**Confidence labels:**
- good: σ ≤ 2.5%
- fair: σ ≤ 5%
- poor: σ ≤ 10%; show the number with "≈"
- none: σ > 10%, fewer than 8 fields, or an unresolved inconsistency. Show no number, only "not measurable: reason".

**Always show:**
- Assumptions that were used rather than measured: ext, c, Cd, ρ_air.
- Flags: `release_unconfirmed`, `catch_extrapolated`, `partial_track`, `frames_only`, `field_order_ambiguous`, `dropped_frame_suspected`, `oblique_view`.

Display speeds to 1 mph and σ to 0.5 mph; keep raw values in JSON. Validate on synthetic videos that 90% intervals cover 85–95% of cases. Never quietly retune k to match one radar reading.

## 6. Critique of the lead's plan

1. **E1 is correct**, but the extension/catch-point uncertainty (about 2.3–3%) is larger than the field-timing error. The calibration step should ask for the pitcher's extension, height or handedness, and review should let the user mark release and catch positions (side view).
2. **E2: affine is unsafe.** Use plumb projection plus a projective map with ρ fitted from the ball's motion. Add the σ_lat term, and reject oblique views (|φ| > about 45°).
3. **E3 should be dropped.** It is mathematically degenerate, not merely fragile; the ball-size variant is too blurred to use.
4. **Splitting into fields** is essential for E1 (it halves timing error) but minor for E2.
5. **Sub-field refinement:** catch, yes (2–4 ms). Release in the behind view, no; it needs user confirmation.