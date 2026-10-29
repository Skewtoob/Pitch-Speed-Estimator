Spec (≤2000 words), written before the code; the experiment scripts it mentions were throwaway prototypes and are not in the repository.

# Study: video I/O, interlacing, output video, UI, packaging

## 0. Experiment results
**Setup.** I made 6-second 720×480 clips with fields rendered at 59.94 Hz and a known ball position x(t). Encoded versions:
- DV in AVI, MOV and raw .dv (bottom field first, BFF)
- MPEG-2 .mpg and .vob (top field first, TFF)
- interlaced H.264 MP4 (TFF)
- H.264 flagged progressive but holding BFF content
- WMV2, HuffYUV, MJPEG
- 30p, blend-, bob- (59.94p) and single-rate-deinterlaced variants

All tests ran on Linux. The Windows facts come from inspecting the real win_amd64 wheels.

1. **OpenCV `VideoCapture` silently returns garbage for interlaced-flagged frames** in the opencv-python(-headless) 4.14 and 5.0 Linux wheels. `read()` still returns ok=True. 179 of 180 frames were wrong on DV, MPEG-2 and interlaced H.264. The cause is the bundled FFmpeg 8 swscale error "Cannot convert interlaced to progressive frames". Version 4.11 was correct. The Windows 5.0.0.93 wheel bundles FFmpeg 7.1 and lacks that error string, so it is probably fine, but this is unverified. **Do not decode with OpenCV.**
2. **PyAV 19.0.2** decoded all 13 files frame-exact against ground truth, gray conversion included. Decode plus field split ran at 800–1400 fps on 4 cores, about 2 minutes per tape-hour.
3. **Timestamps:**
   - OpenCV `FRAME_COUNT` was wrong for WMV (183, actual 180).
   - Raw .dv reports avg_frame_rate=60000/1 (bogus, ffprobe too); `base_rate`/`guessed_rate` are correct.
   - WMV uses a 1 ms timebase, so frame deltas alternate 33/34 ms.
   - MPEG-PS video starts at 0.5334 s and audio at 0.5233 s.
4. **Seeking:** a naive PyAV seek in MPEG-PS lands 11–12 frames late; even a seek to the start lands on the second GOP. What works: seek to target − 1.5 s, decode forward, match by pts, and retry further back (or decode from the start). That gave 30/30 frame-exact results across 5 containers.
5. **Field-order metadata is unreliable.** The H.264 file has stream field_order=tt but per-frame top_field_first=0. DV-AVI reports "unknown". The progressive-flagged file holds BFF content.
6. **Content detector (§2): 11/11 correct.** ffmpeg `idet` was also 11/11. In the hard case (static camera, only the ball moving, noise σ≈8), ours said "uncertain" and `idet` said "Progressive" for the TFF clip (wrong). A ball-track test was decisive: median second difference 3.2 px for the right order vs 95 px for the wrong one.
7. **Field timing check:** with t = t_frame + k·Δ/2 and the detected order, measured ball x matched ground truth to a median 0.6–0.75 px. The wrong order gave 24 px (one field of motion).
8. **Bob display:** a naive `cv2.resize` of fields makes the picture jump 1.0 frame-line vertically between fields. A parity-aware `warpAffine` gives 0.00.
9. **Writers:**
   - OpenCV `avc1`/`H264` fail on Linux. The Windows OpenCV FFmpeg DLL has no libx264, only an OpenH264 loader that needs `openh264-2.5.0-win64.dll`, which is not shipped. MSMF is compiled in but untested.
   - The imageio-ffmpeg 0.6.0 Windows exe is FFmpeg 7.1 built with `--enable-gpl --enable-libx264`.
   - The PyAV Windows wheel ships `libx264-165.dll`.
   - Piping to libx264 runs at 133 fps end-to-end at 960×720p59.94 (decode, warp and text included).
10. **Audio passthrough:** A/V drift was 0.0 ms (beep vs frame) on DV-AVI, .mpg, .vob and WMV with the rebasing in §3. A naive offset gave a 523 ms error, and `-shortest` caused a BrokenPipe.
11. **Wheels:** win_amd64 wheels exist for all dependencies on Python 3.10–3.15. win_arm64 has no opencv-python or imageio-ffmpeg wheels.
12. **The GUI can be tested here:** Xvfb + opencv-python (Qt) + python-xlib XTEST. Measured Qt codes: ←65361 ↑65362 →65363 ↓65364 Home65360 End65367 PgUp65365 PgDn65366 Del65535 Bksp8 Enter13. With no display, `cv2.namedWindow` hard-aborts the process (Qt) rather than raising.

## 1. Decoding (`io/reader.py`)
- **Decoder:** PyAV for all analysis, using `to_ndarray(format="gray")` (luma only, which sidesteps 4:1:1 / interlaced 4:2:0 chroma upsampling). BGR only for display and output.
- **Index pass:** record every pts and cache it in `index.npz`, keyed by file size and mtime. Count frames by decoding; never trust metadata.
- **Timing:** Δ = 1/base_rate.
  - Constant frame rate: if the mean pts step is within 0.2% of Δ and max|pts_i − (pts_0 + iΔ)| < 0.25Δ, set t_i = iΔ exactly. This removes the WMV millisecond jitter.
  - Otherwise use t_i = pts_i − pts_0 and flag `vfr`.
  - pts gaps > 1.5Δ mark dropped frames. Near-identical consecutive frames (mean|ΔY| < 0.3σ) mark capture duplicates.
  - A pitch spanning either gets flag `timing_glitch` and is never auto-accepted.
- **t = 0** at the first video frame, which matches what players show.
- **Random access:** the strategy from §0.4.
- **Containers:** anything FFmpeg demuxes — DV type-2 AVI, raw .dv, DV in MOV, MPEG-2 PS/VOB/TS, MP4/MOV H.264, WMV, HuffYUV/MJPEG AVI. Type-1 DV AVI should work through FFmpeg's internal DV demuxer, but I did not test it.
- **Windows paths:** never pass Unicode paths to cv2 I/O.
- **Pixel aspect ratio (PAR):**
  - Store everything in stored-pixel frame coordinates. Field row r of parity p maps to frame row y = 2r + p.
  - SAR comes from the stream. Defaults: 8:9 for 720/704×480 at 4:3 (10:11 if the user wants strict BT.601), 1:1 for 640×480. `--par` overrides.
  - Doesn't matter for time-of-flight (E1), or for a 1D projective map along the pitch line (E2), because anisotropic scaling is affine and cross-ratios survive.
  - Does matter anywhere x and y are mixed: Euclidean distances, perpendicular projection onto a tilted axis, blob elongation, ball radius, E3 widths. Convert with x_sq = x·SAR.
  - A round ball looks about 2.25:1 wide in a field image, so elongation tests must use square-pixel units.

## 2. Interlace and field order (`io/interlace.py`)
**Detector** (numpy, tested):
- Split T = rows 0::2 and B = rows 1::2, and interpolate T onto B's rows.
- On a motion mask (same-parity difference > 6σ, σ from the MAD), compute mean absolute differences: a = D(T_k, B_k), b = D(B_k, T_k+1), c = D(T_k, B_k+1). All three are cross-parity, so vertical-detail bias cancels.
- Progressive: a ≪ min(b, c) (measured 0.04–0.13). TFF: b/c ≈ 0.37–0.43. BFF: b/c ≈ 2.2–2.7.
- Rules: median a/min(b, c) < 0.6 with ≥60% of votes → progressive; b/c < 0.8 → TFF; > 1.25 → BFF; otherwise uncertain. b and c both ≪ a means field-shifted progressive; re-pair the fields.

**Policy:**
1. Run the detector per 10-second window. Camcorders switch to slow shutter in low light, so segments can turn progressive mid-file.
2. Cross-check with bundled `ffmpeg -vf idet` (parse the last "Multi frame detection" line). If they disagree, the verdict is uncertain.
3. Per pitch, track the ball under both orders and keep the lower median second difference. If that contradicts the global verdict, flag `field_order_conflict`. If it is uncertain everywhere, use frame timing and widen the uncertainty.
4. Metadata is only a hint. `--field-order auto|tff|bff|progressive` overrides.

**Sample index and timing:**
- s = 2i + k, where k is the temporal order within the frame.
- Parity p = k for TFF, 1 − k for BFF.
- t_s = t_i + kΔ/2.
- The field image is `frame[p::2]` at its native 240 rows, not resized.
- The mid-exposure offset is a constant and cancels in all speed estimates.

**Digitization variants:**
- 59.94p bob: one sample per frame. Check for fake 60p made by frame doubling (near-identical pairs) and treat it as 29.97.
- Blend-deinterlaced: frames, but the ball shows up as two half-contrast ghosts one field apart. Detection should expect this; each ghost is a field sample.
- Single-rate deinterlace, camcorder "frame mode" 30p, 352×240: frames.
- Vertically rescaled interlaced video: fields are destroyed. Warn and use frames.
- Telecine: shows as a 5-frame periodicity in a. Warn and use frames.
- PAL (50 fields/s): same code.

**Why it matters:** a wrong order puts the endpoints off by ±1 field (16.7 ms), up to ±4% on a 0.42 s flight.

## 3. Annotated output (`io/writer.py`)
Primary path: pipe BGR frames to imageio-ffmpeg's ffmpeg.
```
ffmpeg -y -f rawvideo -pix_fmt bgr24 -s 960x720 -r 60000/1001 -i - -i SRC
 -map 0:v:0 -map 1:a:0 -af "asetpts=PTS-{shift}/TB,aresample=async=1:first_pts=0"
 -c:a aac -b:a 160k -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p
 -profile:v high -movflags +faststart -t {dur} OUT.mp4
```
- shift = video_start − container_start, because the ffmpeg CLI rebases each input to its container start.
- Don't use `-shortest`. Drop the audio arguments when the source has no audio.
- `yuv420p` is mandatory; bgr24 input would otherwise produce 4:4:4. H.264 High with AAC-LC in MP4 should play in the stock Windows player, but I could not test that on Windows.
- Interlaced sources produce one output frame per field (59.94p). Each field is bob-deinterlaced (stretched to full height) with the parity offset, to square-pixel 960×720 (`--video-size`).
- Fallbacks (automatic, logged): PyAV libx264 video-only, then OpenCV `mp4v`, then MJPG AVI. Never OpenCV `avc1`.
- Audio is worth keeping: it costs little. Don't rely on the glove pop for timing: the capture A/V offset is unknown and sound adds about 0.9 ms per foot. At most it is a future cross-check.
- Default render is highlights (release − 2 s to catch + 3 s per pitch; audio via atrim/concat, untested). Full length at 133 fps takes about 27 minutes per tape-hour. Flag: `--render full|highlights|none`.

## 4. UIs (`calib/ui.py`, `review/ui.py`)
**Structure:** a pure `State` with `on_key(name)` / `on_mouse(ev, x, y)`, unit-tested, and a thin cv2 shell that can also run under Xvfb.

**Key normalization (`ui/keys.py`):**
- Use `waitKeyEx`; `waitKey() & 0xFF` turns arrow keys into 0.
- Win32 codes are VK<<16, taken from OpenCV's Win32 backend and not run on Windows: ←2424832 ↑2490368 →2555904 ↓2621440 PgUp2162688 PgDn2228224 End2293760 Home2359296 Del3014656. Qt codes are in §0.12.
- Every arrow binding has a letter twin. `pitchspeed keys` prints raw key codes.

**Window behavior:**
- Exit the loop when `getWindowProperty(WND_PROP_VISIBLE) < 1`, so the window's X button works.
- Use `WINDOW_AUTOSIZE` with our own canvas (default 720×540, square-pixel 4:3, `--ui-scale`) and map coordinates ourselves.
- Windows DPI awareness via ctypes is untested. On Linux, check DISPLAY before opening a window.
- Default view is the bobbed field (what the detector sees). F toggles the full frame. Z toggles an 8× loupe of 31×31 px with a crosshair.

**Calibrate flow:** probe → console asks view and distance (unless flags give them) → window, with the HUD naming the next point. Points:
- Behind view: `rubber_left`, `rubber_right` (front-edge ends, 24 in), `plate_front_left`, `plate_front_right` (17 in), `plate_apex`.
- Side view: `rubber_front`, `plate_front`, `plate_apex`, plus optional `ref:<name>` points with a distance entered at the console.
- Both views: optional `release_hint`, `catch_hint`.

| key | action |
|---|---|
| ←/→, A/D | ∓1 field |
| ↓/↑, S/W | ∓1 s |
| PgDn/PgUp, [ ] | ∓10 s |
| Home/End | start/end |
| trackbar | coarse scrub (debounced 150 ms) |
| L-click / R-click | place point / delete nearest |
| I J K L | nudge selected point 0.25 px |
| Tab | next point |
| Backspace/U | undo |
| N | skip optional point (e.g., plate hidden) |
| M | drag a mask over the date burn-in |
| G | new calibration segment from here (zoom/pan changed) |
| F Z H | field/frame, loupe, help |
| Enter | sanity-check (warnings) + save |
| Esc ×2 | quit without saving |

**Review flow:**
- Per pitch, decode fields from release − 60 to catch + 60 into RAM.
- A bottom strip shows the timeline with R (green) and C (red) ticks plus a text line: "Pitch 3/12 · auto · 00:01:23.457 · fields 4986–5013 · 78±3 mph (avg 74) · TOF · conf 0.82 · flags".

| key | action |
|---|---|
| ←/→, A/D | ±1 field |
| ↓/↑, S/W | ±10 fields |
| PgUp/PgDn, [ ] | prev/next pitch |
| Home/End | go to release/catch |
| R / C | set release/catch to the current field |
| Tab, `,` `.` | pick the R or C marker; nudge it ∓1 field |
| Y/Enter | accept |
| X/Del | reject (kept, status=rejected) |
| N | new manual pitch (release here; then C) |
| L-click / R-click | add/remove a manual ball point at this field |
| E | re-run the tracker constrained to R..C |
| Space, +/- | loop playback, 1/8–1× speed |
| T Z F H | track overlay, loupe, view, help |
| G | side by side: 8 consecutive fields (frames) in a 4×2 grid, all magnified 1.5–8× around one spot (the mouse at G; R-click a tile moves it, +/- zoom). A/D or a click picks the shown field (yellow border), marks act on it; the set slides only when the pick leaves it. G/Esc closes, Space closes and plays |
| U/Backspace | undo (unbounded) |
| Q/Esc | save, re-export CSV, offer re-render |

Every edit recomputes immediately and autosaves `results.json` atomically (write a temp file, then `os.replace`).

## 5. Packaging
- **pyproject** (setuptools, src layout):
  - `requires-python >=3.11`, since 3.10 reaches end of life in Oct 2026.
  - Dependencies: `numpy>=1.26`, `scipy>=1.11`, `av>=12`, `imageio-ffmpeg>=0.5`, `opencv-python>=4.8,<6`; `dev` extra = `pytest`.
  - `[project.scripts] pitchspeed = "pitchspeed.cli:main"`.
- **OpenCV variants:** never install opencv-python and opencv-python-headless together (they share the `cv2` directory). Sandbox development goes in a venv. If the installed cv2 has no GUI, calibrate and review print a fix-it message.
- **User install:** Python 3.13 x64 from python.org (x64 even on ARM laptops) → download the ZIP → double-click `install.bat`:
```
@echo off
cd /d "%~dp0"
py -3.13 -m venv .venv || py -3 -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -e .
.venv\Scripts\pitchspeed --version
pause
```
- **Drag-and-drop:** `pitchspeed.bat` contains `"%~dp0.venv\Scripts\pitchspeed.exe" run "%~1"` then `pause`.
- **CI:** windows-latest and ubuntu, Python 3.11 and 3.13.

## 6. Layout, CLI, outputs
```
src/pitchspeed/{cli,__main__,version}.py
  io/{reader,probe,interlace,fields,writer}.py   calib/{model,ui}.py
  review/{state,ui}.py   ui/{keys,canvas,loupe}.py   output/{tables,overlay}.py
  synth/render.py   detect/ track/ physics/
tests/   windows/{install,pitchspeed}.bat   README.md
```
**Commands:**
- `probe VIDEO`: codec, SAR, timing mode, field-order evidence, dropped/duplicated frames.
- `calibrate VIDEO [--view behind|side] [--distance 60.5|preset] [--frame N]`
- `analyze VIDEO [--calib F] [--start/--end hh:mm:ss] [--field-order] [--par] [--render …] [--no-audio]`
- `review`, `render`, `keys`, `synth OUT` (test videos).
- `run VIDEO`: wizard that chains probe → calibrate if needed → analyze → review → render.

**Output folder** `<video>_pitchspeed/`: `calibration.json`, `results.csv`, `results.json`, `annotated.mp4`, `index.npz`, `log.txt`.

**Distance presets:** 60′6″, 54′, 50′, 46′, custom, labelled "verify for your league".

**`results.csv`** (one row per pitch):
- pitch, status (auto|edited|manual|rejected)
- time_release as hh:mm:ss.sss (milliseconds, which avoids drop-frame ;ff ambiguity), t_release_s, t_catch_s, flight_s
- frame_/field_release, frame_/field_catch, view
- method (tof|side_fit|tof+side_fit|manual_tof)
- v_release_mph, v_release_unc_mph (1σ), v_avg_mph, v_avg_unc_mph
- confidence (0–1)
- flags (semicolon list: field_order_conflict, timing_glitch, release_occluded, catch_extrapolated, low_contrast), notes

**`results.json`** (schema `pitchspeed.results/1`): version; video {path, size, codec, base_rate, SAR, timing mode, dropped/duplicated frames}; interlace {verdict, per-window a/b/c, idet}; a copy of the calibration; params; pitches[] (the CSV fields plus track `[[s, t, x, y, score, manual]]`, per-method estimates, uncertainty budget, edit history).

**`calibration.json`** (schema `pitchspeed.calibration/1`): video fingerprint, view, distance_ft, PAR and its source, segments `[{frame_start, frame_end, points{name:[x,y]}, masks}]`.

**Overlay:**
- Top-left HUD: source time and field number.
- During flight: a "P3" tag and a trail (last 8 samples solid, older ones faded; green = auto, yellow = edited).
- At catch: a card "Pitch 3: 78 mph" with "±3 · release est. · TOF" underneath, held 3 s or until the next release.
- Low confidence shows "~78?" in orange; rejected pitches are not drawn.
- Top-right: a running list "P1 76 · P2 79 · P3 78". Integers on video; one decimal in the CSV.

## Changes to the lead's plan
- Splitting into fields is right. But field order must come from the content plus a per-pitch track check. Metadata, and even `idet`, can be wrong.
- Drop OpenCV decoding entirely (silent garbage frames, §0.1).
- Blend-deinterlaced footage still carries field-rate information in the two ghosts.
- Shape priors on field images must correct for SAR and the halved row count.
- Make highlights the default render; a full-length render is slow.

**Unverified:** Win32 key codes and DPI handling, MSMF H.264, OpenCV 5 decoding on Windows, Type-1 DV AVI, highlight-reel audio concat, Windows playback of `mp4v`.