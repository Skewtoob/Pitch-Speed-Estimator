# Prior art, physics constants and footage facts

Confidence tags:
- **H**: official or primary text, or several sources that agree.
- **M**: a reputable secondary source, or a fact seen only in a search-engine excerpt.
- **L**: a forum post, a vendor claim or a single source.
- **[calc]**: my own arithmetic.
- **[sandbox]**: I tested it with ffmpeg 6.1.1 in throwaway prototype scripts (not in the repository).
- **[est]**: my estimate, with no source behind it.

The network allowlist blocked mlb.com and fangraphs.com, so facts from those sites come from search excerpts and rate M at best.

## 1. Prior art

**Manual frame-marking (the most common approach)**
- **Baseball Radar Gun +** times the flight between frames the user marks.
  - The vendor claims ±0.5 mph with slow-motion video. A user says the result changes a lot if the marker slips "a few frames". [L]
  - It is licensed under patent US 8,803,913, which claims frame marking at 60 fps or more. Its example works out to about 3.5 mph per frame at 60 fps. [L/M]
- **Snap Pitch** and **Bowling Speed Meter**: the user marks release and catch, then enters the distance. [L]
- **psfv** (open source) computes distance ÷ time from release to the ball over the plate. It supports softball only. Its README says the result is "slower than the MLB velocity (VELO)". [H]
- **Forum practice:** count frames at 29.97 fps and add about 1 mph for every 7 ft of flight. One poster says the distance is the main error source. [L]
- **Kinovea** (free, Windows):
  - A 2021 study (Shishov) found about 7% velocity error on 30 Hz video with a 90° view and grid calibration, and worse at a 30° view. [H]
  - Its forum notes that fast objects appear twice after deinterlacing. [L]

**Automated tools and research**
- **arXiv 2003.03856** (331 side-view MLB clips):
  - Mean absolute error was 2.53 mph, with a systematic −2.27 mph.
  - A camera on the opposite side read about 2.5 mph high. The authors blame their assumption that the ball flies in a known vertical plane. [M]
- **MoVVE** (SIGCSE 2015 student poster): finds the ball with Hough circles, gets depth from the ball's pixel radius, at 60 fps. 94 of 112 pitches were within 10% of radar and 51 within 5%. [M]
- **MLB-YouTube** (CVPRW 2018) learns speed from broadcast video. An error of about 3.6 mph is claimed; I did not verify it. [L]
- **Commercial apps:** SpeedUp (side view) claims ±2 mph, and up to 5 mph off radar. PitchLab and Velo claim radar-level accuracy with no published validation. [L]
- **GitHub:**
  - `pascalewalters/baseball-tracking` uses YOLOv3, misses the ball against white pants, and computes no speed. [H]
  - `chonyy/ML-auto-baseball-pitching-overlay` uses YOLOv4 + SORT for overlays and computes no speed. [H]
  - I found no tool built for interlaced SD footage. GitHub search was blocked here, so that is not proof none exists.

**Pitfalls these sources report:**
- Release and end frames are ambiguous.
- Average speed gets confused with peak speed.
- Distance assumptions are wrong.
- The nominal frame rate differs from the actual one; one phone project measured 59.8–60.05 fps. [L]
- Detectors lose the ball against white uniforms.
- Side views carry a bias from the flight-plane assumption.
- Deinterlacing leaves ghost images.

## 2. Flight physics

**Ball and air**
- The ball weighs 5–5¼ oz with a 9–9¼ in circumference (Official Baseball Rules 3.01). [H] That gives a diameter of about 2.86–2.94 in. [calc]
- Standard air density is 1.225 kg/m³ at sea level and 15 °C. [H] Denver is about 82% of that. [M]

**Drag coefficient (Cd)**
- The range is 0.30–0.45, and the seams decide where a ball falls (Nathan & Young). [M]
- A Kagan & Nathan worked example gives 0.338. Users of Nathan's trajectory calculator commonly fix it at 0.35. [M]
- Drag-crisis values as low as 0.16 have been reported. [L/M] A 90 mph pitch is at a Reynolds number of about 1.85×10⁵. [M]
- Cd at youth pitch speeds is unverified and may be higher. [est]

**Speed loss from release to plate**
- About 9–10% over the roughly 55 ft from release. [M]
- Nathan's example: 100 mph at the hand, 99 at 50 ft, 94 midway, 91 at the plate. [M]
- About 8% at Coors Field. [M]
- So the lead's "~8–10%" holds at MLB distance.

**My model check** [calc] (quadratic drag, no gravity, air density 1.2, mass 145 g, diameter 73.8 mm)

| Flight | Cd 0.30 | Cd 0.35 | Cd 0.45 |
|---|---|---|---|
| 54.5 ft: speed loss | 8.4% | 9.8% | 12.4% |
| 54.5 ft: release ÷ average speed | 1.045 | 1.053 | 1.069 |
| 41 ft (youth): speed loss | 6.4% | n/a | 9.5% |
| 41 ft (youth): release ÷ average speed | 1.034 | n/a | 1.051 |

- In this model the loss depends only on Cd × air density × distance.
- Uncertainty in Cd alone adds about ±1.2% to an E1 release speed.

**What "velocity" means**
- Statcast's VELO is the "maximum speed… from release to… home plate", which is reached at release. [M]
- Baseball Savant's `release_speed`:
  - For 2008–16 it is PITCHf/x "adjusted to roughly out-of-hand".
  - From 2017 it is Statcast "out-of-hand". [H/M]
- PITCHf/x measured `start_speed` at 50 ft and `end_speed` at the front of the plate. [M] The 2017 switch to Statcast raised readings by about 0.4–0.6 mph on average. [M]

**Handheld radar guns**
- A gun reads only the speed component along its beam (speed × cosine of the angle), so an off-angle reading can only be low. [M] 10° off costs about 1.5%. [calc]
- The Stalker Sport 2 shows both a peak (release) speed and a plate speed. [M]
- Stalker material quoted on a forum says the JUGS gun reads about 7 ft after release and the Decatur Ra-Gun 30–50 ft after. [L]
- Guns disagree with each other by 1–2 mph. [L] A radar reading remembered from the early 2000s is only a loose check.

## 3. Distances and geometry

**Field geometry**
- **MLB (Official Baseball Rules 2.04):**
  - The rubber is 24 × 6 in, set 60 ft 6 in from the "rear point of home plate". [H]
  - The rule doesn't say which edge of the rubber. Baseball-Reference says the front edge. [M] Either way the ambiguity is at most 0.8%.
- **Home plate:** a 17 in edge faces the pitcher, with 8½ in sides and 12 in rear edges. [H]
  - That makes the plate 17 in deep, so its front edge is 1.417 ft ahead of the back point. [calc; matches PITCHf/x]
- **Mound:**
  - An 18 ft circle centred 59 ft from the back point of the plate.
  - The rubber sits 18 in behind the centre and 10 in above the plate. [M/H]

**Distances by level** (rubber to back point of the plate)

| Level | Distance | Conf. |
|---|---|---|
| High school, college | 60 ft 6 in | M/H |
| Little League Majors and below / Intermediate | 46 / 50 ft | H/M |
| Little League Junior, Senior | 60 ft 6 in (Junior has a local option of 54 ft) | H/M |
| PONY Pinto / Mustang / Bronco / Pony / Colt | 40 / 46 / 50 / 54 / 60 ft 6 in | M |
| Softball NFHS/NCAA · men's fastpitch | 43 · 46 ft | M |
| Softball 12U · 10U | 40 ft (one source says 38) · 35 ft | L/M |

- Little League and softball measure from the front edge of the rubber. [H/M]
- "13U = 54 ft" matches PONY's Pony division. Travel leagues vary [est], so the distance should stay editable.

**Extension**
- The typical MLB pitcher in 2023 released about 6.4 ft in front of the rubber; leaders were near 7.0 ft. [M]
- A coaching blog puts it at about 1.04 × the pitcher's height. [L]
- High-school showcase leaderboards show 5.5–6.8 ft, but these are the top pitchers, not averages. [L]
- I found no youth average.

**Flight time and catch point**
- The average MLB four-seam fastball took 0.41 s from release to plate in 2015. [M]
- A coaching guide puts the catcher about 2 ft behind the plate. [L] I estimate the glove is 1–3 ft behind the back point. [est]

## 4. Camcorders and digitization

**DV (MiniDV and Digital8)**
- **Format:** 720×480 at 30000/1001 fps, interlaced, 4:1:1 colour, 25 Mb/s intraframe. [M/H]
- **Field order:** bottom field first. Apple's Final Cut Pro manual says DV is "always" lower field first. [H/M]
- **Pixel aspect ratio:** 10:11 by the ITU convention or 8:9 by the MPEG-2 convention. [M] ffmpeg reports 8:9. [sandbox]
- **Audio:** usually 48 kHz/16-bit.
  - DV audio isn't clock-locked to video and may slip up to ±⅓ frame. [L/M]
  - Type-2 AVI captures can drift. [L]
  - Check sync before timing a catch from the mitt-pop sound.
- **Metadata:** each frame stores the recording date and time, which ffprobe can show as "Recorded date". [M] This is useful for splitting clips.
- **Hi8/Video8 tapes:** many Digital8 models play these and output DV over FireWire [M], so those transfers are bottom field first too.

**Analog tapes and captures**
- **Resolution:** Video8 and VHS-C are about 240 lines; Hi8 and S-VHS-C about 400. [M]
- **Field order** of an analog capture depends on the capture device and its driver. [L/M]
- **Deinterlacing:** consumer capture sticks like the Elgato output 640×480 H.264 deinterlaced to 30p. [L/M]
- **Dropped frames:** captures made without a time-base corrector often drop or duplicate frames. [L]

**Camera behaviour**
- **Fields:** CCD camcorders sample the two fields about 1/60 s apart [M] and have no rolling-shutter skew. CMOS models from 2005 on, such as the Sony HC1, do skew. [M]
- **Shutter speed varies by model:**
  - One Sharp model's auto mode ranges 1/60–1/250. [M]
  - The Canon ZR90's manual settings run 1/60–1/2000, and Canon recommends 1/2000 for sports. [M]
  - Sony's "Auto Slow Shutter" is on by default on some models and drops to 1/30 in low light. [M/H]
  - Canon's "Frame mode" records 30p inside a 60i signal. [M]
- **Motion blur:** an 85 mph ball moves about 2.1 ft during a 1/60 s exposure. [calc]

## 5. Other accuracy factors and critique of the plan

1. **The E1 endpoints dominate the error.**
   - One field at 59.94 Hz is 4.1% of a 0.41 s flight, about 3.7 mph at 90 mph. For youth flights it is about 3%; at 29.97 Hz it doubles. [calc]
   - It is better to fit the in-flight track and evaluate it at the release and plate planes than to detect the release and catch frames.
2. **Distance terms add up.**
   - ±1 ft of error in extension or in catch point is about ±1.9% each, plus ±1.2% from Cd. [calc]
   - A realistic E1 uncertainty is about ±3–5 mph at MLB speeds. [est]
3. **Neither container metadata nor ffmpeg's interlace-detection filter (idet) can be trusted for field order.** [sandbox]
   - ffprobe reported field order "unknown" for DV in both .dv and .avi files.
   - idet returned 100% "Undetermined" when only a ball-sized object moved.
   - It called a progressive test pattern top field first (49 of 60 frames).
   - It was right when the whole frame moved bottom-field-first.
   - The ball track itself is a better test: with the wrong field order it zig-zags back and forth. [L]
4. **Some modes destroy field timing and must be detected:** deinterlaced 30p, Frame mode, slow shutter, and dropped or duplicated frames.
   - Use presentation timestamps (PTS), not frame indices.
   - A blend deinterlace may still hold two ball ghosts 1/60 s apart. [est]
5. **Side view (E2).**
   - About 2.5 mph of bias from the flight-plane assumption is documented, and the ball flies above the ground line.
   - A ground homography fixes most of this: the rubber and plate corners give at least 4 points, plus handling for ball height. A 2-point affine map is biased when the view is oblique. [est]
   - The homography also absorbs the pixel aspect ratio.
6. **Behind view (E3).**
   - The ball is only about 2–3 px across at roughly 80 ft with a wide-angle SD lens [est], so depth from ball size is not feasible.
   - I agree with the lead that E3 is fragile.
7. **Don't use streak length for speed.** It needs the exposure time, which is unknown. Treat the streak centroid as the mid-exposure position.

## Sources
- **Prior art:**
  - github.com/syncom/psfv
  - github.com/pascalewalters/baseball-tracking
  - github.com/chonyy/ML-auto-baseball-pitching-overlay
  - apps.apple.com/us/app/id1248662974
  - image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8803913
  - sites.google.com/view/snap-pitch/
  - apps.apple.com/app/id6745965041
  - sites.google.com/view/speedupbaseball/speedup/accuracy
  - apps.apple.com/ca/app/id6738223162
  - apps.apple.com/app/id1467688665
  - arxiv.org/pdf/2003.03856
  - gray.cs.ua.edu/pubs/sigcse-2015-matt.pdf
  - arxiv.org/pdf/1804.03247
  - pmc.ncbi.nlm.nih.gov/articles/PMC8544843
  - kinovea.org/en/forum/viewtopic.php?id=33
  - hackernoon.com/we-built-a-cricket-speed-gun-from-a-phone-camera-and-got-the-physics-wrong
  - discussfastpitch.com/threads/pitch-speed-measurement-math-vs-radar-gun.17498
- **Physics and radar:**
  - blogs.fangraphs.com/exploring-the-variation-in-the-drag-coefficient-of-the-baseball
  - physics.csuchico.edu/kagan/professional/papers/drag.pdf
  - sabr.org/latest/lang-a-physics-comparison-of-great-throws-from-years-past/
  - baseballaero.com/2019/05/03/baseball-drag-crisis/
  - baseball.physics.illinois.edu/Denver.html
  - fox21news.com/top-stories/the-science-of-playing-baseball-in-the-mile-high-city/amp/
  - espn.com/mlb/story/_/id/19107750
  - mlb.com/glossary/statcast/velocity
  - baseballsavant.mlb.com/csv-docs
  - physics.csuchico.edu/baseball/Pubs/PitchFX.pdf
  - tht.fangraphs.com/the-physics-of-radar-guns/
  - community.hsbaseballweb.com/topic/stalker-vs-jug
  - betterbaseball.com/stalker-sport-2-radar-gun.html
  - umpirebible.com/OBR16/3.0.htm
- **Distances:**
  - baseballrulesacademy.com/official-rule/mlb/2-04-1-07-pitchers-plate/
  - mlb.com/glossary/rules/field-dimensions
  - baseball-reference.com/bullpen/Rubber
  - baseballrulesacademy.com/official-rule/nfhs/section-2-the-field/
  - littleleague.org/league-officials/field-specifications
  - littleleague.org/downloads/5070-field-conversion
  - en.wikipedia.org/wiki/PONY_Baseball_and_Softball
  - forum.officiating.com/softball/54124-asa-16u-pitching-distance-print.html
  - coversports.com/resources/faq/a-guide-to-softball-field-dimensions
  - mlb.com/news/futures-game-tools-ranked-by-statcast
  - rocklandpeakperformance.com/extension-at-release-why-its-important-and-how-to-maximize-it/
  - mlb.com/news/syndergaard-fernandez-fastballs-harder-to-hit-c171706062
  - chadmoellerbaseball.com/baseball-catchers-set-up/
- **Video:**
  - support.apple.com/guide/motion/aside/motn63ed0ded/mac
  - manualsdir.com/manuals/547697/apple-final-cut-pro-5.html?page=1711
  - tech-pubs.net/sgi_developer/books/007-3524-004/apf.html
  - forum.videohelp.com/threads/312656-Can-someone-EXPLAIN-the-whole-720x480-thing-to-me
  - dvinfo.net/forum/archive/index.php/t-142974.html
  - forum.videohelp.com/showthread.php?p=1793269
  - ffmpeg.org/pipermail/ffmpeg-user/2016-May/032035.html
  - videohelp.com/dvanalog
  - capture.com/pages/vhs-c-to-digital/
  - en.wikipedia.org/wiki/8_mm_video_format
  - forum.videohelp.com/topic257631.html
  - forum.videohelp.com/showthread.php?p=2673506
  - forum.videohelp.com/showthread.php?p=2787382
  - camcorder.fandom.com/wiki/Interlaced?oldid=289
  - dvinfo.net/forum/sony-hvr-a1-hdr-hc-series/77803-hc1-a1u-rolling-shutter-question.html
  - manualslib.com/manual/489448/Sharp-Vl-Z100h.html?page=65
  - manualslib.com/manual/1141184/Canon-Zr-90.html?page=57
  - sony.ca/en/electronics/support/articles/00030763
  - dvinfo.net/forum/archive/index.php/t-478157.html