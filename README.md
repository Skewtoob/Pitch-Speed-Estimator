# pitchspeed

Estimate the speed of every pitch in a baseball (or softball) video, even on
old, low-quality camcorder tapes. You click a few landmarks once, and the tool:

1. finds each pitch in the video,
2. estimates its **release speed** (the number a radar gun shows) with an
   honest **± range**,
3. writes a table (`results.csv`, opens in Excel) with pitch number, time in
   the video and frame numbers, so you can tell which pitch is which,
4. writes an **annotated copy of the video** with each pitch's speed shown on
   screen,
5. lets you **review** every pitch, fix anything it got wrong, and add pitches
   it missed by hand.

It was built for early-2000s tape footage (MiniDV, Digital8, Hi8, VHS-C)
digitized to a file. That footage is *interlaced*: every frame holds two
half-pictures taken 1/60 s apart. The tool splits them, which doubles the
timing precision compared with counting frames.

> **Status:** tested only on computer-generated videos with known speeds
> (see [Accuracy](#accuracy)). Real tapes will be messier. Please check
> results in the review window, and expect some tuning once real footage is
> tried.

---

## Install (Windows)

1. Install **Python 3.10 or newer (64-bit)** from
   [python.org](https://www.python.org/downloads/). The default options are fine.
2. Download this project (green **Code** button → **Download ZIP**) and unzip
   it, e.g. to `C:\pitchspeed`.
3. Double-click **`windows\install.bat`**. It creates a private Python
   environment in `.venv` and installs everything (a few minutes the first time).

To analyze a video, **drag the video file onto `windows\pitchspeed.bat`**.
A console window walks you through the steps below.

Command-line use (from the project folder):

```
.venv\Scripts\pitchspeed run "D:\Tapes\game1.avi"
```

macOS/Linux: `python3 -m venv .venv && .venv/bin/pip install -e .`, then
`.venv/bin/pitchspeed run video.avi`.

## How it works, step by step

`pitchspeed run VIDEO` does four steps. You can also run each one on its own.

### 1. Calibrate (`pitchspeed calibrate VIDEO`)

The console asks where the camera was (**side** or **behind** the plate),
the pitching distance (league preset or your own number), and for side
views which hand the pitcher throws with. Then a window opens on the video.
The bar at the top always says what to click next.

| Camera view | Click (required) | Optional extra clicks |
|---|---|---|
| **Side**: the ball crosses the screen | middle of the **pitching rubber**; middle of **home plate** | ball at **release**; **glove** at the catch; top and bottom of something **vertical** (a pole) |
| **Behind** the plate: looking out at the pitcher | the **ball in the pitcher's hand at release**; the **catcher's glove** where the ball is caught | the two ends of the rubber; the two front corners of the plate (give the tool the ball size) |

Tips:
- Use arrow keys (or A/D) to step one field at a time, W/S for one second,
  `[` `]` for ten seconds. Find a frame where the pitcher is releasing the ball
  before clicking **release**, and one where the ball is in the glove for **catch**.
- **Z** shows an 8× magnifier; I/J/K/L nudge the selected point by ¼ pixel.
- **M**, then drag: mask out a burned-in date/time stamp.
- If the camera was moved or zoomed partway through the tape, go to that
  point and press **G** to start a new calibration segment. Click the
  points again there.
- **Enter** checks your clicks and shows warnings; press **Enter** again to save.
  **H** shows all keys.

The calibration is saved in `<video name>_pitchspeed\calibration.json`
next to the video. You only do this once per video, or per camera setup.

### 2. Analyze (`pitchspeed analyze VIDEO`)

Finds the pitches and prints a table like this:

```
#  time          frames   speed  +/-  90% range  avg  method    label  flags
1  00:00:02.504  75-89       86  1.5  84-88       82  side fit  good   -
2  00:00:08.961  268-288     60    1  58-61       56  side fit  good   -
3  00:00:14.992  449-466     75    1  73-76       71  side fit  good   -
```

- **speed**: release speed in mph (out of the hand, what a radar gun reads).
- **+/-**: one standard deviation. **90% range**: the true speed should be
  inside it 9 times out of 10, if the assumptions listed above the table hold.
- **avg**: average speed from release to catch. The ball loses about 8–10% of its
  speed on the way, so this is lower than the release speed.
- **label**: `good` (±2.5% or better), `fair` (±5%), `poor` (±10%, shown as
  `~78?`), or `n/a` when it cannot be measured honestly (the reason is listed
  under *Notes*).
- **time** and **frames** tell you where the pitch is in the video. The
  annotated video shows the same pitch numbers.

Anything moving that wasn't accepted as a pitch (throws back to the pitcher,
birds, …) is counted, and suspicious moments are listed as *possible missed
pitches* for you to check in review. Nothing is silently dropped.

### 3. Review (`pitchspeed review VIDEO`)

Steps through every pitch and every "possible missed pitch". The main keys:

| Key | Action |
|---|---|
| ←/→ (A/D) | back / forward one field (1/60 s) |
| ↑/↓ (W/S) | back / forward 10 fields |
| `[` / `]` | previous / next pitch |
| **R** | release = this field (the **first** field where the ball is out of the hand) |
| **C** | catch = this field (the **first** field where the ball is in the glove) |
| Tab, `,` `.` | select the R or C marker and nudge it by one field |
| **Y** / Enter | accept the pitch |
| **X** / Del | reject it (kept in the file as "rejected") |
| **N** | add a pitch the tool missed: press N at the release, then C at the catch |
| **E** | re-run the ball tracker between R and C |
| Space | play/pause, loops around the pitch; `+`/`-` change speed |
| U | undo |
| H | all keys |
| Q / Esc | save and quit |

Every edit re-computes that pitch's speed immediately and saves the results.

### 4. Annotated video (`pitchspeed render VIDEO`)

Writes `<video name>_pitchspeed\annotated.mp4` (standard H.264 + AAC. It
should play in the Windows Films & TV app and in VLC; not yet tested on Windows). While a pitch is in flight it shows the pitch
number and the ball's path. At the catch it shows a card with the speed and
its ± range. A running list of the pitches so far stays in the corner.
`--mode highlights` writes only the pitches.

## Output files

All in `<video name>_pitchspeed\` next to the video:

| File | What |
|---|---|
| `results.csv` | one row per pitch (Excel-friendly) |
| `results.json` | everything, including ball tracks and the uncertainty breakdown |
| `annotated.mp4` | the video with speeds overlaid |
| `calibration.json` | your clicks and settings |

## Accuracy

The ± range on each pitch is the tool's own estimate. On simulated videos
the true speed fell inside the 90% range at least 90% of the time. What
limits accuracy:

- **Side view** is the most accurate. Speed is measured from the ball's
  motion across the frame, using the rubber-to-plate distance as a ruler.
  Typical ±1–3%.
- **Behind the plate** can only time the flight from release to catch.
  Two things are uncertain there:
  - where exactly the pitcher let go: the **extension**, assumed about 5.5 ft
    in front of the rubber for high-school distances and less for youth
    fields;
  - the exact catch instant, when the catcher's body hides the ball.

  Expect roughly ±4–8%. Entering the pitcher's real extension
  (`--extension`) helps.
- **Video quality**: the ball is often only 1–3 pixels across. A heavily
  re-compressed file (e.g. downloaded from YouTube) may have erased it
  completely. Use the **original capture file** (DV-AVI or MPEG-2 from the
  capture device) whenever possible.
- **Deinterlaced or 30p files** lose the half-frame timing. The tool detects
  this, falls back to whole frames, and widens the ±.

Results on synthetic test videos (computer-generated, with known speeds) are
in [docs/ACCURACY.md](docs/ACCURACY.md). They show the method works under
realistic noise, blur, shake and compression. They do **not** prove accuracy
on real tapes; that needs real footage, ideally with radar readings.

## Settings you may want

| Option | Use |
|---|---|
| `--level hs / 54 / 50 / ll / 40 / sb43 / sb40 / sb35` or `--distance FT` | pitching distance (rubber to the back point of the plate). Check your league. |
| `--extension FT` | the pitcher's release point in front of the rubber, if you know it |
| `--throws R/L` | the pitcher's throwing hand (improves side views slightly) |
| `--field-order tff/bff/progressive` | override interlace detection (rarely needed) |
| `--start S --end S` | analyze only part of the video (seconds) |
| `--ui-scale 1.5` | bigger click/review windows on high-resolution screens |
| `--debug` | write diagnostic images to `<out>\debug\` |
| `pitchspeed probe VIDEO` | show format, interlacing and field order |

## Troubleshooting

- **"cannot open a window"**: the window library is missing. Reinstall with
  `.venv\Scripts\pip install --force-reinstall opencv-python`. Don't use
  `opencv-python-headless`.
- **Keys don't work in the windows**: run `pitchspeed keys` and send the
  printed codes. Every arrow key also has a letter twin (A/D/W/S).
- **No pitches found**: check the calibration clicks, and try `--debug`.
  Mark pitches by hand in review (**N**, then **C**).
- **Speeds look systematically off**: check the pitching distance, and on
  side views make sure you clicked the *middle* of the rubber and of the plate.

## For developers

- Design and the reasoning behind the methods: [docs/DESIGN.md](docs/DESIGN.md)
  and [docs/design-notes/](docs/design-notes/) (research notes with sources).
- Tests: `pip install -e .[dev]`, then `pytest -m "not slow and not gui"`.
- Synthetic test videos with ground truth: `pitchspeed synth generate --out clip.mpg`
  and `pitchspeed synth suite --n 24 --out-dir suite/`. Score results with
  `pitchspeed synth evaluate suite/`.
