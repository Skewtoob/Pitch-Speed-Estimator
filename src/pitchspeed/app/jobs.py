"""What each button runs: `pitchspeed` command lines, and reading their progress output.

Pure functions (no Qt), so they can be tested without a display.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

from pitchspeed import constants as C

STYLES = ("radar", "broadcast", "bar", "classic")
STYLE_NAMES = {"radar": "Radar gun", "broadcast": "TV broadcast box", "bar": "Sports bar",
               "classic": "Plain text"}
CLI_EXE = "pitchspeed-cli.exe"     # the console program next to the windowed one in the Windows build


@dataclass
class Settings:
    level: str = "hs"
    style: str = "radar"
    audio: bool = True
    ui_scale: float = 1.0
    catch_cue: str = "arrival"

    def validated(self) -> "Settings":
        lv = self.level if self.level in C.LEVELS else "hs"
        st = self.style if self.style in STYLES else "radar"
        sc = min(3.0, max(0.5, float(self.ui_scale or 1.0)))
        cue = self.catch_cue if self.catch_cue in C.CATCH_CUES else "arrival"
        return Settings(lv, st, bool(self.audio), sc, cue)


@dataclass
class Step:
    title: str                    # shown above the progress bar
    args: list                    # arguments after the program
    opens_window: bool = False    # a marking / clicking window opens during this step


@dataclass
class Job:
    name: str
    video: str
    steps: list = field(default_factory=list)


def cli_command() -> list:
    """The program that runs `pitchspeed` commands: pitchspeed-cli.exe next to the Windows
    program, else this Python with -m pitchspeed."""
    if getattr(sys, "frozen", False):
        return [os.path.join(os.path.dirname(sys.executable), CLI_EXE)]
    return [sys.executable, "-u", "-m", "pitchspeed"]


def _scale(s: Settings) -> list:
    return [] if abs(s.ui_scale - 1.0) < 1e-6 else ["--ui-scale", f"{s.ui_scale:g}"]


def _video_step(video: str, mode: str, s: Settings) -> Step:
    title = {"clips": "Writing one clip per pitch", "full": "Writing the full video",
             "highlights": "Writing the highlights video"}[mode]
    a = ["render", video, "--mode", mode, "--style", s.style]
    return Step(title, a + ([] if s.audio else ["--no-audio"]))


def mark_job(video: str, s: Settings) -> Job:
    s = s.validated()
    return Job("Mark pitches by hand", video, [
        Step("Marking: the marking window opens (Q there saves and closes it)",
             ["mark", video, "--level", s.level, "--catch-cue", s.catch_cue, "--render", "none"] + _scale(s),
             opens_window=True),
        _video_step(video, "clips", s)])


def auto_job(video: str, view: str, s: Settings, throws: Optional[str] = None) -> Job:
    s = s.validated()
    if view not in ("side", "behind"):
        raise ValueError(f"view must be side or behind, got {view!r}")
    cal = ["calibrate", video, "--view", view, "--level", s.level] + _scale(s)
    if view == "side" and throws in ("R", "L"):
        cal += ["--throws", throws]
    return Job("Find pitches automatically (experimental)", video, [
        Step("Calibration: click the points in the window that opens, then Enter twice", cal,
             opens_window=True),
        Step("Finding the pitches and their speeds", ["analyze", video, "--render", "none"]),
        Step("Review: check the pitches in the window that opens (Q saves and closes it)",
             ["review", video, "--render", "none"] + _scale(s), opens_window=True),
        _video_step(video, "clips", s)])


def review_job(video: str, s: Settings) -> Job:
    s = s.validated()
    return Job("Check or fix the pitches", video, [
        Step("Review: the window opens (Q saves and closes it)", ["review", video, "--render", "none"] + _scale(s),
             opens_window=True)])


def video_job(video: str, mode: str, s: Settings) -> Job:
    s = s.validated()
    names = {"clips": "Make one clip per pitch", "full": "Make the full video", "highlights": "Make a highlights video"}
    return Job(names[mode], video, [_video_step(video, mode, s)])


PROGRESS_RE = re.compile(r"^\s*([A-Za-z][\w ]*?):\s+([\d.]+)%.*?eta\s+(\d+)s")


def parse_progress(line: str) -> Optional[tuple]:
    """(label, percent, seconds left) from a progress line such as
    '  analyze: 45.0%  (120/266)  elapsed    10s  eta    12s', else None."""
    m = PROGRESS_RE.match(line)
    if not m:
        return None
    return m.group(1), min(100.0, float(m.group(2))), int(m.group(3))


def split_output(buf: str) -> tuple:
    """Split process output into finished lines (on newline or carriage return) and the rest."""
    parts = re.split(r"\r\n|\r|\n", buf)
    return [p for p in parts[:-1]], parts[-1]
