"""Real-window smoke tests under Xvfb: XTEST keys/clicks into the cv2 (Qt) windows.

Each test runs a subprocess (`xvfb-run -a python -c ...gui_driver_main(kind)`)
so that a Qt abort can never take down the pytest process.  Skipped when
xvfb-run or python-xlib is not available, or on non-Linux platforms.
Set PITCHSPEED_GUI_TEST_USE_DISPLAY=1 to use the current $DISPLAY instead
of a private Xvfb (keys are injected there!).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gui

TESTS_DIR = Path(__file__).resolve().parent


def _launcher() -> list[str]:
    if not sys.platform.startswith("linux"):
        pytest.skip("Xvfb GUI tests run on Linux only")
    try:
        import Xlib  # noqa: F401
    except ImportError:
        pytest.skip("python-xlib not installed (dev-only: pip install python-xlib)")
    from pitchspeed.ui.gui import highgui_backend
    if (highgui_backend() or "NONE").upper() == "NONE":
        pytest.skip("OpenCV without highgui")
    if os.environ.get("PITCHSPEED_GUI_TEST_USE_DISPLAY") == "1" and os.environ.get("DISPLAY"):
        return []
    xvfb = shutil.which("xvfb-run")
    if xvfb is None:
        pytest.skip("xvfb-run not available")
    return [xvfb, "-a", "-s", "-screen 0 1280x1024x24"]


def run_driver(kind: str, timeout: float = 90.0) -> dict:
    code = (f"import sys; sys.path.insert(0, {str(TESTS_DIR)!r}); "
            f"import ui_helpers; ui_helpers.gui_driver_main({kind!r})")
    cmd = _launcher() + [sys.executable, "-c", code]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    assert p.returncode == 0 and lines, f"driver failed ({p.returncode}):\n{p.stdout[-2000:]}\n{p.stderr[-3000:]}"
    out = json.loads(lines[-1][len("RESULT "):])
    assert out["errors"] == [], out["errors"]
    return out


def test_calibration_window_clicks_and_keys():
    out = run_driver("calib")
    cf = out["result"]
    assert cf is not None
    pts = cf["segments"][0]["points"]
    assert pts["rubber"] == pytest.approx([400, 201], abs=0.6)
    assert pts["plate"][0] == pytest.approx(100, abs=0.6)
    assert pts["plate"][1] == pytest.approx(300 - 0.25, abs=0.6)          # nudged up by I
    assert cf["segments"][0]["ref_frame"] == 0
    assert "point_frames" in cf["notes"]


def test_calibration_window_closed_asks_and_reopens():
    """X button (WM_DELETE_WINDOW): the clicks are kept; 'e' re-opens the window, then 'y' saves."""
    out = run_driver("calib_close")
    assert out["prompts"] == 2
    cf = out["result"]
    assert cf is not None
    pts = cf["segments"][0]["points"]
    assert pts["rubber"] == pytest.approx([400, 201], abs=0.6)
    assert pts["plate"][1] == pytest.approx(300 - 0.25, abs=0.6)          # nudged in the re-opened window


def test_calibration_window_escape_twice_cancels():
    assert run_driver("calib_cancel")["result"] is None


def test_review_window_session():
    out = run_driver("review")
    nums = [(n, st) for n, st, _ in out["pitches"]]
    assert nums == [(1, "edited"), (0, "rejected"), (2, "auto")]
    assert out["pitches"][0][2] == 102 and out["manual_points"] == 1
    assert out["saved"] >= 3


def test_key_codes_helper():
    seen = run_driver("keys")["seen"]
    names = [n for _, n in seen]
    assert names[:17] == ["left", "up", "right", "down", "pgup", "pgdn", "home", "end", "del", "backspace",
                          "enter", "tab", "a", "z", ",", ".", "space"]
    assert names[-2:] == ["esc", "esc"]
