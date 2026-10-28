"""Key normalization tables and the GUI availability checks (no window needed)."""
from __future__ import annotations

import pytest

from pitchspeed.calib import state as calib_state
from pitchspeed.review import state as review_state
from pitchspeed.ui import gui
from pitchspeed.ui.keys import (COCOA_KEYS, QT_MEASURED, WIN32_CODES, check_bindings, check_letter_twins,
                                describe_code, key_label, normalize_key, resolve)

# Win32 VK<<16 codes listed in docs/design-notes/io_ux.md section 4
IO_UX_WIN32 = {"left": 2424832, "up": 2490368, "right": 2555904, "down": 2621440, "pgup": 2162688,
               "pgdn": 2228224, "end": 2293760, "home": 2359296, "del": 3014656}
# Qt codes listed in io_ux.md section 0.12 (measured there, re-measured here)
IO_UX_QT = {"left": 65361, "up": 65362, "right": 65363, "down": 65364, "home": 65360, "end": 65367,
            "pgup": 65365, "pgdn": 65366, "del": 65535, "backspace": 8, "enter": 13}


@pytest.mark.parametrize("name,code", sorted(IO_UX_WIN32.items()))
def test_win32_codes_from_design_note(name, code):
    assert normalize_key(code) == name
    assert WIN32_CODES[code] == name


@pytest.mark.parametrize("name,code", sorted(IO_UX_QT.items()))
def test_qt_codes_from_design_note(name, code):
    assert normalize_key(code) == name


@pytest.mark.parametrize("name,code", sorted(QT_MEASURED.items()))
def test_qt_measured_codes(name, code):
    assert normalize_key(code) == name


def test_win32_chars_and_function_keys():
    # WM_CHAR delivers characters; Enter/Esc/Tab/Backspace as control characters
    assert normalize_key(13) == "enter"
    assert normalize_key(27) == "esc"
    assert normalize_key(9) == "tab"
    assert normalize_key(8) == "backspace"
    assert normalize_key(32) == "space"
    assert normalize_key(0x70 << 16) == "f1"
    assert normalize_key(0x7B << 16) == "f12"
    assert normalize_key(0x2D << 16) == "insert"
    assert normalize_key(0x10 << 16) is None          # VK_SHIFT is not reported, but never maps


def test_letters_are_lower_case_everywhere():
    for c in "abcdefghijklmnopqrstuvwxyz":
        assert normalize_key(ord(c)) == c
        assert normalize_key(ord(c.upper())) == c   # Win32 with Shift / caps lock


def test_gtk_modifier_bits_are_stripped():
    shift, numlock = 1 << 16, 0x10 << 16
    assert normalize_key(shift | ord("a")) == "a"
    assert normalize_key(numlock | 0xFF51) == "left"
    assert normalize_key(numlock | 0xFF0D) == "enter"
    assert normalize_key(numlock | 0xFFB4) == "4"     # keypad 4 with NumLock (GTK keysym)
    assert normalize_key(0xFFE1) is None              # Shift_L alone
    assert normalize_key(numlock | 0xFFE3) is None    # Control_L alone


def test_keypad_without_numlock_maps_to_navigation():
    assert normalize_key(65430) == "left"     # KP_Left / KP_4, measured under Qt
    assert normalize_key(65429) == "home"
    assert normalize_key(65434) == "pgup"
    assert normalize_key(65439) == "del"
    assert normalize_key(0xFF8D) == "enter"


def test_cocoa_codes():
    assert normalize_key(63232) == "up"
    assert normalize_key(63233) == "down"
    assert normalize_key(63234) == "left"
    assert normalize_key(63235) == "right"
    assert normalize_key(0xF72C) == "pgup"
    assert normalize_key(0xF72D) == "pgdn"
    assert normalize_key(127) == "backspace"
    assert set(COCOA_KEYS.values()) >= {"home", "end", "del"}


def test_no_key_and_unknown_codes():
    assert normalize_key(-1) is None
    assert normalize_key(None) is None
    assert normalize_key(0) is None
    assert normalize_key(0x1234 << 16) is None
    assert normalize_key(5) is None                 # Ctrl+E on Win32: not bound


def test_punctuation():
    for ch in ",.[]+-=/?<>_;'":
        assert normalize_key(ord(ch)) == ch


def test_describe_code():
    assert "Win32" in describe_code(2424832) and "'left'" in describe_code(2424832)
    assert "not mapped" in describe_code(0x1234 << 16)
    assert "modifiers" in describe_code((1 << 16) | 97)


@pytest.mark.parametrize("bindings", [calib_state.BINDINGS, review_state.BINDINGS], ids=["calib", "review"])
def test_binding_tables_are_consistent(bindings):
    assert check_bindings(bindings) == []
    # every key name in a table is produced by some code
    reachable = set(QT_MEASURED) | set(WIN32_CODES.values()) | {normalize_key(c) for c in range(32, 256)}
    reachable |= {"backtab", "f1", "space"}
    for action, keys in bindings.items():
        for k in keys:
            assert k in reachable, (action, k)


@pytest.mark.parametrize("helps,bindings", [(calib_state.HELP_TEXT, calib_state.BINDINGS),
                                            (review_state.HELP_TEXT, review_state.BINDINGS)])
def test_every_action_has_help(helps, bindings):
    assert set(helps) == set(bindings)


def test_letter_twin_check_detects_missing_twin():
    assert check_letter_twins({"go": ["left"]}) == ["action 'go' has an arrow key but no letter twin"]
    assert check_letter_twins({"go": ["left", "a"]}) == []
    assert any("bound to both" in p for p in check_bindings({"x": ["a"], "y": ["a", "right"]}))


def test_resolve_and_label():
    b = {"next": ["right", "d"], "quit": ["q", "esc"]}
    assert resolve("d", b) == "next"
    assert resolve("esc", b) == "quit"
    assert resolve("z", b) is None
    assert resolve(None, b) is None
    assert key_label(["right", "d"]) == "Right/D"
    assert key_label(["pgdn", "]"]) == "PgDn/]"


# ------------------------------------------------------------ gui checks --
def test_gui_problem_no_display_on_linux():
    msg = gui.gui_problem(env={}, platform="linux", backend="QT5", maps="")
    assert msg is not None and "DISPLAY" in msg


def test_gui_problem_wayland_only():
    msg = gui.gui_problem(env={"WAYLAND_DISPLAY": "wayland-0"}, platform="linux", backend="QT5", maps="")
    assert msg is not None and "X11" in msg


def test_gui_problem_headless_opencv():
    msg = gui.gui_problem(env={"DISPLAY": ":0"}, platform="linux", backend="NONE")
    assert msg is not None and "opencv-python-headless" in msg
    assert "pip install opencv-python" in msg


def test_gui_ok_on_windows_and_with_display():
    assert gui.gui_problem(env={}, platform="win32", backend="WIN32UI") is None
    assert gui.gui_problem(env={}, platform="darwin", backend="COCOA") is None
    assert gui.gui_problem(env={"DISPLAY": ":99"}, platform="linux", backend="QT5", maps="") is None


def test_highgui_backend_reported():
    b = gui.highgui_backend()
    assert b is None or isinstance(b, str)


def test_require_gui_raises_without_display(monkeypatch):
    if not __import__("sys").platform.startswith("linux"):
        pytest.skip("DISPLAY check only applies on Linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(gui.GuiUnavailable):
        gui.require_gui()
    # and the window wrapper refuses before touching cv2.namedWindow (which would abort Qt)
    with pytest.raises(gui.GuiUnavailable):
        gui.Window("never")


def test_run_uis_refuse_without_display(monkeypatch):
    import sys
    if not sys.platform.startswith("linux"):
        pytest.skip("DISPLAY check only applies on Linux")
    from ui_helpers import FakeSource, Recorder, make_result
    from pitchspeed.calib.ui import run_calibration_ui
    from pitchspeed.review.ui import run_review_ui
    from pitchspeed.ui.keys import show_key_codes
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    src = FakeSource(n_frames=20)
    with pytest.raises(gui.GuiUnavailable):
        run_calibration_ui(src, src.info, "side")
    rec = Recorder()
    with pytest.raises(gui.GuiUnavailable):
        run_review_ui(make_result(src, windows=((2, 10),), missed=()), src, None, rec.recompute, rec.save)
    with pytest.raises(gui.GuiUnavailable):
        show_key_codes()


# ------------------------------------------- PyAV-before-OpenCV hang (Linux) --
MAPS_AV_FIRST = """\
7f00-7f01 r-xp 00000000 08:01 1 /usr/lib/x86_64-linux-gnu/libxcb.so.1.1.0
7f02-7f03 r-xp 00000000 08:01 2 /site/av.libs/libxcb-5ddf6756.so.1.1.0
7f04-7f05 r-xp 00000000 08:01 3 /site/av.libs/libxcb-shm-0be6dfbf.so.0.0.0
7f06-7f07 rw-p 00000000 00:00 0
"""
MAPS_CV2_FIRST = """\
7f00-7f01 r-xp 00000000 08:01 1 /usr/lib/x86_64-linux-gnu/libxcb.so.1.1.0
7f02-7f03 r-xp 00000000 08:01 2 /site/av.libs/libxcb-5ddf6756.so.1.1.0
7f04-7f05 r-xp 00000000 08:01 3 /site/opencv_python.libs/libxcb-shm-0be6dfbf.so.0.0.0
7f08-7f09 r-xp 00000000 08:01 4 /usr/lib/x86_64-linux-gnu/libxcb-shm.so.0.0.0
"""


def test_xcb_conflict_detection():
    assert gui.xcb_conflict(MAPS_AV_FIRST) == "/site/av.libs"
    assert gui.xcb_conflict(MAPS_CV2_FIRST) is None
    assert gui.xcb_conflict("") is None
    msg = gui.gui_problem(env={"DISPLAY": ":0"}, platform="linux", backend="QT5", maps=MAPS_AV_FIRST)
    assert "PyAV was loaded before OpenCV" in msg and "import cv2 before" in msg
    # not relevant on Windows / macOS
    assert gui.gui_problem(env={}, platform="win32", backend="WIN32UI", maps=MAPS_AV_FIRST) is None


def test_av_before_cv2_refuses_instead_of_hanging(tmp_path):
    """In a fresh process: av imported first -> clear GuiUnavailable, no window, no hang."""
    import subprocess
    import sys
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux only")
    code = ("import av, cv2, sys\n"
            "from pitchspeed.ui import gui\n"
            "try:\n"
            "    gui.require_gui()\n"
            "    print('NO-ERROR')\n"
            "except gui.GuiUnavailable as e:\n"
            "    print('REFUSED', 'PyAV' in str(e))\n")
    env = dict(__import__("os").environ, DISPLAY=":12345")       # never contacted
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, env=env)
    out = p.stdout.strip()
    probe = subprocess.run([sys.executable, "-c", "import av, cv2; from pitchspeed.ui.gui import xcb_conflict; "
                            "print(xcb_conflict())"], capture_output=True, text=True, timeout=60)
    if probe.stdout.strip() == "None":
        pytest.skip("installed av/opencv wheels do not share libxcb sonames here")
    assert out == "REFUSED True", (out, p.stderr[-2000:])
