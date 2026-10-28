"""Portable key names for ``cv2.waitKeyEx`` codes.

``cv2.waitKey() & 0xFF`` turns every arrow key into 0, so the UIs call
``waitKeyEx`` and map the raw code to a small set of names here:

    'left' 'right' 'up' 'down' 'pgup' 'pgdn' 'home' 'end' 'insert' 'del'
    'backspace' 'enter' 'esc' 'tab' 'backtab' 'space' 'f1'..'f12'
    'a'..'z' (always lower case) '0'..'9' and punctuation as the character
    itself (',' '.' '[' ']' '+' '-' '=' '/' '?' ...)

Where the codes come from (one table per OpenCV highgui backend):

* **Win32** (the opencv-python wheels on Windows): special keys arrive as
  ``VK_code << 16`` from WM_KEYDOWN, e.g. left = 0x25 << 16 = 2424832;
  printable keys, Enter (13), Backspace (8), Tab (9) and Esc (27) arrive
  as their character from WM_CHAR.  Taken from OpenCV's window_w32.cpp;
  NOT run on Windows (``pitchspeed keys`` prints the real codes).
* **Qt** (the opencv-python wheels on Linux; measured under Xvfb with XTEST,
  see tests/test_ui_keys.py): printable keys as Latin-1 characters, letters
  always lower case even with Shift; Enter 13, Tab 9, Shift+Tab 11, Esc 27,
  Backspace 8; other keys as X11 keysyms (left 65361 ... delete 65535).
* **GTK** (self-built OpenCV on Linux): X11 keysyms with the modifier state
  in the upper 16 bits (``code |= state << 16``), stripped here.
* **Cocoa** (macOS wheels): NSEvent function-key characters 0xF700...
  (up 63232); Backspace is DEL (127).  From OpenCV's window_cocoa.mm, not
  verified.

Letters are normalized to lower case on every platform (Qt cannot report
Shift for letters), so no binding may depend on the case of a letter.

Every UI binds each arrow key together with a letter twin (A/D, W/S) so
that keyboards or backends that deliver strange arrow codes still work;
``check_letter_twins`` verifies a binding table.  ``show_key_codes()``
(``pitchspeed keys``) opens a small window and prints raw codes and their
names, for troubleshooting on a new machine.
"""
from __future__ import annotations

import sys
from typing import Callable, Iterable, Mapping, Optional

# ----------------------------------------------------------------- tables --
#: Win32 virtual-key codes (delivered as vk << 16 by OpenCV's Win32 backend).
WIN32_VK: dict[int, str] = {
    0x21: "pgup",     # VK_PRIOR
    0x22: "pgdn",     # VK_NEXT
    0x23: "end",
    0x24: "home",
    0x25: "left",
    0x26: "up",
    0x27: "right",
    0x28: "down",
    0x2D: "insert",
    0x2E: "del",
    **{0x70 + i: f"f{i + 1}" for i in range(12)},   # VK_F1 .. VK_F12
}
#: Full Win32 codes as returned by waitKeyEx (e.g. 2424832 = left).
WIN32_CODES: dict[int, str] = {vk << 16: name for vk, name in WIN32_VK.items()}

#: X11 keysyms (Qt's nativeVirtualKey on X11, GTK keyvals).
X11_KEYSYMS: dict[int, str] = {
    0xFF08: "backspace", 0xFF09: "tab", 0xFF0D: "enter", 0xFF1B: "esc",
    0xFE20: "backtab",   # ISO_Left_Tab (Shift+Tab)
    0xFF50: "home", 0xFF51: "left", 0xFF52: "up", 0xFF53: "right", 0xFF54: "down",
    0xFF55: "pgup", 0xFF56: "pgdn", 0xFF57: "end", 0xFF63: "insert", 0xFFFF: "del",
    # keypad with NumLock off
    0xFF8D: "enter", 0xFF95: "home", 0xFF96: "left", 0xFF97: "up", 0xFF98: "right",
    0xFF99: "down", 0xFF9A: "pgup", 0xFF9B: "pgdn", 0xFF9C: "end", 0xFF9E: "insert",
    0xFF9F: "del",
    # keypad operators / digits (GTK reports keypad keysyms, Qt the character)
    0xFFAA: "*", 0xFFAB: "+", 0xFFAD: "-", 0xFFAE: ".", 0xFFAF: "/",
    **{0xFFB0 + i: str(i) for i in range(10)},
    **{0xFFBE + i: f"f{i + 1}" for i in range(12)},   # F1 .. F12
}
#: X11 modifier keysyms: pressing Shift/Ctrl/Alt alone is not a key press.
X11_MODIFIERS = frozenset(range(0xFFE1, 0xFFEF)) | {0xFF7F, 0xFF14, 0xFE03}

#: macOS Cocoa NSEvent function-key characters (unverified).
COCOA_KEYS: dict[int, str] = {
    0xF700: "up", 0xF701: "down", 0xF702: "left", 0xF703: "right",
    **{0xF704 + i: f"f{i + 1}" for i in range(12)},
    0xF727: "insert", 0xF728: "del", 0xF729: "home", 0xF72B: "end",
    0xF72C: "pgup", 0xF72D: "pgdn",
}

#: Control characters with an unambiguous meaning on every backend.
CONTROL_CHARS: dict[int, str] = {
    8: "backspace", 9: "tab", 10: "enter", 11: "backtab", 13: "enter", 27: "esc",
    32: "space", 127: "backspace",
}

#: Codes measured with opencv-python 5.0.0.93 (Qt 5.15) under Xvfb + XTEST.
QT_MEASURED: dict[str, int] = {
    "left": 65361, "up": 65362, "right": 65363, "down": 65364, "home": 65360, "end": 65367,
    "pgup": 65365, "pgdn": 65366, "del": 65535, "insert": 65379, "backspace": 8, "enter": 13,
    "tab": 9, "backtab": 11, "esc": 27, "space": 32, "f1": 65470, "f12": 65481,
    "a": 97, "z": 122, "0": 48, "9": 57, ",": 44, ".": 46, "[": 91, "]": 93, "=": 61, "+": 43,
    "-": 45, "/": 47, "<": 60, ">": 62,
}

ARROWS = ("left", "right", "up", "down")


# ---------------------------------------------------------- normalization --
def normalize_key(code: Optional[int]) -> Optional[str]:
    """Portable name of a ``cv2.waitKeyEx`` code, or None for no/unknown key.

    -1 (timeout), lone modifier keys and unknown codes give None.
    """
    if code is None:
        return None
    code = int(code)
    if code < 0:
        return None
    low, high = code & 0xFFFF, code >> 16
    if low == 0:
        # Win32 special keys: VK << 16 with nothing in the low bits.
        return WIN32_VK.get(high)
    # GTK puts the modifier state into the high bits: ignore it.
    if low in X11_MODIFIERS:
        return None
    if low in X11_KEYSYMS:
        return X11_KEYSYMS[low]
    if low in COCOA_KEYS:
        return COCOA_KEYS[low]
    if low in CONTROL_CHARS:
        return CONTROL_CHARS[low]
    if 33 <= low <= 126 or 161 <= low <= 255:
        return chr(low).lower()
    return None


def describe_code(code: int) -> str:
    """Human readable description of a raw code (for ``pitchspeed keys``)."""
    name = normalize_key(code)
    low, high = code & 0xFFFF, code >> 16
    parts = [f"raw {code:>8d}", f"hex 0x{code:08X}"]
    if low == 0 and high:
        parts.append(f"(Win32 VK 0x{high:02X} << 16)")
    elif high:
        parts.append(f"(low 0x{low:04X}, high bits 0x{high:04X} = modifiers?)")
    parts.append(f"-> {name!r}" if name else "-> (not mapped)")
    return "  ".join(parts)


# --------------------------------------------------------------- bindings --
def resolve(name: Optional[str], bindings: Mapping[str, Iterable[str]]) -> Optional[str]:
    """Action bound to key `name` in ``{action: [key names]}``, or None."""
    if name is None:
        return None
    for action, keys in bindings.items():
        if name in keys:
            return action
    return None


def check_bindings(bindings: Mapping[str, Iterable[str]]) -> list[str]:
    """Problems with a binding table: keys bound twice, arrows without a letter twin."""
    problems = []
    seen: dict[str, str] = {}
    for action, keys in bindings.items():
        for k in keys:
            if k in seen and seen[k] != action:
                problems.append(f"key {k!r} bound to both {seen[k]!r} and {action!r}")
            seen[k] = action
            if k != k.lower():
                problems.append(f"key {k!r} of {action!r} is upper case (letters are normalized to lower)")
    problems += check_letter_twins(bindings)
    return problems


def check_letter_twins(bindings: Mapping[str, Iterable[str]]) -> list[str]:
    """Every action bound to an arrow key must also be bound to a letter."""
    out = []
    for action, keys in bindings.items():
        keys = list(keys)
        if any(k in ARROWS for k in keys) and not any(len(k) == 1 and k.isalpha() for k in keys):
            out.append(f"action {action!r} has an arrow key but no letter twin")
    return out


def key_label(keys: Iterable[str]) -> str:
    """Short label for a list of key names, for help screens ('Left/A')."""
    pretty = {"left": "Left", "right": "Right", "up": "Up", "down": "Down", "pgup": "PgUp",
              "pgdn": "PgDn", "home": "Home", "end": "End", "del": "Del", "backspace": "Bksp",
              "enter": "Enter", "esc": "Esc", "tab": "Tab", "backtab": "Shift+Tab",
              "space": "Space", "insert": "Ins"}
    return "/".join(pretty.get(k, k.upper() if len(k) == 1 else k) for k in keys)


# ---------------------------------------------------------- show_key_codes --
def show_key_codes(max_keys: Optional[int] = None, timeout_s: Optional[float] = None,
                   out: Callable[[str], None] = print) -> list[tuple[int, Optional[str]]]:
    """Open a small window and print the raw ``waitKeyEx`` code of every key.

    For troubleshooting key bindings on a new machine (``pitchspeed keys``).
    Ends on Esc pressed twice in a row, when the window is closed, after
    `max_keys` keys or after `timeout_s` seconds.  Returns the (code, name)
    pairs seen.
    """
    import time

    import numpy as np

    from pitchspeed.ui.gui import Window, require_gui
    from pitchspeed.ui.canvas import draw_text

    require_gui()
    seen: list[tuple[int, Optional[str]]] = []
    img = np.zeros((150, 480, 3), np.uint8)
    lines = ["pitchspeed keys: press keys in THIS window", "codes are printed in the console",
             "Esc twice (or close the window) to quit"]
    for i, t in enumerate(lines):
        draw_text(img, t, (10, 30 + 30 * i), scale=0.55)
    out("Press keys in the 'pitchspeed keys' window; Esc twice or close the window to quit.")
    out(f"platform: {sys.platform}")
    t0 = time.monotonic()
    with Window("pitchspeed keys") as win:
        win.show(img)
        last = None
        while win.is_open():
            if timeout_s is not None and time.monotonic() - t0 > timeout_s:
                break
            code = win.wait_key(50)
            if code == -1:
                continue
            name = normalize_key(code)
            seen.append((code, name))
            out(describe_code(code))
            if name == "esc" and last == "esc":
                break
            last = name
            if max_keys is not None and len(seen) >= max_keys:
                break
    return seen
