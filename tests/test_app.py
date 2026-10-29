"""The desktop program (pitchspeed.app): commands per button, progress parsing, tape status,
and the Qt window itself (offscreen)."""
import os
import sys

import pytest

from pitchspeed.app import jobs as J
from pitchspeed.app.tapes import tape_info


def test_buttons_run_commands_that_ask_no_questions():
    s = J.Settings(level="50", style="bar", audio=False, ui_scale=1.5)
    m = J.mark_job("t.avi", s)
    assert m.steps[0].args == ["mark", "t.avi", "--level", "50", "--catch-cue", "arrival", "--render", "none",
                               "--ui-scale", "1.5"]
    assert m.steps[1].args == ["render", "t.avi", "--mode", "clips", "--style", "bar", "--no-audio"]
    a = J.auto_job("t.avi", "side", s, "L")
    assert a.steps[0].args[:6] == ["calibrate", "t.avi", "--view", "side", "--level", "50"]
    assert "--throws" in a.steps[0].args and [st.args[0] for st in a.steps] == ["calibrate", "analyze", "review", "render"]
    assert J.video_job("t.avi", "full", J.Settings()).steps[0].args == ["render", "t.avi", "--mode", "full",
                                                                       "--style", "radar"]
    assert J.Settings(level="zzz", style="neon", ui_scale=9, catch_cue="x").validated() == \
        J.Settings("hs", "radar", True, 3.0, "arrival")
    assert "--catch-cue" in J.mark_job("t.avi", J.Settings(catch_cue="squeeze")).steps[0].args
    with pytest.raises(ValueError):
        J.auto_job("t.avi", "top", s)


def test_every_command_is_accepted_by_the_command_line():
    from pitchspeed.cli import build_parser
    ap = build_parser()
    s = J.Settings(level="ll", style="broadcast", audio=False, ui_scale=1.25)
    for job in (J.mark_job("t.avi", s), J.auto_job("t.avi", "behind", s), J.auto_job("t.avi", "side", s, "R"),
                J.review_job("t.avi", s), J.video_job("t.avi", "highlights", s)):
        for st in job.steps:
            ap.parse_args(st.args)              # raises SystemExit on an unknown flag or choice


def test_progress_lines_are_read():
    assert J.parse_progress("  clips:  45.0%  (120/266)  elapsed    10s  eta    12s") == ("clips", 45.0, 12)
    assert J.parse_progress("Writing one clip per pitch ...") is None
    lines, rest = J.split_output("a\r  clips: 5.0%  eta 3s\rb\nc")
    assert lines == ["a", "  clips: 5.0%  eta 3s", "b"] and rest == "c"


def test_tape_status(tmp_path):
    v = tmp_path / "game.avi"
    assert tape_info(str(v)).status == "File not found"
    v.write_bytes(b"x")
    ti = tape_info(str(v))
    assert ti.exists and ti.status == "Not started" and ti.pitches == 0 and ti.name == "game.avi"


# ------------------------------------------------------------------ Qt window
@pytest.fixture(scope="module")
def qapp():
    qt = pytest.importorskip("PySide6.QtWidgets")
    old = os.environ.get("QT_QPA_PLATFORM")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"     # only while Qt starts: other tests use a display
    try:
        return qt.QApplication.instance() or qt.QApplication([])
    except Exception as e:                     # noqa: BLE001 - no Qt platform in this environment
        pytest.skip(f"Qt cannot start: {e}")
    finally:
        if old is None:
            os.environ.pop("QT_QPA_PLATFORM", None)
        else:
            os.environ["QT_QPA_PLATFORM"] = old


def _settings(tmp_path):
    from PySide6.QtCore import QSettings
    qs = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    qs.clear()
    return qs


def test_window_lists_tapes_and_remembers_them_and_the_settings(qapp, tmp_path):
    from pitchspeed.app.main import MainWindow, SettingsDialog, save_settings
    v = tmp_path / "tape1.mpg"
    v.write_bytes(b"x")
    qs = _settings(tmp_path)
    w = MainWindow(qs)
    assert w.table.rowCount() == 0 and not w.b_mark.isEnabled()
    assert w.add_videos([str(v), str(v), str(tmp_path / "missing.avi")]) == 1
    assert w.table.rowCount() == 1 and w.selected() == str(v) and w.b_mark.isEnabled()
    assert not w.b_clips.isEnabled()                          # nothing marked yet
    d = SettingsDialog(w.settings)
    d.level.setCurrentIndex(d.level.findData("54"))
    d.styles.button(1).setChecked(True)
    s = d.result_settings()
    assert s.level == "54" and s.style == "broadcast"
    save_settings(qs, s)
    w2 = MainWindow(qs)
    assert w2.videos == [str(v)] and w2.settings.style == "broadcast" and w2.settings.level == "54"


def test_job_dialog_runs_steps_and_shows_progress(qapp, tmp_path):
    from PySide6.QtCore import QEventLoop, QTimer
    from pitchspeed.app.main import JobDialog
    script = tmp_path / "fake.py"
    script.write_text("import sys,time\nprint('step', sys.argv[1])\n"
                      "print('\\r  clips:  50.0%  (1/2)  elapsed 1s  eta 7s', end='', flush=True)\n"
                      "time.sleep(0.2)\nsys.exit(int(sys.argv[2]))\n")
    job = J.Job("Test", "t.avi", [J.Step("one", ["a", "0"]), J.Step("two", ["b", "0"])])
    d = JobDialog(job, command=[sys.executable, str(script)])
    seen = []
    t = QTimer()
    t.timeout.connect(lambda: seen.append((d.bar.maximum(), d.bar.value())))
    t.start(50)
    loop = QEventLoop()
    QTimer.singleShot(8000, loop.quit)
    d.close_btn.setEnabled(False)
    poll = QTimer()
    poll.timeout.connect(lambda: loop.quit() if d.close_btn.isEnabled() else None)
    poll.start(50)
    loop.exec()
    assert d.ok and "step a" in d.log.toPlainText() and "step b" in d.log.toPlainText()
    assert any(mx == 1000 and val == 500 for mx, val in seen)  # the 50 % line moved the bar
    bad = JobDialog(J.Job("Bad", "t.avi", [J.Step("one", ["a", "3"]), J.Step("two", ["b", "0"])]),
                    command=[sys.executable, str(script)])
    loop2 = QEventLoop()
    QTimer.singleShot(8000, loop2.quit)
    p2 = QTimer()
    p2.timeout.connect(lambda: loop2.quit() if bad.close_btn.isEnabled() else None)
    p2.start(50)
    loop2.exec()
    assert not bad.ok and "step b" not in bad.log.toPlainText()     # stops at the failed step


def test_fastball_average_takes_the_fast_group():
    from pitchspeed.app.tapes import fastball_average
    slow = [51, 58, 61, 61, 61, 61, 64, 64, 64, 64, 64, 64, 68, 68]
    fast = [72] * 21 + [76] * 15 + [81] * 5 + [87]
    m, pm, n = fastball_average(slow + fast)
    assert n == 42 and abs(m - 74.86) < 0.1 and 0.5 < pm < 1.5
    m, pm, n = fastball_average([80, 81, 79, 82])            # one group: all of them
    assert n == 4 and abs(m - 80.5) < 1e-9
    assert fastball_average([]) == (None, None, 0) and fastball_average([80])[1] is None
