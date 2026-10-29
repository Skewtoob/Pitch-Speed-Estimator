"""The Pitch Speed main window (PySide6 / Qt).

A list of tapes with their state; buttons run the matching `pitchspeed` command (jobs.py)
in a separate process and show its progress.  Settings (league, speed graphic, sound,
window size) are kept between sessions with QSettings.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

from PySide6.QtCore import QProcess, QProcessEnvironment, QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox,
                               QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGridLayout,
                               QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QRadioButton, QTableWidget,
                               QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

from pitchspeed import __version__
from pitchspeed import constants as C
from pitchspeed.app import jobs as J
from pitchspeed.app.tapes import tape_info

VIDEO_FILTER = "Videos (*.mp4 *.mpg *.mpeg *.avi *.mov *.mkv *.mts *.m2ts *.vob *.wmv *.dv *.ts);;All files (*)"
UI_SCALES = [("Normal", 1.0), ("Large", 1.25), ("Larger", 1.5), ("Largest", 2.0)]
ORG, APP = "PitchSpeed", "PitchSpeed"
STYLE_SHEET = """
QMainWindow, QDialog { background: #f4f5f7; }
QLabel#title { font-size: 22px; font-weight: 600; color: #1d2b4f; }
QLabel#subtitle { color: #5a6275; }
QPushButton { padding: 7px 14px; border-radius: 6px; border: 1px solid #c5cad6; background: white; }
QPushButton:hover { background: #eef1f7; }
QPushButton:disabled { color: #a0a5b0; background: #f0f1f3; }
QPushButton#primary { background: #c8202e; color: white; border: 1px solid #a51a26; font-weight: 600; }
QPushButton#primary:hover { background: #b01c28; }
QPushButton#primary:disabled { background: #e2a3a8; border-color: #e2a3a8; }
QTableWidget { background: white; border: 1px solid #d4d8e1; border-radius: 6px; gridline-color: #eceef2; }
QHeaderView::section { background: #eceff5; padding: 6px; border: none; font-weight: 600; color: #333a4d; }
QProgressBar { border: 1px solid #c5cad6; border-radius: 5px; background: white; height: 18px; text-align: center; }
QProgressBar::chunk { background: #c8202e; border-radius: 4px; }
"""


def resource(name: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    for p in (os.path.join(base, name), os.path.join(os.path.dirname(os.path.abspath(__file__)), name)):
        if os.path.exists(p):
            return p
    return ""


# ------------------------------------------------------------- settings --
def load_settings(qs: QSettings) -> J.Settings:
    s = J.Settings(level=str(qs.value("level", "hs")), style=str(qs.value("style", "radar")),
                   audio=str(qs.value("audio", "true")).lower() in ("true", "1"),
                   ui_scale=float(qs.value("ui_scale", 1.0)))
    return s.validated()


def save_settings(qs: QSettings, s: J.Settings) -> None:
    qs.setValue("level", s.level)
    qs.setValue("style", s.style)
    qs.setValue("audio", "true" if s.audio else "false")
    qs.setValue("ui_scale", s.ui_scale)


def style_preview(style: str, w: int = 300, h: int = 170) -> QPixmap:
    """A small picture of a speed graphic on a dark field background."""
    import numpy as np
    from types import SimpleNamespace as NS
    from pitchspeed.render import overlay as O
    from pitchspeed.types import SpeedEstimate
    W, H = 640, 360
    yy = np.linspace(0, 1, H)[:, None]
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (np.array([60, 110, 50]) * (0.6 + 0.4 * yy)[..., None]).astype(np.uint8)[:, :1, :]
    img[: H // 3] = (150, 120, 90)
    sp = SpeedEstimate(v_release_mph=78.4, sigma_mph=4.3, ci90_mph=(71.0, 85.0), label="fair", method="manual_tof")
    painter = O.PAINTERS[style]((W, H)) if style in O.PAINTERS else O.Painter((W, H))
    blank = img.copy()
    try:
        painter.speed_card(img, 0, NS(number=5, speed=sp, edited=False, status="manual"), 1.0, None)
    except Exception:                                    # noqa: BLE001 - a preview must never crash
        pass
    ys, xs = np.nonzero(np.abs(img.astype(int) - blank.astype(int)).sum(axis=2) > 0)
    if len(xs):                                          # crop in on the graphic, at the dialog's aspect
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        cw = max(xs.max() - xs.min() + 40, (ys.max() - ys.min() + 40) * w / h, 200)
        ch = cw * h / w
        x0 = int(min(max(0, cx - cw / 2), W - cw)); y0 = int(min(max(0, cy - ch / 2), H - ch))
        img = np.ascontiguousarray(img[y0:y0 + int(ch), x0:x0 + int(cw)])
    hh, ww = img.shape[:2]
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    q = QImage(rgb.data, ww, hh, 3 * ww, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(q).scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation)


class SettingsDialog(QDialog):
    def __init__(self, s: J.Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.level = QComboBox()
        for key, lv in C.LEVELS.items():
            self.level.addItem(lv.label, key)
        self.level.setCurrentIndex(max(0, self.level.findData(s.level)))
        form.addRow("Pitching distance:", self.level)
        self.scale = QComboBox()
        for name, v in UI_SCALES:
            self.scale.addItem(name, v)
        self.scale.setCurrentIndex(min(range(len(UI_SCALES)), key=lambda i: abs(UI_SCALES[i][1] - s.ui_scale)))
        form.addRow("Marking window size:", self.scale)
        self.audio = QCheckBox("Keep the sound in the videos it makes")
        self.audio.setChecked(s.audio)
        form.addRow("", self.audio)
        lay.addLayout(form)
        lay.addWidget(QLabel("<b>Speed graphic</b> in the videos it makes:"))
        grid = QGridLayout()
        self.styles = QButtonGroup(self)
        for i, st in enumerate(J.STYLES):
            box = QVBoxLayout()
            rb = QRadioButton(J.STYLE_NAMES[st])
            rb.setProperty("style_key", st)
            rb.setChecked(st == s.style)
            self.styles.addButton(rb, i)
            pic = QLabel()
            pic.setPixmap(style_preview(st))
            box.addWidget(pic)
            box.addWidget(rb)
            w = QWidget()
            w.setLayout(box)
            grid.addWidget(w, i // 2, i % 2)
        lay.addLayout(grid)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def result_settings(self) -> J.Settings:
        b = self.styles.checkedButton()
        return J.Settings(level=self.level.currentData(), style=b.property("style_key") if b else "radar",
                          audio=self.audio.isChecked(), ui_scale=float(self.scale.currentData())).validated()


class AutoDialog(QDialog):
    """Camera position for the automatic search."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Find pitches automatically")
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Automatic works best with a camera on a tripod that doesn't zoom.\n"
                             "You click a few points on the picture once (again if the camera moves).\n"
                             "If the camera moved or zoomed a lot, 'Mark pitches by hand' is more reliable."))
        lay.addWidget(QLabel("<b>Where was the camera?</b>"))
        self.side = QRadioButton("From the side (dugout, fence, along a baseline)")
        self.behind = QRadioButton("Behind home plate / the backstop, looking out at the pitcher")
        self.behind.setChecked(True)
        lay.addWidget(self.side)
        lay.addWidget(self.behind)
        self.throws = QComboBox()
        self.throws.addItems(["Pitcher throws: not sure", "Right-handed", "Left-handed"])
        lay.addWidget(self.throws)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def choice(self) -> tuple:
        view = "side" if self.side.isChecked() else "behind"
        return view, {1: "R", 2: "L"}.get(self.throws.currentIndex())


# ------------------------------------------------------------ job runner --
class JobDialog(QDialog):
    """Runs a Job's steps one after the other and shows their progress."""

    def __init__(self, job: J.Job, parent=None, command: Optional[list] = None):
        super().__init__(parent)
        self.job, self.cmd = job, command or J.cli_command()
        self.i = -1
        self.ok = False
        self.buf = ""
        self.setWindowTitle(f"{job.name} - {os.path.basename(job.video)}")
        self.resize(620, 420)
        lay = QVBoxLayout(self)
        self.step = QLabel()
        self.step.setWordWrap(True)
        f = QFont()
        f.setPointSize(f.pointSize() + 2)
        self.step.setFont(f)
        lay.addWidget(self.step)
        self.bar = QProgressBar()
        lay.addWidget(self.bar)
        self.eta = QLabel(" ")
        self.eta.setObjectName("subtitle")
        lay.addWidget(self.eta)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(4000)
        lay.addWidget(self.log, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        self.stop = QPushButton("Stop")
        self.stop.clicked.connect(self.cancel)
        self.close_btn = QPushButton("Close")
        self.close_btn.setEnabled(False)
        self.close_btn.clicked.connect(self.accept)
        row.addWidget(self.stop)
        row.addWidget(self.close_btn)
        lay.addLayout(row)
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        self.proc.setProcessEnvironment(env)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._finished)
        QTimer.singleShot(0, self._next)

    def _next(self) -> None:
        self.i += 1
        if self.i >= len(self.job.steps):
            self._done(True, "Done.")
            return
        st = self.job.steps[self.i]
        n = len(self.job.steps)
        self.step.setText(f"Step {self.i + 1} of {n}: {st.title}")
        self.bar.setRange(0, 0)                      # busy until the program reports a percentage
        self.eta.setText("A separate window opens for this step; it may take a minute to appear "
                         "while the video is read." if st.opens_window else " ")
        self.log.appendPlainText(f"\n> pitchspeed {' '.join(st.args[:1] + ['...'])}")
        self.proc.start(self.cmd[0], self.cmd[1:] + list(st.args))
        self.proc.waitForStarted(5000)
        self.proc.closeWriteChannel()                # no console: questions take their default answer

    def _read(self) -> None:
        self.buf += bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        lines, self.buf = J.split_output(self.buf)
        for line in lines:
            if not self._progress(line) and line.strip():
                self.log.appendPlainText(line)
        self._progress(self.buf)                 # the newest progress line has no line end yet

    def _progress(self, line: str) -> bool:
        pr = J.parse_progress(line)
        if pr is None:
            return False
        label, pct, eta = pr
        self.bar.setRange(0, 1000)
        self.bar.setValue(int(pct * 10))
        self.bar.setFormat(f"{label}: {pct:.0f}%")
        self.eta.setText(f"About {eta // 60} min {eta % 60} s left" if eta >= 60 else f"About {eta} s left")
        return True

    def _finished(self, code, _status) -> None:
        self._read()
        if self.buf.strip() and J.parse_progress(self.buf) is None:
            self.log.appendPlainText(self.buf)
        self.buf = ""
        if code != 0:
            self._done(False, "Stopped with a problem: see the messages below.")
            return
        self._next()

    def _done(self, ok: bool, text: str) -> None:
        self.ok = ok
        self.step.setText(text)
        self.bar.setRange(0, 1000)
        self.bar.setValue(1000 if ok else self.bar.value())
        self.bar.setFormat("Done" if ok else "Stopped")
        self.eta.setText(" ")
        self.stop.setEnabled(False)
        self.close_btn.setEnabled(True)

    def cancel(self) -> None:
        if self.proc.state() != QProcess.ProcessState.NotRunning:
            self.proc.kill()
            self.proc.waitForFinished(3000)
        self.i = len(self.job.steps)
        self._done(False, "Stopped. Everything marked so far is saved.")

    def closeEvent(self, e) -> None:
        if self.proc.state() != QProcess.ProcessState.NotRunning:
            if QMessageBox.question(self, "Stop?", "This is still running. Stop it?") != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.cancel()
        super().closeEvent(e)


# ----------------------------------------------------------- main window --
class MainWindow(QMainWindow):
    COLS = ["Tape", "Status", "Pitches", "Typical", "Fastest"]

    def __init__(self, qsettings: Optional[QSettings] = None):
        super().__init__()
        self.qs = qsettings or QSettings(ORG, APP)
        self.settings = load_settings(self.qs)
        self.setWindowTitle("Pitch Speed")
        icon = resource("icon.png")
        if icon:
            self.setWindowIcon(QIcon(icon))
        self.resize(900, 560)
        self.setAcceptDrops(True)
        root = QWidget()
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        head = QHBoxLayout()
        titles = QVBoxLayout()
        t = QLabel("Pitch Speed")
        t.setObjectName("title")
        sub = QLabel("Estimate pitch speeds from your videos. Add a tape, then mark its pitches.")
        sub.setObjectName("subtitle")
        titles.addWidget(t)
        titles.addWidget(sub)
        head.addLayout(titles, 1)
        add = QPushButton("Add tapes...")
        add.setObjectName("primary")
        add.clicked.connect(self.add_dialog)
        sett = QPushButton("Settings")
        sett.clicked.connect(self.edit_settings)
        hlp = QPushButton("Help")
        hlp.clicked.connect(self.show_help)
        for b in (add, sett, hlp):
            head.addWidget(b)
        lay.addLayout(head)
        body = QHBoxLayout()
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, len(self.COLS)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.table.doubleClicked.connect(lambda _i: self.run_mark())
        body.addWidget(self.table, 1)
        side = QVBoxLayout()
        self.b_mark = QPushButton("Mark pitches by hand")
        self.b_mark.setObjectName("primary")
        self.b_mark.setToolTip("Any camera angle: you mark each pitch's release and catch.")
        self.b_mark.clicked.connect(self.run_mark)
        self.b_auto = QPushButton("Find pitches automatically...")
        self.b_auto.clicked.connect(self.run_auto)
        self.b_review = QPushButton("Check / fix pitches")
        self.b_review.clicked.connect(self.run_review)
        self.b_clips = QPushButton("Make clips (one per pitch)")
        self.b_clips.clicked.connect(lambda: self.run_video("clips"))
        self.b_full = QPushButton("Make full video")
        self.b_full.clicked.connect(lambda: self.run_video("full"))
        self.b_folder = QPushButton("Open results folder")
        self.b_folder.clicked.connect(self.open_folder)
        self.b_remove = QPushButton("Remove from list")
        self.b_remove.clicked.connect(self.remove_selected)
        self.style_label = QLabel()
        self.style_label.setObjectName("subtitle")
        self.style_label.setWordWrap(True)
        for b in (self.b_mark, self.b_auto, self.b_review, self.b_clips, self.b_full, self.b_folder):
            side.addWidget(b)
        side.addStretch(1)
        side.addWidget(self.style_label)
        side.addWidget(self.b_remove)
        body.addLayout(side)
        lay.addLayout(body, 1)
        self.hint = QLabel("Tip: you can also drag video files onto this window.")
        self.hint.setObjectName("subtitle")
        lay.addWidget(self.hint)
        self.videos: list[str] = [v for v in json.loads(str(self.qs.value("tapes", "[]") or "[]")) if v]
        self.refresh()

    # tapes
    def _save_list(self) -> None:
        self.qs.setValue("tapes", json.dumps(self.videos))

    def add_videos(self, paths) -> int:
        n = 0
        for p in paths:
            p = os.path.abspath(p)
            if os.path.isfile(p) and p not in self.videos:
                self.videos.append(p)
                n += 1
        self._save_list()
        self.refresh(select=self.videos[-1] if n else None)
        return n

    def add_dialog(self) -> None:
        start = str(self.qs.value("last_dir", os.path.expanduser("~")))
        files, _ = QFileDialog.getOpenFileNames(self, "Add tapes", start, VIDEO_FILTER)
        if files:
            self.qs.setValue("last_dir", os.path.dirname(files[0]))
            self.add_videos(files)

    def remove_selected(self) -> None:
        v = self.selected()
        if v and QMessageBox.question(self, "Remove", f"Remove {os.path.basename(v)} from the list?\n"
                                      "Its video and results stay on your computer.") == QMessageBox.StandardButton.Yes:
            self.videos.remove(v)
            self._save_list()
            self.refresh()

    def selected(self) -> Optional[str]:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        it = self.table.item(rows[0].row(), 0)
        return it.data(Qt.ItemDataRole.UserRole) if it else None

    def refresh(self, select: Optional[str] = None) -> None:
        keep = select or self.selected()
        self.table.setRowCount(0)
        for v in self.videos:
            ti = tape_info(v)
            r = self.table.rowCount()
            self.table.insertRow(r)
            name = QTableWidgetItem(ti.name)
            name.setData(Qt.ItemDataRole.UserRole, v)
            name.setToolTip(v)
            cells = [name, QTableWidgetItem(ti.status), QTableWidgetItem(str(ti.pitches) if ti.pitches else "-"),
                     QTableWidgetItem(f"{ti.typical:.0f} mph" if ti.typical else "-"),
                     QTableWidgetItem(f"{ti.fastest:.0f} mph" if ti.fastest else "-")]
            for c, it in enumerate(cells):
                self.table.setItem(r, c, it)
            if v == keep:
                self.table.selectRow(r)
        if self.videos and not self.selected():
            self.table.selectRow(0)
        self.style_label.setText(f"Speed graphic: {J.STYLE_NAMES[self.settings.style]}\n"
                                 f"Distance: {C.LEVELS[self.settings.level].label}")
        self._update_buttons()

    def _update_buttons(self) -> None:
        v = self.selected()
        ti = tape_info(v) if v else None
        has = ti is not None and ti.exists
        done = has and ti.pitches > 0
        for b in (self.b_mark, self.b_auto, self.b_folder, self.b_remove):
            b.setEnabled(v is not None)
        self.b_mark.setEnabled(has)
        self.b_auto.setEnabled(has)
        for b in (self.b_review, self.b_clips, self.b_full):
            b.setEnabled(bool(done))
        self.hint.setText("Tip: you can also drag video files onto this window." if self.videos else
                          "Click 'Add tapes...' (or drag video files here) to start.")

    # actions
    def _run(self, job: J.Job) -> bool:
        dlg = JobDialog(job, self)
        dlg.exec()
        self.refresh(select=job.video)
        return dlg.ok

    def run_mark(self) -> None:
        v = self.selected()
        if v and os.path.isfile(v):
            self._run(J.mark_job(v, self.settings))

    def run_auto(self) -> None:
        v = self.selected()
        if not (v and os.path.isfile(v)):
            return
        d = AutoDialog(self)
        if d.exec() == QDialog.DialogCode.Accepted:
            view, throws = d.choice()
            self._run(J.auto_job(v, view, self.settings, throws))

    def run_review(self) -> None:
        v = self.selected()
        if v:
            self._run(J.review_job(v, self.settings))

    def run_video(self, mode: str) -> None:
        v = self.selected()
        if v:
            self._run(J.video_job(v, mode, self.settings))

    def open_folder(self) -> None:
        v = self.selected()
        if not v:
            return
        ti = tape_info(v)
        target = ti.out_dir if os.path.isdir(ti.out_dir) else os.path.dirname(v)
        QDesktopServices.openUrl(QUrl.fromLocalFile(target))

    def edit_settings(self) -> None:
        d = SettingsDialog(self.settings, self)
        if d.exec() == QDialog.DialogCode.Accepted:
            self.settings = d.result_settings()
            save_settings(self.qs, self.settings)
            self.refresh()

    def show_help(self) -> None:
        QMessageBox.information(self, "Pitch Speed", HELP_TEXT.format(version=__version__))

    # drag and drop
    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        self.add_videos([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])


HELP_TEXT = """Pitch Speed {version}

1. Add a tape (button, or drag video files onto the window).
2. Choose the pitching distance and the speed graphic under Settings.
3. Mark pitches by hand (any camera angle):
   - Space plays and pauses, + / - change speed, J / L jump 1 second.
   - At each pitch, pause and step with A / D to the first picture with the
     ball out of the hand, press N; then the first picture with the ball in
     the glove, press C (B if the batter hit it).
   - G shows the pictures side by side to pick the exact one.
   - Q saves and closes the window; the clips are then made.
4. Results (results.csv, clips) are in the folder next to each video
   ("Open results folder").

On 30 frames-per-second video, one pitch is only measurable to about
+/- 5 mph; averages over many pitches are much more precise.
Full instructions: README.md (in the program folder and on GitHub)."""


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = QApplication(argv)
    app.setApplicationName("Pitch Speed")
    app.setOrganizationName(ORG)
    app.setStyleSheet(STYLE_SHEET)
    if "--selftest" in argv:                     # build check: the window can be made, then quit
        w = MainWindow(QSettings(os.path.join(os.path.expanduser("~"), ".pitchspeed-selftest.ini"),
                                 QSettings.Format.IniFormat))
        d = SettingsDialog(w.settings, w)
        print("selftest ok", flush=True)
        d.deleteLater()
        w.deleteLater()
        QTimer.singleShot(0, app.quit)
        app.exec()                                 # let Qt delete them before the app goes away
        return 0
    w = MainWindow()
    files = [a for a in argv[1:] if os.path.isfile(a)]
    if files:
        w.add_videos(files)
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
