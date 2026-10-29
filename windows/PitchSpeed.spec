# PyInstaller recipe for the Windows program:  pyinstaller windows\PitchSpeed.spec
# Output: dist\PitchSpeed\ with PitchSpeed.exe (the window) and pitchspeed-cli.exe (the
# command line it runs in the background), sharing one _internal folder.
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hidden = [], [], collect_submodules("pitchspeed")
for pkg in ("pitchspeed", "av", "cv2", "imageio_ffmpeg"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hidden += h

common = dict(pathex=["../src"], binaries=binaries, datas=datas, hiddenimports=hidden,
              excludes=["tkinter", "matplotlib", "pytest"])
gui = Analysis(["gui.py"], **common)
cli = Analysis(["app.py"], **common)
gui_exe = EXE(PYZ(gui.pure), gui.scripts, [], exclude_binaries=True, name="PitchSpeed", console=False,
              icon="icon.ico")
cli_exe = EXE(PYZ(cli.pure), cli.scripts, [], exclude_binaries=True, name="pitchspeed-cli", console=True,
              icon="icon.ico")
coll = COLLECT(gui_exe, gui.binaries, gui.datas, cli_exe, cli.binaries, cli.datas, name="PitchSpeed")
