# PyInstaller recipe for the stand-alone Windows program:  pyinstaller windows\PitchSpeed.spec
# Output: dist\PitchSpeed\PitchSpeed.exe (one folder; zip the whole folder to share it).
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hidden = [], [], collect_submodules("pitchspeed")
for pkg in ("pitchspeed", "av", "cv2", "imageio_ffmpeg"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hidden += h

a = Analysis(["app.py"], pathex=["../src"], binaries=binaries, datas=datas, hiddenimports=hidden,
             excludes=["tkinter", "matplotlib", "pytest"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="PitchSpeed", console=True)
coll = COLLECT(exe, a.binaries, a.datas, name="PitchSpeed")
