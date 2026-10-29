"""Entry point of the stand-alone Windows program (PitchSpeed.exe, built with PyInstaller).

Drag one or more video files onto PitchSpeed.exe (or a shortcut to it): each is
run through the guided `pitchspeed run`.  Started with other arguments it behaves
like the `pitchspeed` command (e.g. `PitchSpeed.exe render video.avi --mode clips`).
The console stays open at the end so the results can be read.
"""
import os
import sys


def _pause() -> None:
    try:
        input("\nPress Enter to close this window ...")
    except (EOFError, KeyboardInterrupt):
        pass


def main() -> int:
    import multiprocessing
    multiprocessing.freeze_support()
    import cv2  # noqa: F401  (load OpenCV before PyAV: see pitchspeed.ui.gui)
    from pitchspeed import cli

    args = sys.argv[1:]
    if not args:
        print("Pitch Speed\n\nDrag a video file onto PitchSpeed.exe (or onto the 'Pitch Speed'"
              " shortcut) to measure its pitches.\nSeveral files can be dragged at once.")
        _pause()
        return 0
    if all(os.path.isfile(a) for a in args):
        code = 0
        for video in args:
            print(f"\n===== {os.path.basename(video)} =====")
            try:
                code = cli.main(["run", video]) or code
            except SystemExit as e:              # argparse / fatal messages
                code = e.code if isinstance(e.code, int) else 1
        _pause()
        return int(code)
    try:
        return int(cli.main(args) or 0)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1


if __name__ == "__main__":
    sys.exit(main())
