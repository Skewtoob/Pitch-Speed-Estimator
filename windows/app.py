"""Entry point of the console program (pitchspeed-cli.exe in the Windows build; the main window runs it).

Drag one or more video files onto pitchspeed-cli.exe: each is
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
    for stream in (sys.stdout, sys.stderr):    # show output as it happens when run by the main window
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass
    import multiprocessing
    multiprocessing.freeze_support()
    import cv2  # noqa: F401  (load OpenCV before PyAV: see pitchspeed.ui.gui)
    from pitchspeed import cli

    args = sys.argv[1:]
    if not args:
        print("Pitch Speed (command line)\n\nThis is the program the Pitch Speed window runs in the"
              " background.\nOpen 'Pitch Speed' (PitchSpeed.exe) instead, or drag a video file onto this one.")
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
