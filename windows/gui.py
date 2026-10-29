"""Entry point of the windowed Pitch Speed program (PitchSpeed.exe in the Windows build)."""
import sys


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    from pitchspeed.app.main import main
    sys.exit(main())
