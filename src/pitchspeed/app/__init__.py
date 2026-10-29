"""Pitch Speed desktop program (Qt): a main window over the `pitchspeed` command line.

The window never does the work itself: each button runs the matching `pitchspeed`
command in a separate process (jobs.py) and shows its progress.  The marking,
calibration and review windows are still the OpenCV windows of the command line.
"""
