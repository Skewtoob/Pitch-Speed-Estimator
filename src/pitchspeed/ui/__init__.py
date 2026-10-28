"""Shared pieces of the interactive windows (calibration, review, key-code helper).

keys    portable names for cv2.waitKeyEx codes (Win32 / Qt / GTK / Cocoa)
gui     display checks + thin cv2 window wrapper (never aborts without a display)
canvas  window layout, frame <-> display coordinates, text / markers / help
loupe   8x loupe with crosshair
frames  cached / prefetched sample images
fmt     small text formatters

IMPORTANT (Linux): import cv2 before anything imports av (PyAV), otherwise
OpenCV's Qt window hangs forever; see pitchspeed.ui.gui.
"""
