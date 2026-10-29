"""pitchspeed: estimate baseball pitch speeds from video."""

__version__ = "0.3.0"

# Import OpenCV before PyAV.  On Linux both wheels bundle X11 libraries with
# identical names; if PyAV's copies load first, OpenCV's Qt windows hang.
try:  # pragma: no cover - import side effect only
    import cv2 as _cv2  # noqa: F401
except Exception:  # cv2 missing/broken: the GUI commands report it later
    pass
