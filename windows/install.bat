@echo off
rem One-time setup: creates a private Python environment in .venv and installs pitchspeed.
rem Requires Python 3.10+ (64-bit) from https://www.python.org/downloads/ ("Add python.exe to PATH" is not needed).
cd /d "%~dp0.."
echo Creating Python environment in "%CD%\.venv" ...
py -3 -m venv .venv
if errorlevel 1 (
  python -m venv .venv
  if errorlevel 1 (
    echo.
    echo Could not create the environment. Install Python 3 from https://www.python.org/downloads/ and try again.
    pause
    exit /b 1
  )
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 (
  echo.
  echo Installation failed. Scroll up for the error message.
  pause
  exit /b 1
)
echo.
".venv\Scripts\pitchspeed.exe" --version
echo.
echo Done. Drag a video file onto windows\pitchspeed.bat to analyze it.
pause
