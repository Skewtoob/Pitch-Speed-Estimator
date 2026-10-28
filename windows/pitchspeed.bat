@echo off
rem Drag one or more video files onto this file to run the guided analysis
rem (the first time it asks: find the pitches automatically, or mark them by hand).
rem Several files are done one after the other.
rem You can also run it from a terminal:  windows\pitchspeed.bat "D:\Tapes\game1.avi"
set "PS_EXE=%~dp0..\.venv\Scripts\pitchspeed.exe"
if not exist "%PS_EXE%" (
  echo pitchspeed is not installed yet. Double-click windows\install.bat first.
  pause
  exit /b 1
)
if "%~1"=="" (
  echo Drag a video file onto this .bat file, or run:  pitchspeed.bat "path\to\video.avi"
  pause
  exit /b 1
)
:next_video
if "%~1"=="" goto all_done
echo.
echo ===== "%~nx1" =====
"%PS_EXE%" run "%~1"
shift
goto next_video
:all_done
pause
