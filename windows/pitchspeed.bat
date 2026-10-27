@echo off
rem Drag a video file onto this file to run the guided analysis
rem (calibrate the first time -> find pitches -> review -> annotated video).
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
"%PS_EXE%" run "%~1"
pause
