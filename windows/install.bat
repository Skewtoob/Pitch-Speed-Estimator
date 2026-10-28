@echo off
setlocal
rem One-time setup: creates a private Python environment in .venv (inside this
rem project folder) and installs pitchspeed into it.
rem Needs Python 3.10 or newer, 64-bit, from https://www.python.org/downloads/
cd /d "%~dp0.."
set "ROOT=%CD%"

rem ---- 1. This folder must be writable ("C:\Program Files" is not) ----
set "PROBE=%ROOT%\.pitchspeed_write_test.tmp"
(echo test> "%PROBE%") 2>nul
if not exist "%PROBE%" goto not_writable
del "%PROBE%" >nul 2>&1

rem ---- 2. Find Python 3 ----
set "PY="
py -3 --version >nul 2>&1
if not errorlevel 1 set "PY=py -3"
if defined PY goto have_python
python --version >nul 2>&1
if not errorlevel 1 set "PY=python"
if defined PY goto have_python
goto no_python

:have_python
echo Using:
%PY% --version
echo.
echo Creating the Python environment in "%ROOT%\.venv" ...
%PY% -m venv "%ROOT%\.venv"
if errorlevel 1 goto venv_failed
"%ROOT%\.venv\Scripts\python.exe" -m pip install --upgrade pip
"%ROOT%\.venv\Scripts\python.exe" -m pip install -e "%ROOT%"
if errorlevel 1 goto pip_failed
echo.
"%ROOT%\.venv\Scripts\pitchspeed.exe" --version
call :make_launcher
echo.
echo Done. To analyze a video, drag the video file onto
echo   "%ROOT%\windows\pitchspeed.bat"
if defined LAUNCHER echo or onto "Pitch Speed" on your Desktop.
echo.
pause
exit /b 0

:not_writable
set "DEST=%USERPROFILE%\pitchspeed"
echo.
echo Windows does not let this installer write to:
echo   "%ROOT%"
echo Folders such as Program Files are protected: even on an administrator
echo account, a program may only change them when it was started with
echo "Run as administrator".
echo.
echo Two ways to fix it:
echo   Y = copy pitchspeed to "%DEST%" and install it there [recommended]
echo   N = stop; then either move the folder yourself, or right-click
echo       install.bat and choose "Run as administrator"
echo.
choice /c YN /m "Copy pitchspeed to %DEST% and install it there"
if errorlevel 2 goto copy_declined
robocopy "%ROOT%" "%DEST%" /E /XD .venv /NFL /NDL /NJH /NJS /NP >nul
if errorlevel 8 goto copy_failed
echo.
echo Copied. Continuing the installation in "%DEST%" ...
echo You can delete the old folder afterwards:
echo   "%ROOT%"
echo.
call "%DEST%\windows\install.bat"
exit /b %errorlevel%

:copy_declined
echo.
echo OK, nothing was changed. Either:
echo   - move the whole pitchspeed folder to Documents or C:\pitchspeed and
echo     double-click windows\install.bat again, or
echo   - right-click windows\install.bat and choose "Run as administrator".
echo.
pause
exit /b 1

:copy_failed
echo.
echo Could not copy the files to "%DEST%".
echo Move the whole pitchspeed folder to Documents or C:\pitchspeed yourself,
echo then double-click windows\install.bat again.
echo.
pause
exit /b 1

:no_python
echo.
echo Python 3 was not found on this PC.
echo Install it from https://www.python.org/downloads/ - 64-bit, version 3.10
echo or newer - then double-click windows\install.bat again.
echo.
pause
exit /b 1

:venv_failed
echo.
echo Could not create the Python environment in "%ROOT%\.venv".
echo Scroll up for the error message.
echo.
pause
exit /b 1

:pip_failed
echo.
echo Installing the packages failed. Scroll up for the error message.
echo Common causes: no internet connection, or a brand-new Python version that
echo the packages do not support yet - Python 3.12 or 3.13 is the safest choice.
echo.
pause
exit /b 1

rem ---- Desktop launcher: a tiny .bat that forwards dropped files ----
:make_launcher
set "LAUNCHER="
set "DESKTOP="
for /f "usebackq delims=" %%D in (`powershell -NoProfile -Command "[Environment]::GetFolderPath('Desktop')"`) do set "DESKTOP=%%D"
if not defined DESKTOP set "DESKTOP=%USERPROFILE%\Desktop"
if not exist "%DESKTOP%\" exit /b 0
set "LF=%DESKTOP%\Pitch Speed.bat"
>"%LF%" echo @echo off
>>"%LF%" echo rem Drag a video file onto this icon to analyze it with pitchspeed.
>>"%LF%" echo call "%ROOT%\windows\pitchspeed.bat" %%*
if exist "%LF%" set "LAUNCHER=%LF%"
exit /b 0
