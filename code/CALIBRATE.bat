@echo off
setlocal
REM CLASSICAL SORTER - live calibration, one arm at a time:  CALIBRATE.bat left   then   CALIBRATE.bat right
REM Optional 2nd number = how many points (default 12).  The arm goes LIMP - hold it. Nothing moves by itself.
REM Close every other robot program first (RUN / RECORD / teleop) - the arm and the camera can only be used once.
call "%~dp0..\config.bat"
set SIDE=%1
if "%SIDE%"=="" set SIDE=left
python -u "%~dp0calibrate.py" %SIDE% %2
python "%~dp0model.py"
pause
