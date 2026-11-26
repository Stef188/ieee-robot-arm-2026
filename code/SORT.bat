@echo off
setlocal
REM CLASSICAL (pre-programmed) COLOUR SORTER - NOT allowed in the competition (rulebook 5.4.3.4-5, 5.4.5.10-11).
REM   SORT.bat           sort for up to 300 s
REM   SORT.bat 60        up to 60 s
REM   SORT.bat dry       no motors, no camera: prints what it would do (uses test_images\live_top.png)
REM Needs CALIBRATE.bat left and/or right first. Speed: type  set SORT_SPEED=30  first for slower moves (default 45).
REM Ctrl+C = stop: the arm goes back to rest. Close every other robot program first.
call "%~dp0..\config.bat"
if /i "%1"=="dry" ( python "%~dp0sort.py" --dry & pause & exit /b )
echo Hands away from the arms. Ctrl+C = stop.
pause
python "%~dp0..\tools\awake.py" python -u "%~dp0sort.py" %1
pause
