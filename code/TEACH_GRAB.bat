@echo off
setlocal
REM 26 Nov: teach grab spots by hand where an arm misses.  TEACH_GRAB.bat right   (or left)
REM Put ONE block where it missed -> Enter -> arm goes LIMP -> guide the open gripper around the block -> Enter. q = finish.
call "%~dp0..\config.bat"
python "%~dp0teach_grab.py" %1
pause
