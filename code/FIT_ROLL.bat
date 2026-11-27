@echo off
setlocal
REM 27 Nov (competition): measure each arm's wrist geometry from grabs taught today (cube at DIFFERENT angles, jaws flat on two faces).
call "%~dp0..\config.bat"
python "%~dp0fit_roll.py"
pause
