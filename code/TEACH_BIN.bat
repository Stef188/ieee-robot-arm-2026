@echo off
setlocal
REM 26 Nov: HARD-CODE the bin drop poses by hand.
REM   TEACH_BIN.bat            all 5 bins in a row (left red, left green, right yellow, right blue, right green)
REM   TEACH_BIN.bat left green just one
REM Each time: the arm goes LIMP - hold the gripper right above the MIDDLE of the bin (~5 cm above the rim), Enter.
REM Close every other robot program first.
call "%~dp0..\config.bat"
if "%1"=="" ( python "%~dp0teach_bin.py" all ) else ( python "%~dp0teach_bin.py" %1 %2 )
pause
