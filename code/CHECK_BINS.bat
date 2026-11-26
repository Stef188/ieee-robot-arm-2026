@echo off
setlocal
REM 26 Nov: SETUP-DAY BIN CHECK - top camera only, the arms do NOT move.
REM Shows which arm will drop which colour with the bins as they stand now, and warns about any bin that is not on
REM one of our 4 bin spots. Picture: classical\bin_layouts\live_check.png
REM   CHECK_BINS.bat        (live camera)
REM   CHECK_BINS.bat all    (test all 24 bin orders, no camera)
call "%~dp0..\config.bat"
if /i "%1"=="all" ( python "%~dp0bin_layouts\check_layout.py" all ) else ( python "%~dp0bin_layouts\check_layout.py" live )
pause
