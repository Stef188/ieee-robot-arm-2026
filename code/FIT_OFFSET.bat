@echo off
setlocal
REM 27 Nov (competition): how far the grabs are shifted on THIS table, from the grab spots taught today with
REM TEACH_GRAB.bat left / right. Prints SORT_TRIM_LEFT / SORT_TRIM_RIGHT for the latest sorter and version I. No motors.
call "%~dp0..\config.bat"
python "%~dp0fit_offset.py"
pause
