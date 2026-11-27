@echo off
setlocal
REM 27 Nov (competition): re-measure each arm's TURN correction from all ground truth on this table.
REM   UPDATE_TURNS.bat         show the best values (changes nothing)
REM   UPDATE_TURNS.bat apply   write them into SORT_FAST.bat (shape/dynamic) and version I RUN.bat
call "%~dp0..\config.bat"
python "%~dp0update_turns.py" %*
pause
