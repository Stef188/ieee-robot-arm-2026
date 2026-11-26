@echo off
setlocal
REM Cleaned copy of the NEW-GRIP fold demos only (episode 72 on, see NEW_GRIP_START.txt) -> ieee_fold_new_clean.
REM Same cleaning as CLEAN.bat (cuts standing still at start/end, shortens long pauses, drops empty demos and the
REM hand-picked bad ones in tools\cleandata.py DROP), but the old-style demos 0-71 are left out completely.
REM The recorded data in ieee_fold and the old ieee_fold_clean are never changed. About 45 s per demo.
REM Run it again after recording more new-style demos: it rebuilds ieee_fold_new_clean from scratch.
call "%~dp0..\..\config.bat"
python "%~dp0..\..\tools\awake.py" python "%~dp0..\..\tools\cleandata.py" ieee_fold ieee_fold_new_clean 72
pause
