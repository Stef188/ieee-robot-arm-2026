@echo off
setlocal
REM 26 Nov: ROUND 3 - DYNAMIC SORT. Shape rule (ShapeNet, as SORT_SHAPE.bat) + the judge MOVES the bins / blocks:
REM  - drop pose for wherever the camera sees each bin NOW (learned drop model, drop_model_<arm>.json, from 1,053 demo
REM    releases over 14-17 bin places per arm); a bin on one of our known spots uses that spot's tested pose
REM  - before going down: is the block still there? (moved = choose again, not a miss)
REM  - before letting go: where is the bin NOW? (moved = re-aim)   - no bin memory
REM   SORT_DYNAMIC.bat [seconds]      SORT_DYNAMIC.bat dry
set SORT_RULE=dynamic
if not defined SORT_SHAPE_MIN set SORT_SHAPE_MIN=0.5
call "%~dp0SORT_FAST.bat" %*
