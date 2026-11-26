@echo off
setlocal
REM 25 Nov late: the sorter with the settings of the GOOD RUN (22:35 - both arms grabbed first try), plus the later
REM fixes that really helped. Double-click, or: SORT_BEST.bat [seconds]  (default 300 = 5 minutes)
REM   RIGHT arm: coordinates only - demo calibration + pan correction + trim 0.5 cm left / 1 cm away from the arms,
REM              NO camera adjusting (that run: blue block first try)
REM   LEFT arm:  camera check from high up first, then a small fine-tune ("careful" mode, that run: centred 0.010)
REM   both:      learned coordinate correction (average of earlier camera corrections, once there are 3+),
REM              gripper opens to 65, jaw-based grip check (no more dropping), waits until really above the bin,
REM              high drop, speed 40 deg/s, block -> bin -> home direct at 60 deg/s.
REM Undone: the wrist turn, the 2 cm grab shift, the fast-mode close-in corrections, the right-arm camera adjusting.
REM 26 Nov 00:50: BOTH ARMS AT THE SAME TIME (each takes the blocks it reaches best; a pick into the middle or the
REM other side waits until the other arm is home). set SORT_BOTH=0 first = the old one-arm-at-a-time.
REM A missed block is left: the arm goes to the next one, misses are retried only when nothing else is left.
if not defined SORT_BOTH set SORT_BOTH=1
set SORT_SPEED=40
REM 25 Nov 23:45: WRIST CAMERAS OFF for both arms (user + data: the first position was already centred, the camera
REM re-adjusting then drove the gripper away). Coordinates only: calibration + pan correction + learned correction + trim.
set SORT_ALIGN=0
set SORT_BIN_CLEAR=20
REM 26 Nov 20:00: right-arm trim OFF - its taught grabs are at block centres, the trim aimed 1.1 cm away from them
set SORT_TRIM_RIGHT=0,0
set SORT_APPROACH_OFFSET=0
set S=%1
if "%S%"=="" set S=300
call "%~dp0SORT.bat" %S%
