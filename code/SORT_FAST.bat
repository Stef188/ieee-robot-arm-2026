@echo off
setlocal
REM 26 Nov 01:00: SORT_BEST, but FAST - both arms at the same time + quicker moves, so more tries per minute.
REM   SORT_FAST.bat [seconds]   (default 300)       SORT_BEST.bat = the slower version that worked on 25 Nov.
REM What is faster (the neural network was never the slow part - 0.2 ms per grab; the arm moves are):
REM   travel 40 -> 100 deg/s, going down 16 -> 50 deg/s, lifting at full speed, block -> bin -> home 60 -> 120 deg/s,
REM   gripper opens/closes at its own fast speed (was ~1.6 s), shorter pauses, no long "has it arrived" waits (8 s -> 1.5 s),
REM   grab still starts HIGH (2 lift heights). Miss = known instantly (jaw watched while closing) -> next block.
REM If it starts knocking blocks over: set SORT_DESC=0.4 first (slower way down), or use SORT_BEST.bat.
set SORT_BOTH=1
if not defined SORT_SPEED set SORT_SPEED=100
if not defined SORT_DESC set SORT_DESC=0.5
if not defined SORT_LIFT set SORT_LIFT=1.0
if not defined SORT_TRANSFER_SPEED set SORT_TRANSFER_SPEED=120
if not defined SORT_GRIP_SPEED set SORT_GRIP_SPEED=150
set SORT_CLOSE_WAIT=0.25
set SORT_OPEN_WAIT=0.1
set SORT_WAIT_MAX=1.5
set SORT_UP=2.0
set SORT_ALIGN=0
set SORT_BIN_CLEAR=20
REM 26 Nov 20:00: right-arm trim OFF - its taught grabs are at block centres, the trim aimed 1.1 cm away from them
REM 27 Nov 18:50 (competition): grabs landed ~1 cm to the RIGHT of the blocks -> aim 1 cm LEFT (camera picture)
if not defined SORT_TRIM_LEFT set SORT_TRIM_LEFT=0,0
if not defined SORT_TRIM_RIGHT set SORT_TRIM_RIGHT=0,0
REM 27 Nov (competition): left arm base clamped ~6 deg turned vs the lab (6 grabs taught here, fit_offset.py)
if not defined SORT_PAN_LEFT set SORT_PAN_LEFT=8
if not defined SORT_PAN_RIGHT set SORT_PAN_RIGHT=1
REM 27 Nov (competition): drops reach this many px (about 8 px = 1 cm) further into the bin, towards its BACK wall.
REM Still landing at the front -> raise it (30, 40). Touching the far rim -> lower it (10).
if not defined SORT_DROP_REACH set SORT_DROP_REACH=20
REM 27 Nov (competition): 3 tries per block - retries search 1 cm right, then left (sideways scatter)
if not defined SORT_MAX_TRIES set SORT_MAX_TRIES=3
REM 27 Nov 18:50: an idle arm takes the best positive-points block after 1 s (conflict waits are separate)
if not defined SORT_IDLE_MAX set SORT_IDLE_MAX=1
REM 27 Nov 19:30 (user: go slow on grappling): slower final approach + gentler close
if not defined SORT_ZONE_SPEED set SORT_ZONE_SPEED=10
set SORT_APPROACH_OFFSET=0
set S=%1
if "%S%"=="" set S=300
call "%~dp0SORT.bat" %S%
