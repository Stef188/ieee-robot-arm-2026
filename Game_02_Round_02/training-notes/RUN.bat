@echo off
REM setlocal: these settings (grasp guard, long plans) stay inside this run and never carry over to other tasks.
setlocal
REM GAME 2 ROUND 2 - FOLDING (up to 100 points)
REM Autonomous run. Optional: RUN.bat <seconds> <checkpoint_step>. Ctrl+C = emergency stop.
REM NOENS=1: plans are not averaged, so short moves such as opening the gripper go through.
set NOENS=1
set RUNFPS=
set GRASPCHECK=1
set RUNNAS=
call "%~dp0..\..\run.bat" fold %1 %2
