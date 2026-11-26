@echo off
REM GAME 2 ROUND 1 - POURING (up to 100 points, best of 3 runs)
REM Autonomous run. Optional: RUN.bat <seconds> <checkpoint_step>. Ctrl+C = emergency stop.
REM NOENS=1: plans are not averaged, so short moves such as opening the gripper go through.
set NOENS=1
set RUNFPS=
call "%~dp0..\..\run.bat" pour %1 %2
