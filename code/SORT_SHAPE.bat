@echo off
setlocal
REM 26 Nov: ROUND 2 - SHAPE SORT. Same as SORT_FAST.bat, but every block's bin comes from its SHAPE, recognised by the
REM learned classifier ShapeNet (shape_net.pt, trained by train_shape_net.py on our shape demos):
REM   cube -> red bin, cylinder -> blue bin, dodecahedron -> green bin, pentagonal prism -> yellow bin (rulebook 5.8.1.46)
REM   SORT_SHAPE.bat [seconds]      SORT_SHAPE.bat dry   (no motors)
set SORT_RULE=shape
REM 26 Nov 18:30: ShapeNet trained on the PHOTO SESSION (clean labels): 77%% right on unseen photos; at >= 50%% sure 80%% right,
REM covering 91%% of blocks. Wrong bin = minus points (5.8.1.63); blocks are taken in order of expected points.
if not defined SORT_SHAPE_MIN set SORT_SHAPE_MIN=0.5
call "%~dp0SORT_FAST.bat" %*
