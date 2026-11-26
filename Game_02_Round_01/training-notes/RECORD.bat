@echo off
REM GAME 2 ROUND 1 - POURING (up to 100 points, best of 3 runs)
REM Records 50 more demos (Esc stops and saves any time) (or: RECORD.bat 5). Recipe: HOW_TO.txt in this folder.
echo.
echo GAME 2 ROUND 1 - POURING (up to 100 points, best of 3 runs)
echo HOW TO RECORD:
echo   Up to 2 minutes per demo: press Right arrow as soon as the pour is finished. Aim for 80.
echo   Right arm grasps the cup LOW on its body. Left arm grips the bowl rim and lifts or pulls it a little toward the cup.
echo   Pour slowly over the bowl centre, tap once, set the cup upright, release the bowl, both arms home.
echo   The bowl must be visibly held in EVERY demo, or the model learns to ignore it.
echo   Move cup and bowl +-3 cm inside their start zones between demos. Always the full 125 g of rice.
echo   Finish every demo well under 45 seconds.
echo   Start AND end every demo in the same rest pose, grippers open.
echo   Move smoothly at one steady speed, same strategy every time. One person does all demos of a task if possible.
echo   Left arrow = THROW AWAY this demo and redo it (use it on every sloppy one).
echo   Right arrow = end the demo early / skip the reset wait.   Esc = stop and save. NEVER close the window.
echo.
pause
set N=%1
if "%N%"=="" set N=50
call "%~dp0..\..\record.bat" pour %N%
