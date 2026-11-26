@echo off
REM GAME 2 ROUND 2 - FOLDING (up to 100 points)
REM Records 50 more demos (Esc stops and saves any time) (or: RECORD.bat 5). Recipe: HOW_TO.txt in this folder.
echo.
echo GAME 2 ROUND 2 - FOLDING (up to 100 points)
echo HOW TO RECORD:
echo   up to 90-second demos (Right arrow when home), ONE cloth per demo, aim for 100.
echo   Both arms pinch the two near corners, fold away from the robot, re-grasp the short side, fold again,
echo   both arms drag the folded cloth into the target zone, both home.
echo   Rotate through all four colours and all four start spots. Have 0-3 folded cloths already in the zone.
echo   Put smooth tape on the gripper tips so the microfibre does not snag (rule 5.1.2.5).
echo   Start AND end every demo in the same rest pose, grippers open.
echo   Move smoothly at one steady speed, same strategy every time. One person does all demos of a task if possible.
echo   Left arrow = THROW AWAY this demo and redo it (use it on every sloppy one).
echo   Right arrow = end the demo early / skip the reset wait.   Esc = stop and save. NEVER close the window.
echo.
pause
set N=%1
if "%N%"=="" set N=50
call "%~dp0..\..\record.bat" fold %N%
