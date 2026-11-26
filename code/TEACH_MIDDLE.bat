@echo off
setlocal
REM 26 Nov pm: teach the MIDDLE of the table to BOTH arms, so hand-offs land where both arms grab well.
REM The two arms' demo areas don't overlap (left ends at x 345, right starts at x 376 in the top camera) - the
REM hand-off spot is in that gap. Teach 2-3 spots there with EACH arm (same spots for both):
REM   put ONE block in the middle strip, about halfway between the two arm bases, a little towards the bins.
call "%~dp0..\config.bat"
echo === LEFT arm: put ONE block in the MIDDLE strip, between the two arms ===
python "%~dp0teach_grab.py" left
echo === RIGHT arm: same spots again, now with the right arm ===
python "%~dp0teach_grab.py" right
pause
