@echo off
setlocal
REM 26 Nov: run this AFTER a sorter run (arms at rest). 1) report on the run  2) every grab that worked becomes a new
REM training example for the grab model (learn_from_runs.py) - the next run starts with the improved model.
call "%~dp0..\config.bat"
python "%~dp0analyse_run.py" 1
python "%~dp0learn_from_runs.py"
pause
