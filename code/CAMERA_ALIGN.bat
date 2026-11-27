@echo off
setlocal
REM 27 Nov: top (pole) camera alignment - put it back where it was in the lab, or correct it in software.
REM   CAMERA_ALIGN.bat        live window: C = click the 4 inner corners of the black line -> how to move the camera,
REM                           W = save a software correction, V = raw/corrected view, G = ghost of the lab view, Q = quit
REM   CAMERA_ALIGN.bat save   save the view NOW as the lab reference (in the lab, table clear, bins on their spots)
REM Details at the top of camera_warp.py. Only the camera is used - the arms do not move.
call "D:\Robot stuff for rares\config.bat"
python "%~dp0camera_warp.py" %*
pause
