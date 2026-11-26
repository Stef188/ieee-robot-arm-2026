@echo off
REM ================= EDIT THIS FILE ONLY =================
REM Serial ports. COM numbers differ per laptop/USB socket, so each arm is identified by its
REM USB adapter serial number (below) and its COM port is looked up automatically after the
REM environment starts. The COM values here are only used if that lookup fails.
REM Serials: left follower 5B79015609, right follower 5B14113082, left leader 5B14033452, right leader 5B79015713
REM 18 Nov: left leader now uses the SPARE adapter board 5B14033452. Its original board 5B14114406 is
REM probably fine (the fault was the USB hub); put that serial back here if the original board is refitted.
REM 18 Nov 15:35: left follower board 5B79015609 hung on open; swapped to 5B14114406 (left leader's old board).
REM 19 Nov: confirmed by unplugging each cable: left follower 5B14031045, right follower 5B14113082,
REM left leader 5B79015609, right leader 5B79015713.
REM The hangs were always the UGREEN/Genesys USB hub, never a board. ARM_CHECK.bat warns if a board is on the wrong arm.
set SN_LEFT_FOLLOWER=5B14031045
REM 19 Nov afternoon: right follower board 5B14113082 died (USB fine, servo side dead) - replaced by 5B14114406.
set SN_RIGHT_FOLLOWER=5B14114406
set SN_LEFT_LEADER=5B79015609
set SN_RIGHT_LEADER=5B79015713
REM Fallback = Rares laptop numbering, 18 Nov (james laptop was 5 / 7 / 3 / 6)
set LEFT_FOLLOWER=COM8
set RIGHT_FOLLOWER=COM6
set LEFT_LEADER=COM5
set RIGHT_LEADER=COM3

REM Camera indexes: run 1_find_cameras.bat, look at the saved pictures, fill in.
REM Leave a wrist camera EMPTY (nothing after =) if you do not have it. 17 Nov: camera 218622279738 fixed with new USB 3 extension.
REM WRIST_TYPE=intelrealsense: wrist values are RealSense serial numbers. WRIST_TYPE=opencv: wrist values are indexes.
set WRIST_TYPE=intelrealsense
REM TOP_TYPE: opencv (CAM_TOP = index) or intelrealsense (CAM_TOP = serial). 17 Nov 21:26: top = D405 218622276310 moved overhead.
set TOP_TYPE=intelrealsense
set CAM_TOP=218622276310
REM 19 Nov: left follower RealSense is now 218622273385 (was 218622279738, no longer connected).
set CAM_LEFT_WRIST=218622273385
REM 19 Nov: right follower wrist = Logitech C270 webcam. Webcam numbers change when cables move, so it is
REM found BY NAME every time (CAM_RIGHT_WRIST is filled in below). It runs 640x360: at 640x480 a C270 only gives 24 fps.
set RIGHT_WRIST_TYPE=opencv
set CAM_RIGHT_WRIST_NAME=Logi C270 HD WebCam
set CAM_RIGHT_WRIST=
set CAM_SIDE=
set CAM_W=640
set CAM_H=480
set FPS=30

REM Training batch size (8 is safe for a 6-8 GB GPU; use 16 with 12 GB+)
set BATCH=8
REM ========================================================

REM Install locations: D:\ is used when present (Rares laptop: C: is nearly full), else the user folder (james laptop).
set CONDA_ROOT=%USERPROFILE%\miniforge3
if exist "D:\miniforge3\Scripts\activate.bat" set CONDA_ROOT=D:\miniforge3
set LEROBOT_DIR=%USERPROFILE%\lerobot
if exist "D:\lerobot\pyproject.toml" set LEROBOT_DIR=D:\lerobot
set DATA_DIR=%LEROBOT_DIR%\data
set OUT_DIR=%LEROBOT_DIR%\outputs\train
set FROZEN_DIR=%LEROBOT_DIR%\frozen

REM Keep caches, calibration and temp files next to LeRobot (on D: on the Rares laptop) instead of C:.
REM Calibration then lives in %LEROBOT_DIR%\.cache\huggingface\lerobot\calibration
set HF_HOME=%LEROBOT_DIR%\.cache\huggingface
set TORCH_HOME=%LEROBOT_DIR%\.cache\torch
set PIP_CACHE_DIR=%LEROBOT_DIR%\.cache\pip
if not exist "%LEROBOT_DIR%\.cache\tmp" mkdir "%LEROBOT_DIR%\.cache\tmp"
set TEMP=%LEROBOT_DIR%\.cache\tmp
set TMP=%LEROBOT_DIR%\.cache\tmp

call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ROOT%"
call conda activate lerobot

REM Replace the fallback COM numbers with the ports whose serial numbers match (prints nothing for an unplugged arm).
for /f "usebackq tokens=1,2 delims==" %%a in (`python "%~dp0tools\kit.py" ports`) do set %%a=%%b
REM Find webcams configured by name (DirectShow number, matches backend 700 below).
if defined CAM_RIGHT_WRIST_NAME for /f "usebackq" %%i in (`python "%~dp0tools\kit.py" camindex %CAM_RIGHT_WRIST_NAME%`) do set CAM_RIGHT_WRIST=%%i
if defined CAM_RIGHT_WRIST_NAME if not defined CAM_RIGHT_WRIST echo WARNING: right wrist camera "%CAM_RIGHT_WRIST_NAME%" not found - running WITHOUT it. Do not record until it is plugged in.

set CAM_OPTS=width: %CAM_W%, height: %CAM_H%, fps: %FPS%
set TOP_SPEC=top: {type: opencv, index_or_path: %CAM_TOP%, fourcc: MJPG, %CAM_OPTS%}
if /i "%TOP_TYPE%"=="intelrealsense" set TOP_SPEC=top: {type: intelrealsense, serial_number_or_name: '%CAM_TOP%', %CAM_OPTS%}
REM Wrist cameras at 424x240 so they keep 30 fps even on a USB 2 connection
set WRIST_OPTS=width: 424, height: 240, fps: %FPS%
set ROBOT_ARGS=--robot.type=bi_so_follower --robot.id=bimanual_follower --robot.left_arm_config.port=%LEFT_FOLLOWER% --robot.right_arm_config.port=%RIGHT_FOLLOWER% --robot.left_arm_config.num_read_retries=10 --robot.right_arm_config.num_read_retries=10
if defined CAM_SIDE goto cams_with_side
set ROBOT_ARGS=%ROBOT_ARGS% --robot.cameras="{ %TOP_SPEC%}"
goto cams_wrist
:cams_with_side
set ROBOT_ARGS=%ROBOT_ARGS% --robot.cameras="{ %TOP_SPEC%, side: {type: opencv, index_or_path: %CAM_SIDE%, %CAM_OPTS%}}"
:cams_wrist
REM Each wrist can be a RealSense (serial number) or a webcam (opencv: DirectShow number, 640x360 MJPG).
if not defined LEFT_WRIST_TYPE set LEFT_WRIST_TYPE=%WRIST_TYPE%
if not defined RIGHT_WRIST_TYPE set RIGHT_WRIST_TYPE=%WRIST_TYPE%
set CV_WRIST_OPTS=width: 640, height: 360, fps: %FPS%, fourcc: MJPG, backend: 700
set LW_SPEC=type: intelrealsense, serial_number_or_name: '%CAM_LEFT_WRIST%', %WRIST_OPTS%
if /i "%LEFT_WRIST_TYPE%"=="opencv" set LW_SPEC=type: opencv, index_or_path: %CAM_LEFT_WRIST%, %CV_WRIST_OPTS%
set RW_SPEC=type: intelrealsense, serial_number_or_name: '%CAM_RIGHT_WRIST%', %WRIST_OPTS%
if /i "%RIGHT_WRIST_TYPE%"=="opencv" set RW_SPEC=type: opencv, index_or_path: %CAM_RIGHT_WRIST%, %CV_WRIST_OPTS%
if defined CAM_LEFT_WRIST set ROBOT_ARGS=%ROBOT_ARGS% --robot.left_arm_config.cameras="{ wrist: {%LW_SPEC%}}"
if defined CAM_RIGHT_WRIST set ROBOT_ARGS=%ROBOT_ARGS% --robot.right_arm_config.cameras="{ wrist: {%RW_SPEC%}}"
set TELEOP_ARGS=--teleop.type=bi_so_leader --teleop.id=bimanual_leader --teleop.left_arm_config.port=%LEFT_LEADER% --teleop.right_arm_config.port=%RIGHT_LEADER%
