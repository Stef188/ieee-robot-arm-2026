@echo off
setlocal
REM Continues the NEW-GRIP fold model on THIS laptop (25 Nov): starts from the weights of checkpoint 100000 of
REM ieee_fold_new_clean_cloud_1124_1727 (= 150k total) and trains more on ieee_fold_new_clean (the 43 new-grip demos).
REM Not a true resume (the optimizer state stayed on the deleted Brev machine) - a warm start, which works about the same.
REM Optional: TRAIN_NEW.bat <steps>   (default 50000 = 200k total, about 3 h here with mixed precision).
REM When done, RUN.bat / RUN_SEQ.bat in this folder pick the new model automatically (newest fold model).
REM Do not run the robot while it trains. Laptop on the charger.
call "%~dp0..\..\config.bat"
set STEPS=%1
if "%STEPS%"=="" set STEPS=50000
if "%WORKERS%"=="" set WORKERS=2
set INIT=%OUT_DIR%\ieee_fold_new_clean_cloud_1124_1727\checkpoints\100000\pretrained_model
if not exist "%INIT%\config.json" ( echo Starting model not found: %INIT% & pause & exit /b 1 )
for /f "usebackq" %%i in (`python "%~dp0..\..\tools\kit.py" stamp`) do set STAMP=%%i
set RUN=ieee_fold_new_clean_local_%STAMP%
cd /d "%LEROBOT_DIR%"
echo Continuing fold on ieee_fold_new_clean for %STEPS% steps from %INIT%  -  run %RUN%
python "%~dp0..\..\tools\awake.py" lerobot-train --dataset.repo_id=local/ieee_fold_new_clean --dataset.root=%DATA_DIR%\ieee_fold_new_clean --dataset.image_transforms.enable=true --policy.path=%INIT% --policy.device=cuda --policy.use_amp=true --policy.push_to_hub=false --batch_size=8 --steps=%STEPS% --save_freq=10000 --log_freq=200 --num_workers=%WORKERS% --wandb.enable=false --output_dir=%OUT_DIR%\%RUN% --job_name=%RUN%
echo.
echo Done. RUN.bat / RUN_SEQ.bat in this folder now use it.
pause
