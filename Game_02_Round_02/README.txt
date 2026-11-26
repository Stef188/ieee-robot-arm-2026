GAME 2 - ROUND 2: FOLDING  (imitation learning, LeRobot ACT)

Model    : ACT (Action Chunking Transformer, LeRobot), chunk size 100, ResNet-18 vision backbone
           (ImageNet-pretrained weights = the only pre-trained component).
Training : 200,000 steps in total (150k cloud GPU + 50k on the laptop), batch 8, learning rate 1e-5,
           final loss 0.060, offline joint error 1.39 deg (training-notes/TRAIN_NEW.bat).
Dataset  : ieee_fold_new_clean - 43 teleoperated two-arm demonstrations with the final (deeper) grip,
           55,724 frames at 30 fps, 3 cameras (top, left wrist RealSense, right wrist C270) + joints of both
           arms. Older demos with a different grip were left out on purpose (training-notes/NEW_GRIP_START.txt).
           Cleaned with training-notes/CLEAN_NEW.bat (idle start/end frames trimmed).
Inference: on the laptop wired to the arms only, no network (training-notes/RUN.bat).

models/ and data/: the model (207 MB) and dataset are larger than GitHub's 100 MB file limit; they are kept
on the team laptop and shared with the judges through a secure link on request.
