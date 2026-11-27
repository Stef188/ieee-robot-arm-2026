GAME 2 - ROUND 1: POURING  (imitation learning, LeRobot ACT)

Model    : ACT (Action Chunking Transformer, LeRobot), chunk size 100, ResNet-18 vision backbone
           (ImageNet-pretrained weights = the only pre-trained component), trained from scratch otherwise.
Training : 200,000 steps, batch 8, learning rate 1e-5 (training-notes/TRAIN.bat).
Dataset  : ieee_pour - 75 teleoperated demonstrations (leader/follower SO-ARM100 arms, both arms),
           101,391 frames at 30 fps, 3 cameras: top, left wrist (RealSense), right wrist (C270 webcam),
           plus the joint positions of both arms. Recorded by the team (training-notes/RECORD.bat, HOW_TO.txt).
Inference: on the laptop wired to the arms only, no network (training-notes/RUN.bat).

models/ and data/: the model (207 MB) and dataset (~100k frames) are larger than GitHub's 100 MB file
limit; they are kept on the team laptop and shared with the judges through Google Drive: https://drive.google.com/drive/folders/18T76qVBEAcmT0PJbPxyNHgqOAVj7Pmiq?usp=drive_link (access on request)
