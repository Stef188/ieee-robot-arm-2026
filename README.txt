TEAM: <TEAM NAME>                         (rename this folder TEAM_NAME to the team name)
IEEE Global Robot Championship 2026 - Robot Arm AI Challenge - Game 1: Pick-and-Place
Submission under Rules & Regulations v1.0 FINAL, 5.4.1.8 - 5.4.1.11.   Repository: <PRIVATE GITHUB URL>

START HERE: TECHNICAL_DOCUMENTATION.md - covers items a-h of 5.4.1.10 (architecture, AI models, datasets, training,
code/weights/instructions, autonomous perception + decision-making + manipulation, third-party declaration,
technical report).

AI MODELS
  GraspNet  - learned grasp-pose model (behaviour cloning from leader-follower demonstrations), used for every grab.
              Weights: Game_01_Round_0x/models/grasp_net_<arm>_<grasp|hover>.npz  (code: code/model.py)
  ShapeNet  - learned shape classifier (CNN, 40k parameters), Round 2.  Weights: Game_01_Round_02/models/shape_net.pt
              (code: code/shape_model.py, code/train_shape_net.py)

FOLDERS
  code/                       full source + launchers (SORT_FAST.bat = Round 1, SORT_SHAPE.bat = Round 2,
                              CHECK_BINS.bat, TEACH_*.bat, AFTER_RUN.bat). config.bat = ports/cameras of our setup.
  Game_01_Round_01/ (Colour Sort)
     data/            calib_left.json, calib_right.json = GraspNet training sets (demo grasps + hand-taught + grasps
                      learned from the robot's own successful runs); grab_points.npz; robot_picks_26nov.jsonl = log
                      of every autonomous pick (used by learn_from_runs.py); sag_comp.json
     models/          grasp_net_*.npz
     training-notes/  grasp_net_*.txt/json (training + leave-one-out evaluation), variants*.txt (model study)
     videos/          see videos/README.txt
  Game_01_Round_02/ (Shape Sort)
     data/            photos.npz = 1,710 labelled crops from the photo session (ShapeNet training set);
                      crops_top_only_1630.npz = demo-derived crops (tried, not used - see documentation c);
                      montage_*.png = label check pictures
     models/          shape_net.pt (+ the same grasp_net_*.npz)
     training-notes/  shape_net_photos_*.txt = training + 5-fold cross-validation of the model in use;
                      shape_net_2026*.txt = earlier attempts on the demo data
     videos/          see videos/README.txt

EXTERNAL / LARGE FILES (raw LeRobot datasets, too large for GitHub)
  ieee_colour_v2      (202 episodes, colour-sorting demonstrations):   <SECURE LINK>
  ieee_shape_clean    (229 episodes, shape-sorting demonstrations):    <SECURE LINK>
  ieee_shape_v2       (33 episodes):                                   <SECURE LINK>
