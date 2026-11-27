# Technical Documentation: Bimanual Colour and Shape Sorter

IEEE Global Robot Championship 2026, Robot Arm AI Challenge. Rules & Regulations v1.0 FINAL, sections 5.4.1.8 to 5.4.1.11. Game 1 (Pick-and-Place): Round 1 Colour Sort, Round 2 Shape Sort, Round 3 Dynamic Sort.

Team: `<TEAM NAME>`
Repository: `<PRIVATE GITHUB URL>`
Date: 26 Nov 2026

This document describes the system **as it is actually implemented**. Every number comes from the code, data and logs in this repository.

---

## a. System architecture

### Hardware

| Part | Details |
|---|---|
| Arms | 2 × SO-101 follower arms (LeRobot SO-follower driver), 6 joints each (shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, gripper). Feetech serial-bus servos, USB serial. |
| Leader arms | 2 × SO-101 leader arms. Used **only to record the human demonstrations** (teleoperation), not during autonomous operation. |
| Top camera | Intel RealSense D405 mounted about 93 cm above the table. Colour stream 640×480 at 30 fps. Depth is not used: its noise at that range was measured at about 2.7 cm. |
| Wrist cameras | One per arm. Recorded in the demonstrations but **not used** by the autonomous sorter (see h). |
| Computer | Windows 11 laptop, CPU only. No GPU is needed at run time. |

### Software

| File | Role |
|---|---|
| `vision.py` | Live perception from the top camera. |
| `model.py` | AI model 1, **GraspNet** (grasp poses), plus the arm-map wrapper. |
| `shape_model.py`, `train_shape_net.py` | AI model 2, **ShapeNet** (shape recognition for Round 2) and its training. |
| `shape_photos.py` (`SHAPE_PHOTOS.bat`), `shape_dataset.py` | ShapeNet training data: labelled photo session; crops from the shape demos. |
| `learn_from_runs.py`, `analyse_run.py` (`AFTER_RUN.bat`) | Learning from the robot's own successful grabs, and the per-run report. |
| `train_grasp_net.py` | Training and evaluation of GraspNet. |
| `calib_from_demos.py` | Dataset extraction from the recorded demonstrations. |
| `sort.py` | Decision-making and the manipulation state machine. |
| `hw.py` | Arm and camera drivers (wraps LeRobot `SOFollower` and `pyrealsense2`). |
| `SORT_FAST.bat` / `SORT_SHAPE.bat` / `SORT_DYNAMIC.bat` | Launchers: Round 1 (colour), Round 2 (shape) and Round 3 (dynamic). `SORT_BEST.bat` = slower settings. `CHECK_BINS.bat` = bin check with the camera only. |
| `teach_bin.py`, `teach_grab.py` (`TEACH_BIN.bat`, `TEACH_GRAB.bat`) | Optional hand-taught bin poses and grab spots (see c). |
| `bins_from_good_drops.py` | Rebuilds the bin drop poses from all demo releases. |
| `benchmark_variants.py`, `miss_map.py` | Model-variant benchmark (d) and the map of grab results from the run logs (h). |
| `versions/` | Runnable snapshots of earlier versions, each with its own code, calibration and settings. |

### Data flow (every pick)

```
RealSense colour frame ×3 ──► vision.py: colour segmentation + true shape centre (sub-pixel) + rotation
                                   │  (u, v, angle) of every loose block, position of every bin
                                   ▼
                     sort.py: choose block + arm (decision rules)
                                   │  (u, v)
                                   ▼
             GraspNet (learned, model.py) ──► grasp joints + hover joints (5 per arm)
                                   │  + wrist roll from the block angle, joint-limit clamp
                                   ▼
        sort.py: high start ► descend ► close ► jaw-feedback grip check ► lift ► bin ► open ► rest
                                   │
                                hw.py ──► servo position commands (LeRobot SOFollower)
```

---

## b. AI model

### GraspNet: a learned grasp-pose policy, learned from demonstration

| | |
|---|---|
| **Task** | Map the position of a block in the top-camera image to the arm joint configuration a human operator used to grasp a block there. This is a learned inverse-kinematics / grasp-pose model. No hand-written kinematic model of the arm is used. |
| **Input** | 2 values: the block centre (u, v) in pixels, normalised as ((u−350)/100, (v−330)/100). |
| **Output** | 5 joint angles (shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll base) in degrees, standardised per joint during training. There are two networks per arm, one for the **grasp** pose and one for the **hover** (approach) pose. That makes 4 networks in total. |
| **Architecture** | Multi-layer perceptron 2 → 16 (tanh) → 16 (tanh) → 5, plus a linear skip connection from input to output. Ensemble of 5 members (random seeds 0–4) whose outputs are averaged. |
| **Size** | 415 parameters per member; 2,075 per network; 4 networks (16.6 kB each at 64-bit). |
| **Framework** | NumPy. Forward pass and back-propagation are written by hand in `model.py` (`GraspNet`). Training is deterministic. |
| **Pre-trained components** | None. GraspNet is trained from scratch on the team's own demonstrations. |
| **Weights** | `grasp_net_left_grasp.npz`, `grasp_net_left_hover.npz`, `grasp_net_right_grasp.npz`, `grasp_net_right_hover.npz`. Each file stores the SHA-256 of its training data. If the data changes, the network is retrained automatically. |

### How GraspNet is used at run time

**Position joints.** GraspNet's outputs for shoulder lift, elbow flex and wrist flex are used directly. Together with the shoulder pan, these joints decide where the gripper goes.

**Shoulder pan (which way the arm turns), changed 26 Nov afternoon.** The pan comes from a 4-parameter geometric model fitted to the same demonstrations: pan = a × (direction of the block seen from the arm's base in the top camera) + c, with the base pixel fitted as well (fitted slope a = 0.96 left / 0.90 right, close to the physical 1).
- Leave-one-out pan error: left 1.26° (cubic fit 1.27°, GraspNet 1.84°), right 2.13° (2.08°, 2.61°).
- On held-out grabs nearest the middle of the table, the geometric model was the most accurate: 1.1° / 1.4°, against the cubic fit's 1.4° / 3.2° and GraspNet's 3.1° / 3.7°.
- Outside the demonstrated area, the cubic fit turned *back* (10–15° wrong, about 4–5 cm) and GraspNet over-turned. The geometric model keeps turning the right way, which matters for edge grabs and hand-offs.
- Run logs back this up: grabs where the old guard switched to the cubic fit held only 5 of 35 times, against 12 of 39 for GraspNet.
- `SORT_PAN=model` switches back to the old pan.

**Wrist roll.** The wrist roll comes from a cubic least-squares fit of the same demonstrations, plus the measured block angle:
- The operator turned the wrist inconsistently between demos (right arm: 0–166°). A cube looks the same every 90°.
- The fitted roll is the one validated on the real arms.

**Plausibility guard.**
- For every grasp, the system computes both GraspNet's position joints and the cubic fit's.
- If they disagree by more than 6° on lift, elbow or wrist flex (`SORT_NET_GUARD`), that grasp uses the cubic fit for those joints. Past the edge of the demos, GraspNet's tanh layers flatten out, while the cubic fit continues the trend seen inside.
- This happens mostly at the edges of the demonstrated workspace, where few demos exist.
- Every grasp logs which one was used: `arm pose from: grasp network / hover network (network vs fit X deg)` or `fit (guard)`.
- Coverage over a 1 cm grid of the reachable table area: GraspNet is used at 165 of 208 spots on the left arm and 107 of 188 on the right.

**Roll training target.** The roll is learned relative to the block's rotation: target roll = demonstrated roll − sign × block angle. At run time the measured block angle is added back. The sign (±1) is chosen by leave-one-out on the training data.

### ShapeNet: learned shape recognition (Round 2)

| | |
|---|---|
| **Task** | Classify each block seen by the top camera as cube, cylinder, dodecahedron or pentagonal prism. The shape gives the bin (rulebook 5.8.1.46: cube → red, cylinder → blue, dodecahedron → green, pentagonal prism → yellow). |
| **Input** | A 64×64 px crop around the block (its centre 40×40 px scaled up), as two channels: contrast-normalised greyscale and the block's silhouette mask. The network never sees colour, so it cannot learn a colour shortcut (Round 2 sorts regardless of colour). |
| **Architecture** | Ensemble of 3 identical convolutional networks (seeds 0-2), each: 4 blocks of 3×3 convolution + batch norm + ReLU + 2×2 max-pool (16 → 32 → 48 → 48 channels), global average pooling, dropout 0.3, linear layer to 4 classes. 40,084 parameters. Trained from scratch, no pre-trained weights. |
| **Run time** | About 1 ms per crop on the CPU. Each block is classified in 8 views (4 rotations × mirror) and over several camera frames, and the probabilities are averaged over all looks at that block. |
| **Use** | Points-maximising decision: a wrong bin costs the block's own points (5.8.1.63/72), so the bin is argmax P(shape) × points(shape, size), and a block is placed only when the expected gain EV = 2·P_b·pts_b − Σ_s P_s·pts_s is positive (≥ 0.5 normally, ≥ 0.2 when the arm has nothing better, ≥ 0 as a last resort). Picks are taken in order of EV, so pentagonal prisms (5 points) come first, then dodecahedra and cylinders. |
| **Weights** | `shape_net.pt`. |

### DropModel: drop pose for a bin anywhere (Round 3, moving bins)

| | |
|---|---|
| **Task** | Round 3 (5.8.1.52–55): a judge moves the bins while the robot works. The arm must drop into a bin wherever the camera sees it. |
| **Data** | 1,053 demo releases (`drop_dataset.py`): the joints at every moment a demonstrator let go of a block over a bin, and where that bin was in the top camera. The bins stood at 14 (left arm) and 17 (right arm) different places across the demo sessions. File: `shape_data/drops.npz`. |
| **Model** (`drop_model.py`) | Per arm, least squares: pan = a × (direction of the bin from the fitted arm base) + c; lift, elbow and wrist flex = quadratic in the distance from the base plus linear in the direction. It only answers inside the range of distances and directions it has seen. |
| **Evaluation** | Leave one bin place out: pan error 1.7° (left) and 1.6° (right) at a never-seen place. The other joints (7–22°) are within the demonstrators' own spread at the same place (10–21°), because different arm postures reach the same spot. |
| **Use** (`SORT_DYNAMIC.bat`) | A bin standing on one of the known spots uses that spot's tested pose; anywhere else this model's pose. Before going down, the block is checked in the newest camera picture (moved = choose again, not a miss). Before letting go, the bin is looked up again (moved = re-aim). No bin memory in Round 3. |

### Learning from the robot's own experience

Every pick is logged (`runlog/picks.jsonl`): block position, the joints when the gripper closed, and the result. `learn_from_runs.py` adds each grab that held its block on the first try to the grasp-pose training set (weight 2), with the deep-grip offset removed, and GraspNet retrains (about 3 s). After the first evening of runs this added 33 verified grasps (23 left, 10 right).

### Other models developed by the team (not used in this sorter)

Before this system, the team trained end-to-end visuomotor policies on the same demonstrations:
- **ACT** (Action Chunking Transformer, LeRobot), trained on cloud GPUs;
- **SmolVLA** (LeRobot, fine-tuned from `lerobot/smolvla_base`).

They are described in h because their results drove the final design.

---

## c. Dataset

| | |
|---|---|
| **Origin** | Recorded by the team. Teleoperation of the two follower arms with the two leader arms (LeRobot `lerobot-record`), colour-sorting task: loose blocks into the bin of the same colour. |
| **Raw dataset** | LeRobot dataset `ieee_colour_v2`, 30 fps. Contents: top camera + 2 wrist cameras (video), 12-D follower joint state, 12-D leader action. |
| **Extraction** (`calib_from_demos.py`) | For every episode and arm, the **first real grasp** is located: the gripper command goes from open (>15) to closed (<12) while the arm is reached out. Each grasp gives one sample:<br>• **input** = the top-camera position of the block of the matching colour on that arm's half of the table, taken from the episode's first frame with `vision.py`;<br>• **grasp target** = follower joints 0.2 s after the close starts;<br>• **hover target** = follower joints 0.5 s before the close. |
| **Filtering** | Only samples where exactly one block qualifies (no ambiguous pairing) are kept. Points inconsistent with their neighbours are dropped. |
| **Final size** | **Left arm 104 samples**, **right arm 65 samples**. Each sample comes from a different demonstration episode. |
| **Bin drop poses** | Taken from the same demos: the joints at every moment an arm released a held block over a bin (`bins_from_good_drops.py`). Each bin pose is the median of all its releases: left red 75, left green 92, right yellow 53, right green 30, right blue 13 (263 in total). |
| **Hand-taught additions (optional)** | `teach_grab.py` adds grab spots that a person guides by hand where an arm misses. Each counts 3× in training. `teach_bin.py` fixes a bin pose by hand. Every added point is marked `"taught": true` in the JSON. |
| **Preprocessing** | Inputs are normalised (see b). Outputs are standardised per joint. Wrist roll is made relative to the block angle (folded to ±45° for 90°-symmetric blocks). |
| **Augmentation** | None. |
| **Hand-taught right-arm grasps** | 9 grasps taught by hand on 25 Nov and 5 on 26 Nov (weight 3), added because the right arm had fewer demos (65) and its demonstrated wrist posture differed from a working hand grasp by about 10°. |

### ShapeNet dataset

| | |
|---|---|
| **Photo session (used)** | `shape_photos.py`: the top camera photographs one shape at a time, all 8 objects of that shape (4 colours × small and large) in about 15 arrangements, standing and lying. Every detected block in a photo therefore has a certain label. A background photo taken first excludes the blocks already in the bins. 76 photos, **1,710 labelled crops** (cube 417, cylinder 357, dodecahedron 327, pentagonal prism 609), every colour in every class. File: `shape_data/photos.npz`. |
| **Demo crops (tried, not used)** | `shape_dataset.py` cut 468 crops from the 262 shape demos (`ieee_shape_clean`, `ieee_shape_v2`), labelled by the bin the operator dropped each block into. Two problems made them unusable as the main source: label noise (bins full of mixed blocks, moved between sessions), and a leak (each arm only handled two of the four shapes, so position and arm predict the label). |
| **Augmentation** | Random rotation (0–360°), mirror, ±10 % scale, ±3 px shift, brightness ±25 and contrast ×0.75–1.25. |
| **Files** | `calib_left.json`, `calib_right.json` (the extracted training sets) and the raw LeRobot dataset `ieee_colour_v2`. The raw dataset is too large for GitHub; see the links in e. |

---

## d. Training process

| | |
|---|---|
| **Command** | `python classical\train_grasp_net.py` |
| **Optimiser** | Adam (β1 0.9, β2 0.999), full batch, learning rate 0.02, 3000 steps. |
| **Loss** | Mean-squared error on standardised joint targets, plus L2 weight decay 2e-3 on the weight matrices. |
| **Ensemble** | 5 members (seeds 0–4), averaged. |
| **Time** | About 10 s per arm on a laptop CPU. |
| **Hyperparameter selection** | Leave-one-out grid over hidden size {8, 16, 32}, weight decay {0, 1e-4, 1e-3, 2e-3, 1e-2, 3e-2}, steps {1000, 1500, 2000, 3000} and ensemble size {1, 3, 5}. The best was hidden 16, weight decay 2e-3, 3000 steps, 5 members. |
| **Evaluation** | Leave-one-out: each demonstrated grasp is predicted by a network trained without it. The metric is the mean absolute joint error in degrees. The baseline is a cubic least-squares polynomial fit on the same splits (the mapping used before GraspNet). |
| **Logs** | `training_logs/grasp_net_<date>.txt` and `.json`. |

### Results (leave-one-out, mean absolute joint error)

| Arm | Samples | GraspNet | Cubic baseline |
|---|---|---|---|
| Left | 104 | 7.97° (median 7.38°) | 7.83° (median 6.97°) |
| Right | 65 | 9.64° (median 8.56°) | 10.47° (median 8.51°) |

Per joint, leave-one-out:

| Arm | Model | Pan | Lift | Elbow | Wrist flex | Wrist roll |
|---|---|---|---|---|---|---|
| Left | GraspNet | 1.74° | 4.20° | 7.44° | 6.44° | 20.03° |
| Left | Cubic | 1.27° | 4.12° | 7.25° | 6.41° | 20.07° |
| Right | GraspNet | 2.69° | 5.87° | 7.79° | 6.87° | 24.98° |
| Right | Cubic | 2.08° | 6.69° | 8.61° | 7.60° | 27.35° |

Source: `training_logs/grasp_net_20261126_0022.txt`.

### ShapeNet results (5-fold cross-validation, grouped by photo: test photos never seen in training)

| | Accuracy |
|---|---|
| ShapeNet, 3 networks + 8 views (used) | **79.1 %** (cube 91 %, cylinder 87 %, dodecahedron 87 %, pentagonal prism 62 %); 91 % correct when at least 70 % sure |
| ShapeNet, 1 network | 77.0 % (cube 90 %, cylinder 84 %, dodecahedron 81 %, pentagonal prism 62 %) |
| Hand-written baseline (roundness, corner count, solidity, aspect, size → nearest class mean) | 69.4 % |
| Chance (largest class) | 35.6 % |
| Per colour (no colour shortcut) | red 76 %, yellow 67 %, green 81 %, blue 83 % |
| When at least 70 % sure | 87.6 % correct, covering 62 % of blocks |

Training: Adam, learning rate 2e-3, weight decay 1e-4, 60 epochs, batch 32, cosine schedule, class-balanced cross-entropy. Log: `training_logs/shape_net_photos_<date>.txt`. Earlier attempts on the demo crops only reached 46–49 % (with a hand-written baseline of 52–56 %).

**Live caveat:** on the real table, with blocks touching and the arms in view, accuracy is lower than on the photos. In a 13-block live run at least 2 placements were wrong, both lying cylinders taken for dodecahedra. This led to the stricter per-shape thresholds and the 8-view averaging.

### Compression and model-type study (`benchmark_variants.py`, `benchmark_variants2.py`)

Ten compressed variants of GraspNet were evaluated with the same leave-one-out test. The error shown is on the position joints. Log: `training_logs/variants_20261126_0257.txt`.

| Variant | Left | Right | Size |
|---|---|---|---|
| Current: 5-net ensemble, float64 | 4.95° | 5.81° | 16.6 kB |
| Single net | 4.97° | 5.67° | 3.3 kB |
| Distilled 2-8-8 net | 5.10° | 6.01° | 0.6 kB |
| float16 | 4.95° | 5.81° | 4.2 kB |
| INT8 (symmetric per-tensor) | 4.97° | 5.82° | 1.4–1.8 kB |
| INT4 | 6.11° | 7.02° | 0.5 kB |
| 50% magnitude-pruned + fine-tuned | 4.96° | 5.91° | 4.5 kB |
| 90% magnitude-pruned + fine-tuned | 8.21° | 9.46° | 1.5 kB |
| Lookup table (2 px grid, int16) | 4.95° | 5.81° | 266 kB |
| + rotation and colour inputs | 5.25° | 6.70° | 36.9 kB |

- INT8 and distillation shrink the model 10–27× with no meaningful loss.
- INT4 and 90% pruning are where accuracy breaks down.
- No variant beats the current model, which shows that the remaining error comes from the demonstration data, not from model capacity.
- Variants 1–8 can be run on the arms with `SORT_NET=<name>`; their weights are in `model_variants/`.

A second study compared other model types with the same test (`benchmark_variants2.py`, log `training_logs/variants2_20261126_0309.txt`). Error is on the position joints, left / right arm:
- **On a par with GraspNet (4.9–5.0° / 5.6–5.9°):** wider (64) net, deeper (3-layer) net, ReLU net, 10-net ensemble, outlier-robust (Huber) net, and a Gaussian process. The Gaussian process uses a third of the parameters.
- **Worse:** a dropout net (5.49° / 6.49°), k-nearest-neighbours (5.29° / 6.08°), boosted trees (5.67° / 7.11°) and a random forest (6.50° / 7.67°).
- The Gaussian process's predictive uncertainty only weakly predicted its own errors (correlation +0.05 left, +0.25 right).
- **Conclusion:** model choice changes the result by less than 0.5°, so the error is set by how consistent the demonstrations are. GraspNet was kept.

A large part of this error is genuine variation in the human demonstrations. The operator grasped similar positions with different wrist and elbow postures, so no model can predict an individual demo exactly.

---

## e. Code, configuration, weights and instructions

**Run (autonomous):**
1. Connect both follower arms and the top camera.
2. Run `classical\SORT_BEST.bat` [seconds, default 300].
3. Ctrl+C stops the run and returns the arms to rest.

**Dry run (no motors, no camera):** `classical\SORT.bat dry`

**Retrain the AI model:** `python classical\train_grasp_net.py` (writes weights and logs).

**Rebuild the dataset from demos:** `python classical\calib_from_demos.py ieee_colour_v2`

**Configuration:** environment variables at the top of `SORT_FAST.bat` / `SORT_BEST.bat` / `sort.py`:
- speeds, trims, `SORT_BOTH` (both arms at once);
- `SORT_GRASP` (`net` = GraspNet, the default; `poly` = baseline, for comparison only);
- `SORT_NET=<variant>`: run one of the compressed model variants from d (for example `5_int8`).

Every run logs `"<arm> arm grasp-pose model: neural network (GraspNet)"` to `sort_log.txt`, and saves pictures of each pick to `runlog/`.

**Round 2 (shape sort):** `classical\SORT_SHAPE.bat` (same as Round 1, plus ShapeNet).

**After every run:** `classical\AFTER_RUN.bat`, which reports on the run and learns from its successful grabs.

**Retrain ShapeNet:** `python classical	rain_shape_net.py` (`SHAPE_DEMOS=0` = photo session only, as used).

**Large files:** `<SECURE LINK TO ieee_colour_v2 DATASET>`

---

## f. Autonomous perception, decision-making and manipulation

### Perception (live, every pick)
`vision.py`, top camera:
- HSV colour segmentation of red, yellow, green and blue.
- Table mask and source-area mask.
- Large blobs are bins; small blobs are loose blocks.
- Each block's **true shape centre** comes from the moments of its anti-aliased filled contour at 4× sub-pixel resolution. This works for any shape, not just a bounding box. It is averaged over 3 consecutive frames.
- Rotation comes from the minimum-area rectangle.
- A block counts if its centre lies in the source area. The outline itself is never cut at the border, because a sliced outline shifts its centre. Blobs from 80 to 1400 px are blocks.
- **Shape rule:** an outline stretched 1.5× or more is a *long* block (a lying cylinder). The jaws are then turned to close across its short axis.

### Decision-making
`sort.py`, `choose()`:
- Only blocks that have a bin of the same colour are considered.
- A block is picked only if it is inside an arm's demonstrated workspace, and not already on a bin.
- **Both arms work at the same time** (`run_both`). Each arm runs its own loop. It never takes the block the other arm is going for.
- **Who does which block:** the arm that has the block's bin and can reach it sorts it. If that arm cannot reach the block but the other arm can, the other arm makes a **hand-off**: it puts the block down on a free spot in the middle that both arms reach, and the bin's arm sorts it from there. The two demonstrated areas do not overlap (left arm up to x = 345 in the camera, right arm from x = 376), so the spots lie at the edge of both. `TEACH_MIDDLE.bat` teaches that strip to both arms.
- **Collision rule (no-overlap areas):** each pick reserves the area of table its arm sweeps: the polygon around the arm's base, the block, its bin or hand-off spot, and the arcs it swings through between them. The other arm may work at the same time as long as its own area stays at least 75 px (about 9 cm) away. Otherwise it takes another block, or waits and looks again before moving. A waiting arm's area is kept free, so neither arm is starved. An arm keeps its area until it is really back near its rest pose (within 12°). History: a left–right strip of 3.6 cm let the arms work 8.5 cm apart in the middle; at 6 cm, the arms waited 227 s of a 5-minute run, because the strip blocked picks that were far apart front to back.
- **Bin spots, not bin colours (26 Nov):** each stored drop pose belongs to a bin *spot* in the camera picture. A bin seen within 35 px (about 4 cm) of a stored spot uses that spot's drop pose, drop height and learned sag correction, whatever its colour. Swapping bins between spots, as in the organisers' example picture where red and yellow change places compared with our lab, needs no new data. A bin that is not on any stored spot is not used yet; a learned drop-pose model for arbitrary bin positions is the planned next step.
  `bin_layouts/check_layout.py all` verifies all 24 orders of the four bins on the four spots (`ALL_24_LAYOUTS.txt`). `CHECK_BINS.bat` checks the real bins with the top camera on setup day, without moving the arms.
- **Bin check at start:** a bin pose whose shoulder turn differs by more than 20° from the turn the arm's own demo grabs predict for that direction is not used. This caught the right arm's "green" bin pose, which pointed at the yellow bin.
- **Space around bins:** a block whose centre is within 15 px (about 1.8 cm) of a bin's real rotated outline, which means practically touching it, is left alone. Blocks within 30 px are taken after everything else. A 30 px exclusion was tried and left the arms idle, because most of the excluded blocks were 2–3 cm from a bin. Every drop comes in 1.5 lift heights above the bin and never opens lower than 0.5 lift heights above the demonstrated release pose. If the arm is stopped on the way down, it opens there instead of pushing.
- **Marked area (black line):** only blocks whose centre is at least 5 px inside the inner edge of the black line are taken (`vision.SOURCE_POLY`, measured from the top camera). A block within 30 px of the line is tried at most twice, because every miss can push it over the line. A retry nudges the grab only in a direction that keeps it 3 cm from other blocks and inside the line. On 26 Nov a nudge towards a neighbouring red cube made the right arm grab the red cube and drop it in the yellow bin. Hand-off spots lie at least 3 cm inside the line.
- **Arm bases:** Blocks right by an arm's base are no longer left until last (`SORT_NEAR_BASE_LAST=1` brings that back), because the drag fix keeps them from being swept away.
- **Passive scanning:** the top camera is read continuously by its own thread. An arm about to choose takes the newest picture taken after it asked.
- **No idle arm:** every 0.4 s, an arm with nothing to do looks again. If it finds nothing normal, it looks *wider* at once: blocks up to 40 px outside its demonstrated area instead of 25, and hand-off spots with less free space around them. Every 5 s an idle arm logs why each visible block is not its job (too close to a bin, out of reach, the other arm's job, no free hand-off spot).
- **Left arm drops (26 Nov):** at the green bin the arm turns to the bin's real centre (geometric pan, 13.7° instead of the demo median 9.9°). The left arm opens lower than the right: 0.3 lift heights above the demonstrated release pose at the green bin and 0.5 at the red bin, coming in high first. Peter saw the left arm drop left of and short of the green bin.
- **Up before sideways:** after a grab, a miss or a put-down, the arm must really reach the high pose (within 5°) before it travels. On 26 Nov it set off for home while still low, because of servo lag at 100°/s, and dragged blocks over the black line towards its base.
- **Sag correction at the bins:** stretched out over a bin, the arm sags under its own weight and the block's, and it stopped 3–6° short at every drop. After each drop the shortfall is learned per arm and bin, and the next drop aims past the pose by that much (at most 6° per joint, saved in `sag_comp.json`).
- **Grip check:** the gripper's own position is watched while it closes. Fully closed (below 7) means a miss, detected immediately. Stopped part-way and steady means a block is held.
- **After a miss** the arm moves on to the next block. A missed block comes round again when its turn comes. Each retry nudges the grab by 1 cm (further, nearer, left, right in turn). Blocks are never dropped for good, so both arms keep working until the timer ends.

### Manipulation
- **GraspNet** gives the grasp and hover position joints for the chosen block. The wrist roll and the plausibility guard work as described in b. These are joined into a trajectory with fixed lift offsets.
- The sequence:
  1. rest → high above the block (2 lift heights);
  2. settle there, then straight down at 16°/s to a **deep grip**: 0.2 lift heights (about 1 cm) below the demonstrated grab height, so tall shapes sit deep in the jaws (`SORT_DEEP`). If the table stops the arm first, it closes there;
  3. close, with the grip check described above;
  4. lift → high above the bin;
  5. straight to the drop pose **high above the bin**, wait until still, and just open. The arm never goes down into the bin, because going down to the demonstrated release pose made the arms hit the bins (26 Nov). The drop pose is either a pose hand-taught with `TEACH_BIN.bat`, about 5 cm above the middle of the bin and used exactly, or one lift height above the demonstrated release pose (that one lands about 4 cm short of the bin centre);
     (hand-off instead: high above the middle spot → down to just above the table → open → up → rest);
  6. wait until still → open → back up → rest.
- Travel, lift and bin moves are fast (up to 100–120°/s in `SORT_FAST`). The final approach stays slow, because servo lag at speed costs about 1 cm.
- Every move goes through the LeRobot `SOFollower` driver, with rate-limited joint targets and joint-limit clamping.

**No human input during operation.** The human role is limited to recording the demonstrations beforehand.

---

## g. Third-party declaration

| Item | Use | Licence |
|---|---|---|
| LeRobot (Hugging Face) 0.6.2 | Robot drivers (`SOFollower`), dataset recording/format, and ACT / SmolVLA training for the earlier models | Apache-2.0 |
| `lerobot/smolvla_base` | Pre-trained VLA, fine-tuned in an earlier experiment; not used by this system | Apache-2.0 |
| OpenCV | Image processing in `vision.py` | Apache-2.0 |
| NumPy | GraspNet, maths | BSD-3 |
| pyrealsense2 (Intel RealSense SDK) | Top camera | Apache-2.0 |
| PyTorch | Only inside LeRobot for the earlier ACT/SmolVLA work | BSD-3 |
| AI coding assistant (Anthropic Claude) | Helped write and debug the code and this document, under the team's direction | n/a |

No external datasets were used. All demonstrations were recorded by the team.

---

## h. Technical report: design decisions, development, challenges, limitations

### Development history
1. **End-to-end ACT policies** were trained on the colour demonstrations, up to 200k steps on cloud GPUs. They worked partly. On the real arms, however, they often missed blocks by about 1 cm at positions not covered by the demos (measured with an aim test). They also hovered over empty spots.
2. **SmolVLA** was fine-tuned on the same data. Its offline action error looked good (1.40), but it failed to pick on the real arms.
3. The team therefore **split the problem**:
   - explicit, measurable perception (exact block centre in the camera);
   - a **small learned model** that only has to learn the mapping from camera position to arm pose, from the same human demonstrations.

   This needs far less data than learning the whole task from pixels. It is also easy to check, because each sample's error can be measured.

### Decisions that were tested on the arms
- **Wrist-camera re-centring was removed.** Logged runs showed the first top-camera-based position was already centred. The extra wrist-camera corrections moved the gripper off the block, so the system uses the top camera only.
- **Jaw-feedback grip check** instead of a camera check. The camera check falsely reported misses and dropped held blocks.
- **Bin poses from real releases only.** Averaged poses had included false release moments.
- **True-centre detection** replaced bounding-box centres. On synthetic shapes, the error dropped to about 0.1 mm in the image.
- **A fast final approach was rejected.** Descending at 50°/s with a loose arrival check missed 9 of 10 grabs, so the last approach is slow again.
- **Drop pose.** Adding a lift to the bin pose, to clear the rim, also pulled the gripper 3–5 cm back towards the arm. In photos, every drop landed short of its bin. Descending to the demonstrated release pose fixed that but made the arms hit the bins, so drops are now made high, and the exact pose over each bin is hand-taught (`TEACH_BIN.bat`).
- **Lab test, 26 Nov afternoon (9 sorted in 5 min):** blocks the bin's arm could not reach were left on the table, blocks missed twice were never tried again (so the arms stood still before the time was up), every pick near the middle stopped the other arm, and one bin pose pointed at the wrong bin. This led to the hand-off, the no-overlap strips, the never-give-up retries and the bin check.
- **Vision blind spots.** A mask meant to hide the arm's cables also hid blocks right in front of the left arm, so it was reduced to the cable area only.

### Limitations
- The learned workspace is limited to where the demonstrations grasped, plus about 2.5 cm of margin.
- **Measured from the run logs** (`miss_map.py`, both-arm runs of 26 Nov): grabs at the edge of the demonstrated area succeed far less often (left 1/14, right 5/22) than well inside it (left 2/7, right 6/16). That is why edge spots can be taught by hand.
- Bins must stay roughly where they were in the demonstrations.
- Tall shapes (pentagonal prism, dodecahedron) can slip: their visible top is offset by camera parallax, and they are less stable in the jaws. The deep grip (26 Nov) is meant to help.
- Hand-off spots lie at the edge of both arms' demonstrated areas until the middle strip is taught (`TEACH_MIDDLE.bat`), so grabs there are less reliable.
- Perception is colour-threshold based, so it is sensitive to strong lighting changes.
- The right wrist webcam is overexposed. (This does not affect the sorter, which does not use the wrist cameras.)

### Original contributions
- The demonstration dataset.
- Automatic extraction of grasp samples from raw teleoperation episodes.
- GraspNet and its evaluation against a baseline.
- Sub-pixel true-centre detection.
- The complete autonomous bimanual sorting pipeline.

---

# Game 2: Two-Arm Coordination (Pouring, Folding): end-to-end imitation learning

**a. Architecture.** The same two SO-ARM100 follower arms, laptop and cameras as Game 1. Three cameras: top,
left wrist (Intel RealSense) and right wrist (Logitech C270). During recording, two leader arms are used for
teleoperated demonstrations only. In a scored run, nobody touches anything: the policy reads the 3 camera images
and the 12 joint positions at 30 Hz and outputs the next joint targets for both arms. Everything runs on the
laptop wired to the arms, with no network (5.4.2.2, 5.4.5).

**b. Model.** ACT (Action Chunking with Transformers) from the open-source LeRobot library (Hugging Face).
It is a transformer encoder-decoder with a CVAE that predicts a chunk of 100 future joint targets for both arms
from the images and joint state. The vision backbone is ResNet-18 with ImageNet weights, the only pre-trained
component; everything else is trained from scratch on our own demonstrations.

**c. Datasets.** Both were recorded by the team by teleoperation with leader/follower arms, in LeRobot format
(parquet + video), at 30 fps with 3 cameras plus joints.
- `ieee_pour`: 75 episodes, 101,391 frames.
- `ieee_fold_new_clean`: 43 episodes, 55,724 frames. Only demonstrations with the final, deeper cloth grip are
  used, because the earlier demos with a different grip are inconsistent. Idle frames at the start and end are
  trimmed (`Game_02_Round_02/training-notes/CLEAN_NEW.bat`).
- Augmentation: LeRobot's image augmentation is on (random brightness, contrast, saturation, hue and
  sharpness), plus varied object and cloth positions across the demos.

**d. Training.** AdamW, learning rate 1e-5, batch 8, chunk 100.
- Pouring: 200,000 steps.
- Folding: 200,000 steps (150k on a cloud L40S GPU + 50k on the laptop). Final L1 loss 0.060; offline mean
  joint error on held-out frames 1.39 deg (1.65 deg at 150k).
- Evaluation: offline joint error plus timed runs on the real arms scored with the official rules.
- Training scripts: `Game_02_Round_0x/training-notes/TRAIN*.bat`.

**e. Code and weights.** LeRobot (open source) plus our `RECORD`/`TRAIN`/`RUN` scripts in each round's
`training-notes/`. The weights (207 MB per model) are above GitHub's file limit, so they are shared with the judges on Google Drive:
https://drive.google.com/drive/folders/1gd8YXZgm0Au0U2eF7QpH1w4wVOUOklzp (access on request).

**f. Autonomy.** Perception and decision-making are learned end to end. The policy sees the cup/cloth and both
grippers in the camera images and outputs both arms' motions; there is no hand-written trajectory.

**g. Third party.** LeRobot (Apache-2.0), PyTorch, torchvision ResNet-18 ImageNet weights, OpenCV, pyrealsense2.

**h. Limitations.**
- Folding: the policy cannot recover a dropped cloth, because no demonstration shows a recovery.
- Grasps are tightly clustered around the demonstrated positions.
- Results depend on lighting similar to the recordings.
