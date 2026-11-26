"""Collect calibration points from recorded demos: (pixel of the object in the top camera) <-> (follower joint angles
at the moment the human closed the gripper on it). No robot needed.
    python classical\\grab_points.py [dataset ...]      (default: ieee_colour_v2_clean)
For every grab (gripper command goes from open > 15 to < 12), the object that DISAPPEARS from its spot within
1.5 s is the one that was grabbed. Only clear cases are kept (exactly one object of a colour vanished near the arm).
Writes classical\\grab_points.npz: arm (0 left / 1 right), u, v (pixels), joints (5 arm joints, deg), dataset, demo.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vision  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def top_img(ds, idx):
    t = ds[idx]["observation.images.top"]
    return cv2.cvtColor((t.permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def collect(name):
    ds = LeRobotDataset(f"local/{name}", root=Path(r"D:\lerobot\data") / name)
    names = ds.meta.features["observation.state"]["names"]
    anames = ds.meta.features["action"]["names"]
    out = []
    for e in range(ds.meta.total_episodes):
        a0, a1 = int(ds.meta.episodes["dataset_from_index"][e]), int(ds.meta.episodes["dataset_to_index"][e])
        hf = ds.hf_dataset.select(range(a0, a1))
        act = np.stack([np.asarray(x) for x in hf["action"]])
        st = np.stack([np.asarray(x) for x in hf["observation.state"]])
        for arm, side in enumerate(("left", "right")):
            g = act[:, anames.index(f"{side}_gripper.pos")]
            was_open = False
            for t in range(1, len(g)):
                if g[t - 1] > 15:
                    was_open = True
                if not (was_open and g[t] < 12 and g[t - 1] >= 12):
                    continue
                was_open = False
                t_c = t + 6                                    # jaws need a few frames to close
                if t_c + 45 >= len(g) or t < 3:
                    continue
                before = vision.detect(top_img(ds, a0 + max(t - 20, 0)))["objects"]
                after = vision.detect(top_img(ds, a0 + t_c + 45))["objects"]
                gone = [o for o in before if not any(p[0] == o[0] and np.hypot(p[1] - o[1], p[2] - o[2]) < 15 for p in after)]
                if len(gone) != 1:
                    continue
                o = gone[0]
                q = [st[t_c, names.index(f"{side}_{j}.pos")] for j in JOINTS]
                out.append((arm, o[1], o[2], *q, e, o[3]))
        if e % 20 == 0:
            print(f"   {name}: demo {e}/{ds.meta.total_episodes}, {len(out)} grab points", flush=True)
    return out


if __name__ == "__main__":
    sets = sys.argv[1:] or ["ieee_colour_v2_clean"]
    rows, tags = [], []
    for s in sets:
        r = collect(s)
        rows += r
        tags += [s] * len(r)
    a = np.array(rows, dtype=float)
    np.savez(Path(__file__).parent / "grab_points.npz", arm=a[:, 0].astype(int), u=a[:, 1], v=a[:, 2],
             joints=a[:, 3:8], demo=a[:, 8].astype(int), angle=a[:, 9], dataset=np.array(tags))
    print(f"saved {len(a)} grab points: left {int((a[:, 0] == 0).sum())}, right {int((a[:, 0] == 1).sum())}")
