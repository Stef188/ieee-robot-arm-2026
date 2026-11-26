"""How far does the block move ACROSS the table (top-camera px) while the hand lifts it after a grab (0.7 s = one 'up')?
That sideways/back part of 'up' is removed so 'up' becomes straight up (25 Nov late: the 2.5x higher drop pulled the
gripper back off the bin, and the high start came down diagonally into the block).
    python classical\\measure_lift.py          -> writes "up_px" (du, dv per one 'up') into calib_left/right.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import vision  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


def top(ds, i):
    return cv2.cvtColor((ds[i]["observation.images.top"].permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def blob_of(img, colour, near):
    best = None
    for name, cx, cy, ang, a, w, h in vision.detect(img)["objects"]:
        if name == colour:
            d = np.hypot(cx - near[0], cy - near[1])
            if d < 60 and (best is None or d < best[0]):
                best = (d, cx, cy)
    return None if best is None else np.array(best[1:])


def main():
    ds = LeRobotDataset("local/ieee_colour_v2", root=Path(r"D:\lerobot\data\ieee_colour_v2"))
    an = ds.meta.features["action"]["names"]
    sn = ds.meta.features["observation.state"]["names"]
    for side in ("left", "right"):
        cal = json.loads((HERE / f"calib_{side}.json").read_text())
        si = [sn.index(f"{side}_{j}.pos") for j in ("shoulder_lift", "elbow_flex")]
        moves = []
        for p in cal["points"]:
            e = p["demo"]
            a0, a1 = int(ds.meta.episodes["dataset_from_index"][e]), int(ds.meta.episodes["dataset_to_index"][e])
            act = np.stack([np.asarray(x) for x in ds.hf_dataset.select(range(a0, a1))["action"]])
            st = np.stack([np.asarray(x) for x in ds.hf_dataset.select(range(a0, a1))["observation.state"]])
            g = act[:, an.index(f"{side}_gripper.pos")]
            t, was = None, False
            for k in range(1, len(g)):
                if g[k - 1] > 15:
                    was = True
                if was and g[k] < 12 <= g[k - 1] and k > 20 and np.abs(st[k, si] - st[0, si]).max() > 30:
                    t = k
                    break
            if t is None or t + 27 >= len(g):
                continue
            start = np.array([p["u"], p["v"]])
            b0 = blob_of(top(ds, a0 + t + 6), p["colour"], start)        # block in the jaws, still on the table
            b1 = blob_of(top(ds, a0 + t + 27), p["colour"], start)       # 0.7 s later (one 'up')
            if b0 is not None and b1 is not None:
                moves.append(b1 - b0)
        m = np.array(moves)
        med = np.median(m, 0)
        print(f"{side}: {len(m)} lifts seen | block moves {med.round(1)} px per 'up' (x right, y towards the arms) "
              f"= {np.linalg.norm(med) / 8.3:.1f} cm | spread {m.std(0).round(1)}")
        cal["up_px"] = med.tolist()
        cal["up_px_note"] = "25 Nov late: top-camera move of the block during one 'up' (demo lifts); removed to make 'up' vertical"
        (HERE / f"calib_{side}.json").write_text(json.dumps(cal, indent=1))


if __name__ == "__main__":
    main()
