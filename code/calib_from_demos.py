"""Build calib_left.json / calib_right.json for the sorter FROM THE RECORDED DEMOS - no hand calibration (25 Nov).
    python classical\\calib_from_demos.py [dataset]        (default ieee_colour_v2)
For every demo and arm, its FIRST grab (gripper command open > 15 -> < 12):
  object   = the loose object (top camera, first frame) of the matching colour on THAT ARM'S HALF of the table
             (LEFT arm = LEFT half of the picture, base x~242; RIGHT arm = right half, base x~458 - checked 25 Nov from
             the recordings: when only the left arm moves, the picture changes around x=240;
             only kept if exactly one object qualifies -> no wrong pairings)
  grasp    = follower joints 0.2 s after the close starts (jaws on the object)
  hover    = follower joints 0.5 s BEFORE the close (approaching from above)
  wrist    = where the object sits in that arm's wrist camera at the hover moment (fine-alignment target)
  bin pose = follower joints when that arm next lets go (gripper opens again) = above the bin of the object's colour
Rest pose = median start pose of the demos. Points far from their neighbours (inconsistent demos) are dropped.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import vision  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

AXIS = 350.0
JN = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def bgr(t):
    return cv2.cvtColor((t.permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "ieee_colour_v2"
    ds = LeRobotDataset(f"local/{name}", root=Path(r"D:\lerobot\data") / name)
    an = ds.meta.features["action"]["names"]
    sn = ds.meta.features["observation.state"]["names"]
    out = {s: {"side": s, "source": f"demos of {name}", "points": [], "bins_raw": {}, "starts": [],
               "time": time.strftime("%Y-%m-%d %H:%M")} for s in ("left", "right")}
    for e in range(ds.meta.total_episodes):
        a0, a1 = int(ds.meta.episodes["dataset_from_index"][e]), int(ds.meta.episodes["dataset_to_index"][e])
        hf = ds.hf_dataset.select(range(a0, a1))
        act = np.stack([np.asarray(x) for x in hf["action"]])
        st = np.stack([np.asarray(x) for x in hf["observation.state"]])
        det = vision.detect(bgr(ds[a0]["observation.images.top"]))
        for side in ("left", "right"):
            si = [sn.index(f"{side}_{j}.pos") for j in JN]
            out[side]["starts"].append(st[0, si].tolist())
            g = act[:, an.index(f"{side}_gripper.pos")]
            was_open, t = False, None
            for k in range(1, len(g)):
                if g[k - 1] > 15:
                    was_open = True
                if was_open and g[k] < 12 <= g[k - 1] and k > 20:
                    was_open = False
                    # a real grab: the arm has reached out (lift/elbow far from the start pose), not a close at home
                    if np.abs(st[k, si[1:3]] - st[0, si[1:3]]).max() > 30:
                        t = k
                        break
            if t is None or t + 6 >= len(g):
                continue
            if t + 27 < len(g):          # the lift right after the grab (0.7 s) = this arm's "straight up" in joints
                out[side].setdefault("lifts", []).append((st[t + 27, si] - st[t + 6, si]).tolist())
            half = [o for o in det["objects"] if (o[1] < AXIS) == (side == "left")]
            if len(half) != 1:
                continue
            o = half[0]
            colour = o[0]
            grasp, hover = st[t + 6, si], st[t - 15, si]
            # wrist target: just BEFORE the jaws close (object centred between them) - the sorter aligns at that height
            wimg = bgr(ds[a0 + t - 2][f"observation.images.{side}_wrist"])
            wrist = vision.find_blob(wimg, colour, near=(0.5, 0.6), min_frac=0.003, max_frac=0.5)
            out[side]["points"].append({"u": float(o[1]), "v": float(o[2]), "angle": float(o[3]), "area": float(o[4]),
                                        "colour": colour, "grasp": grasp.tolist(), "hover": hover.tolist(),
                                        "wrist": None if wrist is None else [float(x) for x in wrist], "demo": e})
            # bin pose: the next let-go after this grab (gripper opens by > 6 from its tightest point)
            low = g[t]
            for k in range(t, len(g)):
                low = min(low, g[k])
                if g[k] - low > 6:
                    if colour in det["bins"]:
                        out[side]["bins_raw"].setdefault(colour, []).append(
                            (det["bins"][colour][0], det["bins"][colour][1], st[k, si].tolist()))
                    break
        if e % 25 == 0:
            print(f"   demo {e}/{ds.meta.total_episodes}: left {len(out['left']['points'])} / right "
                  f"{len(out['right']['points'])} points", flush=True)
    for side, d in out.items():
        # 25 Nov: "above the object" = grab pose + this arm's AVERAGE lift (shoulder lift / elbow / wrist flex only; pan,
        # wrist roll and gripper unchanged). The pose 0.5 s before the grab was at grab height (the hand came in low
        # from the side), so the arm swept into the object instead of coming down from above.
        up = np.mean(np.array(d.pop("lifts", [[0] * 6])), 0)
        up[[0, 4, 5]] = 0.0
        d["up"] = up.tolist()
        for p in d["points"]:
            p["hover"] = (np.array(p["grasp"]) + up).tolist()
        print(f"{side}: 'up' (joint change for ~0.7 s of lifting) = {np.round(up[:4], 1)}")
        pts = d["points"]
        # drop inconsistent points: grasp pose far from what its 5 nearest neighbours (in the picture) suggest
        uv = np.array([[p["u"], p["v"]] for p in pts])
        G = np.array([p["grasp"][:4] for p in pts])
        keep = []
        for i in range(len(pts)):
            dist = np.linalg.norm(uv - uv[i], axis=1)
            nn = np.argsort(dist)[1:6]
            keep.append(np.abs(G[i] - np.median(G[nn], 0)).max() < 20)
        d["points"] = [p for p, k in zip(pts, keep) if k]
        d["rest"] = np.median(np.array(d.pop("starts")), 0).tolist()
        d["bins"] = {}
        for colour, rows in d.pop("bins_raw").items():
            if len(rows) >= 2:
                q = np.median(np.array([r[2] for r in rows]), 0)
                d["bins"][colour] = {"u": float(np.median([r[0] for r in rows])), "v": float(np.median([r[1] for r in rows])),
                                     "hover": q.tolist(), "n": len(rows)}
        (HERE / f"calib_{side}.json").write_text(json.dumps(d, indent=1))
        nw = sum(1 for p in d["points"] if p["wrist"])
        print(f"{side}: {len(d['points'])} grab points kept (of {len(pts)}), {nw} with the object seen by the wrist "
              f"camera, bins {sorted(d['bins'])} -> calib_{side}.json")


if __name__ == "__main__":
    main()
