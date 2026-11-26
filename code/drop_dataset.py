"""Training data for the DROP model (Round 3, moving bins) - 26 Nov.
    python classical\\drop_dataset.py [dataset ...]      (default: ieee_shape_clean ieee_shape_v2 ieee_colour_v2)
In the shape demos the bins were moved between sessions, so the releases happened over bins at many places. For every
release (gripper opens after holding a block): the arm joints at that moment + the bin it was over (found by its rim
in the episode's first and last frames, matched by the arm's shoulder pan, as in shape_dataset.py). Output:
classical\\shape_data\\drops.npz  (arm, bin u/v, bin colour, joints[5], dataset, episode).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import model  # noqa: E402
from shape_dataset import episode_bins, top_img  # noqa: E402


def collect(name, maps):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(f"local/{name}", root=Path(r"D:\lerobot\data") / name)
    sn = ds.meta.features["observation.state"]["names"]
    an = ds.meta.features["action"]["names"]
    rows, memory, skipped = [], {}, {}
    for e in range(ds.meta.total_episodes):
        a0, a1 = int(ds.meta.episodes["dataset_from_index"][e]), int(ds.meta.episodes["dataset_to_index"][e])
        hf = ds.hf_dataset.select(range(a0, a1))
        act = np.stack([np.asarray(x) for x in hf["action"]])
        st = np.stack([np.asarray(x) for x in hf["observation.state"]])
        bins = episode_bins([top_img(ds, a0), top_img(ds, a1 - 1)], memory)
        memory = dict(bins)
        for arm, side in enumerate(("left", "right")):
            m = maps[side]
            g = act[:, an.index(f"{side}_gripper.pos")]
            J = np.stack([st[:, sn.index(f"{side}_{j}.pos")] for j in
                          ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")], 1)
            held = False
            for t in range(1, len(g)):
                if g[t] < 12 <= g[t - 1]:
                    held = True
                if not (held and g[t] > 15 >= g[t - 1]):
                    continue
                held = False
                q = J[t]
                cand = sorted((abs(q[0] - m.pan_geo(b[0], b[1])), c) for c, b in bins.items())
                if not cand or cand[0][0] > 9 or (len(cand) > 1 and cand[1][0] < cand[0][0] + 6):
                    skipped["no clear bin"] = skipped.get("no clear bin", 0) + 1
                    continue
                b = bins[cand[0][1]]
                rows.append((arm, float(b[0]), float(b[1]), cand[0][1], *[float(x) for x in q], name, e))
        if e % 25 == 0:
            print(f"   {name}: episode {e}/{ds.meta.total_episodes}, {len(rows)} releases", flush=True)
    print(f"   {name}: {len(rows)} releases | skipped {skipped}", flush=True)
    return rows


if __name__ == "__main__":
    sets = sys.argv[1:] or ["ieee_shape_clean", "ieee_shape_v2", "ieee_colour_v2"]
    maps = {s: model.load(s) for s in ("left", "right")}
    rows = []
    for s in sets:
        rows += collect(s, maps)
    a = rows
    np.savez_compressed(HERE / "shape_data" / "drops.npz", arm=np.array([r[0] for r in a]), u=np.array([r[1] for r in a]),
                        v=np.array([r[2] for r in a]), colour=np.array([r[3] for r in a]),
                        joints=np.array([r[4:9] for r in a]), dataset=np.array([r[9] for r in a]),
                        episode=np.array([r[10] for r in a]))
    for arm in (0, 1):
        k = [r for r in a if r[0] == arm]
        spots = {(round(r[1] / 20) * 20, round(r[2] / 20) * 20) for r in k}
        print(f"{'left' if arm == 0 else 'right'} arm: {len(k)} releases at {len(spots)} different bin places")
