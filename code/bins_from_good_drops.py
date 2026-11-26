"""Rebuild the bin drop poses from GOOD demo drops only (26 Nov).
    python classical\\bins_from_good_drops.py [dataset]      (default ieee_colour_v2)   -> prints, writes bins_check.json
    python classical\\bins_from_good_drops.py --apply        -> also writes the new poses into calib_left/right.json
For every grab -> let-go in every demo (arm reached out, block really held):
  bin      = the current bin whose pose is nearest (pan / lift / elbow) to the let-go pose
  GOOD     = 1.5 s after the let-go, NO extra loose block of that colour lies next to the bin (= it went in)
New pose = median of the GOOD let-gos (the old one stays as bins_before_gooddrops.<colour>).
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
from calib_from_demos import bgr, JN  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

APPLY = "--apply" in sys.argv
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
name = ARGS[0] if ARGS else "ieee_colour_v2"
ds = LeRobotDataset(f"local/{name}", root=Path(r"D:\lerobot\data") / name)
an = ds.meta.features["action"]["names"]
sn = ds.meta.features["observation.state"]["names"]
calib = {s: json.loads((HERE / f"calib_{s}.json").read_text()) for s in ("left", "right")}
ref = {s: {c: np.array((b.get("hover") if not b.get("taught") else b["hover"])[:3]) for c, b in calib[s]["bins"].items()}
       for s in calib}
AFTER = 45          # frames (1.5 s) after the let-go: has the block landed in the bin?
NEAR = 70           # px: a loose block this close to the bin centre = it missed the bin


def top(i):
    return bgr(ds[int(i)]["observation.images.top"])


def near_count(det, colour, bxy):
    return sum(1 for o in det["objects"] if o[0] == colour and np.hypot(o[1] - bxy[0], o[2] - bxy[1]) < NEAR)


rows = {s: {} for s in calib}
t0 = time.time()
for e in range(ds.meta.total_episodes):
    a0, a1 = int(ds.meta.episodes["dataset_from_index"][e]), int(ds.meta.episodes["dataset_to_index"][e])
    hf = ds.hf_dataset.select(range(a0, a1))
    act = np.stack([np.asarray(x) for x in hf["action"]])
    st = np.stack([np.asarray(x) for x in hf["observation.state"]])
    n = len(act)
    for side in ("left", "right"):
        si = [sn.index(f"{side}_{j}.pos") for j in JN]
        g = act[:, an.index(f"{side}_gripper.pos")]
        k, was_open = 1, False
        while k < n:
            if g[k - 1] > 15:
                was_open = True
            if was_open and g[k] < 12 <= g[k - 1] and k > 20 and np.abs(st[k, si[1:3]] - st[0, si[1:3]]).max() > 30:
                t = k                                    # grab
                low, r = g[t], None
                for j in range(t, n):
                    low = min(low, g[j])
                    if g[j] - low > 6:
                        r = j                            # let-go
                        break
                was_open = False
                if r is None:
                    break
                held = st[r - 1, si[5]] > 7 and np.abs(st[r, si[1:3]] - st[0, si[1:3]]).max() > 30
                q = st[r, si]
                if held and ref[side]:
                    colour = min(ref[side], key=lambda c: np.abs(ref[side][c] - q[:3]).max())
                    dist = float(np.abs(ref[side][colour] - q[:3]).max())
                    if dist < 25:
                        before = vision.detect(top(a0 + t))
                        after = vision.detect(top(a0 + min(r + AFTER, n - 1)))
                        if colour in after["bins"]:
                            bxy = after["bins"][colour][:2]
                            good = near_count(after, colour, bxy) <= near_count(before, colour, bxy)
                            rows[side].setdefault(colour, []).append({"ep": e, "q": q.tolist(), "good": bool(good)})
                k = r + 1
                continue
            k += 1
    if e % 20 == 0:
        print(f"demo {e}/{ds.meta.total_episodes}  ({time.time() - t0:.0f} s)", flush=True)

report = {}
for side in rows:
    for colour, rs in rows[side].items():
        Q = np.array([r["q"] for r in rs])
        good = np.array([r["good"] for r in rs])
        old = np.array(calib[side]["bins"][colour]["hover"][:5])
        qg = np.median(Q[good, :5], 0) if good.sum() >= 2 else None
        spread = float(np.abs(Q[good, :4] - qg[:4]).mean()) if qg is not None else None
        report[f"{side} {colour}"] = {"drops": int(len(rs)), "good": int(good.sum()),
                                      "old": np.round(old, 1).tolist(),
                                      "new": None if qg is None else np.round(qg, 1).tolist(), "spread_deg": spread}
        print(f"\n{side} {colour}: {len(rs)} let-gos, {good.sum()} landed in the bin"
              + (f" | old pose {np.round(old, 0).tolist()} -> GOOD-drops pose {np.round(qg, 0).tolist()} "
                 f"(spread {spread:.1f} deg)" if qg is not None else " | too few good drops - unchanged"))
        if APPLY and qg is not None and not calib[side]["bins"][colour].get("taught"):
            calib[side].setdefault("bins_before_gooddrops", {})[colour] = calib[side]["bins"][colour]
            calib[side]["bins"][colour] = dict(calib[side]["bins"][colour], hover=np.median(Q[good], 0).tolist(),
                                               n=int(good.sum()), source="good demo drops")
(HERE / "bins_check.json").write_text(json.dumps(report, indent=1))
if APPLY:
    for s in calib:
        (HERE / f"calib_{s}.json").write_text(json.dumps(calib[s], indent=1))
    print("\nwritten into calib_left.json / calib_right.json")
