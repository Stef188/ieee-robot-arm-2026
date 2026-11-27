"""LEARN FROM EXPERIENCE (26 Nov): every grab that WORKED in a real run becomes a new training example for GraspNet.
    python classical\\learn_from_runs.py            (LEARN.bat) - adds the new successful grabs, retrains, prints a check
    python classical\\learn_from_runs.py --dry      - only shows what it would add
Source: classical\\runlog\\picks.jsonl, written by sort.py for every pick (block pixel, the arm joints when the gripper
closed, the result). A grab that held the block ("done" or "handed") = a verified example "block here -> these joints
grab it". The deep grip is undone first (joints + DEEP x up), so the example uses the same convention as the demo grabs.
Added to calib_<side>.json with weight 2 (source "run <id>"); each pick is used once (calib "learned_picks"). The old
file is kept as calib_<side>_before_learn_<time>.json. GraspNet retrains itself on the next start (~3 s).
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
DRY = "--dry" in sys.argv
W = 2


def aim_now(side, u, v):
    """Where the sorter aims NOW for a block at (u, v): + learned bias, trim and jaw offset (as in sort.py pick()).
    Learned examples are stored at that point, so the sorter's offsets are not counted twice. If those settings change
    later, run this script with --relearn."""
    import math
    import os
    saved = sys.argv
    sys.argv = [saved[0]]
    import sort
    sys.argv = saved
    lb = sort.learned_bias(side)
    if lb is not None:
        u, v = u + lb[0], v + lb[1]
    tr = sort.trim_px(side)
    bu, bv = sort.ARM_BASE_PX[side]
    h = np.array([u - bu, v - bv], float)
    h /= (np.linalg.norm(h) + 1e-9)
    left_of = np.array([h[1], -h[0]])
    tr = tr - left_of * sort.JAW_OFFSET.get(side, 0.0) * sort.PX_PER_CM
    return float(u + tr[0]), float(v + tr[1])


def main():
    recs = [json.loads(l) for l in open(HERE / "runlog" / "picks.jsonl", encoding="utf-8") if l.strip()]
    ok = [r for r in recs if r.get("result") in ("done", "handed") and r.get("q_grasp") and r.get("up")]
    stamp = time.strftime("%H%M")
    for side in ("left", "right"):
        path = HERE / f"calib_{side}.json"
        calib = json.loads(path.read_text())
        if "--relearn" in sys.argv:          # drop earlier learned examples and add them again at today's aim point
            calib["points"] = [p for p in calib["points"] if not str(p.get("source", "")).startswith("run ")]
            calib["learned_picks"] = []
        used = set(calib.get("learned_picks", []))
        n_len = len(calib["points"][0]["grasp"])
        new = []
        for r in ok:
            key = f"{r['run']}:{r['side']}:{r['t']}"
            if r["side"] != side or key in used or int(r.get("tries") or 1) > 1:
                continue        # retries are nudged 1 cm (not recorded) - only first tries are clean examples
            if str(r.get("time", "")) >= "2026-11-27" and "--comp" not in sys.argv:
                continue        # 27 Nov: competition-table grabs carry the arm TURN correction (SORT_PAN_*) - learning
                                # them as lab examples would apply that correction twice (comp_geometry.py uses them)
            q = np.array(r["q_grasp"][:5], float)
            up = np.array(r["up"][:5], float)
            q = q + float(r.get("deep") or 0.0) * up          # undo the deep grip -> demo grab-height convention
            if q[1] < -80 or abs(q[0]) > 120:                   # rest-pose-like or nonsense: skip
                continue
            g = list(q) + [10.0] * (n_len - 5)                  # gripper value is not used by the model
            h = list(q + 2.0 * up) + [10.0] * (n_len - 5)
            # the AIMED pixel (block + this arm's trim / jaw offset / learned bias): the sorter adds those offsets
            # again at run time, so the example must be "aimed point -> joints that worked" (else counted twice)
            au, av = aim_now(side, r["u"], r["v"])
            new.append(dict(u=au, v=av, angle=r["angle"], area=None, colour=r.get("colour"),
                            grasp=[float(x) for x in g], hover=[float(x) for x in h], wrist=None, demo=None,
                            taught=True, w=W, source=f"run {r['run']}"))
            used.add(key)
        print(f"{side}: {len(new)} new successful grabs to learn from "
              f"({sum(1 for p in calib['points'] if str(p.get('source', '')).startswith('run'))} learned before)")
        if not new or DRY:
            continue
        shutil.copy(path, HERE / f"calib_{side}_before_learn_{stamp}.json")
        calib["points"] += new
        calib["learned_picks"] = sorted(used)
        path.write_text(json.dumps(calib, indent=1))
    if DRY:
        return
    import model
    for side in ("left", "right"):
        t = time.time()
        m = model.load(side)
        print(f"{side}: GraspNet retrained ({time.time() - t:.0f} s) on {len(m.uv)} weighted points")


if __name__ == "__main__":
    main()
