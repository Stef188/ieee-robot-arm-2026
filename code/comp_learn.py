"""COMP_LEARN (27 Nov, competition): teach the grab model THIS table, from ground truth - spot by spot.
    COMP_LEARN.bat          -> honest test (each ground-truth grab left out in turn), prints old vs new error
    COMP_LEARN.bat apply    -> adds the ground truth to calib_<side>.json (the network retrains itself at the next start)
Ground truth = grab spots taught today + successful FIRST-TRY grabs on this table since the camera correction.
The arm-turn correction (SORT_PAN_<side> in SORT_FAST.bat) is SUBTRACTED before storing, because the sorter adds it
again at run time. Deep grabs are converted back to the demo grab height (as learn_from_runs.py does).
Re-running replaces the earlier competition examples (no duplicates). Backup: calib_<side>_before_complearn_<time>.json.
"""
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import os  # noqa: E402
for k in ("SORT_PAN_LEFT", "SORT_PAN_RIGHT"):
    os.environ.pop(k, None)
import model  # noqa: E402

TODAY, SINCE, W = "2026-11-27", "2026-11-27 13:07", 3


def pan_setting(side):
    m = re.search(rf"set SORT_PAN_{side.upper()}=([-\d.]+)", (HERE / "SORT_FAST.bat").read_text(errors="replace"))
    return float(m.group(1)) if m else 0.0


def examples(side, calib):
    """(u, v, angle, joints5 in the LAB convention, label)"""
    pan = pan_setting(side)
    up = np.asarray(calib["up"][:5], float) if calib.get("up") else np.zeros(5)
    out = []
    for p in calib.get("points_comp_taught", []):
        if str(p.get("time", "")).startswith(TODAY):
            q = np.asarray(p["grasp"][:5], float)
            q[0] -= pan
            out.append((p["u"], p["v"], p.get("angle", 0.0), q, "taught " + p["time"][11:]))
    for line in open(HERE / "runlog" / "picks.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if (r.get("side") == side and r.get("result") == "done" and r.get("q_grasp") and r.get("time", "") >= SINCE
                and int(r.get("tries") or 1) == 1):
            q = np.asarray(r["q_grasp"][:5], float) + float(r.get("deep") or 0.0) * up    # back to demo grab height
            q[0] -= pan
            out.append((r["u"], r["v"], r.get("angle", 0.0), q, "grab ok " + r["time"][11:16]))
    return out


def as_point(e, n_len, up):
    u, v, a, q, lab = e
    return {"u": float(u), "v": float(v), "angle": float(a), "area": 300.0, "colour": "comp",
            "grasp": list(map(float, q)) + [10.0] * (n_len - 5), "hover": list(map(float, q + 2.0 * up)) + [10.0] * (n_len - 5),
            "wrist": None, "demo": None, "taught": True, "w": W, "source": "comp " + lab, "comp": True}


def build(calib, ex):
    c = json.loads(json.dumps(calib))
    c["points"] = [p for p in c["points"] if not p.get("comp") and not str(p.get("time", "")).startswith(TODAY)]
    n_len = len(c["points"][0]["grasp"])
    up = np.asarray(c["up"][:5], float) if c.get("up") else np.zeros(5)
    c["points"] += [as_point(e, n_len, up) for e in ex]
    return c


def err(m, e):
    u, v, a, q, _ = e
    g = np.asarray(m.poses(u, v, a)[0][:3], float)
    return np.abs(g - q[:3])


def main():
    apply = len(sys.argv) > 1 and sys.argv[1].lower() == "apply"
    for side in ("left", "right"):
        path = HERE / f"calib_{side}.json"
        calib = json.loads(path.read_text())
        ex = examples(side, calib)
        old = model.ArmMap(build(calib, []))
        e_old = np.array([err(old, e) for e in ex])
        e_new = []
        t0 = time.time()
        for i in range(len(ex)):                      # honest: each example scored by a model trained without it
            m = model.ArmMap(build(calib, ex[:i] + ex[i + 1:]))
            e_new.append(err(m, ex[i]))
        e_new = np.array(e_new)
        print(f"\n{side.upper()} arm: {len(ex)} competition examples (turn correction {pan_setting(side):+g} subtracted) "
              f"[{time.time() - t0:.0f} s]")
        for e, a, b in zip(ex, e_old, e_new):
            print(f"   ({e[0]:.0f},{e[1]:.0f}) {e[4]:14s}: pan/lift/elbow error {np.round(a, 0).tolist()} -> {np.round(b, 0).tolist()}")
        print(f"   MEAN error on examples left out: before {e_old.mean():.1f} deg -> with the others learned {e_new.mean():.1f} deg"
              f"   (pan {e_old[:, 0].mean():.1f} -> {e_new[:, 0].mean():.1f}, lift {e_old[:, 1].mean():.1f} -> "
              f"{e_new[:, 1].mean():.1f}, elbow {e_old[:, 2].mean():.1f} -> {e_new[:, 2].mean():.1f})")
        if apply:
            if e_new.mean() < e_old.mean():
                shutil.copy(path, HERE / f"calib_{side}_before_complearn_{time.strftime('%H%M')}.json")
                path.write_text(json.dumps(build(calib, ex), indent=1))
                model.ArmMap(build(calib, ex))        # retrain + save the network now (not at the sorter start)
                print(f"   -> APPLIED: {len(ex)} competition examples are now in calib_{side}.json")
            else:
                print("   -> NOT applied (not better on examples left out)")
    if not apply:
        print("\n(nothing changed - COMP_LEARN.bat apply  writes it)")


if __name__ == "__main__":
    main()
