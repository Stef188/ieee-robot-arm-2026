"""FIT_OFFSET (27 Nov, competition): how far the arms' grabs are shifted on THIS table, from grab spots taught here.
    FIT_OFFSET.bat            -> uses the grab spots taught TODAY (TEACH_GRAB.bat left / right)
At the competition the camera correction lines up the black line, but the arms do not sit exactly where they sat
relative to the line in the lab - every grab lands ~1-2 cm off. Each spot taught here says where the gripper REALLY
has to go for a block seen at pixel (u, v). This finds the one shift (per arm) that makes the sorter's own grab poses
match the taught ones, for the latest sorter AND for version I, and prints the SORT_TRIM values to use.
The taught spots are then moved out of the training points (kept under "points_comp_taught"), so they are not
counted twice (trim + network). Nothing moves; no motors.
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TODAY = time.strftime("%Y-%m-%d")
PX_PER_CM = 8.3
VERSIONS = {"latest sorter (SORT_SHAPE / SORT_DYNAMIC / SORT_FAST)": HERE,
            "version I (colour RUN.bat)": HERE / "versions" / "I_1705_lines_bins_spots"}


def load_model_module(folder):
    spec = importlib.util.spec_from_file_location(f"model_{abs(hash(str(folder)))}", folder / "model.py")
    mod = importlib.util.module_from_spec(spec)
    old, cwd = sys.path[:], os.getcwd()
    sys.path.insert(0, str(folder))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = old
    return mod


def fit(m, pts):
    """Grid search for the pixel shift (du, dv) that makes m's grab pose at (u+du, v+dv) match the taught joints."""
    best = None
    for du in range(-50, 51, 2):
        for dv in range(-50, 51, 2):
            e = []
            for p in pts:
                g = m.poses(p["u"] + du, p["v"] + dv, p.get("angle", 0.0))[0]
                e.append(np.abs(np.asarray(g[:3]) - np.asarray(p["grasp"][:3])))      # pan, lift, elbow
            err = float(np.mean(e))
            if best is None or err < best[0]:
                best = (err, du, dv)
    e0 = float(np.mean([np.abs(np.asarray(m.poses(p["u"], p["v"], p.get("angle", 0.0))[0][:3]) -
                               np.asarray(p["grasp"][:3])) for p in pts]))
    return best, e0


def main():
    out = {}
    for side in ("left", "right"):
        calib_path = HERE / f"calib_{side}.json"
        calib = json.loads(calib_path.read_text())
        today = [p for p in calib.get("points", []) + calib.get("points_comp_taught", [])
                 if p.get("taught") and str(p.get("time", "")).startswith(TODAY)]
        if not today:
            print(f"\n{side.upper()} arm: no grab spots taught today - run TEACH_GRAB.bat {side} first")
            continue
        print(f"\n{side.upper()} arm: {len(today)} spot(s) taught today at "
              + ", ".join(f"({p['u']:.0f},{p['v']:.0f})" for p in today))
        for name, folder in VERSIONS.items():
            mod = load_model_module(folder)
            c = json.loads((folder / f"calib_{side}.json").read_text())
            c["points"] = [p for p in c.get("points", []) if not str(p.get("time", "")).startswith(TODAY)]
            cwd = os.getcwd()
            os.chdir(folder)
            try:
                m = mod.ArmMap(c)
            finally:
                os.chdir(cwd)
            (err, du, dv), e0 = fit(m, today)
            tx, ty = du / PX_PER_CM, dv / PX_PER_CM
            print(f"   {name}: grabs are off by {np.hypot(du, dv) / PX_PER_CM:.1f} cm -> SORT_TRIM_{side.upper()}="
                  f"{tx:.1f},{ty:.1f}   (joint error {e0:.1f} deg -> {err:.1f} deg after the shift)")
            out.setdefault(name, {})[side] = (round(tx, 1), round(ty, 1), round(e0, 1), round(err, 1))
            # 27 Nov: the main error at the competition was a constant TURN of the arm base -> pan offset per arm
            os.environ.pop(f"SORT_PAN_{side.upper()}", None)
            d = np.array([np.asarray(p["grasp"][:2]) - np.asarray(m.poses(p["u"], p["v"], p.get("angle", 0.0))[0][:2])
                          for p in today])
            pan = float(np.median(d[:, 0]))
            print(f"      TURN: the arm must turn {pan:+.1f} deg more (spread {d[:, 0].std():.1f}) -> "
                  f"SORT_PAN_{side.upper()}={pan:.0f}   | lift {np.median(d[:, 1]):+.1f} deg")
        # keep today's spots out of the training points (the trim does the correction)
        calib.setdefault("points_comp_taught", []).extend(p for p in today if p not in calib["points_comp_taught"])
        calib["points"] = [p for p in calib["points"] if p not in today]
        calib_path.write_text(json.dumps(calib, indent=1))
    (HERE / "comp_offset.json").write_text(json.dumps(out, indent=1))
    print("\nSaved in comp_offset.json. Tell Claude these numbers (or paste this window).")


if __name__ == "__main__":
    main()
