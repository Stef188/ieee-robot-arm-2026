"""UPDATE_TURNS (27 Nov, competition): re-measure each arm's TURN correction from all ground truth on this table and
write it into the launchers. No motors, no camera.
    UPDATE_TURNS.bat            -> prints the best SORT_PAN_LEFT / SORT_PAN_RIGHT for both sorters (nothing changed)
    UPDATE_TURNS.bat apply      -> also writes them into SORT_FAST.bat (shape/dynamic) and version I's RUN.bat
Ground truth = grab spots taught today (TEACH_GRAB) + every successful first-try grab on this table since the camera
correction (runlog\\picks.jsonl). More good runs = more data = better numbers. A new value is only written when it
is better on grabs left out of the fit (honest test) and the arm has >= 6 ground-truth grabs.
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for k in ("SORT_PAN_LEFT", "SORT_PAN_RIGHT"):
    os.environ.pop(k, None)
import comp_geometry as G  # noqa: E402
import fit_offset as F  # noqa: E402

TARGETS = {"latest sorter (shape / dynamic)": (HERE, HERE / "SORT_FAST.bat"),
           "version I (colour)": (HERE / "versions" / "I_1705_lines_bins_spots",
                                  HERE / "versions" / "I_1705_lines_bins_spots" / "RUN.bat")}
GRID = np.arange(-15, 15.01, 0.5)


def current(bat, side):
    m = re.search(rf"set SORT_PAN_{side.upper()}=([-\d.]+)", bat.read_text(encoding="utf-8", errors="replace"))
    return float(m.group(1)) if m else 0.0


def main():
    apply = len(sys.argv) > 1 and sys.argv[1].lower() == "apply"
    for name, (folder, bat) in TARGETS.items():
        mod = F.load_model_module(folder)
        print(f"\n{name}:")
        for side in ("left", "right"):
            gts = G.ground_truth(side)
            c = json.loads((folder / f"calib_{side}.json").read_text())
            c["points"] = [p for p in c.get("points", []) if not str(p.get("time", "")).startswith(G.TODAY)]
            cwd = os.getcwd()
            os.chdir(folder)
            try:
                m = mod.ArmMap(c)
            finally:
                os.chdir(cwd)
            base = np.array([float(m.poses(g[0], g[1], g[2])[0][0]) for g in gts])
            truth = np.array([g[3][0] for g in gts])

            def err(idx, pan):
                return float(np.mean(np.abs(base[idx] + pan - truth[idx])))
            allidx = np.arange(len(gts))
            best = float(min(GRID, key=lambda p: err(allidx, p)))
            loo_best = np.mean([abs(base[i] + min(GRID, key=lambda p: err(np.delete(allidx, i), p)) - truth[i])
                                for i in allidx])
            now = current(bat, side)
            loo_now = err(allidx, now)
            print(f"   {side:5s}: {len(gts):2d} ground-truth grabs | set now {now:+.1f} deg (turn error {loo_now:.1f}) | "
                  f"best {best:+.1f} deg (turn error on grabs left out {loo_best:.1f})")
            if apply and len(gts) >= 6 and abs(best - now) >= 0.5 and loo_best < loo_now:
                s = bat.read_text(encoding="utf-8", errors="replace")
                s2 = re.sub(rf"(if not defined SORT_PAN_{side.upper()} set SORT_PAN_{side.upper()}=)[-\d.]+",
                            lambda mm: mm.group(1) + f"{best:g}", s)
                bat.write_text(s2, encoding="utf-8", newline="")
                print(f"          -> written: SORT_PAN_{side.upper()}={best:g} in {bat.name}")
    if not apply:
        print("\n(nothing changed - run  UPDATE_TURNS.bat apply  to write the best values)")


if __name__ == "__main__":
    main()
