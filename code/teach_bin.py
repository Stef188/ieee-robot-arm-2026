"""HARD-CODE the bin drop poses by hand (26 Nov). For each bin, the arm goes limp, you hold the gripper (holding
nothing) right above the MIDDLE of the bin, ~5 cm above the rim, press Enter: that exact pose is saved, and the sorter
drops EVERY block for that bin at exactly that pose (no extra lift, no guessing from the demos).
    python classical\\teach_bin.py all            -> left red, left green, right yellow, right blue, right green
    python classical\\teach_bin.py left green     -> just one
Saved into calib_<side>.json bins.<colour> ("taught": true); the old pose is kept as bins_before_teach.<colour>.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import hw  # noqa: E402
import vision  # noqa: E402

if sys.argv[1:2] == ["all"]:
    jobs = [("left", "red"), ("left", "green"), ("right", "yellow"), ("right", "blue"), ("right", "green")]
else:
    jobs = [(sys.argv[1], sys.argv[2])]

cam = hw.TopCamera()
cam.connect()
det = vision.detect(cam.grab())
cam.disconnect()
print("bins seen:", {k: (round(v[0]), round(v[1])) for k, v in det["bins"].items()})

for side in dict.fromkeys(s for s, _ in jobs):
    path = HERE / f"calib_{side}.json"
    calib = json.loads(path.read_text())
    arm = hw.Arm(side)
    arm.connect()
    try:
        for _, colour in [j for j in jobs if j[0] == side]:
            if colour not in det["bins"]:
                print(f"\n{colour} bin not seen by the top camera - skipped")
                continue
            bu, bv = det["bins"][colour][:2]
            input(f"\n=== {side.upper()} arm -> {colour.upper()} bin ===\n!!! The {side} arm goes LIMP now. Hold it, press Enter. ")
            arm.torque(False)
            input(f">>> Hold the {side} gripper right above the MIDDLE of the {colour} bin, ~5 cm above the rim. "
                  f"Keep it still, press Enter. ")
            q = np.median(np.array([arm.read() for _ in range(10)]), 0)
            print(f"   saved: {np.round(q[:5], 1).tolist()}")
            calib.setdefault("bins_before_teach", {})[colour] = calib["bins"].get(colour)
            calib["bins"][colour] = {"u": float(bu), "v": float(bv), "hover": [float(x) for x in q],
                                     "taught": True, "time": time.strftime("%Y-%m-%d %H:%M")}
            path.write_text(json.dumps(calib, indent=1))
            input(">>> Bring the arm back near its rest pose and HOLD it - Enter = motors on (it holds there). ")
            arm.torque(True)
            arm.send(arm.read())
    finally:
        try:
            arm.torque(True)
        except Exception:
            pass
        arm.disconnect()
print("\nDone. These bins are now hard-coded:", [f"{s} {c}" for s, c in jobs])
