"""HARD-CODE the bin drop poses by hand (26 Nov). For each bin, the arm goes limp, you hold the gripper (holding
nothing) right above the MIDDLE of the bin, ~5 cm above the rim, press Enter: that exact pose is saved, and the sorter
drops EVERY block for that bin at exactly that pose (no extra lift, no guessing from the demos).
    python classical\\teach_bin.py all            -> every bin the camera sees, each with the NEAREST arm
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

import math  # noqa: E402

ARM_BASE_PX = {"left": (242.0, 420.0), "right": (458.0, 430.0)}   # = sort.ARM_BASE_PX
# 27 Nov: also save into version I (the best colour sorter has its own calibration files)
ALSO = [HERE / "versions" / "I_1705_lines_bins_spots"]


def pan_setting(bat, side):
    """27 Nov: the arm-turn correction this sorter adds to every drop (SORT_PAN_<side> in its launcher)."""
    import re
    try:
        m = re.search(rf"set SORT_PAN_{side.upper()}=([-\d.]+)", bat.read_text(errors="replace"))
        return float(m.group(1)) if m else 0.0
    except OSError:
        return 0.0

cam = hw.TopCamera()
cam.connect()
det = vision.detect(cam.grab())
cam.disconnect()
print("bins seen:", {k: (round(v[0]), round(v[1])) for k, v in det["bins"].items()})

if sys.argv[1:2] == ["all"]:
    # 27 Nov (competition): every bin the camera sees, each with the arm NEAREST to it (the lab layout was
    # hard-coded here: left -> red, right -> yellow - at the competition red stands on the right)
    jobs = []
    for colour, b in sorted(det["bins"].items(), key=lambda kv: kv[1][0]):      # left -> right in the camera
        d = {s: math.hypot(b[0] - x, b[1] - y) for s, (x, y) in ARM_BASE_PX.items()}
        jobs.append((min(d, key=d.get), colour))
    jobs.sort(key=lambda j: j[0] != "left")
    print("plan (each bin -> the arm nearest to it):", ", ".join(f"{s} arm -> {c}" for s, c in jobs))
    if input("Enter = start, anything else = stop: ").strip():
        sys.exit(0)
else:
    jobs = [(sys.argv[1], sys.argv[2])]

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
            input(f">>> Hold the {side} gripper above the BACK-CENTRE of the {colour} bin (centre line, towards the far "
                  f"wall, jaws well inside the rim), ~5 cm above the rim. "
                  f"Keep it still, press Enter. ")
            q = np.median(np.array([arm.read() for _ in range(10)]), 0)
            print(f"   saved: {np.round(q[:5], 1).tolist()}")
            calib.setdefault("bins_before_teach", {})[colour] = calib["bins"].get(colour)
            # 27 Nov (competition): the sorter adds its arm-turn correction to every drop - store the pose WITHOUT it
            q_main = np.array(q, float)
            q_main[0] -= pan_setting(HERE / "SORT_FAST.bat", side)
            calib["bins"][colour] = {"u": float(bu), "v": float(bv), "hover": [float(x) for x in q_main],
                                     "taught": True, "time": time.strftime("%Y-%m-%d %H:%M"), "raw_pan": float(q[0])}
            path.write_text(json.dumps(calib, indent=1))
            for d in ALSO:                       # same pose into version I's calibration
                p2 = d / f"calib_{side}.json"
                if p2.exists():
                    c2 = json.loads(p2.read_text())
                    c2.setdefault("bins_before_teach", {})[colour] = c2["bins"].get(colour)
                    e = dict(calib["bins"][colour])
                    qi = np.array(q, float)
                    qi[0] -= pan_setting(d / "RUN.bat", side)      # version I has its own turn setting
                    e["hover"] = [float(x) for x in qi]
                    c2["bins"][colour] = e
                    p2.write_text(json.dumps(c2, indent=1))
                    print(f"   also saved into {d.name}")
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
