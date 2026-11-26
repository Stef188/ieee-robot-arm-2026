"""Teach grab spots by hand where the arm misses (26 Nov) - mostly the EDGES of the area the demos covered.
    python classical\\teach_grab.py right
Loop: put ONE block where the arm missed -> Enter (the top camera finds it) -> the arm goes limp -> guide the open
gripper down around the block exactly as it should grab -> Enter (saved) -> lift the arm away. q = finish.
Each taught spot is added to calib_<side>.json with weight 3 (it counts 3x vs a demo grab); the grasp-pose network
retrains automatically on the next start (~3 s). The old file is kept as calib_<side>_before_teach_<time>.json.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import hw  # noqa: E402
import vision  # noqa: E402

side = sys.argv[1]
path = HERE / f"calib_{side}.json"
calib = json.loads(path.read_text())
shutil.copy(path, HERE / f"calib_{side}_before_teach_{time.strftime('%H%M')}.json")
up = np.array(calib["up"][:6]) if calib.get("up") else np.zeros(6)

cam = hw.TopCamera()
cam.connect()
arm = hw.Arm(side)
arm.connect()
n = 0
try:
    while True:
        k = input(f"\n>>> Put ONE block where the {side} arm missed (hands + arm out of the camera's view), "
                  f"then Enter.  [q = finish] ").strip().lower()
        if k == "q":
            break
        frames = [cam.grab() for _ in range(3)]
        det = vision.detect_avg(frames)
        # 26 Nov: only LOOSE blocks - not the ones inside (full) bins, not outside the black line
        bins = dict(det["bins"])
        for c, bb in vision.find_bins_by_rim(frames[-1]).items():
            bins.setdefault(c, bb[:5])
        import cv2
        boxes = [cv2.boxPoints(((float(bb[0]), float(bb[1])), (float(bb[2]), float(bb[3])), float(bb[4])))
                 .astype(np.float32) for bb in bins.values()]
        objs = [o for o in det["objects"]
                if cv2.pointPolygonTest(vision.SOURCE_POLY, (float(o[1]), float(o[2])), True) > -8
                and not any(cv2.pointPolygonTest(bx, (float(o[1]), float(o[2])), True) > -12 for bx in boxes)
                and o[2] > 290]                                   # the bins stand along the far edge (v < 290)
        if len(objs) != 1:
            print(f"   the camera sees {len(objs)} loose blocks {[(o[0], round(o[1]), round(o[2])) for o in objs]} - "
                  f"leave exactly ONE on the table (others in the bins) and try again")
            continue
        o = objs[0]
        print(f"   {o[0]} block at ({o[1]:.0f},{o[2]:.0f}), rotation {o[3]:.0f}")
        input(f">>> The {side} arm goes LIMP now - hold it. Enter. ")
        arm.torque(False)
        input(">>> Guide the OPEN gripper down around the block, jaws on two opposite sides, at grab height. "
              "Hold still, Enter. ")
        q = np.median(np.array([arm.read() for _ in range(10)]), 0)
        # a cube can be grabbed at 4 wrist turns 90 deg apart (a long block at 2, 180 apart): keep the turn that is
        # closest to what the model already does there, so the taught spot can't scramble the wrist-turn fit
        try:
            import model
            mm = model.load(side)
            r_pred = mm.poses(o[1], o[2], o[3])[0][4]
            per = 180.0 if (min(o[5], o[6]) > 0 and max(o[5], o[6]) / min(o[5], o[6]) >= 1.5) else 90.0
            q[4] = r_pred + ((q[4] - r_pred + per / 2) % per - per / 2)
        except Exception as e:
            print("   (could not align the wrist turn:", e, ")")
        hover = q + 2.0 * up
        calib["points"].append({"u": float(o[1]), "v": float(o[2]), "angle": float(o[3]), "area": float(o[4]),
                                "colour": o[0], "grasp": [float(x) for x in q], "hover": [float(x) for x in hover],
                                "wrist": None, "demo": None, "taught": True, "w": 3})
        path.write_text(json.dumps(calib, indent=1))
        n += 1
        print(f"   saved spot {n}: joints {np.round(q[:5], 1).tolist()}")
        input(">>> Lift the arm back up near its rest pose and HOLD it, Enter = motors on again. ")
        arm.torque(True)
        arm.send(arm.read())
finally:
    try:
        arm.torque(True)
    except Exception:
        pass
    arm.disconnect()
    cam.disconnect()
print(f"\n{n} spot(s) taught for the {side} arm. The network retrains by itself on the next sorter start.")
