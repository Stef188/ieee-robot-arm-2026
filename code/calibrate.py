"""LIVE calibration for the classical sorter (about 5-8 minutes per arm). Run with CALIBRATE.bat.
    python classical\\calibrate.py left|right [points]       (default 12 points)

For one follower arm at a time, torque is switched OFF so you can guide it by hand (HOLD THE ARM - it will sag).
For each point:
  1. put ONE object on the table where this arm should be able to grab it, arm back at rest (out of the picture)
     -> the top camera finds the object (pixel position + rotation)
  2. guide the gripper onto it exactly like a grab: jaws open around the object, tips just above the table -> Enter
  3. lift the gripper straight up ~5 cm (same spot, same wrist angle) -> Enter
     (the wrist camera then notes where the object sits in its picture = the target for the fine alignment)
Spread the points over the whole area this arm should cover (near/far, left/right, every bin area too).
Then, for each bin: hold the gripper ~5 cm above the bin's centre -> Enter (or s + Enter to skip a bin).
Saves classical\\calib_<side>.json. Nothing moves on its own: the arm is only ever moved by your hand.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import hw  # noqa: E402
import vision  # noqa: E402


def ask(msg):
    return input(f"\n>>> {msg}\n    [Enter = done, q = finish] ").strip().lower()


def main():
    side = sys.argv[1] if len(sys.argv) > 1 else "left"
    n_points = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    arm, cam = hw.Arm(side), hw.TopCamera()
    print(f"Connecting the {side} follower arm, the top camera and the {side} wrist camera ...")
    arm.connect()
    cam.connect()
    try:
        wcam = hw.WristCamera(side)
        wcam.connect()
    except Exception as e:
        wcam = None
        print(f"    (wrist camera not available: {e!r} - calibration still works, fine alignment will be off)")
    rest = arm.read()
    print(f"{side} arm rest pose: {np.round(rest, 1)}")
    input(f"\n!!! The {side} arm's motors will go LIMP. Hold the arm, then press Enter. ")
    arm.torque(False)
    data = {"side": side, "rest": rest.tolist(), "points": [], "bins": {}, "time": time.strftime("%Y-%m-%d %H:%M")}
    out = HERE / f"calib_{side}.json"
    try:
        while len(data["points"]) < n_points:
            k = len(data["points"]) + 1
            if ask(f"POINT {k}/{n_points}: put ONE object somewhere the {side} arm should grab it. "
                   f"Arm back at rest, hands out of the picture.") == "q":
                break
            det = vision.detect(cam.grab())
            objs = det["objects"]
            if len(objs) != 1:
                print(f"    I see {len(objs)} objects ({[o[0] for o in objs]}) - need exactly ONE in the black-tape area. Again.")
                continue
            colour, u, v, ang, area, w, h = objs[0]
            print(f"    found a {colour} object at pixel ({u:.0f}, {v:.0f}), rotated {ang:.0f} deg")
            if ask("Guide the gripper ONTO it like a grab: jaws open around it, tips just above the table.") == "q":
                break
            grasp = arm.read()
            if ask("Lift the gripper STRAIGHT UP about 5 cm (same spot, same wrist angle).") == "q":
                break
            hover = arm.read()
            # 25 Nov: where the object sits in the WRIST camera when the gripper is centred above it = the target the
            # sorter's fine alignment steers to
            wrist = vision.find_blob(wcam.grab(), colour) if wcam is not None else None
            if wcam is not None:
                print(f"    wrist camera sees it at {None if wrist is None else tuple(round(x, 3) for x in wrist[:2])}")
            data["points"].append({"u": u, "v": v, "angle": ang, "area": area, "colour": colour,
                                   "grasp": grasp.tolist(), "hover": hover.tolist(),
                                   "wrist": None if wrist is None else [wrist[0], wrist[1], wrist[2]]})
            out.write_text(json.dumps(data, indent=1))          # saved after every point
            print(f"    saved point {k}: grasp {np.round(grasp[:5], 0)}  hover {np.round(hover[:5], 0)}")
        ask("Move the arm back to rest and clear the table of loose objects (bins stay).")
        bins = vision.detect(cam.grab())["bins"]
        print(f"    bins seen: {sorted(bins)}")
        for colour, (bu, bv, bw, bh, ba) in bins.items():
            r = ask(f"Hold the {side} gripper ~5 cm above the CENTRE of the {colour.upper()} bin (s = skip this bin).")
            if r == "q":
                break
            if r == "s":
                continue
            data["bins"][colour] = {"u": bu, "v": bv, "hover": arm.read().tolist()}
            out.write_text(json.dumps(data, indent=1))
            print(f"    saved {colour} bin")
        ask(f"Put the {side} arm back in its REST pose and let go (it stays relaxed there, like after every run).")
    finally:
        try:
            arm.torque(True)
        except Exception:
            pass
        arm.disconnect()
        cam.disconnect()
        if wcam is not None:
            wcam.disconnect()
    print(f"\nDone: {len(data['points'])} points, bins {sorted(data['bins'])} -> {out}")


if __name__ == "__main__":
    main()
