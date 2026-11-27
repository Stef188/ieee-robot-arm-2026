"""FIT_ROLL (27 Nov, competition): measure each arm's WRIST geometry so the jaws line up with a block's faces.
When the gripper points down, the jaw line in the camera picture turns with the wrist roll AND with the arm's heading:
        roll = s * (block face angle) + k * (arm turn, pan) + c          (mod 90: a cube's faces repeat every 90 deg)
s, k, c are fitted per arm from grabs taught TODAY with the jaws carefully FLAT ON TWO FACES (TEACH_GRAB.bat), with
the cube turned to DIFFERENT angles. Honest test: each taught grab is scored while left out. Saved to roll_geo.json
only if it beats the current (learned) wrist model on the same grabs; model.py then uses it for every grab.
    FIT_ROLL.bat            -> fit + report (+ save if better)
No motors, no camera.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ["SORT_GEO_ROLL"] = "0"           # compare against the CURRENT wrist model
for k in ("SORT_PAN_LEFT", "SORT_PAN_RIGHT"):
    os.environ.pop(k, None)
TODAY = time.strftime("%Y-%m-%d")
PER = 90.0


def wrap(x, per=PER):
    return (np.asarray(x, float) + per / 2) % per - per / 2


def taught_today(side):
    c = json.loads((HERE / f"calib_{side}.json").read_text())
    pts = c.get("points", []) + c.get("points_comp_taught", [])
    seen, out = set(), []
    for p in pts:
        if p.get("taught") and str(p.get("time", "")).startswith(TODAY):
            key = (round(p["u"], 1), round(p["v"], 1), round(p["grasp"][4], 1))
            if key not in seen:
                seen.add(key)
                out.append(p)
    return out


def fit(A, P, R, s):
    best = None
    for k in np.arange(-1.5, 1.501, 0.05):
        for c in np.arange(-45, 45, 0.25):
            e = float(np.abs(wrap(R - (s * A + k * P + c))).mean())
            if best is None or e < best[0]:
                best = (e, float(k), float(c))
    return best


def main():
    import comp_geometry as G
    out = {}
    for side in ("left", "right"):
        T = taught_today(side)
        if len(T) < 6:
            print(f"{side}: only {len(T)} taught grabs today - teach at least 6 (cube at different angles)")
            continue
        A = np.array([p["angle"] for p in T]); P = np.array([p["grasp"][0] for p in T]); R = np.array([p["grasp"][4] for p in T])
        spread = float(np.ptp(wrap(A, 90)))
        m = G.arm_model(side)
        cur = np.abs(wrap(R - np.array([m.poses(p["u"], p["v"], p["angle"])[0][4] for p in T])))
        res = {}
        for s in (1.0, -1.0):
            loo = []
            for i in range(len(T)):
                msk = np.ones(len(T), bool); msk[i] = False
                _, k, c = fit(A[msk], P[msk], R[msk], s)
                loo.append(abs(wrap(R[i] - (s * A[i] + k * P[i] + c))))
            res[s] = np.array(loo)
        s = min(res, key=lambda x: res[x].mean())
        e, k, c = fit(A, P, R, s)
        print(f"\n{side.upper()}: {len(T)} taught grabs, cube angles spread over {spread:.0f} deg (want > 50)")
        print(f"   current wrist model : error {cur.mean():.1f} deg  {np.round(cur, 0).tolist()}")
        print(f"   geometric (left out): error {res[s].mean():.1f} deg  {np.round(res[s], 0).tolist()}"
              f"   -> roll = {s:+.0f} x angle {k:+.2f} x pan {c:+.1f}")
        # 27 Nov: used when it beats the current wrist model on grabs LEFT OUT and its arm-turn factor k is physical
        # (about 1: turning the arm turns the jaws by the same amount)
        ok = res[s].mean() < cur.mean() and 0.6 < k < 1.4
        print("   -> USE IT" if ok else "   -> not used (needs: better than now on grabs left out, k between 0.6 and 1.4)")
        if ok:
            out[side] = dict(s=s, k=k, c=c, n=len(T), loo=float(res[s].mean()), time=time.strftime("%Y-%m-%d %H:%M"))
    if out:
        path = HERE / "roll_geo.json"
        old = json.loads(path.read_text()) if path.exists() else {}
        old.update(out)
        path.write_text(json.dumps(old, indent=1))
        print(f"\nsaved {path.name}: {sorted(out)} - every grab now turns the wrist by this geometry (SORT_GEO_ROLL=0 = off)")


if __name__ == "__main__":
    main()
