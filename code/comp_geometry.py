"""COMP GEOMETRY (27 Nov, competition): where each arm REALLY sits on this table, from ground truth.
    python comp_geometry.py           -> compares correction models (honest: each grab scored while left out),
                                         saves the best to comp_xform.json (used by sort.py), prints the numbers
Ground truth = grab spots taught here today (TEACH_GRAB) + every successful grab on this table since the camera
correction (runlog\\picks.jsonl). Each says: a block seen at camera pixel (u, v) needed exactly these joints.
Correction models, per arm, applied to the camera pixel before the arm's grab model is asked:
    none | turn (constant pan offset) | shift (dx, dy) | base move = turn + shift + scale about the arm base
No motors, no camera.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.pop("SORT_PAN_LEFT", None)
os.environ.pop("SORT_PAN_RIGHT", None)
import model  # noqa: E402

TODAY = "2026-11-27"
SINCE = "2026-11-27 13:07"            # camera correction saved 13:06
BASE = {"left": (242.0, 420.0), "right": (458.0, 430.0)}


def xform(p, side, prm):
    """camera pixel -> 'lab-equivalent' pixel: turn th (deg) + scale s about the arm base, then shift (dx, dy)."""
    th, s, dx, dy = prm
    bx, by = BASE[side]
    c, n = math.cos(math.radians(th)), math.sin(math.radians(th))
    x, y = p[0] - bx, p[1] - by
    return (bx + s * (c * x - n * y) + dx, by + s * (n * x + c * y) + dy)


def ground_truth(side):
    c = json.loads((HERE / f"calib_{side}.json").read_text())
    gt = [(p["u"], p["v"], p.get("angle", 0.0), np.asarray(p["grasp"][:3], float), 1.0, "taught")
          for p in c.get("points_comp_taught", []) + c.get("points", [])
          if p.get("taught") and str(p.get("time", "")).startswith(TODAY)]
    for line in open(HERE / "runlog" / "picks.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("side") == side and r.get("result") == "done" and r.get("q_grasp") and r.get("time", "") >= SINCE:
            gt.append((r["u"], r["v"], r.get("angle", 0.0), np.asarray(r["q_grasp"][:3], float), 0.7, "grab ok"))
    return gt


def arm_model(side):
    c = json.loads((HERE / f"calib_{side}.json").read_text())
    c["points"] = [p for p in c.get("points", []) if not str(p.get("time", "")).startswith(TODAY)]
    return model.ArmMap(c)


def err_one(m, side, g, prm, pan_only_off=None):
    u, v, a, q, w, _ = g
    if pan_only_off is not None:
        pose = np.asarray(m.poses(u, v, a)[0][:3], float)
        pose[0] += pan_only_off
    else:
        pose = np.asarray(m.poses(*xform((u, v), side, prm), a)[0][:3], float)
    d = pose - q
    return d


def cost(m, side, gts, prm=None, pan=None):
    return float(np.sum([g[4] * np.mean(np.abs(err_one(m, side, g, prm, pan))) for g in gts]) / sum(g[4] for g in gts))


def fit(m, side, gts, kind):
    if kind == "none":
        return None, None
    if kind == "turn":
        best = min(np.arange(-15, 15.1, 0.5), key=lambda c: cost(m, side, gts, pan=c))
        return None, float(best)
    x0 = {"shift": [0, 0], "base move": [0, 1, 0, 0]}[kind]

    def full(x):
        return (0.0, 1.0, x[0], x[1]) if kind == "shift" else (x[0], x[1], x[2], x[3])

    def f(x):
        prm = full(x)
        reg = 0.0 if kind == "shift" else 0.02 * abs(prm[0]) + 20 * abs(prm[1] - 1)   # keep it physical
        return cost(m, side, gts, prm) + reg
    x0 = np.asarray(x0, float)
    steps = {"shift": [12, 12], "base move": [4, 0.04, 12, 12]}[kind]
    simplex = [x0] + [x0 + np.eye(len(x0))[i] * steps[i] for i in range(len(x0))]
    r = minimize(f, x0, method="Nelder-Mead", options=dict(initial_simplex=np.array(simplex), xatol=0.05, fatol=0.01,
                                                           maxiter=600))
    return full(r.x), None


def evaluate(m, side, gts, kind):
    """leave-one-out: fit on all but one grab, score the one left out (mean |pan, lift, elbow| and |pan|)."""
    e_all, e_pan = [], []
    for i in range(len(gts)):
        rest = gts[:i] + gts[i + 1:]
        prm, pan = fit(m, side, rest, kind)
        d = err_one(m, side, gts[i], prm if prm else (0.0, 1.0, 0.0, 0.0), pan)
        e_all.append(np.mean(np.abs(d)))
        e_pan.append(abs(d[0]))
    return float(np.mean(e_all)), float(np.mean(e_pan))


def main():
    out = {}
    for side in ("left", "right"):
        gts = ground_truth(side)
        m = arm_model(side)
        print(f"\n{side.upper()} arm: {len(gts)} ground-truth grabs "
              f"({sum(1 for g in gts if g[5] == 'taught')} taught, {sum(1 for g in gts if g[5] != 'taught')} successful grabs)")
        res = {}
        for kind in ("none", "turn", "shift", "base move"):
            t0 = time.time()
            res[kind] = evaluate(m, side, gts, kind)
            print(f"   {kind:10s}: error on grabs left out {res[kind][0]:4.1f} deg (turn alone {res[kind][1]:4.1f} deg)"
                  f"   [{time.time() - t0:.0f} s]", flush=True)
        best = min(res, key=lambda k: res[k][0])
        prm, pan = fit(m, side, gts, best)
        print(f"   BEST: {best}" + (f" -> turn {prm[0]:+.1f} deg, scale {prm[1]:.3f}, shift ({prm[2]:+.0f},{prm[3]:+.0f}) px"
                                    if prm else (f" -> constant turn {pan:+.1f} deg" if pan is not None else "")))
        # what it does at each ground-truth spot
        for g in gts:
            d0 = err_one(m, side, g, (0.0, 1.0, 0.0, 0.0), None)
            d1 = err_one(m, side, g, prm if prm else (0.0, 1.0, 0.0, 0.0), pan)
            print(f"      ({g[0]:.0f},{g[1]:.0f}) {g[5]:8s}: pan/lift/elbow error {np.round(d0, 0).tolist()} -> {np.round(d1, 0).tolist()}")
        out[side] = dict(kind=best, prm=list(prm) if prm else None, pan=pan, loo=res, n=len(gts),
                         time=time.strftime("%Y-%m-%d %H:%M"))
    path = HERE / "comp_xform.json"
    if path.exists():
        path.rename(HERE / f"comp_xform_{time.strftime('%H%M%S')}.json")
    path.write_text(json.dumps(out, indent=1))
    print(f"\nsaved {path.name}")


if __name__ == "__main__":
    main()
