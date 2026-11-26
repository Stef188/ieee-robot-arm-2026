"""Arm geometry for the classical sorter, FITTED from recorded grabs (no URDF needed). Separate from the competition kit.

Model per arm (SO-100/101 style, 5 joints + gripper, angles in LeRobot degrees):
  pan     theta = sp*(q_pan  - o_pan)                          turns the arm plane about the vertical
  in the arm plane:  a1 = sl*(q_lift - o_lift), a2 = se*(q_elbow - o_elbow), a3 = sw*(q_wflex - o_wflex)
     r = r0 + L1 cos a1 + L2 cos(a1+a2) + L3 cos(a1+a2+a3)      (horizontal reach of the grasp point)
     z = h0 + L1 sin a1 + L2 sin(a1+a2) + L3 sin(a1+a2+a3)      (height of the grasp point)
  grasp point in the arm frame: (r cos theta, r sin theta, z)
Camera -> table plane in the arm frame: similarity (scale k m/px, rotation phi, optional mirror, base pixel bx, by)
plus a mild perspective term, fitted together with the joint offsets from (pixel, joints-at-grab) pairs, with
RANSAC-style outlier rejection (wrong object/arm pairings in the demos).
python classical\\armmodel.py     -> fits both arms from classical\\grab_points.npz, prints accuracy, saves arm_fit.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
NOMINAL = dict(L1=0.116, L2=0.135, L3=0.10, r0=0.03, h0=0.12)
PNAMES = ["o_pan", "o_lift", "o_elbow", "o_wflex", "L1", "L2", "L3", "r0", "h0", "z_obj",
          "bx", "by", "k", "phi", "px", "py"]


def fk(p, signs, q):
    """q: (n, 4) [pan, lift, elbow, wflex] deg -> (n, 3) grasp point in the arm frame (m)."""
    d = dict(zip(PNAMES, p))
    sp, sl, se, sw = signs
    r2d = math.pi / 180
    th = sp * (q[:, 0] - d["o_pan"]) * r2d
    a1 = sl * (q[:, 1] - d["o_lift"]) * r2d
    a2 = a1 + se * (q[:, 2] - d["o_elbow"]) * r2d
    a3 = a2 + sw * (q[:, 3] - d["o_wflex"]) * r2d
    r = d["r0"] + d["L1"] * np.cos(a1) + d["L2"] * np.cos(a2) + d["L3"] * np.cos(a3)
    z = d["h0"] + d["L1"] * np.sin(a1) + d["L2"] * np.sin(a2) + d["L3"] * np.sin(a3)
    return np.stack([r * np.cos(th), r * np.sin(th), z], 1)


def pix2arm(p, mirror, uv):
    """pixels (n, 2) -> table point in the arm frame (n, 2), metres."""
    d = dict(zip(PNAMES, p))
    du, dv = uv[:, 0] - d["bx"], uv[:, 1] - d["by"]
    w = 1.0 + d["px"] * du * 1e-3 + d["py"] * dv * 1e-3       # mild perspective
    du, dv = du / w, dv / w
    if mirror:
        du = -du
    c, s = math.cos(d["phi"]), math.sin(d["phi"])
    return d["k"] * np.stack([c * du - s * dv, s * du + c * dv], 1)


def residuals(p, signs, mirror, uv, q):
    d = dict(zip(PNAMES, p))
    g = fk(p, signs, q)
    t = pix2arm(p, mirror, uv)
    res = [(g[:, 0] - t[:, 0]), (g[:, 1] - t[:, 1]), 0.5 * (g[:, 2] - d["z_obj"])]
    # L1/L2 pinned hard to the real arm (they set the metric scale; free, the fit shrinks everything to zero),
    # L3 (wrist -> grasp point), r0, h0 kept near nominal, camera scale near 1.2 mm/px
    reg = [30 * (d["L1"] - NOMINAL["L1"]), 30 * (d["L2"] - NOMINAL["L2"]), 3 * (d["L3"] - NOMINAL["L3"]),
           1 * (d["r0"] - NOMINAL["r0"]), 1 * (d["h0"] - NOMINAL["h0"]), 20 * (d["k"] - 0.0012),
           2 * (d["z_obj"] - 0.015), 0.05 * d["px"], 0.05 * d["py"]]
    return np.concatenate(res + [np.array(reg)])


def lm(p, fun, iters=200, lam=1e-2):
    """Small Levenberg-Marquardt with a numerical Jacobian."""
    p = p.astype(float).copy()
    r = fun(p)
    cost = r @ r
    for _ in range(iters):
        J = np.empty((r.size, p.size))
        for i in range(p.size):
            h = 1e-6 * max(1.0, abs(p[i]))
            pp = p.copy()
            pp[i] += h
            J[:, i] = (fun(pp) - r) / h
        A = J.T @ J
        g = J.T @ r
        while True:
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -g)
            pn = p + step
            rn = fun(pn)
            cn = rn @ rn
            if cn < cost:
                p, r, cost, lam = pn, rn, cn, max(lam / 3, 1e-7)
                break
            lam *= 4
            if lam > 1e8:
                return p, cost
        if np.linalg.norm(step) < 1e-9:
            break
    return p, cost


def point_err(p, signs, mirror, uv, q):
    g = fk(p, signs, q)
    t = pix2arm(p, mirror, uv)
    return np.hypot(g[:, 0] - t[:, 0], g[:, 1] - t[:, 1])


def fit_arm(uv, q, base_px, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for signs in [(a, b, c, e) for a in (1, -1) for b in (1, -1) for c in (1, -1) for e in (1, -1)]:
        for mirror in (False, True):
            for trial in range(3):
                p0 = np.array([rng.uniform(-40, 40), rng.uniform(-60, 60), rng.uniform(-60, 60), rng.uniform(-60, 60),
                               NOMINAL["L1"], NOMINAL["L2"], NOMINAL["L3"], NOMINAL["r0"], NOMINAL["h0"], 0.015,
                               base_px[0], base_px[1], 0.0012, rng.uniform(-math.pi, math.pi), 0.0, 0.0])
                keep = np.ones(len(uv), bool)
                p = p0
                for _ in range(4):        # fit, drop the worst pairings, refit
                    p, _ = lm(p, lambda x: residuals(x, signs, mirror, uv[keep], q[keep]), iters=80)
                    e = point_err(p, signs, mirror, uv, q)
                    keep = e < max(0.02, 2.5 * np.median(e[keep]))
                e = point_err(p, signs, mirror, uv, q)
                score = np.median(e[keep]) + 0.05 * (1 - keep.mean())
                if best is None or score < best[0]:
                    best = (score, p, signs, mirror, keep)
    return best


if __name__ == "__main__":
    d = np.load(HERE / "grab_points.npz")
    BASES = {0: (455, 430), 1: (235, 420)}       # rough base pixels; the fit decides (tries both per arm)
    out = {}
    for arm, side in ((0, "left"), (1, "right")):
        m = (d["arm"] == arm) & (d["joints"][:, 1] > -80)
        uv = np.stack([d["u"][m], d["v"][m]], 1)
        q = d["joints"][m][:, :4]
        results = []
        for base in BASES.values():
            results.append(fit_arm(uv, q, base))
        score, p, signs, mirror, keep = min(results, key=lambda b: b[0])
        e = point_err(p, signs, mirror, uv, q)
        # held-out check: refit on 80 %, test on the other 20 % (inliers only)
        idx = np.flatnonzero(keep)
        rng = np.random.default_rng(1)
        test = rng.choice(idx, max(3, len(idx) // 5), replace=False)
        train = np.setdiff1d(idx, test)
        pt, _ = lm(p, lambda x: residuals(x, signs, mirror, uv[train], q[train]), iters=150)
        et = point_err(pt, signs, mirror, uv[test], q[test])
        dd = dict(zip(PNAMES, p))
        print(f"{side:5s}: {keep.sum()}/{len(uv)} grabs kept | fit error median {1000 * np.median(e[keep]):.1f} mm, "
              f"90% {1000 * np.percentile(e[keep], 90):.1f} mm | HELD-OUT median {1000 * np.median(et):.1f} mm, max "
              f"{1000 * et.max():.1f} mm | signs {signs} mirror {mirror} | L {dd['L1']:.3f}/{dd['L2']:.3f}/{dd['L3']:.3f} "
              f"k {1000 * dd['k']:.2f} mm/px base px ({dd['bx']:.0f},{dd['by']:.0f}) z_obj {1000 * dd['z_obj']:.0f} mm")
        out[side] = dict(params=dict(zip(PNAMES, map(float, p))), signs=list(signs), mirror=bool(mirror),
                         fit_mm=float(1000 * np.median(e[keep])), heldout_mm=float(1000 * np.median(et)),
                         n=int(keep.sum()))
    (HERE / "arm_fit.json").write_text(json.dumps(out, indent=1))
    print("saved", HERE / "arm_fit.json")
