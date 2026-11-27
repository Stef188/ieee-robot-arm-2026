"""ARM KINEMATICS from data (27 Nov, competition): forward + inverse kinematics of each SO-101 arm, FITTED from the
recorded grabs (no URDF / kinematics library on this laptop).

Model (per arm), in the vertical plane the arm turns with its pan joint:
    a1 = s1*lift + o1        upper-arm angle from straight up
    a2 = s2*elbow + o2       forearm angle relative to the upper arm
    a3 = s3*wrist_flex + o3  gripper angle relative to the forearm
    reach  x = x0 + L1 sin(a1) + L2 sin(a1+a2) + L3 sin(a1+a2+a3)       (cm, from the pan axis)
    height z = h0 + L1 cos(a1) + L2 cos(a1+a2) + L3 cos(a1+a2+a3)       (cm, 0 = where the jaws grab a block)
Camera pixel -> reach: r = |(u, v) - base| * k0 * (1 + g (v - 380)/100)   (cm; g = perspective of the tilted camera)
Fitted on every recorded grab (gripper at grab height over a known pixel): L1 is fixed at 11.6 cm (only sets the
unit), everything else is solved for. Honest test: fitted on 80 % of the grabs, scored on the other 20 %.

    python arm_kinematics.py       -> fits both arms, prints accuracy, saves arm_kinematics.json
"""
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
L1 = 11.6
BASE0 = {"left": (242.0, 420.0), "right": (458.0, 430.0)}
NAMES = ["L2", "L3", "o1", "o2", "o3", "h0", "x0", "bu", "bv", "k0", "g", "dh"]
# loose physical priors (SO-101 size): forearm ~13 cm, wrist->jaw centre ~10 cm, shoulder axis a few cm from the pan
# axis and ~8-15 cm above the grab height; jaws point DOWN when grabbing (pitch 180 deg); hover = straight above grab
PRIOR = {"L2": (13.0, 3.0), "L3": (10.0, 3.0), "x0": (2.0, 3.0), "h0": (10.0, 6.0)}


def fk(p, s, q):
    """q: (n, 3) lift, elbow, wrist flex (deg) -> reach x, height z (cm), gripper pitch (deg from straight up)."""
    L2, L3, o1, o2, o3, h0, x0 = p[:7]
    a1 = np.radians(s[0] * q[:, 0] + o1)
    a12 = a1 + np.radians(s[1] * q[:, 1] + o2)
    a123 = a12 + np.radians(s[2] * q[:, 2] + o3)
    x = x0 + L1 * np.sin(a1) + L2 * np.sin(a12) + L3 * np.sin(a123)
    z = h0 + L1 * np.cos(a1) + L2 * np.cos(a12) + L3 * np.cos(a123)
    return x, z, np.degrees(a123)


def reach_cm(p, u, v):
    bu, bv, k0, g = p[7:11]
    return np.hypot(np.asarray(u) - bu, np.asarray(v) - bv) * k0 * (1 + g * (np.asarray(v) - 380) / 100)


def ik(p, s, x, z, pitch, ref=None):
    """Joints (lift, elbow, wrist flex) that put the jaws at reach x, height z with gripper pitch `pitch` (deg).
    Two solutions (elbow up / down): the one closest to `ref` (else elbow-up)."""
    L2, L3, o1, o2, o3, h0, x0 = p[:7]
    ph = math.radians(pitch)
    wx, wz = x - x0 - L3 * math.sin(ph), z - h0 - L3 * math.cos(ph)       # wrist point, relative to the shoulder
    d2 = wx * wx + wz * wz
    c2 = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c2) > 1:
        return None                                                        # out of reach
    out = []
    for sgn in (1, -1):
        b = sgn * math.acos(c2)                                            # a2 (elbow bend)
        a1 = math.atan2(wx, wz) - math.atan2(L2 * math.sin(b), L1 + L2 * math.cos(b))
        a3 = ph - a1 - b
        q = np.array([(math.degrees(a1) - o1) / s[0], (math.degrees(b) - o2) / s[1], (math.degrees(a3) - o3) / s[2]])
        q = (q + 180) % 360 - 180
        out.append(q)
    if ref is not None:
        return min(out, key=lambda q: np.abs(q - np.asarray(ref)).sum())
    return out[0]


def load_points(side):
    c = json.loads((HERE / f"calib_{side}.json").read_text())
    P = [p for p in c["points"] if not str(p.get("time", "")).startswith("2026-11-27") and not p.get("comp")]
    U = np.array([p["u"] for p in P]); V = np.array([p["v"] for p in P])
    Q = np.array([p["grasp"][:5] for p in P], float)
    H = np.array([p["hover"][:5] for p in P], float)
    return U, V, Q, H


def residuals(p, s, U, V, Q, H):
    x, z, pitch = fk(p, s, Q[:, 1:4])
    xh, zh, _ = fk(p, s, H[:, 1:4])
    dpitch = (pitch - 180 + 180) % 360 - 180
    pri = [(p[NAMES.index(k)] - m) / sd for k, (m, sd) in PRIOR.items()]
    return np.concatenate([x - reach_cm(p, U, V), z,                       # grab: jaws at the block, grab height
                           0.5 * (xh - x), 0.5 * (zh - p[11]),               # hover: straight above, dh cm up
                           dpitch / 30.0,                                    # jaws roughly pointing down
                           np.array(pri) * 2.0])


def fit(side, U, V, Q, H):
    best = None
    b0 = BASE0[side]
    for s in itertools.product((1, -1), repeat=3):
        for o1, o2, o3 in itertools.product((-90, 0, 90), repeat=3):
            p0 = [13.0, 10.0, o1, o2, o3, 10.0, 2.0, b0[0], b0[1], 0.12, 0.0, 5.0]
            lo = [5, 3, -400, -400, -400, -20, -10, b0[0] - 60, b0[1] - 60, 0.05, -0.5, 1.0]
            hi = [25, 20, 400, 400, 400, 40, 15, b0[0] + 60, b0[1] + 60, 0.3, 0.5, 15.0]
            try:
                r = least_squares(residuals, p0, bounds=(lo, hi), args=(s, U, V, Q, H), loss="soft_l1", f_scale=1.0,
                                  max_nfev=400)
            except Exception:
                continue
            cost = float(np.sum(np.minimum(r.fun ** 2, 25.0)))
            if best is None or cost < best[0]:
                best = (cost, r.x, s)
    return best


def main():
    out = {}
    rng = np.random.default_rng(0)
    for side in ("left", "right"):
        U, V, Q, H = load_points(side)
        idx = rng.permutation(len(U))
        te, tr = idx[: len(U) // 5], idx[len(U) // 5:]
        cost, p, s = fit(side, U[tr], V[tr], Q[tr], H[tr])
        x, z, pitch = fk(p, s, Q[:, 1:4])
        ex = np.abs(x - reach_cm(p, U, V)); ez = np.abs(z)
        print(f"\n{side.upper()} arm: {len(U)} recorded grabs | signs {s} | " + ", ".join(f"{n} {v:.2f}" for n, v in zip(NAMES, p)))
        print(f"   position error on grabs NOT used in the fit: reach {np.median(ex[te]):.2f} cm (90% < {np.percentile(ex[te], 90):.2f}), "
              f"height {np.median(ez[te]):.2f} cm (90% < {np.percentile(ez[te], 90):.2f})")
        print(f"   (on the fitted grabs: reach {np.median(ex[tr]):.2f} cm, height {np.median(ez[tr]):.2f} cm) | grab pitch median "
              f"{np.median(pitch):.0f} deg")
        # final fit on everything
        cost, p, s = fit(side, U, V, Q, H)
        out[side] = dict(p=list(map(float, p)), s=list(s), names=NAMES, L1=L1, n=int(len(U)),
                         test_reach_cm=float(np.median(ex[te])), test_height_cm=float(np.median(ez[te])))
    (HERE / "arm_kinematics.json").write_text(json.dumps(out, indent=1))
    print("\nsaved arm_kinematics.json")


if __name__ == "__main__":
    main()
