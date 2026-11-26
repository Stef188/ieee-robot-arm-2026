"""DropModel: arm drop pose for a bin seen ANYWHERE by the top camera (Round 3 - moving bins). 26 Nov.
    python classical\\drop_model.py        -> fits both arms on classical\\shape_data\\drops.npz, prints leave-one-out
                                           errors, saves drop_model_left.json / drop_model_right.json
Learned from the demo releases (drop_dataset.py: the joints at every moment a demonstrator let go of a block over a bin,
and where that bin was in the camera). In the shape demos the bins stood at many different places.
Model per arm (least squares, a few numbers - the data is ~100s of releases):
  pan          = a * (direction of the bin seen from the fitted arm base) + c
  lift / elbow / wrist flex = quadratic in the distance r from the base + linear in the direction
  wrist roll   = median of the releases
covers(u, v): only inside the range of distances / directions it has seen (+ a small margin) - elsewhere it says no.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
BASE0 = {"left": (242.0, 420.0), "right": (458.0, 430.0)}


def _feat(u, v, bu, bv):
    u, v = np.asarray(u, float), np.asarray(v, float)
    h = np.degrees(np.arctan2(u - bu, bv - v))
    r = np.hypot(u - bu, v - bv)
    return h, r


def _design(h, r, r0):
    x = (r - r0) / 100.0
    return np.stack([np.ones_like(x), x, x * x, h / 45.0], -1)


class DropModel:
    def __init__(self, d):
        self.d = d

    @classmethod
    def load(cls, side):
        return cls(json.loads((HERE / f"drop_model_{side}.json").read_text()))

    def covers(self, u, v):
        d = self.d
        h, r = _feat(u, v, d["bu"], d["bv"])
        return (d["r_min"] - 15 <= float(r) <= d["r_max"] + 15) and (d["h_min"] - 6 <= float(h) <= d["h_max"] + 6)

    def pose(self, u, v):
        d = self.d
        h, r = _feat(u, v, d["bu"], d["bv"])
        pan = d["pan_a"] * float(h) + d["pan_c"]
        X = _design(np.array([float(h)]), np.array([float(r)]), d["r0"])
        rest = (X @ np.array(d["W"]))[0]
        return np.array([pan, rest[0], rest[1], rest[2], d["roll"]], float)


def fit_side(u, v, q, side):
    """Fit one arm; returns (model dict, leave-one-out mean abs error per joint)."""
    bu0, bv0 = BASE0[side]

    def fit(idx):
        best = None
        for bu in np.arange(bu0 - 30, bu0 + 31, 6):             # base pixel: small grid search (pan residual)
            for bv in np.arange(bv0 - 40, bv0 + 21, 6):
                h, _ = _feat(u[idx], v[idx], bu, bv)
                A = np.c_[h, np.ones_like(h)]
                coef, res, *_ = np.linalg.lstsq(A, q[idx, 0], rcond=None)
                err = np.abs(A @ coef - q[idx, 0]).mean()
                if best is None or err < best[0]:
                    best = (err, bu, bv, coef)
        _, bu, bv, (a, c) = best
        h, r = _feat(u[idx], v[idx], bu, bv)
        r0 = float(np.median(r))
        W = np.linalg.lstsq(_design(h, r, r0), q[idx, 1:4], rcond=None)[0]
        return dict(bu=float(bu), bv=float(bv), pan_a=float(a), pan_c=float(c), r0=r0, W=W.tolist(),
                    roll=float(np.median(q[idx, 4])), r_min=float(r.min()), r_max=float(r.max()),
                    h_min=float(h.min()), h_max=float(h.max()), n=int(len(idx)))

    all_idx = np.arange(len(u))
    errs = []
    for i in all_idx:
        mdl = DropModel(fit(all_idx[all_idx != i]))
        errs.append(np.abs(mdl.pose(u[i], v[i])[:4] - q[i, :4]))
    return fit(all_idx), np.array(errs).mean(0)


def main():
    d = np.load(HERE / "shape_data" / "drops.npz", allow_pickle=True)
    for arm, side in enumerate(("left", "right")):
        k = d["arm"] == arm
        u, v, q = d["u"][k], d["v"][k], d["joints"][k]
        if k.sum() < 12:
            print(f"{side}: only {int(k.sum())} releases - no model")
            continue
        mdl, loo = fit_side(u, v, q, side)
        places = len({(round(a / 20), round(b / 20)) for a, b in zip(u, v)})
        print(f"{side}: {int(k.sum())} releases over {places} bin places | leave-one-out error "
              f"(pan, lift, elbow, wrist flex) {np.round(loo, 1).tolist()} deg | covers distance "
              f"{mdl['r_min']:.0f}-{mdl['r_max']:.0f} px, direction {mdl['h_min']:.0f}..{mdl['h_max']:.0f} deg")
        (HERE / f"drop_model_{side}.json").write_text(json.dumps(mdl, indent=1))


if __name__ == "__main__":
    main()
