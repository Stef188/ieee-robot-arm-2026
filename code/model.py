"""Pixel -> joint angles for the classical sorter, from the live calibration (calib_left/right.json).

No arm geometry needed: the joint angles you showed at each calibration point are interpolated smoothly over the
table (thin-plate spline in pixel space), separately for the GRASP pose and the HOVER pose (~5 cm above).
Wrist roll follows the object's rotation: roll = smooth base value + (object angle, folded to +-45 deg for square
sides) with the sign that fits your calibration best.
python classical\\model.py   -> leave-one-out check: how far off each calibration point is predicted from the others
"""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REACH_MARGIN = 25.0          # px: how far outside the calibrated area an object may be and still be grabbed
PAN_GEO = os.environ.get("SORT_PAN", "geo") == "geo"   # 26 Nov pm: geometric pan (ArmMap._fit_pan_geo); SORT_PAN=model = old
SLOT_R = float(os.environ.get("SORT_SLOT_R") or 35)   # px (~4 cm): a bin this close to a stored bin spot uses that spot's drop pose


def fold(angle, period=90.0):
    """Object rotation folded to (-period/2, period/2]: a square looks the same every 90 deg."""
    a = (angle + period / 2) % period - period / 2
    return a


class TPS:
    """Thin-plate spline from (u, v) to a vector, with a little smoothing."""

    def __init__(self, uv, y, smooth=1.0):
        self.c = uv.mean(0)
        self.s = uv.std() + 1e-9
        x = (uv - self.c) / self.s
        n = len(x)
        K = self._k(x, x) + smooth * 1e-3 * np.eye(n)
        P = np.hstack([np.ones((n, 1)), x])
        A = np.zeros((n + 3, n + 3))
        A[:n, :n], A[:n, n:], A[n:, :n] = K, P, P.T
        b = np.zeros((n + 3, y.shape[1]))
        b[:n] = y
        self.w = np.linalg.solve(A, b)
        self.x = x

    @staticmethod
    def _k(a, b):
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            k = d * d * np.log(d)
        return np.nan_to_num(k)

    def __call__(self, uv):
        x = (np.atleast_2d(uv) - self.c) / self.s
        n = len(self.x)
        return self._k(x, self.x) @ self.w[:n] + np.hstack([np.ones((len(x), 1)), x]) @ self.w[n:]


class Poly:
    """Smooth cubic fit (u, v) -> vector, least squares. Used when the points come from DEMOS (many, noisy: the hand
    grabbed with different wrist/elbow postures) - it averages them instead of copying each one like TPS does.
    25 Nov, left arm from 202 colour demos: leave-one-out 5.9 deg (TPS 10.9)."""

    def __init__(self, uv, y, smooth=None):
        self.w, *_ = np.linalg.lstsq(self._f(uv), y, rcond=None)

    @staticmethod
    def _f(uv):
        uv = np.atleast_2d(uv)
        u, v = (uv[:, 0] - 350) / 100, (uv[:, 1] - 330) / 100
        return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v, u ** 3, u * u * v, u * v * v, v ** 3], 1)

    def __call__(self, uv):
        return self._f(uv) @ self.w


class GraspNet:
    """Learned grasp-pose model (learning from demonstration): a small neural network ensemble that maps where the
    block is in the top-camera picture (u, v) to the 5 arm joint angles a human operator used to grab blocks there.
    Trained on the grab moments extracted from the teleoperated colour demos (calib_from_demos.py).
    Architecture per member: 2 -> 16 tanh -> 16 tanh -> 5, plus a linear skip path; outputs standardised per joint.
    Training: full-batch Adam, lr 0.02, 3000 steps, L2 weight decay 2e-3, mean-squared error; 5 members (seeds 0-4)
    averaged. Pure numpy, deterministic. Weights are saved by train_grasp_net.py (grasp_net_<side>.npz)."""

    HIDDEN, STEPS, LR, WD, SEEDS = 16, 3000, 2e-2, 2e-3, 5

    def __init__(self, uv=None, y=None, smooth=None, params=None):
        if params is not None:                      # load saved weights
            self.ym, self.ys, self.nets = params["ym"], params["ys"], params["nets"]
            return
        self.ym, self.ys = y.mean(0), y.std(0) + 1e-6
        x, t = self._x(uv), (y - self.ym) / self.ys
        self.nets = [self._train(x, t, s) for s in range(self.SEEDS)]

    @staticmethod
    def _x(uv):
        uv = np.atleast_2d(np.asarray(uv, float))
        return np.stack([(uv[:, 0] - 350) / 100, (uv[:, 1] - 330) / 100], 1)

    @staticmethod
    def _fwd(p, x):
        h1 = np.tanh(x @ p[0] + p[1])
        h2 = np.tanh(h1 @ p[2] + p[3])
        return h1, h2, h2 @ p[4] + p[5] + x @ p[6]

    def _train(self, x, t, seed):
        H, o, n = self.HIDDEN, t.shape[1], len(x)
        r = np.random.default_rng(seed)
        p = [r.normal(0, 1 / np.sqrt(2), (2, H)), np.zeros(H), r.normal(0, 1 / np.sqrt(H), (H, H)), np.zeros(H),
             r.normal(0, 0.1 / np.sqrt(H), (H, o)), np.zeros(o), np.zeros((2, o))]
        m1 = [np.zeros_like(a) for a in p]
        m2 = [np.zeros_like(a) for a in p]
        for k in range(1, self.STEPS + 1):
            h1, h2, out = self._fwd(p, x)
            g = 2 * (out - t) / n / o                                  # d(MSE)/d(out)
            d2 = (g @ p[4].T) * (1 - h2 ** 2)
            d1 = (d2 @ p[2].T) * (1 - h1 ** 2)
            gs = [x.T @ d1, d1.sum(0), h1.T @ d2, d2.sum(0), h2.T @ g, g.sum(0), x.T @ g]
            for i in (0, 2, 4):
                gs[i] = gs[i] + self.WD * p[i]
            for i in range(len(p)):                                    # Adam
                m1[i] = 0.9 * m1[i] + 0.1 * gs[i]
                m2[i] = 0.999 * m2[i] + 0.001 * gs[i] ** 2
                p[i] -= self.LR * (m1[i] / (1 - 0.9 ** k)) / (np.sqrt(m2[i] / (1 - 0.999 ** k)) + 1e-8)
        return p

    def __call__(self, uv):
        x = self._x(uv)
        return np.mean([self._fwd(p, x)[2] for p in self.nets], 0) * self.ys + self.ym

    def to_npz(self, path, **meta):
        d = {"ym": self.ym, "ys": self.ys, "n_nets": len(self.nets)}
        for i, p in enumerate(self.nets):
            for j, a in enumerate(p):
                d[f"net{i}_p{j}"] = a
        d.update({k: np.asarray(v) for k, v in meta.items()})
        np.savez(path, **d)

    @classmethod
    def from_npz(cls, path):
        z = np.load(path)
        nets = [[z[f"net{i}_p{j}"] for j in range(7)] for i in range(int(z["n_nets"]))]
        return cls(params={"ym": z["ym"], "ys": z["ys"], "nets": nets}), z


class Combined:
    """Joints 0-3 from the network, joint 4 (wrist roll) from the cubic fit."""

    def __init__(self, net, roll_fit):
        self.net, self.roll_fit = net, roll_fit

    GUARD_DEG = float(os.environ.get("SORT_NET_GUARD") or 90.0)   # 26 Nov 22:00: guard OFF - all runs today:
    #   network grabs held left 32/45, right 11/46; "fit (guard)" grabs only left 9/19, right 5/51 (SORT_NET_GUARD=6 = old)

    def __call__(self, uv):
        out = self.net(uv).copy()
        fit = self.roll_fit(uv)
        out[:, 4] = fit[:, 4]
        # 26 Nov "is it shit" guard: if the network disagrees with the cubic fit by more than GUARD_DEG on any
        # position joint, this grab uses the cubic fit instead (logged as "fit (guard)")
        # 26 Nov pm: with the geometric pan (PAN_GEO, see ArmMap.poses) the pan is not taken from either, so only
        # lift / elbow / wrist flex are compared
        j0 = 1 if PAN_GEO else 0
        bad = np.abs(out[:, j0:4] - fit[:, j0:4]).max(1) > self.GUARD_DEG
        out[bad] = fit[bad]
        self.last = "fit (guard)" if bad.any() else "network"
        self.last_diff = float(np.abs(self.net(uv)[:, j0:4] - fit[:, j0:4]).max())
        return out


def load_or_train_net(side, which, uv, y):
    """The saved network for this arm (grasp_net_<side>_<which>.npz) if it was trained on exactly these points,
    else train it now (~10 s) and save it."""
    variant = os.environ.get("SORT_NET")
    if variant:
        # 26 Nov: one of the benchmark variants (benchmark_variants.py -> model_variants\), e.g. SORT_NET=5_int8
        vp = HERE / "model_variants" / f"{variant}_{side}_{which}.npz"
        if vp.exists():
            return GraspNet.from_npz(vp)[0]
        print(f"   SORT_NET={variant}: no {vp.name} - using the normal network")
    path = HERE / f"grasp_net_{side}_{which}.npz"
    h = hashlib.sha256(np.ascontiguousarray(uv, float).tobytes() + np.ascontiguousarray(y, float).tobytes()).hexdigest()
    if path.exists():
        net, z = GraspNet.from_npz(path)
        if str(z["data_sha256"]) == h:
            return net
    net = GraspNet(uv, y)
    net.to_npz(path, data_sha256=h, n_points=len(uv), side=side, which=which,
               arch="2-16-16-5 tanh MLP + linear skip, 5-member ensemble")
    return net


class ArmMap:
    def __init__(self, calib, roll_sign=None):
        self.side = calib["side"]
        self.rest = np.array(calib["rest"])
        # 26 Nov: hand-taught grab spots (teach_grab.py) carry a weight "w" (3) - they count that many times
        pts = [q for q in calib["points"] for _ in range(int(q.get("w") or 1))]
        self.uv = np.array([[p["u"], p["v"]] for p in pts])
        self.ang = np.array([fold(p["angle"]) for p in pts])
        self.grasp = np.array([p["grasp"] for p in pts])
        self.hover = np.array([p["hover"] for p in pts])
        self.bins = calib.get("bins", {})
        self.pan_corr = calib.get("pan_correction")
        self.last_source = None
        self.roll_offset = 0.0               # set by sort.py (SORT_PICK_ROLL_LEFT/RIGHT)
        if "source" not in calib and len(pts) >= 2:
            # 25 Nov: hand calibration of a CUBE: the wrist roll was set differently each time (132, 166, 4, -68, 173,
            # -180 ...). A cube looks the same every 90 deg, so these are all equivalent grabs - fold every roll to the
            # 90-deg window around the most common one, or interpolating between 173 and -180 would spin the wrist.
            r = self.grasp[:, 4]
            fold90 = lambda x: (x + 45) % 90 - 45
            ref = min(r, key=lambda c: np.abs(fold90(r - c)).sum())
            shift = (ref + fold90(r - ref)) - r
            folded = r + shift
            # keep the whole window inside the joint's range (+-180): move all by whole 90-deg turns if needed
            k = min(range(-2, 3), key=lambda k: np.abs(folded + 90 * k).max())
            shift = shift + 90 * k
            self.grasp[:, 4] += shift
            self.hover[:, 4] += shift
            self.roll_ref = float(ref)
        self.Fit = Poly if len(pts) >= 30 else TPS          # many noisy demo points -> smooth fit; hand calibration -> TPS
        # demo-built calibration: "up" = this arm's average lift in joints; the wrist target was taken just above grab
        # height. Hand calibration: no "up" (its hover pose is used), wrist target taken at the hover pose.
        self.up = np.array(calib["up"][:5]) if calib.get("up") else None
        self.from_demos = "source" in calib
        if self.up is None and len(pts):
            # 25 Nov: hand calibration -> "up" = your average lift from the grab pose to the hover pose (~5 cm), lift /
            # elbow / wrist flex only, so the high start (SORT_UP x up) and the high drop work for it too
            d = np.mean(self.hover[:, :5] - self.grasp[:, :5], 0)
            d[[0, 4]] = 0.0
            self.up = d
        # wrist-camera target: where the object sat in the wrist picture with the gripper centred 5 cm above it
        wa = np.array([p["wrist"] for p in pts if p.get("wrist")])
        w = wa[:, :2] if len(wa) else np.zeros((0, 2))
        self.wrist_target = self.wrist_spread = None
        a = wa[:, 2] if len(wa) else np.zeros(0)
        if len(w) >= 3:
            # robust: the points that agree (within 0.08 of the median) set the target; stray detections (bins, jaw
            # colour fringes) are ignored. Needs a clear majority (60 %), else fine alignment stays OFF - steering to a
            # scattered target makes grabs WORSE. 25 Nov demos: left 89 % agree (spread 0.035) = ON, right 32 % = OFF.
            inl = np.linalg.norm(w - np.median(w, 0), axis=1) < 0.08
            med = np.median(w[inl], 0)
            self.wrist_spread = float(np.linalg.norm(w[inl] - med, axis=1).mean())
            self.wrist_agree = float(inl.mean())
            if inl.mean() >= 0.6:
                self.wrist_target = tuple(med)
                a = a[inl]
            else:
                self.wrist_spread = float(np.linalg.norm(w - np.median(w, 0), axis=1).mean())
        # how big the object looks from hover height: only blobs of about this size count (bins are far bigger,
        # table specks / jaw colour fringes far smaller)
        self.wrist_area = float(np.median(a)) if len(a) >= 3 else None
        self.wrist_feature = "centre"
        if calib.get("wrist_edge_target"):
            # 25 Nov, RIGHT arm: its camera loses the block's left side (moving jaw side) - steer by the block's RIGHT
            # edge against the fixed jaw instead (demos: 71 % agree, spread 0.027)
            self.wrist_feature = "edge"
            self.wrist_target = tuple(calib["wrist_edge_target"])
            self.wrist_area = calib.get("wrist_edge_area", 0.15)
            self.wrist_spread, self.wrist_agree = 0.027, 0.71
        if roll_sign is None:      # pick the roll sign that predicts the calibration best (leave-one-out)
            roll_sign = min((1.0, -1.0), key=lambda s: self.loo(s)[0])
        self.roll_sign = roll_sign
        y_g = self.grasp[:, :5].copy()
        y_h = self.hover[:, :5].copy()
        y_g[:, 4] -= roll_sign * self.ang
        y_h[:, 4] -= roll_sign * self.ang
        # 26 Nov: demo-built calibration -> the LEARNED grasp-pose network (GraspNet, trained on the demo grabs) gives the
        # arm pose. SORT_GRASP=poly switches back to the old cubic curve fit.
        self.grasp_model = "poly" if self.Fit is Poly else "tps"
        if self.from_demos and os.environ.get("SORT_GRASP", "net").lower() != "poly":
            # the network gives the POSITION joints (pan, lift, elbow, wrist flex = where the gripper goes); the wrist
            # roll stays on the cubic fit - the demos turned the wrist inconsistently (right arm 0-166 deg, a cube is
            # the same every 90), and the cubic roll is the one proven on the arms (network roll differed by 13 deg)
            self.f_grasp = Combined(load_or_train_net(self.side, "grasp", self.uv, y_g), self.Fit(self.uv, y_g))
            self.f_hover = Combined(load_or_train_net(self.side, "hover", self.uv, y_h), self.Fit(self.uv, y_h))
            self.grasp_model = (f"neural network (GraspNet) for position joints + cubic fit for wrist roll, guard: cubic fit "
                                f"if they differ > {Combined.GUARD_DEG:.0f} deg"
                                + (f" | variant {os.environ['SORT_NET']}" if os.environ.get("SORT_NET") else ""))
        else:
            self.f_grasp = self.Fit(self.uv, y_g)
            self.f_hover = self.Fit(self.uv, y_h)
        allq = np.vstack([self.grasp, self.hover, self.rest[None]] + [np.array(b["hover"])[None] for b in self.bins.values()])
        self.limits = {j: (float(allq[:, i].min() - 10), float(allq[:, i].max() + 10))
                       for i, j in enumerate(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"])}
        self.pan_geo = self._fit_pan_geo() if (PAN_GEO and len(self.uv) >= 8) else None
        if self.pan_geo is not None:
            lo, hi = self.limits["shoulder_pan"]
            self.limits["shoulder_pan"] = (lo - 10, hi + 10)      # the geometric pan is valid a little further out
            self.grasp_model = getattr(self, "grasp_model", "") + " | pan: geometric (angle from the arm base)"
        import cv2
        self.hull = cv2.convexHull(self.uv.astype(np.float32))

    def widen_limits(self, k, k_drop=0.0):
        """Allow the joint values of 'k x up above every calibration grab' (the high start of each grab) and of
        'k_drop x up above every bin pose' (the high drop)."""
        if self.up is None:
            return
        q = self.grasp[:, :5] + k * self.up
        if self.bins and k_drop:
            q = np.vstack([q] + [np.array(b["hover"][:5]) + k_drop * self.up for b in self.bins.values()])
        for i, j in enumerate(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]):
            lo, hi = self.limits[j]
            self.limits[j] = (min(lo, float(q[:, i].min() - 5)), max(hi, float(q[:, i].max() + 5)))

    def _fit_pan_geo(self):
        """26 Nov pm: PAN (which way the arm turns) = a x (direction from the arm base in the top camera) + c, base pixel
        fitted too - 4 numbers fitted to the demo grabs. Leave-one-out: left 1.26 deg (cubic fit 1.27, network 1.84),
        right 2.13 (2.08, 2.61); on held-out grabs nearest the middle it is the best of the three (1.1 / 1.4 deg).
        Outside the demo area it keeps turning the right way - the cubic fit turned BACK there (wrong by 10-15 deg =
        4-5 cm) and the network over-turned."""
        uv, pan = self.uv, self.grasp[:, 0]
        b0 = (242.0, 420.0) if self.side == "left" else (458.0, 430.0)

        def res(x):
            return x[0] * np.degrees(np.arctan2(uv[:, 0] - x[2], x[3] - uv[:, 1])) + x[1] - pan
        x, lam = np.array([1.0, 0.0, b0[0], b0[1]]), 1e-2       # small Levenberg-Marquardt (numpy only)
        for _ in range(200):
            r = res(x)
            J = np.column_stack([(res(x + e) - r) / 1e-4 for e in np.eye(4) * 1e-4])
            A = J.T @ J
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -J.T @ r)
            if (res(x + step) ** 2).sum() < (r ** 2).sum():
                x, lam = x + step, lam * 0.3
                if np.abs(step).max() < 1e-7:
                    break
            else:
                lam *= 10
        self.pan_geo_params = [float(t) for t in x]
        return lambda u, v: float(x[0] * math.degrees(math.atan2(u - x[2], x[3] - v)) + x[1])

    def reach(self, u, v):
        """Signed distance (px) inside the calibrated area (negative = outside)."""
        import cv2
        return cv2.pointPolygonTest(self.hull, (float(u), float(v)), True)

    def can_reach(self, u, v):
        return self.reach(u, v) >= -REACH_MARGIN

    def pan_fix(self, u, v):
        """Optional smooth pan correction from a hand calibration (calib 'pan_correction': [c0, cu, cv] in deg for
        1, (u-450)/50, (v-350)/30). Pan = which way the arm faces the object: one right answer per spot, whatever
        posture the shoulder/elbow use - so hand points can correct it even when their postures differ."""
        c = self.pan_corr
        return 0.0 if c is None else c[0] + c[1] * (u - 450) / 50 + c[2] * (v - 350) / 30

    def poses(self, u, v, angle, period=90.0):
        """(grasp, hover) arm joints (5) for an object at pixel (u, v) rotated `angle` deg. period 90 = square-ish
        block (any of 4 sides is fine); 180 = long block (lying cylinder): the jaws must close ACROSS it."""
        a = fold(angle, period)
        g = self.f_grasp([u, v])[0]
        h = self.f_hover([u, v])[0]
        if isinstance(self.f_grasp, Combined):
            self.last_source = (f"grasp {self.f_grasp.last} / hover {self.f_hover.last} "
                                f"(network vs fit {self.f_grasp.last_diff:.1f} deg)")
        if self.pan_geo is not None:          # 26 Nov pm: geometric pan (see _fit_pan_geo), same for grab and hover
            g[0] = h[0] = self.pan_geo(u, v)
        dp = self.pan_fix(u, v)
        g[0] += dp
        h[0] += dp
        if self.roll_offset:
            # 25 Nov: grab with the wrist turned (default 90 deg) so the wrist camera faces SIDEWAYS, away from the
            # bins. A cube looks the same after 90 deg, so the jaws still close on two faces. If the turn would pass
            # the roll joint's limit (+-170), turn the other way instead (same jaw angle, camera faces the other side).
            r = self.roll_offset if abs(g[4] + self.roll_offset) <= 170 else self.roll_offset - 180 * np.sign(self.roll_offset)
            g[4] += r
            h[4] += r
        g[4] = float(np.clip(g[4], -170, 170))   # never past the roll joint's wrap point
        h[4] = float(np.clip(h[4], -170, 170))
        g[4] += self.roll_sign * a
        h[4] += self.roll_sign * a
        if abs(g[4]) > 170:                    # jaws are the same after a half turn: stay inside the roll joint's range
            d = -180.0 * np.sign(g[4])
            g[4] += d
            h[4] += d
        return g, h

    def bin_slot(self, colour, u, v):
        """26 Nov pm: which of this arm's stored bin SPOTS a bin of `colour` seen at pixel (u, v) is on. The drop poses
        belong to the spot, not the colour: if the organisers swap bins (e.g. red <-> yellow, as in the official
        picture), the arm next to a spot drops whatever bin colour now stands there. None = not on a known spot."""
        # 26 Nov 19:45: any stored spot of this arm, hand-taught ones too (taught bins used to ignore the camera);
        # at the same spot a hand-taught pose wins over the demo one, whatever bin colour stands there
        near = [(math.hypot(s["u"] - u, s["v"] - v) - (5 if s.get("taught") else 0), name)
                for name, s in self.bins.items() if "u" in s]
        near = [t for t in near if t[0] < SLOT_R]
        return min(near)[1] if near else None

    def bin_hover(self, colour, u, v):
        """Hover pose above a bin: the stored pose of the spot it stands on, else interpolated if reachable."""
        s = self.bin_slot(colour, u, v)
        if s is not None:
            return np.array(self.bins[s]["hover"][:5])
        if self.can_reach(u, v):
            return self.poses(u, v, 0.0)[1]
        return None

    def loo(self, roll_sign):
        """Leave-one-out: predict each calibration grasp from the others. Returns (mean joint error, per-point max)."""
        errs = []
        for i in range(len(self.uv)):
            m = np.arange(len(self.uv)) != i
            y = self.grasp[m, :5].copy()
            y[:, 4] -= roll_sign * self.ang[m]
            f = self.Fit(self.uv[m], y)
            p = f(self.uv[i])[0]
            p[4] += roll_sign * self.ang[i]
            errs.append(np.abs(p - self.grasp[i, :5]))
        e = np.array(errs)
        return float(e.mean()), e.max(1)


def load(side):
    p = HERE / f"calib_{side}.json"
    return ArmMap(json.loads(p.read_text())) if p.exists() else None


if __name__ == "__main__":
    for side in ("left", "right"):
        m = load(side)
        if m is None:
            print(f"{side}: no calib_{side}.json yet - run CALIBRATE.bat {side}")
            continue
        mean, per = m.loo(m.roll_sign)
        print(f"{side}: {len(m.uv)} points, bins {sorted(m.bins)} | leave-one-out joint error mean {mean:.1f} deg, "
              f"worst point {per.max():.1f} deg | roll sign {m.roll_sign:+.0f}")
        if m.wrist_target is None and m.wrist_spread is not None:
            print(f"   wrist fine alignment: OFF (targets inconsistent, spread {m.wrist_spread:.2f} > 0.08 - needs a hand calibration)")
        elif m.wrist_target is None:
            print(f"   wrist fine alignment: OFF (fewer than 3 points with the object seen by the wrist camera)")
        else:
            print(f"   wrist fine alignment target {np.round(m.wrist_target, 3)} (picture fractions), calibration spread "
                  f"{m.wrist_spread:.3f} - under ~0.06 is consistent")
