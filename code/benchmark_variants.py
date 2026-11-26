"""10 variants of the grasp-pose network (GraspNet) - compression / quantisation / pruning / distillation (26 Nov).
    python classical\\benchmark_variants.py           -> leave-one-out benchmark, both arms, ~15-30 min on the CPU
Writes training_logs\\variants_<time>.txt/.json + variants_<time>.png, and each variant's weights (trained on ALL
points, grasp + hover) to model_variants\\<variant>_<side>_<grasp|hover>.npz. Variants 1-8 can be run on the arms with
SORT_NET=<name> (model.py); 9 (lookup table) and 10 (extra inputs) are benchmark-only.

Every leave-one-out fold trains the base 5-net ensemble ONCE; the compressed variants are derived from that same
model (fair comparison). Metric: mean absolute joint error (deg) on the left-out demo grab, position joints
(pan, lift, elbow, wrist flex) = what the network drives on the arms, and all 5 joints.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import model  # noqa: E402

JOINTS = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll"]
STEPS, LR, WD = 3000, 2e-2, 2e-3


# ---------- a small numpy MLP: in -> H tanh -> H tanh -> out, + linear skip; optional fixed masks (pruning) ----------
def xin(uv, extra=None):
    uv = np.atleast_2d(np.asarray(uv, float))
    x = np.stack([(uv[:, 0] - 350) / 100, (uv[:, 1] - 330) / 100], 1)
    return x if extra is None else np.hstack([x, extra])


def fwd(p, x):
    h1 = np.tanh(x @ p[0] + p[1])
    h2 = np.tanh(h1 @ p[2] + p[3])
    return h1, h2, h2 @ p[4] + p[5] + x @ p[6]


def train(x, t, H=16, seed=0, steps=STEPS, init=None, masks=None, lr=LR):
    r = np.random.default_rng(seed)
    d, o, n = x.shape[1], t.shape[1], len(x)
    p = init if init is not None else [r.normal(0, 1 / np.sqrt(d), (d, H)), np.zeros(H), r.normal(0, 1 / np.sqrt(H), (H, H)),
                                       np.zeros(H), r.normal(0, 0.1 / np.sqrt(H), (H, o)), np.zeros(o), np.zeros((d, o))]
    p = [a.copy() for a in p]
    m1 = [np.zeros_like(a) for a in p]
    m2 = [np.zeros_like(a) for a in p]
    for k in range(1, steps + 1):
        h1, h2, out = fwd(p, x)
        g = 2 * (out - t) / n / o
        d2 = (g @ p[4].T) * (1 - h2 ** 2)
        d1 = (d2 @ p[2].T) * (1 - h1 ** 2)
        gs = [x.T @ d1, d1.sum(0), h1.T @ d2, d2.sum(0), h2.T @ g, g.sum(0), x.T @ g]
        for i in (0, 2, 4):
            gs[i] = gs[i] + WD * p[i]
        for i in range(7):
            m1[i] = 0.9 * m1[i] + 0.1 * gs[i]
            m2[i] = 0.999 * m2[i] + 0.001 * gs[i] ** 2
            p[i] -= lr * (m1[i] / (1 - 0.9 ** k)) / (np.sqrt(m2[i] / (1 - 0.999 ** k)) + 1e-8)
            if masks is not None and masks[i] is not None:
                p[i] *= masks[i]
    return p


def quant(p, bits=None, fp16=False):
    """Weights rounded to a symmetric per-tensor integer grid (INT8 / INT4) or to float16 - what would be stored."""
    out = []
    for a in p:
        if fp16:
            out.append(a.astype(np.float16).astype(np.float64))
        elif bits:
            s = np.abs(a).max() / (2 ** (bits - 1) - 1) + 1e-12
            out.append(np.round(a / s) * s)
        else:
            out.append(a)
    return out


def prune(p, frac, x, t):
    """Remove the smallest `frac` of the weight-matrix entries (global magnitude), then fine-tune with them held at 0."""
    w = np.concatenate([np.abs(p[i]).ravel() for i in (0, 2, 4, 6)])
    thr = np.quantile(w, frac)
    masks = [((np.abs(a) > thr).astype(float) if i in (0, 2, 4, 6) else None) for i, a in enumerate(p)]
    init = [a * m if m is not None else a for a, m in zip(p, masks)]
    return train(x, t, init=init, masks=masks, steps=800, lr=5e-3)


def nparams(p, nonzero=False):
    return int(sum((np.count_nonzero(a) if nonzero else a.size) for a in p))


class Variant:
    def __init__(self, name, nets, ym, ys, bits=64, extra=False, lut=None, note=""):
        self.name, self.nets, self.ym, self.ys, self.bits, self.extra, self.lut, self.note = name, nets, ym, ys, bits, extra, lut, note

    def __call__(self, uv, ex=None):
        if self.lut is not None:
            return self.lut(uv)
        x = xin(uv, ex if self.extra else None)
        return np.mean([fwd(p, x)[2] for p in self.nets], 0) * self.ys + self.ym

    def size_bytes(self):
        if self.lut is not None:
            return self.lut.grid.size * 2      # int16 grid
        nz = sum(nparams(p, nonzero=True) for p in self.nets)
        return int(nz * self.bits / 8)


class LUT:
    """Variant 9: the network pre-computed on a 2-px grid of the table, bilinear lookup at run time."""

    def __init__(self, f, step=2):
        self.step = step
        self.us, self.vs = np.arange(150, 571, step), np.arange(180, 431, step)
        U, V = np.meshgrid(self.us, self.vs)
        vals = f(np.stack([U.ravel(), V.ravel()], 1)).reshape(len(self.vs), len(self.us), -1)
        self.lo, self.hi = vals.min((0, 1)), vals.max((0, 1))
        self.grid = np.round((vals - self.lo) / (self.hi - self.lo + 1e-9) * 65535 - 32768).astype(np.int16)

    def __call__(self, uv):
        uv = np.atleast_2d(np.asarray(uv, float))
        fx = np.clip((uv[:, 0] - self.us[0]) / self.step, 0, len(self.us) - 1.001)
        fy = np.clip((uv[:, 1] - self.vs[0]) / self.step, 0, len(self.vs) - 1.001)
        i, j = fy.astype(int), fx.astype(int)
        a, b = (fy - i)[:, None], (fx - j)[:, None]
        g = (self.grid.astype(np.float64) + 32768) / 65535 * (self.hi - self.lo + 1e-9) + self.lo
        return (g[i, j] * (1 - a) * (1 - b) + g[i + 1, j] * a * (1 - b) + g[i, j + 1] * (1 - a) * b + g[i + 1, j + 1] * a * b)


def extra_inputs(ang, colours, palette):
    a = np.radians(np.asarray(ang, float) * 4)              # 90-deg symmetric blocks: angle x4 on the circle
    onehot = np.array([[c == q for q in palette] for c in colours], float)
    return np.hstack([np.sin(a)[:, None], np.cos(a)[:, None], onehot])


def build_all(uv, y, ang, cols, palette, hull_pts):
    """All 10 variants trained on (uv, y)."""
    ym, ys = y.mean(0), y.std(0) + 1e-6
    x, t = xin(uv), (y - ym) / ys
    base = [train(x, t, seed=s) for s in range(5)]
    V = {}
    V["1_ensemble_fp64"] = Variant("1_ensemble_fp64", base, ym, ys, 64, note="current: 5 nets, full precision")
    V["2_single_net"] = Variant("2_single_net", base[:1], ym, ys, 64, note="1 net instead of 5")
    # 3: distillation - a 2->8->8 net trained to copy the ensemble on a dense grid inside the demo area
    U, Vv = np.meshgrid(np.arange(150, 571, 6), np.arange(180, 431, 6))
    g = np.stack([U.ravel(), Vv.ravel()], 1)
    g = g[[cv_inside(hull_pts, q) for q in g]]
    teacher = (V["1_ensemble_fp64"](g) - ym) / ys
    V["3_distilled_tiny"] = Variant("3_distilled_tiny", [train(xin(g), teacher, H=8, seed=0, steps=2000)], ym, ys, 32,
                                    note="2-8-8 net trained to copy #1")
    V["4_fp16"] = Variant("4_fp16", [quant(p, fp16=True) for p in base], ym, ys, 16, note="half precision")
    V["5_int8"] = Variant("5_int8", [quant(p, bits=8) for p in base], ym, ys, 8, note="8-bit integer weights")
    V["6_int4"] = Variant("6_int4", [quant(p, bits=4) for p in base], ym, ys, 4, note="4-bit integer weights")
    V["7_pruned50"] = Variant("7_pruned50", [prune(p, 0.5, x, t) for p in base], ym, ys, 32, note="50% weights removed")
    V["8_pruned90"] = Variant("8_pruned90", [prune(p, 0.9, x, t) for p in base], ym, ys, 32, note="90% weights removed")
    V["9_lookup_table"] = Variant("9_lookup_table", None, ym, ys, lut=LUT(V["1_ensemble_fp64"]),
                                  note="pre-computed 2-px grid, int16")
    ex = extra_inputs(ang, cols, palette)
    xe = xin(uv, ex)
    V["10_extra_inputs"] = Variant("10_extra_inputs", [train(xe, t, H=24, seed=s) for s in range(5)], ym, ys, 64,
                                   extra=True, note="+ rotation + colour as inputs, 24 wide")
    return V


def cv_inside(hull_pts, q):
    import cv2
    return cv2.pointPolygonTest(hull_pts, (float(q[0]), float(q[1])), True) >= -10


def main():
    import cv2
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    (HERE / "training_logs").mkdir(exist_ok=True)
    (HERE / "model_variants").mkdir(exist_ok=True)
    lines, report = [], {"date": stamp, "arms": {}}

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say(f"GraspNet variants benchmark {stamp} - leave-one-out on the demo grabs (deg, lower = better)")
    for side in ("left", "right"):
        m = model.load(side)
        uv, ang = m.uv, m.ang
        cols = [p["colour"] for p in json.loads((HERE / f"calib_{side}.json").read_text())["points"]
                for _ in range(int(p.get("w") or 1))]
        palette = sorted(set(cols))
        y = m.grasp[:, :5].copy()
        y[:, 4] -= m.roll_sign * ang
        hull = m.hull
        n = len(uv)
        errs, t0 = {}, time.time()
        for i in range(n):
            keep = np.arange(n) != i
            V = build_all(uv[keep], y[keep], ang[keep], [c for c, k in zip(cols, keep) if k], palette, hull)
            exi = extra_inputs(ang[i:i + 1], [cols[i]], palette)
            for name, v in V.items():
                pred = v(uv[i:i + 1], exi)[0]
                errs.setdefault(name, []).append(np.abs(pred - y[i]))
            if i % 10 == 0:
                say(f"   {side}: fold {i + 1}/{n} ({time.time() - t0:.0f} s)")
        # final models on ALL points (grasp + hover) -> saved weights, size, speed, agreement with #1
        V = build_all(uv, y, ang, cols, palette, hull)
        yh = m.hover[:, :5].copy()
        yh[:, 4] -= m.roll_sign * ang
        Vh = build_all(uv, yh, ang, cols, palette, hull)
        U, Vv = np.meshgrid(np.arange(150, 571, 8), np.arange(180, 431, 8))
        grid = np.stack([U.ravel(), Vv.ravel()], 1)
        grid = grid[[m.reach(*q) >= 0 for q in grid]]
        ref = V["1_ensemble_fp64"](grid)
        arm = {}
        say(f"\n{side.upper()} arm - {n} demo grabs")
        say(f"   {'variant':18s} {'LOO pos.joints':>14s} {'LOO all 5':>10s} {'size':>9s} {'time/grab':>10s} {'vs #1 (max)':>12s}  note")
        for name, v in V.items():
            e = np.array(errs[name])
            exg = extra_inputs(np.zeros(len(grid)), [palette[0]] * len(grid), palette)
            t1 = time.perf_counter()
            for _ in range(200):
                v(grid[:1], exg[:1])
            ms = (time.perf_counter() - t1) / 200 * 1000
            dev = float(np.abs(v(grid, exg)[:, :4] - ref[:, :4]).max())
            arm[name] = {"loo_pos": float(e[:, :4].mean()), "loo_all": float(e.mean()),
                         "loo_per_joint": e.mean(0).round(3).tolist(), "bytes": v.size_bytes(), "ms": ms,
                         "max_dev_vs_1": dev, "note": v.note}
            say(f"   {name:18s} {e[:, :4].mean():14.2f} {e.mean():10.2f} {v.size_bytes():8d}B {ms:9.3f}ms {dev:11.2f}  {v.note}")
            if v.lut is None and not v.extra:
                for which, vv in (("grasp", v), ("hover", Vh[name])):
                    d = {"ym": vv.ym, "ys": vv.ys, "n_nets": len(vv.nets), "bits": vv.bits, "note": vv.note}
                    for k, p in enumerate(vv.nets):
                        for j, a in enumerate(p):
                            d[f"net{k}_p{j}"] = a
                    np.savez(HERE / "model_variants" / f"{name}_{side}_{which}.npz", **d)
        report["arms"][side] = arm
    (HERE / "training_logs" / f"variants_{stamp}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HERE / "training_logs" / f"variants_{stamp}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = list(report["arms"]["left"])
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for k, side in enumerate(("left", "right")):
            a = report["arms"][side]
            ax[k].barh([s.split("_", 1)[1] for s in names], [a[s]["loo_pos"] for s in names], color="#4a7bd0")
            ax[k].invert_yaxis()
            ax[k].set_title(f"{side} arm: leave-one-out error, position joints (deg)")
            ax[k].axvline(a[names[0]]["loo_pos"], color="k", lw=0.8, ls="--")
        fig.tight_layout()
        fig.savefig(HERE / "training_logs" / f"variants_{stamp}.png", dpi=110)
    except Exception as e:
        say(f"(chart skipped: {e})")
    say(f"\nlog: training_logs\\variants_{stamp}.txt | weights: model_variants\\")


if __name__ == "__main__":
    main()
