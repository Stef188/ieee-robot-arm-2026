"""Variants 11-20 of the grasp-pose model: different MODEL TYPES (26 Nov, overnight). Pure numpy.
    python classical\\benchmark_variants2.py      -> training_logs\\variants2_<time>.txt/.json/.png
Same leave-one-out test as benchmark_variants.py (variants 1-10). #1 (current GraspNet) is re-run as the reference.
11 wide net (64)        12 deep net (3 layers)     13 ReLU net            14 10-net ensemble     15 dropout net
16 k-nearest-neighbours 17 Gaussian process (+ "how sure" score)       18 random forest       19 boosted trees
20 outlier-robust net (Huber loss: demos far off the rest count less)
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
import benchmark_variants as B  # noqa: E402


def xin(uv):
    uv = np.atleast_2d(np.asarray(uv, float))
    return np.stack([(uv[:, 0] - 350) / 100, (uv[:, 1] - 330) / 100], 1)


# ---------- generic MLP: any depth/width, tanh or relu, dropout, MSE or Huber, linear skip ----------
def mlp_train(x, t, sizes, act="tanh", drop=0.0, huber=None, seed=0, steps=3000, lr=2e-2, wd=2e-3):
    r = np.random.default_rng(seed)
    dims = [x.shape[1]] + sizes + [t.shape[1]]
    W = [r.normal(0, (0.1 if i == len(dims) - 2 else 1.0) / np.sqrt(dims[i]), (dims[i], dims[i + 1])) for i in range(len(dims) - 1)]
    b = [np.zeros(d) for d in dims[1:]]
    S = np.zeros((x.shape[1], t.shape[1]))
    P = W + b + [S]
    m1 = [np.zeros_like(a) for a in P]
    m2 = [np.zeros_like(a) for a in P]
    f = (np.tanh, lambda z: 1 - np.tanh(z) ** 2) if act == "tanh" else (lambda z: np.maximum(z, 0), lambda z: (z > 0).astype(float))
    L = len(W)
    n, o = len(x), t.shape[1]
    for k in range(1, steps + 1):
        hs, zs, masks = [x], [], []
        for i in range(L):
            z = hs[-1] @ W[i] + b[i]
            zs.append(z)
            if i < L - 1:
                h = f[0](z)
                mk = (r.random(h.shape) > drop) / (1 - drop) if drop else None
                masks.append(mk)
                hs.append(h * mk if mk is not None else h)
        out = zs[-1] + x @ S
        res = out - t
        g = (np.clip(res, -huber, huber) if huber else res) * 2 / n / o
        gW, gb = [None] * L, [None] * L
        d = g
        for i in range(L - 1, -1, -1):
            gW[i] = hs[i].T @ d + wd * W[i]
            gb[i] = d.sum(0)
            if i:
                d = d @ W[i].T
                if masks[i - 1] is not None:
                    d = d * masks[i - 1]
                d = d * f[1](zs[i - 1])
        grads = gW + gb + [x.T @ g]
        for i in range(len(P)):
            m1[i] = 0.9 * m1[i] + 0.1 * grads[i]
            m2[i] = 0.999 * m2[i] + 0.001 * grads[i] ** 2
            P[i] -= lr * (m1[i] / (1 - 0.9 ** k)) / (np.sqrt(m2[i] / (1 - 0.999 ** k)) + 1e-8)
    W, b, S = P[:L], P[L:2 * L], P[-1]

    def pred(xx):
        h = xx
        for i in range(L):
            z = h @ W[i] + b[i]
            h = f[0](z) if i < L - 1 else z
        return h + xx @ S
    pred.n = int(sum(a.size for a in P))
    return pred


class Ens:
    def __init__(self, preds, ym, ys):
        self.p, self.ym, self.ys = preds, ym, ys
        self.n = sum(q.n for q in preds)

    def __call__(self, uv):
        x = xin(uv)
        return np.mean([q(x) for q in self.p], 0) * self.ys + self.ym


class KNN:
    def __init__(self, uv, y, k=5):
        self.x, self.y, self.k, self.n = xin(uv), y, k, y.size + uv.size

    def __call__(self, uv):
        x = xin(uv)
        d = np.linalg.norm(x[:, None] - self.x[None], axis=2)
        idx = np.argsort(d, 1)[:, :self.k]
        w = 1 / (np.take_along_axis(d, idx, 1) + 0.02)
        return (self.y[idx] * w[..., None]).sum(1) / w.sum(1, keepdims=True)


class GP:
    """RBF Gaussian process, length-scale + noise chosen by marginal likelihood. std() = how unsure it is."""

    def __init__(self, uv, y):
        self.x = xin(uv)
        self.ym, self.ys = y.mean(0), y.std(0) + 1e-6
        t = (y - self.ym) / self.ys
        best = None
        for ell in (0.2, 0.3, 0.45, 0.7, 1.0, 1.5):
            for sn in (0.15, 0.25, 0.4, 0.6, 0.8):
                K = self.k(self.x, self.x, ell) + sn ** 2 * np.eye(len(self.x))
                Lc = np.linalg.cholesky(K)
                a = np.linalg.solve(Lc.T, np.linalg.solve(Lc, t))
                ll = -0.5 * (t * a).sum() - t.shape[1] * np.log(np.diag(Lc)).sum()
                if best is None or ll > best[0]:
                    best = (ll, ell, sn, Lc, a)
        _, self.ell, self.sn, self.Lc, self.a = best
        self.n = self.x.size + self.a.size + 2

    @staticmethod
    def k(a, b, ell):
        return np.exp(-0.5 * ((a[:, None] - b[None]) ** 2).sum(2) / ell ** 2)

    def __call__(self, uv):
        return self.k(xin(uv), self.x, self.ell) @ self.a * self.ys + self.ym

    def std(self, uv):
        ks = self.k(xin(uv), self.x, self.ell)
        v = np.linalg.solve(self.Lc, ks.T)
        return np.sqrt(np.maximum(1 - (v ** 2).sum(0), 0))


# ---------- regression trees (multi-output CART), random forest, gradient boosting ----------
def build_tree(x, t, depth, min_leaf, rng=None):
    node = {"v": t.mean(0)}
    if depth == 0 or len(x) < 2 * min_leaf:
        return node
    best = None
    feats = [rng.integers(x.shape[1])] if rng is not None else range(x.shape[1])
    for j in feats:
        o = np.argsort(x[:, j])
        xs, ts = x[o, j], t[o]
        cs, cs2 = np.cumsum(ts, 0), np.cumsum(ts ** 2, 0)
        n = len(xs)
        for i in range(min_leaf, n - min_leaf + 1):
            if xs[i - 1] == xs[i % n]:
                continue
            l_sse = (cs2[i - 1] - cs[i - 1] ** 2 / i).sum()
            r_sse = ((cs2[-1] - cs2[i - 1]) - (cs[-1] - cs[i - 1]) ** 2 / (n - i)).sum()
            if best is None or l_sse + r_sse < best[0]:
                best = (l_sse + r_sse, j, (xs[i - 1] + xs[i]) / 2)
    if best is None:
        return node
    _, j, thr = best
    m = x[:, j] <= thr
    node.update(j=j, thr=thr, l=build_tree(x[m], t[m], depth - 1, min_leaf, rng), r=build_tree(x[~m], t[~m], depth - 1, min_leaf, rng))
    return node


def tree_pred(node, x):
    out = np.zeros((len(x), len(node["v"])))
    for i, q in enumerate(x):
        nd = node
        while "j" in nd:
            nd = nd["l"] if q[nd["j"]] <= nd["thr"] else nd["r"]
        out[i] = nd["v"]
    return out


def count_nodes(nd):
    return 1 + (count_nodes(nd["l"]) + count_nodes(nd["r"]) if "j" in nd else 0)


class Forest:
    def __init__(self, uv, y, n_trees=60, depth=6, seed=0):
        x, rng = xin(uv), np.random.default_rng(seed)
        self.trees = []
        for _ in range(n_trees):
            i = rng.integers(len(x), size=len(x))
            self.trees.append(build_tree(x[i], y[i], depth, 3, rng))
        self.n = sum(count_nodes(tr) for tr in self.trees) * (y.shape[1] + 2)

    def __call__(self, uv):
        x = xin(uv)
        return np.mean([tree_pred(tr, x) for tr in self.trees], 0)


class Boost:
    def __init__(self, uv, y, rounds=120, depth=3, lr=0.1):
        x = xin(uv)
        self.base, self.lr, self.trees = y.mean(0), lr, []
        f = np.tile(self.base, (len(x), 1))
        for _ in range(rounds):
            tr = build_tree(x, y - f, depth, 4)
            self.trees.append(tr)
            f = f + lr * tree_pred(tr, x)
        self.n = sum(count_nodes(tr) for tr in self.trees) * (y.shape[1] + 2)

    def __call__(self, uv):
        x = xin(uv)
        return self.base + self.lr * sum(tree_pred(tr, x) for tr in self.trees)


def build_all(uv, y):
    ym, ys = y.mean(0), y.std(0) + 1e-6
    x, t = xin(uv), (y - ym) / ys
    V = {}
    V["1_current_ensemble"] = Ens([mlp_train(x, t, [16, 16], seed=s) for s in range(5)], ym, ys)
    V["11_wide64"] = Ens([mlp_train(x, t, [64, 64], seed=s) for s in range(3)], ym, ys)
    V["12_deep3"] = Ens([mlp_train(x, t, [16, 16, 16], seed=s) for s in range(3)], ym, ys)
    V["13_relu"] = Ens([mlp_train(x, t, [16, 16], act="relu", seed=s) for s in range(5)], ym, ys)
    V["14_ensemble10"] = Ens([mlp_train(x, t, [16, 16], seed=s) for s in range(10)], ym, ys)
    V["15_dropout"] = Ens([mlp_train(x, t, [32, 32], drop=0.15, seed=s) for s in range(5)], ym, ys)
    V["16_knn5"] = KNN(uv, y)
    V["17_gauss_process"] = GP(uv, y)
    V["18_random_forest"] = Forest(uv, y)
    V["19_boosted_trees"] = Boost(uv, y)
    V["20_robust_huber"] = Ens([mlp_train(x, t, [16, 16], huber=0.5, seed=s) for s in range(5)], ym, ys)
    return V


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    lines, report = [], {"date": stamp, "arms": {}}

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say(f"Grasp-pose model TYPES benchmark (variants 11-20) {stamp} - leave-one-out on the demo grabs (deg, lower = better)")
    for side in ("left", "right"):
        m = model.load(side)
        uv = m.uv
        y = m.grasp[:, :5].copy()
        y[:, 4] -= m.roll_sign * m.ang
        n = len(uv)
        errs, gp_std, t0 = {}, [], time.time()
        for i in range(n):
            keep = np.arange(n) != i
            V = build_all(uv[keep], y[keep])
            for name, v in V.items():
                errs.setdefault(name, []).append(np.abs(v(uv[i:i + 1])[0] - y[i]))
            gp_std.append(float(V["17_gauss_process"].std(uv[i:i + 1])[0]))
            if i % 10 == 0:
                say(f"   {side}: fold {i + 1}/{n} ({time.time() - t0:.0f} s)")
        V = build_all(uv, y)
        U, Vv = np.meshgrid(np.arange(150, 571, 8), np.arange(180, 431, 8))
        grid = np.stack([U.ravel(), Vv.ravel()], 1)
        grid = grid[[m.reach(*q) >= 0 for q in grid]]
        ref = V["1_current_ensemble"](grid)
        arm = {}
        say(f"\n{side.upper()} arm - {n} demo grabs")
        say(f"   {'variant':20s} {'LOO pos.joints':>14s} {'LOO all 5':>10s} {'params':>8s} {'time/grab':>10s} {'vs #1 (max)':>12s}")
        for name, v in V.items():
            e = np.array(errs[name])
            t1 = time.perf_counter()
            for _ in range(50):
                v(grid[:1])
            ms = (time.perf_counter() - t1) / 50 * 1000
            dev = float(np.abs(v(grid)[:, :4] - ref[:, :4]).max())
            arm[name] = {"loo_pos": float(e[:, :4].mean()), "loo_all": float(e.mean()),
                         "loo_per_joint": e.mean(0).round(3).tolist(), "params": int(v.n), "ms": ms, "max_dev_vs_1": dev}
            say(f"   {name:20s} {e[:, :4].mean():14.2f} {e.mean():10.2f} {int(v.n):8d} {ms:9.3f}ms {dev:11.2f}")
        # does the Gaussian process know when it is wrong? (correlation of its 'unsure' score with the real error)
        e17 = np.array(errs["17_gauss_process"])[:, :4].mean(1)
        c = float(np.corrcoef(gp_std, e17)[0, 1])
        hi = np.array(gp_std) > np.median(gp_std)
        say(f"   Gaussian process 'how sure' score vs real error: correlation {c:+.2f}; error where it is UNSURE "
            f"{e17[hi].mean():.2f} deg vs SURE {e17[~hi].mean():.2f} deg")
        arm["gp_uncertainty"] = {"corr": c, "err_unsure": float(e17[hi].mean()), "err_sure": float(e17[~hi].mean())}
        report["arms"][side] = arm
    (HERE / "training_logs" / f"variants2_{stamp}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HERE / "training_logs" / f"variants2_{stamp}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [k for k in report["arms"]["left"] if k != "gp_uncertainty"]
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
        for k, side in enumerate(("left", "right")):
            a = report["arms"][side]
            ax[k].barh([s.split("_", 1)[1] for s in names], [a[s]["loo_pos"] for s in names], color="#3f9f6f")
            ax[k].invert_yaxis()
            ax[k].axvline(a[names[0]]["loo_pos"], color="k", lw=0.8, ls="--")
            ax[k].set_title(f"{side} arm: leave-one-out error, position joints (deg)")
        fig.tight_layout()
        fig.savefig(HERE / "training_logs" / f"variants2_{stamp}.png", dpi=110)
    except Exception as e:
        say(f"(chart skipped: {e})")
    say(f"\nlog: training_logs\\variants2_{stamp}.txt")


if __name__ == "__main__":
    main()
