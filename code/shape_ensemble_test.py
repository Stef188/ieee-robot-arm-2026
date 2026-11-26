"""Does a 3-network ensemble (+ 8-view test-time averaging) beat one ShapeNet? (26 Nov)
Photo-session crops only, 5-fold cross-validation grouped by photo (test photos never seen in training).
    python classical\\shape_ensemble_test.py
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import shape_model as SM  # noqa: E402
import train_shape_net as T  # noqa: E402
import torch  # noqa: E402


def tta_probs(net, X):
    out = []
    with torch.no_grad():
        for c in X:
            views = [np.rot90(c, k).copy() for k in range(4)]
            views += [v[:, ::-1].copy() for v in views]
            x = torch.from_numpy(np.stack([SM.preprocess(v) for v in views]))
            out.append(torch.softmax(net(x), 1).numpy().mean(0))
    return np.array(out)


def main():
    ph = np.load(HERE / "shape_data" / "photos.npz", allow_pickle=True)
    X, y, g = ph["img"], np.array([SM.CLASSES.index(s) for s in ph["shape"]]), ph["photo"]
    ug = np.unique(g)
    np.random.RandomState(0).shuffle(ug)
    P1, P1t, P3t = np.zeros((len(X), 4)), np.zeros((len(X), 4)), np.zeros((len(X), 4))
    t0 = time.time()
    for i, fg in enumerate(np.array_split(ug, 5)):
        te = np.isin(g, fg)
        nets = []
        for seed in range(3):
            torch.manual_seed(seed)
            np.random.seed(seed)
            nets.append(T.train(X[~te], y[~te]))
        P1[te] = T.predict(nets[0], X[te])
        P1t[te] = tta_probs(nets[0], X[te])
        P3t[te] = np.mean([tta_probs(n, X[te]) for n in nets], 0)
        print(f"fold {i + 1}/5 ({time.time() - t0:.0f} s): 1 net {np.mean(P1[te].argmax(1) == y[te]):.3f} | "
              f"1 net + 8 views {np.mean(P1t[te].argmax(1) == y[te]):.3f} | "
              f"3 nets + 8 views {np.mean(P3t[te].argmax(1) == y[te]):.3f}", flush=True)
    for name, P in (("1 net", P1), ("1 net + 8 views", P1t), ("3 nets + 8 views", P3t)):
        acc = np.mean(P.argmax(1) == y)
        per = ", ".join(f"{c[:5]} {np.mean(P[y == k].argmax(1) == k):.2f}" for k, c in enumerate(SM.CLASSES))
        conf = P.max(1)
        cov = ", ".join(f">={t:.0%}: {np.mean(P[conf >= t].argmax(1) == y[conf >= t]):.2f} on {np.mean(conf >= t):.0%}"
                        for t in (0.5, 0.6, 0.7, 0.8))
        print(f"\n{name:17s}: {acc:.3f} | {per}\n   {cov}")


if __name__ == "__main__":
    main()
