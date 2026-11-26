"""Train + evaluate ShapeNet (shape_model.py) on classical\\shape_data\\crops.npz (made by shape_dataset.py). 26 Nov.
    python classical\\train_shape_net.py
1. drops crops where the block is hidden (the centre is not block-coloured, e.g. the arm in the way);
2. 5-fold cross-validation grouped by DEMO EPISODE (test blocks come from demos the network never saw), with
   augmentation (any rotation, flips, +-10 % scale, +-3 px shift, brightness / contrast);
   reports accuracy per shape AND per colour (it must not have learned colour) and a confusion matrix;
3. compares with a hand-written baseline (roundness, corners, solidity, aspect, size -> nearest class mean);
4. trains the final network on everything -> classical\\shape_net.pt, log -> classical\\training_logs\\shape_net_<time>.txt
"""
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import shape_model as SM  # noqa: E402
import vision  # noqa: E402

import torch  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
EPOCHS, BATCH, LR = 60, 32, 2e-3
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def visible(crop, colour):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    c = vision.colour_mask(hsv, colour)[SM.CROP // 2 - 6:SM.CROP // 2 + 6, SM.CROP // 2 - 6:SM.CROP // 2 + 6]
    return (c > 0).mean() >= 0.3


ISO = int(__import__("os").environ.get("SHAPE_ISO") or 20)


def isolated(crop):
    """No other block blob centre within ISO px of the crop centre."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    m = np.zeros(crop.shape[:2], np.uint8)
    for name in vision.COLOURS:
        m |= vision.colour_mask(hsv, name)
    n, lab, st, cen = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8))
    c = SM.CROP / 2
    own = lab[int(c), int(c)]
    return not any(i != own and st[i, cv2.CC_STAT_AREA] >= 40 and np.hypot(*(cen[i] - c)) < ISO for i in range(1, n))


def augment(crop):
    a = np.random.uniform(0, 360)
    s = np.random.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((SM.CROP / 2, SM.CROP / 2), a, s)
    M[:, 2] += np.random.uniform(-3, 3, 2)
    out = cv2.warpAffine(crop, M, (SM.CROP, SM.CROP), borderMode=cv2.BORDER_REPLICATE)
    if np.random.rand() < 0.5:
        out = out[:, ::-1]
    alpha, beta = np.random.uniform(0.75, 1.25), np.random.uniform(-25, 25)
    return np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def train(X, y, epochs=EPOCHS):
    net = SM.make_net()
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    w = np.bincount(y, minlength=len(SM.CLASSES)).astype(np.float32)
    w = torch.tensor(w.sum() / (len(w) * np.maximum(w, 1)))      # balance the classes
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    for ep in range(epochs):
        net.train()
        idx = np.random.permutation(len(X))
        for i in range(0, len(idx), BATCH):
            b = idx[i:i + BATCH]
            xb = torch.from_numpy(np.stack([SM.preprocess(augment(X[j])) for j in b]))
            loss = lossf(net(xb), torch.from_numpy(y[b]).long())
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    net.eval()
    return net


def predict(net, X):
    with torch.no_grad():
        return torch.softmax(net(torch.from_numpy(np.stack([SM.preprocess(c) for c in X]))), 1).numpy()


OUT_W = Path(os.environ["SHAPE_OUT"]) if os.environ.get("SHAPE_OUT") else SM.WEIGHTS


def main():
    d = np.load(HERE / "shape_data" / "crops.npz", allow_pickle=True)
    X, shape, colour = d["img"], d["shape"], d["colour"]
    groups = np.array([f"{a}:{b}" for a, b in zip(d["dataset"], d["episode"])])
    ok = np.array([visible(c, col) for c, col in zip(X, colour)])
    # 26 Nov: only ISOLATED blocks (no other block centre within ISO px) - crowded crops had unreliable labels
    if ISO:
        alone = np.array([isolated(c) for c in X])
        say(f"isolated blocks (nothing else within {ISO} px): {int((ok & alone).sum())} of {int(ok.sum())}")
        ok &= alone
    say(f"ShapeNet training {time.strftime('%Y-%m-%d %H:%M')} | crops {len(X)}, block visible in {ok.sum()} "
        f"(the rest dropped: arm in the way)")
    X, shape, colour, groups = X[ok], shape[ok], colour[ok], groups[ok]
    if os.environ.get("SHAPE_DEMOS", "1") == "0":          # photo-session data only
        X, shape, colour, groups = X[:0], shape[:0], colour[:0], groups[:0]
        say("demo crops NOT used (SHAPE_DEMOS=0)")
    src = np.array(["demo"] * len(X))
    P = HERE / "shape_data" / "photos.npz"           # photo session (shape_photos.py): CLEAN labels
    if P.exists():
        ph = np.load(P, allow_pickle=True)
        X = np.concatenate([X, ph["img"]])
        shape = np.concatenate([shape, ph["shape"]])
        colour = np.concatenate([colour, ph["colour"]])
        groups = np.concatenate([groups, np.array([f"photo:{t}" for t in ph["photo"]])])
        src = np.concatenate([src, np.array(["photo"] * len(ph["img"]))])
        say(f"+ photo session: {len(ph['img'])} crops with clean labels")
    y = np.array([SM.CLASSES.index(s) for s in shape])
    for k, c in enumerate(SM.CLASSES):
        cols = {cc: int(((y == k) & (colour == cc)).sum()) for cc in vision.COLOURS}
        say(f"   {c:17s} {int((y == k).sum()):4d}  by colour {cols}")
    # ---- grouped 5-fold cross-validation
    ug = np.unique(groups)
    rng = np.random.RandomState(0)
    rng.shuffle(ug)
    folds = np.array_split(ug, 5)
    pred = np.full(len(X), -1)
    conf = np.zeros(len(X))
    hand_pred = np.full(len(X), -1)
    F = np.stack([SM.hand_features(c) for c in X])
    for i, fg in enumerate(folds):
        te = np.isin(groups, fg)
        tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        net = train(X[tr], y[tr])
        pp = predict(net, X[te])
        pred[te], conf[te] = pp.argmax(1), pp.max(1)
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
        cent = np.stack([((F[tr][y[tr] == k] - mu) / sd).mean(0) if (y[tr] == k).any() else np.full(F.shape[1], 1e9)
                         for k in range(len(SM.CLASSES))])
        hand_pred[te] = np.argmin(((((F[te] - mu) / sd)[:, None, :] - cent[None]) ** 2).sum(-1), 1)
        say(f"   fold {i + 1}/5: {te.sum()} test crops, accuracy {np.mean(pred[te] == y[te]):.3f} "
            f"(hand-written baseline {np.mean(hand_pred[te] == y[te]):.3f})")
    m = pred >= 0
    say(f"\nCROSS-VALIDATED (unseen demo episodes): ShapeNet {np.mean(pred[m] == y[m]):.3f} | "
        f"hand-written baseline {np.mean(hand_pred[m] == y[m]):.3f} | chance {np.bincount(y).max() / len(y):.3f}")
    if (src == "photo").any():
        k = m & (src == "photo")
        say(f"ON THE PHOTO-SESSION CROPS (clean labels, unseen photos): ShapeNet {np.mean(pred[k] == y[k]):.3f} | "
            f"hand-written {np.mean(hand_pred[k] == y[k]):.3f}")
    say("per shape (ShapeNet): " + ", ".join(f"{c} {np.mean(pred[m & (y == k)] == k):.2f} (n={int((m & (y == k)).sum())})"
                                              for k, c in enumerate(SM.CLASSES) if (m & (y == k)).any()))
    say("per colour (must be similar - no colour shortcut): " +
        ", ".join(f"{c} {np.mean(pred[m & (colour == c)] == y[m & (colour == c)]):.2f} (n={int((m & (colour == c)).sum())})"
                  for c in vision.COLOURS if (m & (colour == c)).any()))
    say("CONFIDENT-ONLY (place a block only when the model is at least this sure; wrong bin = minus points):")
    for th in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        k = m & (conf >= th)
        if k.any():
            say(f"   >= {th:.0%} sure: {np.mean(pred[k] == y[k]):.3f} correct on {int(k.sum())} blocks "
                f"({k.mean():.0%} of all); expected points per placed block ~ {2 * np.mean(pred[k] == y[k]) - 1:+.2f} x its value")
    cm = np.zeros((4, 4), int)
    for a, b in zip(y[m], pred[m]):
        cm[a, b] += 1
    say("confusion (rows = true, cols = predicted, order " + " / ".join(SM.CLASSES) + "):")
    for k in range(4):
        say("   " + " ".join(f"{v:4d}" for v in cm[k]))
    # ---- final model on everything
    net = train(X, y)
    torch.save({"weights": net.state_dict(),
                "meta": {"classes": SM.CLASSES, "n_train": int(len(X)), "trained": time.strftime("%Y-%m-%d %H:%M"),
                         "cv_accuracy": float(np.mean(pred[m] == y[m])),
                         "params": int(sum(p.numel() for p in net.parameters()))}}, OUT_W)
    say(f"\nsaved {OUT_W} ({sum(p.numel() for p in net.parameters())} parameters)")
    out = HERE / "training_logs"
    out.mkdir(exist_ok=True)
    (out / f"shape_net_{os.environ.get('SHAPE_TAG', '')}{time.strftime('%Y%m%d_%H%M')}.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
