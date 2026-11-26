"""WristShapeNet: shape from the WRIST camera close-up at the moment of the grab (26 Nov).
    python classical\\train_shape_wrist.py
Data: classical\\shape_data\\crops.npz from shape_dataset.py - for every labelled demo grab, 4 wrist frames around the
moment the gripper starts closing (the block fills a large part of the picture, many times the top camera's pixels).
Input per frame: greyscale (contrast-normalised) + block mask (any block colour), 90x160, plus which arm (one-hot).
Colour-blind like ShapeNet. 5-fold cross-validation grouped by demo episode; a grab's prediction = average of its 4
frames. Weights -> classical\\shape_wrist_net.pt, log -> classical\\training_logs\\shape_wrist_<time>.txt
"""
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
import torch.nn as nn  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
EPOCHS, BATCH, LR = 40, 32, 2e-3
LOG = []
H, W = 90, 160
WEIGHTS = HERE / "shape_wrist_net.pt"


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def prep(img):
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_BGR2HSV)
    m = np.zeros(img.shape[:2], np.uint8)
    for name in vision.COLOURS:
        m |= vision.colour_mask(hsv, name)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6)
    return np.stack([g, (m > 0).astype(np.float32)])


def augment(img):
    a, s = np.random.uniform(-25, 25), np.random.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), a, s)
    M[:, 2] += np.random.uniform(-6, 6, 2)
    out = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    alpha, beta = np.random.uniform(0.7, 1.3), np.random.uniform(-30, 30)
    return np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(2, 16), blk(16, 32), blk(32, 48), blk(48, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64 + 2, len(SM.CLASSES)))

    def forward(self, x, arm):
        return self.head(torch.cat([self.f(x), nn.functional.one_hot(arm, 2).float()], 1))


def train(X, A, y):
    """X: (n, 4, H, W, 3) frames, A: arm, y: label - every frame is a training sample."""
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    w = np.bincount(y, minlength=4).astype(np.float32)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w.sum() / (4 * np.maximum(w, 1))))
    items = [(i, f) for i in range(len(X)) for f in range(X.shape[1])]
    for ep in range(EPOCHS):
        net.train()
        np.random.shuffle(items)
        for b in range(0, len(items), BATCH):
            bb = items[b:b + BATCH]
            xb = torch.from_numpy(np.stack([prep(augment(X[i, f])) for i, f in bb]))
            ab = torch.tensor([int(A[i]) for i, _ in bb])
            loss = lossf(net(xb, ab), torch.tensor([int(y[i]) for i, _ in bb]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    net.eval()
    return net


def predict(net, X, A):
    with torch.no_grad():
        out = []
        for i in range(len(X)):
            xb = torch.from_numpy(np.stack([prep(f) for f in X[i]]))
            p = torch.softmax(net(xb, torch.full((len(X[i]),), int(A[i]))), 1).mean(0)
            out.append(p.numpy())
        return np.array(out)


def main():
    d = np.load(HERE / "shape_data" / "crops.npz", allow_pickle=True)
    if "wrist" not in d.files:
        sys.exit("crops.npz has no wrist frames - run shape_dataset.py first")
    X, A, shape, colour = d["wrist"], d["arm"].astype(int), d["shape"], d["colour"]
    y = np.array([SM.CLASSES.index(s) for s in shape])
    groups = np.array([f"{a}:{b}" for a, b in zip(d["dataset"], d["episode"])])
    say(f"WristShapeNet training {time.strftime('%Y-%m-%d %H:%M')} | {len(X)} grabs x {X.shape[1]} wrist frames | "
        f"left {int((A == 0).sum())}, right {int((A == 1).sum())}")
    ug = np.unique(groups)
    np.random.RandomState(0).shuffle(ug)
    pred = np.full(len(X), -1)
    conf = np.zeros(len(X))
    for i, fg in enumerate(np.array_split(ug, 5)):
        te = np.isin(groups, fg)
        net = train(X[~te], A[~te], y[~te])
        p = predict(net, X[te], A[te])
        pred[te], conf[te] = p.argmax(1), p.max(1)
        say(f"   fold {i + 1}/5: {int(te.sum())} test grabs, accuracy {np.mean(pred[te] == y[te]):.3f}")
    m = pred >= 0
    say(f"\nCROSS-VALIDATED (unseen demo episodes): WristShapeNet {np.mean(pred[m] == y[m]):.3f} | "
        f"chance {np.bincount(y).max() / len(y):.3f}")
    for arm, nm in ((0, "left"), (1, "right")):
        k = m & (A == arm)
        say(f"   {nm} wrist camera: {np.mean(pred[k] == y[k]):.3f} (n={int(k.sum())})")
    say("per shape: " + ", ".join(f"{c} {np.mean(pred[m & (y == k)] == k):.2f}" for k, c in enumerate(SM.CLASSES)))
    say("per colour (no colour shortcut?): " + ", ".join(
        f"{c} {np.mean(pred[m & (colour == c)] == y[m & (colour == c)]):.2f}" for c in vision.COLOURS))
    for th in (0.5, 0.6, 0.7, 0.8):
        k = m & (conf >= th)
        say(f"   when at least {th:.0%} sure: {np.mean(pred[k] == y[k]):.3f} correct, covers {k.mean():.0%} of grabs")
    net = train(X, A, y)
    torch.save({"weights": net.state_dict(), "meta": {"classes": SM.CLASSES, "n_train": int(len(X)),
                                                      "cv_accuracy": float(np.mean(pred[m] == y[m])),
                                                      "trained": time.strftime("%Y-%m-%d %H:%M")}}, WEIGHTS)
    say(f"saved {WEIGHTS}")
    (HERE / "training_logs").mkdir(exist_ok=True)
    (HERE / "training_logs" / f"shape_wrist_{time.strftime('%Y%m%d_%H%M')}.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
