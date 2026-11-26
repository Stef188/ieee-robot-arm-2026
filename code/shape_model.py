"""ShapeNet: the LEARNED shape classifier for Round 2 (shape sort) - 26 Nov.
Input: a 64x64 top-camera crop around a block. Output: cube / cylinder / dodecahedron / pentagonal_prism.
Colour-blind by design: the network sees (1) the crop in contrast-normalised greyscale and (2) the block's mask
(pixels of ANY of the four block colours), never the colour itself - Round 2 sorts by shape regardless of colour.
Trained by train_shape_net.py on crops cut from our own shape demos (shape_dataset.py), labelled by the bin the human
dropped each block into. Weights: classical\\shape_net.pt.
"""
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
CLASSES = ["cube", "cylinder", "dodecahedron", "pentagonal_prism"]
SHAPE_TO_BIN = {"cube": "red", "cylinder": "blue", "dodecahedron": "green", "pentagonal_prism": "yellow"}  # 5.8.1.46
CROP = 64
WEIGHTS = HERE / "shape_net.pt"


ZOOM = 40          # 26 Nov: the network looks at the central 40x40 px, scaled up to 64x64 (blocks are only 11-30 px)


def preprocess(crop_bgr):
    """64x64 BGR crop -> (2, 64, 64) float32: normalised grey + block mask (any block colour), centre zoomed in."""
    import vision
    if ZOOM and ZOOM < CROP:
        o = (CROP - ZOOM) // 2
        crop_bgr = cv2.resize(crop_bgr[o:o + ZOOM, o:o + ZOOM], (CROP, CROP), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(cv2.GaussianBlur(crop_bgr, (3, 3), 0), cv2.COLOR_BGR2HSV)
    mask = np.zeros(crop_bgr.shape[:2], np.uint8)
    for name in vision.COLOURS:
        mask |= vision.colour_mask(hsv, name)
    # keep only the blob that touches the centre (neighbours / bin rims are not this block)
    n, lab = cv2.connectedComponents((mask > 0).astype(np.uint8))
    c = lab[CROP // 2 - 4:CROP // 2 + 4, CROP // 2 - 4:CROP // 2 + 4]
    ids = [i for i in np.unique(c) if i > 0]
    keep = np.isin(lab, ids).astype(np.float32) if ids else (mask > 0).astype(np.float32)
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6)
    return np.stack([g, keep]).astype(np.float32)


def make_net():
    import torch.nn as nn

    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
    return nn.Sequential(blk(2, 16), blk(16, 32), blk(32, 48), blk(48, 48),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(48, len(CLASSES)))


class ShapeNet:
    """Loads shape_net.pt; probs(crops) -> (n, 4) class probabilities."""

    def __init__(self, path=WEIGHTS):
        import torch
        self.torch = torch
        state = torch.load(path, map_location="cpu")
        weights = state["ensemble"] if "ensemble" in state else [state["weights"]]   # 26 Nov: 3-network ensemble
        self.nets = []
        for w in weights:
            n = make_net()
            n.load_state_dict(w)
            n.eval()
            self.nets.append(n)
        self.net = self.nets[0]
        self.meta = state.get("meta", {})

    def probs(self, crops_bgr, tta=True):
        """26 Nov 19:45: test-time augmentation - every crop is classified 8 times (4 turns x mirror) and averaged."""
        out = []
        for c in crops_bgr:
            views = [np.rot90(c, k).copy() for k in range(4)]
            views += [v[:, ::-1].copy() for v in views] if tta else []
            if not tta:
                views = [c]
            x = self.torch.from_numpy(np.stack([preprocess(v) for v in views]))
            with self.torch.no_grad():
                out.append(np.mean([self.torch.softmax(n(x), 1).numpy().mean(0) for n in self.nets], 0))
        return np.array(out)


def crop_at(img, u, v, n=CROP):
    pad = cv2.copyMakeBorder(img, n, n, n, n, cv2.BORDER_REPLICATE)
    x, y = int(round(u)) + n - n // 2, int(round(v)) + n - n // 2
    return pad[y:y + n, x:x + n].copy()


def hand_features(crop_bgr):
    """Hand-written baseline features (for the comparison in the docs): roundness, corners, solidity, aspect, area."""
    m = (preprocess(crop_bgr)[1] > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros(6, np.float32)
    c = max(cnts, key=cv2.contourArea)
    a, p = cv2.contourArea(c), cv2.arcLength(c, True) + 1e-6
    hull = cv2.convexHull(c)
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    corners = len(cv2.approxPolyDP(c, 0.04 * p, True))
    return np.array([4 * np.pi * a / p ** 2, corners, a / (cv2.contourArea(hull) + 1e-6),
                     max(w, h) / (min(w, h) + 1e-6), a, p], np.float32)
