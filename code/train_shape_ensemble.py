"""Final ShapeNet = 3 networks (seeds 0-2) trained on ALL photo-session crops, averaged at run time with 8 views each.
Cross-validated (shape_ensemble_test.py, grouped by photo): 79.1 % overall, 91 % when >= 70 % sure (1 net: 76.1 / 87 %).
    python classical\shape_train_final.py  ->  shape_net.pt (old one kept as shape_net_single_<time>.pt)
"""
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import shape_model as SM  # noqa: E402
import train_shape_net as T  # noqa: E402
import torch  # noqa: E402

ph = np.load(HERE / "shape_data" / "photos.npz", allow_pickle=True)
X, y = ph["img"], np.array([SM.CLASSES.index(s) for s in ph["shape"]])
states = []
for seed in range(3):
    t = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    states.append(T.train(X, y).state_dict())
    print(f"net {seed + 1}/3 trained ({time.time() - t:.0f} s)", flush=True)
if SM.WEIGHTS.exists():
    shutil.copy(SM.WEIGHTS, HERE / f"shape_net_single_{time.strftime('%H%M')}.pt")
torch.save({"ensemble": states, "meta": {"classes": SM.CLASSES, "n_train": int(len(X)), "nets": 3,
            "cv_accuracy": 0.791, "trained": time.strftime("%Y-%m-%d %H:%M"), "params_per_net": 40084}}, SM.WEIGHTS)
print("saved", SM.WEIGHTS, "(3-network ensemble)")
