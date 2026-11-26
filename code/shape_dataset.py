"""Build the SHAPE-classifier training set from the shape demos (26 Nov). No robot needed, reads the datasets only.
    python classical\\shape_dataset.py [dataset ...]        (default: ieee_shape_clean ieee_shape_v2)
For every demo grab (gripper closes after being open):
  * the GRABBED block = a block that disappears from the table (top camera, as in grab_points.py); if several vanish
    (both arms busy) the one whose direction matches this arm's shoulder pan at the grab (geometric pan model, < 8 deg);
  * its picture = a 64x64 crop from the top camera, from the episode's first frame if it was already there (no arm in
    the way), else from just before the grab;
  * its LABEL = the bin it was dropped into: when the gripper opens again, the arm's shoulder pan says which bin it is
    over (pan of each bin = the arm's geometric pan at the bin centre; must be < 9 deg from one bin and clearly further
    from the others). Bin -> shape with the official Round 2 mapping (rulebook 5.8.1.46): red = cube, blue = cylinder,
    green = dodecahedron, yellow = pentagonal prism.
  * bins are found by their RIM (the shape-demo bins are full of blocks of every colour, so the usual "one big patch
    of colour" bin finder fails); a bin missed in one episode is taken from the previous episodes if the other bins
    are still in the same places (the layout changes between demo sessions).
Writes classical\\shape_data\\crops.npz + one montage picture per class (to check the labels by eye).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import model  # noqa: E402
import vision  # noqa: E402

BIN_TO_SHAPE = {"red": "cube", "blue": "cylinder", "green": "dodecahedron", "yellow": "pentagonal_prism"}
CROP = 64


def top_img(ds, idx):
    t = ds[idx]["observation.images.top"]
    return cv2.cvtColor((t.permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


WRIST_SIZE = (160, 90)        # wrist frames saved at this size (w, h); both wrist cameras are 16:9
WRIST_DT = (-8, -4, 0, 4)     # frames around the moment the gripper starts closing


def wrist_img(ds, idx, side):
    t = ds[idx][f"observation.images.{side}_wrist"]
    im = cv2.cvtColor((t.permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.resize(im, WRIST_SIZE, interpolation=cv2.INTER_AREA)


def crop(img, u, v, n=CROP):
    pad = cv2.copyMakeBorder(img, n, n, n, n, cv2.BORDER_REPLICATE)
    x, y = int(round(u)) + n - n // 2, int(round(v)) + n - n // 2
    return pad[y:y + n, x:x + n].copy()


def full_bins(img, vmax=345):
    """Bins found by their coloured RIM, even when full of blocks: closed colour mask -> convex hull -> the outer band
    of the hull must be mostly that colour. Returns {colour: (cx, cy, w, h, angle, rim_fraction)}."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    top = np.zeros(img.shape[:2], np.uint8)
    top[150:vmax, 150:580] = 255
    found = {}
    for name in vision.COLOURS:
        m = vision.colour_mask(hsv, name) & top
        mc = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        cnts, _ = cv2.findContours(mc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in cnts:
            hull = cv2.convexHull(c)
            ha = cv2.contourArea(hull)
            (cx, cy), (bw, bh), ang = cv2.minAreaRect(hull)
            if ha < 900 or not (28 <= min(bw, bh) and max(bw, bh) <= 95):     # bin-sized (26 Nov: v2 bins ~1000-1700 px)
                continue
            filled = np.zeros_like(m)
            cv2.drawContours(filled, [hull], -1, 255, -1)
            ring = cv2.subtract(filled, cv2.erode(filled, np.ones((9, 9), np.uint8)))
            rim = float((m[ring > 0] > 0).mean())
            if rim < (0.35 if name == "blue" else 0.45):
                continue
            if best is None or ha > best[0]:
                best = (ha, (cx, cy, bw, bh, ang, rim))
        if best:
            found[name] = best[1]
    return found


def episode_bins(imgs, memory):
    """Bins of one episode from several frames; colours still missing come from `memory` (recent episodes) if every bin
    seen now is where memory has it (same layout)."""
    b = {}
    for img in imgs:
        for c, v in full_bins(img).items():
            if c not in b or v[5] > b[c][5]:
                b[c] = v
    same = all(c in memory and np.hypot(v[0] - memory[c][0], v[1] - memory[c][1]) < 20 for c, v in b.items())
    if b and same:
        for c, v in memory.items():
            b.setdefault(c, v)
    return b


def collect(name, maps):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(f"local/{name}", root=Path(r"D:\lerobot\data") / name)
    sn = ds.meta.features["observation.state"]["names"]
    an = ds.meta.features["action"]["names"]
    rows, skipped, memory = [], {}, {}

    def skip(why):
        skipped[why] = skipped.get(why, 0) + 1
    for e in range(ds.meta.total_episodes):
        a0, a1 = int(ds.meta.episodes["dataset_from_index"][e]), int(ds.meta.episodes["dataset_to_index"][e])
        hf = ds.hf_dataset.select(range(a0, a1))
        act = np.stack([np.asarray(x) for x in hf["action"]])
        st = np.stack([np.asarray(x) for x in hf["observation.state"]])
        first, last = top_img(ds, a0), top_img(ds, a1 - 1)
        det0 = vision.detect(first)
        bins = episode_bins([first, last], memory)
        memory = dict(bins)
        for arm, side in enumerate(("left", "right")):
            m = maps[side]
            g = act[:, an.index(f"{side}_gripper.pos")]
            pan = st[:, sn.index(f"{side}_shoulder_pan.pos")]
            was_open = False
            for t in range(1, len(g)):
                if g[t - 1] > 15:
                    was_open = True
                if not (was_open and g[t] < 12 <= g[t - 1]):
                    continue
                was_open = False
                t_c = min(t + 6, len(g) - 1)
                t_after = min(t_c + 45, len(g) - 1)
                if t < 3 or t_after - t_c < 15:
                    skip("grab too near the episode edge")
                    continue
                # before the grab: 1-3 s earlier (at 0.7 s the arm already covers the block); union of 3 looks
                looks = [top_img(ds, a0 + max(t - k, 0)) for k in (30, 60, 90)]
                before_img = looks[0]
                before = []
                for im in looks:
                    for q in vision.detect(im)["objects"]:
                        if not any(p[0] == q[0] and np.hypot(p[1] - q[1], p[2] - q[2]) < 10 for p in before):
                            before.append(q)
                after = vision.detect(top_img(ds, a0 + t_after))["objects"]
                gone = [o for o in before
                        if not any(p[0] == o[0] and np.hypot(p[1] - o[1], p[2] - o[2]) < 15 for p in after)]
                gone = [o for o in gone if m.can_reach(o[1], o[2])]
                p_grab = float(pan[t_c])
                gone = sorted(gone, key=lambda o: abs(m.pan_geo(o[1], o[2]) - p_grab))
                if not gone or abs(m.pan_geo(gone[0][1], gone[0][2]) - p_grab) > 8:
                    skip("no vanished block in the arm's direction")
                    continue
                if len(gone) > 1 and abs(m.pan_geo(gone[1][1], gone[1][2]) - p_grab) < 8:
                    skip("two vanished blocks in the arm's direction")
                    continue
                o = gone[0]
                rel = next((k for k in range(t_c, len(g)) if g[k] > 15), None)
                if rel is None:
                    skip("not released in this episode")
                    continue
                p_rel = float(pan[rel])
                cand = sorted((abs(p_rel - m.pan_geo(b[0], b[1])), c) for c, b in bins.items())
                if not cand or cand[0][0] > 9 or (len(cand) > 1 and cand[1][0] < cand[0][0] + 6):
                    skip("release pan not clearly over one bin")
                    continue
                label_bin = cand[0][1]
                src = first if any(p[0] == o[0] and np.hypot(p[1] - o[1], p[2] - o[2]) < 8 for p in det0["objects"]) \
                    else before_img
                wr = np.stack([wrist_img(ds, a0 + min(max(t + k, 0), len(g) - 1), side) for k in WRIST_DT])
                rows.append(dict(img=crop(src, o[1], o[2]), wrist=wr, t=t, colour=o[0], u=o[1], v=o[2], angle=o[3], area=o[4],
                                 w=o[5] if len(o) > 5 else 0, h=o[6] if len(o) > 6 else 0, arm=arm, episode=e,
                                 dataset=name, bin=label_bin, shape=BIN_TO_SHAPE[label_bin], pan_err=cand[0][0],
                                 n_bins=len(bins)))
        if e % 20 == 0:
            print(f"   {name}: episode {e}/{ds.meta.total_episodes}, {len(rows)} labelled crops, bins now "
                  f"{ {c: (round(v[0]), round(v[1])) for c, v in bins.items()} }", flush=True)
    print(f"   {name}: {len(rows)} labelled crops | skipped: {skipped}", flush=True)
    return rows


def montage(rows, path, n=80):
    tiles = [r["img"] for r in rows[:n]]
    if not tiles:
        return
    cols = 10
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(path), cv2.resize(grid, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST))


if __name__ == "__main__":
    sets = sys.argv[1:] or ["ieee_shape_clean", "ieee_shape_v2"]
    maps = {s: model.load(s) for s in ("left", "right")}
    rows = []
    for s in sets:
        rows += collect(s, maps)
    out = HERE / "shape_data"
    out.mkdir(exist_ok=True)
    if not rows:
        sys.exit("no labelled crops")
    keys = [k for k in rows[0] if k not in ("img", "wrist")]
    np.savez_compressed(out / "crops.npz", img=np.stack([r["img"] for r in rows]), wrist=np.stack([r["wrist"] for r in rows]),
                        **{k: np.array([r[k] for r in rows]) for k in keys})
    print(f"saved {len(rows)} crops -> {out / 'crops.npz'}")
    for sh in sorted(set(r["shape"] for r in rows)):
        rr = [r for r in rows if r["shape"] == sh]
        cols = {}
        for r in rr:
            cols[r["colour"]] = cols.get(r["colour"], 0) + 1
        print(f"   {sh:17s} {len(rr):4d}  colours {cols}")
        montage(rr, out / f"montage_{sh}.png")
