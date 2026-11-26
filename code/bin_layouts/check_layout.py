"""Bin layouts for the colour sorter (26 Nov).
    python classical\\bin_layouts\\check_layout.py all          -> tests ALL 24 orders of the 4 bins on our 4 bin spots
    python classical\\bin_layouts\\check_layout.py live         -> top camera only (NO motors): what the sorter would do
                                                                  with the bins as they stand now + a marked picture
    python classical\\bin_layouts\\check_layout.py <image.jpg>  -> same, from a saved top-camera picture
The drop poses belong to the 4 bin SPOTS (where the bins stood in the demos), not to colours (model.ArmMap.bin_slot):
whatever colour stands on a spot, the arm next to that spot drops it there with that spot's pose, drop height and
learned sag correction. A bin more than SORT_SLOT_R px (35, ~4 cm) from every spot is NOT used (flagged below).
"""
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
CLS = HERE.parent
sys.path.insert(0, str(CLS))
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
sys.argv = [sys.argv[0]]          # sort.py reads the command line when imported
import model  # noqa: E402
import vision  # noqa: E402

COLOURS = ["red", "green", "blue", "yellow"]


def load_maps():
    import sort
    maps = {s: model.load(s) for s in ("left", "right")}
    for m in maps.values():
        m.roll_offset = 0.0
    sort.check_bins(maps)        # drops bad stored bin poses (the right arm's old green one)
    return maps


def spots(maps):
    """All stored bin spots: name -> (u, v, arm)."""
    out = {}
    for side, m in maps.items():
        for name, b in m.bins.items():
            out[name] = (b["u"], b["v"], side)
    return out


def assignment(maps, bins):
    """colour -> (arm, spot) the sorter would use; None = no arm can drop into it."""
    res = {}
    for c, b in bins.items():
        hit = [(side, m.bin_slot(c, b[0], b[1])) for side, m in maps.items()]
        hit = [(s, sl) for s, sl in hit if sl is not None]
        res[c] = hit[0] if len(hit) == 1 else (hit if hit else None)
    return res


def test_all(maps):
    import sort
    sp = spots(maps)
    names = sorted(sp, key=lambda n: sp[n][0])                     # spots left -> right in the camera
    lines = [f"Bin spots (left -> right in the top camera): " +
             ", ".join(f"{n} spot ({sp[n][0]:.0f},{sp[n][1]:.0f}) = {sp[n][2]} arm" for n in names), ""]
    ok_all = True
    for perm in itertools.permutations(COLOURS):
        bins = {c: (sp[n][0], sp[n][1], 60.0, 50.0, 0.0) for c, n in zip(perm, names)}
        a = assignment(maps, bins)
        good = all(isinstance(a[c], tuple) and a[c][1] == n for c, n in zip(perm, names))
        # each arm plans its drops with exactly the spot's stored pose
        for c, n in zip(perm, names):
            side = sp[n][2]
            p = sort.plan((c, 300.0, 330.0, -90.0, 500.0, 25.0, 25.0), maps[side], bins[c][:2])
            good &= p.get("slot") == n and p["bin"] is not None
        ok_all &= good
        lines.append(f"{'OK ' if good else 'BAD'}  order {' / '.join(perm):30s} -> "
                     + ", ".join(f"{c}: {a[c][0]} arm on the {a[c][1]} spot" if isinstance(a[c], tuple) else f"{c}: ???"
                                 for c in perm))
    lines.append("")
    lines.append("ALL 24 ORDERS WORK" if ok_all else "SOME ORDERS FAIL - see BAD lines")
    return "\n".join(lines), ok_all


def check_image(maps, img, out_path):
    det = vision.detect(img)
    for c, b in vision.find_bins_by_rim(img).items():
        det["bins"].setdefault(c, b[:5])
    sp = spots(maps)
    a = assignment(maps, det["bins"])
    vis = img.copy()
    for n, (u, v, side) in sp.items():
        cv2.circle(vis, (int(u), int(v)), int(model.SLOT_R), (200, 200, 200), 1)
        cv2.putText(vis, f"{n} spot ({side})", (int(u) - 30, int(v) - int(model.SLOT_R) - 4), 0, 0.35, (90, 90, 90), 1)
    print("bins seen by the camera:")
    problems = 0
    for c in COLOURS:
        if c not in det["bins"]:
            print(f"   {c:6s} NOT SEEN - check the camera view / lighting")
            problems += 1
            continue
        b = det["bins"][c]
        x = a.get(c)
        if isinstance(x, tuple):
            msg = f"OK  -> {x[0]} arm drops {c} here (pose of the {x[1]} spot)"
            col = (0, 200, 0)
        else:
            d = min(np.hypot(b[0] - u, b[1] - v) for u, v, _ in sp.values())
            msg = f"NOT ON A KNOWN SPOT ({d:.0f} px from the nearest, limit {model.SLOT_R:.0f}) - move it onto a spot, " \
                  f"or teach it: TEACH_BIN.bat <arm> {c}"
            col = (0, 0, 255)
            problems += 1
        print(f"   {c:6s} at ({b[0]:.0f},{b[1]:.0f}): {msg}")
        box = cv2.boxPoints(((b[0], b[1]), (b[2], b[3]), b[4])).astype(np.int32)
        cv2.polylines(vis, [box], True, col, 2)
        cv2.putText(vis, c + (" OK" if col == (0, 200, 0) else " ??"), (int(b[0]) - 20, int(b[1]) + 4), 0, 0.45, col, 2)
    cv2.polylines(vis, [vision.SOURCE_POLY], True, (255, 0, 255), 1)
    cv2.imwrite(str(out_path), vis)
    print(f"\n{'READY: every bin is on a known spot' if not problems else f'{problems} PROBLEM(S) - see above'}"
          f" | picture: {out_path}")
    return problems


if __name__ == "__main__":
    what = MODE
    maps = load_maps()
    if what == "all":
        txt, ok = test_all(maps)
        print(txt)
        (HERE / "ALL_24_LAYOUTS.txt").write_text(txt + "\n", encoding="utf-8")
    elif what == "live":
        import hw
        cam = hw.TopCamera()
        cam.connect()
        try:
            img = [cam.grab() for _ in range(3)][-1]
        finally:
            cam.disconnect()
        check_image(maps, img, HERE / "live_check.png")
    else:
        check_image(maps, cv2.imread(what), HERE / "image_check.png")
