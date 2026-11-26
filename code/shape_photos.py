"""PHOTO SESSION for the shape classifier (26 Nov) - CLEAN labels. Top camera only, the arms do NOT move.
    python classical\\shape_photos.py            (SHAPE_PHOTOS.bat)
For each shape in turn: put ONLY that shape on the table (all 8 of them: 4 colours x small + large), spread out inside
the black line, not touching each other or the bins -> Enter = photo. Every block the camera finds in that photo is
labelled with that shape - no guessing. Rearrange between photos (turn them, lay cylinders / prisms down AND stand
them up, move them near the edges and the middle) - about 10 photos per shape.
Saves classical\\shape_data\\photos.npz (crops + labels, added to on every run) and the full photos in
classical\\shape_data\\photos\\. train_shape_net.py uses them automatically.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import hw  # noqa: E402
import shape_model as SM  # noqa: E402
import vision  # noqa: E402

OUT = HERE / "shape_data"
PHOTOS = OUT / "photos"
NPZ = OUT / "photos.npz"


def table_blocks(det, img, background=()):
    """Loose blocks on the table only: inside the black line and NOT in / at a bin (26 Nov: bins full of other shapes
    were counted - 24 'cubes' instead of 8). Bins found normally AND by their rim (full bins)."""
    bins = dict(det["bins"])
    for c, b in vision.find_bins_by_rim(img).items():
        bins.setdefault(c, b[:5])
    boxes = [cv2.boxPoints(((float(b[0]), float(b[1])), (float(b[2]), float(b[3])), float(b[4]))).astype(np.float32)
             for b in bins.values()]
    out = []
    for o in det["objects"]:
        pt = (float(o[1]), float(o[2]))
        if cv2.pointPolygonTest(vision.SOURCE_POLY, pt, True) < -8:     # photos: a block ON the line still counts
            continue                                                       # (the sorter itself ignores those)
        if any(cv2.pointPolygonTest(bx, pt, True) > 5 for bx in boxes):      # clearly INSIDE a bin (a cube right
            continue                                                           # next to a bin is fine - 26 Nov)
        if any(np.hypot(o[1] - b[0], o[2] - b[1]) < 12 for b in background):   # was already there (bin contents)
            continue
        out.append(o)
    return out


def load():
    if NPZ.exists():
        d = np.load(NPZ, allow_pickle=True)
        return {k: list(d[k]) for k in d.files}
    return {k: [] for k in ("img", "shape", "colour", "u", "v", "photo")}


def main():
    PHOTOS.mkdir(parents=True, exist_ok=True)
    data = load()
    print(f"photo set so far: {len(data['img'])} crops " +
          str({c: sum(1 for s in data['shape'] if s == c) for c in SM.CLASSES}))
    cam = hw.TopCamera()
    cam.connect()
    try:
        print()
        input(">>> BACKGROUND first: take ALL loose blocks off the table (the bins stay exactly as they are, full or "
              "not), hands out of view, Enter. ")
        bg = vision.detect_avg([cam.grab() for _ in range(3)])["objects"]
        background = [(o[1], o[2]) for o in bg]
        print(f"   background: {len(background)} blocks already there (in the bins etc.) - these are never counted")
        start = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in SM.CLASSES else SM.CLASSES[0]
        for shape in SM.CLASSES[SM.CLASSES.index(start):]:      # SHAPE_PHOTOS.bat cylinder = start there
            n_photo = 0
            print(f"\n=== {shape.upper().replace('_', ' ')}S: put ONLY {shape.replace('_', ' ')}s on the table "
                  f"(all colours, small + large), spread out inside the black line ===")
            while True:
                k = input(f">>> Enter = photo  |  n = next shape  |  q = finish   ({shape}, {n_photo} photos so far) "
                          ).strip().lower()
                if k == "q":
                    return
                if k == "n":
                    break
                frames = [cam.grab() for _ in range(3)]
                det = vision.detect_avg(frames)
                objs = table_blocks(det, frames[-1], background)
                if not objs:
                    print("   no blocks found - hands out of the picture? blocks inside the black line?")
                    continue
                if len(objs) > 8:
                    print(f"   WARNING: {len(objs)} blocks found - only up to 8 of one shape exist. Something else is on "
                          f"the table? Photo NOT used - check and press Enter again.")
                    continue
                tag = f"{shape}_{time.strftime('%H%M%S')}"
                img = frames[-1]
                cv2.imwrite(str(PHOTOS / f"{tag}.png"), img)
                for o in objs:
                    for f in frames:                      # 3 slightly different frames per block
                        data["img"].append(SM.crop_at(f, o[1], o[2]))
                        data["shape"].append(shape)
                        data["colour"].append(o[0])
                        data["u"].append(o[1])
                        data["v"].append(o[2])
                        data["photo"].append(tag)
                n_photo += 1
                np.savez_compressed(NPZ, **{k: np.array(v) for k, v in data.items()})
                print(f"   {len(objs)} blocks labelled '{shape}' {sorted(o[0] for o in objs)} -> saved "
                      f"({len(data['img'])} crops in total). Rearrange them for the next photo.")
    finally:
        cam.disconnect()
        print(f"\nphoto set: {len(data['img'])} crops " +
              str({c: sum(1 for s in data['shape'] if s == c) for c in SM.CLASSES}) +
              "\nNext: python classical\\train_shape_net.py")


if __name__ == "__main__":
    main()
