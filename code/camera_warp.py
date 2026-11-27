"""TOP (pole) CAMERA ALIGNMENT - put the camera back where it was in the lab, or correct it in software. 27 Nov.

    CAMERA_ALIGN.bat          live window (keys below)
    CAMERA_ALIGN.bat save     save the camera's view NOW as the reference (do this in the lab, table clear)

Everything in the sorter (calibration, black line, bin spots, arm bases) is in pixels of the LAB camera view.
If the camera sits differently at the competition:
  1. Physically: press C, click the 4 INNER corners of the black line. It tells you which way to move / turn /
     raise / rotate the camera. Adjust, press C again, repeat until it says GOOD (all corners within 4 px).
  2. Software (if it can't get closer): after clicking the corners press W. camera_warp.json is saved and from
     then on EVERY top-camera picture (sort, teach, calibrate, shape photos) is straightened back into the lab view
     before anything looks at it. X removes it again. SORT_CAM_WARP=0 switches it off for one run.
     (Only the classical sorter uses this. The AI policies - pour, fold, AI colour - need the camera physically
     close to the lab position; use the ghost picture for those.)

Keys in the window:
  C  click the 4 inner corners of the black line (top-left, top-right, bottom-right, bottom-left as seen in the picture)
  W  write the software correction from the last 4 clicks      X  delete the software correction
  V  view: raw camera <-> corrected (what the sorter will see)  G  ghost of the lab reference picture on/off
  S  save the current RAW view as the new reference (old one is kept with a time stamp)   Q / Esc  quit
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REF_IMG = HERE / "camera_reference.jpg"
WARP_FILE = HERE / "camera_warp.json"

# the lab view (26 Nov): inner corners of the black line = vision.SOURCE_POLY corners
REF_CORNERS = np.float32([(197, 212), (524, 216), (508, 398), (197, 384)])
CORNER_NAMES = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]
ARM_BASES = {"left": (242, 420), "right": (458, 430)}          # sort.ARM_BASE_PX
BIN_SPOTS = [(224, 266), (305, 250), (408, 256), (502, 280)]    # where the bins stood in the demos
GOOD_PX = 4.0

_H = None
_loaded = False


def load():
    """The saved software correction (3x3, live pixels -> lab pixels) or None."""
    global _H, _loaded
    if not _loaded:
        _loaded = True
        if os.environ.get("SORT_CAM_WARP", "1") != "0" and WARP_FILE.exists():
            try:
                d = json.loads(WARP_FILE.read_text())
                _H = np.array(d["H"], float)
                print(f"   top camera: software alignment ON ({WARP_FILE.name}, {d.get('time', '?')}, "
                      f"corners were up to {d.get('max_px', 0):.0f} px off) - SORT_CAM_WARP=0 to switch it off")
            except Exception as e:
                print(f"   (top camera: could not read {WARP_FILE.name}: {e!r} - no correction)")
    return _H


def apply(img):
    """Straighten a top-camera picture into the lab view (no-op when there is no correction)."""
    H = load()
    if H is None:
        return img
    import cv2
    return cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def homography(clicked):
    import cv2
    return cv2.getPerspectiveTransform(np.float32(clicked), REF_CORNERS)


def advice(clicked):
    """Plain-language instructions to move the camera so the clicked corners land on the lab corners."""
    p, r = np.float32(clicked), REF_CORNERS
    err = np.linalg.norm(p - r, axis=1)
    lines = [f"corners off by {', '.join(f'{e:.0f}' for e in err)} px (max {err.max():.0f})"]
    if err.max() <= GOOD_PX:
        return ["GOOD - camera matches the lab view (all corners within %.0f px). No correction needed." % GOOD_PX], err

    def size(q):
        return ((np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2,
                (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2)
    (w1, h1), (w0, h0) = size(p), size(r)
    dx, dy = (p.mean(0) - r.mean(0))
    rot = math.degrees(math.atan2(p[1][1] - p[0][1], p[1][0] - p[0][0]) - math.atan2(r[1][1] - r[0][1], r[1][0] - r[0][0]))
    s = (w1 / w0 + h1 / h0) / 2
    key1 = (np.linalg.norm(p[1] - p[0]) / max(1, np.linalg.norm(p[2] - p[3])))
    key0 = (np.linalg.norm(r[1] - r[0]) / max(1, np.linalg.norm(r[2] - r[3])))
    if abs(dx) > 2:
        lines.append(f"table appears {abs(dx):.0f} px too far {'RIGHT' if dx > 0 else 'LEFT'} -> turn/slide the camera "
                     f"to the {'RIGHT' if dx > 0 else 'LEFT'} (as seen in the picture)")
    if abs(dy) > 2:
        lines.append(f"table appears {abs(dy):.0f} px too {'LOW (towards the arms)' if dy > 0 else 'HIGH (away from the arms)'}"
                     f" -> point the camera more {'towards the arms' if dy > 0 else 'away from the arms'}")
    if abs(s - 1) > 0.02:
        lines.append(f"table looks {abs(s - 1) * 100:.0f}% {'BIGGER' if s > 1 else 'SMALLER'} -> camera too "
                     f"{'CLOSE: raise it / move it away' if s > 1 else 'FAR: lower it / move it closer'}")
    if abs(rot) > 0.7:
        lines.append(f"picture is turned {abs(rot):.1f} deg {'clockwise' if rot > 0 else 'anticlockwise'} -> rotate the "
                     f"camera {'clockwise' if rot > 0 else 'anticlockwise'} on its mount")
    if abs(key1 / key0 - 1) > 0.03:
        lines.append(f"far edge vs near edge width is {abs(key1 / key0 - 1) * 100:.0f}% different -> the camera's "
                     f"up/down TILT differs; tilt it until the yellow outline's shape matches the black line")
    lines.append("or press W to correct it in software")
    return lines, err


def save_reference(img):
    import cv2
    if REF_IMG.exists():
        REF_IMG.rename(HERE / f"camera_reference_{time.strftime('%m%d_%H%M%S')}.jpg")
    cv2.imwrite(str(REF_IMG), img)
    print(f"reference saved: {REF_IMG}")


def save_warp(clicked, err):
    H = homography(clicked)
    if WARP_FILE.exists():
        WARP_FILE.rename(HERE / f"camera_warp_{time.strftime('%m%d_%H%M%S')}.json")
    WARP_FILE.write_text(json.dumps(dict(H=H.tolist(), clicked=[list(map(float, c)) for c in clicked],
                                         reference=REF_CORNERS.tolist(), max_px=float(err.max()),
                                         time=time.strftime("%Y-%m-%d %H:%M")), indent=1))
    global _H, _loaded
    _H, _loaded = H, True
    # where the arm bases end up: a sanity check (they are physically fixed - should stay close)
    import cv2
    inv = np.linalg.inv(H)
    for side, (u, v) in ARM_BASES.items():
        q = cv2.perspectiveTransform(np.float32([[[u, v]]]), inv)[0, 0]
        print(f"   {side} arm base should now be seen at ({q[0]:.0f},{q[1]:.0f}) in the RAW picture - check it is on the base")
    print(f"software correction saved: {WARP_FILE}")


def draw_overlay(img, cv2):
    import vision
    cv2.polylines(img, [vision.SOURCE_POLY.reshape(-1, 1, 2).astype(np.int32)], True, (0, 255, 255), 1)
    for i, (u, v) in enumerate(REF_CORNERS):
        cv2.circle(img, (int(u), int(v)), 4, (0, 255, 255), 1)
        cv2.putText(img, str(i + 1), (int(u) + 5, int(v) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    for side, (u, v) in ARM_BASES.items():
        cv2.drawMarker(img, (u, v), (255, 0, 255), cv2.MARKER_CROSS, 16, 1)
    for i, (u, v) in enumerate(BIN_SPOTS):
        cv2.circle(img, (u, v), 22, (255, 200, 0), 1)
        cv2.putText(img, f"bin {i + 1}", (u - 14, v + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1)


def main():
    import cv2
    sys.path.insert(0, str(HERE))
    import hw
    cam = hw.TopCamera()
    cam.connect()
    try:
        if len(sys.argv) > 1 and sys.argv[1].lower() == "save":
            save_reference(cam.grab(raw=True))
            return
        ref = cv2.imread(str(REF_IMG)) if REF_IMG.exists() else None
        Z = 1.5
        st = dict(clicks=[], clicking=False, ghost=ref is not None, corrected=False, msg=[], err=None, last=None)
        st["msg"] = ["C = click the 4 inner corners of the black line   G ghost   V raw/corrected   W save correction",
                     "X delete correction   S save view as reference   Q quit"] + \
                    ([] if ref is not None else ["(no reference picture yet - in the lab, table clear: press S)"])

        def on_mouse(ev, x, y, flags, _):
            if ev == cv2.EVENT_LBUTTONDOWN and st["clicking"]:
                st["clicks"].append((x / Z, y / Z))
                if len(st["clicks"]) == 4:
                    st["clicking"] = False
                    st["msg"], st["err"] = advice(st["clicks"])
                    st["last"] = list(st["clicks"])
                    print("\n".join(st["msg"]))
        win = "TOP CAMERA ALIGNMENT"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, on_mouse)
        while True:
            raw = cam.grab(raw=True)
            img = apply(raw) if st["corrected"] else raw.copy()
            if st["ghost"] and ref is not None:
                img = cv2.addWeighted(img, 0.6, ref, 0.4, 0)
            draw_overlay(img, cv2)
            for (u, v) in st["clicks"]:
                cv2.drawMarker(img, (int(u), int(v)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
            big = cv2.resize(img, None, fx=Z, fy=Z)
            head = ([f"CLICK the {CORNER_NAMES[len(st['clicks'])]} inner corner of the black line"] if st["clicking"]
                    else st["msg"])
            head = head + [f"view: {'CORRECTED (what the sorter sees)' if st['corrected'] else 'RAW camera'}"
                           f" | ghost {'on' if st['ghost'] else 'off'} | software correction "
                           f"{'ON' if load() is not None else 'off'}"]
            for i, t in enumerate(head):
                cv2.putText(big, t, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(big, t, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow(win, big)
            k = cv2.waitKey(30) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord("c"):
                st.update(clicks=[], clicking=True, corrected=False)
            elif k == ord("g"):
                st["ghost"] = not st["ghost"]
            elif k == ord("v"):
                st["corrected"] = not st["corrected"]
            elif k == ord("w"):
                if st["last"] is None:
                    st["msg"] = ["press C and click the 4 corners first"]
                else:
                    save_warp(st["last"], st["err"])
                    st["msg"] = ["software correction SAVED - press V to see what the sorter will see"]
            elif k == ord("x"):
                if WARP_FILE.exists():
                    WARP_FILE.rename(HERE / f"camera_warp_removed_{time.strftime('%m%d_%H%M%S')}.json")
                global _H, _loaded
                _H, _loaded = None, True
                st["msg"] = ["software correction removed (old file kept as camera_warp_removed_*.json)"]
            elif k == ord("s"):
                save_reference(raw)
                ref = cv2.imread(str(REF_IMG))
                st["msg"] = ["this view is now the reference picture (ghost)"]
        cv2.destroyAllWindows()
    finally:
        cam.disconnect()


if __name__ == "__main__":
    main()
