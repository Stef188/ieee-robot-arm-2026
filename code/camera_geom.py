"""CAMERA GEOMETRY (27 Nov, competition): exact camera position here and in the lab, from the black line + the
camera's own lens data (RealSense D405 colour intrinsics) - and the block-HEIGHT correction it makes possible.

Why: the camera sees the TOP of a block at an angle, so a 2-5 cm tall block appears shifted from where it really
stands. The camera correction (camera_warp.json) only straightens the flat table. With both camera poses known:
    block top seen here at pixel p, height h  ->  where it really stands on the table (cm)
                                               ->  the pixel where the LAB camera would have seen its top
which is exactly what the grab model learned from. h comes from the block's own outline (its smaller side, in cm).

Black line: 80 x 50 cm OUTSIDE (measured at the competition, 27 Nov). The corners used are the tape's INNER edge;
the tape width comes out of the picture (aspect ratio with the known lens) - a check that everything fits.
    python camera_geom.py      -> prints both camera poses, the tape width, the size of the correction; saves camera_geom.json
"""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
K = np.array([[391.6, 0, 323.1], [0, 390.6, 242.5], [0, 0, 1.0]])       # D405 colour 640x480 (read 27 Nov)
OUT_W, OUT_D = 80.0, 50.0                                                  # cm, outside of the tape
LAB_CORNERS = np.float32([(197, 212), (524, 216), (508, 398), (197, 384)])   # = camera_warp.REF_CORNERS (lab view)
GEOM_FILE = HERE / "camera_geom.json"


def homog(src, dst):
    import cv2
    return cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))


def pose_from_corners(img_corners):
    """Corners TL, TR, BR, BL (image order: far-left, far-right, near-right, near-left) of the inner rectangle ->
    (aspect W/D, pose). Plane coords: x right (cm), y towards the arms (cm), z up out of the table."""
    H0 = homog([(0, 0), (1, 0), (1, 1), (0, 1)], img_corners)
    M = np.linalg.inv(K) @ H0
    m1, m2 = M[:, 0], M[:, 1]
    return np.linalg.norm(m1) / np.linalg.norm(m2), M


def solve(img_corners, W, D):
    """Camera pose for the inner rectangle W x D cm seen at img_corners: R (plane->camera), t (cm)."""
    H = homog([(0, 0), (W, 0), (W, D), (0, D)], img_corners)
    M = np.linalg.inv(K) @ H
    lam = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    r1, r2, t = M[:, 0] * lam, M[:, 1] * lam, M[:, 2] * lam
    if t[2] < 0:                                       # camera in front of the plane
        r1, r2, t = -r1, -r2, -t
    r1 /= np.linalg.norm(r1)
    r2 = r2 - r1 * (r1 @ r2); r2 /= np.linalg.norm(r2)
    R = np.column_stack([r1, r2, np.cross(r1, r2)])
    return R, t


LAB_BASES = ((242.0, 420.0), (458.0, 430.0))        # lab arm bases in the lab picture (sort.ARM_BASE_PX)


def lab_width_from_arms(a_lab):
    """The lab line's size is unknown: choose its width so the two lab arm bases come out 50 cm apart (rule 5.1.3.3)."""
    Ki = np.linalg.inv(K)

    def to_cm(R, t, p):
        c = -R.T @ t
        d = R.T @ (Ki @ np.array([p[0], p[1], 1.0]))
        return (c + (-c[2] / d[2]) * d)[:2]

    def sep(wd):
        R, t = solve(LAB_CORNERS, wd, wd / a_lab)
        return np.linalg.norm(to_cm(R, t, LAB_BASES[0]) - to_cm(R, t, LAB_BASES[1]))
    return float(min(np.arange(55, 95, 0.05), key=lambda wd: abs(sep(wd) - 50.0)))


def cam_centre(R, t):
    return -R.T @ t


def describe(name, R, t, W, D):
    c = cam_centre(R, t)
    zdir = R.T @ np.array([0, 0, 1.0])                 # optical axis in plane coords
    tilt = math.degrees(math.acos(abs(zdir[2])))
    return (f"{name}: camera {abs(c[2]):.1f} cm above the table, over the point ({c[0]:.1f}, {c[1]:.1f}) cm "
            f"(x from the line's left inner edge, y from its far inner edge), tilted {tilt:.0f} deg from straight down")


def main():
    wj = json.loads((HERE / "camera_warp.json").read_text())
    comp = np.float32(wj["clicked"])
    out = {}
    tapes = {}
    for name, cor in (("lab", LAB_CORNERS), ("competition", comp)):
        a, _ = pose_from_corners(cor)
        # inner W/D = a with W = OUT_W - 2 w, D = OUT_D - 2 w  ->  tape width w
        w = (OUT_W - a * OUT_D) / (2 * (1 - a))
        tapes[name] = w
        print(f"{name}: inner rectangle aspect {a:.3f} -> tape width {w:.2f} cm (inner {OUT_W - 2 * w:.1f} x {OUT_D - 2 * w:.1f} cm)")
    # 27 Nov: the LAB line is a different rectangle (aspect 1.75 vs 1.64) - only the competition line's size is known
    w = float(np.clip(tapes["competition"], 0.5, 4.0))
    W, D = OUT_W - 2 * w, OUT_D - 2 * w
    a_lab, _ = pose_from_corners(LAB_CORNERS)
    WL = lab_width_from_arms(a_lab)
    for name, cor, (ww, dd) in (("lab", LAB_CORNERS, (WL, WL / a_lab)), ("competition", comp, (W, D))):
        R, t = solve(cor, ww, dd)
        out[name] = {"R": R.tolist(), "t": t.tolist()}
        print(describe(name, R, t, W, D))
    out.update(tape_cm=w, inner_cm=[W, D], K=K.tolist(), H_warp=wj["H"])
    GEOM_FILE.write_text(json.dumps(out, indent=1))
    G = Geom()
    # check: at height 0 the new mapping must equal the existing camera correction
    errs = []
    for u in range(200, 520, 40):
        for v in range(220, 400, 30):
            errs.append(np.hypot(*(np.array(G.to_lab(u, v, 0.0)) - (u, v))))
    print(f"consistency with the camera correction at table height: mean {np.mean(errs):.2f} px, max {np.max(errs):.2f} px")
    print("block-height correction (pixels the grab aim moves, lab view):")
    for (u, v) in ((260, 330), (350, 300), (450, 330), (350, 370)):
        moves = [np.array(G.to_lab(u, v, h)) - (u, v) for h in (2.0, 3.0, 4.0, 5.0)]
        print(f"   block at ({u},{v}): " + ", ".join(f"h {h:.0f} cm -> ({m[0]:+.1f},{m[1]:+.1f}) px" for h, m in zip((2, 3, 4, 5), moves)))
    print(f"saved {GEOM_FILE.name}")


class Geom:
    """Loaded from camera_geom.json. to_lab(u, v, h): block TOP seen at (u, v) in the corrected (lab-view) picture at
    height h cm -> the pixel where the lab camera would have seen that same top (what the grab model learned)."""

    def __init__(self):
        g = json.loads(GEOM_FILE.read_text())
        self.Rl, self.tl = np.array(g["lab"]["R"]), np.array(g["lab"]["t"])
        self.Rc, self.tc = np.array(g["competition"]["R"]), np.array(g["competition"]["t"])
        self.Hw = np.array(g["H_warp"])            # comp raw -> lab view (table plane)
        self.Hinv = np.linalg.inv(self.Hw)
        self.Ki = np.linalg.inv(K)
        self.cl = cam_centre(self.Rl, self.tl)
        self.cc = cam_centre(self.Rc, self.tc)

    def _ray_plane(self, Rc, tc, cc, pix, h):
        d = Rc.T @ (self.Ki @ np.array([pix[0], pix[1], 1.0]))
        # plane z = h: with the camera on the side of +z or -z, solve along the ray
        s = (h * np.sign(cc[2]) * 0 + h - cc[2]) / d[2] if False else (self._zsign() * h - cc[2]) / d[2]
        return cc + s * d

    def _zsign(self):
        return 1.0 if self.cc[2] > 0 else -1.0

    def to_lab(self, u, v, h):
        p = self.Hinv @ np.array([u, v, 1.0])
        raw = p[:2] / p[2]                                     # where it is in the competition camera's raw picture
        X = self._ray_plane(self.Rc, self.tc, self.cc, raw, h)  # real 3D top of the block (plane coords, cm)
        q = K @ (self.Rl @ X + self.tl)
        return float(q[0] / q[2]), float(q[1] / q[2])

    def size_cm(self, u, v, px):
        """A length of `px` pixels seen around (u, v) in the lab-view picture, in cm on the table."""
        a = self.to_table_lab(u - px / 2, v); b = self.to_table_lab(u + px / 2, v)
        return float(np.linalg.norm(a - b))

    def to_table_lab(self, u, v):
        d = self.Rl.T @ (self.Ki @ np.array([u, v, 1.0]))
        s = (0.0 - self.cl[2]) / d[2]
        return (self.cl + s * d)[:2]


if __name__ == "__main__":
    main()
