"""Top-camera vision for the classical (pre-programmed) colour sorter. Separate from the competition kit.

detect(img_bgr) -> dict with
    table:   convex hull of the white table (only things inside it count)
    bins:    {colour: (cx, cy, w, h, angle)}  large coloured blobs (biggest one per colour)
    objects: [(colour, cx, cy, angle_deg, area_px, w, h)]  small coloured blobs on the table, not inside a bin
Colours: red, yellow, green, blue. All in pixels of the 640x480 top image.
python classical\\vision.py <image.png> [...]   -> writes <image>_vision.png with everything drawn on it
"""
import sys
from pathlib import Path

import cv2
import numpy as np

# HSV ranges (OpenCV: H 0-179). Tuned on setup_ref\top.png and colour v2 demo frames (25 Nov).
COLOURS = {
    "red":    [((0, 110, 20), (8, 255, 255)), ((170, 110, 20), (179, 255, 255))],
    "yellow": [((18, 110, 110), (35, 255, 255))],
    "green":  [((40, 80, 40), (85, 255, 255))],
    "blue":   [((95, 110, 40), (130, 255, 255))],
}
BIN_MIN_AREA = 1500       # px: a bin is a big blob (live 25 Nov: 2000-5600)
OBJ_MIN_AREA = 80         # px: smaller blobs are noise (26 Nov: 120 -> 80, the small red cube is ~116)
OBJ_MAX_AREA = 1400       # px: small objects ~200-400, large ones up to ~1000
DRAW = {"red": (0, 0, 255), "yellow": (0, 220, 255), "green": (0, 200, 0), "blue": (255, 80, 0)}
# Where loose objects may be (inside the black-tape source area, pixels of the 640x480 top image; the top camera has
# not moved more than 1 px since the colour v2 demos). The lower-left corner is left out: the left arm's base and its
# red/black cables sit there. Bins are found anywhere on the table.
# 26 Nov 01:00: the cut-out for the left arm's base was too big (y > 350): it hid a red dodecahedron and two green
# blocks right in front of the left arm. Now only below y 378 (the base + its red cables start at ~y 385).
# 26 Nov pm: the INNER EDGE OF THE BLACK LINE (the marked source area), measured from the top camera (left x 196,
# right 524 at the top -> 508 at the bottom, bottom y 384 on the left -> 398 on the right). sort.py only takes blocks
# whose centre is SORT_LINE_IN px (10) inside it. Old polygon: (205,215),(518,222),(514,400),(300,400),(300,378),(205,378)
SOURCE_POLY = np.array([(197, 212), (524, 216), (519, 300), (513, 360), (508, 398), (430, 398), (350, 394), (310, 391),
                        (250, 386), (197, 384)], np.int32)


TABLE_V = 165   # lab table brightness threshold (25 Nov)


def table_hull(hsv):
    # 27 Nov (competition): their table is greyer/darker (median V ~148 inside the line vs well over 165 in the lab) -
    # the threshold now follows the table's own brightness, measured inside the black line
    src = np.zeros(hsv.shape[:2], np.uint8)
    cv2.fillPoly(src, [SOURCE_POLY], 255)
    src = cv2.erode(src, np.ones((25, 25), np.uint8))
    vals = hsv[..., 2][src > 0]
    vthr = TABLE_V if vals.size == 0 else int(min(TABLE_V, max(90, np.median(vals) - 35)))
    white = cv2.inRange(hsv, (0, 0, vthr), (179, 60, 255))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(white)
    if n < 2:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    pts = cv2.findNonZero((lab == big).astype(np.uint8))
    return cv2.convexHull(pts)


def colour_mask(hsv, name):
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in COLOURS[name]:
        m |= cv2.inRange(hsv, lo, hi)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def true_centre(contour, shape, up=4):
    """TRUE centre of a shape (26 Nov): the centre of mass of its whole FILLED outline (holes and shading inside
    filled), computed on a 4x finer grid so the outline's partial pixels count - sub-pixel, any shape (dodecahedron,
    pentagon, cylinder, cube). Replaces the centre of the rotated box around it, which is off for non-symmetric
    outlines. Returns (x, y) in the picture's pixels."""
    x, y, w, h = cv2.boundingRect(contour)
    pad = 2
    m = np.zeros(((h + 2 * pad) * up, (w + 2 * pad) * up), np.uint8)
    c = ((contour.reshape(-1, 2) - [x - pad, y - pad]) * up + up / 2).astype(np.int32)
    cv2.fillPoly(m, [c], 255, lineType=cv2.LINE_AA)               # anti-aliased: edge pixels count partly
    M = cv2.moments(m, binaryImage=False)
    if M["m00"] <= 0:
        M = cv2.moments(contour)
        return M["m10"] / max(M["m00"], 1e-9), M["m01"] / max(M["m00"], 1e-9)
    # fine-grid pixel j covers original x in [j/up - 1/(2up)...]: undo the +up/2 shift used when scaling the outline
    return (M["m10"] / M["m00"] - up / 2) / up + x - pad, (M["m01"] / M["m00"] - up / 2) / up + y - pad


def detect_avg(frames):
    """detect() on several frames; each object's centre = the average over the frames it was found in (removes the
    pixel flicker of a single picture). Objects are matched across frames by colour + distance (< 8 px)."""
    dets = [detect(f) for f in frames]
    base = dets[-1]
    objs = []
    for o in base["objects"]:
        xs, ys = [o[1]], [o[2]]
        for d in dets[:-1]:
            best = min((q for q in d["objects"] if q[0] == o[0]), key=lambda q: np.hypot(q[1] - o[1], q[2] - o[2]),
                       default=None)
            if best is not None and np.hypot(best[1] - o[1], best[2] - o[2]) < 8:
                xs.append(best[1])
                ys.append(best[2])
        objs.append((o[0], float(np.mean(xs)), float(np.mean(ys))) + tuple(o[3:]))
    base["objects"] = objs
    return base



ROUND_RATIO = 1.15    # widest / narrowest width below this = round-ish top (dodecahedron, standing cylinder / prism)
LONG_RATIO = 1.5      # long side / narrow side at least this = long block (lying cylinder / prism): jaws go across


def grasp_geometry(contour):
    """27 Nov (competition): the jaw direction from the block's real outline, not a bounding box.
    Width of the (convex) outline in every direction 0-179 deg (like calipers):
      * long block  (lying cylinder / prism): returns the LONG axis angle, size (long, narrow) -> sort.py closes the
        jaws ACROSS it (period 180)
      * square-ish (cube): returns the narrowest = face-to-face direction, size (other, narrow) -> jaws flat on 2 faces
      * round-ish top (dodecahedron, standing cylinder / prism): returns angle 0 = do NOT turn the wrist
    Returns (angle deg, w, h) in the same convention as cv2.minAreaRect (w along `angle`)."""
    pts = cv2.convexHull(contour).reshape(-1, 2).astype(np.float64)
    th = np.radians(np.arange(0, 180, 1.0))
    proj = pts @ np.stack([np.cos(th), np.sin(th)])            # (n, 180)
    width = proj.max(0) - proj.min(0)
    i = int(width.argmin())
    w_min, w_max = float(width[i]), float(width.max())
    w_perp = float(width[(i + 90) % 180])
    if w_min <= 0:
        return 0.0, 1.0, 1.0
    if w_perp / w_min >= LONG_RATIO:
        return float((i + 90) % 180), w_perp, w_min           # long axis, (long, narrow)
    if w_max / w_min < ROUND_RATIO:
        m = (w_min + w_max) / 2
        return 0.0, m, m                                        # round-ish: no wrist turn needed
    return float(i), w_perp, w_min                              # square-ish: jaws close along the narrowest direction

def detect(img):
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    hull = table_hull(hsv)
    inside = np.zeros(img.shape[:2], np.uint8)
    if hull is not None:
        cv2.fillPoly(inside, [hull], 255)
        inside = cv2.erode(inside, np.ones((9, 9), np.uint8))
    # 27 Nov (competition): the area inside the black line (+ 3 cm) ALWAYS counts as table, whatever the table colour
    _src = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(_src, [SOURCE_POLY], 255)
    _near_line = cv2.dilate(_src, np.ones((31, 31), np.uint8))
    inside |= _near_line
    bins, objects = {}, []
    bin_masks = np.zeros(img.shape[:2], np.uint8)
    for name in COLOURS:
        m = colour_mask(hsv, name) & inside
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = [c for c in cnts if cv2.contourArea(c) >= BIN_MIN_AREA]
        # 27 Nov (competition): a bin must stand inside / right at the black line - not a blue bag next to the table
        big = [c for c in big if _near_line[min(479, max(0, int(cv2.minAreaRect(c)[0][1]))),
                                            min(639, max(0, int(cv2.minAreaRect(c)[0][0])))] > 0]
        if big:
            c = max(big, key=cv2.contourArea)
            (cx, cy), (w, h), ang = cv2.minAreaRect(c)
            bins[name] = (cx, cy, w, h, ang)
            cv2.drawContours(bin_masks, [cv2.convexHull(c)], -1, 255, -1)
    bin_zone = cv2.dilate(bin_masks, np.ones((15, 15), np.uint8))
    source = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(source, [SOURCE_POLY], 255)
    for name in COLOURS:
        # 26 Nov: the source area only decides WHICH blocks count (by their centre) - it no longer CUTS a block that
        # sits on its border (a sliced outline = a shifted centre; the red dodecahedron at x~210 was cut in half)
        m = colour_mask(hsv, name) & inside & ~bin_zone
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if OBJ_MIN_AREA <= a <= OBJ_MAX_AREA:
                mm = cv2.moments(c)
                if mm["m00"] <= 0 or source[int(mm["m01"] / mm["m00"]), int(mm["m10"] / mm["m00"])] == 0:
                    continue
                ang, w, h = grasp_geometry(c)                      # 27 Nov: shape-aware jaw direction + size
                cx, cy = true_centre(c, img.shape)                  # 26 Nov: TRUE centre, not the box centre
                objects.append((name, cx, cy, ang, a, w, h))
    return {"table": hull, "bins": bins, "objects": objects}


def draw(img, det):
    out = img.copy()
    if det["table"] is not None:
        cv2.polylines(out, [det["table"]], True, (255, 255, 255), 1)
    cv2.polylines(out, [SOURCE_POLY], True, (255, 0, 255), 1)
    for name, (cx, cy, w, h, ang) in det["bins"].items():
        box = cv2.boxPoints(((cx, cy), (w, h), ang)).astype(int)
        cv2.polylines(out, [box], True, DRAW[name], 2)
        cv2.putText(out, f"BIN {name}", (int(cx) - 30, int(cy) - int(h / 2) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, DRAW[name], 1)
    for name, cx, cy, ang, a, w, h in det["objects"]:
        box = cv2.boxPoints(((cx, cy), (w, h), ang)).astype(int)
        cv2.polylines(out, [box], True, (0, 0, 0), 2)
        cv2.circle(out, (int(cx), int(cy)), 3, (0, 0, 0), -1)
        cv2.putText(out, f"{name} {a:.0f}px {ang:.0f}d", (int(cx) + 8, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return out


def find_blob(img, colour, near=None, min_frac=0.002, max_frac=0.45):
    """WRIST camera: the object of `colour` in the picture. Returns (x, y, area_fraction) with x, y as FRACTIONS of
    the picture width/height (so the 424x240 RealSense and the 640x360 webcam use the same numbers), or None.
    Saturation threshold is relaxed (the right wrist webcam is washed out). `near` = (x, y) fractions: pick the blob
    closest to it (the object we are servoing on), else the biggest."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    m = np.zeros((h, w), np.uint8)
    for lo, hi in COLOURS[colour]:
        lo = (lo[0], min(lo[1], 70), min(lo[2], 40))
        m |= cv2.inRange(hsv, lo, hi)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
    best = None
    for i in range(1, n):
        frac = stats[i, cv2.CC_STAT_AREA] / (w * h)
        if not (min_frac <= frac <= max_frac):
            continue
        x, y = cent[i][0] / w, cent[i][1] / h
        key = -frac if near is None else (x - near[0]) ** 2 + (y - near[1]) ** 2
        if best is None or key < best[0]:
            best = (key, (x, y, frac))
    return None if best is None else best[1]


def find_block_edge(img, colour, min_frac=0.002, max_frac=0.45):
    """RIGHT wrist camera (25 Nov): the block is often cut off at the LEFT edge of the picture (the moving jaw side),
    so its centre is unreliable - use its RIGHT edge (next to the fixed jaw, always fully in view) and its vertical
    centre. Returns (right_edge_x, centre_y, area_fraction) as picture fractions, or None. The biggest blob counts."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    m = np.zeros((h, w), np.uint8)
    for lo, hi in COLOURS[colour]:
        m |= cv2.inRange(hsv, (lo[0], min(lo[1], 70), min(lo[2], 40)), hi)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
    best = None
    for i in range(1, n):
        frac = stats[i, cv2.CC_STAT_AREA] / (w * h)
        if min_frac <= frac <= max_frac and (best is None or frac > best[2]):
            x, y, bw, bh = stats[i, :4]
            best = ((x + bw) / w, cent[i][1] / h, frac)
    return best


def find_jaws(img, max_v=70, min_frac=0.004):
    """WRIST camera: the two dark gripper jaws coming in from the bottom of the picture (gripper OPEN).
    Returns ((x_left_tip, y_left_tip), (x_right_tip, y_right_tip)) as picture fractions, or None.
    Each jaw = a dark blob touching the bottom edge; its tip = its highest point."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    dark = ((hsv[..., 2] < max_v) & (hsv[..., 1] < 120)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(dark)
    jaws = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_frac * w * h or y + bh < h - 3:      # must touch the bottom edge
            continue
        ys, xs = np.nonzero(lab == i)
        top = ys.min()
        tip_x = xs[ys <= top + 0.03 * h].mean()
        jaws.append((area, tip_x / w, top / h))
    if len(jaws) < 2:
        return None
    jaws = sorted(sorted(jaws, reverse=True)[:2], key=lambda j: j[1])   # two biggest, left to right
    return (jaws[0][1], jaws[0][2]), (jaws[1][1], jaws[1][2])


if __name__ == "__main__":
    for p in sys.argv[1:]:
        img = cv2.imread(p)
        det = detect(img)
        print(Path(p).name, "| bins:", {k: (round(v[0]), round(v[1])) for k, v in det["bins"].items()},
              "| objects:", [(o[0], round(o[1]), round(o[2]), round(o[4])) for o in det["objects"]])
        cv2.imwrite(str(Path(p).with_name(Path(p).stem + "_vision.png")), draw(img, det))

def find_bins_by_rim(img, vmax=345):
    """26 Nov: bins FULL of mixed blocks (shape sort) show their colour only as a ring (the rim). Closed colour mask ->
    convex hull -> the outer band of the hull must be mostly that colour. {colour: (cx, cy, w, h, angle, rim)}."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    top = np.zeros(img.shape[:2], np.uint8)
    top[150:vmax, 150:580] = 255
    found = {}
    for name in COLOURS:
        m = colour_mask(hsv, name) & top
        mc = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        cnts, _ = cv2.findContours(mc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in cnts:
            hull = cv2.convexHull(c)
            ha = cv2.contourArea(hull)
            (cx, cy), (bw, bh), ang = cv2.minAreaRect(hull)
            if ha < 900 or not (28 <= min(bw, bh) and max(bw, bh) <= 95):
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
