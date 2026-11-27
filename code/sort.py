"""Colour sorter: classical top-camera perception + the LEARNED grasp-pose network GraspNet (model.py, trained on the
teleoperated demos by train_grasp_net.py; 26 Nov) for the arm poses. See TECHNICAL_DOCUMENTATION.md. Run with SORT.bat.
    python classical\\sort.py [seconds] [--dry]

Loop: top camera -> find loose objects + bins (vision.py) -> pick the object the calibrated arms can reach best ->
that arm: rest -> above the object (gripper open) -> FINE ALIGNMENT with its wrist camera (two small test nudges,
then up to 3 corrections until the object sits where it sat during calibration; SORT_ALIGN=0 switches it off) ->
down slowly -> close + squeeze -> up -> above the bin of the same colour -> open -> back to rest. One arm moves at a time (no collisions). Checks with the camera that the object
really left its spot; a missed block goes to the back of the queue (retried only when it is the last one), 2 misses = skipped.
--dry: no motors and no camera: uses classical\\test_images\\live_top.png and prints the plan.
Speeds: SORT_SPEED (45 deg/s travel); descent and lift at 40 % of that. Ctrl+C = stop (arms go back to rest).
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import model  # noqa: E402
import vision  # noqa: E402

DRY = "--dry" in sys.argv
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
DURATION = float(ARGS[0]) if ARGS else 300.0


_WARNED = set()


def choose(det, maps):
    """Best (object, arm side) pair: an object with a matching bin, inside an arm's calibrated area."""
    best = None
    for o in det["objects"]:
        colour, u, v = o[0], o[1], o[2]
        if colour not in det["bins"]:
            continue
        if near_bin(u, v, det, margin=BIN_CLEAR):
            if (round(u), round(v)) not in _WARNED:
                _WARNED.add((round(u), round(v)))
                print(f"   {colour} block at ({u:.0f},{v:.0f}) is ON a bin - skipped (inside the bin outline).")
            continue
        for side, m in maps.items():
            if m is None or not m.can_reach(u, v):
                continue
            bu, bv = det["bins"][colour][:2]
            if m.bin_hover(colour, bu, bv) is None:
                continue
            score = m.reach(u, v)                     # deeper inside the calibrated area = more reliable
            if best is None or score > best[0]:
                best = (score, o, side)
    return best



def choose_next(det, maps, deferred):
    """26 Nov: after a MISS go to the NEXT block, not the same one again. A missed block is only retried when it is
    the last one left. deferred = {(u, v): misses}."""
    def was_missed(o):
        return any(math.hypot(o[1] - d[0], o[2] - d[1]) < 15 for d in deferred)
    fresh = dict(det, objects=[o for o in det["objects"] if not was_missed(o)])
    c = choose(fresh, maps)
    if c is None:
        c = choose(det, maps)
        if c is not None:
            print(f"   only already-missed blocks left - retrying the {c[1][0]} one")
    return c


def note_miss(deferred, skipped, orig):
    """Count a miss for this block; after 2 misses it is skipped for good."""
    key = next((d for d in deferred if math.hypot(orig[1] - d[0], orig[2] - d[1]) < 15), (orig[1], orig[2]))
    deferred[key] = deferred.get(key, 0) + 1
    if deferred[key] >= 2:
        skipped.add((orig[1], orig[2]))
        print("   missed this block twice - skipping it")
    else:
        print("   -> going to the NEXT block first (this one is retried only if it is the last)")

UP_K = float(os.environ.get("SORT_UP") or 2.0)    # start every grab this many "lift heights" above the object
DEEP = float(os.environ.get("SORT_DEEP") or 0.2)  # 26 Nov pm: grab this many lift heights BELOW the demo grab
#   height (~1 cm; one lift height ~5 cm). Deeper grip = pentagons / prisms / cylinders sit deeper in the jaws.
#   SORT_DEEP=0.3 = deeper still; SORT_DEEP=0 = the old height.
PLACE_CLEAR = float(os.environ.get("SORT_PLACE_CLEAR") or 0.15)   # hand-off: let go this far above the table
BIN_GEO = {x.strip() for x in (os.environ.get("SORT_BIN_GEO") or "left:green").split(",") if x.strip()}
#   26 Nov pm: bins (side:colour) whose drop turns to the bin's real centre (geometric pan) instead of the demo median.
#   left green: demo median 9.9 deg, bin centre 13.7 deg - Peter saw the left arm drop LEFT of the green bin.
DROP_UP_BIN = {"left:green": 0.5, "left:red": 0.7}   # 26 Nov pm: deeper drops for the left arm, but (Peter: the arms
#   must NEVER touch the bins) not below 0.5 lift heights above the demo let-go pose (which was at the rim). The
#   sag correction now stops the drops landing short, which is what "deeper" was for.
BIN_APPROACH = float(os.environ.get("SORT_BIN_APPROACH") or 1.5)   # every drop comes in this high above the bin
#   (lift heights over the demo let-go pose), then goes straight down to its drop height - clear of the rims   # 26 Nov pm (Peter: go deeper into the green bin): lower drop = also less pulled back
#   towards the arm. Override per bin: SORT_DROP_UP_LEFT_GREEN=0.3 etc.


DROP_PAN_BIN = {"left:green": 5.0}   # deg extra turn at the drop, + = right in the top camera (both arms: pan grows
#   with x). 26 Nov pm: left green landed left of the bin at 9.9 and still at 13.7 -> 18.7. Override per bin:
#   SORT_DROP_PAN_LEFT_GREEN=7 (more right) / 3 (less).


def drop_pan_for(side, colour):
    e = os.environ.get(f"SORT_DROP_PAN_{side.upper()}_{colour.upper()}")
    if e:
        return float(e)
    return DROP_PAN_BIN.get(f"{side}:{colour}", 0.0)


DROP_MIN = float(os.environ.get("SORT_DROP_MIN") or 0.5)   # never open lower than this (lift heights above let-go)


def drop_up_for(side, colour):
    """Lift heights above the demo let-go pose at which this arm opens over this bin."""
    e = os.environ.get(f"SORT_DROP_UP_{side.upper()}_{colour.upper()}")
    k = float(e) if e else DROP_UP_BIN.get(f"{side}:{colour}", DROP_UP)
    return max(k, DROP_MIN)
DROP_UP = float(os.environ.get("SORT_DROP_UP") or 1.0)   # 25 Nov: drop this many lift heights ABOVE the recorded bin
#   pose (the demo let-go was low, at the rim: the gripper hit the edge; 2.5 keeps the WRIST CAMERAS clear too -
#   they stick out above / beside the jaws). The block is released from up there and falls in.
BIN_CLEAR = float(os.environ.get("SORT_BIN_CLEAR") or 20)   # 26 Nov 16:10: 30 -> 15 px from the bin's real outline
#   (30 made the arms idle - most "too close" blocks were 2-3 cm from a bin). Blocks between BIN_CLEAR and BIN_LAST
#   px are still taken, but after everything else.
BIN_LAST = float(os.environ.get("SORT_BIN_LAST") or 30)
_BIN_CLEAR_OLD = float(os.environ.get("SORT_BIN_CLEAR") or 30)   # px: blocks closer than this to a bin's edge are left alone
#   (26 Nov pm: 30 px ~3.6 cm - the open jaws hit the yellow bin going for blocks right beside it; was 12, 0 in the bats)
#   (25 Nov: was 40; the cameras are now kept off the bins by turning the wrist instead, see DROP_ROLL)
DROP_ROLL = {"left": float(os.environ.get("SORT_DROP_ROLL_LEFT") or 0.0),
             "right": float(os.environ.get("SORT_DROP_ROLL_RIGHT") or 0.0)}   # 25 Nov late: wrist turn UNDONE (user: it made grabs worse)
#   25 Nov: while carrying to the bin and dropping, the wrist turns this much (deg) so the wrist camera points
#   SIDEWAYS ("east"), not at the bin - it can never hit the bin. Capped at +-170 (the roll joint wraps at 180).
#   If an arm turns its camera the wrong way, set e.g. SORT_DROP_ROLL_RIGHT=-90.
PICK_ROLL = {"left": float(os.environ.get("SORT_PICK_ROLL_LEFT") or DROP_ROLL["left"]),
             "right": float(os.environ.get("SORT_PICK_ROLL_RIGHT") or DROP_ROLL["right"])}
#   25 Nov: GRABS too are done with the wrist turned this much (camera sideways for every block). 0 = off.


def above_pose(m, g, h, frac):
    """frac = 0 -> the grab pose itself, 1 -> one lift height above it (grab + frac x up).
    Hand calibration at exactly frac 1: its own recorded hover pose (where its wrist target was taken)."""
    if not m.from_demos and abs(frac - 1.0) < 1e-6:
        return h
    if m.up is not None:
        return g + frac * m.up
    return h if frac >= 1 else g + frac * (h - g)


LONG_ASPECT = float(os.environ.get("SORT_LONG_ASPECT") or 1.5)   # outline this stretched = long block (lying cylinder)
LONG_TURN = float(os.environ.get("SORT_LONG_TURN") or 0.0)       # 0 or 90: flips which way the jaws close on long
#   blocks. 26 Nov: the demos did not settle it (28 of 51 long-block grabs favour 0) - if the jaws close ALONG a lying
#   cylinder instead of across it, set SORT_LONG_TURN=90


def shape_of(o):
    """Rough shape from the top-camera outline: 'long' (lying cylinder, jaws must go across) or 'block'."""
    if len(o) >= 7 and min(o[5], o[6]) > 0 and max(o[5], o[6]) / min(o[5], o[6]) >= LONG_ASPECT:
        return "long"
    return "block"


def plan(o, m, bin_xy):
    colour, u, v, ang = o[0], o[1], o[2], o[3]
    if shape_of(o) == "long":
        long_axis = ang if o[5] >= o[6] else ang + 90          # minAreaRect: width side lies along `ang`
        g, h = m.poses(u, v, long_axis + 90 - LONG_TURN, period=180.0)   # jaws ACROSS the long axis
    else:
        g, h = m.poses(u, v, ang)
    # Round 3: a bin standing on one of our known spots uses that spot's tested pose; anywhere else the learned drop
    # model (drop_model.py) gives the pose for wherever the camera sees the bin
    # 27 Nov (competition): BIN TRACKING in every round - the drop follows where the camera sees the bin NOW, aimed at
    # the BACK-MIDDLE of the bin (BIN_BACK px past its centre, away from this arm: drops landed short). On a known spot:
    # that spot's tested pose, moved by the drop model's change between the spot and the bin's real place.
    track = DYNAMIC or BIN_TRACK
    dm = drop_model(m.side) if track else None
    on_spot = hasattr(m, "bin_slot") and m.bin_slot(colour, *bin_xy) is not None
    target = tuple(bin_xy[:2])
    if track and BIN_BACK:
        bx, by = ARM_BASE_PX[m.side]
        dd = math.hypot(bin_xy[0] - bx, bin_xy[1] - by)
        if dd > 1:
            target = (bin_xy[0] + BIN_BACK * (bin_xy[0] - bx) / dd, bin_xy[1] + BIN_BACK * (bin_xy[1] - by) / dd)
    if dm is not None:
        # the drop model's lift is a quadratic in the distance from the base: beyond its peak (few demo drops that far
        # out) it turns back down - never aim further out than the peak
        W = np.asarray(dm.d["W"], float)
        if W[2][0] < 0:
            r_peak = dm.d["r0"] + 100.0 * (-W[1][0] / (2 * W[2][0])) - 3
            bx, by = dm.d["bu"], dm.d["bv"]
            rt = math.hypot(target[0] - bx, target[1] - by)
            if rt > r_peak > 0:
                target = (bx + (target[0] - bx) * r_peak / rt, by + (target[1] - by) * r_peak / rt)
    pan_done = False
    if dm is not None and on_spot:
        bh = m.bin_hover(colour, *bin_xy)
        sp = m.bins.get(m.bin_slot(colour, *bin_xy), {})
        if bh is not None and "u" in sp and dm.covers(*target) and dm.covers(sp["u"], sp["v"]):
            bh = np.asarray(bh, float) + (dm.pose(*target) - dm.pose(sp["u"], sp["v"]))
    elif dm is not None and dm.covers(*target):
        bh = dm.pose(*target)
    else:
        bh = m.bin_hover(colour, *bin_xy)
        pan_done = bh is not None and not on_spot         # interpolated by m.poses(): turn correction already in
    rest = m.rest[:5]
    high = above_pose(m, g, h, UP_K)               # HIGH start: well above the object, then straight down
    # 26 Nov pm: per-bin settings belong to the bin SPOT (model.bin_slot), so swapped bin colours keep the drop height,
    # extra turn and sag correction learned for that place
    slot = (m.bin_slot(colour, *bin_xy) if hasattr(m, "bin_slot") else None) or colour
    taught = bool(m.bins.get(slot, {}).get("taught"))     # 26 Nov: hand-taught bin pose (teach_bin.py) = used exactly
    k_drop = drop_up_for(m.side, slot)
    drop = bh + k_drop * m.up if (bh is not None and m.up is not None and not taught) else bh   # HIGH drop: clear of the bin rim
    bin_high = bh + max(k_drop, BIN_APPROACH) * m.up if (bh is not None and m.up is not None and not taught) else drop
    if taught and bh is not None and m.up is not None:
        # 26 Nov 19:45: a hand-taught (e.g. fully stretched) drop pose is approached from above, never sideways
        bin_high = np.array(bh, float) + TAUGHT_APPROACH * m.up
    if drop is not None and not taught and f"{m.side}:{slot}" in BIN_GEO and getattr(m, "pan_geo", None):
        # 26 Nov pm: turn to the bin's REAL centre (geometric pan at the bin seen by the camera), not the demo median
        drop = np.array(drop, float)
        bin_high = np.array(bin_high, float)
        drop[0] = bin_high[0] = m.pan_geo(*target)
    pan_adj = drop_pan_for(m.side, slot)
    if drop is not None and not taught and pan_adj:
        # 26 Nov pm (Peter: left green still lands LEFT of the bin at the bin-centre turn - the block sits against
        # one jaw, off the gripper's centre line): extra turn, + = to the right in the top camera
        drop = np.array(drop, float)
        bin_high = np.array(bin_high, float)
        drop[0] += pan_adj
        bin_high[0] += pan_adj
    if drop is not None:
        drop = np.array(drop, float)
        r = DROP_ROLL.get(m.side, 90.0)
        if abs(drop[4] + r) > 170:
            r = r - 180 * np.sign(r)        # other way round (joint limit), camera still sideways
        drop[4] = float(drop[4] + r)                  # camera sideways, not at the bin
    # 26 Nov: bin_low = EXACTLY the pose the human let go at in the demos (or the hand-taught one). The high drop
    # (+ DROP_UP x lift) also pulls the gripper ~3-5 cm back towards the arm - every drop landed short. So: come in
    # high (clears the rim), go down to bin_low, open there.
    bin_low = None if bh is None else np.array(bh, float)
    if bin_low is not None and drop is not None:
        bin_low[4] = drop[4]
    # 26 Nov pm (Peter: pentagons were gripped too shallow): EVERY grab goes DEEP lift heights below the demo grab
    # height (straight down: "up" is the vertical lift of the demos). The way down stops early if the table is in
    # the way (stall check), so it can't push into the table for long.
    g_deep = g - DEEP * m.up if (m.up is not None and DEEP) else g
    if bin_high is not None and drop is not None:
        bin_high = np.array(bin_high, float)
        bin_high[4] = drop[4]
    # 27 Nov (competition): the arm base is turned vs the lab - the same turn correction as the grabs (SORT_PAN_<side>)
    po = 0.0 if pan_done else pan_offset(m.side)
    if po:
        dpo = np.array([po, 0, 0, 0, 0], float)
        drop = None if drop is None else np.asarray(drop, float) + dpo
        bin_low = None if bin_low is None else np.asarray(bin_low, float) + dpo
        bin_high = None if bin_high is None else np.asarray(bin_high, float) + dpo
    return dict(hover=high, grasp=g_deep, grasp_demo=g, bin=drop, bin_low=bin_low, bin_high=bin_high, slot=slot,
                mid1=(high + rest) / 2, mid2=None if drop is None else (drop + rest) / 2, rest=rest,
                low=above_pose(m, g, h, 0.25))     # just above the object (last check before closing)


def plan_place(m, u, v, held_grasp):
    """26 Nov pm, HAND-OFF: put the held block down at pixel (u, v) for the other arm. Same wrist turn as it was picked
    with (no turning with a block in the jaws); let go PLACE_CLEAR lift heights above where it was gripped."""
    g, h = m.poses(u, v, 0.0)
    g[4] = held_grasp[4]
    h[4] = held_grasp[4]
    high = above_pose(m, g, h, UP_K)
    place = g - DEEP * m.up + PLACE_CLEAR * m.up if m.up is not None else g
    return dict(hover=high, place=place)


def with_grip(q5, grip):
    return list(q5) + [grip]


# ---------- wrist-camera fine alignment (25 Nov) ----------------------------------------------------------------------
ALIGN_TOL = 0.025       # wrist-picture fraction (~1 cm at hover height): close enough, grab
ALIGN_PROBE = 10.0      # top-camera px: size of the two test nudges that measure which way the wrist picture moves
ALIGN_STEP = 30.0       # px: biggest single correction
ALIGN_TOTAL = 45.0      # px: never move further than this from where the top camera said (~5 cm)
ALIGN_ITERS = 3
MODE = os.environ.get("SORT_MODE", "fast")     # fast = one look + one correction if far off; careful = two-stage centring
FAR_OFF = float(os.environ.get("SORT_FAR_OFF") or 0.06)   # wrist-picture fraction: only then correct (25 Nov late: 0.10 -> 0.06)
FAR_OFF_EDGE = float(os.environ.get("SORT_FAR_OFF_EDGE") or 0.05)   # right arm (block edge vs fixed jaw, very consistent)
TRANSFER_SPEED = float(os.environ.get("SORT_TRANSFER_SPEED") or 60.0)   # deg/s: block -> bin -> home (all high)
APPROACH_CM = float(os.environ.get("SORT_APPROACH_OFFSET") or 2.0)   # cm: the look + the way down are this far
#   towards the STATIONARY jaw side; at grab height the gripper slides back onto the exact spot (25 Nov late).
#   (The earlier version shifted the GRAB itself 2 cm - wrong, it missed; removed.)
CLOSE_TOL = float(os.environ.get("SORT_CLOSE_TOL") or 0.12)   # before closing: block within this of the camera target
CAMERA_BLIND = {"right": {"yellow"}}   # 25 Nov: the right wrist webcam is so overexposed that yellow on white is
#   invisible -> for these the jaw check (stopped part-way = holding) decides instead of the camera
HOLD_MIN = float(os.environ.get("SORT_HOLD_MIN") or 7.0)   # gripper reading above this after squeezing = holding
#   (25 Nov demos: jaws on an object read ~20-27, closed on nothing ~1-4)
CENTRE_OK = 0.04       # wrist-picture fraction: worse than this after both stages = not centred (retry, then skip)


def align(uv0, measure, target, J=None, log=print, tol=None, step=None, total=None, iters=None, probe=None):
    """Steer the hover point (top-camera pixels) until the object sits at `target` in the wrist camera.
    measure(uv) -> moves to hover above uv, returns the wrist blob (x, y) in picture fractions, or None.
    Returns (best uv, J used, final error) - J (2x2: wrist fraction per top pixel) can be reused for the next object."""
    tol = ALIGN_TOL if tol is None else tol
    step = ALIGN_STEP if step is None else step
    total = ALIGN_TOTAL if total is None else total
    iters = ALIGN_ITERS if iters is None else iters
    probe = ALIGN_PROBE if probe is None else probe
    uv0 = np.asarray(uv0, float)
    t = np.asarray(target, float)
    b0 = measure(uv0)
    if b0 is None:
        log("   align: object not seen by the wrist camera - grabbing where the top camera says")
        return uv0, J, None
    e0 = np.asarray(b0[:2]) - t
    best = (np.linalg.norm(e0), uv0)
    if best[0] <= tol:
        log(f"   align: already centred ({best[0]:.3f})")
        return uv0, J, best[0]
    if J is None:                                  # learn how the wrist picture moves for a nudge in u and in v
        b1 = measure(uv0 + [probe, 0])
        b2 = measure(uv0 + [0, probe])
        if b1 is None or b2 is None:
            log("   align: lost the object while probing - grabbing where the top camera says")
            return uv0, None, best[0]
        J = np.column_stack([(np.asarray(b1[:2]) - b0[:2]) / probe, (np.asarray(b2[:2]) - b0[:2]) / probe])
        if abs(np.linalg.det(J)) < 1e-6:
            log("   align: probing gave no usable direction - grabbing where the top camera says")
            return uv0, None, best[0]
    uv, e = uv0.copy(), e0
    for _ in range(iters):
        d = -np.linalg.solve(J, e)
        n = np.linalg.norm(d)
        if n > step:
            d *= step / n
        nxt = uv + d
        off = nxt - uv0
        if np.linalg.norm(off) > total:
            nxt = uv0 + off * total / np.linalg.norm(off)
        b = measure(nxt)
        if b is None:
            break
        uv, e = nxt, np.asarray(b[:2]) - t
        if np.linalg.norm(e) < best[0]:
            best = (np.linalg.norm(e), uv.copy())
        if best[0] <= tol:
            break
    log(f"   align: wrist error {np.linalg.norm(e0):.3f} -> {best[0]:.3f}, moved {np.linalg.norm(best[1] - uv0):.0f} px")
    if best[0] > np.linalg.norm(e0) * 0.9:       # did not help: forget J so the next object measures it afresh
        J = None
    return best[1], J, best[0]


def run_dry():
    import cv2
    img = cv2.imread(str(HERE / "test_images" / "live_top.png"))
    maps = {s: model.load(s) for s in ("left", "right")}
    for s, mm in maps.items():
        if mm is not None:
            mm.roll_offset = PICK_ROLL[s]
    if not any(maps.values()):
        print("No calibration yet (calib_left.json / calib_right.json). Run CALIBRATE.bat first.")
        return
    det = vision.detect(img)
    print("bins:", {k: (round(v[0]), round(v[1])) for k, v in det["bins"].items()})
    print("objects:", [(o[0], round(o[1]), round(o[2])) for o in det["objects"]])
    c = choose(det, maps)
    if c is None:
        print("nothing reachable to sort")
        return
    _, o, side = c
    p = plan(o, maps[side], det["bins"][o[0]][:2])
    print(f"would take the {o[0]} object at ({o[1]:.0f},{o[2]:.0f}) with the {side} arm:")
    print(f"   arm model: {maps[side].grasp_model}")
    print(f"   arm pose from: {maps[side].last_source}")
    for k in ("rest", "mid1", "hover", "grasp", "mid2", "bin"):
        print(f"   {k:6s} {np.round(p[k], 1)}")


PX_PER_CM = 8.3        # top camera, at the table: ~1.2 mm per pixel
ARM_BASE_PX = {"left": (242.0, 420.0), "right": (458.0, 430.0)}   # arm bases in the top-camera picture (25 Nov)
JAW_OFFSET = {"left": float(os.environ.get("SORT_JAW_OFFSET_LEFT") or 0.0),
              "right": float(os.environ.get("SORT_JAW_OFFSET_RIGHT") or 0.0)}   # 26 Nov 20:00: 1 cm -> 0 (see TRIM)
#   cm, towards the STATIONARY jaw (right of the base->block line). 25 Nov late: right arm +1 cm (its stationary
#   jaw landed ~1 cm inside the block). If it goes the wrong way: SORT_JAW_OFFSET_RIGHT=-1


def trim_px(side):
    """SORT_TRIM_LEFT / SORT_TRIM_RIGHT = "x,y" in cm, added to where that arm grabs (after centring):
    x + = towards the RIGHT of the top-camera picture, y + = towards the ARMS (down the picture)."""
    s = os.environ.get(f"SORT_TRIM_{side.upper()}", "")
    try:
        x, y = (float(t) for t in s.split(","))
        return np.array([x, y]) * PX_PER_CM
    except Exception:
        return np.zeros(2)


class _Tee:
    """print() goes to the window AND to classical\\sort_log.txt (so the run can be checked afterwards)."""

    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.out.flush()


CORR_FILE = HERE / "corrections.csv"


def record_correction(side, orig, final):
    """After the camera centred a block: how far (top-camera px) the true spot was from where the arm first went.
    The average of these is applied automatically next time (learned_bias) so the arm starts at better coordinates."""
    new = not CORR_FILE.exists()
    with open(CORR_FILE, "a", encoding="utf-8") as f:
        if new:
            f.write("time,side,u,v,du,dv" + chr(10))
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{side},{orig[1]:.1f},{orig[2]:.1f},"
                f"{final[1] - orig[1]:.2f},{final[2] - orig[2]:.2f}" + chr(10))


def learned_bias(side, min_n=3):
    """Median camera correction (top px) of this arm from earlier runs, or None if fewer than min_n."""
    if os.environ.get("SORT_LEARN", "1") != "1" or not CORR_FILE.exists():
        return None
    rows = [l.strip().split(",") for l in CORR_FILE.read_text(encoding="utf-8").splitlines()[1:] if l.strip()]
    d = np.array([[float(r[4]), float(r[5])] for r in rows if r[1] == side])
    if len(d) < min_n:
        return None
    return np.median(d, 0)


BIN_MAX_SIDE = float(os.environ.get("SORT_BIN_MAX_SIDE") or 66)   # px: a real bin is never longer than this


def near_bin(u, v, det, margin=15):
    """Is top-camera pixel (u, v) on, or within `margin` px of, any bin's outline? 26 Nov pm: measured to the bin's
    real (rotated) rectangle - the old circle (half the longer side) let the corners of a turned bin stick out, and
    the open jaws hit them (Peter: the arms must NEVER touch the bins)."""
    import cv2
    for b in det["bins"].values():
        cx, cy, w, h = b[:4]
        w, h = min(w, BIN_MAX_SIDE), min(h, BIN_MAX_SIDE)     # a merged (bin + blocks) outline is cut to a real bin
        ang = b[4] if len(b) > 4 else 0.0
        box = cv2.boxPoints(((float(cx), float(cy)), (float(w), float(h)), float(ang))).astype(np.float32)
        if cv2.pointPolygonTest(box, (float(u), float(v)), True) > -margin:
            return True
    return False


def inside_line(u, v):
    """px the point lies INSIDE the black line of the marked area (negative = outside). vision.SOURCE_POLY = the
    line's inner edge, measured from the top camera on 26 Nov."""
    import cv2
    return cv2.pointPolygonTest(vision.SOURCE_POLY, (float(u), float(v)), True)


def save_pic(img, name, marks=()):
    """runlog\\<time>_<name>.jpg with circles: (x, y fractions or pixels, colour BGR, label)."""
    import cv2
    d = HERE / "runlog"
    d.mkdir(exist_ok=True)
    out = img.copy()
    h, w = out.shape[:2]
    for x, y, col, lab in marks:
        px, py = (int(x * w), int(y * h)) if x <= 1.5 and y <= 1.5 else (int(x), int(y))
        cv2.circle(out, (px, py), 9, col, 2)
        cv2.putText(out, lab, (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    cv2.imwrite(str(d / f"{time.strftime('%H%M%S')}_{name}.jpg"), out)


def run():
    import hw
    sys.stdout = _Tee(HERE / "sort_log.txt")
    print(f"\n===== SORT run {time.strftime('%Y-%m-%d %H:%M:%S')} | mode {MODE} | up x{UP_K} | drop up x{DROP_UP} | speed {hw.MAX_SPEED} | trims "
          f"left {os.environ.get('SORT_TRIM_LEFT', '-')} right {os.environ.get('SORT_TRIM_RIGHT', '-')} =====")
    maps = {s: model.load(s) for s in ("left", "right")}
    for s, mm in maps.items():
        if mm is not None:
            mm.roll_offset = PICK_ROLL[s]
            print(f"{s} arm grasp-pose model: {mm.grasp_model}")
    if not any(maps.values()):
        print("No calibration yet. Run CALIBRATE.bat left / right first.")
        return
    cam = hw.TopCamera()
    arms = {s: hw.Arm(s) for s, m in maps.items() if m is not None}
    cam.connect()
    wcams, Js = {}, {}
    for s in arms:
        if (maps[s].wrist_target is not None and os.environ.get("SORT_ALIGN", "1") == "1"
                and os.environ.get(f"SORT_ALIGN_{s.upper()}", "1") == "1"):
            try:
                wcams[s] = hw.WristCamera(s)
                wcams[s].connect()
                print(f"{s} wrist camera on: fine alignment ON (target {np.round(maps[s].wrist_target, 3)})")
            except Exception as e:
                wcams.pop(s, None)
                print(f"{s} wrist camera failed ({e!r}): fine alignment OFF for this arm")
        else:
            print(f"{s}: fine alignment OFF")
    for s, a in arms.items():
        a.connect()
        maps[s].widen_limits(UP_K, DROP_UP)
        a.limits = dict(maps[s].limits, gripper=(0.0, 80.0))
        a.limits["wrist_roll"] = (-172.0, 172.0)    # the 90-deg wrist turns (pick + drop) need the full range
    rest = {s: a.read() for s, a in arms.items()}
    print("rest poses:", {s: np.round(q[:5], 0).tolist() for s, q in rest.items()})
    sp = hw.MAX_SPEED
    done, misses, skipped = 0, 0, set()
    deferred = {}          # 26 Nov: missed blocks -> retried only when nothing else is left
    t_end = time.time() + DURATION
    try:
        while time.time() < t_end:
            # 26 Nov: TRUE centres (centre of mass of each shape's filled outline, sub-pixel), averaged over 3 pictures
            frames = [cam.grab() for _ in range(3)]
            top_img = frames[-1]
            det = vision.detect_avg(frames)
            det["objects"] = [o for o in det["objects"]
                              if not any(math.hypot(o[1] - s[0], o[2] - s[1]) < 12 for s in skipped)]
            c = choose_next(det, maps, deferred)
            if c is None:
                # 26 Nov: one empty look used to end the run (00:26 run: 0 sorted though 7 blocks were there a minute
                # later) - look again 3 times, 1 s apart, and save what the camera saw
                for _ in range(3):
                    time.sleep(1.0)
                    frames = [cam.grab() for _ in range(3)]
                    top_img = frames[-1]
                    det = vision.detect_avg(frames)
                    det["objects"] = [o for o in det["objects"]
                                      if not any(math.hypot(o[1] - s[0], o[2] - s[1]) < 12 for s in skipped)]
                    c = choose_next(det, maps, deferred)
                    if c is not None:
                        break
            if c is None:
                save_pic(top_img, "nothing_to_sort", [(o[1], o[2], (255, 0, 255), o[0]) for o in det["objects"]])
                print(f"nothing (more) to sort that the arms can reach | camera saw bins {sorted(det['bins'])}, "
                      f"objects {[(o[0], round(o[1]), round(o[2])) for o in det['objects']]} (picture in runlog)")
                break
            _, o, side = c
            arm, m = arms[side], maps[side]
            colour = o[0]
            p = plan(o, m, det["bins"][colour][:2])
            print(f"\n[{time.strftime('%H:%M:%S')}] {colour} object at ({o[1]:.0f},{o[2]:.0f}) rot {o[3]:.0f} -> {side} arm"
                  + (" | LONG block (lying cylinder): jaws across it" if shape_of(o) == "long" else ""))
            orig = o
            lb = learned_bias(side)
            if lb is not None and not near_bin(o[1] + lb[0], o[2] + lb[1], det):
                o = (o[0], o[1] + lb[0], o[2] + lb[1]) + tuple(o[3:])
                p = plan(o, m, det["bins"][colour][:2])
                print(f"   learned correction for the {side} arm applied first: {lb[0] / PX_PER_CM:+.1f} cm right, {lb[1] / PX_PER_CM:+.1f} cm towards the arms")
            if m.last_source:
                print(f"   arm pose from: {m.last_source}")
            save_pic(top_img, f"{side}_top_target", [(o[1], o[2], (255, 0, 255), f"{colour} target")])
            arm.move_to(with_grip(p["mid1"], hw.GRIP_OPEN), sp, tol=99, settle=0.0)
            arm.move_to(with_grip(p["hover"], hw.GRIP_OPEN), sp, tol=4, settle=0.1, timeout=1.5)   # HIGH above the object
            if side in wcams:
                wc = wcams[side]

                def measure_at(frac):
                    def measure(uv, _m=m, _o=o, _arm=arm, _wc=wc, _c=colour):
                        g, h = _m.poses(uv[0], uv[1], _o[3])
                        _arm.move_to(with_grip(above_pose(_m, g, h, frac), hw.GRIP_OPEN), sp * 0.5, tol=2.0,
                                     settle=0.2, timeout=1.5)
                        a = _m.wrist_area or 0.02
                        wimg = _wc.grab()
                        if _m.wrist_feature == "edge":     # right arm: block's right edge vs the fixed jaw
                            b = vision.find_block_edge(wimg, _c, min_frac=0.01, max_frac=0.6)
                        else:
                            b = vision.find_blob(wimg, _c, near=_m.wrist_target, min_frac=0.4 * a, max_frac=2.0 * a)
                        tx, ty = _m.wrist_target
                        marks = [(tx, ty, (0, 255, 0), 'TARGET')] + ([(b[0], b[1], (255, 0, 255), 'object')] if b else [])
                        save_pic(wimg, f"{side}_wrist_h{frac:.2f}_{'seen' if b else 'NOT-SEEN'}", marks)
                        if b:
                            print(f"      wrist h{frac:.2f}: object at ({b[0]:.3f},{b[1]:.3f}), target ({tx:.3f},{ty:.3f}), "
                                  f"off by ({b[0] - tx:+.3f},{b[1] - ty:+.3f})")
                        else:
                            print(f"      wrist h{frac:.2f}: object NOT seen")
                        return b
                    return measure

                if MODE == "fast":
                    # FAST: ONE wrist look just above the block, ONE correction only if far off, then grab.
                    # 25 Nov late (user): the look AND the way down are APPROACH_CM off-centre towards the STATIONARY
                    # jaw's side, so the stationary jaw comes down BESIDE the block, not on it. Only at grab height
                    # does the gripper slide those cm back onto the exact spot the camera worked out, and grab there.
                    meas = measure_at(0.25)
                    ref = m.wrist_area or 0.05
                    J = Js.get(side)
                    if J is None:                          # first block for this arm: learn which way the picture moves
                        b0, b1, b2 = meas((o[1], o[2])), meas((o[1] + 6, o[2])), meas((o[1], o[2] + 6))
                        if b0 is not None and b1 is not None and b2 is not None:
                            J = np.column_stack([(np.asarray(b1[:2]) - b0[:2]) / 6, (np.asarray(b2[:2]) - b0[:2]) / 6])
                            if abs(np.linalg.det(J)) < 1e-6:
                                J = None
                        Js[side] = J
                    sh = np.zeros(2)
                    if J is not None and APPROACH_CM > 0:
                        # towards the stationary jaw = the block appears further LEFT in the wrist picture
                        # (both wrist cameras: moving jaw = left of the picture, stationary jaw = right)
                        sh = np.linalg.solve(J, np.array([-1.0, 0.0]))
                        sh = sh / (np.linalg.norm(sh) + 1e-9) * APPROACH_CM * PX_PER_CM
                        if near_bin(o[1] + sh[0], o[2] + sh[1], det):
                            sh = np.zeros(2)
                    bl = meas((o[1] + sh[0], o[2] + sh[1]))
                    if bl is not None and not (0.25 * ref <= bl[2] <= 3.0 * ref):
                        print(f"   wrist camera: blob size {bl[2]:.3f} is not a block (expected ~{ref:.3f}) - ignoring it")
                        bl = None
                    if bl is None:
                        print("   wrist camera can't see a block here - going down to look again before closing")
                    else:
                        expected = np.asarray(m.wrist_target) + (J @ sh if J is not None else 0)   # seen from the offset
                        e = np.asarray(bl[:2]) - expected
                        en = float(np.linalg.norm(e))
                        far = FAR_OFF_EDGE if m.wrist_feature == "edge" else FAR_OFF
                        if en <= far:
                            print(f"   close enough ({en:.3f} <= {far})")
                        elif J is None:
                            print(f"   {en:.3f} off but no usable direction - using the top-camera position")
                        else:
                            d = -np.linalg.solve(J, e)
                            n = float(np.linalg.norm(d))
                            if n > 30:
                                d *= 30 / n
                            nu, nv = o[1] + float(d[0]), o[2] + float(d[1])
                            if near_bin(nu, nv, det):
                                print("   correction would go next to a BIN - ignored")
                            else:
                                o = (o[0], nu, nv) + tuple(o[3:])
                                print(f"   {en:.3f} off - ONE correction of {n:.0f} px")
                    p = plan(o, m, det["bins"][colour][:2])
                    if sh.any():
                        p_off = plan((o[0], o[1] + sh[0], o[2] + sh[1]) + tuple(o[3:]), m, det["bins"][colour][:2])
                        arm.move_to(with_grip(p_off["low"], hw.GRIP_OPEN), sp * 0.5, tol=3, settle=0.1, timeout=2)
                        print(f"   came down {APPROACH_CM:.1f} cm towards the stationary jaw's side - sliding onto the spot at grab height")
                    else:
                        arm.move_to(with_grip(p["low"], hw.GRIP_OPEN), sp * 0.5, tol=3, settle=0.1, timeout=2)
                else:
                    ok, err = False, None
                    for attempt in (1, 2):
                        uv, Js[side], err = align((o[1], o[2]), measure_at(0.6 if m.from_demos else 1.0), m.wrist_target,
                                                  Js.get(side), tol=0.05 if m.from_demos else None)
                        if not near_bin(uv[0], uv[1], det):
                            o = (o[0], float(uv[0]), float(uv[1])) + tuple(o[3:])
                        if m.from_demos:
                            uv, _, err2 = align((o[1], o[2]), measure_at(0.25), m.wrist_target, None,
                                                tol=0.015, step=12, total=25, iters=4, probe=6.0)
                            if not near_bin(uv[0], uv[1], det):
                                o = (o[0], float(uv[0]), float(uv[1])) + tuple(o[3:])
                            err = err2 if err2 is not None else err
                        if err is None or err <= CENTRE_OK:
                            ok = True
                            break
                        print(f"   NOT CENTRED (wrist error {err:.3f} > {CENTRE_OK}) - back up and try again" if attempt == 1
                              else f"   still not centred ({err:.3f}) - skipping this object instead of knocking it")
                        p = plan(o, m, det["bins"][colour][:2])
                        arm.move_to(with_grip(p["hover"], hw.GRIP_OPEN), sp * 0.5)
                    if not ok:
                        skipped.add((orig[1], orig[2]))
                        arm.move_to(with_grip(p["mid1"], hw.GRIP_OPEN), sp)
                        arm.move_to(rest[side], sp, tol=99)
                        continue
                    if err is not None:
                        print(f"   CENTRED between the jaws (wrist error {err:.3f})")
                        record_correction(side, orig, o)
                    p = plan(o, m, det["bins"][colour][:2])
                    arm.move_to(with_grip(p["low"], hw.GRIP_OPEN), sp * 0.4)    # straight down from where it was checked
            tr = trim_px(side)
            jo = JAW_OFFSET.get(side, 0.0)
            if jo:
                # 25 Nov late (user): right arm's STATIONARY jaw came down ~1 cm INSIDE the block. Move the grab spot jo cm
                # towards the stationary-jaw side = to the RIGHT as seen from the arm base looking at the block (= right
                # in that arm's wrist picture). Worked out per block: 90 deg right of the base -> block line.
                bu, bv = ARM_BASE_PX[side]
                hdg = np.array([o[1] - bu, o[2] - bv], float)
                hdg /= (np.linalg.norm(hdg) + 1e-9)
                right = np.array([-hdg[1], hdg[0]])          # picture y points down: (-hy, hx) = right-hand side
                tr = tr + right * jo * PX_PER_CM
                print(f"   {side} arm: grab spot {jo:+.1f} cm towards the stationary jaw")
            if tr.any():
                o = (o[0], o[1] + float(tr[0]), o[2] + float(tr[1])) + tuple(o[3:])
                p = plan(o, m, det["bins"][colour][:2])
                print(f"   trim {side}: grab moved {tr[0] / PX_PER_CM:+.1f} cm right, {tr[1] / PX_PER_CM:+.1f} cm towards the arms")
                arm.move_to(with_grip(p["low"], hw.GRIP_OPEN), sp * 0.4)
            arm.move_to(with_grip(p["grasp"], hw.GRIP_OPEN), sp * 0.4, tol=2.5, settle=0.1, timeout=2)
            save_pic(cam.grab(), f"{side}_top_at_grab", [(orig[1], orig[2], (255, 0, 255), 'object was here')])
            if side in wcams:
                # 25 Nov: never close on nothing - at grab height the block must be between the jaws (wrist camera,
                # right size). Otherwise: don't close, go back up, try again (2nd time: skip it).
                wimg = wcams[side].grab()
                ref = m.wrist_area or 0.05
                fb = vision.find_block_edge(wimg, colour, min_frac=0.35 * ref, max_frac=3.0 * ref) if m.wrist_feature == "edge"                     else vision.find_blob(wimg, colour, near=m.wrist_target, min_frac=0.35 * ref, max_frac=3.0 * ref)
                blind = colour in CAMERA_BLIND.get(side, set())
                if fb is not None:
                    off = float(np.linalg.norm(np.asarray(fb[:2]) - np.asarray(m.wrist_target)))
                    if off > CLOSE_TOL:
                        print(f"   block seen but NOT between the jaws ({off:.3f} from the target > {CLOSE_TOL})")
                        fb = None
                save_pic(wimg, f"{side}_wrist_before_close_{'block' if fb else 'NO-BLOCK'}", [(fb[0], fb[1], (255, 0, 255), 'block')] if fb else [])
                if fb is None and blind:
                    print(f"   (the {side} wrist camera can't see {colour} - grabbing at the calibrated spot, the jaws will tell)")
                elif fb is None:
                    misses += 1
                    print(f"   NO block of the right size between the jaws - NOT closing, back up")
                    arm.move_to(with_grip(p["hover"], hw.GRIP_OPEN), sp * 0.6, tol=99, settle=0.05)
                    arm.move_to(rest[side], TRANSFER_SPEED, tol=99, settle=0.1)
                    note_miss(deferred, skipped, orig)
                    continue
                else:
                    print(f"   block between the jaws (size {fb[2]:.3f}, {off:.3f} from the target) - closing")
            arm.move_to(with_grip(p["grasp"], hw.GRIP_CLOSED), sp, tol=99, settle=0.0)
            time.sleep(0.4)
            # 25 Nov: is there an object between the jaws? Ask the GRIPPER, not the top camera (the camera still sees
            # the object right above its old spot while it is held -> it said "missed" and opened: the drop bug).
            # Holding = the jaws stopped part-way (object between them); empty = they closed all the way.
            jaw = arm.read()[5]
            if jaw < HOLD_MIN:
                misses += 1
                print(f"   MISSED: gripper closed on nothing (jaw {jaw:.1f} < {HOLD_MIN}) - opening")
                arm.move_to(with_grip(p["grasp"], hw.GRIP_OPEN), sp, tol=99)
                arm.move_to(with_grip(p["hover"], hw.GRIP_OPEN), sp * 0.5)
                arm.move_to(with_grip(p["mid1"], hw.GRIP_OPEN), sp)
                arm.move_to(rest[side], sp, tol=99)
                note_miss(deferred, skipped, orig)
                continue
            print(f"   GRIPPED (jaw stopped at {jaw:.1f}: object between the jaws)")
            arm.move_to(with_grip(p["hover"], hw.GRIP_CLOSED), sp * 0.6, tol=99, settle=0.05)  # straight up, still closed
            jaw = arm.read()[5]
            if jaw < HOLD_MIN:
                print(f"   DROPPED while lifting (jaw {jaw:.1f}) - back home, will try again")
                arm.move_to(with_grip(p["mid1"], hw.GRIP_OPEN), sp)
                arm.move_to(rest[side], sp, tol=99)
                continue
            misses = 0
            # gripper stays CLOSED the whole way until it is above the bin
            # 25 Nov: DIRECT from high above the block to high above the bin (both high = clear of the table), fast
            # wait until the arm has REALLY arrived above the bin (at 60 deg/s the arm lags its commands; it used to
            # open while still on the way)
            err = arm.move_to(with_grip(p["bin"], hw.GRIP_CLOSED), TRANSFER_SPEED, tol=3.0, settle=0.2, timeout=3.0)
            if err > 3.0:
                print(f"   arm still {err:.1f} deg from the bin pose after 3 s - opening anyway")
            jaw = arm.read()[5]
            print(f"   above the {colour} bin, " + (f"still holding (jaw {jaw:.1f}) - releasing" if jaw >= HOLD_MIN
                                                       else f"object LOST on the way (jaw {jaw:.1f})"))
            bx = det["bins"][colour]
            save_pic(cam.grab(), f"{side}_top_at_drop_{colour}", [(bx[0], bx[1], (0, 255, 255), f"{colour} bin centre")])
            arm.move_to(with_grip(p["bin"], hw.GRIP_OPEN), TRANSFER_SPEED, tol=99, settle=0.0)
            time.sleep(0.25)
            arm.move_to(rest[side], TRANSFER_SPEED, tol=99, settle=0.1)          # straight home
            done += 1
            print(f"   sorted ({done} so far)")
    except KeyboardInterrupt:
        print("\nSTOP pressed - arms back to rest")
    finally:
        for s, a in arms.items():
            try:
                a.move_to(rest[s], hw.MAX_SPEED, tol=99, timeout=3)
            except Exception:
                pass
            a.disconnect()
        cam.disconnect()
        for wc in wcams.values():
            wc.disconnect()
    print(f"\nSORTED {done} objects.")


# ---------- BOTH ARMS AT THE SAME TIME (26 Nov; reworked 26 Nov pm after the lab test) ---------------------------------
# 26 Nov pm (Peter's test, 9 sorted in 5 min): blocks were SEEN but nobody took them (the arm with the right bin could
# not reach them), missed blocks were skipped for good (so the arms stood still before the 5 min were up), every pick
# near the middle made the other arm wait at home, and the right arm's "green bin" pose pointed at the YELLOW bin.
# Now:
#  * NO-OVERLAP ZONES instead of "wait at home": a pick reserves the strip of table (top-camera x) between the arm's
#    base, the block and the bin, plus SORT_GAP px each side. The other arm may do anything outside that strip at the
#    same time; it only waits when its block really is inside. While waiting it takes another block if it has one.
#  * HAND-OFF: a block whose bin belongs to the arm that CAN'T reach it is moved by the other arm to a free spot in the
#    middle (both arms reach it), then sorted from there. (The two arms' demo areas don't overlap - left x <= 345,
#    right x >= 376 - so the middle strip is at the edge of both; TEACH_MIDDLE.bat teaches it for both arms.)
#  * NEVER GIVE UP: a missed block goes to the back of the queue; when it comes round again the grab is nudged 1 cm
#    (further, nearer, left, right in turn). The arms keep going until the timer runs out.
#  * BIN CHECK at start: a bin pose that turns the arm a different way from where that bin is (> 20 deg) is not used.
CROSS_PX = {"left": float(os.environ.get("SORT_CROSS_LEFT") or 320), "right": float(os.environ.get("SORT_CROSS_RIGHT") or 380)}
SEP = float(os.environ.get("SORT_SEP") or 75)      # px (~9 cm): closest the two arms' areas may come (see strip()).
#   26 Nov pm: replaces the left-right strip (SORT_GAP), which made the arms wait 227 s of a 5-min run.
GAP = float(os.environ.get("SORT_GAP") or 50)      # (old, only printed) px (~6 cm): each arm's strip was this much wider on both sides.
#   26 Nov pm (Peter: don't let them hit each other in the middle): 30 -> 50. With 30 both arms worked 8.5 cm apart
#   in the middle (grippers + wrist cameras + open jaws stick out). 50 = at least ~12 cm between the two grab spots;
#   left picks into its green bin and right picks into its blue bin still fit side by side.
UP_TOL = float(os.environ.get("SORT_UP_TOL") or 5)   # deg: after a grab / miss / put-down the arm must really be UP
#   (within this of the high pose) before it travels sideways (26 Nov pm: it dragged blocks over the black line)
SAG_LEARN = os.environ.get("SORT_SAG", "1") == "1"
SAG_MAX = float(os.environ.get("SORT_SAG_MAX") or 6)   # deg: largest sag correction per joint
SAG_FILE = HERE / "sag_comp.json"
NEAR_LAST = os.environ.get("SORT_NEAR_BASE_LAST", "0") == "1"


def load_sag():
    try:
        import json
        return json.loads(SAG_FILE.read_text())
    except Exception:
        return {}


def save_sag(d):
    try:
        import json
        SAG_FILE.write_text(json.dumps(d, indent=1))
    except Exception:
        pass


FINISH_S = float(os.environ.get("SORT_FINISH_S") or 12)   # s: no new pick in the last 12 s, so the arms are done and
#   back at rest when the 5 minutes are up (26 Nov: Peter - the run must end at 5 min)
HOME_TOL = float(os.environ.get("SORT_HOME_TOL") or 12)   # deg: an arm only frees its strip once it is REALLY back
#   near its rest pose (pan / lift / elbow within this) - at speed it lags behind its commands
HANDOFF = os.environ.get("SORT_HANDOFF", "1") == "1"
DYNAMIC = os.environ.get("SORT_RULE", "colour").lower() == "dynamic"   # Round 3: shape rule + MOVING bins / blocks
BIN_TRACK = os.environ.get("SORT_BIN_TRACK", "1") == "1"   # 27 Nov: every round drops where the bin is NOW (camera)
BIN_BACK = float(os.environ.get("SORT_BIN_BACK") or 10)    # px (~1.2 cm): aim past the bin centre, away from the arm


def pan_offset(side):
    """27 Nov (competition): SORT_PAN_LEFT / SORT_PAN_RIGHT - the arm base is turned vs the lab (degrees)."""
    try:
        return float(os.environ.get(f"SORT_PAN_{side.upper()}") or 0)
    except ValueError:
        return 0.0
SHAPE_RULE = os.environ.get("SORT_RULE", "colour").lower() in ("shape", "dynamic")   # Round 2/3: sort by SHAPE
SHAPE_MIN = float(os.environ.get("SORT_SHAPE_MIN") or 0.5)   # below this confidence the block waits for a better look
BIN_MEMORY = os.environ.get("SORT_BIN_MEMORY", "0" if DYNAMIC else "1") == "1"   # off in Round 3: bins move   # a bin the camera loses (full of mixed blocks) stays
#   where it was last seen
WIDE_MARGIN = float(os.environ.get("SORT_WIDE_MARGIN") or 25)   # px: when an arm has nothing else, it also takes blocks
#   this far outside its demo area (normal: 25). 26 Nov pm: the idle left arm had red blocks at -30 and -33 px.
# 27 Nov (competition): after the turn correction the misses are SIDEWAYS scatter (~1 cm, either side) - so a retry
# searches sideways first (right, then left, as seen from the arm), then further / closer
NUDGE_CM = [(0, 0), (0, -1), (0, 1), (1, 0), (-1, 0)]   # retry n of a missed block: (further from the arm, to its left)
NUDGE_CLEAR = float(os.environ.get("SORT_NUDGE_CLEAR") or 24)   # px (~3 cm): a nudged grab stays this far from other blocks
LINE_IN = float(os.environ.get("SORT_LINE_IN") or 5)     # px: a block's centre must be this far INSIDE the black line
LINE_NEAR = float(os.environ.get("SORT_LINE_NEAR") or 30)   # px: "near the line" (a miss can push it over)
LINE_TRIES = int(os.environ.get("SORT_LINE_TRIES") or 2)    # tries for a block near the line, then it is left alone
DESC = float(os.environ.get("SORT_DESC") or 0.4)          # 26 Nov: speed going DOWN onto the block (x travel speed)
LIFT = float(os.environ.get("SORT_LIFT") or 0.6)          # speed lifting the block (x travel speed)
CLOSE_WAIT = float(os.environ.get("SORT_CLOSE_WAIT") or 0.4)   # s after closing before the jaw check
OPEN_WAIT = float(os.environ.get("SORT_OPEN_WAIT") or 0.25)    # s after opening above the bin
DROP_LOW = os.environ.get("SORT_DROP_LOW", "0") == "1"   # 26 Nov pm: OFF - the arm hit the bins going down to the
#   demo let-go pose. Now it opens high above the bin (DROP_UP lift heights above the let-go pose, or exactly at a
#   hand-taught bin pose from TEACH_BIN.bat).
NEAR_BASE = float(os.environ.get("SORT_NEAR_BASE") or 75)   # px (~9 cm): a block this close to an arm's own base is
#   left until last for that arm (26 Nov pm: the left arm kept missing blocks right beside its base)
ZONE_SPEED = float(os.environ.get("SORT_ZONE_SPEED") or 16.0)   # deg/s: the last part straight down onto the block
#   (= the speed of the 25 Nov runs that grabbed well). The fast mode keeps this - speed only where it is safe.
WAIT_MAX = float(os.environ.get("SORT_WAIT_MAX") or 8.0)       # s: longest wait for the arm to "arrive" (sagging arm)


SHAPE_POINTS = {"cube": (1, 2), "cylinder": (2, 3), "dodecahedron": (3, 4), "pentagonal_prism": (5, 5)}   # 5.8.1.71
LARGE_AREA = float(os.environ.get("SORT_LARGE_AREA") or 450)
EV_MIN = float(os.environ.get("SORT_EV_MIN") or 0.5)          # expected points to place a block normally
EV_RELAX = float(os.environ.get("SORT_EV_RELAX") or 0.2)      # ... when this arm has nothing better to do
EV_DESPERATE = float(os.environ.get("SORT_EV_DESPERATE") or 0.0)   # ... last resort (idle > 2 s): never a losing bet    # px: outline bigger than this = a LARGE block


TAUGHT_APPROACH = float(os.environ.get("SORT_TAUGHT_APPROACH") or 1.0)   # lift heights above a taught drop pose
SHAPE_MIN_PRISM = float(os.environ.get("SORT_SHAPE_MIN_PRISM") or 0.75)   # 26 Nov 21:55: prisms stay strict
SHAPE_MIN_DODECA = float(os.environ.get("SORT_SHAPE_MIN_DODECA") or 0.6)   # 26 Nov 19:45: 7 of 13 placements in the
#   19:32 run were dodecahedron guesses at 45-65 %, at least 2 of them lying cylinders   # 26 Nov 18:35: "prism" is ShapeNet's least
#   reliable answer - in the first shape run every wrong block in the yellow bin was a prism guess at 50-54 %
SHAPE_RELAX = float(os.environ.get("SORT_SHAPE_RELAX") or 0.4)   # 3-net model: softer, better-calibrated scores            # when an arm has nothing sure left, it also
SHAPE_RELAX_PRISM = float(os.environ.get("SORT_SHAPE_RELAX_PRISM") or 0.75)   # takes blocks down to these (all attempted)
MAX_TRIES = int(os.environ.get("SORT_MAX_TRIES") or 2)   # tries per block, then it is left alone
IDLE_MAX = float(os.environ.get("SORT_IDLE_MAX") or 2)    # s: 26 Nov 22:00 (Peter: never idle more than 2 s) - after
#   this an arm takes a LAST-RESORT pick: best shape guess >= DESPERATE_MIN, 2 more tries for missed blocks, only blocks
#   touching a bin (BIN_TOUCH) skipped, reach DESPERATE_MARGIN px outside its demo area. The black line still applies.
DESPERATE_MIN = float(os.environ.get("SORT_DESPERATE_MIN") or 0.3)
DESPERATE_MARGIN = float(os.environ.get("SORT_DESPERATE_MARGIN") or 40)
BIN_TOUCH = float(os.environ.get("SORT_BIN_TOUCH") or 12)
BLOCK_DEG = float(os.environ.get("SORT_BLOCK_DEG") or 8)  # arm stopped this far from the grab pose = it HIT something
#   (a bin, a block, the other arm) - 26 Nov: right-arm grabs stopped 13-52 deg short next to bins
MISS_LAST = int(os.environ.get("SORT_MISS_LAST") or 4)   # a block missed this often goes behind every other block
SHAPE_RANK = {"pentagonal_prism": 4, "dodecahedron": 3, "cylinder": 2, "cube": 1}   # shape round: picking order
LINE_EDGE = float(os.environ.get("SORT_LINE_EDGE") or 4)   # px: the block's EDGE (not just its centre) must be this
#   far inside the black line (26 Nov 23:30: no block touching the line is picked)


def shape_min(shape, relaxed=False):
    if shape == "pentagonal_prism":
        return SHAPE_RELAX_PRISM if relaxed else SHAPE_MIN_PRISM
    if shape == "dodecahedron":
        return max(SHAPE_RELAX, SHAPE_MIN_DODECA - 0.1) if relaxed else SHAPE_MIN_DODECA
    return SHAPE_RELAX if relaxed else SHAPE_MIN


def shape_value(o):
    """Expected points for placing block o by its predicted shape: (2 x confidence - 1) x points (wrong bin = minus)."""
    if len(o) > 10:
        return float(o[10])                    # the points-maximising EV from classify_shapes
    if len(o) < 10:
        return 0.0
    small, large = SHAPE_POINTS.get(o[8], (1, 1))
    pts = large if (o[4] or 0) > LARGE_AREA else small
    return (2 * float(o[9]) - 1) * pts


MOVED_PX = float(os.environ.get("SORT_MOVED_PX") or 18)
BIN_WAIT = float(os.environ.get("SORT_BIN_WAIT") or 3)   # s: Round 3 - wait above for a bin that is being carried   # Round 3: block no longer within this of where it was = moved
_DROP_MODELS = {}


def drop_model(side):
    """Round 3: the learned drop model of this arm (drop_model.py, drop_model_<side>.json) or None."""
    if side not in _DROP_MODELS:
        try:
            import drop_model as DM
            _DROP_MODELS[side] = DM.DropModel.load(side)
        except Exception as e:
            print(f"   (no drop model for the {side} arm: {e!r} - bins only on the stored spots)")
            _DROP_MODELS[side] = None
    return _DROP_MODELS[side]


def bin_ok(m, colour, bu, bv):
    """Can this arm drop into a bin of `colour` seen at (bu, bv)? Round 3: anywhere its drop model covers."""
    if DYNAMIC or BIN_TRACK:
        if hasattr(m, "bin_slot") and m.bin_slot(colour, bu, bv) is not None:
            return True                                   # on one of this arm's known spots
        dm = drop_model(m.side)
        if dm is not None and dm.covers(bu, bv):
            # 26 Nov 21:40: the NEARER arm serves a bin (the right arm stretching ~20 cm to the green bin sagged short
            # towards the arms); blocks for the other side are handed over
            other = "right" if m.side == "left" else "left"
            mine = math.hypot(bu - ARM_BASE_PX[m.side][0], bv - ARM_BASE_PX[m.side][1])
            theirs = math.hypot(bu - ARM_BASE_PX[other][0], bv - ARM_BASE_PX[other][1])
            odm = drop_model(other)
            if odm is not None and odm.covers(bu, bv) and theirs < mine - 10:
                return False
            return True
        return False
    return m.bin_hover(colour, bu, bv) is not None      # (also: a bin on one of this arm's known spots)


def classify_shapes(net, frames, det, memory=None, frozen=()):
    """Round 2: every block gets its SHAPE from ShapeNet (average over the camera frames); its target bin is the one
    for that shape (rulebook 5.8.1.46). Object tuple becomes (target bin, u, v, angle, area, w, h, real colour, shape,
    confidence); target "?" = not sure enough yet (looked at again on the next scan)."""
    import shape_model as SM
    objs = []
    for o in det["objects"]:
        o = tuple(o) + (0.0,) * max(0, 7 - len(o))
        p = net.probs([SM.crop_at(f, o[1], o[2]) for f in frames]).mean(0)
        if memory is not None:
            # 26 Nov: AVERAGE the block's shape over every look so far (same spot +-10 px) - one lucky scan must not
            # push a block over the confidence threshold
            e = next((e for e in memory if math.hypot(e[0] - o[1], e[1] - o[2]) < 10), None)
            if e is None:
                e = [o[1], o[2], np.zeros(len(p)), 0]
                memory.append(e)
            if e[3] > 0 and any(math.hypot(o[1] - f[0], o[2] - f[1]) < 30 for f in frozen):
                p = e[2] / e[3]        # 26 Nov: an arm is going for it - the camera now sees the GRIPPER, not the block
            else:
                e[0], e[1], e[2], e[3] = o[1], o[2], e[2] + p, e[3] + 1
                p = e[2] / e[3]
        # 26 Nov 22:15: POINTS-MAXIMISING bin. A wrong bin costs the block's OWN points (5.8.1.63/72), so the best bin
        # is argmax P(shape) x points(shape, size), and the expected gain of placing it there is
        # EV = 2 x P_b x pts_b - sum_s P_s x pts_s. This puts prisms first, then dodecahedra, then cylinders, and keeps
        # a likely prism out of the cube bin (a mistake there costs 5).
        # 26 Nov 22:25: the bin is the MOST LIKELY shape (the points-weighted choice dumped unsure blocks in the yellow
        # bin - the model's prism probabilities are not reliable on the real table). Expected points only decide the
        # ORDER (prisms / dodecahedra first) and whether a block is worth placing: EV = 2 x P_b x pts_b - sum P_s x pts_s
        large = (o[4] or 0) > LARGE_AREA
        pts = np.array([SHAPE_POINTS[c][1 if large else 0] for c in SM.CLASSES], float)
        pv = p * pts
        k = int(p.argmax())
        ev = float(2 * pv[k] - pv.sum())
        shape = SM.CLASSES[k]
        ok = p[k] >= shape_min(shape) and ev >= EV_MIN
        objs.append(((SM.SHAPE_TO_BIN[shape] if ok else "?"),) + tuple(o[1:7]) + (o[0], shape, float(p[k]), ev))
    return dict(det, objects=objs)


def strip(side, *pts):
    """26 Nov pm: the AREA of table an arm sweeps for a pick (top-camera px polygon): its base, the block, the bin /
    hand-off spot, and the arcs it swings through between them (around its base). Replaces the left-right strip
    (with a 12 cm strip the arms waited 227 s of a 5-min run - it blocked picks far apart front-to-back)."""
    import cv2
    bx, by = ARM_BASE_PX[side]
    P = [(bx, by)] + [(float(u), float(v)) for u, v in pts]
    tg = [(float(u), float(v)) for u, v in pts]
    for (u1, v1), (u2, v2) in zip(tg, tg[1:]):
        a1, a2 = math.atan2(v1 - by, u1 - bx), math.atan2(v2 - by, u2 - bx)
        da = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        r1, r2 = math.hypot(u1 - bx, v1 - by), math.hypot(u2 - bx, v2 - by)
        for t in (0.25, 0.5, 0.75):
            a, r = a1 + t * da, r1 + t * (r2 - r1)            # swings round its base while reaching in / out
            P.append((bx + r * math.cos(a), by + r * math.sin(a)))
    return cv2.convexHull(np.array(P, np.float32))


def _seg_cross(p1, p2, p3, p4):
    def o(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return o(p1, p2, p3) * o(p1, p2, p4) < 0 and o(p3, p4, p1) * o(p3, p4, p2) < 0


def zone_gap(a, b):
    """Smallest distance (px) between two arm areas; 0 if they touch or cross."""
    import cv2
    A, B = a.reshape(-1, 2), b.reshape(-1, 2)
    d = min([-cv2.pointPolygonTest(b, (float(x), float(y)), True) for x, y in A]
            + [-cv2.pointPolygonTest(a, (float(x), float(y)), True) for x, y in B])
    if d <= 0:
        return 0.0
    for i in range(len(A)):
        for j in range(len(B)):
            if _seg_cross(A[i], A[(i + 1) % len(A)], B[j], B[(j + 1) % len(B)]):
                return 0.0
    return float(d)


def overlap(a, b):
    """True = the two arms' areas come closer than SEP px (they could hit each other)."""
    return a is not None and b is not None and zone_gap(a, b) < SEP


def check_bins(maps):
    """Drop bin poses that point the arm the wrong way. Expected turn (pan) for a bin = the same pan-vs-direction line
    the arm's own demo grabs follow. 26 Nov pm: the right arm's 'green' bin had pan +9.5 (= towards its yellow bin)
    where the green bin needs about -45 -> Peter's green prism went into the yellow bin."""
    for side, m in maps.items():
        if m is None or len(m.uv) < 5:
            continue
        bu, bv = ARM_BASE_PX[side]
        head = np.degrees(np.arctan2(m.uv[:, 0] - bu, bv - m.uv[:, 1]))
        k, c = np.linalg.lstsq(np.c_[head, np.ones(len(head))], m.grasp[:, 0], rcond=None)[0]
        for colour in list(m.bins):
            b = m.bins[colour]
            want = k * math.degrees(math.atan2(b["u"] - bu, bv - b["v"])) + c
            if abs(b["hover"][0] - want) > 20:
                print(f"   BIN CHECK: {side} arm's {colour} bin pose turns the arm to {b['hover'][0]:.0f} deg, but the "
                      f"{colour} bin is at about {want:.0f} deg - NOT used (this arm hands {colour} blocks over instead)")
                del m.bins[colour]


def handoff_spots(maps, giver, recv, margin=None):
    """Grid of table spots BOTH arms reach, best first (the receiver's grab there matters most)."""
    margin = model.REACH_MARGIN if margin is None else margin
    import cv2
    out = []
    for u in range(290, 431, 5):
        for v in range(290, 400, 5):
            if cv2.pointPolygonTest(vision.SOURCE_POLY, (float(u), float(v)), True) < 25:   # 3 cm inside the line
                continue
            rg, rr = maps[giver].reach(u, v), maps[recv].reach(u, v)
            # 26 Nov pm: the arm that PUTS the block down stays inside its normal reach (25 px) - only the receiver's
            # margin is widened (the right arm's put-downs at 40 px out were bad)
            if rg < -model.REACH_MARGIN:
                continue
            # (26 Nov pm: no extra slack any more - only 5 spots, one stuck block blocked them all. Ping-pong is
            # prevented instead by never handing a block off again from where a hand-off put it down)
            if rg >= -margin and rr >= -margin:
                out.append((min(rg, rr) + 0.3 * rr, u, v))
    out.sort(reverse=True)
    return [(u, v) for _, u, v in out]


def misses_at(deferred, u, v):
    return max([n for d, n in deferred.items() if math.hypot(u - d[0], v - d[1]) < 15], default=0)


def choose_task(side, det, maps, deferred, other_claim, busy, spots, put_down=(), margin=None, clear=35, why=None,
                relax_shape=False, desperate=False):
    """Best job for THIS arm. busy = table strips this arm must stay out of (the other arm's current pick, or the one
    it is waiting for). Returns (task, conflict) - conflict True = every job of this arm is inside a busy strip.
    margin = how far (px) outside an arm's demo area a block still counts as reachable; clear = free space (px) a
    hand-off spot needs around it. why (dict) collects why blocks were not taken (for the "idle" message)."""
    margin = model.REACH_MARGIN if margin is None else margin
    why = {} if why is None else why

    def no(reason):
        why[reason] = why.get(reason, 0) + 1
    other = "right" if side == "left" else "left"
    mo = maps.get(other)
    tasks = []
    for o in det["objects"]:
        colour, u, v = o[0], o[1], o[2]
        if colour == "?":
            ev = o[10] if len(o) > 10 else -9
            conf_ok = len(o) > 9 and o[9] >= shape_min(o[8], relaxed=True)
            if relax_shape and len(o) > 9 and ((conf_ok and ev >= EV_RELAX) or (desperate and conf_ok and ev >= EV_DESPERATE)):
                import shape_model as SM
                o = (SM.SHAPE_TO_BIN[o[8]],) + tuple(o[1:])      # nothing sure left: take its best guess
                colour = o[0]
            else:
                no("shape not sure yet")
                continue
        if colour not in det["bins"]:
            no("no bin of its colour")
            continue
        if near_bin(u, v, det, margin=BIN_CLEAR):          # 26 Nov 22:25: also in the last-resort pass (it grabbed a bin)
            no("too close to a bin")
            continue
        half = max(o[5] or 0, o[6] or 0) / 2.0 if len(o) > 6 else 0.0
        if inside_line(u, v) < max(LINE_IN, half + LINE_EDGE):
            # 26 Nov pm (Peter: don't pick up objects outside the space): on or outside the black line = not ours
            # 26 Nov 23:30: measured to the block's EDGE - a block TOUCHING the line is not picked either
            no("on / outside the black line")
            continue
        if misses_at(deferred, u, v) >= MAX_TRIES + (2 if desperate else 0):
            # 26 Nov 19:00: all of today's runs - first tries work ~40 %, retries ~7 % (15 of ~210); one cube got 13
            # tries in a row. A block is tried at most MAX_TRIES times, then left alone.
            no(f"missed {MAX_TRIES}x - left alone")
            continue
        if inside_line(u, v) < LINE_NEAR and misses_at(deferred, u, v) >= LINE_TRIES:
            # every miss near the line can push the block over it - after LINE_TRIES misses, leave it
            no("near the black line, missed twice")
            continue
        if other_claim is not None and math.hypot(u - other_claim[0], v - other_claim[1]) < 25:
            no("the other arm is taking it")
            continue
        bu, bv = det["bins"][colour][:2]
        m = maps[side]
        if m.reach(u, v) < -margin:
            no("out of this arm's reach")
            continue
        has_me = bin_ok(m, colour, bu, bv)
        has_other = mo is not None and bin_ok(mo, colour, bu, bv)
        other_reach = (mo.reach(u, v) >= -margin) if mo is not None else False
        n = misses_at(deferred, u, v)
        near = math.hypot(u - ARM_BASE_PX[side][0], v - ARM_BASE_PX[side][1]) < NEAR_BASE   # right by this arm's base
        by_bin = near_bin(u, v, det, margin=BIN_LAST)        # close-ish to a bin: taken, but after everything else
        if has_me:
            if has_other and other_reach and mo.reach(u, v) > m.reach(u, v):
                no("the other arm's job")
                continue                                   # both have this bin: the closer arm does it
            tasks.append(dict(kind="sort", o=o, bin_xy=(bu, bv), score=m.reach(u, v), misses=n, near=near, by_bin=by_bin,
                              iv=strip(side, (u, v), (bu, bv))))
        elif not has_other or other_reach:
            no("the other arm's job")
        elif any(math.hypot(u - s[0], v - s[1]) < 30 for s in put_down):
            no("put down for the other arm")
        elif HANDOFF and spots:
            # the arm with the bin can't reach it -> this arm moves it to a free middle spot the other arm reaches
            spot = next((s for s in spots
                         if not near_bin(s[0], s[1], det, margin=25)
                         and all(math.hypot(s[0] - q[1], s[1] - q[2]) > clear for q in det["objects"] if q is not o)
                         and (other_claim is None or math.hypot(s[0] - other_claim[0], s[1] - other_claim[1]) > 45)),
                        None)
            if spot is not None:
                tasks.append(dict(kind="handoff", o=o, spot=spot, score=m.reach(u, v) - 20, misses=n, near=near, by_bin=by_bin,
                                  iv=strip(side, (u, v), spot)))
            else:
                no("hand-off: no free middle spot")
    if not tasks:
        return None, False
    # order: fewest misses, then deepest inside the arm's area. 26 Nov pm (Peter: grab the blocks by the bases more):
    # blocks by an arm's base are no longer left till last (SORT_NEAR_BASE_LAST=1 = they are again)
    if SHAPE_RULE:   # Round 2: most EXPECTED POINTS first (a 5-min round only fits ~10-12 picks)
        # 26 Nov 23:30 (user): prism > dodecahedron > cylinder > cube (large before small), the CLOSEST to this arm's
        # base first within a shape; hand-offs ranked the same (a prism on the wrong side is passed before an own cube);
        # a missed block goes behind every untried one (retries work ~7 %, first tries ~40 %) and comes round at the end
        tasks.sort(key=lambda t: (t["misses"] > 0, -SHAPE_RANK.get(t["o"][8] if len(t["o"]) > 8 else "", 0),
                                  t.get("by_bin", False), -((t["o"][4] or 0) > LARGE_AREA),
                                  math.hypot(t["o"][1] - ARM_BASE_PX[side][0], t["o"][2] - ARM_BASE_PX[side][1])))
    else:
        tasks.sort(key=lambda t: (t["misses"] >= MISS_LAST, t.get("by_bin", False), (t["near"] if NEAR_LAST else False),
                                  t["misses"], -t["score"]))
    free = [t for t in tasks if not any(overlap(t["iv"], b) for b in busy)]
    return (free[0], False) if free else (tasks[0], True)


def run_both():
    import threading
    import hw
    sys.stdout = _Tee(HERE / "sort_log.txt")
    print(f"\n===== SORT run BOTH ARMS v2 {time.strftime('%Y-%m-%d %H:%M:%S')} | up x{UP_K} | deep {DEEP} | speed "
          f"{hw.MAX_SPEED} | arms kept {SEP:.0f} px apart | hand-off {'on' if HANDOFF else 'off'} | trims left "
          f"{os.environ.get('SORT_TRIM_LEFT', '-')} right {os.environ.get('SORT_TRIM_RIGHT', '-')} =====")
    maps = {s: model.load(s) for s in ("left", "right")}
    for s, mm in maps.items():
        if mm is not None:
            mm.roll_offset = PICK_ROLL[s]
            print(f"{s} arm grasp-pose model: {mm.grasp_model}")
    check_bins(maps)
    for s, mm in maps.items():
        if mm is not None:
            print(f"{s} arm bins: {sorted(mm.bins)}")
    spots, spots_wide = {}, {}
    if all(maps.values()):
        for g, r in (("left", "right"), ("right", "left")):
            spots[g] = handoff_spots(maps, g, r)
            spots_wide[g] = handoff_spots(maps, g, r, WIDE_MARGIN)
            print(f"hand-off {g} -> {r}: {len(spots[g])} middle spots" + (f", best {spots[g][0]}" if spots[g] else
                  " - NONE (run TEACH_MIDDLE.bat)"))
    cam = hw.TopCamera()
    arms = {s: hw.Arm(s) for s, m in maps.items() if m is not None}
    cam.connect()
    for s, a in arms.items():
        a.connect()
        maps[s].widen_limits(UP_K, DROP_UP)
        if (DYNAMIC or BIN_TRACK) and drop_model(s) is not None and maps[s].up is not None:
            # 26 Nov 21:40: the drop model's poses (and the high approach) must not be clipped by the joint limits
            dm, upv = drop_model(s), np.asarray(maps[s].up[:5])
            Q = [dm.pose(u, v) + k * upv for u in range(180, 541, 15) for v in range(220, 341, 15) if dm.covers(u, v)
                 for k in (DROP_UP, BIN_APPROACH)]
            if Q:
                Q = np.array(Q)
                for i, j in enumerate(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]):
                    lo, hi = maps[s].limits[j]
                    maps[s].limits[j] = (min(lo, float(Q[:, i].min()) - 5), max(hi, float(Q[:, i].max()) + 5))
        a.limits = dict(maps[s].limits, gripper=(0.0, 80.0))
        a.limits["wrist_roll"] = (-172.0, 172.0)
    rest = {s: a.read() for s, a in arms.items()}
    print("rest poses:", {s: np.round(q[:5], 0).tolist() for s, q in rest.items()})
    sp = hw.MAX_SPEED
    cam_lock, cond = threading.Lock(), threading.Condition()
    active = {s: None for s in arms}            # table strip each arm is using right now
    want = {s: None for s in arms}              # strip an arm is waiting for (the other one keeps out of it)
    claimed = {s: None for s in arms}
    deferred = {}
    put_down = []                               # where hand-offs put blocks down (never handed off again from there)
    stats = {"done": 0, "handed": 0, "miss": 0, "wait_s": 0.0}
    stop = threading.Event()
    last_rec = {}
    run_id = time.strftime("%Y%m%d_%H%M%S")
    sag = load_sag()
    if sag:
        print("sag corrections from earlier runs:", {k: np.round(v, 1).tolist() for k, v in sag.items()})
    t_start = time.time()
    t_end = t_start + DURATION

    # 26 Nov pm (Peter: passively scan for objects): the top camera is read ALL THE TIME by its own thread; an arm
    # that wants to choose takes the newest picture that was taken after it asked (so never a stale one)
    scan = {"t": 0.0, "img": None, "det": None}
    bin_mem = {}
    bin_hist = []                               # bins of the last 3 scans (Round 3: a move must be seen twice)
    shape_mem = []                              # per-block running average of the shape probabilities
    shape_net = None
    if SHAPE_RULE:
        import shape_model as SM
        shape_net = SM.ShapeNet()
        print(("ROUND 3 - DYNAMIC SORT (moving bins/blocks, learned drop model)" if DYNAMIC else "ROUND 2 - SHAPE SORT")
              + f": ShapeNet {SM.WEIGHTS.name} ({shape_net.meta}) | "
              f"bins: {SM.SHAPE_TO_BIN}")
    scan_cond = threading.Condition()

    def scanner():
        while not stop.is_set():
            t0 = time.time()
            try:
                with cam_lock:
                    frames = [cam.grab() for _ in range(3)]
                det = vision.detect_avg(frames)
                if len(det["bins"]) < 4:              # full bins of mixed blocks: find them by their rim
                    for c, b in vision.find_bins_by_rim(frames[-1]).items():
                        det["bins"].setdefault(c, b[:5])
                if BIN_MEMORY:
                    # 26 Nov 19:00: keep each bin's LARGEST outline seen at that spot - a full bin can look smaller
                    # (its contents then looked like loose blocks and the arm went for them / hit the bin)
                    for c, b in det["bins"].items():
                        # 26 Nov 20:00: an outline bigger than a real bin = the bin merged with blocks touching it ->
                        # ignore it (the "largest outline" rule made the blue bin swallow half the table)
                        if max(b[2], b[3]) > BIN_MAX_SIDE + 6:
                            continue
                        bin_mem[c] = b
                    for c, b in bin_mem.items():
                        cur = det["bins"].get(c)
                        if cur is None or math.hypot(cur[0] - b[0], cur[1] - b[1]) < 30:
                            det["bins"][c] = b
                if shape_net is not None:
                    det = classify_shapes(shape_net, frames, det, shape_mem,
                                          frozen=[c for c in claimed.values() if c is not None])
            except Exception as e:
                print(f"   camera scan failed ({e!r}) - trying again")
                time.sleep(0.2)
                continue
            with scan_cond:
                scan.update(t=t0, img=frames[-1], det=det)
                bin_hist.append(dict(det["bins"]))
                del bin_hist[:-3]
                scan_cond.notify_all()

    def look():
        asked = time.time()
        with scan_cond:
            scan_cond.wait_for(lambda: scan["t"] >= asked or stop.is_set(), timeout=3.0)
            det = scan["det"] or {"bins": {}, "objects": []}
            return scan["img"], dict(det, objects=list(det["objects"]))

    def snap(name, marks):
        with cam_lock:
            img = cam.grab()
        save_pic(img, name, marks)

    def say(side, msg):
        print(f"[{time.strftime('%H:%M:%S')} {time.time() - t_start:5.1f}s {side.upper():5s}] {msg}")

    def pick(side, task, top_img, det):
        arm, m = arms[side], maps[side]
        o = task["o"]
        colour, orig = o[0], o
        n = task["misses"]
        say(side, (f"{o[8]} ({o[7]}, {o[9]:.0%} sure, +{o[10]:.1f} pts expected) -> {colour} bin, at" if len(o) > 10 else
                   f"{o[8]} ({o[7]}, {o[9]:.0%} sure) -> {colour} bin, at" if len(o) > 9 else f"{colour} object at")
                  + f" ({o[1]:.0f},{o[2]:.0f}) rot {o[3]:.0f}"
                  + (" | LONG block: jaws across it" if shape_of(o) == "long" else "")
                  + (f" | HAND-OFF to the middle {task['spot']}" if task["kind"] == "handoff" else "")
                  + (" | right by the arm's base" if task.get("near") else "")
                  + (" | close to a bin (done last, careful)" if task.get("by_bin") else "")
                  + (" | LAST-RESORT pick (idle > 2 s)" if task.get("last_resort") else
                     (" | WIDER look (nothing else left)" if task.get("wide") else ""))
                  + (f" | try {n + 1}" if n else ""))
        lb = learned_bias(side)
        if lb is not None and not near_bin(o[1] + lb[0], o[2] + lb[1], det):
            o = (o[0], o[1] + lb[0], o[2] + lb[1]) + tuple(o[3:])
        tr = trim_px(side)
        bu, bv = ARM_BASE_PX[side]
        hdg = np.array([o[1] - bu, o[2] - bv], float)
        hdg /= (np.linalg.norm(hdg) + 1e-9)
        left_of = np.array([hdg[1], -hdg[0]])                 # 90 deg to the left of the base -> block line
        jo = JAW_OFFSET.get(side, 0.0)
        if jo:
            tr = tr - left_of * jo * PX_PER_CM
        if n:                                              # retry: nudge the grab 1 cm, a different way each time
            # 26 Nov pm: the right arm's 7th try nudged 1 cm TOWARDS a red cube 2 cm away, grabbed the red cube and
            # put it in the yellow bin. A nudge may never bring the jaws closer to another block (or the line):
            others = [(q[1], q[2]) for q in det["objects"] if math.hypot(q[1] - orig[1], q[2] - orig[2]) > 6]
            nud = None
            for k in range(len(NUDGE_CM)):
                a, b = NUDGE_CM[(n + k) % len(NUDGE_CM)]
                d = (hdg * a + left_of * b) * PX_PER_CM
                gu, gv = orig[1] + d[0], orig[2] + d[1]
                if (a or b) and all(math.hypot(gu - x, gv - y) >= NUDGE_CLEAR for x, y in others) \
                        and inside_line(gu, gv) >= LINE_IN:
                    nud = (a, b, d)
                    break
            if nud is None:
                say(side, f"   retry {n}: no safe direction to nudge (other blocks / the line too close) - same spot")
            else:
                a, b, d = nud
                tr = tr + d
                say(side, f"   retry {n}: grab nudged {a:+d} cm further / {b:+d} cm left (seen from the arm)")
        if tr.any():
            o = (o[0], o[1] + float(tr[0]), o[2] + float(tr[1])) + tuple(o[3:])
        bin_xy = task.get("bin_xy") or det["bins"][colour][:2]
        p = plan(o, m, bin_xy)
        # 26 Nov: full record of this pick for the learning loop (learn_from_runs.py / analyse_run.py)
        last_rec[side] = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), run=run_id, t=round(time.time() - t_start, 1),
                              side=side, kind=task["kind"], rule="shape" if SHAPE_RULE else "colour",
                              u=float(orig[1]), v=float(orig[2]), u_aim=float(o[1]), v_aim=float(o[2]),
                              angle=float(orig[3]), target=colour, colour=(orig[7] if len(orig) > 9 else colour),
                              shape=(orig[8] if len(orig) > 9 else None),
                              conf=(float(orig[9]) if len(orig) > 9 else None), tries=int(task["misses"]) + 1,
                              wide=bool(task.get("wide")), long=shape_of(orig) == "long",
                              grasp_cmd=[float(x) for x in p["grasp"]], grasp_demo=[float(x) for x in p["grasp_demo"]],
                              deep=DEEP, up=None if m.up is None else [float(x) for x in m.up],
                              slot=p.get("slot"), source=m.last_source)
        if m.last_source:
            say(side, f"   arm pose from: {m.last_source}")
        save_pic(top_img, f"{side}_top_target", [(o[1], o[2], (255, 0, 255), f"{colour} target")])
        if DYNAMIC:
            # Round 3: the judge may have moved the block - check the newest camera picture BEFORE the arm sets off
            # (26 Nov 21:55: checked above the block, the arm itself hid it -> 9 false 'moved' aborts)
            d_now = scan.get("det") or {"objects": []}
            here = [q for q in d_now["objects"] if math.hypot(q[1] - orig[1], q[2] - orig[2]) < MOVED_PX
                    and (q[7] if len(q) > 9 else q[0]) == (orig[7] if len(orig) > 9 else orig[0])]
            if not here:
                say(side, "   the block is not there any more (moved?) - choosing again")
                return "moved"
        arm.move_to(with_grip(p["mid1"], hw.GRIP_OPEN), sp, tol=99, settle=0.0)
        # high above the block, settle, straight down at ZONE_SPEED (the careful zone-in that grabbed well), then the
        # last bit DEEP below the demo grab height; stops at once if the table is in the way
        arm.move_to(with_grip(p["hover"], hw.GRIP_OPEN), sp, tol=3, settle=0.15, timeout=2.0)
        arm.move_to(with_grip(p["low"], hw.GRIP_OPEN), ZONE_SPEED, tol=3, settle=0.05, timeout=3.0)
        arm.move_to(with_grip(p["grasp"], hw.GRIP_OPEN), ZONE_SPEED * 0.8, tol=2.0, settle=0.15, timeout=2.5, stall=True)
        q = arm.read()
        last_rec[side]["q_grasp"] = [float(x) for x in q[:5]]        # where the arm REALLY was when it closed
        short = float(np.abs(np.asarray(q[1:4]) - np.asarray(p["grasp"][1:4])).max())
        if short > BLOCK_DEG:
            # it hit something on the way down (bin rim, a block, the other arm): don't close, go back up, never again
            say(side, f"   BLOCKED {short:.0f} deg above the grab pose (hit something) - backing off, block left alone")
            snap(f"{side}_top_BLOCKED", [(orig[1], orig[2], (0, 0, 255), 'blocked here')])
            arm.move_to(with_grip(p["hover"], hw.GRIP_OPEN), sp * 0.5, tol=UP_TOL, settle=0.0, timeout=2.0)
            arm.move_to(rest[side], TRANSFER_SPEED, tol=99, settle=0.05)
            return "blocked"
        # 26 Nov 23:30 (user: go deep but don't press the gripper on the block): if the arm stopped before the deep
        # grab pose (table / block top), close where it IS - commanding the deeper pose kept the servos pressing down
        hold = np.asarray(q[:5], float) if short > 1.5 else np.asarray(p["grasp"], float)
        arm.send(np.r_[hold, hw.GRIP_CLOSED])
        t0, prev, steady, jaw = time.time(), None, 0, q[5]
        while time.time() - t0 < CLOSE_WAIT + 0.6:
            jaw = arm.read()[5]
            if jaw < HOLD_MIN:
                break
            steady = steady + 1 if prev is not None and abs(jaw - prev) < 0.6 else 0
            prev = jaw
            if steady >= 3 and time.time() - t0 > 0.12 and jaw < hw.GRIP_OPEN - 10:
                break
            time.sleep(0.02)
        last_rec[side]["jaw"] = float(jaw)
        if jaw < HOLD_MIN:
            say(side, f"   MISSED (jaw {jaw:.1f}) - next block, this one comes round again later")
            snap(f"{side}_top_MISSED", [(orig[1], orig[2], (255, 0, 255), 'object was here')])
            # 26 Nov pm: go UP and really GET there before travelling home - it used to head home while still low
            # (servo lag) and dragged blocks towards its base, over the black line
            arm.move_to(with_grip(p["hover"], hw.GRIP_OPEN), sp, tol=UP_TOL, settle=0.0, timeout=1.5)
            arm.move_to(rest[side], TRANSFER_SPEED, tol=99, settle=0.05)
            return "miss"
        say(side, f"   GRIPPED (jaw {jaw:.1f}, after {time.time() - t0:.2f} s)")
        arm.move_to(with_grip(p["hover"], hw.GRIP_CLOSED), sp * LIFT, tol=UP_TOL, settle=0.0, timeout=1.5)   # really up
        jaw = arm.read()[5]
        if jaw < HOLD_MIN:
            say(side, f"   DROPPED while lifting (jaw {jaw:.1f})")
            arm.move_to(rest[side], TRANSFER_SPEED, tol=99, settle=0.1)
            return "miss"
        if task["kind"] == "handoff":
            su, sv = task["spot"]
            pp = plan_place(m, su, sv, p["grasp"])
            arm.move_to(with_grip(pp["hover"], hw.GRIP_CLOSED), TRANSFER_SPEED, tol=4, settle=0.1, timeout=2.0)
            arm.move_to(with_grip(pp["place"], hw.GRIP_CLOSED), ZONE_SPEED * 1.5, tol=2.5, settle=0.1, timeout=2.5,
                        stall=True)
            arm.move_to(with_grip(pp["place"], hw.GRIP_OPEN), TRANSFER_SPEED, tol=99, settle=0.0)
            time.sleep(OPEN_WAIT)
            arm.move_to(with_grip(pp["hover"], hw.GRIP_OPEN), TRANSFER_SPEED, tol=UP_TOL, settle=0.0, timeout=1.5)
            arm.move_to(rest[side], TRANSFER_SPEED, tol=99, settle=0.1)
            say(side, f"   put down in the middle at ({su},{sv}) for the other arm")
            return "handed"
        # 26 Nov pm (Peter: the arms hit the bins going down): straight to the drop pose HIGH above the bin, wait until
        # still (opening while moving throws the block), JUST OPEN, go home. No going down into the bin.
        # (SORT_DROP_LOW=1 = the old way: down to the demo let-go pose first.)
        if DYNAMIC or BIN_TRACK:
            # Round 3: the judge may have moved the bin during the carry - re-aim at where the camera sees it NOW
            b_now = (scan.get("det") or {"bins": {}})["bins"].get(colour)
            if b_now is None:
                # 26 Nov 21:50: the judge may be CARRYING the bin (swapping bins) - wait above for it to reappear
                say(side, f"   the {colour} bin is not in view (being moved?) - waiting up to {BIN_WAIT:.0f} s")
                t_w = time.time()
                while b_now is None and time.time() - t_w < BIN_WAIT and not stop.is_set():
                    time.sleep(0.2)
                    b_now = (scan.get("det") or {"bins": {}})["bins"].get(colour)
                time.sleep(0.4)                                   # let a second look confirm the new place
            b_prev = bin_hist[-2].get(colour) if len(bin_hist) >= 2 else None
            confirmed = (b_now is not None and b_prev is not None and max(b_now[2], b_now[3]) <= BIN_MAX_SIDE + 6
                         and math.hypot(b_now[0] - b_prev[0], b_now[1] - b_prev[1]) < 12)
            if b_now is not None and not confirmed and math.hypot(b_now[0] - bin_xy[0], b_now[1] - bin_xy[1]) > 10:
                say(side, f"   the {colour} bin looks moved but not in two looks in a row - keeping the old aim")
            if confirmed and not bin_ok(m, colour, b_now[0], b_now[1]) and spots.get(side):
                # swapped to the OTHER arm's side: don't stretch across - put the block down in the middle for it
                det_now = scan.get("det") or {"objects": [], "bins": {}}
                spot = next((sp_ for sp_ in spots[side] if not near_bin(sp_[0], sp_[1], det_now, margin=25)
                             and all(math.hypot(sp_[0] - q[1], sp_[1] - q[2]) > 30 for q in det_now["objects"])), None)
                if spot is not None:
                    say(side, f"   the {colour} bin moved to the other arm's side - handing the block over at {spot}")
                    pp = plan_place(m, spot[0], spot[1], p["grasp"])
                    arm.move_to(with_grip(pp["hover"], hw.GRIP_CLOSED), TRANSFER_SPEED, tol=4, settle=0.1, timeout=2.0)
                    arm.move_to(with_grip(pp["place"], hw.GRIP_CLOSED), ZONE_SPEED * 1.5, tol=2.5, settle=0.1, timeout=2.5,
                                stall=True)
                    arm.move_to(with_grip(pp["place"], hw.GRIP_OPEN), TRANSFER_SPEED, tol=99, settle=0.0)
                    time.sleep(OPEN_WAIT)
                    arm.move_to(with_grip(pp["hover"], hw.GRIP_OPEN), TRANSFER_SPEED, tol=UP_TOL, settle=0.0, timeout=1.5)
                    arm.move_to(rest[side], TRANSFER_SPEED, tol=99, settle=0.1)
                    task["spot"] = spot
                    return "handed"
            if confirmed and math.hypot(b_now[0] - bin_xy[0], b_now[1] - bin_xy[1]) > 10:
                p2 = plan(o, m, b_now[:2])
                if p2.get("bin") is not None:
                    say(side, f"   the {colour} bin moved {math.hypot(b_now[0] - bin_xy[0], b_now[1] - bin_xy[1]):.0f} px"
                              f" - re-aiming at ({b_now[0]:.0f},{b_now[1]:.0f})")
                    p["bin"], p["bin_high"], p["bin_low"], p["slot"] = p2["bin"], p2.get("bin_high"), p2.get("bin_low"), p2.get("slot")
                    bin_xy = b_now[:2]
        lower = p.get("bin_high") is not None and np.abs(np.asarray(p["bin_high"]) - p["bin"]).max() > 0.5
        # 26 Nov pm: SAG. Every drop "stopped 3-6 deg short": stretched out over a bin, the arm sags under its own
        # weight + the block and lets go short of / lower than the bin pose. Learned per arm + bin: aim past the pose by
        # what it fell short last time (max SAG_MAX deg per joint), saved in sag_comp.json across runs.
        key = f"{side}:{p.get('slot') or colour}"     # sag learned per bin SPOT
        comp = np.array(sag.get(key, [0.0] * 4) + [0.0], float)
        aim = np.asarray(p["bin"], float) + comp
        if lower:     # a lower drop (e.g. left green): come in high first, then down to the drop pose
            arm.move_to(with_grip(np.asarray(p["bin_high"], float) + comp, hw.GRIP_CLOSED), TRANSFER_SPEED, tol=4,
                        settle=0.0, timeout=2.0)
            arm.move_to(with_grip(aim, hw.GRIP_CLOSED), ZONE_SPEED * 2, tol=3, settle=0.2, timeout=2.5, stall=True)
        else:
            arm.move_to(with_grip(aim, hw.GRIP_CLOSED), TRANSFER_SPEED, tol=3, settle=0.2, timeout=2.5, stall=True)
        q_at = arm.read()[:5]
        short = np.asarray(p["bin"], float)[:4] - q_at[:4]          # how far it really is from the bin pose
        err = float(np.abs(short).max())
        if SAG_LEARN and arm.read()[5] >= HOLD_MIN:
            sag[key] = [float(x) for x in np.clip(comp[:4] + 0.7 * short, -SAG_MAX, SAG_MAX)]
            save_sag(sag)
        target = aim
        if DROP_LOW and p.get("bin_low") is not None:
            target = p["bin_low"]
            err = arm.move_to(with_grip(target, hw.GRIP_CLOSED), ZONE_SPEED * 2, tol=2.5, settle=0.3, timeout=3.0, stall=True)
        jaw = arm.read()[5]
        say(side, f"   above the {colour} bin " + (f"(on the {p['slot']} bin's spot) " if p.get('slot') not in (None, colour) else "")
            + (f"(still {err:.1f} deg from the pose) " if err > 3.0 else "")
            + (f"[sag correction {np.round(comp[:4], 1).tolist()}] " if np.abs(comp).max() > 0.2 else "")
            + ("- opening" if jaw >= HOLD_MIN else f"- object LOST on the way (jaw {jaw:.1f})"))
        # 26 Nov pm: photo of EVERY drop (runlog\<time>_<side>_DROP_<colour>.jpg, bin centre marked) to see where it lets go
        bx = det["bins"].get(colour)
        if bx is not None:
            snap(f"{side}_DROP_{colour}", [(bx[0], bx[1], (0, 255, 255), f"{colour} bin centre")])
        arm.move_to(with_grip(target, hw.GRIP_OPEN), TRANSFER_SPEED, tol=99, settle=0.0)
        time.sleep(OPEN_WAIT)
        if target is not p["bin"]:
            arm.move_to(with_grip(p["bin"], hw.GRIP_OPEN), TRANSFER_SPEED, tol=99, settle=0.0)   # back up, clear of the rim
        if lower:
            arm.move_to(with_grip(p["bin_high"], hw.GRIP_OPEN), TRANSFER_SPEED, tol=99, settle=0.0)   # back up first
        arm.move_to(rest[side], TRANSFER_SPEED, tol=99, settle=0.1)
        return "done" if jaw >= HOLD_MIN else "miss"

    def worker(side):
        other = "right" if side == "left" else "left"
        idle_since, last_why = None, 0.0
        while not stop.is_set() and time.time() < t_end - FINISH_S:   # no new pick that can't finish in time
            top_img, det = look()
            with cond:
                busy = [active.get(other), want.get(other)]
                why = {}
                task, conflict = choose_task(side, det, maps, deferred, claimed.get(other), busy, spots.get(side),
                                             put_down, why=why)
                if task is None:
                    # 26 Nov pm (Peter: an arm sat watching while blocks were left): nothing normal to do -> look
                    # WIDER at once: blocks a little further outside the demo area (WIDE_MARGIN px instead of 25)
                    # and hand-off spots with less free space around them
                    why = {}
                    task, conflict = choose_task(side, det, maps, deferred, claimed.get(other), busy,
                                                 spots_wide.get(side), put_down, margin=WIDE_MARGIN, clear=25, why=why,
                                                 relax_shape=SHAPE_RULE)
                    if task is not None:
                        task["wide"] = True
                if task is None and idle_since is not None and time.time() - idle_since >= IDLE_MAX:
                    why = {}
                    task, conflict = choose_task(side, det, maps, deferred, claimed.get(other), busy,
                                                 spots_wide.get(side), put_down, margin=DESPERATE_MARGIN, clear=25,
                                                 why=why, relax_shape=SHAPE_RULE, desperate=True)
                    if task is not None:
                        task["wide"] = True
                        task["last_resort"] = True
                if task is None:
                    want[side] = None
                elif conflict:
                    # every job of this arm is in the other arm's strip: wait until that pick is over, then LOOK AGAIN
                    # (the other arm may have moved the very block)
                    if overlap(task["iv"], active.get(other)):
                        want[side] = task["iv"]
                        t_w = time.time()
                        cond.wait_for(lambda: not overlap(task["iv"], active.get(other)) or stop.is_set(), timeout=20)
                        stats["wait_s"] += time.time() - t_w
                    else:                                  # only the other arm's WAITING strip: let it go first
                        cond.wait_for(lambda: want.get(other) is None or stop.is_set(), timeout=1.0)
                    continue
                else:
                    active[side], want[side], claimed[side] = task["iv"], None, (task["o"][1], task["o"][2])
                    cond.notify_all()
            if task is None:
                # idle: look again every 0.4 s; say WHY every 5 s (so a block that is left can be explained)
                if idle_since is None or time.time() - last_why > 5.0:
                    if idle_since is None:
                        idle_since = time.time()
                        save_pic(top_img, f"{side}_nothing_to_sort", [(q[1], q[2], (255, 0, 255), q[0]) for q in det["objects"]])
                    last_why = time.time()
                    say(side, f"idle {time.time() - idle_since:.0f} s, nothing it can do (bins {sorted(maps[side].bins)}) | "
                              f"{len(det['objects'])} blocks seen: "
                              + (", ".join(f"{n} {k}" for k, n in sorted(why.items(), key=lambda t: -t[1])) or "none"))
                time.sleep(0.4)
                continue
            idle_since = None
            o = task["o"]
            try:
                result = pick(side, task, top_img, det)
            except Exception as e:
                say(side, f"ERROR {e!r} - this arm stops")
                stop.set()
                raise
            finally:
                # 26 Nov pm: keep the strip until this arm is REALLY back home (it lags its commands at speed) - only
                # then may the other arm move into the middle
                t_h = time.time()
                while not stop.is_set() and time.time() - t_h < 2.5:
                    try:
                        if np.abs(arms[side].read()[:3] - rest[side][:3]).max() < HOME_TOL:
                            break
                    except Exception:
                        break
                    time.sleep(0.03)
                with cond:
                    active[side], claimed[side] = None, None
                    cond.notify_all()
            try:
                r = dict(last_rec.get(side) or {}, result=result)
                with open(HERE / "runlog" / "picks.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(r) + chr(10))
            except Exception as e:
                say(side, f"   (could not write the pick record: {e!r})")
            with cond:
                if result in ("done", "handed"):           # that spot is clear now: forget its old misses
                    gone = [(o[1], o[2])] + ([task["spot"]] if result == "handed" else [])
                    for d in [d for d in deferred if any(math.hypot(d[0] - g[0], d[1] - g[1]) < 20 for g in gone)]:
                        del deferred[d]
                if result == "done":
                    stats["done"] += 1
                    say(side, f"sorted ({stats['done']} so far, both arms)")
                elif result == "handed":
                    stats["handed"] += 1
                    put_down.append(task["spot"])
                elif result == "moved":
                    pass                                     # Round 3: the judge moved it - not a miss
                else:
                    stats["miss"] += 1
                    key = next((d for d in deferred if math.hypot(o[1] - d[0], o[2] - d[1]) < 15), (o[1], o[2]))
                    deferred[key] = deferred.get(key, 0) + (99 if result == "blocked" else 1)
        say(side, "time is up (no new picks in the last %.0f s, so everything is finished at the end)" % FINISH_S if time.time() >= t_end - FINISH_S else "stopped")

    scan_thread = threading.Thread(target=scanner, daemon=True, name="camera")
    scan_thread.start()
    threads = [threading.Thread(target=worker, args=(s,), daemon=True, name=s) for s in arms]
    try:
        for t in threads:
            t.start()
            time.sleep(0.3)
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.2)
    except KeyboardInterrupt:
        print("\nSTOP pressed - arms back to rest")
        stop.set()
        with cond:
            cond.notify_all()
        for t in threads:
            t.join(timeout=8)
    finally:
        stop.set()
        for s, a in arms.items():
            try:
                a.move_to(rest[s], hw.MAX_SPEED, tol=99, timeout=3)
            except Exception:
                pass
            a.disconnect()
        scan_thread.join(timeout=2)
        cam.disconnect()
    print(f"\nSORTED {stats['done']} objects (both arms) | hand-offs {stats['handed']} | misses {stats['miss']} | "
          f"time spent waiting for the other arm {stats['wait_s']:.0f} s")


def plan_dry_both(img_path=None):
    """--dry for the two-arm version: what each arm would do first, no motors, no camera."""
    import cv2
    img = cv2.imread(str(img_path or HERE / "test_images" / "live_top.png"))
    maps = {s: model.load(s) for s in ("left", "right")}
    for s, mm in maps.items():
        if mm is not None:
            mm.roll_offset = PICK_ROLL[s]
    check_bins(maps)
    spots = {g: handoff_spots(maps, g, r) for g, r in (("left", "right"), ("right", "left"))}
    for g in spots:
        print(f"hand-off {g}: {len(spots[g])} spots, best {spots[g][:3]}")
    det = vision.detect(img)
    if SHAPE_RULE:
        import shape_model as SM
        det = classify_shapes(SM.ShapeNet(), [img], det)
        print("shapes:", [(o[7], o[8], f"{o[9]:.0%}", "->", o[0]) for o in det["objects"]])
    print("bins seen:", {k: (round(v[0]), round(v[1])) for k, v in det["bins"].items()})
    print("objects:", [(o[0], round(o[1]), round(o[2])) for o in det["objects"]])
    active = {"left": None, "right": None}
    for side in ("left", "right"):
        other = "right" if side == "left" else "left"
        t, conflict = choose_task(side, det, maps, {}, None, [active[other]], spots[side])
        if t is None:
            print(f"{side}: nothing")
            continue
        active[side] = t["iv"]
        o = t["o"]
        print(f"{side}: {t['kind']} {o[0]} at ({o[1]:.0f},{o[2]:.0f}) strip {np.round(t['iv'], 0)}"
              + (f" -> spot {t['spot']}" if t["kind"] == "handoff" else "") + (" (WAITS)" if conflict else " (at once)"))
        p = plan(o, maps[side], det["bins"][o[0]][:2])
        print(f"   grasp {np.round(p['grasp'], 1)} (demo height {np.round(p['grasp_demo'], 1)})")
        if t["kind"] == "handoff":
            pp = plan_place(maps[side], *t["spot"], p["grasp"])
            print(f"   place {np.round(pp['place'], 1)}")


if __name__ == "__main__":
    if DRY:
        plan_dry_both() if os.environ.get("SORT_BOTH", "1") == "1" else run_dry()
    elif os.environ.get("SORT_BOTH", "1") == "1":
        run_both()
    else:
        run()
