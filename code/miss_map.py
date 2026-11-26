"""Miss map (26 Nov): every grab attempt in sort_log.txt (+ the saved versions' logs) -> where on the table each arm
grabs well / misses. Writes test_images\\miss_map.png and prints the worst zones (= where to use TEACH_GRAB).
    python classical\\miss_map.py
"""
import re
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
LOGS = [HERE / "sort_log.txt"] + sorted((HERE / "versions").glob("*/sort_log.txt"))
START1 = re.compile(r"^\[(\d\d:\d\d:\d\d)\] (\w+) object at \((\d+),(\d+)\).*-> (left|right) arm")
START2 = re.compile(r"^\[(\d\d:\d\d:\d\d) (LEFT|RIGHT)\s*\] (\w+) object at \((\d+),(\d+)\)")
GOOD = ("GRIPPED", "sorted (")
BAD = ("MISSED", "NO block", "DROPPED", "skipping this object", "not centred")


def attempts():
    out = []
    for log in LOGS:
        if not log.exists():
            continue
        run, cur = None, {}
        for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "===== SORT run" in line:
                run = line.split("SORT run")[1].split("|")[0].strip()
                wrist = None
                cur = {}
                continue
            if "fine alignment ON" in line:
                wrist = True
            m1, m2 = START1.match(line), START2.match(line)
            if m1 or m2:
                if m1:
                    t, colour, u, v, side = m1.groups()
                else:
                    t, side, colour, u, v = m2.groups()
                    side = side.lower()
                cur[side] = dict(run=run, t=t, side=side, colour=colour, u=int(u), v=int(v), ok=None,
                                 wrist=bool(wrist), log=log.parent.name)
                out.append(cur[side])
                continue
            tag = re.match(r"^\[\d\d:\d\d:\d\d (LEFT|RIGHT)\s*\]", line)
            side = tag.group(1).lower() if tag else (next(iter(cur)) if len(cur) == 1 else None)
            if tag is None and len(cur) > 1:
                side = list(cur)[-1]
            a = cur.get(side) if side else None
            if a is None or a["ok"] is not None:
                continue
            if any(k in line for k in GOOD):
                a["ok"] = True
            elif any(k in line for k in BAD):
                a["ok"] = False
    return [a for a in out if a["ok"] is not None]


def main():
    A = attempts()
    img = cv2.imread(str(HERE / "test_images" / "now_top.png"))
    if img is None:
        img = np.full((480, 640, 3), 235, np.uint8)
    img = (img * 0.55 + 255 * 0.45).astype(np.uint8)
    for a in A:
        col = (0, 170, 0) if a["ok"] else (0, 0, 230)
        if a["side"] == "left":
            cv2.circle(img, (a["u"], a["v"]), 5, col, 2)
        else:
            cv2.drawMarker(img, (a["u"], a["v"]), col, cv2.MARKER_SQUARE, 9, 2)
    cv2.putText(img, "green = grabbed, red = missed | circle = LEFT arm, square = RIGHT arm", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.imwrite(str(HERE / "test_images" / "miss_map.png"), img)
    print(f"{len(A)} grab attempts with a result, from {len(LOGS)} log(s)")
    for side in ("left", "right"):
        s = [a for a in A if a["side"] == side]
        if not s:
            continue
        ok = sum(a["ok"] for a in s)
        print(f"\n{side.upper()} arm: {ok}/{len(s)} grabbed ({100 * ok / len(s):.0f}%)")
        for c in sorted({a["colour"] for a in s}):
            cs = [a for a in s if a["colour"] == c]
            print(f"   {c:6s}: {sum(a['ok'] for a in cs)}/{len(cs)}")
        # zones: 40 px (~5 cm) squares
        z = {}
        for a in s:
            k = (a["u"] // 40 * 40, a["v"] // 40 * 40)
            z.setdefault(k, []).append(a["ok"])
        worst = sorted(((sum(not x for x in r), len(r), k) for k, r in z.items() if len(r) >= 2), reverse=True)[:5]
        print("   worst zones (top-camera px, 40x40 = ~5x5 cm): " + "; ".join(
            f"x {k[0]}-{k[0] + 40} y {k[1]}-{k[1] + 40}: {m}/{n} missed" for m, n, k in worst if m))
    print("\nmap: classical\\test_images\\miss_map.png")


if __name__ == "__main__":
    main()
