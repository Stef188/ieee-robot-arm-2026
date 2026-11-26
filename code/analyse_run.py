"""Report on the last sorter run(s) from classical\\runlog\\picks.jsonl (26 Nov).
    python classical\\analyse_run.py [n_runs]      (AFTER_RUN.bat runs this, then learn_from_runs.py)
Per run: picks per arm, grabs that held / missed / hit something, first tries vs retries, where on the table the arms
miss (camera grid), and in shape mode every placement with the model's guess and confidence.
"""
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent


def pct(a, b):
    return f"{a}/{b} ({a / b:.0%})" if b else "-"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    recs = [json.loads(l) for l in open(HERE / "runlog" / "picks.jsonl", encoding="utf-8") if l.strip()]
    runs = sorted(set(r["run"] for r in recs))[-n:]
    for run in runs:
        R = [r for r in recs if r["run"] == run]
        held = lambda r: r["result"] in ("done", "handed")
        print(f"\n===== run {run} ({R[0]['rule']} sort) - {len(R)} picks =====")
        print(f"   sorted into bins {sum(r['result'] == 'done' for r in R)}, handed over {sum(r['result'] == 'handed' for r in R)}"
              f", missed {sum(r['result'] == 'miss' for r in R)}, hit something {sum(r['result'] == 'blocked' for r in R)}")
        for side in ("left", "right"):
            S = [r for r in R if r["side"] == side]
            f1 = [r for r in S if int(r.get("tries") or 1) == 1]
            rt = [r for r in S if int(r.get("tries") or 1) > 1]
            print(f"   {side:5s}: grabs held {pct(sum(map(held, S)), len(S))} | first tries {pct(sum(map(held, f1)), len(f1))}"
                  f" | retries {pct(sum(map(held, rt)), len(rt))}")
        grid = collections.defaultdict(lambda: [0, 0])
        for r in R:
            k = (r["side"], int(r["u"] // 60) * 60, int(r["v"] // 40) * 40)
            grid[k][0 if held(r) else 1] += 1
        bad = sorted(((v[1], k) for k, v in grid.items() if v[1] >= 2 and v[0] == 0), reverse=True)
        if bad:
            print("   spots where an arm never held a block (camera x, y area):")
            for misses, (side, u, v) in bad[:6]:
                print(f"      {side:5s} x {u}-{u + 60}, y {v}-{v + 40}: {misses} misses")
        if R[0]["rule"] == "shape":
            placed = [r for r in R if held(r)]
            print("   shape placements (model guess, confidence -> bin):")
            for r in placed:
                print(f"      {r['side']:5s} {r.get('shape') or '?':17s} {(r.get('conf') or 0):.0%}  {r['colour']:6s} block -> {r['target']} bin")
            low = [r for r in placed if (r.get("conf") or 0) < 0.65]
            print(f"   {len(low)} of {len(placed)} placements were below 65 % sure (most likely to be in the wrong bin)")


if __name__ == "__main__":
    main()
