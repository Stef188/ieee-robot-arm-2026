"""Train the learned grasp-pose networks (GraspNet, model.py) for both arms and write the training log.

    python classical\\train_grasp_net.py          -> trains, saves grasp_net_<side>_<grasp|hover>.npz, writes
                                                    training_logs\\grasp_net_<date>.txt (+ .json)
    python classical\\train_grasp_net.py --quick  -> skip the leave-one-out evaluation (training only)

Data: calib_<side>.json, built by calib_from_demos.py from the teleoperated colour-sorting demonstrations
(the frame where the human closed the gripper on a block: top-camera block position -> arm joint angles).
Evaluation: leave-one-out - each grab is predicted by a network trained WITHOUT it, compared with the cubic
least-squares baseline (Poly) on the same split.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import model  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def loo_errors(m, Fit):
    errs = []
    for i in range(len(m.uv)):
        keep = np.arange(len(m.uv)) != i
        y = m.grasp[keep, :5].copy()
        y[:, 4] -= m.roll_sign * m.ang[keep]
        p = Fit(m.uv[keep], y)(m.uv[i])[0]
        p[4] += m.roll_sign * m.ang[i]
        errs.append(np.abs(p - m.grasp[i, :5]))
    return np.array(errs)


def main():
    quick = "--quick" in sys.argv
    logdir = HERE / "training_logs"
    logdir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    lines, report = [], {"date": stamp, "hyperparameters": {
        "hidden": model.GraspNet.HIDDEN, "steps": model.GraspNet.STEPS, "lr": model.GraspNet.LR,
        "weight_decay": model.GraspNet.WD, "ensemble": model.GraspNet.SEEDS, "optimizer": "Adam (full batch)",
        "loss": "MSE on per-joint standardised targets", "inputs": "(u-350)/100, (v-330)/100 top-camera px",
        "outputs": JOINTS}, "arms": {}}

    def say(t=""):
        print(t, flush=True)
        lines.append(t)

    say(f"GraspNet training {stamp}")
    say(json.dumps(report["hyperparameters"]))
    for side in ("left", "right"):
        calib = json.loads((HERE / f"calib_{side}.json").read_text())
        for which in ("grasp", "hover"):           # force a fresh train + save
            (HERE / f"grasp_net_{side}_{which}.npz").unlink(missing_ok=True)
        t0 = time.time()
        m = model.ArmMap(calib)
        demos = sorted({p.get("demo") for p in calib["points"] if p.get("demo") is not None})
        say(f"\n{side} arm: {len(m.uv)} grab samples from {len(demos)} demonstration episodes, roll sign "
            f"{m.roll_sign:+.0f}, trained in {time.time() - t0:.1f} s -> grasp_net_{side}_grasp.npz / _hover.npz "
            f"(model in use: {m.grasp_model})")
        y = m.grasp[:, :5].copy()
        y[:, 4] -= m.roll_sign * m.ang
        fit_err = np.abs(m.f_grasp(m.uv) - y)
        say(f"  training-set error (deg)  mean {fit_err.mean():.2f} | per joint "
            + ", ".join(f"{j} {e:.2f}" for j, e in zip(JOINTS, fit_err.mean(0))))
        arm = {"samples": int(len(m.uv)), "episodes": len(demos), "train_mean_err_deg": float(fit_err.mean())}
        if not quick:
            t0 = time.time()
            e_nn = loo_errors(m, model.GraspNet)
            e_po = loo_errors(m, model.Poly)
            say(f"  leave-one-out error (deg) NETWORK mean {e_nn.mean():.2f} median {np.median(e_nn.mean(1)):.2f} | "
                f"baseline cubic fit mean {e_po.mean():.2f} median {np.median(e_po.mean(1)):.2f}  ({time.time() - t0:.0f} s)")
            say("  per joint NETWORK  " + ", ".join(f"{j} {e:.2f}" for j, e in zip(JOINTS, e_nn.mean(0))))
            say("  per joint baseline " + ", ".join(f"{j} {e:.2f}" for j, e in zip(JOINTS, e_po.mean(0))))
            arm.update(loo_mean_deg_network=float(e_nn.mean()), loo_mean_deg_baseline=float(e_po.mean()),
                       loo_per_joint_network=e_nn.mean(0).round(3).tolist(),
                       loo_per_joint_baseline=e_po.mean(0).round(3).tolist())
        report["arms"][side] = arm
    (logdir / f"grasp_net_{stamp}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (logdir / f"grasp_net_{stamp}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nlog: training_logs\\grasp_net_{stamp}.txt")


if __name__ == "__main__":
    main()
