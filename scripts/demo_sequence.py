"""The bimanual sequence by the scripted experts, as one presentation video (out/sequence.mp4).

Stages, each started where the previous one left the scene:
1. left hand brings the cup while the right opens the cap and keeps it (at once)
2. right hand pours into the cup the left holds, keeping the cap (re-grip and lift by the rl_hold policy)
3. both put down: the bottle back where it started, the cup beside it (at once)
4. right hand puts the cap back on and closes it while the left holds the bottle (with --recap)

Each frame shows the outside view and the head camera side by side (video.Composer), with the
stage and what each hand is doing.

    .venv/bin/python scripts/demo_sequence.py [--seed N] [--recap]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dexhand_sim.tasks import BottleSteady, RecapHeld, run_experts  # noqa: E402
from dexhand_sim.video import Composer  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recap", action="store_true")
    ap.add_argument("--policy", type=Path, default=Path("outputs/rl_hold/final.zip"),
                    help="the rl_hold policy that re-grips and lifts the bottle keeping the cap")
    ap.add_argument("--stand", action="store_true", help="the fallback: the cap waits on its stand while a full grasp pours")
    args = ap.parse_args()
    if args.stand:
        return stand_sequence(args)
    recap = RecapHeld(policy=args.policy)  # builds every task of the sequence on one model
    pb, pc = recap.place_bottle, recap.place_cup
    pour = pb.pour
    cap, cup = pour.cap_task, pour.cup_task
    m = recap.model
    comp = Composer(m, recap.observer)
    frames: list[np.ndarray] = []
    rng = np.random.default_rng(args.seed)

    # Stage 1 is run here (not through prepare) to film it.
    from dexhand_sim.scene import set_home

    d = mujoco.MjData(m)
    set_home(m, d)
    cup.randomize(d, rng)
    cap.randomize(d, rng)
    mujoco.mj_forward(m, d)
    ep_cup, ep_cap = cup.begin(d), cap.begin(d)
    thread = ep_cap.extra["thread"]

    def film(stage: str, lines, control=lambda: "制御: 見本動作(スクリプト)") -> callable:
        def on_frame(step: int) -> None:
            frames.append(comp.frame(d, stage, lines() + [control()]))
        return on_frame

    def right_cap() -> str:
        if thread.state == "on":
            return f"右手: キャップを回す ({np.rad2deg(thread.turned):.0f}/180度)"
        return "右手: キャップを外して保持"

    ok1 = run_experts([(cup, ep_cup), (cap, ep_cap)],
                      film("段階1: 準備(左右同時)", lambda: ["左手: コップを注ぐ位置へ運ぶ", right_cap()]))
    ep_pour = pour.begin(d)
    ep_pour.extra.update(ep_cap.extra)
    policy_until = d.time + 3.0  # PourHeld.policy_lift: HoldLiftEnv.EPISODE_S
    ok2 = run_experts([(pour, ep_pour)], film("段階2: キャップを持ったまま注ぐ", lambda: [
        "左手: コップを保持", f"右手: 握り直して持ち上げ注ぐ ({pour.poured(ep_pour)}/{len(pour.bottle.particles)}粒)"],
        lambda: "制御: 強化学習の方策(握り直し)" if d.time < policy_until else "制御: 見本動作(スクリプト)"))
    eb, ec = pb.begin(d), pc.begin(d)
    eb.extra.update(ep_cap.extra)
    ec.extra.update(ep_cap.extra)
    ok3 = run_experts([(pb, eb), (pc, ec)], film("段階3: 置く(左右同時)", lambda: [
        "左手: コップを机に置いて離す", "右手: ボトルを元の位置に置く"]))
    results = {"準備(コップ, キャップ)": ok1, "注ぐ": ok2, "置く(ボトル, コップ)": ok3}
    if args.recap:
        steady = BottleSteady(model=m)
        steady.AFTER_ENGAGE = True
        er, es = recap.begin(d), steady.begin(d)
        er.extra.update(ep_cap.extra)
        es.extra.update(ep_cap.extra)
        closing = {"on": "締まっている", "released": "首へ運ぶ", "engaged": "首に乗せて回している", "closed": "閉まった"}
        results["閉める"], results["ボトルを押さえる"] = run_experts([(recap, er), (steady, es)], film("段階4: キャップを閉める", lambda: [
            "左手: ボトルを押さえる", f"右手: 持っているキャップを閉める ({closing[thread.state]})"]))
    for k, v in results.items():
        print(k, v)
    print(f"poured {pour.poured(ep_pour)}/{len(pour.bottle.particles)}")

    import imageio.v3 as iio

    OUT.mkdir(exist_ok=True)
    iio.imwrite(OUT / "sequence.mp4", np.stack(frames), fps=30, codec="libx264", plugin="pyav", out_pixel_format="yuv420p")
    print(f"video: out/sequence.mp4 ({len(frames) / 30:.1f} s)")


def stand_sequence(args) -> None:
    """Stages with the cap stand: prepare (cup and cap at once), cap onto the stand, full-grasp pour,
    both put down (at once), and the recap from the stand with --recap."""
    from dexhand_sim.scene import set_home
    from dexhand_sim.tasks import RecapFromStand

    recap = RecapFromStand()
    put, pour, pb, pc = recap.put, recap.pour, recap.place_bottle, recap.place_cup
    cap, cup = put.cap_task, put.cup_task
    m = recap.model
    comp = Composer(m, recap.observer)
    frames: list[np.ndarray] = []
    rng = np.random.default_rng(args.seed)
    d = mujoco.MjData(m)
    set_home(m, d)
    cup.randomize(d, rng)
    cap.randomize(d, rng)
    mujoco.mj_forward(m, d)
    ep_cup, ep_cap = cup.begin(d), cap.begin(d)
    thread = ep_cap.extra["thread"]

    def film(stage: str, lines):
        def on_frame(step: int) -> None:
            frames.append(comp.frame(d, stage, lines() + ["制御: 見本動作(スクリプト)"]))
        return on_frame

    def right_cap() -> str:
        if thread.state == "on":
            return f"右手: キャップを回す ({np.rad2deg(thread.turned):.0f}/180度)"
        return "右手: キャップを外して保持"

    results = {}
    results["準備"] = run_experts([(cup, ep_cup), (cap, ep_cap)],
                                film("段階1: 準備(左右同時)", lambda: ["左手: コップを注ぐ位置へ運ぶ", right_cap()]))
    e = put.begin(d)
    e.extra.update(ep_cap.extra)
    results["キャップを台へ"] = run_experts([(put, e)], film("段階2: キャップを台に置く", lambda: ["左手: コップを保持", "右手: キャップを台に置く"]))
    ep_pour = pour.begin(d)
    results["注ぐ"] = run_experts([(pour, ep_pour)], film("段階3: 注ぐ", lambda: [
        "左手: コップを保持", f"右手: ボトルを握って注ぐ ({pour.poured(ep_pour)}/{len(pour.bottle.particles)}粒)"]))
    eb, ec = pb.begin(d), pc.begin(d)
    eb.extra.update(ep_cap.extra)
    ec.extra.update(ep_cap.extra)
    results["置く"] = run_experts([(pb, eb), (pc, ec)], film("段階4: 置く(左右同時)", lambda: [
        "左手: コップを机に置いて離す", "右手: ボトルを元の位置に置く"]))
    if args.recap:
        from dexhand_sim.tasks import BottleSteady

        steady = BottleSteady(model=m)
        er, es = recap.begin(d), steady.begin(d)
        er.extra.update(ep_cap.extra)
        es.extra.update(ep_cap.extra)
        closing = {"on": "締まっている", "released": "外れている", "engaged": "首に乗せて回している", "closed": "閉まった"}
        results["閉める"], results["ボトルを押さえる"] = run_experts([(recap, er), (steady, es)], film("段階5: キャップを閉める", lambda: [
            "左手: ボトルを押さえる", f"右手: 台のキャップを戻して閉める ({closing[thread.state]}, {np.rad2deg(thread.closing):.0f}/90度)"]))
    for k, v in results.items():
        print(k, v)
    print(f"poured {pour.poured(ep_pour)}/{len(pour.bottle.particles)}")
    import imageio.v3 as iio

    OUT.mkdir(exist_ok=True)
    iio.imwrite(OUT / "sequence_stand.mp4", np.stack(frames), fps=30, codec="libx264", plugin="pyav", out_pixel_format="yuv420p")
    print(f"video: out/sequence_stand.mp4 ({len(frames) / 30:.1f} s)")


if __name__ == "__main__":
    main()
