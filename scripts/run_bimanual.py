"""Run the bimanual pour with trained policies only: the left hand brings the cup, then the right hand pours.

Each phase is one single-hand policy (scripts/train_act.sh) seeing what it
was trained on; the hand not in the phase holds its last targets. The switch
from the cup phase to the pour phase happens after cup_bring's time limit
(a fixed hand-over, standing in for a judge that looks at the head camera).

    .venv/bin/python scripts/run_bimanual.py \\
        --cup-policy outputs/act_cup_bring/checkpoints/last/pretrained_model \\
        --pour-policy outputs/act_pour/checkpoints/last/pretrained_model --episodes 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_policy import load_policy  # noqa: E402

from dexhand_sim.dataset import IMAGE_KEY  # noqa: E402
from dexhand_sim.tasks import CUP_GOAL, FPS, Episode, Pour, Task  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cup-policy", type=Path, required=True)
    ap.add_argument("--pour-policy", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20_000)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    from lerobot.utils.control_utils import predict_action

    pour = Pour()
    cup = pour.cup_task
    m = pour.model
    sub = round(1 / FPS / m.opt.timestep)
    phases = [(cup, *load_policy(args.cup_policy, args.device)), (pour, *load_policy(args.pour_policy, args.device))]
    renderer = mujoco.Renderer(m, 480, 640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.44, -0.06, 0.82]
    camera.distance, camera.azimuth, camera.elevation = 0.8, 160.0, -20.0
    results = []
    for i in range(args.episodes):
        # The pour scene's start (cup on the table, cap off), without the
        # scripted cup hand-over that Pour.reset adds for single-task runs.
        ep = Task.reset(pour, np.random.default_rng(args.seed + i))
        frames = []
        for task, policy, pre, post in phases:
            policy.reset()
            side = Episode(m, ep.data, task.joints, task.actuators, task.qadr)
            for _ in range(int(task.seconds * FPS)):
                image = task.observer.head_image(ep.data)
                obs = {IMAGE_KEY: image, "observation.state": side.state()}
                action = predict_action(obs, policy, torch.device(args.device), pre, post, use_amp=False, task=task.instruction)
                ep.data.ctrl[side.actuators] = action.squeeze(0).cpu().numpy()
                for _ in range(sub):
                    task.step(ep)
                renderer.update_scene(ep.data, camera=camera)
                frames.append(np.concatenate([renderer.render(), image], axis=1))
            if task is cup:
                held = cup.success(Episode(m, ep.data, cup.joints, cup.actuators, cup.qadr))
                print(f"episode {i}: cup {'at' if held else 'NOT at'} the pour position "
                      f"({1000 * np.linalg.norm(ep.data.body('cup').xpos - CUP_GOAL):.0f} mm off)", flush=True)
        ok = pour.success(ep)
        results.append(ok)
        print(f"episode {i}: {'success' if ok else 'failure'}, poured {pour.poured(ep)}/{len(pour.bottle.particles)}", flush=True)
        import imageio.v3 as iio

        OUT.mkdir(exist_ok=True)
        iio.imwrite(OUT / f"bimanual_{i}.mp4", np.stack(frames), fps=FPS, codec="libx264", plugin="pyav", out_pixel_format="yuv420p")
    print(f"RESULT bimanual: {sum(results)}/{len(results)} successful")


if __name__ == "__main__":
    main()
