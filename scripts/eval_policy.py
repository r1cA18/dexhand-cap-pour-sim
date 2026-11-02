"""Run a trained policy closed-loop in the simulation and measure its success rate.

The policy sees what it was trained on (the head camera and its side's 27
joints) and its 27 outputs become that side's position targets, held for
1/FPS s each. Starts are drawn from a seed range the recordings did not use.

    .venv/bin/python scripts/eval_policy.py --task cup_bring \\
        --policy outputs/act_cup_bring/checkpoints/last/pretrained_model --episodes 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dexhand_sim.dataset import IMAGE_KEY  # noqa: E402
from dexhand_sim.tasks import FPS, TASKS  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"


def load_policy(path: Path, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=str(path), preprocessor_overrides={"device_processor": {"device": device}}
    )
    return policy, pre, post


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), required=True)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=10_000, help="first start seed (recordings use 0 upward)")
    ap.add_argument("--videos", type=int, default=3, help="episodes to write to out/")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--close", action="store_true", help="film the bottle's cap up close instead of the overview")
    args = ap.parse_args()

    from lerobot.utils.control_utils import predict_action

    policy, pre, post = load_policy(args.policy, args.device)
    task = TASKS[args.task]()
    m = task.model
    sub = round(1 / FPS / m.opt.timestep)
    renderer = mujoco.Renderer(m, 480, 640)
    results = []
    for i in range(args.episodes):
        ep = task.reset(np.random.default_rng(args.seed + i))
        policy.reset()
        frames = []
        camera: str | mujoco.MjvCamera = "overview"
        if args.close:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [*ep.data.body("pet").xpos[:2], 0.88]
            camera.distance, camera.azimuth, camera.elevation = 0.30, -150.0, -25.0
        t0 = time.perf_counter()
        for _ in range(int(task.seconds * FPS)):
            image = task.observer.head_image(ep.data)
            obs = {IMAGE_KEY: image, "observation.state": ep.state()}
            action = predict_action(obs, policy, torch.device(args.device), pre, post, use_amp=False, task=task.instruction)
            ep.data.ctrl[ep.actuators] = action.squeeze(0).cpu().numpy()
            for _ in range(sub):
                task.step(ep)
            if i < args.videos:
                renderer.update_scene(ep.data, camera=camera)
                frames.append(np.concatenate([renderer.render(), image], axis=1))
        ok = task.success(ep)
        results.append(ok)
        print(f"episode {i}: {'success' if ok else 'failure'} ({time.perf_counter() - t0:.0f} s)", flush=True)
        if frames:
            import imageio.v3 as iio

            OUT.mkdir(exist_ok=True)
            iio.imwrite(OUT / f"eval_{args.task}_{i}.mp4", np.stack(frames), fps=FPS, codec="libx264", plugin="pyav",
                        out_pixel_format="yuv420p")
    rate = sum(results) / len(results)
    print(f"RESULT {args.task}: {sum(results)}/{len(results)} successful ({100 * rate:.0f}%)")
    (OUT / f"eval_{args.task}.json").write_text(json.dumps({"policy": str(args.policy), "success": results}, indent=1))


if __name__ == "__main__":
    main()
