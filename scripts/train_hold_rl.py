"""Train the rl_hold policy with PPO (stable-baselines3) on parallel CPU environments.

    .venv/bin/python scripts/train_hold_rl.py --bank data/hold_bank.npz --envs 22 --steps 3000000 --out outputs/rl_hold
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_env(bank: str, seed: int):
    def thunk():
        from dexhand_sim.rl_hold import HoldLiftEnv

        return HoldLiftEnv(bank, seed=seed)
    return thunk


def main() -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--envs", type=int, default=22)
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    env = VecMonitor(SubprocVecEnv([make_env(args.bank, i) for i in range(args.envs)]))

    class Success(BaseCallback):
        """Logs the success rate and mean lift of the episodes finished since the last rollout."""

        def __init__(self) -> None:
            super().__init__()
            self.done: list[tuple[bool, float]] = []

        def _on_step(self) -> bool:
            for info, done in zip(self.locals["infos"], self.locals["dones"]):
                if done:
                    self.done.append((info.get("success", False), info.get("lift", 0.0)))
            return True

        def _on_rollout_end(self) -> None:
            if self.done:
                s = np.mean([x[0] for x in self.done])
                lift = np.mean([x[1] for x in self.done])
                self.logger.record("hold/success", s)
                self.logger.record("hold/lift_m", lift)
                print(f"steps {self.num_timesteps}: episodes {len(self.done)} success {100 * s:.0f}% lift {1000 * lift:.0f} mm", flush=True)
                self.done = []

    model = PPO("MlpPolicy", env, n_steps=256, batch_size=2816, n_epochs=8, learning_rate=3e-4, gamma=0.99,
                gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, policy_kwargs={"net_arch": [256, 256]},
                device="cpu", verbose=0, tensorboard_log=str(args.out / "tb"))
    model.learn(args.steps, callback=[Success(), CheckpointCallback(max(200_000 // args.envs, 1), str(args.out / "ckpt"), name_prefix="ppo")])
    model.save(args.out / "final")
    print("saved", args.out / "final.zip")


if __name__ == "__main__":
    main()
