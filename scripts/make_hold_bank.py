"""Save start states for rl_hold: the scene at the end of the cap_open expert, one per seed that succeeds.

    .venv/bin/python scripts/make_hold_bank.py --count 64 --out data/hold_bank.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dexhand_sim.rl_hold import STATE_SPEC, build_rl_model  # noqa: E402
from dexhand_sim.tasks import CapOpen, run_expert  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=64)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    m = build_rl_model()
    task = CapOpen(model=m)
    states, seeds = [], []
    seed = args.first_seed
    t0 = time.perf_counter()
    while len(states) < args.count:
        ep = task.reset(np.random.default_rng(seed))
        if run_expert(task, ep):
            st = np.zeros(mujoco.mj_stateSize(m, STATE_SPEC))
            mujoco.mj_getState(m, ep.data, st, STATE_SPEC)
            states.append(st)
            seeds.append(seed)
        print(f"seed {seed}: {len(states)}/{args.count} ({time.perf_counter() - t0:.0f} s)", flush=True)
        seed += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, states=np.stack(states), seeds=np.array(seeds))
    print(f"saved {len(states)} states to {args.out}")


if __name__ == "__main__":
    main()
