"""Record scripted-expert demonstrations of one task as a LeRobot v3 dataset.

The observations are the head camera image and the joint states; the actions
are the joint targets. Episodes the expert fails are discarded.

    .venv/bin/python scripts/record_expert.py --task cup_bring --episodes 50 --root data/cup_bring
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dexhand_sim.dataset import Recorder  # noqa: E402
from dexhand_sim.tasks import TASKS, run_expert  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--policy", type=Path, help="the rl_hold policy, for pour_held (outputs/rl_hold/final.zip)")
    args = ap.parse_args()
    if args.root.exists():
        if not args.overwrite:
            sys.exit(f"{args.root} exists; pass --overwrite to replace it")
        shutil.rmtree(args.root)

    task = TASKS[args.task](policy=args.policy) if args.policy else TASKS[args.task]()
    rec = Recorder(task, args.root)
    rng = np.random.default_rng(args.seed)
    kept = tried = 0
    t0 = time.perf_counter()
    while kept < args.episodes:
        tried += 1
        ep = task.reset(rng)
        ok = run_expert(task, ep, rec.frame)
        if ok:
            rec.save()
            kept += 1
        else:
            rec.discard()
        print(f"episode {tried}: {'kept' if ok else 'discarded'} ({kept}/{args.episodes}, {time.perf_counter() - t0:.0f} s)", flush=True)
    rec.finish()
    print(f"dataset {args.root}: {rec.ds.num_episodes} episodes, {rec.ds.num_frames} frames; expert success {kept}/{tried}")


if __name__ == "__main__":
    main()
