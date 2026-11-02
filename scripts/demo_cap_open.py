"""Scripted cap opening with the right hand alone (tasks.CapOpen's expert), recorded to out/cap_open.mp4.

The motion follows how a person opened a bottle one-handed in a recorded
video, read by eye (see tasks.CapOpen): hold the neck and shoulder with the
middle, ring and little fingers, turn the cap between thumb and index in
strokes, then pinch the free cap and draw it up off the neck.

    .venv/bin/python scripts/demo_cap_open.py [--seed N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dexhand_sim.tasks import CapOpen, run_expert  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    task = CapOpen()
    ep = task.reset(np.random.default_rng(args.seed))
    renderer = mujoco.Renderer(task.model, 480, 640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [*ep.data.body("pet").xpos[:2], 0.88]
    camera.distance, camera.azimuth, camera.elevation = 0.30, -150.0, -25.0
    frames = []
    turned = []

    def on_frame(ep, action) -> None:
        renderer.update_scene(ep.data, camera=camera)
        frames.append(np.concatenate([renderer.render(), task.observer.head_image(ep.data)], axis=1))
        turned.append(np.rad2deg(ep.extra["thread"].turned))

    ok = run_expert(task, ep, on_frame)
    thread = ep.extra["thread"]
    off, up = task.cap_lift(ep)
    released_at = next((i for i, t in enumerate(turned) if t >= 179.9), None)
    print(f"cap turned {np.rad2deg(thread.turned):.0f} deg, released {thread.released}"
          + (f" at {released_at / 30:.1f} s" if released_at is not None else ""))
    print(f"cap {1000 * up:.1f} mm above where it started, {1000 * off:.1f} mm off the bottle's axis, "
          f"in the fingers {task.cap_in_fingers(ep)}")
    tilt = np.rad2deg(np.arccos(np.clip(ep.data.body("pet").xmat[8], -1, 1)))
    print(f"bottle tilt {tilt:.0f} deg")
    print("RESULT:", "SUCCESS" if ok else "FAILURE")

    import imageio.v3 as iio

    OUT.mkdir(exist_ok=True)
    iio.imwrite(OUT / "cap_open.mp4", np.stack(frames), fps=30, codec="libx264", plugin="pyav", out_pixel_format="yuv420p")
    print(f"video: out/cap_open.mp4 ({len(frames) / 30:.1f} s)")


if __name__ == "__main__":
    main()
