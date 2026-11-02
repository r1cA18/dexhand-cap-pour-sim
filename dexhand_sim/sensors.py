"""What a policy may observe: only what the real robot can sense.

- The head camera image (Intel RealSense D435 color, 640x480, at its design
  pose).
- The robot's own joint state: both arms, both hands, the lift and the neck.

Object poses, contacts and anything else the simulator knows are NOT
observations; they are used only to score an episode. There are no
wrist cameras.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .scene import ARM_JOINTS, HEAD_SHELL_GROUP, SIDES, Names

HEAD_CAMERA = "head"


def proprio_joint_names(model: mujoco.MjModel) -> list[str]:
    """Arm joints (left then right), hand joints (left then right), lift, neck pan, neck tilt."""
    names = [Names.arm_joint(side, j) for side in SIDES for j in ARM_JOINTS]
    for side in SIDES:
        prefix = Names.hand_prefix(side)
        names += [model.joint(i).name for i in range(model.njnt) if model.joint(i).name.startswith(prefix)]
    return names + ["lift", "neck_pan", "neck_tilt"]


class Observer:
    def __init__(self, model: mujoco.MjModel, width: int = 640, height: int = 480) -> None:
        self.model = model
        self.renderer = mujoco.Renderer(model, height, width)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[HEAD_SHELL_GROUP] = 0  # the camera looks out through the head's shell
        self.names = proprio_joint_names(model)
        self.qadr = np.array([model.joint(n).qposadr[0] for n in self.names])
        self.dadr = np.array([model.joint(n).dofadr[0] for n in self.names])

    def head_image(self, data: mujoco.MjData) -> np.ndarray:
        self.renderer.update_scene(data, camera=HEAD_CAMERA, scene_option=self.option)
        return self.renderer.render().copy()

    def observe(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        return {
            "observation.images.head": self.head_image(data),
            "observation.state": data.qpos[self.qadr].astype(np.float32),
            "observation.velocity": data.qvel[self.dadr].astype(np.float32),
        }
