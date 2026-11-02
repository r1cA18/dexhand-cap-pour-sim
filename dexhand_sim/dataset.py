"""Write episodes as a LeRobot v3 dataset (lerobot 0.5).

One dataset per single-hand task. Per frame, at tasks.FPS:
- observation.images.base_0_rgb: the head camera (the conventional name for the base
  camera of the robot), 480x640 RGB, stored as video
- observation.state: that side's 27 joint positions (tasks.side_joints)
- action: the 27 position targets commanded at that frame
- task: the task's instruction string
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .tasks import FPS, Episode, Task

IMAGE_KEY = "observation.images.base_0_rgb"


def features(task: Task) -> dict:
    h, w = task.observer.renderer.height, task.observer.renderer.width
    names = list(task.joints)
    return {
        IMAGE_KEY: {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
        "action": {"dtype": "float32", "shape": (len(names),), "names": names},
    }


class Recorder:
    def __init__(self, task: Task, root: Path, repo_id: str | None = None, vcodec: str = "h264") -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.task = task
        self.ds = LeRobotDataset.create(
            repo_id=repo_id or f"local/{task.name}",
            fps=FPS,
            features=features(task),
            root=root,
            robot_type="dual_arm_wuji_hand2",
            use_videos=True,
            vcodec=vcodec,
        )

    def frame(self, ep: Episode, action: np.ndarray) -> None:
        self.ds.add_frame({
            IMAGE_KEY: self.task.observer.head_image(ep.data),
            "observation.state": ep.state(),
            "action": action.astype(np.float32),
            "task": self.task.instruction,
        })

    def save(self) -> None:
        self.ds.save_episode()

    def discard(self) -> None:
        self.ds.clear_episode_buffer()

    def finish(self) -> None:
        self.ds.finalize()
