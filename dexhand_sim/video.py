"""Presentation video frames: a wide outside view of the whole scene beside what the robot's camera sees.

Each frame is 1600x720:
- left, 1120x720: an outside view framing the robot, the table and both hands
- right, 480 wide: the head camera (the only camera the policies observe, as
  sensors.Observer renders it), with a caption, and below it a text panel
  with the current stage and what each hand is doing
"""

from __future__ import annotations

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .sensors import Observer

W, H = 1600, 720
OUT_W = 1120
SIDE_W = W - OUT_W


def _font(size: int) -> ImageFont.ImageFont:
    for path in ("/System/Library/Fonts/Hiragino Sans GB.ttc", "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
                 "/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


class Composer:
    def __init__(self, model: mujoco.MjModel, observer: Observer | None = None) -> None:
        self.model = model
        self.observer = observer or Observer(model)
        self.outside = mujoco.Renderer(model, H, OUT_W)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.42, -0.02, 0.83]
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 1.1, -150.0, -24.0
        self.font = _font(22)
        self.small = _font(18)

    def frame(self, data: mujoco.MjData, stage: str = "", lines: list[str] = ()) -> np.ndarray:
        self.outside.update_scene(data, camera=self.camera)
        canvas = Image.new("RGB", (W, H), (18, 18, 22))
        canvas.paste(Image.fromarray(self.outside.render()), (0, 0))
        head = Image.fromarray(self.observer.head_image(data)).resize((SIDE_W, SIDE_W * 3 // 4))
        canvas.paste(head, (OUT_W, 34))
        draw = ImageDraw.Draw(canvas)
        draw.text((OUT_W + 10, 6), "頭カメラ(方策が見る映像)", font=self.small, fill=(230, 230, 230))
        draw.text((12, 8), "外からの視点", font=self.small, fill=(230, 230, 230))
        y = 34 + SIDE_W * 3 // 4 + 16
        if stage:
            draw.text((OUT_W + 10, y), stage, font=self.font, fill=(255, 214, 102))
            y += 36
        for line in lines:
            draw.text((OUT_W + 10, y), line, font=self.small, fill=(220, 220, 220))
            y += 28
        draw.text((OUT_W + 10, H - 30), f"t = {data.time:5.1f} s", font=self.small, fill=(160, 160, 160))
        return np.asarray(canvas)
