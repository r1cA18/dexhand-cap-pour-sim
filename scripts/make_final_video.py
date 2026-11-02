"""Cut the submission video (out/final/final.mp4) from clips already rendered under out/.

Every segment becomes 1280x720 at 30 fps: a title card, or a clip scaled into the top 1280x480
with a caption band under it. The segment list is SEGMENTS below; edit it and re-run.

    .venv/bin/python scripts/make_final_video.py
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
W, H, CLIP_H, FPS = 1280, 720, 480, 30
BOLD = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
REG = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
BG, FG, ACCENT, DIM = (18, 20, 26), (236, 238, 242), (255, 196, 92), (150, 156, 168)

# ("card", seconds, title, [lines]) or ("clip", path, caption title, [lines], start, duration, speed)
SEGMENTS: list[tuple] = [
    ("card", 6, "5指ハンドでキャップを開けて注ぐ", [
        "双腕ロボットの両手に20関節の5指ハンドを付けたシミュレーション",
        "右手: 親指と人差し指でキャップを回して外す",
        "左手: 同時にコップを注ぐ位置へ運ぶ",
        "右手: ボトルを傾けて中身(粒)を注ぐ"]),
    ("card", 7, "なぜ5指が必要か", [
        "ボトルを握ったまま 指先だけでキャップを回す",
        "2指グリッパでは手首ごと回すしかない",
        "観測は頭カメラの画像と自分の関節角だけ(物体の位置は使わない)"]),
    ("clip", "hand_capture_overlay.mp4", "1. 人の手順を読み取る", [
        "人がキャップを開ける動画をコマ送りで観察",
        "中指以下で首を握り 親指と人差し指の側面で挟んで逆向きに滑らせる"], 0, 11.8, 1.0),
    ("clip", "cap_open.mp4", "2. 見本動作(スクリプト)", [
        "親指と人差し指の3点をキャップ外周に置き 40度ずつずらす",
        "初期配置6/6でキャップが外れる (1.5倍速)"], 0, 14, 1.5),
    ("clip", "final/cap_open_ft4k/eval_cap_open_0.mp4", "3. 模倣学習(ACT)の方策: キャップ開け", [
        "見本55本で学習 入力は右の頭カメラ画像と関節角のみ",
        "学習に使っていない初期配置で5/10成功 (1.5倍速)"], 0, 16, 1.5),
    ("clip", "final/bimanual_ft/bimanual_1.mp4", "4. 模倣学習(ACT)の方策2つで両手の注ぎ", [
        "左手のコップ運び(8秒)の後 右手の注ぎ(16秒)へ切り替え 方策だけで実行",
        "学習に使っていない初期配置で4/4成功 30粒中23〜26粒"], 0, 24, 1.0),
    ("clip", "final/sequence_rl_seed0.mp4", "5. 強化学習(PPO)で握り直して 持ったまま注ぐ", [
        "見本動作で失敗した握り直し(3秒間)だけをPPOで学習 成功率98%",
        "キャップを指先に挟んだまま注いで置く 30粒中20粒"], 0, 37.2, 1.0),
    ("clip", "sequence_stand.mp4", "6. 全段の見本動作(スクリプト) キャップを閉めるまで", [
        "キャップを台に置いて注ぎ 左手でボトルを押さえて右手首を回して閉める",
        "(2倍速)"], 0, 36.8, 2.0),
    ("card", 9, "結果", [
        "ACT  キャップ開け 5/10   両手の注ぎ(コップ運び→注ぐ) 4/4",
        "PPO  キャップを挟んだまま握り直して持ち上げる 98%",
        "通し(スクリプト+PPO)  キャップ開け→持ったまま注ぐ→置く 成功",
        "観測は頭カメラ画像と関節角のみ  シミュレータ: MuJoCo 3.16"]),
]


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def card(path: Path, title: str, lines: list[str]) -> None:
    im = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(im)
    dr.text((90, 150), title, font=font(BOLD, 50), fill=ACCENT)
    y = 260
    for line in lines:
        dr.text((90, y), line, font=font(REG, 32), fill=FG)
        y += 56
    im.save(path)


def band(path: Path, title: str, lines: list[str]) -> None:
    im = Image.new("RGB", (W, H - CLIP_H), BG)
    dr = ImageDraw.Draw(im)
    dr.text((40, 22), title, font=font(BOLD, 34), fill=ACCENT)
    y = 82
    for line in lines:
        dr.text((40, y), line, font=font(REG, 26), fill=FG)
        y += 42
    im.save(path)


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def render(segments: list[tuple], out: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        parts = []
        for i, seg in enumerate(segments):
            part = tmp_dir / f"{i:02d}.mp4"
            png = tmp_dir / f"{i:02d}.png"
            if seg[0] == "card":
                _, seconds, title, lines = seg
                card(png, title, lines)
                run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", str(png), "-t", str(seconds),
                     "-r", str(FPS), "-pix_fmt", "yuv420p", "-c:v", "libx264", str(part)])
            else:
                _, src, title, lines, start, duration, speed = seg
                band(png, title, lines)
                vf = (f"[0:v]setpts=PTS/{speed},scale={W}:{CLIP_H}:force_original_aspect_ratio=decrease,"
                      f"pad={W}:{H}:(ow-iw)/2:0:color=0x12141a,fps={FPS}[v];[v][1:v]overlay=0:{CLIP_H}")
                run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(duration * speed),
                     "-i", str(OUT / src), "-i", str(png), "-filter_complex", vf, "-an",
                     "-pix_fmt", "yuv420p", "-c:v", "libx264", str(part)])
            parts.append(part)
        listing = tmp_dir / "list.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))
        run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
             "-c", "copy", str(out)])


def main() -> None:
    out = OUT / "final" / "final.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    render(SEGMENTS, out)
    print(out)


if __name__ == "__main__":
    main()
