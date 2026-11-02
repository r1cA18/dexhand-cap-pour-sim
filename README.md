# 5指ハンドの双腕ロボットによるキャップ開けと注ぎ(MuJoCo)

双腕ロボットの上半身の両腕にWuji Hand 2.1(5指20関節)を付けたMuJoCoのシミュレーションです。右手でペットボトルのキャップを親指と人差し指で回して外し、左手でコップを運び、右手で中身(粒30個)を注ぎます。方策の観測は頭部カメラ画像と関節角だけです。

## 構成

| パス | 内容 |
| --- | --- |
| `dexhand_sim/scene.py` | ロボットと両手とカメラを1つのMuJoCoモデルに組む |
| `dexhand_sim/props.py` `objects.py` | ボトル(ねじ状態を持つキャップ)とコップと粒と台 |
| `dexhand_sim/sensors.py` | 方策の観測(頭カメラ画像と関節角)と評価用の物体状態 |
| `dexhand_sim/ik.py` | 腕と指先の逆運動学 |
| `dexhand_sim/tasks.py` | 課題の定義、成功判定、スクリプトの見本動作 |
| `dexhand_sim/dataset.py` | 見本動作をLeRobotDataset v3で書き出す |
| `dexhand_sim/rl_hold.py` | 握り直しの強化学習環境(Gymnasium) |
| `dexhand_sim/video.py` | 外からの視点と頭カメラを並べた動画 |
| `scripts/` | 録画、学習、評価、動画作成の入口 |
| `outputs/rl_hold/final.zip` | 握り直しのPPO学習済み重み |
| `assets/` | ハンドのMJCF(MIT)。出典は`SOURCE.txt` |

## 準備

```bash
uv venv --python 3.12 .venv
uv pip install -e ".[train,rl]"
```

## 再現手順

スクリプトの見本動作(GPU不要):

```bash
.venv/bin/python scripts/demo_cap_open.py --seed 0          # キャップ開け -> out/cap_open.mp4
.venv/bin/python scripts/demo_sequence.py --seed 0          # キャップ開け、PPOで握り直し、持ったまま注ぐ、置く
.venv/bin/python scripts/demo_sequence.py --seed 0 --stand --recap   # キャップを台に置いて注ぎ、閉めるまで
```

模倣学習(ACT):

```bash
.venv/bin/python scripts/record_expert.py --task cap_open --episodes 55 --root data/cap_open
scripts/train_act.sh cap_open                                             # CUDA、batch 32、30,000 step
scripts/train_act.sh cap_open --policy.device=mps --batch_size=8 --steps=4000 \
  --policy.path=<学習済みのpretrained_model>                              # Macでの追加学習
.venv/bin/python scripts/eval_policy.py --task cap_open --policy outputs/act_cap_open/checkpoints/last/pretrained_model --episodes 10 --close
.venv/bin/python scripts/run_bimanual.py --cup-policy <cup_bringの重み> --pour-policy <pourの重み> --episodes 4
```

強化学習(PPO):

```bash
.venv/bin/python scripts/make_hold_bank.py --count 64 --out data/hold_bank.npz
.venv/bin/python scripts/train_hold_rl.py --bank data/hold_bank.npz --envs 22 --steps 3000000 --out outputs/rl_hold
```

## 結果

| 課題 | 方式 | 成功 |
| --- | --- | --- |
| `cup_bring`(左手) | ACT | 4/4 |
| `cap_open`(右手) | ACT | 5/10 |
| 両手の注ぎ(`cup_bring`の後に`pour`) | ACT 2つ | 4/4(30粒中23〜26粒) |
| 握り直して持ち上げる | PPO | 98% |
| 通し(キャップ開け、握り直し、注ぐ、置く) | スクリプト+PPO | 成功(30粒中20粒) |
