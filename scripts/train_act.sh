#!/usr/bin/env bash
# Train an ACT policy on one task's LeRobot v3 dataset with lerobot-train.
#
#   scripts/train_act.sh <task> [extra lerobot-train args...]
#
# Reads data/<task>, writes outputs/act_<task>. Hyperparameters are the
# ACT preset (chunk 30, kl_weight 50, lr 5e-5). Defaults suit an RTX 5090
# (batch 32, 30k steps, CUDA); on a Mac pass e.g.
#   --policy.device=mps --batch_size=8 --steps=2000
# Later arguments override earlier ones.
set -euo pipefail
task=${1:?usage: train_act.sh <task> [lerobot-train args...]}
shift
root=$(cd "$(dirname "$0")/.." && pwd)
exec "$root/.venv/bin/lerobot-train" \
  --dataset.repo_id="local/${task}" \
  --dataset.root="$root/data/${task}" \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.kl_weight=50 \
  --policy.optimizer_lr=5e-5 \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --batch_size=32 \
  --steps=30000 \
  --save_freq=5000 \
  --log_freq=100 \
  --num_workers=4 \
  --wandb.enable=false \
  --output_dir="$root/outputs/act_${task}" \
  --job_name="act_${task}" \
  "$@"
