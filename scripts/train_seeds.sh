#!/usr/bin/env bash
# Tier 2: retrain the 36 paper checkpoints (3 algos x 2 tasks x 2 planes x 3 seeds), sequentially.
# Idempotent: a checkpoint that already exists is skipped. Hours per run on an RTX 4090; see README.
set -euo pipefail
cd "$(dirname "$0")/.."
for algo in ppo sac td3; do for task in altitude attitude; do for plane in Airship_V7 Volantex_Ranger; do for seed in 0 1 2; do
  ck="checkpoints/${algo}_${task}_${plane}_s${seed}.pt"
  if [ -f "$ck" ]; then echo "SKIP $ck"; continue; fi
  echo "=== $algo $task $plane s$seed  $(date +%H:%M:%S)"
  .venv/bin/falcons train --algo "$algo" --task "$task" --plane "$plane" --seed "$seed"
done; done; done; done
