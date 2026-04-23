#!/usr/bin/env bash
# Stage 1: Memory Construction for Mem0 on LoCoMo.
# It samples 2 trajectories from the full dataset and processes them
# with a single process and 2 workers.
# Please modify the variables below to fit your setup.
# ========================================================
memory_type="Mem0"
dataset_type="LoCoMo"
dataset_path="datasets/locomo/data/locomo10.json"
config_path="examples/evaluate_mem0_on_locomo/mem0_config.json"
num_workers=10
sample_size=10
log_dir="mem0_logs"
token_cost_file="token_cost_mem0"
# ========================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

[ ! -d "$log_dir" ] && mkdir -p "$log_dir"

log_file="${log_dir}/process_1.log"
[ ! -f "$log_file" ] && touch "$log_file"

nohup python memory_construction.py \
    --memory-type "$memory_type" \
    --dataset-type "$dataset_type" \
    --dataset-path "$dataset_path" \
    --config-path "$config_path" \
    --num-workers "$num_workers" \
    --sample-size "$sample_size" \
    --token-cost-save-filename "$token_cost_file" \
    > "$log_file" 2>&1 &
echo $! > "${log_dir}/process_1.pid"
