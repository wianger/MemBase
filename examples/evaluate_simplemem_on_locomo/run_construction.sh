#!/usr/bin/env bash
# Stage 1: Memory Construction for SimpleMem on LoCoMo.
# Please modify the variables below to fit your setup.
# ========================================================
memory_type="SimpleMem"
dataset_type="LoCoMo"
dataset_path="YOUR_DATASET_PATH"
config_path="examples/evaluate_simplemem_on_locomo/simplemem_config.json"
num_workers=2
sample_size=2
log_dir="simplemem_logs"
token_cost_file="token_cost_simplemem"
# ========================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

[ ! -d "$log_dir" ] && mkdir -p "$log_dir"

log_file="${log_dir}/process_1.log"
[ ! -f "$log_file" ] && touch "$log_file"

nohup python3 memory_construction.py \
    --memory-type "$memory_type" \
    --dataset-type "$dataset_type" \
    --dataset-path "$dataset_path" \
    --config-path "$config_path" \
    --num-workers "$num_workers" \
    --sample-size "$sample_size" \
    --token-cost-save-filename "$token_cost_file" \
    > "$log_file" 2>&1 &
echo $! > "${log_dir}/process_1.pid"

