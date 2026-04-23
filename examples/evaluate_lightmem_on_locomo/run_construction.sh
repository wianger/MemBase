#!/usr/bin/env bash
# Stage 1: Memory Construction for LightMem on LoCoMo.
# Please modify the variables below to fit your setup.
# ========================================================
memory_type="LightMem"
dataset_type="LoCoMo"
dataset_path="dataset/locomo/data/locomo10.json"
config_path="examples/evaluate_lightmem_on_locomo/lightmem_config.json"
num_workers=1
sample_size=10
log_dir="output/lightmem_logs"
token_cost_file="${log_dir}/token_cost_lightmem"
# ========================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

[ ! -d "$log_dir" ] && mkdir -p "$log_dir"

log_file="${log_dir}/construction.log"
[ ! -f "$log_file" ] && touch "$log_file"

python memory_construction.py \
    --memory-type "$memory_type" \
    --dataset-type "$dataset_type" \
    --dataset-path "$dataset_path" \
    --config-path "$config_path" \
    --num-workers "$num_workers" \
    --sample-size "$sample_size" \
    --token-cost-save-filename "$token_cost_file" > "$log_file" 2>&1
