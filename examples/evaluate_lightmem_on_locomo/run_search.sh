#!/usr/bin/env bash
# Stage 2: Memory Retrieval for LightMem on LoCoMo.
# Please modify the variables below to fit your setup.
# ========================================================
memory_type="LightMem"
dataset_type="LoCoMo"
dataset_path="lightmem_output/LoCoMo_stage_1.json"
config_path="examples/evaluate_lightmem_on_locomo/lightmem_config.json"
num_workers=2
top_k=10
# ========================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

python memory_search.py \
    --memory-type "$memory_type" \
    --dataset-type "$dataset_type" \
    --dataset-path "$dataset_path" \
    --dataset-standardized \
    --config-path "$config_path" \
    --num-workers "$num_workers" \
    --top-k "$top_k"
