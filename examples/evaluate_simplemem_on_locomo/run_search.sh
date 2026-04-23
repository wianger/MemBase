#!/usr/bin/env bash
# Stage 2: Memory Retrieval for SimpleMem on LoCoMo.
# Please modify the variables below to fit your setup.
# ========================================================
memory_type="SimpleMem"
dataset_type="LoCoMo"
dataset_path="simplemem_output/LoCoMo_stage_1.json"
config_path="examples/evaluate_simplemem_on_locomo/simplemem_config.json"
num_workers=2
top_k=10
# ========================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

python3 memory_search.py \
    --memory-type "$memory_type" \
    --dataset-type "$dataset_type" \
    --dataset-path "$dataset_path" \
    --dataset-standardized \
    --config-path "$config_path" \
    --num-workers "$num_workers" \
    --top-k "$top_k"

