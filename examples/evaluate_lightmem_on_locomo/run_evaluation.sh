#!/usr/bin/env bash
# Stage 3: Question Answering & Evaluation for LightMem on LoCoMo.
# Please modify the variables below to fit your setup.
# ========================================================
search_results_path="lightmem_output/10_0_2.json"
dataset_type="LoCoMo"
qa_model="gpt-4.1-mini"
judge_model="gpt-4.1-mini"
qa_batch_size=4
judge_batch_size=4
api_config_path="examples/evaluate_lightmem_on_locomo/api_config.json"
# ========================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

python memory_evaluation.py \
    --search-results-path "$search_results_path" \
    --dataset-type "$dataset_type" \
    --qa-model "$qa_model" \
    --judge-model "$judge_model" \
    --qa-batch-size "$qa_batch_size" \
    --judge-batch-size "$judge_batch_size" \
    --api-config-path "$api_config_path"
