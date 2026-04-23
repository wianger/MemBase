#!/usr/bin/env bash
# Stage 3: Question Answering & Evaluation for LightMem on LoCoMo.
# Please modify the variables below to fit your setup.
# ========================================================
search_results_path="lightmem_output/10_0_10.json"
dataset_type="LoCoMo"
qa_model="deepseek-chat"
judge_model="deepseek-chat"
qa_batch_size=4
judge_batch_size=4
api_config_path="examples/evaluate_lightmem_on_locomo/api_config.json"
prompt_template="examples/evaluate_lightmem_on_locomo/qa_prompt.py:get_lightmem_qa_prompt"
# ========================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

log_file="output/lightmem_logs/evaluation.log"
[ ! -f "$log_file" ] && touch "$log_file"

python memory_evaluation.py \
    --search-results-path "$search_results_path" \
    --dataset-type "$dataset_type" \
    --qa-model "$qa_model" \
    --judge-model "$judge_model" \
    --qa-batch-size "$qa_batch_size" \
    --judge-batch-size "$judge_batch_size" \
    --api-config-path "$api_config_path" \
    --prompt-template "$prompt_template" > "$log_file" 2>&1
