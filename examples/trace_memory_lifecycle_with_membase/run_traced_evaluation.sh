#!/usr/bin/env bash
# ================================================================
#  Unified Stage 3: Evaluation
#
#  Loads the per-user search execution graph, traces the
#  prompt -> LLM answer -> judge step, and saves the updated graph.
#
#  The search-results file is expected at:
#     <save_dir>/<top_k>_<start_idx>_<end_idx>.json
#  where <save_dir> is read from the chosen config and
#  start_idx/end_idx match the range used in Stage 2
#  (defaults: start_idx=0, end_idx=sample_size).
#
#  METHOD   : mem0 | evermemos | rag | long_context
#  DATASET  : locomo | longmemeval
#             (realmem is evaluated online during construction)
# ================================================================
METHOD="evermemos"
DATASET="locomo"

top_k=10
start_idx=0
end_idx=1                     # typically equal to construction's `sample_size`

qa_model="gpt-4.1-mini"
judge_model="claude-opus-4-5"
qa_batch_size=4
judge_batch_size=4
api_config_path="examples/trace_memory_lifecycle_with_membase/configs/api_config.json"
# ================================================================

set -euo pipefail
cd "$(dirname "$0")/../.."

example_dir="examples/trace_memory_lifecycle_with_membase"

config_path="${example_dir}/configs/${METHOD}_config.json"
if [[ ! -f "$config_path" ]]; then
    echo "Config not found: $config_path" >&2
    exit 1
fi

# ---- Dataset -> dataset type ----
case "$DATASET" in
  locomo)      dataset_type="LoCoMo"      ;;
  longmemeval) dataset_type="LongMemEval" ;;
  realmem)
    echo "DATASET='realmem' is evaluated online during construction;" >&2
    echo "the evaluation stage is not applicable." >&2
    exit 1
    ;;
  *) echo "Unknown DATASET: '$DATASET'" >&2; exit 1 ;;
esac

# ---- Output layout (single source of truth: config's save_dir) ----
save_dir=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['save_dir'])" "$config_path")
log_dir="${example_dir}/logs/${METHOD}_${DATASET}"
traced_data_save_dir="${save_dir}/traced_data"
search_results_path="${save_dir}/${top_k}_${start_idx}_${end_idx}.json"

if [[ ! -f "$search_results_path" ]]; then
    echo "Search results not found: $search_results_path" >&2
    echo "Run run_traced_search.sh first (and check top_k/start_idx/end_idx)." >&2
    exit 1
fi

mkdir -p "$log_dir" "$traced_data_save_dir"
log_file="${log_dir}/traced_evaluation.log"

nohup python memory_evaluation.py \
    --search-results-path "$search_results_path" \
    --dataset-type "$dataset_type" \
    --qa-model "$qa_model" \
    --judge-model "$judge_model" \
    --qa-batch-size "$qa_batch_size" \
    --judge-batch-size "$judge_batch_size" \
    --api-config-path "$api_config_path" \
    --traced-data-save-dir "$traced_data_save_dir" \
    --tracing \
    > "$log_file" 2>&1 &

echo $! > "${log_dir}/traced_evaluation.pid"
echo "Traced evaluation started. PID: $(cat "${log_dir}/traced_evaluation.pid")"
echo "Profile : ${METHOD} / ${DATASET}"
echo "Log     : $log_file"
echo "Results : $search_results_path"
echo "Traces  : $traced_data_save_dir"
