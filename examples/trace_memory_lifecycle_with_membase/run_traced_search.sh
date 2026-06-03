#!/usr/bin/env bash
# ================================================================
#  Unified Stage 2: Memory Search
#
#  Loads the per-user construction execution graph, traces the
#  query -> retrieval step, and saves the updated graph.
#
#  The stage-1 dataset file is expected at:
#     <save_dir>/<DatasetType>_stage_1.json
#  where <save_dir> is read from the chosen config.
#
#  METHOD   : mem0 | evermemos | rag | long_context
#  DATASET  : locomo | longmemeval
#             (realmem does online evaluation during construction,
#              so search/evaluation stages are not applicable)
# ================================================================
METHOD="evermemos"
DATASET="locomo"

num_workers=1
top_k=10
# ================================================================

set -euo pipefail
cd "$(dirname "$0")/../.."

example_dir="examples/trace_memory_lifecycle_with_membase"

# ---- Method -> memory type ----
case "$METHOD" in
  mem0)         memory_type="Mem0"         ;;
  evermemos)    memory_type="EverMemOS"    ;;
  rag)          memory_type="NaiveRAG"     ;;
  long_context) memory_type="Long-Context" ;;
  *) echo "Unknown METHOD: '$METHOD'" >&2; exit 1 ;;
esac

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
    echo "DATASET='realmem' does online evaluation during construction;" >&2
    echo "the search stage is not applicable." >&2
    exit 1
    ;;
  *) echo "Unknown DATASET: '$DATASET'" >&2; exit 1 ;;
esac

# ---- Output layout (single source of truth: config's save_dir) ----
save_dir=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['save_dir'])" "$config_path")
log_dir="${example_dir}/logs/${METHOD}_${DATASET}"
traced_data_save_dir="${save_dir}/traced_data"
dataset_path="${save_dir}/${dataset_type}_stage_1.json"

if [[ ! -f "$dataset_path" ]]; then
    echo "Stage-1 dataset not found: $dataset_path" >&2
    echo "Run run_traced_construction.sh first." >&2
    exit 1
fi

mkdir -p "$log_dir" "$traced_data_save_dir"
log_file="${log_dir}/traced_search.log"

nohup python memory_search.py \
    --memory-type "$memory_type" \
    --dataset-type "$dataset_type" \
    --dataset-path "$dataset_path" \
    --dataset-standardized \
    --config-path "$config_path" \
    --num-workers "$num_workers" \
    --top-k "$top_k" \
    --traced-data-save-dir "$traced_data_save_dir" \
    --tracing \
    > "$log_file" 2>&1 &

echo $! > "${log_dir}/traced_search.pid"
echo "Traced search started. PID: $(cat "${log_dir}/traced_search.pid")"
echo "Profile : ${METHOD} / ${DATASET}"
echo "Log     : $log_file"
echo "Output  : $save_dir"
echo "Traces  : $traced_data_save_dir"
