#!/usr/bin/env bash
# ================================================================
#  Unified Stage 1: Memory Construction
#
#  Pick a profile via the two variables below; everything else is
#  derived from a consistent layout:
#     examples/trace_memory_lifecycle_with_membase/configs/<METHOD>_config.json
#     examples/trace_memory_lifecycle_with_membase/data/<DatasetDir>/...
#     <save_dir>/...          (taken from the chosen config)
#
#  METHOD   : mem0 | evermemos | rag | long_context
#  DATASET  : locomo | realmem | longmemeval
# ================================================================
METHOD="evermemos"
DATASET="locomo"

num_workers=1
sample_size=1
seed=42
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

# ---- Dataset -> dataset type / path ----
case "$DATASET" in
  locomo)
    dataset_type="LoCoMo"
    dataset_path="${example_dir}/data/LoCoMo/locomo10.json"
    ;;
  realmem)
    dataset_type="RealMem"
    dataset_path="${example_dir}/data/RealMem"
    ;;
  longmemeval)
    dataset_type="LongMemEval"
    dataset_path="${example_dir}/data/LongMemEval/longmemeval_s_cleaned.json"
    ;;
  *) echo "Unknown DATASET: '$DATASET'" >&2; exit 1 ;;
esac

# ---- Output layout (single source of truth: config's save_dir) ----
save_dir=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['save_dir'])" "$config_path")
log_dir="${example_dir}/logs/${METHOD}_${DATASET}"
traced_data_save_dir="${save_dir}/traced_data"
token_cost_file="${save_dir}/token_cost_traced_${METHOD}"

mkdir -p "$log_dir" "$traced_data_save_dir"
log_file="${log_dir}/traced_construction.log"

# ---- Optional online-eval config (only used by RealMem) ----
extra_args=()
if [[ "$DATASET" == "realmem" ]]; then
    extra_args+=(--online-eval-config-path "${example_dir}/configs/realmem_eval_config.json")
fi

nohup python memory_construction.py \
    --memory-type "$memory_type" \
    --dataset-type "$dataset_type" \
    --dataset-path "$dataset_path" \
    --config-path "$config_path" \
    --num-workers "$num_workers" \
    --seed "$seed" \
    --sample-size "$sample_size" \
    --token-cost-save-filename "$token_cost_file" \
    --traced-data-save-dir "$traced_data_save_dir" \
    --rerun \
    --tracing \
    "${extra_args[@]}" \
    > "$log_file" 2>&1 &

echo $! > "${log_dir}/traced_construction.pid"
echo "Traced construction started. PID: $(cat "${log_dir}/traced_construction.pid")"
echo "Profile : ${METHOD} / ${DATASET}"
echo "Log     : $log_file"
echo "Output  : $save_dir"
echo "Traces  : $traced_data_save_dir"
