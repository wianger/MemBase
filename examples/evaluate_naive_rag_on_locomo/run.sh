#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

construction_script="examples/evaluate_naive_rag_on_locomo/run_construction.sh"
search_script="examples/evaluate_naive_rag_on_locomo/run_search.sh"
evaluation_script="examples/evaluate_naive_rag_on_locomo/run_evaluation.sh"

log_dir="naive_rag_logs"
pid_file="${log_dir}/process_1.pid"
log_file="${log_dir}/process_1.log"
stage1_output="naive_rag_output/LoCoMo_stage_1.json"

bash "$construction_script"

if [[ ! -f "$pid_file" ]]; then
    echo "Stage 1 did not create PID file: $pid_file" >&2
    exit 1
fi

pid="$(cat "$pid_file")"

echo "Waiting for Stage 1 output: $stage1_output"
while [[ ! -f "$stage1_output" ]]; do
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "Stage 1 exited before producing $stage1_output" >&2
        if [[ -f "$log_file" ]]; then
            echo "--- Last 200 lines of $log_file ---" >&2
            tail -n 200 "$log_file" >&2
        fi
        exit 1
    fi
    sleep 5
done

wait "$pid" || {
    echo "Stage 1 failed. See $log_file for details." >&2
    exit 1
}

bash "$search_script"
bash "$evaluation_script"
