#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

construction_script="examples/evaluate_simplemem_on_locomo/run_construction.sh"
search_script="examples/evaluate_simplemem_on_locomo/run_search.sh"
evaluation_script="examples/evaluate_simplemem_on_locomo/run_evaluation.sh"

bash "$construction_script"
bash "$search_script"
bash "$evaluation_script"
