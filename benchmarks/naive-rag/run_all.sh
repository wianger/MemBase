#!/usr/bin/env bash
# One-click full pipeline for Naive-RAG benchmark:
#   uv sync -> Stage 1 (construction) -> Stage 2 (search) -> Stage 3 (evaluation)
#
# Usage:
#   cd benchmarks/naive-rag
#   ./run_all.sh
#
# Public environment overrides:
#   LLM_BASE_URL, EMBEDDING_BASE_URL, LLM_MODEL, EMBEDDING_MODEL, API_KEY
#   DATASET_PATH, NUM_WORKERS, TOP_K, QA_BATCH_SIZE, JUDGE_BATCH_SIZE, SAMPLE_SIZE
#   EMBEDDING_DIM (optional; skips auto-detection if set)
#   TOKENIZER_PATH (optional; if unset, prefer <repo_root>/models/<LLM_MODEL>, else fallback to LLM_MODEL)
#   HF_HUB_OFFLINE (defaults to 1 to avoid tokenizer download/network timeout)
#   LITELLM_LOCAL_MODEL_COST_MAP (defaults to true to avoid remote cost-map fetch noise)
#   NOTE: if SAMPLE_SIZE is unset/empty, run on full dataset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ==========================
# Defaults (override by env)
# ==========================
LLM_BASE_URL="${LLM_BASE_URL:-http://10.46.131.226:8000/v1}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://10.46.131.226:8001/v1}"
LLM_MODEL="${LLM_MODEL:-qwen3.5-0.8b}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-qwen3-embedding-0.6b}"
API_KEY="${API_KEY:-EMPTY}"

DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/datasets/locomo/data/locomo10.json}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TOP_K="${TOP_K:-10}"
QA_BATCH_SIZE="${QA_BATCH_SIZE:-4}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-4}"
SAMPLE_SIZE="${SAMPLE_SIZE:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
LITELLM_LOCAL_MODEL_COST_MAP="${LITELLM_LOCAL_MODEL_COST_MAP:-true}"

# If TOKENIZER_PATH is not explicitly provided, prefer local model files under
# <repo_root>/models/<llm_model>. Fallback to model name for litellm/tiktoken.
if [[ -z "${TOKENIZER_PATH}" ]]; then
  LOCAL_TOKENIZER_PATH="${REPO_ROOT}/models/${LLM_MODEL}"
  if [[ -d "${LOCAL_TOKENIZER_PATH}" ]]; then
    TOKENIZER_PATH="${LOCAL_TOKENIZER_PATH}"
  else
    TOKENIZER_PATH="${LLM_MODEL}"
  fi
fi

SAVE_DIR_REL="benchmarks/naive-rag/output"
SAVE_DIR_ABS="${REPO_ROOT}/${SAVE_DIR_REL}"
RUN_DIR="${SCRIPT_DIR}/.run"
RUNTIME_CONFIG="${RUN_DIR}/naive_rag_config.runtime.json"
RUNTIME_API_CONFIG="${RUN_DIR}/api_config.runtime.json"

NO_PROXY_HOSTS="10.46.131.226,127.0.0.1,localhost"
PYTHON_ENV_PREFIX=(
  env
  -u http_proxy
  -u https_proxy
  -u HTTP_PROXY
  -u HTTPS_PROXY
  -u ALL_PROXY
  -u all_proxy
  "NO_PROXY=${NO_PROXY_HOSTS}"
  "no_proxy=${NO_PROXY_HOSTS}"
  "HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
  "LITELLM_LOCAL_MODEL_COST_MAP=${LITELLM_LOCAL_MODEL_COST_MAP}"
)

mkdir -p "${RUN_DIR}" "${SAVE_DIR_ABS}"

echo "==> Repo root: ${REPO_ROOT}"
echo "==> Benchmark dir: ${SCRIPT_DIR}"
echo "==> Dataset: ${DATASET_PATH}"
echo "==> LLM endpoint/model: ${LLM_BASE_URL} / ${LLM_MODEL}"
echo "==> Embedding endpoint/model: ${EMBEDDING_BASE_URL} / ${EMBEDDING_MODEL}"
echo "==> Tokenizer path: ${TOKENIZER_PATH}"
echo "==> HF_HUB_OFFLINE: ${HF_HUB_OFFLINE}"
echo "==> LITELLM_LOCAL_MODEL_COST_MAP: ${LITELLM_LOCAL_MODEL_COST_MAP}"
echo

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "ERROR: dataset file does not exist: ${DATASET_PATH}"
  exit 1
fi

if [[ -n "${SAMPLE_SIZE}" ]] && ! [[ "${SAMPLE_SIZE}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: SAMPLE_SIZE must be a positive integer when provided, got: ${SAMPLE_SIZE}"
  exit 1
fi

if [[ -n "${SAMPLE_SIZE}" ]] && [[ "${SAMPLE_SIZE}" -le 0 ]]; then
  echo "ERROR: SAMPLE_SIZE must be > 0 when provided, got: ${SAMPLE_SIZE}"
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not installed or not in PATH."
  exit 1
fi

echo "==> [0/6] Resolve effective sample size"
if [[ -n "${SAMPLE_SIZE}" ]]; then
  EFFECTIVE_SAMPLE_SIZE="${SAMPLE_SIZE}"
  echo "Use SAMPLE_SIZE from env: ${EFFECTIVE_SAMPLE_SIZE}"
else
  DATASET_SIZE="$(
    cd "${SCRIPT_DIR}" && \
    DATASET_PATH="${DATASET_PATH}" \
    "${PYTHON_ENV_PREFIX[@]}" uv run python - <<'PY'
import json
import os
import sys

path = os.environ["DATASET_PATH"]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, list):
    size = len(data)
elif isinstance(data, dict) and isinstance(data.get("trajectories"), list):
    size = len(data["trajectories"])
else:
    print(
        "ERROR: unsupported dataset JSON format. "
        "Expected a raw list dataset or standardized dataset with 'trajectories'.",
        file=sys.stderr,
    )
    sys.exit(2)

if size <= 0:
    print("ERROR: dataset is empty.", file=sys.stderr)
    sys.exit(3)

print(size)
PY
  )" || {
    echo "ERROR: failed to infer dataset size from ${DATASET_PATH}."
    echo "       Please set SAMPLE_SIZE explicitly and retry."
    exit 1
  }
  EFFECTIVE_SAMPLE_SIZE="$(echo "${DATASET_SIZE}" | tail -n 1 | tr -d '[:space:]')"
  if [[ -z "${EFFECTIVE_SAMPLE_SIZE}" ]] || ! [[ "${EFFECTIVE_SAMPLE_SIZE}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: failed to parse inferred dataset size."
    echo "Output:"
    echo "${DATASET_SIZE}"
    echo "Please set SAMPLE_SIZE explicitly and retry."
    exit 1
  fi
  echo "SAMPLE_SIZE is unset -> run full dataset with inferred size: ${EFFECTIVE_SAMPLE_SIZE}"
fi
echo

echo "==> [1/7] uv sync"
(cd "${SCRIPT_DIR}" && uv sync)

if [[ -n "${EMBEDDING_DIM:-}" ]]; then
  RETRIEVER_DIM="${EMBEDDING_DIM}"
  echo "==> [2/7] Use EMBEDDING_DIM from env: ${RETRIEVER_DIM}"
else
  echo "==> [2/7] Auto-detect embedding dimension from endpoint"
  DETECT_OUTPUT="$(
    cd "${SCRIPT_DIR}" && \
    EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL}" \
    API_KEY="${API_KEY}" \
    EMBEDDING_MODEL="${EMBEDDING_MODEL}" \
    "${PYTHON_ENV_PREFIX[@]}" uv run python - <<'PY'
import json
import os
import sys
from openai import OpenAI

base_url = os.environ["EMBEDDING_BASE_URL"]
api_key = os.environ["API_KEY"]
model = os.environ["EMBEDDING_MODEL"]
client = OpenAI(base_url=base_url, api_key=api_key)

resp = client.embeddings.create(
    model=model,
    input="dimension probe",
)
vec = resp.data[0].embedding
if vec is None or len(vec) == 0:
    print("ERROR: empty embedding vector", file=sys.stderr)
    sys.exit(2)

print(json.dumps({"dim": len(vec)}))
PY
  )" || {
    echo "ERROR: failed to detect embedding dimension from ${EMBEDDING_BASE_URL}."
    echo "       Please set EMBEDDING_DIM manually and retry."
    exit 1
  }
  RETRIEVER_DIM="$(echo "${DETECT_OUTPUT}" | sed -n 's/.*"dim"[[:space:]]*:[[:space:]]*\([0-9]\+\).*/\1/p' | tail -n 1)"
  if [[ -z "${RETRIEVER_DIM}" ]]; then
    echo "ERROR: unable to parse embedding dimension from output:"
    echo "${DETECT_OUTPUT}"
    echo "Please set EMBEDDING_DIM manually and retry."
    exit 1
  fi
  echo "Detected embedding dim: ${RETRIEVER_DIM}"
fi

echo "==> [3/7] Generate runtime config files (no mutation to baseline files)"
cat > "${RUNTIME_CONFIG}" <<EOF
{
    "user_id": "guest",
    "save_dir": "${SAVE_DIR_REL}",
    "max_tokens": null,
    "num_overlap_msgs": 0,
    "message_separator": "\\n",
    "deferred": false,
    "llm_model": "${TOKENIZER_PATH}",
    "retriever_name_or_path": "openai:${EMBEDDING_MODEL}",
    "retriever_dim": ${RETRIEVER_DIM},
    "embedding_kwargs": {
        "api_key": "${API_KEY}",
        "base_url": "${EMBEDDING_BASE_URL}"
    }
}
EOF

cat > "${RUNTIME_API_CONFIG}" <<EOF
{
    "api_keys": ["${API_KEY}"],
    "base_urls": ["${LLM_BASE_URL}"]
}
EOF

echo "Runtime config: ${RUNTIME_CONFIG}"
echo "Runtime api config: ${RUNTIME_API_CONFIG}"
echo

echo "==> [4/7] Stage 1 - memory construction"
STAGE1_CMD=(
  "${PYTHON_ENV_PREFIX[@]}" uv run --project "${SCRIPT_DIR}" python memory_construction.py
  --memory-type "NaiveRAG"
  --dataset-type "LoCoMo"
  --dataset-path "${DATASET_PATH}"
  --config-path "${RUNTIME_CONFIG}"
  --sample-size "${EFFECTIVE_SAMPLE_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --tokenizer-path "${TOKENIZER_PATH}"
  --token-cost-save-filename "${SAVE_DIR_ABS}/token_cost_naive_rag"
)
(
  cd "${REPO_ROOT}" && \
  "${STAGE1_CMD[@]}"
)

TOKEN_COST_JSON="${SAVE_DIR_ABS}/token_cost_naive_rag.json"
if [[ -f "${TOKEN_COST_JSON}" ]] && grep -Eq '^[[:space:]]*\{[[:space:]]*\}[[:space:]]*$' "${TOKEN_COST_JSON}"; then
  echo "NOTE: ${TOKEN_COST_JSON} is {}."
  echo "      This is expected for NaiveRAG in current MemBase: token_cost tracks patched LLM calls,"
  echo "      while NaiveRAG construction mainly performs embedding/indexing without LLM-call patch specs."
fi

STAGE1_DATASET="${SAVE_DIR_ABS}/LoCoMo_stage_1.json"
if [[ ! -f "${STAGE1_DATASET}" ]]; then
  echo "ERROR: Stage 1 output dataset not found: ${STAGE1_DATASET}"
  exit 1
fi

# Search uses index range [start_idx, end_idx), and stage1 sampled size is SAMPLE_SIZE.
START_IDX=0
END_IDX="${EFFECTIVE_SAMPLE_SIZE}"

echo "==> [5/7] Stage 2 - memory search"
STAGE2_CMD=(
  "${PYTHON_ENV_PREFIX[@]}" uv run --project "${SCRIPT_DIR}" python memory_search.py
  --memory-type "NaiveRAG"
  --dataset-type "LoCoMo"
  --dataset-path "${STAGE1_DATASET}"
  --dataset-standardized
  --config-path "${RUNTIME_CONFIG}"
  --num-workers "${NUM_WORKERS}"
  --top-k "${TOP_K}"
  --start-idx "${START_IDX}"
  --end-idx "${END_IDX}"
  --token-cost-save-filename "${SAVE_DIR_ABS}/token_cost_naive_rag"
  --tokenizer-path "${TOKENIZER_PATH}"
)
(
  cd "${REPO_ROOT}" && \
  "${STAGE2_CMD[@]}"
)

SEARCH_RESULTS="${SAVE_DIR_ABS}/${TOP_K}_${START_IDX}_${END_IDX}.json"
if [[ ! -f "${SEARCH_RESULTS}" ]]; then
  echo "ERROR: Stage 2 output not found: ${SEARCH_RESULTS}"
  exit 1
fi

echo "==> [6/7] Stage 3 - evaluation (metrics: f1 bleu llm_judge)"
STAGE3_CMD=(
  "${PYTHON_ENV_PREFIX[@]}" uv run --project "${SCRIPT_DIR}" python memory_evaluation.py
  --search-results-path "${SEARCH_RESULTS}"
  --dataset-type "LoCoMo"
  --qa-model "${LLM_MODEL}"
  --judge-model "${LLM_MODEL}"
  --qa-batch-size "${QA_BATCH_SIZE}"
  --judge-batch-size "${JUDGE_BATCH_SIZE}"
  --api-config-path "${RUNTIME_API_CONFIG}"
  --metrics f1 bleu llm_judge
)
(
  cd "${REPO_ROOT}" && \
  "${STAGE3_CMD[@]}"
)

EVAL_RESULTS="${SEARCH_RESULTS%.json}_evaluation.json"
TOKEN_COST_JSON="${SAVE_DIR_ABS}/token_cost_naive_rag.json"
TOKEN_COST_SUMMARY="${SAVE_DIR_ABS}/token_cost_summary.md"

echo "==> [7/7] Token cost summary"
if [[ -f "${TOKEN_COST_JSON}" ]]; then
  SUMMARY_CMD=(
    "${PYTHON_ENV_PREFIX[@]}" uv run --project "${SCRIPT_DIR}" python "${SCRIPT_DIR}/summarize_token_cost.py"
    --input "${TOKEN_COST_JSON}"
    --format "markdown"
    --save "${TOKEN_COST_SUMMARY}"
  )
  (
    cd "${REPO_ROOT}" && \
    "${SUMMARY_CMD[@]}"
  )
else
  echo "WARNING: token cost file not found, skip summary: ${TOKEN_COST_JSON}"
fi

echo
echo "✅ Done."
echo "Stage 1 dataset: ${STAGE1_DATASET}"
echo "Stage 2 results: ${SEARCH_RESULTS}"
echo "Stage 3 results: ${EVAL_RESULTS}"
if [[ -f "${TOKEN_COST_SUMMARY}" ]]; then
  echo "Token cost summary: ${TOKEN_COST_SUMMARY}"
fi
