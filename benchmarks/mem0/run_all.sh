#!/usr/bin/env bash
# One-click full pipeline for Mem0 benchmark:
#   uv sync -> Stage 1 (construction) -> Stage 2 (search) -> Stage 3 (evaluation)
#
# Usage:
#   cd benchmarks/mem0
#   ./run_all.sh
#
# Public environment overrides:
#   LLM_BASE_URL, EMBEDDING_BASE_URL, LLM_MODEL, EMBEDDING_MODEL, API_KEY
#   EMBEDDER_PROVIDER (default lmstudio; optional openai)
#   ENABLE_RERANKER (default 1)
#   RERANKER_PROVIDER (default llm_reranker)
#   RERANKER_MODEL (default qwen3-reranker-8b)
#   RERANKER_LLM_PROVIDER (default openai; only for llm_reranker)
#   RERANKER_BASE_URL (default LLM_BASE_URL; only for llm_reranker + openai)
#   RERANKER_API_KEY (default API_KEY)
#   RERANKER_TOP_K (default TOP_K)
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
EMBEDDER_PROVIDER="${EMBEDDER_PROVIDER:-lmstudio}"

DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/datasets/locomo/data/locomo10.json}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TOP_K="${TOP_K:-10}"
QA_BATCH_SIZE="${QA_BATCH_SIZE:-4}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-4}"
ENABLE_RERANKER="${ENABLE_RERANKER:-1}"
RERANKER_PROVIDER="${RERANKER_PROVIDER:-llm_reranker}"
RERANKER_MODEL="${RERANKER_MODEL:-qwen3-reranker-8b}"
RERANKER_LLM_PROVIDER="${RERANKER_LLM_PROVIDER:-openai}"
RERANKER_BASE_URL="${RERANKER_BASE_URL:-${LLM_BASE_URL}}"
RERANKER_API_KEY="${RERANKER_API_KEY:-${API_KEY}}"
RERANKER_TOP_K="${RERANKER_TOP_K:-${TOP_K}}"
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

SAVE_DIR_REL="benchmarks/mem0/output"
SAVE_DIR_ABS="${REPO_ROOT}/${SAVE_DIR_REL}"
RUN_DIR="${SCRIPT_DIR}/.run"
RUNTIME_CONFIG="${RUN_DIR}/mem0_config.runtime.json"
RUNTIME_API_CONFIG="${RUN_DIR}/api_config.runtime.json"

PYTHON_ENV_PREFIX=(
  env
  "HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
  "LITELLM_LOCAL_MODEL_COST_MAP=${LITELLM_LOCAL_MODEL_COST_MAP}"
)

mkdir -p "${RUN_DIR}" "${SAVE_DIR_ABS}"

echo "==> Repo root: ${REPO_ROOT}"
echo "==> Benchmark dir: ${SCRIPT_DIR}"
echo "==> Dataset: ${DATASET_PATH}"
echo "==> LLM endpoint/model: ${LLM_BASE_URL} / ${LLM_MODEL}"
echo "==> Embedding endpoint/model: ${EMBEDDING_BASE_URL} / ${EMBEDDING_MODEL}"
echo "==> Embedder provider: ${EMBEDDER_PROVIDER}"
if [[ "${ENABLE_RERANKER}" == "1" ]]; then
  echo "==> Reranker: enabled (${RERANKER_PROVIDER})"
  echo "==> Reranker model/top-k: ${RERANKER_MODEL} / ${RERANKER_TOP_K}"
  if [[ "${RERANKER_PROVIDER}" == "llm_reranker" ]]; then
    echo "==> Reranker LLM provider/base URL: ${RERANKER_LLM_PROVIDER} / ${RERANKER_BASE_URL}"
  fi
else
  echo "==> Reranker: disabled"
fi
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

if [[ "${ENABLE_RERANKER}" != "0" && "${ENABLE_RERANKER}" != "1" ]]; then
  echo "ERROR: ENABLE_RERANKER must be 0 or 1, got: ${ENABLE_RERANKER}"
  exit 1
fi

if [[ "${ENABLE_RERANKER}" == "1" ]]; then
  case "${RERANKER_PROVIDER}" in
    cohere|sentence_transformer|zero_entropy|llm_reranker|huggingface)
      ;;
    *)
      echo "ERROR: unsupported RERANKER_PROVIDER: ${RERANKER_PROVIDER}"
      echo "       Supported values: cohere, sentence_transformer, zero_entropy, llm_reranker, huggingface"
      exit 1
      ;;
  esac

  if ! [[ "${RERANKER_TOP_K}" =~ ^[0-9]+$ ]] || [[ "${RERANKER_TOP_K}" -le 0 ]]; then
    echo "ERROR: RERANKER_TOP_K must be a positive integer, got: ${RERANKER_TOP_K}"
    exit 1
  fi

  # mem0ai's llm_reranker + openai provider reads base URL from OPENAI_BASE_URL env.
  if [[ "${RERANKER_PROVIDER}" == "llm_reranker" ]] && [[ "${RERANKER_LLM_PROVIDER}" != "openai" ]]; then
    echo "ERROR: in this script, llm_reranker currently only supports RERANKER_LLM_PROVIDER=openai."
    echo "       got RERANKER_LLM_PROVIDER=${RERANKER_LLM_PROVIDER}"
    exit 1
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not installed or not in PATH."
  exit 1
fi

echo "==> [0/7] Resolve effective sample size"
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
  EFFECTIVE_EMBEDDING_DIM="${EMBEDDING_DIM}"
  echo "==> [2/7] Use EMBEDDING_DIM from env: ${EFFECTIVE_EMBEDDING_DIM}"
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
  EFFECTIVE_EMBEDDING_DIM="$(echo "${DETECT_OUTPUT}" | sed -n 's/.*"dim"[[:space:]]*:[[:space:]]*\([0-9]\+\).*/\1/p' | tail -n 1)"
  if [[ -z "${EFFECTIVE_EMBEDDING_DIM}" ]]; then
    echo "ERROR: unable to parse embedding dimension from output:"
    echo "${DETECT_OUTPUT}"
    echo "Please set EMBEDDING_DIM manually and retry."
    exit 1
  fi
  echo "Detected embedding dim: ${EFFECTIVE_EMBEDDING_DIM}"
fi

echo "==> [3/7] Generate runtime config files (no mutation to baseline files)"

case "${EMBEDDER_PROVIDER}" in
  lmstudio)
    EMBEDDING_CONFIG_JSON="$(cat <<EOF
{
        "lmstudio_base_url": "${EMBEDDING_BASE_URL}",
        "api_key": "${API_KEY}"
    }
EOF
)"
    ;;
  openai)
    EMBEDDING_CONFIG_JSON="$(cat <<EOF
{
        "openai_base_url": "${EMBEDDING_BASE_URL}",
        "api_key": "${API_KEY}"
    }
EOF
)"
    ;;
  *)
    echo "ERROR: unsupported EMBEDDER_PROVIDER: ${EMBEDDER_PROVIDER}"
    echo "       Supported values: lmstudio, openai"
    exit 1
    ;;
esac

RERANKER_PROVIDER_JSON="null"
RERANKER_CONFIG_JSON="{}"
if [[ "${ENABLE_RERANKER}" == "1" ]]; then
  RERANKER_PROVIDER_JSON="\"${RERANKER_PROVIDER}\""
  case "${RERANKER_PROVIDER}" in
    llm_reranker)
      RERANKER_CONFIG_JSON="$(cat <<EOF
{
        "provider": "${RERANKER_LLM_PROVIDER}",
        "model": "${RERANKER_MODEL}",
        "api_key": "${RERANKER_API_KEY}",
        "top_k": ${RERANKER_TOP_K},
        "temperature": 0.0,
        "max_tokens": 64
    }
EOF
)"
      ;;
    *)
      RERANKER_CONFIG_JSON="$(cat <<EOF
{
        "model": "${RERANKER_MODEL}",
        "api_key": "${RERANKER_API_KEY}",
        "top_k": ${RERANKER_TOP_K}
    }
EOF
)"
      ;;
  esac
fi

cat > "${RUNTIME_CONFIG}" <<EOF
{
    "user_id": "guest",
    "save_dir": "${SAVE_DIR_REL}",
    "llm_provider": "openai",
    "llm_model": "${LLM_MODEL}",
    "llm_config": {
        "openai_base_url": "${LLM_BASE_URL}",
        "api_key": "${API_KEY}",
        "temperature": 0.0
    },
    "embedder_provider": "${EMBEDDER_PROVIDER}",
    "embedding_model": "${EMBEDDING_MODEL}",
    "embedding_model_dims": ${EFFECTIVE_EMBEDDING_DIM},
    "embedding_config": ${EMBEDDING_CONFIG_JSON},
    "reranker_provider": ${RERANKER_PROVIDER_JSON},
    "reranker_config": ${RERANKER_CONFIG_JSON},
    "graph_store_provider": "kuzu",
    "graph_store_config": {}
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
  --memory-type "Mem0"
  --dataset-type "LoCoMo"
  --dataset-path "${DATASET_PATH}"
  --config-path "${RUNTIME_CONFIG}"
  --sample-size "${EFFECTIVE_SAMPLE_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --tokenizer-path "${TOKENIZER_PATH}"
  --token-cost-save-filename "${SAVE_DIR_ABS}/token_cost_mem0"
)
(
  cd "${REPO_ROOT}" && \
  "${STAGE1_CMD[@]}"
)

STAGE1_DATASET="${SAVE_DIR_ABS}/LoCoMo_stage_1.json"
if [[ ! -f "${STAGE1_DATASET}" ]]; then
  echo "ERROR: Stage 1 output dataset not found: ${STAGE1_DATASET}"
  exit 1
fi

START_IDX=0
END_IDX="${EFFECTIVE_SAMPLE_SIZE}"

echo "==> [5/7] Stage 2 - memory search"
STAGE2_CMD=(
  "${PYTHON_ENV_PREFIX[@]}"
)
if [[ "${ENABLE_RERANKER}" == "1" ]] && [[ "${RERANKER_PROVIDER}" == "llm_reranker" ]]; then
  STAGE2_CMD+=(
    "OPENAI_BASE_URL=${RERANKER_BASE_URL}"
    "OPENAI_API_KEY=${RERANKER_API_KEY}"
  )
fi
STAGE2_CMD+=(
  uv run --project "${SCRIPT_DIR}" python memory_search.py
  --memory-type "Mem0"
  --dataset-type "LoCoMo"
  --dataset-path "${STAGE1_DATASET}"
  --dataset-standardized
  --config-path "${RUNTIME_CONFIG}"
  --num-workers "${NUM_WORKERS}"
  --top-k "${TOP_K}"
  --start-idx "${START_IDX}"
  --end-idx "${END_IDX}"
  --tokenizer-path "${TOKENIZER_PATH}"
  --token-cost-save-filename "${SAVE_DIR_ABS}/token_cost_mem0"
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
TOKEN_COST_JSON="${SAVE_DIR_ABS}/token_cost_mem0.json"
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
echo "Token cost: ${TOKEN_COST_JSON}"
if [[ -f "${TOKEN_COST_SUMMARY}" ]]; then
  echo "Token cost summary: ${TOKEN_COST_SUMMARY}"
fi
