#!/usr/bin/env bash
# One-click full pipeline for LightMem benchmark:
#   uv sync -> Stage 1 (construction) -> Stage 2 (search) -> Stage 3 (evaluation)
#
# Usage:
#   cd benchmarks/lightmem
#   ./run_all.sh
#
# Public environment overrides:
#   LLM_BASE_URL, LLM_MODEL, API_KEY
#   EMBEDDING_BASE_URL, EMBEDDING_MODEL, EMBEDDING_DIMENSION, EMBEDDING_DEVICE
#   EMBEDDING_API_KEY (optional; defaults to API_KEY)
#   LLMLINGUA_MODEL_PATH, LLMLINGUA_DEVICE_MAP, LLMLINGUA_COMPRESS_RATE
#   MEMORY_MANAGER_NAME, EXTRACTION_MODE, ENABLE_OFFLINE_UPDATE
#   CONSTRUCT_QUEUE_TOP_K, CONSTRUCT_QUEUE_KEEP_TOP_N, CONSTRUCT_QUEUE_WORKERS
#   OFFLINE_UPDATE_SCORE_THRESHOLD, OFFLINE_UPDATE_WORKERS
#   LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_TOP_P
#   LLM_API_POOL_SIZE (default 16; OpenAI client pool size for Stage3 parallel requests)
#   DATASET_PATH, NUM_WORKERS, TOP_K, QA_BATCH_SIZE, JUDGE_BATCH_SIZE, SAMPLE_SIZE
#   TOKENIZER_PATH (optional; if unset, prefer <repo_root>/models/<LLM_MODEL>, else fallback to LLM_MODEL)
#   EMBEDDING_TOKENIZER_PATH (optional; if unset, prefer local embedding path, else EMBEDDING_MODEL)
#   HF_HUB_OFFLINE (defaults to 0 to allow first-time model download)
#   TRANSFORMERS_OFFLINE (defaults to HF_HUB_OFFLINE)
#   LITELLM_LOCAL_MODEL_COST_MAP (defaults to true to avoid remote cost-map fetch noise)
#   NOTE: if SAMPLE_SIZE is unset/empty, run on full dataset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ==========================
# Defaults (override by env)
# ==========================
LLM_BASE_URL="${LLM_BASE_URL:-http://10.77.110.187:8000/v1}"
LLM_MODEL="${LLM_MODEL:-qwen3.5-9b}"
API_KEY="${API_KEY:-EMPTY}"

EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://10.77.110.187:8001/v1}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-qwen3-embedding-8b}"
EMBEDDING_DIMENSION="${EMBEDDING_DIMENSION:-}"
EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-${API_KEY}}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cpu}"
EMBEDDING_MODEL_KWARGS_JSON="${EMBEDDING_MODEL_KWARGS_JSON:-{}}"

LLMLINGUA_MODEL_PATH="${LLMLINGUA_MODEL_PATH:-microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank}"
LLMLINGUA_DEVICE_MAP="${LLMLINGUA_DEVICE_MAP:-cpu}"
LLMLINGUA_USE_V2="${LLMLINGUA_USE_V2:-1}"
LLMLINGUA_COMPRESS_RATE="${LLMLINGUA_COMPRESS_RATE:-0.6}"
TOPIC_SEGMENTER_MODEL_NAME="${TOPIC_SEGMENTER_MODEL_NAME:-llmlingua-2}"

MEMORY_MANAGER_NAME="${MEMORY_MANAGER_NAME:-openai}"
EXTRACTION_MODE="${EXTRACTION_MODE:-flat}"
ENABLE_OFFLINE_UPDATE="${ENABLE_OFFLINE_UPDATE:-1}"
CONSTRUCT_QUEUE_TOP_K="${CONSTRUCT_QUEUE_TOP_K:-20}"
CONSTRUCT_QUEUE_KEEP_TOP_N="${CONSTRUCT_QUEUE_KEEP_TOP_N:-10}"
CONSTRUCT_QUEUE_WORKERS="${CONSTRUCT_QUEUE_WORKERS:-8}"
OFFLINE_UPDATE_SCORE_THRESHOLD="${OFFLINE_UPDATE_SCORE_THRESHOLD:-0.9}"
OFFLINE_UPDATE_WORKERS="${OFFLINE_UPDATE_WORKERS:-5}"

LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-16000}"
LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.0}"
LLM_TOP_P="${LLM_TOP_P:-0.1}"

PRE_COMPRESS="${PRE_COMPRESS:-1}"
TOPIC_SEGMENT="${TOPIC_SEGMENT:-1}"
PRECOMP_TOPIC_SHARED="${PRECOMP_TOPIC_SHARED:-1}"
MESSAGES_USE="${MESSAGES_USE:-user_only}"
METADATA_GENERATE="${METADATA_GENERATE:-1}"
TEXT_SUMMARY="${TEXT_SUMMARY:-1}"
EXTRACT_THRESHOLD="${EXTRACT_THRESHOLD:-0.1}"
INDEX_STRATEGY="${INDEX_STRATEGY:-embedding}"
RETRIEVE_STRATEGY="${RETRIEVE_STRATEGY:-embedding}"
UPDATE_MODE="${UPDATE_MODE:-offline}"
QDRANT_COLLECTION_NAME="${QDRANT_COLLECTION_NAME:-memory_entries}"
QDRANT_ON_DISK="${QDRANT_ON_DISK:-1}"

LLM_API_POOL_SIZE="${LLM_API_POOL_SIZE:-16}"

DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/datasets/locomo/data/locomo10.json}"
NUM_WORKERS="${NUM_WORKERS:-16}"
TOP_K="${TOP_K:-10}"
QA_BATCH_SIZE="${QA_BATCH_SIZE:-16}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-16}"
SAMPLE_SIZE="${SAMPLE_SIZE:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
EMBEDDING_TOKENIZER_PATH="${EMBEDDING_TOKENIZER_PATH:-}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-${HF_HUB_OFFLINE}}"
LITELLM_LOCAL_MODEL_COST_MAP="${LITELLM_LOCAL_MODEL_COST_MAP:-true}"

resolve_local_model_dir() {
  local model_ref="$1"
  local candidate=""
  local model_basename=""
  local model_flattened=""
  local model_ref_lower=""
  local model_basename_lower=""
  local model_flattened_lower=""

  if [[ -d "${model_ref}" ]]; then
    echo "${model_ref}"
    return 0
  fi

  candidate="${REPO_ROOT}/models/${model_ref}"
  if [[ -d "${candidate}" ]]; then
    echo "${candidate}"
    return 0
  fi

  model_basename="$(basename "${model_ref}")"
  candidate="${REPO_ROOT}/models/${model_basename}"
  if [[ -d "${candidate}" ]]; then
    echo "${candidate}"
    return 0
  fi

  model_flattened="${model_ref//\//--}"
  candidate="${REPO_ROOT}/models/${model_flattened}"
  if [[ -d "${candidate}" ]]; then
    echo "${candidate}"
    return 0
  fi

  model_ref_lower="${model_ref,,}"
  candidate="${REPO_ROOT}/models/${model_ref_lower}"
  if [[ -d "${candidate}" ]]; then
    echo "${candidate}"
    return 0
  fi

  model_basename_lower="${model_basename,,}"
  candidate="${REPO_ROOT}/models/${model_basename_lower}"
  if [[ -d "${candidate}" ]]; then
    echo "${candidate}"
    return 0
  fi

  model_flattened_lower="${model_flattened,,}"
  candidate="${REPO_ROOT}/models/${model_flattened_lower}"
  if [[ -d "${candidate}" ]]; then
    echo "${candidate}"
    return 0
  fi

  return 1
}

if [[ -z "${TOKENIZER_PATH}" ]]; then
  if LOCAL_TOKENIZER_PATH="$(resolve_local_model_dir "${LLM_MODEL}")"; then
    TOKENIZER_PATH="${LOCAL_TOKENIZER_PATH}"
  else
    TOKENIZER_PATH="${LLM_MODEL}"
  fi
fi

if [[ -z "${EMBEDDING_BASE_URL}" ]]; then
  if LOCAL_EMBEDDING_MODEL_PATH="$(resolve_local_model_dir "${EMBEDDING_MODEL}")"; then
    EMBEDDING_MODEL="${LOCAL_EMBEDDING_MODEL_PATH}"
  fi

  if [[ -z "${EMBEDDING_TOKENIZER_PATH}" ]]; then
    if [[ -d "${EMBEDDING_MODEL}" ]]; then
      EMBEDDING_TOKENIZER_PATH="${EMBEDDING_MODEL}"
    elif LOCAL_EMBEDDING_TOKENIZER_PATH="$(resolve_local_model_dir "${EMBEDDING_MODEL}")"; then
      EMBEDDING_TOKENIZER_PATH="${LOCAL_EMBEDDING_TOKENIZER_PATH}"
    else
      EMBEDDING_TOKENIZER_PATH="${EMBEDDING_MODEL}"
    fi
  fi
else
  if [[ -z "${EMBEDDING_TOKENIZER_PATH}" ]]; then
    if LOCAL_EMBEDDING_TOKENIZER_PATH="$(resolve_local_model_dir "${EMBEDDING_MODEL}")"; then
      EMBEDDING_TOKENIZER_PATH="${LOCAL_EMBEDDING_TOKENIZER_PATH}"
    else
      EMBEDDING_TOKENIZER_PATH="${EMBEDDING_MODEL}"
    fi
  fi
fi

if LOCAL_LLMLINGUA_MODEL_PATH="$(resolve_local_model_dir "${LLMLINGUA_MODEL_PATH}")"; then
  LLMLINGUA_MODEL_PATH="${LOCAL_LLMLINGUA_MODEL_PATH}"
fi

SAVE_DIR_REL="benchmarks/lightmem/output"
SAVE_DIR_ABS="${REPO_ROOT}/${SAVE_DIR_REL}"
RUN_DIR="${SCRIPT_DIR}/.run"
RUNTIME_CONFIG="${RUN_DIR}/lightmem_config.runtime.json"
RUNTIME_API_CONFIG="${RUN_DIR}/api_config.runtime.json"

extract_host() {
  local url="$1"
  echo "${url}" | sed -E 's#^[a-zA-Z]+://([^/:]+).*#\1#'
}

LLM_HOST="$(extract_host "${LLM_BASE_URL}")"
EMBEDDING_HOST="$(extract_host "${EMBEDDING_BASE_URL}")"
NO_PROXY_HOSTS="127.0.0.1,localhost"
if [[ -n "${LLM_HOST}" ]]; then
  NO_PROXY_HOSTS="${LLM_HOST},${NO_PROXY_HOSTS}"
fi
if [[ -n "${EMBEDDING_HOST}" ]]; then
  NO_PROXY_HOSTS="${EMBEDDING_HOST},${NO_PROXY_HOSTS}"
fi

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
  "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
  "LITELLM_LOCAL_MODEL_COST_MAP=${LITELLM_LOCAL_MODEL_COST_MAP}"
)

mkdir -p "${RUN_DIR}" "${SAVE_DIR_ABS}"

echo "==> Repo root: ${REPO_ROOT}"
echo "==> Benchmark dir: ${SCRIPT_DIR}"
echo "==> Dataset: ${DATASET_PATH}"
echo "==> LLM endpoint/model: ${LLM_BASE_URL} / ${LLM_MODEL}"
if [[ -n "${EMBEDDING_BASE_URL}" ]]; then
  echo "==> Embedding endpoint/model: ${EMBEDDING_BASE_URL} / ${EMBEDDING_MODEL}"
else
  echo "==> Embedding model: ${EMBEDDING_MODEL}"
fi
echo "==> LLMLingua model path: ${LLMLINGUA_MODEL_PATH}"
echo "==> Memory manager / extraction mode: ${MEMORY_MANAGER_NAME} / ${EXTRACTION_MODE}"
echo "==> Offline update enabled: ${ENABLE_OFFLINE_UPDATE}"
echo "==> Tokenizer path: ${TOKENIZER_PATH}"
echo "==> Embedding tokenizer path: ${EMBEDDING_TOKENIZER_PATH}"
echo "==> HF_HUB_OFFLINE: ${HF_HUB_OFFLINE}"
echo "==> TRANSFORMERS_OFFLINE: ${TRANSFORMERS_OFFLINE}"
echo "==> LITELLM_LOCAL_MODEL_COST_MAP: ${LITELLM_LOCAL_MODEL_COST_MAP}"
echo

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "ERROR: dataset file does not exist: ${DATASET_PATH}"
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not installed or not in PATH."
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is not installed or not in PATH."
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

for INT_VAR in \
  LLM_API_POOL_SIZE NUM_WORKERS TOP_K QA_BATCH_SIZE JUDGE_BATCH_SIZE \
  LLM_MAX_TOKENS CONSTRUCT_QUEUE_TOP_K CONSTRUCT_QUEUE_KEEP_TOP_N \
  CONSTRUCT_QUEUE_WORKERS OFFLINE_UPDATE_WORKERS; do
  VALUE="${!INT_VAR}"
  if ! [[ "${VALUE}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ${INT_VAR} must be a positive integer, got: ${VALUE}"
    exit 1
  fi
  if [[ "${VALUE}" -le 0 ]]; then
    echo "ERROR: ${INT_VAR} must be > 0, got: ${VALUE}"
    exit 1
  fi
done

if [[ -n "${EMBEDDING_DIMENSION}" ]]; then
  if ! [[ "${EMBEDDING_DIMENSION}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: EMBEDDING_DIMENSION must be a positive integer, got: ${EMBEDDING_DIMENSION}"
    exit 1
  fi
  if [[ "${EMBEDDING_DIMENSION}" -le 0 ]]; then
    echo "ERROR: EMBEDDING_DIMENSION must be > 0, got: ${EMBEDDING_DIMENSION}"
    exit 1
  fi
fi

for BOOL_VAR in PRE_COMPRESS TOPIC_SEGMENT PRECOMP_TOPIC_SHARED METADATA_GENERATE TEXT_SUMMARY QDRANT_ON_DISK LLMLINGUA_USE_V2 ENABLE_OFFLINE_UPDATE; do
  VALUE="${!BOOL_VAR}"
  if [[ "${VALUE}" != "0" && "${VALUE}" != "1" ]]; then
    echo "ERROR: ${BOOL_VAR} must be 0 or 1, got: ${VALUE}"
    exit 1
  fi
done

case "${MEMORY_MANAGER_NAME}" in
  openai|deepseek|vllm|ollama|transformers|vllm_offline)
    ;;
  *)
    echo "ERROR: unsupported MEMORY_MANAGER_NAME: ${MEMORY_MANAGER_NAME}"
    echo "       Supported values: openai, deepseek, vllm, ollama, transformers, vllm_offline"
    exit 1
    ;;
esac

case "${EXTRACTION_MODE}" in
  flat|event)
    ;;
  *)
    echo "ERROR: EXTRACTION_MODE must be flat or event, got: ${EXTRACTION_MODE}"
    exit 1
    ;;
esac

case "${MESSAGES_USE}" in
  user_only|assistant_only|hybrid)
    ;;
  *)
    echo "ERROR: MESSAGES_USE must be user_only, assistant_only, or hybrid, got: ${MESSAGES_USE}"
    exit 1
    ;;
esac

case "${INDEX_STRATEGY}" in
  embedding|context|hybrid)
    ;;
  *)
    echo "ERROR: INDEX_STRATEGY must be embedding, context, or hybrid, got: ${INDEX_STRATEGY}"
    exit 1
    ;;
esac

case "${RETRIEVE_STRATEGY}" in
  embedding|context|hybrid)
    ;;
  *)
    echo "ERROR: RETRIEVE_STRATEGY must be embedding, context, or hybrid, got: ${RETRIEVE_STRATEGY}"
    exit 1
    ;;
esac

case "${UPDATE_MODE}" in
  offline|online)
    ;;
  *)
    echo "ERROR: UPDATE_MODE must be offline or online, got: ${UPDATE_MODE}"
    exit 1
    ;;
esac

if ! [[ "${LLM_TEMPERATURE}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "ERROR: LLM_TEMPERATURE must be a non-negative number, got: ${LLM_TEMPERATURE}"
  exit 1
fi
if ! [[ "${LLM_TOP_P}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "ERROR: LLM_TOP_P must be a positive number in (0, 1], got: ${LLM_TOP_P}"
  exit 1
fi
if ! awk "BEGIN{exit !(${LLM_TOP_P} > 0 && ${LLM_TOP_P} <= 1)}"; then
  echo "ERROR: LLM_TOP_P must be in (0,1], got: ${LLM_TOP_P}"
  exit 1
fi
if ! [[ "${EXTRACT_THRESHOLD}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "ERROR: EXTRACT_THRESHOLD must be a non-negative number, got: ${EXTRACT_THRESHOLD}"
  exit 1
fi
if ! [[ "${LLMLINGUA_COMPRESS_RATE}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "ERROR: LLMLINGUA_COMPRESS_RATE must be a positive number in (0, 1], got: ${LLMLINGUA_COMPRESS_RATE}"
  exit 1
fi
if ! awk "BEGIN{exit !(${LLMLINGUA_COMPRESS_RATE} > 0 && ${LLMLINGUA_COMPRESS_RATE} <= 1)}"; then
  echo "ERROR: LLMLINGUA_COMPRESS_RATE must be in (0,1], got: ${LLMLINGUA_COMPRESS_RATE}"
  exit 1
fi
if ! [[ "${OFFLINE_UPDATE_SCORE_THRESHOLD}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "ERROR: OFFLINE_UPDATE_SCORE_THRESHOLD must be a number in [0, 1], got: ${OFFLINE_UPDATE_SCORE_THRESHOLD}"
  exit 1
fi
if ! awk "BEGIN{exit !(${OFFLINE_UPDATE_SCORE_THRESHOLD} >= 0 && ${OFFLINE_UPDATE_SCORE_THRESHOLD} <= 1)}"; then
  echo "ERROR: OFFLINE_UPDATE_SCORE_THRESHOLD must be in [0,1], got: ${OFFLINE_UPDATE_SCORE_THRESHOLD}"
  exit 1
fi

if [[ -z "${EMBEDDING_BASE_URL}" ]] && [[ "${HF_HUB_OFFLINE}" == "1" ]] && [[ ! -d "${EMBEDDING_MODEL}" ]]; then
  echo "ERROR: HF_HUB_OFFLINE=1 but EMBEDDING_MODEL is not a local directory: ${EMBEDDING_MODEL}"
  echo "       Please either:"
  echo "       1) set EMBEDDING_MODEL to a local model path, or"
  echo "       2) set HF_HUB_OFFLINE=0 (and TRANSFORMERS_OFFLINE=0) to allow downloading, or"
  echo "       3) set EMBEDDING_BASE_URL to use a remote embedding endpoint."
  if [[ -d "${REPO_ROOT}/models" ]]; then
    echo "       Local candidates under ${REPO_ROOT}/models:"
    find "${REPO_ROOT}/models" -mindepth 1 -maxdepth 2 -type d | sed 's#^#         - #'
  fi
  exit 1
fi

if [[ "${HF_HUB_OFFLINE}" == "1" ]] && [[ ! -d "${LLMLINGUA_MODEL_PATH}" ]]; then
  echo "ERROR: HF_HUB_OFFLINE=1 but LLMLINGUA_MODEL_PATH is not a local directory: ${LLMLINGUA_MODEL_PATH}"
  echo "       Please either:"
  echo "       1) set LLMLINGUA_MODEL_PATH to a local model path, or"
  echo "       2) set HF_HUB_OFFLINE=0 (and TRANSFORMERS_OFFLINE=0) to allow downloading."
  if [[ -d "${REPO_ROOT}/models" ]]; then
    echo "       Local candidates under ${REPO_ROOT}/models:"
    find "${REPO_ROOT}/models" -mindepth 1 -maxdepth 2 -type d | sed 's#^#         - #'
  fi
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

if [[ -n "${EMBEDDING_DIMENSION}" ]]; then
  EFFECTIVE_EMBEDDING_DIMENSION="${EMBEDDING_DIMENSION}"
  echo "==> [2/7] Use EMBEDDING_DIMENSION from env: ${EFFECTIVE_EMBEDDING_DIMENSION}"
elif [[ -n "${EMBEDDING_BASE_URL}" ]]; then
  echo "==> [2/7] Auto-detect embedding dimension from endpoint"
  DETECT_OUTPUT="$(
    cd "${SCRIPT_DIR}" && \
    EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL}" \
    EMBEDDING_API_KEY="${EMBEDDING_API_KEY}" \
    EMBEDDING_MODEL="${EMBEDDING_MODEL}" \
    "${PYTHON_ENV_PREFIX[@]}" uv run python - <<'PY'
import json
import os
import sys
import httpx
from openai import OpenAI

base_url = os.environ["EMBEDDING_BASE_URL"]
api_key = os.environ["EMBEDDING_API_KEY"]
model = os.environ["EMBEDDING_MODEL"]
client = OpenAI(
    base_url=base_url,
    api_key=api_key,
    http_client=httpx.Client(trust_env=False),
)

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
    echo "       Please set EMBEDDING_DIMENSION manually and retry."
    exit 1
  }
  EFFECTIVE_EMBEDDING_DIMENSION="$(echo "${DETECT_OUTPUT}" | sed -n 's/.*"dim"[[:space:]]*:[[:space:]]*\([0-9]\+\).*/\1/p' | tail -n 1)"
  if [[ -z "${EFFECTIVE_EMBEDDING_DIMENSION}" ]]; then
    echo "ERROR: unable to parse embedding dimension from output:"
    echo "${DETECT_OUTPUT}"
    echo "Please set EMBEDDING_DIMENSION manually and retry."
    exit 1
  fi
  echo "Detected embedding dim: ${EFFECTIVE_EMBEDDING_DIMENSION}"
else
  EFFECTIVE_EMBEDDING_DIMENSION="384"
  echo "==> [2/7] Use local embedding default dimension: ${EFFECTIVE_EMBEDDING_DIMENSION}"
fi

echo "==> [3/7] Generate runtime config files (no mutation to baseline files)"
cat > "${RUNTIME_CONFIG}" <<EOF_JSON
{
    "user_id": "guest",
    "save_dir": "${SAVE_DIR_REL}",
    "api_key": "${API_KEY}",
    "llm_base_url": "${LLM_BASE_URL}",
    "llm_model": "${LLM_MODEL}",
    "memory_manager_name": "${MEMORY_MANAGER_NAME}",
    "llm_max_tokens": ${LLM_MAX_TOKENS},
    "temperature": ${LLM_TEMPERATURE},
    "top_p": ${LLM_TOP_P},
    "embedding_base_url": $([[ -n "${EMBEDDING_BASE_URL}" ]] && printf '"%s"' "${EMBEDDING_BASE_URL}" || echo null),
    "embedding_api_key": $([[ -n "${EMBEDDING_API_KEY}" ]] && printf '"%s"' "${EMBEDDING_API_KEY}" || echo null),
    "embedding_model": "${EMBEDDING_MODEL}",
    "embedding_dimension": ${EFFECTIVE_EMBEDDING_DIMENSION},
    "embedding_device": "${EMBEDDING_DEVICE}",
    "embedding_model_kwargs": ${EMBEDDING_MODEL_KWARGS_JSON},
    "retrieve_limit_default": ${TOP_K},
    "qdrant_collection_name": "${QDRANT_COLLECTION_NAME}",
    "qdrant_on_disk": $([[ "${QDRANT_ON_DISK}" == "1" ]] && echo true || echo false),
    "pre_compress": $([[ "${PRE_COMPRESS}" == "1" ]] && echo true || echo false),
    "topic_segment": $([[ "${TOPIC_SEGMENT}" == "1" ]] && echo true || echo false),
    "precomp_topic_shared": $([[ "${PRECOMP_TOPIC_SHARED}" == "1" ]] && echo true || echo false),
    "messages_use": "${MESSAGES_USE}",
    "metadata_generate": $([[ "${METADATA_GENERATE}" == "1" ]] && echo true || echo false),
    "text_summary": $([[ "${TEXT_SUMMARY}" == "1" ]] && echo true || echo false),
    "extract_threshold": ${EXTRACT_THRESHOLD},
    "index_strategy": "${INDEX_STRATEGY}",
    "retrieve_strategy": "${RETRIEVE_STRATEGY}",
    "update_mode": "${UPDATE_MODE}",
    "extraction_mode": "${EXTRACTION_MODE}",
    "llmlingua_model_path": "${LLMLINGUA_MODEL_PATH}",
    "llmlingua_device_map": "${LLMLINGUA_DEVICE_MAP}",
    "llmlingua_use_v2": $([[ "${LLMLINGUA_USE_V2}" == "1" ]] && echo true || echo false),
    "llmlingua_compress_rate": ${LLMLINGUA_COMPRESS_RATE},
    "topic_segmenter_model_name": "${TOPIC_SEGMENTER_MODEL_NAME}",
    "enable_offline_update": $([[ "${ENABLE_OFFLINE_UPDATE}" == "1" ]] && echo true || echo false),
    "construct_queue_top_k": ${CONSTRUCT_QUEUE_TOP_K},
    "construct_queue_keep_top_n": ${CONSTRUCT_QUEUE_KEEP_TOP_N},
    "construct_queue_workers": ${CONSTRUCT_QUEUE_WORKERS},
    "offline_update_score_threshold": ${OFFLINE_UPDATE_SCORE_THRESHOLD},
    "offline_update_workers": ${OFFLINE_UPDATE_WORKERS}
}
EOF_JSON

API_KEYS_JSON="$(
  API_KEY="${API_KEY}" \
  LLM_API_POOL_SIZE="${LLM_API_POOL_SIZE}" \
  python3 - <<'PY'
import json
import os

n = int(os.environ["LLM_API_POOL_SIZE"])
print(json.dumps([os.environ["API_KEY"]] * n, ensure_ascii=False))
PY
)"

BASE_URLS_JSON="$(
  LLM_BASE_URL="${LLM_BASE_URL}" \
  LLM_API_POOL_SIZE="${LLM_API_POOL_SIZE}" \
  python3 - <<'PY'
import json
import os

n = int(os.environ["LLM_API_POOL_SIZE"])
print(json.dumps([os.environ["LLM_BASE_URL"]] * n, ensure_ascii=False))
PY
)"

cat > "${RUNTIME_API_CONFIG}" <<EOF_JSON
{
    "api_keys": ${API_KEYS_JSON},
    "base_urls": ${BASE_URLS_JSON}
}
EOF_JSON

echo "Runtime config: ${RUNTIME_CONFIG}"
echo "Runtime api config: ${RUNTIME_API_CONFIG}"
echo "LLM API pool size (Stage3): ${LLM_API_POOL_SIZE}"
echo

echo "==> [3/7] Stage 1 - memory construction"
STAGE1_CMD=(
  "${PYTHON_ENV_PREFIX[@]}" uv run --project "${SCRIPT_DIR}" python memory_construction.py
  --memory-type "LightMem"
  --dataset-type "LoCoMo"
  --dataset-path "${DATASET_PATH}"
  --config-path "${RUNTIME_CONFIG}"
  --sample-size "${EFFECTIVE_SAMPLE_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --tokenizer-path "${TOKENIZER_PATH}"
  --embedding-tokenizer-path "${EMBEDDING_TOKENIZER_PATH}"
  --token-cost-save-filename "${SAVE_DIR_ABS}/token_cost_lightmem"
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

echo "==> [4/7] Stage 2 - memory search"
STAGE2_CMD=(
  "${PYTHON_ENV_PREFIX[@]}" uv run --project "${SCRIPT_DIR}" python memory_search.py
  --memory-type "LightMem"
  --dataset-type "LoCoMo"
  --dataset-path "${STAGE1_DATASET}"
  --dataset-standardized
  --config-path "${RUNTIME_CONFIG}"
  --num-workers "${NUM_WORKERS}"
  --top-k "${TOP_K}"
  --start-idx "${START_IDX}"
  --end-idx "${END_IDX}"
  --token-cost-save-filename "${SAVE_DIR_ABS}/token_cost_lightmem"
  --tokenizer-path "${TOKENIZER_PATH}"
  --embedding-tokenizer-path "${EMBEDDING_TOKENIZER_PATH}"
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

echo "==> [5/7] Stage 3 - evaluation (metrics: f1 bleu llm_judge)"
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
TOKEN_COST_JSON="${SAVE_DIR_ABS}/token_cost_lightmem.json"
TOKEN_COST_SUMMARY="${SAVE_DIR_ABS}/token_cost_summary.md"

echo "==> [6/7] Token cost summary"
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
