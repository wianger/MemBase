#!/usr/bin/env bash
# Check LLM and embedding endpoints before running the full Naive-RAG benchmark.
#
# Usage:
#   cd benchmarks/naive-rag
#   ./check_endpoints.sh
#
# Public environment overrides:
#   LLM_BASE_URL, EMBEDDING_BASE_URL, LLM_MODEL, EMBEDDING_MODEL, API_KEY
#   TIMEOUT_SECONDS, CHAT_PROMPT, EMBEDDING_INPUT
set -euo pipefail

LLM_BASE_URL="${LLM_BASE_URL:-http://10.77.110.187:8000/v1}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://10.77.110.187:8001/v1}"
LLM_MODEL="${LLM_MODEL:-qwen3.5-9b}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-qwen3-embedding-8b}"
API_KEY="${API_KEY:-EMPTY}"

TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-20}"
CHAT_PROMPT="${CHAT_PROMPT:-你好！}"
EMBEDDING_INPUT="${EMBEDDING_INPUT:-这是一段需要转换为向量的文本。}"

extract_host() {
  local url="$1"
  echo "${url}" | sed -E 's#^[a-zA-Z]+://([^/:]+).*#\1#'
}

LLM_HOST="$(extract_host "${LLM_BASE_URL}")"
EMBEDDING_HOST="$(extract_host "${EMBEDDING_BASE_URL}")"
NO_PROXY_HOSTS="${LLM_HOST},${EMBEDDING_HOST},127.0.0.1,localhost"

CURL_PREFIX=(
  env
  -u http_proxy
  -u https_proxy
  -u HTTP_PROXY
  -u HTTPS_PROXY
  -u ALL_PROXY
  -u all_proxy
  "NO_PROXY=${NO_PROXY_HOSTS}"
  "no_proxy=${NO_PROXY_HOSTS}"
  curl
  --noproxy "*"
  --connect-timeout "${TIMEOUT_SECONDS}"
  --max-time "${TIMEOUT_SECONDS}"
  -sS
)

AUTH_HEADERS=()
if [[ -n "${API_KEY}" && "${API_KEY}" != "EMPTY" ]]; then
  AUTH_HEADERS=(-H "Authorization: Bearer ${API_KEY}")
fi

run_json_check() {
  local name="$1"
  local method="$2"
  local url="$3"
  local payload="${4:-}"

  local tmp_body
  tmp_body="$(mktemp)"
  local status

  if [[ "${method}" == "GET" ]]; then
    status="$(
      "${CURL_PREFIX[@]}" \
        -X GET \
        "${AUTH_HEADERS[@]}" \
        -H "Content-Type: application/json" \
        -o "${tmp_body}" \
        -w "%{http_code}" \
        "${url}"
    )"
  else
    status="$(
      "${CURL_PREFIX[@]}" \
        -X POST \
        "${AUTH_HEADERS[@]}" \
        -H "Content-Type: application/json" \
        -d "${payload}" \
        -o "${tmp_body}" \
        -w "%{http_code}" \
        "${url}"
    )"
  fi

  if [[ "${status}" =~ ^2 ]]; then
    local preview
    preview="$(head -c 240 "${tmp_body}" | tr '\n' ' ')"
    echo "✅ ${name} (HTTP ${status})"
    echo "   Response preview: ${preview}"
  else
    echo "❌ ${name} failed (HTTP ${status})"
    echo "   URL: ${url}"
    echo "   Response:"
    cat "${tmp_body}"
    rm -f "${tmp_body}"
    exit 1
  fi

  rm -f "${tmp_body}"
}

LLM_MODELS_URL="${LLM_BASE_URL%/}/models"
EMBEDDING_MODELS_URL="${EMBEDDING_BASE_URL%/}/models"
CHAT_URL="${LLM_BASE_URL%/}/chat/completions"
EMBEDDING_URL="${EMBEDDING_BASE_URL%/}/embeddings"

CHAT_PAYLOAD="$(cat <<EOF
{
  "model": "${LLM_MODEL}",
  "messages": [
    {"role": "user", "content": "${CHAT_PROMPT}"}
  ],
  "temperature": 0.0,
  "max_tokens": 64
}
EOF
)"

EMBEDDING_PAYLOAD="$(cat <<EOF
{
  "model": "${EMBEDDING_MODEL}",
  "input": "${EMBEDDING_INPUT}"
}
EOF
)"

echo "== Endpoint check =="
echo "LLM base URL      : ${LLM_BASE_URL}"
echo "Embedding base URL: ${EMBEDDING_BASE_URL}"
echo "LLM model         : ${LLM_MODEL}"
echo "Embedding model   : ${EMBEDDING_MODEL}"
echo "NO_PROXY          : ${NO_PROXY_HOSTS}"
echo

run_json_check "LLM /models" "GET" "${LLM_MODELS_URL}"
run_json_check "Embedding /models" "GET" "${EMBEDDING_MODELS_URL}"
run_json_check "LLM chat completion" "POST" "${CHAT_URL}" "${CHAT_PAYLOAD}"
run_json_check "Embedding vectorization" "POST" "${EMBEDDING_URL}" "${EMBEDDING_PAYLOAD}"

echo
echo "🎉 All endpoint checks passed."
