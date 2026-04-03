# Mem0 Benchmark (One-Click)

这个目录提供 Mem0 的独立 benchmark 资产：

- `pyproject.toml`: 使用 `uv` 管理独立 Python 环境
- `mem0_config.json`: 基线 memory 配置（不在运行时改写）
- `api_config.json`: 基线 API 配置（不在运行时改写）
- `check_endpoints.sh`: 端点连通性与模型可用性检查
- `run_all.sh`: 一键运行三阶段（构建、检索、评测）
- `summarize_token_cost.py`: token 统计汇总工具

## 快速开始

建议先检查端点，再跑全流程：

```bash
cd benchmarks/mem0
chmod +x check_endpoints.sh run_all.sh
./check_endpoints.sh
./run_all.sh
```

仅执行端点检查：

```bash
cd benchmarks/mem0
./check_endpoints.sh
```

默认行为：

- 数据集：`datasets/locomo/data/locomo10.json`
- LLM endpoint：`http://10.46.131.226:8000/v1`
- Embedding endpoint：`http://10.46.131.226:8001/v1`
- LLM model：`qwen3.5-0.8b`
- Embedding model：`qwen3-embedding-0.6b`
- Embedder provider：`lmstudio`（默认，避免 vLLM embedding 因 `dimensions` 参数触发 matryoshka 报错）
- API key：`EMPTY`
- `SAMPLE_SIZE` 未设置时自动跑全量数据集
- 评测指标：`f1 bleu llm_judge`

## 环境变量覆盖

`run_all.sh` 支持通过环境变量覆盖默认值：

- `LLM_BASE_URL`（默认 `http://10.46.131.226:8000/v1`）
- `EMBEDDING_BASE_URL`（默认 `http://10.46.131.226:8001/v1`）
- `LLM_MODEL`（默认 `qwen3.5-0.8b`）
- `EMBEDDING_MODEL`（默认 `qwen3-embedding-0.6b`）
- `EMBEDDER_PROVIDER`（默认 `lmstudio`，可选 `openai`）
- `ENABLE_RERANKER`（默认 `1`，设为 `0` 关闭）
- `RERANKER_PROVIDER`（默认 `llm_reranker`）
- `RERANKER_MODEL`（默认 `qwen3-reranker-8b`）
- `RERANKER_LLM_PROVIDER`（默认 `openai`，当前仅 `llm_reranker` 场景使用）
- `RERANKER_BASE_URL`（默认 `LLM_BASE_URL`，用于 `llm_reranker + openai`）
- `RERANKER_API_KEY`（默认 `API_KEY`）
- `RERANKER_TOP_K`（默认 `TOP_K`）
- `API_KEY`（默认 `EMPTY`）
- `DATASET_PATH`（默认 LoCoMo10）
- `NUM_WORKERS`（默认 `8`）
- `TOP_K`（默认 `10`）
- `QA_BATCH_SIZE`（默认 `4`）
- `JUDGE_BATCH_SIZE`（默认 `4`）
- `SAMPLE_SIZE`（可选；不设置则跑全量，设置后按采样量跑）
- `EMBEDDING_DIM`（可选；提供后跳过自动探测向量维度）
- `TOKENIZER_PATH`（可选；不设置时优先使用 `<repo_root>/models/<LLM_MODEL>`，否则回退到 `LLM_MODEL`）
- `HF_HUB_OFFLINE`（默认 `1`；设为 `0` 允许联网从 HuggingFace 拉 tokenizer）
- `LITELLM_LOCAL_MODEL_COST_MAP`（默认 `true`；设为 `false` 时 LiteLLM 会尝试联网更新模型 cost map）

`check_endpoints.sh` 支持：

- `LLM_BASE_URL`
- `EMBEDDING_BASE_URL`
- `LLM_MODEL`
- `EMBEDDING_MODEL`
- `API_KEY`
- `TIMEOUT_SECONDS`

示例：

```bash
cd benchmarks/mem0
TOP_K=20 \
SAMPLE_SIZE=5 \
./run_all.sh
```

全量 + 并发 8 示例：

```bash
cd benchmarks/mem0
NUM_WORKERS=8 ./run_all.sh
```

开启 reranker（使用 `qwen3-reranker-8b`）示例：

```bash
cd benchmarks/mem0
ENABLE_RERANKER=1 \
RERANKER_PROVIDER=llm_reranker \
RERANKER_MODEL=qwen3-reranker-8b \
RERANKER_BASE_URL=http://10.46.131.226:8002/v1 \
./run_all.sh
```

## 输出文件

默认输出目录：`benchmarks/mem0/output`

- Stage 1: `benchmarks/mem0/output/LoCoMo_stage_1.json`
- Stage 2: `benchmarks/mem0/output/<top_k>_0_<sample_size>.json`
- Stage 3: `benchmarks/mem0/output/<top_k>_0_<sample_size>_evaluation.json`
- Token cost: `benchmarks/mem0/output/token_cost_mem0.json`
- Token cost summary（自动生成）: `benchmarks/mem0/output/token_cost_summary.md`

## 代理处理说明

`run_all.sh` 内部对每次 Python 调用都使用以下前缀，确保绕过代理访问内网服务：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
NO_PROXY=10.46.131.226,127.0.0.1,localhost \
no_proxy=10.46.131.226,127.0.0.1,localhost \
...
```

## 维度探测说明

脚本会优先使用 `EMBEDDING_DIM`；如果未设置，会调用 embedding endpoint 做一次请求并自动读取向量维度填入运行时配置。  
若探测失败，脚本会报错退出并提示手动设置 `EMBEDDING_DIM`。

## vLLM Embedding 兼容性说明

当 embedding 服务是 vLLM 且模型不支持 matryoshka（例如 `qwen3-embedding-0.6b`）时，
Mem0 的 `openai` embedder 会在请求中携带 `dimensions` 参数，从而触发 400 报错。

为避免该问题，`run_all.sh` 默认使用 `EMBEDDER_PROVIDER=lmstudio`（仍走 OpenAI 兼容 API）：

- `embedder_provider = "lmstudio"`
- `embedding_config.lmstudio_base_url = <EMBEDDING_BASE_URL>`

如果你确认 embedding 服务支持 `dimensions`，可手动切回：

```bash
cd benchmarks/mem0
EMBEDDER_PROVIDER=openai ./run_all.sh
```

## Reranker 说明

`run_all.sh` 已支持可选 reranker 配置，默认开启（`ENABLE_RERANKER=1`）。  
当开启并使用 `RERANKER_PROVIDER=llm_reranker` 时，脚本会把 reranker 配置写入运行时 `mem0_config.runtime.json`，并在 Stage2 搜索时通过环境变量注入 `OPENAI_BASE_URL/OPENAI_API_KEY` 给 reranker 使用。

说明：

- 如果你当前还没有部署 reranker 服务，可临时关闭：`ENABLE_RERANKER=0 ./run_all.sh`。
- 如果 `ENABLE_RERANKER=1` 但对应服务未就绪，Stage2 可能出现 rerank 失败并回退原始检索结果（由 mem0 内部处理）。

## Token Cost 汇总工具

`run_all.sh` 在流程末尾会自动调用 `summarize_token_cost.py` 生成 `output/token_cost_summary.md`。

也可以手动执行：

```bash
cd benchmarks/mem0
uv run python summarize_token_cost.py \
  --input output/token_cost_mem0.json \
  --format markdown
```
