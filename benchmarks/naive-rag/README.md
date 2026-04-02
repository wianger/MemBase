# Naive-RAG Benchmark (One-Click)

这个目录提供 Naive-RAG 的独立 benchmark 资产：

- `pyproject.toml`: 使用 `uv` 管理独立 Python 环境
- `naive_rag_config.json`: 基线 memory 配置（不在运行时改写）
- `api_config.json`: 基线 API 配置（不在运行时改写）
- `check_endpoints.sh`: 端点连通性与模型可用性检查
- `run_all.sh`: 一键运行三阶段（构建、检索、评测）

## 快速开始

建议先检查端点，再跑全流程：

```bash
cd benchmarks/naive-rag
chmod +x check_endpoints.sh run_all.sh
./check_endpoints.sh
./run_all.sh
```

仅执行端点检查：

```bash
cd benchmarks/naive-rag
./check_endpoints.sh
```

默认行为：

- 数据集：`datasets/locomo/data/locomo10.json`
- LLM endpoint：`http://10.46.131.226:8000/v1`
- Embedding endpoint：`http://10.46.131.226:8001/v1`
- LLM model：`qwen3.5-0.8b`
- Embedding model：`qwen3-embedding-0.6b`
- API key：`EMPTY`
- `SAMPLE_SIZE` 未设置时自动跑全量数据集
- `HF_HUB_OFFLINE=1`（默认离线 tokenizer，避免访问 HuggingFace 超时）
- 评测指标：`f1 bleu llm_judge`

## 环境变量覆盖

`run_all.sh` 支持通过环境变量覆盖默认值：

- `LLM_BASE_URL`（默认 `http://10.46.131.226:8000/v1`）
- `EMBEDDING_BASE_URL`（默认 `http://10.46.131.226:8001/v1`）
- `LLM_MODEL`（默认 `qwen3.5-0.8b`）
- `EMBEDDING_MODEL`（默认 `qwen3-embedding-0.6b`）
- `API_KEY`（默认 `EMPTY`）
- `DATASET_PATH`（默认 LoCoMo10）
- `NUM_WORKERS`（默认 `2`）
- `TOP_K`（默认 `10`）
- `QA_BATCH_SIZE`（默认 `4`）
- `JUDGE_BATCH_SIZE`（默认 `4`）
- `SAMPLE_SIZE`（可选；不设置则跑全量，设置后按采样量跑）
- `EMBEDDING_DIM`（可选；提供后跳过自动探测向量维度）
- `TOKENIZER_PATH`（可选；不设置时优先使用 `<repo_root>/models/<LLM_MODEL>`，否则回退到 `LLM_MODEL`）
- `HF_HUB_OFFLINE`（默认 `1`；设为 `0` 允许联网从 HuggingFace 拉 tokenizer）

`check_endpoints.sh` 支持：

- `LLM_BASE_URL`
- `EMBEDDING_BASE_URL`
- `LLM_MODEL`
- `EMBEDDING_MODEL`
- `API_KEY`
- `TIMEOUT_SECONDS`

示例：

```bash
cd benchmarks/naive-rag
LLM_MODEL=qwen3.5-7b \
TOP_K=20 \
SAMPLE_SIZE=5 \
./run_all.sh
```

全量 + 并发 8 示例：

```bash
cd benchmarks/naive-rag
NUM_WORKERS=8 ./run_all.sh
```

如果你本机有本地 tokenizer 目录，建议显式指定（最稳）：

```bash
cd benchmarks/naive-rag
NUM_WORKERS=8 \
TOKENIZER_PATH=/path/to/local/tokenizer \
./run_all.sh
```

## 输出文件

默认输出目录：`benchmarks/naive-rag/output`

- Stage 1: `benchmarks/naive-rag/output/LoCoMo_stage_1.json`
- Stage 2: `benchmarks/naive-rag/output/<top_k>_0_<sample_size>.json`
- Stage 3: `benchmarks/naive-rag/output/<top_k>_0_<sample_size>_evaluation.json`
- Token cost: `benchmarks/naive-rag/output/token_cost_naive_rag.json`
- Token cost summary（自动生成）: `benchmarks/naive-rag/output/token_cost_summary.md`

说明：`NaiveRAG` 的 `token_cost_naive_rag.json` 可能是 `{}`。  
这是因为当前实现的 token 统计主要依赖对 LLM 调用的 patch 监控，而 NaiveRAG 构建阶段核心是 embedding/indexing，不一定触发被监控的 LLM 调用。
本目录的一键脚本已补充检索阶段（Stage2）的 embedding token 统计，统计单位为每个检索 query 的 token 数，并写回同一个 token cost 文件（按 embedding 模型名归档，`op_type=embedding_retrieval_query`）。
此外，Stage1 已补充 embedding 入库 token 统计：

- `embedding_ingestion_message`：每条 message 文本的 token 计数。
- `embedding_ingestion_document`：每次真正入库（触发 embedding/index）的 document 文本 token 计数（含 `flush` 时的尾部文档）。

## Naive-RAG 流程说明

- Stage1（`memory_construction.py`）：构建记忆索引，不负责最终问答生成。
- Stage2（`memory_search.py`）：对每个问题检索记忆片段。
- Stage3（`memory_evaluation.py`）：把“问题 + 检索到的记忆”喂给 `qa-model` 生成答案，再按指标评测（`f1/bleu/llm_judge`）。

## Token Cost 汇总工具

新增脚本：`summarize_token_cost.py`，可把 `token_cost_naive_rag.json` 自动汇总成表格。
`run_all.sh` 在流程末尾会自动调用该脚本生成 `output/token_cost_summary.md`。

示例：

```bash
cd benchmarks/naive-rag
uv run python summarize_token_cost.py \
  --input output/token_cost_naive_rag.json \
  --format markdown
```

保存到文件：

```bash
cd benchmarks/naive-rag
uv run python summarize_token_cost.py \
  --input output/token_cost_naive_rag.json \
  --format markdown \
  --save output/token_cost_summary.md
```

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
