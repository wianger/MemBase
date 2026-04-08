# SimpleMem Benchmark (One-Click)

这个目录提供 SimpleMem 的独立 benchmark 资产：

- `pyproject.toml`: 使用 `uv` 管理独立 Python 环境
- `simplemem_config.json`: 基线 memory 配置（不在运行时改写）
- `api_config.json`: 基线 API 配置（不在运行时改写）
- `check_endpoints.sh`: 端点连通性与模型可用性检查
- `run_all.sh`: 一键运行三阶段（构建、检索、评测）
- `summarize_token_cost.py`: token 统计汇总工具

## 快速开始

建议先检查 LLM 端点，再跑全流程：

```bash
cd benchmarks/simplemem
chmod +x check_endpoints.sh run_all.sh
./check_endpoints.sh
./run_all.sh
```

仅执行端点检查：

```bash
cd benchmarks/simplemem
./check_endpoints.sh
```

默认行为：

- 数据集：`datasets/locomo/data/locomo10.json`
- LLM endpoint：`http://10.46.131.226:8000/v1`
- LLM model：`qwen3.5-0.8b`
- Embedding model：`Qwen/Qwen3-Embedding-0.6B`
- API key：`EMPTY`
- 默认 embedding endpoint：`http://10.46.131.226:8001/v1`
- 可通过 `EMBEDDING_BASE_URL` 覆盖为其他 OpenAI-compatible `/embeddings` 服务
- `SAMPLE_SIZE` 未设置时自动跑全量数据集
- `planning/reflection/parallel` 默认开启
- 评测指标：`f1 bleu llm_judge`

## 环境变量覆盖

`run_all.sh` 支持通过环境变量覆盖默认值：

- `LLM_BASE_URL`（默认 `http://10.46.131.226:8000/v1`）
- `EMBEDDING_BASE_URL`（默认 `http://10.46.131.226:8001/v1`）
- `LLM_MODEL`（默认 `qwen3.5-0.8b`）
- `EMBEDDING_MODEL`（默认 `Qwen/Qwen3-Embedding-0.6B`）
- `API_KEY`（默认 `EMPTY`）
- `EMBEDDING_API_KEY`（可选；默认回退到 `API_KEY`）
- `EMBEDDING_DIMENSION`（可选；远端模式下未设置时自动探测，本地模式下默认使用 `1024`）
- `LLM_API_POOL_SIZE`（默认 `16`；Stage3 QA/Judge 的 OpenAI client pool 大小）
- `ENABLE_PLANNING`（默认 `1`）
- `ENABLE_REFLECTION`（默认 `1`）
- `MAX_REFLECTION_ROUNDS`（默认 `2`）
- `ENABLE_PARALLEL_PROCESSING`（默认 `1`）
- `MAX_PARALLEL_WORKERS`（默认 `16`）
- `ENABLE_PARALLEL_RETRIEVAL`（默认 `1`）
- `MAX_RETRIEVAL_WORKERS`（默认 `8`）
- `WINDOW_SIZE`（默认 `40`）
- `OVERLAP_SIZE`（默认 `2`）
- `SEMANTIC_TOP_K`（默认 `25`）
- `KEYWORD_TOP_K`（默认 `5`）
- `STRUCTURED_TOP_K`（默认 `5`）
- `MEMORY_TABLE_NAME`（默认 `memory_entries`）
- `USE_JSON_FORMAT`（默认 `0`）
- `USE_STREAMING`（默认 `0`）
- `ENABLE_THINKING`（默认 `0`）
- `DATASET_PATH`（默认 LoCoMo10）
- `NUM_WORKERS`（默认 `8`）
- `TOP_K`（默认 `10`）
- `QA_BATCH_SIZE`（默认 `4`）
- `JUDGE_BATCH_SIZE`（默认 `4`）
- `SAMPLE_SIZE`（可选；不设置则跑全量，设置后按采样量跑）
- `TOKENIZER_PATH`（可选；不设置时优先使用 `<repo_root>/models/<LLM_MODEL>`，否则回退到 `LLM_MODEL`）
- `EMBEDDING_TOKENIZER_PATH`（可选；不设置时优先使用 `<repo_root>/models/<EMBEDDING_MODEL>`，否则回退到 `EMBEDDING_MODEL`）
- `HF_HUB_OFFLINE`（默认 `1`；设为 `0` 允许联网从 HuggingFace 拉 tokenizer）
- `LITELLM_LOCAL_MODEL_COST_MAP`（默认 `true`；设为 `false` 时 LiteLLM 会尝试联网更新模型 cost map）

`check_endpoints.sh` 支持：

- `LLM_BASE_URL`
- `LLM_MODEL`
- `API_KEY`
- `EMBEDDING_BASE_URL`
- `EMBEDDING_MODEL`
- `EMBEDDING_API_KEY`
- `TIMEOUT_SECONDS`
- `CHAT_PROMPT`

示例：

```bash
cd benchmarks/simplemem
TOP_K=20 \
SAMPLE_SIZE=5 \
./run_all.sh
```

关闭 reflection 示例：

```bash
cd benchmarks/simplemem
ENABLE_REFLECTION=0 ./run_all.sh
```

## 输出文件

默认输出目录：`benchmarks/simplemem/output`

- Stage 1: `benchmarks/simplemem/output/LoCoMo_stage_1.json`
- Stage 2: `benchmarks/simplemem/output/<top_k>_0_<sample_size>.json`
- Stage 3: `benchmarks/simplemem/output/<top_k>_0_<sample_size>_evaluation.json`
- Token cost: `benchmarks/simplemem/output/token_cost_simplemem.json`
- Token cost summary（自动生成）: `benchmarks/simplemem/output/token_cost_summary.md`

## Embedding 说明

SimpleMem 现在支持两种 embedding 模式：

- 默认模式：默认走 `EMBEDDING_BASE_URL=http://10.46.131.226:8001/v1`，复用外部 OpenAI-compatible embedding 服务（例如 vLLM）。
- 本地模式：如果你想退回本地加载，可显式清空 `EMBEDDING_BASE_URL`，此时由 `simplemem` 内部通过本地 `sentence-transformers` 模型加载。

建议：

- 本地模式下，将 `EMBEDDING_MODEL` 设为本机已可用的 HuggingFace 模型名或本地模型路径。
- 本地离线路径模式下，优先确保路径中存在可用 tokenizer/model 文件。
- 远端模式下，确保 `EMBEDDING_MODEL` 是服务端可识别的模型 ID；若未手动设置 `EMBEDDING_DIMENSION`，脚本会自动探测返回向量维度。

远端 embedding 示例：

```bash
cd benchmarks/simplemem
EMBEDDING_BASE_URL=http://10.46.131.226:8001/v1 \
EMBEDDING_MODEL=qwen3-embedding-0.6b \
EMBEDDING_DIMENSION=1024 \
./run_all.sh
```

## Provider 行为说明

在 MemBase 的 `SimpleMemLayer` 中，`delete/update` 采用严格失败语义：

- `delete(memory_id)`：返回 `False`，并打印“不支持该原生能力”的日志
- `update(memory_id, **kwargs)`：返回 `False`，并打印“不支持该原生能力”的日志

这与 SimpleMem 0.1.0 的原生 API 能力边界保持一致，不做近似语义替代实现。

## Token Cost 汇总工具

`run_all.sh` 在流程末尾会自动调用 `summarize_token_cost.py` 生成 `output/token_cost_summary.md`。

也可以手动执行：

```bash
cd benchmarks/simplemem
uv run python summarize_token_cost.py \
  --input output/token_cost_simplemem.json \
  --format markdown
```
