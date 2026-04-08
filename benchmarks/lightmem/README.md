# LightMem Benchmark (MemBase)

该目录提供 `LightMem` 在 MemBase 上的一键基准脚本，流程与 `mem0 / naive-rag / simplemem` 保持一致：

1. Stage 1: `memory_construction.py`
2. Stage 2: `memory_search.py`
3. Stage 3: `memory_evaluation.py`
4. Token cost 汇总

## 版本策略

采用可运行优先 + commit 固定：

- `lightmem @ git+https://github.com/zjunlp/LightMem@a19ea88df47c73fd2f55d27f64616467ef576a81`

说明：PyPI 当前仅有 `lightmem==0.0.0`，接口不完整，不能稳定支撑该 benchmark 流程。

## 快速开始

```bash
cd benchmarks/lightmem
./check_endpoints.sh
./run_all.sh
```

## 常用环境变量

- LLM:
  - `LLM_BASE_URL` (default: `http://10.46.131.226:8000/v1`)
  - `LLM_MODEL` (default: `qwen3.5-0.8b`)
  - `API_KEY` (default: `EMPTY`)
- Embedding:
  - `EMBEDDING_BASE_URL` (default: `http://10.46.131.226:8001/v1`)
  - `EMBEDDING_MODEL` (default: `qwen3-embedding-0.6b`)
  - `EMBEDDING_API_KEY` (optional; defaults to `API_KEY`)
  - `EMBEDDING_DIMENSION` (optional; remote mode auto-detects when unset)
  - `EMBEDDING_DEVICE` (default: `cpu`)
- LLMLingua:
  - `LLMLINGUA_MODEL_PATH` (default: `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`)
  - `LLMLINGUA_DEVICE_MAP` (default: `cpu`)
  - `LLMLINGUA_COMPRESS_RATE` (default: `0.6`)
- LightMem 行为:
  - `MEMORY_MANAGER_NAME` (default: `openai`)
  - `EXTRACTION_MODE` (default: `flat`)
  - `ENABLE_OFFLINE_UPDATE` (default: `1`)
  - `CONSTRUCT_QUEUE_TOP_K` / `CONSTRUCT_QUEUE_KEEP_TOP_N` / `CONSTRUCT_QUEUE_WORKERS`
  - `OFFLINE_UPDATE_SCORE_THRESHOLD` / `OFFLINE_UPDATE_WORKERS`
- 通用评测:
  - `DATASET_PATH`, `SAMPLE_SIZE`, `NUM_WORKERS`, `TOP_K`
  - `QA_BATCH_SIZE`, `JUDGE_BATCH_SIZE`
- 离线相关:
  - `HF_HUB_OFFLINE` (default: `0`，允许首次下载默认 LLMLingua 模型)
  - `TRANSFORMERS_OFFLINE` (default: 跟随 `HF_HUB_OFFLINE`)

## 离线运行注意事项

当 `HF_HUB_OFFLINE=1` 且你显式清空 `EMBEDDING_BASE_URL` 使用本地 embedding 时：

- 未设置 `EMBEDDING_BASE_URL` 时，`EMBEDDING_MODEL` 必须是本地目录（脚本会自动尝试从 `<repo_root>/models` 解析）。
- `LLMLINGUA_MODEL_PATH` 也必须是本地目录。
- 若二者不是本地目录，`run_all.sh` 会提前报错并退出。

## Embedding 模式说明

LightMem 现在支持两种 embedding 模式：

- 默认模式：默认走 `EMBEDDING_BASE_URL=http://10.46.131.226:8001/v1`，复用外部 OpenAI-compatible `/embeddings` 服务。
- 本地模式：如果你想退回本地 embedding，可显式清空 `EMBEDDING_BASE_URL`。

维度规则：

- 如果显式设置了 `EMBEDDING_DIMENSION`，脚本直接使用该值。
- 如果未设置且使用远端模式，脚本会自动调用 embedding endpoint 探测向量维度。
- 如果未设置且使用本地模式，脚本使用当前基线默认值 `384`。

远端 embedding 示例：

```bash
cd benchmarks/lightmem
EMBEDDING_BASE_URL=http://10.46.131.226:8001/v1 \
EMBEDDING_MODEL=qwen3-embedding-0.6b \
./run_all.sh
```

## 输出文件

默认输出到 `benchmarks/lightmem/output`：

- Stage 1 数据：`LoCoMo_stage_1.json`
- Stage 2 检索：`<topk>_0_<n>.json`
- Stage 3 评测：`<topk>_0_<n>_evaluation.json`
- Token 成本：`token_cost_lightmem.json`
- Token 汇总：`token_cost_summary.md`

## 说明

- `LightMemLayer.delete/update` 采用严格失败策略：返回 `False` 并打印明确日志，不做近似语义替代。
- `run_all.sh` 运行时会写 `.run/lightmem_config.runtime.json` 与 `.run/api_config.runtime.json`，不会修改基线配置文件。
