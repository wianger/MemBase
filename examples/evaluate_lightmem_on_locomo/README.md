# Evaluate LightMem on LoCoMo

This example walks through evaluating the external **LightMem** adapter on the **LoCoMo** dataset using MemBase's three-stage pipeline.

## Prerequisites

- Python environment with the LightMem adapter dependencies:

```bash
pip install -r envs/lightmem_requirements.txt
pip install -e /home/wiang/LightMem
```

- A LoCoMo raw dataset file downloaded locally.
- A local or remote model path for LLMLingua-2.
- A DeepSeek API key for LightMem construction and MemBase evaluation.
- A local vLLM embedding service serving `Qwen3-Embedding-8B`.

## Step 1: Start the Embedding Model with vLLM

Start the local embedding service before running the pipeline:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-Embedding-8B \
    --port 8008 \
    --served-model-name Qwen3-Embedding-8B \
    --gpu-memory-utilization 0.6 \
    --hf_overrides '{"is_matryoshka": true}'
```

This exposes an OpenAI-compatible embedding endpoint at `http://localhost:8008/v1`.

## Step 2: Configure LightMem

Edit [`lightmem_config.json`](lightmem_config.json) and replace:

- `YOUR_DEEPSEEK_API_KEY`
- `llmlingua_model_path`

The full list of supported fields lives in [`membase/configs/lightmem.py`](/home/wiang/MemBase/membase/configs/lightmem.py).

This example uses:

- `deepseek-chat` for construction
- `Qwen3-Embedding-8B` via local vLLM for embeddings
- DeepSeek API again for Stage 3 QA and judge
- A LightMem-specific LoCoMo QA prompt for Stage 3

By default, each user's memory is stored under:

- `<save_dir>/<user_id>/config.json`
- `<save_dir>/<user_id>/qdrant/`

This matches MemBase's per-user runner layout and avoids cross-user collisions.

## Step 3: Run Memory Construction

```bash
bash examples/evaluate_lightmem_on_locomo/run_construction.sh
```

This samples 2 trajectories and saves the standardized dataset to `lightmem_output/LoCoMo_stage_1.json`.

## Step 4: Run Memory Search

This stage reads the standardized dataset produced by Stage 1 and excludes adversarial questions via [`question_filter.py`](question_filter.py).

```bash
bash examples/evaluate_lightmem_on_locomo/run_search.sh
```

This stage reads the standardized dataset produced by Stage 1 and writes retrieval results to `lightmem_output/<top_k>_<start>_<end>.json`.

> **Note**: The `dataset_path` in the search and evaluation scripts must point to the standardized dataset produced by Stage 1, not the original raw data file.

## Step 5: Run Evaluation

```bash
bash examples/evaluate_lightmem_on_locomo/run_evaluation.sh
```

This keeps using MemBase's QA and judge pipeline with `deepseek-chat`. The LightMem adapter only handles memory construction and retrieval, but the Stage 3 example now passes a dedicated LoCoMo QA prompt through [`qa_prompt.py`](qa_prompt.py) so the evaluation prompt is closer to LightMem's intended task framing.

## Notes

- This integration is **external dependency + adapter**, not a vendored copy of LightMem.
- v1 targets the root text pipeline with `pre_compress`, `topic_segment`, `offline update`, local Qdrant persistence, and optional summary construction.
- Summary retrieval is optional. When `enable_summary=true`, the adapter runs LightMem summarization during `flush()` and mixes summary hits into the stage 2 retrieval budget.
- If you want a strict cross-method ablation with one shared Stage 3 prompt, remove `--prompt-template` from [`run_evaluation.sh`](run_evaluation.sh) to fall back to MemBase's default QA prompt.
