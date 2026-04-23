# Evaluate SimpleMem on LoCoMo

This example walks through evaluating the vendored **SimpleMem** text memory layer on the **LoCoMo** dataset using MemBase's three-stage pipeline.

## Prerequisites

- Python environment with the SimpleMem dependencies installed:

```bash
pip install -r envs/simplemem_requirements.txt
```

- A LoCoMo raw dataset file downloaded locally.
- A DeepSeek API key for SimpleMem construction and MemBase evaluation.
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

## Step 2: Configure SimpleMem

Edit [`simplemem_config.json`](simplemem_config.json) and replace:

- `YOUR_DEEPSEEK_API_KEY`
- `YOUR_DATASET_PATH` in [`run_construction.sh`](run_construction.sh)

The full list of supported fields lives in [`membase/configs/simplemem.py`](/home/wiang/MemBase/membase/configs/simplemem.py).

This example uses:

- `deepseek-chat` for construction
- `Qwen3-Embedding-8B` via local vLLM for embeddings
- DeepSeek API again for Stage 3 QA and judge
- A SimpleMem-specific LoCoMo QA prompt for Stage 3

By default, each user's memory is stored under:

- `<save_dir>/<user_id>/config.json`
- `<save_dir>/<user_id>/lancedb/`

This matches MemBase's per-user runner layout and avoids cross-user data collisions.

## Step 3: Run Memory Construction

```bash
bash examples/evaluate_simplemem_on_locomo/run_construction.sh
```

This samples 2 trajectories and saves the standardized dataset to `simplemem_output/LoCoMo_stage_1.json`.

## Step 4: Run Memory Search

This stage reads the standardized dataset produced by Stage 1 and excludes adversarial questions via [`question_filter.py`](question_filter.py).

```bash
bash examples/evaluate_simplemem_on_locomo/run_search.sh
```

This stage reads the standardized dataset produced by Stage 1 and writes retrieval results to `simplemem_output/<top_k>_<start>_<end>.json`.

> **Note**: The `dataset_path` in the search and evaluation scripts must point to the standardized dataset produced by Stage 1, not the original raw data file.

## Step 5: Run Evaluation

```bash
bash examples/evaluate_simplemem_on_locomo/run_evaluation.sh
```

This keeps using MemBase's QA and judge pipeline with `deepseek-chat`. The vendored SimpleMem integration only handles memory construction and retrieval, but the Stage 3 example now passes a dedicated LoCoMo QA prompt through [`qa_prompt.py`](qa_prompt.py) so the answer style better matches SimpleMem's retrieval context.

The prompt is intentionally adapted for MemBase: it asks for direct short answers and does not require JSON output, because MemBase Stage 3 sends the answer text directly into its judge pipeline.

## Notes

- If you want a strict cross-method ablation with one shared Stage 3 prompt, remove `--prompt-template` from [`run_evaluation.sh`](run_evaluation.sh) to fall back to MemBase's default QA prompt.
