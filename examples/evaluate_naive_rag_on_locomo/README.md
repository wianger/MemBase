# Evaluate NaiveRAG on LoCoMo

This example walks through evaluating the **NaiveRAG** memory layer on the **LoCoMo** dataset using MemBase's standard three-stage pipeline.

## Prerequisites

- Python environment with the RAG dependencies installed:

```bash
pip install -r envs/rag_requirements.txt
```

- A LoCoMo raw dataset file downloaded locally.
- A local vLLM embedding service serving `Qwen3-Embedding-8B`.
- A DeepSeek API key for MemBase evaluation in Stage 3.

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

## Step 2: Configure NaiveRAG

Edit [`naive_rag_config.json`](naive_rag_config.json) and replace:

- `YOUR_DATASET_PATH` in [`run_construction.sh`](run_construction.sh)
- `YOUR_DEEPSEEK_API_KEY` in [`api_config.json`](api_config.json)

The full list of supported fields lives in [`membase/configs/naive_rag.py`](/home/wiang/MemBase/membase/configs/naive_rag.py).

This example uses a simple default profile:

- local vLLM embeddings via `openai:Qwen3-Embedding-8B`
- no deferred chunking
- no explicit `max_tokens`
- `num_overlap_msgs=0`

`llm_model` is only used for tokenizer selection in NaiveRAG. It does not trigger construction-time API calls in this example.

By default, each user's memory is stored under:

- `<save_dir>/<user_id>/config.json`
- `<save_dir>/<user_id>/buffer_state.json`
- `<save_dir>/<user_id>/<user_id>.pkl`

This matches MemBase's per-user runner layout and avoids cross-user data collisions.

## Step 3: Run Memory Construction

```bash
bash examples/evaluate_naive_rag_on_locomo/run_construction.sh
```

This samples 2 trajectories and saves the standardized dataset to `naive_rag_output/LoCoMo_stage_1.json`.

## Step 4: Run Memory Search

This stage reads the standardized dataset produced by Stage 1 and excludes adversarial questions via [`question_filter.py`](question_filter.py).

```bash
bash examples/evaluate_naive_rag_on_locomo/run_search.sh
```

The retrieval results are written to `naive_rag_output/<top_k>_<start>_<end>.json`.

> **Note**: The `dataset_path` in the search and evaluation scripts must point to the standardized dataset produced by Stage 1, not the original raw data file.

## Step 5: Run Evaluation

```bash
bash examples/evaluate_naive_rag_on_locomo/run_evaluation.sh
```

This stage keeps using MemBase's own QA and judge pipeline with `deepseek-chat`. NaiveRAG only handles memory construction and retrieval.
