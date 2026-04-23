# Evaluate NaiveRAG on LoCoMo

This example walks through evaluating the **NaiveRAG** memory layer on the **LoCoMo** dataset using MemBase's standard three-stage pipeline.

## Prerequisites

- Python environment with the RAG dependencies installed:

```bash
pip install -r envs/rag_requirements.txt
```

- A LoCoMo raw dataset file downloaded locally.
- A local HuggingFace embedding setup available for Stage 1 and Stage 2.
- An OpenAI-compatible API key and optional base URL for the LLM used by MemBase evaluation in Stage 3.

## Step 1: Configure NaiveRAG

Edit [`naive_rag_config.json`](naive_rag_config.json) and replace:

- `YOUR_DATASET_PATH` in [`run_construction.sh`](run_construction.sh)
- `YOUR_OPENAI_API_KEY`
- `YOUR_OPENAI_API_BASE` in [`api_config.json`](api_config.json)

The full list of supported fields lives in [`membase/configs/naive_rag.py`](/home/wiang/MemBase/membase/configs/naive_rag.py).

This example uses a simple default profile:

- local HuggingFace embeddings via `huggingface:all-MiniLM-L6-v2`
- no deferred chunking
- no explicit `max_tokens`
- `num_overlap_msgs=0`

By default, each user's memory is stored under:

- `<save_dir>/<user_id>/config.json`
- `<save_dir>/<user_id>/buffer_state.json`
- `<save_dir>/<user_id>/<user_id>.pkl`

This matches MemBase's per-user runner layout and avoids cross-user data collisions.

## Step 2: Run Memory Construction

```bash
bash examples/evaluate_naive_rag_on_locomo/run_construction.sh
```

This samples 2 trajectories and saves the standardized dataset to `naive_rag_output/LoCoMo_stage_1.json`.

## Step 3: Run Memory Search

This stage reads the standardized dataset produced by Stage 1 and excludes adversarial questions via [`question_filter.py`](question_filter.py).

```bash
bash examples/evaluate_naive_rag_on_locomo/run_search.sh
```

The retrieval results are written to `naive_rag_output/<top_k>_<start>_<end>.json`.

> **Note**: The `dataset_path` in the search and evaluation scripts must point to the standardized dataset produced by Stage 1, not the original raw data file.

## Step 4: Run Evaluation

```bash
bash examples/evaluate_naive_rag_on_locomo/run_evaluation.sh
```

This stage keeps using MemBase's own QA and judge pipeline. NaiveRAG only handles memory construction and retrieval.
