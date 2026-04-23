# Evaluate SimpleMem on LoCoMo

This example walks through evaluating the vendored **SimpleMem** text memory layer on the **LoCoMo** dataset using MemBase's three-stage pipeline.

## Prerequisites

- Python environment with the SimpleMem dependencies installed:

```bash
pip install -r envs/simplemem_requirements.txt
```

- A LoCoMo raw dataset file downloaded locally.
- An OpenAI-compatible API key and optional base URL for the LLM used by SimpleMem and MemBase evaluation.

## Step 1: Configure SimpleMem

Edit [`simplemem_config.json`](simplemem_config.json) and replace:

- `YOUR_OPENAI_API_KEY`
- `YOUR_OPENAI_API_BASE`
- `YOUR_DATASET_PATH` in [`run_construction.sh`](run_construction.sh)

The full list of supported fields lives in [`membase/configs/simplemem.py`](/home/wiang/MemBase/membase/configs/simplemem.py).

By default, each user's memory is stored under:

- `<save_dir>/<user_id>/config.json`
- `<save_dir>/<user_id>/lancedb/`

This matches MemBase's per-user runner layout and avoids cross-user data collisions.

## Step 2: Run Memory Construction

```bash
bash examples/evaluate_simplemem_on_locomo/run_construction.sh
```

This samples 2 trajectories and saves the standardized dataset to `simplemem_output/LoCoMo_stage_1.json`.

## Step 3: Run Memory Search

```bash
bash examples/evaluate_simplemem_on_locomo/run_search.sh
```

This stage reads the standardized dataset produced by Stage 1 and writes retrieval results to `simplemem_output/<top_k>_<start>_<end>.json`.

## Step 4: Run Evaluation

```bash
bash examples/evaluate_simplemem_on_locomo/run_evaluation.sh
```

This keeps using MemBase's QA and judge pipeline. The vendored SimpleMem integration only handles memory construction and retrieval.

