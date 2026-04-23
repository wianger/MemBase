# Evaluate LightMem on LoCoMo

This example walks through evaluating the external **LightMem** adapter on the **LoCoMo** dataset using MemBase's three-stage pipeline.

## Prerequisites

- Python environment with the LightMem adapter dependencies:

```bash
pip install -r envs/lightmem_requirements.txt
pip install -e /home/wiang/LightMem
```

- A LoCoMo raw dataset file downloaded locally.
- A local or remote model path for LLMLingua-2 and the embedding model you want to use.
- An OpenAI-compatible API key and optional base URL for LightMem construction and MemBase evaluation.

## Step 1: Configure LightMem

Edit [`lightmem_config.json`](lightmem_config.json) and replace:

- `YOUR_OPENAI_API_KEY`
- `YOUR_OPENAI_API_BASE`
- `llmlingua_model_path`
- `embedding_model`
- `embedding_device` if needed

The full list of supported fields lives in [`membase/configs/lightmem.py`](/home/wiang/MemBase/membase/configs/lightmem.py).

By default, each user's memory is stored under:

- `<save_dir>/<user_id>/config.json`
- `<save_dir>/<user_id>/qdrant/`

This matches MemBase's per-user runner layout and avoids cross-user collisions.

## Step 2: Run Memory Construction

```bash
bash examples/evaluate_lightmem_on_locomo/run_construction.sh
```

This samples 2 trajectories and saves the standardized dataset to `lightmem_output/LoCoMo_stage_1.json`.

## Step 3: Run Memory Search

```bash
bash examples/evaluate_lightmem_on_locomo/run_search.sh
```

This stage reads the standardized dataset produced by Stage 1 and writes retrieval results to `lightmem_output/<top_k>_<start>_<end>.json`.

## Step 4: Run Evaluation

```bash
bash examples/evaluate_lightmem_on_locomo/run_evaluation.sh
```

This keeps using MemBase's QA and judge pipeline. The LightMem adapter only handles memory construction and retrieval.

## Notes

- This integration is **external dependency + adapter**, not a vendored copy of LightMem.
- v1 targets the root text pipeline with `pre_compress`, `topic_segment`, `offline update`, local Qdrant persistence, and optional summary construction.
- Summary retrieval is optional. When `enable_summary=true`, the adapter runs LightMem summarization during `flush()` and mixes summary hits into the stage 2 retrieval budget.
