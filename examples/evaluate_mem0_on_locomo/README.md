# Evaluate Mem0 on LoCoMo

This example walks through evaluating the **Mem0** memory layer on the **LoCoMo** dataset using the native **Mem0 v2** API shape. It covers how to configure Mem0 with `deepseek-chat` for construction and evaluation, how to deploy `Qwen3-Embedding-8B` locally with vLLM, how to use a custom question-answering prompt derived from the [Mem0 paper](https://arxiv.org/pdf/2504.19413), and how to filter specific question types at the retrieval stage.

---

## Prerequisites

- **Python >= 3.12** with the Mem0 environment (`pip install -r envs/mem0_requirements.txt`). The requirements file installs Mem0 from the upstream GitHub `v2.0.0` tag rather than a PyPI prerelease.
- **vLLM** installed in the environment (`pip install vllm`).
- **Qwen3-Embedding-8B** downloaded locally. See the [example](../download_models/) for how to download it.
- A **DeepSeek API key** for construction and evaluation.
- If you want Mem0 v2's hybrid retrieval path locally, install the extras listed in [`envs/mem0_requirements.txt`](../../envs/mem0_requirements.txt). `fastembed` is required for BM25 sparse encoding, and Mem0 v2 may expect a local spaCy English model such as `en_core_web_sm` depending on your environment.

---

## Step 1: Serve the Embedding Model with vLLM

Start the vLLM embedding server before running the pipeline:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve pretrained_models/Qwen3-Embedding-8B \
    --port 8008 \
    --served-model-name Qwen3-Embedding-8B \
    --gpu-memory-utilization 0.6 \
    --hf_overrides '{"is_matryoshka": true}'
```

This exposes an OpenAI-compatible embedding endpoint at `http://localhost:8008/v1`. The Mem0 config uses the `openai` embedder provider to connect to it.

> **Note**: The `--hf_overrides '{"is_matryoshka": true}'` flag is required for Qwen3-Embedding models to enable Matryoshka representation support.

## Step 2: Download the Dataset

Download the LoCoMo dataset from GitHub:

> https://github.com/snap-research/locomo/tree/main/data

Save it to a local path, e.g., `/path/to/locomo.json`.

## Step 3: Configure Mem0

See [`mem0_config.json`](mem0_config.json). The full list of configuration fields can be found in `membase/configs/mem0.py`.

### Embedding Configuration

Mem0 uses the `openai` embedder provider to connect to any OpenAI-compatible embedding API (including vLLM). The `embedding_config` section specifies the vLLM endpoint:

```json
"embedder_provider": "openai",
"embedding_model": "Qwen3-Embedding-8B",
"embedding_model_dims": 4096,
"embedding_config": {
    "openai_base_url": "http://localhost:8008/v1",
    "api_key": "EMPTY"
}
```

`embedding_model` must match the `--served-model-name` in the vLLM command. `api_key` can be `"EMPTY"` for local servers.

### LLM Configuration

In this example, `llm_provider` is set to `"openai"`, and the API key and base URL in [`mem0_config.json`](mem0_config.json) point to DeepSeek's OpenAI-compatible API:

```json
"llm_model": "deepseek-chat",
"llm_config": {
    "openai_base_url": "https://api.deepseek.com",
    "api_key": "YOUR_DEEPSEEK_API_KEY"
}
```

### Custom Instructions

Mem0 v2 replaces the old v1 fact-extraction / graph-store prompt wiring with a unified `custom_instructions` field. Use it to bias extraction behavior without depending on removed v1 prompt fields:

```json
"custom_instructions": "Extract durable memories from the dialogue with correct timestamps and speaker attribution."
```

### Vector Store

Mem0 uses **Qdrant in local mode** (`on_disk=True`) by default, which stores data directly on disk without requiring an external Qdrant server. Mem0 v2 also maintains an internal entity collection to support hybrid retrieval. Each user's memory is stored in its own directory, so multiple processes can safely run in parallel as long as they handle non-overlapping trajectory ranges.

## Step 4: Run Memory Construction (Stage 1)

Edit [`run_construction.sh`](run_construction.sh) to set your dataset path, then run:

```bash
bash examples/evaluate_mem0_on_locomo/run_construction.sh
```

This samples 2 trajectories from the full LoCoMo dataset via `--sample-size` and processes them with 2 workers in a single process. The sampled dataset is automatically saved to `mem0_output/LoCoMo_stage_1.json` in standardized format for subsequent stages.

## Step 5: Run Memory Retrieval (Stage 2)

This stage reads the standardized dataset saved by Stage 1 (with `--dataset-standardized`).

Adversarial questions are excluded via `--question-filter-path`, which points to [`question_filter.py`](question_filter.py). You can modify this filter to target other question types (see `membase/datasets/locomo.py` for all available types).

```bash
bash examples/evaluate_mem0_on_locomo/run_search.sh
```

> **Note**: The `dataset_path` in the search and evaluation scripts must point to the **standardized** dataset produced by Stage 1, not the original raw data file. Because Stage 1 samples a subset of trajectories from the full dataset, memory is only constructed for that subset. The standardized file records exactly which trajectories are sampled, so subsequent stages must use it to stay consistent.

## Step 6: Run Evaluation (Stage 3)

Edit [`run_evaluation.sh`](run_evaluation.sh) and run:

```bash
bash examples/evaluate_mem0_on_locomo/run_evaluation.sh
```

This example uses the upstream Mem0 LoCoMo **non-graph** answer prompt via `--prompt-template`, together with a custom `--message-builder` that sends the rendered prompt as a single `system` message, matching the official evaluation style more closely. The helper is defined in [`qa_prompt.py`](qa_prompt.py).

In upstream Mem0 evaluation code there are three related answer prompt variants:

- `ANSWER_PROMPT_GRAPH`: the older graph-aware variant that expects per-speaker graph relations in addition to memories
- `ANSWER_PROMPT`: the non-graph two-speaker variant, which is the one this MemBase example now follows
- `ANSWER_PROMPT_ZEP`: a separate single-stream prompt used by the Zep baseline

Because the MemBase Mem0 v2 migration intentionally removed graph-memory support, this example aligns with the official **non-graph** prompt rather than the graph variant.

> **Note**: Make sure to set `--dataset-type LoCoMo` so that the correct judge prompt template and response parser are used.
>
> **Fairness Note**: This example keeps Mem0's official-style LoCoMo QA formatting by default. If you want a stricter apples-to-apples comparison with the other methods, remove both `--prompt-template` and `--message-builder` from [`run_evaluation.sh`](run_evaluation.sh).
