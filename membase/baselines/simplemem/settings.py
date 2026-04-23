from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(slots=True)
class SimpleMemSettings:
    """Runtime settings for the vendored SimpleMem backend."""

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    llm_model: str = "gpt-4.1-mini"
    embedding_provider: Literal["sentence_transformer", "openai"] = "sentence_transformer"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dimension: int = 1024
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_model_kwargs: dict[str, Any] | None = None
    enable_thinking: bool = False
    use_streaming: bool = True
    use_json_format: bool = False
    window_size: int = 40
    overlap_size: int = 2
    semantic_top_k: int = 25
    keyword_top_k: int = 5
    structured_top_k: int = 5
    lancedb_path: str = "./lancedb_data"
    memory_table_name: str = "memory_entries"
    enable_parallel_processing: bool = True
    max_parallel_workers: int = 16
    enable_parallel_retrieval: bool = True
    max_retrieval_workers: int = 8
    enable_planning: bool = True
    enable_reflection: bool = True
    max_reflection_rounds: int = 2
