import os

from pydantic import Field, JsonValue, model_validator
from typing import Literal, Self

from .base import MemBaseConfig


class SimpleMemConfig(MemBaseConfig):
    """Configuration for the vendored SimpleMem text backend."""

    api_key: str | None = Field(
        default=None,
        description="OpenAI-compatible API key for SimpleMem's LLM calls.",
    )
    base_url: str | None = Field(
        default=None,
        description="Optional OpenAI-compatible base URL.",
    )
    llm_model: str = Field(
        default="gpt-4.1-mini",
        description="LLM model used for memory construction and retrieval planning.",
    )
    embedding_provider: Literal["sentence_transformer", "openai"] = Field(
        default="sentence_transformer",
        description=(
            "Embedding backend used by SimpleMem. "
            "`sentence_transformer` loads the model locally, while `openai` "
            "calls an OpenAI-compatible embeddings endpoint."
        ),
    )
    embedding_model: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B",
        description="Embedding model name or path.",
    )
    embedding_dimension: int = Field(
        default=1024,
        ge=1,
        description="Embedding dimension for the configured embedding model.",
    )
    embedding_api_key: str | None = Field(
        default=None,
        description="API key for OpenAI-compatible embedding endpoints.",
    )
    embedding_base_url: str | None = Field(
        default=None,
        description="Base URL for OpenAI-compatible embedding endpoints.",
    )
    embedding_model_kwargs: dict[str, JsonValue] = Field(
        default_factory=dict,
        description=(
            "Additional keyword arguments for the embedding backend. "
            "For `sentence_transformer`, common keys include `model_kwargs` "
            "or `encode_kwargs`. For `openai`, extra SDK kwargs may be provided."
        ),
    )
    embedding_context_length: int = Field(
        default=32768,
        ge=1,
        description="Documented context length for the embedding model.",
    )
    enable_thinking: bool = Field(
        default=False,
        description="Enable Qwen-compatible deep thinking mode.",
    )
    use_streaming: bool = Field(
        default=True,
        description="Enable streaming responses for LLM calls.",
    )
    use_json_format: bool = Field(
        default=False,
        description="Request JSON responses from OpenAI-compatible APIs when supported.",
    )
    window_size: int = Field(
        default=40,
        ge=1,
        description="Number of dialogues per construction window.",
    )
    overlap_size: int = Field(
        default=2,
        ge=0,
        description="Window overlap size for context continuity.",
    )
    semantic_top_k: int = Field(default=25, ge=1)
    keyword_top_k: int = Field(default=5, ge=1)
    structured_top_k: int = Field(default=5, ge=1)
    db_path: str | None = Field(
        default=None,
        description="LanceDB storage path. Defaults to <save_dir>/lancedb.",
    )
    table_name: str | None = Field(
        default=None,
        description="LanceDB table name. Defaults to 'memory_entries'.",
    )
    enable_parallel_processing: bool = Field(default=True)
    max_parallel_workers: int = Field(default=16, ge=1)
    enable_parallel_retrieval: bool = Field(default=True)
    max_retrieval_workers: int = Field(default=8, ge=1)
    enable_planning: bool = Field(default=True)
    enable_reflection: bool = Field(default=True)
    max_reflection_rounds: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def _populate_paths(self) -> Self:
        if self.db_path is None:
            self.db_path = os.path.join(self.save_dir, "lancedb")
        if self.table_name is None:
            self.table_name = "memory_entries"
        return self

    def get_llm_models(self) -> list[str]:
        return [self.llm_model]
