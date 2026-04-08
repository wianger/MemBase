from .base import MemBaseConfig
from pydantic import Field, model_validator
from typing import Self


class SimpleMemConfig(MemBaseConfig):
    """Configuration for SimpleMem provider."""

    api_key: str = Field(
        default="EMPTY",
        description=(
            "API key for the OpenAI-compatible LLM endpoint used by SimpleMem."
        ),
    )
    llm_base_url: str | None = Field(
        default=None,
        description=(
            "Optional base URL for OpenAI-compatible chat completion endpoint."
        ),
    )
    llm_model: str = Field(
        default="gpt-4.1-mini",
        description="LLM model name used by SimpleMem.",
    )

    embedding_base_url: str | None = Field(
        default=None,
        description=(
            "Optional base URL for an OpenAI-compatible embedding endpoint. "
            "When set, SimpleMem uses remote embedding API calls instead of "
            "loading a local sentence-transformers model."
        ),
    )
    embedding_api_key: str | None = Field(
        default=None,
        description=(
            "Optional API key for the remote embedding endpoint. "
            "Defaults to `api_key` when omitted."
        ),
    )
    embedding_model: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B",
        description=(
            "Embedding model name or path used by SimpleMem. "
            "By default it is loaded locally via sentence-transformers; "
            "when `embedding_base_url` is set it is treated as the remote model ID."
        ),
    )
    embedding_dimension: int = Field(
        default=1024,
        ge=1,
        description=(
            "Embedding dimension metadata used by SimpleMem config. "
            "In remote endpoint mode this must match the returned vector size."
        ),
    )

    memory_table_name: str = Field(
        default="memory_entries",
        description="Table name inside the user-local LanceDB directory.",
    )

    window_size: int = Field(
        default=40,
        ge=1,
        description="Dialogue window size for memory extraction.",
    )
    overlap_size: int = Field(
        default=2,
        ge=0,
        description="Window overlap size for memory extraction.",
    )

    semantic_top_k: int = Field(
        default=25,
        ge=1,
        description="Semantic retrieval depth inside SimpleMem hybrid retriever.",
    )
    keyword_top_k: int = Field(
        default=5,
        ge=1,
        description="Keyword retrieval depth inside SimpleMem hybrid retriever.",
    )
    structured_top_k: int = Field(
        default=5,
        ge=1,
        description="Structured retrieval depth inside SimpleMem hybrid retriever.",
    )

    enable_planning: bool = Field(
        default=True,
        description="Whether SimpleMem query planning is enabled.",
    )
    enable_reflection: bool = Field(
        default=True,
        description="Whether SimpleMem reflection rounds are enabled.",
    )
    max_reflection_rounds: int = Field(
        default=2,
        ge=0,
        description="Maximum reflection rounds for SimpleMem retrieval.",
    )

    enable_parallel_processing: bool = Field(
        default=True,
        description="Whether SimpleMem enables parallel processing in memory building.",
    )
    max_parallel_workers: int = Field(
        default=16,
        ge=1,
        description="Max parallel workers for memory building.",
    )
    enable_parallel_retrieval: bool = Field(
        default=True,
        description="Whether SimpleMem enables parallel retrieval planning/search.",
    )
    max_retrieval_workers: int = Field(
        default=8,
        ge=1,
        description="Max parallel workers for retrieval.",
    )

    enable_thinking: bool = Field(
        default=False,
        description="Whether to pass thinking mode controls to the backend LLM client.",
    )
    use_streaming: bool = Field(
        default=False,
        description="Whether to use streaming responses in SimpleMem LLM calls.",
    )
    use_json_format: bool = Field(
        default=False,
        description=(
            "Whether SimpleMem sub-prompts request JSON mode via response_format=json_object."
        ),
    )

    @model_validator(mode="after")
    def _normalize_optional_endpoint_fields(self) -> Self:
        if isinstance(self.embedding_base_url, str) and self.embedding_base_url.strip() == "":
            self.embedding_base_url = None
        if isinstance(self.embedding_api_key, str) and self.embedding_api_key.strip() == "":
            self.embedding_api_key = None
        return self

    def uses_remote_embedding(self) -> bool:
        return (
            isinstance(self.embedding_base_url, str)
            and self.embedding_base_url.strip() != ""
        )

    def get_llm_models(self) -> list[str]:
        return [self.llm_model]

    def get_embedding_models(self) -> list[str]:
        return [self.embedding_model]
