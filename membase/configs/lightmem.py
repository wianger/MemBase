from .base import MemBaseConfig
from pydantic import Field, model_validator
from typing import Any, Literal
from typing import Self


class LightMemConfig(MemBaseConfig):
    """Configuration for LightMem provider."""

    api_key: str = Field(
        default="EMPTY",
        description="API key for the configured LightMem memory manager.",
    )
    llm_base_url: str | None = Field(
        default=None,
        description="Optional base URL for the LLM endpoint.",
    )
    llm_model: str = Field(
        default="gpt-4.1-mini",
        description="LLM model name for metadata extraction and offline updates.",
    )
    memory_manager_name: Literal[
        "openai", "deepseek", "vllm", "ollama", "transformers", "vllm_offline"
    ] = Field(
        default="openai",
        description="LightMem memory manager backend.",
    )
    llm_max_tokens: int = Field(
        default=16000,
        ge=1,
        description="Max tokens for LightMem manager generation calls.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        description="Sampling temperature for the memory manager.",
    )
    top_p: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description="Top-p parameter for manager generation calls.",
    )

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description=(
            "Embedding model path/name for LightMem text embedder. "
            "By default it is loaded locally; when `embedding_base_url` is set "
            "it is treated as the remote model ID for an OpenAI-compatible endpoint."
        ),
    )
    embedding_base_url: str | None = Field(
        default=None,
        description=(
            "Optional base URL for an OpenAI-compatible embedding endpoint. "
            "When set, LightMem uses the upstream `openai` text embedder backend "
            "instead of the local `huggingface` backend."
        ),
    )
    embedding_api_key: str | None = Field(
        default=None,
        description=(
            "Optional API key for the remote embedding endpoint. "
            "Defaults to `api_key` when omitted."
        ),
    )
    embedding_dimension: int = Field(
        default=384,
        ge=1,
        description="Embedding dimension used by embedder and Qdrant collections.",
    )
    embedding_device: str = Field(
        default="cpu",
        description="Device hint for embedding model kwargs, e.g. cpu/cuda.",
    )
    embedding_model_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra kwargs forwarded to sentence-transformers model init.",
    )

    retrieve_limit_default: int = Field(
        default=10,
        ge=1,
        description="Default retrieval limit used when caller does not provide k.",
    )
    qdrant_collection_name: str = Field(
        default="memory_entries",
        description="Qdrant collection name used for LightMem retrieval index.",
    )
    qdrant_on_disk: bool = Field(
        default=True,
        description="Whether to persist Qdrant local storage on disk.",
    )

    pre_compress: bool = Field(default=True, description="Enable LLMLingua pre-compression.")
    topic_segment: bool = Field(default=True, description="Enable topic segmentation.")
    precomp_topic_shared: bool = Field(
        default=True,
        description="Share compressor model with topic segmenter when possible.",
    )
    messages_use: Literal["user_only", "assistant_only", "hybrid"] = Field(
        default="user_only",
        description="Message role filtering mode in extraction.",
    )
    metadata_generate: bool = Field(
        default=True,
        description="Whether to generate metadata/facts via LLM extraction.",
    )
    text_summary: bool = Field(
        default=True,
        description="Whether to keep summarized form alongside memory.",
    )
    extract_threshold: float = Field(
        default=0.1,
        ge=0.0,
        description="Extraction trigger threshold used by LightMem.",
    )
    index_strategy: Literal["embedding", "context", "hybrid"] = Field(
        default="embedding",
        description="LightMem indexing strategy.",
    )
    retrieve_strategy: Literal["embedding", "context", "hybrid"] = Field(
        default="embedding",
        description="LightMem retrieval strategy.",
    )
    update_mode: Literal["offline", "online"] = Field(
        default="offline",
        description="LightMem update mode.",
    )
    extraction_mode: Literal["flat", "event"] = Field(
        default="flat",
        description="LightMem extraction mode; default matches upstream LoCoMo script.",
    )

    llmlingua_model_path: str = Field(
        default="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        description="Model path/name for LLMLingua-2 pre-compressor.",
    )
    llmlingua_device_map: str = Field(
        default="cpu",
        description="Device map for LLMLingua model loading.",
    )
    llmlingua_use_v2: bool = Field(
        default=True,
        description="Whether to enable LLMLingua-2 mode in PromptCompressor.",
    )
    llmlingua_compress_rate: float = Field(
        default=0.6,
        gt=0.0,
        le=1.0,
        description="Compression rate for LLMLingua pre-compression.",
    )
    topic_segmenter_model_name: str = Field(
        default="llmlingua-2",
        description="Topic segmenter model name.",
    )

    enable_offline_update: bool = Field(
        default=True,
        description="Whether flush triggers queue construction + offline update.",
    )
    construct_queue_top_k: int = Field(
        default=20,
        ge=1,
        description="Top-k neighbors when constructing update queues.",
    )
    construct_queue_keep_top_n: int = Field(
        default=10,
        ge=1,
        description="Keep top-n candidates in update queues.",
    )
    construct_queue_workers: int = Field(
        default=8,
        ge=1,
        description="Worker threads for queue construction.",
    )
    offline_update_score_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for offline memory updates.",
    )
    offline_update_workers: int = Field(
        default=5,
        ge=1,
        description="Worker threads for offline update.",
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
