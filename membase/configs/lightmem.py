import os

from pydantic import Field, JsonValue, PrivateAttr, model_validator
from typing import Any, Literal, Self

from .base import MemBaseConfig


class LightMemConfig(MemBaseConfig):
    """Configuration for the external LightMem adapter."""

    _qdrant_path_was_default: bool = PrivateAttr(default=False)
    _collection_name_was_default: bool = PrivateAttr(default=False)
    _summary_collection_name_was_default: bool = PrivateAttr(default=False)

    api_key: str | None = Field(
        default=None,
        description="OpenAI-compatible API key for LightMem construction calls.",
    )
    base_url: str | None = Field(
        default=None,
        description="Optional OpenAI-compatible base URL for LightMem construction calls.",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model used by LightMem's memory manager.",
    )
    llm_temperature: float = Field(default=0.1)
    llm_max_tokens: int = Field(default=16000, ge=1)
    llm_top_p: float = Field(default=0.1)

    pre_compress: bool = Field(default=True)
    llmlingua_model_path: str = Field(
        default="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        description="Local path or supported model id for LLMLingua-2.",
    )
    compress_rate: float = Field(default=0.8)
    compress_target_token: int = Field(default=-1)
    topic_segment: bool = Field(default=True)
    precomp_topic_shared: bool = Field(default=True)

    messages_use: Literal["user_only", "assistant_only", "hybrid"] = Field(
        default="user_only"
    )
    metadata_generate: bool = Field(default=True)
    text_summary: bool = Field(default=True)
    extraction_mode: Literal["flat", "event"] = Field(default="flat")
    extract_threshold: float = Field(default=0.1)

    embedding_provider: Literal["huggingface", "openai"] = Field(
        default="huggingface"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name or local path.",
    )
    embedding_dims: int = Field(default=384, ge=1)
    embedding_device: str = Field(default="cpu")
    embedding_model_kwargs: dict[str, JsonValue] = Field(default_factory=dict)
    embedding_api_key: str | None = Field(default=None)
    embedding_base_url: str | None = Field(default=None)

    enable_summary: bool = Field(default=False)
    summary_top_k: int = Field(default=5, ge=1)
    summary_time_window: int = Field(default=3600, ge=1)
    summary_top_k_seeds: int = Field(default=15, ge=1)
    summary_enable_cross_event: bool = Field(default=True)
    summary_retrieval_scope: Literal["global", "historical"] = Field(default="global")

    qdrant_path: str | None = Field(
        default=None,
        description="Local Qdrant persistence directory. Defaults to <save_dir>/qdrant.",
    )
    collection_name: str | None = Field(
        default=None,
        description="Main memory collection name. Defaults to 'memory_entries'.",
    )
    summary_collection_name: str | None = Field(
        default=None,
        description="Summary collection name. Defaults to 'memory_summaries'.",
    )

    @model_validator(mode="after")
    def _populate_paths(self) -> Self:
        default_qdrant_path = os.path.join(self.save_dir, "qdrant")
        if self.qdrant_path is None or (
            getattr(self, "_qdrant_path_was_default", False)
            and self.qdrant_path != default_qdrant_path
        ):
            self._qdrant_path_was_default = True
            self.qdrant_path = default_qdrant_path
        else:
            self._qdrant_path_was_default = False

        if self.collection_name is None:
            self._collection_name_was_default = True
            self.collection_name = "memory_entries"
        else:
            self._collection_name_was_default = False

        if self.summary_collection_name is None:
            self._summary_collection_name_was_default = True
            self.summary_collection_name = "memory_summaries"
        else:
            self._summary_collection_name_was_default = False
        return self

    def get_llm_models(self) -> list[str]:
        return [self.llm_model]

    def build_lightmem_config(self) -> dict[str, Any]:
        """Build the nested configuration dict expected by LightMem."""
        text_embedder_configs: dict[str, Any] = {
            "model": self.embedding_model,
            "embedding_dims": self.embedding_dims,
        }
        if self.embedding_provider == "huggingface":
            text_embedder_configs["model_kwargs"] = {
                "device": self.embedding_device,
                **self.embedding_model_kwargs,
            }
        else:
            if self.embedding_api_key is not None:
                text_embedder_configs["api_key"] = self.embedding_api_key
            if self.embedding_base_url is not None:
                text_embedder_configs["openai_base_url"] = self.embedding_base_url
            if self.embedding_model_kwargs:
                text_embedder_configs["model_kwargs"] = dict(self.embedding_model_kwargs)

        cfg: dict[str, Any] = {
            "pre_compress": self.pre_compress,
            "pre_compressor": {
                "model_name": "llmlingua-2",
                "configs": {
                    "llmlingua_config": {
                        "model_name": self.llmlingua_model_path,
                        "device_map": self.embedding_device,
                        "use_llmlingua2": True,
                    },
                    "compress_config": {
                        "instruction": "",
                        "rate": self.compress_rate,
                        "target_token": self.compress_target_token,
                    },
                },
            },
            "topic_segment": self.topic_segment,
            "precomp_topic_shared": self.precomp_topic_shared,
            "topic_segmenter": {
                "model_name": "llmlingua-2",
            },
            "messages_use": self.messages_use,
            "metadata_generate": self.metadata_generate,
            "text_summary": self.text_summary,
            "memory_manager": {
                "model_name": "openai",
                "configs": {
                    "model": self.llm_model,
                    "api_key": self.api_key,
                    "openai_base_url": self.base_url,
                    "temperature": self.llm_temperature,
                    "max_tokens": self.llm_max_tokens,
                    "top_p": self.llm_top_p,
                },
            },
            "extract_threshold": self.extract_threshold,
            "index_strategy": "embedding",
            "text_embedder": {
                "model_name": self.embedding_provider,
                "configs": text_embedder_configs,
            },
            "retrieve_strategy": "embedding",
            "embedding_retriever": {
                "model_name": "qdrant",
                "configs": {
                    "collection_name": self.collection_name,
                    "embedding_model_dims": self.embedding_dims,
                    "path": self.qdrant_path,
                    "on_disk": True,
                },
            },
            "update": "offline",
            "extraction_mode": self.extraction_mode,
        }

        if self.enable_summary:
            cfg["summary_retriever"] = {
                "model_name": "qdrant",
                "configs": {
                    "collection_name": self.summary_collection_name,
                    "embedding_model_dims": self.embedding_dims,
                    "path": self.qdrant_path,
                    "on_disk": True,
                },
            }

        return cfg
