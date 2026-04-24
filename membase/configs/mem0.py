import os
from typing import Any, Literal, Self

from pydantic import ConfigDict, Field, JsonValue, model_validator

from .base import MemBaseConfig


class Mem0Config(MemBaseConfig):
    """The default configuration for Mem0."""

    model_config = ConfigDict(extra="forbid")

    llm_provider: Literal[
        "ollama", "openai", "groq", "together", "aws_bedrock", "litellm", 
        "azure_openai", "openai_structured", "anthropic", "azure_openai_structured", 
        "gemini", "deepseek", "xai", "sarvam", "lmstudio", "vllm", "langchain", 
    ] = Field(
        default="openai",
        description=(
            "LLM provider for Mem0. Common providers include `'openai'`, `'ollama'`, "
            "`'anthropic'`, `'groq'`, `'litellm'`, etc. "
            "See https://docs.mem0.ai/components/llms/overview for the full list."
        ),
    )
    llm_model: str = Field(
        default="gpt-4.1-mini",
        description="LLM model name.",
        examples=["gpt-4.1-mini", "grok-3-beta", "claude-sonnet-4-20250514"],
    )
    llm_config: dict[str, JsonValue] = Field(
        default_factory=dict,
        description=(
            "Additional keyword arguments forwarded to the Mem0 LLM configuration. "
            "Common keys include `api_key`, `openai_base_url`, `temperature`, "
            "`max_tokens`, `top_p`, etc."
        ),
        examples=[
            {
                "api_key": "sk-...",
                "openai_base_url": "https://api.openai.com/v1",
                "temperature": 0.0,
                "max_tokens": 4096,
            },
        ],
    )

    embedder_provider: Literal[
        "openai", "ollama", "huggingface", "azure_openai", "gemini", "vertexai", 
        "together", "lmstudio", "langchain", "aws_bedrock", "fastembed", 
    ] = Field(
        default="huggingface",
        description=(
            "Embedder provider. `'huggingface'` loads the model locally via "
            "`sentence-transformers`. `'openai'` calls the OpenAI embedding API. "
            "`'ollama'` calls the Ollama embedding API. "
            "See https://docs.mem0.ai/components/embedders/overview for the full list."
        ),
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Embedding model name or path.",
        examples=["all-MiniLM-L6-v2", "text-embedding-3-small", "models/text-embedding-004"],
    )
    embedding_model_dims: int = Field(
        default=384,
        description=(
            "Embedding dimension. It must match the chosen `embedding_model`."
        ),
    )
    embedding_config: dict[str, JsonValue] = Field(
        default_factory=dict,
        description=(
            "Additional keyword arguments forwarded to the Mem0 embedder configuration. "
            "For the `'huggingface'` provider, a common key is `model_kwargs` "
            "(e.g., `{'device': 'cuda'}`). For the `'openai'` provider, common keys "
            "include `api_key`, `openai_base_url`, etc."
        ),
        examples=[
            {"model_kwargs": {"device": "cuda"}},
            {"api_key": "sk-...", "openai_base_url": "https://api.openai.com/v1"},
        ],
    )

    collection_name: str | None = Field(
        default=None,
        description=(
            "Qdrant collection name. If not provided, it defaults to `user_id`. "
            "When running multiple users in the same Qdrant instance, each user should "
            "have a distinct collection name to avoid data conflicts."
        ),
    ) 

    history_db_path: str | None = Field(
        default=None,
        description=(
            "Path to the SQLite history database that records the operation history "
            "(ADD / UPDATE / DELETE) of every memory unit. If not provided, it "
            "defaults to `<save_dir>/history.db`. When running in parallel, each "
            "user memory layer instance must use a separate `history_db_path` to "
            "avoid SQLite file-level lock contention."
        ),
    )

    reranker_provider: Literal[
        "cohere", "sentence_transformer", "zero_entropy", "llm_reranker", "huggingface",
    ] | None = Field(
        default=None,
        description=(
            "Reranker provider. Common providers include `'cohere'`, "
            "`'sentence_transformer'`, `'huggingface'`, `'llm_reranker'`, `'zero_entropy'`. "
            "`None` disables reranking. "
            "See https://docs.mem0.ai/components/rerankers/overview for the full list."
        ),
    )
    reranker_config: dict[str, JsonValue] = Field(
        default_factory=dict,
        description=(
            "Additional keyword arguments forwarded to the reranker configuration. "
            "Common keys include `model`, `api_key`, `top_k`, etc."
        ),
        examples=[
            {"model": "rerank-english-v3.0", "api_key": "cohere-api-key", "top_k": 5},
            {"model": "gpt-4o-mini", "provider": "openai", "api_key": "openai-api-key", "top_k": 5},
        ],
    )

    custom_instructions: str | None = Field(
        default=None,
        description=(
            "Additional extraction instructions forwarded to Mem0 v2's "
            "`custom_instructions` configuration field."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_v1_fields(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values

        legacy_fields = {
            "graph_store_provider",
            "graph_store_config",
            "custom_fact_extraction_prompt",
            "custom_update_memory_prompt",
        }
        present = sorted(field for field in legacy_fields if field in values)
        if present:
            raise ValueError(
                "Mem0 v2 no longer supports the legacy v1 config field(s): "
                f"{', '.join(present)}. Use `custom_instructions` and rebuild memory from scratch."
            )
        return values

    @model_validator(mode="after")
    def _force_persistence(self) -> Self:
        """Ensure persistence paths are set for Qdrant and history DB."""
        if self.collection_name is None:
            self.collection_name = self.user_id

        if self.history_db_path is None:
            self.history_db_path = os.path.join(self.save_dir, "history.db")
        return self

    def get_llm_models(self) -> list[str]:
        return [self.llm_model]

    def build_mem0_config(self) -> dict[str, Any]:
        """Build the nested configuration dictionary expected 
        by mem0's internal configuration interface.

        Returns:
            `dict[str, Any]`: 
                The Mem0 configuration dictionary.
        """
        cfg = {
            "version": "v1.1",
            "llm": {
                "provider": self.llm_provider,
                "config": {
                    "model": self.llm_model,
                    **self.llm_config,
                },
            },
            "embedder": {
                "provider": self.embedder_provider,
                "config": {
                    "model": self.embedding_model,
                    "embedding_dims": self.embedding_model_dims,
                    **self.embedding_config,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": self.collection_name,
                    "embedding_model_dims": self.embedding_model_dims,
                    "path": self.save_dir,
                    "on_disk": True,
                },
            },
            "history_db_path": self.history_db_path,
        }

        if self.custom_instructions is not None:
            cfg["custom_instructions"] = self.custom_instructions

        if self.reranker_provider is not None:
            cfg["reranker"] = {
                "provider": self.reranker_provider,
                "config": self.reranker_config or None,
            }

        return cfg
