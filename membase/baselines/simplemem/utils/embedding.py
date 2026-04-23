from __future__ import annotations

from typing import Any

import numpy as np
from openai import OpenAI

from ..settings import SimpleMemSettings


class EmbeddingModel:
    """Embedding wrapper for the vendored SimpleMem backend."""

    def __init__(
        self,
        settings: SimpleMemSettings,
        model_name: str | None = None,
        use_optimization: bool = True,
    ) -> None:
        self.settings = settings
        self.model_name = model_name or settings.embedding_model
        self.use_optimization = use_optimization
        self.model_type = settings.embedding_provider
        self.supports_query_prompt = False
        self.dimension = settings.embedding_dimension
        self._init_model()

    def _init_model(self) -> None:
        if self.model_type == "openai":
            self._init_openai_client()
            return

        self._init_sentence_transformer()

    def _init_openai_client(self) -> None:
        self.client = OpenAI(
            api_key=self.settings.embedding_api_key,
            base_url=self.settings.embedding_base_url,
        )
        self.supports_query_prompt = False

    def _init_sentence_transformer(self) -> None:
        from sentence_transformers import SentenceTransformer

        model_kwargs = {}
        encode_kwargs = {}
        if self.settings.embedding_model_kwargs:
            model_kwargs = dict(
                self.settings.embedding_model_kwargs.get("model_kwargs", {})
            )
            encode_kwargs = dict(
                self.settings.embedding_model_kwargs.get("encode_kwargs", {})
            )
        self._encode_kwargs = encode_kwargs

        if self.model_name.startswith("qwen3"):
            qwen3_models = {
                "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
                "qwen3-4b": "Qwen/Qwen3-Embedding-4B",
                "qwen3-8b": "Qwen/Qwen3-Embedding-8B",
            }
            model_path = qwen3_models.get(self.model_name, self.model_name)
            try:
                if self.use_optimization:
                    optimized_model_kwargs = {
                        "attn_implementation": "flash_attention_2",
                        "device_map": "auto",
                        **model_kwargs,
                    }
                    self.model = SentenceTransformer(
                        model_path,
                        model_kwargs=optimized_model_kwargs,
                        tokenizer_kwargs={"padding_side": "left"},
                        trust_remote_code=True,
                    )
                else:
                    self.model = SentenceTransformer(
                        model_path,
                        model_kwargs=model_kwargs or None,
                        trust_remote_code=True,
                    )
                self.model_type = "qwen3_sentence_transformer"
                self.supports_query_prompt = (
                    hasattr(self.model, "prompts")
                    and "query" in getattr(self.model, "prompts", {})
                )
            except Exception:
                if model_kwargs:
                    self.model = SentenceTransformer(
                        model_path,
                        model_kwargs=model_kwargs,
                        trust_remote_code=True,
                    )
                else:
                    self.model = SentenceTransformer(model_path, trust_remote_code=True)
                self.model_type = "qwen3_sentence_transformer"
        else:
            if model_kwargs:
                self.model = SentenceTransformer(self.model_name, model_kwargs=model_kwargs)
            else:
                self.model = SentenceTransformer(self.model_name)

        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str] | str, is_query: bool = False) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if self.model_type == "openai":
            return self._encode_openai(texts)
        if (
            self.model_type == "qwen3_sentence_transformer"
            and self.supports_query_prompt
            and is_query
        ):
            try:
                kwargs = {
                    "prompt_name": "query",
                    "show_progress_bar": False,
                    "normalize_embeddings": True,
                    **getattr(self, "_encode_kwargs", {}),
                }
                return self.model.encode(
                    texts,
                    **kwargs,
                )
            except Exception:
                pass
        kwargs = {
            "show_progress_bar": False,
            "normalize_embeddings": True,
            **getattr(self, "_encode_kwargs", {}),
        }
        return self.model.encode(
            texts,
            **kwargs,
        )

    def _encode_openai(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
            **(self.settings.embedding_model_kwargs or {}),
        )
        vectors = [item.embedding for item in response.data]
        return np.asarray(vectors, dtype=np.float32)

    def encode_single(self, text: str, is_query: bool = False) -> np.ndarray:
        return self.encode([text], is_query=is_query)[0]

    def encode_query(self, queries: list[str]) -> np.ndarray:
        return self.encode(queries, is_query=True)

    def encode_documents(self, documents: list[str]) -> np.ndarray:
        return self.encode(documents, is_query=False)
