from __future__ import annotations

from typing import Any

import numpy as np

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
        self.model_type = "sentence_transformer"
        self.supports_query_prompt = False
        self.dimension = settings.embedding_dimension
        self._init_model()

    def _init_model(self) -> None:
        from sentence_transformers import SentenceTransformer

        if self.model_name.startswith("qwen3"):
            qwen3_models = {
                "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
                "qwen3-4b": "Qwen/Qwen3-Embedding-4B",
                "qwen3-8b": "Qwen/Qwen3-Embedding-8B",
            }
            model_path = qwen3_models.get(self.model_name, self.model_name)
            try:
                if self.use_optimization:
                    self.model = SentenceTransformer(
                        model_path,
                        model_kwargs={
                            "attn_implementation": "flash_attention_2",
                            "device_map": "auto",
                        },
                        tokenizer_kwargs={"padding_side": "left"},
                        trust_remote_code=True,
                    )
                else:
                    self.model = SentenceTransformer(
                        model_path,
                        trust_remote_code=True,
                    )
                self.model_type = "qwen3_sentence_transformer"
                self.supports_query_prompt = (
                    hasattr(self.model, "prompts")
                    and "query" in getattr(self.model, "prompts", {})
                )
            except Exception:
                self.model = SentenceTransformer(model_path, trust_remote_code=True)
                self.model_type = "qwen3_sentence_transformer"
        else:
            self.model = SentenceTransformer(self.model_name)

        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str] | str, is_query: bool = False) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if (
            self.model_type == "qwen3_sentence_transformer"
            and self.supports_query_prompt
            and is_query
        ):
            try:
                return self.model.encode(
                    texts,
                    prompt_name="query",
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
            except Exception:
                pass
        return self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def encode_single(self, text: str, is_query: bool = False) -> np.ndarray:
        return self.encode([text], is_query=is_query)[0]

    def encode_query(self, queries: list[str]) -> np.ndarray:
        return self.encode(queries, is_query=True)

    def encode_documents(self, documents: list[str]) -> np.ndarray:
        return self.encode(documents, is_query=False)

