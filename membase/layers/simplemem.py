import functools
import json
import os
from dataclasses import replace
from typing import Any, ClassVar
import numpy as np
from openai import OpenAI

from simplemem import SimpleMemConfig as UpstreamSimpleMemConfig
from simplemem import SimpleMemSystem
from simplemem import set_config as set_simplemem_config

from .base import MemBaseLayer
from ..configs.simplemem import SimpleMemConfig
from ..model_types.dataset import Message
from ..model_types.memory import MemoryEntry
from ..utils import (
    PatchSpec,
    make_attr_patch,
    token_monitor,
    MonkeyPatcher,
    CostStateManager,
)


class _RemoteEmbeddingModel:
    """OpenAI-compatible embedding adapter for upstream SimpleMem."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        dimension: int,
        **_: Any,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.model_type = "openai_compatible"
        self.supports_query_prompt = False
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def _normalize_vectors(self, vectors: list[list[float]]) -> np.ndarray:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=0)

        # Match sentence-transformers' normalize_embeddings=True behavior.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        arr = arr / norms

        if arr.shape[1] != self.dimension:
            raise ValueError(
                "Remote embedding dimension mismatch: "
                f"expected {self.dimension}, got {arr.shape[1]}"
            )
        return arr

    def encode(self, texts: list[str] | str, is_query: bool = False) -> np.ndarray:
        del is_query
        if isinstance(texts, str):
            texts = [texts]
        if len(texts) == 0:
            return np.zeros((0, self.dimension), dtype=np.float32)

        resp = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        vectors = [item.embedding for item in resp.data]
        return self._normalize_vectors(vectors)

    def encode_single(self, text: str, is_query: bool = False) -> np.ndarray:
        return self.encode([text], is_query=is_query)[0]

    def encode_query(self, queries: list[str]) -> np.ndarray:
        return self.encode(queries, is_query=True)

    def encode_documents(self, documents: list[str]) -> np.ndarray:
        return self.encode(documents, is_query=False)


def _build_shared_simplemem_config(config: SimpleMemConfig) -> UpstreamSimpleMemConfig:
    """Build process-level SimpleMem config with shared-only fields.

    Do not include user-specific storage paths here; those are passed per-system
    instance via ``db_path`` to avoid cross-user concurrency interference.
    """
    default_cfg = UpstreamSimpleMemConfig()
    overrides = {
        "openai_api_key": config.api_key,
        "openai_base_url": config.llm_base_url,
        "llm_model": config.llm_model,
        "embedding_model": config.embedding_model,
        "embedding_dimension": config.embedding_dimension,
        "memory_table_name": config.memory_table_name,
        "window_size": config.window_size,
        "overlap_size": config.overlap_size,
        "semantic_top_k": config.semantic_top_k,
        "keyword_top_k": config.keyword_top_k,
        "structured_top_k": config.structured_top_k,
        "enable_thinking": config.enable_thinking,
        "use_streaming": config.use_streaming,
        "use_json_format": config.use_json_format,
        "enable_parallel_processing": config.enable_parallel_processing,
        "max_parallel_workers": config.max_parallel_workers,
        "enable_parallel_retrieval": config.enable_parallel_retrieval,
        "max_retrieval_workers": config.max_retrieval_workers,
        "enable_planning": config.enable_planning,
        "enable_reflection": config.enable_reflection,
        "max_reflection_rounds": config.max_reflection_rounds,
    }

    # Prefer dataclass-style immutable update, fallback to setattr for compatibility.
    try:
        return replace(default_cfg, **overrides)
    except Exception:
        for key, value in overrides.items():
            if hasattr(default_cfg, key):
                setattr(default_cfg, key, value)
        return default_cfg


def _extract_simplemem_messages(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    """Normalize SimpleMem LLM client messages for token monitor."""
    messages = kwargs.get("messages", args[0] if len(args) > 0 else None)
    if isinstance(messages, list):
        normalized: list[dict[str, Any]] = []
        for item in messages:
            if isinstance(item, dict):
                role = str(item.get("role", "user"))
                content = str(item.get("content", ""))
                normalized.append({"role": role, "content": content})
            else:
                normalized.append({"role": "user", "content": str(item)})
        return normalized
    return []


def _detect_simplemem_op_type(messages: list[dict[str, Any]]) -> str:
    """Classify SimpleMem internal LLM call by prompt shape."""
    if len(messages) == 0:
        return "generation"
    first = messages[0]
    system_text = str(first.get("content", ""))
    if "professional information extraction assistant" in system_text:
        return "generation"
    if "query analysis assistant" in system_text:
        return "retrieval_query_analysis"
    if "search query generation assistant" in system_text:
        return "retrieval_query_generation"
    if "information adequacy evaluator" in system_text:
        return "retrieval_adequacy_check"
    if "search strategy assistant" in system_text:
        return "retrieval_additional_queries"
    if "professional Q&A assistant" in system_text:
        return "answer_generation"
    return "generation"


def _entry_get(item: Any, key: str, default: Any = None) -> Any:
    """Read a field from either object-style or dict-style SimpleMem entries."""
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


class SimpleMemLayer(MemBaseLayer):
    layer_type: ClassVar[str] = "SimpleMem"

    def __init__(self, config: SimpleMemConfig) -> None:
        self._init_layer(config)
        self.config = config
        self.embedding_model_name = config.embedding_model

    def _init_layer(self, config: SimpleMemConfig) -> None:
        os.makedirs(config.save_dir, exist_ok=True)

        # Shared process-level config for upstream package.
        set_simplemem_config(_build_shared_simplemem_config(config))

        system_ctor = functools.partial(
            SimpleMemSystem,
            api_key=config.api_key,
            model=config.llm_model,
            base_url=config.llm_base_url,
            db_path=config.save_dir,
            table_name=config.memory_table_name,
            clear_db=False,
            enable_thinking=config.enable_thinking,
            use_streaming=config.use_streaming,
            enable_planning=config.enable_planning,
            enable_reflection=config.enable_reflection,
            max_reflection_rounds=config.max_reflection_rounds,
            enable_parallel_processing=config.enable_parallel_processing,
            max_parallel_workers=config.max_parallel_workers,
            enable_parallel_retrieval=config.enable_parallel_retrieval,
            max_retrieval_workers=config.max_retrieval_workers,
        )

        if config.uses_remote_embedding():
            embedding_api_key = config.embedding_api_key or config.api_key
            getter, setter = make_attr_patch(
                __import__("simplemem.system", fromlist=["EmbeddingModel"]),
                "EmbeddingModel",
            )

            def _remote_embedding_wrapper(_original: Any):
                @functools.wraps(_original)
                def _factory(*args: Any, **kwargs: Any) -> _RemoteEmbeddingModel:
                    model_name = kwargs.pop("model_name", None)
                    return _RemoteEmbeddingModel(
                        model_name=model_name or config.embedding_model,
                        base_url=str(config.embedding_base_url),
                        api_key=embedding_api_key,
                        dimension=config.embedding_dimension,
                        **kwargs,
                    )

                return _factory

            spec = PatchSpec(
                name="simplemem.system.EmbeddingModel",
                getter=getter,
                setter=setter,
                wrapper=_remote_embedding_wrapper,
            )
            with MonkeyPatcher([spec]):
                self.memory_layer = system_ctor()
        else:
            # User-specific path is passed directly to system constructor.
            self.memory_layer = system_ctor()

    def add_message(self, message: Message, **kwargs: Any) -> None:
        try:
            self.memory_layer.add_dialogue(
                speaker=message.name,
                content=message.content,
                timestamp=message.timestamp,
            )
        except Exception as e:
            print(
                "Error in add_message method in SimpleMemLayer: "
                f"\n\t{e.__class__.__name__}: {e}"
            )

    def add_messages(self, messages: list[Message], **kwargs: Any) -> None:
        for message in messages:
            self.add_message(message, **kwargs)

    def flush(self) -> None:
        self.memory_layer.finalize()

    def retrieve(self, query: str, k: int = 10, **kwargs: Any) -> list[MemoryEntry]:
        reflection_flag = kwargs.pop("enable_reflection", None)
        retrieve_kwargs: dict[str, Any] = {"query": query}
        if reflection_flag is not None:
            retrieve_kwargs["enable_reflection"] = reflection_flag

        patch_specs: list[PatchSpec] = []
        try:
            CostStateManager.get(self.config.llm_model)
            patch_specs = self.get_patch_specs()
        except Exception:
            patch_specs = []

        try:
            if len(patch_specs) > 0:
                with MonkeyPatcher(patch_specs):
                    memories = self.memory_layer.hybrid_retriever.retrieve(**retrieve_kwargs)
            else:
                memories = self.memory_layer.hybrid_retriever.retrieve(**retrieve_kwargs)
        except Exception as e:
            print(
                "Error in retrieve method in SimpleMemLayer: "
                f"\n\t{e.__class__.__name__}: {e}"
            )
            return []

        if memories is None:
            return []

        memory_items = list(memories) if not isinstance(memories, list) else memories

        outputs: list[MemoryEntry] = []
        for item in memory_items[:k]:
            content = str(
                _entry_get(
                    item,
                    "lossless_restatement",
                    _entry_get(item, "content", ""),
                )
            )
            keywords = _entry_get(item, "keywords", [])
            persons = _entry_get(item, "persons", [])
            entities = _entry_get(item, "entities", [])
            timestamp = _entry_get(item, "timestamp")
            location = _entry_get(item, "location")
            topic = _entry_get(item, "topic")

            metadata = {
                "entry_id": _entry_get(item, "entry_id"),
                "keywords": list(keywords) if isinstance(keywords, (list, tuple, set)) else [],
                "timestamp": timestamp,
                "location": location,
                "persons": list(persons) if isinstance(persons, (list, tuple, set)) else [],
                "entities": list(entities) if isinstance(entities, (list, tuple, set)) else [],
                "topic": topic,
            }

            parts = [f"Memory: {content}"]
            if isinstance(timestamp, str) and timestamp.strip() != "":
                parts.append(f"Time: {timestamp}")
            if isinstance(location, str) and location.strip() != "":
                parts.append(f"Location: {location}")
            if isinstance(persons, (list, tuple, set)) and len(persons) > 0:
                parts.append(f"Persons: {', '.join(str(p) for p in persons)}")
            if isinstance(entities, (list, tuple, set)) and len(entities) > 0:
                parts.append(f"Entities: {', '.join(str(e) for e in entities)}")
            if isinstance(topic, str) and topic.strip() != "":
                parts.append(f"Topic: {topic}")

            outputs.append(
                MemoryEntry(
                    content=content,
                    formatted_content="\n".join(parts),
                    metadata=metadata,
                )
            )
        return outputs

    def delete(self, memory_id: str) -> bool:
        print(
            "SimpleMemLayer.delete is not supported by upstream SimpleMem API. "
            f"Requested memory_id={memory_id}."
        )
        return False

    def update(self, memory_id: str, **kwargs: Any) -> bool:
        print(
            "SimpleMemLayer.update is not supported by upstream SimpleMem API. "
            f"Requested memory_id={memory_id}."
        )
        return False

    def save_memory(self) -> None:
        os.makedirs(self.config.save_dir, exist_ok=True)
        config_path = os.path.join(self.config.save_dir, "config.json")
        config_dict = {
            "layer_type": self.layer_type,
            **self.config.model_dump(mode="python"),
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)

    def load_memory(self, user_id: str | None = None) -> bool:
        if user_id is None:
            user_id = self.config.user_id

        config_path = os.path.join(self.config.save_dir, "config.json")
        if not os.path.exists(config_path):
            return False

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        if user_id != config_dict["user_id"]:
            raise ValueError(
                f"The user id in the config file ({config_dict['user_id']}) "
                f"does not match the user id ({user_id}) in the function call."
            )

        config = SimpleMemConfig(**config_dict)
        self._init_layer(config)
        self.config = config
        self.embedding_model_name = config.embedding_model

        # Verify there is at least one persisted memory entry.
        try:
            entries = self.memory_layer.get_all_memories()
            return len(entries) > 0
        except Exception:
            return False

    def get_patch_specs(self) -> list[PatchSpec]:
        llm_client = getattr(self.memory_layer, "llm_client", None)
        if llm_client is None or not hasattr(llm_client, "chat_completion"):
            return []
        getter, setter = make_attr_patch(llm_client, "chat_completion")

        monitor_wrapper = token_monitor(
            extract_model_name=lambda *args, **kwargs: (
                self.config.llm_model,
                {},
            ),
            extract_input_dict=lambda *args, **kwargs: {
                "messages": _extract_simplemem_messages(*args, **kwargs),
                "metadata": {
                    "op_type": _detect_simplemem_op_type(
                        _extract_simplemem_messages(*args, **kwargs)
                    )
                },
            },
            extract_output_dict=lambda result: {
                "messages": result if isinstance(result, str) else str(result),
            },
        )

        def _simplemem_chat_completion_wrapper(func):
            monitored_func = monitor_wrapper(func)

            @functools.wraps(monitored_func)
            def wrapped(*args, **kwargs):
                return monitored_func(*args, **kwargs)

            return wrapped

        spec = PatchSpec(
            name=f"{llm_client.__class__.__name__}.chat_completion",
            getter=getter,
            setter=setter,
            wrapper=_simplemem_chat_completion_wrapper,
        )
        return [spec]
