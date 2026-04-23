import importlib
import json
import logging
import os
import shutil
from typing import Any, ClassVar

from .base import MemBaseLayer
from ..configs.lightmem import LightMemConfig
from ..model_types.dataset import Message
from ..model_types.memory import MemoryEntry
from ..utils import PatchSpec, make_attr_patch, token_monitor


LOGGER = logging.getLogger(__name__)


def _require_lightmem() -> tuple[type[Any], type[Any]]:
    try:
        lightmemory_cls = importlib.import_module(
            "lightmem.memory.lightmem"
        ).LightMemory
        openai_manager_cls = importlib.import_module(
            "lightmem.factory.memory_manager.openai"
        ).OpenaiManager
        return lightmemory_cls, openai_manager_cls
    except ImportError as exc:
        raise ImportError(
            "LightMem is not installed. Install it first with "
            "`pip install -e /home/wiang/LightMem` or another compatible "
            "`lightmem` package source before using `--memory-type LightMem`."
        ) from exc


def _collection_exists(retriever: Any, collection_name: str) -> bool:
    for collection in retriever.list_cols().collections:
        if collection.name == collection_name:
            return True
    return False


def _safe_collection_exists(retriever: Any | None, collection_name: str) -> bool:
    if retriever is None:
        return False
    try:
        return _collection_exists(retriever, collection_name)
    except Exception:
        return False


def _format_lightmem_memory(payload: dict[str, Any], *, source: str) -> str:
    if source == "summary":
        parts = [f"[SUMMARY] {payload.get('summary', payload.get('memory', ''))}"]
        time_range = payload.get("time_range") or {}
        if time_range.get("start") or time_range.get("end"):
            parts.append(
                f"Time Range: {time_range.get('start', '')} -> {time_range.get('end', '')}"
            )
        if payload.get("entry_count") is not None:
            parts.append(f"Covered Entries: {payload['entry_count']}")
        return "\n".join(parts)

    parts = [payload.get("memory", "")]
    if payload.get("time_stamp"):
        parts.append(f"Time: {payload['time_stamp']}")
    if payload.get("weekday"):
        parts.append(f"Weekday: {payload['weekday']}")
    if payload.get("speaker_name"):
        parts.append(f"Speaker: {payload['speaker_name']}")
    if payload.get("topic_summary"):
        parts.append(f"Topic Summary: {payload['topic_summary']}")
    if payload.get("category"):
        parts.append(f"Category: {payload['category']}")
    if payload.get("subcategory"):
        parts.append(f"Subcategory: {payload['subcategory']}")
    return "\n".join(parts)


class LightMemLayer(MemBaseLayer):
    layer_type: ClassVar[str] = "LightMem"
    ingest_granularity: ClassVar[str] = "session"

    def __init__(self, config: LightMemConfig) -> None:
        self.config = config
        self.memory_layer: Any | None = None
        self._fresh_initialized = False
        self._manager_patch_target: Any | None = None

    def _ensure_lightmemory(self, mode: str) -> Any:
        if self.memory_layer is not None:
            return self.memory_layer

        LightMemory, _ = _require_lightmem()

        if mode == "fresh" and not self._fresh_initialized:
            if os.path.isdir(self.config.qdrant_path):
                shutil.rmtree(self.config.qdrant_path)
            self._fresh_initialized = True

        os.makedirs(self.config.qdrant_path, exist_ok=True)
        self.memory_layer = LightMemory.from_config(self.config.build_lightmem_config())
        self._manager_patch_target = getattr(self.memory_layer, "manager", None)
        return self.memory_layer

    def _to_lightmem_turns(
        self,
        messages: list[Message],
        session_started_at: str,
    ) -> list[list[dict[str, Any]]]:
        filtered = []
        for message in messages:
            if message.role == "system":
                LOGGER.warning(
                    "Skip system message '%s' when constructing LightMem session.",
                    message.id,
                )
                continue
            filtered.append(
                {
                    "role": message.role,
                    "content": message.content,
                    "speaker_id": message.metadata.get("speaker_id", message.name),
                    "speaker_name": message.name,
                    "time_stamp": session_started_at,
                }
            )

        turns: list[list[dict[str, Any]]] = []
        pending_user: dict[str, Any] | None = None
        for message in filtered:
            if message["role"] == "user":
                if pending_user is not None:
                    turns.append(
                        [
                            pending_user,
                            {
                                "role": "assistant",
                                "content": "",
                                "speaker_id": pending_user["speaker_id"],
                                "speaker_name": pending_user["speaker_name"],
                                "time_stamp": session_started_at,
                            },
                        ]
                    )
                pending_user = message
                continue

            if pending_user is None:
                continue

            turns.append([pending_user, message])
            pending_user = None

        if pending_user is not None:
            turns.append(
                [
                    pending_user,
                    {
                        "role": "assistant",
                        "content": "",
                        "speaker_id": pending_user["speaker_id"],
                        "speaker_name": pending_user["speaker_name"],
                        "time_stamp": session_started_at,
                    },
                ]
            )
        return turns

    def add_message(self, message: Message, **kwargs: Any) -> None:
        raise NotImplementedError(
            "LightMem construction is session-level in MemBase. Use `add_messages()`."
        )

    def add_messages(self, messages: list[Message], **kwargs: Any) -> None:
        session_started_at = kwargs["session_started_at"]
        is_last_session = kwargs.get("is_last_session", False)
        turns = self._to_lightmem_turns(messages, session_started_at)

        if not turns:
            return

        system = self._ensure_lightmemory(mode="fresh")
        for turn_idx, turn_messages in enumerate(turns):
            is_last_turn = is_last_session and turn_idx == len(turns) - 1
            system.add_memory(
                messages=turn_messages,
                force_segment=is_last_turn,
                force_extract=is_last_turn,
            )

    def _retrieve_main_entries(self, query: str, limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        system = self._ensure_lightmemory(mode="load")
        query_vector = system.text_embedder.embed(query)
        return system.embedding_retriever.search(
            query_vector=query_vector,
            limit=limit,
            return_full=True,
        )

    def _retrieve_summary_entries(self, query: str, limit: int) -> list[dict[str, Any]]:
        if not self.config.enable_summary or limit <= 0:
            return []
        system = self._ensure_lightmemory(mode="load")
        if not hasattr(system, "summary_retriever"):
            return []
        if not _safe_collection_exists(
            getattr(system, "summary_retriever", None),
            self.config.summary_collection_name,
        ):
            return []
        query_vector = system.text_embedder.embed(query)
        return system.summary_retriever.search(
            query_vector=query_vector,
            limit=limit,
            return_full=True,
        )

    def retrieve(self, query: str, k: int = 10, **kwargs: Any) -> list[MemoryEntry]:
        summary_budget = min(self.config.summary_top_k, k) if self.config.enable_summary else 0
        summary_hits = self._retrieve_summary_entries(query, summary_budget)
        main_hits = self._retrieve_main_entries(query, max(k - len(summary_hits), 0))

        outputs: list[MemoryEntry] = []
        for result in summary_hits:
            payload = result.get("payload", {})
            outputs.append(
                MemoryEntry(
                    content=payload.get("summary", payload.get("memory", "")),
                    formatted_content=_format_lightmem_memory(payload, source="summary"),
                    metadata={
                        "entry_id": result.get("id"),
                        "score": result.get("score"),
                        "summary_id": result.get("id"),
                        "time_range": payload.get("time_range"),
                        "entry_count": payload.get("entry_count"),
                        "seed_count": payload.get("seed_count"),
                        "covered_entry_ids": payload.get("covered_entry_ids", []),
                        "seed_entry_ids": payload.get("seed_entry_ids", []),
                        "source": "summary",
                    },
                )
            )

        for result in main_hits:
            payload = result.get("payload", {})
            outputs.append(
                MemoryEntry(
                    content=payload.get("memory", ""),
                    formatted_content=_format_lightmem_memory(payload, source="memory"),
                    metadata={
                        "entry_id": result.get("id"),
                        "score": result.get("score"),
                        "time_stamp": payload.get("time_stamp"),
                        "weekday": payload.get("weekday"),
                        "speaker_id": payload.get("speaker_id"),
                        "speaker_name": payload.get("speaker_name"),
                        "topic_id": payload.get("topic_id"),
                        "topic_summary": payload.get("topic_summary"),
                        "category": payload.get("category"),
                        "subcategory": payload.get("subcategory"),
                        "memory_class": payload.get("memory_class"),
                        "original_memory": payload.get("original_memory"),
                        "compressed_memory": payload.get("compressed_memory"),
                        "source": "memory",
                    },
                )
            )

        return outputs[:k]

    def delete(self, memory_id: str) -> bool:
        raise NotImplementedError(
            "LightMem integration is append-only in MemBase v1."
        )

    def update(self, memory_id: str, **kwargs: Any) -> bool:
        raise NotImplementedError(
            "LightMem integration is append-only in MemBase v1."
        )

    def save_memory(self) -> None:
        os.makedirs(self.config.save_dir, exist_ok=True)
        config_path = os.path.join(self.config.save_dir, "config.json")
        config_dict = {
            "layer_type": self.layer_type,
            **self.config.model_dump(mode="python"),
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=4)

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

        config = LightMemConfig(**config_dict)
        self.config = config
        self.memory_layer = None
        self._manager_patch_target = None
        self._fresh_initialized = False

        if not os.path.isdir(config.qdrant_path):
            return False

        system = self._ensure_lightmemory(mode="load")
        if not _collection_exists(system.embedding_retriever, config.collection_name):
            self.memory_layer = None
            return False
        return True

    def flush(self) -> None:
        if not self.config.enable_summary:
            return
        system = self._ensure_lightmemory(mode="fresh")
        system.summarize(
            process_all=True,
            time_window=self.config.summary_time_window,
            enable_cross_event=self.config.summary_enable_cross_event,
            retrieval_scope=self.config.summary_retrieval_scope,
            top_k_seeds=self.config.summary_top_k_seeds,
        )

    def get_patch_specs(self) -> list[PatchSpec]:
        self._ensure_lightmemory(mode="fresh")
        if self._manager_patch_target is None:
            return []
        getter, setter = make_attr_patch(self._manager_patch_target, "generate_response")
        return [
            PatchSpec(
                name=f"{self._manager_patch_target.__class__.__name__}.generate_response",
                getter=getter,
                setter=setter,
                wrapper=token_monitor(
                    extract_model_name=lambda *args, **kwargs: (
                        self.config.llm_model,
                        {},
                    ),
                    extract_input_dict=lambda *args, **kwargs: {
                        "messages": kwargs.get("messages", args[0] if args else []),
                        "metadata": {"op_type": "generation"},
                    },
                    extract_output_dict=lambda response: {
                        "messages": response[0] if isinstance(response, tuple) else response,
                    },
                ),
            )
        ]

    def cleanup(self) -> None:
        self.memory_layer = None
        self._manager_patch_target = None
