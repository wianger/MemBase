import importlib
import json
import os
from typing import Any, ClassVar

from .base import MemBaseLayer
from ..baselines.simplemem.main import SimpleMemSystem
from ..baselines.simplemem.settings import SimpleMemSettings
from ..baselines.simplemem.utils.llm_client import LLMClient
from ..configs.simplemem import SimpleMemConfig
from ..model_types.dataset import Message
from ..model_types.memory import MemoryEntry
from ..utils import PatchSpec, make_attr_patch, token_monitor


def _format_simplemem_memory(entry: Any) -> str:
    parts = [entry.lossless_restatement]
    if entry.timestamp:
        parts.append(f"Time: {entry.timestamp}")
    if entry.location:
        parts.append(f"Location: {entry.location}")
    if entry.persons:
        parts.append(f"Persons: {', '.join(entry.persons)}")
    if entry.entities:
        parts.append(f"Entities: {', '.join(entry.entities)}")
    if entry.topic:
        parts.append(f"Topic: {entry.topic}")
    return "\n".join(parts)


class SimpleMemLayer(MemBaseLayer):
    layer_type: ClassVar[str] = "SimpleMem"

    def __init__(self, config: SimpleMemConfig) -> None:
        self.config = config
        self.settings = self._build_settings(config)
        self._llm_client = LLMClient(settings=self.settings)
        self.memory_layer: SimpleMemSystem | None = None
        self._fresh_initialized = False

    @staticmethod
    def _build_settings(config: SimpleMemConfig) -> SimpleMemSettings:
        return SimpleMemSettings(
            openai_api_key=config.api_key,
            openai_base_url=config.base_url,
            llm_model=config.llm_model,
            embedding_model=config.embedding_model,
            embedding_dimension=config.embedding_dimension,
            enable_thinking=config.enable_thinking,
            use_streaming=config.use_streaming,
            use_json_format=config.use_json_format,
            window_size=config.window_size,
            overlap_size=config.overlap_size,
            semantic_top_k=config.semantic_top_k,
            keyword_top_k=config.keyword_top_k,
            structured_top_k=config.structured_top_k,
            lancedb_path=config.db_path,
            memory_table_name=config.table_name,
            enable_parallel_processing=config.enable_parallel_processing,
            max_parallel_workers=config.max_parallel_workers,
            enable_parallel_retrieval=config.enable_parallel_retrieval,
            max_retrieval_workers=config.max_retrieval_workers,
            enable_planning=config.enable_planning,
            enable_reflection=config.enable_reflection,
            max_reflection_rounds=config.max_reflection_rounds,
        )

    @staticmethod
    def _connect_lancedb(path: str) -> Any:
        return importlib.import_module("lancedb").connect(path)

    def _ensure_system(self, mode: str) -> SimpleMemSystem:
        if self.memory_layer is not None:
            return self.memory_layer

        clear_db = False
        if mode == "fresh" and not self._fresh_initialized:
            clear_db = True
            self._fresh_initialized = True

        self.memory_layer = SimpleMemSystem(
            settings=self.settings,
            llm_client=self._llm_client,
            db_path=self.config.db_path,
            table_name=self.config.table_name,
            clear_db=clear_db,
            enable_planning=self.config.enable_planning,
            enable_reflection=self.config.enable_reflection,
            max_reflection_rounds=self.config.max_reflection_rounds,
            enable_parallel_processing=self.config.enable_parallel_processing,
            max_parallel_workers=self.config.max_parallel_workers,
            enable_parallel_retrieval=self.config.enable_parallel_retrieval,
            max_retrieval_workers=self.config.max_retrieval_workers,
        )
        return self.memory_layer

    def add_message(self, message: Message, **kwargs: Any) -> None:
        system = self._ensure_system(mode="fresh")
        system.add_dialogue(
            speaker=message.name,
            content=message.content,
            timestamp=message.timestamp,
            role=message.role,
        )

    def add_messages(self, messages: list[Message], **kwargs: Any) -> None:
        for message in messages:
            self.add_message(message, **kwargs)

    def retrieve(self, query: str, k: int = 10, **kwargs: Any) -> list[MemoryEntry]:
        system = self._ensure_system(mode="load")
        semantic_top_k = max(k, self.config.semantic_top_k)
        keyword_top_k = max(k, self.config.keyword_top_k)
        structured_top_k = max(k, self.config.structured_top_k)
        results = system.hybrid_retriever.retrieve(
            query=query,
            semantic_top_k=semantic_top_k,
            keyword_top_k=keyword_top_k,
            structured_top_k=structured_top_k,
        )
        outputs = []
        for entry in results[:k]:
            outputs.append(
                MemoryEntry(
                    content=entry.lossless_restatement,
                    formatted_content=_format_simplemem_memory(entry),
                    metadata={
                        "entry_id": entry.entry_id,
                        "keywords": entry.keywords,
                        "timestamp": entry.timestamp,
                        "location": entry.location,
                        "persons": entry.persons,
                        "entities": entry.entities,
                        "topic": entry.topic,
                    },
                )
            )
        return outputs

    def delete(self, memory_id: str) -> bool:
        raise NotImplementedError(
            "SimpleMem integration is append-only in MemBase v1."
        )

    def update(self, memory_id: str, **kwargs: Any) -> bool:
        raise NotImplementedError(
            "SimpleMem integration is append-only in MemBase v1."
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

        config = SimpleMemConfig(**config_dict)
        self.config = config
        self.settings = self._build_settings(config)
        self._llm_client = LLMClient(settings=self.settings)
        self.memory_layer = None
        self._fresh_initialized = False

        if not os.path.isdir(config.db_path):
            return False

        db = self._connect_lancedb(config.db_path)
        if config.table_name not in db.table_names():
            return False

        self._ensure_system(mode="load")
        return True

    def flush(self) -> None:
        system = self._ensure_system(mode="fresh")
        system.finalize()

    def get_patch_specs(self) -> list[PatchSpec]:
        getter, setter = make_attr_patch(self._llm_client, "chat_completion")
        return [
            PatchSpec(
                name=f"{self._llm_client.__class__.__name__}.chat_completion",
                getter=getter,
                setter=setter,
                wrapper=token_monitor(
                    extract_model_name=lambda *args, **kwargs: (
                        self.config.llm_model,
                        {},
                    ),
                    extract_input_dict=lambda *args, **kwargs: {
                        "messages": kwargs.get("messages", args[0] if args else []),
                        "metadata": {"op_type": "generation, update, retrieval-planning"},
                    },
                    extract_output_dict=lambda response: {
                        "messages": response,
                    },
                ),
            )
        ]

    def cleanup(self) -> None:
        self.memory_layer = None
