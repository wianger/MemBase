import functools
import json
import os
from typing import Any, ClassVar
import httpx
from openai import OpenAI

from lightmem.memory.lightmem import LightMemory
from lightmem.memory.prompts import (
    LoCoMo_Event_Binding_factual,
    LoCoMo_Event_Binding_relational,
    METADATA_GENERATE_PROMPT_locomo,
)

from .base import MemBaseLayer
from ..configs.lightmem import LightMemConfig
from ..model_types.dataset import Message
from ..model_types.memory import MemoryEntry
from ..utils import (
    MonkeyPatcher,
    PatchSpec,
    make_attr_patch,
    token_monitor,
)
from lightmem.factory.text_embedder.factory import TextEmbedderFactory as UpstreamTextEmbedderFactory


class _RemoteLightMemEmbedder:
    """OpenAI-compatible embedder adapter for upstream LightMem."""

    def __init__(self, config: LightMemConfig) -> None:
        self.config = config
        self.client = OpenAI(
            api_key=config.embedding_api_key or config.api_key,
            base_url=config.embedding_base_url,
            http_client=httpx.Client(trust_env=False),
        )
        self.total_calls = 0
        self.total_tokens = 0

    def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        if isinstance(text, list) and len(text) == 0:
            return []
        resp = self.client.embeddings.create(
            model=self.config.embedding_model,
            input=text,
        )
        self.total_calls += 1
        self.total_tokens += getattr(resp.usage, "total_tokens", 0)
        if isinstance(text, list):
            return [item.embedding for item in resp.data]
        return resp.data[0].embedding

    def get_stats(self) -> dict[str, int]:
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
        }


def _as_message_records(message: Message) -> list[dict[str, Any]]:
    """Convert MemBase Message into one LightMem turn (user + empty assistant)."""
    base = {
        "speaker_id": message.name,
        "speaker_name": message.name,
        "time_stamp": message.timestamp,
    }
    return [
        {
            **base,
            "role": "user",
            "content": message.content,
        },
        {
            **base,
            "role": "assistant",
            "content": "",
        },
    ]


def _detect_lightmem_op_type(messages: list[dict[str, Any]]) -> str:
    if len(messages) == 0:
        return "generation"
    first = messages[0]
    system_text = str(first.get("content", ""))
    if "Personal Information Extractor" in system_text:
        return "generation"
    if "Relational Memory Extractor" in system_text:
        return "generation_relational"
    if "Target memory:" in str(messages[-1].get("content", "")):
        return "update"
    if "summar" in system_text.lower():
        return "summarize"
    return "generation"


def _build_lightmem_config(config: LightMemConfig) -> dict[str, Any]:
    model_kwargs = {
        "device": config.embedding_device,
        **(config.embedding_model_kwargs or {}),
    }

    llmlingua_cfg: dict[str, Any] = {
        "model_name": config.llmlingua_model_path,
        "device_map": config.llmlingua_device_map,
        "use_llmlingua2": config.llmlingua_use_v2,
    }

    cfg = {
        "pre_compress": config.pre_compress,
        "pre_compressor": {
            "model_name": "llmlingua-2",
            "configs": {
                "llmlingua_config": llmlingua_cfg,
                "compress_config": {
                    "instruction": "",
                    "rate": config.llmlingua_compress_rate,
                    "target_token": -1,
                },
            },
        },
        "topic_segment": config.topic_segment,
        "precomp_topic_shared": config.precomp_topic_shared,
        "topic_segmenter": {
            "model_name": config.topic_segmenter_model_name,
        },
        "messages_use": config.messages_use,
        "metadata_generate": config.metadata_generate,
        "text_summary": config.text_summary,
        "memory_manager": {
            "model_name": config.memory_manager_name,
            "configs": {
                "model": config.llm_model,
                "api_key": config.api_key,
                "max_tokens": config.llm_max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "openai_base_url": config.llm_base_url,
                "deepseek_base_url": config.llm_base_url,
                "vllm_base_url": config.llm_base_url,
            },
        },
        "extract_threshold": config.extract_threshold,
        "index_strategy": config.index_strategy,
        "text_embedder": {
            "model_name": "huggingface",
            "configs": {
                "model": config.embedding_model,
                "embedding_dims": config.embedding_dimension,
                "model_kwargs": model_kwargs,
            },
        },
        "retrieve_strategy": config.retrieve_strategy,
        "embedding_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": config.qdrant_collection_name,
                "embedding_model_dims": config.embedding_dimension,
                "path": config.save_dir,
                "on_disk": config.qdrant_on_disk,
            },
        },
        "update": config.update_mode,
        "extraction_mode": config.extraction_mode,
        # Avoid noisy files in benchmark output unless users explicitly enable logs.
        "logging": {
            "level": "WARNING",
            "console_enabled": False,
            "file_enabled": False,
        },
    }
    return cfg


class LightMemLayer(MemBaseLayer):
    layer_type: ClassVar[str] = "LightMem"

    def __init__(self, config: LightMemConfig) -> None:
        self.config = config
        self.embedding_model_name = config.embedding_model
        self._last_turn_messages: list[dict[str, Any]] | None = None
        self._init_layer(config)

    def _init_layer(self, config: LightMemConfig) -> None:
        os.makedirs(config.save_dir, exist_ok=True)
        if config.uses_remote_embedding():
            getter, setter = make_attr_patch(UpstreamTextEmbedderFactory, "from_config")

            def _remote_text_embedder_wrapper(_original: Any):
                def _factory(*args: Any, **kwargs: Any) -> _RemoteLightMemEmbedder:
                    del args, kwargs
                    return _RemoteLightMemEmbedder(config)

                return _factory

            spec = PatchSpec(
                name="TextEmbedderFactory.from_config",
                getter=getter,
                setter=setter,
                wrapper=_remote_text_embedder_wrapper,
            )
            with MonkeyPatcher([spec]):
                self.memory_layer = LightMemory.from_config(_build_lightmem_config(config))
        else:
            self.memory_layer = LightMemory.from_config(_build_lightmem_config(config))

    @staticmethod
    def _config_signature(config: LightMemConfig) -> dict[str, Any]:
        """Build a stable config signature for reload decisions."""
        return config.model_dump(mode="python")

    def add_message(self, message: Message, **kwargs: Any) -> None:
        turn_messages = _as_message_records(message)
        self._last_turn_messages = turn_messages

        prompt_arg: str | dict[str, str]
        if self.config.extraction_mode == "event":
            prompt_arg = {
                "factual": LoCoMo_Event_Binding_factual,
                "relational": LoCoMo_Event_Binding_relational,
            }
        else:
            prompt_arg = METADATA_GENERATE_PROMPT_locomo

        try:
            self.memory_layer.add_memory(
                messages=turn_messages,
                METADATA_GENERATE_PROMPT=prompt_arg,
                force_segment=False,
                force_extract=False,
            )
        except Exception as e:
            print(
                "Error in add_message method in LightMemLayer: "
                f"\n\t{e.__class__.__name__}: {e}"
            )

    def add_messages(self, messages: list[Message], **kwargs: Any) -> None:
        for message in messages:
            self.add_message(message, **kwargs)

    def flush(self) -> None:
        if self._last_turn_messages is not None:
            try:
                prompt_arg: str | dict[str, str]
                if self.config.extraction_mode == "event":
                    prompt_arg = {
                        "factual": LoCoMo_Event_Binding_factual,
                        "relational": LoCoMo_Event_Binding_relational,
                    }
                else:
                    prompt_arg = METADATA_GENERATE_PROMPT_locomo
                self.memory_layer.add_memory(
                    messages=self._last_turn_messages,
                    METADATA_GENERATE_PROMPT=prompt_arg,
                    force_segment=True,
                    force_extract=True,
                )
            except Exception as e:
                print(
                    "Error while forcing final segmentation/extraction in LightMemLayer.flush: "
                    f"\n\t{e.__class__.__name__}: {e}"
                )

        if self.config.enable_offline_update:
            try:
                self.memory_layer.construct_update_queue_all_entries(
                    top_k=self.config.construct_queue_top_k,
                    keep_top_n=self.config.construct_queue_keep_top_n,
                    max_workers=self.config.construct_queue_workers,
                )
                self.memory_layer.offline_update_all_entries(
                    score_threshold=self.config.offline_update_score_threshold,
                    max_workers=self.config.offline_update_workers,
                )
            except Exception as e:
                print(
                    "Error in LightMem offline update flow during flush: "
                    f"\n\t{e.__class__.__name__}: {e}"
                )

    def retrieve(self, query: str, k: int = 10, **kwargs: Any) -> list[MemoryEntry]:
        limit = k if k is not None else self.config.retrieve_limit_default
        try:
            raw_result = self.memory_layer.retrieve(query=query, limit=limit)
        except Exception as e:
            print(
                "Error in retrieve method in LightMemLayer: "
                f"\n\t{e.__class__.__name__}: {e}"
            )
            return []

        if raw_result is None:
            return []

        if isinstance(raw_result, str):
            lines = [line.strip() for line in raw_result.split("\n") if line.strip() != ""]
        elif isinstance(raw_result, list):
            lines = [str(item).strip() for item in raw_result if str(item).strip() != ""]
        else:
            text = str(raw_result).strip()
            lines = [text] if text != "" else []

        return [
            MemoryEntry(
                content=line,
                formatted_content=line,
                metadata={
                    "provider": "LightMem",
                    "source": "embedding_retriever",
                },
            )
            for line in lines[:limit]
        ]

    def delete(self, memory_id: str) -> bool:
        print(
            "LightMemLayer.delete is not supported by upstream LightMem API. "
            f"Requested memory_id={memory_id}."
        )
        return False

    def update(self, memory_id: str, **kwargs: Any) -> bool:
        print(
            "LightMemLayer.update is not supported by upstream LightMem API. "
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

        config = LightMemConfig(**config_dict)
        current_layer = getattr(self, "memory_layer", None)
        current_config = getattr(self, "config", None)
        should_reinit = (
            current_layer is None
            or current_config is None
            or self._config_signature(current_config) != self._config_signature(config)
        )

        # Keep runtime view aligned with persisted config.
        self.config = config
        self.embedding_model_name = config.embedding_model

        # Avoid opening the same local Qdrant path twice in one process.
        if should_reinit:
            self.cleanup()
            self._init_layer(config)

        try:
            entries = self.memory_layer.embedding_retriever.get_all(
                with_vectors=False,
                with_payload=True,
            )
            return len(entries) > 0
        except Exception:
            return False

    def cleanup(self) -> None:
        memory_layer = getattr(self, "memory_layer", None)
        if memory_layer is None:
            return

        retriever = getattr(memory_layer, "embedding_retriever", None)
        client = getattr(retriever, "client", None)
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def get_patch_specs(self) -> list[PatchSpec]:
        manager = getattr(self.memory_layer, "manager", None)
        if manager is None or not hasattr(manager, "generate_response"):
            return []

        getter, setter = make_attr_patch(manager, "generate_response")

        monitor_wrapper = token_monitor(
            extract_model_name=lambda *args, **kwargs: (self.config.llm_model, {}),
            extract_input_dict=lambda *args, **kwargs: {
                "messages": kwargs.get("messages", args[0] if len(args) > 0 else []),
                "metadata": {
                    "op_type": _detect_lightmem_op_type(
                        kwargs.get("messages", args[0] if len(args) > 0 else [])
                        if isinstance(kwargs.get("messages", args[0] if len(args) > 0 else []), list)
                        else []
                    )
                },
            },
            extract_output_dict=lambda result: {
                "messages": (
                    result[0]
                    if isinstance(result, tuple) and len(result) > 0
                    else result
                ),
            },
        )

        def _lightmem_generate_response_wrapper(func):
            monitored_func = monitor_wrapper(func)

            @functools.wraps(monitored_func)
            def wrapped(*args, **kwargs):
                return monitored_func(*args, **kwargs)

            return wrapped

        return [
            PatchSpec(
                name=f"{manager.__class__.__name__}.generate_response",
                getter=getter,
                setter=setter,
                wrapper=_lightmem_generate_response_wrapper,
            )
        ]
