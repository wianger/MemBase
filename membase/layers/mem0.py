import json
import os
import warnings
from importlib.util import find_spec
from typing import Any, ClassVar

os.environ["MEM0_TELEMETRY"] = "False"

from mem0 import Memory

from .base import MemBaseLayer
from ..configs.mem0 import Mem0Config
from ..model_types.dataset import Message
from ..model_types.memory import MemoryEntry
from ..utils import PatchSpec, make_attr_patch, token_monitor


class Mem0Layer(MemBaseLayer):
    layer_type: ClassVar[str] = "Mem0"

    def __init__(self, config: Mem0Config) -> None:
        self._init_layer(config)
        self.config = config

    def _init_layer(self, config: Mem0Config) -> None:
        mem0_config = config.build_mem0_config()
        self.memory_layer = Memory.from_config(mem0_config)
        self._warn_hybrid_retrieval_status()

    @staticmethod
    def _module_available(module_name: str) -> bool:
        try:
            return find_spec(module_name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    @classmethod
    def _spacy_model_available(cls) -> bool:
        if not cls._module_available("spacy"):
            return False

        try:
            import spacy

            return bool(spacy.util.is_package("en_core_web_sm"))
        except Exception:
            return cls._module_available("en_core_web_sm")

    def _detect_hybrid_retrieval_status(self) -> tuple[str, list[str], list[str]]:
        available_components = ["semantic vector search"]
        degraded_reasons = []

        vector_store = getattr(self.memory_layer, "vector_store", None)
        has_keyword_search = hasattr(vector_store, "keyword_search")
        has_bm25_slot = bool(getattr(vector_store, "_has_bm25_slot", False))
        fastembed_installed = self._module_available("fastembed")
        spacy_installed = self._module_available("spacy")
        spacy_model_installed = self._spacy_model_available()

        bm25_ready = False
        if has_keyword_search and has_bm25_slot:
            try:
                bm25_ready = vector_store._get_bm25_encoder() is not None
            except Exception:
                bm25_ready = False

        if bm25_ready:
            available_components.append("BM25 keyword search")
        else:
            if not has_keyword_search:
                degraded_reasons.append("vector store does not expose BM25 keyword search")
            elif not has_bm25_slot:
                degraded_reasons.append(
                    "current Qdrant collection has no 'bm25' sparse slot; use a fresh rebuild to enable BM25"
                )
            elif not fastembed_installed:
                degraded_reasons.append("fastembed is not installed")
            else:
                degraded_reasons.append("BM25 encoder is unavailable")

        if spacy_installed and spacy_model_installed:
            available_components.append("entity boost")
        else:
            if not spacy_installed:
                degraded_reasons.append("spaCy is not installed")
            elif not spacy_model_installed:
                degraded_reasons.append("spaCy model 'en_core_web_sm' is not installed")
            else:
                degraded_reasons.append("entity extraction runtime is unavailable")

        status = "full" if len(available_components) == 3 else "degraded"
        return status, available_components, degraded_reasons

    def _warn_hybrid_retrieval_status(self) -> None:
        status, available_components, degraded_reasons = self._detect_hybrid_retrieval_status()
        self.hybrid_retrieval_status = status
        self.hybrid_retrieval_components = available_components

        message = (
            f"Mem0 hybrid retrieval status: {status} "
            f"(available: {', '.join(available_components)})"
        )
        if degraded_reasons:
            message += f". Missing or disabled pieces: {'; '.join(degraded_reasons)}."
        else:
            message += "."

        warnings.warn(message, RuntimeWarning, stacklevel=2)

    @staticmethod
    def _format_message_content(message: Message) -> str:
        return (
            f"{message.content}\nBelow is this message's metadata:\n"
            f"Speaker Name: {message.name}\n"
            f"Speaker Role: {message.role}\n"
        )

    def add_message(self, message: Message, **kwargs: Any) -> None:
        self.memory_layer.add(
            messages={
                "content": self._format_message_content(message),
                "role": message.role,
                "name": message.name,
            },
            user_id=self.config.user_id,
            metadata={
                "timestamp": message.timestamp,
                "speakers": message.name,
            },
            **kwargs,
        )

    def add_messages(self, messages: list[Message], **kwargs: Any) -> None:
        message_level = kwargs.pop("message_level", True)
        if message_level not in [True, False]:
            raise TypeError(
                "`message_level` must be a boolean to indicate whether the messages "
                "are added to the memory layer message by message or as a whole."
            )

        if message_level or len(messages) < 2:
            for message in messages:
                self.add_message(message, **kwargs)
            return

        new_messages = []
        for message in messages:
            new_messages.append(
                {
                    "id": message.id,
                    "content": self._format_message_content(message),
                    "role": message.role,
                    "name": message.name,
                    "timestamp": message.timestamp,
                    "metadata": message.metadata,
                }
            )

        self.memory_layer.add(
            messages=new_messages,
            user_id=self.config.user_id,
            metadata={
                "timestamp": f"[{messages[0].timestamp}, {messages[-1].timestamp}]",
                "speakers": ", ".join(sorted({message.name for message in messages})),
            },
            **kwargs,
        )

    def retrieve(self, query: str, k: int = 10, **kwargs: Any) -> list[MemoryEntry]:
        search_kwargs = dict(kwargs)
        filters = dict(search_kwargs.pop("filters", {}) or {})
        filters.setdefault("user_id", self.config.user_id)

        result = self.memory_layer.search(
            query=query,
            top_k=k,
            filters=filters,
            **search_kwargs,
        )

        outputs = []
        for item in result["results"]:
            content = item["memory"]
            metadata = {key: value for key, value in item.items() if key != "memory"}
            nested_metadata = metadata.get("metadata", {})

            parts = [f"Memory: {content}"]
            if nested_metadata.get("timestamp"):
                parts.append(f"Time: {nested_metadata['timestamp']}")

            outputs.append(
                MemoryEntry(
                    content=content,
                    metadata=metadata,
                    formatted_content="\n".join(parts),
                )
            )

        return outputs

    def delete(self, memory_id: str) -> bool:
        try:
            self.memory_layer.delete(memory_id)
            return True
        except Exception as e:
            print(f"Error in delete method in Mem0Layer: \n\t{e.__class__.__name__}: {e}")
            return False

    def update(self, memory_id: str, **kwargs: Any) -> bool:
        if "data" not in kwargs:
            raise KeyError("`data` is required in `kwargs` for Mem0 layer.")
        data = kwargs.pop("data")
        metadata = kwargs.pop("metadata", None)
        try:
            self.memory_layer.update(memory_id, data, metadata=metadata)
            return True
        except Exception as e:
            print(f"Error in update method in Mem0Layer: \n\t{e.__class__.__name__}: {e}")
            return False

    def save_memory(self) -> None:
        os.makedirs(self.config.save_dir, exist_ok=True)
        config_path = os.path.join(self.config.save_dir, "config.json")
        config_dict = {
            "layer_type": self.layer_type,
            **self.config.model_dump(mode="python"),
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)

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

        persisted_layer_type = config_dict.pop("layer_type", None)
        if persisted_layer_type is not None and persisted_layer_type != self.layer_type:
            raise ValueError(
                f"The layer type in the config file ({persisted_layer_type}) "
                f"does not match the current memory layer ({self.layer_type})."
            )

        config = Mem0Config(**config_dict)

        self.cleanup()
        self._init_layer(config)
        self.config = config

        try:
            existing = self.memory_layer.get_all(
                filters={"user_id": user_id},
                top_k=1,
            )
            memories = existing["results"]
            return len(memories) > 0
        except Exception:
            return False

    def get_patch_specs(self) -> list[PatchSpec]:
        getter, setter = make_attr_patch(self.memory_layer.llm, "generate_response")
        spec = PatchSpec(
            name=f"{self.memory_layer.llm.__class__.__name__}.generate_response",
            getter=getter,
            setter=setter,
            wrapper=token_monitor(
                extract_model_name=lambda *args, **kwargs: (
                    self.config.llm_model,
                    {},
                ),
                extract_input_dict=lambda *args, **kwargs: {
                    "messages": kwargs.get("messages", args[0] if len(args) > 0 else ""),
                    "metadata": {
                        "op_type": "generation",
                    },
                },
                extract_output_dict=lambda result: {
                    "messages": result if isinstance(result, str) else [
                        {
                            "role": "assistant",
                            **result,
                        }
                    ],
                },
            ),
        )
        return [spec]

    def cleanup(self) -> None:
        client = getattr(self.memory_layer.vector_store, "client", None)
        if client is not None and hasattr(client, "close"):
            client.close()
