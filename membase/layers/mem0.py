import os
os.environ["MEM0_TELEMETRY"] = "False" # Disable telemetry.

import ast
import functools
import json
import re
from mem0 import Memory
from mem0.configs.base import MemoryConfig
from mem0.memory.storage import SQLiteManager
from mem0.utils.factory import (
    EmbedderFactory,
    GraphStoreFactory,
    LlmFactory,
    VectorStoreFactory,
    RerankerFactory,
)
from .base import MemBaseLayer
from ..utils import (
    PatchSpec,
    make_attr_patch,
    token_monitor,
)
from ..configs.mem0 import Mem0Config
from ..model_types.memory import MemoryEntry
from ..model_types.dataset import Message
from typing import Any, ClassVar


_MEM0_UPDATE_PROMPT_MARKER = "The new retrieved facts are mentioned in the triple backticks."
_GRAPH_SOURCE_KEYS = ("source", "subject", "from", "src", "head")
_GRAPH_RELATIONSHIP_KEYS = ("relationship", "relation", "predicate", "rel", "edge")
_GRAPH_DESTINATION_KEYS = ("destination", "object", "to", "dst", "target", "tail")


def _extract_mem0_messages_from_call(*args: Any, **kwargs: Any) -> list[dict[str, Any]] | None:
    """Extract the `messages` argument from a Mem0 llm.generate_response call."""
    messages = kwargs.get("messages", args[0] if len(args) > 0 else None)
    if isinstance(messages, list):
        return messages
    return None


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _safe_empty_update_payload_json() -> str:
    """Return a no-op update payload that Mem0 can parse reliably."""
    return json.dumps({"memory": []}, ensure_ascii=False)


def _extract_allowed_memory_ids_from_prompt(prompt: str) -> set[str]:
    """Extract allowed old-memory IDs from Mem0 update prompt code blocks."""
    id_to_text = _extract_old_memory_id_text_map_from_prompt(prompt)
    return set(id_to_text.keys())


def _extract_old_memory_id_text_map_from_prompt(prompt: str) -> dict[str, str]:
    """Extract old-memory id->text mapping from Mem0 update prompt code blocks."""
    code_blocks = re.findall(r"```(.*?)```", prompt, flags=re.S)
    for block in code_blocks:
        candidate = block.strip()
        if not candidate:
            continue
        try:
            parsed = ast.literal_eval(candidate)
        except Exception:
            continue
        if isinstance(parsed, list):
            id_to_text: dict[str, str] = {}
            for item in parsed:
                if not isinstance(item, dict) or "id" not in item:
                    continue
                memory_id = str(item["id"])
                memory_text = item.get("text", "")
                if not isinstance(memory_text, str):
                    memory_text = str(memory_text)
                id_to_text[memory_id] = memory_text
            if len(id_to_text) > 0:
                return id_to_text
    return {}


def _normalize_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return " ".join(text.strip().lower().split())


def _repair_action_id_by_old_memory(
    old_memory_to_id: dict[str, str],
    old_memory_text: str,
) -> str | None:
    """Try to recover target id from ``old_memory`` text via exact normalized match."""
    if not isinstance(old_memory_text, str) or old_memory_text.strip() == "":
        return None
    normalized_old = _normalize_text(old_memory_text)

    # Exact normalized match only (no fuzzy matching).
    if normalized_old in old_memory_to_id:
        return old_memory_to_id[normalized_old]
    return None


def _try_parse_mem0_update_payload(response: str) -> dict[str, Any] | None:
    """Best-effort parser for Mem0 update payloads.

    Some models prepend/append extra text, wrap JSON in fenced blocks, or return a
    leading valid JSON object followed by free-form text. This helper attempts a few
    safe parse strategies before giving up.
    """
    if not isinstance(response, str):
        return None

    candidates: list[str] = []
    cleaned = _strip_code_fences(response)
    if cleaned != "":
        candidates.append(cleaned)

    # Common case: assistant returns ```json ... ``` with surrounding prose.
    for block in re.findall(r"```(?:json)?\s*(.*?)```", response, flags=re.S | re.I):
        block = block.strip()
        if block != "":
            candidates.append(block)

    # Fallback: slice from the first "{" to the last "}".
    left = response.find("{")
    right = response.rfind("}")
    if left != -1 and right > left:
        candidates.append(response[left:right + 1].strip())

    seen: set[str] = set()
    decoder = json.JSONDecoder()
    for candidate in candidates:
        if candidate in seen or candidate == "":
            continue
        seen.add(candidate)

        parsed: Any | None = None
        try:
            parsed = json.loads(candidate)
        except Exception:
            # Try to parse a leading JSON object and ignore trailing non-JSON text.
            try:
                parsed, _ = decoder.raw_decode(candidate)
            except Exception:
                continue

        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            # Some models omit the top-level wrapper and return memory actions directly.
            return {"memory": parsed}
    return None


def _sanitize_mem0_update_actions(
    messages: list[dict[str, Any]] | None,
    response: str,
) -> str:
    """Sanitize invalid UPDATE/DELETE/NONE ids returned by LLM for Mem0 update calls.

    Mem0 maps existing UUIDs to temporary string IDs ("0", "1", ...). If the model
    returns an out-of-range ID (e.g., "1" when only "0" exists), Mem0 raises
    KeyError during action application. This sanitizer downgrades those invalid-id
    actions to ADD, which is id-agnostic in Mem0's implementation.
    """
    if not isinstance(response, str):
        return response
    if not isinstance(messages, list) or len(messages) == 0:
        return response
    if not isinstance(messages[0], dict):
        return response

    prompt = messages[0].get("content", "")
    if not isinstance(prompt, str) or _MEM0_UPDATE_PROMPT_MARKER not in prompt:
        return response

    cleaned = _strip_code_fences(response)
    if cleaned == "":
        print(
            "⚠️ Mem0 update-action sanitizer received empty/blank update payload. "
            "Fallback to no-op update actions."
        )
        return _safe_empty_update_payload_json()

    payload = _try_parse_mem0_update_payload(response)
    if payload is None:
        preview = cleaned.replace("\n", " ")[:240]
        print(
            "⚠️ Mem0 update-action sanitizer failed to parse JSON payload. "
            "Fallback to no-op update actions. "
            f"Payload preview: {preview}"
        )
        return _safe_empty_update_payload_json()

    memory_actions = payload.get("memory")
    if not isinstance(memory_actions, list):
        print(
            "⚠️ Mem0 update-action sanitizer found payload without a valid `memory` list. "
            "Fallback to no-op update actions."
        )
        return _safe_empty_update_payload_json()

    id_to_text = _extract_old_memory_id_text_map_from_prompt(prompt)
    allowed_ids = set(id_to_text.keys())
    old_memory_to_id = {
        _normalize_text(text): memory_id
        for memory_id, text in id_to_text.items()
        if isinstance(text, str) and text.strip() != ""
    }

    repaired = 0
    downgraded = 0
    for action in memory_actions:
        if not isinstance(action, dict):
            continue
        event = str(action.get("event", "")).upper()
        if event not in {"UPDATE", "DELETE", "NONE"}:
            continue
        action_id = str(action.get("id", ""))
        if action_id in allowed_ids:
            continue

        repaired_id = _repair_action_id_by_old_memory(
            old_memory_to_id=old_memory_to_id,
            old_memory_text=str(action.get("old_memory", "")),
        )
        if repaired_id is not None and repaired_id in allowed_ids:
            action["id"] = repaired_id
            repaired += 1
            continue

        # Unknown id and cannot recover: downgrade safely to avoid KeyError in mem0.
        if event == "UPDATE":
            # Preserve the newly extracted fact when update target is invalid.
            action["event"] = "ADD"
            action["id"] = "NEW"
            action.pop("old_memory", None)
        else:
            # For DELETE/NONE with unknown target, skip by turning into NONE.
            action["event"] = "NONE"
            action["id"] = "SKIP"
            action.pop("old_memory", None)
        downgraded += 1

    if repaired == 0 and downgraded == 0:
        # Even when no ID fix is needed, return canonical JSON to avoid
        # downstream parse failures from trailing prose or code fences.
        return json.dumps(payload, ensure_ascii=False)

    print(
        "⚠️ Mem0 update-action sanitizer handled invalid memory IDs: "
        f"repaired={repaired}, downgraded={downgraded}."
    )
    return json.dumps(payload, ensure_ascii=False)


def _pick_graph_field(item: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    """Pick the first non-empty value from candidate keys."""
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text != "":
            return text
    return None


def _sanitize_graph_entity_items(entity_list: Any) -> list[dict[str, str]]:
    """Normalize graph-entity items and drop malformed entries.

    Some OpenAI-compatible models can return partially valid tool arguments in graph
    extraction (e.g., missing ``relationship``). Upstream Mem0 assumes strict keys and
    raises ``KeyError``. This sanitizer normalizes common aliases and skips invalid rows.
    """
    if not isinstance(entity_list, list):
        return []

    sanitized: list[dict[str, str]] = []
    dropped = 0
    for item in entity_list:
        if not isinstance(item, dict):
            dropped += 1
            continue

        source = _pick_graph_field(item, _GRAPH_SOURCE_KEYS)
        relationship = _pick_graph_field(item, _GRAPH_RELATIONSHIP_KEYS)
        destination = _pick_graph_field(item, _GRAPH_DESTINATION_KEYS)

        if source is None or relationship is None or destination is None:
            dropped += 1
            continue

        sanitized.append(
            {
                "source": source.lower().replace(" ", "_"),
                "relationship": relationship.lower().replace(" ", "_"),
                "destination": destination.lower().replace(" ", "_"),
            }
        )

    if dropped > 0:
        print(
            "⚠️ Mem0 graph-entity sanitizer dropped malformed relation items: "
            f"dropped={dropped}, kept={len(sanitized)}."
        )
    return sanitized


def _patch_mem0_graph_entity_sanitizer(graph: Any) -> None:
    """Patch Mem0 graph helper to tolerate malformed relationship items."""
    if graph is None or not hasattr(graph, "_remove_spaces_from_entities"):
        return

    def _safe_remove_spaces_from_entities(entity_list: Any) -> list[dict[str, str]]:
        return _sanitize_graph_entity_items(entity_list)

    graph._remove_spaces_from_entities = _safe_remove_spaces_from_entities


class Mem0Memory(Memory):
    """A thin subclass of ``mem0.Memory`` that skips telemetry initialization.

    The upstream ``Memory.__init__`` creates an additional Qdrant collection
    (``mem0migrations``) solely for anonymous usage telemetry via PostHog.
    This is unnecessary for evaluation and introduces extra I/O and potential
    lock contention when running multiple instances in parallel."""

    def __init__(self, config: MemoryConfig = MemoryConfig()) -> None:
        self.config = config

        self.custom_fact_extraction_prompt = self.config.custom_fact_extraction_prompt
        self.custom_update_memory_prompt = self.config.custom_update_memory_prompt
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, 
            self.config.vector_store.config,
        )
        self.llm = LlmFactory.create(
            self.config.llm.provider, 
            self.config.llm.config,
        )
        self.db = SQLiteManager(self.config.history_db_path)
        self.collection_name = self.config.vector_store.config.collection_name
        self.api_version = self.config.version

        # Initialize reranker if configured.
        self.reranker = None
        if config.reranker:
            self.reranker = RerankerFactory.create(
                config.reranker.provider,
                config.reranker.config,
            )

        self.enable_graph = False

        if self.config.graph_store.config:
            provider = self.config.graph_store.provider
            self.graph = GraphStoreFactory.create(provider, self.config)
            self.enable_graph = True
        else:
            self.graph = None

        # Telemetry is intentionally skipped. Set the attribute to `None`. 
        self._telemetry_vector_store = None


class Mem0Layer(MemBaseLayer):

    layer_type: ClassVar[str] = "Mem0"

    def __init__(self, config: Mem0Config) -> None:
        """Create an interface of Mem0. The implementation is based on the 
        third-party library `mem0ai`."""
        self._init_layer(config)
        self.config = config

    def _init_layer(self, config: Mem0Config) -> None:
        """Initialize the Mem0 layer.

        Mem0 natively manages persistence via its Qdrant backend (``on_disk=True``), 
        so no additional serialization is required.
        
        Args:
            config (`Mem0Config`): 
                The configuration for the Mem0 layer.
        """
        mem0_config = config.build_mem0_config()
        self.memory_layer = Mem0Memory.from_config(mem0_config)
        _patch_mem0_graph_entity_sanitizer(getattr(self.memory_layer, "graph", None))

    def add_message(self, message: Message, **kwargs: Any) -> None:
        # Note that Mem0 does't use name field directly. 
        # Therefore, we incorporate the name information into the message content 
        # to retain speaker identity.
        text = (
            f"{message.content}\nBelow is this message's metadata:\n"
            f"Speaker Name: {message.name}\n"
            f"Speaker Role: {message.role}\n"
        )

        # Following Mem0's implementation (https://github.com/mem0ai/mem0/blob/main/evaluation/src/memzero/add.py#L83).
        try:
            self.memory_layer.add(
                messages={
                    "content": text,
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
        except Exception as e:
            print(f"Error in add_message method in Mem0Layer: \n\t{e.__class__.__name__}: {e}")

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
        else:
            new_messages = [] 
            for message in messages:
                msg_dict = message.model_dump(mode="python")
                msg_dict["content"] = (
                    f"{message.content}\nBelow is this message's metadata:\n"
                    f"Speaker Name: {message.name}\n"
                    f"Speaker Role: {message.role}\n"
                )
                new_messages.append(msg_dict)
            
            self.memory_layer.add(
                messages=new_messages,
                user_id=self.config.user_id,
                metadata={
                    "timestamp": f"[{messages[0].timestamp}, {messages[-1].timestamp}]",
                    "speakers": ", ".join(
                        sorted(
                            set(
                                [message.name for message in messages]
                            )
                        )
                    ),
                },
                **kwargs, 
            )

    def retrieve(self, query: str, k: int = 10, **kwargs: Any) -> list[MemoryEntry]:
        result = self.memory_layer.search(
            query=query,
            user_id=self.config.user_id,
            limit=k,
            **kwargs,
        )

        memories = result["results"]
        relations = result.get("relations")

        graph_text = ""
        if relations:
            graph_text = "\n".join(
                ["### Graph Relations:"] + [str(rel) for rel in relations]
            )

        outputs = []
        for item in memories:
            content = item["memory"]
            metadata = {k: v for k, v in item.items() if k != "memory"}
            nested_metadata = metadata.get("metadata", {})

            parts = [f"Memory: {content}"]
            if nested_metadata.get("timestamp"):
                parts.append(f"Time: {nested_metadata['timestamp']}")
            if graph_text:
                parts.append(graph_text)
            formatted = "\n".join(parts)

            outputs.append(
                MemoryEntry(
                    content=content,
                    metadata=metadata,
                    formatted_content=formatted,
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
        try:
            self.memory_layer.update(memory_id, data)
            return True
        except Exception as e:
            print(f"Error in update method in Mem0Layer: \n\t{e.__class__.__name__}: {e}")
            return False

    def save_memory(self) -> None:
        os.makedirs(self.config.save_dir, exist_ok=True)

        # Write config.json.
        config_path = os.path.join(self.config.save_dir, "config.json")
        config_dict = {
            "layer_type": self.layer_type,
            **self.config.model_dump(mode="python"),
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)

        # The Qdrant vector store (on_disk=True) and the SQLite history DB persist
        # automatically, so no additional serialization is needed here.

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

        config = Mem0Config(**config_dict)

        # Release the existing Qdrant client's file lock before re-initialization.
        self.cleanup()

        self._init_layer(config)
        self.config = config

        # Verify that the Qdrant store actually contains data for this user.
        try:
            existing = self.memory_layer.get_all(user_id=user_id, limit=1)
            memories = existing["results"]
            return len(memories) > 0
        except Exception:
            return False

    def get_patch_specs(self) -> list[PatchSpec]:
        # # See https://github.com/mem0ai/mem0/blob/v1.0.5/mem0/configs/prompts.py#L62.
        # kwargs.get(
        #     "messages", 
        #     args[0] if len(args) > 0 else ""
        # )[0]["content"].startswith(
        #     "You are a Personal Information Organizer"
        # ) 
        # # See https://github.com/mem0ai/mem0/blob/v1.0.5/mem0/configs/prompts.py#L123. 
        # or kwargs.get(
        #     "messages", 
        #     args[0] if len(args) > 0 else ""
        # )[0]["content"].startswith(
        #     "You are an Assistant Information Organizer"
        # )
        # # See https://github.com/mem0ai/mem0/blob/v1.0.5/mem0/memory/main.py#L426. 
        # or kwargs.get(
        #     "messages", 
        #     args[0] if len(args) > 0 else ""
        # )[0]["content"].startswith(
        #     self.config.custom_fact_extraction_prompt
        # ) 
        # # See https://github.com/mem0ai/mem0/blob/v1.0.5/mem0/memory/kuzu_memory.py#L231. 
        # or kwargs.get(
        #     "messages", 
        #     args[0] if len(args) > 0 else ""
        # )[0]["content"].startswith(
        #     "You are a smart assistant who understands entities and their types in a given text"
        # ) 
        # # See https://github.com/mem0ai/mem0/blob/v1.0.5/mem0/memory/kuzu_memory.py#L266. 
        # or (
        #     "You are an advanced algorithm designed to extract structured information " 
        #     "from text to construct knowledge graphs"
        # ) in kwargs.get(
        #     "messages", 
        #     args[0] if len(args) > 0 else ""
        # )[0]["content"]
        getter, setter = make_attr_patch(self.memory_layer.llm, "generate_response")
        monitor_wrapper = token_monitor(
            extract_model_name=lambda *args, **kwargs: (
                self.config.llm_model, 
                {
                    # See https://github.com/mem0ai/mem0/blob/v1.0.5/mem0/memory/kuzu_memory.py#L235.
                    # The graph version of Mem0 uses tools to extract entities and their relations. 
                    "tools": kwargs.get("tools"),
                }
            ),
            # The update-memory prompt is easier to identify than the fact-extraction prompt.
            extract_input_dict=lambda *args, **kwargs: {
                "messages": kwargs.get("messages", args[0] if len(args) > 0 else ""),
                "metadata": {
                    "op_type": (
                        "update" if (
                                "The new retrieved facts are mentioned in the triple backticks. " 
                                "You have to analyze the new retrieved facts and determine whether " 
                                "these facts should be added, updated, or deleted in the memory."
                            ) in kwargs.get(
                                "messages", 
                                args[0] if len(args) > 0 else [{"content": ""}]
                            )[0]["content"]
                        else "generation"
                    )
                },
            },
            # The result may be a plain string or an OpenAI-compatible message dictionary
            # (e.g., {"role": "assistant", "content": "..."}).
            extract_output_dict=lambda result: {
                "messages": result if isinstance(result, str) else [
                    {
                        "role": "assistant",
                        **result,
                    }
                ],
            },
        )

        def _mem0_generate_response_wrapper(func):
            monitored_func = monitor_wrapper(func)

            @functools.wraps(monitored_func)
            def wrapped(*args, **kwargs):
                result = monitored_func(*args, **kwargs)
                messages = _extract_mem0_messages_from_call(*args, **kwargs)
                return _sanitize_mem0_update_actions(messages, result)

            return wrapped

        spec = PatchSpec(
            name=f"{self.memory_layer.llm.__class__.__name__}.generate_response",
            getter=getter,
            setter=setter,
            wrapper=_mem0_generate_response_wrapper,
        )
        return [spec]

    def cleanup(self) -> None:
        """Release the Qdrant local client's exclusive file lock."""
        client = self.memory_layer.vector_store.client
        client.close()
