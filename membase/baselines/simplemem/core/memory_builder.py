from __future__ import annotations

import concurrent.futures

from ..database.vector_store import VectorStore
from ..models.memory_entry import Dialogue, MemoryEntry
from ..settings import SimpleMemSettings
from ..utils.llm_client import LLMClient


class MemoryBuilder:
    """SimpleMem memory builder for online construction."""

    def __init__(
        self,
        settings: SimpleMemSettings,
        llm_client: LLMClient,
        vector_store: VectorStore,
        window_size: int | None = None,
        enable_parallel_processing: bool | None = None,
        max_parallel_workers: int | None = None,
    ) -> None:
        self.settings = settings
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.window_size = window_size or settings.window_size
        self.overlap_size = settings.overlap_size
        self.step_size = max(1, self.window_size - self.overlap_size)
        self.enable_parallel_processing = (
            settings.enable_parallel_processing
            if enable_parallel_processing is None
            else enable_parallel_processing
        )
        self.max_parallel_workers = (
            settings.max_parallel_workers
            if max_parallel_workers is None
            else max_parallel_workers
        )
        self.dialogue_buffer: list[Dialogue] = []
        self.processed_count = 0
        self.previous_entries: list[MemoryEntry] = []

    def add_dialogue(self, dialogue: Dialogue, auto_process: bool = True) -> None:
        self.dialogue_buffer.append(dialogue)
        if auto_process and len(self.dialogue_buffer) >= self.window_size:
            self.process_window()

    def add_dialogues(
        self,
        dialogues: list[Dialogue],
        auto_process: bool = True,
    ) -> None:
        if self.enable_parallel_processing and len(dialogues) > self.window_size * 2:
            self.add_dialogues_parallel(dialogues)
            return
        for dialogue in dialogues:
            self.add_dialogue(dialogue, auto_process=False)
        if auto_process:
            while len(self.dialogue_buffer) >= self.window_size:
                self.process_window()

    def add_dialogues_parallel(self, dialogues: list[Dialogue]) -> None:
        pre_existing = list(self.dialogue_buffer)
        windows_to_process: list[list[Dialogue]] = []
        try:
            self.dialogue_buffer.extend(dialogues)
            pos = 0
            while pos + self.window_size <= len(self.dialogue_buffer):
                windows_to_process.append(
                    self.dialogue_buffer[pos : pos + self.window_size]
                )
                pos += self.step_size
            remaining = self.dialogue_buffer[pos:]
            if remaining:
                windows_to_process.append(remaining)
            self.dialogue_buffer = []
            if windows_to_process:
                self._process_windows_parallel(windows_to_process)
        except Exception:
            if not self.dialogue_buffer:
                self.dialogue_buffer = pre_existing + list(dialogues)
            while len(self.dialogue_buffer) >= self.window_size:
                self.process_window()

    def process_window(self) -> None:
        if not self.dialogue_buffer:
            return
        window = self.dialogue_buffer[: self.window_size]
        self.dialogue_buffer = self.dialogue_buffer[self.step_size :]
        entries = self._generate_memory_entries(window)
        if entries:
            self.vector_store.add_entries(entries)
            self.previous_entries = entries
            self.processed_count += len(window)

    def process_remaining(self) -> None:
        if not self.dialogue_buffer:
            return
        entries = self._generate_memory_entries(self.dialogue_buffer)
        if entries:
            self.vector_store.add_entries(entries)
            self.processed_count += len(self.dialogue_buffer)
        self.dialogue_buffer = []

    def _generate_memory_entries(self, dialogues: list[Dialogue]) -> list[MemoryEntry]:
        dialogue_text = "\n".join([str(d) for d in dialogues])
        context = ""
        if self.previous_entries:
            context = (
                "\n[Previous Window Memory Entries (for reference to avoid duplication)]\n"
            )
            for entry in self.previous_entries[:3]:
                context += f"- {entry.lossless_restatement}\n"
        prompt = self._build_extraction_prompt(dialogue_text, context)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional information extraction assistant, "
                    "skilled at extracting structured, unambiguous information "
                    "from conversations. You must output valid JSON format."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        max_retries = 3
        response = ""
        for attempt in range(max_retries):
            try:
                response_format = (
                    {"type": "json_object"} if self.settings.use_json_format else None
                )
                response = self.llm_client.chat_completion(
                    messages,
                    temperature=0.1,
                    response_format=response_format,
                )
                return self._parse_llm_response(response)
            except Exception:
                if attempt == max_retries - 1:
                    return []
        return []

    def _build_extraction_prompt(self, dialogue_text: str, context: str) -> str:
        return f"""
Your task is to extract all valuable information from the following dialogues and convert them into structured memory entries.

{context}

[Current Window Dialogues]
{dialogue_text}

[Requirements]
1. Complete Coverage: Generate enough memory entries to ensure ALL information in the dialogues is captured
2. Force Disambiguation: Absolutely prohibit pronouns and relative time references
3. Lossless Information: Each entry's lossless_restatement must be a complete, independent, understandable sentence
4. Precise Extraction:
   - keywords: Core keywords
   - timestamp: Absolute time in ISO 8601 format if explicit time is mentioned
   - location: Specific location name if mentioned
   - persons: All person names mentioned
   - entities: Companies, products, organizations, etc.
   - topic: The topic of this information

Return ONLY a JSON array in the following format:
[
  {{
    "lossless_restatement": "Complete unambiguous restatement",
    "keywords": ["keyword1", "keyword2"],
    "timestamp": "YYYY-MM-DDTHH:MM:SS or null",
    "location": "location name or null",
    "persons": ["name1", "name2"],
    "entities": ["entity1", "entity2"],
    "topic": "topic phrase"
  }}
]
"""

    def _parse_llm_response(self, response: str) -> list[MemoryEntry]:
        data = self.llm_client.extract_json(response)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array but got: {type(data)}")
        return [
            MemoryEntry(
                lossless_restatement=item["lossless_restatement"],
                keywords=item.get("keywords", []),
                timestamp=item.get("timestamp"),
                location=item.get("location"),
                persons=item.get("persons", []),
                entities=item.get("entities", []),
                topic=item.get("topic"),
            )
            for item in data
        ]

    def _process_windows_parallel(self, windows: list[list[Dialogue]]) -> None:
        all_entries: list[MemoryEntry] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_parallel_workers
        ) as executor:
            futures = [
                executor.submit(self._generate_memory_entries, window)
                for window in windows
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    all_entries.extend(future.result())
                except Exception:
                    continue
        if all_entries:
            self.vector_store.add_entries(all_entries)
            self.processed_count += sum(len(window) for window in windows)
            self.previous_entries = all_entries[-10:]

