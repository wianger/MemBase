from __future__ import annotations

from ..simplemem.core.hybrid_retriever import HybridRetriever
from ..simplemem.core.memory_builder import MemoryBuilder
from ..simplemem.database.vector_store import VectorStore
from ..simplemem.models.memory_entry import Dialogue, MemoryEntry
from ..simplemem.settings import SimpleMemSettings
from ..simplemem.utils.embedding import EmbeddingModel
from ..simplemem.utils.llm_client import LLMClient


class SimpleMemSystem:
    """SimpleMem text-only system adapted for MemBase."""

    def __init__(
        self,
        settings: SimpleMemSettings,
        llm_client: LLMClient | None = None,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        db_path: str | None = None,
        table_name: str | None = None,
        clear_db: bool = False,
        enable_thinking: bool | None = None,
        use_streaming: bool | None = None,
        enable_planning: bool | None = None,
        enable_reflection: bool | None = None,
        max_reflection_rounds: int | None = None,
        enable_parallel_processing: bool | None = None,
        max_parallel_workers: int | None = None,
        enable_parallel_retrieval: bool | None = None,
        max_retrieval_workers: int | None = None,
    ) -> None:
        self.settings = settings
        self.llm_client = llm_client or LLMClient(
            settings=settings,
            api_key=api_key,
            model=model,
            base_url=base_url,
            enable_thinking=enable_thinking,
            use_streaming=use_streaming,
        )
        self.embedding_model = EmbeddingModel(settings=settings)
        self.vector_store = VectorStore(
            settings=settings,
            db_path=db_path,
            embedding_model=self.embedding_model,
            table_name=table_name,
        )

        if clear_db:
            self.vector_store.clear()

        self.memory_builder = MemoryBuilder(
            settings=settings,
            llm_client=self.llm_client,
            vector_store=self.vector_store,
            enable_parallel_processing=enable_parallel_processing,
            max_parallel_workers=max_parallel_workers,
        )
        self.hybrid_retriever = HybridRetriever(
            settings=settings,
            llm_client=self.llm_client,
            vector_store=self.vector_store,
            enable_planning=enable_planning,
            enable_reflection=enable_reflection,
            max_reflection_rounds=max_reflection_rounds,
            enable_parallel_retrieval=enable_parallel_retrieval,
            max_retrieval_workers=max_retrieval_workers,
        )

    def add_dialogue(
        self,
        speaker: str,
        content: str,
        timestamp: str | None = None,
        role: str | None = None,
    ) -> None:
        dialogue_id = (
            self.memory_builder.processed_count
            + len(self.memory_builder.dialogue_buffer)
            + 1
        )
        dialogue = Dialogue(
            dialogue_id=dialogue_id,
            speaker=speaker,
            content=content,
            timestamp=timestamp,
            role=role,
        )
        self.memory_builder.add_dialogue(dialogue)

    def add_dialogues(self, dialogues: list[Dialogue]) -> None:
        self.memory_builder.add_dialogues(dialogues)

    def finalize(self) -> None:
        self.memory_builder.process_remaining()

    def get_all_memories(self) -> list[MemoryEntry]:
        return self.vector_store.get_all_entries()

