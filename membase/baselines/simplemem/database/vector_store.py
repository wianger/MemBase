from __future__ import annotations

import os
from typing import Any

import lancedb
import pyarrow as pa

from ..models.memory_entry import MemoryEntry
from ..settings import SimpleMemSettings
from ..utils.embedding import EmbeddingModel


class VectorStore:
    """Multi-view indexing store backed by LanceDB."""

    def __init__(
        self,
        settings: SimpleMemSettings,
        db_path: str | None = None,
        embedding_model: EmbeddingModel | None = None,
        table_name: str | None = None,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        self.settings = settings
        self.db_path = db_path or settings.lancedb_path
        self.embedding_model = embedding_model or EmbeddingModel(settings)
        self.table_name = table_name or settings.memory_table_name
        self.table = None
        self._fts_initialized = False
        self._is_cloud_storage = self.db_path.startswith(("gs://", "s3://", "az://"))

        if self._is_cloud_storage:
            self.db = lancedb.connect(self.db_path, storage_options=storage_options)
        else:
            os.makedirs(self.db_path, exist_ok=True)
            self.db = lancedb.connect(self.db_path)

        self._init_table()

    def _init_table(self) -> None:
        schema = pa.schema(
            [
                pa.field("entry_id", pa.string()),
                pa.field("lossless_restatement", pa.string()),
                pa.field("keywords", pa.list_(pa.string())),
                pa.field("timestamp", pa.string()),
                pa.field("location", pa.string()),
                pa.field("persons", pa.list_(pa.string())),
                pa.field("entities", pa.list_(pa.string())),
                pa.field("topic", pa.string()),
                pa.field(
                    "vector",
                    pa.list_(pa.float32(), self.embedding_model.dimension),
                ),
            ]
        )

        if self.table_name not in self.db.table_names():
            self.table = self.db.create_table(self.table_name, schema=schema)
        else:
            self.table = self.db.open_table(self.table_name)

    def _init_fts_index(self) -> None:
        if self._fts_initialized:
            return
        try:
            if self._is_cloud_storage:
                self.table.create_fts_index(
                    "lossless_restatement",
                    use_tantivy=False,
                    replace=True,
                )
            else:
                self.table.create_fts_index(
                    "lossless_restatement",
                    use_tantivy=True,
                    tokenizer_name="en_stem",
                    replace=True,
                )
            self._fts_initialized = True
        except Exception:
            pass

    def _results_to_entries(self, results: list[dict[str, Any]]) -> list[MemoryEntry]:
        entries = []
        for result in results:
            entries.append(
                MemoryEntry(
                    entry_id=result["entry_id"],
                    lossless_restatement=result["lossless_restatement"],
                    keywords=list(result.get("keywords") or []),
                    timestamp=result.get("timestamp") or None,
                    location=result.get("location") or None,
                    persons=list(result.get("persons") or []),
                    entities=list(result.get("entities") or []),
                    topic=result.get("topic") or None,
                )
            )
        return entries

    def add_entries(self, entries: list[MemoryEntry]) -> None:
        if not entries:
            return
        restatements = [entry.lossless_restatement for entry in entries]
        vectors = self.embedding_model.encode_documents(restatements)

        data = []
        for entry, vector in zip(entries, vectors):
            data.append(
                {
                    "entry_id": entry.entry_id,
                    "lossless_restatement": entry.lossless_restatement,
                    "keywords": entry.keywords,
                    "timestamp": entry.timestamp or "",
                    "location": entry.location or "",
                    "persons": entry.persons,
                    "entities": entry.entities,
                    "topic": entry.topic or "",
                    "vector": vector.tolist(),
                }
            )
        self.table.add(data)
        if not self._fts_initialized:
            self._init_fts_index()

    def semantic_search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        try:
            if self.table.count_rows() == 0:
                return []
            query_vector = self.embedding_model.encode_single(query, is_query=True)
            results = self.table.search(query_vector.tolist()).limit(top_k).to_list()
            return self._results_to_entries(results)
        except Exception:
            return []

    def keyword_search(self, keywords: list[str], top_k: int = 3) -> list[MemoryEntry]:
        try:
            if not keywords or self.table.count_rows() == 0:
                return []
            query = " ".join(keywords)
            results = self.table.search(query).limit(top_k).to_list()
            return self._results_to_entries(results)
        except Exception:
            return []

    def structured_search(
        self,
        persons: list[str] | None = None,
        timestamp_range: tuple[str, str] | None = None,
        location: str | None = None,
        entities: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[MemoryEntry]:
        try:
            if self.table.count_rows() == 0:
                return []
            if not any([persons, timestamp_range, location, entities]):
                return []

            conditions: list[str] = []
            if persons:
                values = ", ".join([f"'{p}'" for p in persons])
                conditions.append(f"array_has_any(persons, make_array({values}))")
            if location:
                safe_location = location.replace("'", "''")
                conditions.append(f"location LIKE '%{safe_location}%'")
            if entities:
                values = ", ".join([f"'{e}'" for e in entities])
                conditions.append(f"array_has_any(entities, make_array({values}))")
            if timestamp_range:
                start_time, end_time = timestamp_range
                conditions.append(
                    f"timestamp >= '{start_time}' AND timestamp <= '{end_time}'"
                )

            query = self.table.search().where(" AND ".join(conditions), prefilter=True)
            if top_k is not None:
                query = query.limit(top_k)
            return self._results_to_entries(query.to_list())
        except Exception:
            return []

    def get_all_entries(self) -> list[MemoryEntry]:
        return self._results_to_entries(self.table.to_arrow().to_pylist())

    def optimize(self) -> None:
        self.table.optimize()

    def clear(self) -> None:
        self.db.drop_table(self.table_name)
        self._fts_initialized = False
        self._init_table()

