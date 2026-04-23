from __future__ import annotations

import concurrent.futures
from datetime import timedelta
from typing import Any

import dateparser

from ..database.vector_store import VectorStore
from ..models.memory_entry import MemoryEntry
from ..settings import SimpleMemSettings
from ..utils.llm_client import LLMClient


class HybridRetriever:
    """Intent-aware hybrid retriever used by the vendored SimpleMem backend."""

    def __init__(
        self,
        settings: SimpleMemSettings,
        llm_client: LLMClient,
        vector_store: VectorStore,
        semantic_top_k: int | None = None,
        keyword_top_k: int | None = None,
        structured_top_k: int | None = None,
        enable_planning: bool | None = None,
        enable_reflection: bool | None = None,
        max_reflection_rounds: int | None = None,
        enable_parallel_retrieval: bool | None = None,
        max_retrieval_workers: int | None = None,
    ) -> None:
        self.settings = settings
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.semantic_top_k = semantic_top_k or settings.semantic_top_k
        self.keyword_top_k = keyword_top_k or settings.keyword_top_k
        self.structured_top_k = structured_top_k or settings.structured_top_k
        self.enable_planning = (
            settings.enable_planning if enable_planning is None else enable_planning
        )
        self.enable_reflection = (
            settings.enable_reflection
            if enable_reflection is None
            else enable_reflection
        )
        self.max_reflection_rounds = (
            settings.max_reflection_rounds
            if max_reflection_rounds is None
            else max_reflection_rounds
        )
        self.enable_parallel_retrieval = (
            settings.enable_parallel_retrieval
            if enable_parallel_retrieval is None
            else enable_parallel_retrieval
        )
        self.max_retrieval_workers = (
            settings.max_retrieval_workers
            if max_retrieval_workers is None
            else max_retrieval_workers
        )

    def retrieve(
        self,
        query: str,
        enable_reflection: bool | None = None,
        semantic_top_k: int | None = None,
        keyword_top_k: int | None = None,
        structured_top_k: int | None = None,
    ) -> list[MemoryEntry]:
        semantic_limit = semantic_top_k or self.semantic_top_k
        keyword_limit = keyword_top_k or self.keyword_top_k
        structured_limit = structured_top_k or self.structured_top_k
        if self.enable_planning:
            return self._retrieve_with_planning(
                query=query,
                enable_reflection=enable_reflection,
                semantic_top_k=semantic_limit,
                keyword_top_k=keyword_limit,
                structured_top_k=structured_limit,
            )
        return self.vector_store.semantic_search(query, top_k=semantic_limit)

    def _retrieve_with_planning(
        self,
        query: str,
        enable_reflection: bool | None,
        semantic_top_k: int,
        keyword_top_k: int,
        structured_top_k: int,
    ) -> list[MemoryEntry]:
        information_plan = self._analyze_information_requirements(query)
        search_queries = self._generate_targeted_queries(query, information_plan)

        if self.enable_parallel_retrieval and len(search_queries) > 1:
            all_results = self._execute_parallel_searches(
                search_queries,
                semantic_top_k,
            )
        else:
            all_results = []
            for search_query in search_queries:
                all_results.extend(
                    self.vector_store.semantic_search(search_query, top_k=semantic_top_k)
                )

        query_analysis = self._analyze_query(query)
        all_results.extend(
            self._keyword_search(query, query_analysis, keyword_top_k)
        )
        all_results.extend(
            self._structured_search(query_analysis, structured_top_k)
        )
        merged_results = self._merge_and_deduplicate_entries(all_results)

        should_use_reflection = (
            self.enable_reflection
            if enable_reflection is None
            else enable_reflection
        )
        if should_use_reflection:
            return self._retrieve_with_intelligent_reflection(
                query,
                merged_results,
                information_plan,
                semantic_top_k,
            )
        return merged_results

    def _analyze_query(self, query: str) -> dict[str, Any]:
        prompt = f"""
Analyze the following query and extract key information:

Query: {query}

Please extract:
1. keywords: List of keywords
2. persons: Person names mentioned
3. time_expression: Time expression if any
4. location: Location if any
5. entities: Entities

Return ONLY JSON:
{{
  "keywords": ["keyword1", "keyword2"],
  "persons": ["name1"],
  "time_expression": "time expression or null",
  "location": "location or null",
  "entities": ["entity1"]
}}
"""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a query analysis assistant. You must output valid "
                    "JSON format."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        for attempt in range(3):
            try:
                response_format = (
                    {"type": "json_object"} if self.settings.use_json_format else None
                )
                response = self.llm_client.chat_completion(
                    messages,
                    temperature=0.1,
                    response_format=response_format,
                )
                return self.llm_client.extract_json(response)
            except Exception:
                if attempt == 2:
                    return {
                        "keywords": [query],
                        "persons": [],
                        "time_expression": None,
                        "location": None,
                        "entities": [],
                    }
        return {
            "keywords": [query],
            "persons": [],
            "time_expression": None,
            "location": None,
            "entities": [],
        }

    def _keyword_search(
        self,
        query: str,
        query_analysis: dict[str, Any],
        top_k: int,
    ) -> list[MemoryEntry]:
        keywords = query_analysis.get("keywords", [])
        if not keywords:
            keywords = [query]
        return self.vector_store.keyword_search(keywords, top_k=top_k)

    def _structured_search(
        self,
        query_analysis: dict[str, Any],
        top_k: int,
    ) -> list[MemoryEntry]:
        persons = query_analysis.get("persons", [])
        location = query_analysis.get("location")
        entities = query_analysis.get("entities", [])
        time_expression = query_analysis.get("time_expression")
        timestamp_range = (
            self._parse_time_range(time_expression) if time_expression else None
        )
        if not any([persons, location, entities, timestamp_range]):
            return []
        return self.vector_store.structured_search(
            persons=persons or None,
            location=location,
            entities=entities or None,
            timestamp_range=timestamp_range,
            top_k=top_k,
        )

    def _parse_time_range(self, time_expression: str) -> tuple[str, str] | None:
        try:
            parsed_date = dateparser.parse(
                time_expression,
                settings={"PREFER_DATES_FROM": "past"},
            )
            if parsed_date is None:
                return None
            start_time = parsed_date.replace(hour=0, minute=0, second=0)
            end_time = parsed_date.replace(hour=23, minute=59, second=59)
            if "week" in time_expression.lower() or "周" in time_expression:
                start_time = start_time - timedelta(days=7)
                end_time = end_time + timedelta(days=7)
            return (start_time.isoformat(), end_time.isoformat())
        except Exception:
            return None

    def _merge_and_deduplicate_entries(
        self,
        entries: list[MemoryEntry],
    ) -> list[MemoryEntry]:
        seen_ids = set()
        merged = []
        for entry in entries:
            if entry.entry_id not in seen_ids:
                seen_ids.add(entry.entry_id)
                merged.append(entry)
        return merged

    def _execute_parallel_searches(
        self,
        search_queries: list[str],
        semantic_top_k: int,
    ) -> list[MemoryEntry]:
        all_results: list[MemoryEntry] = []
        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_retrieval_workers
            ) as executor:
                futures = [
                    executor.submit(
                        self.vector_store.semantic_search,
                        query,
                        semantic_top_k,
                    )
                    for query in search_queries
                ]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        all_results.extend(future.result())
                    except Exception:
                        continue
        except Exception:
            for query in search_queries:
                all_results.extend(
                    self.vector_store.semantic_search(query, top_k=semantic_top_k)
                )
        return all_results

    def _analyze_information_requirements(self, query: str) -> dict[str, Any]:
        prompt = f"""
Analyze the following question and determine what specific information is required to answer it comprehensively.

Question: {query}

Return ONLY JSON:
{{
  "question_type": "type of question",
  "key_entities": ["entity1", "entity2"],
  "required_info": [
    {{
      "info_type": "what kind of information",
      "description": "specific information needed",
      "priority": "high"
    }}
  ],
  "relationships": ["relationship1"],
  "minimal_queries_needed": 2
}}
"""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an intelligent information requirement analyst. "
                    "You must output valid JSON format."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            response_format = (
                {"type": "json_object"} if self.settings.use_json_format else None
            )
            response = self.llm_client.chat_completion(
                messages,
                temperature=0.2,
                response_format=response_format,
            )
            return self.llm_client.extract_json(response)
        except Exception:
            return {
                "question_type": "general",
                "key_entities": [query],
                "required_info": [
                    {
                        "info_type": "general",
                        "description": "relevant information",
                        "priority": "high",
                    }
                ],
                "relationships": [],
                "minimal_queries_needed": 1,
            }

    def _generate_targeted_queries(
        self,
        original_query: str,
        information_plan: dict[str, Any],
    ) -> list[str]:
        prompt = f"""
Based on the information requirements analysis, generate the minimal set of targeted search queries needed to gather the required information.

Original Question: {original_query}
Information Requirements Analysis: {information_plan}

Return ONLY JSON:
{{
  "reasoning": "Brief explanation of the query strategy",
  "queries": ["query 1", "query 2"]
}}
"""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a query generation specialist. You must output "
                    "valid JSON format."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            response_format = (
                {"type": "json_object"} if self.settings.use_json_format else None
            )
            response = self.llm_client.chat_completion(
                messages,
                temperature=0.3,
                response_format=response_format,
            )
            result = self.llm_client.extract_json(response)
            queries = result.get("queries", [original_query])
            if original_query not in queries:
                queries.insert(0, original_query)
            return queries[:4]
        except Exception:
            return [original_query]

    def _retrieve_with_intelligent_reflection(
        self,
        query: str,
        initial_results: list[MemoryEntry],
        information_plan: dict[str, Any],
        semantic_top_k: int,
    ) -> list[MemoryEntry]:
        current_results = initial_results
        for _ in range(self.max_reflection_rounds):
            if not current_results:
                break
            completeness_status = self._analyze_information_completeness(
                query,
                current_results,
                information_plan,
            )
            if completeness_status == "complete":
                break
            additional_queries = self._generate_missing_info_queries(
                query,
                current_results,
                information_plan,
            )
            if not additional_queries:
                break
            additional_results: list[MemoryEntry] = []
            if self.enable_parallel_retrieval and len(additional_queries) > 1:
                additional_results = self._execute_parallel_searches(
                    additional_queries,
                    semantic_top_k,
                )
            else:
                for add_query in additional_queries:
                    additional_results.extend(
                        self.vector_store.semantic_search(
                            add_query,
                            top_k=semantic_top_k,
                        )
                    )
            current_results = self._merge_and_deduplicate_entries(
                current_results + additional_results
            )
        return current_results

    def _analyze_information_completeness(
        self,
        query: str,
        current_results: list[MemoryEntry],
        information_plan: dict[str, Any],
    ) -> str:
        prompt = f"""
Analyze whether the provided information is sufficient to answer the original question.

Original Question: {query}
Required Information Types: {information_plan.get("required_info", [])}
Current Available Information: {self._format_contexts_for_check(current_results)}

Return ONLY JSON:
{{
  "assessment": "complete" OR "incomplete",
  "reasoning": "Brief explanation",
  "missing_info_types": ["missing"],
  "coverage_percentage": 85
}}
"""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an information completeness evaluator. You must "
                    "output valid JSON format."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            response_format = (
                {"type": "json_object"} if self.settings.use_json_format else None
            )
            response = self.llm_client.chat_completion(
                messages,
                temperature=0.1,
                response_format=response_format,
            )
            result = self.llm_client.extract_json(response)
            return result.get("assessment", "incomplete")
        except Exception:
            return "incomplete"

    def _generate_missing_info_queries(
        self,
        original_query: str,
        current_results: list[MemoryEntry],
        information_plan: dict[str, Any],
    ) -> list[str]:
        prompt = f"""
Based on the original question and current available information, generate targeted search queries for the missing information.

Original Question: {original_query}
Required Information Types: {information_plan.get("required_info", [])}
Current Available Information: {self._format_contexts_for_check(current_results)}

Return ONLY JSON:
{{
  "missing_analysis": "what is missing",
  "targeted_queries": ["query 1", "query 2"]
}}
"""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a missing information query generator. You must "
                    "output valid JSON format."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            response_format = (
                {"type": "json_object"} if self.settings.use_json_format else None
            )
            response = self.llm_client.chat_completion(
                messages,
                temperature=0.3,
                response_format=response_format,
            )
            result = self.llm_client.extract_json(response)
            return result.get("targeted_queries", [])
        except Exception:
            return []

    def _format_contexts_for_check(self, contexts: list[MemoryEntry]) -> str:
        formatted = []
        for idx, entry in enumerate(contexts, start=1):
            parts = [f"[Info {idx}] {entry.lossless_restatement}"]
            if entry.timestamp:
                parts.append(f"Time: {entry.timestamp}")
            formatted.append(" | ".join(parts))
        return "\n".join(formatted)

