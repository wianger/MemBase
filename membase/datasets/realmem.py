import json
import os
import re
from datetime import datetime, timedelta
from pydantic import (
    Field,
    PrivateAttr,
    model_validator,
)
from .online_base import OnlineEvalEnv, OnlineMemBaseDataset
from ..model_types.dataset import (
    Trajectory,
    Session,
    Message,
)
from ..model_types.evaluation import OnlineEvalResult, MetricResult
from ..model_types.memory import MemoryEntry
from ..layers.base import MemBaseLayer
from ..inference_utils.operators import QuestionAnsweringOperator, LLMExactMatch
from typing import Any, Self


class RealMemEvalEnv(OnlineEvalEnv):
    """Evaluation environment for RealMem online evaluation."""

    top_k: int = Field(
        default=10,
        description="Number of memories to retrieve per task.",
    )
    retrieval_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra keyword arguments forwarded to memory layer's retrieve method.",
    )

    qa_model: str = Field(
        default="gpt-4.1-mini",
        description="Model name for question-answering generation.",
    )
    judge_model: str = Field(
        default="gpt-4.1-mini",
        description="Model name for the LLM-as-a-judge.",
    )

    # Note that in the official implementation, the temperature for 
    # the question-answering generation is set to 0.7.
    generation_kwargs: dict[str, Any] = Field(
        default_factory=lambda: {"temperature": 0.0},
        description="Generation parameters forwarded to the LLM operators.",
    )

    api_keys: list[str] | None = Field(
        default=None,
        description=(
            "API keys for the LLM operator. "
            "If provided, they take precedence over ``api_config_path``."
        ),
    )
    base_urls: list[str] | None = Field(
        default=None,
        description=(
            "Base URLs for the LLM operator. "
            "If provided, they take precedence over ``api_config_path``."
        ),
    )
    api_config_path: str | None = Field(
        default=None,
        description="Path to the API config file.",
    )

    # Private operators initialised during validation.
    _qa_operator: QuestionAnsweringOperator = PrivateAttr()
    _judge_operator: LLMExactMatch = PrivateAttr()

    @model_validator(mode="after")
    def _init_operators(self) -> Self:
        """Resolve API credentials and build the LLM operators."""
        interface_kwargs = self._resolve_interface_kwargs()

        self._qa_operator = QuestionAnsweringOperator(
            prompt_name="realmem-question-answering",
            model_name=self.qa_model,
            **interface_kwargs,
        )
        self._judge_operator = LLMExactMatch(
            prompt_name="realmem-lm-score",
            model_name=self.judge_model,
            **interface_kwargs,
        )
        return self

    def _resolve_interface_kwargs(self) -> dict[str, Any]:
        """Build the interface keyword arguments for the LLM operator."""
        if self.api_keys is not None and self.base_urls is not None:
            return {
                "api_keys": self.api_keys,
                "base_urls": self.base_urls,
            }

        if self.api_config_path is not None:
            with open(self.api_config_path, "r") as f:
                cfg = json.load(f)
            return {
                "api_keys": cfg["api_keys"], 
                "base_urls": cfg["base_urls"],
            }

        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is not None:
            return {
                "api_keys": [api_key],
                "base_urls": [
                    os.environ.get("OPENAI_API_BASE")
                ],
            }

        return {}

    @property
    def qa_operator(self) -> QuestionAnsweringOperator:
        """Get the question-answering operator."""
        return self._qa_operator

    @property
    def judge_operator(self) -> LLMExactMatch:
        """Get the LLM-as-a-judge operator."""
        return self._judge_operator


class RealMem(OnlineMemBaseDataset):
    """Dataset wrapper for RealMem."""

    @classmethod
    def read_raw_data(cls, path: str) -> Self:
        """Read RealMem data from a directory, JSON file, or JSONL file.

        Args:
            path (`str`):
                Path to a directory of json files, a single JSON
                file, or a JSONL file where each line is one persona's
                JSON object.

        Returns:
            `Self`:
                The dataset instance constructed from the raw data.
        """
        # Firstly, load persona objects from the given path.
        persona_objects = cls._load_persona_objects(path)

        # Secondly, construct trajectories.
        trajectories = []

        for source_name, data in persona_objects:
            # Extract the trajectory metadata.
            traj_meta = data["_metadata"]

            person_name = traj_meta["person_name"]
            dialogues = data["dialogues"]
            sessions = []

            # A mapping from the source evidence content to its corresponding message ID.
            source_evidence_map = {} 

            for dialogue in dialogues:
                session_id = dialogue["session_identifier"]
                session_uuid = dialogue["session_uuid"]

                raw_time = dialogue["current_time"]
                # The raw data only provides the date
                # so we set the session time to 12:00 PM on that day.
                session_dt = datetime.strptime(raw_time, "%Y-%m-%d (%A)") + timedelta(hours=12)
                session_ts = session_dt.isoformat()

                turns = dialogue["dialogue_turns"]
                messages = [] 

                for t_idx, turn in enumerate(turns, start=1):
                    # `Assistant` -> `assistant`
                    # `User` -> `user`
                    role = turn["speaker"].lower()
                    content = turn["content"]

                    # Some messages don't contain `is_query` key.
                    is_task = turn.get("is_query", False)
                    name = role if role == "assistant" else person_name

                    msg_meta = {
                        "is_task": is_task,
                    }
                    if is_task:
                        # Get the corresponding source evidence. 
                        msg_meta["evidence"] = [] 
                        for memory in turns[t_idx]["memory_used"]:
                            memory_session_uuid = memory["session_uuid"]
                            memory_content = memory["content"]
                            evidence_key = f"{memory_session_uuid}:{memory_content}"

                            # If this query has an abnormal source evidence, we filter it out.
                            if evidence_key not in source_evidence_map: 
                                msg_meta["is_task"] = False 
                                break 

                            msg_meta["evidence"].append(
                                source_evidence_map[evidence_key]
                            )

                        if msg_meta["is_task"]:
                            msg_meta["query_id"] = turn["query_id"]
                            msg_meta["topic"] = turn["topic"]
                            msg_meta["question_type"] = turn["category_name"]
                            msg_meta["session_type"] = turn["session_type"]

                            # The reference trajectory is the assistant's response.   
                            golden_answer = turns[t_idx]["content"]                       
                            msg_meta["ref_traj"] = golden_answer
                        else:
                            del msg_meta["evidence"]

                    messages.append(
                        Message(
                            id=f"{session_uuid}-{t_idx}",
                            name=name,
                            role=role,
                            content=content,
                            timestamp=session_ts,
                            metadata=msg_meta,
                        )
                    )

                extracted = dialogue["extracted_memory"].copy() 
                session_meta = {
                    "extracted_memory": extracted,
                    "session_identifier": session_id,
                } 
                for extracted_memory in extracted:
                    extracted_memory_content = extracted_memory["content"]
                    source_turn = extracted_memory["source_turn"]

                    # Currently, source evidence with `source_turn == -1` is temporarily filtered out.
                    # Any questions that rely on this source evidence will be excluded from evaluation.
                    if source_turn == -1: 
                        continue 
                    elif source_turn == 0:
                        source_turn = 1  

                    # We have checked that each source evidence key is unique.
                    # Each extracted memory's `session_uuid` is the same as the 
                    # current session's `session_uuid`.
                    key = f"{session_uuid}:{extracted_memory_content}"
                    source_evidence_map[key] = f"{session_uuid}-{source_turn}"

                sessions.append(
                    Session(
                        id=session_uuid,
                        messages=messages,
                        metadata=session_meta,
                    )
                )

            traj_meta["source"] = source_name
            trajectories.append(
                Trajectory(
                    id=f"realmem-{person_name}",
                    sessions=sorted(sessions),
                    metadata=traj_meta,
                )
            )

        return cls(
            trajectories=trajectories,
            qa_pair_lists=[[] for _ in trajectories],
        )

    def _generate_metadata(self) -> dict[str, Any]:
        meta = {
            "name": "RealMem",
            "paper": "RealMem: Benchmarking LLMs in Real-World Memory-Driven Interaction",
            "codebase_url": "https://github.com/AvatarMemory/RealMemBench",
            "size": len(self),
            "total_sessions": 0,
            "total_messages": 0,
            "total_tasks": 0,
        }

        question_type_stats = {}
        session_type_stats = {}
        topic_stats = {}

        for trajectory, _ in self:
            meta["total_sessions"] += len(trajectory)
            for session in trajectory:
                meta["total_messages"] += len(session)
                for msg in session:
                    if not msg.metadata.get("is_task"):
                        continue
                    meta["total_tasks"] += 1
                    qtype = msg.metadata["question_type"]
                    question_type_stats[qtype] = question_type_stats.get(qtype, 0) + 1
                    stype = msg.metadata["session_type"]
                    session_type_stats[stype] = session_type_stats.get(stype, 0) + 1
                    topic = msg.metadata["topic"]
                    topic_stats[topic] = topic_stats.get(topic, 0) + 1

        meta["question_type_stats"] = question_type_stats
        meta["session_type_stats"] = session_type_stats
        meta["topic_stats"] = topic_stats

        n_traj = len(self)
        n_sessions = meta["total_sessions"]
        if n_traj > 0 and n_sessions > 0:
            meta["avg_session_per_trajectory"] = n_sessions / n_traj
            meta["avg_message_per_session"] = meta["total_messages"] / n_sessions
            meta["avg_task_per_trajectory"] = meta["total_tasks"] / n_traj
        else:
            meta["avg_session_per_trajectory"] = 0.0
            meta["avg_message_per_session"] = 0.0
            meta["avg_task_per_trajectory"] = 0.0

        return meta

    @classmethod
    def online_evaluate(
        cls,
        messages: Message | list[Message] | Session,
        layer: MemBaseLayer,
        env: RealMemEvalEnv,
    ) -> list[OnlineEvalResult]:
        if isinstance(messages, (list, Session)):
            if len(messages) != 1:
                raise ValueError(
                    "RealMem online evaluation only supports a single task message at a time. "
                    f"However, {len(messages)} task messages are provided."
                )
            message = messages[0]
        else:
            message = messages

        question = message.content
        if "ref_traj" not in message.metadata:
            raise ValueError(
                "The message does not contain the golden answer. "
                "Please check the data format."
            )

        # 1. Retrieve related memories before updating the memory layer.
        retrieved_memories = layer.retrieve(
            question, 
            k=env.top_k, 
            **env.retrieval_kwargs,
        )
        if len(retrieved_memories) == 0:
            retrieved_memories = [
                MemoryEntry(
                    content="[NO RETRIEVED MEMORIES]",
                    formatted_content="[NO RETRIEVED MEMORIES]",
                    metadata={},
                )
            ]

        # See https://github.com/AvatarMemory/RealMemBench/blob/67afd0891d603adcc4458ff0449df306ef296b7a/eval/run_generation.py#L12. 
        context_parts = []
        for i, entry in enumerate(retrieved_memories, start=1):
            text = entry.formatted_content or entry.content
            context_parts.append(f"---- idx {i} ----\n{text}")
        context = "\n\n".join(context_parts)

        # 2. Generate answer via question-answering model.
        qa_responses = env.qa_operator(
            [question], 
            [context],
            batch_size=1,
            aggregate=False,
            **env.generation_kwargs,
        )
        generated = qa_responses[0].get("processed_content")

        # 3. Build the rollout.
        augmented_msg_dict = message.model_dump(mode="python")
        augmented_msg_dict["metadata"]["original_message_content"] = question
        augmented_msg_dict["metadata"]["retrieved_memories"] = [
            entry.model_dump(mode="python") 
            for entry in retrieved_memories
        ]
        # It is the actual message content that the model sees.
        augmented_content = env.qa_operator.prompt.substitute(
            question=question, 
            context=context,
        )
        augmented_msg_dict["content"] = augmented_content
        augmented_msg = Message.model_validate(augmented_msg_dict)

        # 4. Get the assistant's generated response.
        response_msg = Message(
            name="assistant",
            role="assistant",
            content=generated,
            timestamp=message.timestamp,
        )

        rollout = [augmented_msg, response_msg]

        # 5. Judge against golden answer.
        golden_answer = message.metadata["ref_traj"]
        metrics = {}

        judge_responses = env.judge_operator(
            [question], 
            [[golden_answer]], 
            [generated],
            batch_size=1,
            aggregate=False,
            **env.generation_kwargs,
        )
        raw_judge = judge_responses[0].get("processed_content")
        metrics["llm_judge"] = MetricResult(
            value=cls.parse_judge_response(raw_judge),
            metadata={"judge_response": raw_judge},
        )

        # 6. Update the memory layer.
        layer.add_message(message)

        return [OnlineEvalResult(metrics=metrics, rollout=rollout)]

    @classmethod
    def parse_judge_response(cls, content: str) -> float:
        """Convert the raw text output from the judge model into a correctness score.

        Args:
            content (`str`):
                The raw text content returned by the judge model.

        Returns:
            `float`:
                `1.0` if the prediction is judged correct, `0.0` otherwise.
        """
        # See https://github.com/AvatarMemory/RealMemBench/blob/main/eval/compute_llm_metrics_for_realmem.py#L114. 
        try:
            m = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
            if m:
                obj = json.loads(m.group(1))
                return float(obj.get("score", 0))
            m = re.search(r'\{[^{}]*"score"[^{}]*\}', content, re.DOTALL)
            if m:
                obj = json.loads(m.group(0))
                return float(obj.get("score", 0))
        except Exception as e:
            print("An error occurs when the judge response is parsed: ", e)
        return 0.0

    @classmethod
    def _load_persona_objects(
        cls,
        path: str,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Load persona JSON objects from a directory, file, or JSONL.

        Args:
            path (`str`):
                Path to a directory of json files, a single JSON
                file, or a JSONL file where each line is one persona's
                JSON object.

        Returns:
            `list[tuple[str, dict[str, Any]]]`:
                A list of tuples, each containing the source filename and 
                the persona's JSON object.
        """
        raw = []

        if os.path.isdir(path):
            for fname in os.listdir(path):
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(path, fname)
                with open(fpath, "r", encoding="utf-8") as fh:
                    raw.append(
                        (fname, json.load(fh))
                    )
        elif path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as fh:
                for line_idx, line in enumerate(fh, start=1):
                    raw.append(
                        (f"line_{line_idx}", json.loads(line))
                    )
        else:
            fname = os.path.basename(path)
            with open(path, "r", encoding="utf-8") as fh:
                raw.append(
                    (fname, json.load(fh))
                )

        raw.sort(key=lambda t: t[1]["_metadata"]["person_name"])
        return raw

