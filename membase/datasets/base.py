from __future__ import annotations
from ..inference_utils.operators import LLMExactMatch
from ..model_types.dataset import MemoryDataset, QuestionAnswerPair
from typing import Any


class MemBaseDataset(MemoryDataset):
    """Intermediate base for all MemBase dataset implementations.

    It provides a default (empty) metadata generator, a JSON-based ``read_raw_data``
    loader that leverages Pydantic deserialization, and a ``save_dataset`` method
    for persisting the dataset to disk.  Dataset-specific subclasses should
    override ``read_raw_data`` to parse their own raw formats and may override
    ``_generate_metadata`` for richer statistics.
    """

    @classmethod
    def get_judge_template_name(cls, qa_pair: QuestionAnswerPair) -> str:
        """Get the judge prompt template name for a question-answer pair.

        Subclasses can overwrite this method to customize the LLM-as-a-Judge prompt template.

        Args:
            qa_pair (`QuestionAnswerPair`):
                The question-answer pair to get the judge prompt template name for.

        Returns:
            `str`:
                The name of the judge prompt template.
        """
        return qa_pair.metadata.get("question_type", "default-exact-match")

    @classmethod
    def parse_judge_response(cls, content: str) -> float:
        """Convert the raw text output from the judge model into a correctness score.

        The default behaviour checks whether the word `"yes"` appears in the
        lowercased response. Subclasses can override this method to accommodate
        different judge formats.

        Args:
            content (`str`):
                The raw text content returned by the judge model.

        Returns:
            `float`:
                `1.0` if the prediction is judged correct, `0.0` otherwise.
        """
        return float("yes" in content.lower())

    @classmethod
    def evaluate(
        cls,
        qa_pairs: list[QuestionAnswerPair],
        predictions: list[str],
        judge_model: str = "gpt-4.1-mini",
        judge_batch_size: int = 4,
        **kwargs: Any,
    ) -> list[dict[str, dict[str, Any]]]:
        """Evaluate the predictions against the golden answers for each question-answer pair.

        This base implementation uses an LLM-as-a-Judge approach to determine whether
        each prediction is correct by comparing it against the golden answers. The judge
        prompt template is resolved via ``get_judge_template_name``, and the raw judge
        output is converted to a correctness score via ``parse_judge_response``. Subclasses
        can override either of these class methods to customize the judging behaviour.

        Subclasses can also override this method to incorporate additional evaluation
        metrics beyond accuracy. For example, a dataset that annotates source evidence
        in each question-answer pair could compute retrieval recall@k by comparing the
        retrieved results against the ground-truth evidence IDs. When overriding, call
        ``super().evaluate(...)`` first to obtain the base accuracy results, then merge
        the extra per-pair metrics into each result dictionary::

            @classmethod
            def evaluate(cls, qa_pairs, predictions, **kwargs):
                results = super().evaluate(qa_pairs, predictions, **kwargs)
                retrieval_results = kwargs.get("retrieval_results", [])
                for i, qa_pair in enumerate(qa_pairs):
                    evidence_ids = set(qa_pair.metadata.get("evidence_ids", []))
                    retrieved_ids = set(r["id"] for r in retrieval_results[i])
                    hit = len(evidence_ids & retrieved_ids)
                    k = len(retrieval_results[i]) if retrieval_results else 1
                    results[i][f"recall@{k}"] = {
                        "value": hit / len(evidence_ids) if evidence_ids else 0.0,
                        "metadata": {},
                    }
                return results

        Args:
            qa_pairs (`list[QuestionAnswerPair]`):
                The question-answer pairs to evaluate.
            predictions (`list[str]`):
                The predicted answers, one per question-answer pair.
            judge_model (`str`, defaults to `"gpt-4.1-mini"`):
                The model name or path used for the LLM judge.
            judge_batch_size (`int`, defaults to `4`):
                Batch size for the judge model inference.
            **kwargs (`Any`):
                Remaining keyword arguments are forwarded to the LLM interface
                constructor. If `api_key`, `api_keys`, `base_url` or `base_urls` is present,
                an OpenAI-compatible API backend is used and the remaining arguments correspond
                to those accepted by `openai.OpenAI`. Otherwise, a local vLLM backend is assumed and
                the arguments correspond to `vllm.LLM` constructor parameters. An optional `generation_config`
                dictionary can be included to supply generation-time parameters (mapping to the chat completions
                request body for OpenAI, or `vllm.SamplingParams` for vLLM). If `generation_config` is not provided,
                `{"temperature": 0.0}` is used by default for deterministic judging.

        Returns:
            `list[dict[str, dict[str, Any]]]`:
                Per-pair evaluation results containing the metrics.
        """
        if len(qa_pairs) != len(predictions):
            raise ValueError(
                f"The number of question-answer pairs ({len(qa_pairs)}) and predictions "
                f"({len(predictions)}) must be the same."
            )

        # Separate generation-time config from interface constructor kwargs.
        generation_config = {"temperature": 0.0}
        generation_config.update(kwargs.pop("generation_config", {}))

        # Group question-answer pairs by their judge template name so that pairs sharing
        # the same prompt can be evaluated in a single batched call.
        groups = {}
        for idx, qa_pair in enumerate(qa_pairs):
            template_name = cls.get_judge_template_name(qa_pair)
            groups.setdefault(template_name, []).append(idx)

        results = [{} for _ in range(len(qa_pairs))]

        # Use the first group's template to initialize the operator; subsequent
        # groups switch the prompt via `set_prompt`.
        first_template = next(iter(groups))
        judge_operator = LLMExactMatch(
            prompt_name=first_template,
            model_name=judge_model,
            **kwargs,
        )

        for template_name, indices in groups.items():
            judge_operator.set_prompt(template_name)

            batch_questions = [qa_pairs[i].question for i in indices]
            batch_golden_answers = [qa_pairs[i].golden_answers for i in indices]
            batch_predictions = [predictions[i] for i in indices]

            judge_responses = judge_operator(
                batch_questions,
                batch_golden_answers,
                batch_predictions,
                batch_size=judge_batch_size,
                aggregate=False,
                **generation_config,
            )

            for local_pos, global_idx in enumerate(indices):
                content = judge_responses[local_pos].get("processed_content")
                if content is None:
                    raise ValueError(
                        "The judge model's response for question "
                        f"'{qa_pairs[global_idx].question}' is empty."
                    )
                results[global_idx] = {
                    "llm_judge": {
                        "value": cls.parse_judge_response(content),
                        "metadata": {"judge_response": content},
                    },
                }

        # Print the evaluation summary.
        cls.print_evaluation_summary(results, qa_pairs)

        return results

    @classmethod
    def print_evaluation_summary(
        cls,
        results: list[dict[str, dict[str, Any]]],
        qa_pairs: list[QuestionAnswerPair],
    ) -> None:
        """Print a summary of evaluation results grouped by metric and question type.

        This function prints the overall average value and a per-question-type breakdown
        (when more than one question type exists).

        Args:
            results (`list[dict[str, dict[str, Any]]]`):
                Per-pair evaluation results. Each element maps metric names to
                dictionaries containing at least a value key.
            qa_pairs (`list[QuestionAnswerPair]`):
                The corresponding question-answer pairs, used to extract
                question types from metadata.
        """
        if not results:
            return

        # Collect all metric names (preserving insertion order).
        metric_names = list(
            dict.fromkeys(
                key for result in results for key in result.keys()
            )
        )

        # Build question-type groups.
        question_type_groups = {}
        for idx, qa_pair in enumerate(qa_pairs):
            qtype = qa_pair.metadata.get("question_type", "unknown")
            question_type_groups.setdefault(qtype, []).append(idx)

        for metric in metric_names:
            values = [results[i][metric]["value"] for i in range(len(results))]
            overall = sum(values) / len(values)
            print(f"[{metric}] Overall: {overall:.4f}")

            if len(question_type_groups) > 1:
                for qtype, indices in sorted(question_type_groups.items()):
                    avg = sum(results[i][metric]["value"] for i in indices) / len(indices)
                    print(f"  {qtype}: {avg:.4f} ({len(indices)} questions)")
            print()
