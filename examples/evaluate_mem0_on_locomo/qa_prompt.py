import json
import string

from membase.model_types.dataset import QuestionAnswerPair
from membase.model_types.memory import MemoryEntry


def get_mem0_qa_prompt() -> string.Template:
    """Return the upstream Mem0 LoCoMo non-graph QA prompt."""
    return string.Template(
        "You are an intelligent memory assistant tasked with retrieving accurate information from "
        "conversation memories.\n\n"
        "# CONTEXT:\n"
        "You have access to memories from two speakers in a conversation. These memories contain "
        "timestamped information that may be relevant to answering the question.\n\n"
        "# INSTRUCTIONS:\n"
        "1. Carefully analyze all provided memories from both speakers\n"
        "2. Pay special attention to the timestamps to determine the answer\n"
        "3. If the question asks about a specific event or fact, look for direct evidence in the memories\n"
        "4. If the memories contain contradictory information, prioritize the most recent memory\n"
        "5. If there is a question about time references (like 'last year', 'two months ago', etc.), "
        "calculate the actual date based on the memory timestamp. For example, if a memory from "
        "4 May 2022 mentions 'went to India last year,' then the trip occurred in 2021.\n"
        "6. Always convert relative time references to specific dates, months, or years. For example, "
        "convert 'last year' to '2022' or 'two months ago' to 'March 2023' based on the memory "
        "timestamp. Ignore the reference while answering the question.\n"
        "7. Focus only on the content of the memories from both speakers. Do not confuse character "
        "names mentioned in memories with the actual users who created those memories.\n"
        "8. The answer should be less than 5-6 words.\n\n"
        "# APPROACH (Think step by step):\n"
        "1. First, examine all memories that contain information related to the question\n"
        "2. Examine the timestamps and content of these memories carefully\n"
        "3. Look for explicit mentions of dates, times, locations, or events that answer the question\n"
        "4. If the answer requires calculation (e.g., converting relative time references), show your work\n"
        "5. Formulate a precise, concise answer based solely on the evidence in the memories\n"
        "6. Double-check that your answer directly addresses the question asked\n"
        "7. Ensure your final answer is specific and avoids vague time references\n\n"
        "Memories for user $speaker_1_user_id:\n\n"
        "$speaker_1_memories\n\n"
        "Memories for user $speaker_2_user_id:\n\n"
        "$speaker_2_memories\n\n"
        "Question: $question\n\n"
        "Answer:\n"
    )


def _format_memory_for_official_prompt(memory: MemoryEntry) -> str:
    timestamp = None

    nested_metadata = memory.metadata.get("metadata")
    if isinstance(nested_metadata, dict):
        timestamp = nested_metadata.get("timestamp")

    if timestamp is None:
        timestamp = memory.metadata.get("timestamp")
    if timestamp is None:
        timestamp = "Unknown time"

    content = memory.content.strip() if memory.content else "[NO RETRIEVED MEMORIES]"
    return f"{timestamp}: {content}"


def _split_memories_by_speaker(
    qa_pair: QuestionAnswerPair,
    retrieved_memories: list[MemoryEntry],
) -> tuple[str, list[str], str, list[str]]:
    speaker_names = list(qa_pair.metadata.get("speaker_names", []))
    speaker_1 = speaker_names[0] if len(speaker_names) > 0 else "speaker_1"
    speaker_2 = speaker_names[1] if len(speaker_names) > 1 else "speaker_2"

    speaker_1_memories = []
    speaker_2_memories = []
    fallback_memories = []

    lowered_speaker_1 = speaker_1.lower()
    lowered_speaker_2 = speaker_2.lower()

    for memory in retrieved_memories:
        content = memory.content or ""
        nested_metadata = memory.metadata.get("metadata", {})
        speakers = nested_metadata.get("speakers")
        formatted_memory = _format_memory_for_official_prompt(memory)

        if isinstance(speakers, str):
            lowered_speakers = speakers.lower()
            matched_speaker_1 = lowered_speaker_1 in lowered_speakers
            matched_speaker_2 = lowered_speaker_2 in lowered_speakers

            if matched_speaker_1 and not matched_speaker_2:
                speaker_1_memories.append(formatted_memory)
                continue
            if matched_speaker_2 and not matched_speaker_1:
                speaker_2_memories.append(formatted_memory)
                continue

        lowered_content = content.lower()
        matched_speaker_1 = lowered_speaker_1 in lowered_content
        matched_speaker_2 = lowered_speaker_2 in lowered_content

        if matched_speaker_1 and not matched_speaker_2:
            speaker_1_memories.append(formatted_memory)
        elif matched_speaker_2 and not matched_speaker_1:
            speaker_2_memories.append(formatted_memory)
        else:
            fallback_memories.append(formatted_memory)

    for idx, memory in enumerate(fallback_memories):
        if idx % 2 == 0:
            speaker_1_memories.append(memory)
        else:
            speaker_2_memories.append(memory)

    if not speaker_1_memories:
        speaker_1_memories = ["[NO RETRIEVED MEMORIES]"]
    if not speaker_2_memories:
        speaker_2_memories = ["[NO RETRIEVED MEMORIES]"]

    return speaker_1, speaker_1_memories, speaker_2, speaker_2_memories


def build_mem0_official_messages(
    qa_pair: QuestionAnswerPair,
    retrieved_memories: list[MemoryEntry],
) -> list[dict[str, str]]:
    """Build a system-only message list aligned with upstream Mem0 LoCoMo evaluation."""
    prompt = get_mem0_qa_prompt()
    speaker_1, speaker_1_memories, speaker_2, speaker_2_memories = _split_memories_by_speaker(
        qa_pair,
        retrieved_memories,
    )

    rendered_prompt = prompt.substitute(
        speaker_1_user_id=speaker_1,
        speaker_1_memories=json.dumps(speaker_1_memories, indent=4, ensure_ascii=False),
        speaker_2_user_id=speaker_2,
        speaker_2_memories=json.dumps(speaker_2_memories, indent=4, ensure_ascii=False),
        question=qa_pair.question,
    )

    return [{"role": "system", "content": rendered_prompt}]
