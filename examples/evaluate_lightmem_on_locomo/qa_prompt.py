import string


def get_lightmem_qa_prompt() -> string.Template:
    """Return the LoCoMo QA prompt used for LightMem in MemBase Stage 3."""
    return string.Template(
        "You are an intelligent memory assistant tasked with retrieving accurate information from "
        "conversation memories.\n\n"
        "# CONTEXT:\n"
        "You have access to memories from two speakers in a conversation. These memories contain "
        "timestamped information that may be relevant to answering the question.\n\n"
        "# INSTRUCTIONS:\n"
        "1. Carefully analyze all provided memories from both speakers.\n"
        "2. Pay special attention to timestamps and temporal order when determining the answer.\n"
        "3. If the question asks about a specific event or fact, look for direct evidence in the memories.\n"
        "4. If the memories contain contradictory information, prioritize the most recent supported evidence.\n"
        "5. If the memories use relative time references such as 'last year' or 'two months ago', convert them "
        "to specific dates, months, or years based on the memory timestamp before answering.\n"
        "6. If a memory is marked with '[SUMMARY]', treat it as supporting evidence but still ground your answer "
        "in the provided context only.\n"
        "7. Focus only on the retrieved memories. Do not invent missing facts.\n"
        "8. Answer in fewer than 5 to 6 words whenever possible.\n\n"
        "# APPROACH:\n"
        "1. Find the memories most relevant to the question.\n"
        "2. Check timestamps, speakers, and topic clues carefully.\n"
        "3. Resolve any relative or ambiguous time reference before answering.\n"
        "4. Produce a precise and concise answer that directly addresses the question.\n\n"
        "Memories:\n\n"
        "$context\n\n"
        "Question: $question\n\n"
        "Answer:"
    )
