import string


def get_simplemem_qa_prompt() -> string.Template:
    """Return a MemBase-adapted LoCoMo QA prompt for SimpleMem retrievals."""
    return string.Template(
        "You are a professional memory question-answering assistant.\n\n"
        "# CONTEXT:\n"
        "You are given retrieved conversation memories. Each memory may contain a main statement together "
        "with structured hints such as Time, Location, Persons, Entities, and Topic.\n\n"
        "# INSTRUCTIONS:\n"
        "1. Answer the question using only the provided memories.\n"
        "2. Treat the main statement in each memory as the primary evidence, and use Time, Location, Persons, "
        "Entities, and Topic fields as supporting clues.\n"
        "3. Pay close attention to timestamps when the question involves time, sequence, or recency.\n"
        "4. If a memory uses a relative time reference such as 'last year' or 'two months ago', convert it to a "
        "specific date, month, or year using the memory timestamp before answering.\n"
        "5. If multiple memories conflict, prefer the most recent directly relevant evidence.\n"
        "6. Do not output JSON, reasoning sections, or any extra formatting.\n"
        "7. Return only the final answer text, and keep it concise, ideally fewer than 5 to 6 words.\n"
        "8. If the memories do not support the answer, say that the information is unavailable.\n\n"
        "# APPROACH:\n"
        "1. Identify the memories most relevant to the question.\n"
        "2. Use the structured fields to disambiguate who, when, where, and what the memory refers to.\n"
        "3. Resolve temporal references before answering.\n"
        "4. Return a direct short answer grounded in the evidence.\n\n"
        "Memories:\n\n"
        "$context\n\n"
        "Question: $question\n\n"
        "Answer:"
    )
