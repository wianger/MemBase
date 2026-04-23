from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    """A SimpleMem memory unit with structured metadata."""

    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    lossless_restatement: str = Field(
        ...,
        description=(
            "Self-contained fact with resolved references and absolute time "
            "expressions whenever available."
        ),
    )
    keywords: list[str] = Field(default_factory=list)
    timestamp: str | None = None
    location: str | None = None
    persons: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    topic: str | None = None


class Dialogue(BaseModel):
    """A raw dialogue line consumed by SimpleMem."""

    dialogue_id: int
    speaker: str
    content: str
    timestamp: str | None = None
    role: Literal["user", "assistant", "system"] | None = None

    def __str__(self) -> str:
        time_str = f"[{self.timestamp}] " if self.timestamp else ""
        role_str = f" (role: {self.role})" if self.role is not None else ""
        return f"{time_str}{self.speaker}{role_str}: {self.content}"

