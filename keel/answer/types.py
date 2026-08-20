"""Data types shared by the answer engine, the agent loop and their callers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from keel.providers.base import ChunkHit


@dataclass
class User:
    """The requesting user. `tags` are the ACL tags the user carries; a chunk is visible when
    the user holds at least one of the chunk's tags."""

    user_id: str
    tags: list[str] = field(default_factory=lambda: ["public"])


@dataclass
class Citation:
    """One [n] marker in an answer, resolved to the chunk it points at."""

    n: int
    chunk_id: int
    source: str
    title: str | None
    page: int | None
    heading: str | None
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "title": self.title,
            "page": self.page,
            "heading": self.heading,
            "snippet": self.snippet,
        }


@dataclass
class Answer:
    """The answer engine's result. `text` is prose, or a canonical JSON string in JSON mode
    (with the parsed object under `data`). `refused` is True when the sources hold no answer."""

    text: str
    citations: list[Citation]
    refused: bool
    retrieved: list[ChunkHit]
    prompt_tokens: int
    output_tokens: int
    latency_ms: int
    request_id: str
    data: Any = None
    error: str | None = None


@runtime_checkable
class Retriever(Protocol):
    """What the answer engine and the search_docs tool need from retrieval: hits already filtered
    to what a user carrying `user_tags` may see, best first. Matches keel.retrieval.hybrid.Retriever."""

    def retrieve(self, query: str, user_tags: list[str] | None = None) -> list[ChunkHit]: ...
