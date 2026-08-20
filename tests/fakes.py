"""Fakes shared across test lanes: a scripted LLM, a canned retriever, and in-memory ledger and log
sinks. Nothing here touches the network.

Usage:
    llm = FakeLLM(["plain reply", tool_call_reply("calculator", {"expression": "2+2"}), "final"])
    retriever = FakeRetriever([make_hit(1, "Band 2 needs three quotes.", score=0.9)])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from keel.providers.base import ChatMessage, ChatResult, ChunkHit, ToolSpec


def make_hit(
    chunk_id: int,
    text: str,
    *,
    score: float = 0.9,
    source: str = "fixtures/corpus/northbank-council-procurement.md",
    title: str | None = "Northbank City Council Procurement Guide",
    heading: str | None = None,
    page: int | None = None,
    tags: tuple[str, ...] | list[str] = ("public",),
    document_id: int = 1,
    quarantined: bool = False,
) -> ChunkHit:
    """A ChunkHit with sensible defaults; override what the test cares about."""
    return ChunkHit(
        chunk_id=chunk_id,
        score=score,
        text=text,
        document_id=document_id,
        source=source,
        title=title,
        heading=heading,
        page=page,
        acl_tags=list(tags),
        quarantined=quarantined,
    )


def tool_call_reply(name: str, arguments: dict[str, Any], call_id: str | None = None, content: str = "") -> ChatResult:
    """A ChatResult in which the model asks for one tool call."""
    return ChatResult(
        content=content,
        tool_calls=[{"id": call_id or f"call_{name}", "name": name, "arguments": dict(arguments)}],
        prompt_tokens=10,
        output_tokens=5,
        model="fake-model",
    )


class FakeLLM:
    """LLMProvider that replays scripted responses in order and records every call.

    Each scripted item may be a ChatResult or a plain string (returned as content with no tool
    calls). Running out of script raises AssertionError so a runaway loop fails loudly.
    """

    name = "fake"

    def __init__(self, responses: list[ChatResult | str] | None = None) -> None:
        self.responses: list[ChatResult | str] = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> ChatResult:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": list(tools) if tools else [],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_schema": json_schema,
            }
        )
        if not self.responses:
            raise AssertionError("FakeLLM has no scripted response left for this call")
        scripted = self.responses.pop(0)
        if isinstance(scripted, str):
            return ChatResult(content=scripted, prompt_tokens=10, output_tokens=5, model="fake-model")
        return scripted

    def healthy(self) -> bool:
        return True


class FakeRetriever:
    """Retriever that returns the given hits, filtered by ACL tags, best score first."""

    def __init__(self, hits: list[ChunkHit] | None = None) -> None:
        self.hits: list[ChunkHit] = list(hits or [])
        self.calls: list[dict[str, Any]] = []

    def retrieve(self, query: str, user_tags: list[str] | None = None, k: int | None = None) -> list[ChunkHit]:
        self.calls.append({"query": query, "user_tags": user_tags, "k": k})
        visible = [h for h in self.hits if user_tags is None or set(h.acl_tags) & set(user_tags)]
        ranked = sorted(visible, key=lambda h: h.score, reverse=True)
        return ranked if k is None else ranked[:k]


@dataclass
class FakeLedger:
    """Collects ledger entries as (kind, request_id, payload)."""

    entries: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)

    def append(self, kind: str, request_id: str, payload: dict[str, Any]) -> None:
        self.entries.append((kind, request_id, payload))

    def kinds(self) -> list[str]:
        return [kind for kind, _, _ in self.entries]


@dataclass
class FakeLog:
    """Collects inference-log records as dicts."""

    records: list[dict[str, Any]] = field(default_factory=list)

    def record(self, **fields: Any) -> None:
        self.records.append(dict(fields))
