"""Provider contracts. Everything above this layer (retrieval, answer, agent, evals) talks only to
these protocols, so the on-premise and Azure profiles are interchangeable at config time.

Implementations: providers/local.py (llama-server + fastembed + SQLite), providers/azure.py
(Azure OpenAI + Azure AI Search, DefaultAzureCredential), providers/aws.py (documented stub).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ChatMessage:
    role: str  # system | user | assistant | tool
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


@dataclass
class ToolSpec:
    """OpenAI-style function tool. `write` tools never execute unattended."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema
    write: bool = False


@dataclass
class ChatResult:
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # [{id, name, arguments(dict)}]
    prompt_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    raw: Any = None


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> ChatResult: ...

    def healthy(self) -> bool: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass
class ChunkHit:
    chunk_id: int
    score: float
    text: str
    document_id: int
    source: str
    title: str | None
    heading: str | None
    page: int | None
    acl_tags: list[str]
    quarantined: bool = False


@runtime_checkable
class VectorIndex(Protocol):
    """Stores chunk embeddings and answers similarity queries. The local implementation reads the
    SQLite chunks table; the Azure implementation is an Azure AI Search index.

    ACL filtering is the caller's job in the local implementation (retrieval/hybrid.py filters before
    generation); the Azure implementation pushes the filter into the search query.
    """

    name: str

    def upsert(self, chunk_ids: list[int], vectors: list[list[float]]) -> None: ...

    def search(self, vector: list[float], k: int, allowed_tags: list[str] | None = None) -> list[ChunkHit]: ...

    def count(self) -> int: ...


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: list[ChunkHit]) -> list[ChunkHit]: ...
