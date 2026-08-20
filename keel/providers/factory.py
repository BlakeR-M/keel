"""Builds the whole application context from Settings. This is the one place the deploy profile
(local | azure | aws) is decided; everything above it (CLI, web, evals) asks for `build_context()`
and never imports a concrete provider.

Order matters: air-gap enforcement is installed first, then the store, then providers.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from keel.agent.approvals import ApprovalQueue
from keel.agent.loop import AgentLoop
from keel.agent.policy import Policy
from keel.agent.tools import ToolRegistry, default_registry
from keel.airgap import enforce_from_settings
from keel.answer.engine import AnswerEngine
from keel.config import Settings, get_settings
from keel.db import connect
from keel.observe.log import InferenceLog
from keel.providers.base import EmbeddingProvider, LLMProvider, Reranker, VectorIndex
from keel.retrieval.hybrid import Retriever
from keel.safety.injection import screen as injection_screen
from keel.safety.ledger import Ledger

DEFAULT_POLICY: dict[str, Any] = {
    "allowed_tools": ["search_docs", "calculator", "sql_readonly", "create_ticket"],
    "write_tools_require_approval": True,
    "tool_arg_rules": {"sql_readonly": {"max_rows": 50}},
    "max_tool_calls_per_request": 6,
}


@dataclass
class AppContext:
    settings: Settings
    conn: sqlite3.Connection
    llm: LLMProvider
    embedder: EmbeddingProvider
    index: VectorIndex
    reranker: Reranker | None
    retriever: Retriever
    ledger: Ledger
    log: InferenceLog
    registry: ToolRegistry
    policy: Policy
    approvals: ApprovalQueue
    answer_engine: AnswerEngine
    agent_loop: AgentLoop
    profile: str
    extras: dict[str, Any] = field(default_factory=dict)

    def screen(self, text: str) -> tuple[bool, str | None]:
        """Injection screen used at ingest time (heuristic layer; the judge is opt-in via `keel ingest --judge`)."""
        result = injection_screen(text)
        return bool(result.quarantined), result.reason

    def screen_with_judge(self, text: str) -> tuple[bool, str | None]:
        """Heuristics plus the LLM judge; slower, used when `--judge` is passed at ingest."""
        from keel.safety.injection import screen_with_judge

        result = screen_with_judge(text, self.llm)
        return bool(result.quarantined), result.reason

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


def _load_policy(settings: Settings) -> dict[str, Any]:
    """Policy file at <data_dir>/policy.yaml wins; otherwise the default policy."""
    path = Path(settings.data_dir) / "policy.yaml"
    if path.exists():
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            return loaded
    return dict(DEFAULT_POLICY)


def _local_providers(settings: Settings, conn: sqlite3.Connection):
    from keel.providers.local_embed import FastembedEmbeddings
    from keel.providers.local_index import SqliteVectorIndex
    from keel.providers.local_llm import OpenAICompatibleLLM
    from keel.providers.local_rerank import FastembedReranker

    llm = OpenAICompatibleLLM(
        base_url=settings.local_llm_base_url,
        model=settings.local_llm_model,
        api_key=settings.local_llm_api_key,
        timeout=settings.local_llm_timeout,
        name="llama-server",
    )
    embedder = FastembedEmbeddings(settings.local_embed_model)
    index = SqliteVectorIndex(conn, embed_model=settings.local_embed_model)
    reranker = FastembedReranker(settings.local_rerank_model) if settings.rerank else None
    return llm, embedder, index, reranker


def _azure_providers(settings: Settings, conn: sqlite3.Connection):
    from keel.providers.azure import AzureOpenAIChat, AzureOpenAIEmbeddings, AzureSearchIndex
    from keel.providers.local_index import fetch_hits  # noqa: F401  (row loader helper lives here)
    from keel.providers.local_rerank import FastembedReranker

    if not settings.azure_openai_endpoint or not settings.azure_search_endpoint:
        raise RuntimeError(
            "Azure profile needs KEEL_AZURE_OPENAI_ENDPOINT and KEEL_AZURE_SEARCH_ENDPOINT. "
            "See deploy/azure/README.md."
        )
    llm = AzureOpenAIChat(
        settings.azure_openai_endpoint,
        settings.azure_openai_chat_deployment,
        settings.azure_openai_api_version,
    )
    embedder = AzureOpenAIEmbeddings(
        settings.azure_openai_endpoint,
        settings.azure_openai_embed_deployment,
        settings.azure_openai_api_version,
    )

    def row_loader(chunk_ids: list[int]):
        marks = ",".join("?" for _ in chunk_ids)
        rows = conn.execute(
            f"""SELECT c.id AS chunk_id, c.document_id, c.text, c.heading, c.page, c.acl_tags,
                       c.quarantined, c.embedding, d.source, d.title
                FROM chunks c JOIN documents d ON d.id = c.document_id
                WHERE c.id IN ({marks})""",
            tuple(chunk_ids),
        ).fetchall()
        return [dict(r) for r in rows]

    index = AzureSearchIndex(
        settings.azure_search_endpoint,
        settings.azure_search_index,
        embedder.dim,
        row_loader=row_loader,
    )
    # Reranking stays local and CPU-only in both profiles; it is cheap and keeps the ACL-filtered
    # candidate set on the appliance.
    reranker = FastembedReranker(settings.local_rerank_model) if settings.rerank else None
    return llm, embedder, index, reranker


def build_context(settings: Settings | None = None, *, conn: sqlite3.Connection | None = None) -> AppContext:
    """Wire the appliance for the configured profile."""
    settings = settings or get_settings()
    enforce_from_settings(settings)
    settings.ensure_dirs()
    conn = conn or connect(settings.db_path)

    if settings.profile == "local":
        llm, embedder, index, reranker = _local_providers(settings, conn)
    elif settings.profile == "azure":
        llm, embedder, index, reranker = _azure_providers(settings, conn)
    elif settings.profile == "aws":
        from keel.providers.aws import BedrockChat  # raises NotImplementedError with the README pointer

        BedrockChat(region="ap-southeast-2", model_id="stub")
        raise RuntimeError("unreachable")
    else:  # pragma: no cover
        raise RuntimeError(f"unknown profile {settings.profile!r}")

    retriever = Retriever(conn, settings, embedder, index, reranker)
    ledger = Ledger(conn)
    log = InferenceLog(conn)
    approvals = ApprovalQueue(conn, ledger=ledger)
    registry = default_registry(
        settings,
        retriever=retriever,
        sql_path=None,
        sql_tables=(),
        http_hosts=(),
    )
    policy = Policy(_load_policy(settings), registry=registry, airgap=settings.airgap)
    answer_engine = AnswerEngine(llm, retriever, settings, ledger=ledger, log=log)
    agent_loop = AgentLoop(llm, registry, policy, approvals, settings, ledger=ledger, log=log)

    return AppContext(
        settings=settings,
        conn=conn,
        llm=llm,
        embedder=embedder,
        index=index,
        reranker=reranker,
        retriever=retriever,
        ledger=ledger,
        log=log,
        registry=registry,
        policy=policy,
        approvals=approvals,
        answer_engine=answer_engine,
        agent_loop=agent_loop,
        profile=settings.profile,
    )
