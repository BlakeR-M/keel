"""Hybrid retrieval: BM25 and vector search, ACL-filtered before fusion, reciprocal rank fusion, then
an optional cross-encoder rerank.

Permission filtering happens before generation, never after: both candidate lists are filtered by the
user's ACL tags at the store, and a second check here drops anything the store let through.
Quarantined chunks never reach the candidate set.

Scores on returned hits are relevance estimates in [0, 1], the scale `settings.min_relevance` is set
against. With a reranker, the score is the sigmoid of the cross-encoder logit. Without one, it is the
RRF score normalised by the best score a chunk ranked first in every list would get, which says how
well ranked a chunk was rather than how relevant it is; keep `rerank` on for a meaningful refusal gate.
"""

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

from keel.config import Settings
from keel.providers.base import ChunkHit, EmbeddingProvider, Reranker, VectorIndex
from keel.retrieval.bm25 import fts_search

RRF_K = 60


@dataclass
class RetrievedChunk(ChunkHit):
    """A `ChunkHit` with the fusion details kept alongside the final score."""

    fused: float = 0.0  # raw reciprocal rank fusion score
    bm25_rank: int | None = None  # 1-based rank in the BM25 list, None when absent
    vector_rank: int | None = None  # 1-based rank in the vector list, None when absent
    bm25_score: float | None = None  # negated bm25() rank from FTS5
    vector_score: float | None = None  # cosine similarity
    rerank_score: float | None = None  # raw reranker output when reranking ran


def rrf_fuse(ranked_lists: Sequence[Sequence[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal rank fusion: each list contributes 1 / (k + rank) for every id it holds."""
    scores: dict[int, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] += 1.0 / (k + rank)
    return dict(scores)


def rrf_order(ranked_lists: Sequence[Sequence[int]], k: int = RRF_K) -> list[int]:
    """Ids ordered by fused score, highest first; ties break on the smaller id."""
    scores = rrf_fuse(ranked_lists, k)
    return sorted(scores, key=lambda item: (-scores[item], item))


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    ex = math.exp(value)
    return ex / (1.0 + ex)


def max_score(hits: Iterable[ChunkHit]) -> float:
    """Best relevance score among hits, 0.0 when there are none. Compare with `settings.min_relevance`."""
    return max((float(h.score) for h in hits), default=0.0)


def allowed(hit: ChunkHit, user_tags: frozenset[str]) -> bool:
    """A user may see a chunk when the two tag sets share at least one tag and the chunk is not quarantined."""
    return not hit.quarantined and not user_tags.isdisjoint(hit.acl_tags)


class Retriever:
    """Hybrid retriever over one SQLite store."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        settings: Settings,
        embedder: EmbeddingProvider,
        index: VectorIndex,
        reranker: Reranker | None = None,
    ) -> None:
        self.conn = conn
        self.settings = settings
        self.embedder = embedder
        self.index = index
        self.reranker = reranker

    def retrieve(
        self, query: str, user_tags: Sequence[str] | None = None, k: int | None = None
    ) -> list[RetrievedChunk]:
        """Top-k chunks the user is entitled to, best first. `user_tags` None means an anonymous caller
        holding only the `public` tag; an empty list matches nothing."""
        tags = frozenset(user_tags) if user_tags is not None else frozenset({"public"})
        if not tags:
            return []
        final_k = k if k is not None else self.settings.top_k_final
        if final_k <= 0 or not query or not query.strip():
            return []
        tag_list = sorted(tags)

        bm25_hits = [
            h for h in fts_search(self.conn, query, self.settings.top_k_bm25, tag_list) if allowed(h, tags)
        ]
        query_vector = self.embedder.embed_query(query)
        vector_hits = [
            h
            for h in self.index.search(query_vector, self.settings.top_k_vector, tag_list)
            if allowed(h, tags)
        ]

        candidates = self._fuse(bm25_hits, vector_hits)
        if not candidates:
            return []

        if self.reranker is not None and self.settings.rerank:
            reranked = self.reranker.rerank(query, candidates)
            scored = [
                replace(h, rerank_score=float(h.score), score=sigmoid(float(h.score))) for h in reranked
            ]
            scored.sort(key=lambda h: (-h.score, h.chunk_id))
        else:
            best = len([lst for lst in (bm25_hits, vector_hits) if lst]) / (RRF_K + 1)
            scored = [replace(h, score=h.fused / best if best > 0 else 0.0) for h in candidates]
        return scored[:final_k]

    def _fuse(self, bm25_hits: list[ChunkHit], vector_hits: list[ChunkHit]) -> list[RetrievedChunk]:
        by_id: dict[int, ChunkHit] = {}
        bm25_pos: dict[int, tuple[int, float]] = {}
        vector_pos: dict[int, tuple[int, float]] = {}
        for rank, hit in enumerate(bm25_hits, start=1):
            bm25_pos.setdefault(hit.chunk_id, (rank, hit.score))
            by_id.setdefault(hit.chunk_id, hit)
        for rank, hit in enumerate(vector_hits, start=1):
            vector_pos.setdefault(hit.chunk_id, (rank, hit.score))
            by_id.setdefault(hit.chunk_id, hit)
        fused = rrf_fuse([[h.chunk_id for h in bm25_hits], [h.chunk_id for h in vector_hits]])
        ordered = sorted(fused, key=lambda cid: (-fused[cid], cid))
        results: list[RetrievedChunk] = []
        for cid in ordered:
            base = by_id[cid]
            bm = bm25_pos.get(cid)
            vec = vector_pos.get(cid)
            results.append(
                RetrievedChunk(
                    chunk_id=base.chunk_id,
                    score=fused[cid],
                    text=base.text,
                    document_id=base.document_id,
                    source=base.source,
                    title=base.title,
                    heading=base.heading,
                    page=base.page,
                    acl_tags=list(base.acl_tags),
                    quarantined=base.quarantined,
                    fused=fused[cid],
                    bm25_rank=bm[0] if bm else None,
                    vector_rank=vec[0] if vec else None,
                    bm25_score=bm[1] if bm else None,
                    vector_score=vec[1] if vec else None,
                )
            )
        return results


def build_local_retriever(conn: sqlite3.Connection, settings: Settings) -> Retriever:
    """Wire the local profile: fastembed embeddings, the SQLite cosine index and the fastembed
    cross-encoder (left out when `settings.rerank` is off)."""
    from keel.providers.local_embed import FastembedEmbeddings
    from keel.providers.local_index import SqliteVectorIndex
    from keel.providers.local_rerank import FastembedReranker

    embedder = FastembedEmbeddings(settings.local_embed_model, local_files_only=settings.airgap)
    index = SqliteVectorIndex(conn, embed_model=embedder.name)
    reranker = (
        FastembedReranker(settings.local_rerank_model, local_files_only=settings.airgap)
        if settings.rerank
        else None
    )
    return Retriever(conn, settings, embedder, index, reranker)
