"""Retrieval: hybrid BM25 + vector search with ACL filtering before fusion, RRF and reranking."""

from keel.retrieval.bm25 import fts_search, sanitise_query
from keel.retrieval.hybrid import (
    RRF_K,
    RetrievedChunk,
    Retriever,
    build_local_retriever,
    max_score,
    rrf_fuse,
    rrf_order,
)

__all__ = [
    "RRF_K",
    "RetrievedChunk",
    "Retriever",
    "build_local_retriever",
    "fts_search",
    "max_score",
    "rrf_fuse",
    "rrf_order",
    "sanitise_query",
]
