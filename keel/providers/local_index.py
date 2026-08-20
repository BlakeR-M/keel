"""SQLite-backed vector index. Implements `VectorIndex` over the `chunks` table.

Embeddings live in `chunks.embedding` as float32 little-endian blobs. Search loads every
non-quarantined embedding into a numpy matrix once, keeps it in memory, and scores by cosine
similarity. The cache refreshes after `upsert()` and whenever the table's fingerprint changes (row
count, highest id, the quarantine flags and the ACL tag columns). The rows a search returns are read
fresh from SQLite, and their current tags and quarantine flag are checked again: when the cached mask
proves stale (a tag narrowed or a chunk quarantined behind the cache's back) the cache is dropped and
the search runs once more over fresh data, so a caller never receives a chunk its tags no longer cover.

Ceiling: brute-force cosine is comfortable to roughly one hundred thousand chunks (384-d float32
is 1.5 KB per chunk, so 100k chunks hold 150 MB and a query is one matrix-vector product). Beyond
that, swap this class for a sqlite-vec or pgvector index behind the same `VectorIndex` protocol.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass

import numpy as np

from keel.db import transaction
from keel.providers.base import ChunkHit

log = logging.getLogger(__name__)

HIT_COLUMNS = (
    "c.id AS chunk_id, c.text, c.document_id, c.heading, c.page, c.acl_tags, c.quarantined, d.source, d.title"
)
HIT_FROM = "chunks c JOIN documents d ON d.id = c.document_id"


def parse_tags(raw: str | None) -> list[str]:
    """Decode a JSON ACL tag list from a chunk or document row."""
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [str(tag) for tag in value] if isinstance(value, list) else []


def hit_from_row(row: sqlite3.Row, score: float) -> ChunkHit:
    """Build a `ChunkHit` from a row selected with `HIT_COLUMNS` from `HIT_FROM`."""
    return ChunkHit(
        chunk_id=int(row["chunk_id"]),
        score=float(score),
        text=row["text"],
        document_id=int(row["document_id"]),
        source=row["source"],
        title=row["title"],
        heading=row["heading"],
        page=row["page"],
        acl_tags=parse_tags(row["acl_tags"]),
        quarantined=bool(row["quarantined"]),
    )


def fetch_hits(conn: sqlite3.Connection, scored_ids: list[tuple[int, float]]) -> list[ChunkHit]:
    """Load chunk rows for (chunk_id, score) pairs, preserving the given order."""
    if not scored_ids:
        return []
    ids = [cid for cid, _ in scored_ids]
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT {HIT_COLUMNS} FROM {HIT_FROM} WHERE c.id IN ({placeholders})", ids
    ).fetchall()
    by_id = {int(row["chunk_id"]): row for row in rows}
    return [hit_from_row(by_id[cid], score) for cid, score in scored_ids if cid in by_id]


@dataclass
class _Matrix:
    ids: np.ndarray  # int64, shape (n,)
    vectors: np.ndarray  # float32, shape (n, dim), rows normalised to unit length
    tags: list[frozenset[str]]


class SqliteVectorIndex:
    """Cosine-similarity search over chunk embeddings stored in SQLite."""

    name = "sqlite-cosine"

    def __init__(self, conn: sqlite3.Connection, *, embed_model: str | None = None) -> None:
        self.conn = conn
        self.embed_model = embed_model
        self._matrices: dict[int, _Matrix] | None = None
        self._fingerprint: tuple | None = None

    # ------------------------------------------------------------------ writes

    def upsert(self, chunk_ids: list[int], vectors: list[list[float]]) -> None:
        """Store one embedding per existing chunk row and drop the in-memory cache."""
        if len(chunk_ids) != len(vectors):
            raise ValueError(f"{len(chunk_ids)} chunk ids and {len(vectors)} vectors; the counts must match")
        if not chunk_ids:
            return
        dims = {len(vec) for vec in vectors}
        if len(dims) != 1:
            raise ValueError(f"vectors of mixed dimensions {sorted(dims)}; one embedding model per upsert")
        rows = [
            (np.asarray(vec, dtype="<f4").tobytes(), self.embed_model, int(cid))
            for cid, vec in zip(chunk_ids, vectors, strict=True)
        ]
        if self.conn.in_transaction:
            self._write(rows)
        else:
            with transaction(self.conn):
                self._write(rows)
        self._matrices = None
        self._fingerprint = None

    def _write(self, rows: list[tuple[bytes, str | None, int]]) -> None:
        cursor = self.conn.executemany("UPDATE chunks SET embedding = ?, embed_model = ? WHERE id = ?", rows)
        if cursor.rowcount != len(rows):
            wanted = {cid for _, _, cid in rows}
            placeholders = ",".join("?" for _ in wanted)
            present = {
                int(r[0])
                for r in self.conn.execute(
                    f"SELECT id FROM chunks WHERE id IN ({placeholders})", list(wanted)
                )
            }
            missing = sorted(wanted - present)
            raise KeyError(f"chunk ids without a row in the chunks table: {missing}")

    def invalidate(self) -> None:
        """Drop the cached matrix so the next search reloads from SQLite."""
        self._matrices = None
        self._fingerprint = None

    # ------------------------------------------------------------------ reads

    def count(self) -> int:
        """Number of chunks that carry an embedding."""
        return int(self.conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0])

    def search(self, vector: list[float], k: int, allowed_tags: list[str] | None = None) -> list[ChunkHit]:
        """Top-k chunks by cosine similarity. Quarantined chunks are never candidates. With
        `allowed_tags`, only chunks sharing at least one tag are scored; an empty list matches nothing;
        None applies no ACL filter."""
        if k <= 0:
            return []
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query))
        query = query / norm if norm > 0 else query
        allowed = None if allowed_tags is None else frozenset(allowed_tags)
        if allowed is not None and not allowed:
            return []
        hits, stale = self._search_cached(query, k, allowed)
        if stale:
            # The rows came back with tags or a quarantine flag the cached mask no longer matches:
            # reload from SQLite and answer from fresh data.
            self.invalidate()
            hits, _ = self._search_cached(query, k, allowed)
        return hits

    def _search_cached(
        self, query: np.ndarray, k: int, allowed: frozenset[str] | None
    ) -> tuple[list[ChunkHit], bool]:
        """Score against the cached matrix; return (hits the current rows still permit, cache was stale)."""
        matrix = self._matrices_by_dim().get(int(query.shape[0]))
        if matrix is None or matrix.ids.size == 0:
            return [], False
        if allowed is None:
            candidates = np.arange(matrix.ids.size)
        else:
            mask = np.fromiter(
                (not tags.isdisjoint(allowed) for tags in matrix.tags), dtype=bool, count=len(matrix.tags)
            )
            candidates = np.flatnonzero(mask)
            if candidates.size == 0:
                return [], False
        scores = matrix.vectors[candidates] @ query
        take = min(k, candidates.size)
        top = (
            np.argpartition(-scores, take - 1)[:take]
            if take < candidates.size
            else np.arange(candidates.size)
        )
        order = top[np.lexsort((matrix.ids[candidates[top]], -scores[top]))]
        scored = [(int(matrix.ids[candidates[i]]), float(scores[i])) for i in order]
        hits = fetch_hits(self.conn, scored)
        fresh = [
            h for h in hits if not h.quarantined and (allowed is None or not allowed.isdisjoint(h.acl_tags))
        ]
        return fresh, len(fresh) != len(hits)

    # ------------------------------------------------------------------ cache

    def _current_fingerprint(self) -> tuple:
        row = self.conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(id), 0), COALESCE(SUM(quarantined), 0), "
            "COALESCE(SUM(id * quarantined), 0), COALESCE(SUM(length(acl_tags)), 0), "
            "COALESCE(SUM(id * length(acl_tags)), 0) "
            "FROM chunks WHERE embedding IS NOT NULL"
        ).fetchone()
        return tuple(int(v) for v in row)

    def _matrices_by_dim(self) -> dict[int, _Matrix]:
        fingerprint = self._current_fingerprint()
        if self._matrices is None or fingerprint != self._fingerprint:
            self._matrices = self._load()
            self._fingerprint = fingerprint
        return self._matrices

    def _load(self) -> dict[int, _Matrix]:
        rows = self.conn.execute(
            "SELECT id, embedding, acl_tags FROM chunks WHERE quarantined = 0 AND embedding IS NOT NULL ORDER BY id"
        ).fetchall()
        grouped: dict[int, list[sqlite3.Row]] = {}
        for row in rows:
            blob = row["embedding"]
            if len(blob) % 4 != 0 or len(blob) == 0:
                log.warning("chunk %s has an embedding blob of %d bytes; skipped", row["id"], len(blob))
                continue
            grouped.setdefault(len(blob) // 4, []).append(row)
        if len(grouped) > 1:
            log.warning(
                "chunk embeddings have mixed dimensions %s; each query uses the matching set", sorted(grouped)
            )
        matrices: dict[int, _Matrix] = {}
        for dim, group in grouped.items():
            vectors = np.frombuffer(b"".join(r["embedding"] for r in group), dtype="<f4").reshape(
                len(group), dim
            )
            vectors = vectors.astype(np.float32, copy=True)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors /= norms
            matrices[dim] = _Matrix(
                ids=np.fromiter((int(r["id"]) for r in group), dtype=np.int64, count=len(group)),
                vectors=vectors,
                tags=[frozenset(parse_tags(r["acl_tags"])) for r in group],
            )
        return matrices
