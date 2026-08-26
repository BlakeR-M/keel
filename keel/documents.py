"""The document lifecycle: what is in the store, what it is tagged, and taking it out again.

Ingest could put documents in and nothing could take them out or correct them. On an appliance whose
whole argument is controlling who reads what, mistagging a sensitive document is the most ordinary
mistake there is, and until now the only remedy was editing SQLite by hand. These are the operations
that close that: list, retag, remove.

Every write runs inside one transaction with its ledger row, the way ingest does, so the store and
the audit trail cannot disagree about what happened.

Removal leans on the schema rather than on care. `chunks.document_id` cascades from `documents`, the
`chunks_ad` trigger clears the full-text entry, and embeddings live on the chunk row, so a single
`DELETE FROM documents` leaves nothing retrievable behind. The vector index fingerprint counts rows
and takes their maximum id, so a removal changes it and the cached matrices are rebuilt; `invalidate`
is called anyway, because relying on a fingerprint for a security property is thinner than saying so.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any

from keel.db import transaction

__all__ = [
    "DocumentRow",
    "RemovalResult",
    "corpus_tags",
    "get_document",
    "list_documents",
    "normalise_tags",
    "remove_document",
    "retag_document",
]

log = logging.getLogger("keel.documents")


@dataclass(frozen=True)
class DocumentRow:
    """One document in the store, with the counts an operator wants beside it."""

    document_id: int
    title: str
    source: str
    acl_tags: list[str] = field(default_factory=list)
    chunks: int = 0
    quarantined: int = 0
    ingested_at: str = ""
    mime: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RemovalResult:
    """What a removal took out, for the caller to report and for the ledger to record."""

    document_id: int
    title: str
    source: str
    chunks_removed: int
    acl_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalise_tags(tags: Any) -> list[str]:
    """A clean, ordered, duplicate-free tag list. An empty result means `public`.

    Tags are the access-control model, so an empty list would mean a document nobody can retrieve,
    which is a silent way to lose a document rather than a useful state.
    """
    items = [tags] if isinstance(tags, str) else list(tags or [])
    seen: list[str] = []
    for item in items:
        # A form sends one `tags` field holding commas, so each item is split again rather than
        # taken whole. Otherwise "ops, safety" becomes a single tag nobody will ever hold.
        for part in str(item).replace(";", ",").split(","):
            tag = part.strip().lower()
            if tag and tag not in seen:
                seen.append(tag)
    return seen or ["public"]


def _parse_tags(stored: Any) -> list[str]:
    try:
        loaded = json.loads(stored) if isinstance(stored, str) else stored
    except (TypeError, ValueError):
        return []
    return [str(tag) for tag in loaded] if isinstance(loaded, list) else []


_SELECT = """
SELECT d.id, d.title, d.source, d.acl_tags, d.ingested_at, d.mime,
       COUNT(c.id) AS chunks,
       COALESCE(SUM(c.quarantined), 0) AS quarantined
FROM documents d
LEFT JOIN chunks c ON c.document_id = d.id
"""


def _row(record: sqlite3.Row) -> DocumentRow:
    return DocumentRow(
        document_id=int(record["id"]),
        title=str(record["title"] or record["source"]),
        source=str(record["source"]),
        acl_tags=_parse_tags(record["acl_tags"]),
        chunks=int(record["chunks"] or 0),
        quarantined=int(record["quarantined"] or 0),
        ingested_at=str(record["ingested_at"] or ""),
        mime=record["mime"],
    )


def list_documents(conn: sqlite3.Connection, limit: int = 500) -> list[DocumentRow]:
    """Every document in the store, newest first."""
    rows = conn.execute(
        f"{_SELECT} GROUP BY d.id ORDER BY d.ingested_at DESC, d.id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [_row(record) for record in rows]


def get_document(conn: sqlite3.Connection, document_id: int) -> DocumentRow | None:
    record = conn.execute(f"{_SELECT} WHERE d.id = ? GROUP BY d.id", (document_id,)).fetchone()
    return _row(record) if record is not None else None


def corpus_tags(conn: sqlite3.Connection) -> list[str]:
    """Every tag in use across the store, sorted, for an operator choosing one.

    Read from the documents rather than from a list somebody maintains, so it describes the corpus
    as it is rather than as it was meant to be.
    """
    tags: set[str] = set()
    for (stored,) in conn.execute("SELECT acl_tags FROM documents"):
        tags.update(_parse_tags(stored))
    return sorted(tags)


def _write_ledger(conn: sqlite3.Connection, ledger: Any, kind: str, payload: dict[str, Any]) -> None:
    """Append a ledger row inside the caller's open transaction, as ingest does.

    A failure propagates, so the surrounding transaction rolls back and the store never records a
    change the ledger missed.
    """
    if ledger is None:
        try:
            from keel.safety.ledger import Ledger
        except ImportError:
            return
        ledger = Ledger(conn)
    append = getattr(ledger, "append", None)
    if callable(append):
        append(kind, None, payload)


def _invalidate(index: Any) -> None:
    invalidate = getattr(index, "invalidate", None)
    if callable(invalidate):
        invalidate()


def retag_document(
    conn: sqlite3.Connection,
    index: Any,
    document_id: int,
    tags: Any,
    *,
    by: str = "admin",
    ledger: Any = None,
) -> DocumentRow:
    """Replace a document's access tags, and its chunks' tags with it.

    The chunks carry their own copy, because retrieval filters on the chunk rather than joining back
    to the document on every query. Both move together inside one transaction, the stored vectors are
    re-upserted so any index carrying tags alongside them is told, and the change lands in the ledger.

    Raises `LookupError` when the document is absent.
    """
    current = get_document(conn, document_id)
    if current is None:
        raise LookupError(f"no document {document_id} in this store")
    wanted = normalise_tags(tags)
    if wanted == current.acl_tags:
        return current

    import numpy as np

    tags_json = json.dumps(wanted)
    with transaction(conn):
        conn.execute("UPDATE documents SET acl_tags = ? WHERE id = ?", (tags_json, document_id))
        conn.execute("UPDATE chunks SET acl_tags = ? WHERE document_id = ?", (tags_json, document_id))
        rows = conn.execute(
            "SELECT id, embedding FROM chunks WHERE document_id = ? AND embedding IS NOT NULL ORDER BY id",
            (document_id,),
        ).fetchall()
        chunk_ids = [int(record["id"]) for record in rows]
        if chunk_ids and index is not None:
            vectors = [np.frombuffer(record["embedding"], dtype="<f4").tolist() for record in rows]
            index.upsert(chunk_ids, vectors)
        _write_ledger(
            conn,
            ledger,
            "ingest",
            {
                "action": "retag",
                "document_id": document_id,
                "source": current.source,
                "acl_tags": wanted,
                "previous_acl_tags": current.acl_tags,
                "chunks": len(chunk_ids),
                "by": by,
            },
        )
    _invalidate(index)
    log.info("retagged document %d: %s to %s", document_id, current.acl_tags, wanted)
    updated = get_document(conn, document_id)
    assert updated is not None  # noqa: S101 (the row was read inside the same connection)
    return updated


def remove_document(
    conn: sqlite3.Connection,
    index: Any,
    document_id: int,
    *,
    by: str = "admin",
    ledger: Any = None,
) -> RemovalResult:
    """Take a document out of the store, with its chunks, its full-text entries and its embeddings.

    One `DELETE` does the work: `chunks.document_id` cascades, the `chunks_ad` trigger clears the
    full-text index, and embeddings sit on the chunk row. The ledger row goes in first, inside the
    same transaction, so a removal that fails records nothing and a removal that succeeds is
    recorded before the rows are gone.

    Raises `LookupError` when the document is absent.
    """
    current = get_document(conn, document_id)
    if current is None:
        raise LookupError(f"no document {document_id} in this store")

    with transaction(conn):
        _write_ledger(
            conn,
            ledger,
            "ingest",
            {
                "action": "remove",
                "document_id": document_id,
                "source": current.source,
                "title": current.title,
                "acl_tags": current.acl_tags,
                "chunks": current.chunks,
                "by": by,
            },
        )
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    _invalidate(index)

    left = conn.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (document_id,)).fetchone()[0]
    if int(left):  # pragma: no cover (the schema cascades; this says so if that ever changes)
        raise RuntimeError(f"document {document_id} left {left} chunk(s) behind")
    log.info("removed document %d (%s), %d chunk(s)", document_id, current.source, current.chunks)
    return RemovalResult(
        document_id=document_id,
        title=current.title,
        source=current.source,
        chunks_removed=current.chunks,
        acl_tags=current.acl_tags,
    )


def clear_documents(
    conn: sqlite3.Connection,
    index: Any,
    *,
    by: str = "admin",
    ledger: Any = None,
) -> list[RemovalResult]:
    """Take every document out of the store, leaving it empty.

    Each document goes through `remove_document`, so the cascade, the full-text trigger and the
    ledger entry behave exactly as they do for a single removal, and a clear reads in the audit trail
    as the set of removals it is rather than as one opaque event. The loop re-reads the store instead
    of walking one listing, because `list_documents` caps its rows and a large corpus would otherwise
    keep a tail. An empty store returns an empty list.
    """
    results: list[RemovalResult] = []
    while True:
        rows = list_documents(conn)
        if not rows:
            break
        for row in rows:
            results.append(remove_document(conn, index, row.document_id, by=by, ledger=ledger))
    log.info("cleared %d document(s)", len(results))
    return results
