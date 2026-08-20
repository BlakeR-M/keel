"""Ingest pipeline: load, chunk, screen, embed, store.

`ingest_path()` is idempotent by document checksum: a source whose bytes are already in the store is
reported as a duplicate and adds no chunks. When the duplicate arrives with an explicit `acl_tags` that
differs from the stored tags, the document and every chunk are retagged in one transaction and the index
refreshed, so an operator can narrow (or widen) who may read a document by ingesting it again; a
re-ingest without explicit tags leaves the stored tags alone. `ingest_manifest()` reads the
`fixtures/corpus.yaml` format (a `documents:` list of `path`, `title`, `acl_tags`).

An optional `screen(text) -> (quarantined, reason)` callable runs before anything is stored, over three
views of the document: each chunk together with its section heading (the heading rides into every prompt
inside the source label), each pair of adjacent chunks (an instruction split across two sections scores
low in each half and high together), and the document title, which is shown with every passage of the
document. Flagged chunks are written with `quarantined = 1` and stay out of retrieval; a flagged title is
replaced by the file or URL name and the original kept in the document's `meta`. The safety package
supplies the real screen; the default stores every chunk unflagged.

The ledger row for an ingest is written inside the same transaction as the document, so a ledger that
cannot be written rolls the ingest back and no document exists without its audit row. The row names
every chunk the screen held back and why, and a replaced title.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import numpy as np
import yaml

from keel.config import Settings
from keel.db import transaction
from keel.ingest.chunk import Chunk, chunk_document
from keel.ingest.loaders import LoadedDoc, is_url, load
from keel.providers.base import EmbeddingProvider, VectorIndex

log = logging.getLogger(__name__)

ScreenFn = Callable[[str], tuple[bool, str | None]]
DEFAULT_ACL_TAGS: tuple[str, ...] = ("public",)

Flag = tuple[bool, str | None]


@dataclass
class IngestResult:
    """Outcome of ingesting one source."""

    document_id: int
    chunks_added: int
    skipped_duplicate: bool
    source: str
    checksum: str
    title: str | None = None
    quarantined: int = 0
    acl_tags: list[str] = field(default_factory=list)
    tags_updated: bool = False  # a duplicate whose ACL tags were changed to `acl_tags`
    previous_acl_tags: list[str] | None = None  # the tags a retag replaced
    title_flagged: str | None = None  # why the title was replaced by the file name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ingest_path(
    conn: sqlite3.Connection,
    settings: Settings,
    embedder: EmbeddingProvider,
    index: VectorIndex,
    source: str | Path,
    *,
    title: str | None = None,
    acl_tags: Sequence[str] | None = None,
    screen: ScreenFn | None = None,
    meta: dict[str, Any] | None = None,
    ledger: Any = None,
) -> IngestResult:
    """Ingest one file path or URL. Returns a duplicate result, adding nothing, when the same bytes
    were ingested before; a duplicate with explicit `acl_tags` different from the stored ones is
    retagged. `ledger` is an object with `append(kind, request_id, payload)`; by default the safety
    package's ledger records the ingest when that package is available."""
    doc = load(source, settings=settings)
    existing = _find_document(conn, doc.checksum)
    if existing is not None:
        stored_tags = _document_tags(conn, existing)
        if acl_tags is None or _normalise_tags(acl_tags) == stored_tags:
            return IngestResult(
                existing, 0, True, doc.source, doc.checksum, title or doc.title, acl_tags=stored_tags
            )
        return _retag_document(conn, index, existing, stored_tags, _normalise_tags(acl_tags), doc, ledger)

    tags = _normalise_tags(acl_tags)
    doc_title, title_reason = _screen_title(title or doc.title, _fallback_title(doc), screen)
    chunks = chunk_document(doc, settings)
    flags = _screen_chunks(chunks, screen)
    vectors = embedder.embed([chunk.text for chunk in chunks]) if chunks else []
    if len(vectors) != len(chunks):
        raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks")

    document_meta = {
        "kind": doc.kind,
        "pages": len(doc.pages),
        "size_bytes": doc.size_bytes,
        "embed_model": getattr(embedder, "name", None),
        **(meta or {}),
    }
    if title_reason is not None:
        document_meta["title_quarantined"] = {"title": title or doc.title, "reason": title_reason}
    quarantined = [i for i, (flagged, _) in enumerate(flags) if flagged]
    try:
        with transaction(conn):
            document_id = _insert_document(
                conn, doc.source, doc_title, doc.checksum, doc.mime, tags, document_meta
            )
            chunk_ids = _insert_chunks(conn, document_id, chunks, flags, tags)
            if chunk_ids:
                index.upsert(chunk_ids, vectors)
            # One ledger row per ingest. The quarantine details ride on it when the screen flagged
            # anything, so the audit trail names every chunk held back and why.
            payload: dict[str, Any] = {
                "source": doc.source,
                "checksum": doc.checksum,
                "chunks_added": len(chunks),
            }
            if quarantined:
                payload["quarantined"] = len(quarantined)
                payload["quarantined_chunk_ids"] = [chunk_ids[i] for i in quarantined]
                payload["quarantine_reasons"] = [flags[i][1] for i in quarantined]
            if title_reason is not None:
                payload["title_replaced"] = {"title": title or doc.title, "reason": title_reason}
            _write_ledger(conn, ledger, "ingest", payload)
    except sqlite3.IntegrityError:
        existing = _find_document(conn, doc.checksum)
        if existing is None:
            raise
        return IngestResult(
            existing, 0, True, doc.source, doc.checksum, doc_title, acl_tags=_document_tags(conn, existing)
        )

    log.info(
        "ingested %s: document %d, %d chunks, %d quarantined",
        doc.source,
        document_id,
        len(chunks),
        len(quarantined),
    )
    return IngestResult(
        document_id,
        len(chunks),
        False,
        doc.source,
        doc.checksum,
        doc_title,
        len(quarantined),
        acl_tags=tags,
        title_flagged=title_reason,
    )


def ingest_manifest(
    conn: sqlite3.Connection,
    settings: Settings,
    embedder: EmbeddingProvider,
    index: VectorIndex,
    manifest_path: str | Path,
    *,
    screen: ScreenFn | None = None,
    ledger: Any = None,
) -> list[IngestResult]:
    """Ingest every entry of a corpus manifest (`documents:` list of `path` or `url`, `title`, `acl_tags`).
    Relative paths resolve against the manifest's directory, then its parent, then the working directory."""
    manifest = Path(manifest_path)
    data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    entries = data.get("documents") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"{manifest}: expected a top-level 'documents' list")
    results: list[IngestResult] = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{manifest}: documents[{position}] must be a mapping")
        source = _resolve_source(manifest, entry, position)
        tags = entry.get("acl_tags")
        if tags is not None and (not isinstance(tags, list) or not all(isinstance(t, str) for t in tags)):
            raise ValueError(f"{manifest}: documents[{position}].acl_tags must be a list of strings")
        extra = {k: v for k, v in entry.items() if k not in {"path", "url", "title", "acl_tags"}}
        results.append(
            ingest_path(
                conn,
                settings,
                embedder,
                index,
                source,
                title=entry.get("title"),
                acl_tags=tags,
                screen=screen,
                meta=extra or None,
                ledger=ledger,
            )
        )
    return results


# ----------------------------------------------------------------------------- screening


def _screen_text(chunk: Chunk) -> str:
    """What the model sees of a chunk: its section heading (via the source label) and its text."""
    return f"{chunk.heading}\n\n{chunk.text}" if chunk.heading else chunk.text


def _screen_chunks(chunks: list[Chunk], screen: ScreenFn | None) -> list[Flag]:
    """One (quarantined, reason) per chunk. Each chunk is screened with its heading; then every pair of
    adjacent chunks that both passed is screened together, so an instruction split across two sections
    is caught with its context, and both halves are flagged."""
    if screen is None:
        return [(False, None) for _ in chunks]
    flags: list[Flag] = [screen(_screen_text(chunk)) for chunk in chunks]
    for i in range(len(chunks) - 1):
        if flags[i][0] or flags[i + 1][0]:
            continue
        flagged, reason = screen(f"{_screen_text(chunks[i])}\n\n{_screen_text(chunks[i + 1])}")
        if flagged:
            joint = f"together with the adjacent section: {reason or 'flagged'}"
            flags[i] = (True, joint)
            flags[i + 1] = (True, joint)
    return flags


def _screen_title(title: str | None, fallback: str, screen: ScreenFn | None) -> tuple[str | None, str | None]:
    """The title to store and, when the screen flags it, the reason it was replaced by `fallback`.
    A title that already is the fallback (a plain text file named after itself) is not screened."""
    if screen is None or not title or title == fallback:
        return title, None
    flagged, reason = screen(title)
    if not flagged:
        return title, None
    log.warning("title of %r flagged by the injection screen; stored as %r", title, fallback)
    return fallback, reason or "flagged"


def _fallback_title(doc: LoadedDoc) -> str:
    """The name a document falls back to when its own title cannot be shown: the file stem or the URL path stem."""
    if is_url(doc.source):
        parts = urlparse(doc.source)
        return PurePosixPath(parts.path).stem or parts.hostname or doc.source
    return Path(doc.source).stem or doc.source


# ----------------------------------------------------------------------------- helpers


def _normalise_tags(acl_tags: Sequence[str] | None) -> list[str]:
    tags = [str(t).strip() for t in (acl_tags if acl_tags is not None else DEFAULT_ACL_TAGS)]
    tags = [t for t in tags if t]
    if not tags:
        raise ValueError("acl_tags must name at least one tag; use ['public'] for open documents")
    return sorted(set(tags))


def _resolve_source(manifest: Path, entry: dict[str, Any], position: int) -> str:
    raw = entry.get("path") or entry.get("url")
    if not raw or not isinstance(raw, str):
        raise ValueError(f"{manifest}: documents[{position}] needs a 'path' or 'url'")
    if is_url(raw):
        return raw
    candidate = Path(raw)
    if candidate.is_absolute():
        return str(candidate)
    for base in (manifest.parent, manifest.parent.parent, Path.cwd()):
        resolved = base / candidate
        if resolved.exists():
            return str(resolved)
    raise FileNotFoundError(
        f"{manifest}: documents[{position}] path {raw!r} was not found relative to the manifest or the working directory"
    )


def _find_document(conn: sqlite3.Connection, checksum: str) -> int | None:
    row = conn.execute("SELECT id FROM documents WHERE checksum = ?", (checksum,)).fetchone()
    return int(row[0]) if row else None


def _document_tags(conn: sqlite3.Connection, document_id: int) -> list[str]:
    row = conn.execute("SELECT acl_tags FROM documents WHERE id = ?", (document_id,)).fetchone()
    try:
        value = json.loads(row[0]) if row and row[0] else []
    except ValueError:
        value = []
    return sorted({str(t) for t in value}) if isinstance(value, list) else []


def _insert_document(
    conn: sqlite3.Connection,
    source: str,
    title: str | None,
    checksum: str,
    mime: str,
    tags: list[str],
    meta: dict[str, Any],
) -> int:
    cursor = conn.execute(
        "INSERT INTO documents (source, title, checksum, mime, acl_tags, meta) VALUES (?, ?, ?, ?, ?, ?)",
        (source, title, checksum, mime, json.dumps(tags), json.dumps(meta, default=str)),
    )
    return int(cursor.lastrowid)


def _insert_chunks(
    conn: sqlite3.Connection,
    document_id: int,
    chunks: list[Chunk],
    flags: list[Flag],
    tags: list[str],
) -> list[int]:
    tags_json = json.dumps(tags)
    ids: list[int] = []
    for chunk, (quarantined, reason) in zip(chunks, flags, strict=True):
        cursor = conn.execute(
            "INSERT INTO chunks (document_id, ordinal, text, heading, page, char_start, char_end, checksum, "
            "acl_tags, quarantined, quarantine_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                document_id,
                chunk.ordinal,
                chunk.text,
                chunk.heading,
                chunk.page,
                chunk.char_start,
                chunk.char_end,
                hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
                tags_json,
                1 if quarantined else 0,
                reason if quarantined else None,
            ),
        )
        ids.append(int(cursor.lastrowid))
    return ids


def _retag_document(
    conn: sqlite3.Connection,
    index: VectorIndex,
    document_id: int,
    previous: list[str],
    tags: list[str],
    doc: LoadedDoc,
    ledger: Any,
) -> IngestResult:
    """Replace the ACL tags on a stored document and all its chunks, re-upsert the chunks so every
    index (the local cache, a remote index) carries the new tags, and record the change in the ledger."""
    tags_json = json.dumps(tags)
    with transaction(conn):
        conn.execute("UPDATE documents SET acl_tags = ? WHERE id = ?", (tags_json, document_id))
        conn.execute("UPDATE chunks SET acl_tags = ? WHERE document_id = ?", (tags_json, document_id))
        rows = conn.execute(
            "SELECT id, embedding FROM chunks WHERE document_id = ? AND embedding IS NOT NULL ORDER BY id",
            (document_id,),
        ).fetchall()
        chunk_ids = [int(r["id"]) for r in rows]
        if chunk_ids:
            vectors = [np.frombuffer(r["embedding"], dtype="<f4").tolist() for r in rows]
            index.upsert(chunk_ids, vectors)
        _write_ledger(
            conn,
            ledger,
            "ingest",
            {
                "action": "retag",
                "source": doc.source,
                "checksum": doc.checksum,
                "document_id": document_id,
                "acl_tags": tags,
                "previous_acl_tags": previous,
                "chunks": len(chunk_ids),
            },
        )
    invalidate = getattr(index, "invalidate", None)
    if callable(invalidate):
        invalidate()
    log.info("retagged %s: document %d, %s -> %s", doc.source, document_id, previous, tags)
    stored_title = conn.execute("SELECT title FROM documents WHERE id = ?", (document_id,)).fetchone()
    return IngestResult(
        document_id,
        0,
        True,
        doc.source,
        doc.checksum,
        stored_title[0] if stored_title else None,
        acl_tags=tags,
        tags_updated=True,
        previous_acl_tags=previous,
    )


def _write_ledger(conn: sqlite3.Connection, ledger: Any, kind: str, payload: dict[str, Any]) -> None:
    """Append a ledger row inside the caller's open transaction. `ledger` is any object with
    `append(kind, request_id, payload)`; when None, the safety package's `Ledger` is used if it is
    importable. A failure propagates so the surrounding ingest transaction rolls back."""
    if ledger is None:
        try:
            from keel.safety.ledger import Ledger
        except ImportError:
            return
        ledger = Ledger(conn)
    ledger.append(kind, None, payload)
