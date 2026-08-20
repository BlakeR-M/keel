"""Startup ingest for a hosted demo: fill an empty store from a corpus manifest.

`ensure_demo_corpus(ctx, manifest)` ingests the manifest when the store holds zero documents and
leaves a populated store alone. Ingest itself is idempotent by checksum, so calling it again against a
populated store would also add nothing; the document count check keeps startup quick and keeps a
store someone has curated by hand exactly as it is. The web app calls this from its lifespan when
`KEEL_BOOTSTRAP_CORPUS` names a manifest (see docs/web.md).
"""

from __future__ import annotations

import logging
from pathlib import Path

from keel.ingest.pipeline import IngestResult, ingest_manifest
from keel.providers.factory import AppContext

log = logging.getLogger("keel.bootstrap")


def document_count(ctx: AppContext) -> int:
    row = ctx.conn.execute("SELECT COUNT(*) FROM documents").fetchone()
    return int(row[0] or 0) if row else 0


def ensure_demo_corpus(ctx: AppContext, manifest: str | Path) -> list[IngestResult]:
    """Ingest `manifest` through the injection screen when the store is empty; return the results,
    an empty list when the store already held documents. A missing manifest raises FileNotFoundError."""
    path = Path(manifest)
    if not path.is_file():
        raise FileNotFoundError(f"bootstrap corpus manifest {path} does not exist")
    existing = document_count(ctx)
    if existing:
        log.info("bootstrap: store already holds %d documents, manifest %s left alone", existing, path)
        return []
    results = ingest_manifest(
        ctx.conn, ctx.settings, ctx.embedder, ctx.index, path, screen=ctx.screen, ledger=ctx.ledger
    )
    log.info(
        "bootstrap: ingested %d documents, %d chunks, %d quarantined from %s",
        len(results),
        sum(r.chunks_added for r in results),
        sum(r.quarantined for r in results),
        path,
    )
    return results


__all__ = ["ensure_demo_corpus", "document_count"]
