"""Ingest: load PDF, DOCX, Markdown, HTML and text sources, chunk them by section, embed and store."""

from keel.ingest.chunk import Chunk, chunk_document, chunk_pages, chunk_text
from keel.ingest.errors import AirgapViolation, IngestError, UnsupportedSource
from keel.ingest.loaders import LoadedDoc, PageText, load
from keel.ingest.pipeline import IngestResult, ScreenFn, ingest_manifest, ingest_path

__all__ = [
    "AirgapViolation",
    "Chunk",
    "IngestError",
    "IngestResult",
    "LoadedDoc",
    "PageText",
    "ScreenFn",
    "UnsupportedSource",
    "chunk_document",
    "chunk_pages",
    "chunk_text",
    "ingest_manifest",
    "ingest_path",
    "load",
]
