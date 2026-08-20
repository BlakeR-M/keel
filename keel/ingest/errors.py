"""Errors raised by the ingest package."""

from __future__ import annotations


class IngestError(Exception):
    """Base class for ingest failures."""


class AirgapViolation(IngestError):
    """Raised when air-gap mode is on and a source would reach a host other than the local machine."""

    def __init__(self, url: str, host: str) -> None:
        self.url = url
        self.host = host
        super().__init__(
            f"air-gap mode is on: fetching {url!r} would reach {host!r}; only 127.0.0.1 and localhost are reachable"
        )


class UnsupportedSource(IngestError):
    """Raised when a source's format cannot be recognised as PDF, DOCX, Markdown, HTML or text."""
