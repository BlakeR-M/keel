"""Taking a document from a browser, safely, before the ingest pipeline sees it.

Uploading is the one place where a person hands Keel bytes it did not go and fetch itself, so the
checks live here rather than inside a route handler, and they run before anything touches the disk.

Three things are decided here and nowhere else. The name is reduced to its final component, so a
filename carrying directories reaches the store as a plain name. The suffix has to be one the loaders
actually read, taken from their own map so the two cannot drift apart. And the size cap is enforced
while the stream is read rather than after, so an oversized upload stops partway instead of arriving
in memory first.

Everything that survives is written into a directory of its own that the caller removes, and from
there it is an ordinary `ingest_path` call: the same chunking, the same injection screen, the same
ledger row a file ingested from the command line gets.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import BinaryIO

from keel.ingest.loaders import KIND_BY_SUFFIX

__all__ = [
    "ALLOWED_SUFFIXES",
    "MAX_UPLOAD_BYTES",
    "MAX_NAME_LENGTH",
    "UploadRejected",
    "StagedUpload",
    "describe_limits",
    "safe_name",
    "stage_upload",
]

#: Exactly what the loaders can read, taken from their own table so a new format reaches uploads for
#: free and a removed one stops being accepted at the same moment.
ALLOWED_SUFFIXES: frozenset[str] = frozenset(KIND_BY_SUFFIX)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_NAME_LENGTH = 120
_CHUNK = 64 * 1024

# A filename reduced to what a store can hold: word characters, spaces, dots, hyphens. Anything else
# becomes a hyphen rather than being dropped, so two files cannot silently collapse to one name.
_UNSAFE = re.compile(r"[^\w .-]+", re.UNICODE)
_RUNS = re.compile(r"-{2,}")


class UploadRejected(Exception):
    """An upload that never reaches the disk, carrying the sentence to show the person who sent it."""


@dataclass(frozen=True)
class StagedUpload:
    """A file written to a directory of the caller's making, ready for `ingest_path`."""

    path: Path
    name: str
    size: int


def describe_limits() -> str:
    """One line naming what an upload may be, for a form and for a refusal."""
    kinds = ", ".join(sorted(suffix.lstrip(".") for suffix in ALLOWED_SUFFIXES))
    return f"{kinds}, up to {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"


def safe_name(raw: str | None) -> str:
    """The final component of `raw`, reduced to a plain filename with a suffix the loaders read.

    Raises `UploadRejected` when there is nothing usable left. Directory components go before
    anything else is considered, so a name like `../../etc/passwd` is judged as `passwd`, which then
    fails the suffix check the way any other unreadable file would.
    """
    candidate = (raw or "").strip()
    if not candidate:
        raise UploadRejected("That upload arrived without a filename.")
    # Both separators, whatever the sending platform used, then the final component.
    candidate = PurePath(candidate.replace("\\", "/")).name
    candidate = unicodedata.normalize("NFKC", candidate).strip()
    if not candidate or candidate in {".", ".."}:
        raise UploadRejected("That filename has no name in it.")

    suffix = Path(candidate).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise UploadRejected(
            f"Keel reads {describe_limits()}. That file is {suffix or 'without a suffix'}."
        )

    stem = _RUNS.sub("-", _UNSAFE.sub("-", Path(candidate).stem)).strip(" .-")
    if not stem:
        raise UploadRejected("That filename is punctuation only.")
    return f"{stem[:MAX_NAME_LENGTH]}{suffix}"


def stage_upload(stream: BinaryIO, filename: str | None, directory: Path) -> StagedUpload:
    """Write an upload into `directory` under a safe name, refusing past the size cap.

    The cap is checked as the stream is read, so an oversized file stops partway and the partial
    write is removed. The caller owns `directory` and is responsible for removing it.
    """
    name = safe_name(filename)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / name
    size = 0
    try:
        with destination.open("wb") as out:
            while True:
                block = stream.read(_CHUNK)
                if not block:
                    break
                size += len(block)
                if size > MAX_UPLOAD_BYTES:
                    raise UploadRejected(f"That file is larger than {describe_limits()}.")
                out.write(block)
    except UploadRejected:
        destination.unlink(missing_ok=True)
        raise
    if size == 0:
        destination.unlink(missing_ok=True)
        raise UploadRejected("That file is empty.")
    return StagedUpload(path=destination, name=name, size=size)
