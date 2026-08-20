"""Section-aware chunking.

Text is split on Markdown ATX headings first (HTML and DOCX loaders emit those), then paragraphs inside
each section are packed into chunks of at most `chunk_size` characters with `chunk_overlap` characters of
sentence-aligned overlap. A paragraph longer than the chunk size is packed sentence by sentence; only a
single sentence longer than the chunk size is cut at a word boundary. Every chunk carries the nearest
heading, the page number and its character span within the page text. The output is a pure function
of the input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from keel.config import Settings
from keel.ingest.loaders import LoadedDoc, PageText


@dataclass(frozen=True)
class Chunk:
    """One retrieval unit. `char_start`/`char_end` index into the page text the chunk came from."""

    ordinal: int
    text: str
    heading: str | None
    page: int | None
    char_start: int
    char_end: int


@dataclass(frozen=True)
class Section:
    """A heading and the character span of its body within the page text."""

    heading: str | None
    start: int
    end: int


_HEADING_LINE = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*\n?$")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?][\"')\]])\s+|(?<=[.!?])\s+")
_FENCE = ("```", "~~~")

Span = tuple[int, int]


def split_sections(text: str, inherited_heading: str | None = None) -> list[Section]:
    """Split text at ATX headings. The span before the first heading carries `inherited_heading`
    (the heading still open from the previous page). Headings inside fenced code blocks are ignored."""
    sections: list[Section] = []
    heading = inherited_heading
    body_start = 0
    in_fence = False
    pos = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith(_FENCE):
            in_fence = not in_fence
        elif not in_fence:
            match = _HEADING_LINE.match(line)
            if match:
                sections.append(Section(heading, body_start, pos))
                heading = match.group(2).strip() or heading
                body_start = pos + len(line)
        pos += len(line)
    sections.append(Section(heading, body_start, len(text)))
    return sections


def paragraph_spans(text: str, start: int, end: int) -> list[Span]:
    """Spans of paragraphs (runs of non-blank lines) inside text[start:end], trimmed of surrounding whitespace."""
    spans: list[Span] = []
    pos = start
    para_start: int | None = None
    last_end = start
    for line in text[start:end].splitlines(keepends=True):
        line_start = pos
        pos += len(line)
        if line.strip():
            if para_start is None:
                para_start = line_start + (len(line) - len(line.lstrip()))
            last_end = line_start + len(line.rstrip())
        elif para_start is not None:
            spans.append((para_start, last_end))
            para_start = None
    if para_start is not None:
        spans.append((para_start, last_end))
    return spans


def sentence_spans(text: str, start: int, end: int) -> list[Span]:
    """Spans of sentences inside text[start:end]. Breaks after . ! ? followed by whitespace."""
    spans: list[Span] = []
    cursor = start
    for match in _SENTENCE_BREAK.finditer(text, start, end):
        if match.start() > cursor:
            spans.append((cursor, match.start()))
        cursor = match.end()
    if cursor < end:
        spans.append((cursor, end))
    return spans


def _word_spans(text: str, start: int, end: int, size: int) -> list[Span]:
    """Cut text[start:end] into pieces of at most `size` characters at whitespace where possible."""
    spans: list[Span] = []
    cursor = start
    while end - cursor > size:
        cut = cursor + size
        window = text[cursor:cut]
        last_space = max(window.rfind(" "), window.rfind("\n"), window.rfind("\t"))
        if last_space > 0:
            cut = cursor + last_space
        spans.append((cursor, cut))
        cursor = cut
        while cursor < end and text[cursor].isspace():
            cursor += 1
    if cursor < end:
        spans.append((cursor, end))
    return spans


def _units(text: str, start: int, end: int, size: int) -> list[Span]:
    """Packing units for a section: paragraphs, or sentences when a paragraph exceeds `size`, or word
    pieces when a single sentence exceeds `size`."""
    units: list[Span] = []
    for p_start, p_end in paragraph_spans(text, start, end):
        if p_end - p_start <= size:
            units.append((p_start, p_end))
            continue
        for s_start, s_end in sentence_spans(text, p_start, p_end):
            if s_end - s_start <= size:
                units.append((s_start, s_end))
            else:
                units.extend(_word_spans(text, s_start, s_end, size))
    return units


def _overlap_start(text: str, start: int, end: int, overlap: int) -> int | None:
    """Start of the trailing whole sentences of text[start:end] that fit within `overlap` characters."""
    if overlap <= 0:
        return None
    for s_start, _ in sentence_spans(text, start, end):
        if end - s_start <= overlap:
            return s_start
    return None


def pack_spans(text: str, units: list[Span], chunk_size: int, overlap: int) -> list[Span]:
    """Greedily pack units into spans of at most `chunk_size` characters. When a chunk closes, the next
    chunk starts with the trailing sentences of the closed chunk that fit within `overlap`."""
    chunks: list[Span] = []
    current: Span | None = None
    for u_start, u_end in units:
        if current is None:
            current = (u_start, u_end)
            continue
        if u_end - current[0] <= chunk_size:
            current = (current[0], u_end)
            continue
        chunks.append(current)
        carry = _overlap_start(text, current[0], current[1], overlap)
        if carry is not None and u_end - carry <= chunk_size:
            current = (carry, u_end)
        else:
            current = (u_start, u_end)
    if current is not None:
        chunks.append(current)
    return chunks


def _chunk_page(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    page: int | None,
    inherited_heading: str | None,
    first_ordinal: int,
) -> tuple[list[Chunk], str | None]:
    """Chunk one page of text. Returns the chunks and the heading still open at the end of the page."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive number of characters")
    overlap = max(0, min(chunk_overlap, chunk_size // 2))
    chunks: list[Chunk] = []
    ordinal = first_ordinal
    sections = split_sections(text, inherited_heading)
    for section in sections:
        units = _units(text, section.start, section.end, chunk_size)
        for start, end in pack_spans(text, units, chunk_size, overlap):
            body = text[start:end]
            if not body.strip():
                continue
            chunks.append(
                Chunk(
                    ordinal=ordinal,
                    text=body,
                    heading=section.heading,
                    page=page,
                    char_start=start,
                    char_end=end,
                )
            )
            ordinal += 1
    return chunks, sections[-1].heading


def chunk_text(text: str, *, chunk_size: int, chunk_overlap: int, page: int | None = None) -> list[Chunk]:
    """Chunk a single block of text (one page)."""
    chunks, _ = _chunk_page(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        page=page,
        inherited_heading=None,
        first_ordinal=0,
    )
    return chunks


def chunk_pages(pages: list[PageText], *, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Chunk a sequence of pages. Headings carry across page boundaries; chunks never span pages."""
    chunks: list[Chunk] = []
    heading: str | None = None
    for page in pages:
        page_chunks, heading = _chunk_page(
            page.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            page=page.page,
            inherited_heading=heading,
            first_ordinal=len(chunks),
        )
        chunks.extend(page_chunks)
    return chunks


def chunk_document(doc: LoadedDoc, settings: Settings) -> list[Chunk]:
    """Chunk a loaded document with the sizes from settings."""
    return chunk_pages(doc.pages, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
