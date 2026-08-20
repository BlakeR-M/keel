"""Source loaders: turn a path or URL into plain text pages.

`load()` accepts PDF, DOCX, Markdown, HTML and plain text. Every loader returns a `LoadedDoc` whose
`pages` hold normalised text (LF line endings, UTF-8). HTML and DOCX headings become Markdown ATX
headings (`## Heading`) so the chunker can split on sections for every format.

Format detection order: file extension, then HTTP content-type (URLs), then leading bytes.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from keel.config import Settings, get_settings
from keel.ingest.errors import AirgapViolation, UnsupportedSource

log = logging.getLogger(__name__)

USER_AGENT = "Keel/0.1"
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

KIND_BY_SUFFIX: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".md": "md",
    ".markdown": "md",
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
    ".txt": "txt",
    ".text": "txt",
}
KIND_BY_MIME: dict[str, str] = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/html": "html",
    "application/xhtml+xml": "html",
    "text/markdown": "md",
    "text/x-markdown": "md",
    "text/plain": "txt",
}
MIME_BY_KIND: dict[str, str] = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "html": "text/html",
    "md": "text/markdown",
    "txt": "text/plain",
}


@dataclass
class PageText:
    """Text of one page. `page` is 1-based for paginated sources and None for flat ones."""

    page: int | None
    text: str


@dataclass
class LoadedDoc:
    """A source turned into text pages, with the checksum that makes re-ingest idempotent."""

    source: str
    title: str
    mime: str
    kind: str  # pdf | docx | md | html | txt
    checksum: str  # sha256 of raw_bytes
    raw_bytes: bytes = field(repr=False)
    pages: list[PageText] = field(default_factory=list)

    @property
    def size_bytes(self) -> int:
        return len(self.raw_bytes)

    @property
    def text(self) -> str:
        """All pages joined with blank lines."""
        return "\n\n".join(p.text for p in self.pages)


Parser = Callable[[bytes], tuple[str | None, list[PageText]]]


def is_url(source: str) -> bool:
    return source.lower().startswith(("http://", "https://"))


def load(source: str | Path, *, settings: Settings | None = None) -> LoadedDoc:
    """Load a file path or http(s) URL into a `LoadedDoc`.

    URL fetches honour air-gap mode: with `settings.airgap` on, any host other than the local machine
    raises `AirgapViolation` before a connection is opened.
    """
    src = str(source)
    if is_url(src):
        raw, content_type = _fetch(src, settings or get_settings())
        suffix = PurePosixPath(urlparse(src).path).suffix.lower()
        kind = detect_kind(suffix, content_type, raw)
        fallback_title = PurePosixPath(urlparse(src).path).stem or urlparse(src).hostname or src
    else:
        path = Path(source)
        raw = path.read_bytes()
        kind = detect_kind(path.suffix.lower(), None, raw)
        fallback_title = path.stem
    title, pages = _PARSERS[kind](raw)
    return LoadedDoc(
        source=src,
        title=title or fallback_title,
        mime=MIME_BY_KIND[kind],
        kind=kind,
        checksum=hashlib.sha256(raw).hexdigest(),
        raw_bytes=raw,
        pages=pages,
    )


def detect_kind(suffix: str, content_type: str | None, raw: bytes) -> str:
    """Pick the loader kind from extension, then content-type, then leading bytes."""
    if suffix in KIND_BY_SUFFIX:
        return KIND_BY_SUFFIX[suffix]
    if content_type:
        mime = content_type.split(";", 1)[0].strip().lower()
        if mime in KIND_BY_MIME:
            return KIND_BY_MIME[mime]
    head = raw[:2048].lstrip()
    if head.startswith(b"%PDF"):
        return "pdf"
    if raw[:2] == b"PK" and b"word/document.xml" in raw[:65536]:
        return "docx"
    lowered = head[:512].lower()
    if lowered.startswith(b"<!doctype html") or b"<html" in lowered or b"<body" in lowered:
        return "html"
    if _looks_like_text(head):
        return "txt"
    raise UnsupportedSource(
        f"cannot recognise the format of this source (suffix={suffix!r}, content-type={content_type!r})"
    )


def _looks_like_text(head: bytes) -> bool:
    if not head:
        return True
    if b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


# ----------------------------------------------------------------------------- fetching


def _refuse_remote_hosts(request: Any) -> None:
    """httpx request hook for air-gap mode: every hop, redirects included, must stay on the local machine."""
    host = request.url.host
    if host not in LOCAL_HOSTS:
        raise AirgapViolation(str(request.url), host)


def _fetch(url: str, settings: Settings) -> tuple[bytes, str | None]:
    """GET a URL and return (bytes, content-type). Air-gap mode refuses non-local hosts, redirects included."""
    host = urlparse(url).hostname or ""
    if settings.airgap and host not in LOCAL_HOSTS:
        raise AirgapViolation(url, host)
    import httpx

    hooks = {"request": [_refuse_remote_hosts]} if settings.airgap else {}
    with httpx.Client(
        follow_redirects=True, timeout=30.0, headers={"User-Agent": USER_AGENT}, event_hooks=hooks
    ) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.content, response.headers.get("content-type")


# ----------------------------------------------------------------------------- parsers


def _decode(raw: bytes) -> str:
    text = raw.decode("utf-8-sig", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


_ATX_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)


def _parse_txt(raw: bytes) -> tuple[str | None, list[PageText]]:
    return None, [PageText(page=None, text=_decode(raw))]


def _parse_md(raw: bytes) -> tuple[str | None, list[PageText]]:
    text = _decode(raw)
    title = None
    for match in _ATX_HEADING.finditer(text):
        heading = match.group(2).strip()
        if len(match.group(1)) == 1:
            title = heading
            break
        title = title or heading
    return title, [PageText(page=None, text=text)]


def _parse_pdf(raw: bytes) -> tuple[str | None, list[PageText]]:
    from io import BytesIO

    from pypdf import PdfReader

    reader = PdfReader(BytesIO(raw))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:  # pragma: no cover - depends on the PDF
            log.warning("PDF is encrypted and the empty password was rejected: %s", exc)
    title = None
    try:
        meta = reader.metadata
        if meta is not None and meta.title:
            title = str(meta.title).strip() or None
    except Exception as exc:  # pragma: no cover - malformed metadata
        log.warning("PDF metadata unreadable: %s", exc)
    pages: list[PageText] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - malformed content stream
            log.warning("PDF page %d text extraction failed: %s", number, exc)
            text = ""
        pages.append(PageText(page=number, text=text.replace("\r\n", "\n").replace("\r", "\n")))
    return title, pages


def _docx_heading_level(style_name: str) -> int | None:
    name = (style_name or "").strip().lower()
    if name == "title":
        return 1
    if name.startswith("heading"):
        digits = "".join(ch for ch in name if ch.isdigit())
        level = int(digits) if digits else 1
        return max(1, min(level, 6))
    return None


def _parse_docx(raw: bytes) -> tuple[str | None, list[PageText]]:
    from io import BytesIO

    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(BytesIO(raw))
    title: str | None = None
    try:
        title = (document.core_properties.title or "").strip() or None
    except Exception:  # pragma: no cover - missing core properties part
        title = None
    blocks: list[str] = []
    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            text = " ".join(item.text.split())
            if not text:
                continue
            level = _docx_heading_level(item.style.name if item.style is not None else "")
            if level is not None:
                if title is None:
                    title = text
                blocks.append(f"{'#' * level} {text}")
            else:
                blocks.append(text)
        elif isinstance(item, Table):
            for row in item.rows:
                cells = [" ".join(cell.text.split()) for cell in row.cells]
                line = " | ".join(cells).strip()
                if line:
                    blocks.append(line)
    return title, [PageText(page=None, text="\n\n".join(blocks))]


_HTML_DROP = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        "nav",
        "aside",
        "footer",
        "iframe",
        "svg",
        "canvas",
        "form",
        "button",
        "input",
        "select",
        "textarea",
        "option",
        "head",
        "meta",
        "link",
        "title",
    }
)
_HTML_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_HTML_BLOCKS = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "main",
        "header",
        "ul",
        "ol",
        "dl",
        "dt",
        "dd",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "caption",
        "blockquote",
        "pre",
        "figure",
        "figcaption",
        "address",
        "details",
        "summary",
        "hr",
        "fieldset",
        "legend",
        "body",
        "html",
    }
)


def html_to_text(root: Any) -> str:
    """Render a BeautifulSoup tree as Markdown-flavoured text: ATX headings, dash bullets, pipe-separated
    table cells. Navigation, scripts, styles and form controls are left out."""
    from bs4.element import Comment, Declaration, Doctype, NavigableString, ProcessingInstruction

    out: list[str] = []

    def render(node: Any) -> None:
        for child in node.children:
            if isinstance(child, (Comment, Declaration, Doctype, ProcessingInstruction)):
                continue
            if isinstance(child, NavigableString):
                out.append(str(child))
                continue
            name = (child.name or "").lower()
            if name in _HTML_DROP:
                continue
            if name in _HTML_HEADINGS:
                heading = " ".join(child.get_text(" ").split())
                if heading:
                    out.append(f"\n\n{'#' * _HTML_HEADINGS[name]} {heading}\n\n")
                continue
            if name == "br":
                out.append("\n")
            elif name == "li":
                out.append("\n- ")
                render(child)
                out.append("\n")
            elif name in ("td", "th"):
                render(child)
                out.append(" | ")
            elif name == "tr":
                out.append("\n")
                render(child)
                out.append("\n")
            elif name in _HTML_BLOCKS:
                out.append("\n\n")
                render(child)
                out.append("\n\n")
            else:
                render(child)

    render(root)
    lines = []
    for line in "".join(out).split("\n"):
        line = " ".join(line.split())
        if line.endswith(" |"):
            line = line[:-2]
        lines.append(line)
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _parse_html(raw: bytes) -> tuple[str | None, list[PageText]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw, "lxml")
    title = None
    if soup.title is not None and soup.title.string:
        title = " ".join(soup.title.string.split()) or None
    if title is None:
        h1 = soup.find("h1")
        if h1 is not None:
            title = " ".join(h1.get_text(" ").split()) or None
    root = soup.find("main") or soup.find("article") or soup.body or soup
    return title, [PageText(page=None, text=html_to_text(root))]


_PARSERS: dict[str, Parser] = {
    "txt": _parse_txt,
    "md": _parse_md,
    "pdf": _parse_pdf,
    "docx": _parse_docx,
    "html": _parse_html,
}
