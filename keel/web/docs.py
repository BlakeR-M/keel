"""The repository's Markdown documentation, rendered as pages of the site.

The ten documents under `docs/` are the reference material a reader wants after the landing page:
the architecture, the CLI, the web app, the evaluation harness, the on-premise and Azure
deployments, the threat model, the adversarial review and the demo script. They live as Markdown in
the repository so they stay reviewable in a diff, and this module serves the same files as pages so
a visitor never has to leave for GitHub to read them.

Only files that sit directly in `docs/` and whose name is a plain slug are reachable, so a request
path can name nothing outside the directory. Links are rewritten as the page renders: a sibling
`.md` file becomes another page here, and a path that reaches out of `docs/` (source files, tests,
fixtures) becomes a link into the repository on GitHub.

The rendered HTML is cached per slug against the file's modification time, so a page costs one
Markdown pass per edit rather than one per request.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path

import markdown

__all__ = [
    "DOCS_DIR",
    "GITHUB_BLOB",
    "GITHUB_REPO",
    "Doc",
    "index",
    "page",
    "rewrite_links",
    "slugs",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"

GITHUB_REPO = "https://github.com/BlakeR-M/keel"
GITHUB_BLOB = f"{GITHUB_REPO}/blob/main"

SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

#: The reading order on the index. Files present in `docs/` but absent here follow, sorted by slug,
#: so a new document appears without an edit to this list.
READING_ORDER: tuple[str, ...] = (
    "tutorial",
    "architecture",
    "security-review",
    "threat-model",
    "cli",
    "web",
    "evals",
    "onprem",
    "deploy-azure",
    "demo-script",
)

#: One line per document, shown under its title on the index. A document with no entry shows the
#: first sentence of its own body instead.
BLURBS: dict[str, str] = {
    "tutorial": "Install it, load documents, ask a question, run the agent, verify the ledger.",
    "architecture": "Every component, the data flow through a question, and the ledger hash recipe.",
    "security-review": "The pre-publish adversarial review: 105 attack tests, 27 findings, and the test that proves each fix.",
    "threat-model": "Assets, trust boundaries, the attacks considered, and the control that answers each one.",
    "cli": "Every command, with the flags and the output each one prints.",
    "web": "The routes, the identity model behind a reverse proxy, and the demo flags.",
    "evals": "The golden set, the scored metrics, and the regression gate that fails a release.",
    "onprem": "Running the appliance on one machine, with and without a network.",
    "deploy-azure": "The client-tenancy deployment on managed identity, with local key auth disabled.",
    "demo-script": "The recorded walkthrough, shot by shot.",
}

TITLE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
HREF = re.compile(r'href="([^"]+)"')
FIRST_SENTENCE = re.compile(r"^([A-Z][^\n]*?[.!?])(?:\s|$)", re.MULTILINE)

_EXTENSIONS = ("tables", "fenced_code", "sane_lists", "attr_list", "toc")


@dataclass(frozen=True)
class Doc:
    """One rendered documentation page."""

    slug: str
    title: str
    blurb: str
    html: str
    source: str
    """Path of the Markdown file relative to the repository root, for the "edit on GitHub" link."""


_lock = threading.Lock()
_cache: dict[str, tuple[float, Doc]] = {}


def _path(slug: str) -> Path | None:
    """The Markdown file for `slug`, or None when the slug names nothing readable in `docs/`.

    A slug is lower-case words joined by hyphens, so it can hold no separator, no parent reference
    and no drive letter. The resolved path is checked against `DOCS_DIR` as well, which closes a
    symlink pointing out of the directory.
    """
    if not SLUG.match(slug):
        return None
    candidate = DOCS_DIR / f"{slug}.md"
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    if resolved.parent != DOCS_DIR.resolve() or not resolved.is_file():
        return None
    return resolved


def slugs() -> list[str]:
    """Every readable document, in reading order, with unlisted files sorted after."""
    if not DOCS_DIR.is_dir():
        return []
    found = {path.stem for path in DOCS_DIR.glob("*.md") if SLUG.match(path.stem)}
    ordered = [slug for slug in READING_ORDER if slug in found]
    return ordered + sorted(found - set(ordered))


def rewrite_links(html: str) -> str:
    """Point relative links at this site or at the repository on GitHub.

    A sibling document becomes a page here (`architecture.md` to `/docs/architecture`, fragment
    kept). Anything reaching out of `docs/` is a source file, a test or a fixture, so it becomes a
    link into the repository. Absolute URLs, fragments and mail links are left as written.
    """

    def replace(match: re.Match[str]) -> str:
        href = match.group(1)
        if not href or href.startswith(("http://", "https://", "#", "mailto:", "/")):
            return match.group(0)
        target, _, fragment = href.partition("#")
        suffix = f"#{fragment}" if fragment else ""
        if target.endswith(".md"):
            stem = Path(target).stem
            if "/" not in target.strip("./") and SLUG.match(stem):
                return f'href="/docs/{stem}{suffix}"'
        path = target
        while path.startswith(("../", "./")):
            path = path.split("/", 1)[1] if "/" in path else ""
        if path.startswith("../") or not path:
            return match.group(0)
        prefix = "" if target.startswith("../") else "docs/"
        return f'href="{GITHUB_BLOB}/{prefix}{path}{suffix}"'

    return HREF.sub(replace, html)


def _blurb(slug: str, text: str) -> str:
    if slug in BLURBS:
        return BLURBS[slug]
    body = TITLE.sub("", text, count=1).strip()
    match = FIRST_SENTENCE.search(body)
    return match.group(1) if match else ""


def _render(slug: str, path: Path) -> Doc:
    text = path.read_text(encoding="utf-8")
    title_match = TITLE.search(text)
    title = title_match.group(1) if title_match else slug.replace("-", " ").capitalize()
    renderer = markdown.Markdown(extensions=list(_EXTENSIONS), output_format="html")
    html = rewrite_links(renderer.convert(text))
    return Doc(slug=slug, title=title, blurb=_blurb(slug, text), html=html, source=f"docs/{slug}.md")


def page(slug: str) -> Doc | None:
    """The rendered document for `slug`, or None when it names nothing in `docs/`.

    The result is cached against the file's modification time, so editing a document during a local
    run shows the edit on the next request without a restart.
    """
    path = _path(slug)
    if path is None:
        return None
    stamp = path.stat().st_mtime
    with _lock:
        cached = _cache.get(slug)
        if cached is not None and cached[0] == stamp:
            return cached[1]
    doc = _render(slug, path)
    with _lock:
        _cache[slug] = (stamp, doc)
    return doc


def index() -> list[Doc]:
    """Every document in reading order, rendered, for the documentation index."""
    pages = [page(slug) for slug in slugs()]
    return [doc for doc in pages if doc is not None]
