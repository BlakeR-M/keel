"""Fetch a small public, CC BY 4.0 demo corpus into demo-corpus/ and write demo-corpus/manifest.yaml.

Sources: the Australian Government Style Manual (stylemanual.gov.au) and the Australian Signals
Directorate's Essential Eight explainer (cyber.gov.au). Both publish under Creative Commons Attribution
4.0; the manifest records the attribution and the source URL of every page.

Politeness: at most 15 pages, one request at a time with a short pause, a named User-Agent, and any
error skips that page. The script refuses to run when KEEL_AIRGAP is set to 1.

Usage:  python scripts/fetch_demo_corpus.py [--out demo-corpus] [--limit 15]
Then:   keel ingest --manifest demo-corpus/manifest.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from keel.config import Settings  # noqa: E402

USER_AGENT = "Keel-demo/0.1"
PAUSE_SECONDS = 1.0
TIMEOUT_SECONDS = 20.0
MAX_PAGES = 15

STYLE_MANUAL = "https://www.stylemanual.gov.au"
CYBER = "https://www.cyber.gov.au"

PAGES: list[tuple[str, str]] = [
    # (url, attribution)
    (
        f"{STYLE_MANUAL}/writing-and-designing-content/clear-language-and-writing-style/plain-language-and-word-choice",
        "Style Manual",
    ),
    (
        f"{STYLE_MANUAL}/writing-and-designing-content/clear-language-and-writing-style/sentences",
        "Style Manual",
    ),
    (
        f"{STYLE_MANUAL}/writing-and-designing-content/clear-language-and-writing-style/voice-and-tone",
        "Style Manual",
    ),
    (f"{STYLE_MANUAL}/structuring-content/headings", "Style Manual"),
    (f"{STYLE_MANUAL}/structuring-content/lists", "Style Manual"),
    (f"{STYLE_MANUAL}/structuring-content/paragraphs", "Style Manual"),
    (f"{STYLE_MANUAL}/grammar-punctuation-and-conventions/punctuation/commas", "Style Manual"),
    (f"{STYLE_MANUAL}/grammar-punctuation-and-conventions/punctuation/apostrophes", "Style Manual"),
    (f"{STYLE_MANUAL}/grammar-punctuation-and-conventions/punctuation/quotation-marks", "Style Manual"),
    (f"{STYLE_MANUAL}/grammar-punctuation-and-conventions/numbers-and-measurements/numerals", "Style Manual"),
    (
        f"{STYLE_MANUAL}/grammar-punctuation-and-conventions/numbers-and-measurements/dates-and-time",
        "Style Manual",
    ),
    (f"{STYLE_MANUAL}/grammar-punctuation-and-conventions/numbers-and-measurements/currency", "Style Manual"),
    (f"{STYLE_MANUAL}/grammar-punctuation-and-conventions/spelling", "Style Manual"),
    (
        f"{CYBER}/business-government/asds-cyber-security-frameworks/essential-eight/essential-eight-explained",
        "Essential Eight",
    ),
    (
        f"{CYBER}/resources-business-and-government/essential-cyber-security/essential-eight/essential-eight-explained",
        "Essential Eight",
    ),
]

ATTRIBUTION = {
    "Style Manual": "Australian Government Style Manual, Commonwealth of Australia, CC BY 4.0",
    "Essential Eight": "Australian Signals Directorate, cyber.gov.au, Commonwealth of Australia, CC BY 4.0",
}


def airgap_on() -> bool:
    return Settings().airgap


def slug_for(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/") or "index"
    slug = re.sub(r"[^a-z0-9]+", "-", f"{parsed.hostname}-{path}".lower()).strip("-")
    return slug[:120]


def page_title(html: bytes, fallback: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    if soup.title is not None and soup.title.string:
        return " ".join(soup.title.string.split())
    h1 = soup.find("h1")
    if h1 is not None:
        return " ".join(h1.get_text(" ").split()) or fallback
    return fallback


def yaml_quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def fetch_all(out_dir: Path, limit: int) -> list[dict[str, str]]:
    import httpx

    entries: list[dict[str, str]] = []
    seen_checksums: set[str] = set()
    with httpx.Client(
        follow_redirects=True, timeout=TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT}
    ) as client:
        for url, group in PAGES[:limit]:
            try:
                response = client.get(url)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "html" not in content_type:
                    print(f"skip (content-type {content_type!r}): {url}")
                    continue
                body = response.content
            except Exception as exc:  # any failure skips the page
                print(f"skip ({type(exc).__name__}: {exc}): {url}")
                time.sleep(PAUSE_SECONDS)
                continue
            checksum = hashlib.sha256(body).hexdigest()
            if checksum in seen_checksums:
                print(f"skip (same content as an earlier page): {url}")
                continue
            seen_checksums.add(checksum)
            name = f"{slug_for(url)}.html"
            (out_dir / name).write_bytes(body)
            entries.append(
                {
                    "path": f"{out_dir.name}/{name}",
                    "title": page_title(body, name),
                    "source_url": str(response.url),
                    "attribution": ATTRIBUTION[group],
                }
            )
            print(f"saved: {name} ({len(body)} bytes)")
            time.sleep(PAUSE_SECONDS)
    return entries


def write_manifest(out_dir: Path, entries: list[dict[str, str]]) -> Path:
    lines = [
        "# Demo corpus manifest, written by scripts/fetch_demo_corpus.py.",
        "# Pages are published under Creative Commons Attribution 4.0 by the Commonwealth of Australia;",
        "# each entry names its source URL and attribution. Ingest with: keel ingest --manifest demo-corpus/manifest.yaml",
        "documents:",
    ]
    for entry in entries:
        lines.append(f"  - path: {yaml_quote(entry['path'])}")
        lines.append(f"    title: {yaml_quote(entry['title'])}")
        lines.append("    acl_tags: [public]")
        lines.append(f"    source_url: {yaml_quote(entry['source_url'])}")
        lines.append(f"    attribution: {yaml_quote(entry['attribution'])}")
    manifest = out_dir / "manifest.yaml"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default=str(REPO_ROOT / "demo-corpus"), help="output directory")
    parser.add_argument(
        "--limit", type=int, default=MAX_PAGES, help=f"maximum pages to fetch (at most {MAX_PAGES})"
    )
    args = parser.parse_args(argv)

    if airgap_on():
        print("KEEL_AIRGAP is on: this script fetches public web pages and stays idle in air-gap mode.")
        return 2
    limit = max(1, min(args.limit, MAX_PAGES))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = fetch_all(out_dir, limit)
    manifest = write_manifest(out_dir, entries)
    print(f"{len(entries)} page(s) saved; manifest at {manifest}")
    return 0 if entries else 1


if __name__ == "__main__":
    raise SystemExit(main())
