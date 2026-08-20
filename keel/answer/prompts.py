"""Prompts for grounded answering. Short and explicit: the local model is a 3B."""

from __future__ import annotations

import json
from typing import Any

from keel.providers.base import ChunkHit

REFUSAL = "That is not in the documents I have access to."

SYSTEM_PROMPT = f"""You answer questions using only the numbered sources provided in the message.

Rules:
1. Use only facts stated in the sources. Add nothing from memory or general knowledge.
2. Write the answer as one or more complete sentences that state the facts in words. Then, at the end of each sentence, cite the source it came from as a number in square brackets, for example: Three written quotes are required. [1]
3. A citation on its own is never an answer. Every citation follows a sentence that states the fact.
4. If the sources do not contain the answer, reply with exactly this sentence and nothing else: {REFUSAL}
5. Keep the answer short and direct. Plain sentences, no preamble."""

BARE_CITATION_RETRY = """That reply contained only citation markers. State the answer in words as a full sentence, then add the citation, for example: The retention period is seven years. [2]"""

JSON_INSTRUCTION = """Reply with a single JSON object and nothing else. It must match this JSON schema:
{schema}
Fill every field from the sources. Leave out markdown, code fences and commentary."""

RETRY_INSTRUCTION = """That reply was invalid: {error}
Reply again with only a JSON object that matches the schema."""

SNIPPET_CHARS = 240


def source_label(hit: ChunkHit) -> str:
    """Human-readable label for a source: title or file name, with page and heading when known."""
    name = hit.title or hit.source.replace("\\", "/").rsplit("/", 1)[-1]
    parts = [name]
    if hit.page:
        parts.append(f"p.{hit.page}")
    if hit.heading:
        parts.append(hit.heading)
    return ", ".join(parts)


def build_source_block(hits: list[ChunkHit]) -> str:
    """Number the sources from 1 and lay them out one after another."""
    blocks = [f"[{n}] {source_label(hit)}\n{hit.text.strip()}" for n, hit in enumerate(hits, start=1)]
    return "\n\n".join(blocks)


def build_user_prompt(question: str, hits: list[ChunkHit], json_schema: dict[str, Any] | None = None) -> str:
    """The user turn: sources, the question, and the JSON instruction when a schema is given."""
    text = f"Sources:\n\n{build_source_block(hits)}\n\nQuestion: {question.strip()}"
    if json_schema is not None:
        text += "\n\n" + JSON_INSTRUCTION.format(schema=json.dumps(json_schema))
    return text


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """The first `limit` characters of a chunk, on one line."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"
