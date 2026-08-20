"""Grounded answering: retrieve, gate on relevance, generate with numbered sources, resolve [n]
citations back to chunks. JSON mode validates the model output against a schema and retries once.

Ledger and log hooks are duck-typed: `ledger.append(kind, request_id, payload)` and
`log.record(**fields)`. Pass None to run without either.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from keel.answer.prompts import (
    BARE_CITATION_RETRY,
    REFUSAL,
    RETRY_INSTRUCTION,
    SYSTEM_PROMPT,
    build_user_prompt,
    snippet,
)
from keel.answer.schema import validate
from keel.answer.types import Answer, Citation, User
from keel.providers.base import ChatMessage, ChatResult, ChunkHit
from keel.retrieval.hybrid import allowed

_CITATION_RE = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*(.*?)\s*```$", re.DOTALL)
_TOP_CITATIONS_WHEN_UNMARKED = 2


class AnswerEngine:
    """Answers a question from the corpus a user is entitled to see, with citations or a refusal."""

    def __init__(self, llm: Any, retriever: Any, settings: Any, ledger: Any = None, log: Any = None) -> None:
        self.llm = llm
        self.retriever = retriever
        self.settings = settings
        self.ledger = ledger
        self.log = log

    # ------------------------------------------------------------------ public

    def answer(self, question: str, user: User, *, json_schema: dict[str, Any] | None = None) -> Answer:
        """Answer `question` for `user`. Refuses without calling the model when retrieval finds
        nothing relevant enough; otherwise returns cited prose, or validated JSON when
        `json_schema` is given."""
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        self._ledger(
            "request",
            request_id,
            {
                "mode": "answer",
                "question": question,
                "user_id": user.user_id,
                "user_tags": list(user.tags),
                "json_mode": json_schema is not None,
            },
        )

        hits = list(self.retriever.retrieve(question, list(user.tags)))
        # The retriever filters by ACL and quarantine already; the same test runs again here, at the
        # generation boundary, so a retriever that leaks (a bug, another backend) still puts nothing the
        # user may not read into the prompt or the citations.
        sources = [h for h in hits if allowed(h, frozenset(user.tags))]
        self._ledger(
            "retrieval",
            request_id,
            {
                "chunk_ids": [h.chunk_id for h in sources],
                "scores": [round(float(h.score), 4) for h in sources],
                "quarantined_ids": [h.chunk_id for h in hits if h.quarantined],
            },
        )

        top_score = max((float(h.score) for h in sources), default=None)
        if top_score is None or top_score < self.settings.min_relevance:
            answer = Answer(REFUSAL, [], True, sources, 0, 0, 0, request_id)
            model = ""
        elif json_schema is not None:
            answer, model = self._answer_json(question, sources, json_schema, request_id)
        else:
            answer, model = self._answer_text(question, sources, request_id)

        answer.latency_ms = int((time.perf_counter() - started) * 1000)
        self._ledger(
            "answer",
            request_id,
            {
                "text": answer.text,
                "refused": answer.refused,
                "citations": [c.chunk_id for c in answer.citations],
                "prompt_tokens": answer.prompt_tokens,
                "output_tokens": answer.output_tokens,
                "latency_ms": answer.latency_ms,
                "model": model,
                "error": answer.error,
            },
        )
        self._record(
            request_id=request_id,
            user_id=user.user_id,
            user_tags=list(user.tags),
            mode="answer",
            question=question,
            retrieved_ids=[h.chunk_id for h in sources],
            tool_calls=None,
            answer=answer.text,
            refused=answer.refused,
            citations=[c.to_dict() for c in answer.citations],
            latency_ms=answer.latency_ms,
            prompt_tokens=answer.prompt_tokens,
            output_tokens=answer.output_tokens,
            provider=getattr(self.llm, "name", ""),
            model=model,
        )
        return answer

    # ------------------------------------------------------------------ modes

    def _answer_text(self, question: str, sources: list[ChunkHit], request_id: str) -> tuple[Answer, str]:
        messages = [
            ChatMessage("system", SYSTEM_PROMPT),
            ChatMessage("user", build_user_prompt(question, sources)),
        ]
        result = self._chat(messages)
        text = result.content.strip()
        prompt_tokens, output_tokens = result.prompt_tokens, result.output_tokens
        if is_bare_citation(text):
            # The small model sometimes emits "[1]" and stops. Ask once for the sentence.
            retry_messages = messages + [
                ChatMessage("assistant", text),
                ChatMessage("user", BARE_CITATION_RETRY),
            ]
            retry = self._chat(retry_messages)
            prompt_tokens += retry.prompt_tokens
            output_tokens += retry.output_tokens
            if retry.content.strip() and not is_bare_citation(retry.content.strip()):
                text = retry.content.strip()
                result = retry
        if is_refusal(text):
            answer = Answer(REFUSAL, [], True, sources, prompt_tokens, output_tokens, 0, request_id)
            return answer, result.model
        citations = resolve_citations(text, sources)
        if not citations:
            citations = top_citations(sources)
        answer = Answer(text, citations, False, sources, prompt_tokens, output_tokens, 0, request_id)
        return answer, result.model

    def _answer_json(
        self, question: str, sources: list[ChunkHit], schema: dict[str, Any], request_id: str
    ) -> tuple[Answer, str]:
        messages = [
            ChatMessage("system", SYSTEM_PROMPT),
            ChatMessage("user", build_user_prompt(question, sources, json_schema=schema)),
        ]
        first = self._chat(messages, json_schema=schema)
        prompt_tokens, output_tokens, model = first.prompt_tokens, first.output_tokens, first.model
        data, error = parse_json_output(first.content, schema)
        raw = first.content
        if error is not None:
            messages.append(ChatMessage("assistant", first.content))
            messages.append(ChatMessage("user", RETRY_INSTRUCTION.format(error=error)))
            second = self._chat(messages, json_schema=schema)
            prompt_tokens += second.prompt_tokens
            output_tokens += second.output_tokens
            model = second.model or model
            data, error = parse_json_output(second.content, schema)
            raw = second.content

        citations = top_citations(sources)
        if error is not None:
            answer = Answer(
                raw.strip(),
                citations,
                False,
                sources,
                prompt_tokens,
                output_tokens,
                0,
                request_id,
                error=error,
            )
            return answer, model
        text = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        answer = Answer(
            text, citations, False, sources, prompt_tokens, output_tokens, 0, request_id, data=data
        )
        return answer, model

    # ------------------------------------------------------------------ helpers

    def _chat(self, messages: list[ChatMessage], json_schema: dict[str, Any] | None = None) -> ChatResult:
        return self.llm.chat(
            messages,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_output_tokens,
            json_schema=json_schema,
        )

    def _ledger(self, kind: str, request_id: str, payload: dict[str, Any]) -> None:
        if self.ledger is not None:
            self.ledger.append(kind, request_id, payload)

    def _record(self, **fields: Any) -> None:
        if self.log is not None:
            self.log.record(**fields)


# ---------------------------------------------------------------------- pure functions


_CITATION_ONLY = re.compile(r"^[\s\[\]\d,;.\-]*$")


def is_bare_citation(text: str) -> bool:
    """True when the reply is nothing but citation markers and punctuation, for example "[1]" or "[1] [3]"."""
    return bool(text) and bool(_CITATION_ONLY.match(text)) and "[" in text


def is_refusal(text: str) -> bool:
    """True when the model reply is the refusal sentence (citation markers and trailing punctuation
    aside), or opens with it."""
    core = _normalise(REFUSAL)
    return _normalise(text).startswith(core)


def _normalise(text: str) -> str:
    stripped = _CITATION_RE.sub("", text)
    return " ".join(stripped.lower().split()).strip(" .!")


def citation_numbers(text: str) -> list[int]:
    """Source numbers cited as [n] or [n, m], in order of first appearance, without repeats."""
    seen: list[int] = []
    for group in _CITATION_RE.findall(text):
        for part in group.split(","):
            n = int(part.strip())
            if n not in seen:
                seen.append(n)
    return seen


def make_citation(n: int, hit: ChunkHit) -> Citation:
    return Citation(
        n=n,
        chunk_id=hit.chunk_id,
        source=hit.source,
        title=hit.title,
        page=hit.page,
        heading=hit.heading,
        snippet=snippet(hit.text),
    )


def resolve_citations(text: str, sources: list[ChunkHit]) -> list[Citation]:
    """Map the [n] markers in `text` to sources. Numbers outside 1..len(sources) are dropped."""
    return [make_citation(n, sources[n - 1]) for n in citation_numbers(text) if 1 <= n <= len(sources)]


def top_citations(sources: list[ChunkHit], count: int = _TOP_CITATIONS_WHEN_UNMARKED) -> list[Citation]:
    """Citations for the first `count` sources, used when the model marked none."""
    return [make_citation(n, hit) for n, hit in enumerate(sources[:count], start=1)]


def parse_json_output(content: str, schema: dict[str, Any]) -> tuple[Any, str | None]:
    """Decode the model's JSON reply and validate it. Returns (data, None) on success or
    (None, error) describing what to fix."""
    text = content.strip()
    fenced = _FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        data = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None, "the reply was not JSON"
        try:
            data = json.loads(text[start : end + 1])
        except ValueError as exc:
            return None, f"the reply was not valid JSON ({exc.msg})"
    problems = validate(data, schema)
    if problems:
        return None, "; ".join(problems)
    return data, None
