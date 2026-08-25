"""Pure helpers behind the web routes: demo users and tag parsing, answer text to HTML with citation
chips, JSON shapes for the API, sparkline geometry for the admin trend, and the loopback test behind
the admin guard. Nothing here touches the app, the store or the model, so every function is unit
testable on its own.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import urlencode

from markupsafe import Markup, escape

from keel.agent.loop import AgentResult
from keel.answer.types import Answer, Citation, User

DEMO_USERS: dict[str, list[str]] = {
    "public": ["public"],
    "hr-officer": ["public", "hr"],
}
"""Users the question box offers. Identity is self-asserted in this demo, on loopback only; see docs/web.md
and `keel.web.app.trusted_identity`."""

DEFAULT_TAGS: tuple[str, ...] = ("public",)
DEFAULT_USER_ID = "public"

_MARKER_RE = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")
_TAG_SPLIT_RE = re.compile(r"[,\s]+")

SPARK_WIDTH = 240
SPARK_HEIGHT = 48
SPARK_PAD = 4


# ---------------------------------------------------------------------- users and tags


def parse_tags(raw: Any) -> list[str]:
    """Tags from a JSON list, a comma or space separated string, or a list of such strings: trimmed,
    in first-seen order, without repeats. Anything else yields an empty list."""
    pieces: list[str] = []
    if isinstance(raw, str):
        pieces = _TAG_SPLIT_RE.split(raw)
    elif isinstance(raw, (list, tuple, set)):
        for item in raw:
            pieces.extend(_TAG_SPLIT_RE.split(str(item)))
    seen: list[str] = []
    for piece in pieces:
        tag = piece.strip()
        if tag and tag not in seen:
            seen.append(tag)
    return seen


def resolve_user(user_id: Any, tags: Any = None) -> User:
    """The user a request runs as: a demo user's tags plus any extra tags supplied, or the supplied
    tags alone for an unknown user id, or `public` when nothing else applies."""
    uid = str(user_id or "").strip() or DEFAULT_USER_ID
    combined = list(DEMO_USERS.get(uid, []))
    for tag in parse_tags(tags):
        if tag not in combined:
            combined.append(tag)
    if not combined:
        combined = list(DEFAULT_TAGS)
    return User(user_id=uid, tags=combined)


def anonymous_user() -> User:
    """The user a request runs as when nothing trustworthy names one: `public`, carrying `public`."""
    return User(user_id=DEFAULT_USER_ID, tags=list(DEFAULT_TAGS))


def demo_user(user_id: Any) -> User:
    """The user a request runs as under the hosted demo: one of the demo users with its fixed tags, or
    `public` for any other id. Extra tags from the request play no part."""
    uid = str(user_id or "").strip()
    if uid in DEMO_USERS:
        return User(user_id=uid, tags=list(DEMO_USERS[uid]))
    return anonymous_user()


def proxy_user(user_id: Any, tags: Any) -> User:
    """The user an authenticating reverse proxy asserts through its identity headers: the tags exactly
    as given (the proxy is authoritative, demo users play no part), `public` when it names none."""
    uid = str(user_id or "").strip() or DEFAULT_USER_ID
    return User(user_id=uid, tags=parse_tags(tags) or list(DEFAULT_TAGS))


def user_json(user: User) -> dict[str, Any]:
    return {"user_id": user.user_id, "tags": list(user.tags)}


# ---------------------------------------------------------------------- citations


def source_url(chunk_id: int, user: User) -> str:
    """Link to the source viewer that carries the user's tags so the ACL check there matches."""
    return f"/source/{chunk_id}?" + urlencode({"user": user.user_id, "tags": ",".join(user.tags)})


def chip_label(citation: Citation) -> str:
    """`title · heading · p.N`, dropping the parts a citation lacks."""
    name = citation.title or citation.source.replace("\\", "/").rsplit("/", 1)[-1]
    parts = [name]
    if citation.heading:
        parts.append(citation.heading)
    if citation.page:
        parts.append(f"p.{citation.page}")
    return " · ".join(parts)


@dataclass
class Chip:
    """One citation rendered for the page."""

    n: int
    chunk_id: int
    label: str
    url: str
    snippet: str


def chips_for(citations: list[Citation], user: User) -> list[Chip]:
    return [
        Chip(
            n=c.n,
            chunk_id=c.chunk_id,
            label=chip_label(c),
            url=source_url(c.chunk_id, user),
            snippet=c.snippet,
        )
        for c in citations
    ]


def answer_html(text: str, citations: list[Citation], user: User) -> Markup:
    """The answer as safe HTML: the text is escaped first, then every `[n]` marker that names a
    citation becomes a chip linking to its source. Markers with no matching citation stay as text."""
    by_n = {c.n: c for c in citations}
    escaped = str(escape(text))

    def replace(match: re.Match[str]) -> str:
        numbers = [int(part) for part in match.group(1).split(",")]
        if not all(n in by_n for n in numbers):
            return match.group(0)
        chips = []
        for n in numbers:
            citation = by_n[n]
            url = escape(source_url(citation.chunk_id, user))
            title = escape(chip_label(citation))
            chips.append(f'<a class="cite" href="{url}" title="{title}">[{n}]</a>')
        return "".join(chips)

    return Markup(_MARKER_RE.sub(replace, escaped))


# ---------------------------------------------------------------------- JSON shapes


def answer_json(answer: Answer, user: User) -> dict[str, Any]:
    """The answer engine's result as the API returns it."""
    return {
        "request_id": answer.request_id,
        "mode": "answer",
        "user": user_json(user),
        "text": answer.text,
        "refused": bool(answer.refused),
        "citations": [c.to_dict() for c in answer.citations],
        "retrieved": [
            {
                "chunk_id": h.chunk_id,
                "score": round(float(h.score), 4),
                "title": h.title,
                "heading": h.heading,
                "page": h.page,
            }
            for h in answer.retrieved
        ],
        "prompt_tokens": answer.prompt_tokens,
        "output_tokens": answer.output_tokens,
        "latency_ms": answer.latency_ms,
        "data": answer.data,
        "error": answer.error,
    }


def agent_json(result: AgentResult, user: User) -> dict[str, Any]:
    """The agent loop's result as the API returns it."""
    return {
        "request_id": result.request_id,
        "mode": "agent",
        "user": user_json(user),
        "text": result.text,
        "refused": False,
        "steps": list(result.steps),
        "refused_tools": list(result.refused_tools),
        "prompt_tokens": result.prompt_tokens,
        "output_tokens": result.output_tokens,
        "latency_ms": result.latency_ms,
    }


def step_view(step: dict[str, Any]) -> dict[str, Any]:
    """One agent step flattened for the template: tool, arguments, a state word and the outcome."""
    decision = step.get("decision") or {}
    if "queued_id" in step:
        state, outcome = "queued", f"queued for approval #{step['queued_id']}"
    elif not decision.get("allowed", True):
        state, outcome = "refused", str(step.get("result") or f"refused: {decision.get('reason', '')}")
    else:
        state, outcome = "ran", str(step.get("result") or "")
    return {
        "tool": step.get("tool", ""),
        "arguments": step.get("arguments") or {},
        "reason": decision.get("reason", ""),
        "state": state,
        "outcome": outcome,
    }


# ---------------------------------------------------------------------- admin guard


def is_loopback(host: str) -> bool:
    """True for 127.0.0.0/8, ::1 and the name `localhost`; the addresses the admin page trusts."""
    name = (host or "").strip().lower()
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


# ---------------------------------------------------------------------- sparklines


@dataclass
class Sparkline:
    """One small chart for the admin trend, ready for the SVG template."""

    key: str
    label: str
    unit: str
    values: list[float | None]
    points: str = ""
    dots: list[tuple[float, float]] = field(default_factory=list)
    last: float | None = None
    peak: float | None = None
    present: bool = False

    @property
    def last_text(self) -> str:
        """The newest value with its unit, or a dot when the series is empty."""
        return f"{format_value(self.last)}{self.unit}" if self.last is not None else "·"


def format_value(value: float) -> str:
    """Whole numbers with thousands separators, other values with the digits they need."""
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:g}"


def sparkline_geometry(
    values: list[float | None], width: int = SPARK_WIDTH, height: int = SPARK_HEIGHT, pad: int = SPARK_PAD
) -> tuple[str, list[tuple[float, float]]]:
    """Polyline points and dot centres for the values that are present, scaled to the box.
    A single present value sits mid-height; a flat series draws a mid-height line."""
    present = [(i, v) for i, v in enumerate(values) if v is not None]
    if not present:
        return "", []
    n = len(values)
    lo = min(v for _, v in present)
    hi = max(v for _, v in present)
    span_x = width - 2 * pad
    span_y = height - 2 * pad
    dots: list[tuple[float, float]] = []
    for i, v in present:
        x = pad + (span_x * i / (n - 1) if n > 1 else span_x / 2)
        y = pad + (span_y * (1 - (v - lo) / (hi - lo)) if hi > lo else span_y / 2)
        dots.append((round(x, 1), round(y, 1)))
    points = " ".join(f"{x},{y}" for x, y in dots)
    return points, dots


def build_sparkline(key: str, label: str, unit: str, values: list[float | None]) -> Sparkline:
    points, dots = sparkline_geometry(values)
    present = [v for v in values if v is not None]
    return Sparkline(
        key=key,
        label=label,
        unit=unit,
        values=values,
        points=points,
        dots=dots,
        last=present[-1] if present else None,
        peak=max(present) if present else None,
        present=bool(present),
    )


def day_labels(days: int, today: date | None = None) -> list[str]:
    """ISO dates for the last `days` days ending today (UTC), oldest first."""
    end = today or datetime.now(UTC).date()
    return [(end - timedelta(days=days - 1 - i)).isoformat() for i in range(days)]


def trend_sparklines(
    daily: list[dict[str, Any]], days: int = 14, today: date | None = None
) -> list[Sparkline]:
    """Sparklines for the admin trend from `InferenceLog.daily_summary()` rows: requests per day,
    refusal rate, average latency, and mean judge groundedness and relevance. Days with no requests
    count zero requests and leave the rate and latency series empty for that day."""
    labels = day_labels(days, today)
    by_day = {str(row.get("day")): row for row in daily}
    requests: list[float | None] = []
    refusal_rate: list[float | None] = []
    latency: list[float | None] = []
    groundedness: list[float | None] = []
    relevance: list[float | None] = []
    for day in labels:
        row = by_day.get(day)
        count = float(row.get("requests") or 0) if row else 0.0
        requests.append(count)
        if row and count > 0:
            refusal_rate.append(round(100.0 * float(row.get("refused") or 0) / count, 1))
            avg = row.get("avg_latency_ms")
            latency.append(round(float(avg), 0) if avg is not None else None)
        else:
            refusal_rate.append(None)
            latency.append(None)
        for series, key in ((groundedness, "groundedness"), (relevance, "relevance")):
            value = row.get(key) if row else None
            series.append(round(float(value), 3) if value is not None else None)
    return [
        build_sparkline("requests", "Requests per day", "", requests),
        build_sparkline("refusal_rate", "Refusal rate", "%", refusal_rate),
        build_sparkline("latency", "Average latency", "ms", latency),
        build_sparkline("groundedness", "Mean groundedness (judge)", "", groundedness),
        build_sparkline("relevance", "Mean relevance (judge)", "", relevance),
    ]
