"""Typed tools and the registry the agent loop calls them through.

Built-ins: `search_docs` (retrieval under the current user's ACL tags), `calculator` (arithmetic
via the ast, nothing else), `sql_readonly` (SELECT-only SQLite with a table allowlist, a row cap
and a read-only authorizer), `http_get` (host allowlist, off entirely under air-gap) and
`create_ticket`, a write tool that only ever runs through the approval queue.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import operator
import re
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from keel.answer.prompts import snippet, source_label
from keel.answer.schema import validate
from keel.answer.types import User
from keel.providers.base import ChunkHit, ToolSpec
from keel.retrieval.hybrid import allowed


class ToolError(Exception):
    """A tool declined or failed. The message is safe to hand back to the model."""


@dataclass
class ToolContext:
    """Per-request facts a tool may need, passed to functions that declare a `context` parameter."""

    user: User | None = None
    request_id: str = ""


@dataclass
class _Registration:
    spec: ToolSpec
    fn: Callable[..., Any]
    wants_context: bool


class ToolRegistry:
    """Holds ToolSpecs with their implementations and runs calls with argument validation.

    `policy` is the Policy last bound to this registry (`Policy(config, registry=...)` sets it). The
    agent loop checks its own policy before every call; the approval queue re-checks this one when it
    executes an approved call later, so an approval never outlives the policy that permitted it.
    """

    def __init__(self) -> None:
        self._tools: dict[str, _Registration] = {}
        self.policy: Any = None

    def register(self, spec: ToolSpec, fn: Callable[..., Any]) -> None:
        wants_context = "context" in inspect.signature(fn).parameters
        self._tools[spec.name] = _Registration(spec, fn, wants_context)

    def get(self, name: str) -> ToolSpec | None:
        reg = self._tools.get(name)
        return reg.spec if reg else None

    def specs(self) -> list[ToolSpec]:
        return [reg.spec for reg in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def is_write(self, name: str) -> bool:
        spec = self.get(name)
        return bool(spec and spec.write)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def call(self, name: str, arguments: dict[str, Any], context: ToolContext | None = None) -> str:
        """Validate `arguments` against the tool's schema and run it. Raises ToolError for an
        unknown tool, invalid arguments or a failure inside the tool."""
        reg = self._tools.get(name)
        if reg is None:
            raise ToolError(f"tool '{name}' is unknown")
        problems = validate(arguments, reg.spec.parameters)
        if problems:
            raise ToolError("invalid arguments: " + "; ".join(problems))
        kwargs = dict(arguments)
        if reg.wants_context:
            kwargs["context"] = context or ToolContext()
        try:
            result = reg.fn(**kwargs)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc
        return result if isinstance(result, str) else json.dumps(result, default=str)


# ---------------------------------------------------------------------- search_docs

SEARCH_DOCS_SPEC = ToolSpec(
    name="search_docs",
    description="Search the document corpus the current user may read. Returns numbered passages.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "What to look for, in plain words."}},
        "required": ["query"],
        "additionalProperties": False,
    },
)


def search_docs_tool(retriever: Any, max_hits: int = 5, snippet_chars: int = 400) -> tuple[ToolSpec, Callable[..., str]]:
    """search_docs bound to a retriever. Passages come back numbered [1]..[n]."""

    def search_docs(query: str, *, context: ToolContext) -> str:
        tags = list(context.user.tags) if context.user else ["public"]
        # The retriever filters by ACL and quarantine; the same test again here keeps a leaky
        # retriever from handing the model a passage the user may not read.
        entitled = frozenset(tags)
        hits: list[ChunkHit] = [h for h in retriever.retrieve(query, tags) if allowed(h, entitled)][:max_hits]
        if not hits:
            return "no matching passages"
        return "\n\n".join(
            f"[{n}] {source_label(h)}\n{snippet(h.text, snippet_chars)}" for n, h in enumerate(hits, start=1)
        )

    return SEARCH_DOCS_SPEC, search_docs


# ---------------------------------------------------------------------- calculator

CALCULATOR_SPEC = ToolSpec(
    name="calculator",
    description="Evaluate an arithmetic expression with + - * / ** % and parentheses. Numbers only.",
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "For example 2*(3+4)"}},
        "required": ["expression"],
        "additionalProperties": False,
    },
)

_BINARY_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_EXPONENT = 4096
_MAX_EXPRESSION_CHARS = 500
# Integers are bounded by size as well as by exponent: nested powers keep every exponent under the cap
# while the base grows, and ((2**4096)**4096)**4096 never returns. The bound is checked before a power
# is computed (from the base's bit length and the exponent) and after every operation.
_MAX_RESULT_BITS = 65_536
_TOO_LARGE = "result is too large"


def calculate(expression: str) -> int | float:
    """Evaluate arithmetic on numbers only. Names, calls, attributes and every other node are
    rejected with ToolError; so are results that are not finite real numbers."""
    if len(expression) > _MAX_EXPRESSION_CHARS:
        raise ToolError("expression is too long")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"expression could not be parsed: {exc.msg}") from exc
    try:
        return _eval_node(tree.body)
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
    except OverflowError as exc:
        raise ToolError(_TOO_LARGE) from exc


def _checked(value: Any) -> int | float:
    """Refuse anything the calculator must never hand back: complex numbers, inf, nan, oversized ints."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError("result is not a real number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ToolError(_TOO_LARGE if math.isinf(value) else "result is undefined")
    if isinstance(value, int) and value.bit_length() > _MAX_RESULT_BITS:
        raise ToolError(_TOO_LARGE)
    return value


def _eval_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return _checked(node.value)  # 1e999 parses to inf; refuse it at the source
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _checked(_UNARY_OPS[type(node.op)](_eval_node(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > _MAX_EXPONENT:
                raise ToolError(f"exponent above {_MAX_EXPONENT} is out of range")
            if isinstance(left, int) and isinstance(right, int) and right > 0:
                if abs(left).bit_length() * right > _MAX_RESULT_BITS:
                    raise ToolError(_TOO_LARGE)
        return _checked(_BINARY_OPS[type(node.op)](left, right))
    raise ToolError(f"only numbers, + - * / ** % and parentheses are allowed (found {type(node).__name__})")


def calculator_tool() -> tuple[ToolSpec, Callable[..., str]]:
    def calculator(expression: str) -> str:
        value = calculate(expression)
        if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
            value = int(value)
        return str(value)

    return CALCULATOR_SPEC, calculator


# ---------------------------------------------------------------------- sql_readonly

SQL_READONLY_SPEC = ToolSpec(
    name="sql_readonly",
    description="Run one read-only SELECT against the reporting database. Only allowlisted tables are readable.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "A single SELECT statement."}},
        "required": ["query"],
        "additionalProperties": False,
    },
)

_SQL_TOKEN_RE = re.compile(r'"[^"]*"|`[^`]*`|\[[^\]]*\]|\'[^\']*\'|[A-Za-z_][A-Za-z0-9_.]*|\S')
_SQL_KEYWORDS = {
    "as", "cross", "except", "from", "full", "group", "having", "inner", "intersect", "join", "left",
    "limit", "natural", "on", "order", "outer", "right", "select", "union", "using", "where", "window",
}
_SQL_STEP_BUDGET = 2000  # progress-handler calls, each after 10000 VM steps
_SQL_MAX_VALUE_BYTES = 1_000_000  # SQLITE_LIMIT_LENGTH: the largest string or blob one value may hold
_SQL_MAX_RESULT_CHARS = 2_000_000  # the serialised result handed back to the model
# Functions with no place in a reporting query: allocation primitives and extension loading.
_SQL_DENIED_FUNCTIONS = frozenset({"zeroblob", "randomblob", "load_extension"})


def strip_sql_comments(sql: str) -> str:
    """Remove -- and /* */ comments and surrounding whitespace."""
    without_block = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    without_line = re.sub(r"--[^\n]*", " ", without_block)
    return without_line.strip()


def referenced_tables(sql: str) -> list[str]:
    """Identifiers that follow FROM (comma lists included) and JOIN, quotes removed."""
    tokens = _SQL_TOKEN_RE.findall(sql)
    tables: list[str] = []
    i = 0
    while i < len(tokens):
        keyword = tokens[i].lower()
        if keyword not in ("from", "join"):
            i += 1
            continue
        j = i + 1
        while j < len(tokens):
            name = tokens[j]
            if not _is_identifier(name) or name.lower() in _SQL_KEYWORDS:
                break  # a subquery or stray punctuation; an inner FROM is visited later in the scan
            tables.append(_unquote_identifier(name))
            j += 1
            if j < len(tokens) and tokens[j].lower() == "as":
                j += 1
            if j < len(tokens) and _is_identifier(tokens[j]) and tokens[j].lower() not in _SQL_KEYWORDS:
                j += 1  # alias
            if keyword == "join" or j >= len(tokens) or tokens[j] != ",":
                break
            j += 1  # past the comma, on to the next table in the FROM list
        i = j
    return tables


def _unquote_identifier(token: str) -> str:
    if len(token) >= 2 and token[0] in "\"`[" and token[-1] in "\"`]":
        return token[1:-1]
    return token


def _is_identifier(token: str) -> bool:
    return bool(re.match(r'^(?:"[^"]*"|`[^`]*`|\[[^\]]*\]|[A-Za-z_][A-Za-z0-9_.]*)$', token))


def check_select_query(sql: str, allowed_tables: Iterable[str] | None) -> str | None:
    """Return the reason a query is refused, or None when it is a single SELECT over allowed
    tables. `allowed_tables=None` skips the table check."""
    text = strip_sql_comments(sql)
    if not re.match(r"select\b", text, re.IGNORECASE):
        return "only SELECT statements are allowed"
    if ";" in text:
        return "semicolons are not allowed; send one statement"
    if allowed_tables is None:
        return None
    allowed = {t.lower() for t in allowed_tables}
    for table in referenced_tables(text):
        if table.lower() not in allowed:
            return f"table '{table}' is outside the allowlist"
    return None


def sql_readonly_tool(
    db_path: str | Path, tables: Iterable[str], max_rows: int = 50
) -> tuple[ToolSpec, Callable[..., str]]:
    """sql_readonly bound to one SQLite file, an allowlist of tables and a row cap."""
    allowed = {t.lower() for t in tables}
    path = Path(db_path)

    def authorizer(action: int, arg1: str | None, arg2: str | None, dbname: str | None, source: str | None) -> int:
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION:
            return sqlite3.SQLITE_DENY if (arg2 or "").lower() in _SQL_DENIED_FUNCTIONS else sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ and (arg1 or "").lower() in allowed:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    def sql_readonly(query: str) -> str:
        problem = check_select_query(query, allowed)
        if problem:
            raise ToolError(problem)
        if not path.exists():
            raise ToolError("the reporting database is unavailable")
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            conn.set_authorizer(authorizer)
            # One value may hold at most _SQL_MAX_VALUE_BYTES: hex(zeroblob(1e9)) fails inside the
            # engine before it allocates, rather than after a gigabyte and half a minute.
            conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, _SQL_MAX_VALUE_BYTES)
            budget = {"left": _SQL_STEP_BUDGET}

            def watchdog() -> int:
                budget["left"] -= 1
                return 1 if budget["left"] < 0 else 0

            conn.set_progress_handler(watchdog, 10000)
            try:
                cursor = conn.execute(strip_sql_comments(query))
                rows = cursor.fetchmany(max_rows + 1)
            except sqlite3.Error as exc:
                raise ToolError(f"query failed: {exc}") from exc
            columns = [d[0] for d in cursor.description or []]
        finally:
            conn.close()
        truncated = len(rows) > max_rows
        rows = rows[:max_rows]
        payload = {
            "columns": columns,
            "rows": [dict(zip(columns, row, strict=True)) for row in rows],
            "row_count": len(rows),
            "truncated": truncated,
        }
        text = json.dumps(payload, default=str)
        if len(text) > _SQL_MAX_RESULT_CHARS:
            raise ToolError("result is too large to hand back; select fewer or narrower columns")
        return text

    return SQL_READONLY_SPEC, sql_readonly


# ---------------------------------------------------------------------- http_get

HTTP_GET_SPEC = ToolSpec(
    name="http_get",
    description="Fetch a web page or API response by GET. Only allowlisted hosts are reachable.",
    parameters={
        "type": "object",
        "properties": {"url": {"type": "string", "description": "A full http or https URL."}},
        "required": ["url"],
        "additionalProperties": False,
    },
)

AIRGAP_REFUSAL = "air-gap mode is on: outbound HTTP stays off"


def host_problem(url: str, allowed_hosts: Iterable[str] | None) -> str | None:
    """Return why `url` is refused, or None when its host is on the allowlist.
    `allowed_hosts=None` skips the host check; an empty list refuses every host."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return "url must be a full http or https address"
    if allowed_hosts is None:
        return None
    if parts.hostname.lower() not in {h.lower() for h in allowed_hosts}:
        return f"host '{parts.hostname}' is outside the allowlist"
    return None


def http_get_tool(
    hosts: Iterable[str], *, airgap: bool = False, timeout: float = 10.0, max_chars: int = 4000
) -> tuple[ToolSpec, Callable[..., str]]:
    """http_get bound to a host allowlist. Under air-gap the tool refuses before any connection."""
    allowed = [h.lower() for h in hosts]

    def http_get(url: str) -> str:
        if airgap:
            raise ToolError(AIRGAP_REFUSAL)
        problem = host_problem(url, allowed)
        if problem:
            raise ToolError(problem)
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            response = client.get(url)
        return f"HTTP {response.status_code}\n{response.text[:max_chars]}"

    return HTTP_GET_SPEC, http_get


# ---------------------------------------------------------------------- create_ticket (write)

CREATE_TICKET_SPEC = ToolSpec(
    name="create_ticket",
    description="Create a support ticket. This is a write action: it waits for a person to approve it.",
    parameters={
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short ticket title."},
            "body": {"type": "string", "description": "What needs doing and why."},
        },
        "required": ["title", "body"],
        "additionalProperties": False,
    },
    write=True,
)


@dataclass
class TicketBook:
    """Where the example write tool records tickets once a person has approved them."""

    created: list[dict[str, str]] = field(default_factory=list)


def create_ticket_tool(book: TicketBook | None = None) -> tuple[ToolSpec, Callable[..., str]]:
    store = book or TicketBook()

    def create_ticket(title: str, body: str) -> str:
        store.created.append({"title": title, "body": body})
        return f"ticket created: {title} (#{len(store.created)})"

    return CREATE_TICKET_SPEC, create_ticket


# ---------------------------------------------------------------------- assembly


def default_registry(
    settings: Any,
    *,
    retriever: Any = None,
    sql_path: str | Path | None = None,
    sql_tables: Iterable[str] = (),
    sql_max_rows: int = 50,
    http_hosts: Iterable[str] = (),
    tickets: TicketBook | None = None,
) -> ToolRegistry:
    """The built-in tool set. search_docs needs a retriever and sql_readonly a database path;
    each is left out when its dependency is absent."""
    registry = ToolRegistry()
    registry.register(*calculator_tool())
    if retriever is not None:
        registry.register(*search_docs_tool(retriever))
    if sql_path is not None:
        registry.register(*sql_readonly_tool(sql_path, sql_tables, sql_max_rows))
    registry.register(*http_get_tool(http_hosts, airgap=bool(getattr(settings, "airgap", False))))
    registry.register(*create_ticket_tool(tickets))
    return registry
