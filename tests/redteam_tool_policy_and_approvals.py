"""Adversarial tests against tool policy and approvals.

Each test encodes one concrete attack: inputs, state, and the safe behaviour we expect. A test that
passes is evidence the control holds. A test marked ``xfail(strict=True, reason="FINDING: ...")``
records a control gap the builder missed: it asserts the *safe* behaviour, which does not hold today,
so it fails and stays green under xfail. When the implementation is fixed the test will pass, turn
into an XPASS, and fail the suite until the marker is removed. That is deliberate: the xfails are the
work order for the fix phase.

Targets: keel/agent/policy.py, keel/agent/tools.py, keel/agent/approvals.py, keel/agent/loop.py and
the admin approval routes in keel/web/app.py.
"""

from __future__ import annotations

import math
import sqlite3
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from keel.agent import (
    AgentLoop,
    ApprovalQueue,
    Policy,
    TicketBook,
    ToolContext,
    ToolError,
    ToolRegistry,
    default_registry,
)
from keel.agent.tools import (
    calculate,
    check_select_query,
    http_get_tool,
    sql_readonly_tool,
)
from keel.answer.types import User
from keel.config import Settings
from keel.db import connect
from keel.web import app as webapp
from tests.fakes import FakeLedger, FakeLLM, FakeLog, FakeRetriever, make_hit, tool_call_reply

POLICY_CONFIG: dict[str, Any] = {
    "allowed_tools": ["search_docs", "calculator", "sql_readonly", "http_get", "create_ticket"],
    "write_tools_require_approval": True,
    "tool_arg_rules": {
        "sql_readonly": {"tables": ["users", "orders"], "max_rows": 50},
        "http_get": {"hosts": ["example.com"]},
    },
    "max_tool_calls_per_request": 6,
}


# ---------------------------------------------------------------------- fixtures


@pytest.fixture
def reporting_db(tmp_path: Path) -> Path:
    path = tmp_path / "reporting.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, total REAL)")
    conn.execute("CREATE TABLE secrets (id INTEGER PRIMARY KEY, code TEXT)")
    conn.executemany("INSERT INTO users (name) VALUES (?)", [(f"user{i}",) for i in range(120)])
    conn.execute("INSERT INTO secrets (code) VALUES ('PELICAN-7741')")
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def registry(reporting_db: Path) -> ToolRegistry:
    return default_registry(
        Settings(airgap=False),
        retriever=FakeRetriever([make_hit(1, "Band 2 needs three written quotes.")]),
        sql_path=reporting_db,
        sql_tables=["users", "orders"],
        sql_max_rows=50,
        http_hosts=["example.com"],
    )


@pytest.fixture
def policy(registry: ToolRegistry) -> Policy:
    return Policy(POLICY_CONFIG, registry=registry, airgap=False)


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    connection = connect(tmp_path / "keel.db")
    yield connection
    connection.close()


def build_loop(
    llm: FakeLLM, registry: ToolRegistry, conn: sqlite3.Connection, config: dict[str, Any] | None = None
) -> tuple[AgentLoop, ApprovalQueue]:
    settings = Settings(temperature=0.0, max_output_tokens=200, airgap=False)
    policy = Policy(config or POLICY_CONFIG, registry=registry, airgap=settings.airgap)
    approvals = ApprovalQueue(conn, ledger=FakeLedger())
    loop = AgentLoop(llm, registry, policy, approvals, settings, ledger=FakeLedger(), log=FakeLog())
    return loop, approvals


# ====================================================================== calculator
# The calculator claims "numbers only" and evaluates a restricted AST. Attacks probe both the
# escape surface (names, calls, attributes) and resource use (exponent, nesting, magnitude).


def test_calculator_rejects_import_and_call() -> None:
    """CONTROL: __import__('os').system(...) is a Call node and must be refused."""
    with pytest.raises(ToolError, match="only numbers"):
        calculate("__import__('os').system('dir')")


def test_calculator_rejects_dunder_class_traversal() -> None:
    """CONTROL: the ().__class__.__bases__ sandbox-escape shape is refused (Call/Attribute nodes)."""
    with pytest.raises(ToolError):
        calculate("().__class__.__bases__[0].__subclasses__()")
    with pytest.raises(ToolError):
        calculate("(1).__class__")


def test_calculator_refuses_huge_right_exponent() -> None:
    """CONTROL: 9**9**9 has right operand 9**9 = 387420489 > the 4096 exponent cap, so it is refused
    before Python attempts the astronomically large power."""
    with pytest.raises(ToolError, match="exponent"):
        calculate("9**9**9")
    with pytest.raises(ToolError, match="exponent"):
        calculate("2 ** 100000")


def test_calculator_rejects_unicode_digits() -> None:
    """CONTROL: Arabic-Indic and full-width digit look-alikes do not parse as Python number tokens,
    so they are refused rather than silently coerced."""
    with pytest.raises(ToolError, match="parsed"):
        calculate("\u0661\u0662\u0663+\u0664")  # ١٢٣+٤
    with pytest.raises(ToolError, match="parsed"):
        calculate("\uff12+\uff12")  # ２+２


def test_calculator_deep_nesting_is_bounded_by_length_cap() -> None:
    """CONTROL: 5000-deep parentheses exceed the 500-char cap and are refused, so the recursive
    evaluator never blows the Python stack. Nesting that fits the cap stays under the recursion limit."""
    with pytest.raises(ToolError, match="too long"):
        calculate("(" * 5000 + "1" + ")" * 5000)
    assert calculate("(" * 200 + "1" + ")" * 200) == 1


def test_calculator_stacked_power_is_not_a_dos() -> None:
    """ATTACK (fixed): nest powers so no single exponent exceeds 4096 yet the result is astronomical.
    SAFE behaviour: the calculator bounds the total magnitude (checked from the base's bit length and
    the exponent before the power is computed) and refuses. The fast variant below would produce a
    ~2MB integer; the deeper ((2**4096)**4096)**4096 never returned before the fix."""
    with pytest.raises(ToolError):
        calculate("(2**4096)**4096")
    with pytest.raises(ToolError):
        calculate("((2**4096)**4096)**4096")


def test_calculator_does_not_return_complex() -> None:
    """ATTACK (fixed): (-8)**0.5 yields a complex number. SAFE behaviour: refuse, since the tool
    promises real numbers only."""
    with pytest.raises(ToolError):
        calculate("(-8)**0.5")


def test_calculator_stays_finite() -> None:
    """ATTACK (fixed): produce non-finite floats. SAFE behaviour: every accepted result is finite; an
    expression whose value would be inf or nan is refused with ToolError (the fix the finding
    prescribed), and a refusal hands the model nothing non-finite either."""
    for expr in ("1e999", "1e308*10", "1e999-1e999"):
        try:
            value = calculate(expr)
        except ToolError:
            continue  # refused: nothing non-finite reaches the model
        assert isinstance(value, (int, float)) and math.isfinite(value), f"{expr} -> {value}"


# ====================================================================== sql_readonly
# A read-only SELECT tool with a table allowlist, a row cap, a read-only URI, a statement-level
# authorizer and a progress-handler watchdog. Attacks probe the parser, the authorizer and resource use.


def test_sql_refuses_semicolon_and_trailing_ddl(policy: Policy) -> None:
    """CONTROL: a stacked 'SELECT ...; DROP TABLE ...' is refused for the semicolon."""
    decision = policy.check("sql_readonly", {"query": "SELECT * FROM users; DROP TABLE users"})
    assert decision.allowed is False
    assert "semicolon" in decision.reason


def test_sql_refuses_cte_prefix(policy: Policy) -> None:
    """CONTROL: 'WITH x AS (...) SELECT ...' does not begin with SELECT and is refused, which also
    blocks WITH RECURSIVE run-forever queries."""
    assert policy.check("sql_readonly", {"query": "WITH x AS (SELECT * FROM secrets) SELECT * FROM x"}).allowed is False
    assert policy.check("sql_readonly", {"query": "WITH x AS (SELECT 1) SELECT * FROM users"}).allowed is False


def test_sql_refuses_subquery_on_disallowed_table(policy: Policy) -> None:
    """CONTROL: a disallowed table hidden in a FROM subquery is still caught."""
    for query in (
        "SELECT * FROM (SELECT code FROM secrets) t",
        "SELECT (SELECT code FROM secrets) AS c FROM users",
    ):
        decision = policy.check("sql_readonly", {"query": query})
        assert decision.allowed is False and "secrets" in decision.reason


def test_sql_refuses_pragma_and_attach(policy: Policy) -> None:
    """CONTROL: PRAGMA and ATTACH do not begin with SELECT and are refused."""
    assert policy.check("sql_readonly", {"query": "PRAGMA table_info(users)"}).allowed is False
    assert policy.check("sql_readonly", {"query": "ATTACH DATABASE 'x.db' AS y"}).allowed is False


def test_sql_refuses_sqlite_master(policy: Policy) -> None:
    """CONTROL: the schema table sqlite_master is not on the allowlist, in either case."""
    assert policy.check("sql_readonly", {"query": "select * from sqlite_master"}).allowed is False
    assert policy.check("sql_readonly", {"query": "SELECT * FROM SQLITE_MASTER"}).allowed is False


def test_sql_refuses_comment_and_whitespace_tricks(policy: Policy) -> None:
    """CONTROL: an inline comment inside the SELECT keyword breaks the statement (refused), and a
    comment cannot smuggle a disallowed table past the check."""
    assert policy.check("sql_readonly", {"query": "SEL/**/ECT * FROM users"}).allowed is False
    assert policy.check("sql_readonly", {"query": "SELECT\n--\n* FROM secrets"}).allowed is False
    assert policy.check("sql_readonly", {"query": "SELECT * FROM/**/secrets"}).allowed is False


def test_sql_refuses_join_to_disallowed_table(policy: Policy) -> None:
    """CONTROL: reaching secrets by joining from an allowlisted table is caught."""
    decision = policy.check("sql_readonly", {"query": "SELECT * FROM users u JOIN secrets s ON s.id = u.id"})
    assert decision.allowed is False and "secrets" in decision.reason


def test_sql_refuses_union_exfiltration(policy: Policy) -> None:
    """CONTROL: a UNION that appends rows from a disallowed table (a row-cap and ACL bypass attempt)
    is refused because the second SELECT's table is off the allowlist."""
    decision = policy.check(
        "sql_readonly", {"query": "SELECT id, name FROM users UNION SELECT id, code FROM secrets"}
    )
    assert decision.allowed is False and "secrets" in decision.reason


def test_sql_authorizer_blocks_reads_the_parser_misses(reporting_db: Path) -> None:
    """CONTROL: the parenthesised-table trick 'FROM users, (secrets)' slips past the FROM-list parser
    but the statement authorizer denies the read of secrets at execution time."""
    _, fn = sql_readonly_tool(reporting_db, tables=["users"], max_rows=5)
    with pytest.raises(ToolError, match="query failed"):
        fn("SELECT * FROM users, (secrets)")


def test_sql_load_extension_is_blocked_at_the_connection(reporting_db: Path) -> None:
    """CONTROL WITH NOTE: check_select_query ALLOWS 'SELECT load_extension(...)' because the statement
    has no FROM tables to fail the allowlist. Execution is still blocked because Python's sqlite3
    ships with extension loading disabled, so the authorizer/engine returns 'not authorized'. The
    parser gap is covered only by that second layer; a deployment that ever calls
    enable_load_extension(True) would turn this into arbitrary code loading."""
    assert check_select_query("SELECT load_extension('foo')", ["users"]) is None  # parser lets it through
    _, fn = sql_readonly_tool(reporting_db, tables=["users"], max_rows=5)
    with pytest.raises(ToolError, match="not authorized"):
        fn("SELECT load_extension('foo')")


def test_sql_readonly_caps_single_value_size(reporting_db: Path) -> None:
    """ATTACK (fixed): allocate a huge value in a single-row SELECT. SAFE behaviour: refuse. The
    connection's SQLITE_LIMIT_LENGTH caps any one value at a megabyte inside the engine, zeroblob and
    randomblob are denied by the authorizer, and the serialised result is capped as well."""
    _, fn = sql_readonly_tool(reporting_db, tables=["users"], max_rows=50)
    with pytest.raises(ToolError):
        fn("SELECT hex(zeroblob(4000000)) AS b")
    with pytest.raises(ToolError):
        fn("SELECT randomblob(1000000000) AS b")
    with pytest.raises(ToolError):
        fn("SELECT hex(zeroblob(2000)) AS b")  # a modest allocation, still an allocation primitive


# ====================================================================== http_get
# Host-allowlisted GET, off entirely under air-gap. Attacks probe scheme confusion, host-parse
# tricks and redirect following.


def test_http_refuses_dangerous_schemes(policy: Policy) -> None:
    """CONTROL: file://, gopher://, ftp:// and javascript: are not http/https and are refused."""
    for url in ("file:///etc/passwd", "file://C:/Windows/win.ini", "gopher://example.com/", "ftp://example.com/", "javascript:alert(1)"):
        assert policy.check("http_get", {"url": url}).allowed is False


def test_http_userinfo_does_not_spoof_the_host(policy: Policy) -> None:
    """CONTROL: 'http://allowed@evil.com/' and 'http://example.com@evil.com/' resolve to host evil.com,
    which is off the allowlist. The userinfo segment cannot smuggle an allowed name."""
    for url in ("http://allowed@evil.com/", "http://example.com@evil.com/", "http://evil.com#@example.com/"):
        decision = policy.check("http_get", {"url": url})
        assert decision.allowed is False and "evil.com" in decision.reason


def test_http_refuses_ip_literals_and_metadata(policy: Policy) -> None:
    """CONTROL: an IPv6 literal, loopback and the cloud metadata IP are all off a hostname allowlist."""
    for url in ("http://[::1]/", "http://127.0.0.1/", "http://169.254.169.254/latest/meta-data/"):
        assert policy.check("http_get", {"url": url}).allowed is False


def test_http_refuses_suffix_lookalike_host(policy: Policy) -> None:
    """CONTROL: 'example.com.evil.com' is a different host and is refused (no suffix match)."""
    assert policy.check("http_get", {"url": "http://example.com.evil.com/"}).allowed is False


def test_http_does_not_follow_redirect_to_disallowed_host() -> None:
    """CONTROL: the tool builds its client with follow_redirects=False, so a 302 pointing at a host
    off the allowlist is returned as a bare 302, not chased. A local server issues the redirect; the
    tool must not fetch evil.example.net. DNS rebinding (the host resolving to a new address between
    the allowlist check and the socket connect) is not reproducible in a unit test and is noted in the
    report, not asserted here."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            self.send_response(302)
            self.send_header("Location", "http://evil.example.net/secret")
            self.end_headers()

        def log_message(self, *args: Any) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _, fn = http_get_tool(["127.0.0.1"], airgap=False, timeout=5)
        result = fn(f"http://127.0.0.1:{port}/redir")
    finally:
        server.shutdown()
    assert result.startswith("HTTP 302")
    assert "evil.example.net" not in result and "secret" not in result


# ====================================================================== registry and policy boundary


def test_registry_rejects_off_schema_arguments(registry: ToolRegistry) -> None:
    """CONTROL: extra keys (additionalProperties: false) and wrong types are refused before the tool
    body runs."""
    with pytest.raises(ToolError, match="invalid arguments"):
        registry.call("calculator", {"expression": "1+1", "extra": "x"})
    with pytest.raises(ToolError, match="invalid arguments"):
        registry.call("calculator", {"expression": 5})
    with pytest.raises(ToolError, match="invalid arguments"):
        registry.call("calculator", {})


def test_policy_is_case_and_whitespace_sensitive_on_tool_names(policy: Policy) -> None:
    """CONTROL: 'Calculator', 'CALCULATOR' and 'calculator ' (trailing space) are not the allowlisted
    name and are refused. The allowlist does not fold case or strip, so a model cannot dodge it by
    perturbing the tool name."""
    assert policy.check("calculator", {"expression": "1+1"}).allowed is True
    for name in ("Calculator", "CALCULATOR", "calculator ", " calculator", "calculator\t"):
        assert policy.check(name, {"expression": "1+1"}).allowed is False


def test_policy_missing_allowlist_denies_every_tool() -> None:
    """CONTROL: a config with no allowed_tools key denies all tools (fail closed), rather than
    defaulting to allow-all."""
    empty = Policy({})
    assert empty.allowed_tools == []
    assert empty.check("calculator", {"expression": "1"}).allowed is False
    assert empty.check("create_ticket", {"title": "t", "body": "b"}).allowed is False


# ====================================================================== agent loop budget


def test_many_tool_calls_in_one_turn_stay_within_budget(registry: ToolRegistry, conn: sqlite3.Connection) -> None:
    """CONTROL: max_tool_calls_per_request counts every call across a single assistant turn, not just
    across turns. A turn that emits ten calculator calls under a budget of three executes exactly
    three and refuses the rest, so a model cannot bypass the budget by batching calls into one turn."""
    reply = tool_call_reply("calculator", {"expression": "1+1"}, call_id="c0")
    for i in range(1, 10):
        reply.tool_calls.append({"id": f"c{i}", "name": "calculator", "arguments": {"expression": "1+1"}})
    llm = FakeLLM([reply, "done"])
    loop, _ = build_loop(llm, registry, conn, {**POLICY_CONFIG, "max_tool_calls_per_request": 3})

    result = loop.run("spam calculator", User("u1", ["public"]))

    executed = [s for s in result.steps if s.get("result") == "2"]
    refused = [s for s in result.steps if str(s.get("result", "")).startswith("refused: the tool call budget")]
    assert len(executed) == 3
    assert len(refused) == 7


def test_unregistered_but_allowlisted_tool_is_refused(registry: ToolRegistry, conn: sqlite3.Connection) -> None:
    """CONTROL: a tool named in the allowlist but absent from the registry is refused as unknown, not
    executed."""
    llm = FakeLLM([tool_call_reply("ghost", {}), "done"])
    loop, _ = build_loop(llm, registry, conn, {**POLICY_CONFIG, "allowed_tools": ["calculator", "ghost"]})
    result = loop.run("q", User("u1", ["public"]))
    assert result.steps[0]["result"] == "refused: tool 'ghost' is unknown"


def test_write_tool_emitted_twice_queues_two_rows_and_runs_neither(conn: sqlite3.Connection) -> None:
    """CONTROL: two create_ticket calls in one turn create two independent pending approvals and
    execute nothing. Double-queueing does not become double execution; each row still needs its own
    human decision."""
    book = TicketBook()
    registry_with_book = default_registry(Settings(airgap=False), tickets=book)
    reply = tool_call_reply("create_ticket", {"title": "A", "body": "one"}, call_id="t0")
    reply.tool_calls.append({"id": "t1", "name": "create_ticket", "arguments": {"title": "B", "body": "two"}})
    llm = FakeLLM([reply, "both queued"])
    loop, approvals = build_loop(llm, registry_with_book, conn)

    result = loop.run("raise two tickets", User("u1", ["public"]))

    queued = [s for s in result.steps if "queued_id" in s]
    assert len(queued) == 2
    assert len({s["queued_id"] for s in queued}) == 2
    assert len(approvals.list("pending")) == 2
    assert book.created == []


# ====================================================================== approval queue


@pytest.fixture
def tickets() -> TicketBook:
    return TicketBook()


def test_double_approve_and_double_execute_are_refused(conn: sqlite3.Connection) -> None:
    """CONTROL: a call can be approved once and executed once. Re-deciding or re-executing raises,
    and the write side effect happens a single time."""
    book = TicketBook()
    registry = default_registry(Settings(airgap=False), tickets=book)
    queue = ApprovalQueue(conn, ledger=FakeLedger())
    approval_id = queue.enqueue("req-1", "create_ticket", {"title": "Printer", "body": "Toner"})

    queue.decide(approval_id, approved=True, by="blake")
    with pytest.raises(ValueError, match="approved"):
        queue.decide(approval_id, approved=True, by="blake")  # second approve

    queue.execute(approval_id, registry, ToolContext())
    with pytest.raises(ValueError, match="executed"):
        queue.execute(approval_id, registry, ToolContext())  # second execute
    assert book.created == [{"title": "Printer", "body": "Toner"}]  # ran exactly once


def test_execute_of_a_tool_removed_from_the_registry_fails_closed(conn: sqlite3.Connection) -> None:
    """CONTROL: if the tool named on an approved row is no longer registered, execution returns an
    error string rather than crashing or running something else."""
    registry = default_registry(Settings(airgap=False))  # no create_ticket… but has calculator
    queue = ApprovalQueue(conn, ledger=FakeLedger())
    approval_id = queue.enqueue("req-2", "vanished_tool", {"x": 1})
    queue.decide(approval_id, approved=True, by="blake")
    result = queue.execute(approval_id, registry, ToolContext())
    assert result.startswith("error:") and "unknown" in result


def test_execute_re_checks_current_policy(conn: sqlite3.Connection) -> None:
    """ATTACK (fixed): queue a write call, then tighten the policy to forbid that tool, then approve
    and execute. SAFE behaviour: the now-forbidden tool must not run, so no ticket is recorded. The
    queue re-checks the policy bound to the registry (or one passed to execute) and stores the
    refusal as the row's result."""
    book = TicketBook()
    registry = default_registry(Settings(airgap=False), tickets=book)
    queue = ApprovalQueue(conn, ledger=FakeLedger())
    approval_id = queue.enqueue("req-3", "create_ticket", {"title": "T", "body": "B"})
    queue.decide(approval_id, approved=True, by="blake")

    live_policy = Policy({"allowed_tools": ["calculator"]}, registry=registry)  # create_ticket now denied
    assert live_policy.check("create_ticket", {"title": "T", "body": "B"}).allowed is False

    result = queue.execute(approval_id, registry, ToolContext())
    assert book.created == []  # nothing the live policy denies should have run
    assert result.startswith("refused:") and "allowlist" in result
    assert queue.get(approval_id)["status"] == "executed"

    # An explicit policy handed to execute() wins over the bound one, in both directions.
    second = queue.enqueue("req-4", "create_ticket", {"title": "T2", "body": "B2"})
    queue.decide(second, approved=True, by="blake")
    permissive = Policy({"allowed_tools": ["create_ticket"]})
    assert queue.execute(second, registry, ToolContext(), policy=permissive).startswith("ticket created")
    assert book.created == [{"title": "T2", "body": "B2"}]


# ====================================================================== admin approval routes


@pytest.fixture
def admin_client() -> Iterator[TestClient]:
    """A TestClient over the real app with a lightweight loopback context, so require_admin passes
    without building the model stack. Only the fields the id-validation paths touch are present."""

    class _Settings:
        host = "127.0.0.1"

    class _Approvals:
        def get(self, approval_id: int) -> None:
            return None

        def list(self, status: str | None = None) -> list[dict[str, Any]]:
            return []

    class _Ctx:
        settings = _Settings()
        approvals = _Approvals()

    saved = getattr(webapp.app.state, "ctx", None)
    webapp.app.state.ctx = _Ctx()
    webapp.app.state.ctx_owned = False
    with TestClient(webapp.app) as client:
        yield client
    webapp.app.state.ctx = saved


def test_approval_route_rejects_non_integer_id(admin_client: TestClient) -> None:
    """CONTROL: a non-integer approval id is rejected by path validation (422), never reaching the
    queue. A well-formed but unknown id is a clean 404."""
    assert admin_client.post("/admin/approvals/abc/approve").status_code == 422
    assert admin_client.post("/admin/approvals/3.5/approve").status_code == 422
    assert admin_client.post("/admin/approvals/abc/reject").status_code == 422
    assert admin_client.post("/admin/approvals/999999/approve", headers={"accept": "application/json"}).status_code == 404
