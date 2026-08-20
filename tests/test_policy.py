"""Policy at the tool boundary and the built-in tools' own guards."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import pytest

from keel.agent.policy import Decision, Policy
from keel.agent.tools import (
    AIRGAP_REFUSAL,
    ToolContext,
    ToolError,
    ToolRegistry,
    calculate,
    calculator_tool,
    check_select_query,
    create_ticket_tool,
    default_registry,
    http_get_tool,
    referenced_tables,
    search_docs_tool,
    sql_readonly_tool,
)
from keel.config import Settings
from keel.providers.base import ToolSpec
from tests.fakes import FakeRetriever, make_hit

POLICY_CONFIG = {
    "allowed_tools": ["search_docs", "calculator", "sql_readonly", "http_get", "create_ticket"],
    "write_tools_require_approval": True,
    "tool_arg_rules": {
        "sql_readonly": {"tables": ["users", "orders"], "max_rows": 50},
        "http_get": {"hosts": ["example.com"]},
    },
    "max_tool_calls_per_request": 6,
}


@pytest.fixture
def reporting_db(tmp_path: Path) -> Path:
    path = tmp_path / "reporting.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, total REAL)")
    conn.execute("CREATE TABLE secrets (id INTEGER PRIMARY KEY, code TEXT)")
    conn.executemany("INSERT INTO users (name) VALUES (?)", [(f"user{i}",) for i in range(120)])
    conn.executemany("INSERT INTO orders (user_id, total) VALUES (?, ?)", [(i, i * 1.5) for i in range(1, 11)])
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


# ---------------------------------------------------------------------- allowlist and approvals


def test_disallowed_tool_is_refused_with_a_reason(policy, caplog):
    with caplog.at_level(logging.WARNING, logger="keel.policy"):
        decision = policy.check("delete_everything", {})
    assert decision == Decision(False, "tool 'delete_everything' is outside this deployment's allowlist")
    assert decision.needs_approval is False
    assert "delete_everything" in caplog.text


def test_tool_missing_from_allowlist_is_refused_even_when_registered(registry):
    narrow = Policy({"allowed_tools": ["calculator"]}, registry=registry)
    assert narrow.check("calculator", {"expression": "1+1"}).allowed is True
    decision = narrow.check("http_get", {"url": "https://example.com/"})
    assert decision.allowed is False and "allowlist" in decision.reason


def test_write_tool_needs_approval(policy):
    decision = policy.check("create_ticket", {"title": "Printer", "body": "Out of toner"})
    assert decision.allowed is True
    assert decision.needs_approval is True
    assert "approve" in decision.reason


def test_write_flag_override_without_registry():
    bare = Policy({"allowed_tools": ["create_ticket"]})
    assert bare.check("create_ticket", {}, write=True).needs_approval is True
    assert bare.check("create_ticket", {}).needs_approval is False


def test_write_tools_run_directly_when_approval_is_switched_off(registry):
    relaxed = Policy({"allowed_tools": ["create_ticket"], "write_tools_require_approval": False}, registry=registry)
    decision = relaxed.check("create_ticket", {"title": "t", "body": "b"})
    assert decision.allowed is True and decision.needs_approval is False


def test_policy_loads_from_yaml(tmp_path: Path, registry):
    path = tmp_path / "policy.yaml"
    path.write_text(
        "allowed_tools: [calculator]\nmax_tool_calls_per_request: 2\n"
        "tool_arg_rules:\n  sql_readonly: {tables: [users]}\n",
        encoding="utf-8",
    )
    loaded = Policy.from_yaml(path, registry=registry)
    assert loaded.allowed_tools == ["calculator"]
    assert loaded.max_tool_calls_per_request == 2
    assert loaded.rules_for("sql_readonly") == {"tables": ["users"]}
    assert loaded.check("calculator", {"expression": "1"}).allowed is True


def test_defaults_are_safe():
    empty = Policy({})
    assert empty.allowed_tools == []
    assert empty.write_tools_require_approval is True
    assert empty.max_tool_calls_per_request == 6
    assert empty.check("calculator", {"expression": "1"}).allowed is False


# ---------------------------------------------------------------------- sql_readonly


def test_sql_policy_rejects_update(policy):
    decision = policy.check("sql_readonly", {"query": "UPDATE users SET name = 'x'"})
    assert decision.allowed is False
    assert decision.reason == "only SELECT statements are allowed"


def test_sql_policy_rejects_non_allowlisted_table(policy):
    decision = policy.check("sql_readonly", {"query": "SELECT code FROM secrets"})
    assert decision.allowed is False
    assert decision.reason == "table 'secrets' is outside the allowlist"


def test_sql_policy_rejects_semicolons_and_join_to_hidden_table(policy):
    assert policy.check("sql_readonly", {"query": "SELECT * FROM users; DROP TABLE users"}).allowed is False
    joined = policy.check("sql_readonly", {"query": "SELECT * FROM users u JOIN secrets s ON s.id = u.id"})
    assert joined.allowed is False and "secrets" in joined.reason


def test_sql_policy_accepts_select_on_allowed_table(policy):
    decision = policy.check(
        "sql_readonly", {"query": "-- top users\nSELECT u.name, o.total FROM users u, orders AS o WHERE o.user_id = u.id"}
    )
    assert decision == Decision(True, "allowed")


def test_sql_tool_caps_rows(registry):
    out = json.loads(registry.call("sql_readonly", {"query": "SELECT id, name FROM users ORDER BY id"}))
    assert out["columns"] == ["id", "name"]
    assert out["row_count"] == 50 and len(out["rows"]) == 50
    assert out["truncated"] is True
    assert out["rows"][0] == {"id": 1, "name": "user0"}

    small = json.loads(registry.call("sql_readonly", {"query": "SELECT count(*) AS n FROM orders"}))
    assert small["rows"] == [{"n": 10}] and small["truncated"] is False


def test_sql_tool_refuses_hidden_tables_even_inside_subqueries(registry):
    with pytest.raises(ToolError, match="outside the allowlist"):
        registry.call("sql_readonly", {"query": "SELECT (SELECT code FROM secrets) AS c FROM users"})
    with pytest.raises(ToolError, match="only SELECT"):
        registry.call("sql_readonly", {"query": "PRAGMA table_info(users)"})


def test_sql_tool_authorizer_blocks_reads_the_parser_misses(reporting_db):
    spec, fn = sql_readonly_tool(reporting_db, tables=["users"], max_rows=5)
    with pytest.raises(ToolError, match="query failed"):
        fn("SELECT * FROM users WHERE name IN (SELECT code FROM (secrets))")
    with pytest.raises(ToolError, match="query failed"):
        fn("SELECT * FROM users, (secrets)")
    with pytest.raises(ToolError, match="outside the allowlist"):
        fn("SELECT * FROM sqlite_master")


def test_sql_tool_is_read_only_at_the_connection(reporting_db):
    _, fn = sql_readonly_tool(reporting_db, tables=["users"], max_rows=5)
    assert json.loads(fn("SELECT count(*) AS n FROM users"))["rows"] == [{"n": 120}]
    check = sqlite3.connect(reporting_db)
    assert check.execute("SELECT count(*) FROM users").fetchone()[0] == 120
    check.close()


@pytest.mark.parametrize(
    "sql, tables",
    [
        ("SELECT * FROM users", ["users"]),
        ("select a.x from users a, orders b join items i on i.id = b.id", ["users", "orders", "items"]),
        ('SELECT * FROM "users" u LEFT JOIN [orders] o ON o.user_id = u.id', ["users", "orders"]),
        ("SELECT * FROM (SELECT * FROM users) t", ["users"]),
        ("SELECT 1", []),
    ],
)
def test_referenced_tables(sql, tables):
    assert referenced_tables(sql) == tables


def test_check_select_query_strips_comments():
    assert check_select_query("/* c */ SELECT 1 -- trailing", None) is None
    assert check_select_query("SELECT * FROM users", []) == "table 'users' is outside the allowlist"
    assert check_select_query("  DELETE FROM users", ["users"]) == "only SELECT statements are allowed"
    assert check_select_query("SELECT * FROM users;", ["users"]) == "semicolons are not allowed; send one statement"


# ---------------------------------------------------------------------- calculator


def test_calculator_evaluates_arithmetic(registry):
    assert registry.call("calculator", {"expression": "2*(3+4)"}) == "14"
    assert registry.call("calculator", {"expression": "1234*5678"}) == "7006652"
    assert registry.call("calculator", {"expression": "7 / 2"}) == "3.5"
    assert registry.call("calculator", {"expression": "-3 ** 2 % 5"}) == "1"  # Python semantics: -(3**2) % 5
    assert registry.call("calculator", {"expression": "(-3) ** 2"}) == "9"
    assert calculate("2 ** 10") == 1024


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",
        "os.system('dir')",
        "().__class__",
        "abs(-1)",
        "x + 1",
        "'a' * 3",
        "[1, 2]",
        "1 if 2 else 3",
        "lambda: 1",
        "2 ** 100000",
        "1 << 3",
    ],
)
def test_calculator_rejects_everything_but_arithmetic(expression):
    with pytest.raises(ToolError):
        calculate(expression)


def test_calculator_reports_division_by_zero_and_syntax_errors():
    with pytest.raises(ToolError, match="division by zero"):
        calculate("1/0")
    with pytest.raises(ToolError, match="parsed"):
        calculate("2 *")


# ---------------------------------------------------------------------- http_get and air-gap


def test_http_get_policy_refuses_under_airgap(registry):
    sealed = Policy(POLICY_CONFIG, registry=registry, airgap=True)
    decision = sealed.check("http_get", {"url": "https://example.com/page"})
    assert decision.allowed is False
    assert decision.reason == AIRGAP_REFUSAL


def test_http_get_tool_refuses_under_airgap_before_any_connection():
    _, fn = http_get_tool(["example.com"], airgap=True)
    with pytest.raises(ToolError, match="air-gap"):
        fn("https://example.com/")


def test_http_get_policy_and_tool_enforce_host_allowlist(policy):
    assert policy.check("http_get", {"url": "https://example.com/a"}).allowed is True
    denied = policy.check("http_get", {"url": "https://evil.example.net/a"})
    assert denied.allowed is False and "evil.example.net" in denied.reason
    assert policy.check("http_get", {"url": "ftp://example.com/a"}).allowed is False
    assert policy.check("http_get", {"url": "not a url"}).allowed is False

    _, fn = http_get_tool(["example.com"], airgap=False)
    with pytest.raises(ToolError, match="outside the allowlist"):
        fn("https://evil.example.net/a")


# ---------------------------------------------------------------------- registry and other tools


def test_registry_validates_arguments_and_reports_unknown_tools(registry):
    with pytest.raises(ToolError, match="unknown"):
        registry.call("teleport", {})
    with pytest.raises(ToolError, match="invalid arguments"):
        registry.call("calculator", {})
    with pytest.raises(ToolError, match="invalid arguments"):
        registry.call("calculator", {"expression": 5})
    assert registry.get("create_ticket").write is True
    assert registry.is_write("calculator") is False
    assert "calculator" in registry and "teleport" not in registry
    assert {s.name for s in registry.specs()} == {"search_docs", "calculator", "sql_readonly", "http_get", "create_ticket"}


def test_search_docs_uses_the_current_users_tags():
    retriever = FakeRetriever(
        [
            make_hit(1, "Band 2 needs three written quotes.", score=0.9),
            make_hit(2, "Band E pays up to $198,000. PELICAN-7741", score=0.95, tags=("hr",)),
        ]
    )
    registry = ToolRegistry()
    registry.register(*search_docs_tool(retriever))
    public = registry.call("search_docs", {"query": "quotes"}, ToolContext(user=None))
    assert "[1]" in public and "PELICAN" not in public
    assert retriever.calls[-1]["user_tags"] == ["public"]

    from keel.answer.types import User

    hr = registry.call("search_docs", {"query": "bands"}, ToolContext(user=User("hr1", ["public", "hr"])))
    assert "PELICAN" in hr and "[2]" in hr
    assert registry.call("search_docs", {"query": "x"}, ToolContext(user=User("u", ["nothing"]))) == "no matching passages"


def test_create_ticket_records_only_when_called():
    spec, fn = create_ticket_tool()
    assert spec.write is True
    assert fn(title="Printer", body="Toner").startswith("ticket created: Printer")


def test_custom_tool_registration_and_json_results():
    registry = ToolRegistry()
    spec = ToolSpec("weather", "Weather stub", {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})
    registry.register(spec, lambda city: {"city": city, "temp_c": 21})
    assert json.loads(registry.call("weather", {"city": "Canberra"})) == {"city": "Canberra", "temp_c": 21}
    _, calc = calculator_tool()
    assert calc(expression="3*3") == "9"
