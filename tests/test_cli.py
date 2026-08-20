"""CLI tests through typer's CliRunner. Every command runs against a temporary data directory with a
scripted FakeLLM standing in for llama-server; embeddings (fastembed, cached) and the SQLite index are
real. Nothing here touches the network."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from click.testing import Result
from typer.testing import CliRunner

from keel.answer.prompts import REFUSAL
from keel.answer.types import Citation
from keel.cli import app, format_citation
from keel.providers.factory import build_context
from keel.providers.local_index import SqliteVectorIndex
from keel.safety.ledger import verify_file
from tests.conftest import REPO_ROOT
from tests.fakes import FakeLLM, tool_call_reply

runner = CliRunner()

Invoke = Callable[..., Result]

PROCUREMENT_QUESTION = "How many written quotes does a $20,000 purchase need at Northbank Council?"
RESTRICTED_QUESTION = "What is the confidential review code for the 2026 pay round?"


# ----------------------------------------------------------------------------- fixtures


@pytest.fixture
def fake_llm() -> FakeLLM:
    """The scripted model every command in a test talks to. Tests append responses before invoking."""
    return FakeLLM()


@pytest.fixture
def cli(settings, fake_llm: FakeLLM, embedder, monkeypatch: pytest.MonkeyPatch) -> Invoke:
    """Invoke the CLI in-process against the temporary data directory.

    `keel.providers.factory._local_providers` is replaced so `build_context()` wires the FakeLLM, the
    real fastembed embedder, a real SqliteVectorIndex and no reranker; no server is needed.
    """

    def local_providers(settings: Any, conn: Any) -> tuple[Any, Any, Any, None]:
        return fake_llm, embedder, SqliteVectorIndex(conn, embed_model=embedder.name), None

    monkeypatch.setattr("keel.providers.factory._local_providers", local_providers)

    def invoke(*args: str) -> Result:
        return runner.invoke(app, list(args))

    return invoke


def ok(result: Result) -> Result:
    """Assert a zero exit code with the output (and any exception) in the failure message."""
    assert result.exit_code == 0, f"exit {result.exit_code}\n{result.output}\n{result.exception!r}"
    return result


def ingest_fixtures(cli: Invoke, fixture_manifest: Path) -> Result:
    return ok(cli("ingest", "--manifest", str(fixture_manifest)))


def json_lines(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ----------------------------------------------------------------------------- entry points


@pytest.mark.parametrize("module", ["keel", "keel.cli"])
def test_python_dash_m_entry_points_print_help(module: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    for command in (
        "ingest",
        "ask",
        "agent",
        "approvals",
        "verify-ledger",
        "status",
        "serve",
        "eval",
        "export-log",
    ):
        assert command in proc.stdout


def test_no_arguments_prints_help(cli: Invoke) -> None:
    result = cli()
    assert "Usage" in result.output
    assert "ingest" in result.output


def test_profile_option_rejects_unknown_profile(cli: Invoke) -> None:
    result = cli("--profile", "bogus", "status")
    assert result.exit_code == 2
    assert "profile must be one of local, azure, aws" in result.output


def test_data_dir_option_points_the_store_elsewhere(cli: Invoke, tmp_path: Path) -> None:
    other = tmp_path / "elsewhere"
    result = ok(cli("--data-dir", str(other), "status"))
    assert f"data dir: {other.resolve()}" in result.output
    assert (other / "keel.db").exists()


# ----------------------------------------------------------------------------- ingest


def test_ingest_manifest_reports_five_documents_then_duplicates(cli: Invoke, fixture_manifest: Path) -> None:
    first = ingest_fixtures(cli, fixture_manifest)
    assert sum(1 for line in first.output.splitlines() if line.startswith("ingested ")) == 5
    assert "5 documents: 5 new, 0 duplicate;" in first.output
    assert "1 quarantined" in first.output  # the planted injection fixture

    second = ingest_fixtures(cli, fixture_manifest)
    assert sum(1 for line in second.output.splitlines() if line.startswith("duplicate ")) == 5
    assert "5 documents: 0 new, 5 duplicate; 0 chunks added; 0 quarantined" in second.output


def test_ingest_single_file_with_title_and_tags(cli: Invoke, tmp_path: Path, conn) -> None:
    doc = tmp_path / "leave.md"
    doc.write_text("# Leave\n\nStaff get four weeks of annual leave each year.\n", encoding="utf-8")
    result = ok(cli("ingest", str(doc), "--title", "Leave Policy", "--tags", "hr, staff"))
    assert "1 documents: 1 new, 0 duplicate;" in result.output
    row = conn.execute("SELECT title, acl_tags FROM documents").fetchone()
    assert row["title"] == "Leave Policy"
    assert json.loads(row["acl_tags"]) == ["hr", "staff"]


def test_ingest_without_sources_or_manifest_is_a_usage_error(cli: Invoke) -> None:
    result = cli("ingest")
    assert result.exit_code == 2
    assert "--manifest" in result.output


def test_ingest_reports_a_missing_file_and_continues(cli: Invoke, tmp_path: Path) -> None:
    good = tmp_path / "good.txt"
    good.write_text("The office opens at nine.\n", encoding="utf-8")
    result = cli("ingest", str(tmp_path / "missing.txt"), str(good))
    assert result.exit_code == 1
    assert "failed" in result.output and "missing.txt" in result.output
    assert "1 documents: 1 new" in result.output


def test_ingest_judge_flag_asks_the_model_about_each_chunk(
    cli: Invoke, fake_llm: FakeLLM, tmp_path: Path
) -> None:
    doc = tmp_path / "hours.txt"
    doc.write_text("The office opens at nine and closes at five.\n", encoding="utf-8")
    fake_llm.responses.append("no, plain information for a reader")

    result = ok(cli("ingest", str(doc), "--judge"))

    assert "1 documents: 1 new, 0 duplicate; 1 chunks added; 0 quarantined" in result.output
    assert fake_llm.call_count == 1  # one chunk, one judge call


# ----------------------------------------------------------------------------- ask


def test_format_citation_leaves_out_absent_parts() -> None:
    full = Citation(n=1, chunk_id=3, source="a/b.pdf", title="Guide", page=4, heading="Bands", snippet="")
    assert format_citation(full) == "[1] Guide · Bands · p.4 · a/b.pdf"
    bare = Citation(n=2, chunk_id=4, source="notes.txt", title=None, page=None, heading=None, snippet="")
    assert format_citation(bare) == "[2] notes.txt"


def test_ask_prints_answer_citations_and_status(
    cli: Invoke, fake_llm: FakeLLM, fixture_manifest: Path
) -> None:
    ingest_fixtures(cli, fixture_manifest)
    fake_llm.responses.append("Three written quotes are required [1].")

    result = ok(cli("ask", PROCUREMENT_QUESTION, "--user", "u1", "--tags", "public"))

    lines = result.output.splitlines()
    assert lines[0] == "Three written quotes are required [1]."
    citation_lines = [line for line in lines if line.startswith("[1] ")]
    assert citation_lines, result.output
    assert re.match(
        r"^\[1\] Northbank City Council Procurement Guide · .+ · .*northbank-council-procurement\.md$",
        citation_lines[0],
    )
    assert re.search(
        r"^status: answered · 1 citation · \d+ ms · request [0-9a-f]{32}$", result.output, re.MULTILINE
    )
    assert fake_llm.call_count == 1


def test_ask_prints_refusal_state(cli: Invoke, fake_llm: FakeLLM, fixture_manifest: Path) -> None:
    ingest_fixtures(cli, fixture_manifest)
    fake_llm.responses.append(REFUSAL)

    result = ok(cli("ask", RESTRICTED_QUESTION))

    assert result.output.startswith(REFUSAL)
    assert re.search(r"^status: refused · \d+ ms · request [0-9a-f]{32}$", result.output, re.MULTILINE)
    assert "PELICAN" not in result.output


def test_ask_raw_prints_the_answer_as_json(cli: Invoke, fake_llm: FakeLLM, fixture_manifest: Path) -> None:
    ingest_fixtures(cli, fixture_manifest)
    fake_llm.responses.append("Three written quotes [1].")

    result = ok(cli("ask", PROCUREMENT_QUESTION, "--raw"))

    data = json.loads(result.output)
    assert data["text"] == "Three written quotes [1]."
    assert data["refused"] is False
    assert data["citations"][0]["n"] == 1
    assert data["retrieved"] and "chunk_id" in data["retrieved"][0]
    assert re.fullmatch(r"[0-9a-f]{32}", data["request_id"])


def test_ask_json_schema_returns_validated_json(
    cli: Invoke, fake_llm: FakeLLM, fixture_manifest: Path, tmp_path: Path
) -> None:
    ingest_fixtures(cli, fixture_manifest)
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps({"type": "object", "properties": {"quotes": {"type": "integer"}}, "required": ["quotes"]}),
        encoding="utf-8",
    )
    fake_llm.responses.append('{"quotes": 3}')

    result = ok(cli("ask", PROCUREMENT_QUESTION, "--json-schema", str(schema_path)))

    assert result.output.splitlines()[0] == '{"quotes":3}'
    assert fake_llm.calls[0]["json_schema"] == {
        "type": "object",
        "properties": {"quotes": {"type": "integer"}},
        "required": ["quotes"],
    }


def test_ask_json_schema_file_must_hold_an_object(cli: Invoke, tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    schema_path.write_text("[1, 2]", encoding="utf-8")
    result = cli("ask", "anything", "--json-schema", str(schema_path))
    assert result.exit_code == 2
    assert "JSON object" in result.output


# ----------------------------------------------------------------------------- agent


def test_agent_prints_text_and_step_table(cli: Invoke, fake_llm: FakeLLM) -> None:
    fake_llm.responses.extend(
        [tool_call_reply("calculator", {"expression": "1234*5678"}), "1234*5678 is 7006652."]
    )

    result = ok(cli("agent", "What is 1234*5678?", "--max-steps", "3"))

    lines = result.output.splitlines()
    assert lines[0] == "1234*5678 is 7006652."
    assert re.search(r"^#\s+tool\s+decision\s+result$", result.output, re.MULTILINE)
    assert re.search(r"^1\s+calculator\s+allowed\s+7006652$", result.output, re.MULTILINE)
    assert re.search(r"^1 steps · \d+ ms · request [0-9a-f]{32}$", result.output, re.MULTILINE)


def test_agent_shows_queued_and_refused_calls(cli: Invoke, fake_llm: FakeLLM) -> None:
    fake_llm.responses.extend(
        [
            tool_call_reply("create_ticket", {"title": "Printer down", "body": "Level 2 printer is jammed."}),
            tool_call_reply("http_get", {"url": "https://example.com/"}),
            "The ticket awaits approval.",
        ]
    )

    result = ok(cli("agent", "Open a ticket about the printer, then fetch example.com"))

    assert re.search(r"^1\s+create_ticket\s+queued\s+approval id 1$", result.output, re.MULTILINE)
    assert re.search(
        r"^2\s+http_get\s+refused\s+refused: tool 'http_get' is outside", result.output, re.MULTILINE
    )
    assert "refused tools: http_get" in result.output

    listed = ok(cli("approvals", "list", "--status", "pending"))
    assert "create_ticket" in listed.output and "Printer down" in listed.output


# ----------------------------------------------------------------------------- approvals


def test_approvals_list_approve_and_reject(cli: Invoke, fixture_manifest: Path) -> None:
    empty = ok(cli("approvals", "list"))
    assert empty.output.strip() == "no approvals"

    ctx = build_context()
    try:
        first = ctx.approvals.enqueue("req-1", "create_ticket", {"title": "Printer down", "body": "jammed"})
        second = ctx.approvals.enqueue(
            "req-2", "create_ticket", {"title": "Coffee", "body": "machine broken"}
        )
    finally:
        ctx.close()

    listed = ok(cli("approvals", "list", "--status", "pending"))
    rows = [line for line in listed.output.splitlines()[1:] if line.strip()]
    assert len(rows) == 2
    assert rows[0].startswith(f"{first} ") and "pending" in rows[0] and "create_ticket" in rows[0]

    approved = ok(cli("approvals", "approve", str(first), "--by", "blake"))
    assert f"approved {first} (create_ticket) by blake" in approved.output
    assert "executed: ticket created: Printer down (#1)" in approved.output

    again = cli("approvals", "approve", str(first))
    assert again.exit_code == 1
    assert f"approval {first} is executed" in again.output

    rejected = ok(cli("approvals", "reject", str(second), "--by", "blake"))
    assert f"rejected {second} (create_ticket) by blake" in rejected.output

    executed = ok(cli("approvals", "list", "--status", "executed"))
    assert f"{first} " in executed.output and "blake" in executed.output
    rejected_list = ok(cli("approvals", "list", "--status", "rejected"))
    assert f"{second} " in rejected_list.output
    assert ok(cli("approvals", "list", "--status", "pending")).output.strip() == "no pending approvals"


def test_approvals_list_rejects_an_unknown_status(cli: Invoke) -> None:
    result = cli("approvals", "list", "--status", "bogus")
    assert result.exit_code == 2
    assert "status must be one of" in result.output


def test_approvals_approve_unknown_id_exits_1(cli: Invoke) -> None:
    result = cli("approvals", "approve", "99")
    assert result.exit_code == 1
    assert "approval 99 does not exist" in result.output


# ----------------------------------------------------------------------------- verify-ledger


def test_verify_ledger_intact_and_export_verifies_offline(
    cli: Invoke, fixture_manifest: Path, tmp_path: Path
) -> None:
    ingest_fixtures(cli, fixture_manifest)
    export = tmp_path / "ledger.jsonl"

    result = ok(cli("verify-ledger", "--export", str(export)))

    assert re.search(
        r"^ledger: intact · 5 rows checked · head seq 5 · head [0-9a-f]{64}$", result.output, re.MULTILINE
    )
    assert f"exported 5 rows to {export}" in result.output
    assert "export verifies: intact · 5 rows checked" in result.output
    assert export.exists()
    assert len(json_lines(export.read_text(encoding="utf-8"))) == 5
    assert verify_file(export).ok


def test_verify_ledger_broken_chain_exits_1(cli: Invoke, fixture_manifest: Path, conn) -> None:
    ingest_fixtures(cli, fixture_manifest)
    conn.execute("UPDATE ledger SET payload = '{\"tampered\":true}' WHERE seq = 2")

    result = cli("verify-ledger")

    assert result.exit_code == 1
    assert "ledger: broken at seq 2 · seq 2: hash mismatch" in result.output


def test_verify_ledger_empty_store_is_intact(cli: Invoke) -> None:
    result = ok(cli("verify-ledger"))
    assert "ledger: intact · 0 rows checked · head seq 0" in result.output


# ----------------------------------------------------------------------------- status


def test_status_prints_counts(
    cli: Invoke, fake_llm: FakeLLM, fixture_manifest: Path, conn, data_dir: Path
) -> None:
    ingest_fixtures(cli, fixture_manifest)
    fake_llm.responses.append(REFUSAL)
    ok(cli("ask", RESTRICTED_QUESTION))
    chunks = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    result = ok(cli("status"))

    assert "profile: local" in result.output
    assert f"data dir: {data_dir.resolve()}" in result.output
    assert "air-gap: off" in result.output
    assert "llm: healthy · fake" in result.output
    assert "documents: 5" in result.output
    assert f"chunks: {chunks} (1 quarantined)" in result.output
    assert re.search(r"^ledger: \d+ rows · head seq \d+$", result.output, re.MULTILINE)
    assert re.search(
        r"^inference: 1 requests · 1 refused · avg \d+ ms · \d+ prompt tokens · \d+ output tokens$",
        result.output,
        re.MULTILINE,
    )


# ----------------------------------------------------------------------------- export-log


def test_export_log_prints_json_lines_oldest_first(
    cli: Invoke, fake_llm: FakeLLM, fixture_manifest: Path
) -> None:
    ingest_fixtures(cli, fixture_manifest)
    fake_llm.responses.extend(["Three written quotes are required [1].", REFUSAL])
    ok(cli("ask", PROCUREMENT_QUESTION, "--user", "u1"))
    ok(cli("ask", RESTRICTED_QUESTION, "--user", "u2"))

    result = ok(cli("export-log", "--limit", "10"))

    rows = json_lines(result.output)
    assert [row["user_id"] for row in rows] == ["u1", "u2"]
    assert rows[0]["mode"] == "answer" and rows[0]["question"] == PROCUREMENT_QUESTION
    assert rows[0]["refused"] is False and rows[1]["refused"] is True
    assert rows[0]["citations"][0]["n"] == 1

    limited = ok(cli("export-log", "--limit", "1"))
    assert [row["user_id"] for row in json_lines(limited.output)] == ["u2"]
    assert json_lines(ok(cli("export-log", "--mode", "agent")).output) == []


# ----------------------------------------------------------------------------- serve


def test_serve_hands_the_import_string_to_uvicorn(cli: Invoke, monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    seen: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda application, **kwargs: seen.update(app=application, **kwargs))

    result = ok(cli("serve", "--host", "0.0.0.0", "--port", "9000"))

    assert seen == {"app": "keel.web.app:app", "host": "0.0.0.0", "port": 9000}
    assert "keel web: http://0.0.0.0:9000" in result.output


# ----------------------------------------------------------------------------- eval


@dataclass
class _EvalResult:
    """The shape keel.evals.run.EvalResult exposes: a summary, per-item detail, report paths, the gate."""

    summary: dict[str, Any]
    items: list[dict[str, Any]]
    report_html_path: Path
    report_json_path: Path
    gate_passed: bool


def _fake_evals_run(seen: dict[str, Any], gate_passed: bool = False) -> types.ModuleType:
    """A stand-in for keel.evals.run: run_eval records its keyword arguments; promote_baseline records the call."""

    def run_eval(ctx: Any, **kwargs: Any) -> _EvalResult:
        seen.update(kwargs)
        return _EvalResult(
            summary={"hit_at_k": 0.5, "groundedness": 0.9},
            items=[{"id": "q1", "answer": "long per-item detail that stays out of the terminal"}],
            report_html_path=Path("reports/latest.html"),
            report_json_path=Path("reports/latest.json"),
            gate_passed=gate_passed,
        )

    def promote_baseline(result: _EvalResult) -> Path:
        seen["promoted"] = result.report_json_path
        return Path("reports/baseline.json")

    module = types.ModuleType("keel.evals.run")
    module.run_eval = run_eval  # type: ignore[attr-defined]
    module.promote_baseline = promote_baseline  # type: ignore[attr-defined]
    return module


def test_eval_calls_run_eval_and_gates(cli: Invoke, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setitem(sys.modules, "keel.evals.run", _fake_evals_run(seen))

    gated = cli("eval", "--golden", "golden.yaml", "--report", "reports", "--gate")
    assert gated.exit_code == 1, gated.output
    assert seen == {"golden_path": Path("golden.yaml"), "report_dir": Path("reports")}
    assert '"hit_at_k": 0.5' in gated.output
    assert "latest.html" in gated.output
    assert "long per-item detail" not in gated.output
    assert "gate: failed" in gated.output

    seen.clear()
    ungated = ok(cli("eval"))
    assert seen == {}
    assert "gate: failed" in ungated.output


def test_eval_passes_judge_baseline_and_promote_through(cli: Invoke, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setitem(sys.modules, "keel.evals.run", _fake_evals_run(seen, gate_passed=True))

    result = ok(cli("eval", "--no-judge", "--baseline", "old.json", "--promote", "--gate"))

    assert seen == {
        "judge": False,
        "baseline_path": Path("old.json"),
        "promoted": Path("reports/latest.json"),
    }
    assert "gate: passed" in result.output
    assert "baseline: reports" in result.output


def test_eval_generate_drafts_golden_items(
    cli: Invoke, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setitem(sys.modules, "keel.evals.run", _fake_evals_run(seen))

    def generate_golden(ctx: Any, n: int, out_path: Path, **kwargs: Any) -> list[str]:
        seen.update(n=n, out_path=out_path)
        return ["item"] * n

    golden_module = types.ModuleType("keel.evals.golden")
    golden_module.generate_golden = generate_golden  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "keel.evals.golden", golden_module)
    out = tmp_path / "draft.yaml"

    result = ok(cli("eval", "--generate", "3", "--out", str(out)))

    assert seen == {"n": 3, "out_path": out}  # run_eval was never called
    assert f"drafted 3 golden items to {out}" in result.output


def test_eval_without_the_evals_module_says_so(cli: Invoke, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keel.evals.run", None)  # makes the import raise ImportError
    result = cli("eval")
    assert result.exit_code == 1
    assert "keel.evals.run" in result.output
