"""The `keel` command line: ingest, ask, agent, approvals, verify-ledger, status, serve, eval, export-log.

Importing this module is cheap. Every command builds the application context (`build_context()`)
inside its own body, so `keel --help` loads nothing heavier than typer. The global `--data-dir` and
`--profile` options set `KEEL_DATA_DIR` and `KEEL_PROFILE` before settings load, which is why they
sit on the app callback rather than on each command.

Run as `keel` (console script), `python -m keel` or `python -m keel.cli`.
"""

from __future__ import annotations

import dataclasses
import getpass
import json
import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

if TYPE_CHECKING:  # heavy imports stay out of the module load path
    from keel.answer.types import Answer, Citation
    from keel.providers.factory import AppContext

PROFILES = ("local", "azure", "aws")
DEFAULT_USER = "cli"
DEFAULT_TAGS = "public"
SEPARATOR = " · "  # middle dot with spaces: '[1] title · heading · p.N · source'
RESULT_CHARS = 72  # width of the result column in the agent step table

app = typer.Typer(
    name="keel",
    help="Keel: sovereign retrieval and agent appliance. Ingest documents, ask questions, run the agent, audit the ledger.",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
approvals_app = typer.Typer(help="Approve, reject and list queued write tool calls.", no_args_is_help=True)
app.add_typer(approvals_app, name="approvals")


# ---------------------------------------------------------------------- global options


@app.callback()
def main(
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            "-d",
            help="Data directory holding keel.db and policy.yaml (sets KEEL_DATA_DIR before settings load).",
            show_default=False,
        ),
    ] = None,
    profile: Annotated[
        str | None,
        typer.Option(
            "--profile", help="Deploy profile: local, azure or aws (sets KEEL_PROFILE).", show_default=False
        ),
    ] = None,
) -> None:
    """Global options, applied before any command reads its settings."""
    if data_dir is not None:
        os.environ["KEEL_DATA_DIR"] = str(data_dir)
    if profile is not None:
        if profile not in PROFILES:
            raise typer.BadParameter(f"profile must be one of {', '.join(PROFILES)}", param_hint="--profile")
        os.environ["KEEL_PROFILE"] = profile
    if data_dir is not None or profile is not None:
        from keel.config import reset_settings

        reset_settings()


# ---------------------------------------------------------------------- shared helpers


@contextmanager
def _context() -> Iterator[AppContext]:
    """Build the application context for one command and close its store afterwards.

    A configuration problem (for example the Azure profile without its endpoints) ends the command
    with the message on stderr and exit code 1 rather than a traceback.
    """
    from keel.providers.factory import build_context

    try:
        ctx = build_context()
    except RuntimeError as exc:
        _fail(str(exc))
    try:
        yield ctx
    finally:
        ctx.close()


def _fail(message: str, code: int = 1) -> None:
    """Print `message` to stderr and exit with `code`."""
    typer.echo(message, err=True)
    raise typer.Exit(code=code)


def _split_tags(raw: str) -> list[str]:
    """Turn 'public, hr' into ['public', 'hr']; an empty string means no tags."""
    return [part.strip() for part in raw.split(",") if part.strip()]


def _user(user_id: str, tags: str) -> Any:
    from keel.answer.types import User

    return User(user_id=user_id, tags=_split_tags(tags) or [DEFAULT_TAGS])


def _json(value: Any) -> str:
    """Compact, stable JSON for one output line."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _one_line(text: str, limit: int) -> str:
    """Collapse whitespace and cap the length for table cells."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 3].rstrip() + "..."


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a plain, fixed-width text table (deterministic in any terminal width)."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip()]
    for row in rows:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
    return "\n".join(lines)


def format_citation(citation: Citation) -> str:
    """One citation line: '[n] title · heading · p.N · source', with absent parts left out."""
    parts: list[str] = []
    if citation.title:
        parts.append(citation.title)
    if citation.heading:
        parts.append(citation.heading)
    if citation.page:
        parts.append(f"p.{citation.page}")
    parts.append(citation.source)
    return f"[{citation.n}] " + SEPARATOR.join(parts)


def answer_to_dict(answer: Answer) -> dict[str, Any]:
    """The Answer as plain data (retrieved chunks included), for `keel ask --raw`."""
    return dataclasses.asdict(answer)


EVAL_FIELDS = (
    "summary",
    "regressions",
    "report_html_path",
    "report_json_path",
    "baseline_path",
    "generated_at",
)


def _describe(result: Any) -> str:
    """Render an eval result for the terminal: the summary and report paths when the result carries
    them (per-item detail stays in the JSON report), else the whole object as JSON or text."""
    known = {name: getattr(result, name) for name in EVAL_FIELDS if hasattr(result, name)}
    if known:
        return json.dumps(known, ensure_ascii=False, indent=2, default=str)
    if hasattr(result, "to_dict"):
        return json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str)
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return json.dumps(dataclasses.asdict(result), ensure_ascii=False, indent=2, default=str)
    if isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    return str(result)


def _default_by() -> str:
    try:
        return getpass.getuser()
    except Exception:  # no account name available (some service contexts)
        return DEFAULT_USER


# ---------------------------------------------------------------------- ingest


@app.command()
def ingest(
    sources: Annotated[
        list[str] | None,
        typer.Argument(
            help="File paths or http(s) URLs: PDF, DOCX, Markdown, HTML or plain text.", show_default=False
        ),
    ] = None,
    manifest: Annotated[
        Path | None,
        typer.Option(
            "--manifest",
            "-m",
            help="Corpus manifest (documents: path, title, acl_tags), e.g. fixtures/corpus.yaml.",
            show_default=False,
        ),
    ] = None,
    title: Annotated[
        str | None,
        typer.Option(
            "--title",
            help="Title for the ingested source(s); the document's own title otherwise.",
            show_default=False,
        ),
    ] = None,
    tags: Annotated[
        str | None,
        typer.Option(
            "--tags",
            help=(
                "Comma-separated ACL tags for the source(s); a reader needs one of them. A new source "
                "defaults to public; a source already stored keeps its tags unless --tags names others, "
                "in which case the document and its chunks are retagged."
            ),
            show_default=False,
        ),
    ] = None,
    judge: Annotated[
        bool,
        typer.Option(
            "--judge", help="Screen every chunk with the LLM judge as well as the heuristics (slower)."
        ),
    ] = False,
) -> None:
    """Ingest files, URLs or a manifest into the store. Re-ingesting the same bytes reports a duplicate and adds nothing; with --tags naming different tags it retags the stored document instead."""
    if not sources and manifest is None:
        _fail("give at least one path or URL, or --manifest <corpus.yaml>", code=2)
    import httpx

    from keel.ingest.errors import IngestError
    from keel.ingest.pipeline import IngestResult, ingest_manifest, ingest_path

    # Everything a source can fail with short of a bug: unreadable file, bad manifest entry, unsupported
    # format, air-gap refusal (either layer), or a failed fetch.
    source_errors = (OSError, ValueError, RuntimeError, IngestError, httpx.HTTPError)
    results: list[IngestResult] = []
    failures = 0
    with _context() as ctx:
        screen = ctx.screen_with_judge if judge else ctx.screen
        if manifest is not None:
            try:
                results.extend(
                    ingest_manifest(
                        ctx.conn,
                        ctx.settings,
                        ctx.embedder,
                        ctx.index,
                        manifest,
                        screen=screen,
                        ledger=ctx.ledger,
                    )
                )
            except source_errors as exc:
                _fail(f"manifest {manifest}: {exc}")
        for source in sources or []:
            try:
                results.append(
                    ingest_path(
                        ctx.conn,
                        ctx.settings,
                        ctx.embedder,
                        ctx.index,
                        source,
                        title=title,
                        acl_tags=_split_tags(tags) if tags is not None else None,
                        screen=screen,
                        ledger=ctx.ledger,
                    )
                )
            except source_errors as exc:
                failures += 1
                typer.echo(f"failed     {source}  ({exc})", err=True)
    _report_ingest(results)
    if failures:
        raise typer.Exit(code=1)


def _report_ingest(results: Iterable[Any]) -> None:
    """One line per source, then a summary line."""
    rows = list(results)
    for result in rows:
        if getattr(result, "tags_updated", False):
            typer.echo(
                f"retagged   {result.source}  (document {result.document_id} already stored, 0 chunks added; "
                f"ACL tags {', '.join(result.previous_acl_tags or [])} -> {', '.join(result.acl_tags)})"
            )
        elif result.skipped_duplicate:
            typer.echo(
                f"duplicate  {result.source}  (document {result.document_id} already stored, 0 chunks added)"
            )
        else:
            quarantined = f", {result.quarantined} quarantined" if result.quarantined else ""
            title_note = (
                ", title replaced (flagged by the screen)" if getattr(result, "title_flagged", None) else ""
            )
            typer.echo(
                f"ingested   {result.source}  (document {result.document_id}, {result.chunks_added} chunks{quarantined}{title_note})"
            )
    new = sum(1 for r in rows if not r.skipped_duplicate)
    duplicates = len(rows) - new
    chunks = sum(r.chunks_added for r in rows)
    quarantined_total = sum(r.quarantined for r in rows)
    typer.echo(
        f"{len(rows)} documents: {new} new, {duplicates} duplicate; {chunks} chunks added; {quarantined_total} quarantined"
    )


# ---------------------------------------------------------------------- ask


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="The question to answer from the corpus.")],
    user: Annotated[
        str, typer.Option("--user", "-u", help="User id recorded on the request.")
    ] = DEFAULT_USER,
    tags: Annotated[
        str, typer.Option("--tags", help="Comma-separated ACL tags the user carries.")
    ] = DEFAULT_TAGS,
    json_schema: Annotated[
        Path | None,
        typer.Option(
            "--json-schema",
            help="JSON schema file; the answer comes back as validated JSON.",
            show_default=False,
        ),
    ] = None,
    raw: Annotated[
        bool, typer.Option("--raw", help="Print the whole Answer as JSON instead of the readable form.")
    ] = False,
) -> None:
    """Answer a question with citations, or refuse when the sources hold no answer. Exit 0 either way."""
    schema: dict[str, Any] | None = None
    if json_schema is not None:
        try:
            schema = json.loads(json_schema.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            _fail(f"--json-schema {json_schema}: {exc}", code=2)
        if not isinstance(schema, dict):
            _fail(f"--json-schema {json_schema}: the file must hold a JSON object", code=2)

    with _context() as ctx:
        answer = ctx.answer_engine.answer(question, _user(user, tags), json_schema=schema)

    if raw:
        typer.echo(json.dumps(answer_to_dict(answer), ensure_ascii=False, indent=2, default=str))
    else:
        _print_answer(answer)
    if answer.error:
        raise typer.Exit(code=1)


def _print_answer(answer: Answer) -> None:
    typer.echo(answer.text)
    if answer.citations:
        typer.echo("")
        for citation in answer.citations:
            typer.echo(format_citation(citation))
    typer.echo("")
    tail = f"{answer.latency_ms} ms{SEPARATOR}request {answer.request_id}"
    if answer.error:
        typer.echo(f"status: error{SEPARATOR}{answer.error}{SEPARATOR}{tail}")
    elif answer.refused:
        typer.echo(f"status: refused{SEPARATOR}{tail}")
    else:
        count = len(answer.citations)
        typer.echo(f"status: answered{SEPARATOR}{count} citation{'s' if count != 1 else ''}{SEPARATOR}{tail}")


# ---------------------------------------------------------------------- agent


@app.command()
def agent(
    question: Annotated[str, typer.Argument(help="The request for the tool-calling agent.")],
    user: Annotated[
        str, typer.Option("--user", "-u", help="User id recorded on the request.")
    ] = DEFAULT_USER,
    tags: Annotated[
        str, typer.Option("--tags", help="Comma-separated ACL tags the user carries.")
    ] = DEFAULT_TAGS,
    max_steps: Annotated[
        int, typer.Option("--max-steps", min=1, help="Model turns before the loop stops.")
    ] = 6,
) -> None:
    """Run the agent: the model calls tools under policy, write tools queue for approval, and the final text prints with a step table."""
    with _context() as ctx:
        result = ctx.agent_loop.run(question, _user(user, tags), max_steps=max_steps)

    typer.echo(result.text)
    typer.echo("")
    if result.steps:
        rows = [
            [str(n), step.get("tool") or "", _decision_label(step), _step_result(step)]
            for n, step in enumerate(result.steps, start=1)
        ]
        typer.echo(_table(["#", "tool", "decision", "result"], rows))
    else:
        typer.echo("steps: none (the model answered without tools)")
    refused = f"{SEPARATOR}refused tools: {', '.join(result.refused_tools)}" if result.refused_tools else ""
    typer.echo(
        f"{len(result.steps)} steps{refused}{SEPARATOR}{result.latency_ms} ms{SEPARATOR}request {result.request_id}"
    )


def _decision_label(step: dict[str, Any]) -> str:
    decision = step.get("decision") or {}
    if not decision.get("allowed", False):
        return "refused"
    if decision.get("needs_approval"):
        return "queued"
    return "allowed"


def _step_result(step: dict[str, Any]) -> str:
    if "queued_id" in step:
        return f"approval id {step['queued_id']}"
    return _one_line(step.get("result") or "", RESULT_CHARS)


# ---------------------------------------------------------------------- approvals


@approvals_app.command("list")
def approvals_list(
    status: Annotated[
        str | None,
        typer.Option("--status", help="Filter: pending, approved, rejected or executed.", show_default=False),
    ] = None,
) -> None:
    """List queued write tool calls, oldest first."""
    with _context() as ctx:
        try:
            rows = ctx.approvals.list(status)
        except ValueError as exc:
            _fail(str(exc), code=2)
    if not rows:
        typer.echo(f"no {status} approvals" if status else "no approvals")
        return
    table = [
        [
            str(row["id"]),
            row["status"],
            row["tool"],
            row["ts"] or "",
            row.get("decided_by") or "",
            _one_line(_json(row["arguments"]), RESULT_CHARS),
        ]
        for row in rows
    ]
    typer.echo(_table(["id", "status", "tool", "requested", "decided by", "arguments"], table))


@approvals_app.command("approve")
def approvals_approve(
    approval_id: Annotated[int, typer.Argument(help="Approval id from `keel approvals list`.")],
    by: Annotated[
        str | None,
        typer.Option("--by", help="Who decides; defaults to the OS user name.", show_default=False),
    ] = None,
) -> None:
    """Approve a pending call and execute it through the tool registry."""
    from keel.agent.tools import ToolContext

    with _context() as ctx:
        try:
            row = ctx.approvals.decide(approval_id, True, by or _default_by())
            typer.echo(f"approved {approval_id} ({row['tool']}) by {row['decided_by']}")
            result = ctx.approvals.execute(
                approval_id, ctx.registry, ToolContext(request_id=row["request_id"]), policy=ctx.policy
            )
        except ValueError as exc:
            _fail(str(exc))
    typer.echo(f"executed: {result}")


@approvals_app.command("reject")
def approvals_reject(
    approval_id: Annotated[int, typer.Argument(help="Approval id from `keel approvals list`.")],
    by: Annotated[
        str | None,
        typer.Option("--by", help="Who decides; defaults to the OS user name.", show_default=False),
    ] = None,
) -> None:
    """Reject a pending call; it stays on record and never runs."""
    with _context() as ctx:
        try:
            row = ctx.approvals.decide(approval_id, False, by or _default_by())
        except ValueError as exc:
            _fail(str(exc))
    typer.echo(f"rejected {approval_id} ({row['tool']}) by {row['decided_by']}")


# ---------------------------------------------------------------------- verify-ledger


@app.command("verify-ledger")
def verify_ledger(
    export: Annotated[
        Path | None,
        typer.Option(
            "--export",
            help="Also write the ledger as JSONL here and verify the file offline.",
            show_default=False,
        ),
    ] = None,
) -> None:
    """Recompute the hash chain of the audit ledger. Exit 1 when any link is broken."""
    from keel.safety.ledger import verify_file

    with _context() as ctx:
        result = ctx.ledger.verify()
        head_seq = _ledger_head_seq(ctx)
        head = ctx.ledger.head()
        exported = None
        if export is not None:
            try:
                exported = ctx.ledger.export(export)
            except OSError as exc:
                _fail(f"--export {export}: {exc}")

    broken = not result.ok
    if result.ok:
        typer.echo(
            f"ledger: intact{SEPARATOR}{result.checked} rows checked{SEPARATOR}head seq {head_seq}{SEPARATOR}head {head}"
        )
    else:
        typer.echo(
            f"ledger: broken at seq {result.first_bad_seq}{SEPARATOR}{result.reason}{SEPARATOR}{result.checked} rows checked"
        )
    if export is not None:
        typer.echo(f"exported {exported} rows to {export}")
        file_result = verify_file(export)
        if file_result.ok:
            typer.echo(f"export verifies: intact{SEPARATOR}{file_result.checked} rows checked")
        else:
            broken = True
            typer.echo(
                f"export verifies: broken at seq {file_result.first_bad_seq}{SEPARATOR}{file_result.reason}"
            )
    if broken:
        raise typer.Exit(code=1)


def _ledger_head_seq(ctx: AppContext) -> int:
    row = ctx.conn.execute("SELECT COALESCE(MAX(seq), 0) FROM ledger").fetchone()
    return int(row[0]) if row else 0


# ---------------------------------------------------------------------- status


@app.command()
def status() -> None:
    """Show the profile, data directory, air-gap state, LLM health, corpus counts, ledger head and inference totals."""
    with _context() as ctx:
        settings = ctx.settings
        profile = ctx.profile
        healthy = ctx.llm.healthy()
        documents = int(ctx.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
        chunk_row = ctx.conn.execute("SELECT COUNT(*), COALESCE(SUM(quarantined), 0) FROM chunks").fetchone()
        chunks, quarantined = int(chunk_row[0]), int(chunk_row[1])
        ledger_rows = ctx.ledger.count()
        head_seq = _ledger_head_seq(ctx)
        totals = ctx.log.totals()
        llm_name = getattr(ctx.llm, "name", type(ctx.llm).__name__)
        llm_model = getattr(ctx.llm, "model", "") or ""
        llm_url = getattr(ctx.llm, "base_url", "") or ""

    typer.echo(f"profile: {profile}")
    typer.echo(f"data dir: {Path(settings.data_dir).resolve()}")
    typer.echo(f"air-gap: {'on' if settings.airgap else 'off'}")
    llm_parts = [llm_name] + [p for p in (llm_model, llm_url) if p]
    typer.echo(f"llm: {'healthy' if healthy else 'unreachable'}{SEPARATOR}{SEPARATOR.join(llm_parts)}")
    typer.echo(f"documents: {documents}")
    typer.echo(f"chunks: {chunks} ({quarantined} quarantined)")
    typer.echo(f"ledger: {ledger_rows} rows{SEPARATOR}head seq {head_seq}")
    typer.echo(
        f"inference: {totals.get('requests', 0)} requests{SEPARATOR}{totals.get('refused', 0)} refused"
        f"{SEPARATOR}avg {int(totals.get('avg_latency_ms') or 0)} ms"
        f"{SEPARATOR}{totals.get('prompt_tokens', 0)} prompt tokens{SEPARATOR}{totals.get('output_tokens', 0)} output tokens"
    )


# ---------------------------------------------------------------------- serve


@app.command()
def serve(
    host: Annotated[
        str | None,
        typer.Option("--host", help="Bind address; defaults to KEEL_HOST (127.0.0.1).", show_default=False),
    ] = None,
    port: Annotated[
        int | None, typer.Option("--port", help="Port; defaults to KEEL_PORT (8400).", show_default=False)
    ] = None,
) -> None:
    """Start the web app (chat, citations, admin page) with uvicorn."""
    import uvicorn

    from keel.config import get_settings

    settings = get_settings()
    bind_host = host or settings.host
    bind_port = port if port is not None else settings.port
    typer.echo(f"keel web: http://{bind_host}:{bind_port}")
    uvicorn.run("keel.web.app:app", host=bind_host, port=bind_port)


# ---------------------------------------------------------------------- eval


@app.command("eval")
def eval_command(
    golden: Annotated[
        Path | None,
        typer.Option(
            "--golden", help="Golden question set (YAML); the evals default otherwise.", show_default=False
        ),
    ] = None,
    report: Annotated[
        Path | None,
        typer.Option("--report", help="Directory for the HTML report and JSON summary.", show_default=False),
    ] = None,
    gate: Annotated[bool, typer.Option("--gate", help="Exit 1 when the regression gate fails.")] = False,
    judge: Annotated[
        bool,
        typer.Option(
            "--judge/--no-judge",
            help="Run the LLM judge on answered items. Off, the retrieval, refusal and string checks finish in seconds.",
        ),
    ] = True,
    baseline: Annotated[
        Path | None,
        typer.Option(
            "--baseline",
            help="Baseline summary to gate against; <report>/baseline.json otherwise.",
            show_default=False,
        ),
    ] = None,
    promote: Annotated[
        bool,
        typer.Option(
            "--promote",
            help="After the run, copy its summary to <report>/baseline.json for later runs to gate against.",
        ),
    ] = False,
    generate: Annotated[
        int | None,
        typer.Option(
            "--generate",
            min=1,
            help="Draft this many golden items from random corpus chunks with the model into --out, then stop.",
            show_default=False,
        ),
    ] = None,
    out: Annotated[
        Path, typer.Option("--out", help="Where --generate writes its draft (editable YAML).")
    ] = Path("fixtures/golden-generated.yaml"),
) -> None:
    """Run the golden set through the appliance and report retrieval, groundedness, relevance, refusals, latency and tokens."""
    try:
        from keel.evals.run import promote_baseline, run_eval
    except ImportError:
        _fail("keel eval needs the evals module (keel.evals.run); it is absent from this checkout.")

    if generate is not None:
        from keel.evals.golden import generate_golden

        with _context() as ctx:
            items = generate_golden(ctx, generate, out)
        typer.echo(
            f"drafted {len(items)} golden items to {out}; edit them by hand, then run: keel eval --golden {out}"
        )
        return

    kwargs: dict[str, Any] = {}
    if golden is not None:
        kwargs["golden_path"] = golden
    if report is not None:
        kwargs["report_dir"] = report
    if not judge:
        kwargs["judge"] = False
    if baseline is not None:
        kwargs["baseline_path"] = baseline
    with _context() as ctx:
        result = run_eval(ctx, **kwargs)
        promoted = promote_baseline(result) if promote else None

    typer.echo(_describe(result))
    passed = getattr(result, "gate_passed", None)
    if passed is not None:
        typer.echo(f"gate: {'passed' if passed else 'failed'}")
    if promoted is not None:
        typer.echo(f"baseline: {promoted}")
    if gate and passed is False:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------- export-log


@app.command("export-log")
def export_log(
    limit: Annotated[
        int, typer.Option("--limit", "-n", min=1, help="How many of the most recent rows to print.")
    ] = 50,
    mode: Annotated[
        str | None,
        typer.Option("--mode", help="Only rows of this mode: answer, agent or eval.", show_default=False),
    ] = None,
) -> None:
    """Print the most recent inference log rows as JSON lines, oldest first."""
    with _context() as ctx:
        rows = ctx.log.recent(limit=limit, mode=mode)
    for row in reversed(rows):
        typer.echo(_json(row))


if __name__ == "__main__":  # python -m keel.cli
    app()
