"""The eval runner. `run_eval()` asks every golden item through the answer engine as the eval user,
scores retrieval, refusal and string checks, judges the answered items, attaches the scores to the
inference log, writes `reports/eval-<timestamp>.json` and `.html` (plus `latest.json` and
`latest.html`), and holds the summary against a baseline when one exists.

`promote_baseline()` copies the latest summary into `baseline.json` so the next run is gated on it.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from keel.answer.prompts import build_source_block
from keel.answer.types import Answer, User
from keel.evals.golden import GoldenItem, load_golden, validate
from keel.evals.judge import Judge
from keel.evals.metrics import (
    DEFAULT_THRESHOLDS,
    HIT_KS,
    GateResult,
    aggregate,
    compare,
    found_strings,
    hit_at_k,
    missing_strings,
    reciprocal_rank,
    refusal_correct,
)
from keel.evals.report import METHODOLOGY, write_report

log = logging.getLogger("keel.evals.run")

EVAL_USER_ID = "eval"
BASELINE_NAME = "baseline.json"
LATEST_JSON = "latest.json"
LATEST_HTML = "latest.html"


@dataclass
class EvalResult:
    """What one eval run produced. `items` are the per-item result dicts the JSON report holds."""

    summary: dict[str, Any]
    items: list[dict[str, Any]]
    report_html_path: Path
    report_json_path: Path
    gate_passed: bool
    regressions: list[dict[str, Any]] = field(default_factory=list)
    baseline_path: Path | None = None
    thresholds: dict[str, float] = field(default_factory=dict)
    generated_at: str = ""

    @property
    def report_dir(self) -> Path:
        return self.report_json_path.parent


# ----------------------------------------------------------------------------- run


def run_eval(
    ctx: Any,
    golden_path: str | Path = "fixtures/golden.yaml",
    report_dir: str | Path = "reports",
    *,
    judge: bool = True,
    baseline_path: str | Path | None = None,
    thresholds: dict[str, float] | None = None,
) -> EvalResult:
    """Run the golden set through `ctx.answer_engine`, write the reports and gate against the
    baseline. `baseline_path` defaults to `<report_dir>/baseline.json` when that file exists;
    `thresholds` default to `DEFAULT_THRESHOLDS`. Raises ValueError when the golden set is malformed."""
    golden_file = Path(golden_path)
    items = load_golden(golden_file)
    problems = validate(items)
    if problems:
        raise ValueError(f"{golden_file}: " + "; ".join(problems))

    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    limits = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    judge_obj = Judge.from_context(ctx) if judge else None

    log.info("eval: %d items from %s (judge %s)", len(items), golden_file, "on" if judge_obj else "off")
    results = [evaluate_item(ctx, item, judge_obj) for item in items]
    summary = aggregate(results)

    baseline_file = _resolve_baseline(baseline_path, out_dir)
    baseline_summary = _load_baseline_summary(baseline_file) if baseline_file else None
    gate: GateResult = compare(summary, baseline_summary, limits)

    generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = _unique_path(out_dir, f"eval-{stamp}", ".json")
    html_path = json_path.with_suffix(".html")

    payload: dict[str, Any] = {
        "generated_at": generated_at,
        "golden_path": str(golden_file),
        "profile": str(getattr(ctx, "profile", "") or getattr(ctx.settings, "profile", "")),
        "provider": str(getattr(ctx.llm, "name", "") or ""),
        "model": str(getattr(ctx.llm, "model", "") or ""),
        "judge_names": judge_obj.names if judge_obj else [],
        "thresholds": limits,
        "baseline_path": str(baseline_file) if baseline_file else None,
        "gate": gate.to_dict(),
        "summary": summary,
        "items": results,
        "methodology": METHODOLOGY,
        "report_json_path": str(json_path),
        "report_html_path": str(html_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_report(payload, html_path)
    shutil.copyfile(json_path, out_dir / LATEST_JSON)
    shutil.copyfile(html_path, out_dir / LATEST_HTML)
    log.info(
        "eval: hit@3 %s, groundedness %s, refusal_correct %s, gate %s -> %s",
        summary.get("hit_at_3"),
        summary.get("groundedness"),
        summary.get("refusal_correct"),
        "passed" if gate.passed else "failed",
        json_path,
    )
    return EvalResult(
        summary=summary,
        items=results,
        report_html_path=html_path,
        report_json_path=json_path,
        gate_passed=gate.passed,
        regressions=[r.to_dict() for r in gate.regressions],
        baseline_path=baseline_file,
        thresholds=limits,
        generated_at=generated_at,
    )


def promote_baseline(result: EvalResult) -> Path:
    """Copy the run's `latest.json` (falling back to its own JSON report) to `baseline.json` in the
    same report directory and return the baseline path."""
    source = result.report_dir / LATEST_JSON
    if not source.exists():
        source = result.report_json_path
    target = result.report_dir / BASELINE_NAME
    shutil.copyfile(source, target)
    return target


# ----------------------------------------------------------------------------- one item


def evaluate_item(ctx: Any, item: GoldenItem, judge: Judge | None) -> dict[str, Any]:
    """Ask one item and score it. Engine failures are recorded on the item, never raised."""
    user = User(user_id=EVAL_USER_ID, tags=list(item.user_tags))
    started = time.perf_counter()
    try:
        answer = ctx.answer_engine.answer(item.question, user)
        error: str | None = answer.error
        _mark_eval_mode(ctx, answer.request_id)
    except Exception as exc:  # one broken item must not end the run
        log.warning("eval item %s failed: %s", item.id, exc)
        answer = None
        error = f"{type(exc).__name__}: {exc}"
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    result = score_item(item, answer, elapsed_ms, error)

    if judge is not None and answer is not None and not item.expect_refusal and not answer.refused:
        context = build_source_block(answer.retrieved)
        verdict = judge.score(item.question, answer.text, context, item.expected_answer)
        result["groundedness"] = verdict.get("groundedness")
        result["relevance"] = verdict.get("relevance")
        result["correctness"] = verdict.get("correctness")
        result["judge_reasons"] = dict(verdict.get("reasons") or {})
        result["judge"] = verdict
        result["judge_error"] = verdict.get("error")

    if answer is not None:
        _attach_judge(ctx, answer.request_id, result)
    return result


def score_item(item: GoldenItem, answer: Answer | None, elapsed_ms: int, error: str | None) -> dict[str, Any]:
    """Retrieval, refusal and string checks for one item; judge fields start empty."""
    text = answer.text if answer is not None else ""
    refused = bool(answer.refused) if answer is not None else False
    hits = list(answer.retrieved) if answer is not None else []
    titles = [hit_title(h) for h in hits]
    missing = missing_strings(text, item.must_include)
    found = found_strings(text, item.must_not_include)
    include_pass = (not missing) if item.must_include else None
    exclude_pass = (not found) if item.must_not_include else None
    result: dict[str, Any] = {
        "id": item.id,
        "question": item.question,
        "user_tags": list(item.user_tags),
        "expect_refusal": item.expect_refusal,
        "expected_answer": item.expected_answer,
        "expected_sources": list(item.expected_sources),
        "answer": text,
        "refused": refused,
        "refusal_correct": refusal_correct(item.expect_refusal, refused) if error is None else False,
        "retrieved_titles": titles,
        "retrieved_ids": [h.chunk_id for h in hits],
        "citations": [c.to_dict() for c in answer.citations] if answer is not None else [],
        "reciprocal_rank": reciprocal_rank(item.expected_sources, titles),
        "must_include_missing": missing,
        "must_not_include_found": found,
        "must_include_pass": include_pass,
        "must_not_include_pass": exclude_pass,
        "checks_pass": include_pass is not False and exclude_pass is not False and error is None,
        "groundedness": None,
        "relevance": None,
        "correctness": None,
        "judge_reasons": {},
        "judge": None,
        "judge_error": None,
        "latency_ms": int(answer.latency_ms) if answer is not None and answer.latency_ms else elapsed_ms,
        "prompt_tokens": int(answer.prompt_tokens) if answer is not None else 0,
        "output_tokens": int(answer.output_tokens) if answer is not None else 0,
        "request_id": answer.request_id if answer is not None else None,
        "error": error,
    }
    for k in HIT_KS:
        result[f"hit_at_{k}"] = hit_at_k(item.expected_sources, titles, k)
    return result


def hit_title(hit: Any) -> str:
    """A chunk's document title, or its file name when the title is empty."""
    title = getattr(hit, "title", None)
    if title:
        return str(title)
    source = str(getattr(hit, "source", "") or "")
    return source.replace("\\", "/").rsplit("/", 1)[-1]


# ----------------------------------------------------------------------------- helpers


def _mark_eval_mode(ctx: Any, request_id: str | None) -> None:
    """Re-label the inference log row the engine wrote as mode 'eval' so `keel export-log --mode eval` finds it."""
    conn = getattr(ctx, "conn", None)
    if conn is None or not request_id:
        return
    try:
        conn.execute("UPDATE inference_log SET mode='eval' WHERE request_id=?", (request_id,))
    except Exception:  # a missing table in a fake context is not an eval failure
        return


def _attach_judge(ctx: Any, request_id: str, result: dict[str, Any]) -> None:
    """Write the eval scores onto the inference log row for this request. The admin page's daily
    trend reads `groundedness` and `relevance` from this JSON."""
    log_obj = getattr(ctx, "log", None)
    if log_obj is None or not hasattr(log_obj, "attach_judge"):
        return
    scores = {
        "groundedness": result.get("groundedness"),
        "relevance": result.get("relevance"),
        "correctness": result.get("correctness"),
        "refusal_correct": result.get("refusal_correct"),
        "hit_at_3": result.get("hit_at_3"),
        "checks_pass": result.get("checks_pass"),
        "reasons": result.get("judge_reasons") or {},
        "judges": (result.get("judge") or {}).get("judges") if result.get("judge") else {},
        "golden_id": result.get("id"),
        "eval": True,
    }
    try:
        log_obj.attach_judge(request_id, scores)
    except Exception as exc:  # the log is a side channel; the eval result stands without it
        log.warning("attach_judge failed for %s: %s", request_id, exc)


def _resolve_baseline(baseline_path: str | Path | None, out_dir: Path) -> Path | None:
    """An explicit baseline must exist; otherwise `<report_dir>/baseline.json` is used when present."""
    if baseline_path is not None:
        explicit = Path(baseline_path)
        if not explicit.exists():
            raise FileNotFoundError(f"baseline {explicit} does not exist")
        return explicit
    candidate = out_dir / BASELINE_NAME
    return candidate if candidate.exists() else None


def _load_baseline_summary(path: Path) -> dict[str, Any] | None:
    """The `summary` of a saved report, or the whole file when it is already a bare summary.
    A missing or unreadable baseline compares as no baseline (logged, never raised)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("baseline %s unreadable, comparing against nothing: %s", path, exc)
        return None
    if isinstance(data, dict) and isinstance(data.get("summary"), dict):
        return data["summary"]
    return data if isinstance(data, dict) else None


def _unique_path(directory: Path, stem: str, suffix: str) -> Path:
    """`<stem><suffix>`, or `<stem>-2<suffix>` and so on when a run in the same second exists."""
    candidate = directory / f"{stem}{suffix}"
    counter = 2
    while candidate.exists() or candidate.with_suffix(".html").exists():
        candidate = directory / f"{stem}-{counter}{suffix}"
        counter += 1
    return candidate
