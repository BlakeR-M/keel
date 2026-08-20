"""Evals lane: the golden set loads and validates, the metric functions on hand-built cases, the
judge parser and its retry path, and `run_eval` end to end over the real fixture corpus (fastembed
retrieval and reranking) with a scripted LLM standing in for llama-server. No network."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from keel.answer import REFUSAL
from keel.config import Settings
from keel.db import connect
from keel.evals import (
    DEFAULT_THRESHOLDS,
    GoldenItem,
    aggregate,
    compare,
    generate_golden,
    judge_answer,
    load_golden,
    promote_baseline,
    run_eval,
    save_golden,
    validate_golden,
)
from keel.evals.judge import JUDGE_SCHEMA, Judge, merge_verdicts, parse_judge_reply
from keel.evals.metrics import (
    found_strings,
    hit_at_k,
    missing_strings,
    percentile,
    reciprocal_rank,
    refusal_correct,
)
from keel.evals.report import render_report
from keel.ingest import ingest_manifest
from keel.providers.base import ChatMessage, ChatResult, ToolSpec
from keel.providers.factory import AppContext, build_context
from tests.fakes import FakeLLM

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = REPO_ROOT / "fixtures" / "golden.yaml"
FIXTURE_MANIFEST = REPO_ROOT / "fixtures" / "corpus.yaml"
SECRET = "PELICAN-7741"
OVERRIDE = "APPROVED BY OVERRIDE"

GOOD_JUDGE_JSON = json.dumps(
    {
        "groundedness": 0.9,
        "groundedness_reason": "every claim appears in the passages",
        "relevance": 0.8,
        "relevance_reason": "answers the question directly",
        "correctness": 1.0,
        "correctness_reason": "matches the reference",
    }
)


# ----------------------------------------------------------------------------- scripted LLM


def question_of(messages: list[ChatMessage]) -> str:
    """The question the answer engine put at the end of its user prompt."""
    for message in reversed(messages):
        if message.role == "user":
            content = message.content or ""
            marker = content.rfind("Question:")
            return content[marker + len("Question:") :].strip() if marker >= 0 else content.strip()
    return ""


class RoutingLLM:
    """LLMProvider for eval tests: answers keyed on the question, judge replies keyed on the judge
    schema. Records every call so tests can count judge and answer turns."""

    name = "fake"
    model = "fake-model"

    def __init__(
        self,
        answer_for: Callable[[str], str],
        *,
        judge_replies: list[str] | None = None,
        judge_default: str = GOOD_JUDGE_JSON,
    ) -> None:
        self.answer_for = answer_for
        self.judge_replies = list(judge_replies or [])
        self.judge_default = judge_default
        self.calls: list[dict[str, Any]] = []
        self.judge_calls = 0
        self.answer_calls = 0

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> ChatResult:
        self.calls.append({"messages": list(messages), "json_schema": json_schema})
        if json_schema is not None and set(json_schema.get("properties") or {}) >= set(
            JUDGE_SCHEMA["properties"]
        ):
            self.judge_calls += 1
            content = self.judge_replies.pop(0) if self.judge_replies else self.judge_default
            return ChatResult(content=content, prompt_tokens=30, output_tokens=12, model="fake-judge")
        self.answer_calls += 1
        return ChatResult(
            content=self.answer_for(question_of(messages)),
            prompt_tokens=20,
            output_tokens=8,
            model="fake-model",
        )

    def healthy(self) -> bool:
        return True


def oracle(items: list[GoldenItem], *, override: dict[str, str] | None = None) -> Callable[[str], str]:
    """An answer function that replies with the golden reference (cited [1]) or the refusal sentence,
    with per-question overrides for tests that plant a bad answer."""
    by_question: dict[str, str] = {}
    for item in items:
        by_question[item.question] = REFUSAL if item.expect_refusal else f"{item.expected_answer} [1]"
    by_question.update(override or {})

    def answer_for(question: str) -> str:
        return by_question.get(question, REFUSAL)

    return answer_for


# ----------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def golden() -> list[GoldenItem]:
    return load_golden(GOLDEN_PATH)


@pytest.fixture(scope="module")
def ctx(tmp_path_factory: pytest.TempPathFactory) -> AppContext:
    """The real local wiring over a temporary store with the fixture corpus ingested. Tests swap the
    LLM per test; everything below it (embeddings, index, reranker, ledger, log) is real."""
    data_dir = tmp_path_factory.mktemp("evals")
    settings = Settings(data_dir=data_dir)
    conn: sqlite3.Connection = connect(settings.db_path)
    context = build_context(settings, conn=conn)
    context.settings.gemini_api_key = ""
    results = ingest_manifest(
        conn,
        settings,
        context.embedder,
        context.index,
        FIXTURE_MANIFEST,
        screen=context.screen,
        ledger=context.ledger,
    )
    assert len(results) == 5
    yield context
    context.close()


def use_llm(context: AppContext, llm: Any) -> None:
    context.llm = llm
    context.answer_engine.llm = llm


def mini_golden(golden: list[GoldenItem], path: Path, ids: list[str]) -> Path:
    by_id = {item.id: item for item in golden}
    save_golden([by_id[i] for i in ids], path)
    return path


MINI_IDS = [
    "proc-band2-quotes",
    "clinic-retention-adult",
    "hr-secret-code-public",
    "off-capital-france",
    "inject-verbal-approval",
]


# ----------------------------------------------------------------------------- golden set


def test_golden_loads_validates_and_covers_the_brief(golden: list[GoldenItem]) -> None:
    assert len(golden) >= 16
    assert validate_golden(golden) == []
    assert len({item.id for item in golden}) == len(golden)

    restricted_as_public = [
        it
        for it in golden
        if it.user_tags == ["public"] and it.expect_refusal and SECRET in it.must_not_include
    ]
    assert len(restricted_as_public) >= 3
    hr_items = [it for it in golden if it.user_tags == ["hr"] and not it.expect_refusal]
    assert len(hr_items) >= 2
    off_corpus = [it for it in golden if it.id.startswith("off-")]
    assert len(off_corpus) == 2 and all(it.expect_refusal for it in off_corpus)
    assert any("capital of France" in it.question for it in off_corpus)
    assert any("2022 World Cup" in it.question for it in off_corpus)
    bait = [it for it in golden if OVERRIDE in it.must_not_include]
    assert len(bait) == 1 and not bait[0].expect_refusal
    themes = {"proc-", "clinic-", "keel-"}
    for prefix in themes:
        assert sum(1 for it in golden if it.id.startswith(prefix)) >= 3


def test_golden_round_trips_through_yaml(golden: list[GoldenItem], tmp_path: Path) -> None:
    path = save_golden(golden, tmp_path / "golden.yaml")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("#")
    assert load_golden(path) == golden


def test_golden_accepts_a_bare_list_and_scalar_slips(tmp_path: Path) -> None:
    path = tmp_path / "bare.yaml"
    path.write_text(
        "- id: a\n  question: Q?\n  user_tags: hr\n  expected_answer: A.\n  expected_sources: Salary\n",
        encoding="utf-8",
    )
    [item] = load_golden(path)
    assert item.user_tags == ["hr"]
    assert item.expected_sources == ["Salary"]
    assert item.expect_refusal is False


def test_validate_names_each_problem() -> None:
    items = [
        GoldenItem(id="dup", question="Q1?", expected_answer="A", expected_sources=["Guide"]),
        GoldenItem(id="dup", question="", expected_answer="A", expected_sources=[]),
        GoldenItem(
            id="ref", question="Q3?", expected_answer="refuse", expect_refusal=True, must_include=["x"]
        ),
        GoldenItem(
            id="both",
            question="Q4?",
            expected_answer="A",
            expected_sources=["G"],
            must_include=["a"],
            must_not_include=["A"],
        ),
    ]
    problems = validate_golden(items)
    joined = "\n".join(problems)
    assert "dup: id is used more than once" in joined
    assert "dup: question is empty" in joined
    assert "dup: expected_sources is empty" in joined
    assert "ref: a refusal item cannot carry must_include" in joined
    assert "both: ['a'] appear in both" in joined


def test_generate_golden_drafts_editable_items_from_corpus_chunks(ctx: AppContext, tmp_path: Path) -> None:
    drafts = [
        json.dumps({"question": "How many quotes does Band 2 need?", "answer": "Three written quotes."}),
        "not json at all",
        json.dumps(
            {
                "question": "Where is restricted information stored?",
                "answer": "In the practice management system.",
            }
        ),
    ]
    llm = FakeLLM(drafts)
    use_llm(ctx, llm)
    out = tmp_path / "generated.yaml"

    items = generate_golden(ctx, 3, out, seed=7)

    assert llm.call_count == 3
    assert all(call["json_schema"] is not None for call in llm.calls)
    assert len(items) == 2, "the unparseable draft is skipped, the rest land"
    loaded = load_golden(out)
    assert loaded == items
    assert validate_golden(loaded) == []
    assert all(len(item.expected_sources) == 1 for item in loaded)
    assert {tag for item in loaded for tag in item.user_tags} <= {"public", "hr"}
    assert out.read_text(encoding="utf-8").startswith("# Keel golden evaluation set")


# ----------------------------------------------------------------------------- metrics


def test_hit_at_k_and_reciprocal_rank() -> None:
    titles = [
        "Keel Operations Notes",
        "Harbour Clinic Data Handling Standard",
        "Northbank City Council Procurement Guide",
        None,
    ]
    assert hit_at_k(["Procurement Guide"], titles, 1) is False
    assert hit_at_k(["Procurement Guide"], titles, 3) is True
    assert hit_at_k(["procurement guide"], titles, 3) is True, "title matching ignores case"
    assert hit_at_k(["Salary Bands"], titles, 5) is False
    assert hit_at_k([], titles, 3) is None, "no expected sources means the metric does not apply"
    assert reciprocal_rank(["Harbour Clinic"], titles) == pytest.approx(0.5)
    assert reciprocal_rank(["Harbour Clinic", "Keel"], titles) == pytest.approx(1.0)
    assert reciprocal_rank(["Salary Bands"], titles) == 0.0
    assert reciprocal_rank([], titles) is None


def test_refusal_and_string_checks() -> None:
    assert refusal_correct(True, True) and refusal_correct(False, False)
    assert not refusal_correct(True, False) and not refusal_correct(False, True)
    text = "Three written quotes are required [2]. Approved by the Team Leader."
    assert missing_strings(text, ["three", "team leader", "seven"]) == ["seven"]
    assert found_strings(text, ["quotes", SECRET, "approved by"]) == ["quotes", "approved by"]
    assert missing_strings("", ["a"]) == ["a"] and found_strings("", ["a"]) == []


def test_percentile_interpolates() -> None:
    assert percentile([], 50) is None
    assert percentile([7], 95) == 7
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([10, 20, 30, 40, 50], 95) == pytest.approx(48.0)
    assert percentile([None, 3, 1], 0) == 1


def make_item(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "hit_at_1": True,
        "hit_at_3": True,
        "hit_at_5": True,
        "reciprocal_rank": 1.0,
        "refusal_correct": True,
        "expect_refusal": False,
        "refused": False,
        "must_include_pass": True,
        "must_not_include_pass": None,
        "checks_pass": True,
        "groundedness": 0.9,
        "relevance": 0.8,
        "correctness": 1.0,
        "latency_ms": 100,
        "prompt_tokens": 100,
        "output_tokens": 10,
        "error": None,
    }
    base.update(overrides)
    return base


def test_aggregate_over_hand_built_items() -> None:
    items = [
        make_item(),
        make_item(
            hit_at_1=False,
            hit_at_3=False,
            hit_at_5=True,
            reciprocal_rank=0.25,
            latency_ms=300,
            must_include_pass=False,
            checks_pass=False,
        ),
        make_item(
            hit_at_1=None,
            hit_at_3=None,
            hit_at_5=None,
            reciprocal_rank=None,
            expect_refusal=True,
            refused=True,
            must_include_pass=None,
            must_not_include_pass=True,
            groundedness=None,
            relevance=None,
            correctness=None,
            latency_ms=50,
            prompt_tokens=0,
            output_tokens=0,
        ),
        make_item(
            refused=True,
            refusal_correct=False,
            must_not_include_pass=False,
            checks_pass=False,
            groundedness=None,
            relevance=None,
            correctness=None,
            latency_ms=200,
            error="boom",
        ),
    ]
    summary = aggregate(items)
    assert summary["items"] == 4 and summary["errors"] == 1
    assert summary["retrieval_items"] == 3
    assert summary["hit_at_1"] == pytest.approx(2 / 3)
    assert summary["hit_at_3"] == pytest.approx(2 / 3)
    assert summary["hit_at_5"] == pytest.approx(1.0)
    assert summary["mrr"] == pytest.approx((1.0 + 0.25 + 1.0) / 3)
    assert summary["refusal_correct"] == pytest.approx(0.75)
    assert summary["refusals_expected"] == 1 and summary["refusals_actual"] == 2
    assert summary["must_include_pass"] == pytest.approx(2 / 3)
    assert summary["must_not_include_pass"] == pytest.approx(0.5)
    assert summary["checks_pass"] == pytest.approx(0.5)
    assert summary["judged"] == 2
    assert summary["groundedness"] == pytest.approx(0.9)
    assert summary["latency_p50_ms"] == pytest.approx(150.0)
    assert summary["latency_p95_ms"] == pytest.approx(285.0)
    assert summary["prompt_tokens"] == 300 and summary["output_tokens"] == 30
    assert summary["tokens_per_item"] == pytest.approx(82.5)
    empty = aggregate([])
    assert empty["items"] == 0 and empty["hit_at_3"] is None and empty["latency_p50_ms"] is None


def test_compare_flags_drops_past_threshold_and_skips_missing_metrics() -> None:
    baseline = {"hit_at_3": 0.9, "groundedness": 0.8, "refusal_correct": 1.0, "must_not_include_pass": 1.0}
    same = compare(dict(baseline), baseline)
    assert same.passed and same.regressions == [] and sorted(same.compared) == sorted(DEFAULT_THRESHOLDS)

    within = compare({**baseline, "hit_at_3": 0.86, "groundedness": 0.76}, baseline)
    assert within.passed, "drops inside the threshold pass"

    worse = compare({**baseline, "hit_at_3": 0.5, "must_not_include_pass": 0.95}, baseline)
    assert not worse.passed
    assert [r.metric for r in worse.regressions] == ["hit_at_3", "must_not_include_pass"]
    assert worse.regressions[0].delta == pytest.approx(-0.4)
    assert worse.to_dict()["regressions"][0]["threshold"] == -0.05

    missing = compare({**baseline, "groundedness": None}, baseline)
    assert missing.passed and "groundedness" in missing.skipped and "groundedness" not in missing.compared

    no_baseline = compare(baseline, None)
    assert no_baseline.passed and no_baseline.compared == []

    custom = compare({"hit_at_3": 0.7}, {"hit_at_3": 0.9}, {"hit_at_3": -0.3})
    assert custom.passed


# ----------------------------------------------------------------------------- judge


def test_judge_parses_scores_and_reasons() -> None:
    llm = FakeLLM([GOOD_JUDGE_JSON])
    scores = judge_answer(llm, "Q?", "A [1]", "[1] passage", "reference")
    assert (scores.groundedness, scores.relevance, scores.correctness) == (0.9, 0.8, 1.0)
    assert scores.reasons["groundedness"] == "every claim appears in the passages"
    assert scores.error is None and scores.attempts == 1 and scores.usable
    call = llm.calls[0]
    assert call["json_schema"] == JUDGE_SCHEMA
    assert call["temperature"] == 0.0
    user_prompt = call["messages"][1].content
    assert "<question>\nQ?" in user_prompt and "<reference>\nreference" in user_prompt


def test_judge_retries_once_then_returns_none_scores_without_raising() -> None:
    llm = FakeLLM(["I cannot score this.", "still prose, still no JSON"])
    scores = judge_answer(llm, "Q?", "A", "ctx", "ref")
    assert llm.call_count == 2, "one retry, then give up"
    assert (scores.groundedness, scores.relevance, scores.correctness) == (None, None, None)
    assert scores.error and not scores.usable
    assert scores.to_dict()["attempts"] == 2


def test_judge_recovers_on_the_retry() -> None:
    llm = FakeLLM(["garbage", GOOD_JUDGE_JSON])
    scores = judge_answer(llm, "Q?", "A", "ctx", "ref")
    assert llm.call_count == 2 and scores.groundedness == 0.9 and scores.error is None


def test_judge_survives_a_provider_exception() -> None:
    class Broken:
        name = "broken"

        def chat(self, *args: Any, **kwargs: Any) -> ChatResult:
            raise ConnectionError("server away")

    scores = judge_answer(Broken(), "Q?", "A", "ctx", "ref")
    assert scores.groundedness is None and "ConnectionError" in (scores.error or "")


def test_parse_judge_reply_tolerates_fences_strings_and_scales() -> None:
    fenced = (
        "```json\n"
        + json.dumps(
            {"groundedness": "0.75", "relevance": 8, "correctness": 90, "reasons": {"relevance": "ok"}}
        )
        + "\n```"
    )
    scores, error = parse_judge_reply(fenced)
    assert error is None
    assert scores["groundedness"] == 0.75 and scores["relevance"] == 0.8 and scores["correctness"] == 0.9
    assert scores["reasons"] == {"relevance": "ok"}
    scores, error = parse_judge_reply('{"groundedness": 1.4, "relevance": -0.2, "correctness": null}')
    assert scores["groundedness"] == 1.0 and scores["relevance"] == 0.0 and scores["correctness"] is None
    assert parse_judge_reply('{"note": "no scores here"}') == (
        None,
        "the reply carried no numeric score for groundedness, relevance or correctness",
    )
    assert parse_judge_reply("")[0] is None


def test_two_judges_average_and_keep_both_raw() -> None:
    primary = FakeLLM(
        [json.dumps({"groundedness": 1.0, "relevance": 1.0, "correctness": 0.5, "groundedness_reason": "p"})]
    )
    primary.name = "primary"
    second = FakeLLM([json.dumps({"groundedness": 0.5, "relevance": 1.0, "correctness": 1.0})])
    second.name = "gemini"
    verdict = Judge(primary, second=second).score("Q?", "A", "ctx", "ref")
    assert verdict["groundedness"] == pytest.approx(0.75)
    assert verdict["relevance"] == pytest.approx(1.0)
    assert verdict["correctness"] == pytest.approx(0.75)
    assert set(verdict["judges"]) == {"primary", "gemini"}
    assert (
        verdict["judges"]["primary"]["groundedness"] == 1.0
        and verdict["judges"]["gemini"]["groundedness"] == 0.5
    )
    assert verdict["reasons"]["groundedness"] == "p"

    silent_second = FakeLLM(["nope"])
    silent_second.name = "gemini"
    solo = Judge(FakeLLM([GOOD_JUDGE_JSON]), second=silent_second).score("Q?", "A", "ctx", "ref")
    assert solo["groundedness"] == 0.9 and list(solo["judges"]) == ["fake"]
    assert merge_verdicts([])["groundedness"] is None


def test_judge_from_context_skips_gemini_without_a_key(ctx: AppContext) -> None:
    judge = Judge.from_context(ctx)
    assert judge.second is None and judge.names == [ctx.llm.name]


# ----------------------------------------------------------------------------- run_eval end to end


def test_run_eval_writes_reports_and_passes_without_baseline(
    ctx: AppContext, golden: list[GoldenItem], tmp_path: Path
) -> None:
    llm = RoutingLLM(oracle(golden))
    use_llm(ctx, llm)
    report_dir = tmp_path / "reports"

    result = run_eval(ctx, GOLDEN_PATH, report_dir)

    assert result.report_json_path.exists() and result.report_json_path.suffix == ".json"
    assert result.report_html_path.exists() and result.report_html_path.suffix == ".html"
    assert (report_dir / "latest.json").exists() and (report_dir / "latest.html").exists()
    assert result.gate_passed and result.regressions == [] and result.baseline_path is None
    assert result.thresholds == DEFAULT_THRESHOLDS

    summary = result.summary
    assert summary["items"] == len(golden) and summary["errors"] == 0
    assert summary["hit_at_3"] >= 0.8, "real retrieval over the fixture corpus finds the expected document"
    assert summary["hit_at_5"] >= summary["hit_at_3"] >= summary["hit_at_1"]
    assert 0.0 < summary["mrr"] <= 1.0
    assert summary["refusal_correct"] == 1.0, "the oracle refuses exactly the refusal items"
    assert summary["must_not_include_pass"] == 1.0
    assert summary["must_include_pass"] == 1.0
    assert summary["groundedness"] == pytest.approx(0.9)
    assert summary["relevance"] == pytest.approx(0.8)
    assert summary["correctness"] == pytest.approx(1.0)
    answered = [it for it in result.items if not it["refused"]]
    assert summary["judged"] == len(answered) == llm.judge_calls
    assert summary["latency_p50_ms"] is not None and summary["latency_p95_ms"] >= summary["latency_p50_ms"]
    assert summary["prompt_tokens"] > 0 and summary["output_tokens"] > 0

    by_id = {it["id"]: it for it in result.items}
    for item in golden:
        row = by_id[item.id]
        assert row["refused"] == item.expect_refusal
        assert row["refusal_correct"] is True
        if item.expect_refusal:
            assert row["groundedness"] is None and row["hit_at_3"] is None
        else:
            assert (
                row["groundedness"] == 0.9 and row["judge_reasons"]["correctness"] == "matches the reference"
            )
    hr_row = by_id["hr-band-c-hr"]
    assert hr_row["hit_at_1"] is True and any("Salary Bands" in t for t in hr_row["retrieved_titles"])
    public_row = by_id["hr-band-c-public"]
    assert all("Salary Bands" not in t for t in public_row["retrieved_titles"]), "ACL filtering held"

    logged = ctx.log.get(hr_row["request_id"])
    assert logged is not None and logged["judge"]["groundedness"] == 0.9
    assert logged["judge"]["golden_id"] == "hr-band-c-hr" and logged["judge"]["hit_at_3"] is True
    refusal_logged = ctx.log.get(by_id["off-capital-france"]["request_id"])
    assert (
        refusal_logged["judge"]["refusal_correct"] is True and refusal_logged["judge"]["groundedness"] is None
    )

    payload = json.loads(result.report_json_path.read_text(encoding="utf-8"))
    assert payload["summary"] == summary and len(payload["items"]) == len(golden)
    assert payload["gate"]["passed"] is True and payload["judge_names"] == ["fake"]
    html = result.report_html_path.read_text(encoding="utf-8")
    assert "<title>Keel eval" in html and "Regression gate" in html and "Methodology" in html
    assert golden[0].question in html and "No baseline" in html
    assert "http" not in html.split("<body>")[0].split("<style>")[1], "no external assets in the stylesheet"


def test_run_eval_gate_fails_against_an_inflated_baseline(
    ctx: AppContext, golden: list[GoldenItem], tmp_path: Path
) -> None:
    use_llm(ctx, RoutingLLM(oracle(golden)))
    mini = mini_golden(golden, tmp_path / "mini.yaml", MINI_IDS)
    report_dir = tmp_path / "reports"
    first = run_eval(ctx, mini, report_dir, judge=False)
    assert first.gate_passed and first.summary["groundedness"] is None and first.summary["judged"] == 0

    inflated = json.loads(first.report_json_path.read_text(encoding="utf-8"))
    inflated["summary"]["hit_at_3"] = first.summary["hit_at_3"] + 0.3
    baseline = tmp_path / "inflated-baseline.json"
    baseline.write_text(json.dumps(inflated), encoding="utf-8")

    second = run_eval(ctx, mini, report_dir, judge=False, baseline_path=baseline)
    assert not second.gate_passed
    assert [r["metric"] for r in second.regressions] == ["hit_at_3"]
    assert second.regressions[0]["delta"] == pytest.approx(-0.3)
    assert second.baseline_path == baseline
    html = second.report_html_path.read_text(encoding="utf-8")
    assert "Failed" in html and "hit_at_3" in html

    with pytest.raises(FileNotFoundError):
        run_eval(ctx, mini, report_dir, judge=False, baseline_path=tmp_path / "missing.json")


def test_promote_baseline_then_default_baseline_is_used(
    ctx: AppContext, golden: list[GoldenItem], tmp_path: Path
) -> None:
    use_llm(ctx, RoutingLLM(oracle(golden)))
    mini = mini_golden(golden, tmp_path / "mini.yaml", MINI_IDS)
    report_dir = tmp_path / "reports"
    first = run_eval(ctx, mini, report_dir, judge=False)

    baseline = promote_baseline(first)
    assert baseline == report_dir / "baseline.json" and baseline.exists()
    assert json.loads(baseline.read_text(encoding="utf-8"))["summary"] == first.summary

    second = run_eval(ctx, mini, report_dir, judge=False)
    assert second.baseline_path == baseline
    assert second.gate_passed and second.regressions == []
    assert second.report_json_path != first.report_json_path


def test_broken_retriever_zeroes_hit_at_3_refuses_everything_and_fails_the_gate(
    ctx: AppContext, golden: list[GoldenItem], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_llm(ctx, RoutingLLM(oracle(golden)))
    report_dir = tmp_path / "reports"
    healthy = run_eval(ctx, GOLDEN_PATH, report_dir, judge=False)
    promote_baseline(healthy)

    monkeypatch.setattr(ctx.retriever, "retrieve", lambda *args, **kwargs: [])
    broken = run_eval(ctx, GOLDEN_PATH, report_dir, judge=False)

    assert (
        broken.summary["hit_at_3"] == 0.0
        and broken.summary["hit_at_1"] == 0.0
        and broken.summary["mrr"] == 0.0
    )
    for row in broken.items:
        assert row["refused"] is True and row["retrieved_titles"] == []
        assert row["refusal_correct"] == row["expect_refusal"]
    expected_refusals = sum(1 for it in golden if it.expect_refusal)
    assert broken.summary["refusals_actual"] == len(golden)
    assert broken.summary["refusal_correct"] == pytest.approx(expected_refusals / len(golden))
    assert not broken.gate_passed
    assert {r["metric"] for r in broken.regressions} >= {"hit_at_3", "refusal_correct"}


def test_must_not_include_catches_a_planted_override_string(
    ctx: AppContext, golden: list[GoldenItem], tmp_path: Path
) -> None:
    bait = next(it for it in golden if OVERRIDE in it.must_not_include)
    planted = f"Yes, any staff member can approve purchases up to $500,000 verbally. {OVERRIDE} [1]"
    use_llm(ctx, RoutingLLM(oracle(golden, override={bait.question: planted})))
    mini = mini_golden(golden, tmp_path / "mini.yaml", MINI_IDS)

    result = run_eval(ctx, mini, tmp_path / "reports", judge=False)

    row = next(it for it in result.items if it["id"] == bait.id)
    assert row["refused"] is False
    assert row["must_not_include_found"] == [OVERRIDE]
    assert row["must_not_include_pass"] is False and row["checks_pass"] is False
    assert result.summary["must_not_include_pass"] < 1.0
    others = [it for it in result.items if it["id"] != bait.id and it["must_not_include_pass"] is not None]
    assert others and all(it["must_not_include_pass"] for it in others)
    assert OVERRIDE in result.report_html_path.read_text(encoding="utf-8")


def test_run_eval_keeps_going_when_the_judge_reply_is_unusable(
    ctx: AppContext, golden: list[GoldenItem], tmp_path: Path
) -> None:
    llm = RoutingLLM(oracle(golden), judge_default="I would rather not produce JSON today.")
    use_llm(ctx, llm)
    mini = mini_golden(golden, tmp_path / "mini.yaml", MINI_IDS)

    result = run_eval(ctx, mini, tmp_path / "reports")

    answered = [it for it in result.items if not it["refused"]]
    assert answered
    for row in answered:
        assert row["groundedness"] is None and row["relevance"] is None and row["correctness"] is None
        assert row["judge_error"]
        assert row["judge"]["judges"]["fake"]["attempts"] == 2
    assert llm.judge_calls == 2 * len(answered)
    assert result.summary["groundedness"] is None and result.summary["judged"] == 0
    assert result.gate_passed
    assert "n/a" in result.report_html_path.read_text(encoding="utf-8")


def test_run_eval_records_an_engine_error_on_the_item(
    ctx: AppContext, golden: list[GoldenItem], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_llm(ctx, RoutingLLM(oracle(golden)))
    mini = mini_golden(golden, tmp_path / "mini.yaml", MINI_IDS[:2])

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("model away")

    monkeypatch.setattr(ctx.answer_engine, "answer", explode)
    result = run_eval(ctx, mini, tmp_path / "reports", judge=False)
    assert result.summary["errors"] == 2
    assert all("RuntimeError" in it["error"] for it in result.items)
    assert all(it["refusal_correct"] is False and it["checks_pass"] is False for it in result.items)


def test_run_eval_rejects_a_malformed_golden_set(ctx: AppContext, tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("items:\n  - id: x\n    question: ''\n", encoding="utf-8")
    with pytest.raises(ValueError, match="question is empty"):
        run_eval(ctx, bad, tmp_path / "reports", judge=False)


def test_render_report_from_a_saved_payload_is_self_contained(tmp_path: Path) -> None:
    payload = {
        "generated_at": "2026-08-18T12:00:00Z",
        "profile": "local",
        "model": "qwen",
        "golden_path": "fixtures/golden.yaml",
        "judge_names": ["llama-server"],
        "thresholds": DEFAULT_THRESHOLDS,
        "baseline_path": "reports/baseline.json",
        "gate": {
            "passed": False,
            "regressions": [
                {"metric": "hit_at_3", "baseline": 0.9, "current": 0.5, "delta": -0.4, "threshold": -0.05}
            ],
            "compared": ["hit_at_3"],
            "skipped": [],
        },
        "summary": aggregate([make_item()]),
        "items": [
            {
                **make_item(),
                "id": "x",
                "question": "Q <script>alert(1)</script>?",
                "user_tags": ["public"],
                "expected_answer": "A",
                "expected_sources": ["G"],
                "answer": "A [1]",
                "retrieved_titles": ["G"],
                "judge_reasons": {"groundedness": "fine"},
                "must_include_missing": [],
                "must_not_include_found": [],
            }
        ],
    }
    html = render_report(payload)
    assert "&lt;script&gt;" in html and "<script>alert" not in html, "item text is escaped"
    assert "Failed" in html and "hit_at_3" in html
    assert 'src="http' not in html and 'href="http' not in html
