"""Evaluation harness: golden set, LLM judge, metrics, HTML report and the regression gate.

Usage:
    from keel.evals import run_eval, promote_baseline, load_golden
    result = run_eval(ctx)                # fixtures/golden.yaml -> reports/eval-<stamp>.{json,html}
    promote_baseline(result)              # reports/latest.json -> reports/baseline.json
"""

from keel.evals.golden import GoldenItem, generate_golden, load_golden, save_golden
from keel.evals.golden import validate as validate_golden
from keel.evals.judge import Judge, JudgeScores, judge_answer
from keel.evals.metrics import DEFAULT_THRESHOLDS, GateResult, Regression, aggregate, compare
from keel.evals.report import render_report, write_report
from keel.evals.run import EvalResult, evaluate_item, promote_baseline, run_eval

__all__ = [
    "DEFAULT_THRESHOLDS",
    "EvalResult",
    "GateResult",
    "GoldenItem",
    "Judge",
    "JudgeScores",
    "Regression",
    "aggregate",
    "compare",
    "evaluate_item",
    "generate_golden",
    "judge_answer",
    "load_golden",
    "promote_baseline",
    "render_report",
    "run_eval",
    "save_golden",
    "validate_golden",
    "write_report",
]
