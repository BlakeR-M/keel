"""Evaluation metrics: retrieval hit@k and MRR against expected source titles, refusal correctness,
must_include / must_not_include string checks, latency percentiles and token counts. `aggregate()`
folds per-item results into one summary; `compare()` holds a summary against a baseline and names
every metric that dropped by more than its threshold.

Per-item results are plain dicts (the JSON report keeps the same shape) so a summary can be rebuilt
from a saved report without the objects that produced it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

HIT_KS: tuple[int, ...] = (1, 3, 5)

# Allowed change per metric between baseline and current, as an absolute delta on a 0..1 scale.
# A negative value is the drop that is still tolerated; 0.0 means any drop fails the gate.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "hit_at_3": -0.05,
    "groundedness": -0.05,
    "refusal_correct": -0.05,
    "must_not_include_pass": 0.0,
}

_EPSILON = 1e-9


# ----------------------------------------------------------------------------- retrieval


def _norm(text: str | None) -> str:
    return " ".join((text or "").lower().split())


def title_matches(expected: str, title: str | None) -> bool:
    """True when the expected substring occurs in the title, ignoring case and extra whitespace."""
    needle = _norm(expected)
    return bool(needle) and needle in _norm(title)


def first_hit_rank(expected_sources: Sequence[str], retrieved_titles: Sequence[str | None]) -> int | None:
    """1-based rank of the first retrieved title matching any expected source, or None."""
    for rank, title in enumerate(retrieved_titles, start=1):
        if any(title_matches(expected, title) for expected in expected_sources):
            return rank
    return None


def hit_at_k(expected_sources: Sequence[str], retrieved_titles: Sequence[str | None], k: int) -> bool | None:
    """True when an expected source appears in the top `k` retrieved titles. None when the item
    names no expected sources (retrieval metrics do not apply to it)."""
    if not expected_sources:
        return None
    return first_hit_rank(expected_sources, list(retrieved_titles)[:k]) is not None


def reciprocal_rank(expected_sources: Sequence[str], retrieved_titles: Sequence[str | None]) -> float | None:
    """1 / rank of the first matching title, 0.0 when none matched, None when not applicable."""
    if not expected_sources:
        return None
    rank = first_hit_rank(expected_sources, retrieved_titles)
    return 1.0 / rank if rank else 0.0


# ----------------------------------------------------------------------------- behaviour checks


def refusal_correct(expected: bool, actual: bool) -> bool:
    """True when the system refused exactly when the golden item said it should."""
    return bool(expected) == bool(actual)


def missing_strings(text: str, must_include: Iterable[str]) -> list[str]:
    """The must_include strings absent from `text`, compared without case."""
    haystack = (text or "").lower()
    return [s for s in must_include if s and s.lower() not in haystack]


def found_strings(text: str, must_not_include: Iterable[str]) -> list[str]:
    """The must_not_include strings present in `text`, compared without case."""
    haystack = (text or "").lower()
    return [s for s in must_not_include if s and s.lower() in haystack]


# ----------------------------------------------------------------------------- statistics


def mean(values: Iterable[float | int | None]) -> float | None:
    """Arithmetic mean of the non-None values, None when there are none."""
    present = [float(v) for v in values if v is not None]
    return sum(present) / len(present) if present else None


def rate(values: Iterable[bool | None]) -> float | None:
    """Share of True among the non-None values, None when there are none."""
    present = [bool(v) for v in values if v is not None]
    return sum(present) / len(present) if present else None


def percentile(values: Iterable[float | int | None], p: float) -> float | None:
    """The p-th percentile (0..100) with linear interpolation between ranks, None when empty."""
    present = sorted(float(v) for v in values if v is not None)
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    position = (len(present) - 1) * (max(0.0, min(100.0, p)) / 100.0)
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return present[lo]
    weight = position - lo
    return present[lo] * (1.0 - weight) + present[hi] * weight


# ----------------------------------------------------------------------------- aggregate


def aggregate(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Fold per-item result dicts into a summary. Rates over checks that apply to no item are None
    rather than 1.0, so a missing check reads as missing rather than perfect."""
    summary: dict[str, Any] = {
        "items": len(items),
        "errors": sum(1 for it in items if it.get("error")),
        "retrieval_items": sum(1 for it in items if it.get("hit_at_3") is not None),
    }
    for k in HIT_KS:
        summary[f"hit_at_{k}"] = rate(it.get(f"hit_at_{k}") for it in items)
    summary["mrr"] = mean(it.get("reciprocal_rank") for it in items)
    summary["refusal_correct"] = rate(it.get("refusal_correct") for it in items)
    summary["refusals_expected"] = sum(1 for it in items if it.get("expect_refusal"))
    summary["refusals_actual"] = sum(1 for it in items if it.get("refused"))
    summary["must_include_pass"] = rate(it.get("must_include_pass") for it in items)
    summary["must_not_include_pass"] = rate(it.get("must_not_include_pass") for it in items)
    summary["checks_pass"] = rate(it.get("checks_pass") for it in items)
    summary["judged"] = sum(1 for it in items if it.get("groundedness") is not None)
    for metric in ("groundedness", "relevance", "correctness"):
        summary[metric] = mean(it.get(metric) for it in items)
    latencies = [it.get("latency_ms") for it in items]
    summary["latency_p50_ms"] = percentile(latencies, 50)
    summary["latency_p95_ms"] = percentile(latencies, 95)
    summary["latency_mean_ms"] = mean(latencies)
    summary["prompt_tokens"] = int(sum(int(it.get("prompt_tokens") or 0) for it in items))
    summary["output_tokens"] = int(sum(int(it.get("output_tokens") or 0) for it in items))
    summary["tokens_per_item"] = (
        (summary["prompt_tokens"] + summary["output_tokens"]) / len(items) if items else None
    )
    return summary


# ----------------------------------------------------------------------------- gate


@dataclass
class Regression:
    """One metric that dropped past its threshold."""

    metric: str
    baseline: float
    current: float
    delta: float
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GateResult:
    """Outcome of comparing a summary against a baseline. `compared` lists the metrics that were
    checked; `skipped` lists metrics absent or None on either side, which cannot fail the gate."""

    passed: bool
    regressions: list[Regression] = field(default_factory=list)
    compared: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "regressions": [r.to_dict() for r in self.regressions],
            "compared": list(self.compared),
            "skipped": list(self.skipped),
        }


def compare(
    summary: dict[str, Any],
    baseline_summary: dict[str, Any] | None,
    thresholds: dict[str, float] | None = None,
) -> GateResult:
    """Hold `summary` against `baseline_summary`. A metric regresses when
    `current - baseline < threshold` (thresholds are zero or negative). With no baseline the gate
    passes and compares nothing."""
    limits = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    if not baseline_summary:
        return GateResult(passed=True, skipped=sorted(limits))
    result = GateResult(passed=True)
    for metric, threshold in limits.items():
        before = baseline_summary.get(metric)
        now = summary.get(metric)
        if not _is_number(before) or not _is_number(now):
            result.skipped.append(metric)
            continue
        result.compared.append(metric)
        delta = float(now) - float(before)
        if delta < float(threshold) - _EPSILON:
            result.regressions.append(Regression(metric, float(before), float(now), delta, float(threshold)))
    result.passed = not result.regressions
    return result


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
