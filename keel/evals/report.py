"""The HTML eval report: one self-contained page (inline CSS, no scripts, no external assets) with
summary tiles, the regression gate, a per-item table and the methodology. `render_report()` takes the
same payload `run.py` writes as JSON, so a report can be re-rendered from a saved run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, select_autoescape

METHODOLOGY = (
    "Each golden item is asked through the same answer engine a user reaches, as the eval user carrying "
    "the item's ACL tags. Retrieval hit@k and MRR compare the titles of the chunks the engine retrieved "
    "(after permission filtering and quarantine) with the item's expected sources. Refusal correctness "
    "compares whether the engine refused with whether the item says it should. must_include and "
    "must_not_include are case-insensitive substring checks on the answer text; must_not_include carries "
    "the leak and override checks. Groundedness, relevance and correctness come from an LLM judge (the "
    "deployment's own model, plus Gemini as a second opinion when a key is configured) that reads the "
    "question, the retrieved passages, the answer and the reference answer and returns three scores in "
    "0..1 with reasons; refusals are not judged. Latency covers retrieval and generation for one item. "
    "The gate compares this run's summary with the saved baseline and fails when a gated metric drops by "
    "more than its threshold."
)

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Keel eval {{ generated_at }}</title>
<style>
  :root {
    --bg: #0f1216; --panel: #161b22; --line: #262d37; --text: #d7dde5; --muted: #8b95a3;
    --accent: #7aa2c4; --good: #58a874; --warn: #d0a24a; --bad: #d16b6b;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 1240px; margin: 0 auto; padding: 32px 24px 64px; }
  h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; }
  h2 { font-size: 16px; font-weight: 600; margin: 36px 0 12px; color: var(--text); }
  .sub { color: var(--muted); margin: 0 0 24px; }
  .sub code { color: var(--accent); }
  .tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 12px; }
  .tile { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; }
  .tile .label { color: var(--muted); font-size: 12px; letter-spacing: .02em; }
  .tile .value { font-size: 24px; font-weight: 600; margin-top: 4px; font-variant-numeric: tabular-nums; }
  .tile .note { color: var(--muted); font-size: 12px; margin-top: 2px; }
  .gate { border-radius: 8px; padding: 14px 16px; border: 1px solid var(--line); background: var(--panel); }
  .gate.pass { border-color: var(--good); }
  .gate.fail { border-color: var(--bad); }
  .gate strong { font-size: 15px; }
  .pill { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 12px; border: 1px solid var(--line); color: var(--muted); }
  .pill.pass { color: var(--good); border-color: var(--good); }
  .pill.fail { color: var(--bad); border-color: var(--bad); }
  .pill.na { color: var(--muted); }
  .wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }
  table { border-collapse: collapse; width: 100%; min-width: 1100px; background: var(--panel); }
  th, td { text-align: left; vertical-align: top; padding: 9px 10px; border-bottom: 1px solid var(--line); }
  th { color: var(--muted); font-weight: 500; font-size: 12px; white-space: nowrap; position: sticky; top: 0; background: var(--panel); }
  td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
  td.q { min-width: 220px; }
  td .id { color: var(--muted); font-size: 12px; display: block; }
  td .exp { color: var(--muted); font-size: 12px; display: block; margin-top: 4px; }
  td .got { display: block; margin-top: 4px; white-space: pre-wrap; }
  td .reason { color: var(--muted); font-size: 12px; display: block; margin-top: 4px; }
  td .tags { color: var(--accent); font-size: 12px; }
  .missing { color: var(--bad); }
  p.method { color: var(--muted); max-width: 900px; }
  footer { color: var(--muted); font-size: 12px; margin-top: 40px; }
</style>
</head>
<body>
<main>
  <h1>Keel evaluation report</h1>
  <p class="sub">{{ generated_at }} &middot; profile <code>{{ profile }}</code> &middot; model <code>{{ model or "unknown" }}</code>
     &middot; golden set <code>{{ golden_path }}</code> &middot; {{ summary["items"] }} items
     {% if judge_names %}&middot; judge {{ judge_names | join(" + ") }}{% else %}&middot; judge off{% endif %}</p>

  <div class="tiles">
    <div class="tile"><div class="label">hit@1</div><div class="value">{{ pct(summary["hit_at_1"]) }}</div><div class="note">{{ summary["retrieval_items"] }} retrieval items</div></div>
    <div class="tile"><div class="label">hit@3</div><div class="value">{{ pct(summary["hit_at_3"]) }}</div></div>
    <div class="tile"><div class="label">hit@5</div><div class="value">{{ pct(summary["hit_at_5"]) }}</div></div>
    <div class="tile"><div class="label">MRR</div><div class="value">{{ num(summary["mrr"]) }}</div></div>
    <div class="tile"><div class="label">groundedness</div><div class="value">{{ num(summary["groundedness"]) }}</div><div class="note">{{ summary["judged"] }} judged</div></div>
    <div class="tile"><div class="label">relevance</div><div class="value">{{ num(summary["relevance"]) }}</div></div>
    <div class="tile"><div class="label">correctness</div><div class="value">{{ num(summary["correctness"]) }}</div></div>
    <div class="tile"><div class="label">refusal correct</div><div class="value">{{ pct(summary["refusal_correct"]) }}</div><div class="note">{{ summary["refusals_actual"] }} refused, {{ summary["refusals_expected"] }} expected</div></div>
    <div class="tile"><div class="label">must_include pass</div><div class="value">{{ pct(summary["must_include_pass"]) }}</div></div>
    <div class="tile"><div class="label">must_not_include pass</div><div class="value">{{ pct(summary["must_not_include_pass"]) }}</div></div>
    <div class="tile"><div class="label">latency p50</div><div class="value">{{ ms(summary["latency_p50_ms"]) }}</div><div class="note">p95 {{ ms(summary["latency_p95_ms"]) }}</div></div>
    <div class="tile"><div class="label">tokens</div><div class="value">{{ summary["prompt_tokens"] + summary["output_tokens"] }}</div><div class="note">{{ summary["prompt_tokens"] }} prompt, {{ summary["output_tokens"] }} output</div></div>
    {% if summary["errors"] %}<div class="tile"><div class="label">errors</div><div class="value missing">{{ summary["errors"] }}</div></div>{% endif %}
  </div>

  <h2>Regression gate</h2>
  <div class="gate {{ 'pass' if gate["passed"] else 'fail' }}">
    {% if baseline_path %}
      <strong>{{ "Passed" if gate["passed"] else "Failed" }}</strong> against baseline <code>{{ baseline_path }}</code>.
      Compared: {{ gate["compared"] | join(", ") if gate["compared"] else "nothing" }}.
      {% if gate["skipped"] %}Skipped (missing on one side): {{ gate["skipped"] | join(", ") }}.{% endif %}
      {% if gate["regressions"] %}
      <div class="wrap" style="margin-top:12px">
      <table style="min-width:0">
        <thead><tr><th>metric</th><th>baseline</th><th>current</th><th>delta</th><th>allowed</th></tr></thead>
        <tbody>
        {% for r in gate["regressions"] %}
          <tr><td>{{ r["metric"] }}</td><td class="num">{{ num(r["baseline"]) }}</td><td class="num">{{ num(r["current"]) }}</td>
              <td class="num missing">{{ "%+.3f" | format(r["delta"]) }}</td><td class="num">{{ "%+.3f" | format(r["threshold"]) }}</td></tr>
        {% endfor %}
        </tbody>
      </table>
      </div>
      {% endif %}
    {% else %}
      <strong>No baseline.</strong> The gate passes by default; promote this run (<code>promote_baseline</code>, which copies latest.json to baseline.json) to make it the baseline for the next run.
    {% endif %}
    <div class="note" style="color:var(--muted);margin-top:8px">Thresholds:
      {% for k, v in thresholds.items() %}<code>{{ k }} {{ "%+.2f" | format(v) }}</code>{% if not loop.last %}, {% endif %}{% endfor %}</div>
  </div>

  <h2>Items</h2>
  <div class="wrap">
  <table>
    <thead>
      <tr>
        <th>question / expected / got</th>
        <th>tags</th>
        <th>refusal</th>
        <th>hit@3</th>
        <th>ground.</th>
        <th>relev.</th>
        <th>correct.</th>
        <th>checks</th>
        <th>latency</th>
      </tr>
    </thead>
    <tbody>
    {% for it in items %}
      <tr>
        <td class="q">
          <span class="id">{{ it["id"] }}</span>
          {{ it["question"] }}
          <span class="exp">expected: {{ it["expected_answer"] }}{% if it["expected_sources"] %} &middot; sources: {{ it["expected_sources"] | join(", ") }}{% endif %}</span>
          <span class="got">{% if it["error"] %}<span class="missing">error: {{ it["error"] }}</span>{% else %}{{ it["answer"] }}{% endif %}</span>
          {% if it["retrieved_titles"] %}<span class="exp">retrieved: {{ it["retrieved_titles"] | join(" | ") }}</span>{% endif %}
          {% if it["judge_reasons"] %}<span class="reason">{% for k, v in it["judge_reasons"].items() %}{{ k }}: {{ v }}{% if not loop.last %} &middot; {% endif %}{% endfor %}</span>{% endif %}
        </td>
        <td><span class="tags">{{ it["user_tags"] | join(", ") }}</span></td>
        <td>
          {{ "refused" if it["refused"] else "answered" }}<br>
          <span class="pill {{ 'pass' if it["refusal_correct"] else 'fail' }}">{{ "as expected" if it["refusal_correct"] else ("expected refusal" if it["expect_refusal"] else "expected answer") }}</span>
        </td>
        <td>{{ flag(it["hit_at_3"]) }}{% if it["reciprocal_rank"] is not none %}<br><span class="id">rr {{ num(it["reciprocal_rank"]) }}</span>{% endif %}</td>
        <td class="num">{{ num(it["groundedness"]) }}</td>
        <td class="num">{{ num(it["relevance"]) }}</td>
        <td class="num">{{ num(it["correctness"]) }}</td>
        <td>
          {{ flag(it["checks_pass"]) }}
          {% if it["must_include_missing"] %}<br><span class="missing">missing: {{ it["must_include_missing"] | join(", ") }}</span>{% endif %}
          {% if it["must_not_include_found"] %}<br><span class="missing">found: {{ it["must_not_include_found"] | join(", ") }}</span>{% endif %}
        </td>
        <td class="num">{{ ms(it["latency_ms"]) }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  </div>

  <h2>Methodology</h2>
  <p class="method">{{ methodology }}</p>

  <footer>Keel eval &middot; report generated {{ generated_at }} &middot; JSON twin: <code>{{ json_name }}</code></footer>
</main>
</body>
</html>
"""


def _num(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.0f}%"


def _ms(value: Any) -> str:
    if value is None:
        return "n/a"
    ms = float(value)
    return f"{ms / 1000.0:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


def _flag(value: Any) -> Any:
    from markupsafe import Markup

    if value is None:
        return Markup('<span class="pill na">n/a</span>')
    if value:
        return Markup('<span class="pill pass">pass</span>')
    return Markup('<span class="pill fail">fail</span>')


_ENV = Environment(autoescape=select_autoescape(default=True, default_for_string=True))
_ENV.globals.update(num=_num, pct=_pct, ms=_ms, flag=_flag)
_REPORT = _ENV.from_string(_TEMPLATE)


def render_report(payload: dict[str, Any]) -> str:
    """Render the report page from a run payload (the same dict `run.py` saves as JSON)."""
    gate = payload.get("gate") or {"passed": True, "regressions": [], "compared": [], "skipped": []}
    return _REPORT.render(
        generated_at=payload.get("generated_at", ""),
        profile=payload.get("profile", ""),
        model=payload.get("model", ""),
        golden_path=payload.get("golden_path", ""),
        judge_names=payload.get("judge_names") or [],
        summary=payload.get("summary") or {},
        gate=gate,
        baseline_path=payload.get("baseline_path"),
        thresholds=payload.get("thresholds") or {},
        items=payload.get("items") or [],
        methodology=payload.get("methodology") or METHODOLOGY,
        json_name=Path(str(payload.get("report_json_path") or "")).name,
    )


def write_report(payload: dict[str, Any], path: str | Path) -> Path:
    """Render and write the report; returns the path written."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(render_report(payload), encoding="utf-8")
    return file
