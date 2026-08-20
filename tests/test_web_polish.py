"""Structural checks on the polished web UI (Feat 14): the footer that names Keel with the profile and
air-gap state, the inline SVG favicon, calm copy free of hype words and em dashes, fluid CSS with hover
and focus states and tables that scroll inside their own frame, and the markup the fetch swap relies on.

Pages render through the app's own Jinja environment with a hand-built context, and the chat page
through a TestClient over a stub AppContext, so nothing here loads a model or opens a store.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from markupsafe import Markup

from keel.answer.prompts import REFUSAL
from keel.answer.types import User
from keel.web import views
from keel.web.app import STATIC_DIR, TEMPLATES_DIR, _env, app

HYPE_WORDS = [
    "revolutionary",
    "seamless",
    "cutting-edge",
    "world-class",
    "game-changing",
    "powerful",
    "effortless",
]
EM_DASH = "\u2014"

UI_FILES: list[Path] = sorted(TEMPLATES_DIR.glob("*.html")) + sorted(STATIC_DIR.iterdir())


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------- fixtures


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A TestClient over the app with a stub context installed, so the chat page renders with the
    profile and air-gap state and the lifespan handler leaves the model and the store alone."""
    app.state.ctx = SimpleNamespace(profile="local", settings=SimpleNamespace(airgap=True))
    app.state.ctx_owned = False
    with TestClient(app) as test_client:
        yield test_client
    app.state.ctx = None


def admin_context(**overrides: object) -> dict[str, object]:
    """The context admin.html expects, in the shape `keel.web.app.admin_context` builds."""
    today = datetime.now(UTC).date().isoformat()
    daily = [{"day": today, "requests": 3, "refused": 1, "avg_latency_ms": 1200.0}]
    context: dict[str, object] = {
        "active": "admin",
        "profile": "local",
        "totals": {
            "requests": 3,
            "refused": 1,
            "avg_latency_ms": 1200,
            "prompt_tokens": 2100,
            "output_tokens": 140,
        },
        "counts": {"documents": 5, "chunks": 27, "quarantined": 1, "pending_approvals": 0, "ledger_seq": 12},
        "recent": [],
        "sparklines": views.trend_sparklines(daily, 14),
        "trend_days": 14,
        "quarantine": [],
        "pending": [],
        "decided": [],
        "verify": None,
        "notice": None,
        "settings": {
            "host": "127.0.0.1",
            "port": 8400,
            "profile": "local",
            "airgap": True,
            "llm_model": "qwen2.5-3b-instruct",
            "min_relevance": 0.15,
            "guarded": False,
        },
    }
    context.update(overrides)
    return context


def result_context(**overrides: object) -> dict[str, object]:
    """The `r` context the result partial expects for a refused answer."""
    context: dict[str, object] = {
        "mode": "answer",
        "question": "What is the capital of France?",
        "user": User("public", ["public"]),
        "answer": SimpleNamespace(refused=True, error=None),
        "html": Markup(REFUSAL),
        "chips": [],
        "request_id": "abcdef0123456789",
        "latency_ms": 12,
        "prompt_tokens": 0,
        "output_tokens": 0,
    }
    context.update(overrides)
    return context


# ---------------------------------------------------------------------- copy


@pytest.mark.parametrize("path", UI_FILES, ids=lambda p: p.name)
def test_ui_copy_carries_no_hype_words(path: Path) -> None:
    text = read(path).lower()
    found = [word for word in HYPE_WORDS if word in text]
    assert found == [], f"{path.name} uses {found}"


@pytest.mark.parametrize("path", UI_FILES, ids=lambda p: p.name)
def test_ui_copy_carries_no_em_dashes(path: Path) -> None:
    assert EM_DASH not in read(path), f"{path.name} contains an em dash"


# ---------------------------------------------------------------------- base layout: favicon and footer


def test_base_template_declares_an_inline_svg_favicon() -> None:
    base = read(TEMPLATES_DIR / "base.html")
    match = re.search(r'<link rel="icon"[^>]*href="(data:image/svg\+xml,[^"]+)"', base)
    assert match, "base.html should carry an inline SVG favicon"
    assert "%3Csvg" in match.group(1)
    assert 'name="viewport"' in base


def test_base_template_carries_description_and_open_graph_tags(client: TestClient) -> None:
    """Pasted into LinkedIn, Slack or an email, the link unfurls with a title and a summary."""
    html = client.get("/").text
    assert '<meta name="description"' in html
    assert '<meta property="og:title" content="Keel">' in html
    assert '<meta property="og:description"' in html
    assert '<meta property="og:type" content="website">' in html
    assert '<meta name="twitter:card" content="summary">' in html


def test_chat_page_shows_favicon_footer_and_status(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    assert 'rel="icon" type="image/svg+xml" href="data:image/svg+xml,' in html
    footer = html[html.index("<footer") : html.index("</footer>")]
    assert "Keel" in footer
    assert "profile" in footer and ">local<" in footer
    assert "Answers stay on this appliance" in footer
    # the mode control keeps both radio values for the plain form post
    assert 'name="mode" value="answer"' in html and 'name="mode" value="agent"' in html
    assert 'class="seg"' in html
    # the form ends at the submit row: the old hint line (identity echo, latency note, shortcut ad)
    # was removed on purpose, so the page explains its widgets with the widgets themselves
    assert "Runs as" not in html and "<kbd>" not in html


def test_admin_page_footer_reports_profile_and_air_gap() -> None:
    html = _env.get_template("admin.html").render(**admin_context())
    footer = html[html.index("<footer") : html.index("</footer>")]
    assert "air-gap on" in footer and ">local<" in footer
    off = _env.get_template("admin.html").render(
        **admin_context(settings={**admin_context()["settings"], "airgap": False})
    )
    assert "air-gap off" in off[off.index("<footer") :]


# ---------------------------------------------------------------------- admin structure


def test_admin_page_tiles_facts_and_labelled_sparklines() -> None:
    html = _env.get_template("admin.html").render(**admin_context())
    assert html.count('class="tile"') == 10
    assert 'class="facts"' in html
    for fact in ("profile", "model", "listening on", "admin access", "air-gap", "refusal below"):
        assert f'<span class="k">{fact}</span>' in html
    assert html.count('class="spark-label"') == 5
    assert "14 days ago" in html and "today" in html
    assert "peak 3" in html  # requests per day, one day of data
    assert "Requests per day, last 14 days, latest 3, peak 3" in html
    assert "appears once an eval run attaches judge scores" in html
    assert 'class="table-wrap"' not in html  # nothing recent, so no tables render
    assert "The queue is clear." in html


def test_admin_tables_render_inside_a_scrolling_frame() -> None:
    recent = [
        {
            "request_id": "req-1",
            "ts": "2026-08-18T10:00:00Z",
            "mode": "answer",
            "user_id": "public",
            "question": "How many quotes?",
            "refused": False,
            "latency_ms": 900,
            "prompt_tokens": 500,
            "output_tokens": 20,
        }
    ]
    html = _env.get_template("admin.html").render(**admin_context(recent=recent))
    assert '<div class="table-wrap">' in html
    assert '<table class="grid">' in html


# ---------------------------------------------------------------------- result partial


def test_refusal_renders_calm_and_marked() -> None:
    html = _env.get_template("_result.html").render(r=result_context())
    assert 'class="result refused"' in html
    assert 'class="tag state-refused">refusal<' in html
    assert "Try another user or a question the corpus covers." in html
    assert "state-error" not in html


def test_agent_steps_render_as_cards_with_tool_and_state() -> None:
    steps = [
        {
            "tool": "calculator",
            "arguments": {"expression": "2*(3+4)"},
            "state": "ran",
            "outcome": "14",
            "reason": "",
        },
        {
            "tool": "create_ticket",
            "arguments": {"title": "Renew"},
            "state": "queued",
            "outcome": "queued for approval #1",
            "reason": "",
        },
    ]
    context = result_context(mode="agent", agent=SimpleNamespace(text="Done."), steps=steps)
    html = _env.get_template("_result.html").render(r=context)
    assert html.count('class="step step-') == 2
    assert 'class="tool">calculator<' in html and 'class="tag state-ran">ran<' in html
    assert 'class="tag state-queued">queued<' in html
    assert "decide on the admin page" in html


# ---------------------------------------------------------------------- stylesheet and script


def test_css_is_fluid_with_hover_and_focus_states() -> None:
    css = read(STATIC_DIR / "keel.css")
    for token in ("--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--accent", "--font", "--mono"):
        assert f"{token}:" in css
    assert re.search(r"\.table-wrap\s*\{[^}]*overflow-x:\s*auto", css)
    assert "@media (max-width: 760px)" in css and "@media (max-width: 480px)" in css
    assert "button:hover" in css and "button:focus-visible" in css
    assert "a.cite:hover" in css and ".chip:hover" in css
    assert "prefers-reduced-motion" in css
    # fluid layout: max-widths and grids carry the layout; a three-digit pixel width would pin a box
    fixed = re.findall(r"(?<![-\w])width:\s*\d{3,}px", css)
    assert fixed == [], fixed


def test_script_swaps_partials_and_shows_a_working_state() -> None:
    js = read(STATIC_DIR / "keel.js")
    assert '"X-Keel-Partial": "1"' in js
    assert 'aria-busy="true"' in js and 'class="working"' in js
    assert "requestSubmit" in js and "ctrlKey" in js
