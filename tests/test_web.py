"""Web app tests: health, chat page, JSON API, citation chips, refusal state, source viewer ACL,
admin page, approvals, quarantine release, ledger verify and export, and the admin token guard.

The AppContext is built with `build_context()` after `factory._local_providers` is swapped for a
FakeLLM plus the real (cached) fastembed models, then the fixture corpus is ingested into a temporary
data directory. No network, no llama-server.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from keel.answer.prompts import REFUSAL
from keel.answer.types import Citation, User
from keel.config import Settings
from keel.ingest import ingest_manifest
from keel.providers import factory
from keel.providers.factory import AppContext, build_context
from keel.providers.local_index import SqliteVectorIndex
from keel.safety.ledger import verify_file
from keel.web import views
from keel.web.app import ADMIN_TOKEN_ENV, ADMIN_TOKEN_HEADER, app, llm_healthy
from tests.fakes import FakeLLM, tool_call_reply

FIXTURE_MANIFEST = Path(__file__).resolve().parent.parent / "fixtures" / "corpus.yaml"
QUOTES_QUESTION = "How many quotes are needed for a purchase of $20,000?"
QUOTES_ANSWER = "A purchase of $20,000 needs three written quotes [1]."


# ---------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def web_ctx(tmp_path_factory: pytest.TempPathFactory, embedder, reranker) -> Iterator[AppContext]:
    """One AppContext for the module: FakeLLM, real local embeddings and reranker, fixture corpus
    ingested through the injection screen so the planted supplier note lands in quarantine."""
    data_dir = tmp_path_factory.mktemp("web")
    settings = Settings(data_dir=data_dir, airgap=False)

    def fake_local_providers(settings_: Settings, conn):
        return FakeLLM(), embedder, SqliteVectorIndex(conn, embed_model=embedder.name), reranker

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(factory, "_local_providers", fake_local_providers)
        ctx = build_context(settings)
    results = ingest_manifest(
        ctx.conn,
        ctx.settings,
        ctx.embedder,
        ctx.index,
        FIXTURE_MANIFEST,
        screen=ctx.screen,
        ledger=ctx.ledger,
    )
    assert len(results) == 5
    yield ctx
    ctx.close()


@pytest.fixture
def client(web_ctx: AppContext) -> Iterator[TestClient]:
    """A TestClient over the module app with the shared context installed and the LLM script cleared."""
    web_ctx.llm.responses.clear()
    app.state.ctx = web_ctx
    app.state.ctx_owned = False
    with TestClient(app) as test_client:
        yield test_client
    app.state.ctx = None


@pytest.fixture
def llm(web_ctx: AppContext) -> FakeLLM:
    return web_ctx.llm


def restricted_chunk_id(ctx: AppContext) -> int:
    row = ctx.conn.execute(
        "SELECT id FROM chunks WHERE EXISTS (SELECT 1 FROM json_each(acl_tags) WHERE value = 'hr') ORDER BY id"
    ).fetchone()
    assert row is not None
    return int(row[0])


def quarantined_chunk_id(ctx: AppContext) -> int:
    row = ctx.conn.execute("SELECT id FROM chunks WHERE quarantined = 1 ORDER BY id").fetchone()
    assert row is not None, "the fixture corpus should leave the planted injection in quarantine"
    return int(row[0])


# ---------------------------------------------------------------------- health and pages


def test_health_reports_status_and_store_counts(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["profile"] == "local"
    assert body["llm"] is True
    assert body["documents"] == 5
    assert body["chunks"] > 5
    assert body["quarantined"] >= 1
    assert body["ledger_seq"] >= 5


def test_head_is_served_on_the_chat_page_and_health(client: TestClient) -> None:
    """Link checkers and unfurl bots probe with HEAD and read a 405 as a dead page."""
    for path in ("/", "/health"):
        response = client.head(path)
        assert response.status_code == 200, path


def test_health_renders_as_a_page_for_a_browser_and_json_for_monitors(client: TestClient) -> None:
    """The nav link lands a person on a page; monitors and curl keep the JSON shape."""
    page = client.get("/health", headers={"Accept": "text/html"})
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "documents" in page.text and "ledger" in page.text
    body = client.get("/health").json()
    assert body["status"] == "ok"


def test_admin_refusal_renders_as_a_page_for_a_browser(
    client: TestClient, web_ctx: AppContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A visitor clicking Admin without the token reads a styled page naming the guard, and an API
    caller keeps the JSON error shape."""
    original = web_ctx.settings.host
    monkeypatch.delenv(ADMIN_TOKEN_ENV, raising=False)
    web_ctx.settings.host = "0.0.0.0"
    try:
        page = client.get("/admin", headers={"Accept": "text/html"})
        assert page.status_code == 401
        assert page.headers["content-type"].startswith("text/html")
        assert ADMIN_TOKEN_HEADER in page.text
        api = client.get("/admin", headers={"Accept": "application/json"})
        assert api.status_code == 401
        assert ADMIN_TOKEN_HEADER in api.json()["error"]
    finally:
        web_ctx.settings.host = original


def test_llm_probe_gives_up_after_the_timeout() -> None:
    class Slow:
        def healthy(self) -> bool:
            import time

            time.sleep(1.0)
            return True

    class Broken:
        def healthy(self) -> bool:
            raise ConnectionError("down")

    assert llm_healthy(Slow(), timeout=0.05) is False
    assert llm_healthy(Broken(), timeout=1.0) is False
    assert llm_healthy(FakeLLM(), timeout=1.0) is True


def test_chat_page_renders_the_question_box(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    assert 'name="question"' in html
    assert 'name="user_id"' in html and 'value="hr-officer"' in html
    assert 'name="tags"' in html
    assert 'value="agent"' in html and 'value="answer"' in html
    assert "/static/keel.css" in html and "/static/keel.js" in html


def test_static_assets_are_served(client: TestClient) -> None:
    assert client.get("/static/keel.css").status_code == 200
    assert client.get("/static/keel.js").status_code == 200


# ---------------------------------------------------------------------- ask


def test_api_ask_returns_json_with_citations(client: TestClient, llm: FakeLLM) -> None:
    llm.responses.append(QUOTES_ANSWER)
    response = client.post("/api/ask", json={"question": QUOTES_QUESTION, "user_id": "public", "tags": []})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "answer"
    assert body["refused"] is False
    assert body["text"] == QUOTES_ANSWER
    assert body["citations"] and body["citations"][0]["n"] == 1
    assert body["citations"][0]["title"] == "Northbank City Council Procurement Guide"
    assert body["retrieved"] and body["retrieved"][0]["chunk_id"] == body["citations"][0]["chunk_id"]
    assert body["user"] == {"user_id": "public", "tags": ["public"]}
    assert body["request_id"] and body["latency_ms"] >= 0
    assert llm.call_count == 1


def test_ask_form_post_renders_partial_with_citation_chips(client: TestClient, llm: FakeLLM) -> None:
    llm.responses.append(QUOTES_ANSWER)
    response = client.post(
        "/ask",
        data={"question": QUOTES_QUESTION, "user_id": "public", "mode": "answer"},
        headers={"X-Keel-Partial": "1"},
    )
    assert response.status_code == 200
    html = response.text
    assert "<html" not in html
    assert 'class="cite"' in html and ">[1]</a>" in html
    assert 'class="chip"' in html and "Northbank City Council Procurement Guide" in html
    assert "/source/" in html
    assert "prompt tokens" in html and "output tokens" in html
    assert 'class="result"' in html


def test_ask_plain_form_post_renders_the_full_page(client: TestClient, llm: FakeLLM) -> None:
    llm.responses.append(QUOTES_ANSWER)
    response = client.post("/ask", data={"question": QUOTES_QUESTION, "user_id": "public"})
    assert response.status_code == 200
    assert "<html" in response.text
    assert 'name="question"' in response.text
    assert 'class="cite"' in response.text


def test_refusal_state_is_marked(client: TestClient, llm: FakeLLM) -> None:
    llm.responses.append(REFUSAL)
    response = client.post(
        "/ask", data={"question": QUOTES_QUESTION, "user_id": "public"}, headers={"X-Keel-Partial": "1"}
    )
    assert response.status_code == 200
    assert 'class="result refused"' in response.text
    assert REFUSAL in response.text

    llm.responses.append(REFUSAL)
    body = client.post("/api/ask", json={"question": QUOTES_QUESTION}).json()
    assert body["refused"] is True and body["citations"] == []


def test_ask_rejects_an_empty_question_and_a_bad_mode(client: TestClient) -> None:
    assert client.post("/api/ask", json={"question": "  "}).status_code == 400
    assert client.post("/api/ask", json={"question": "x", "mode": "other"}).status_code == 400
    partial = client.post("/ask", data={"question": ""}, headers={"X-Keel-Partial": "1"})
    assert partial.status_code == 400 and 'class="result error"' in partial.text


def test_ask_reports_a_model_failure_as_502(client: TestClient, llm: FakeLLM) -> None:
    body = client.post("/api/ask", json={"question": QUOTES_QUESTION})
    assert body.status_code == 502
    assert "send the question again" in body.json()["error"]


def test_answer_html_escapes_text_and_turns_markers_into_chips() -> None:
    user = User("public", ["public"])
    citations = [Citation(1, 42, "docs/a.md", "Guide", 3, "Quotes", "snippet")]
    html = views.answer_html("<b>Three</b> quotes [1], see also [2].", citations, user)
    assert "&lt;b&gt;Three&lt;/b&gt;" in html
    assert 'href="/source/42?user=public&amp;tags=public"' in html
    assert "Guide · Quotes · p.3" in html
    assert "[2]" in html and html.count('class="cite"') == 1


# ---------------------------------------------------------------------- agent


def test_api_agent_runs_tools_and_queues_write_calls(
    client: TestClient, llm: FakeLLM, web_ctx: AppContext
) -> None:
    llm.responses.extend(
        [
            tool_call_reply("calculator", {"expression": "2*(3+4)"}, call_id="c1"),
            tool_call_reply(
                "create_ticket", {"title": "Renew contract", "body": "Band 2 renewal"}, call_id="c2"
            ),
            "The result is 14. The ticket awaits approval.",
        ]
    )
    response = client.post(
        "/api/agent", json={"question": "Compute 2*(3+4) and raise a ticket", "user_id": "public"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "agent"
    assert body["text"].startswith("The result is 14.")
    assert [s["tool"] for s in body["steps"]] == ["calculator", "create_ticket"]
    assert body["steps"][0]["result"] == "14"
    queued_id = body["steps"][1]["queued_id"]
    assert web_ctx.approvals.get(queued_id)["status"] == "pending"

    llm.responses.extend(
        [tool_call_reply("create_ticket", {"title": "Second", "body": "Also waits"}, call_id="c3"), "Queued."]
    )
    partial = client.post(
        "/ask", data={"question": "Raise another ticket", "mode": "agent"}, headers={"X-Keel-Partial": "1"}
    )
    assert partial.status_code == 200
    assert "queued for approval #" in partial.text
    assert 'class="tool">create_ticket' in partial.text


def test_admin_approve_executes_the_queued_ticket(
    client: TestClient, llm: FakeLLM, web_ctx: AppContext
) -> None:
    llm.responses.extend(
        [tool_call_reply("create_ticket", {"title": "Approve me", "body": "Please"}, call_id="c1"), "Queued."]
    )
    body = client.post("/api/agent", json={"question": "Raise a ticket", "user_id": "public"}).json()
    approval_id = body["steps"][0]["queued_id"]
    assert web_ctx.approvals.get(approval_id)["status"] == "pending"

    response = client.post(f"/admin/approvals/{approval_id}/approve", data={"by": "blake"})
    assert response.status_code == 200  # redirected to /admin
    assert response.url.path == "/admin"
    row = web_ctx.approvals.get(approval_id)
    assert row["status"] == "executed"
    assert row["decided_by"] == "blake"
    assert row["result"].startswith("ticket created: Approve me")

    again = client.post(f"/admin/approvals/{approval_id}/approve", headers={"accept": "application/json"})
    assert again.status_code == 409


def test_admin_reject_marks_the_row_and_runs_nothing(
    client: TestClient, llm: FakeLLM, web_ctx: AppContext
) -> None:
    llm.responses.extend(
        [tool_call_reply("create_ticket", {"title": "Reject me", "body": "Please"}, call_id="c1"), "Queued."]
    )
    body = client.post("/api/agent", json={"question": "Raise a ticket"}).json()
    approval_id = body["steps"][0]["queued_id"]
    response = client.post(f"/admin/approvals/{approval_id}/reject", headers={"accept": "application/json"})
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    assert web_ctx.approvals.get(approval_id)["result"] is None
    assert client.post("/admin/approvals/999999/reject").status_code == 404


# ---------------------------------------------------------------------- source viewer and ACL


def test_source_enforces_acl_tags(client: TestClient, web_ctx: AppContext) -> None:
    chunk_id = restricted_chunk_id(web_ctx)
    assert client.get(f"/source/{chunk_id}", params={"tags": "public"}).status_code == 403
    assert client.get(f"/source/{chunk_id}").status_code == 403
    allowed = client.get(f"/source/{chunk_id}", params={"tags": "hr"})
    assert allowed.status_code == 200
    assert "Northbank Salary Bands" in allowed.text
    as_user = client.get(f"/source/{chunk_id}", params={"user": "hr-officer"})
    assert as_user.status_code == 200
    as_json = client.get(f"/source/{chunk_id}", params={"tags": "hr"}, headers={"accept": "application/json"})
    assert as_json.json()["acl_tags"] == ["hr"]


def test_source_unknown_chunk_is_404(client: TestClient) -> None:
    assert client.get("/source/999999", params={"tags": "public"}).status_code == 404


def test_public_chunk_shows_text_and_tags(client: TestClient, web_ctx: AppContext) -> None:
    row = web_ctx.conn.execute(
        "SELECT id FROM chunks WHERE acl_tags = '[\"public\"]' AND quarantined = 0 ORDER BY id"
    ).fetchone()
    response = client.get(f"/source/{row[0]}", params={"user": "public"})
    assert response.status_code == 200
    assert 'class="tag acl">public' in response.text
    assert "<pre>" in response.text


# ---------------------------------------------------------------------- admin page


def test_admin_page_lists_requests_totals_trend_and_sections(client: TestClient, llm: FakeLLM) -> None:
    llm.responses.append(QUOTES_ANSWER)
    client.post("/api/ask", json={"question": QUOTES_QUESTION, "user_id": "hr-officer"})
    response = client.get("/admin")
    assert response.status_code == 200
    html = response.text
    assert QUOTES_QUESTION[:40] in html
    assert 'id="totals"' in html and 'id="trend"' in html and 'id="recent"' in html
    assert 'id="approvals"' in html and 'id="quarantine"' in html and 'id="ledger"' in html
    assert "<polyline" in html
    assert "Requests per day" in html and "Refusal rate" in html
    assert "/admin/ledger/export" in html
    assert "/admin/quarantine/" in html


def test_admin_request_detail_shows_retrieval_citations_and_ledger(client: TestClient, llm: FakeLLM) -> None:
    llm.responses.append(QUOTES_ANSWER)
    body = client.post("/api/ask", json={"question": QUOTES_QUESTION}).json()
    response = client.get(f"/admin/request/{body['request_id']}")
    assert response.status_code == 200
    html = response.text
    assert QUOTES_QUESTION in html
    assert f"chunk {body['citations'][0]['chunk_id']}" in html
    assert "Northbank City Council Procurement Guide" in html
    assert ">retrieval<" in html and ">answer<" in html
    assert client.get("/admin/request/nope").status_code == 404
    as_json = client.get(
        f"/admin/request/{body['request_id']}", headers={"accept": "application/json"}
    ).json()
    assert as_json["request_id"] == body["request_id"]
    assert [row["kind"] for row in as_json["ledger"]] == ["request", "retrieval", "answer"]


def test_ledger_verify_shows_ok(client: TestClient) -> None:
    response = client.post("/admin/ledger/verify")
    assert response.status_code == 200
    assert "chain intact" in response.text
    assert 'class="tag state-ok">ok<' in response.text
    as_json = client.post("/admin/ledger/verify", headers={"accept": "application/json"}).json()
    assert as_json["ok"] is True and as_json["checked"] > 0


def test_ledger_export_is_verifiable_jsonl(client: TestClient, tmp_path: Path, web_ctx: AppContext) -> None:
    response = client.get("/admin/ledger/export")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert "keel-ledger.jsonl" in response.headers["content-disposition"]
    lines = [line for line in response.text.splitlines() if line.strip()]
    assert len(lines) == web_ctx.ledger.count()
    first = json.loads(lines[0])
    assert set(first) == {"seq", "ts", "kind", "request_id", "prev_hash", "hash", "payload"}
    path = tmp_path / "ledger.jsonl"
    path.write_text(response.text, encoding="utf-8")
    assert verify_file(path).ok


def test_quarantine_release_flips_the_flag_and_appends_a_ledger_row(
    client: TestClient, web_ctx: AppContext
) -> None:
    chunk_id = quarantined_chunk_id(web_ctx)
    before = web_ctx.ledger.count()
    admin_html = client.get("/admin").text
    assert f"/admin/quarantine/{chunk_id}/release" in admin_html

    response = client.post(f"/admin/quarantine/{chunk_id}/release", data={"by": "blake"})
    assert response.status_code == 200 and response.url.path == "/admin"
    flag = web_ctx.conn.execute("SELECT quarantined FROM chunks WHERE id = ?", (chunk_id,)).fetchone()[0]
    assert flag == 0
    entries = list(web_ctx.ledger.entries())
    assert len(entries) == before + 1
    last = entries[-1]
    assert last.kind == "quarantine"
    assert last.payload == {"chunk_id": chunk_id, "action": "release", "by": "blake"}
    assert web_ctx.ledger.verify().ok
    assert f"/admin/quarantine/{chunk_id}/release" not in client.get("/admin").text

    again = client.post(f"/admin/quarantine/{chunk_id}/release", headers={"accept": "application/json"})
    assert again.status_code == 409
    assert client.post("/admin/quarantine/999999/release").status_code == 404
    # put it back so other tests keep a quarantined chunk to look at
    web_ctx.conn.execute("UPDATE chunks SET quarantined = 1 WHERE id = ?", (chunk_id,))


# ---------------------------------------------------------------------- admin guard


def test_admin_guard_requires_token_beyond_loopback(
    client: TestClient, web_ctx: AppContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = web_ctx.settings.host
    monkeypatch.delenv(ADMIN_TOKEN_ENV, raising=False)
    web_ctx.settings.host = "0.0.0.0"
    try:
        assert client.get("/admin").status_code == 401
        assert client.post("/admin/ledger/verify").status_code == 401
        assert client.get("/admin/ledger/export").status_code == 401
        assert client.get("/admin", headers={ADMIN_TOKEN_HEADER: "anything"}).status_code == 401
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "s3cret")
        assert client.get("/admin", headers={ADMIN_TOKEN_HEADER: "wrong"}).status_code == 401
        assert client.get("/admin", headers={ADMIN_TOKEN_HEADER: "s3cret"}).status_code == 200
        # chat, health and source stay open
        assert client.get("/health").status_code == 200
        assert client.get("/").status_code == 200
    finally:
        web_ctx.settings.host = original
    assert client.get("/admin").status_code == 200


@pytest.mark.parametrize(
    ("host", "loopback"),
    [
        ("127.0.0.1", True),
        ("localhost", True),
        ("::1", True),
        ("127.0.0.5", True),
        ("0.0.0.0", False),
        ("10.0.0.4", False),
        ("", False),
    ],
)
def test_is_loopback(host: str, loopback: bool) -> None:
    assert views.is_loopback(host) is loopback


# ---------------------------------------------------------------------- pure helpers


def test_resolve_user_merges_demo_tags_and_extras() -> None:
    assert views.resolve_user(None, None) == User("public", ["public"])
    assert views.resolve_user("hr-officer", "").tags == ["public", "hr"]
    assert views.resolve_user("public", "finance, legal finance").tags == ["public", "finance", "legal"]
    assert views.resolve_user("guest", ["ops"]).tags == ["ops"]
    assert views.resolve_user("guest", []).tags == ["public"]


def test_trend_sparklines_cover_fourteen_days_and_skip_missing_values() -> None:
    from datetime import date

    today = date(2026, 8, 18)
    daily = [
        {
            "day": "2026-08-17",
            "requests": 4,
            "refused": 1,
            "avg_latency_ms": 1200.0,
            "groundedness": None,
            "relevance": None,
        },
        {
            "day": "2026-08-18",
            "requests": 2,
            "refused": 0,
            "avg_latency_ms": 800.0,
            "groundedness": 0.9,
            "relevance": 0.8,
        },
    ]
    sparks = {s.key: s for s in views.trend_sparklines(daily, 14, today)}
    assert len(sparks["requests"].values) == 14
    assert sparks["requests"].values[-2:] == [4.0, 2.0] and sparks["requests"].values[0] == 0.0
    assert sparks["refusal_rate"].values[-2:] == [25.0, 0.0] and sparks["refusal_rate"].values[0] is None
    assert sparks["latency"].last == 800.0 and sparks["latency"].last_text == "800ms"
    assert sparks["refusal_rate"].last_text == "0%" and sparks["relevance"].last_text == "0.8"
    assert sparks["groundedness"].present and sparks["groundedness"].values[-1] == 0.9
    assert views.trend_sparklines([], 14, today)[3].last_text == "·"
    assert views.format_value(1234.0) == "1,234" and views.format_value(12.5) == "12.5"
    assert sparks["requests"].points.count(",") == 14
    assert views.sparkline_geometry([None, None]) == ("", [])
    points, dots = views.sparkline_geometry([1.0])
    assert len(dots) == 1 and points


# ---------------------------------------------------------------------- hosted demo posture


def _hr_chunk_ids(ctx: AppContext) -> set[int]:
    rows = ctx.conn.execute("SELECT id FROM chunks WHERE acl_tags = '[\"hr\"]'").fetchall()
    return {int(r[0]) for r in rows}


SECRET_QUESTION = "What is the confidential review code for the 2026 pay round?"


@pytest.mark.parametrize("demo_identity", [True, False], ids=["demo-identity-on", "demo-identity-off"])
def test_demo_identity_beyond_loopback(
    client: TestClient,
    web_ctx: AppContext,
    llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
    demo_identity: bool,
) -> None:
    """With KEEL_DEMO_IDENTITY=1 the demo user picker is honoured beyond loopback: hr-officer retrieves the
    hr chunk. With it off the same request runs as public. Admin needs the token either way."""
    restricted = _hr_chunk_ids(web_ctx)
    assert restricted
    monkeypatch.delenv(ADMIN_TOKEN_ENV, raising=False)
    original_host, original_flag = web_ctx.settings.host, web_ctx.settings.demo_identity
    web_ctx.settings.host = "0.0.0.0"
    web_ctx.settings.demo_identity = demo_identity
    try:
        llm.responses.append("The code is PELICAN-7741 [1].")
        response = client.post(
            "/api/ask", json={"question": SECRET_QUESTION, "user_id": "hr-officer", "tags": ["hr"]}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        retrieved = {r["chunk_id"] for r in body["retrieved"]}
        if demo_identity:
            assert body["user"] == {"user_id": "hr-officer", "tags": ["public", "hr"]}
            assert retrieved & restricted, body["retrieved"]
        else:
            assert body["user"] == {"user_id": "public", "tags": ["public"]}
            assert not (retrieved & restricted), body["retrieved"]
        assert client.get("/admin").status_code == 401
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "s3cret")
        assert client.get("/admin", headers={ADMIN_TOKEN_HEADER: "s3cret"}).status_code == 200
        page = client.get("/")
        assert page.status_code == 200
        assert ('id="demo-banner"' in page.text) is demo_identity
        assert ('id="demo-compare"' in page.text) is demo_identity
        assert ('id="intro"' in page.text) is demo_identity
        if demo_identity:
            assert SECRET_QUESTION in page.text, "the compare button should carry the restricted question"
            assert "How it runs internal operations" in page.text, "the intro dialog carries the four blocks"
    finally:
        web_ctx.settings.host = original_host
        web_ctx.settings.demo_identity = original_flag


def test_demo_identity_ignores_extra_tags_and_unknown_users(
    client: TestClient, web_ctx: AppContext, llm: FakeLLM
) -> None:
    """Under the demo flag only the picker's users carry tags: an unknown id with `hr` in the body is public."""
    restricted = _hr_chunk_ids(web_ctx)
    original_host, original_flag = web_ctx.settings.host, web_ctx.settings.demo_identity
    web_ctx.settings.host = "0.0.0.0"
    web_ctx.settings.demo_identity = True
    try:
        llm.responses.append("Answer [1].")
        response = client.post(
            "/api/ask", json={"question": SECRET_QUESTION, "user_id": "nobody", "tags": ["hr"]}
        )
        body = response.json()
        assert body["user"] == {"user_id": "public", "tags": ["public"]}
        assert not ({r["chunk_id"] for r in body["retrieved"]} & restricted)
        llm.responses.append("Answer [1].")
        response = client.post(
            "/api/ask", json={"question": SECRET_QUESTION, "user_id": "public", "tags": ["hr"]}
        )
        assert response.json()["user"]["tags"] == ["public"]
    finally:
        web_ctx.settings.host = original_host
        web_ctx.settings.demo_identity = original_flag


def test_web_app_exposes_no_ingest_route_and_every_write_sits_under_admin() -> None:
    """The read-only demo posture: nothing on the web ingests, and approve, reject and quarantine
    release are admin routes (token beyond loopback)."""
    routes = []
    for route in app.routes:
        included = getattr(route, "original_router", None)
        routes.extend(included.routes if included is not None else [route])
    paths = {getattr(route, "path", "") for route in routes}
    assert paths >= {"/", "/ask", "/admin"}
    assert not any("ingest" in path for path in paths)
    writers = [
        path
        for route in routes
        for path in [getattr(route, "path", "")]
        if "POST" in (getattr(route, "methods", None) or set())
    ]
    assert set(writers) == {
        "/ask",
        "/api/ask",
        "/api/agent",
        "/admin/approvals/{approval_id}/approve",
        "/admin/approvals/{approval_id}/reject",
        "/admin/quarantine/{chunk_id}/release",
        "/admin/ledger/verify",
    }
