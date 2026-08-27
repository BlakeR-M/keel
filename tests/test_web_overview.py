"""The overview page, the documentation pages and the two live control demonstrations.

The overview page is the front door: a reader arriving cold from a link lands there rather than in
the question box. These tests hold that shape in place. They cover what the page must say, that the
documentation renders from the repository's own Markdown with its links rewritten, that a request
path can name nothing outside `docs/`, and that the air-gap probe and the redaction demonstration
report what the underlying controls actually did.

No network and no model: the probe refuses every connection by design, and redaction is a pure
function.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from keel.airgap import airgapped
from keel.web import airgap_probe, docs
from keel.web.airgap_probe import LAYERS, clean_host, probe
from keel.web.app import COMPARE_QUESTION, REDACT_MAX_CHARS, app

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTSIDE_HOST = "data.attacker.example"


def _empty_store() -> sqlite3.Connection:
    """An in-memory store with the one table the question box reads to report the corpus."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, source TEXT, title TEXT)")
    return conn


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A TestClient over a stub context, so nothing here loads a model or opens a store.

    These routes read templates, Markdown and a few settings. `host` is loopback so the demo user
    picker is honoured and the overview renders the comparison, which is what it does for anyone
    running the appliance on their own machine. `front_page` is pinned to the overview so `/` serves
    the page these tests are about; what `/` serves by default is covered separately below.
    """
    app.state.ctx = SimpleNamespace(
        profile="local",
        settings=SimpleNamespace(
            airgap=True, host="127.0.0.1", demo_identity=False, front_page="overview"
        ),
        conn=_empty_store(),
    )
    app.state.ctx_owned = False
    with TestClient(app) as test_client:
        yield test_client
    app.state.ctx = None


@pytest.fixture(autouse=True)
def _clear_probe_limit() -> Iterator[None]:
    """The probe is rate limited per caller. Every test in this file starts from a clean slate so the
    order tests run in cannot decide whether one of them sees a 429."""
    from keel.web import app as web_app

    web_app._probe_last.clear()
    yield
    web_app._probe_last.clear()


# ---------------------------------------------------------------------- the overview page


def test_overview_leads_with_the_security_posture_rather_than_the_question_box(
    client: TestClient,
) -> None:
    """A stranger following a link reads what Keel is and what it refuses, before any form.

    The heading states what Keel is rather than instructing the reader to do something. This page is
    a reference for someone deciding whether to look further, so the register stays descriptive.
    """
    html = client.get("/").text
    assert "Keel is a retrieval and agent appliance" in html
    for claim in (
        "sovereign",
        "air-gap",
        "hash-chained ledger",
        "managed identity",
        "105",
        "27",
    ):
        assert claim in html.lower() or claim in html, claim
    assert '<form id="ask-form"' not in html, "the question box belongs to /chat"
    assert "/docs/security-review" in html
    assert "https://github.com/BlakeR-M/keel" in html


def test_overview_carries_all_three_live_demonstrations(client: TestClient) -> None:
    html = client.get("/").text
    assert 'id="permissions"' in html and 'id="demo-compare"' in html
    assert 'id="airgap"' in html and 'id="airgap-form"' in html
    assert 'id="redaction"' in html and 'id="redact-form"' in html
    assert COMPARE_QUESTION in html, "the comparison button carries the restricted question"


def test_overview_lists_the_documentation_it_can_render(client: TestClient) -> None:
    html = client.get("/").text
    for slug in ("tutorial", "architecture", "security-review", "threat-model"):
        assert f"/docs/{slug}" in html, slug


def test_navigation_reaches_every_public_page(client: TestClient) -> None:
    html = client.get("/").text
    for href in ('href="/"', 'href="/chat"', 'href="/docs"', 'href="/health"'):
        assert href in html, href


def test_the_public_pages_never_link_to_the_operator_surface() -> None:
    """The admin routes need a token beyond loopback, so an unconditional link to them is a wall a
    reader walks into.

    Two templates are allowed to link there and both do it conditionally, each covered by its own
    tests below: the question box offers the upload where uploading is possible at all, and the
    header carries an Admin item where the console answers without a token.
    """
    templates = Path(__file__).resolve().parent.parent / "keel" / "web" / "templates"
    # The operator pages link among themselves, which is what they are for. `chat.html` and
    # `base.html` are the visitor-facing pair allowed to, and only behind a condition.
    allowed = {"admin.html", "admin_request.html", "connection.html", "chat.html", "base.html"}
    offenders = [
        path.name
        for path in sorted(templates.glob("*.html"))
        if path.name not in allowed and '"/admin' in path.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"these visitor-facing templates link to /admin: {offenders}"


def test_the_question_box_offers_the_upload_only_where_uploading_is_possible() -> None:
    """On the hosted demo there is no ingest route, so pointing a reader at one would be a dead end.
    The same panel then names the command line instead."""
    with _client_with(demo_identity=False) as client:
        html = client.get("/chat").text
    from keel.web.app import web_ingest_enabled

    if web_ingest_enabled():
        assert 'href="/admin#documents"' in html
    else:
        # The header may still carry its own Admin item on loopback, so this asks about the panel's
        # link rather than about the page holding the string anywhere.
        assert 'href="/admin#documents"' not in html
        assert "keel ingest" in html


def test_the_question_box_names_the_next_steps_for_somebody_who_just_installed_it() -> None:
    """Whoever cloned Keel arrives wanting to point it at their own work. The panel answers that on
    the page they land on rather than leaving it to the documentation."""
    with _client_with(demo_identity=False) as client:
        html = client.get("/chat").text
    assert 'id="getting-started"' in html
    for step in ("Add your documents", "Connect a model", "your own Azure", "who reads what"):
        assert step in html, step
    assert "keel setup --profile azure" in html
    assert "/docs/setup" in html


def test_the_hosted_demo_leaves_the_getting_started_panel_out() -> None:
    """A visitor to the demo is reading, rather than setting anything up."""
    with _client_with(demo_identity=True) as client:
        html = client.get("/chat").text
    assert 'id="getting-started"' not in html


def test_the_identity_controls_appear_only_where_they_are_honoured() -> None:
    """Beyond loopback and away from the demo, identity comes from the reverse proxy and the form is
    ignored. Offering a user select there would be offering a control that does nothing."""
    with _client_with(demo_identity=False, host="0.0.0.0") as client:
        proxied = client.get("/chat").text
    with _client_with(demo_identity=False, host="127.0.0.1") as client:
        local = client.get("/chat").text
    assert 'name="user_id"' not in proxied
    assert "come from the reverse proxy" in proxied
    assert 'name="user_id"' in local


# ---------------------------------------------------------------------- documentation pages


def test_docs_index_lists_every_document_in_reading_order(client: TestClient) -> None:
    response = client.get("/docs")
    assert response.status_code == 200
    html = response.text
    slugs = docs.slugs()
    assert "tutorial" in slugs and "security-review" in slugs
    positions = [html.index(f'/docs/{slug}"') for slug in slugs]
    assert positions == sorted(positions), "the index should follow the reading order"


def test_a_document_renders_its_markdown_with_links_rewritten(client: TestClient) -> None:
    response = client.get("/docs/security-review")
    assert response.status_code == 200
    html = response.text
    assert "<table>" in html and html.count("<tr>") > 20, "the findings table renders as a table"
    # a path reaching out of docs/ becomes a link into the repository
    assert f"{docs.GITHUB_BLOB}/tests/redteam_acl_and_retrieval_leakage.py" in html
    # and the page names the file it came from
    assert "docs/security-review.md" in html


def test_sibling_document_links_stay_on_the_site(client: TestClient) -> None:
    html = client.get("/docs/tutorial").text
    assert 'href="/docs/architecture"' in html
    assert 'href="/docs/security-review"' in html
    assert f'href="{docs.GITHUB_BLOB}/fixtures/corpus.yaml"' in html


def test_documents_carry_previous_and_next_links(client: TestClient) -> None:
    ordered = docs.slugs()
    middle = ordered[1]
    html = client.get(f"/docs/{middle}").text
    assert f'/docs/{ordered[0]}"' in html and f'/docs/{ordered[2]}"' in html


def test_the_two_open_findings_stay_visible_on_the_published_review(client: TestClient) -> None:
    """The review publishes its unfixed findings. A page that quietly dropped them would be worse
    than no page, so the count and the reason both have to survive rendering."""
    html = client.get("/docs/security-review").text
    assert "Twenty-five are fixed" in html
    assert "xfail" in html
    assert "keeps its xfail" in html
    overview = client.get("/").text
    assert "open by choice" in overview


@pytest.mark.parametrize(
    "slug",
    ["nothing-here", "..", "../README", "../../etc/passwd", "..%2f..%2fREADME", "Security-Review", ""],
)
def test_a_slug_can_name_nothing_outside_the_docs_directory(client: TestClient, slug: str) -> None:
    """A hostile slug reaches the index or a 404, and never renders a file.

    Some of these normalise before routing sees them: `/docs/..` resolves to the overview and
    `/docs/` to the index. Those are fine places to land. What matters is that no request path
    renders a document, so the assertion is about the body rather than only the status.
    """
    response = client.get(f"/docs/{slug}")
    assert response.status_code in (200, 307, 404), f"{slug} returned {response.status_code}"
    if response.status_code == 200:
        assert '<article class="prose">' not in response.text, f"{slug} rendered a document"


def test_docs_module_refuses_a_slug_that_is_not_a_plain_word() -> None:
    for slug in ("../README", "..", "docs/architecture", "arch itecture", "Architecture", "a/b"):
        assert docs.page(slug) is None, slug


def test_rewrite_links_leaves_absolute_and_fragment_links_alone() -> None:
    html = (
        '<a href="https://example.com/x">a</a>'
        '<a href="#section">b</a>'
        '<a href="mailto:someone@example.com">c</a>'
        '<a href="/chat">d</a>'
    )
    assert docs.rewrite_links(html) == html


def test_rewrite_links_maps_siblings_here_and_everything_else_to_the_repository() -> None:
    rewritten = docs.rewrite_links(
        '<a href="architecture.md">a</a>'
        '<a href="security-review.md#findings">b</a>'
        '<a href="../keel/airgap.py">c</a>'
        '<a href="../tests/test_airgap.py">d</a>'
    )
    assert 'href="/docs/architecture"' in rewritten
    assert 'href="/docs/security-review#findings"' in rewritten
    assert f'href="{docs.GITHUB_BLOB}/keel/airgap.py"' in rewritten
    assert f'href="{docs.GITHUB_BLOB}/tests/test_airgap.py"' in rewritten


# ---------------------------------------------------------------------- the air-gap probe


@pytest.mark.parametrize(
    "raw", ["data.attacker.example", "8.8.8.8", "localhost", "  Example.COM  ", "::1", "a-b.example."]
)
def test_clean_host_accepts_a_bare_host(raw: str) -> None:
    assert clean_host(raw) is not None


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "http://evil.example/x",
        "evil.example/path",
        "evil.example:443",
        "user@evil.example",
        "two words",
        "line\nbreak",
        "a" * 300,
        "-leading.example",
    ],
)
def test_clean_host_refuses_anything_that_is_not_a_bare_host(raw: str) -> None:
    assert clean_host(raw) is None


def test_every_guarded_layer_refuses_a_host_outside_the_allow_list() -> None:
    """The claim the overview page makes, checked against the guard itself rather than the copy."""
    with airgapped():
        report = probe(OUTSIDE_HOST)
    assert report["guard"] is True
    assert report["allowed"] is False
    assert [attempt["layer"] for attempt in report["attempts"]] == [name for name, _ in LAYERS]
    for attempt in report["attempts"]:
        assert attempt["refused"] is True, attempt
        assert attempt["via"] == attempt["layer"], attempt
        assert "Air-gap mode refused" in attempt["detail"]
    assert report["refused"] == len(LAYERS)


def test_an_address_reports_that_no_lookup_was_needed_rather_than_a_layer_letting_it_past() -> None:
    """An IP literal never reaches a resolver, so counting name resolution as a layer that failed to
    refuse would misreport the control. The connect guards still refuse it."""
    with airgapped():
        report = probe("8.8.8.8")
    by_layer = {attempt["layer"]: attempt for attempt in report["attempts"]}
    assert by_layer["dns"]["outcome"] == "no-lookup"
    assert by_layer["dns"]["refused"] is False
    for layer in ("socket", "asyncio", "urllib", "httpx"):
        assert by_layer[layer]["refused"] is True, layer
    assert "nothing to look up" in report["summary"]


def test_an_allow_listed_host_is_answered_from_the_policy_without_a_connection() -> None:
    """Loopback is on the allow list, so the guard would permit it. The probe reports that from the
    policy and connects to nothing, which keeps this route from becoming a way to reach loopback."""
    with airgapped():
        report = probe("localhost")
    assert report["allowed"] is True
    assert report["attempts"] == []
    assert "allow list" in report["summary"]


def test_the_probe_runs_in_a_child_process_under_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end through the subprocess, which is how the route runs it. The parent's own guard state
    is untouched: this process never had air-gap on and still does not."""
    from keel import airgap

    assert airgap.is_enabled() is False
    report = airgap_probe.run(OUTSIDE_HOST)
    assert airgap.is_enabled() is False
    assert report["guard"] is True
    assert report["refused"] == len(LAYERS)


def test_airgap_probe_route_reports_every_refusal(client: TestClient) -> None:
    response = client.post("/api/airgap-probe", json={"host": OUTSIDE_HOST})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["host"] == OUTSIDE_HOST
    assert body["refused"] == len(LAYERS)
    assert all(attempt["refused"] for attempt in body["attempts"])


def test_airgap_probe_route_refuses_anything_that_is_not_a_host(client: TestClient) -> None:
    response = client.post("/api/airgap-probe", json={"host": "http://evil.example/steal"})
    assert response.status_code == 400
    assert "error" in response.json()


def test_airgap_probe_route_runs_a_bounded_number_of_children_at_once() -> None:
    """A child process costs memory while it lives and the app already holds two embedding models,
    so the route hands back a 429 rather than letting a burst of visitors crowd the box."""
    from keel.web import app as web_app

    assert web_app.PROBE_MAX_CONCURRENT >= 1
    held = [web_app._probe_slots.acquire(blocking=False) for _ in range(web_app.PROBE_MAX_CONCURRENT)]
    try:
        assert all(held)
        assert web_app._probe_slots.acquire(blocking=False) is False
    finally:
        for _ in held:
            web_app._probe_slots.release()


def test_airgap_probe_route_is_rate_limited_per_caller(client: TestClient) -> None:
    first = client.post("/api/airgap-probe", json={"host": OUTSIDE_HOST})
    second = client.post("/api/airgap-probe", json={"host": OUTSIDE_HOST})
    assert first.status_code == 200
    assert second.status_code == 429
    assert "error" in second.json()


# ---------------------------------------------------------------------- redaction


def test_redact_route_replaces_australian_identifiers_and_leaves_ordinary_numbers(
    client: TestClient,
) -> None:
    text = (
        "Reach Jordan at jordan.ellery@example.com or 0412 345 678. "
        "Tax file number 123 456 782, Medicare 2123 45670 1, ABN 51 824 753 556. "
        "Purchase order 987654321 and invoice 4417 stay as written."
    )
    body = client.post("/api/redact", json={"text": text}).json()
    for kind in ("email", "phone_au", "tfn", "medicare", "abn"):
        assert body["counts"].get(kind) == 1, kind
    assert "[REDACTED:tfn]" in body["redacted"] and "[REDACTED:abn]" in body["redacted"]
    assert "987654321" in body["redacted"], "a purchase order is not an identifier"
    assert "4417" in body["redacted"], "an invoice number is not an identifier"
    assert "jordan.ellery@example.com" not in body["redacted"]


def test_redact_route_reports_the_span_of_every_finding(client: TestClient) -> None:
    text = "ABN 51 824 753 556 belongs to nobody."
    body = client.post("/api/redact", json={"text": text}).json()
    assert body["findings"], body
    start, end = body["findings"][0]["start"], body["findings"][0]["end"]
    assert text[start:end] == "51 824 753 556"


def test_redact_route_caps_the_text_it_accepts(client: TestClient) -> None:
    response = client.post("/api/redact", json={"text": "x" * (REDACT_MAX_CHARS + 1)})
    assert response.status_code == 400
    assert "error" in response.json()


def test_redact_route_returns_text_unchanged_when_nothing_matches(client: TestClient) -> None:
    body = client.post("/api/redact", json={"text": "Nothing identifying here at all."}).json()
    assert body["redacted"] == "Nothing identifying here at all."
    assert body["findings"] == []
    assert body["counts"] == {}


# ---------------------------------------------------------------------- packaging


def test_the_image_carries_the_documentation_the_site_renders() -> None:
    """The docs routes read Markdown from the repository, so `docs/` has to ship with the image.
    Without this the pages 404 in a container and nowhere else, which is the worst place to find out."""
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY docs ./docs" in dockerfile
    ignored = {
        line.strip()
        for line in (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert "docs" not in ignored


def test_every_document_named_in_the_reading_order_exists() -> None:
    """A slug in the reading order with no file behind it would leave a dead card on the index."""
    missing = [slug for slug in docs.READING_ORDER if not (docs.DOCS_DIR / f"{slug}.md").is_file()]
    assert missing == [], missing


# ---------------------------------------------------------------------- what `/` serves


def _client_with(**settings: object) -> TestClient:
    """A TestClient whose deployment settings are the ones under test."""
    base = {"airgap": False, "host": "127.0.0.1", "demo_identity": False, "front_page": "auto"}
    base.update(settings)
    app.state.ctx = SimpleNamespace(
        profile="local", settings=SimpleNamespace(**base), conn=_empty_store()
    )
    app.state.ctx_owned = False
    return TestClient(app)


def test_the_hosted_demo_leads_with_the_overview() -> None:
    """A visitor arriving from a link has to be told what this is before a form makes any sense."""
    with _client_with(demo_identity=True) as client:
        html = client.get("/").text
    assert "Keel is a retrieval and agent appliance" in html
    assert '<form id="ask-form"' not in html


def test_an_installed_appliance_leads_with_the_question_box() -> None:
    """Whoever cloned and ran Keel came to use it. Landing them on the pitch reads as a missing app,
    which is the confusion this setting exists to remove."""
    with _client_with(demo_identity=False) as client:
        html = client.get("/").text
    assert '<form id="ask-form"' in html
    assert 'name="question"' in html


@pytest.mark.parametrize(
    ("front_page", "expects_form"), [("overview", False), ("chat", True)]
)
def test_the_setting_overrides_the_deployment(front_page: str, expects_form: bool) -> None:
    with _client_with(demo_identity=True, front_page=front_page) as client:
        html = client.get("/").text
    assert ('<form id="ask-form"' in html) is expects_form


@pytest.mark.parametrize("demo_identity", [True, False])
def test_the_overview_is_always_reachable_at_about(demo_identity: bool) -> None:
    """Wherever `/` points, the explanation keeps a permanent home."""
    with _client_with(demo_identity=demo_identity) as client:
        response = client.get("/about")
    assert response.status_code == 200
    assert "Keel is a retrieval and agent appliance" in response.text


@pytest.mark.parametrize("demo_identity", [True, False])
def test_the_question_box_is_always_reachable_at_chat(demo_identity: bool) -> None:
    with _client_with(demo_identity=demo_identity) as client:
        response = client.get("/chat")
    assert response.status_code == 200
    assert '<form id="ask-form"' in response.text


def test_the_navigation_names_the_page_the_front_door_serves() -> None:
    """The first nav entry is whatever `/` is, so it never sends a reader to a page they are on."""
    with _client_with(demo_identity=True) as client:
        demo_nav = client.get("/").text
    with _client_with(demo_identity=False) as client:
        installed_nav = client.get("/").text
    assert ">Overview<" in demo_nav and ">Demo<" in demo_nav
    assert ">About<" not in demo_nav, "the overview is the front door here, so /about stays quiet"
    assert ">Ask<" in installed_nav and ">About<" in installed_nav


def test_an_empty_store_says_so_rather_than_refusing_without_explanation() -> None:
    """A fresh install refuses every question for want of sources. Saying why beats looking broken."""
    with _client_with(demo_identity=False) as client:
        html = client.get("/chat").text
    assert "holds no documents yet" in html
    assert "keel ingest --manifest fixtures/corpus.yaml" in html
    assert "/docs/setup" in html


def test_the_getting_started_panel_is_open_on_a_first_visit() -> None:
    """Whoever just installed Keel should read the next steps without hunting for them. It stays
    open in the markup, so a first visit and a visit with scripting off both show it, and the script
    folds it away for good once they collapse it once."""
    with _client_with(demo_identity=False) as client:
        html = client.get("/chat").text
    panel = html[html.index('id="getting-started"') : html.index("</summary>")]
    assert " open" in panel, "the panel should be open before anyone has collapsed it"


# ---------------------------------------------------------------------- reaching the documents


def test_the_header_offers_the_admin_page_wherever_it_opens() -> None:
    """Documents live on the admin page, and until now the only route to them was one inline link
    inside a panel written to fold away for good. An appliance whose whole job is answering from your
    documents has to keep managing them one click away."""
    with _client_with(demo_identity=False) as client:
        html = client.get("/").text
    assert 'href="/admin"' in html
    assert "Admin" in html


def test_the_header_leaves_the_admin_page_out_beyond_loopback() -> None:
    """`require_admin` opens the console on loopback and wants a token header no browser sends
    anywhere else, so a link offered there is a 401 a reader walks into."""
    with _client_with(demo_identity=False, host="0.0.0.0") as client:
        html = client.get("/").text
    assert 'href="/admin"' not in html


def test_the_hosted_demo_carries_no_admin_item() -> None:
    """The published site listens beyond loopback, so its header stays as a visitor finds it."""
    with _client_with(demo_identity=True, host="0.0.0.0") as client:
        html = client.get("/").text
    assert 'href="/admin"' not in html


def test_the_page_gives_a_reader_a_way_to_get_in_touch() -> None:
    """The page names its author and, for a while, offered no way to reach him. Somebody who reads it
    and wants to talk should not have to go searching for the name they just read."""
    with _client_with(demo_identity=True) as client:
        html = client.get("/").text
    assert "mailto:" in html, "the overview should carry an address a reader can write to"
    assert "github.com/BlakeR-M" in html


def test_every_page_carries_the_contact_in_its_footer() -> None:
    """Not only the overview: a reader who arrives on a documentation page should find it there too."""
    with _client_with(demo_identity=True) as client:
        for path in ("/chat", "/docs", "/docs/setup"):
            assert "mailto:" in client.get(path).text, path
