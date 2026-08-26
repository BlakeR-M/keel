"""The document lifecycle: what is in the store, what it is tagged, and taking it out again.

Two of these matter more than the rest, and both are about what happens after a change rather than
about the change itself. A document retagged out of somebody's reach has to stop reaching them, and
a removed document has to stop being retrievable by any path: the keyword index, the vector index,
and the source viewer alike. Those are asserted against a real store with real retrieval rather than
by reading the SQL.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from keel.cli import app as cli_app
from keel.config import Settings, reset_settings
from keel.documents import (
    clear_documents,
    corpus_tags,
    get_document,
    list_documents,
    normalise_tags,
    remove_document,
    retag_document,
)
from keel.providers import factory
from keel.providers.factory import AppContext, build_context
from keel.providers.local_index import SqliteVectorIndex
from keel.web.app import app
from tests.fakes import FakeLLM

runner = CliRunner()

FIXTURE_MANIFEST = Path(__file__).resolve().parent.parent / "fixtures" / "corpus.yaml"
QUOTES_QUESTION = "How many written quotes does a $20,000 purchase need?"


@pytest.fixture
def ctx(tmp_path: Path, embedder, reranker, monkeypatch: pytest.MonkeyPatch) -> Iterator[AppContext]:  # noqa: ANN001
    """A real store holding the fixture corpus, with a fake model and no network.

    Function scoped rather than module scoped: these tests remove documents, and a shared store would
    make each one depend on the order the others ran in.
    """
    settings = Settings(data_dir=tmp_path, airgap=False)
    # The command-line tests at the foot of this file go through `_context()`, which reads the cached
    # module-level settings rather than the object built here. Without clearing that cache every
    # `runner.invoke` in the file acts on whichever store was cached first, and a count asserted after
    # another test mutated that store comes back wrong.
    monkeypatch.setenv("KEEL_DATA_DIR", str(tmp_path))
    reset_settings()

    def fake_local_providers(settings_: Settings, conn):  # noqa: ANN001, ANN202
        return FakeLLM(), embedder, SqliteVectorIndex(conn, embed_model=embedder.name), reranker

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(factory, "_local_providers", fake_local_providers)
        context = build_context(settings)
    from keel.ingest import ingest_manifest

    ingest_manifest(
        context.conn,
        context.settings,
        context.embedder,
        context.index,
        FIXTURE_MANIFEST,
        screen=context.screen,
        ledger=context.ledger,
    )
    yield context
    context.close()
    reset_settings()


@pytest.fixture
def client(ctx: AppContext) -> Iterator[TestClient]:
    app.state.ctx = ctx
    app.state.ctx_owned = False
    with TestClient(app) as test_client:
        yield test_client
    app.state.ctx = None


def procurement_id(ctx: AppContext) -> int:
    row = ctx.conn.execute(
        "SELECT id FROM documents WHERE source LIKE '%northbank-council-procurement%'"
    ).fetchone()
    assert row is not None
    return int(row[0])


def retrieves(ctx: AppContext, question: str, tags: list[str]) -> set[int]:
    """Chunk ids the retriever returns for a caller holding `tags`."""
    return {int(hit.chunk_id) for hit in ctx.retriever.retrieve(question, tags)}


def chunks_of(ctx: AppContext, document_id: int) -> set[int]:
    """This document's own chunk ids.

    Retrieval returns the best chunks across the whole corpus, so comparing whole result sets would
    compare other documents too. The question is only ever about these ids.
    """
    return {
        int(row[0])
        for row in ctx.conn.execute("SELECT id FROM chunks WHERE document_id = ?", (document_id,))
    }


# ---------------------------------------------------------------------- tags


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("hr, finance", ["hr", "finance"]),
        ("HR ,  Finance , hr", ["hr", "finance"]),
        ("public;hr", ["public", "hr"]),
        (["Ops", "ops", " "], ["ops"]),
        ("", ["public"]),
        ([], ["public"]),
        (None, ["public"]),
    ],
)
def test_tags_are_cleaned_and_an_empty_list_becomes_public(given: object, expected: list[str]) -> None:
    """An empty tag list would mean a document nobody can retrieve, which loses it quietly rather
    than saying so. `public` is the honest default."""
    assert normalise_tags(given) == expected


# ---------------------------------------------------------------------- listing


def test_the_store_lists_what_it_holds(ctx: AppContext) -> None:
    rows = list_documents(ctx.conn)
    assert len(rows) == 5
    by_title = {row.title: row for row in rows}
    assert by_title["Northbank Salary Bands"].acl_tags == ["hr"]
    assert by_title["Northbank City Council Procurement Guide"].chunks > 0
    assert any(row.quarantined for row in rows), "the planted injection should show as quarantined"


def test_the_tags_in_use_come_from_the_documents_themselves(ctx: AppContext) -> None:
    assert corpus_tags(ctx.conn) == ["hr", "public"]


# ---------------------------------------------------------------------- retagging


def test_retagging_moves_the_document_out_of_reach_of_the_old_tag(ctx: AppContext) -> None:
    """The security property. A document retagged to `hr` stops reaching a `public` reader, and the
    chunks move with the document because retrieval filters on the chunk."""
    document_id = procurement_id(ctx)
    mine = chunks_of(ctx, document_id)
    assert mine & retrieves(ctx, QUOTES_QUESTION, ["public"]), (
        "the fixture should be reachable by a public reader to begin with"
    )

    retag_document(ctx.conn, ctx.index, document_id, "hr", by="tester", ledger=ctx.ledger)

    assert not (mine & retrieves(ctx, QUOTES_QUESTION, ["public"])), (
        "a public reader still reaches the retagged chunks"
    )
    assert mine & retrieves(ctx, QUOTES_QUESTION, ["hr"]), "an hr reader should now reach them"


def test_retagging_updates_the_chunks_as_well_as_the_document(ctx: AppContext) -> None:
    document_id = procurement_id(ctx)
    retag_document(ctx.conn, ctx.index, document_id, "ops, safety", ledger=ctx.ledger)
    stored = get_document(ctx.conn, document_id)
    assert stored is not None and stored.acl_tags == ["ops", "safety"]
    distinct = {
        row[0]
        for row in ctx.conn.execute("SELECT DISTINCT acl_tags FROM chunks WHERE document_id = ?", (document_id,))
    }
    assert distinct == {'["ops", "safety"]'}


def test_retagging_lands_in_the_ledger_and_leaves_the_chain_intact(ctx: AppContext) -> None:
    from keel.safety.ledger import Ledger

    before = ctx.ledger.count()
    retag_document(ctx.conn, ctx.index, procurement_id(ctx), "ops", by="tester", ledger=ctx.ledger)
    assert ctx.ledger.count() > before
    assert Ledger(ctx.conn).verify().ok


def test_retagging_to_the_same_tags_changes_nothing(ctx: AppContext) -> None:
    before = ctx.ledger.count()
    row = retag_document(ctx.conn, ctx.index, procurement_id(ctx), "public", ledger=ctx.ledger)
    assert row.acl_tags == ["public"]
    assert ctx.ledger.count() == before, "a change that changes nothing should record nothing"


def test_retagging_a_document_that_is_absent_says_so(ctx: AppContext) -> None:
    with pytest.raises(LookupError):
        retag_document(ctx.conn, ctx.index, 9999, "public", ledger=ctx.ledger)


# ---------------------------------------------------------------------- removal


def test_a_removed_document_is_retrievable_by_no_path(ctx: AppContext) -> None:
    """The other security property. Removal has to clear the keyword index and the vector index too,
    since a chunk left in either is a chunk that can still reach a prompt."""
    document_id = procurement_id(ctx)
    mine = chunks_of(ctx, document_id)
    assert mine & retrieves(ctx, QUOTES_QUESTION, ["public"])

    removed = remove_document(ctx.conn, ctx.index, document_id, by="tester", ledger=ctx.ledger)
    assert removed.chunks_removed == len(mine)

    assert get_document(ctx.conn, document_id) is None
    assert chunks_of(ctx, document_id) == set()
    assert not (mine & retrieves(ctx, QUOTES_QUESTION, ["public"])), "removed chunks still retrievable"
    assert not (mine & retrieves(ctx, QUOTES_QUESTION, ["public", "hr", "ops"])), (
        "removed chunks reachable by a reader holding every tag"
    )


def test_removal_clears_the_full_text_index(ctx: AppContext) -> None:
    document_id = procurement_id(ctx)
    remove_document(ctx.conn, ctx.index, document_id, ledger=ctx.ledger)
    orphans = ctx.conn.execute(
        "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'procurement'"
    ).fetchone()[0]
    remaining = ctx.conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE text LIKE '%procurement%'"
    ).fetchone()[0]
    assert orphans <= remaining, "the full-text index holds rows the chunks table no longer has"


def test_removal_lands_in_the_ledger_before_the_rows_go(ctx: AppContext) -> None:
    """The ledger row describes something that will not exist a moment later, so it has to carry
    enough to say what left."""
    from keel.safety.ledger import Ledger

    document_id = procurement_id(ctx)
    remove_document(ctx.conn, ctx.index, document_id, by="tester", ledger=ctx.ledger)
    assert Ledger(ctx.conn).verify().ok
    rows = ctx.conn.execute("SELECT payload FROM ledger ORDER BY seq DESC LIMIT 5").fetchall()
    text = " ".join(str(row[0]) for row in rows)
    assert "remove" in text and "northbank-council-procurement" in text


def test_removing_a_document_that_is_absent_says_so(ctx: AppContext) -> None:
    with pytest.raises(LookupError):
        remove_document(ctx.conn, ctx.index, 9999, ledger=ctx.ledger)


# ---------------------------------------------------------------------- the routes


def test_the_admin_page_lists_the_documents_with_their_tags(client: TestClient) -> None:
    html = client.get("/admin").text
    assert "Northbank Salary Bands" in html
    assert 'action="/admin/documents/' in html
    assert "Retag" in html and "Remove" in html


def test_retagging_through_the_route(client: TestClient, ctx: AppContext) -> None:
    document_id = procurement_id(ctx)
    response = client.post(
        f"/admin/documents/{document_id}/retag",
        data={"tags": "ops, safety", "by": "tester"},
        headers={"accept": "application/json"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["acl_tags"] == ["ops", "safety"]


def test_removing_through_the_route(client: TestClient, ctx: AppContext) -> None:
    document_id = procurement_id(ctx)
    response = client.post(
        f"/admin/documents/{document_id}/remove",
        data={"by": "tester"},
        headers={"accept": "application/json"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["chunks_removed"] > 0
    assert get_document(ctx.conn, document_id) is None


@pytest.mark.parametrize("action", ["retag", "remove"])
def test_a_document_that_is_absent_answers_404(client: TestClient, action: str) -> None:
    response = client.post(
        f"/admin/documents/9999/{action}", data={"tags": "public"}, headers={"accept": "application/json"}
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------- the connection page


def test_the_connection_page_reports_what_keel_is_pointed_at(client: TestClient) -> None:
    response = client.get("/admin/connection")
    assert response.status_code == 200
    html = response.text
    assert "Connection" in html
    assert "model endpoint" in html, "the doctor checks should appear"
    assert "data directory" in html


def test_the_connection_page_refuses_to_switch_to_a_host_off_this_machine(client: TestClient) -> None:
    """A form that accepts any endpoint is an exfiltration lever: point the model somewhere you
    control and every question and every retrieved passage follows it."""
    response = client.post(
        "/admin/connection",
        data={"base_url": "https://attacker.example/v1", "model": "anything"},
        headers={"accept": "application/json"},
    )
    assert response.status_code == 400
    assert "this machine" in response.text


@pytest.mark.parametrize(
    "base_url",
    ["http://10.1.2.3:11434/v1", "http://model.example.internal:8081/v1", "https://api.openai.com/v1"],
)
def test_every_off_machine_endpoint_is_refused(client: TestClient, base_url: str) -> None:
    response = client.post(
        "/admin/connection",
        data={"base_url": base_url, "model": "m"},
        headers={"accept": "application/json"},
    )
    assert response.status_code == 400


def test_switching_needs_both_an_endpoint_and_a_model(client: TestClient) -> None:
    response = client.post(
        "/admin/connection", data={"base_url": ""}, headers={"accept": "application/json"}
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------- the commands


def test_the_command_line_lists_the_same_documents(ctx: AppContext, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEEL_DATA_DIR", str(ctx.settings.data_dir))
    result = runner.invoke(cli_app, ["documents", "list"])
    assert result.exit_code == 0, result.output
    assert "Northbank Salary Bands" in result.output
    assert "5 documents" in result.output


def test_the_command_line_retags(ctx: AppContext, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEEL_DATA_DIR", str(ctx.settings.data_dir))
    document_id = procurement_id(ctx)
    result = runner.invoke(cli_app, ["documents", "retag", str(document_id), "--tags", "ops", "--by", "t"])
    assert result.exit_code == 0, result.output
    assert "ops" in result.output


def test_the_command_line_removes_only_with_confirmation(
    ctx: AppContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removal is the one irreversible action, so the prompt is the default and `--yes` is the opt out."""
    monkeypatch.setenv("KEEL_DATA_DIR", str(ctx.settings.data_dir))
    document_id = procurement_id(ctx)
    declined = runner.invoke(cli_app, ["documents", "remove", str(document_id)], input="n\n")
    assert declined.exit_code != 0
    assert get_document(ctx.conn, document_id) is not None, "declining should leave the document alone"

    accepted = runner.invoke(cli_app, ["documents", "remove", str(document_id), "--yes"])
    assert accepted.exit_code == 0, accepted.output


def test_the_command_line_says_so_when_the_document_is_absent(
    ctx: AppContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KEEL_DATA_DIR", str(ctx.settings.data_dir))
    result = runner.invoke(cli_app, ["documents", "remove", "9999", "--yes"])
    assert result.exit_code == 1
    assert "9999" in result.output


# ---------------------------------------------------------------------- clearing


def test_clearing_empties_the_store_and_leaves_nothing_behind(ctx: AppContext) -> None:
    """The same property removal has, asked of the whole store at once: no documents, no chunks, and
    no full-text rows pointing at chunks that are gone."""
    before = list_documents(ctx.conn)
    assert len(before) > 1, "the fixture corpus should hold several documents for this to mean much"

    removed = clear_documents(ctx.conn, ctx.index, by="tester", ledger=ctx.ledger)

    assert len(removed) == len(before)
    assert list_documents(ctx.conn) == []
    assert int(ctx.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]) == 0
    assert int(ctx.conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]) == 0


def test_a_cleared_store_retrieves_nothing_at_all(ctx: AppContext) -> None:
    """Retrieval is where it matters. A store that reports itself empty while still answering from
    the chunks it used to hold would be the worst version of this."""
    assert retrieves(ctx, QUOTES_QUESTION, ["public"]), "the question should reach the corpus first"
    clear_documents(ctx.conn, ctx.index, ledger=ctx.ledger)
    assert retrieves(ctx, QUOTES_QUESTION, ["public"]) == set()


def test_clearing_records_each_document_separately_and_leaves_the_chain_intact(ctx: AppContext) -> None:
    """One entry per document rather than one for the clear, so the audit trail says what left."""
    from keel.safety.ledger import Ledger

    documents = len(list_documents(ctx.conn))
    before = ctx.ledger.count()
    clear_documents(ctx.conn, ctx.index, by="tester", ledger=ctx.ledger)
    assert ctx.ledger.count() == before + documents
    assert Ledger(ctx.conn).verify().ok


def test_clearing_a_store_that_is_already_empty_changes_nothing(ctx: AppContext) -> None:
    clear_documents(ctx.conn, ctx.index, ledger=ctx.ledger)
    before = ctx.ledger.count()
    assert clear_documents(ctx.conn, ctx.index, ledger=ctx.ledger) == []
    assert ctx.ledger.count() == before, "a clear with nothing to clear should record nothing"


def test_the_command_line_clears_only_with_confirmation(
    ctx: AppContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Emptying the store is the largest irreversible action here, so the prompt is the default."""
    monkeypatch.setenv("KEEL_DATA_DIR", str(ctx.settings.data_dir))
    declined = runner.invoke(cli_app, ["documents", "clear"], input="n\n")
    assert declined.exit_code != 0
    assert list_documents(ctx.conn), "declining should leave every document in place"

    held = len(list_documents(ctx.conn))
    accepted = runner.invoke(cli_app, ["documents", "clear", "--yes"])
    assert accepted.exit_code == 0, accepted.output
    assert f"cleared {held} document(s)" in accepted.output
    assert list_documents(ctx.conn) == []


def test_the_command_line_says_so_when_there_is_nothing_to_clear(
    ctx: AppContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KEEL_DATA_DIR", str(ctx.settings.data_dir))
    assert runner.invoke(cli_app, ["documents", "clear", "--yes"]).exit_code == 0
    again = runner.invoke(cli_app, ["documents", "clear"])
    assert again.exit_code == 0, again.output
    assert "no documents already" in again.output
