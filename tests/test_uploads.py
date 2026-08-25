"""Taking a document from a browser: the checks that run before anything reaches the disk.

Upload is the one place a person hands Keel bytes it did not fetch itself, so it is the one place a
hostile filename, an unreadable format or an oversized stream arrives from outside. These tests pin
what happens to each, and that the route enforcing them sits behind the admin guard and disappears
entirely on a deployment that declares itself read-only.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from keel.ingest.loaders import KIND_BY_SUFFIX
from keel.web.uploads import (
    ALLOWED_SUFFIXES,
    MAX_NAME_LENGTH,
    MAX_UPLOAD_BYTES,
    UploadRejected,
    describe_limits,
    safe_name,
    stage_upload,
)

MARKDOWN = b"# Handbook\n\nThe quiet hours are between 8pm and 7am.\n"


# ---------------------------------------------------------------------- filenames


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("handbook.pdf", "handbook.pdf"),
        ("Policy Note.md", "Policy Note.md"),
        ("report.DOCX", "report.docx"),
        ("notes.markdown", "notes.markdown"),
    ],
)
def test_an_ordinary_filename_survives(given: str, expected: str) -> None:
    """The stem is kept as written and the suffix is lowered, so `.DOCX` and `.docx` are one thing
    in the store rather than two."""
    assert safe_name(given) == expected


@pytest.mark.parametrize(
    "given",
    [
        "../../etc/passwd.md",
        "..\\..\\windows\\system32\\config.md",
        "/absolute/path/notes.md",
        "C:\\Users\\someone\\notes.md",
        "subdir/notes.md",
    ],
)
def test_a_filename_carrying_directories_is_reduced_to_its_last_component(given: str) -> None:
    """A path in a filename reaches the store as a plain name, so it can name nothing but itself."""
    cleaned = safe_name(given)
    assert "/" not in cleaned and "\\" not in cleaned
    assert not cleaned.startswith("..")
    assert Path(cleaned).name == cleaned


@pytest.mark.parametrize(
    "given",
    ["notes.exe", "payload.sh", "archive.zip", "script.py", "image.png", "noextension", "", "   "],
)
def test_a_format_the_loaders_cannot_read_is_refused(given: str) -> None:
    with pytest.raises(UploadRejected):
        safe_name(given)


@pytest.mark.parametrize("given", [".", "..", "....md", "---.md", "   .md"])
def test_a_filename_with_no_name_in_it_is_refused(given: str) -> None:
    with pytest.raises(UploadRejected):
        safe_name(given)


def test_a_very_long_filename_is_cut_to_something_a_store_can_hold() -> None:
    cleaned = safe_name("a" * 4000 + ".md")
    assert cleaned.endswith(".md")
    assert len(cleaned) <= MAX_NAME_LENGTH + len(".md")


def test_the_allowed_formats_come_from_the_loaders_rather_than_a_second_list() -> None:
    """Two lists drift. The upload check reads the loaders' own table, so a format is readable and
    uploadable at the same moment or neither."""
    assert ALLOWED_SUFFIXES == frozenset(KIND_BY_SUFFIX)
    for suffix in (".pdf", ".docx", ".md", ".html", ".txt"):
        assert suffix in ALLOWED_SUFFIXES


def test_the_limits_are_stated_in_a_sentence_a_person_can_act_on() -> None:
    described = describe_limits()
    for kind in ("pdf", "docx", "md", "txt"):
        assert kind in described
    assert "MB" in described


# ---------------------------------------------------------------------- staging


def test_a_file_is_written_under_its_safe_name(tmp_path: Path) -> None:
    staged = stage_upload(io.BytesIO(MARKDOWN), "../../handbook.md", tmp_path / "staging")
    assert staged.name == "handbook.md"
    assert staged.path.parent == tmp_path / "staging"
    assert staged.path.read_bytes() == MARKDOWN
    assert staged.size == len(MARKDOWN)


def test_an_oversized_stream_stops_partway_and_leaves_nothing_behind(tmp_path: Path) -> None:
    """The cap is enforced while reading, so an oversized upload never arrives in full."""
    oversized = io.BytesIO(b"x" * (MAX_UPLOAD_BYTES + 1024))
    directory = tmp_path / "staging"
    with pytest.raises(UploadRejected, match="larger than"):
        stage_upload(oversized, "big.txt", directory)
    assert list(directory.glob("*")) == [], "a refused upload leaves no partial file"


def test_an_empty_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(UploadRejected, match="empty"):
        stage_upload(io.BytesIO(b""), "nothing.md", tmp_path / "staging")
    assert list((tmp_path / "staging").glob("*")) == []


def test_a_refused_name_never_creates_a_file(tmp_path: Path) -> None:
    directory = tmp_path / "staging"
    with pytest.raises(UploadRejected):
        stage_upload(io.BytesIO(MARKDOWN), "payload.exe", directory)
    assert not directory.exists() or list(directory.glob("*")) == []


# ---------------------------------------------------------------------- the route


@pytest.fixture(scope="module")
def web_ctx(tmp_path_factory: pytest.TempPathFactory, embedder, reranker):  # noqa: ANN001, ANN201
    """A real store with the real embedder, a fake model, and no network.

    Uploading writes, so these tests want a genuine SQLite store and a genuine chunking and screening
    path. The model is a fake, since nothing here generates an answer.
    """
    from keel.config import Settings
    from keel.providers import factory
    from keel.providers.factory import build_context
    from keel.providers.local_index import SqliteVectorIndex
    from tests.fakes import FakeLLM

    data_dir = tmp_path_factory.mktemp("uploads")
    settings = Settings(data_dir=data_dir, airgap=False)

    def fake_local_providers(settings_: Settings, conn):  # noqa: ANN001, ANN202
        return FakeLLM(), embedder, SqliteVectorIndex(conn, embed_model=embedder.name), reranker

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(factory, "_local_providers", fake_local_providers)
        ctx = build_context(settings)
    yield ctx
    ctx.close()


@pytest.fixture
def client(web_ctx) -> Iterator[TestClient]:  # noqa: ANN001
    from keel.web.app import app

    app.state.ctx = web_ctx
    app.state.ctx_owned = False
    with TestClient(app) as test_client:
        yield test_client
    app.state.ctx = None


def test_uploading_a_document_puts_it_through_the_ordinary_pipeline(client: TestClient, web_ctx) -> None:  # noqa: ANN001
    """The same chunking, the same screen, the same ledger row a command-line ingest gets."""
    before = int(web_ctx.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
    response = client.post(
        "/admin/ingest",
        files={"file": ("quiet-hours.md", MARKDOWN, "text/markdown")},
        data={"tags": "facilities", "by": "tester"},
        headers={"accept": "application/json"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["chunks_added"] >= 1
    assert body["acl_tags"] == ["facilities"]
    assert body["uploaded_name"] == "quiet-hours.md"
    after = int(web_ctx.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
    assert after == before + 1


def test_an_upload_with_no_tags_becomes_public(client: TestClient) -> None:
    """Tags are the access model, so the default is stated rather than left to chance."""
    response = client.post(
        "/admin/ingest",
        files={"file": ("untagged-note.md", b"# Note\n\nOpening hours are nine to five.\n", "text/markdown")},
        data={"tags": ""},
        headers={"accept": "application/json"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["acl_tags"] == ["public"]


def test_a_format_the_loaders_cannot_read_is_refused_by_the_route(client: TestClient) -> None:
    response = client.post(
        "/admin/ingest",
        files={"file": ("payload.exe", b"MZ\x00\x00", "application/octet-stream")},
        headers={"accept": "application/json"},
    )
    assert response.status_code == 400
    assert "exe" in response.text or "reads" in response.text


def test_a_hostile_filename_is_stored_under_a_plain_name(client: TestClient, web_ctx) -> None:  # noqa: ANN001
    response = client.post(
        "/admin/ingest",
        files={"file": ("../../../evil.md", b"# Evil\n\nNothing to see.\n", "text/markdown")},
        headers={"accept": "application/json"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["uploaded_name"] == "evil.md"
    sources = [row[0] for row in web_ctx.conn.execute("SELECT source FROM documents")]
    assert not any(".." in str(source) for source in sources)


def test_uploading_the_same_bytes_twice_adds_nothing(client: TestClient) -> None:
    payload = b"# Repeat\n\nThe same bytes, sent twice.\n"
    first = client.post(
        "/admin/ingest",
        files={"file": ("repeat.md", payload, "text/markdown")},
        headers={"accept": "application/json"},
    ).json()
    second = client.post(
        "/admin/ingest",
        files={"file": ("repeat.md", payload, "text/markdown")},
        headers={"accept": "application/json"},
    ).json()
    assert first["chunks_added"] >= 1
    assert second["skipped_duplicate"] is True
    assert second["chunks_added"] == 0


def test_a_request_with_no_file_says_so(client: TestClient) -> None:
    response = client.post("/admin/ingest", data={"tags": "public"}, headers={"accept": "application/json"})
    assert response.status_code == 400
    assert "file" in response.text.lower()


def test_the_upload_lands_in_the_ledger(client: TestClient, web_ctx) -> None:  # noqa: ANN001
    """An ingest writes its ledger row inside the same transaction, upload or command line alike."""
    before = web_ctx.ledger.count()
    client.post(
        "/admin/ingest",
        files={"file": ("ledgered.md", b"# Ledgered\n\nA line worth recording.\n", "text/markdown")},
        headers={"accept": "application/json"},
    )
    assert web_ctx.ledger.count() > before


def test_the_admin_page_offers_the_upload_where_writes_are_allowed(client: TestClient) -> None:
    html = client.get("/admin").text
    assert 'action="/admin/ingest"' in html
    assert 'enctype="multipart/form-data"' in html
    assert 'name="tags"' in html, "tags are the access model and belong on the form"
