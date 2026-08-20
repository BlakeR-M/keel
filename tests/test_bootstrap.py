"""Startup ingest: `ensure_demo_corpus` fills an empty store from a manifest and leaves a populated
store alone; the web app runs it from its lifespan when KEEL_BOOTSTRAP_CORPUS is set."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from keel.bootstrap import document_count, ensure_demo_corpus
from keel.config import Settings
from keel.providers import factory
from keel.providers.factory import AppContext, build_context
from keel.providers.local_index import SqliteVectorIndex
from keel.web.app import app
from tests.fakes import FakeLLM

FIXTURE_MANIFEST = Path(__file__).resolve().parent.parent / "fixtures" / "corpus.yaml"


@pytest.fixture
def empty_ctx(tmp_path: Path, embedder, reranker):
    settings = Settings(data_dir=tmp_path, airgap=False, bootstrap_corpus=FIXTURE_MANIFEST)

    def fake_local_providers(settings_: Settings, conn):
        return FakeLLM(), embedder, SqliteVectorIndex(conn, embed_model=embedder.name), reranker

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(factory, "_local_providers", fake_local_providers)
        ctx = build_context(settings)
    yield ctx
    ctx.close()


def test_ensure_demo_corpus_ingests_once_and_then_leaves_the_store_alone(empty_ctx: AppContext) -> None:
    assert document_count(empty_ctx) == 0
    results = ensure_demo_corpus(empty_ctx, FIXTURE_MANIFEST)
    assert len(results) == 5
    assert sum(r.chunks_added for r in results) == 27
    assert sum(r.quarantined for r in results) == 1
    assert document_count(empty_ctx) == 5
    assert ensure_demo_corpus(empty_ctx, FIXTURE_MANIFEST) == []
    assert document_count(empty_ctx) == 5


def test_ensure_demo_corpus_names_a_missing_manifest(empty_ctx: AppContext, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ensure_demo_corpus(empty_ctx, tmp_path / "absent.yaml")
    assert document_count(empty_ctx) == 0


def test_lifespan_bootstraps_the_corpus_when_configured(empty_ctx: AppContext) -> None:
    app.state.ctx = empty_ctx
    app.state.ctx_owned = False
    try:
        with TestClient(app) as client:
            health = client.get("/health").json()
            assert health["documents"] == 5 and health["chunks"] == 27 and health["quarantined"] == 1
    finally:
        app.state.ctx = None
