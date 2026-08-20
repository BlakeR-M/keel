"""Shared test fixtures: a temporary data directory, settings pointed at it, a fresh SQLite store, and
session-scoped local embedding and reranking models (cached on disk; no network)."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from keel.config import Settings, get_settings, reset_settings
from keel.db import connect

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_MANIFEST = REPO_ROOT / "fixtures" / "corpus.yaml"


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """An empty data directory for one test."""
    path = tmp_path / "data"
    path.mkdir()
    return path


@pytest.fixture
def settings(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    """Settings whose data_dir is the temporary directory. `get_settings()` returns the same object."""
    monkeypatch.setenv("KEEL_DATA_DIR", str(data_dir))
    reset_settings()
    yield get_settings()
    reset_settings()


@pytest.fixture
def conn(settings: Settings) -> Iterator[sqlite3.Connection]:
    """A fresh SQLite store with the schema applied, closed after the test."""
    connection = connect(settings.db_path)
    yield connection
    connection.close()


@pytest.fixture
def fixture_manifest() -> Path:
    """Path to fixtures/corpus.yaml."""
    return FIXTURE_MANIFEST


@pytest.fixture(scope="session")
def embedder():
    """The local fastembed embedding model, loaded once per session."""
    from keel.providers.local_embed import FastembedEmbeddings

    return FastembedEmbeddings(Settings().local_embed_model)


@pytest.fixture(scope="session")
def reranker():
    """The local fastembed cross-encoder, loaded once per session."""
    from keel.providers.local_rerank import FastembedReranker

    return FastembedReranker(Settings().local_rerank_model)
