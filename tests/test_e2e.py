"""End-to-end through the CLI against the real local stack: llama-server on 127.0.0.1:8081, fastembed
embeddings and reranker, SQLite. Skips cleanly when the server is unreachable.

One fresh temporary data directory per module: ingest the fixture corpus, ask a public question and a
restricted one, run the agent with the calculator, then verify the ledger the run produced.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from keel.answer.prompts import REFUSAL
from keel.cli import app
from keel.config import reset_settings
from tests.conftest import FIXTURE_MANIFEST

LLM_MODELS_URL = "http://127.0.0.1:8081/v1/models"

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _server_up(url: str = LLM_MODELS_URL) -> bool:
    try:
        return httpx.get(url, timeout=2).status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _server_up(), reason=f"llama-server is not reachable at {LLM_MODELS_URL}"),
]

runner = CliRunner()


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """A fresh data directory with the fixture corpus ingested through the CLI, shared by the module."""
    path = tmp_path_factory.mktemp("e2e") / "data"
    patch = pytest.MonkeyPatch()
    patch.setenv("KEEL_DATA_DIR", str(path))
    reset_settings()
    result = runner.invoke(app, ["--data-dir", str(path), "ingest", "--manifest", str(FIXTURE_MANIFEST)])
    assert result.exit_code == 0, result.output
    assert "5 documents: 5 new, 0 duplicate;" in result.output
    yield path
    patch.undo()
    reset_settings()


def keel(data_dir: Path, *args: str):
    result = runner.invoke(app, ["--data-dir", str(data_dir), *args])
    assert result.exit_code == 0, f"exit {result.exit_code}\n{result.output}\n{result.exception!r}"
    return result


def test_ask_procurement_question_is_answered_with_a_northbank_citation(data_dir: Path) -> None:
    result = keel(
        data_dir, "ask", "How many written quotes does a $20,000 purchase need at Northbank Council?"
    )
    assert "three" in result.output.lower()
    assert re.search(r"^\[1\] Northbank City Council Procurement Guide · ", result.output, re.MULTILINE)
    assert "status: answered" in result.output


def test_ask_restricted_question_as_public_user_is_refused_without_leaking(data_dir: Path) -> None:
    result = keel(
        data_dir,
        "ask",
        "What is the confidential review code for the 2026 pay round?",
        "--user",
        "pat",
        "--tags",
        "public",
    )
    assert result.output.startswith(REFUSAL)
    assert "status: refused" in result.output
    assert "PELICAN" not in result.output


def test_agent_uses_the_calculator(data_dir: Path) -> None:
    result = keel(data_dir, "agent", "What is 1234*5678? Use the calculator tool.")
    assert "7006652" in result.output.replace(",", "")
    assert re.search(r"^\d+\s+calculator\s+allowed\s+7006652$", result.output, re.MULTILINE)


def test_verify_ledger_after_the_run_is_intact(data_dir: Path) -> None:
    result = keel(data_dir, "verify-ledger")
    assert re.search(
        r"^ledger: intact · \d+ rows checked · head seq \d+ · head [0-9a-f]{64}$", result.output, re.MULTILINE
    )
