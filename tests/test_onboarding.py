"""First-run discovery, `.env` writing and the preflight behind `keel doctor`.

Keel ships no model and no cloud account, so the first ten minutes are configuration rather than
code: find the chat server the operator already runs, write which one to use, and say plainly what
is standing in the way when something is unready. These tests pin that behaviour.

A stub HTTP server on a loopback port stands in for Ollama, LM Studio, llama.cpp or vLLM. All four
speak the same OpenAI-compatible `/v1/models`, which is the whole reason one probe finds any of them.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from typer.testing import CliRunner

from keel.cli import app
from keel.config import Settings
from keel.onboarding import (
    KNOWN_SERVERS,
    Check,
    discover,
    merge_env,
    probe_endpoint,
    run_checks,
)
from tests.fakes import FakeLLM

runner = CliRunner()

SERVED_MODELS = ["llama3.1:8b", "qwen2.5-3b-instruct"]


class _ModelsHandler(BaseHTTPRequestHandler):
    """Answers `/v1/models` the way every OpenAI-compatible server does."""

    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        if self.path.rstrip("/").endswith("/models"):
            body = json.dumps({"data": [{"id": name} for name in SERVED_MODELS]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        return


@pytest.fixture
def stub_server() -> Iterator[str]:
    """A loopback OpenAI-compatible endpoint, yielded as its base URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ModelsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------- probing


def test_probe_reads_the_model_list_a_server_reports(stub_server: str) -> None:
    reachable, models, detail = probe_endpoint(stub_server)
    assert reachable is True
    assert list(models) == SERVED_MODELS
    assert "2 model(s)" in detail


def test_probe_reports_a_closed_port_rather_than_raising() -> None:
    """Every failure here is something to show the reader, so none of them escape as an exception."""
    reachable, models, detail = probe_endpoint("http://127.0.0.1:9/v1", timeout=1.0)
    assert reachable is False
    assert models == ()
    assert detail, "a failure should carry a reason"


def test_probe_is_refused_by_the_air_gap_for_a_host_outside_the_allow_list() -> None:
    """Under air-gap the probe is guarded like any other connection, and reports the refusal."""
    from keel.airgap import airgapped

    with airgapped():
        reachable, _, detail = probe_endpoint("http://data.attacker.example/v1", timeout=1.0)
    assert reachable is False
    assert "Airgap" in detail or "air-gap" in detail.lower()


def test_discovery_finds_a_server_on_a_known_port(
    stub_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("keel.onboarding.KNOWN_SERVERS", (("Stub", stub_server),))
    found = discover(timeout=2.0)
    assert len(found) == 1
    assert found[0].base_url == stub_server
    assert found[0].first_model == SERVED_MODELS[0]


def test_the_known_server_list_covers_the_runtimes_people_actually_have() -> None:
    names = {name for name, _ in KNOWN_SERVERS}
    assert {"Ollama", "LM Studio", "llama.cpp", "vLLM"} <= names
    for _, base_url in KNOWN_SERVERS:
        assert base_url.startswith("http://127.0.0.1:"), "discovery probes this machine only"
        assert base_url.endswith("/v1")


# ---------------------------------------------------------------------- writing .env


def test_merge_env_writes_the_values_it_is_given(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    merge_env(env, {"KEEL_PROFILE": "local", "KEEL_LOCAL_LLM_MODEL": "llama3.1:8b"})
    text = env.read_text(encoding="utf-8")
    assert "KEEL_PROFILE=local" in text
    assert "KEEL_LOCAL_LLM_MODEL=llama3.1:8b" in text


def test_merge_env_updates_in_place_and_keeps_what_the_operator_wrote(tmp_path: Path) -> None:
    """A second `keel setup` should change the line rather than append a shadowed duplicate."""
    env = tmp_path / ".env"
    env.write_text(
        "# my notes\nKEEL_PROFILE=local\nKEEL_LOCAL_LLM_MODEL=old-model\nKEEL_PORT=9000\n",
        encoding="utf-8",
    )
    merge_env(env, {"KEEL_LOCAL_LLM_MODEL": "new-model"})
    text = env.read_text(encoding="utf-8")
    assert text.count("KEEL_LOCAL_LLM_MODEL=") == 1
    assert "KEEL_LOCAL_LLM_MODEL=new-model" in text
    assert "# my notes" in text, "comments survive"
    assert "KEEL_PORT=9000" in text, "settings this command has no opinion on survive"


# ---------------------------------------------------------------------- preflight


def named(checks: list[Check], name: str) -> Check:
    matching = [check for check in checks if check.name == name]
    assert matching, f"no check named {name}: {[c.name for c in checks]}"
    return matching[0]


def test_every_failing_check_names_a_fix(tmp_path: Path) -> None:
    """A preflight that only says something is wrong leaves the reader where it found them."""
    settings = Settings(data_dir=tmp_path, local_llm_base_url="http://127.0.0.1:9/v1")
    for check in run_checks(settings):
        if not check.ok:
            assert check.fix, f"{check.name} fails without naming a fix"


def test_an_unreachable_model_endpoint_is_reported_with_its_url(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, local_llm_base_url="http://127.0.0.1:9/v1")
    check = named(run_checks(settings), "model endpoint")
    assert check.ok is False
    assert "127.0.0.1:9" in check.detail
    assert "keel setup" in check.fix


def test_a_model_name_the_endpoint_does_not_serve_is_caught(tmp_path: Path, stub_server: str) -> None:
    """The quietest first-run failure is the right server with a model name off by one character."""
    settings = Settings(data_dir=tmp_path, local_llm_base_url=stub_server, local_llm_model="not-there")
    checks = run_checks(settings)
    assert named(checks, "model endpoint").ok is True
    name_check = named(checks, "model name")
    assert name_check.ok is False
    assert "llama3.1:8b" in name_check.detail


def test_a_model_name_the_endpoint_serves_passes(tmp_path: Path, stub_server: str) -> None:
    settings = Settings(data_dir=tmp_path, local_llm_base_url=stub_server, local_llm_model="llama3.1:8b")
    assert named(run_checks(settings), "model name").ok is True


def test_air_gap_with_the_model_host_outside_the_allow_list_is_caught(tmp_path: Path) -> None:
    """Air-gap on plus a model on another host refuses every call. Better to say so before the
    first question than to let it look like the model is broken."""
    settings = Settings(
        data_dir=tmp_path, airgap=True, local_llm_base_url="http://model.example.internal:8081/v1"
    )
    check = named(run_checks(settings), "air-gap allow list")
    assert check.ok is False
    assert "KEEL_AIRGAP_ALLOW_HOSTS" in check.fix


def test_air_gap_with_a_loopback_model_stays_quiet(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, airgap=True, local_llm_base_url="http://127.0.0.1:8081/v1")
    assert [c for c in run_checks(settings) if c.name == "air-gap allow list"] == []


def test_an_empty_store_tells_you_how_to_fill_it(tmp_path: Path) -> None:
    check = named(run_checks(Settings(data_dir=tmp_path)), "corpus")
    assert check.ok is False
    assert "ingest" in check.fix


def test_the_azure_profile_checks_the_endpoints_it_needs(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, profile="azure")
    checks = run_checks(settings)
    endpoint = named(checks, "Azure OpenAI endpoint")
    assert endpoint.ok is False
    assert "KEEL_AZURE_OPENAI_ENDPOINT" in endpoint.detail
    assert "deploy.ps1" in endpoint.fix, "the fix should point at the script that creates it"


# ---------------------------------------------------------------------- the commands


def test_setup_writes_the_endpoint_it_was_given(tmp_path: Path, stub_server: str) -> None:
    env = tmp_path / ".env"
    result = runner.invoke(
        app,
        ["setup", "--base-url", stub_server, "--model", "llama3.1:8b", "--no-ingest", "--env-file", str(env)],
    )
    assert result.exit_code == 0, result.output
    text = env.read_text(encoding="utf-8")
    assert f"KEEL_LOCAL_LLM_BASE_URL={stub_server}" in text
    assert "KEEL_LOCAL_LLM_MODEL=llama3.1:8b" in text
    assert "KEEL_PROFILE=local" in text


def test_setup_refuses_an_endpoint_that_is_not_answering(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    result = runner.invoke(
        app,
        ["setup", "--base-url", "http://127.0.0.1:9/v1", "--model", "x", "--no-ingest", "--env-file", str(env)],
    )
    assert result.exit_code == 1
    assert "out of reach" in result.output
    assert not env.exists(), "a failed setup should leave no half-written configuration"


def test_setup_for_azure_writes_the_endpoints_and_no_key(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    result = runner.invoke(
        app,
        [
            "setup", "--profile", "azure",
            "--azure-openai-endpoint", "https://example-openai.openai.azure.com",
            "--azure-search-endpoint", "https://example-search.search.windows.net",
            "--chat-deployment", "gpt-4o-mini", "--embed-deployment", "text-embedding-3-small",
            "--no-ingest", "--env-file", str(env), "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    text = env.read_text(encoding="utf-8")
    assert "KEEL_PROFILE=azure" in text
    assert "example-openai.openai.azure.com" in text
    assert "KEY" not in text.upper().replace("KEEL_", ""), "the Azure profile writes no key anywhere"


def test_setup_guides_when_nothing_is_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The most common first run: no chat server yet. Name the ones that work and how to start them."""
    monkeypatch.setattr("keel.onboarding.KNOWN_SERVERS", ())
    result = runner.invoke(app, ["setup", "--no-ingest", "--env-file", str(tmp_path / ".env")])
    assert result.exit_code == 1
    for runtime in ("Ollama", "LM Studio", "llama.cpp"):
        assert runtime in result.output, runtime
    assert "--base-url" in result.output


def test_doctor_exits_one_while_something_wants_attention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KEEL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("KEEL_LOCAL_LLM_BASE_URL", "http://127.0.0.1:9/v1")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "model endpoint" in result.output
    assert "want attention" in result.output


def test_doctor_is_listed_in_the_help_so_a_stuck_reader_finds_it() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "setup" in result.output and "doctor" in result.output


# ---------------------------------------------------------------------- serving


def test_the_browser_opens_only_where_a_person_is_sitting() -> None:
    """A server bound to 0.0.0.0 has its operator somewhere else entirely, so launching a browser on
    the box would be noise at best. Loopback is the case where it helps."""
    from keel.cli import _serves_a_person

    assert _serves_a_person("127.0.0.1") is True
    assert _serves_a_person("localhost") is True
    assert _serves_a_person("::1") is True
    assert _serves_a_person("0.0.0.0") is False
    assert _serves_a_person("10.1.2.3") is False


def test_serve_leaves_the_browser_alone_unless_asked() -> None:
    """`--open` is opt-in on `keel serve`, so a scripted or containerised start stays quiet."""
    import inspect

    from keel.cli import serve

    default = inspect.signature(serve).parameters["open_browser"].default
    assert default is False


def test_up_opens_the_browser_by_default() -> None:
    """`keel up` is the command a person runs by hand after cloning, so it finishes by showing them
    the thing they just started."""
    import inspect

    from keel.cli import up

    assert inspect.signature(up).parameters["open_browser"].default is True


def test_up_is_listed_first_among_the_commands_a_new_reader_needs() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("up", "setup", "doctor", "serve"):
        assert command in result.output, command


def test_opening_a_browser_survives_a_machine_with_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A headless box raises from webbrowser.open. The URL is printed either way, so the failure
    stays quiet rather than taking the server down with it."""
    import webbrowser

    from keel.cli import _open_browser_when_ready

    def explode(_url: str) -> bool:
        raise RuntimeError("no browser here")

    monkeypatch.setattr(webbrowser, "open", explode)
    _open_browser_when_ready("http://127.0.0.1:8400", delay=0.01)
    import time

    time.sleep(0.2)  # let the timer thread run and swallow its own failure


def test_up_loads_the_fixture_corpus_and_empty_leaves_it_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embedder, reranker  # noqa: ANN001
) -> None:
    """Two runs of the same command, so the flag is read against what it changes.

    Whoever clones Keel to look at it wants documents to ask about, and whoever clones it to answer
    from their own work wants an empty store. The fixtures stay on disk under `fixtures/` either way.
    """
    from keel import cli as cli_module
    from keel import onboarding
    from keel.config import get_settings, reset_settings
    from keel.documents import list_documents
    from keel.providers import factory
    from keel.providers.local_index import SqliteVectorIndex

    def fake_local_providers(settings_, conn):  # noqa: ANN001, ANN202
        return FakeLLM(), embedder, SqliteVectorIndex(conn, embed_model=embedder.name), reranker

    monkeypatch.setattr(factory, "_local_providers", fake_local_providers)
    # No model server to find, so the run neither probes the network nor writes a `.env`.
    monkeypatch.setattr(onboarding, "probe_endpoint", lambda *a, **k: (True, "", []))
    monkeypatch.setattr(onboarding, "discover", lambda *a, **k: [])
    served: list[bool] = []
    monkeypatch.setattr(cli_module, "_run_server", lambda *a: served.append(True))

    def run(data_dir: Path, *flags: str) -> str:
        monkeypatch.setenv("KEEL_DATA_DIR", str(data_dir))
        reset_settings()
        result = runner.invoke(app, ["up", "--no-open", *flags])
        assert result.exit_code == 0, result.output
        return result.output

    def documents(data_dir: Path) -> int:
        monkeypatch.setenv("KEEL_DATA_DIR", str(data_dir))
        reset_settings()
        context = factory.build_context(get_settings())
        try:
            return len(list_documents(context.conn))
        finally:
            context.close()

    empty_dir = tmp_path / "empty"
    run(empty_dir, "--empty")
    assert documents(empty_dir) == 0, "--empty should arrive at a store holding nothing"

    loaded_dir = tmp_path / "loaded"
    run(loaded_dir)
    assert documents(loaded_dir) > 0, "the default run should load the fixture corpus"

    assert served == [True, True], "both runs should still start the server"
    reset_settings()


def test_up_keeps_the_fixture_corpus_on_disk_either_way() -> None:
    """`--empty` is about the store rather than the repository: the fixtures are what the tests and
    the access-control comparison run against, so they stay where they are."""
    manifest = Path(__file__).resolve().parent.parent / "fixtures" / "corpus.yaml"
    assert manifest.is_file()
