"""Air-gap egress guard: refusals happen before any packet leaves, loopback keeps working, and
disabling puts every original back. Unit tests use fakes and a loopback HTTP server; the two
integration tests touch the local llama-server and the fastembed cache and skip when absent.
"""

from __future__ import annotations

import asyncio
import os
import socket
import tempfile
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
import yaml

from keel import airgap
from keel.airgap import (
    AirgapViolation,
    airgap_async_transport,
    airgap_transport,
    airgapped,
    enforce,
    enforce_from_settings,
    guard_httpx_client,
    is_enabled,
    is_host_allowed,
)
from keel.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]
LLAMA_MODELS_URL = "http://127.0.0.1:8081/v1/models"


# --------------------------------------------------------------------------------------------------
# Fixtures


@pytest.fixture(autouse=True)
def _guard_off(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every test starts and ends with the guard off.

    Requesting `monkeypatch` orders this teardown before monkeypatch's own undo, so tests that
    pre-patch socket functions get them restored in the right sequence.
    """
    enforce(False)
    yield
    enforce(False)


class _OkHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 (stdlib signature)
        return


class _QuietServer(ThreadingHTTPServer):
    """Loopback server that stays silent when a test connects and closes without sending a request."""

    def handle_error(self, request: object, client_address: object) -> None:
        return


@pytest.fixture(scope="module")
def local_server() -> Iterator[tuple[str, int]]:
    """A loopback HTTP server; yields (base_url, port)."""
    server = _QuietServer(("127.0.0.1", 0), _OkHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", port
    finally:
        server.shutdown()
        server.server_close()


class _Recorder:
    """Stands in for an original network function and records every call."""

    def __init__(self, result: object = None) -> None:
        self.calls: list[tuple] = []
        self.result = result

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.calls.append(args)
        return self.result


# --------------------------------------------------------------------------------------------------
# Socket layer


def test_socket_connect_is_refused_before_the_original_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_connect = _Recorder()
    fake_connect_ex = _Recorder(result=0)
    monkeypatch.setattr(socket.socket, "connect", fake_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", fake_connect_ex)
    enforce(True)

    with socket.socket() as sock:
        with pytest.raises(AirgapViolation) as info:
            sock.connect(("1.1.1.1", 80))
        assert info.value.host == "1.1.1.1"
        assert info.value.port == 80
        with pytest.raises(AirgapViolation):
            sock.connect(("example.com", 443))
        with pytest.raises(AirgapViolation):
            sock.connect_ex(("1.1.1.1", 80))
    assert fake_connect.calls == []
    assert fake_connect_ex.calls == []

    with socket.socket() as sock:
        sock.connect(("127.0.0.1", 9))
        assert sock.connect_ex(("localhost", 9)) == 0
    assert [call[1] for call in fake_connect.calls] == [("127.0.0.1", 9)]
    assert [call[1] for call in fake_connect_ex.calls] == [("localhost", 9)]

    enforce(False)
    assert socket.socket.connect is fake_connect
    assert socket.socket.connect_ex is fake_connect_ex


def test_udp_sendto_is_refused_before_the_original_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_sendto = _Recorder(result=1)
    monkeypatch.setattr(socket.socket, "sendto", fake_sendto)
    enforce(True)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        with pytest.raises(AirgapViolation):
            sock.sendto(b"x", ("8.8.8.8", 53))
        with pytest.raises(AirgapViolation):
            sock.sendto(b"x", 0, ("8.8.8.8", 53))
        assert fake_sendto.calls == []
        assert sock.sendto(b"x", ("127.0.0.1", 53)) == 1
    assert len(fake_sendto.calls) == 1


def test_create_connection_checks_the_name_before_resolving(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_create = _Recorder(result="socket-stand-in")
    monkeypatch.setattr(socket, "create_connection", fake_create)
    enforce(True)

    with pytest.raises(AirgapViolation):
        socket.create_connection(("example.com", 443), timeout=1)
    assert fake_create.calls == []
    assert socket.create_connection(("localhost", 8081), 1) == "socket-stand-in"
    assert fake_create.calls == [(("localhost", 8081), 1)]

    enforce(False)
    assert socket.create_connection is fake_create


def test_loopback_socket_connection_succeeds_while_enabled(local_server: tuple[str, int]) -> None:
    _, port = local_server
    enforce(True)
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(b"GET / HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
        assert sock.recv(64).startswith(b"HTTP/1.0 200")


# --------------------------------------------------------------------------------------------------
# httpx


def test_httpx_transport_refuses_external_hosts_before_sending(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_create = _Recorder()
    monkeypatch.setattr(socket, "create_connection", fake_create)
    enforce(True)
    with httpx.Client(transport=airgap_transport()) as client:
        with pytest.raises(AirgapViolation) as info:
            client.get("https://example.com")
    assert info.value.via == "httpx"
    assert fake_create.calls == []


def test_httpx_transport_guards_even_when_enforce_is_off() -> None:
    assert not is_enabled()
    with httpx.Client(transport=airgap_transport()) as client:
        with pytest.raises(AirgapViolation):
            client.get("https://example.com")


def test_httpx_transport_allows_loopback(local_server: tuple[str, int]) -> None:
    base_url, _ = local_server
    enforce(True)
    with httpx.Client(transport=airgap_transport(), timeout=5) as client:
        response = client.get(base_url + "/")
    assert response.status_code == 200
    assert response.text == "ok"


def test_httpx_transport_to_closed_loopback_port_is_a_connect_error_and_not_a_violation() -> None:
    enforce(True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
    with httpx.Client(transport=airgap_transport(), timeout=2) as client:
        try:
            client.get(f"http://127.0.0.1:{closed_port}/")
        except AirgapViolation:  # pragma: no cover - the assertion below names the failure
            pytest.fail("loopback was refused by the air-gap guard")
        except httpx.HTTPError:
            pass


def test_plain_httpx_client_is_refused_at_the_socket_layer(
    monkeypatch: pytest.MonkeyPatch, local_server: tuple[str, int]
) -> None:
    base_url, _ = local_server
    enforce(True)
    with httpx.Client(timeout=5) as client:
        with pytest.raises(AirgapViolation):
            client.get("https://example.com")
        assert client.get(base_url + "/").text == "ok"


def test_guard_httpx_client_installs_a_request_hook_on_sync_and_async_clients() -> None:
    enforce(True)
    with httpx.Client() as client:
        guard_httpx_client(client)
        assert len(client.event_hooks["request"]) == 1
        with pytest.raises(AirgapViolation):
            client.get("http://10.0.0.1/")

    async def run_async() -> None:
        async with httpx.AsyncClient() as client:
            guard_httpx_client(client)
            with pytest.raises(AirgapViolation):
                await client.get("https://example.com")
        async with httpx.AsyncClient(transport=airgap_async_transport()) as client:
            with pytest.raises(AirgapViolation):
                await client.get("https://example.com")

    asyncio.run(run_async())


# --------------------------------------------------------------------------------------------------
# urllib


def test_urllib_urlopen_refuses_external_hosts_before_any_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_create = _Recorder()
    monkeypatch.setattr(socket, "create_connection", fake_create)
    enforce(True)
    with pytest.raises(AirgapViolation) as info:
        urllib.request.urlopen("https://example.com", timeout=2)
    assert info.value.via == "urllib"
    with pytest.raises(AirgapViolation):
        urllib.request.urlopen(urllib.request.Request("http://example.com/path"), timeout=2)
    with pytest.raises(AirgapViolation):
        urllib.request.build_opener().open("https://example.com")
    assert fake_create.calls == []


def test_urllib_allows_loopback(local_server: tuple[str, int]) -> None:
    base_url, _ = local_server
    enforce(True)
    with urllib.request.urlopen(base_url + "/", timeout=5) as response:
        assert response.status == 200
        assert response.read() == b"ok"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{closed_port}/", timeout=2)
    except AirgapViolation:  # pragma: no cover - the assertion below names the failure
        pytest.fail("loopback was refused by the air-gap guard")
    except urllib.error.URLError:
        pass


# --------------------------------------------------------------------------------------------------
# asyncio


def test_asyncio_sock_connect_is_guarded(local_server: tuple[str, int]) -> None:
    _, port = local_server
    enforce(True)

    async def run() -> None:
        loop = asyncio.get_running_loop()
        with socket.socket() as sock:
            sock.setblocking(False)
            with pytest.raises(AirgapViolation) as info:
                await loop.sock_connect(sock, ("1.1.1.1", 80))
            assert info.value.via == "asyncio"
        with socket.socket() as sock:
            sock.setblocking(False)
            await loop.sock_connect(sock, ("127.0.0.1", port))

    asyncio.run(run())


# --------------------------------------------------------------------------------------------------
# Switching on and off


def test_enable_and_disable_are_idempotent_and_restore_originals() -> None:
    originals = {
        "connect": socket.socket.connect,
        "connect_ex": socket.socket.connect_ex,
        "sendto": socket.socket.sendto,
        "create_connection": socket.create_connection,
        "opener_open": urllib.request.OpenerDirector.open,
        "selector_sock_connect": asyncio.selector_events.BaseSelectorEventLoop.sock_connect,
        "proactor_sock_connect": asyncio.proactor_events.BaseProactorEventLoop.sock_connect,
    }
    assert "connect" not in vars(socket.socket)

    enforce(True)
    enforce(True)
    assert is_enabled()
    guarded_connect = socket.socket.connect
    assert guarded_connect is not originals["connect"]
    enforce(True, allow_hosts=("127.0.0.1", "localhost", "::1", "llama"))
    assert socket.socket.connect is guarded_connect
    assert "llama" in airgap.allowed_hosts()

    enforce(False)
    enforce(False)
    assert not is_enabled()
    assert socket.socket.connect is originals["connect"]
    assert "connect" not in vars(socket.socket)
    assert socket.socket.connect_ex is originals["connect_ex"]
    assert socket.socket.sendto is originals["sendto"]
    assert socket.create_connection is originals["create_connection"]
    assert urllib.request.OpenerDirector.open is originals["opener_open"]
    assert asyncio.selector_events.BaseSelectorEventLoop.sock_connect is originals["selector_sock_connect"]
    assert asyncio.proactor_events.BaseProactorEventLoop.sock_connect is originals["proactor_sock_connect"]
    assert airgap.allowed_hosts() == ("127.0.0.1", "::1", "localhost")

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
    with socket.socket() as sock, pytest.raises(OSError):
        sock.settimeout(1)
        sock.connect(("127.0.0.1", closed_port))


def test_offline_env_is_set_while_enabled_and_restored_after(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    enforce(True)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "0"
    enforce(False)
    assert "HF_HUB_OFFLINE" not in os.environ
    assert os.environ["TRANSFORMERS_OFFLINE"] == "0"


def test_airgapped_context_manager_restores_the_previous_state() -> None:
    with airgapped():
        assert is_enabled()
        with pytest.raises(AirgapViolation):
            socket.create_connection(("example.com", 443))
    assert not is_enabled()

    enforce(True, allow_hosts=("127.0.0.1", "localhost", "::1", "llama"))
    with airgapped(allow_hosts=("127.0.0.1",)):
        assert airgap.allowed_hosts() == ("127.0.0.1",)
    assert is_enabled()
    assert "llama" in airgap.allowed_hosts()


def test_enforce_from_settings_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEEL_AIRGAP", "1")
    monkeypatch.setenv("KEEL_AIRGAP_ALLOW_HOSTS", "llama, Vector-Store")
    assert enforce_from_settings(Settings()) is True
    assert is_enabled()
    assert is_host_allowed("llama")
    assert is_host_allowed("vector-store")
    assert not is_host_allowed("example.com")

    monkeypatch.setenv("KEEL_AIRGAP", "0")
    assert enforce_from_settings(Settings()) is False
    assert not is_enabled()


def test_enable_logs_one_line(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("INFO", logger="keel.airgap"):
        enforce(True)
        enforce(True)
    lines = [record.getMessage() for record in caplog.records if record.name == "keel.airgap"]
    assert len(lines) == 1
    assert lines[0].startswith("Air-gap mode on")


# --------------------------------------------------------------------------------------------------
# Host policy


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "127.5.5.5",
        "::1",
        "[::1]",
        "::ffff:127.0.0.1",
        "localhost",
        "LOCALHOST",
        "localhost.",
        "",
        None,
    ],
)
def test_loopback_and_allow_list_names_pass(host: str | None) -> None:
    assert is_host_allowed(host)


@pytest.mark.parametrize(
    "host", ["1.1.1.1", "10.0.0.1", "0.0.0.0", "example.com", "llama", "::", "fe80::1%eth0"]
)
def test_other_hosts_are_refused(host: str) -> None:
    assert not is_host_allowed(host)


def test_extra_allow_host_and_its_resolved_addresses_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []

    def fake_getaddrinfo(host: str, port: object, *args: object, **kwargs: object) -> list:
        asked.append(host)
        if host == "llama":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("172.18.0.3", 0))]
        raise socket.gaierror("stand-in resolver knows llama only")

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    hosts = ("127.0.0.1", "localhost", "::1", "llama")
    assert is_host_allowed("llama", hosts)
    assert is_host_allowed("LLAMA", hosts)
    assert is_host_allowed("172.18.0.3", hosts)
    assert not is_host_allowed("172.18.0.4", hosts)
    assert not is_host_allowed("172.18.0.3")
    assert asked == ["llama"]


def test_violation_message_names_the_target_and_the_fix() -> None:
    enforce(True)
    with pytest.raises(AirgapViolation) as info:
        socket.create_connection(("example.com", 443))
    text = str(info.value)
    assert "example.com:443" in text
    assert "127.0.0.1" in text
    assert "KEEL_AIRGAP" in text


# --------------------------------------------------------------------------------------------------
# Deploy files


def test_compose_files_parse_and_declare_airgap() -> None:
    compose = yaml.safe_load(
        (REPO_ROOT / "deploy" / "onprem" / "docker-compose.yml").read_text(encoding="utf-8")
    )
    assert set(compose["services"]) == {"llama", "keel"}
    keel_env = compose["services"]["keel"]["environment"]
    assert keel_env["KEEL_AIRGAP"] == "1"
    assert keel_env["KEEL_AIRGAP_ALLOW_HOSTS"] == "llama"
    assert keel_env["KEEL_LOCAL_LLM_BASE_URL"] == "http://llama:8081/v1"
    assert keel_env["HF_HUB_OFFLINE"] == "1"
    assert "--jinja" in compose["services"]["llama"]["command"]
    assert "--port 8081" in compose["services"]["llama"]["command"]
    assert "keel-data" in compose["volumes"]

    gpu = yaml.safe_load(
        (REPO_ROOT / "deploy" / "onprem" / "docker-compose.gpu.yml").read_text(encoding="utf-8")
    )
    assert "server-cuda" in gpu["services"]["llama"]["image"]

    # The host-model stack brings no model of its own and reaches the runtime already on the machine.
    # The guard stays on there too, with the host bridge as the one destination it permits.
    host_model = yaml.safe_load(
        (REPO_ROOT / "deploy" / "onprem" / "docker-compose.host-model.yml").read_text(encoding="utf-8")
    )
    assert set(host_model["services"]) == {"keel"}, "this stack runs Keel alone"
    host_env = host_model["services"]["keel"]["environment"]
    assert host_env["KEEL_AIRGAP"] == "1"
    assert host_env["KEEL_AIRGAP_ALLOW_HOSTS"] == "host.docker.internal"
    assert "host.docker.internal" in host_env["KEEL_LOCAL_LLM_BASE_URL"]
    assert host_env["KEEL_BOOTSTRAP_CORPUS"] == "fixtures/corpus.yaml", "it answers a question after up"
    assert "host.docker.internal:host-gateway" in host_model["services"]["keel"]["extra_hosts"], (
        "Linux needs this line for host.docker.internal to resolve at all"
    )


# --------------------------------------------------------------------------------------------------
# Integration: local llama-server and the fastembed cache keep working under the guard


def _llama_server_reachable() -> bool:
    try:
        with urllib.request.urlopen(LLAMA_MODELS_URL, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


@pytest.mark.integration
def test_local_llama_server_stays_reachable_under_airgap() -> None:
    if not _llama_server_reachable():
        pytest.skip("llama-server is not answering on 127.0.0.1:8081")
    with airgapped():
        with httpx.Client(transport=airgap_transport(), timeout=10) as client:
            response = client.get(LLAMA_MODELS_URL)
        assert response.status_code == 200
        assert response.json()["data"]


@pytest.mark.integration
def test_fastembed_loads_from_the_local_cache_under_airgap() -> None:
    cache_dir = Path(os.environ.get("FASTEMBED_CACHE_PATH", Path(tempfile.gettempdir()) / "fastembed_cache"))
    if not (cache_dir / "models--qdrant--bge-small-en-v1.5-onnx-q").exists():
        pytest.skip(f"fastembed cache for bge-small-en-v1.5 is absent at {cache_dir}")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from fastembed import TextEmbedding

    with airgapped():
        model = TextEmbedding("BAAI/bge-small-en-v1.5")
        vectors = list(model.embed(["air-gap smoke test"]))
    assert len(vectors) == 1
    assert len(vectors[0]) == 384
