"""Air-gap egress guard.

`enforce(True)` installs process-wide guards so every outbound connection to a host outside the
allow list is refused with `AirgapViolation` before a packet leaves the box:

- socket layer: `socket.socket.connect`, `socket.socket.connect_ex`, `socket.socket.sendto` and
  `socket.create_connection` check the destination host. Hostnames are checked as written, before
  any DNS lookup.
- name resolution: `socket.getaddrinfo`, `gethostbyname`, `gethostbyname_ex` and `gethostbyaddr`
  refuse names outside the allow list, so no DNS packet leaves the box for a host that could never
  be connected to (a lookup of `<data>.attacker.example` is an exfiltration channel of its own).
  IP literals, allow-listed names, an empty host and passive (bind) lookups pass; asyncio's resolver
  runs `socket.getaddrinfo` in its executor and is covered by the same guard.
- asyncio: `loop.sock_connect` on both event loop families (selector on Linux, proactor on Windows,
  where the proactor connects through ConnectEx and skips `socket.connect`).
- httpx: `airgap_transport()` wraps a transport and `guard_httpx_client(client)` installs a request
  hook; both refuse before sending.
- urllib: `urllib.request.OpenerDirector.open` checks the URL host before any handler runs.

Loopback addresses (127.0.0.0/8, ::1 and the IPv4-mapped form) and the names in `allow_hosts`
(default 127.0.0.1, localhost, ::1) pass. A name in the allow list also permits the addresses it
resolves to, so a compose sibling such as `llama` works for clients that resolve first and connect
second. Unix-domain sockets, `file:` and `data:` URLs carry no network host and pass.

Enabling also sets HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 when unset, so fastembed and
huggingface_hub read the local model cache and skip their metadata requests.

`enforce(False)` restores every original. Both directions are idempotent. `airgapped()` is a
context manager for tests. The app entry points call `enforce_from_settings()` once at startup; it
reads `Settings.airgap` (env KEEL_AIRGAP) and the optional env KEEL_AIRGAP_ALLOW_HOSTS (comma
separated extra hosts, for example `llama` in the compose stack).
"""

from __future__ import annotations

import asyncio.proactor_events
import asyncio.selector_events
import functools
import ipaddress
import logging
import os
import socket
import threading
import time
import urllib.request
from collections.abc import Callable, Coroutine, Iterable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar
from urllib.parse import urlsplit

import httpx

from keel.config import Settings, get_settings

__all__ = [
    "DEFAULT_ALLOW_HOSTS",
    "AirgapViolation",
    "AirgapTransport",
    "AsyncAirgapTransport",
    "airgap_async_request_hook",
    "airgap_async_transport",
    "airgap_request_hook",
    "airgap_transport",
    "airgapped",
    "allowed_hosts",
    "enforce",
    "enforce_from_settings",
    "guard_httpx_client",
    "is_enabled",
    "is_host_allowed",
]

log = logging.getLogger("keel.airgap")

DEFAULT_ALLOW_HOSTS: tuple[str, ...] = ("127.0.0.1", "localhost", "::1")
ALLOW_HOSTS_ENV = "KEEL_AIRGAP_ALLOW_HOSTS"

_OFFLINE_ENV: tuple[str, ...] = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
_RESOLVE_TTL_SECONDS = 30.0
_IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
_HttpxClient = TypeVar("_HttpxClient", httpx.Client, httpx.AsyncClient)


class AirgapViolation(RuntimeError):
    """Raised when air-gap mode refuses an outbound connection to a host outside the allow list."""

    def __init__(
        self, host: str, port: int | None = None, via: str = "socket", allowed: Iterable[str] = ()
    ) -> None:
        self.host = host
        self.port = port
        self.via = via
        target = host if port is None else f"{host}:{port}"
        allowed_text = ", ".join(sorted(allowed)) or ", ".join(DEFAULT_ALLOW_HOSTS)
        action = f"name lookup of {target}" if via == "dns" else f"{via} connection to {target}"
        super().__init__(
            f"Air-gap mode refused the {action}. Allowed hosts: {allowed_text}. "
            f"Set KEEL_AIRGAP=0 to reach other hosts, or list the host in {ALLOW_HOSTS_ENV}."
        )


class _State:
    """Process-wide guard state: the allow list, saved originals and the resolution cache."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.enabled = False
        self.allow_hosts: frozenset[str] = _normalise_hosts(DEFAULT_ALLOW_HOSTS)
        self.originals: list[tuple[Any, str, bool, Any]] = []
        self.env_set: list[str] = []
        self.resolved: dict[str, tuple[float, frozenset[_IPAddress]]] = {}
        self.getaddrinfo: Callable[..., Any] = socket.getaddrinfo  # the resolver behind the guard


def _normalise_host(host: str) -> str:
    text = host.strip().lower()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return text.rstrip(".")


def _normalise_hosts(hosts: Iterable[str]) -> frozenset[str]:
    return frozenset(_normalise_host(h) for h in hosts if h and h.strip())


_state = _State()


# --------------------------------------------------------------------------------------------------
# Host policy


def _as_ip(host: str) -> _IPAddress | None:
    candidate = host.split("%", 1)[0]
    try:
        return ipaddress.ip_address(candidate)
    except ValueError:
        return None


def _allow_set(allow_hosts: Iterable[str] | None) -> frozenset[str]:
    if allow_hosts is None:
        return _state.allow_hosts
    return _normalise_hosts(allow_hosts)


def _resolved_allowed_ips(allowed: frozenset[str]) -> set[_IPAddress]:
    """Addresses that the named entries of the allow list resolve to, cached for a short while."""
    ips: set[_IPAddress] = set()
    now = time.monotonic()
    # With the guard on, socket.getaddrinfo is the guard itself; resolve through the saved original.
    resolver = _state.getaddrinfo if _state.enabled else socket.getaddrinfo
    for name in allowed:
        if name == "localhost" or _as_ip(name) is not None:
            continue
        entry = _state.resolved.get(name)
        if entry is None or entry[0] < now:
            found: set[_IPAddress] = set()
            try:
                for info in resolver(name, None):
                    ip = _as_ip(str(info[4][0]))
                    if ip is not None:
                        found.add(ip)
            except OSError:
                pass
            entry = (now + _RESOLVE_TTL_SECONDS, frozenset(found))
            _state.resolved[name] = entry
        ips |= entry[1]
    return ips


def is_host_allowed(host: str | None, allow_hosts: Iterable[str] | None = None) -> bool:
    """Return True when a connection to `host` may proceed.

    `allow_hosts` defaults to the process allow list (the defaults when the guard is off). An empty
    host means a unix-domain socket or a URL scheme with no network host, which passes.
    """
    if host is None or host == "":
        return True
    allowed = _allow_set(allow_hosts)
    text = _normalise_host(host)
    if text in allowed:
        return True
    ip = _as_ip(text)
    if ip is None:
        return False
    if ip.is_loopback:
        return True
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None and mapped.is_loopback:
        return True
    return ip in _resolved_allowed_ips(allowed)


def _refuse_unless_allowed(
    host: str | None, port: int | None, via: str, allow_hosts: Iterable[str] | None = None
) -> None:
    if not is_host_allowed(host, allow_hosts):
        raise AirgapViolation(str(host), port, via, _allow_set(allow_hosts))


def _address_host_port(sock: Any, address: Any) -> tuple[str | None, int | None]:
    """Extract (host, port) from a socket address; (None, None) for unix-domain forms."""
    af_unix = getattr(socket, "AF_UNIX", None)
    if af_unix is not None and getattr(sock, "family", None) == af_unix:
        return None, None
    if isinstance(address, str | bytes | bytearray | os.PathLike):
        return None, None
    if isinstance(address, tuple) and len(address) >= 2:
        raw_host = address[0]
        if isinstance(raw_host, bytes | bytearray):
            raw_host = bytes(raw_host).decode("ascii", "replace")
        if isinstance(raw_host, str):
            port = address[1] if isinstance(address[1], int) else None
            return raw_host, port
    return "<unrecognised address form>", None


# --------------------------------------------------------------------------------------------------
# Patching


def _patch(owner: Any, name: str, replacement: Callable[..., Any]) -> None:
    was_in_dict = name in vars(owner)
    _state.originals.append((owner, name, was_in_dict, getattr(owner, name)))
    setattr(owner, name, replacement)


def _restore_all() -> None:
    for owner, name, was_in_dict, value in reversed(_state.originals):
        if was_in_dict:
            setattr(owner, name, value)
        else:
            try:
                delattr(owner, name)
            except AttributeError:
                pass
    _state.originals.clear()


def _install_socket_guards() -> None:
    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex
    orig_sendto = socket.socket.sendto
    orig_create_connection = socket.create_connection

    @functools.wraps(orig_connect)
    def connect(self: socket.socket, address: Any) -> Any:
        _refuse_unless_allowed(*_address_host_port(self, address), via="socket")
        return orig_connect(self, address)

    @functools.wraps(orig_connect_ex)
    def connect_ex(self: socket.socket, address: Any) -> Any:
        _refuse_unless_allowed(*_address_host_port(self, address), via="socket")
        return orig_connect_ex(self, address)

    @functools.wraps(orig_sendto)
    def sendto(self: socket.socket, *args: Any) -> Any:
        if len(args) >= 2:
            _refuse_unless_allowed(*_address_host_port(self, args[-1]), via="socket")
        return orig_sendto(self, *args)

    @functools.wraps(orig_create_connection)
    def create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
        _refuse_unless_allowed(*_address_host_port(None, address), via="socket")
        return orig_create_connection(address, *args, **kwargs)

    _patch(socket.socket, "connect", connect)
    _patch(socket.socket, "connect_ex", connect_ex)
    _patch(socket.socket, "sendto", sendto)
    _patch(socket, "create_connection", create_connection)


def _refuse_lookup_unless_allowed(host: Any, port: int | None = None) -> None:
    """Refuse a name-resolution call for a host outside the allow list. An absent host (a wildcard
    lookup) and an IP literal (no DNS involved; the connect guard judges it) pass."""
    if host is None:
        return
    name = bytes(host).decode("ascii", "replace") if isinstance(host, bytes | bytearray) else str(host)
    if not name or _as_ip(_normalise_host(name)) is not None:
        return
    _refuse_unless_allowed(name, port, via="dns")


def _install_resolver_guards() -> None:
    orig_getaddrinfo = socket.getaddrinfo
    orig_gethostbyname = socket.gethostbyname
    orig_gethostbyname_ex = socket.gethostbyname_ex
    orig_gethostbyaddr = socket.gethostbyaddr
    passive = getattr(socket, "AI_PASSIVE", 0)

    @functools.wraps(orig_getaddrinfo)
    def getaddrinfo(
        host: Any, port: Any, family: int = 0, type: int = 0, proto: int = 0, flags: int = 0
    ) -> Any:
        if not (flags & passive):  # a passive lookup binds a listener; nothing leaves the box
            _refuse_lookup_unless_allowed(host, port if isinstance(port, int) else None)
        return orig_getaddrinfo(host, port, family, type, proto, flags)

    @functools.wraps(orig_gethostbyname)
    def gethostbyname(host: Any) -> Any:
        _refuse_lookup_unless_allowed(host)
        return orig_gethostbyname(host)

    @functools.wraps(orig_gethostbyname_ex)
    def gethostbyname_ex(host: Any) -> Any:
        _refuse_lookup_unless_allowed(host)
        return orig_gethostbyname_ex(host)

    @functools.wraps(orig_gethostbyaddr)
    def gethostbyaddr(address: Any) -> Any:
        # A reverse lookup of an address outside the allow list is a DNS query about that address.
        name = (
            bytes(address).decode("ascii", "replace")
            if isinstance(address, bytes | bytearray)
            else str(address)
        )
        _refuse_unless_allowed(name, None, via="dns")
        return orig_gethostbyaddr(address)

    _state.getaddrinfo = orig_getaddrinfo
    _patch(socket, "getaddrinfo", getaddrinfo)
    _patch(socket, "gethostbyname", gethostbyname)
    _patch(socket, "gethostbyname_ex", gethostbyname_ex)
    _patch(socket, "gethostbyaddr", gethostbyaddr)


def _guarded_sock_connect(
    original: Callable[..., Coroutine[Any, Any, Any]],
) -> Callable[..., Coroutine[Any, Any, Any]]:
    @functools.wraps(original)
    async def sock_connect(self: Any, sock: Any, address: Any) -> Any:
        _refuse_unless_allowed(*_address_host_port(sock, address), via="asyncio")
        return await original(self, sock, address)

    return sock_connect


def _install_asyncio_guards() -> None:
    loops = (
        asyncio.selector_events.BaseSelectorEventLoop,
        asyncio.proactor_events.BaseProactorEventLoop,
    )
    for loop_cls in loops:
        _patch(loop_cls, "sock_connect", _guarded_sock_connect(loop_cls.sock_connect))


def _url_host_port(url: Any) -> tuple[str | None, int | None]:
    if isinstance(url, urllib.request.Request):
        url = url.full_url
    if not isinstance(url, str):
        return None, None
    parts = urlsplit(url)
    try:
        port = parts.port
    except ValueError:
        port = None
    return parts.hostname, port


def _install_urllib_guard() -> None:
    orig_open = urllib.request.OpenerDirector.open

    @functools.wraps(orig_open)
    def open_(self: urllib.request.OpenerDirector, fullurl: Any, *args: Any, **kwargs: Any) -> Any:
        _refuse_unless_allowed(*_url_host_port(fullurl), via="urllib")
        return orig_open(self, fullurl, *args, **kwargs)

    _patch(urllib.request.OpenerDirector, "open", open_)


def _set_offline_env() -> None:
    for key in _OFFLINE_ENV:
        if key not in os.environ:
            os.environ[key] = "1"
            _state.env_set.append(key)


def _restore_env() -> None:
    for key in _state.env_set:
        os.environ.pop(key, None)
    _state.env_set.clear()


# --------------------------------------------------------------------------------------------------
# Public switches


def enforce(enabled: bool, allow_hosts: Iterable[str] = DEFAULT_ALLOW_HOSTS) -> None:
    """Switch the process-wide egress guard on or off.

    Repeated calls are idempotent: enabling twice keeps one set of guards (a changed allow list is
    applied in place), disabling when off is a no-op. Enabling logs one line naming the allow list.
    """
    hosts = _normalise_hosts(allow_hosts)
    with _state.lock:
        if enabled:
            if _state.enabled:
                if hosts != _state.allow_hosts:
                    _state.allow_hosts = hosts
                    _state.resolved.clear()
                    log.info("Air-gap allow list updated: %s", ", ".join(sorted(hosts)))
                return
            _state.allow_hosts = hosts
            _state.resolved.clear()
            _install_socket_guards()
            _install_resolver_guards()
            _install_asyncio_guards()
            _install_urllib_guard()
            _set_offline_env()
            _state.enabled = True
            log.info("Air-gap mode on: outbound connections go to %s only.", ", ".join(sorted(hosts)))
            return
        _state.resolved.clear()
        if not _state.enabled:
            return
        _restore_all()
        _restore_env()
        _state.getaddrinfo = socket.getaddrinfo
        _state.allow_hosts = _normalise_hosts(DEFAULT_ALLOW_HOSTS)
        _state.enabled = False
        log.debug("Air-gap mode off: original network functions restored.")


def is_enabled() -> bool:
    """Return True while the process-wide guard is installed."""
    return _state.enabled


def allowed_hosts() -> tuple[str, ...]:
    """The current allow list, sorted."""
    return tuple(sorted(_state.allow_hosts))


def _split_hosts(text: str) -> list[str]:
    return [part for part in text.replace(";", ",").replace(" ", ",").split(",") if part]


def enforce_from_settings(settings: Settings | None = None) -> bool:
    """Apply `Settings.airgap` (env KEEL_AIRGAP) to the process and return the resulting state.

    Extra hosts come from env KEEL_AIRGAP_ALLOW_HOSTS (comma separated) on top of the defaults.
    Call this once at app startup (web app, CLI).
    """
    current = settings if settings is not None else get_settings()
    extra = _split_hosts(os.environ.get(ALLOW_HOSTS_ENV, ""))
    enforce(bool(current.airgap), (*DEFAULT_ALLOW_HOSTS, *extra))
    return is_enabled()


@contextmanager
def airgapped(allow_hosts: Iterable[str] = DEFAULT_ALLOW_HOSTS) -> Iterator[None]:
    """Enable the guard for the duration of a block, then put back the previous state. For tests."""
    with _state.lock:
        previous_enabled = _state.enabled
        previous_hosts = tuple(_state.allow_hosts)
    enforce(True, allow_hosts)
    try:
        yield
    finally:
        if previous_enabled:
            enforce(True, previous_hosts)
        else:
            enforce(False)


# --------------------------------------------------------------------------------------------------
# httpx helpers


def _check_httpx_request(request: httpx.Request, allow_hosts: Iterable[str] | None) -> None:
    _refuse_unless_allowed(request.url.host or None, request.url.port, via="httpx", allow_hosts=allow_hosts)


class AirgapTransport(httpx.BaseTransport):
    """Sync httpx transport that refuses hosts outside the allow list before the inner transport sends.

    The guard applies whenever this transport is used, with `allow_hosts` or the process allow list.
    """

    def __init__(
        self, inner: httpx.BaseTransport | None = None, allow_hosts: Iterable[str] | None = None
    ) -> None:
        self._inner = inner if inner is not None else httpx.HTTPTransport()
        self._allow_hosts = None if allow_hosts is None else tuple(allow_hosts)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        _check_httpx_request(request, self._allow_hosts)
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()


class AsyncAirgapTransport(httpx.AsyncBaseTransport):
    """Async httpx transport that refuses hosts outside the allow list before the inner transport sends."""

    def __init__(
        self, inner: httpx.AsyncBaseTransport | None = None, allow_hosts: Iterable[str] | None = None
    ) -> None:
        self._inner = inner if inner is not None else httpx.AsyncHTTPTransport()
        self._allow_hosts = None if allow_hosts is None else tuple(allow_hosts)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        _check_httpx_request(request, self._allow_hosts)
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


def airgap_transport(
    inner: httpx.BaseTransport | None = None, allow_hosts: Iterable[str] | None = None
) -> AirgapTransport:
    """Build a sync transport for `httpx.Client(transport=...)` that refuses non-allowed hosts before sending."""
    return AirgapTransport(inner, allow_hosts)


def airgap_async_transport(
    inner: httpx.AsyncBaseTransport | None = None, allow_hosts: Iterable[str] | None = None
) -> AsyncAirgapTransport:
    """Build an async transport for `httpx.AsyncClient(transport=...)` that refuses non-allowed hosts."""
    return AsyncAirgapTransport(inner, allow_hosts)


def airgap_request_hook(allow_hosts: Iterable[str] | None = None) -> Callable[[httpx.Request], None]:
    """Build a request event hook for `httpx.Client(event_hooks={"request": [hook]})`."""
    hosts = None if allow_hosts is None else tuple(allow_hosts)

    def hook(request: httpx.Request) -> None:
        _check_httpx_request(request, hosts)

    return hook


def airgap_async_request_hook(
    allow_hosts: Iterable[str] | None = None,
) -> Callable[[httpx.Request], Coroutine[Any, Any, None]]:
    """Build a request event hook for `httpx.AsyncClient(event_hooks={"request": [hook]})`."""
    hosts = None if allow_hosts is None else tuple(allow_hosts)

    async def hook(request: httpx.Request) -> None:
        _check_httpx_request(request, hosts)

    return hook


def guard_httpx_client(client: _HttpxClient, allow_hosts: Iterable[str] | None = None) -> _HttpxClient:
    """Install the air-gap request hook on an existing httpx client, ahead of its other hooks.

    Returns the same client. Works for `httpx.Client` and `httpx.AsyncClient`, including the clients
    passed to the OpenAI SDK as `http_client=`.
    """
    hook: Callable[..., Any]
    if isinstance(client, httpx.AsyncClient):
        hook = airgap_async_request_hook(allow_hosts)
    else:
        hook = airgap_request_hook(allow_hosts)
    hooks = client.event_hooks
    hooks["request"] = [hook, *hooks.get("request", [])]
    client.event_hooks = hooks
    return client
