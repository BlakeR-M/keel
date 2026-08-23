"""A live air-gap demonstration: attempt to reach a host, at every layer, with the guard installed.

Air-gap mode is the control that is hardest to believe from a description and easiest to believe
from a refusal you asked for yourself. This module attempts a real outbound connection to a host a
visitor names, through each of the five layers `keel.airgap` guards, and reports what stopped it.

The attempts run in a **child process** started with `KEEL_AIRGAP=1`, for two reasons. The guard is
process-wide, so installing it in a web worker would take the model connection down for whoever else
is mid-question. And a child that enforces the guard from its first line is the honest version of
the claim: nothing in it has ever been able to reach the host.

Nothing here can become an outbound request gadget. A host outside the allow list cannot be reached,
which is the property being demonstrated. A host inside the allow list is reported from the policy
alone and never connected to, so loopback stays out of reach as well.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any

__all__ = [
    "LAYERS",
    "MAX_HOST_LENGTH",
    "PROBE_PORT",
    "PROBE_TIMEOUT_SECONDS",
    "Attempt",
    "clean_host",
    "probe",
    "run",
]

PROBE_PORT = 443
PROBE_TIMEOUT_SECONDS = 20.0
MAX_HOST_LENGTH = 253

#: Each layer, in the order a real request would reach it, with the line that explains why it is
#: guarded separately from the one before it.
LAYERS: tuple[tuple[str, str], ...] = (
    (
        "dns",
        "Name resolution. A lookup of a host that could never be connected to is an exfiltration "
        "channel of its own, so the name is refused before a DNS packet leaves.",
    ),
    (
        "socket",
        "The socket layer. socket.connect, connect_ex, sendto and create_connection check the "
        "destination, and a hostname is checked as written, ahead of any lookup.",
    ),
    (
        "asyncio",
        "The event loop. sock_connect is guarded on the selector and proactor loops, because the "
        "proactor connects through ConnectEx and never calls socket.connect.",
    ),
    (
        "urllib",
        "urllib.request. OpenerDirector.open reads the URL host before any handler runs.",
    ),
    (
        "httpx",
        "httpx. A guarded transport and a request hook both refuse ahead of sending, which covers "
        "the client handed to the OpenAI SDK.",
    ),
)

# A hostname label or an IP literal. Deliberately narrow: this string is passed to a resolver and
# shown back on a page, so anything with whitespace, a scheme, a path or a credential is rejected
# rather than cleaned up into something the visitor did not type.
HOSTNAME = re.compile(r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)(?:\.(?!-)[A-Za-z0-9-]{1,63}(?<!-))*\.?$")
IPV6 = re.compile(r"^[0-9A-Fa-f:]{2,45}$")


@dataclass(frozen=True)
class Attempt:
    """What one layer did when it was pointed at the host.

    `layer` is the call that was made and `via` is the guard that answered it. They usually match.
    Where they differ the outer guard reached the host name first, which is the defence in depth
    working rather than a mismatch: a name that resolves nowhere reachable is refused as a name.
    """

    layer: str
    note: str
    refused: bool
    via: str
    detail: str
    outcome: str = "refused"
    """`refused`, `no-lookup` (an address rather than a name, so the connect guards judge it),
    `completed` (the attempt went through) or `error` (something other than the guard stopped it)."""


def clean_host(raw: str) -> str | None:
    """Normalise what a visitor typed into a bare host, or None when it is not one.

    A URL, an email address, a port, a path or anything with whitespace is rejected rather than
    trimmed into a different host than the one on screen.
    """
    text = (raw or "").strip()
    if not text or len(text) > MAX_HOST_LENGTH:
        return None
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if any(character.isspace() or ord(character) < 0x20 for character in text):
        return None
    if HOSTNAME.match(text) or IPV6.match(text):
        return text.rstrip(".").lower()
    return None


# ------------------------------------------------------------------------------------ the child


def _attempts(host: str) -> list[Attempt]:
    """Run one real attempt per layer. Only called with the guard installed."""
    import asyncio
    import socket
    import urllib.request

    import httpx

    from keel.airgap import AirgapViolation, airgap_transport

    url = f"https://{host}/"

    def dns() -> None:
        socket.getaddrinfo(host, PROBE_PORT)

    def sock() -> None:
        socket.create_connection((host, PROBE_PORT), timeout=2).close()

    def loop() -> None:
        # sock_connect rather than open_connection: it is the method the guard patches, and it
        # takes the address without a lookup, so this exercises the event loop guard itself.
        async def connect() -> None:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.setblocking(False)
            try:
                await asyncio.get_running_loop().sock_connect(client, (host, PROBE_PORT))
            finally:
                client.close()

        asyncio.run(connect())

    def urllib_open() -> None:
        with urllib.request.urlopen(url, timeout=2) as response:  # noqa: S310 (guarded above)
            response.read(1)

    def httpx_send() -> None:
        with httpx.Client(transport=airgap_transport(), timeout=2) as client:
            client.get(url)

    calls = {"dns": dns, "socket": sock, "asyncio": loop, "urllib": urllib_open, "httpx": httpx_send}
    literal = _is_ip_literal(host)
    results: list[Attempt] = []
    for layer, note in LAYERS:
        if layer == "dns" and literal:
            # An address needs no lookup, so no DNS packet was ever going to leave. Say that,
            # rather than counting a layer that had nothing to refuse as a layer that let it past.
            results.append(
                Attempt(
                    layer,
                    note,
                    False,
                    "none",
                    f"{host} is an address rather than a name, so there is nothing to look up. "
                    "The connect guards below judge it instead.",
                    outcome="no-lookup",
                )
            )
            continue
        try:
            calls[layer]()
        except AirgapViolation as violation:
            results.append(Attempt(layer, note, True, violation.via, str(violation)))
        except Exception as error:  # noqa: BLE001 (reported rather than raised: this is a report)
            # The guard should have refused first. Say what happened instead of claiming a refusal.
            detail = f"{type(error).__name__}: {error}".strip().splitlines()[0]
            results.append(Attempt(layer, note, False, "none", detail, outcome="error"))
        else:
            results.append(
                Attempt(
                    layer,
                    note,
                    False,
                    "none",
                    "the attempt completed without a refusal",
                    outcome="completed",
                )
            )
    return results


def _is_ip_literal(host: str) -> bool:
    import ipaddress

    try:
        ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False
    return True


def probe(host: str) -> dict[str, Any]:
    """Point every guarded layer at `host` and report what each one did.

    Call this only inside a process where the guard is installed (`enforce(True)` or `airgapped()`).
    An allow-listed host is answered from the policy and never connected to.
    """
    from keel.airgap import allowed_hosts, is_enabled, is_host_allowed

    body: dict[str, Any] = {
        "host": host,
        "guard": is_enabled(),
        "allow_hosts": list(allowed_hosts()),
        "allowed": is_host_allowed(host),
    }
    if body["allowed"]:
        body["attempts"] = []
        body["summary"] = (
            f"{host} is on the allow list, so the guard permits it. Keel reports that from the "
            "policy and makes no connection."
        )
        return body
    attempts = _attempts(host)
    body["attempts"] = [asdict(attempt) for attempt in attempts]
    refused = sum(1 for attempt in attempts if attempt.refused)
    tried = [attempt for attempt in attempts if attempt.outcome != "no-lookup"]
    body["refused"] = refused
    body["layers"] = len(tried)
    if refused == len(tried) and len(tried) == len(attempts):
        summary = f"Every one of the {refused} guarded layers refused {host} before a packet left the process."
    elif refused == len(tried):
        summary = (
            f"All {refused} layers that apply refused {host} before a packet left the process. "
            f"{host} is an address rather than a name, so name resolution had nothing to look up."
        )
    else:
        summary = f"{refused} of {len(tried)} layers refused {host}. Read each line below."
    body["summary"] = summary
    return body


def main(argv: list[str] | None = None) -> int:
    """`python -m keel.web.airgap_probe <host>`: install the guard, probe, print JSON.

    The guard is installed through `enforce_from_settings`, so the allow list the report names is the
    one this deployment actually runs with rather than a demonstration default. The parent forces
    `KEEL_AIRGAP=1` in the child's environment, and `KEEL_AIRGAP_ALLOW_HOSTS` is inherited as set.
    """
    from keel.airgap import enforce_from_settings

    args = sys.argv[1:] if argv is None else argv
    host = clean_host(args[0]) if args else None
    if host is None:
        print(json.dumps({"error": "a bare hostname or IP address is needed"}))
        return 2
    enforce_from_settings()
    print(json.dumps(probe(host)))
    return 0


# ------------------------------------------------------------------------------------ the parent


def run(raw_host: str, timeout: float = PROBE_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Probe `raw_host` in a child process under the guard and return its report.

    The returned mapping carries `error` instead of `attempts` when the host is not a bare host or
    the child fails to report.
    """
    host = clean_host(raw_host)
    if host is None:
        return {
            "error": "That is not a host. Type a bare hostname or IP address, "
            "for example data.attacker.example or 8.8.8.8."
        }
    env = dict(os.environ, KEEL_AIRGAP="1", PYTHONIOENCODING="utf-8")
    try:
        completed = subprocess.run(  # noqa: S603 (fixed argv, no shell, host validated above)
            [sys.executable, "-m", "keel.web.airgap_probe", host],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(_repo_root()),
        )
    except subprocess.TimeoutExpired:
        return {"error": f"The probe of {host} ran past {timeout:.0f} seconds and was stopped."}
    except OSError as error:
        return {"error": f"The probe could not start: {type(error).__name__}."}
    try:
        body = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": f"The probe of {host} returned nothing readable."}
    if not isinstance(body, dict):
        return {"error": f"The probe of {host} returned nothing readable."}
    return body


def _repo_root() -> Any:
    from pathlib import Path

    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    raise SystemExit(main())
