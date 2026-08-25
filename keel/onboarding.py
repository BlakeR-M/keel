"""First-run discovery and preflight checks: point Keel at what you already run.

Keel brings no model and no cloud account. It runs against whatever you already have, which makes
the first ten minutes the part most likely to go wrong: a server on a port nobody remembers, a model
name that differs by one character from the one on disk, an Azure endpoint with no role assignment
behind it. This module removes the guesswork from that.

`discover()` looks for a chat server already running on this machine. Ollama, LM Studio, llama.cpp
and vLLM all expose the same OpenAI-compatible `/v1/models`, so one probe finds any of them and
reads back the model names they actually serve. `keel setup` offers what it found and writes the
answer to `.env`.

`run_checks()` is the preflight behind `keel doctor`. Every check ends in a sentence naming the fix
rather than a stack trace, because the failures here are configuration rather than bugs.

Nothing in this module reaches past the machine it runs on unless the operator configured a remote
endpoint themselves, and under air-gap mode a probe of a host outside the allow list is refused by
`keel.airgap` like any other connection. That refusal is reported as a finding rather than swallowed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from keel.config import Settings

__all__ = [
    "KNOWN_SERVERS",
    "Check",
    "Discovered",
    "discover",
    "env_lines",
    "merge_env",
    "probe_endpoint",
    "run_checks",
]

#: Chat servers people already run, with the port each one listens on by default. Every entry speaks
#: the OpenAI-compatible API, so the same probe reads the model list from all of them.
KNOWN_SERVERS: tuple[tuple[str, str], ...] = (
    ("Ollama", "http://127.0.0.1:11434/v1"),
    ("LM Studio", "http://127.0.0.1:1234/v1"),
    ("llama.cpp", "http://127.0.0.1:8081/v1"),
    ("llama.cpp", "http://127.0.0.1:8080/v1"),
    ("vLLM", "http://127.0.0.1:8000/v1"),
    ("Text generation web UI", "http://127.0.0.1:5000/v1"),
)

PROBE_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class Discovered:
    """A chat server answering on this machine, with the models it reports."""

    name: str
    base_url: str
    models: tuple[str, ...]

    @property
    def first_model(self) -> str:
        return self.models[0] if self.models else ""


@dataclass(frozen=True)
class Check:
    """One preflight result. `fix` is a sentence the reader can act on."""

    name: str
    ok: bool
    detail: str
    fix: str = ""

    @property
    def mark(self) -> str:
        return "ok" if self.ok else "check"


# ------------------------------------------------------------------------------------ discovery


def probe_endpoint(
    base_url: str, api_key: str = "local", timeout: float = PROBE_TIMEOUT_SECONDS
) -> tuple[bool, tuple[str, ...], str]:
    """Ask an OpenAI-compatible endpoint for its model list.

    Returns (reachable, model ids, detail). A refusal from the air-gap guard, a closed port and an
    HTTP error are all reported through `detail` rather than raised, because every one of them is
    something the caller wants to show the reader rather than crash on.
    """
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = httpx.get(url, headers=headers, timeout=timeout)
    except Exception as error:  # noqa: BLE001 (every failure here is a finding, including AirgapViolation)
        return False, (), f"{type(error).__name__}: {str(error).splitlines()[0][:160]}"
    if response.status_code >= 400:
        return False, (), f"HTTP {response.status_code} from {url}"
    try:
        payload: Any = response.json()
    except ValueError:
        return False, (), f"{url} answered with something other than JSON"
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return True, (), f"{url} answered, with no model list in the response"
    models = tuple(str(row.get("id")) for row in rows if isinstance(row, dict) and row.get("id"))
    return True, models, f"{len(models)} model(s) at {base_url}"


def discover(timeout: float = PROBE_TIMEOUT_SECONDS) -> list[Discovered]:
    """Every chat server answering on this machine, in the order `KNOWN_SERVERS` lists them.

    A port that answers with no models still counts as found: the server is up and the reader may
    want to know that before they go looking for it elsewhere.
    """
    found: list[Discovered] = []
    seen: set[str] = set()
    for name, base_url in KNOWN_SERVERS:
        if base_url in seen:
            continue
        reachable, models, _ = probe_endpoint(base_url, timeout=timeout)
        if reachable:
            seen.add(base_url)
            found.append(Discovered(name=name, base_url=base_url, models=models))
    return found


# ------------------------------------------------------------------------------------ .env writing


def env_lines(values: dict[str, str]) -> list[str]:
    """`KEY=value` lines, sorted, for the settings a first run needs to record."""
    return [f"{key}={value}" for key, value in sorted(values.items())]


def merge_env(path: Path, values: dict[str, str]) -> str:
    """Write `values` into the `.env` at `path`, keeping every other line as the operator wrote it.

    A key already present is rewritten in place, so a second `keel setup` updates the file rather
    than appending a second copy that the first one shadows. Comments and blank lines survive.
    """
    existing: list[str] = []
    if path.is_file():
        existing = path.read_text(encoding="utf-8").splitlines()
    remaining = dict(values)
    out: list[str] = []
    for line in existing:
        key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else ""
        if key and key in remaining:
            out.append(f"{key}={remaining.pop(key)}")
        else:
            out.append(line)
    if remaining:
        if out and out[-1].strip():
            out.append("")
        out.append("# Written by `keel setup`")
        out.extend(env_lines(remaining))
    text = "\n".join(out).rstrip("\n") + "\n"
    path.write_text(text, encoding="utf-8")
    return text


# ------------------------------------------------------------------------------------ preflight


def _check_store(settings: Settings) -> Check:
    data_dir = Path(settings.data_dir)
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        probe = data_dir / ".keel-write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as error:
        return Check(
            "data directory",
            False,
            f"{data_dir} is not writable: {type(error).__name__}",
            "Point KEEL_DATA_DIR at a directory this user owns, or create it and grant write access.",
        )
    return Check("data directory", True, f"{data_dir} is writable")


def _check_local_model(settings: Settings) -> list[Check]:
    base_url = settings.local_llm_base_url
    wanted = settings.local_llm_model
    reachable, models, detail = probe_endpoint(base_url, settings.local_llm_api_key)
    if not reachable:
        found = discover()
        if found:
            names = ", ".join(f"{d.name} at {d.base_url}" for d in found)
            fix = f"Something is answering elsewhere on this machine: {names}. Run `keel setup` to point at it."
        else:
            fix = (
                "Start a chat server and point KEEL_LOCAL_LLM_BASE_URL at it. Ollama, LM Studio, "
                "llama.cpp and vLLM all work. Then run `keel setup`."
            )
        return [Check("model endpoint", False, f"{base_url} is out of reach. {detail}", fix)]

    checks = [Check("model endpoint", True, detail)]
    if models and wanted not in models:
        listed = ", ".join(models[:6]) + (" ..." if len(models) > 6 else "")
        checks.append(
            Check(
                "model name",
                False,
                f"KEEL_LOCAL_LLM_MODEL is {wanted!r}, and that endpoint serves: {listed}",
                "Set KEEL_LOCAL_LLM_MODEL to one of those, or run `keel setup` to choose one.",
            )
        )
    elif models:
        checks.append(Check("model name", True, f"{wanted} is served by that endpoint"))
    else:
        checks.append(
            Check(
                "model name",
                True,
                f"{base_url} lists no models, so {wanted!r} is taken on trust",
                "Ask a question to confirm it answers.",
            )
        )
    return checks


def _check_azure(settings: Settings) -> list[Check]:
    checks: list[Check] = []
    for label, value, env in (
        ("Azure OpenAI endpoint", settings.azure_openai_endpoint, "KEEL_AZURE_OPENAI_ENDPOINT"),
        ("Azure AI Search endpoint", settings.azure_search_endpoint, "KEEL_AZURE_SEARCH_ENDPOINT"),
    ):
        if value:
            checks.append(Check(label, True, value))
        else:
            checks.append(
                Check(
                    label,
                    False,
                    f"{env} is empty",
                    f"Set {env} to the resource in your own subscription. "
                    "`deploy/azure/deploy.ps1` creates both and prints them.",
                )
            )
    try:
        from azure.identity import DefaultAzureCredential
    except ImportError:
        checks.append(
            Check(
                "Azure credential",
                False,
                "azure-identity is absent",
                'Install the cloud extra: pip install -e ".[azure]"',
            )
        )
        return checks
    try:
        DefaultAzureCredential().get_token("https://cognitiveservices.azure.com/.default")
    except Exception as error:  # noqa: BLE001 (a credential failure is a finding to report)
        checks.append(
            Check(
                "Azure credential",
                False,
                f"DefaultAzureCredential returned nothing: {type(error).__name__}",
                "Run `az login`, or attach a managed identity to the host running Keel.",
            )
        )
    else:
        checks.append(Check("Azure credential", True, "DefaultAzureCredential resolved a token"))
    return checks


def _check_airgap(settings: Settings) -> list[Check]:
    from urllib.parse import urlsplit

    from keel.airgap import ALLOW_HOSTS_ENV, DEFAULT_ALLOW_HOSTS, is_host_allowed

    if not settings.airgap:
        return [Check("air-gap", True, "off, so outbound connections are unrestricted")]
    extra = [h for h in os.environ.get(ALLOW_HOSTS_ENV, "").replace(";", ",").split(",") if h.strip()]
    allowed = (*DEFAULT_ALLOW_HOSTS, *[h.strip() for h in extra])
    checks = [Check("air-gap", True, f"on, allowing {', '.join(allowed)}")]
    host = urlsplit(settings.local_llm_base_url).hostname or ""
    if settings.profile == "local" and host and not is_host_allowed(host, allowed):
        checks.append(
            Check(
                "air-gap allow list",
                False,
                f"the model host {host} sits outside the allow list, so every model call is refused",
                f"Add it: {ALLOW_HOSTS_ENV}={host}. Loopback is always allowed.",
            )
        )
    return checks


def _check_corpus(settings: Settings) -> Check:
    store = Path(settings.data_dir) / "keel.db"
    if not store.is_file():
        return Check(
            "corpus",
            False,
            "no store yet",
            "Load the fixture corpus: keel ingest --manifest fixtures/corpus.yaml",
        )
    import sqlite3

    try:
        with sqlite3.connect(store) as conn:
            documents = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
    except sqlite3.Error as error:
        return Check("corpus", False, f"the store is unreadable: {error}", "Delete it and ingest again.")
    if documents == 0:
        return Check(
            "corpus",
            False,
            "the store holds no documents",
            "Load the fixture corpus: keel ingest --manifest fixtures/corpus.yaml",
        )
    return Check("corpus", True, f"{documents} document(s) in the store")


def run_checks(settings: Settings | None = None) -> list[Check]:
    """Everything a first run needs to be right, each failure carrying its fix.

    Ordered the way a reader debugs: the store first, then the air gap that would refuse the model
    call, then the model itself, then whether there is anything to ask about.
    """
    current = settings if settings is not None else Settings()
    checks: list[Check] = [_check_store(current)]
    checks.extend(_check_airgap(current))
    if current.profile == "azure":
        checks.extend(_check_azure(current))
    else:
        checks.extend(_check_local_model(current))
    checks.append(_check_corpus(current))
    return checks
