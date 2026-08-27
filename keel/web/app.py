"""The Keel web app: question box with citation chips, source viewer with ACL enforcement, admin page
(recent requests, totals, 14-day trend, quarantine, approvals, ledger verify and export) and a health
probe. `app` is the ASGI entry point (`uvicorn keel.web.app:app`, or `keel serve`).

The AppContext comes from `keel.providers.factory.build_context()`: built once at startup by the
lifespan handler, or on the first request when the server skips lifespan events, and kept on
`app.state.ctx`. Tests set `app.state.ctx` before the first request and the app uses that instead.
Nothing at import time touches the model or the store.

Admin routes are open on loopback. When `settings.host` is any other address they require the
`X-Keel-Admin-Token` header to equal the `KEEL_ADMIN_TOKEN` environment variable.

Identity follows the same line. On loopback the question box's user select and tags field are honoured as
given (the demo of permission filtering). Beyond loopback nothing in a request body or query string can
name a user or a tag: identity comes only from an authenticating reverse proxy that sends `X-Keel-User`
and `X-Keel-Tags` together with `X-Keel-Proxy-Token` equal to `KEEL_PROXY_TOKEN`; every other request
runs as `public`. See `trusted_identity`.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from keel import __version__
from keel.agent.tools import ToolContext
from keel.answer.types import User
from keel.bootstrap import ensure_demo_corpus
from keel.db import transaction
from keel.documents import corpus_tags, list_documents
from keel.providers.factory import AppContext, build_context
from keel.providers.local_index import parse_tags as parse_chunk_tags
from keel.safety.ledger import VerifyResult
from keel.safety.pii import DEFAULT_KINDS, redact
from keel.web import airgap_probe
from keel.web import docs as web_docs
from keel.web.views import (
    DEMO_USERS,
    agent_json,
    anonymous_user,
    answer_html,
    answer_json,
    chips_for,
    demo_user,
    is_loopback,
    parse_tags,
    proxy_user,
    resolve_user,
    source_url,
    step_view,
    trend_sparklines,
    user_json,
)

log = logging.getLogger("keel.web")

WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

ADMIN_TOKEN_ENV = "KEEL_ADMIN_TOKEN"
ADMIN_TOKEN_HEADER = "X-Keel-Admin-Token"
PROXY_TOKEN_ENV = "KEEL_PROXY_TOKEN"
PROXY_TOKEN_HEADER = "X-Keel-Proxy-Token"
IDENTITY_USER_HEADER = "X-Keel-User"
IDENTITY_TAGS_HEADER = "X-Keel-Tags"
PARTIAL_HEADER = "X-Keel-Partial"
LLM_PROBE_SECONDS = 2.0
RECENT_LIMIT = 50
TREND_DAYS = 14
MODES = ("answer", "agent")
DEFAULT_ACTOR = "admin"

#: The restricted question the overview page asks as both demo users. It names a value that lives in
#: an `hr`-tagged fixture document, so `public` is refused and `hr-officer` is answered with a citation.
COMPARE_QUESTION = "What is the confidential review code for the 2026 pay round?"

#: Starting text for the redaction demonstration. Every identifier here is invented and passes its own
#: check digit, which is the point: shape alone would match far more than this.
REDACT_SAMPLE = (
    "Contractor onboarding note. Reach Jordan Ellery at jordan.ellery@example.com or 0412 345 678.\n"
    "Tax file number 123 456 782, Medicare 2123 45670 1, ABN 51 824 753 556.\n"
    "Purchase order 987654321 and invoice 4417 stay as written: neither is an identifier."
)
REDACT_MAX_CHARS = 4000

#: One air-gap probe per this many seconds per client address. The probe starts a child process, so
#: this keeps a held-down button from becoming a way to spend the appliance's CPU.
PROBE_MIN_SECONDS = 3.0

#: Probes running at once, across every caller. A child process costs memory while it lives, and the
#: app already holds two embedding models, so a burst of visitors waits rather than crowds the box.
PROBE_MAX_CONCURRENT = 2

_ctx_lock = threading.Lock()


# ---------------------------------------------------------------------- templates


def _thousands(value: Any) -> str:
    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return "0"


def _clock(ts: Any) -> str:
    """`2026-08-18T12:34:56.789Z` shown as `2026-08-18 12:34:56`."""
    text = str(ts or "")
    return text[:19].replace("T", " ")


def _short(text: Any, limit: int = 90) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _pretty(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _percent(part: Any, whole: Any) -> str:
    try:
        whole_f = float(whole)
        return f"{100.0 * float(part) / whole_f:.0f}%" if whole_f > 0 else "0%"
    except (TypeError, ValueError):
        return "0%"


_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True, trim_blocks=True, lstrip_blocks=True
)
_env.filters.update(
    {"thousands": _thousands, "clock": _clock, "short": _short, "pretty": _pretty, "percent": _percent}
)
_env.globals.update(
    {
        "version": __version__,
        "demo_users": DEMO_USERS,
        "github": web_docs.GITHUB_REPO,
        "contact_email": web_docs.CONTACT_EMAIL,
        "author_github": web_docs.AUTHOR_GITHUB,
    }
)


def render(name: str, *, status: int = 200, **context: Any) -> HTMLResponse:
    """Render a template to an HTML response. Every value is escaped unless it is a Markup.

    The footer reads `profile` and `airgap`, and the header reads `front` and `admin_open`. All of
    them are filled from the app context when the caller did not pass them, so every page shows the
    same status line and the same navigation. `admin_open` follows the same rule `require_admin`
    enforces: the console is open on loopback, and beyond loopback it wants a token header no browser
    sends, so the link appears exactly where a visitor can follow it.
    """
    ctx = getattr(app.state, "ctx", None)
    if ctx is not None:
        context.setdefault("profile", ctx.profile)
        context.setdefault("airgap", ctx.settings.airgap)
        context.setdefault("demo", getattr(ctx.settings, "demo_identity", False))
        context.setdefault("front", front_page(ctx))
        context.setdefault("admin_open", is_loopback(ctx.settings.host))
    return HTMLResponse(_env.get_template(name).render(**context), status_code=status)


# ---------------------------------------------------------------------- app context


def get_ctx(request: Request) -> AppContext:
    """The AppContext on `app.state.ctx`, built on first use when nothing has set it yet."""
    state = request.app.state
    ctx = getattr(state, "ctx", None)
    if ctx is None:
        with _ctx_lock:
            ctx = getattr(state, "ctx", None)
            if ctx is None:
                ctx = build_context()
                state.ctx = ctx
                state.ctx_owned = True
    return ctx


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Build the context at startup unless one is already installed; close what this app built."""
    state = application.state
    if getattr(state, "ctx", None) is None:
        state.ctx = await run_in_threadpool(build_context)
        state.ctx_owned = True
        log.info("Keel web ready: profile %s, data dir %s", state.ctx.profile, state.ctx.settings.data_dir)
    manifest = getattr(getattr(state.ctx, "settings", None), "bootstrap_corpus", None)
    if manifest is not None:
        await run_in_threadpool(ensure_demo_corpus, state.ctx, manifest)
    try:
        yield
    finally:
        if getattr(state, "ctx_owned", False):
            state.ctx.close()
            state.ctx = None
            state.ctx_owned = False


app = FastAPI(title="Keel", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------- request helpers


def wants_json(request: Request) -> bool:
    """JSON for the /api paths and for clients that ask for it in Accept."""
    if request.url.path.startswith("/api/"):
        return True
    return "application/json" in request.headers.get("accept", "").lower()


def wants_partial(request: Request) -> bool:
    return request.headers.get(PARTIAL_HEADER, "") == "1"


async def read_payload(request: Request) -> dict[str, Any]:
    """The request body as one dict, from a JSON object or a form.

    Every form field is carried through, because a hardcoded list of names silently drops whatever a
    newer route adds. Repeated `tags` fields collect into a list; an absent body yields an empty dict.
    """
    content_type = request.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        try:
            data = await request.json()
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}
    form = await request.form()
    payload: dict[str, Any] = {key: form.get(key) for key in form}
    if "tags" in form:
        payload["tags"] = list(form.getlist("tags"))
    return payload


def _header_token_matches(request: Request, env_name: str, header: str) -> bool:
    """True when the request carries `header` equal to the non-empty environment variable `env_name`."""
    expected = os.environ.get(env_name, "")
    given = request.headers.get(header, "")
    return bool(expected and given and secrets.compare_digest(expected.encode(), given.encode()))


def trusted_identity(request: Request, user_id: Any = None, tags: Any = None) -> User:
    """The user a request runs as.

    On loopback identity is self-asserted, the demo: `user_id` and `tags` (from the body or the query
    string) resolve against the demo users. Beyond loopback the body and query string are ignored,
    since anyone can write them: identity comes from the reverse proxy through `X-Keel-User` and
    `X-Keel-Tags`, honoured only when the request also carries `X-Keel-Proxy-Token` equal to
    `KEEL_PROXY_TOKEN` (an unset token means no proxy is trusted). With `KEEL_DEMO_IDENTITY=1` (the
    hosted demo of the fixture corpus, never a real deployment) the demo user picker is honoured
    beyond loopback as well: `user_id` resolves to one of the demo users and its fixed tags, extra tags
    are ignored. Any other request runs as `public`.
    """
    ctx = get_ctx(request)
    if is_loopback(ctx.settings.host):
        return resolve_user(user_id, tags)
    if _header_token_matches(request, PROXY_TOKEN_ENV, PROXY_TOKEN_HEADER):
        return proxy_user(
            request.headers.get(IDENTITY_USER_HEADER), request.headers.get(IDENTITY_TAGS_HEADER)
        )
    if getattr(ctx.settings, "demo_identity", False):
        return demo_user(user_id)
    return anonymous_user()


def user_from_query(request: Request) -> User:
    return trusted_identity(request, request.query_params.get("user"), request.query_params.get("tags"))


def actor_from(payload: dict[str, Any]) -> str:
    return str(payload.get("by") or "").strip() or DEFAULT_ACTOR


def error_response(request: Request, status: int, message: str, *, partial_ok: bool = True) -> Response:
    """One error shape for every route: JSON, an HTML fragment for the question box, or an error page."""
    if wants_json(request):
        return JSONResponse({"error": message}, status_code=status)
    if partial_ok and wants_partial(request):
        return render("_error.html", status=status, message=message)
    return render("error.html", status=status, message=message, status_code=status, active="")


@app.exception_handler(StarletteHTTPException)
async def http_exception_as_page(request: Request, exc: StarletteHTTPException) -> Response:
    """Raised errors take the same shape as returned ones: a styled page for a browser, JSON for an
    API caller. Without this, a visitor clicking Admin without the token reads a bare JSON detail."""
    return error_response(request, exc.status_code, str(exc.detail))


def prefers_html(request: Request) -> bool:
    """A browser navigation asks for text/html in Accept; monitors and curl do not."""
    return "text/html" in request.headers.get("accept", "").lower()


def redirect_admin(fragment: str = "") -> RedirectResponse:
    return RedirectResponse(url=f"/admin#{fragment}" if fragment else "/admin", status_code=303)


# ---------------------------------------------------------------------- store reads


def _count(ctx: AppContext, sql: str, *args: Any) -> int:
    row = ctx.conn.execute(sql, args).fetchone()
    return int(row[0] or 0) if row else 0


def store_counts(ctx: AppContext) -> dict[str, int]:
    return {
        "documents": _count(ctx, "SELECT COUNT(*) FROM documents"),
        "chunks": _count(ctx, "SELECT COUNT(*) FROM chunks"),
        "quarantined": _count(ctx, "SELECT COUNT(*) FROM chunks WHERE quarantined = 1"),
        "ledger_seq": _count(ctx, "SELECT COALESCE(MAX(seq), 0) FROM ledger"),
        "pending_approvals": _count(ctx, "SELECT COUNT(*) FROM approvals WHERE status = 'pending'"),
    }


def llm_healthy(llm: Any, timeout: float = LLM_PROBE_SECONDS) -> bool:
    """`llm.healthy()` with a hard time limit: a probe still running after `timeout` seconds counts as
    unhealthy and is left to finish on its own daemon thread."""
    box: dict[str, bool] = {}

    def probe() -> None:
        try:
            box["ok"] = bool(llm.healthy())
        except Exception:
            box["ok"] = False

    thread = threading.Thread(target=probe, name="keel-llm-probe", daemon=True)
    thread.start()
    thread.join(timeout)
    return box.get("ok", False)


def load_chunk(ctx: AppContext, chunk_id: int) -> dict[str, Any] | None:
    row = ctx.conn.execute(
        """SELECT c.id AS chunk_id, c.text, c.heading, c.page, c.ordinal, c.acl_tags, c.quarantined,
                  c.quarantine_reason, c.document_id, d.source, d.title
           FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.id = ?""",
        (int(chunk_id),),
    ).fetchone()
    if row is None:
        return None
    chunk = dict(row)
    chunk["acl_tags"] = parse_chunk_tags(chunk.get("acl_tags"))
    chunk["quarantined"] = bool(chunk.get("quarantined"))
    return chunk


def quarantine_rows(ctx: AppContext) -> list[dict[str, Any]]:
    rows = ctx.conn.execute(
        """SELECT c.id AS chunk_id, c.quarantine_reason, c.heading, c.acl_tags, d.source, d.title
           FROM chunks c JOIN documents d ON d.id = c.document_id
           WHERE c.quarantined = 1 ORDER BY c.id"""
    ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["acl_tags"] = parse_chunk_tags(item.get("acl_tags"))
        item["url"] = source_url(item["chunk_id"], User("admin", item["acl_tags"] or ["public"]))
        out.append(item)
    return out


def ledger_rows_for(ctx: AppContext, request_id: str) -> list[dict[str, Any]]:
    rows = ctx.conn.execute(
        "SELECT seq, ts, kind, payload FROM ledger WHERE request_id = ? ORDER BY seq", (request_id,)
    ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        try:
            item["payload"] = json.loads(item["payload"])
        except (TypeError, ValueError):
            pass
        out.append(item)
    return out


def approvals_for(ctx: AppContext, request_id: str) -> list[dict[str, Any]]:
    return [row for row in ctx.approvals.list() if row.get("request_id") == request_id]


def upload_context() -> dict[str, Any]:
    """What the admin page needs to offer an upload, or to leave the section out entirely."""
    if not web_ingest_enabled():
        return {"ingest_enabled": False, "upload_limits": "", "upload_accept": ""}
    from keel.web.uploads import ALLOWED_SUFFIXES, describe_limits

    return {
        "ingest_enabled": True,
        "upload_limits": describe_limits(),
        "upload_accept": ",".join(sorted(ALLOWED_SUFFIXES)),
    }


def admin_context(
    ctx: AppContext,
    *,
    verify: VerifyResult | None = None,
    notice: str | None = None,
    request: Request | None = None,
) -> dict[str, Any]:
    """Everything the admin page shows, gathered in one place so every admin route renders alike.
    `request`, when given, supplies the address the app is actually being reached on."""
    listen_host = (request.url.hostname if request is not None else None) or ctx.settings.host
    listen_port = (request.url.port if request is not None else None) or ctx.settings.port
    totals = ctx.log.totals()
    counts = store_counts(ctx)
    approvals = ctx.approvals.list()
    return {
        "active": "admin",
        "profile": ctx.profile,
        "totals": totals,
        "counts": counts,
        "recent": ctx.log.recent(RECENT_LIMIT),
        "sparklines": trend_sparklines(ctx.log.daily_summary(TREND_DAYS), TREND_DAYS),
        "trend_days": TREND_DAYS,
        "quarantine": quarantine_rows(ctx),
        "pending": [row for row in approvals if row["status"] == "pending"],
        "decided": [row for row in reversed(approvals) if row["status"] != "pending"][:10],
        "verify": verify,
        "notice": notice,
        "settings": {
            "host": listen_host,
            "port": listen_port,
            "profile": ctx.settings.profile,
            "airgap": ctx.settings.airgap,
            "llm_model": ctx.settings.local_llm_model
            if ctx.profile == "local"
            else ctx.settings.azure_openai_chat_deployment,
            "min_relevance": ctx.settings.min_relevance,
            "guarded": not is_loopback(ctx.settings.host),
        },
        "documents": [row.to_dict() for row in list_documents(ctx.conn)],
        "corpus_tags": corpus_tags(ctx.conn),
        **upload_context(),
    }


# ---------------------------------------------------------------------- health


# HEAD is registered alongside GET on / and /health for link checkers and unfurl bots, which probe
# with HEAD and read a 405 as a dead page.
@app.get("/health")
@app.head("/health")
def health(request: Request) -> Response:
    """Liveness plus the numbers a monitor wants: profile, model reachability, corpus size, ledger
    head. Monitors and curl get JSON; a browser following the nav link gets the same numbers as a
    page."""
    ctx = get_ctx(request)
    counts = store_counts(ctx)
    body = {
        "status": "ok",
        "profile": ctx.profile,
        "llm": llm_healthy(ctx.llm),
        "documents": counts["documents"],
        "chunks": counts["chunks"],
        "quarantined": counts["quarantined"],
        "ledger_seq": counts["ledger_seq"],
    }
    if prefers_html(request) and not wants_json(request):
        return render("health.html", active="health", health=body)
    return JSONResponse(body)


# ---------------------------------------------------------------------- overview and documentation


def identity_picker(ctx: AppContext) -> bool:
    """True when this deployment honours the demo user picker, so the comparison can run in a browser.

    That is loopback, where identity is self-asserted by design, and the hosted demo of the fixture
    corpus (`KEEL_DEMO_IDENTITY=1`). Everywhere else identity comes from the operator's proxy and the
    overview page says so instead of offering a picker that would be ignored.
    """
    host = getattr(ctx.settings, "host", "")
    return is_loopback(host) or bool(getattr(ctx.settings, "demo_identity", False))


def front_page(ctx: AppContext) -> str:
    """`overview` or `chat`: what `/` serves on this deployment.

    `KEEL_FRONT_PAGE` decides it outright when set. On `auto`, the hosted demo of the fixture corpus
    leads with the overview, because a visitor arriving from a link needs orienting before a form.
    Every other deployment leads with the question box, because whoever installed Keel came to use it.
    The overview stays at `/about` in both cases.
    """
    choice = str(getattr(ctx.settings, "front_page", "auto") or "auto")
    if choice in ("overview", "chat"):
        return choice
    return "overview" if getattr(ctx.settings, "demo_identity", False) else "chat"


def overview_page(ctx: AppContext) -> HTMLResponse:
    """The overview: what Keel is, the three live demonstrations, the way into the documentation."""
    return render(
        "landing.html",
        active="overview",
        profile=ctx.profile,
        picker=identity_picker(ctx),
        compare_question=COMPARE_QUESTION,
        redact_sample=REDACT_SAMPLE,
        docs=web_docs.index(),
    )


@app.get("/", response_class=HTMLResponse)
@app.head("/", response_class=HTMLResponse)
def root(request: Request) -> HTMLResponse:
    """The front door, which depends on who deployed this. See `front_page`."""
    ctx = get_ctx(request)
    if front_page(ctx) == "overview":
        return overview_page(ctx)
    return chat_page(ctx)


@app.get("/about", response_class=HTMLResponse)
@app.head("/about", response_class=HTMLResponse)
def about(request: Request) -> HTMLResponse:
    """The overview, always reachable whichever page `/` serves."""
    return overview_page(get_ctx(request))


@app.get("/docs", response_class=HTMLResponse)
@app.head("/docs", response_class=HTMLResponse)
def docs_index(request: Request) -> HTMLResponse:
    """Every document in `docs/`, rendered as pages of this site rather than left on GitHub."""
    return render("docs_index.html", active="docs", profile=get_ctx(request).profile, docs=web_docs.index())


@app.get("/docs/{slug}", response_class=HTMLResponse)
@app.head("/docs/{slug}", response_class=HTMLResponse)
def docs_page(request: Request, slug: str) -> Response:
    """One document. An unknown slug is a 404 through the usual error page rather than a stack trace."""
    doc = web_docs.page(slug)
    if doc is None:
        return error_response(request, 404, "That document is not one of the pages here.", partial_ok=False)
    ordered = web_docs.slugs()
    position = ordered.index(slug) if slug in ordered else -1
    return render(
        "doc.html",
        active="docs",
        profile=get_ctx(request).profile,
        doc=doc,
        prev=_neighbour(ordered, position - 1) if position > 0 else None,
        next=_neighbour(ordered, position + 1) if position >= 0 else None,
    )


def _neighbour(ordered: list[str], index: int) -> web_docs.Doc | None:
    """The document at `index` in reading order, or None when the index falls off either end."""
    if 0 <= index < len(ordered):
        return web_docs.page(ordered[index])
    return None


# ---------------------------------------------------------------------- chat


FIXTURE_MARKER = "northbank-council-procurement"


def corpus_state(ctx: AppContext) -> tuple[int, bool]:
    """(document count, whether the fixture corpus is loaded).

    The sample questions on the question box name Northbank Council and the Harbour Clinic, so they
    help only where those documents are present. Somebody running Keel over their own files gets the
    box without the fixture hints.
    """
    documents = int(ctx.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
    if documents == 0:
        return 0, False
    row = ctx.conn.execute(
        "SELECT 1 FROM documents WHERE source LIKE ? LIMIT 1", (f"%{FIXTURE_MARKER}%",)
    ).fetchone()
    return documents, row is not None


def chat_page(
    ctx: AppContext, *, form: dict[str, Any] | None = None, result: dict[str, Any] | None = None
) -> HTMLResponse:
    """The question box, with the submitted values kept in the form and any result rendered in place.

    `picker` decides whether the identity controls render at all. Beyond loopback and away from the
    hosted demo, `trusted_identity` takes the user and the tags from the reverse proxy and ignores
    anything in the form, so offering a select there would be offering a control that does nothing.
    """
    values = {"question": "", "user_id": "public", "tags": "", "mode": "answer"}
    if form:
        values.update({k: str(v or "") for k, v in form.items() if k in values})
    documents, fixtures = corpus_state(ctx)
    return render(
        "chat.html",
        active="chat",
        profile=ctx.profile,
        form=values,
        result=result,
        documents=documents,
        fixture_corpus=fixtures,
        picker=identity_picker(ctx),
        ingest_enabled=web_ingest_enabled(),
    )


@app.get("/chat", response_class=HTMLResponse)
@app.head("/chat", response_class=HTMLResponse)
def chat(request: Request) -> HTMLResponse:
    """The appliance: question box, user select, free tags, answer or agent mode."""
    return chat_page(get_ctx(request))


def result_context(mode: str, question: str, user: User, outcome: Any) -> dict[str, Any]:
    """Template context for the result partial in either mode."""
    if mode == "agent":
        return {
            "mode": mode,
            "question": question,
            "user": user,
            "agent": outcome,
            "steps": [step_view(step) for step in outcome.steps],
            "request_id": outcome.request_id,
            "latency_ms": outcome.latency_ms,
            "prompt_tokens": outcome.prompt_tokens,
            "output_tokens": outcome.output_tokens,
        }
    return {
        "mode": mode,
        "question": question,
        "user": user,
        "answer": outcome,
        "html": answer_html(outcome.text, outcome.citations, user),
        "chips": chips_for(outcome.citations, user),
        "request_id": outcome.request_id,
        "latency_ms": outcome.latency_ms,
        "prompt_tokens": outcome.prompt_tokens,
        "output_tokens": outcome.output_tokens,
    }


async def run_question(request: Request, *, force_mode: str | None = None) -> Response:
    """Shared body of /ask, /api/ask and /api/agent: parse, run the engine or the agent off the event
    loop, and reply as JSON, an HTML partial, or the whole page with the result in place."""
    ctx = get_ctx(request)
    payload = await read_payload(request)
    question = str(payload.get("question") or "").strip()
    mode = force_mode or str(payload.get("mode") or "answer").strip().lower()
    user = trusted_identity(request, payload.get("user_id"), payload.get("tags"))
    if mode not in MODES:
        return error_response(request, 400, "mode is one of: answer, agent")
    if not question:
        return error_response(request, 400, "Type a question first.")
    try:
        if mode == "agent":
            outcome = await run_in_threadpool(ctx.agent_loop.run, question, user)
        else:
            outcome = await run_in_threadpool(ctx.answer_engine.answer, question, user)
    except Exception as exc:
        log.exception("%s request failed", mode)
        endpoint = getattr(ctx.llm, "base_url", None) or "the model endpoint"
        message = (
            f"The model at {endpoint} is unreachable right now ({type(exc).__name__}). "
            "Start llama-server, then send the question again."
        )
        return error_response(request, 502, message)
    if wants_json(request):
        body = agent_json(outcome, user) if mode == "agent" else answer_json(outcome, user)
        return JSONResponse(body)
    context = result_context(mode, question, user, outcome)
    if wants_partial(request):
        return render("_result.html", r=context)
    form = {
        "question": question,
        "user_id": user.user_id,
        "tags": str(payload.get("tags") or ""),
        "mode": mode,
    }
    if isinstance(payload.get("tags"), list):
        form["tags"] = ", ".join(str(t) for t in payload["tags"])
    return chat_page(ctx, form=form, result=context)


@app.post("/ask")
async def ask(request: Request) -> Response:
    """Answer or agent run from a form or JSON body {question, user_id, tags, mode}."""
    return await run_question(request)


@app.post("/api/ask")
async def api_ask(request: Request) -> Response:
    """JSON answer for {question, user_id, tags, mode?}."""
    return await run_question(request)


@app.post("/api/agent")
async def api_agent(request: Request) -> Response:
    """JSON agent run for {question, user_id, tags}."""
    return await run_question(request, force_mode="agent")


# ---------------------------------------------------------------------- live control demonstrations
#
# Two POST routes that change nothing. They exist so a reader can exercise a control rather than read
# a claim about it. Neither touches the store, the model or the ledger.


_probe_lock = threading.Lock()
_probe_last: dict[str, float] = {}
_probe_slots = threading.Semaphore(PROBE_MAX_CONCURRENT)


def _probe_allowed(caller: str) -> bool:
    """One probe per caller per `PROBE_MIN_SECONDS`. The probe starts a child process, so a held-down
    button is worth slowing down; the map is bounded so it cannot grow into a memory leak."""
    now = time.monotonic()
    with _probe_lock:
        if len(_probe_last) > 1024:
            cutoff = now - PROBE_MIN_SECONDS
            for key in [k for k, seen in _probe_last.items() if seen < cutoff]:
                del _probe_last[key]
        previous = _probe_last.get(caller)
        if previous is not None and now - previous < PROBE_MIN_SECONDS:
            return False
        _probe_last[caller] = now
        return True


@app.post("/api/airgap-probe")
async def api_airgap_probe(request: Request) -> Response:
    """Attempt a connection to a named host, under the air-gap guard, and report what each layer did.

    The attempts run in a child process started with `KEEL_AIRGAP=1`, so this worker's own guard state
    stays as the operator set it and nobody else's request is affected. A host outside the allow list
    cannot be reached, which is the property being demonstrated, and a host inside it is answered from
    the policy without a connection. See `keel.web.airgap_probe`.
    """
    payload = await read_payload(request)
    caller = request.client.host if request.client else "unknown"
    if not _probe_allowed(caller):
        return JSONResponse(
            {"error": f"One probe every {PROBE_MIN_SECONDS:.0f} seconds. Try again shortly."},
            status_code=429,
        )
    host = str(payload.get("host") or "")
    if not _probe_slots.acquire(blocking=False):
        return JSONResponse(
            {"error": "The probe is busy with another visitor. Try again in a moment."},
            status_code=429,
        )
    try:
        body = await run_in_threadpool(airgap_probe.run, host)
    finally:
        _probe_slots.release()
    return JSONResponse(body, status_code=400 if "error" in body else 200)


@app.post("/api/redact")
async def api_redact(request: Request) -> Response:
    """Redact the submitted text and report what was found and where.

    `keel.safety.pii.redact` is a pure function of its input: no store, no model, no network. The text
    is neither kept nor logged, and the response carries the spans so a reader can see which characters
    each check digit actually claimed.
    """
    payload = await read_payload(request)
    text = str(payload.get("text") or "")
    if len(text) > REDACT_MAX_CHARS:
        return JSONResponse(
            {"error": f"Send at most {REDACT_MAX_CHARS} characters."}, status_code=400
        )
    redacted, findings = redact(text)
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding["kind"]] = counts.get(finding["kind"], 0) + 1
    return JSONResponse(
        {
            "redacted": redacted,
            "findings": [
                {"kind": finding["kind"], "start": finding["span"][0], "end": finding["span"][1]}
                for finding in findings
            ],
            "counts": counts,
            "kinds": list(DEFAULT_KINDS),
        }
    )


# ---------------------------------------------------------------------- source viewer


@app.get("/source/{chunk_id}")
def source(request: Request, chunk_id: int) -> Response:
    """One chunk with its document, heading, page and ACL tags. The caller's tags (query `user` and
    `tags` on loopback, the proxy identity headers beyond it) must share a tag with the chunk;
    otherwise 403 with a message that names nothing about the chunk. Unknown ids give 404."""
    ctx = get_ctx(request)
    user = user_from_query(request)
    chunk = load_chunk(ctx, chunk_id)
    if chunk is None:
        return error_response(request, 404, f"Chunk {chunk_id} is unknown to this store.", partial_ok=False)
    if not set(chunk["acl_tags"]) & set(user.tags):
        return error_response(
            request,
            403,
            f"User {user.user_id} carries {', '.join(user.tags)}; this chunk is outside those entitlements.",
            partial_ok=False,
        )
    if wants_json(request):
        return JSONResponse({**chunk, "user": user_json(user)})
    return render("source.html", active="chat", profile=ctx.profile, chunk=chunk, user=user)


# ---------------------------------------------------------------------- admin


def require_admin(request: Request) -> None:
    """Admin routes are open on loopback. Beyond loopback the request must carry the admin token."""
    ctx = get_ctx(request)
    if is_loopback(ctx.settings.host):
        return
    if _header_token_matches(request, ADMIN_TOKEN_ENV, ADMIN_TOKEN_HEADER):
        return
    raise HTTPException(
        status_code=401,
        detail=f"Admin routes need the {ADMIN_TOKEN_HEADER} header matching {ADMIN_TOKEN_ENV} when Keel listens beyond loopback.",
    )


admin = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])


@admin.get("", response_class=HTMLResponse)
def admin_page(request: Request) -> HTMLResponse:
    """Recent requests, totals, the 14-day trend, quarantine, approvals and ledger controls."""
    ctx = get_ctx(request)
    return render("admin.html", **admin_context(ctx, request=request))


@admin.get("/request/{request_id}")
def admin_request(request: Request, request_id: str) -> Response:
    """One logged request: retrieved chunk ids, citations, tool calls, judge scores, ledger rows."""
    ctx = get_ctx(request)
    row = ctx.log.get(request_id)
    if row is None:
        return error_response(
            request, 404, f"Request {request_id} is not in the inference log.", partial_ok=False
        )
    user = User(str(row.get("user_id") or "public"), list(row.get("user_tags") or ["public"]))
    retrieved = [
        {"chunk_id": cid, "url": source_url(int(cid), user)}
        for cid in (row.get("retrieved_ids") or [])
        if isinstance(cid, int)
    ]
    citations = row.get("citations") or []
    for citation in citations:
        if isinstance(citation, dict) and isinstance(citation.get("chunk_id"), int):
            citation["url"] = source_url(citation["chunk_id"], user)
    steps = [step_view(step) for step in (row.get("tool_calls") or []) if isinstance(step, dict)]
    if wants_json(request):
        return JSONResponse(
            {**row, "ledger": ledger_rows_for(ctx, request_id), "approvals": approvals_for(ctx, request_id)}
        )
    return render(
        "admin_request.html",
        active="admin",
        profile=ctx.profile,
        row=row,
        retrieved=retrieved,
        citations=citations,
        steps=steps,
        ledger=ledger_rows_for(ctx, request_id),
        approvals=approvals_for(ctx, request_id),
    )


@admin.post("/approvals/{approval_id}/approve")
async def approve(request: Request, approval_id: int) -> Response:
    """Approve a pending write call and run it through the registry; the result lands on the row."""
    ctx = get_ctx(request)
    payload = await read_payload(request)
    row = ctx.approvals.get(approval_id)
    if row is None:
        return error_response(request, 404, f"Approval {approval_id} does not exist.", partial_ok=False)
    try:
        ctx.approvals.decide(approval_id, True, actor_from(payload))
    except ValueError as exc:
        return error_response(request, 409, str(exc), partial_ok=False)
    context = ToolContext(user=None, request_id=str(row.get("request_id") or ""))
    result = await run_in_threadpool(
        ctx.approvals.execute, approval_id, ctx.registry, context, policy=ctx.policy
    )
    if wants_json(request):
        return JSONResponse({**(ctx.approvals.get(approval_id) or {}), "result": result})
    return redirect_admin("approvals")


@admin.post("/approvals/{approval_id}/reject")
async def reject(request: Request, approval_id: int) -> Response:
    """Reject a pending write call. Nothing runs."""
    ctx = get_ctx(request)
    payload = await read_payload(request)
    if ctx.approvals.get(approval_id) is None:
        return error_response(request, 404, f"Approval {approval_id} does not exist.", partial_ok=False)
    try:
        row = ctx.approvals.decide(approval_id, False, actor_from(payload))
    except ValueError as exc:
        return error_response(request, 409, str(exc), partial_ok=False)
    if wants_json(request):
        return JSONResponse(row)
    return redirect_admin("approvals")


@admin.post("/quarantine/{chunk_id}/release")
async def release_quarantine(request: Request, chunk_id: int) -> Response:
    """Clear a chunk's quarantine flag so retrieval may return it again, and record who did it
    as a ledger row of kind `quarantine`."""
    ctx = get_ctx(request)
    payload = await read_payload(request)
    chunk = load_chunk(ctx, chunk_id)
    if chunk is None:
        return error_response(request, 404, f"Chunk {chunk_id} is unknown to this store.", partial_ok=False)
    if not chunk["quarantined"]:
        return error_response(
            request, 409, f"Chunk {chunk_id} is already out of quarantine.", partial_ok=False
        )
    by = actor_from(payload)
    with transaction(ctx.conn):
        ctx.conn.execute("UPDATE chunks SET quarantined = 0 WHERE id = ?", (int(chunk_id),))
        entry = ctx.ledger.append(
            "quarantine", None, {"chunk_id": int(chunk_id), "action": "release", "by": by}
        )
    if wants_json(request):
        return JSONResponse(
            {"chunk_id": int(chunk_id), "quarantined": False, "by": by, "ledger_seq": entry.seq}
        )
    return redirect_admin("quarantine")


@admin.post("/ledger/verify")
def verify_ledger(request: Request) -> Response:
    """Recompute the whole hash chain and show the result on the admin page."""
    ctx = get_ctx(request)
    result = ctx.ledger.verify()
    if wants_json(request):
        return JSONResponse(
            {
                "ok": result.ok,
                "checked": result.checked,
                "first_bad_seq": result.first_bad_seq,
                "reason": result.reason,
            }
        )
    return render("admin.html", **admin_context(ctx, verify=result, request=request))


def _ledger_lines(ctx: AppContext) -> Iterator[str]:
    for entry in ctx.ledger.entries():
        record = {
            "seq": entry.seq,
            "ts": entry.ts,
            "kind": entry.kind,
            "request_id": entry.request_id,
            "prev_hash": entry.prev_hash,
            "hash": entry.hash,
            "payload": entry.payload,
        }
        yield json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"


@admin.get("/ledger/export")
def export_ledger(request: Request) -> StreamingResponse:
    """The ledger as JSONL, one row per line, in the shape `keel.safety.ledger.verify_file` reads."""
    ctx = get_ctx(request)
    headers = {"Content-Disposition": 'attachment; filename="keel-ledger.jsonl"'}
    return StreamingResponse(_ledger_lines(ctx), media_type="application/x-ndjson", headers=headers)


# ---------------------------------------------------------------------- ingest from the browser
#
# This route exists only where the deployment permits writes. `KEEL_DEMO_READONLY=1` leaves it
# unregistered rather than registered-and-refusing, so the hosted demo of the fixture corpus has no
# ingest path at all and a reader can confirm that from the route table rather than from a promise.


def web_ingest_enabled() -> bool:
    """True when this deployment registers the browser ingest route.

    Read once at import, because the answer decides the shape of the application rather than the
    outcome of a request. `keel.config.Settings` reads the environment and the `.env` beside it and
    touches neither the model nor the store, which keeps import cheap.
    """
    from keel.config import Settings

    return not Settings().demo_readonly


if web_ingest_enabled():

    @admin.post("/ingest")
    async def admin_ingest(request: Request) -> Response:
        """Take a document from the browser and put it through the ordinary ingest pipeline.

        Behind the admin guard, so beyond loopback it needs the admin token like every other write.
        The file is checked and staged by `keel.web.uploads` before anything touches the store, then
        ingested by the same `ingest_path` the command line calls: the same chunking, the same
        injection screen, the same ledger row. Tags are the access-control model, so they are taken
        from the form rather than guessed at, and an upload with none is `public`.
        """
        import shutil
        import tempfile

        from keel.ingest import ingest_path
        from keel.web.uploads import UploadRejected, describe_limits, stage_upload

        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "filename"):
            return error_response(request, 400, "Choose a file to ingest.", partial_ok=False)

        tags = parse_tags(form.get("tags")) or ["public"]
        title = str(form.get("title") or "").strip() or None
        actor = str(form.get("by") or "").strip() or DEFAULT_ACTOR

        staging = Path(tempfile.mkdtemp(prefix="keel-upload-"))
        try:
            try:
                staged = await run_in_threadpool(stage_upload, upload.file, upload.filename, staging)
            except UploadRejected as rejected:
                return error_response(request, 400, str(rejected), partial_ok=False)

            ctx = get_ctx(request)
            try:
                result = await run_in_threadpool(
                    ingest_path,
                    ctx.conn,
                    ctx.settings,
                    ctx.embedder,
                    ctx.index,
                    staged.path,
                    title=title,
                    acl_tags=tags,
                    screen=ctx.screen,
                    ledger=ctx.ledger,
                    meta={"uploaded_by": actor, "original_name": staged.name, "bytes": staged.size},
                )
            except Exception as error:  # noqa: BLE001 (a bad document is the sender's problem to read)
                log.warning("upload of %s could not be ingested: %s", staged.name, error)
                return error_response(
                    request,
                    400,
                    f"{staged.name} could not be read as a document. Keel reads {describe_limits()}.",
                    partial_ok=False,
                )
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        body = result.to_dict()
        body["uploaded_name"] = staged.name
        if wants_json(request):
            return JSONResponse(body)
        return redirect_admin("#documents")


    @admin.post("/documents/{document_id}/retag")
    async def admin_retag(request: Request, document_id: int) -> Response:
        """Change a document's access tags, and its chunks' tags with it.

        Mistagging a sensitive document is the ordinary mistake on an appliance like this, and until
        this route existed the only remedy was editing SQLite. Both tables move inside one
        transaction with a ledger row, so the correction is audited like the ingest was.
        """
        from keel.documents import retag_document

        payload = await read_payload(request)
        ctx = get_ctx(request)
        try:
            updated = await run_in_threadpool(
                retag_document,
                ctx.conn,
                ctx.index,
                document_id,
                payload.get("tags", ""),
                by=actor_from(payload),
                ledger=ctx.ledger,
            )
        except LookupError:
            return error_response(request, 404, f"No document {document_id} in this store.", partial_ok=False)
        if wants_json(request):
            return JSONResponse(updated.to_dict())
        return redirect_admin("#documents")

    @admin.post("/documents/{document_id}/remove")
    async def admin_remove_document(request: Request, document_id: int) -> Response:
        """Take a document out of the store, with its chunks, full-text entries and embeddings.

        The ledger row is written before the rows go, inside the same transaction, so a removal is
        recorded even though the thing it describes is gone. See `keel.documents.remove_document`
        for why one DELETE is enough to leave nothing retrievable behind.
        """
        from keel.documents import remove_document

        payload = await read_payload(request)
        ctx = get_ctx(request)
        try:
            removed = await run_in_threadpool(
                remove_document,
                ctx.conn,
                ctx.index,
                document_id,
                by=actor_from(payload),
                ledger=ctx.ledger,
            )
        except LookupError:
            return error_response(request, 404, f"No document {document_id} in this store.", partial_ok=False)
        if wants_json(request):
            return JSONResponse(removed.to_dict())
        return redirect_admin("#documents")

    @admin.post("/connection")
    async def admin_connection_switch(request: Request) -> Response:
        """Point Keel at a different model server, from the ones answering on this machine.

        Loopback only, and deliberately so. A form that accepts any endpoint is an exfiltration
        lever: whoever reaches it could point the model at a host they control and read every
        question and every retrieved passage. Restricting the choice to what is already listening on
        this machine keeps the convenience and removes the lever.

        The running providers are updated in place so the change takes effect on the next question,
        and `.env` is written so it survives a restart.
        """
        from keel.onboarding import merge_env, probe_endpoint
        from keel.providers.local_llm import OpenAICompatibleLLM
        from keel.web.views import is_loopback

        payload = await read_payload(request)
        base_url = str(payload.get("base_url") or "").strip()
        model = str(payload.get("model") or "").strip()
        host = urlsplit(base_url).hostname or ""
        if not base_url or not model:
            return error_response(request, 400, "Name an endpoint and a model.", partial_ok=False)
        if not is_loopback(host):
            return error_response(
                request,
                400,
                "Keel switches only to a model server on this machine. An endpoint elsewhere is a "
                "deployment decision, set through KEEL_LOCAL_LLM_BASE_URL where it can be reviewed.",
                partial_ok=False,
            )
        reachable, models, detail = await run_in_threadpool(probe_endpoint, base_url)
        if not reachable:
            return error_response(request, 400, f"{base_url} is out of reach: {detail}", partial_ok=False)
        if models and model not in models:
            listed = ", ".join(models[:6])
            return error_response(
                request, 400, f"{base_url} serves {listed}, and not {model}.", partial_ok=False
            )

        ctx = get_ctx(request)
        swapped = OpenAICompatibleLLM(
            base_url=base_url, model=model, api_key=ctx.settings.local_llm_api_key,
            timeout=ctx.settings.local_llm_timeout,
        )
        # The engine and the loop each hold their own reference from build time, so all three move
        # together. A request already mid-call finishes on the provider it started with.
        ctx.llm = swapped
        ctx.answer_engine.llm = swapped
        ctx.agent_loop.llm = swapped
        ctx.settings.local_llm_base_url = base_url
        ctx.settings.local_llm_model = model
        values = {"KEEL_LOCAL_LLM_BASE_URL": base_url, "KEEL_LOCAL_LLM_MODEL": model}
        await run_in_threadpool(merge_env, Path(".env"), values)
        log.info("model switched to %s at %s", model, base_url)
        if wants_json(request):
            return JSONResponse({"base_url": base_url, "model": model, "applied": True})
        return RedirectResponse("/admin/connection", status_code=303)


@admin.get("/connection", response_class=HTMLResponse)
def admin_connection(request: Request) -> HTMLResponse:
    """What Keel is pointed at, whether it answers, and what to do when it does not.

    The same checks `keel doctor` runs, on a page, for whoever deployed this with `docker compose up`
    and has no terminal on the box. Probing costs a couple of seconds, which is why it sits here
    rather than on every admin page load.
    """
    from keel.onboarding import discover, run_checks
    from keel.web.views import is_loopback

    ctx = get_ctx(request)
    checks = run_checks(ctx.settings)
    found = discover() if ctx.profile == "local" else []
    return render(
        "connection.html",
        active="admin",
        profile=ctx.profile,
        checks=[
            {"name": c.name, "ok": c.ok, "detail": c.detail, "fix": c.fix, "mark": c.mark} for c in checks
        ],
        wanting=sum(1 for c in checks if not c.ok),
        discovered=[
            {"name": d.name, "base_url": d.base_url, "models": list(d.models)}
            for d in found
            if is_loopback(urlsplit(d.base_url).hostname or "")
        ],
        current={
            "profile": ctx.profile,
            "base_url": ctx.settings.local_llm_base_url,
            "model": ctx.settings.local_llm_model,
            "azure_openai": ctx.settings.azure_openai_endpoint,
            "azure_search": ctx.settings.azure_search_endpoint,
            "chat_deployment": ctx.settings.azure_openai_chat_deployment,
        },
        can_switch=web_ingest_enabled() and ctx.profile == "local",
    )

app.include_router(admin)


# ---------------------------------------------------------------------- misc


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


__all__ = ["app", "get_ctx", "llm_healthy", "require_admin"]
