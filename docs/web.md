# Web app: overview, demonstration, documentation, admin

`keel/web/app.py` exposes `app`, a FastAPI application with server-rendered Jinja2 pages, one small
vanilla JavaScript file and one stylesheet. There is no build step: a reviewer runs it, opens a browser
and reads the templates.

The site has four surfaces. `/` is the overview, which is where a reader arriving from a link lands:
what Keel is, three demonstrations that run the production code paths live, and the way into the
reference material. `/chat` is the appliance itself. `/docs` renders the Markdown under `docs/` as
pages of the site, so the reference material and the source never drift apart. `/admin` is the
operator's surface and needs the token beyond loopback.

## Run

```powershell
# Everything (llama-server + web app), idempotent
.\deploy\onprem\run.ps1

# Web app alone, when llama-server is already answering on 127.0.0.1:8081
.\.venv\Scripts\python.exe -m uvicorn keel.web.app:app --host 127.0.0.1 --port 8400
# or, once the CLI is installed
keel serve
```

Then open <http://127.0.0.1:8400/>. `KEEL_DATA_DIR` picks the store; ingest the fixture corpus first
(`keel ingest --manifest fixtures/corpus.yaml`, or `demo.ps1` does the whole thing) or the
demonstration page refuses every question because there is nothing to retrieve. The overview,
the documentation and the air-gap and redaction demonstrations all work against an empty store.

The app builds its `AppContext` (`keel.providers.factory.build_context()`) once, at startup in the
lifespan handler, and keeps it on `app.state.ctx`. Servers that skip lifespan events get the same
context built on the first request. Tests set `app.state.ctx` themselves before the first request
and the app uses that instead. Import time touches neither the model nor the store.

## Routes

| Method and path | What it does |
| --- | --- |
| `GET /` | Overview: what Keel is, the permission comparison, the air-gap probe, the redaction demonstration, how a question travels, the review summary, the control table, the deployment shapes and the documentation index. The comparison appears where the demo user picker is honoured, which is loopback and `KEEL_DEMO_IDENTITY=1`. |
| `GET /chat` | The appliance: question box, user select (`public` [public], `hr-officer` [public, hr]), free extra tags, mode toggle answer or agent, submit. Results render in place. |
| `GET /docs` | The documentation index, built from the Markdown files in `docs/`, in the reading order `keel.web.docs.READING_ORDER` names. |
| `GET /docs/{slug}` | One document, rendered from `docs/{slug}.md`. A slug is lower-case words joined by hyphens, so it can name nothing outside the directory; the resolved path is checked against `docs/` as well, which closes a symlink pointing out. Sibling `.md` links become pages here and anything reaching out of `docs/` becomes a link into the repository on GitHub. |
| `POST /ask` | Runs the question. Body is a form or JSON `{question, user_id, tags, mode}`. Replies with the HTML result partial when the request carries `X-Keel-Partial: 1` (what `keel.js` sends), JSON when `Accept: application/json`, otherwise the full chat page with the result in place (plain form post, JavaScript off). |
| `POST /api/ask` | Same body, always JSON: `{request_id, mode, user, text, refused, citations[], retrieved[], prompt_tokens, output_tokens, latency_ms, data, error}`. |
| `POST /api/agent` | Same body with mode forced to agent, always JSON: `{request_id, mode, user, text, refused, steps[], refused_tools[], prompt_tokens, output_tokens, latency_ms}`. Each step carries `tool`, `arguments`, `decision`, and either `result` or `queued_id`. |
| `GET /source/{chunk_id}` | The chunk text with document title, source, heading, page, position and ACL tags. Query `user` and `tags` name the caller; a chunk whose tags share none with the caller's is 403, an unknown id is 404. `Accept: application/json` returns the row as JSON. |
| `GET /health` | `{"status": "ok", "profile", "llm", "documents", "chunks", "quarantined", "ledger_seq"}`. `llm` is `llm.healthy()` under a two second guard: a probe still running after that counts as unhealthy. |
| `GET /admin` | Totals tiles, 14-day trend (inline SVG sparklines: requests, refusal rate, average latency, mean groundedness and relevance once judge scores exist), recent requests table, approvals, quarantine, ledger controls. |
| `GET /admin/request/{request_id}` | One logged request: user and tags, answer, retrieved chunk ids, citations, tool calls, judge scores, approvals it raised, and every ledger row that carries its request id. |
| `POST /admin/approvals/{id}/approve` | `ctx.approvals.decide(id, True, by)` then `execute` through the registry; the result lands on the row. Optional form field `by` (default `admin`). Redirects to `/admin#approvals`, or JSON with `Accept: application/json`. A row that is no longer pending answers 409. |
| `POST /admin/approvals/{id}/reject` | `decide(id, False, by)`. Nothing runs. |
| `POST /admin/quarantine/{chunk_id}/release` | Clears the chunk's quarantine flag so retrieval may return it again and appends a ledger row of kind `quarantine` with payload `{chunk_id, action: "release", by}`. Both writes happen in one transaction. |
| `POST /admin/ledger/verify` | Recomputes the whole hash chain and shows the `VerifyResult` on the admin page (JSON with `Accept: application/json`). |
| `GET /admin/ledger/export` | The ledger as JSONL, `application/x-ndjson`, one row per line in the shape `keel.safety.ledger.verify_file()` reads offline. |
| `POST /api/airgap-probe` | Body `{host}`. Attempts a connection to `host` at every guarded layer, in a child process started with `KEEL_AIRGAP=1` and this deployment's own `KEEL_AIRGAP_ALLOW_HOSTS`, so the allow list the report names is the one the appliance actually runs with, and returns `{host, guard, allow_hosts, allowed, attempts[], refused, layers, summary}`. Each attempt carries the layer, the guard that answered (`via`), the outcome and the refusal text. The guard is installed in the child rather than in the worker, so one visitor's probe cannot take the model connection away from another visitor's question. A host outside the allow list is unreachable, which is the property being demonstrated, and a host inside it is answered from the policy with no connection made. One probe per caller every three seconds; a faster caller gets 429, and a value that is not a bare host gets 400. |
| `POST /api/redact` | Body `{text}`. Runs `keel.safety.pii.redact` and returns `{redacted, findings[], counts, kinds}`. A pure function of its argument: no store, no model, no network, and the text is neither kept nor logged. Text longer than 4000 characters gets 400. |
| `GET /static/keel.css`, `GET /static/keel.js` | The stylesheet and the script. |

### Users and tags

Identity on the demonstration page is self-asserted: the user select and the extra tags field decide which ACL
tags a request carries, and every retrieval, citation link and source view honours those tags. That
is the right demo of permission filtering before generation, and the wrong thing to expose to the
open internet as-is. Put the app behind an authenticating reverse proxy that maps real users to tags
before anyone outside the team reaches it.

### Identity beyond loopback

On loopback (the demo) the user and tags come from the request itself. When `KEEL_HOST` is any other
address, the app ignores `user_id` and `tags` in the body and query string, since anyone can write
them, and takes identity only from a reverse proxy: the proxy sets `X-Keel-User` and `X-Keel-Tags`
(comma-separated) and proves itself with `X-Keel-Proxy-Token` equal to the `KEEL_PROXY_TOKEN`
environment variable. Requests without a valid proxy token run as `public`. An unset
`KEEL_PROXY_TOKEN` means no proxy is trusted and every non-loopback request is `public`.

```bash
curl -H "X-Keel-Proxy-Token: $KEEL_PROXY_TOKEN" -H "X-Keel-User: emma" -H "X-Keel-Tags: public,owner"      -X POST http://host:8400/api/ask -d '{"question": "..."}' -H "Content-Type: application/json"
```


### Hosted demo identity

`KEEL_DEMO_IDENTITY=1` (`Settings.demo_identity`) exists for one purpose: the public hosted demo of
the fixture corpus at keel.flow-through.com.au, where the point is to let a visitor switch between
`public` and `hr-officer` and watch permission filtering work on documents that hold nothing real.
With the flag on and the host beyond loopback, `trusted_identity` honours the demo user picker:
`user_id` resolves to one of the demo users (`public` [public], `hr-officer` [public, hr]) with that
user's fixed tags, and the extra tags field is ignored; a proxy identity with a valid
`X-Keel-Proxy-Token` still wins, and any other id runs as `public`. The admin guard is unaffected:
`/admin` still needs `X-Keel-Admin-Token` beyond loopback. **Never set this for a real deployment**:
it lets anyone on the network claim the `hr` tag. The chat page shows a banner naming the demo when
the flag is on, with an "Ask as both users" button that fires the restricted pay-round question as
`public` and as `hr-officer` through the same `/ask` path a typed question takes, and renders the
refusal and the cited answer side by side.

`KEEL_DEMO_READONLY=1` (`Settings.demo_readonly`) declares the demo's read-only posture. The web app
has no ingest route at all (`tests/test_web.py::test_web_app_exposes_no_ingest_route_and_every_write_sits_under_admin`
pins the full list of POST routes), and every write that exists, approve, reject and quarantine
release, sits under the admin guard; the agent's `create_ticket` still queues, and approving it needs
the admin token. Ingest happens through the CLI, or once at startup:

`KEEL_BOOTSTRAP_CORPUS=fixtures/corpus.yaml` (`Settings.bootstrap_corpus`) makes the lifespan
handler call `keel.bootstrap.ensure_demo_corpus`, which ingests the manifest through the injection
screen when the store holds zero documents and leaves a populated store alone. Ingest is idempotent
by checksum anyway; the count check keeps restarts quick.

`KEEL_LOCAL_LLM_TIMEOUT` (default 120 seconds) is the per-call timeout on the local model client;
the hosted demo runs a CPU-only server and sets it to 300.

Citation chips link to `/source/{chunk_id}?user=…&tags=…` with the asking user's tags, so a chip
opens for the user who earned it and answers 403 for anyone carrying fewer tags.

### Rendering and escaping

Templates autoescape everything. Answer text is escaped first, then each `[n]` marker that names a
citation is converted into a chip on the server; markers with no matching citation stay as plain
text. Agent replies render as text. Nothing from the store, the model or the request reaches the page
unescaped.

## Admin guard

The admin page has no login. On loopback that is deliberate: the appliance listens on `127.0.0.1` and
whoever can reach it already owns the box. The guard is config-free and applies the moment
`settings.host` (`KEEL_HOST`) is anything other than a loopback address (`127.0.0.0/8`, `::1`,
`localhost`): every `/admin` route then requires the header `X-Keel-Admin-Token` equal to the
environment variable `KEEL_ADMIN_TOKEN`, and answers 401 otherwise. An unset `KEEL_ADMIN_TOKEN` with a
non-loopback host locks the admin routes entirely, which is the safe failure. Chat, `/source`,
`/health` and the API stay open in both cases; the reverse proxy above is where those get their auth.

```powershell
$env:KEEL_HOST = '0.0.0.0'; $env:KEEL_ADMIN_TOKEN = 'a-long-random-string'
curl -H "X-Keel-Admin-Token: a-long-random-string" http://host:8400/admin
```

No secret is ever rendered: the admin page shows host, port, profile, model name, air-gap state and
the refusal threshold, and nothing from `.env`.

## Design notes

Dark, calm, system font stack, readable at 390 px and 1280 px. The result area on the chat page is
swapped by `fetch` so the page stays put while the CPU model answers; refusals get an amber left
border and a short note, agent steps list tool, arguments, decision state and result (or
`queued for approval #id` with a link to the admin page). The 14-day trend is five inline SVG
sparklines computed server-side (`keel/web/views.py`, `trend_sparklines`); no chart library.

## Tests

`tests/test_web.py` runs the app under FastAPI's `TestClient` with `build_context()` wired to a
`FakeLLM` and the real cached fastembed models, over the fixture corpus in a temporary data directory.
No network and no llama-server are needed.
