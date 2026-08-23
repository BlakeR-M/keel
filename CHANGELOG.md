# Changelog

All notable changes to Keel. The format follows Keep a Changelog; versions follow semantic versioning.

## [Unreleased]

### Added

- **The site leads with the security posture.** `GET /` is now an overview page rather than the
  question box: what Keel is, three demonstrations that run the production code paths live, how a
  question travels, the review summary with its open findings, a control table mapping every claim
  to its code and its proving test, the three deployment shapes, a quickstart and the documentation
  index. The appliance itself moved to `GET /chat` and is unchanged. A reader arriving cold from a
  link reads the posture before any form (`landing.html`, `keel/web/app.py`).
- **The permission comparison is the hero of the overview** rather than a button on the chat page.
  One click asks the restricted pay-round question as `public` and as `hr-officer` through the same
  `/ask` path a typed question takes, and the panel under it names where the filter runs, what the
  refusal costs, and the second check at the generation boundary.
- **The air gap is demonstrable, not only described** (`POST /api/airgap-probe`,
  `keel/web/airgap_probe.py`): a visitor names a host and Keel attempts a real connection at every
  guarded layer, reporting the layer, the guard that answered and the refusal text from the
  exception itself. The attempts run in a child process started with `KEEL_AIRGAP=1`, so a worker's
  own guard state is untouched and one visitor's probe cannot take the model connection away from
  another visitor's question. A host outside the allow list is unreachable, which is the property
  being shown; a host inside it is answered from the policy with no connection made. An address
  reports that name resolution had nothing to look up rather than counting as a layer that let it
  past. One probe per caller every three seconds.
- **Redaction is demonstrable too** (`POST /api/redact`): text in, redacted text and the span of
  every finding out, through the same `keel.safety.pii.redact` the pipeline uses. The sample text
  makes the point that shape matching alone would miss, keeping a purchase order and an invoice
  number intact beside a tax file number, a Medicare number and an ABN that pass their check digits.
  A pure function: no store, no model, no network, and the text is neither kept nor logged.
- **The documentation is part of the site** (`GET /docs`, `GET /docs/{slug}`, `keel/web/docs.py`):
  the Markdown under `docs/` renders as pages, with sibling links kept on the site and links reaching
  out of `docs/` pointed at the repository. A slug is lower-case words joined by hyphens and the
  resolved path is checked against `docs/`, so a request path can name nothing outside it. The
  Dockerfile now copies `docs/` into the image, with a test that says so.
- **[`docs/tutorial.md`](docs/tutorial.md)**: install, load the fixture corpus, ask a question you are
  not entitled to, run the agent and approve its write, verify and export the ledger, bring your own
  documents, serve it, turn the network off, and measure the answers.
- **Hosted demo** (`deploy/railway/`): a two-service Railway layout, `keel-llm` (llama.cpp CPU server
  fetching Qwen2.5-3B-Instruct Q4_K_M onto a volume) and `keel` (the root Dockerfile, now non-root
  with an entrypoint that honours `PORT`, data on `/data`, `railway.json` with a `/health` check).
  `KEEL_DEMO_IDENTITY=1` honours the demo user picker beyond loopback (fixed tags per
  demo user, extra tags ignored, admin guard unchanged) with a banner naming the demo;
  `KEEL_DEMO_READONLY=1` declares the read-only posture (a test pins every POST route);
  `KEEL_BOOTSTRAP_CORPUS` ingests a manifest into an empty store at startup (`keel/bootstrap.py`);
  `KEEL_LOCAL_LLM_TIMEOUT` reaches the model client. Both demo flags are for the hosted demo of the
  fixture corpus and never for a real deployment.
- **One-click permission compare**, since moved to the overview page and covered above.
- **HEAD on every page a link can point at** (`/`, `/chat`, `/docs`, `/docs/{slug}`, `/health`) for link checkers and unfurl bots that read a 405 as a dead page;
  **meta description and Open Graph tags** in `base.html` so a pasted link unfurls with a title
  and a summary.
- **Health as a page for people, JSON for machines**: a browser following the nav link gets a
  styled page with the same numbers; monitors and curl keep the exact JSON shape they had.
- **Raised errors render like returned ones**: an app-level handler routes `HTTPException` through
  the shared error shape, so a visitor clicking Admin without the token reads a styled page naming
  the guard instead of a bare JSON detail, and API callers asking for JSON keep JSON.
- **First-arrival intro on the hosted demo**: a dismissible dialog stating what Keel is, what it is
  for, how it is built, and how it runs a business's internal operations. Shown once per browser
  (the seen flag is written at show time), reachable again from the banner's "About this demo"
  link, absent without JavaScript, and demo-gated like the banner and compare.

### Fixed

- **Docs squared with the code** (2026-08-20 review): the adversarial findings table
  published as `docs/security-review.md` (the README linked to a table that had been dropped); test
  counts corrected to the measured 585 tests across 18 files; the open-findings sentence now names
  one medium and one low; the Railway sizing paragraph carries measured latency (three to nine
  seconds per cited answer over the public URL); the README opens with three short sentences.
- **Chat page copy trimmed**: the hint line under the form (identity echo, latency note, keyboard
  shortcut ad) is gone, along with the script that maintained it; Ctrl+Enter still sends the form,
  unadvertised. The working state says what it is doing and counts the seconds, nothing more.

## [0.1.0] - 2026-08-18

First release: the sovereign RAG and agent appliance, end to end on one machine with a local model,
code-complete for Azure against mocked SDK clients, with evaluation, policy and audit built in.

### Added

- **Ingest** (`keel/ingest/`): PDF, DOCX, Markdown, HTML and plain text loaders; section-aware
  chunking with overlap; every chunk carries source, heading or page, checksum and ACL tags;
  re-ingest is idempotent by document checksum; manifest ingest (`documents:` list with `acl_tags`);
  a screen hook that quarantines flagged chunks at ingest.
- **Retrieval** (`keel/retrieval/`): hybrid BM25 (SQLite FTS5) plus vector search (fastembed
  `bge-small-en-v1.5`, embeddings in SQLite), ACL filtering before reciprocal rank fusion, optional
  cross-encoder rerank (`ms-marco-MiniLM-L-6-v2`), relevance scores in the unit interval.
- **Answer** (`keel/answer/`): grounded answers with `[n]` citations resolved to chunks, a refusal
  path below `min_relevance` that never calls the model, JSON mode with schema validation and one
  retry.
- **Agent** (`keel/agent/`): tool-calling loop with typed tool schemas (`search_docs`, `calculator`,
  `sql_readonly`, `http_get`, `create_ticket`); policy at the tool boundary (allowlist, argument
  rules, call budget); write tools wait in an approval queue and run only after a person approves.
- **Safety** (`keel/safety/`): indirect prompt-injection screen (weighted heuristics plus an optional
  LLM judge) with quarantine; Australian-format PII detection and redaction with check digits;
  hash-chained ledger with `keel verify-ledger` and offline verification of exports.
- **Evals** (`keel/evals/`): golden Q/A set (drafted from the corpus, hand-editable YAML), hit@k,
  groundedness, relevance and refusal metrics, latency and tokens, LLM-as-judge with the local model
  and Gemini as an optional second judge, HTML report and JSON summary, a regression gate that fails
  on a drop past the threshold.
- **Observability** (`keel/observe/`, `keel/web/`): inference log per request (user, tags, retrieved
  chunk ids, tool calls, answer, judge scores, latency, tokens) and an admin page with recent requests,
  a fourteen-day trend, the quarantine list with release, the approval queue with approve and reject,
  and ledger verify and export.
- **Web and CLI**: FastAPI and Jinja chat page with citation chips and the refusal state; `keel`
  command line with `ingest`, `ask`, `agent`, `approvals` (list, approve, reject), `verify-ledger`,
  `eval`, `export-log`, `status` and `serve`.
- **On-premise deploy** (`deploy/onprem/`): native `run.ps1` and `run.sh` that start llama-server and
  the app without Docker; Docker Compose stack (app, llama.cpp server, data volume, GPU overlay);
  `KEEL_AIRGAP=1` egress guard at the socket, asyncio, urllib and httpx layers with a test suite that
  proves it; `demo.ps1` and `make demo` from a fresh clone.
- **Azure deploy** (`deploy/azure/`): Bicep for Container Apps, Azure OpenAI deployments, Azure AI
  Search, Key Vault, a user-assigned managed identity and optional private endpoints; `deploy.ps1`
  with what-if, deploy and a smoke test; Azure providers behind the same `LLMProvider`,
  `EmbeddingProvider` and `VectorIndex` contracts, unit-tested against mocked SDK clients,
  `DefaultAzureCredential` and no keys.
- **AWS stub** (`keel/providers/aws.py`, `deploy/aws/README.md`): interfaces and the mapping to
  Bedrock and OpenSearch Serverless; honest about being a stub.
- **CI** (`.github/workflows/ci.yml`): ruff, unit tests without an LLM, `bicep build` and `bicep
  lint` from the release binary, and the eval gate against fakes with report upload.
- **Docs**: `README.md`, `docs/onprem.md`, `deploy/azure/README.md`, `docs/threat-model.md`,
  `SECURITY.md`, this changelog.
- **Front door and walkthroughs**: the top-level `README.md` written for a reviewer (controls with
  the code and the test behind each, the sixty-second demo, architecture diagrams, the evaluation
  numbers from the 2026-08-18 run, what is stubbed or unverified, the repository map);
  `docs/README.md` as the documentation index; `docs/architecture.md` (module walkthrough, provider
  contracts, data model, the `/api/ask` and agent-loop lifecycles, ledger kinds);
  `docs/demo-script.md` (ninety-second recording script and questions to try);
  `docs/deploy-azure.md` (pointer to `deploy/azure/README.md`).

### Security review before publish

- Three adversarial test files (`tests/redteam_*.py`, 105 attack tests, 174 cases once
  parametrized) were written against the ACL,
  tool-policy, approvals, ledger, injection-screening and air-gap paths; 27 findings, 25 fixed in
  the same release and two accepted as strict xfails, one medium covered by the ingest LLM judge
  and one low releasable from the admin quarantine list (`docs/security-review.md`): ledger append deadlock
  with a shared-connection reader; self-asserted identity beyond loopback (now `X-Keel-User` /
  `X-Keel-Tags` behind `KEEL_PROXY_TOKEN`); calculator stacked-power resource attack; injections in
  headings and titles; silent no-op when re-ingesting with narrower tags; SQL result value caps and
  denied `zeroblob`/`randomblob`/`load_extension`; approval execution re-checks the live policy; DNS
  lookups covered by the air-gap guard; ledger hash material changed to a canonical JSON array of the
  four chained fields so field boundaries are explicit.
- Answer engine retries once when the model returns only a citation marker; the eval harness showed
  the fix (must_include 0.73 to 1.00, groundedness 0.85 to 1.00 on the 3B).

### Known limits in this release

- Identity is self-asserted on loopback (a user picker, no login); beyond loopback it comes only from
  a reverse proxy holding `KEEL_PROXY_TOKEN`. A built-in login is the next release.
- The Azure profile is code-complete and unit-tested against mocks; the first live deployment
  happens on a machine with the Azure CLI signed in.
- The Docker Compose stack is validated by a YAML parser and review; the first `docker compose up`
  on a Docker host is the remaining check.
- The AWS profile is a stub with interfaces and a README.
- The client overlays' user lists are mirrored into the web app's user table by hand at install; a
  loader for `keel.yaml` users is a follow-up.
- The PII redactor is a module and a redaction pass; a `--redact` flag on `keel ingest` is a follow-up.
