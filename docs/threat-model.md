# Threat model

What Keel protects, who it protects it from, where the trust boundaries sit, which control answers
each threat and the file that implements it, what stays as residual risk, and how to verify every
control from a fresh shell. `SECURITY.md` is the short public version; this page is the working one.

## Assets

| Asset | Where it lives | Why it matters |
| --- | --- | --- |
| Client documents and their chunks (text, headings, pages) | `documents`, `chunks` tables in `<data_dir>/keel.db`; source files under the client's corpus folder | The confidential content the appliance exists to answer from |
| ACL tags on documents and chunks | `documents.acl_tags`, `chunks.acl_tags` | The permission model; wrong tags mean wrong disclosure |
| Embeddings and the FTS index | `chunks.embedding`, `chunks_fts` | Derived from the documents; leak the same information |
| Inference log | `inference_log` | Who asked what and what came back; personal information by nature |
| Ledger | `ledger` | The audit trail; its integrity is the point |
| Approval queue | `approvals` | Pending write actions and who decided them |
| Tool policy | `<data_dir>/policy.yaml` | Decides what the agent may do; an edit widens the blast radius |
| The model endpoint | llama-server on 127.0.0.1:8081, or Azure OpenAI | Availability of answers; in Azure, a billable resource |
| The machine and its login | The on-premise PC or the Container App | Everything above sits behind it |
| Cloud identity (Azure profile) | User-assigned managed identity | The only credential; there are no keys |

## Actors

| Actor | Trust | What they might do |
| --- | --- | --- |
| Staff user in a role (owner, coordinator, carer; owner, front desk, trainer) | Trusted for their tags | Ask about documents outside their role, on purpose or by accident |
| Curious or malicious insider | Same access as their role | Try to widen access by picking another user, phrasing questions to leak, or reading the disk |
| Document author (including suppliers and third parties whose files are ingested) | Untrusted content | Plant instructions aimed at the model inside a document |
| The LLM | Untrusted component | Hallucinate, misread a passage, follow injected instructions, propose harmful tool calls |
| Tools and their backends (SQLite reporting export, HTTP hosts) | Constrained by policy | Return large or hostile results; be used to read beyond intent |
| Operator or admin | Trusted on the machine | Approve write actions, release quarantine, edit policy, back up and restore |
| Network attacker | Untrusted, outside | Reach the app or the model port, or receive data the box sends out |
| Anyone with the database file | Untrusted | Read the corpus, alter the log or ledger |

## Trust boundaries

1. **Documents in.** Files cross from "content someone wrote" to "chunks the model may see". Controls
   at the boundary: the injection screen (quarantine), tags applied from the manifest, checksum
   idempotence, and the PII redactor for a pass before ingest. Under air-gap, remote URLs are refused
   at load.
2. **Users to requests.** A person becomes a `User(user_id, tags)`. In the 0.1.x build identity is
   self-asserted through the user picker; the boundary that holds is the machine's login and the
   loopback bind. Everything downstream (retrieval, tools, log) treats the tags as authoritative.
3. **Retrieval to generation.** Only chunks that pass the ACL check and are not quarantined are
   fused, reranked and placed in the prompt. Below `min_relevance` the engine refuses without calling
   the model.
4. **The model to tools.** Everything the model emits is untrusted. A proposed tool call meets the
   policy (allowlist, argument rules, write flag), then the registry validates arguments against the
   tool's schema, then the tool's own guard runs (AST-only arithmetic, SELECT-only SQL under a read-only
   authoriser and a step budget, host allowlist for HTTP). Write tools stop at the approval queue.
5. **Admin.** Approvals, quarantine release, ledger verify and export sit behind `require_admin`:
   open on loopback, token beyond it. The policy file and the database are files on disk behind the
   operating system's permissions.
6. **The box to the network.** With `KEEL_AIRGAP=1` the process refuses outbound connections other
   than loopback and the named allow list. In the Azure profile the boundary is the VNet and private
   endpoints when enabled, and managed identity for every call.

## Threats and controls

| # | Threat | Control | Implemented in | Verified by |
| --- | --- | --- | --- | --- |
| T1 | A user retrieves, or the answer cites, a chunk outside their tags | ACL filter on both candidate lists before fusion, a second check on fused hits, tags carried on every request; refusal when the entitled set is empty or weak | `keel/retrieval/hybrid.py` (`allowed`, `Retriever.retrieve`), `keel/answer/engine.py`, `keel/agent/tools.py` (`search_docs` runs under the user's tags) | `tests/test_retrieval.py` (public user never sees the HR document; unentitled and quarantined hits dropped before fusion), `tests/test_answer.py` (ACL-filtered hits lead to refusal; retriever gets the user's tags), `tests/test_agent.py` (search_docs runs under the user's tags) |
| T2 | Indirect prompt injection: a document instructs the model | Heuristic screen with weighted patterns and an optional LLM judge at ingest; flagged chunks stored `quarantined = 1`, excluded from BM25 and vector candidates and from the prompt; admin release is recorded | `keel/safety/injection.py`, `keel/ingest/pipeline.py` (screen hook), `keel/retrieval/hybrid.py`, `keel/web/app.py` (release with ledger row) | `tests/test_safety.py` (planted fixture quarantined with reason; clean fixtures pass; base64, hidden unicode, template tokens flagged; judge combination), `tests/test_ingest.py` (screen hook marks quarantined chunks), `tests/test_answer.py` (quarantined hits stay out of the prompt) |
| T3 | The model calls a tool outside the deployment's intent, or with dangerous arguments | Per-deployment allowlist, argument rules, tool-call budget; registry schema validation; calculator evaluates numbers and operators only; `sql_readonly` is SELECT-only over allowlisted tables with a read-only authoriser, row cap and step budget; `http_get` needs an allowlisted host and is off under air-gap | `keel/agent/policy.py`, `keel/agent/tools.py`, `keel/agent/loop.py` | `tests/test_policy.py` (disallowed tool refused and logged; SQL table allowlist; DDL and multi-statement refused; host allowlist; air-gap refusal), `tests/test_agent.py` (unknown tool refused and the loop continues; budget enforced) |
| T4 | A write action runs unattended | Write tools are queued as `pending`; a person approves or rejects; an approved call runs once and its result is stored; every transition is a ledger row with the decider | `keel/agent/approvals.py`, `keel/agent/policy.py` (`needs_approval`), `keel/agent/loop.py`, `keel/web/app.py` (approve and reject routes) | `tests/test_agent.py` (write tool queued and never executed; decide and execute; reject path; write tool without a queue refused), `tests/test_web.py` |
| T5 | Data leaves the box over the network (model, tool, library or dependency phoning home) | Air-gap guard at socket, asyncio, urllib and httpx layers; loopback and named hosts only; offline flags for the model cache; `http_get` refuses before connecting | `keel/airgap.py`, `keel/agent/tools.py`, `deploy/onprem/docker-compose.yml` (`KEEL_AIRGAP=1`, allow list `llama`) | `tests/test_airgap.py` (socket, UDP, create_connection, httpx transport and plain client, urllib, asyncio all refused; loopback allowed; enable and disable restore originals; compose files declare air-gap), `tests/test_ingest.py` (remote URL refused under air-gap) |
| T6 | The audit trail is altered, reordered or trimmed | Hash chain over kind, request id, canonical payload and previous hash from a genesis value; verify names the first broken link; export verifies offline with the standard library | `keel/safety/ledger.py` | `tests/test_safety.py` (append and verify; tampered payload detected; export then verify; tampered export fails; concurrent appends keep one chain) |
| T7 | Personal information in the corpus reaches a user or the log | Narrow tags for documents that hold it; deterministic PII detection and redaction with check-digit validation as a pass before ingest; per-deployment rules in the operator's runbook | `keel/safety/pii.py` | `tests/test_safety.py` (TFN, Medicare, ABN, card, phone, email detection and redaction) |
| T8 | Unauthorised admin action (approve, release quarantine, export the ledger) | Admin routes open on loopback only; beyond loopback a constant-time token check; the appliance binds to loopback by default | `keel/web/app.py` (`require_admin`), `keel/web/views.py` (`is_loopback`), `deploy/onprem/run.ps1` (`--host 127.0.0.1`) | `tests/test_web.py` |
| T9 | Ungrounded or wrong answers presented with confidence | Refusal below `min_relevance` without calling the model; citations resolved to real chunk ids; uncited answers get their top sources attached; JSON mode validates against a schema with one retry; eval harness with a regression gate | `keel/answer/engine.py`, `keel/answer/schema.py`, `keel/evals/` | `tests/test_answer.py` (refusal paths; citation mapping; JSON retry), `tests/test_evals.py` (broken retriever fails the gate; must-not-include catches a planted override string) |
| T10 | Cloud credentials leak or are misused | No keys anywhere: `DefaultAzureCredential`, user-assigned managed identity, local auth disabled on Azure OpenAI and Search, private endpoints as an option | `keel/providers/azure.py`, `deploy/azure/main.bicep` | `tests/test_azure_provider.py` (providers against mocked SDK clients; credential path), CI `bicep build` and `bicep lint` |
| T11 | A hostile file at ingest (oversized, malformed, remote) | Format loaders with size accounting, checksum idempotence, chunk size limits; remote URLs refused under air-gap | `keel/ingest/loaders.py`, `keel/ingest/pipeline.py`, `keel/airgap.py` | `tests/test_ingest.py` (every format from disk; duplicate re-ingest adds nothing; remote URL refused under air-gap) |
| T12 | The tool policy is loosened without anyone noticing | Policy is one file in the data directory, versioned alongside the deployment and tested; refusals are logged | `keel/providers/factory.py` (`_load_policy`) | `tests/test_policy.py` (search_docs allowed, create_ticket queued, http_get denied, SQL allowlist, empty host list refuses every host) |

## Residual risks

- **Self-asserted identity in the 0.1.x build.** A person at the machine can pick another user in the
  picker. The compensating control is physical: one appliance per business, loopback only, on a
  machine with the business's own login. Per-person login is the next release.
- **Novel injection phrasings.** The screen is a heuristic layer with an optional judge; a passage
  written to slip past both can reach the prompt. The blast radius is bounded by the ACL (the model
  sees only what the user may see anyway), the tool policy and the approval queue.
- **Model correctness.** A fluent misreading is still possible. Citations and the refusal gate reduce
  it; the eval harness measures it; the person reading the answer decides.
- **The database file.** SQLite is unencrypted at rest. Disk encryption and file permissions carry
  this risk; backups need the same care.
- **In-process air-gap.** Other processes and unusual connection paths are outside the guard. A host
  firewall or a `--network none` container gives the hard boundary.
- **Ledger tail.** A shorter chain that ends earlier still verifies. Recording the head hash outside
  the machine at each backup closes this.
- **Availability.** No rate limits; a CPU-bound model can be kept busy by one user. Acceptable for a
  single-business appliance; a reverse proxy with limits when exposure grows.
- **Supply chain.** The GGUF model, the ONNX embedding models and the Python dependencies are
  downloaded artifacts. Pin versions, record checksums, and build the container from a staged model
  cache for air-gapped hosts.
- **Reporting exports.** `sql_readonly` protects the appliance from writes and from unlisted tables;
  it cannot know whether an allowlisted table itself holds personal rows. Allowlist aggregate views only.

## How to verify each control

Every command runs from the repository root with the project interpreter (`.venv\Scripts\python.exe`
on Windows, `.venv/bin/python` elsewhere) and needs no network and no model server; integration tests
skip on their own when llama-server is absent.

| Control | Command |
| --- | --- |
| ACL before generation (T1) | `python -m pytest tests/test_retrieval.py tests/test_answer.py -q` |
| Injection quarantine (T2) | `python -m pytest tests/test_safety.py -q -k "quarantin or flagged or judge or fixture"` |
| Tool policy and tool guards (T3, T12) | `python -m pytest tests/test_policy.py -q` |
| Approval queue (T4) | `python -m pytest tests/test_agent.py -q -k "queued or approval or write_tool"` |
| Air-gap (T5, T11) | `python -m pytest tests/test_airgap.py tests/test_ingest.py -q -k "airgap or refused or compose"` |
| Ledger integrity (T6) | `python -m pytest tests/test_safety.py -q -k ledger`, then on a live store: `keel verify-ledger` |
| PII redaction (T7) | `python -m pytest tests/test_safety.py -q -k "tfn or email or card or abn or detect"` |
| Admin guard (T8) | `python -m pytest tests/test_web.py -q -k admin` |
| Grounding, refusal, JSON mode, eval gate (T9) | `python -m pytest tests/test_answer.py tests/test_evals.py -q` |
| Azure profile without keys (T10) | `python -m pytest tests/test_azure_provider.py -q`; `bicep build deploy/azure/main.bicep` |
| Everything at once | `python -m pytest -m "not integration" -q` (what CI runs) |
