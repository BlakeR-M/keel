# Security policy

Keel is a retrieval-augmented generation and agent appliance built to run where the documents live:
on one machine with nothing leaving it, or inside a client's own Azure tenancy with managed identity
and no keys. This file says which versions receive fixes, how to report a problem, what the appliance
defends against, what it leaves to you, and the checklist for a secure deployment. The full threat
model with the file that implements each control lives in `docs/threat-model.md`.

## Supported versions

| Version | Supported |
| --- | --- |
| 0.1.x (current) | Yes: security fixes land on `main` and are tagged. |
| Anything earlier | Pre-release scaffolding; upgrade to 0.1.x. |

Keel is early software. Fixes ship as patch releases on the current minor
version; there is no long-term support branch yet.

## Reporting a vulnerability

Report privately, and give the maintainer time to fix before anything is public:

1. Preferred: GitHub private vulnerability reporting on the repository (Security tab, "Report a
   vulnerability"). It opens a private advisory thread with the maintainer.
2. Otherwise: email service@flow-through.com.au with "Keel security" in the subject.

Include the version or commit, the deployment profile (local or azure), steps to reproduce, and what
an attacker gains. You will hear back within five working days with a first assessment, and again when
a fix is available.
Please keep client documents and personal information out of the report; a redacted reproduction on
the fixture corpus (`fixtures/corpus/`) is ideal.

## What the appliance defends

- **Permission-filtered retrieval, before generation.** Every chunk carries ACL tags; every request
  carries a user with tags; chunks the user is not entitled to are dropped inside retrieval, before
  fusion and before anything reaches the model. Refusal is the outcome when the entitled set holds no
  answer (`keel/retrieval/hybrid.py`, `keel/answer/engine.py`).
- **Policy at the tool boundary and an approval queue.** Every tool call the model proposes passes a
  per-deployment allowlist and argument rules (SELECT-only SQL with a table allowlist and a read-only
  authoriser, HTTP host allowlist, arithmetic-only calculator). Tools marked `write` are queued for a
  person and run only after approval, never unattended (`keel/agent/policy.py`,
  `keel/agent/approvals.py`, `keel/agent/tools.py`).
- **Injection quarantine.** Retrieved text is data. A heuristic screen, with an optional LLM judge,
  runs on every chunk at ingest; flagged chunks are stored quarantined, stay out of retrieval and the
  prompt, and are listed on the admin page with the reason (`keel/safety/injection.py`).
- **Air-gap mode.** With `KEEL_AIRGAP=1` the process refuses every outbound connection other than
  loopback and the named allow list, at the socket, asyncio, urllib and httpx layers, before a packet
  leaves; the embedding models load from the local cache (`keel/airgap.py`).
- **Tamper-evident ledger.** Every request, retrieval set, tool call, approval, ingest and answer is
  appended to a hash-chained ledger; `keel verify-ledger` names the first broken link, and an
  exported chain verifies offline with nothing but Python (`keel/safety/ledger.py`).
- **PII redactor.** Deterministic detection and redaction of Australian identifiers (TFN, Medicare,
  ABN, phone, email, cards) with check-digit validation, for a redaction pass over documents that
  need it before they enter the corpus (`keel/safety/pii.py`).
- **No keys in the cloud profile.** The Azure profile authenticates with `DefaultAzureCredential`
  (managed identity in Azure); the Bicep template disables local key auth on Azure OpenAI and Azure
  AI Search (`keel/providers/azure.py`, `deploy/azure/main.bicep`).
- **Inference log for every request**, with user, tags, retrieved chunk ids, tool calls, answer,
  latency and tokens, so a question about "who saw what" has an answer (`keel/observe/log.py`).

## What the appliance leaves to you

- **It is not a data loss prevention system.** Keel decides what a user may retrieve. Once an entitled
  user has an answer on screen, copying it, photographing it or emailing it is outside Keel's reach.
  Tag documents by the narrowest role that needs them; the ACL is only as good as the tags.
- **The admin page trusts localhost by default.** Admin routes (approvals, quarantine release, ledger
  verify and export) are open to any request arriving on loopback. Beyond loopback they require the
  `X-Keel-Admin-Token` header matching `KEEL_ADMIN_TOKEN`. Anyone with a shell on the appliance is an
  admin; protect the machine.
- **Identity is self-asserted in the 0.1.x build.** The demonstration page offers a user picker; there is no
  password. The guard is the machine's own login plus the loopback bind. Per-person login is
  a later release; until then, one appliance per business on a machine that business controls.
- **The LLM can still be wrong.** Citations point at the chunks the answer drew on and the refusal gate
  stops guessing when retrieval is weak, but a fluent answer can still misread a passage. Read the
  cited source for anything that matters, and run `keel eval` after model or corpus changes.
- **Heuristics are one layer.** The injection screen catches the planted fixture and the common
  patterns; a novel phrasing may pass. ACL filtering, the tool policy and approvals limit what a
  passed injection can achieve.
- **The air-gap guard is in-process.** It covers what Keel and the libraries in its process do. Other
  processes on the machine, and libraries that resolve first and connect through paths outside the
  standard socket layer, are the host firewall's job. For a hard boundary add an egress-deny rule or
  run the container with no network.
- **The database file is the crown jewels.** `keel.db` holds documents, chunks, embeddings, the log
  and the ledger in plain SQLite. Anyone who can read the file can read the corpus. Encrypt the disk,
  restrict the data directory, and treat backups the same way.
- **The ledger proves integrity, not confidentiality, and only up to its head.** Rows removed from
  the tail leave a valid shorter chain; anchor `head()` externally when that matters.
- **No rate limiting or multi-tenancy.** One appliance serves one business; a busy loop can starve
  the CPU-bound model. Put a reverse proxy with limits in front when exposure grows.
- **Model files are third-party artifacts.** Verify the checksum of any GGUF you download and keep
  the model directory read-only to the app.

## Secure deployment checklist

On-premise:

- [ ] `KEEL_AIRGAP=1`, and the app bound to `127.0.0.1` (`KEEL_HOST` default). Prove it with
      `pytest tests/test_airgap.py -q` on the box and one live check from `docs/onprem.md`.
- [ ] `KEEL_DATA_DIR` on an encrypted disk, readable by the service account only; backups in a
      folder the owner holds, same protection.
- [ ] `policy.yaml` in the data directory reviewed with the owner: `write_tools_require_approval:
      true`, `http_get` off the allowlist unless a host is named, `sql_readonly` tables limited to
      aggregate views.
- [ ] Documents tagged by the narrowest role; a user from the narrowest role tested for a refusal on
      a wider role's document before handover.
- [ ] Documents with personal information either kept out of the corpus, or tagged narrow and
      redacted with the PII redactor before ingest, as the owner decides in writing.
- [ ] `KEEL_ADMIN_TOKEN` set to a long random value the moment the app listens beyond loopback.
- [ ] `keel verify-ledger` scheduled weekly and after every restore; the ledger head recorded
      somewhere outside the machine at each backup.
- [ ] Model file checksum recorded; llama-server started by `deploy/onprem/run.ps1` with the model
      directory read-only.
- [ ] Windows Update on; the appliance user is a standard account; the install account is separate.

Azure profile:

- [ ] Managed identity only; local key auth stays disabled on Azure OpenAI and Azure AI Search (the
      template's default). No secrets in the container environment.
- [ ] `enablePrivateEndpoints=true` for a client that needs the services off the public network.
- [ ] Ingress limited to the client's identity provider or IP range; `KEEL_ADMIN_TOKEN` set.
- [ ] Log Analytics retention agreed with the client; the inference log holds question text.
- [ ] `keel eval` run against the golden set after every model or deployment change, report kept.
