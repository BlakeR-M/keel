# Tutorial

This walks a fresh clone up to a running appliance answering questions about your own documents,
with the controls visible at each step. Everything here runs on one machine. Nothing reaches an
external service, and the last section turns the network off entirely and shows the same commands
still working.

Reading time is about fifteen minutes. Running it takes about the same, plus a model download if you
want generated answers rather than retrieval alone.

## Before you start

- Python 3.11 or newer.
- About 500 MB of disk for the two embedding models, which download once and then read from cache.
- A model server if you want generated answers. Any OpenAI-compatible endpoint works, and
  [`deploy/onprem/`](../deploy/onprem/run.sh) starts a llama.cpp server for you. Retrieval, permission
  filtering, screening, the ledger and the approval queue all work with no model at all.

## 1. Install

```bash
git clone https://github.com/BlakeR-M/keel.git
cd keel
python -m venv .venv
.venv/bin/pip install -e ".[dev]"        # .venv\Scripts\pip on Windows
```

Confirm the appliance can see itself:

```bash
keel status
```

That prints the profile, the data directory, the air-gap state, whether a model is reachable, the
corpus counts and the ledger head. A fresh install reports an empty store, which is correct.

Run the suite while you are here. It is the fastest way to know the clone is sound:

```bash
.venv/bin/python -m pytest -m "not integration"
```

## 2. Load the fixture corpus

The repository ships five small documents that hold nothing real: a council procurement policy, a
clinic data-handling policy, an HR salary-bands document tagged `hr`, an operations note, and a
supplier note with a prompt injection planted in it on purpose.

```bash
keel ingest --manifest fixtures/corpus.yaml
```

Read the output rather than skipping it. Three things happened:

- Each document was split into section-aware chunks, and each chunk inherited its document's
  entitlement tags. The salary-bands chunks carry `hr`, so only a user holding that tag can retrieve
  them.
- Every chunk was screened for prompt injection **together with its own heading and the document
  title**, because an instruction written into a heading rides into every prompt through the source
  label. The planted supplier note trips the screen and lands in quarantine.
- One ledger row was written inside the same transaction as the ingest. A ledger failure rolls the
  ingest back, so the store and the audit trail cannot disagree.

Quarantined chunks stay out of retrieval until a person releases them from the admin page. That is
the intended posture: a suspect document is visible to an operator and invisible to the model.

## 3. Ask a question, then ask one you are not entitled to

```bash
keel ask 'How many written quotes does a $20,000 purchase need at Northbank Council?' --tags public
```

You get an answer with a citation to the exact chunk it came from. Now the restricted one, as a user
holding only the `public` tag:

```bash
keel ask 'What is the confidential review code for the 2026 pay round?' --tags public
```

Keel says the sources do not cover the question. Read what that refusal costs: nothing entitled
cleared the relevance line, so no prompt was built and no model call was made at all. There is no
generated text for a cleverly worded follow-up to steer.

Add the tag and ask again:

```bash
keel ask 'What is the confidential review code for the 2026 pay round?' --tags public,hr
```

Now it answers, with a citation.

The difference between those two runs is not a filter applied to the answer. The entitlement filter
runs inside the BM25 query and inside the vector query, before the two result sets are fused and
before the reranker sees anything, so an unentitled chunk was never a candidate. A second check runs
at the generation boundary, added as finding A7 of the security review, so a retrieval bug on its own
is not enough to put a restricted chunk in front of the model.

## 4. Run the agent, and approve its write

Agent mode lets the model call five typed tools under a policy that is written down rather than
implied: `search_docs`, `calculator`, `sql_readonly`, `http_get` and `create_ticket`.

```bash
keel agent "Create a support ticket titled 'Printer down' saying the level 2 printer is jammed."
```

The step table shows what the model called and what came back. The ticket does not exist yet. Ticket
creation is a write tool, so it queued:

```bash
keel approvals list --status pending
keel approvals approve 1 --by owner
```

The policy is re-checked at the moment of execution rather than at the moment of approval, which was
finding P3 of the review. An approval cannot outlive the policy that permitted it: tighten the policy
after someone approves an action and the approval turns into a recorded refusal instead of a write.

The read-only SQL tool is worth trying as well. It runs under a SQLite authorizer with statement and
result-size caps, so it can read and cannot write, and it cannot be talked into allocating its way
through the machine's memory.

`http_get` is the one tool that reaches outside the box, so it is the one to look at closely. It is
bound to a host allowlist, an empty allowlist refuses every host, and with `KEEL_AIRGAP=1` it refuses
before any connection is attempted at all. Section 8 turns that on.

## 5. Verify the audit trail

```bash
keel verify-ledger
```

Every request, retrieval, answer, tool call and approval appended a row that hashes a canonical JSON
array of the chained fields. Because the field boundaries are explicit in the hash material, moving a
character from one field into the next changes the hash rather than producing the same one, which was
finding L3.

Export it and verify the file on its own, which is the shape an assessor asks for:

```bash
keel verify-ledger --export ledger.jsonl
```

That writes the ledger as JSON lines and then verifies the written file offline with the same
function, so the export is checked rather than assumed. The format is strict JSON: it refuses to
write `NaN` or `Infinity`, so a strict parser on the other end never chokes on it, and a file resaved
with a byte-order mark still verifies. The inference log exports separately with
`keel export-log`.

## 6. Bring your own documents

Point ingest at one or more files, at an http or https URL, or at a manifest. PDF, DOCX, Markdown,
HTML and plain text all load.

```bash
keel ingest ./handbook.pdf --tags public
keel ingest ./salary-bands.docx --tags hr,finance
keel ingest ./policy-a.md ./policy-b.md --tags public
keel ingest https://example.gov.au/procurement-policy --tags public
```

Tags are the whole access-control model, so decide them deliberately. A tag is a capability: a user
holding `hr` can retrieve any chunk tagged `hr`. Re-ingesting the same bytes reports a duplicate and
adds nothing, and re-ingesting with different `--tags` retags the stored document and its chunks in
one transaction rather than silently doing nothing, which was finding A2.

For anything larger than a handful of files, write a manifest instead: a YAML list of documents with
their paths, titles and tags, which keeps the tagging decision in a file you can review rather than in
shell history. Copy [`fixtures/corpus.yaml`](../fixtures/corpus.yaml) and edit it, then run
`keel ingest --manifest your-corpus.yaml`.

Turn on the language-model judge at ingest when the documents come from somewhere you do not control:

```bash
keel ingest --manifest supplier-notes.yaml --judge
```

The heuristics catch trigger words and structure. The judge catches paraphrased instructions that
carry no trigger word, which is the open medium finding in the review, recorded honestly rather than
quietly closed.

## 7. Serve it

```bash
keel serve                    # http://127.0.0.1:8400
```

On loopback the user picker on the demonstration page is honoured as given, because loopback identity
is the operator's own login. Beyond loopback nothing in a request body or query string can name a
user or a tag. Identity comes from an authenticating reverse proxy through `X-Keel-User` and
`X-Keel-Tags`, honoured only when the request also carries `X-Keel-Proxy-Token` matching
`KEEL_PROXY_TOKEN`, and every other request runs as `public`. That was finding A1. Set the admin
token as well before you bind to anything other than loopback:

```bash
export KEEL_ADMIN_TOKEN="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export KEEL_HOST=0.0.0.0
keel serve
```

The admin page carries the quarantine list, the approval queue, ledger verification and export, and a
fourteen-day trend. See [the web documentation](web.md) for the routes and the identity model in full.

## 8. Turn the network off

```bash
export KEEL_AIRGAP=1
keel status
```

The status line now reports air-gap on. From here every outbound connection to a host outside the
allow list is refused before a packet leaves, at five layers: name resolution, the socket, the event
loop, `urllib` and `httpx`. Name resolution is guarded on its own because a lookup of
`<secret>.attacker.example` carries the secret to a name server whether or not the connection ever
happens, which was finding G1.

Loopback stays reachable, so a model server on the same machine keeps working. Name a compose sibling
or another allowed host explicitly:

```bash
export KEEL_AIRGAP_ALLOW_HOSTS=llama
```

Enabling air-gap mode also sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so the embedding and
reranking models read the local cache and skip their metadata requests. Warm that cache once with the
network on, and every later run works with the cable out.

Confirm it for yourself rather than taking the paragraph on faith. The overview page runs exactly this
guard against any host you name, and reports what each layer did.

## 9. Measure the answers

```bash
keel eval
```

The golden set runs through the production answer path and scores retrieval, groundedness, relevance,
refusals, leak strings and judged quality, then writes HTML and JSON reports. The gate fails the run
when a metric drops past its threshold, and continuous integration runs it on every change, so answer
quality is measured rather than asserted. See [the evaluation guide](evals.md).

## Where to go next

- [Architecture](architecture.md) for the whole path through a question, and the ledger hash recipe.
- [Threat model](threat-model.md) for the attacks the design answers, and the control for each one.
- [Adversarial review](security-review.md) for all 27 findings, the fix for each, and the two that
  stay open on purpose.
- [On premise](onprem.md) for the Compose stack, the GPU variant and offline operation.
- [Azure](deploy-azure.md) for a deployment inside a client tenancy on managed identity with no keys.
- [CLI reference](cli.md) for every command and flag.
