# The `keel` command line

`keel` is the appliance's console: ingest documents, ask questions, run the agent, decide approvals,
audit the ledger, check status, start the web app, run the evals and export the inference log. It is
installed as the `keel` console script by `pip install -e .`; `python -m keel` and `python -m keel.cli`
run the same app from a checkout without installing.

Every command builds the application context inside its own body, so `keel --help` is instant. Every
command honours the same environment as the web app (`KEEL_*` variables, `.env` file) and the same
data directory: one SQLite file holding documents, chunks, the inference log, approvals and the ledger.

## Global options

| Option | Effect |
| --- | --- |
| `--data-dir PATH`, `-d PATH` | Data directory for `keel.db` and `policy.yaml`. Sets `KEEL_DATA_DIR` before settings load. Default `./data`. |
| `--profile local\|azure\|aws` | Deploy profile. Sets `KEEL_PROFILE`. Default `local`. |
| `-h`, `--help` | Help for the app or for any command. |

Global options come before the command name:

```powershell
keel --data-dir D:\keel-data status
```

Air-gap mode is a setting, not a flag: `KEEL_AIRGAP=1` turns the in-process egress guard on for every
command, and `keel status` reports it as `air-gap: on`.

## Commands

### `keel up`

The whole first run in one command: find a model server, load the fixture corpus, start the web app
and open it. Each step is skipped once it has happened, so a second run just starts the server.

```bash
keel up [--no-open]
```

This is the command to run after cloning. It lands on the question box rather than the overview,
because whoever installed Keel came to use it. See [`setup.md`](setup.md) for the pieces separately.

### `keel setup`

Point Keel at a model you already run, or at your own Azure resources, and write it to `.env`.

```bash
keel setup                                              # search this machine and choose
keel setup --base-url http://127.0.0.1:11434/v1 --model llama3.1:8b
keel setup --profile azure --azure-openai-endpoint https://<resource>.openai.azure.com \
           --azure-search-endpoint https://<search>.search.windows.net \
           --chat-deployment gpt-4o-mini --embed-deployment text-embedding-3-small
```

With no options it probes the ports Ollama, LM Studio, llama.cpp and vLLM listen on, since all four
answer the same OpenAI-compatible `/v1/models`, and reads back the models each one serves. Choosing
one writes `KEEL_LOCAL_LLM_BASE_URL` and `KEEL_LOCAL_LLM_MODEL`, then loads the fixture corpus when
the store is empty. `--yes` takes the first server found without asking, `--no-ingest` skips the
corpus, and `--env-file` writes somewhere other than `.env`. The Azure profile writes no key: the
deployment runs on a user-assigned managed identity.

Exit 1 when nothing is answering, or when the endpoint named is out of reach.

### `keel doctor`

Check the configuration a first run depends on, and name the fix for anything unready.

```bash
keel doctor
```

Reads settings without building the application context, so it still reports when the model endpoint
or the store is the thing standing in the way. Checks the data directory is writable, the air-gap
allow list covers the model host, the endpoint answers, the configured model is one that endpoint
serves, and the store holds documents. Exit 1 when a check wants attention.

```
profile: local
   ok  data directory  D:\keel\data is writable
   ok  air-gap         off, so outbound connections are unrestricted
check  model endpoint  http://127.0.0.1:8081/v1 is out of reach. ConnectTimeout: timed out
                       Something is answering elsewhere on this machine: Ollama at
                       http://127.0.0.1:11434/v1. Run `keel setup` to point at it.
   ok  corpus          5 document(s) in the store
```

The same checks appear on a page at `/admin/connection`, for a deployment with no terminal on the box.

### `keel documents`

List what the store holds, change a document's access tags, or take one out.

```bash
keel documents list [--json]
keel documents retag <id> --tags hr,finance [--by NAME]
keel documents remove <id> [--yes] [--by NAME]
```

`list` prints each document with its access tags, chunk count and quarantine count, newest first,
and ends with the tags in use across the store.

`retag` replaces a document's tags and its chunks' tags together, inside one transaction. Retrieval
filters on the chunk, so both have to move or the change would be cosmetic. Tags are the whole
access-control model: a reader holding any one of them can retrieve the document, and an empty list
becomes `public` rather than a document nobody can reach.

`remove` takes the document out with its chunks, its full-text entries and its embeddings. It prompts
first, because it is the one irreversible action here; `--yes` opts out for a script. The ledger row
is written before the rows go, inside the same transaction, so the audit trail keeps a description of
what left.

Both write to the ledger and both are available in the browser under Documents on the admin page.
Exit 1 when the document id is absent.

### `keel ingest`

Ingest files, URLs or a manifest. Formats: PDF, DOCX, Markdown, HTML and plain text. Every chunk is
screened for prompt injection before it is stored; a flagged chunk is written as quarantined and stays
out of retrieval. Re-ingesting the same bytes reports a duplicate and adds nothing, so a rerun is safe.

```
keel ingest <path-or-url>... [--title TEXT] [--tags public,hr] [--judge]
keel ingest --manifest fixtures/corpus.yaml [--judge]
```

- `--tags` are the ACL tags a reader must hold one of (default `public`).
- `--title` names the document; otherwise the document's own title (or file name) is used.
- `--judge` runs the LLM judge on every chunk as well as the heuristics. Slower; use it for corpora
  where hidden instructions are a real concern.
- `--manifest` reads a `documents:` list of `path` or `url`, `title` and `acl_tags` (see `fixtures/corpus.yaml`).

Example, the fixture corpus, then the same again to show idempotence:

```powershell
keel ingest --manifest fixtures/corpus.yaml
# ingested   fixtures\corpus\northbank-council-procurement.md  (document 1, 7 chunks)
# ...
# ingested   fixtures\corpus\supplier-note-injected.md  (document 5, 4 chunks, 1 quarantined)
# 5 documents: 5 new, 0 duplicate; 27 chunks added; 1 quarantined

keel ingest --manifest fixtures/corpus.yaml
# duplicate  fixtures\corpus\northbank-council-procurement.md  (document 1 already stored, 0 chunks added)
# ...
# 5 documents: 0 new, 5 duplicate; 0 chunks added; 0 quarantined
```

Exit codes: 0 when every source ingested or was a duplicate; 1 when a source failed (the failure is
printed and the other sources still ingest); 2 when neither sources nor `--manifest` were given.

### `keel ask`

Answer a question from the corpus the user is entitled to see. Prints the answer, then one line per
citation as `[n] title · heading · p.N · source` (parts absent from the chunk are left out), then a
status line: `answered` with the citation count, or `refused` when the sources hold no answer. Exit 0
either way.

```
keel ask "<question>" [--user ID] [--tags public,hr] [--json-schema schema.json] [--raw]
```

- `--user` and `--tags` are the requesting user; `--tags` decides which chunks retrieval may see
  (default `public`). Permission filtering happens before generation.
- `--json-schema` switches to JSON mode: the answer is a JSON object validated against the schema in
  the file, with one retry on an invalid reply. When the second reply is still invalid the status line
  says `error` and the exit code is 1.
- `--raw` prints the whole `Answer` as JSON (text, citations, refusal flag, retrieved chunks, tokens,
  latency, request id) for scripting.

```powershell
keel ask "How many written quotes does a $20,000 purchase need at Northbank Council?"
# Three written quotes are required [1].
#
# [1] Northbank City Council Procurement Guide · Thresholds · fixtures\corpus\northbank-council-procurement.md
#
# status: answered · 1 citation · 2179 ms · request ca4c51a96aab4e83a6327743b74538fc

keel ask "What is the confidential review code for the 2026 pay round?" --user pat --tags public
# That is not in the documents I have access to.
#
# status: refused · 833 ms · request e61f0eea638a4c1c94d7f4a6958e7883
```

### `keel agent`

Run the tool-calling agent. The model may call `search_docs`, `calculator`, `sql_readonly`, `http_get`
and `create_ticket`, each call checked by the policy at the tool boundary: refused calls go back to the
model as refused, write tools (`create_ticket`) are queued for a person and reported as queued, the
rest run. Prints the final text and a step table (tool, decision, result or approval id).

```
keel agent "<question>" [--user ID] [--tags public,hr] [--max-steps 6]
```

```powershell
keel agent "Create a support ticket titled 'Printer down' saying the level 2 printer is jammed."
# The support ticket titled 'Printer down' has been queued for approval.
#
# #  tool           decision  result
# 1  create_ticket  queued    approval id 1
# 1 steps · 4557 ms · request d5535646b8b34121a9a3b46d4b23ef20
```

### `keel approvals`

Work the approval queue. A queued write call runs only after `approve`.

```
keel approvals list [--status pending|approved|rejected|executed]
keel approvals approve <id> [--by NAME]
keel approvals reject <id> [--by NAME]
```

`approve` records the decision, then executes the call through the tool registry and prints the tool's
result; the row moves to `executed`. `reject` records the decision and the call never runs. `--by`
defaults to the OS user name. Deciding a call that is no longer pending prints the reason and exits 1.

```powershell
keel approvals list --status pending
# id  status   tool           requested                 decided by  arguments
# 1   pending  create_ticket  2026-08-18T13:44:06.542Z              {"body": "The level 2 printer is jammed.", "title": "Printer down"}

keel approvals approve 1 --by owner
# approved 1 (create_ticket) by owner
# executed: ticket created: Printer down (#1)
```

### `keel verify-ledger`

Recompute the hash chain of the audit ledger (every request, retrieval set, tool call, approval, answer
and ingest). Prints `intact` with the row count and head hash, or `broken` with the first bad sequence
number and the reason. Exit 1 when any link is broken.

```
keel verify-ledger [--export ledger.jsonl]
```

`--export` writes the ledger as JSONL and verifies the file offline with `verify_file`, the same
function an auditor runs with nothing but Python and the file.

```powershell
keel verify-ledger --export D:\keel-data\ledger.jsonl
# ledger: intact · 22 rows checked · head seq 22 · head d244671d097122a3f2b1fafaf7cdf23d4dbc31a4ca3d076bea9e6e01e8b48417
# exported 22 rows to D:\keel-data\ledger.jsonl
# export verifies: intact · 22 rows checked
```

### `keel status`

One screen of facts: profile, data directory, air-gap state, LLM health (a `GET /models` against the
configured server), document and chunk counts with the quarantined share, ledger size and head
sequence, and inference totals from the log.

```powershell
keel status
# profile: local
# data dir: D:\keel-data
# air-gap: off
# llm: healthy · llama-server · qwen2.5-3b-instruct · http://127.0.0.1:8081/v1
# documents: 5
# chunks: 27 (1 quarantined)
# ledger: 22 rows · head seq 22
# inference: 4 requests · 1 refused · avg 2711 ms · 3402 prompt tokens · 130 output tokens
```

### `keel serve`

Start the web app (chat with citation chips, refusal state, admin page) with uvicorn. Host and port
default to `KEEL_HOST` (127.0.0.1) and `KEEL_PORT` (8400).

```powershell
keel serve --host 127.0.0.1 --port 8400
# keel web: http://127.0.0.1:8400
```

### `keel eval`

Run the golden question set through the appliance and report retrieval hit@k, groundedness, relevance,
refusal correctness, latency and tokens, with an HTML report and a JSON summary (see `docs/evals.md`
for the method). The terminal shows the summary, any regressions and the report paths; per-item detail
lives in the JSON report. `--gate` turns the regression gate into the exit code (1 when a gated score
dropped past its threshold against the baseline), which is how CI uses it.

```
keel eval [--golden fixtures/golden.yaml] [--report reports/] [--gate] [--no-judge] [--baseline baseline.json] [--promote]
keel eval --generate N [--out fixtures/golden-generated.yaml]
```

- `--no-judge` skips the LLM judge; retrieval, refusal and string checks still run.
- `--baseline` names the summary to gate against; by default `<report>/baseline.json` when it exists.
- `--promote` copies this run's summary to `<report>/baseline.json` so later runs gate against it.
- `--generate N` drafts N golden items from random corpus chunks with the model into `--out` (editable
  YAML) and stops; run `keel eval --golden <that file>` once you have edited them.

```powershell
keel eval --report reports --gate
# {
#   "summary": {"items": 22, "hit_at_3": 1.0, "mrr": 1.0, "refusal_correct": 1.0, "groundedness": 0.9, ...},
#   "regressions": [],
#   "report_html_path": "reports\\eval-20260818T135633Z.html",
#   "report_json_path": "reports\\eval-20260818T135633Z.json",
#   "baseline_path": "reports\\baseline.json",
#   "generated_at": "2026-08-18T13:56:33Z"
# }
# gate: passed
```

### `keel export-log`

Print inference log rows as JSON lines, oldest first, for shipping to a log store or a spreadsheet.
Each row carries the request id, user and tags, mode, question, retrieved chunk ids, tool calls,
answer, refusal flag, citations, latency, tokens, provider and model, plus judge scores when an eval
attached them.

```
keel export-log [--limit 50] [--mode answer|agent|eval]
```

```powershell
keel export-log --limit 200 --mode answer > answers.jsonl
```

## Exit codes at a glance

| Code | Meaning |
| --- | --- |
| 0 | The command completed. For `ask` this includes a refusal, which is a correct outcome. |
| 1 | The command ran and found a problem: a broken ledger, a failed source, an approval that is no longer pending, a JSON answer that failed validation twice, an eval gate failure with `--gate`, or a missing evals module. |
| 2 | Usage: an unknown profile, no source for `ingest`, an unreadable schema file, an unknown approval status. |
