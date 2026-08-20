# Architecture

Keel is one Python package, `keel/`, with a small number of layers that only ever talk downward:
the entry points (CLI, web app, eval runner) ask `keel.providers.factory.build_context()` for an
`AppContext`; the context wires retrieval, the answer engine, the agent loop, the ledger and the log
against whichever profile the settings name; and every model, embedding, vector-store and reranker
call goes through the protocols in `keel/providers/base.py`. Everything persistent lives in one SQLite
file. This page walks the modules in dependency order, then the data model, then the two request
lifecycles, then the ledger.

The two diagrams in the top-level [`README.md`](../README.md#architecture) show the same shape at a
glance.

## Modules

### `keel/config.py`

`Settings` is a pydantic-settings model read from `KEEL_*` environment variables and an optional
`.env` file, with a working default for every field so a fresh clone runs the on-premise demo with no
configuration. It carries the profile (`local`, `azure`, `aws`), the data directory, the air-gap flag,
the retrieval knobs (`chunk_size` 800 characters with `chunk_overlap` 120, `top_k_bm25` and
`top_k_vector` 20 each, `top_k_final` 6, `rerank` on, `min_relevance` 0.15), the local model endpoint
and model names, the Azure endpoints and deployment names (endpoints only, no keys), the generation
limits (`max_output_tokens` 700, `temperature` 0.1), the optional Gemini judge key, and the web bind
address. `get_settings()` caches one instance; `reset_settings()` drops it, which the CLI's global
`--data-dir` and `--profile` options and the tests use after changing the environment.

### `keel/db.py`

The SQLite schema and the two helpers everything uses to reach it: `connect(path)` opens the file
with `check_same_thread=False`, `sqlite3.Row` rows and autocommit, and applies the schema idempotently
(WAL journal, foreign keys on, `CREATE ... IF NOT EXISTS` throughout, so migrations are additive
statements appended to `SCHEMA`); `transaction(conn)` is a context manager that issues `BEGIN`,
`COMMIT` and `ROLLBACK` around a block. The data model is described in full below.

### `keel/airgap.py`

The egress guard. `enforce(True)` patches `socket.socket.connect`, `connect_ex` and `sendto`,
`socket.create_connection`, `asyncio` `loop.sock_connect` on both event-loop families, and
`urllib.request.OpenerDirector.open`, and provides an httpx transport and request hook, so every
outbound connection to a host outside loopback and the allow list raises `AirgapViolation` before a
packet leaves; host names are checked as written, before DNS, and a name on the allow list also
permits the addresses it resolves to (the compose stack allows `llama`). It also sets `HF_HUB_OFFLINE`
and `TRANSFORMERS_OFFLINE` so fastembed reads its cache. `enforce(False)` restores every original;
`enforce_from_settings()` is what the entry points call first, reading `Settings.airgap` and
`KEEL_AIRGAP_ALLOW_HOSTS`.

### `keel/providers/base.py`

The contracts. Four runtime-checkable protocols and four dataclasses:

- `LLMProvider`: `name`; `chat(messages, *, tools, temperature, max_tokens, json_schema) -> ChatResult`;
  `healthy() -> bool`.
- `EmbeddingProvider`: `name`, `dim`; `embed(texts) -> list[list[float]]`; `embed_query(text) -> list[float]`.
- `VectorIndex`: `name`; `upsert(chunk_ids, vectors)`; `search(vector, k, allowed_tags) -> list[ChunkHit]`;
  `count()`.
- `Reranker`: `name`; `rerank(query, hits) -> list[ChunkHit]`.
- `ChatMessage(role, content, name, tool_call_id, tool_calls)`, `ToolSpec(name, description,
  parameters, write)`, `ChatResult(content, tool_calls, prompt_tokens, output_tokens, model, raw)`
  and `ChunkHit(chunk_id, score, text, document_id, source, title, heading, page, acl_tags,
  quarantined)`.

`ToolSpec.write` is the flag the policy reads to route a call to the approval queue. `ChunkHit.score`
is the relevance estimate the refusal gate compares with `min_relevance`. Nothing above this module
imports a concrete provider; the local, Azure and AWS profiles are interchangeable at configuration
time because they all implement these shapes.

### `keel/providers/local_llm.py`, `local_embed.py`, `local_index.py`, `local_rerank.py`

The local profile. `OpenAICompatibleLLM` maps `ChatMessage`, `ToolSpec` and `json_schema` onto the
OpenAI chat-completions API for any compatible server (llama-server here), parses tool calls with
JSON-decoded arguments and usage counts into `ChatResult`, asks for `response_format=json_schema` and
falls back to a schema instruction in the prompt the first time a server rejects it, and reports
`healthy()` from `GET /models`; given a prebuilt `openai.AzureOpenAI` client it serves the Azure
profile too. `FastembedEmbeddings` and `FastembedReranker` load `BAAI/bge-small-en-v1.5` (384
dimensions) and `Xenova/ms-marco-MiniLM-L-6-v2` as ONNX models on the CPU, lazily on first use, and
from the local cache only under air-gap. `SqliteVectorIndex` keeps embeddings in `chunks.embedding` as
float32 blobs, loads every non-quarantined embedding into one numpy matrix (refreshed after `upsert`
and whenever the table's fingerprint changes) and scores by cosine similarity, filtering by tags and
quarantine in the query; its stated ceiling is about one hundred thousand chunks, and the swap path is
sqlite-vec or pgvector behind the same protocol.

### `keel/providers/azure.py` and `keel/providers/aws.py`

The Azure profile: `AzureOpenAIChat` wraps `OpenAICompatibleLLM` around an `AzureOpenAI` client
built with `DefaultAzureCredential` and the Cognitive Services token scope; `AzureOpenAIEmbeddings`
batches texts through the embeddings deployment and checks the dimension; `AzureSearchIndex` creates
the index (HNSW vector profile plus the chunk metadata fields), upserts documents, and pushes the ACL
filter into the query as OData
(`quarantined eq false and acl_tags/any(t: search.in(t, 'public,hr', ','))`) so the filter runs
inside the search service. The Azure SDK packages are an optional extra (`pip install "keel[azure]"`);
the module imports without them and names the install command when a provider is constructed. The
reranker stays local in both profiles, so the ACL-filtered candidate set is reranked on the appliance.
`aws.py` declares `BedrockChat`, `BedrockEmbeddings` and `OpenSearchServerlessIndex` against the same
contracts, validates configuration, and raises `NotImplementedError` pointing at
[`deploy/aws/README.md`](../deploy/aws/README.md).

### `keel/providers/factory.py`

`build_context()` is the one place the profile is decided. In order: enforce air-gap from settings,
ensure the data directory, open the store, build the providers for the profile (or raise a
`RuntimeError` naming the missing Azure endpoints, or the AWS stub error), then wire `Retriever`,
`Ledger`, `InferenceLog`, `ApprovalQueue`, the default tool registry (calculator, `search_docs`
bound to the retriever, `http_get` bound to an empty host list and the air-gap flag, `create_ticket`;
`sql_readonly` only when a reporting database path is given), the `Policy` from
`<data_dir>/policy.yaml` or `DEFAULT_POLICY`, the `AnswerEngine` and the `AgentLoop`, all sharing the
same ledger and log. `AppContext` also exposes the ingest-time injection screen (`screen`, heuristics
only; `screen_with_judge`, heuristics plus the model) and `close()`.

### `keel/ingest/loaders.py`, `chunk.py`, `pipeline.py`, `errors.py`

`load(source)` reads a file path or an http(s) URL into a `LoadedDoc` (source, title, MIME, kind,
SHA-256 checksum, raw bytes, pages): the kind is detected by extension, then content type, then
leading bytes; PDF pages come through pypdf, DOCX headings, paragraphs and tables through python-docx,
HTML through lxml with navigation chrome dropped and headings kept as Markdown ATX lines, Markdown and
text as they are; a remote URL under air-gap raises `AirgapViolation` before any connection.
`chunk_document` splits each page on ATX headings (code fences ignored), packs paragraphs into chunks
of at most `chunk_size` characters with sentence-aligned overlap, and stamps every chunk with its
nearest heading, page and character span; the output is a pure function of the input. `ingest_path`
loads, checks the checksum against `documents` (a match returns a duplicate result and adds nothing),
chunks, runs the optional screen on every chunk, embeds, and writes the document, the chunks (with
`quarantined` and `quarantine_reason` where flagged) and the vectors in one transaction, then appends
an `ingest` ledger row. `ingest_manifest` walks a `documents:` list of `path` or `url`, `title` and
`acl_tags`, resolving relative paths against the manifest's directory. `IngestError`,
`AirgapViolation` and `UnsupportedSource` are the failure types the CLI reports per source.

### `keel/retrieval/bm25.py` and `hybrid.py`

`fts_search` runs BM25 over the `chunks_fts` FTS5 table; `sanitise_query` turns arbitrary user text
into a safe MATCH expression (quoted tokens joined by OR) so operators and stray quotes never raise,
and the ACL and quarantine conditions sit inside the SQL. `Retriever.retrieve(query, user_tags, k)`
runs the BM25 and vector searches with the tag list, drops anything the store let through with a
second `allowed()` check (tags intersect and not quarantined), fuses the two lists with reciprocal
rank fusion (`k` 60), and, with a reranker, rescores the fused candidates with the cross-encoder and
maps the logit through a sigmoid so `score` is a relevance estimate in the unit interval; without a
reranker the score is normalised RRF, which ranks well and gates poorly, so `rerank` stays on. Empty
tags return nothing; `None` tags mean an anonymous `public` caller. `RetrievedChunk` extends
`ChunkHit` with the per-list ranks and scores for anyone debugging a ranking.

### `keel/answer/prompts.py`, `schema.py`, `types.py`, `engine.py`

`prompts.py` holds the short system prompt written for a 3B model (use only the numbered sources,
state the fact in words then cite it, a bare citation is never an answer, refuse with one fixed
sentence), the refusal sentence, the bare-citation retry, the JSON-mode instruction and the retry
instruction, and `build_source_block`, which numbers the retrieved chunks from 1. `schema.py` is a
small JSON-schema validator (`type`, `properties`, `required`, `additionalProperties: false`, `enum`,
`items`, `minimum`, `maximum`) used for JSON-mode answers and tool arguments alike. `types.py` defines
`User(user_id, tags)`, `Citation(n, chunk_id, source, title, page, heading, snippet)` and
`Answer(text, citations, refused, retrieved, prompt_tokens, output_tokens, latency_ms, request_id,
data, error)`. `AnswerEngine.answer` is the request lifecycle described below; its pure helpers
(`citation_numbers`, `resolve_citations`, `top_citations`, `is_refusal`, `is_bare_citation`,
`parse_json_output`) are unit-tested on their own.

### `keel/agent/tools.py`, `policy.py`, `approvals.py`, `loop.py`

`ToolRegistry` holds `ToolSpec`s with their implementations and runs a call only after validating the
arguments against the tool's schema; the built-ins are `search_docs` (retrieval under the calling
user's tags, numbered passages back), `calculator` (an AST walk that accepts numbers and arithmetic
operators and nothing else), `sql_readonly` (one SELECT over allowlisted tables, run through a
read-only SQLite URI with an authoriser that permits SELECT, functions and reads of the allowlisted
tables and denies everything else, a progress-handler step budget and a row cap), `http_get` (host allowlist,
refused outright under air-gap) and `create_ticket`, the example write tool. `Policy.check(name,
arguments)` applies the deployment's allowlist, the per-tool argument rules (SELECT-only SQL and the
table allowlist, the HTTP host allowlist and the air-gap refusal) and the write flag, returning a
`Decision(allowed, reason, needs_approval)`; policies load from YAML and default to safe values.
`ApprovalQueue` stores write calls as `pending` rows, `decide`s them to `approved` or `rejected` with
the decider's name, and `execute`s an approved call once through the registry, storing the result and
marking the row `executed`; every transition is a ledger row. `AgentLoop.run` is the second lifecycle
described below.

### `keel/safety/injection.py`, `pii.py`, `ledger.py`

`screen(text)` scores a passage against a table of weighted patterns (instruction overrides, text
addressed to an AI, requests to reveal secrets, dictated answers, role markers and chat-template
tokens, hidden HTML comments, base64-like blobs, hidden unicode, a high density of AI-directed
imperatives), combines the weights as a noisy-OR and quarantines at 0.5; `screen_with_judge` adds a
yes-or-no LLM classifier and quarantines when either layer says so. `pii.py` detects and redacts
Australian identifiers (TFN, Medicare, ABN, card numbers, phone numbers, email addresses) with
check-digit validation so ordinary numbers pass; both functions are pure. `Ledger` is the
tamper-evident audit trail: `append(kind, request_id, payload)` writes one row whose SHA-256 covers
one canonical JSON array `[prev_hash, kind, request_id, canonical_payload]` (field boundaries stay explicit and a NULL request_id stays distinct from an empty one), computed in Python inside a `BEGIN IMMEDIATE`
transaction under a per-connection lock, so two writers can never chain from the same predecessor
and a reader on the same connection can never deadlock an append; `verify()` recomputes the
chain and names the first broken link; `export()` writes JSONL that `verify_file()` checks with
nothing but the standard library. Timestamps and sequence numbers sit outside the hash; order is
bound by the chain itself.

### `keel/observe/log.py`

`InferenceLog.record(**fields)` upserts one row per request into `inference_log` (the answer engine
and the agent loop call it with the request id, user and tags, mode, question, retrieved chunk ids,
tool calls, answer, refusal flag, citations, latency, tokens, provider and model), `attach_judge`
stores the eval scores on the row, and the read side (`recent`, `get`, `daily_summary`, `totals`)
feeds the admin page's tiles, fourteen-day sparklines and request detail, `keel status` and
`keel export-log`.

### `keel/evals/golden.py`, `judge.py`, `metrics.py`, `run.py`, `report.py`

`golden.py` loads, validates, saves and drafts the golden set (`fixtures/golden.yaml`: question,
user tags, reference answer, expected sources, `must_include`, `must_not_include`, `expect_refusal`).
`judge.py` asks the deployment's own model in JSON-schema mode for groundedness, relevance and
correctness with reasons, retries once, and averages with a Gemini second judge when a key is set
and the box is not air-gapped. `metrics.py` computes hit@k and MRR, refusal correctness, the string
checks, latency percentiles and tokens, `aggregate`s per-item results into a summary and `compare`s a
summary with a baseline against thresholds (`hit_at_3`, `groundedness` and `refusal_correct` may
each drop up to five points; `must_not_include_pass` may not drop at all). `run.py` asks every item through
`ctx.answer_engine`, scores it, judges answered items, attaches scores to the inference log, writes
`eval-<stamp>.json` and `.html` plus `latest.json` and `latest.html`, and gates; `promote_baseline`
copies the latest summary to `baseline.json`. `report.py` renders the JSON payload as one
self-contained HTML page. [`evals.md`](evals.md) has the method and the numbers.

### `keel/web/app.py`, `views.py`, `templates/`, `static/`

The FastAPI app: chat page (`GET /`), `POST /ask` (HTML partial, JSON, or the full page depending
on the request), `POST /api/ask` and `POST /api/agent` (always JSON), `GET /source/{chunk_id}` (the
chunk with its tags, 403 when the caller's tags share none of them), `GET /health`, and the admin
router (`/admin`, `/admin/request/{id}`, approve and reject, quarantine release, ledger verify and
export) behind `require_admin`, which is open on loopback and needs `X-Keel-Admin-Token` beyond it.
The `AppContext` is built once in the lifespan handler or on first request and kept on
`app.state.ctx`; the model and the agent run in a thread pool off the event loop. `views.py` is pure
helpers: the demo users and tag parsing, answer text to escaped HTML with `[n]` markers turned into
citation chips, the JSON shapes, the step view for agent results, the loopback test and the sparkline
geometry. Templates are Jinja with autoescape on; the client side is one stylesheet and one small
script that swaps the result partial in place. [`web.md`](web.md) lists every route.

### `keel/cli.py`

The `keel` command line on typer, also reachable as `python -m keel` and `python -m keel.cli`. The
global `--data-dir` and `--profile` options set `KEEL_DATA_DIR` and `KEEL_PROFILE` before settings
load; every command builds its own `AppContext` inside its body so `--help` is instant. Commands:
`ingest`, `ask`, `agent`, `approvals list|approve|reject`, `verify-ledger`, `status`, `serve`, `eval`,
`export-log`. [`cli.md`](cli.md) documents each with its output and exit codes.

## Data model (`keel/db.py`)

One SQLite file, `<data_dir>/keel.db`, in WAL mode with foreign keys on.

| Table | Columns | Notes |
| --- | --- | --- |
| `documents` | `id`, `source`, `title`, `checksum` (unique), `mime`, `acl_tags` (JSON list, default `["public"]`), `ingested_at`, `meta` (JSON) | The checksum is the SHA-256 of the raw bytes and is what makes ingest idempotent. |
| `chunks` | `id`, `document_id` (cascade delete), `ordinal`, `text`, `heading`, `page`, `char_start`, `char_end`, `checksum`, `acl_tags`, `quarantined` (0 or 1), `quarantine_reason`, `embedding` (float32 little-endian blob), `embed_model`; unique on (`document_id`, `ordinal`) | Tags are copied from the document at ingest so retrieval filters on the chunk row alone. |
| `chunks_fts` | FTS5 external-content table over `text` and `heading`, `porter unicode61` tokeniser, kept in step by insert, update and delete triggers | BM25 comes from FTS5's `bm25()` rank. |
| `inference_log` | `id`, `ts`, `request_id` (unique), `user_id`, `user_tags`, `mode` (`answer`, `agent`, `eval`), `question`, `retrieved_ids`, `tool_calls`, `answer`, `refused`, `citations`, `latency_ms`, `prompt_tokens`, `output_tokens`, `judge`, `provider`, `model` | One row per request; JSON columns hold lists and the judge scores. |
| `approvals` | `id`, `ts`, `request_id`, `tool`, `arguments` (JSON), `status` (`pending`, `approved`, `rejected`, `executed`), `decided_at`, `decided_by`, `result` | Write tool calls wait here for a person. |
| `ledger` | `seq`, `ts`, `kind`, `request_id`, `payload` (canonical JSON), `prev_hash`, `hash` (unique) | The hash chain; see the ledger section. |

Everything Keel knows is in this file, so a backup is one consistent copy of it and a restore is
putting it back followed by `keel verify-ledger`.

## Request lifecycle: `POST /api/ask`

1. `read_payload` takes `{question, user_id, tags, mode}` from JSON or a form; `resolve_user` turns
   the user id and tags into a `User`: a demo user's tags plus any extra tags supplied, the supplied
   tags alone for an unknown id, or `public` when nothing else applies. An empty question or an
   unknown mode is a 400.
2. `ctx.answer_engine.answer(question, user)` runs in a thread pool. It mints a request id and
   appends a `request` ledger row (mode, question, user id, tags, JSON-mode flag).
3. `retriever.retrieve(question, user.tags)`: BM25 and vector candidates are fetched with the tag list
   in the query, filtered again with `allowed()`, fused with RRF, reranked, and returned with unit
   scores. Quarantined hits are dropped. A `retrieval` ledger row records the chunk ids and scores.
4. The gate: when there is no hit, or the top score is under `min_relevance`, the answer is the fixed
   refusal sentence with no citations, and the model is never called.
5. Otherwise the model receives the system prompt and the numbered source block plus the question.
   A reply that is nothing but citation markers is retried once with the bare-citation instruction.
   A reply that opens with the refusal sentence is a refusal. Otherwise `[n]` markers are resolved
   to `Citation`s (out-of-range numbers dropped) and an unmarked answer gets its top two sources.
   In JSON mode the reply is parsed and validated against the caller's schema, retried once with the
   validation error, and reported as `error` when the second reply is still invalid.
6. An `answer` ledger row (text, refused, cited chunk ids, tokens, latency, model, error) and one
   `inference_log` row are written; the `Answer` is returned.
7. `answer_json` serialises it: `{request_id, mode, user, text, refused, citations[], retrieved[],
   prompt_tokens, output_tokens, latency_ms, data, error}`. Citations carry a `/source/{chunk_id}`
   URL with the asking user's tags, so a chip opens for the user who earned it and answers 403 for
   anyone carrying fewer tags. A model that is unreachable surfaces as a 502 naming the endpoint.

The same engine serves `keel ask` and the eval runner; nothing in the eval path is mocked.

## Request lifecycle: the agent loop

1. `AgentLoop.run(question, user, max_steps=6)` (behind `POST /api/agent`, the chat page in agent
   mode, and `keel agent`) mints a request id, builds a `ToolContext(user, request_id)`, reads the
   registry's `ToolSpec`s and appends a `request` ledger row.
2. The system prompt lists every tool with its description and marks write tools; the user turn is
   the question. Up to `max_steps` model turns follow, each with the tool specs attached.
3. When the model replies with plain content the loop ends and that text is the answer. When it
   proposes tool calls, each call goes through `_handle_call`: the per-request budget is checked,
   then `policy.check(name, arguments)` (allowlist, argument rules, write flag), then the registry
   membership, then whether a write call has a queue to go to. A `tool_call` ledger row records the
   tool, arguments and decision.
4. A refused call goes back to the model as `refused: <reason>` and is listed in `refused_tools`. A
   write call is enqueued in `approvals` as `pending`, an `approval` ledger row is written, and the
   model sees `queued for approval (id N)`. An allowed call runs through `registry.call` (schema
   validation, then the tool's own guard), its result is truncated to 6,000 characters, sent back as
   the tool message, and recorded as a `tool_result` ledger row.
5. When the turns run out the answer is a fixed step-limit sentence and the steps so far stand.
6. An `answer` ledger row and one `inference_log` row (mode `agent`, `tool_calls` = the steps) are
   written; `AgentResult(text, steps, refused_tools, tokens, latency_ms, request_id)` is returned and
   serialised by `agent_json` with each step's tool, arguments, decision, and result or `queued_id`.
7. Later, a person decides: `keel approvals approve <id>` or `POST /admin/approvals/{id}/approve`
   calls `ApprovalQueue.decide` (an `approval` ledger row with the decider) and then `execute`, which
   runs the stored call once through the registry, stores the result on the row, marks it `executed`
   and writes a final `approval` ledger row. `reject` records the decision and nothing runs.

## Ledger kinds

Every kind is written by exactly the code named here, and `keel verify-ledger` recomputes the chain
across all of them in sequence order.

| Kind | Written by | Payload |
| --- | --- | --- |
| `request` | answer engine, agent loop | mode, question, user id, user tags (and the JSON-mode flag for answers) |
| `retrieval` | answer engine | chunk ids and scores placed before the model, and the quarantined ids dropped |
| `tool_call` | agent loop | tool, arguments, the policy decision (allowed, reason, needs approval) |
| `tool_result` | agent loop | tool and the result text handed back to the model |
| `answer` | answer engine, agent loop | text, refused flag, cited chunk ids (answers) or step count and refused tools (agent), tokens, latency, model, error |
| `approval` | agent loop and approval queue | approval id, tool, and the transition: `pending` with arguments, `approved` or `rejected` with the decider, `executed` with the result |
| `ingest` | ingest pipeline | source, checksum, chunks added |
| `quarantine` | admin quarantine release | chunk id, `release`, who did it |

The hash recipe is `sha256(prev_hash + "|" + kind + "|" + request_id + "|" + canonical_payload)`
from a genesis of sixty-four zeros; `canonical_payload` is `json.dumps(payload, sort_keys=True,
separators=(",", ":"), ensure_ascii=False)`. A chain proves nothing about rows removed from its tail,
so record `Ledger.head()` outside the machine at each backup when that matters.
