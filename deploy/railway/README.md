# Hosted demo on Railway

The public demo at <https://keel.flow-through.com.au> runs the fixture corpus (five documents, one
of them HR-only, one with a planted injection that lands in quarantine) on Railway as two services
in one project, joined over the private network. It uses no model API: generation runs on a
CPU-only llama.cpp server inside the project, on Railway compute alone.

```
                    public domain                    private network
  visitor  ----->  keel  (this repo, Dockerfile)  ----->  keel-llm  (deploy/railway/llm)
                    /data volume: keel.db                   /models volume: the GGUF
```

## Services

### `keel-llm`: the model

Built from [`llm/Dockerfile`](llm/Dockerfile) (build context: the repo root, selected on the service
with `RAILWAY_DOCKERFILE_PATH=deploy/railway/llm/Dockerfile`) on top of
`ghcr.io/ggml-org/llama.cpp:server` (CPU build). On first start [`llm/entrypoint.sh`](llm/entrypoint.sh) downloads
`Qwen/Qwen2.5-3B-Instruct-GGUF` `q4_k_m` (about 2 GB) into `/models`, a volume, then runs
`llama-server --host 0.0.0.0 --port 8081 --jinja --alias qwen2.5-3b-instruct -c 8192 -t <nproc> --parallel 2`.
Later starts reuse the file. Health: `GET /health` (llama-server's own) or `GET /v1/models`.

Sizing: about 3 GB of RAM (2 GB weights plus the 8k context for two parallel slots) and as many
vCPUs as the plan gives; set `LLAMA_THREADS` to the plan's vCPU count. Measured over the public URL
on 2026-08-20 with `LLAMA_THREADS=8` (five questions, `curl -w '%{time_total}'`): a short cited
answer lands in three to nine seconds end to end, and a refusal returns in about two seconds with
zero model calls. The first answer after a deploy takes longer while the model loads, which is the
reason `KEEL_LOCAL_LLM_TIMEOUT` stays raised on the app.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_URL` | the Qwen2.5-3B Q4_K_M file on Hugging Face | Where the GGUF comes from on first start |
| `MODEL_FILE` | `qwen2.5-3b-instruct-q4_k_m.gguf` | File name under `/models` |
| `MODEL_ALIAS` | `qwen2.5-3b-instruct` | Model name the API reports; matches the app's `KEEL_LOCAL_LLM_MODEL` |
| `LLAMA_CTX` | `8192` | Context length |
| `LLAMA_PARALLEL` | `2` | Concurrent slots |
| `LLAMA_THREADS` | `nproc` | CPU threads. Set it to the plan's vCPU count on Railway: `nproc` reports the host's cores (48 on the demo) |
| `LLAMA_PORT` | `8081` | Port when `PORT` is unset. Railway injects `PORT=8080` even for a private service, so the demo sets `PORT=8081` on the service to match the app's base URL |
| `LLAMA_EXTRA_ARGS` | empty | Extra llama-server flags |

The service has no public domain. The app reaches it at
`http://keel-llm.railway.internal:8081/v1`.

### `keel`: the app

Built from the root [`Dockerfile`](../../Dockerfile) with [`railway.json`](../../railway.json)
(builder `DOCKERFILE` with no explicit path, so `RAILWAY_DOCKERFILE_PATH` on `keel-llm` still wins;
healthcheck `/health` with a 600 second timeout, long enough for the embedding models to warm on the
app and for the model download plus load on `keel-llm`, which shares the file; restart on failure). [`entrypoint.sh`](entrypoint.sh) binds uvicorn to `0.0.0.0` on Railway's `PORT` (8400
elsewhere) with proxy headers honoured. The entrypoint starts as root, makes `/data` (the volume,
which Railway mounts root-owned) writable by the unprivileged `keel` user, and drops to that user with
`setpriv` before uvicorn starts; the SQLite store lives on the `/data` volume; the two fastembed models are baked into the image at build time, so
`HF_HUB_OFFLINE=1` is safe and the first request needs no download.

| Variable | Value on the demo | Meaning |
| --- | --- | --- |
| `KEEL_PROFILE` | `local` | Local providers: llama-server for generation, fastembed on CPU for embeddings and reranking, SQLite store |
| `KEEL_HOST` | `0.0.0.0` | Bind address; anything beyond loopback turns the admin guard on |
| `KEEL_DATA_DIR` | `/data` | Store, on the volume |
| `KEEL_LOCAL_LLM_BASE_URL` | `http://keel-llm.railway.internal:8081/v1` | The model service over the private network |
| `KEEL_LOCAL_LLM_MODEL` | `qwen2.5-3b-instruct` | Must equal the server's `--alias` |
| `KEEL_LOCAL_LLM_TIMEOUT` | `300` | Seconds per model call; the CPU server is slow |
| `KEEL_DEMO_IDENTITY` | `1` | Honour the demo user picker beyond loopback. **Hosted demo of the fixture corpus only; never set for a real deployment** ([docs/web.md](../../docs/web.md)) |
| `KEEL_DEMO_READONLY` | `1` | Declares the read-only posture: no web ingest route, every write under the admin guard |
| `KEEL_BOOTSTRAP_CORPUS` | `fixtures/corpus.yaml` | Ingest the fixture manifest at startup when the store is empty |
| `KEEL_ADMIN_TOKEN` | a 32-byte hex secret | `/admin` needs `X-Keel-Admin-Token` equal to it |
| `KEEL_PROXY_TOKEN` | unset | No reverse proxy asserts identity on the demo |
| `KEEL_AIRGAP` | `1` | Egress guard on. Every outbound connection to a host outside the allow list is refused at five layers before a packet leaves |
| `KEEL_AIRGAP_ALLOW_HOSTS` | `keel-llm.railway.internal` | The one host the app may reach: the model service over the private network. Loopback is always allowed |
| `HF_HUB_OFFLINE` | `1` | fastembed reads only the baked cache |
| `PORT` | set by Railway | The port uvicorn listens on |

## Deploying from a clean project

```bash
railway link -p <project> -e production

# model service: the repo root is the build context, RAILWAY_DOCKERFILE_PATH picks its Dockerfile,
# volume at /models
railway add --service keel-llm
railway service link keel-llm
railway volume add --mount-path /models
railway variables --set RAILWAY_DOCKERFILE_PATH=deploy/railway/llm/Dockerfile --set PORT=8081 --set LLAMA_THREADS=8
railway up --service keel-llm -d

# app service: repo root, volume at /data, variables from the table above
railway add --service keel
railway service link keel
railway volume add --mount-path /data
railway variables --set KEEL_PROFILE=local --set KEEL_HOST=0.0.0.0 ...   # see the table
railway variables --set KEEL_AIRGAP=1 --set KEEL_AIRGAP_ALLOW_HOSTS=keel-llm.railway.internal
railway up --service keel -d
railway domain --service keel                                   # a *.up.railway.app URL
railway domain keel.flow-through.com.au --service keel          # prints the CNAME target and a TXT verify record
```

Watch `railway logs --service keel-llm` for the model download (once, a couple of minutes), then
`railway logs --service keel` for `bootstrap: ingested 5 documents, 27 chunks, 1 quarantined`.

## Checking it

```bash
BASE=https://keel.flow-through.com.au
curl -s $BASE/health
# {"status":"ok","profile":"local","llm":true,"documents":5,"chunks":27,"quarantined":1,"ledger_seq":...}

curl -s -X POST $BASE/api/ask -H 'Content-Type: application/json' \
  -d '{"question":"How many written quotes does a $20,000 purchase need at Northbank Council?","user_id":"public"}'
# a cited answer: three written quotes [1]

Q='What is the confidential review code for the 2026 pay round?'
curl -s -X POST $BASE/api/ask -H 'Content-Type: application/json' -d "{\"question\":\"$Q\",\"user_id\":\"public\"}"
# refused: the HR document is outside the public user's tags
curl -s -X POST $BASE/api/ask -H 'Content-Type: application/json' -d "{\"question\":\"$Q\",\"user_id\":\"hr-officer\"}"
# answered: PELICAN-7741, cited to Northbank Salary Bands

curl -s -o /dev/null -w '%{http_code}\n' $BASE/admin                                   # 401
curl -s -o /dev/null -w '%{http_code}\n' -H "X-Keel-Admin-Token: $KEEL_ADMIN_TOKEN" $BASE/admin   # 200

curl -s -X POST $BASE/api/airgap-probe -H 'Content-Type: application/json' \
  -d '{"host":"data.attacker.example"}'
# every guarded layer refuses it: dns, socket, asyncio, urllib, httpx

curl -s -o /dev/null -w '%{http_code}\n' $BASE/docs/architecture   # 200: docs/ ships with the image
```

The footer on every page reports the air-gap state, so `KEEL_AIRGAP=1` is also what keeps the demo
from telling a visitor that the appliance's strongest control is switched off. With the allow list
holding only `keel-llm.railway.internal`, the demo genuinely runs under the guard: the app reaches
the model service over the private network and nothing else.

## What the demo is and is not

It shows permission filtering before generation (one click on the overview page asks the restricted
question as `public` and `hr-officer`, side by side), the air-gap guard refusing a host the visitor
names, PII redaction over Australian identifiers, citations that open in the source viewer,
refusal below the relevance line with zero model calls, the injection quarantine, the approval queue
for the agent's write tool, and the hash-chained ledger, on documents that hold nothing real. Identity is the
demo picker (`KEEL_DEMO_IDENTITY=1`), the store is read-only from the web, and admin needs the token.
A real deployment runs on the client's own machine or tenancy with identity from their proxy or
login, and never sets the demo flags; see [docs/onprem.md](../../docs/onprem.md) and
[deploy/azure/README.md](../azure/README.md).
