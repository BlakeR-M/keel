# On-premise deployment

Keel runs on one box with nothing leaving it: a llama.cpp server for generation, fastembed models on
the CPU for embeddings and reranking, and a single SQLite file for documents, chunks, the inference
log, approvals and the ledger. This guide covers the native runner (what runs tonight, no Docker), the
Docker Compose stack, the air-gap guard and how to prove it, model swaps, backups and upgrades.

## Hardware notes

Reference machine (Canberra, 2026-08-18): Windows 10, 32 logical cores, RTX 5080 with 16 GB VRAM.

- **Tonight the GPU is busy** with an unrelated training run, so Keel runs the local LLM on the CPU:
  Qwen2.5-3B-Instruct Q4_K_M (`D:\models\qwen2.5-3b-instruct-q4_k_m.gguf`, about 2 GB), `-ngl 0`,
  roughly 8 to 20 tokens per second depending on prompt length. Answers with citations take a few
  seconds; that is the demo budget.
- **When the GPU is free**, swap to Qwen3.5-9B Q5_K_M (`D:\models\Qwen3.5-9B-Q5_K_M.gguf`, about
  6.5 GB) with `-ngl 99` for full offload. Generation runs several times faster than the CPU path
  and answer quality rises noticeably. Nothing in Keel changes; only the server flags do.
- Embeddings and reranking stay on the CPU in every configuration: `BAAI/bge-small-en-v1.5` (384
  dimensions, ONNX) and `Xenova/ms-marco-MiniLM-L-6-v2`. Together they need about 150 MB of disk in
  the fastembed cache and a few hundred MB of RAM.
- Sizing guide for a client appliance: 8 cores and 16 GB RAM run the 3B model comfortably on CPU; a
  12 GB or larger NVIDIA card runs the 9B model fully offloaded. Disk: the model files plus the corpus
  (SQLite grows by a few KB per chunk, embedding included).

## Native runner (Windows and Linux, no Docker)

`deploy/onprem/run.ps1` (Windows) and `deploy/onprem/run.sh` (Linux, macOS, Git Bash) start
llama-server on 127.0.0.1:8081, wait for `/v1/models` to answer, then start the web app with
`python -m uvicorn keel.web.app:app --host 127.0.0.1 --port 8400`. Both are idempotent (a service that
already answers is left alone), run the processes in the background, and write PIDs and logs to
`data/`. `stop.ps1` / `stop.sh` end exactly the processes the runner started; a llama-server that was
already running beforehand is left as it is.

```powershell
# Windows: fresh clone to a running demo (creates .venv, installs, starts, ingests, prints the URL)
.\demo.ps1

# Day to day
.\deploy\onprem\run.ps1            # start llama-server and the web app
.\deploy\onprem\run.ps1 -SkipWeb   # llama-server only
.\deploy\onprem\stop.ps1           # stop what run.ps1 started
```

```bash
# Linux / macOS / Git Bash
make demo                          # same fresh-clone path
make up && make down               # deploy/onprem/run.sh and stop.sh
```

Configuration is by environment variable, every value with a default:

| Variable | Default | Meaning |
| --- | --- | --- |
| `KEEL_LLAMA_SERVER` | `D:\llama.cpp\bin\llama-server.exe` (Windows), `llama-server` on PATH (Linux) | llama-server binary |
| `KEEL_MODEL_PATH` | `D:\models\qwen2.5-3b-instruct-q4_k_m.gguf` (Windows), `./models/qwen2.5-3b-instruct-q4_k_m.gguf` (Linux) | GGUF model |
| `KEEL_NGL` | `0` | layers offloaded to the GPU |
| `KEEL_LLAMA_CTX` | `8192` | context length |
| `KEEL_LLAMA_THREADS` | logical processor count | CPU threads |
| `KEEL_LLAMA_ALIAS` | `qwen2.5-3b-instruct` | model name the API reports; matches `KEEL_LOCAL_LLM_MODEL` |
| `KEEL_LLAMA_PORT` | `8081` | llama-server port |
| `KEEL_LLAMA_EXTRA_ARGS` | empty | extra llama-server flags, for example `-np 4` |
| `KEEL_PYTHON` | `.venv` interpreter | interpreter for the web app |
| `KEEL_DATA_DIR` | `<repo>/data` | pid files, logs, and `keel.db` |
| `KEEL_AIRGAP` | `0` | `1` turns on the egress guard in the app |

Logs: `data/llama-server.err` (llama.cpp writes its log to stderr and flushes in blocks, so the file
fills a little behind the console) and `data/keel-web.err` (uvicorn).

## Model swap

1. Put the GGUF file where the box can read it.
2. Point the runner at it and choose the offload:

   ```powershell
   $env:KEEL_MODEL_PATH = 'D:\models\Qwen3.5-9B-Q5_K_M.gguf'
   $env:KEEL_NGL = '99'
   .\deploy\onprem\stop.ps1
   .\deploy\onprem\run.ps1
   ```

3. Keep the alias. llama-server serves the model under `KEEL_LLAMA_ALIAS` (default
   `qwen2.5-3b-instruct`) and Keel asks for `KEEL_LOCAL_LLM_MODEL` (same default), so a swap needs no
   Keel change. Change both together when you want the served name to reflect the file, for example
   `qwen3.5-9b`.
4. Tool calling relies on `--jinja` (the model's own chat template). Every Qwen 2.5 and 3.5 instruct
   GGUF works with it; for another family, confirm the template carries tool-call support before
   pointing the agent at it.
5. Compose: set `KEEL_MODEL_FILE` (file name under the models directory) and `KEEL_NGL` in the shell
   or a `.env` next to the compose file, then `docker compose up -d`.

## Air-gap mode

`KEEL_AIRGAP=1` makes the app call `keel.airgap.enforce_from_settings()` at startup, which installs
process-wide guards: `socket.socket.connect` / `connect_ex` / `sendto`, `socket.create_connection`,
`asyncio` `loop.sock_connect`, `urllib.request` openers, and an httpx transport / request hook for
clients built with `keel.airgap.airgap_transport()` or `guard_httpx_client()`. Any connection to a host
outside the allow list is refused with `keel.airgap.AirgapViolation` before a packet leaves. Loopback
(127.0.0.0/8, ::1) and the names in the allow list (`127.0.0.1`, `localhost`, `::1`, plus anything in
`KEEL_AIRGAP_ALLOW_HOSTS`, for example `llama` in the compose stack) pass. Names are checked as
written, before DNS. Enabling also sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` so fastembed
reads the local model cache and makes no metadata requests. The app logs one line at startup:
`Air-gap mode on: outbound connections go to 127.0.0.1, ::1, localhost only.`

What the guard covers: everything Keel itself does (llama-server over httpx, fastembed from cache,
SQLite on disk) plus any library in the process that connects through the standard socket, asyncio,
urllib or httpx paths. What it leaves to the host: DNS lookups issued by a library that resolves first
and connects second (the connect is still refused), and processes other than Keel. For a hard boundary
add a host firewall rule or run the container with no network (below).

### How to prove it

1. **Unit tests** (no network needed, a few seconds):

   ```powershell
   .venv\Scripts\python.exe -m pytest tests\test_airgap.py -q
   ```

   They show a socket connect to `1.1.1.1:80` refused before the original `connect` runs, `httpx` and
   `urllib` refusing `https://example.com` before any handler or transport sends, loopback requests
   succeeding under the guard, and disable putting every original back. Two integration tests confirm
   the local llama-server on 127.0.0.1:8081 and the fastembed cache keep working with the guard on;
   they skip when either is absent.

2. **Live check** in a running app or a shell with the guard on:

   ```powershell
   $env:KEEL_AIRGAP = '1'
   .venv\Scripts\python.exe -c "from keel.airgap import enforce_from_settings; enforce_from_settings(); import httpx; httpx.get('https://example.com')"
   # -> keel.airgap.AirgapViolation: Air-gap mode refused the socket connection to example.com:443 ...
   ```

3. **Container with no network** (Docker host). Ingest the corpus first so the data volume holds the
   chunks and embeddings, then start the keel image with `--network none`: it boots from the baked
   model cache, serves retrieval, the ledger and the admin page from the volume, and reports the LLM
   as unreachable for answers, which is the expected result with no network at all:

   ```bash
   docker compose -f deploy/onprem/docker-compose.yml up -d --build
   docker compose -f deploy/onprem/docker-compose.yml exec keel keel ingest --manifest fixtures/corpus.yaml
   docker compose -f deploy/onprem/docker-compose.yml stop keel
   docker run --rm --network none -v keel-data:/app/data -e KEEL_AIRGAP=1 keel-keel keel verify-ledger
   docker run --rm --network none -e KEEL_AIRGAP=1 keel-keel python -c \
     "from keel.airgap import enforce_from_settings; enforce_from_settings(); import httpx; httpx.get('https://example.com')"
   # -> AirgapViolation from the guard; with the network removed, the socket layer has nowhere to go either
   ```

   For a serving stack that stays isolated, keep the compose file as shipped (in-process guard, allow
   list `llama`) and add an egress-deny rule for the Docker bridge at the host firewall; that is a
   host-level control outside what this repository can verify.

## Docker Compose

`deploy/onprem/docker-compose.yml` runs two services and one named volume:

- `llama`: `ghcr.io/ggml-org/llama.cpp:server` with `-m /models/<file> --host 0.0.0.0 --port 8081
  --jinja`, the models directory mounted read-only, and a health check on `/health`.
- `keel`: built from the root `Dockerfile` (python:3.11-slim, non-root user, port 8400), with
  `KEEL_LOCAL_LLM_BASE_URL=http://llama:8081/v1`, `KEEL_AIRGAP=1` and `KEEL_AIRGAP_ALLOW_HOSTS=llama`.
  Waits for `llama` to be healthy. Port 8400 is published on 127.0.0.1 by default (`KEEL_BIND` changes
  the bind address).
- `keel-data`: the SQLite store at `/app/data`.

```bash
mkdir -p deploy/onprem/models && cp /path/to/model.gguf deploy/onprem/models/model.gguf
docker compose -f deploy/onprem/docker-compose.yml up -d --build
docker compose -f deploy/onprem/docker-compose.yml exec keel keel ingest --manifest fixtures/corpus.yaml
# GPU: add -f deploy/onprem/docker-compose.gpu.yml (server-cuda image, -ngl 99, NVIDIA Container Toolkit)
```

Docker is absent on the reference machine, so the compose files are validated with a YAML parser
(`make compose-check` and `tests/test_airgap.py::test_compose_files_parse_and_declare_airgap`) and by
review; the first `docker compose up` on a Docker host is the remaining check. Pin the image tags named
in the file header before production use. The image bakes the fastembed models at build time
(`PREFETCH_MODELS=1`); for a build host with no network, stage a cache in `deploy/onprem/model-cache/`
and build with `--build-arg PREFETCH_MODELS=0`.

## Backups

Everything Keel knows lives in one file, `data/keel.db` (plus `keel.db-wal` and `keel.db-shm` while
the app runs). Back it up with a consistent snapshot:

```powershell
# hot backup with SQLite's own copy (safe while the app runs)
.venv\Scripts\python.exe -c "import sqlite3; s=sqlite3.connect('data/keel.db'); d=sqlite3.connect('backups/keel-2026-08-18.db'); s.backup(d); d.close()"
```

or stop the app and copy the file. Restore by stopping the app and putting the file back. The ledger
inside the file is hash-chained; `keel verify-ledger` after a restore confirms the chain is intact.
Documents are re-ingestable from their sources at any time (ingest is idempotent by checksum), so the
database is the only state worth backing up. Compose: `docker run --rm -v keel-data:/data -v
$PWD:/backup alpine cp /data/keel.db /backup/`.

## Upgrades

1. Back up `data/keel.db` (above).
2. `git pull`, then `.venv\Scripts\python.exe -m pip install -e .[dev]` (or `make install`).
3. Schema changes are additive and applied on connect; there is no separate migration step.
4. Run `.venv\Scripts\python.exe -m pytest -q` and `keel eval` against the golden set; the eval
   regression gate is the signal that a model or retriever change moved quality.
5. `stop` then `run` the native stack, or `docker compose up -d --build` for the compose stack.
6. Model upgrades follow the model swap steps; re-run `keel eval` afterwards and keep the report.
