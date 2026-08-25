# Setup: bring your own model, bring your own cloud

Keel ships no model and no cloud account. It runs against whatever you already have, which keeps the
licensing simple and keeps your documents on infrastructure you control. This page covers the four
ways people point it at something, and what to do when a first run stalls.

The short version:

```bash
git clone https://github.com/BlakeR-M/keel.git
cd keel
python -m venv .venv && .venv/bin/pip install -e ".[dev]"   # .venv\Scripts\pip on Windows

keel setup      # finds a chat server you already run and writes .env
keel doctor     # confirms the configuration, and names the fix for anything unready
keel serve      # http://127.0.0.1:8400
```

`keel setup` looks for a chat server answering on this machine. Ollama, LM Studio, llama.cpp and vLLM
all expose the same OpenAI-compatible `/v1/models`, so whichever you have is found and its model list
read back to you. Choosing one writes `KEEL_LOCAL_LLM_BASE_URL` and `KEEL_LOCAL_LLM_MODEL` into
`.env`, then loads the fixture corpus so there is something to ask straight away.

## What Keel downloads

One thing, once: the embedding and reranking models that retrieval uses, about 150 MB total
(`BAAI/bge-small-en-v1.5` and `Xenova/ms-marco-MiniLM-L-6-v2`, both ONNX on the CPU through
fastembed). They land in the fastembed cache on first use and are read from disk after that.

Warm that cache while you have a network, and every later run works with the cable out. Air-gap mode
sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for you, so a warm cache plus `KEEL_AIRGAP=1` is
a complete offline install. The Docker image bakes both models in at build time, so a container needs
no download at all.

Everything else is yours to provide.

## Option 1: a model on your own machine

The default, and the one that keeps every document and every question on your hardware.

| Runtime | Start it | Endpoint `keel setup` looks for |
| --- | --- | --- |
| [Ollama](https://ollama.com) | `ollama serve`, then `ollama pull llama3.1:8b` | `http://127.0.0.1:11434/v1` |
| [LM Studio](https://lmstudio.ai) | Load a model, switch on the local server | `http://127.0.0.1:1234/v1` |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | `llama-server -m model.gguf --port 8081 --jinja` | `http://127.0.0.1:8081/v1` |
| [vLLM](https://docs.vllm.ai) | `vllm serve <model>` | `http://127.0.0.1:8000/v1` |

Then:

```bash
keel setup
```

To skip the search and name one directly:

```bash
keel setup --base-url http://127.0.0.1:11434/v1 --model llama3.1:8b
```

Any OpenAI-compatible server works, including ones not in that table. If it answers `GET /v1/models`
and `POST /v1/chat/completions`, Keel can use it.

**Model size and answer quality.** Retrieval, the entitlement filter, refusals, screening, the ledger
and the approval queue all behave the same whatever model you attach. The model affects the wording of
answers. A 3B model is enough to see the system work; 7B and up reads noticeably better. `keel eval`
scores whichever one you attach, so you can measure the difference rather than guess at it.

## Option 2: a hosted OpenAI-compatible endpoint

OpenAI, OpenRouter, Groq, Together and most others speak the same API, so they work with the same two
settings plus a key:

```bash
keel setup --base-url https://api.openai.com/v1 --model gpt-4o-mini --api-key sk-...
```

**Read this before choosing it.** A hosted model means the passages retrieved for each question leave
your network and reach that provider. That is a reasonable trade for a small business with ordinary
documents, and it is the wrong choice where the documents are the reason you wanted Keel. Air-gap mode
refuses those hosts by design, so `KEEL_AIRGAP=1` and a hosted model are mutually exclusive, and that
is deliberate rather than an oversight.

The entitlement filter still runs before generation either way, so the provider only ever receives
passages the person asking was entitled to read.

**Anthropic's API** uses a different request shape, so pointing `--base-url` at it directly falls over.
Reach it through a proxy that presents an OpenAI-compatible surface, or open an issue if a first-class
provider would be useful to you.

## Option 3: your own Azure tenancy

The cloud profile runs inside your subscription on a user-assigned managed identity, with no key in
the code and no key in the configuration. The Bicep template disables local key authentication on both
Azure OpenAI and Azure AI Search, so a key is not merely unused, it is turned off.

If you already have the resources:

```bash
keel setup --profile azure \
  --azure-openai-endpoint https://<resource>.openai.azure.com \
  --azure-search-endpoint https://<search>.search.windows.net \
  --chat-deployment gpt-4o-mini \
  --embed-deployment text-embedding-3-small
```

If you want Keel to create them in your subscription:

```powershell
az login
.\deploy\azure\deploy.ps1 -ResourceGroup keel-rg -Location australiaeast -WhatIf   # preview
.\deploy\azure\deploy.ps1 -ResourceGroup keel-rg -Location australiaeast           # deploy
```

The script checks the Azure CLI, previews with `az deployment group what-if`, creates the resource
group, deploys the template, prints the endpoints and smoke-tests `/health`. Edit `image` in
[`deploy/azure/main.bicepparam`](../deploy/azure/main.bicepparam) first so it points at your own
container registry. Full detail lives in [`deploy/azure/README.md`](../deploy/azure/README.md).

Install the cloud extra so the credential libraries are present:

```bash
pip install -e ".[azure]"
```

## Option 4: Docker, with your own model container

`deploy/onprem/docker-compose.yml` runs a llama.cpp server and Keel side by side, with the app under
`KEEL_AIRGAP=1` so the only host it can reach is the model container.

```bash
# put a GGUF at deploy/onprem/models/model.gguf, or set KEEL_MODELS_DIR and KEEL_MODEL_FILE
docker compose -f deploy/onprem/docker-compose.yml up -d --build
docker compose -f deploy/onprem/docker-compose.yml exec keel keel ingest --manifest fixtures/corpus.yaml
```

Swap in `docker-compose.gpu.yml` for CUDA offload. [`docs/onprem.md`](onprem.md) covers the model
swap, sizing, backups and the network-level air-gap proof with `docker run --network none`.

## When a first run stalls

```bash
keel doctor
```

Every check names the fix rather than printing a stack trace:

```
profile: local
   ok  data directory  D:\keel\data is writable
   ok  air-gap         off, so outbound connections are unrestricted
check  model endpoint  http://127.0.0.1:8081/v1 is out of reach. ConnectTimeout: timed out
                       Something is answering elsewhere on this machine: Ollama at
                       http://127.0.0.1:11434/v1. Run `keel setup` to point at it.
   ok  corpus          5 document(s) in the store
```

The four that catch almost everything:

| Symptom | What it usually is |
| --- | --- |
| Model endpoint out of reach | The server is on a different port. `keel doctor` probes the well-known ones and tells you where it found something. |
| Model name rejected, endpoint fine | `KEEL_LOCAL_LLM_MODEL` differs from what the server serves, often by a tag such as `:8b`. `keel doctor` lists the real names. |
| Every question refused | The store holds no documents. Run `keel ingest --manifest fixtures/corpus.yaml`, or ingest your own. |
| Air-gap on, every model call refused | The model is on a host outside the allow list. Add it with `KEEL_AIRGAP_ALLOW_HOSTS`, or move the model to loopback. |

Refusals that say *"the sources do not cover the question"* are the entitlement filter and the
relevance gate working. That answer costs zero model calls, so it appears instantly even with no model
attached at all.

## Settings worth knowing

`keel setup` writes only what it needs. Everything else has a working default and lives in
[`.env.example`](../.env.example).

| Setting | Default | What it decides |
| --- | --- | --- |
| `KEEL_PROFILE` | `local` | `local` for your own model server, `azure` for your own tenancy |
| `KEEL_LOCAL_LLM_BASE_URL` | `http://127.0.0.1:8081/v1` | The OpenAI-compatible endpoint |
| `KEEL_LOCAL_LLM_MODEL` | `qwen2.5-3b-instruct` | The model name that endpoint serves |
| `KEEL_LOCAL_LLM_TIMEOUT` | `120` | Seconds per model call. Raise it on a slow CPU |
| `KEEL_DATA_DIR` | `./data` | Where the SQLite store and the ledger live |
| `KEEL_AIRGAP` | `0` | `1` refuses every outbound connection outside the allow list |
| `KEEL_AIRGAP_ALLOW_HOSTS` | unset | Extra hosts the guard permits, comma separated |
| `KEEL_HOST` | `127.0.0.1` | Binding beyond loopback turns the admin guard on |
| `KEEL_ADMIN_TOKEN` | unset | Needed for `/admin` once bound beyond loopback |
| `KEEL_PROXY_TOKEN` | unset | Lets an authenticating reverse proxy assert identity |
| `KEEL_MIN_RELEVANCE` | `0.15` | Below this fused score Keel refuses rather than guesses |

## Next

- [Tutorial](tutorial.md) walks the whole appliance, from ingest through the agent to the ledger.
- [Web app](web.md) covers the routes and the identity model behind a reverse proxy.
- [On premise](onprem.md) covers Compose, GPU offload and offline operation.
- [Architecture](architecture.md) is the module-by-module walkthrough.
