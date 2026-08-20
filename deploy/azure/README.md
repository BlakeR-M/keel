# Keel on Azure

One Bicep template stands up the Azure profile of Keel: the app on Container Apps, Azure OpenAI for
generation and embeddings, Azure AI Search for the vector index, Key Vault for anything a client adds
later, and one user-assigned managed identity that replaces every key. `deploy.ps1` previews or runs
the deployment and smoke-tests the result.

Status: template, script and provider code are complete and unit-tested against mocked SDK clients
(`tests/test_azure_provider.py`); `bicep build` and `bicep lint` pass on both templates. The first
live deployment happens on a machine with the Azure CLI signed in.

## What gets created

| Resource | Name pattern | Purpose |
| --- | --- | --- |
| Resource group | `keel-rg` (script parameter) | Holds everything below |
| User-assigned managed identity | `keel-id` | The app's identity; every role below is granted to it |
| Log Analytics workspace | `keel-logs` | Container Apps console and system logs, 30-day retention |
| Container Apps environment | `keel-env` | Consumption workload profile; VNet-injected when private endpoints are on |
| Container App | `keel-app` | Runs the Keel image, external HTTPS ingress to port 8400, `/health` liveness and readiness probes, one replica by default |
| Azure OpenAI account | `keel-oai-<hash>` | Kind OpenAI, SKU S0, local (key) auth disabled |
| Azure OpenAI deployments | `gpt-4o-mini`, `text-embedding-3-small` | Chat and embedding deployments on the Standard SKU (in-region inference) |
| Azure AI Search | `keel-search-<hash>` | Basic tier, one replica and partition, semantic ranker off, key auth disabled |
| Key Vault | `kv-keel-<hash>` | RBAC authorisation, soft delete, purge protection; empty on day one |
| Role assignments | (GUIDs) | Cognitive Services OpenAI User on the account; Search Index Data Contributor and Search Service Contributor on Search; Key Vault Secrets User on the vault |
| VNet, subnets, private endpoints, private DNS zones | `keel-vnet`, `keel-*-pe`, `privatelink.*` | Only with `enablePrivateEndpoints=true` |

The `<hash>` is `uniqueString(resourceGroup().id)`, so names are stable per resource group and
globally unique where Azure requires it.

Outputs: `appUrl`, `appName`, `openaiEndpoint`, `searchEndpoint`, `identityClientId`, `keyVaultUri`,
`resourceGroupName`.

## Prerequisites

- Azure CLI: `winget install -e --id Microsoft.AzureCLI`, then open a new terminal.
- A signed-in account with Owner or Contributor plus User Access Administrator on the target
  subscription (role assignments need the latter).
- Azure OpenAI access enabled on the subscription, and quota for the two models in the region.
- A container image. Build it from the repo root with `deploy/onprem/Dockerfile` and push it to any
  registry, or use a published image. Set `image` in `main.bicepparam`; for a private registry also
  set `registryServer` and grant the identity AcrPull on that registry.

## Commands

```powershell
az login
cd deploy\azure
# edit main.bicepparam: image at minimum
.\deploy.ps1 -WhatIf                                     # preview: az deployment group what-if
.\deploy.ps1 -ResourceGroup keel-rg -Location australiaeast
```

The script:

1. Checks `az` is installed (prints the winget command and exits 1 when it is missing).
2. Checks `az account show` (prints `az login` and exits 1 when signed out).
3. With `-WhatIf`: runs `az deployment group what-if` against the existing group and stops.
4. Otherwise: `az group create`, `az deployment group create --parameters main.bicepparam`, prints the
   outputs, then requests `<appUrl>/health` up to five times, fifteen seconds apart.

Exit codes: 0 healthy, 1 a prerequisite is missing, 2 the deployment succeeded and the app has not
answered `/health` yet (the script prints the `az containerapp logs show` command to watch it come up).

One-shot alternative at subscription scope (creates the group as part of the deployment):

```powershell
az deployment sub create --location australiaeast --template-file deploy/azure/subscription.bicep --parameters image=ghcr.io/<owner>/keel:0.1.0
```

Validate the templates without an Azure account (Bicep CLI: `winget install -e --id Microsoft.Bicep`):

```powershell
bicep build deploy/azure/main.bicep
bicep lint deploy/azure/main.bicep
bicep build-params deploy/azure/main.bicepparam
```

## How identity replaces keys

- The Container App carries the user-assigned identity, and the template sets `AZURE_CLIENT_ID` to
  that identity's client id. `DefaultAzureCredential` in `keel/providers/azure.py` reads it and obtains
  tokens from the managed identity endpoint inside the container.
- Azure OpenAI is called with `openai.AzureOpenAI(azure_ad_token_provider=...)` using the scope
  `https://cognitiveservices.azure.com/.default`; the account has `disableLocalAuth: true`, so keys
  cannot be used even by mistake.
- Azure AI Search clients take the same credential; the service has `disableLocalAuth: true`.
- Key Vault uses RBAC (`enableRbacAuthorization: true`) with the identity as Secrets User.
- On a workstation the same code path uses your `az login` session. Grant your user the same three
  data-plane roles on the OpenAI account and the search service to run Keel locally against them.

Configuration handed to the container (all from `keel/config.py`): `KEEL_PROFILE=azure`,
`KEEL_HOST=0.0.0.0`, `KEEL_PORT=8400`, `KEEL_DATA_DIR=/data`, `KEEL_AZURE_OPENAI_ENDPOINT`,
`KEEL_AZURE_OPENAI_API_VERSION`, `KEEL_AZURE_OPENAI_CHAT_DEPLOYMENT`,
`KEEL_AZURE_OPENAI_EMBED_DEPLOYMENT`, `KEEL_AZURE_SEARCH_ENDPOINT`, `KEEL_AZURE_SEARCH_INDEX`, plus
`AZURE_CLIENT_ID`. No key or connection string appears anywhere.

## Private endpoints flag

`enablePrivateEndpoints=true` (in `main.bicepparam`) adds:

- a VNet (`10.40.0.0/16` by default) with an `apps` subnet delegated to `Microsoft.App/environments`
  and a `private-endpoints` subnet;
- private endpoints for the OpenAI account, the search service and the vault, each with a private
  DNS zone (`privatelink.openai.azure.com`, `privatelink.search.windows.net`,
  `privatelink.vaultcore.azure.net`) linked to the VNet;
- `publicNetworkAccess: Disabled` and deny-by-default network ACLs on those three services;
- VNet injection for the Container Apps environment, so the app resolves the private addresses.

The app's own ingress stays public HTTPS in both modes; set `internal: true` in the environment's
`vnetConfiguration` and front it with your own gateway when the UI must stay private too. Switching
the flag on an existing deployment recreates the Container Apps environment (VNet injection is set at
creation), so choose the posture before the first deploy or plan a redeploy.

## Cost notes (approximate, list prices, australiaeast, August 2026; the pricing calculator is authoritative)

- Azure OpenAI S0 is pay-as-you-go per token: gpt-4o-mini around USD 0.15 per million input tokens
  and USD 0.60 per million output tokens; text-embedding-3-small around USD 0.02 per million tokens.
  A pilot corpus of a few thousand chunks embeds for cents; chat spend follows usage.
- Azure AI Search basic is a fixed monthly charge, about USD 75 (about AUD 110) per month for one
  search unit. This is the largest fixed line item; the free tier is left out because its 50 MB of
  storage and shared capacity fit a smoke test and a client corpus outgrows it quickly.
- Container Apps consumption bills vCPU-seconds and GiB-seconds. One always-on replica at 1 vCPU and
  2 GiB comes to roughly USD 70 (about AUD 105) per month after the monthly free grant; scaling
  `minReplicas` to 0 trades a cold start for a near-zero idle bill.
- Log Analytics: pay per GB ingested (about USD 2.30 per GB) with a small daily free allowance; a
  single app stays low.
- Key Vault: cents per ten thousand operations; empty vault, no charge.
- Private endpoints: about USD 0.01 per hour each (three endpoints, about USD 22 per month) plus a
  small charge per private DNS zone.

## After deploy: ingest and evaluate against the deployed instance

The container starts with an empty store. Two ways to load a corpus and run the evaluation harness
against the Azure providers:

1. Inside the container (uses the managed identity, needs no extra roles):

   ```powershell
   az containerapp exec --name keel-app --resource-group keel-rg --command "keel ingest --manifest fixtures/corpus.yaml"
   az containerapp exec --name keel-app --resource-group keel-rg --command "keel eval"
   ```

   The report lands under `/data` in the container; copy it out with `az containerapp exec` and
   `cat`, or serve it from the admin page.

2. From a workstation against the same services (uses your `az login` session):

   ```powershell
   az role assignment create --assignee <your-upn> --role "Cognitive Services OpenAI User" --scope <openai resource id>
   az role assignment create --assignee <your-upn> --role "Search Index Data Contributor" --scope <search resource id>
   az role assignment create --assignee <your-upn> --role "Search Service Contributor" --scope <search resource id>
   $env:KEEL_PROFILE = "azure"
   $env:KEEL_AZURE_OPENAI_ENDPOINT = "<openaiEndpoint output>"
   $env:KEEL_AZURE_SEARCH_ENDPOINT = "<searchEndpoint output>"
   keel ingest --manifest fixtures/corpus.yaml
   keel eval
   ```

   `keel eval --help` lists the report and gate options; the top-level README describes the
   evaluation method. Both routes exercise Azure OpenAI and Azure AI Search exactly as the deployed
   app does; the ledger and inference log stay in the SQLite store of whichever process ran them.

## What is not done

- No API Management AI gateway in front of Azure OpenAI (token quotas, per-consumer keys, model
  routing) and no Web Application Firewall or Front Door in front of the app. Both bolt on later
  without touching the template's resources.
- Storage for the SQLite store: `/data` is the container's ephemeral filesystem, so the ledger,
  inference log and approvals reset when the replica restarts. Keep `maxReplicas` at 1 until the store
  moves; Azure Files over SMB is not a fit for SQLite in WAL mode, so the path forward is a managed
  database (Azure Database for PostgreSQL) behind the same store interface.
- No customer-managed keys, no diagnostic settings beyond Container Apps logs, no Defender plans, no
  budget alert. Add these per client policy.
- No live deployment has run yet from this repository; the first run is `deploy.ps1 -WhatIf` on a
  signed-in machine.

## Files

- `main.bicep`: the resource-group scope template (all resources, role assignments, optional network).
- `subscription.bicep`: small subscription-scope parent that creates the group and calls `main.bicep`.
- `main.bicepparam`: parameters; edit `image` first.
- `deploy.ps1`: prerequisite checks, what-if, deploy, outputs, `/health` smoke test.
