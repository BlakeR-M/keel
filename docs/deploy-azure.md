# Deploying Keel to Azure

The authoritative guide is [`deploy/azure/README.md`](../deploy/azure/README.md), next to the
template it describes. This page is the short pointer.

What the Azure profile is: the same Keel image and code, with `KEEL_PROFILE=azure`, using Azure
OpenAI for chat and embeddings, Azure AI Search for the vector index (the ACL filter runs inside the
search query), Key Vault for anything a client adds later, and one user-assigned managed identity in
place of every key. Reranking, the SQLite store, the ledger, the inference log and the approval queue
stay on the appliance exactly as in the local profile. The provider code is
[`keel/providers/azure.py`](../keel/providers/azure.py); the contracts it implements are in
[`keel/providers/base.py`](../keel/providers/base.py).

Files under [`deploy/azure/`](../deploy/azure/README.md):

| File | Purpose |
| --- | --- |
| `main.bicep` | Resource-group scope template: Container App and environment, Log Analytics, Azure OpenAI account with `gpt-4o-mini` and `text-embedding-3-small` deployments, Azure AI Search (basic), Key Vault, the managed identity and its role assignments, and the VNet, private endpoints and private DNS zones behind `enablePrivateEndpoints`. |
| `subscription.bicep` | Subscription-scope wrapper that creates the resource group and calls `main.bicep`. |
| `main.bicepparam` | Parameters; set `image` first. |
| `deploy.ps1` | Checks for `az` and a signed-in account, previews with `-WhatIf`, creates the group, deploys, prints the outputs and polls `/health`. |

The shortest path, once the Azure CLI is installed and signed in:

```powershell
az login
cd deploy\azure
.\deploy.ps1 -WhatIf
.\deploy.ps1 -ResourceGroup keel-rg -Location australiaeast
```

Validation without an account, which is what CI runs:

```powershell
bicep build deploy/azure/main.bicep
bicep lint deploy/azure/main.bicep
```

Status at 0.1.0: the template, the script and the providers are complete and unit-tested against
mocked SDK clients ([`tests/test_azure_provider.py`](../tests/test_azure_provider.py)); `bicep build`
and `bicep lint` pass; no live deployment has run from this repository, because no Azure credentials
exist on the reference machine. The first run is `deploy.ps1 -WhatIf` on a signed-in workstation.
The README's "What is not done" section lists the gaps to plan for (ephemeral `/data` in the
container, no API Management gateway, no customer-managed keys) and the cost notes give the
approximate monthly figures for the basic footprint. The secure deployment checklist for the Azure
profile is in [`SECURITY.md`](../SECURITY.md).
