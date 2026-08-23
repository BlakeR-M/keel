// Parameters for deploy/azure/main.bicep. Copy, edit, and pass with `--parameters main.bicepparam`.
// Every parameter left out keeps the default declared in main.bicep.
using './main.bicep'

// Image that runs Keel. Build with deploy/onprem/Dockerfile and push to any registry, or use a
// published image. For a private registry also set registryServer and grant the identity AcrPull.
param image = 'ghcr.io/<your-org>/keel:latest'
param registryServer = ''

param namePrefix = 'keel'
param location = 'australiaeast'

// Azure OpenAI deployments. Standard keeps inference in-region.
param chatModelName = 'gpt-4o-mini'
param chatModelVersion = '2024-07-18'
param chatDeploymentName = 'gpt-4o-mini'
param chatCapacity = 10
param embedModelName = 'text-embedding-3-small'
param embedModelVersion = '1'
param embedDeploymentName = 'text-embedding-3-small'
param embedCapacity = 30
param openaiDeploymentSku = 'Standard'
param openaiApiVersion = '2024-10-21'

// Azure AI Search
param searchSku = 'basic'
param searchIndexName = 'keel-chunks'

// Network posture. true adds a VNet, private endpoints and private DNS zones and closes public
// access on OpenAI, Search and Key Vault.
param enablePrivateEndpoints = false

// App sizing. One replica keeps the SQLite store consistent.
param minReplicas = 1
param maxReplicas = 1
param containerCpu = '1.0'
param containerMemory = '2Gi'

param tags = {
  app: 'keel'
  owner: 'flow-through-logic'
}
