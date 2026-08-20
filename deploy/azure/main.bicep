// Keel on Azure: Container Apps + Azure OpenAI + Azure AI Search + Key Vault, one user-assigned managed
// identity and no keys anywhere. Scope: resource group (deploy.ps1 creates the group first;
// subscription.bicep is the small parent for a one-shot subscription-scope deployment).
//
// API versions (checked against the ARM reference, August 2026):
//   Microsoft.ManagedIdentity/userAssignedIdentities        2023-01-31
//   Microsoft.OperationalInsights/workspaces                 2023-09-01
//   Microsoft.App/managedEnvironments, containerApps         2024-03-01
//   Microsoft.CognitiveServices/accounts (+deployments)      2024-10-01
//   Microsoft.Search/searchServices                          2023-11-01
//   Microsoft.KeyVault/vaults                                2023-07-01
//   Microsoft.Authorization/roleAssignments                  2022-04-01
//   Microsoft.Network/virtualNetworks, privateEndpoints      2023-11-01
//   Microsoft.Network/privateDnsZones (+links)               2020-06-01

targetScope = 'resourceGroup'

// ------------------------------------------------------------------------------------------------
// Parameters
// ------------------------------------------------------------------------------------------------

@description('Prefix for resource names: lowercase letters and digits, 3 to 12 characters.')
@minLength(3)
@maxLength(12)
param namePrefix string = 'keel'

@description('Region for every resource. Azure OpenAI model availability differs by region; australiaeast carries gpt-4o-mini and text-embedding-3-small on the Standard SKU.')
param location string = resourceGroup().location

@description('Container image that runs Keel, for example ghcr.io/<owner>/keel:0.1.0. Build it with deploy/onprem/Dockerfile or point at any registry image.')
param image string

@description('Registry server for a private image (for example myregistry.azurecr.io). The managed identity needs AcrPull on that registry. Leave empty for a public image.')
param registryServer string = ''

@description('Chat model name and version for the Azure OpenAI deployment.')
param chatModelName string = 'gpt-4o-mini'
param chatModelVersion string = '2024-07-18'
param chatDeploymentName string = 'gpt-4o-mini'

@description('Chat deployment capacity in thousands of tokens per minute.')
@minValue(1)
param chatCapacity int = 10

@description('Embedding model name and version for the Azure OpenAI deployment.')
param embedModelName string = 'text-embedding-3-small'
param embedModelVersion string = '1'
param embedDeploymentName string = 'text-embedding-3-small'

@description('Embedding deployment capacity in thousands of tokens per minute.')
@minValue(1)
param embedCapacity int = 30

@description('Deployment SKU. Standard keeps inference inside the region (data residency); GlobalStandard offers more quota and routes inference across regions.')
@allowed(['Standard', 'GlobalStandard', 'DataZoneStandard'])
param openaiDeploymentSku string = 'Standard'

@description('Azure OpenAI data-plane API version handed to the app (KEEL_AZURE_OPENAI_API_VERSION).')
param openaiApiVersion string = '2024-10-21'

@description('Azure AI Search SKU. basic suits a single-tenant appliance; standard adds partitions and replicas.')
@allowed(['basic', 'standard'])
param searchSku string = 'basic'

@description('Name of the search index Keel creates on first run (KEEL_AZURE_SEARCH_INDEX).')
param searchIndexName string = 'keel-chunks'

@description('When true: a VNet with an apps subnet and a private-endpoint subnet, private endpoints plus private DNS zones for OpenAI, Search and Key Vault, public network access disabled on those three, and the Container Apps environment injected into the VNet.')
param enablePrivateEndpoints bool = false

param vnetAddressPrefix string = '10.40.0.0/16'
param appSubnetPrefix string = '10.40.0.0/23'
param privateEndpointSubnetPrefix string = '10.40.2.0/24'

@description('Replica bounds. The SQLite store lives on the container filesystem, so one replica keeps a single consistent store; raise maxReplicas once the store moves to a shared service.')
@minValue(0)
param minReplicas int = 1
@minValue(1)
param maxReplicas int = 1

param containerCpu string = '1.0'
param containerMemory string = '2Gi'

param tags object = {
  app: 'keel'
}

// ------------------------------------------------------------------------------------------------
// Names and constants
// ------------------------------------------------------------------------------------------------

var suffix = toLower(uniqueString(resourceGroup().id))
var identityName = '${namePrefix}-id'
var logsName = '${namePrefix}-logs'
var envName = '${namePrefix}-env'
var appName = '${namePrefix}-app'
var openaiName = '${namePrefix}-oai-${suffix}'
var searchName = '${namePrefix}-search-${suffix}'
var vaultName = 'kv-${namePrefix}-${take(suffix, 8)}'
var vnetName = '${namePrefix}-vnet'
var appSubnetName = 'apps'
var privateEndpointSubnetName = 'private-endpoints'
var searchEndpoint = 'https://${searchName}.search.windows.net'

// Built-in role definition ids.
var roleIds = {
  cognitiveServicesOpenAIUser: '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'
  searchIndexDataContributor: '8ebe5a00-799e-43f5-93ac-243d3dce84a7'
  searchServiceContributor: '7ca78c08-252a-4471-8644-bb5ff32d4ba0'
  keyVaultSecretsUser: '4633458b-17de-408a-b874-0445c86b69e6'
}

// Private link targets, one entry per service that gets a private endpoint.
var privateLinks = [
  {
    name: 'openai'
    zone: 'privatelink.openai.azure.com'
    groupId: 'account'
  }
  {
    name: 'search'
    zone: 'privatelink.search.windows.net'
    groupId: 'searchService'
  }
  {
    name: 'vault'
    zone: 'privatelink.vaultcore.azure.net'
    groupId: 'vault'
  }
]

// ------------------------------------------------------------------------------------------------
// Identity and logging
// ------------------------------------------------------------------------------------------------

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: identityName
  location: location
  tags: tags
}

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logsName
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// ------------------------------------------------------------------------------------------------
// Network (only with enablePrivateEndpoints)
// ------------------------------------------------------------------------------------------------

resource vnet 'Microsoft.Network/virtualNetworks@2023-11-01' = if (enablePrivateEndpoints) {
  name: vnetName
  location: location
  tags: tags
  properties: {
    addressSpace: {
      addressPrefixes: [vnetAddressPrefix]
    }
    subnets: [
      {
        name: appSubnetName
        properties: {
          addressPrefix: appSubnetPrefix
          delegations: [
            {
              name: 'Microsoft.App.environments'
              properties: {
                serviceName: 'Microsoft.App/environments'
              }
            }
          ]
        }
      }
      {
        name: privateEndpointSubnetName
        properties: {
          addressPrefix: privateEndpointSubnetPrefix
        }
      }
    ]
  }
}

// ------------------------------------------------------------------------------------------------
// Azure OpenAI: account plus chat and embedding deployments (deployments are created one at a time)
// ------------------------------------------------------------------------------------------------

resource openai 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: openaiName
  location: location
  tags: tags
  kind: 'OpenAI'
  sku: {
    name: 'S0'
  }
  properties: {
    customSubDomainName: openaiName
    disableLocalAuth: true
    publicNetworkAccess: enablePrivateEndpoints ? 'Disabled' : 'Enabled'
    networkAcls: {
      defaultAction: enablePrivateEndpoints ? 'Deny' : 'Allow'
    }
  }
}

resource chatDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openai
  name: chatDeploymentName
  sku: {
    name: openaiDeploymentSku
    capacity: chatCapacity
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: chatModelName
      version: chatModelVersion
    }
    versionUpgradeOption: 'OnceNewDefaultVersionAvailable'
  }
}

resource embedDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openai
  name: embedDeploymentName
  sku: {
    name: openaiDeploymentSku
    capacity: embedCapacity
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: embedModelName
      version: embedModelVersion
    }
    versionUpgradeOption: 'OnceNewDefaultVersionAvailable'
  }
  dependsOn: [chatDeployment]
}

// ------------------------------------------------------------------------------------------------
// Azure AI Search
// ------------------------------------------------------------------------------------------------

resource search 'Microsoft.Search/searchServices@2023-11-01' = {
  name: searchName
  location: location
  tags: tags
  sku: {
    name: searchSku
  }
  properties: {
    replicaCount: 1
    partitionCount: 1
    hostingMode: 'default'
    semanticSearch: 'disabled'
    disableLocalAuth: true
    publicNetworkAccess: enablePrivateEndpoints ? 'disabled' : 'enabled'
  }
}

// ------------------------------------------------------------------------------------------------
// Key Vault (RBAC, purge protection) for anything a client adds later
// ------------------------------------------------------------------------------------------------

resource vault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: vaultName
  location: location
  tags: tags
  properties: {
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 90
    enablePurgeProtection: true
    publicNetworkAccess: enablePrivateEndpoints ? 'Disabled' : 'Enabled'
    networkAcls: {
      defaultAction: enablePrivateEndpoints ? 'Deny' : 'Allow'
      bypass: 'AzureServices'
    }
  }
}

// ------------------------------------------------------------------------------------------------
// Role assignments for the identity (identity replaces keys)
// ------------------------------------------------------------------------------------------------

resource openaiUserRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(openai.id, identity.id, roleIds.cognitiveServicesOpenAIUser)
  scope: openai
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleIds.cognitiveServicesOpenAIUser)
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource searchIndexRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(search.id, identity.id, roleIds.searchIndexDataContributor)
  scope: search
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleIds.searchIndexDataContributor)
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource searchServiceRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(search.id, identity.id, roleIds.searchServiceContributor)
  scope: search
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleIds.searchServiceContributor)
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource vaultSecretsRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(vault.id, identity.id, roleIds.keyVaultSecretsUser)
  scope: vault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleIds.keyVaultSecretsUser)
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// ------------------------------------------------------------------------------------------------
// Container Apps environment and the Keel app
// ------------------------------------------------------------------------------------------------

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: envName
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
    workloadProfiles: [
      {
        name: 'Consumption'
        workloadProfileType: 'Consumption'
      }
    ]
    vnetConfiguration: enablePrivateEndpoints
      ? {
          infrastructureSubnetId: resourceId('Microsoft.Network/virtualNetworks/subnets', vnetName, appSubnetName)
          internal: false
        }
      : null
    zoneRedundant: false
  }
  dependsOn: [vnet]
}

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: appName
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${identity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: env.id
    workloadProfileName: 'Consumption'
    configuration: {
      ingress: {
        external: true
        targetPort: 8400
        transport: 'auto'
        allowInsecure: false
      }
      registries: empty(registryServer)
        ? []
        : [
            {
              server: registryServer
              identity: identity.id
            }
          ]
    }
    template: {
      containers: [
        {
          name: 'keel'
          image: image
          resources: {
            cpu: json(containerCpu)
            memory: containerMemory
          }
          env: [
            { name: 'KEEL_PROFILE', value: 'azure' }
            { name: 'KEEL_HOST', value: '0.0.0.0' }
            { name: 'KEEL_PORT', value: '8400' }
            { name: 'KEEL_DATA_DIR', value: '/data' }
            { name: 'KEEL_AZURE_OPENAI_ENDPOINT', value: openai.properties.endpoint }
            { name: 'KEEL_AZURE_OPENAI_API_VERSION', value: openaiApiVersion }
            { name: 'KEEL_AZURE_OPENAI_CHAT_DEPLOYMENT', value: chatDeploymentName }
            { name: 'KEEL_AZURE_OPENAI_EMBED_DEPLOYMENT', value: embedDeploymentName }
            { name: 'KEEL_AZURE_SEARCH_ENDPOINT', value: searchEndpoint }
            { name: 'KEEL_AZURE_SEARCH_INDEX', value: searchIndexName }
            // DefaultAzureCredential picks this user-assigned identity through AZURE_CLIENT_ID.
            { name: 'AZURE_CLIENT_ID', value: identity.properties.clientId }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: 8400
              }
              initialDelaySeconds: 15
              periodSeconds: 30
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/health'
                port: 8400
              }
              initialDelaySeconds: 5
              periodSeconds: 10
            }
          ]
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
      }
    }
  }
  // Roles and model deployments land before the first container boots.
  dependsOn: [
    chatDeployment
    embedDeployment
    openaiUserRole
    searchIndexRole
    searchServiceRole
    vaultSecretsRole
    search
  ]
}

// ------------------------------------------------------------------------------------------------
// Private endpoints and DNS (only with enablePrivateEndpoints)
// ------------------------------------------------------------------------------------------------

resource dnsZones 'Microsoft.Network/privateDnsZones@2020-06-01' = [
  for link in privateLinks: if (enablePrivateEndpoints) {
    name: link.zone
    location: 'global'
    tags: tags
  }
]

resource dnsLinks 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = [
  for (link, i) in privateLinks: if (enablePrivateEndpoints) {
    parent: dnsZones[i]
    name: '${vnetName}-link'
    location: 'global'
    tags: tags
    properties: {
      registrationEnabled: false
      virtualNetwork: {
        id: vnet.id
      }
    }
  }
]

var privateLinkTargets = [openai.id, search.id, vault.id]

resource privateEndpoints 'Microsoft.Network/privateEndpoints@2023-11-01' = [
  for (link, i) in privateLinks: if (enablePrivateEndpoints) {
    name: '${namePrefix}-${link.name}-pe'
    location: location
    tags: tags
    properties: {
      subnet: {
        id: resourceId('Microsoft.Network/virtualNetworks/subnets', vnetName, privateEndpointSubnetName)
      }
      privateLinkServiceConnections: [
        {
          name: '${namePrefix}-${link.name}'
          properties: {
            privateLinkServiceId: privateLinkTargets[i]
            groupIds: [link.groupId]
          }
        }
      ]
    }
    dependsOn: [vnet]
  }
]

resource dnsZoneGroups 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2023-11-01' = [
  for (link, i) in privateLinks: if (enablePrivateEndpoints) {
    parent: privateEndpoints[i]
    name: 'default'
    properties: {
      privateDnsZoneConfigs: [
        {
          name: link.name
          properties: {
            privateDnsZoneId: dnsZones[i].id
          }
        }
      ]
    }
  }
]

// ------------------------------------------------------------------------------------------------
// Outputs
// ------------------------------------------------------------------------------------------------

output appUrl string = 'https://${app.properties.configuration.ingress.fqdn}'
output appName string = app.name
output openaiEndpoint string = openai.properties.endpoint
output searchEndpoint string = searchEndpoint
output identityClientId string = identity.properties.clientId
output keyVaultUri string = vault.properties.vaultUri
output resourceGroupName string = resourceGroup().name
