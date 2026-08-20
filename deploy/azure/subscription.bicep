// Small subscription-scope parent: creates the resource group and deploys main.bicep into it in one
// step. Use it when a pipeline holds subscription-level rights; deploy.ps1 uses main.bicep directly
// after creating the group with `az group create`.
//
//   az deployment sub create --location australiaeast --template-file deploy/azure/subscription.bicep \
//     --parameters image=ghcr.io/<owner>/keel:0.1.0
//
// API versions: Microsoft.Resources/resourceGroups 2024-03-01

targetScope = 'subscription'

@description('Name of the resource group to create or reuse.')
param resourceGroupName string = 'keel-rg'

@description('Region for the resource group and everything inside it.')
param location string = 'australiaeast'

@description('Prefix for resource names inside the group.')
@minLength(3)
@maxLength(12)
param namePrefix string = 'keel'

@description('Container image that runs Keel.')
param image string

@description('Registry server for a private image; empty for a public image.')
param registryServer string = ''

@description('Adds a VNet, private endpoints and private DNS, and disables public access on OpenAI, Search and Key Vault.')
param enablePrivateEndpoints bool = false

param tags object = {
  app: 'keel'
}

resource rg 'Microsoft.Resources/resourceGroups@2024-03-01' = {
  name: resourceGroupName
  location: location
  tags: tags
}

module keel 'main.bicep' = {
  name: 'keel-${uniqueString(deployment().name)}'
  scope: rg
  params: {
    namePrefix: namePrefix
    location: location
    image: image
    registryServer: registryServer
    enablePrivateEndpoints: enablePrivateEndpoints
    tags: tags
  }
}

output appUrl string = keel.outputs.appUrl
output openaiEndpoint string = keel.outputs.openaiEndpoint
output searchEndpoint string = keel.outputs.searchEndpoint
output identityClientId string = keel.outputs.identityClientId
output keyVaultUri string = keel.outputs.keyVaultUri
output resourceGroupName string = rg.name
