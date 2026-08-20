# Keel on AWS (stub)

Status: **stub only.** `keel/providers/aws.py` declares `BedrockChat`, `BedrockEmbeddings` and
`OpenSearchServerlessIndex` against the same contracts as the local and Azure profiles
(`keel/providers/base.py`). Each constructor validates its configuration and raises
`NotImplementedError` pointing here. There is no infrastructure code for AWS in this release. The
Azure profile (`deploy/azure/`) is the reference for what a complete cloud profile looks like: managed
identity instead of keys, private endpoints as a flag, one script that previews, deploys and smoke-tests.

## Mapping

| Keel component | Azure profile (built) | AWS equivalent (planned) | Notes |
| --- | --- | --- | --- |
| `LLMProvider.chat` | Azure OpenAI chat deployment | Amazon Bedrock `converse` (bedrock-runtime) | Tools map to `toolConfig`; tool calls arrive as `toolUse` blocks with dict input; usage from `usage.inputTokens` and `usage.outputTokens`. JSON-schema mode uses one forced tool whose input schema is the target schema. |
| `LLMProvider.healthy` | GET `/openai/models` with the identity | Bedrock `get_foundation_model` or a one-token `converse` | |
| `EmbeddingProvider.embed` | Azure OpenAI embeddings deployment | Bedrock `invoke_model` with Titan Text Embeddings V2 (`dimensions` 256, 512, 1024) or Cohere Embed v3 (1024, up to 96 texts per call) | `embed_query` sends Cohere `input_type: search_query`. |
| `VectorIndex` | Azure AI Search index, HNSW profile, OData ACL filter | OpenSearch Serverless vector collection: `knn_vector` field (hnsw, faiss, cosinesimil), keyword `acl_tags`, boolean `quarantined`; kNN query with a bool filter (`term quarantined=false`, `terms acl_tags`) | Writes via `_bulk` keyed by chunk id; `count` via `_count`. |
| App hosting | Container Apps (consumption, external ingress, `/health` probes) | ECS Fargate behind an Application Load Balancer, or App Runner for the smallest footprint | Same image from `deploy/onprem/Dockerfile`. |
| Identity | User-assigned managed identity, `DefaultAzureCredential` | IAM task role (ECS) or instance role (App Runner) through the default boto3 credential chain | Policies: `bedrock:InvokeModel` and `bedrock:Converse` on the chosen model ARNs, `aoss:APIAccessAll` on the collection plus a collection data access policy for the role. |
| Secrets | Key Vault (RBAC, purge protection), empty on day one | AWS Secrets Manager (or SSM Parameter Store SecureString) | Same posture: nothing to store while every call is role-based. |
| Logs | Log Analytics workspace | CloudWatch Logs (awslogs driver) | |
| Private networking flag | VNet, private endpoints, private DNS | VPC with interface endpoints for `bedrock-runtime` and `aoss`, private subnets for the tasks, an internal ALB when the UI stays private | |
| Infrastructure as code | Bicep (`main.bicep`, `subscription.bicep`) | CDK (Python) or CloudFormation; SAM is a fit for the App Runner variant | Not written. |
| Deploy script | `deploy.ps1` (az checks, what-if, deploy, `/health` smoke test) | A `deploy.ps1` on `aws sts get-caller-identity`, `cdk diff`, `cdk deploy`, then the same `/health` loop | Not written. |

## What exists

- `keel/providers/aws.py`: three stub classes with docstrings that name the exact AWS API behind each
  contract method, plus configuration validation (region, model id, HTTPS collection endpoint,
  positive dimension).
- `tests/test_azure_provider.py::TestAwsStubs`: proves valid configuration raises
  `NotImplementedError` with this file's path in the message and invalid configuration raises
  `ValueError` first.

## What a real AWS profile needs

1. `boto3` and `opensearch-py` as an `aws` optional dependency in `pyproject.toml`.
2. Bodies for the three classes following the docstring mapping; unit tests against stubbed clients
   the way `tests/test_azure_provider.py` fakes the Azure SDK.
3. `keel/config.py` fields for region, model ids and the collection endpoint (`KEEL_AWS_*`).
4. IaC and a deploy script mirroring `deploy/azure/`.

Region note for Australian clients: Bedrock model availability in `ap-southeast-2` (Sydney) is
narrower than in `us-east-1`; check the model list for the region before choosing model ids, and
prefer in-region models when data residency is part of the brief.
