"""AWS profile: documented stubs for Amazon Bedrock and OpenSearch Serverless.

Status: stub only. Each class validates its configuration and then raises NotImplementedError
pointing at deploy/aws/README.md, which maps every Keel component to the AWS service that would
back it. The class docstrings map each contract method (keel/providers/base.py) to the AWS API call
that implements it, so the port is a matter of filling in the bodies.
"""

from __future__ import annotations

from typing import Any

from keel.providers.base import ChatMessage, ChatResult, ChunkHit, ToolSpec

AWS_STUB_POINTER = (
    "The AWS profile is a stub in this release. See deploy/aws/README.md for the mapping and status."
)


def _stub_error(component: str) -> NotImplementedError:
    return NotImplementedError(f"{component} is not implemented. {AWS_STUB_POINTER}")


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required.")
    return value.strip()


class BedrockChat:
    """LLMProvider stub for Amazon Bedrock.

    Contract mapping:
      chat()    -> bedrock-runtime `converse` (messages, system, toolConfig from ToolSpec, inferenceConfig
                   {temperature, maxTokens}); tool calls come back as `toolUse` content blocks with
                   `input` already a dict; usage from `usage.inputTokens` / `usage.outputTokens`.
                   JSON-schema mode: a single forced tool whose input schema is the target schema
                   (`toolChoice: {tool: {name}}`), the tool input being the structured answer.
      healthy() -> bedrock `get_foundation_model(modelIdentifier=model_id)` (control plane), or a
                   one-token `converse` when the caller wants an end-to-end check.
    Auth: the default boto3 credential chain (task role on ECS/App Runner, profile locally). No keys
    in configuration.
    """

    name = "bedrock-chat"

    def __init__(self, region: str, model_id: str) -> None:
        self.region = _require_text(region, "AWS region")
        self.model_id = _require_text(model_id, "Bedrock model id")
        raise _stub_error("BedrockChat")

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> ChatResult:
        """Would call bedrock-runtime `converse` (see class docstring)."""
        raise _stub_error("BedrockChat.chat")

    def healthy(self) -> bool:
        """Would call bedrock `get_foundation_model` (see class docstring)."""
        raise _stub_error("BedrockChat.healthy")


class BedrockEmbeddings:
    """EmbeddingProvider stub for Amazon Bedrock embedding models.

    Contract mapping:
      embed()       -> bedrock-runtime `invoke_model` per text (Titan: body {"inputText", "dimensions",
                       "normalize"}; Cohere: body {"texts": [...], "input_type": "search_document"} which
                       accepts up to 96 texts per call), parsing `embedding` / `embeddings` from the JSON body.
      embed_query() -> the same call with Cohere `input_type: "search_query"`, or `embed([text])[0]` for Titan.
    `dim` follows the model: Titan Text Embeddings V2 offers 256, 512 or 1024; Cohere Embed v3 is 1024.
    """

    name = "bedrock-embeddings"

    def __init__(self, region: str, model_id: str, dim: int = 1024) -> None:
        self.region = _require_text(region, "AWS region")
        self.model_id = _require_text(model_id, "Bedrock embedding model id")
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("Embedding dim must be a positive integer.")
        self.dim = dim
        raise _stub_error("BedrockEmbeddings")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Would call bedrock-runtime `invoke_model` (see class docstring)."""
        raise _stub_error("BedrockEmbeddings.embed")

    def embed_query(self, text: str) -> list[float]:
        """Would call bedrock-runtime `invoke_model` with the query input type (see class docstring)."""
        raise _stub_error("BedrockEmbeddings.embed_query")


class OpenSearchServerlessIndex:
    """VectorIndex stub for an Amazon OpenSearch Serverless vector collection.

    Contract mapping:
      ensure_index()     -> PUT /{index} with `index.knn: true` and a mapping holding a `knn_vector`
                            field (dimension = dim, method hnsw / engine faiss / space cosinesimil) plus
                            keyword fields for acl_tags, boolean quarantined, integer chunk_id and document_id.
      upsert_documents() -> `_bulk` index actions keyed by chunk id (opensearch-py `helpers.bulk`);
                            upsert(chunk_ids, vectors) follows the Azure design: rows via a row loader,
                            or a clear error pointing at upsert_documents.
      search()           -> POST /{index}/_search with `knn: {vector: {vector, k, filter}}` where the
                            filter is a bool query: `term quarantined=false` plus, when allowed_tags is
                            given, `terms acl_tags: [...]` (an efficient pre-filter on the faiss engine).
      count()            -> GET /{index}/_count.
    Auth: SigV4 request signing with the task role (AWSV4SignerAuth, service name "aoss") and a data
    access policy on the collection granting the role index read and write.
    """

    name = "opensearch-serverless"

    def __init__(self, endpoint: str, index_name: str, dim: int, region: str) -> None:
        self.endpoint = _require_text(endpoint, "OpenSearch Serverless collection endpoint")
        if not self.endpoint.startswith("https://"):
            raise ValueError("OpenSearch Serverless collection endpoint must start with https://.")
        self.index_name = _require_text(index_name, "OpenSearch index name")
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("Vector dim must be a positive integer.")
        self.dim = dim
        self.region = _require_text(region, "AWS region")
        raise _stub_error("OpenSearchServerlessIndex")

    def upsert(self, chunk_ids: list[int], vectors: list[list[float]]) -> None:
        """Would issue `_bulk` index actions (see class docstring)."""
        raise _stub_error("OpenSearchServerlessIndex.upsert")

    def search(self, vector: list[float], k: int, allowed_tags: list[str] | None = None) -> list[ChunkHit]:
        """Would run a filtered kNN `_search` (see class docstring)."""
        raise _stub_error("OpenSearchServerlessIndex.search")

    def count(self) -> int:
        """Would call `_count` (see class docstring)."""
        raise _stub_error("OpenSearchServerlessIndex.count")
