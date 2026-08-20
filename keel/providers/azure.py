"""Azure profile providers: Azure OpenAI for chat and embeddings, Azure AI Search for the vector index.

Authentication is DefaultAzureCredential throughout: the user-assigned managed identity inside Azure
(the container reads AZURE_CLIENT_ID), `az login` on a workstation. Configuration carries endpoints
and deployment names only.

The Azure SDK packages (azure-identity, azure-search-documents) are optional: `pip install "keel[azure]"`.
This module imports cleanly without them; constructing a provider then raises a RuntimeError that
names the install command. The openai package is a core dependency and is imported directly.
Chat mapping is shared with the local profile (`keel.providers.local_llm`). Tests inject fake clients
through the `client`, `search_client` and `index_client` arguments.
"""

from __future__ import annotations

import importlib
import json
from array import array
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from openai import AzureOpenAI

from keel.providers.base import ChatMessage, ChatResult, ChunkHit, ToolSpec
from keel.providers.local_llm import OpenAICompatibleLLM

AZURE_INSTALL_HINT = 'Install the Azure extras with: pip install "keel[azure]"'
COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"

VECTOR_FIELD = "vector"
HNSW_ALGORITHM_NAME = "keel-hnsw"
VECTOR_PROFILE_NAME = "keel-hnsw-profile"
INDEX_FIELD_NAMES: tuple[str, ...] = (
    "id",
    "chunk_id",
    "document_id",
    "text",
    "heading",
    "page",
    "source",
    "title",
    "acl_tags",
    "quarantined",
    VECTOR_FIELD,
)
SELECT_FIELDS: tuple[str, ...] = tuple(f for f in INDEX_FIELD_NAMES if f != VECTOR_FIELD)

UPSERT_HINT = (
    "AzureSearchIndex.upsert(chunk_ids, vectors) needs chunk text and metadata for the search index. "
    "Call upsert_documents(rows) with the chunk rows joined to their document (chunk_id, document_id, "
    "text, heading, page, source, title, acl_tags, quarantined, vector), or construct the index with "
    "row_loader=<callable(chunk_ids) -> rows> so upsert() can fetch them."
)


def _import_module(module: str) -> Any:
    """Imports `module` at call time, naming the install command when the package is absent."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise RuntimeError(f"{module} is required for the Azure profile. {AZURE_INSTALL_HINT}") from exc


def _import(module: str, attr: str) -> Any:
    """Imports `attr` from `module` at call time (see `_import_module`)."""
    return getattr(_import_module(module), attr)


def default_credential() -> Any:
    """Returns a DefaultAzureCredential (managed identity in Azure, `az login` locally)."""
    return _import("azure.identity", "DefaultAzureCredential")()


# ---------------------------------------------------------------------------------------------------
# Azure OpenAI
# ---------------------------------------------------------------------------------------------------


class AzureOpenAIChat:
    """LLMProvider backed by an Azure OpenAI chat deployment.

    Auth: `openai.AzureOpenAI(azure_ad_token_provider=get_bearer_token_provider(credential,
    "https://cognitiveservices.azure.com/.default"))` with DefaultAzureCredential when no credential
    is given. Pass `client` to substitute a fake in tests.

    Message, tool and response mapping is shared with the local profile through
    `keel.providers.local_llm.OpenAICompatibleLLM`, so both profiles speak the same wire format:
    ChatMessage to chat-completions dicts, ToolSpec to function tools, tool calls back with
    JSON-decoded arguments, usage counts on ChatResult.
    """

    name = "azure-openai"

    def __init__(
        self,
        endpoint: str,
        deployment: str,
        api_version: str,
        credential: Any | None = None,
        client: Any | None = None,
    ) -> None:
        if not endpoint:
            raise ValueError("Azure OpenAI endpoint is required (KEEL_AZURE_OPENAI_ENDPOINT).")
        if not deployment:
            raise ValueError("Azure OpenAI chat deployment name is required.")
        self.endpoint = endpoint.rstrip("/")
        self.deployment = deployment
        self.api_version = api_version
        self._client = (
            client if client is not None else _make_openai_client(self.endpoint, api_version, credential)
        )
        self._llm = OpenAICompatibleLLM(model=deployment, client=self._client, name=self.name)

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> ChatResult:
        """Runs one chat completion against the deployment and maps the reply to ChatResult."""
        return self._llm.chat(
            messages, tools=tools, temperature=temperature, max_tokens=max_tokens, json_schema=json_schema
        )

    def healthy(self) -> bool:
        """True when the account answers GET /openai/models with this credential."""
        return self._llm.healthy()


class AzureOpenAIEmbeddings:
    """EmbeddingProvider backed by an Azure OpenAI embeddings deployment (text-embedding-3-small by default).

    `dim` is sent as the `dimensions` request parameter, so a deployment of a text-embedding-3 model
    can be sized to match another profile's index. Vectors of any other length raise ValueError.
    """

    name = "azure-openai-embeddings"

    def __init__(
        self,
        endpoint: str,
        deployment: str,
        api_version: str,
        dim: int = 1536,
        credential: Any | None = None,
        client: Any | None = None,
        batch_size: int = 128,
    ) -> None:
        if not endpoint:
            raise ValueError("Azure OpenAI endpoint is required (KEEL_AZURE_OPENAI_ENDPOINT).")
        if not deployment:
            raise ValueError("Azure OpenAI embedding deployment name is required.")
        if dim <= 0:
            raise ValueError("Embedding dim must be a positive integer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        self.endpoint = endpoint.rstrip("/")
        self.deployment = deployment
        self.api_version = api_version
        self.dim = dim
        self.batch_size = batch_size
        self._client = (
            client if client is not None else _make_openai_client(self.endpoint, api_version, credential)
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embeds texts in batches of `batch_size`, preserving input order."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self._client.embeddings.create(model=self.deployment, input=batch, dimensions=self.dim)
            ordered = sorted(response.data, key=lambda item: item.index)
            for item in ordered:
                vector = [float(x) for x in item.embedding]
                if len(vector) != self.dim:
                    raise ValueError(
                        f"Deployment {self.deployment!r} returned {len(vector)}-d vectors; this provider is configured for {self.dim}-d."
                    )
                vectors.append(vector)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embeds one query string."""
        return self.embed([text])[0]


def _make_openai_client(endpoint: str, api_version: str, credential: Any | None) -> AzureOpenAI:
    """Builds an AzureOpenAI client that fetches Entra tokens through the credential (no API key)."""
    get_bearer_token_provider = _import("azure.identity", "get_bearer_token_provider")
    credential = credential if credential is not None else default_credential()
    token_provider = get_bearer_token_provider(credential, COGNITIVE_SERVICES_SCOPE)
    return AzureOpenAI(
        azure_endpoint=endpoint, api_version=api_version, azure_ad_token_provider=token_provider
    )


# ---------------------------------------------------------------------------------------------------
# Azure AI Search
# ---------------------------------------------------------------------------------------------------


class AzureSearchIndex:
    """VectorIndex backed by an Azure AI Search index with an HNSW vector profile.

    Documents mirror the SQLite chunks table joined to documents: id (string key, the chunk id),
    chunk_id, document_id, text, heading, page, source, title, acl_tags, quarantined, vector.

    Writing: `upsert_documents(rows)` takes those rows straight from SQLite (acl_tags may be a JSON
    string, quarantined may be 0/1, the vector may arrive as `vector` (list of floats) or `embedding`
    (list or float32 little-endian bytes)). The contract method `upsert(chunk_ids, vectors)` carries
    no text or metadata; it works when the index was built with `row_loader` (a callable from chunk
    ids to rows) and otherwise raises RuntimeError pointing at `upsert_documents`.

    Reading: `search()` runs a VectorizedQuery with an OData filter that keeps quarantined chunks out
    and, when `allowed_tags` is given, keeps only chunks carrying at least one of the tags. Filtering
    happens inside the service before ranking (vector_filter_mode preFilter).
    """

    name = "azure-search"

    def __init__(
        self,
        endpoint: str,
        index_name: str,
        dim: int,
        credential: Any | None = None,
        search_client: Any | None = None,
        index_client: Any | None = None,
        row_loader: Callable[[list[int]], Iterable[Mapping[str, Any]]] | None = None,
        batch_size: int = 500,
    ) -> None:
        if not endpoint:
            raise ValueError("Azure AI Search endpoint is required (KEEL_AZURE_SEARCH_ENDPOINT).")
        if not index_name:
            raise ValueError("Azure AI Search index name is required (KEEL_AZURE_SEARCH_INDEX).")
        if dim <= 0:
            raise ValueError("Vector dim must be a positive integer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        self.endpoint = endpoint.rstrip("/")
        self.index_name = index_name
        self.dim = dim
        self.batch_size = batch_size
        self._row_loader = row_loader
        if search_client is None or index_client is None:
            credential = credential if credential is not None else default_credential()
        self._search_client = (
            search_client
            if search_client is not None
            else _import("azure.search.documents", "SearchClient")(self.endpoint, index_name, credential)
        )
        self._index_client = (
            index_client
            if index_client is not None
            else _import("azure.search.documents.indexes", "SearchIndexClient")(self.endpoint, credential)
        )

    # -- schema -------------------------------------------------------------------------------------

    def build_index_definition(self) -> Any:
        """Returns the SearchIndex definition (fields plus the HNSW vector profile)."""
        models = _import_module("azure.search.documents.indexes.models")
        types = models.SearchFieldDataType
        fields = [
            models.SimpleField(name="id", type=types.String, key=True),
            models.SimpleField(name="chunk_id", type=types.Int64, filterable=True),
            models.SimpleField(name="document_id", type=types.Int64, filterable=True),
            models.SearchableField(name="text"),
            models.SimpleField(name="heading", type=types.String),
            models.SimpleField(name="page", type=types.Int32),
            models.SimpleField(name="source", type=types.String),
            models.SimpleField(name="title", type=types.String),
            models.SimpleField(name="acl_tags", type=types.Collection(types.String), filterable=True),
            models.SimpleField(name="quarantined", type=types.Boolean, filterable=True),
            models.SearchField(
                name=VECTOR_FIELD,
                type=types.Collection(types.Single),
                searchable=True,
                vector_search_dimensions=self.dim,
                vector_search_profile_name=VECTOR_PROFILE_NAME,
            ),
        ]
        vector_search = models.VectorSearch(
            algorithms=[models.HnswAlgorithmConfiguration(name=HNSW_ALGORITHM_NAME)],
            profiles=[
                models.VectorSearchProfile(
                    name=VECTOR_PROFILE_NAME, algorithm_configuration_name=HNSW_ALGORITHM_NAME
                )
            ],
        )
        return models.SearchIndex(name=self.index_name, fields=fields, vector_search=vector_search)

    def ensure_index(self) -> None:
        """Creates the index or updates it in place to the current definition (idempotent)."""
        self._index_client.create_or_update_index(self.build_index_definition())

    # -- writes -------------------------------------------------------------------------------------

    def upsert(self, chunk_ids: list[int], vectors: list[list[float]]) -> None:
        """Upserts chunks by id when a `row_loader` was supplied; otherwise raises RuntimeError.

        The rows from the loader carry the text and metadata; `vectors` supplies the embeddings in
        the same order as `chunk_ids`.
        """
        if self._row_loader is None:
            raise RuntimeError(UPSERT_HINT)
        if len(chunk_ids) != len(vectors):
            raise ValueError("chunk_ids and vectors must have the same length.")
        by_id = {int(cid): vec for cid, vec in zip(chunk_ids, vectors, strict=True)}
        rows = []
        for row in self._row_loader(list(chunk_ids)):
            merged = dict(row)
            chunk_id = int(merged["chunk_id"])
            if chunk_id not in by_id:
                raise ValueError(
                    f"row_loader returned chunk {chunk_id}, which is outside the requested chunk_ids."
                )
            merged["vector"] = by_id[chunk_id]
            rows.append(merged)
        self.upsert_documents(rows)

    def upsert_documents(self, rows: Iterable[Mapping[str, Any]]) -> int:
        """Uploads chunk rows to the index in batches of `batch_size`; returns the number uploaded."""
        documents = [self.row_to_document(row) for row in rows]
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start : start + self.batch_size]
            results = self._search_client.upload_documents(batch)
            failed = [r.key for r in results or [] if getattr(r, "succeeded", True) is False]
            if failed:
                raise RuntimeError(f"Azure AI Search rejected {len(failed)} document(s): {failed[:10]}")
        return len(documents)

    def row_to_document(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Maps one chunk row (SQLite shape) to a search document."""
        vector = row.get("vector")
        if vector is None:
            vector = row.get("embedding")
        if vector is None:
            raise ValueError(f"Chunk row {row.get('chunk_id')!r} carries no vector or embedding.")
        vector = _as_float_list(vector)
        if len(vector) != self.dim:
            raise ValueError(
                f"Chunk {row.get('chunk_id')!r} vector is {len(vector)}-d; the index is {self.dim}-d."
            )
        chunk_id = int(row["chunk_id"])
        page = row.get("page")
        return {
            "id": str(chunk_id),
            "chunk_id": chunk_id,
            "document_id": int(row["document_id"]),
            "text": row.get("text") or "",
            "heading": row.get("heading"),
            "page": int(page) if page is not None else None,
            "source": row.get("source") or "",
            "title": row.get("title"),
            "acl_tags": _as_tag_list(row.get("acl_tags")),
            "quarantined": bool(row.get("quarantined", False)),
            VECTOR_FIELD: vector,
        }

    # -- reads --------------------------------------------------------------------------------------

    def search(self, vector: list[float], k: int, allowed_tags: list[str] | None = None) -> list[ChunkHit]:
        """Returns the k nearest non-quarantined chunks the caller may read, as ChunkHit."""
        if allowed_tags is not None and len(allowed_tags) == 0:
            return []
        vectorized_query_cls = _import("azure.search.documents.models", "VectorizedQuery")
        query = vectorized_query_cls(vector=list(vector), k_nearest_neighbors=k, fields=VECTOR_FIELD)
        results = self._search_client.search(
            search_text=None,
            vector_queries=[query],
            filter=build_filter(allowed_tags),
            vector_filter_mode="preFilter",
            select=list(SELECT_FIELDS),
            top=k,
        )
        return [_hit_from_document(doc) for doc in results]

    def count(self) -> int:
        """Returns the number of documents in the index."""
        return int(self._search_client.get_document_count())


def build_filter(allowed_tags: list[str] | None) -> str:
    """Builds the OData filter: quarantined chunks out, and with tags, only chunks carrying one of them.

    Example: `quarantined eq false and acl_tags/any(t: search.in(t, 'public,hr', ','))`.
    Tags are plain words; a tag containing a comma is rejected because the comma delimits the list.
    """
    clause = "quarantined eq false"
    if allowed_tags is None:
        return clause
    for tag in allowed_tags:
        if "," in tag:
            raise ValueError(f"ACL tag {tag!r} contains a comma; tags are plain words.")
    joined = ",".join(tag.replace("'", "''") for tag in allowed_tags)
    return f"{clause} and acl_tags/any(t: search.in(t, '{joined}', ','))"


def _hit_from_document(doc: Mapping[str, Any]) -> ChunkHit:
    page = doc.get("page")
    return ChunkHit(
        chunk_id=int(doc["chunk_id"]),
        score=float(doc.get("@search.score") or 0.0),
        text=doc.get("text") or "",
        document_id=int(doc["document_id"]),
        source=doc.get("source") or "",
        title=doc.get("title"),
        heading=doc.get("heading"),
        page=int(page) if page is not None else None,
        acl_tags=_as_tag_list(doc.get("acl_tags")),
        quarantined=bool(doc.get("quarantined", False)),
    )


def _as_tag_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        loaded = json.loads(value)
        return [str(t) for t in loaded]
    return [str(t) for t in value]


def _as_float_list(value: Any) -> list[float]:
    if isinstance(value, (bytes, bytearray, memoryview)):
        floats = array("f")
        floats.frombytes(bytes(value))
        return [float(x) for x in floats]
    return [float(x) for x in value]
