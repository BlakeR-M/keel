"""Unit tests for the Azure providers (fake SDK clients, no network) and the AWS stubs."""

from __future__ import annotations

import importlib
import sys
from array import array
from types import SimpleNamespace
from unittest import mock

import pytest

from keel.providers import aws
from keel.providers import azure as azure_mod
from keel.providers.azure import AzureOpenAIChat, AzureOpenAIEmbeddings, AzureSearchIndex, build_filter
from keel.providers.base import (
    ChatMessage,
    ChatResult,
    ChunkHit,
    EmbeddingProvider,
    LLMProvider,
    ToolSpec,
    VectorIndex,
)

ENDPOINT = "https://example.openai.azure.com"
SEARCH_ENDPOINT = "https://example.search.windows.net"
API_VERSION = "2024-10-21"


# ---------------------------------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------------------------------


def _chat_response(content: str | None, tool_calls: list | None = None, prompt: int = 12, output: int = 5):
    """A chat completion in the wire shape (the shared mapper reads SDK objects and plain dicts alike)."""
    return {
        "choices": [{"message": {"role": "assistant", "content": content, "tool_calls": tool_calls}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": output},
        "model": "gpt-4o-mini-2024-07-18",
    }


def _tool_call(call_id: str, name: str, arguments: str) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _fake_openai_client() -> mock.MagicMock:
    return mock.MagicMock(name="AzureOpenAI")


def _fake_embeddings_response(dim: int, count: int):
    data = [SimpleNamespace(index=i, embedding=[float(i)] * dim) for i in reversed(range(count))]
    return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=3, total_tokens=3))


# ---------------------------------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------------------------------


class TestAzureOpenAIChat:
    def test_maps_messages_tools_and_parses_tool_calls_and_usage(self):
        client = _fake_openai_client()
        client.chat.completions.create.return_value = _chat_response(
            None, [_tool_call("call_1", "search_docs", '{"query": "salary bands", "k": 3}')]
        )
        provider = AzureOpenAIChat(ENDPOINT, "gpt-4o-mini", API_VERSION, client=client)
        assert isinstance(provider, LLMProvider)

        messages = [
            ChatMessage(role="system", content="You answer from the corpus."),
            ChatMessage(role="user", content="What are the salary bands?"),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[{"id": "call_0", "name": "search_docs", "arguments": {"query": "bands"}}],
            ),
            ChatMessage(role="tool", content="Band A: $62,000", tool_call_id="call_0", name="search_docs"),
        ]
        tools = [
            ToolSpec(
                name="search_docs",
                description="Search the corpus.",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ]
        result = provider.chat(messages, tools=tools, temperature=0.1, max_tokens=200)

        assert isinstance(result, ChatResult)
        create = client.chat.completions.create
        create.assert_called_once()
        kwargs = create.call_args.kwargs
        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["temperature"] == 0.1
        assert kwargs["max_tokens"] == 200
        assert "response_format" not in kwargs
        assert kwargs["messages"] == [
            {"role": "system", "content": "You answer from the corpus."},
            {"role": "user", "content": "What are the salary bands?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "search_docs", "arguments": '{"query": "bands"}'},
                    }
                ],
            },
            {"role": "tool", "content": "Band A: $62,000", "tool_call_id": "call_0"},
        ]
        assert kwargs["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "search_docs",
                    "description": "Search the corpus.",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
            }
        ]
        assert result.content == ""
        assert result.tool_calls == [
            {"id": "call_1", "name": "search_docs", "arguments": {"query": "salary bands", "k": 3}}
        ]
        assert result.prompt_tokens == 12
        assert result.output_tokens == 5
        assert result.model == "gpt-4o-mini-2024-07-18"
        assert result.raw is create.return_value
        assert provider.name == "azure-openai"

    def test_plain_answer_without_tools_omits_tools_kwarg(self):
        client = _fake_openai_client()
        client.chat.completions.create.return_value = _chat_response("Band A runs from $62,000 to $71,000.")
        provider = AzureOpenAIChat(ENDPOINT, "gpt-4o-mini", API_VERSION, client=client)

        result = provider.chat([ChatMessage(role="user", content="Band A?")])

        kwargs = client.chat.completions.create.call_args.kwargs
        assert "tools" not in kwargs
        assert "temperature" not in kwargs
        assert result.content == "Band A runs from $62,000 to $71,000."
        assert result.tool_calls == []

    def test_json_schema_maps_to_json_schema_response_format(self):
        client = _fake_openai_client()
        client.chat.completions.create.return_value = _chat_response('{"band": "A"}')
        provider = AzureOpenAIChat(ENDPOINT, "gpt-4o-mini", API_VERSION, client=client)
        schema = {"title": "BandAnswer", "type": "object", "properties": {"band": {"type": "string"}}}

        provider.chat([ChatMessage(role="user", content="Band?")], json_schema=schema)

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "out", "schema": schema},
        }

    def test_wire_shaped_tool_calls_pass_through_and_bad_json_is_kept_raw(self):
        client = _fake_openai_client()
        client.chat.completions.create.return_value = _chat_response(
            None, [_tool_call("c2", "calc", "{not json")]
        )
        provider = AzureOpenAIChat(ENDPOINT, "gpt-4o-mini", API_VERSION, client=client)
        wire_call = _tool_call("c1", "calc", '{"x": 1}')

        result = provider.chat([ChatMessage(role="assistant", content="", tool_calls=[wire_call])])

        sent = client.chat.completions.create.call_args.kwargs["messages"]
        assert sent == [{"role": "assistant", "content": "", "tool_calls": [wire_call]}]
        assert result.tool_calls == [
            {"id": "c2", "name": "calc", "arguments": {}, "arguments_raw": "{not json"}
        ]

    def test_healthy_reflects_client_outcome(self):
        client = _fake_openai_client()
        provider = AzureOpenAIChat(ENDPOINT, "gpt-4o-mini", API_VERSION, client=client)
        assert provider.healthy() is True
        client.models.list.assert_called_once_with()
        client.models.list.side_effect = RuntimeError("401")
        assert provider.healthy() is False

    def test_missing_endpoint_is_rejected(self):
        with pytest.raises(ValueError):
            AzureOpenAIChat("", "gpt-4o-mini", API_VERSION, client=_fake_openai_client())


# ---------------------------------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------------------------------


class TestAzureOpenAIEmbeddings:
    def test_embed_returns_dim_length_vectors_in_input_order(self):
        client = _fake_openai_client()
        client.embeddings.create.return_value = _fake_embeddings_response(dim=8, count=3)
        provider = AzureOpenAIEmbeddings(
            ENDPOINT, "text-embedding-3-small", API_VERSION, dim=8, client=client
        )
        assert isinstance(provider, EmbeddingProvider)

        vectors = provider.embed(["a", "b", "c"])

        assert len(vectors) == 3
        assert all(len(v) == 8 for v in vectors)
        assert [v[0] for v in vectors] == [0.0, 1.0, 2.0]
        kwargs = client.embeddings.create.call_args.kwargs
        assert kwargs["model"] == "text-embedding-3-small"
        assert kwargs["input"] == ["a", "b", "c"]
        assert kwargs["dimensions"] == 8

    def test_embed_batches_and_query_uses_first_vector(self):
        client = _fake_openai_client()
        client.embeddings.create.side_effect = [
            _fake_embeddings_response(dim=4, count=2),
            _fake_embeddings_response(dim=4, count=1),
        ]
        provider = AzureOpenAIEmbeddings(
            ENDPOINT, "text-embedding-3-small", API_VERSION, dim=4, client=client, batch_size=2
        )

        vectors = provider.embed(["a", "b", "c"])
        assert len(vectors) == 3
        assert client.embeddings.create.call_count == 2

        client.embeddings.create.side_effect = None
        client.embeddings.create.return_value = _fake_embeddings_response(dim=4, count=1)
        assert provider.embed_query("q") == [0.0, 0.0, 0.0, 0.0]

    def test_wrong_dimension_is_rejected(self):
        client = _fake_openai_client()
        client.embeddings.create.return_value = _fake_embeddings_response(dim=5, count=1)
        provider = AzureOpenAIEmbeddings(
            ENDPOINT, "text-embedding-3-small", API_VERSION, dim=8, client=client
        )
        with pytest.raises(ValueError):
            provider.embed(["a"])


# ---------------------------------------------------------------------------------------------------
# Search index
# ---------------------------------------------------------------------------------------------------


def _index(dim: int = 4, **kwargs) -> tuple[AzureSearchIndex, mock.MagicMock, mock.MagicMock]:
    search_client = mock.MagicMock(name="SearchClient")
    index_client = mock.MagicMock(name="SearchIndexClient")
    idx = AzureSearchIndex(
        SEARCH_ENDPOINT, "keel-chunks", dim, search_client=search_client, index_client=index_client, **kwargs
    )
    return idx, search_client, index_client


def _row(chunk_id: int, dim: int = 4, **overrides) -> dict:
    row = {
        "chunk_id": chunk_id,
        "document_id": 1,
        "text": f"chunk {chunk_id}",
        "heading": "Bands",
        "page": None,
        "source": "fixtures/corpus/hr-salary-bands.md",
        "title": "Northbank Salary Bands",
        "acl_tags": '["hr"]',
        "quarantined": 0,
        "vector": [0.1] * dim,
    }
    row.update(overrides)
    return row


class TestAzureSearchIndex:
    def test_ensure_index_creates_expected_fields_and_vector_profile(self):
        pytest.importorskip("azure.search.documents")
        idx, _, index_client = _index(dim=4)
        assert isinstance(idx, VectorIndex)

        idx.ensure_index()

        index_client.create_or_update_index.assert_called_once()
        definition = index_client.create_or_update_index.call_args.args[0]
        assert definition.name == "keel-chunks"
        names = [f.name for f in definition.fields]
        assert names == [
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
            "vector",
        ]
        by_name = {f.name: f for f in definition.fields}
        assert by_name["id"].key is True
        assert by_name["chunk_id"].filterable is True
        assert by_name["document_id"].filterable is True
        assert by_name["text"].searchable is True
        assert by_name["acl_tags"].filterable is True
        assert str(by_name["acl_tags"].type) == "Collection(Edm.String)"
        assert by_name["quarantined"].filterable is True
        assert str(by_name["vector"].type) == "Collection(Edm.Single)"
        assert by_name["vector"].vector_search_dimensions == 4
        assert by_name["vector"].vector_search_profile_name == "keel-hnsw-profile"
        assert [p.name for p in definition.vector_search.profiles] == ["keel-hnsw-profile"]
        assert definition.vector_search.profiles[0].algorithm_configuration_name == "keel-hnsw"
        assert [a.name for a in definition.vector_search.algorithms] == ["keel-hnsw"]
        assert definition.vector_search.algorithms[0].kind == "hnsw"

    def test_upsert_documents_batches_to_upload_documents(self):
        idx, search_client, _ = _index(dim=4, batch_size=2)
        rows = [_row(1), _row(2, quarantined=1, acl_tags=["public"]), _row(3, page=7)]

        uploaded = idx.upsert_documents(rows)

        assert uploaded == 3
        assert search_client.upload_documents.call_count == 2
        first, second = [call.args[0] for call in search_client.upload_documents.call_args_list]
        assert [d["id"] for d in first] == ["1", "2"]
        assert [d["id"] for d in second] == ["3"]
        assert first[0] == {
            "id": "1",
            "chunk_id": 1,
            "document_id": 1,
            "text": "chunk 1",
            "heading": "Bands",
            "page": None,
            "source": "fixtures/corpus/hr-salary-bands.md",
            "title": "Northbank Salary Bands",
            "acl_tags": ["hr"],
            "quarantined": False,
            "vector": [0.1, 0.1, 0.1, 0.1],
        }
        assert first[1]["quarantined"] is True
        assert first[1]["acl_tags"] == ["public"]
        assert second[0]["page"] == 7

    def test_upsert_documents_accepts_float32_blob_embeddings(self):
        idx, search_client, _ = _index(dim=3)
        blob = array("f", [1.0, 2.0, 3.0]).tobytes()
        row = _row(9, dim=3)
        del row["vector"]
        row["embedding"] = blob

        idx.upsert_documents([row])

        doc = search_client.upload_documents.call_args.args[0][0]
        assert doc["vector"] == [1.0, 2.0, 3.0]

    def test_upsert_documents_reports_rejected_documents(self):
        idx, search_client, _ = _index(dim=4)
        search_client.upload_documents.return_value = [
            SimpleNamespace(key="1", succeeded=False, error_message="boom")
        ]
        with pytest.raises(RuntimeError, match="rejected 1"):
            idx.upsert_documents([_row(1)])

    def test_upsert_without_row_loader_points_at_upsert_documents(self):
        idx, _, _ = _index(dim=4)
        with pytest.raises(RuntimeError, match="upsert_documents"):
            idx.upsert([1], [[0.1] * 4])

    def test_upsert_with_row_loader_merges_vectors_into_rows(self):
        def loader(chunk_ids: list[int]) -> list[dict]:
            return [_row(cid, vector=None) for cid in chunk_ids]

        idx, search_client, _ = _index(dim=2, row_loader=loader)

        idx.upsert([4, 5], [[1.0, 0.0], [0.0, 1.0]])

        docs = search_client.upload_documents.call_args.args[0]
        assert [(d["id"], d["vector"]) for d in docs] == [("4", [1.0, 0.0]), ("5", [0.0, 1.0])]

    def test_search_builds_acl_filter_and_maps_hits(self):
        pytest.importorskip("azure.search.documents")
        idx, search_client, _ = _index(dim=4)
        search_client.search.return_value = [
            {
                "@search.score": 0.83,
                "chunk_id": 42,
                "document_id": 3,
                "text": "Band A: $62,000 to $71,000.",
                "heading": "Bands",
                "page": None,
                "source": "fixtures/corpus/hr-salary-bands.md",
                "title": "Northbank Salary Bands",
                "acl_tags": ["hr"],
                "quarantined": False,
            }
        ]

        hits = idx.search([0.1, 0.2, 0.3, 0.4], k=5, allowed_tags=["public", "hr"])

        search_client.search.assert_called_once()
        kwargs = search_client.search.call_args.kwargs
        assert kwargs["filter"] == "quarantined eq false and acl_tags/any(t: search.in(t, 'public,hr', ','))"
        assert kwargs["search_text"] is None
        assert kwargs["top"] == 5
        assert kwargs["vector_filter_mode"] == "preFilter"
        assert "vector" not in kwargs["select"]
        query = kwargs["vector_queries"][0]
        assert query.vector == [0.1, 0.2, 0.3, 0.4]
        assert query.k_nearest_neighbors == 5
        assert query.fields == "vector"
        assert hits == [
            ChunkHit(
                chunk_id=42,
                score=0.83,
                text="Band A: $62,000 to $71,000.",
                document_id=3,
                source="fixtures/corpus/hr-salary-bands.md",
                title="Northbank Salary Bands",
                heading="Bands",
                page=None,
                acl_tags=["hr"],
                quarantined=False,
            )
        ]

    def test_search_without_tags_filters_quarantine_only_and_empty_tags_returns_nothing(self):
        pytest.importorskip("azure.search.documents")
        idx, search_client, _ = _index(dim=2)
        search_client.search.return_value = []

        idx.search([0.5, 0.5], k=3)
        assert search_client.search.call_args.kwargs["filter"] == "quarantined eq false"

        search_client.search.reset_mock()
        assert idx.search([0.5, 0.5], k=3, allowed_tags=[]) == []
        search_client.search.assert_not_called()

    def test_build_filter_escapes_quotes_and_rejects_commas(self):
        assert (
            build_filter(["o'brien"])
            == "quarantined eq false and acl_tags/any(t: search.in(t, 'o''brien', ','))"
        )
        with pytest.raises(ValueError):
            build_filter(["a,b"])

    def test_count_uses_document_count(self):
        idx, search_client, _ = _index()
        search_client.get_document_count.return_value = 17
        assert idx.count() == 17


# ---------------------------------------------------------------------------------------------------
# Credentials and import guard
# ---------------------------------------------------------------------------------------------------


class TestCredentials:
    def test_chat_uses_default_azure_credential_when_no_client_is_injected(self):
        pytest.importorskip("azure.identity")
        pytest.importorskip("openai")
        with mock.patch("azure.identity.DefaultAzureCredential") as credential_cls:
            provider = AzureOpenAIChat(ENDPOINT, "gpt-4o-mini", API_VERSION)
        credential_cls.assert_called_once_with()
        assert provider._client is not None

    def test_search_index_uses_default_azure_credential_once_for_both_clients(self):
        pytest.importorskip("azure.identity")
        pytest.importorskip("azure.search.documents")
        with (
            mock.patch("azure.identity.DefaultAzureCredential") as credential_cls,
            mock.patch("azure.search.documents.SearchClient") as search_cls,
            mock.patch("azure.search.documents.indexes.SearchIndexClient") as index_cls,
        ):
            AzureSearchIndex(SEARCH_ENDPOINT, "keel-chunks", 4)
        credential_cls.assert_called_once_with()
        search_cls.assert_called_once_with(SEARCH_ENDPOINT, "keel-chunks", credential_cls.return_value)
        index_cls.assert_called_once_with(SEARCH_ENDPOINT, credential_cls.return_value)

    def test_explicit_credential_skips_default_credential(self):
        pytest.importorskip("azure.identity")
        pytest.importorskip("openai")
        with mock.patch("azure.identity.DefaultAzureCredential") as credential_cls:
            AzureOpenAIEmbeddings(
                ENDPOINT, "text-embedding-3-small", API_VERSION, credential=mock.MagicMock()
            )
        credential_cls.assert_not_called()


class _BlockAzureFinder:
    """Meta-path finder that reports every azure.* package as absent."""

    @staticmethod
    def find_spec(name, path=None, target=None):
        if name == "azure" or name.startswith("azure."):
            raise ImportError(f"No module named {name!r}")
        return None


class TestImportGuard:
    def test_module_imports_without_azure_packages_and_names_the_install(self, monkeypatch):
        for name in list(sys.modules):
            if name == "azure" or name.startswith("azure."):
                monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.setattr(sys, "meta_path", [_BlockAzureFinder, *sys.meta_path])

        module = importlib.reload(azure_mod)
        with pytest.raises(RuntimeError, match=r"pip install \"keel\[azure\]\""):
            module.AzureOpenAIChat(ENDPOINT, "gpt-4o-mini", API_VERSION)
        with pytest.raises(RuntimeError, match=r"pip install \"keel\[azure\]\""):
            module.AzureSearchIndex(SEARCH_ENDPOINT, "keel-chunks", 4)


# ---------------------------------------------------------------------------------------------------
# AWS stubs
# ---------------------------------------------------------------------------------------------------


class TestAwsStubs:
    @pytest.mark.parametrize(
        "factory",
        [
            lambda: aws.BedrockChat("ap-southeast-2", "amazon.nova-lite-v1:0"),
            lambda: aws.BedrockEmbeddings("ap-southeast-2", "amazon.titan-embed-text-v2:0", dim=1024),
            lambda: aws.OpenSearchServerlessIndex(
                "https://abc.ap-southeast-2.aoss.amazonaws.com", "keel", 1024, "ap-southeast-2"
            ),
        ],
    )
    def test_valid_config_raises_not_implemented_with_readme_pointer(self, factory):
        with pytest.raises(NotImplementedError, match=r"deploy/aws/README\.md"):
            factory()

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: aws.BedrockChat("", "amazon.nova-lite-v1:0"),
            lambda: aws.BedrockChat("ap-southeast-2", ""),
            lambda: aws.BedrockEmbeddings("ap-southeast-2", "amazon.titan-embed-text-v2:0", dim=0),
            lambda: aws.OpenSearchServerlessIndex("http://plain", "keel", 1024, "ap-southeast-2"),
            lambda: aws.OpenSearchServerlessIndex(
                "https://abc.aoss.amazonaws.com", "", 1024, "ap-southeast-2"
            ),
        ],
    )
    def test_invalid_config_is_rejected_before_the_stub_error(self, factory):
        with pytest.raises(ValueError):
            factory()

    def test_stub_pointer_names_the_readme(self):
        assert "deploy/aws/README.md" in aws.AWS_STUB_POINTER
