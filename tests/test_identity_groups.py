"""Directory groups from the authenticating proxy: `X-Keel-Groups` mapped to tags by `KEEL_GROUP_TAGS`.

The pure mapping is tested on its own, then end to end through the source viewer beyond loopback: a
user whose Entra ID group maps to `hr` opens an hr chunk, a user in an unmapped group does not, and the
groups header earns nothing without the proxy token. Same AppContext recipe as tests/test_web.py.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from keel.config import Settings
from keel.ingest import ingest_manifest
from keel.providers import factory
from keel.providers.factory import AppContext, build_context
from keel.providers.local_index import SqliteVectorIndex
from keel.web.app import PROXY_TOKEN_ENV, app
from keel.web.views import group_tags_for, proxy_user
from tests.fakes import FakeLLM

FIXTURE_MANIFEST = Path(__file__).resolve().parent.parent / "fixtures" / "corpus.yaml"
HR_GROUP = "6f1c2a9e-3b4d-4e5f-8a7b-0c1d2e3f4a5b"
FINANCE_GROUP = "Finance Staff"
GROUP_MAP = {HR_GROUP: "hr", FINANCE_GROUP: ["finance", "procurement"]}
PROXY_TOKEN = "test-proxy-token"


# ---------------------------------------------------------------------- the mapping


def test_mapped_groups_earn_their_tags_in_order_without_repeats() -> None:
    groups = f"{FINANCE_GROUP},{HR_GROUP},{HR_GROUP}"
    assert group_tags_for(groups, GROUP_MAP) == ["finance", "procurement", "hr"]


def test_group_matching_ignores_case() -> None:
    assert group_tags_for(HR_GROUP.upper(), GROUP_MAP) == ["hr"]
    assert group_tags_for("finance staff", GROUP_MAP) == ["finance", "procurement"]


def test_an_unmapped_group_grants_nothing() -> None:
    assert group_tags_for("Domain Admins", GROUP_MAP) == []
    assert group_tags_for(HR_GROUP, {}) == []
    assert group_tags_for(HR_GROUP, None) == []


def test_proxy_user_combines_explicit_tags_and_group_tags() -> None:
    user = proxy_user("emma", "public", HR_GROUP, GROUP_MAP)
    assert user.user_id == "emma"
    assert user.tags == ["public", "hr"]


def test_proxy_user_with_only_unmapped_groups_is_public() -> None:
    assert proxy_user("emma", None, "Domain Admins", GROUP_MAP).tags == ["public"]


# ---------------------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def groups_ctx(tmp_path_factory: pytest.TempPathFactory, embedder, reranker) -> Iterator[AppContext]:
    settings = Settings(data_dir=tmp_path_factory.mktemp("groups"), airgap=False, group_tags=GROUP_MAP)

    def fake_local_providers(settings_: Settings, conn):
        return FakeLLM(), embedder, SqliteVectorIndex(conn, embed_model=embedder.name), reranker

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(factory, "_local_providers", fake_local_providers)
        ctx = build_context(settings)
    ingest_manifest(
        ctx.conn, ctx.settings, ctx.embedder, ctx.index, FIXTURE_MANIFEST, screen=ctx.screen, ledger=ctx.ledger
    )
    yield ctx
    ctx.close()


@pytest.fixture
def remote_client(groups_ctx: AppContext, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A client against the app as deployed beyond loopback, trusting a proxy with PROXY_TOKEN."""
    monkeypatch.setenv(PROXY_TOKEN_ENV, PROXY_TOKEN)
    original = groups_ctx.settings.host
    groups_ctx.settings.host = "0.0.0.0"
    app.state.ctx = groups_ctx
    app.state.ctx_owned = False
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.state.ctx = None
        groups_ctx.settings.host = original


def hr_chunk_id(ctx: AppContext) -> int:
    row = ctx.conn.execute(
        "SELECT id FROM chunks WHERE EXISTS (SELECT 1 FROM json_each(acl_tags) WHERE value = 'hr') ORDER BY id"
    ).fetchone()
    assert row is not None
    return int(row[0])


def source_as(client: TestClient, chunk_id: int, headers: dict[str, str]):
    return client.get(f"/source/{chunk_id}", headers={"accept": "application/json", **headers})


def test_a_user_in_the_mapped_hr_group_opens_an_hr_chunk(
    groups_ctx: AppContext, remote_client: TestClient
) -> None:
    response = source_as(
        remote_client,
        hr_chunk_id(groups_ctx),
        {"X-Keel-Proxy-Token": PROXY_TOKEN, "X-Keel-User": "emma", "X-Keel-Groups": HR_GROUP},
    )
    assert response.status_code == 200, response.text
    assert "hr" in response.json()["user"]["tags"]


def test_a_user_in_an_unmapped_group_is_refused_the_hr_chunk(
    groups_ctx: AppContext, remote_client: TestClient
) -> None:
    response = source_as(
        remote_client,
        hr_chunk_id(groups_ctx),
        {"X-Keel-Proxy-Token": PROXY_TOKEN, "X-Keel-User": "sam", "X-Keel-Groups": "Domain Admins"},
    )
    assert response.status_code in (401, 403), response.text


def test_groups_earn_nothing_without_the_proxy_token(
    groups_ctx: AppContext, remote_client: TestClient
) -> None:
    response = source_as(
        remote_client, hr_chunk_id(groups_ctx), {"X-Keel-User": "emma", "X-Keel-Groups": HR_GROUP}
    )
    assert response.status_code in (401, 403), response.text
