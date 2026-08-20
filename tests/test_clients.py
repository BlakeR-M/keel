"""Client overlays (clients/<id>/): the tool policy loads and decides as intended, the overlay is
well-formed and points at files that exist, the manifest template tags every document, the runbook
covers the operating essentials, and no client data has crept into the repository.

Also guards the CI workflow and .dockerignore this lane owns: valid YAML with the expected jobs, and
the ignore entries that keep data and models out of the image.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from keel.agent.policy import Policy
from keel.agent.tools import ToolRegistry, default_registry
from keel.config import Settings
from keel.safety.pii import detect_only
from tests.fakes import FakeRetriever, make_hit

REPO_ROOT = Path(__file__).resolve().parent.parent
CLIENTS_DIR = REPO_ROOT / "clients"
CLIENTS = ("example-agency", "example-gym")

OVERLAY_FILES = ("keel.yaml", "policy.yaml", "corpus.manifest.yaml", "runbook.md", "needs-from-client.md")
REQUIRED_FILE_KEYS = ("policy", "manifest", "runbook", "checklist")


def read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(params=CLIENTS)
def client_dir(request: pytest.FixtureRequest) -> Path:
    """The overlay directory for one client."""
    return CLIENTS_DIR / request.param


@pytest.fixture
def overlay(client_dir: Path) -> dict[str, Any]:
    return read_yaml(client_dir / "keel.yaml")


@pytest.fixture
def manifest(client_dir: Path) -> dict[str, Any]:
    return read_yaml(client_dir / "corpus.manifest.yaml")


@pytest.fixture
def registry() -> ToolRegistry:
    """The built-in tool set with a canned retriever, so write flags come from the real ToolSpecs."""
    return default_registry(
        Settings(airgap=True),
        retriever=FakeRetriever([make_hit(1, "Carers sign in at the start of every shift.")]),
    )


@pytest.fixture
def policy(client_dir: Path, registry: ToolRegistry) -> Policy:
    return Policy.from_yaml(client_dir / "policy.yaml", registry=registry, airgap=True)


# ---------------------------------------------------------------------- policy.yaml


def test_overlay_directory_is_complete(client_dir: Path):
    missing = [name for name in OVERLAY_FILES if not (client_dir / name).is_file()]
    assert missing == [], f"{client_dir.name} is missing {missing}"


def test_policy_allows_search_docs(policy: Policy):
    decision = policy.check("search_docs", {"query": "shift sign-in"})
    assert decision.allowed is True
    assert decision.needs_approval is False


def test_policy_queues_create_ticket_for_approval(policy: Policy):
    decision = policy.check(
        "create_ticket", {"title": "Update handbook", "body": "Section 3 is out of date."}
    )
    assert decision.allowed is True
    assert decision.needs_approval is True
    assert policy.write_tools_require_approval is True


def test_policy_denies_a_tool_outside_the_allowlist(policy: Policy):
    assert "http_get" not in policy.allowed_tools
    decision = policy.check("http_get", {"url": "https://example.com"})
    assert decision.allowed is False
    assert "allowlist" in decision.reason

    unknown = policy.check("delete_everything", {})
    assert unknown.allowed is False


def test_policy_sql_rules_allow_listed_tables_only(policy: Policy):
    tables = policy.rules_for("sql_readonly").get("tables")
    assert isinstance(tables, list) and tables, "sql_readonly needs a table allowlist"
    assert policy.rules_for("sql_readonly").get("max_rows") == 50

    listed = policy.check("sql_readonly", {"query": f"SELECT * FROM {tables[0]}"})
    assert listed.allowed is True

    unlisted = policy.check("sql_readonly", {"query": "SELECT * FROM members"})
    assert unlisted.allowed is False
    assert "members" in unlisted.reason


def test_policy_http_rules_refuse_every_host_when_switched_on(client_dir: Path, registry: ToolRegistry):
    """The http_get rules block is a safe default: an empty host list refuses every host even if a
    later edit adds http_get to the allowlist and the box leaves air-gap."""
    config = read_yaml(client_dir / "policy.yaml")
    assert config["tool_arg_rules"]["http_get"]["hosts"] == []
    config["allowed_tools"] = [*config["allowed_tools"], "http_get"]
    switched_on = Policy(config, registry=registry, airgap=False)
    decision = switched_on.check("http_get", {"url": "https://example.com/page"})
    assert decision.allowed is False
    assert "example.com" in decision.reason


def test_policy_has_a_tool_call_budget(policy: Policy):
    assert isinstance(policy.max_tool_calls_per_request, int)
    assert 1 <= policy.max_tool_calls_per_request <= 10


# ---------------------------------------------------------------------- keel.yaml


def test_overlay_parses_with_the_shared_layout(overlay: dict[str, Any], client_dir: Path):
    for key in ("client", "settings", "roles", "users", "files"):
        assert key in overlay, f"{client_dir.name}/keel.yaml lacks the '{key}' block"
    assert overlay["client"]["id"] == client_dir.name
    assert isinstance(overlay["client"]["name"], str) and overlay["client"]["name"]


def test_overlay_settings_name_a_profile_and_data_dir(overlay: dict[str, Any]):
    settings = overlay["settings"]
    assert settings["profile"] in ("local", "azure")
    assert isinstance(settings["data_dir"], str) and settings["data_dir"]
    assert settings["airgap"] in (0, 1, True, False)
    # Every settings key is a Settings field, so the runbook's KEEL_ mapping resolves.
    unknown = set(settings) - set(Settings.model_fields)
    assert unknown == set(), f"settings keys with no Settings field: {sorted(unknown)}"


def test_overlay_references_existing_files(overlay: dict[str, Any], client_dir: Path):
    files = overlay["files"]
    for key in REQUIRED_FILE_KEYS:
        assert key in files, f"files.{key} is missing"
        target = client_dir / files[key]
        assert target.is_file(), f"files.{key} points at {target}, which does not exist"


def test_overlay_roles_are_tag_lists_and_the_owner_holds_every_tag(overlay: dict[str, Any]):
    roles = overlay["roles"]
    assert isinstance(roles, dict) and roles
    all_tags: set[str] = set()
    for role, tags in roles.items():
        assert isinstance(tags, list) and tags, f"role {role} carries no tags"
        assert all(isinstance(t, str) and t for t in tags), f"role {role} has a non-string tag"
        all_tags.update(tags)
    assert "owner" in roles
    assert set(roles["owner"]) == all_tags, "the owner role holds every tag any role holds"
    assert "public" in all_tags


def test_overlay_users_hold_known_roles_and_unique_ids(overlay: dict[str, Any]):
    users = overlay["users"]
    assert isinstance(users, list) and users
    ids = [u["user_id"] for u in users]
    assert len(ids) == len(set(ids)), "user ids repeat"
    for user in users:
        assert user["role"] in overlay["roles"], f"user {user['user_id']} has unknown role {user['role']}"
    assert any(u["role"] == "owner" for u in users)


# ---------------------------------------------------------------------- corpus.manifest.yaml


def test_manifest_parses_with_tagged_documents(manifest: dict[str, Any]):
    documents = manifest["documents"]
    assert isinstance(documents, list) and documents
    for position, entry in enumerate(documents):
        assert isinstance(entry, dict), f"documents[{position}] is not a mapping"
        assert isinstance(entry.get("path"), str) and entry["path"], f"documents[{position}] has no path"
        assert isinstance(entry.get("title"), str) and entry["title"], f"documents[{position}] has no title"
        tags = entry.get("acl_tags")
        assert isinstance(tags, list) and tags, f"documents[{position}] acl_tags must be a non-empty list"
        assert all(isinstance(t, str) and t for t in tags), f"documents[{position}] has a non-string tag"


def test_manifest_tags_are_all_reachable_by_some_role(manifest: dict[str, Any], overlay: dict[str, Any]):
    role_tags = {t for tags in overlay["roles"].values() for t in tags}
    for entry in manifest["documents"]:
        stray = set(entry["acl_tags"]) - role_tags
        assert stray == set(), f"{entry['path']} carries tags no role holds: {sorted(stray)}"


def test_manifest_paths_are_placeholders_not_repo_files(manifest: dict[str, Any], client_dir: Path):
    """The template stays a template: relative example paths, none of them present in the repo."""
    for entry in manifest["documents"]:
        path = Path(entry["path"])
        assert not path.is_absolute(), f"{entry['path']} is absolute; templates use relative example paths"
        assert not (client_dir / path).exists(), (
            f"{entry['path']} exists in the repo; client documents stay out"
        )


# ---------------------------------------------------------------------- runbook.md and checklist


def test_runbook_covers_approvals_acl_and_backup(client_dir: Path):
    text = (client_dir / "runbook.md").read_text(encoding="utf-8")
    for topic in ("approval", "ACL", "backup", "run.ps1", "ingest", "admin page", "verify-ledger"):
        assert topic.lower() in text.lower(), f"{client_dir.name}/runbook.md says nothing about {topic}"


def test_checklist_asks_for_documents_people_machine_and_approver(client_dir: Path):
    text = (client_dir / "needs-from-client.md").read_text(encoding="utf-8").lower()
    for topic in ("document", "role", "machine", "approve", "backup"):
        assert topic in text, f"{client_dir.name}/needs-from-client.md says nothing about {topic}"


# ---------------------------------------------------------------------- no client data in the repo


def _client_text_files() -> list[Path]:
    return sorted(p for p in CLIENTS_DIR.rglob("*") if p.is_file() and p.suffix in {".md", ".yaml", ".yml"})


@pytest.mark.parametrize("path", _client_text_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_client_files_carry_no_personal_identifiers(path: Path):
    text = path.read_text(encoding="utf-8")
    findings = detect_only(text)
    assert findings == [], f"{path.relative_to(REPO_ROOT)} contains {[f['kind'] for f in findings]}"
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text), "an email address slipped into a client file"


def test_clients_readme_states_the_no_client_data_rule():
    text = (CLIENTS_DIR / "README.md").read_text(encoding="utf-8")
    assert "client data never enters this repository" in text.lower()
    for client in CLIENTS:
        assert f"clients/{client}/" in text


# ---------------------------------------------------------------------- CI workflow and .dockerignore


def test_ci_workflow_is_valid_yaml_with_the_expected_jobs():
    workflow = read_yaml(REPO_ROOT / ".github" / "workflows" / "ci.yml")
    triggers = workflow.get("on") if "on" in workflow else workflow.get(True)  # YAML 1.1 reads `on` as True
    assert triggers is not None and "push" in triggers and "pull_request" in triggers
    jobs = workflow["jobs"]
    for job in ("lint", "test", "bicep", "eval-gate"):
        assert job in jobs, f"ci.yml lacks the {job} job"
        assert jobs[job]["runs-on"] == "ubuntu-latest"
    steps_text = yaml.safe_dump(jobs)
    assert "ruff check" in steps_text
    assert "not integration" in steps_text
    assert "bicep build deploy/azure/main.bicep" in steps_text
    assert "tests/test_evals.py" in steps_text
    assert "upload-artifact" in steps_text


def test_dockerignore_keeps_data_and_models_out_of_the_image():
    entries = {
        line.strip()
        for line in (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    for required in (
        ".venv",
        "data",
        ".git",
        "__pycache__",
        "*.db",
        "deploy/onprem/models",
        "reports",
        "demo-corpus",
        ".pytest_cache",
    ):
        assert required in entries, f".dockerignore lacks {required}"


def test_threat_model_names_test_files_that_exist():
    text = (REPO_ROOT / "docs" / "threat-model.md").read_text(encoding="utf-8")
    named = set(re.findall(r"tests/(test_\w+\.py)", text))
    assert named, "the threat model names no test files"
    missing = sorted(name for name in named if not (REPO_ROOT / "tests" / name).is_file())
    assert missing == [], f"threat model names tests that do not exist: {missing}"
