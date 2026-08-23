"""Repository guards: the CI workflow runs the jobs it claims, the image excludes
data and models, and the threat model names test files that exist.

These checks belong to the repository rather than to any deployment, so they stay
here while deployment-shaped configuration lives with the deployment.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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
