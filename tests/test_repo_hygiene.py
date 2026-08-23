"""Repository guards: the CI workflow runs the jobs it claims, the image excludes
data and models, and the threat model names test files that exist.

These checks belong to the repository rather than to any deployment, so they stay
here while deployment-shaped configuration lives with the deployment.
"""

from __future__ import annotations

import re
import subprocess
import sys
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


def collected_counts() -> tuple[int, int]:
    """(tests, files) as pytest collects them, from a child process.

    A number in the README is a claim like any other, and this repository's whole argument is that
    its claims are checkable. Counting statically would miss parametrised cases, so this asks pytest.
    """
    completed = subprocess.run(  # noqa: S603 (fixed argv, no shell)
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=300,
    )
    per_file = re.findall(r"^tests/\S+\.py: (\d+)$", completed.stdout, re.MULTILINE)
    assert per_file, f"pytest collected nothing readable:\n{completed.stdout[-2000:]}"
    return sum(int(count) for count in per_file), len(per_file)


def test_the_readme_reports_the_number_of_tests_there_actually_are():
    tests, files = collected_counts()
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"The full suite is ([\d,]+) tests across (\d+) files", readme)
    assert match, "the README should state the suite size"
    assert int(match.group(1).replace(",", "")) == tests, f"the README says {match.group(1)}, pytest collects {tests}"
    assert int(match.group(2)) == files, f"the README says {match.group(2)} files, pytest collects {files}"


def test_the_overview_page_reports_the_same_number_as_the_readme():
    """The landing page states the suite size to a reader who never opens the repository. It has to
    agree with the README rather than drift into a rounder, friendlier number."""
    tests, files = collected_counts()
    page = (REPO_ROOT / "keel" / "web" / "templates" / "landing.html").read_text(encoding="utf-8")
    match = re.search(r'<span class="stat-n">(\d+)</span><span class="stat-k">tests, (\d+) files', page)
    assert match, "the overview page should state the suite size"
    assert (int(match.group(1)), int(match.group(2))) == (tests, files)


def test_the_adversarial_case_count_matches_the_red_team_files():
    """105 attack tests expanding to 174 cases is the headline number on the overview page and in the
    README. It comes from the three red-team files and nowhere else."""
    completed = subprocess.run(  # noqa: S603 (fixed argv, no shell)
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            *[str(path) for path in sorted((REPO_ROOT / "tests").glob("redteam_*.py"))],
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=300,
    )
    cases = sum(int(n) for n in re.findall(r"^tests/\S+\.py: (\d+)$", completed.stdout, re.MULTILINE))
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"of which ([\d,]+) are adversarial cases", readme)
    assert match, "the README should state the adversarial case count"
    assert int(match.group(1).replace(",", "")) == cases, f"README says {match.group(1)}, pytest collects {cases}"
