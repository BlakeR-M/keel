"""Repository guards: the CI workflow runs the jobs it claims, the image excludes
data and models, and the threat model names test files that exist.

These checks belong to the repository rather than to any deployment, so they stay
here while deployment-shaped configuration lives with the deployment.
"""

from __future__ import annotations

import functools
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


@functools.lru_cache(maxsize=1)
def _collect() -> str:
    """One child-process collection, shared by every check below.

    A number in the README is a claim like any other, and this repository's whole argument is that
    its claims are checkable. Counting statically would miss parametrised cases, so this asks pytest,
    once: the collection costs a few seconds and several checks want the same answer.
    """
    completed = subprocess.run(  # noqa: S603 (fixed argv, no shell)
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "--quiet",
            "-p",
            "no:cacheprovider",
            "--override-ini=addopts=",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=300,
    )
    assert "::" in completed.stdout, "pytest collected nothing readable"
    return completed.stdout


def collected_node_ids() -> list[str]:
    return [line.strip() for line in _collect().splitlines() if "::" in line]


def collected_counts() -> tuple[int, int]:
    """(tests, files) as pytest collects them."""
    nodes = collected_node_ids()
    return len(nodes), len({node.split("::", 1)[0] for node in nodes})


def collected_test_names() -> set[str]:
    """Every collected test function by leaf name, parametrisation suffix dropped.

    Node ids carry the class for class-nested tests, so the leaf is the last `::` segment.
    """
    names = {node.split("[", 1)[0].split("::")[-1].strip() for node in collected_node_ids()}
    return {name for name in names if name.startswith("test_")}


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
    cases = sum(1 for node in collected_node_ids() if node.startswith("tests/redteam_"))
    functions = 0
    for path in sorted((REPO_ROOT / "tests").glob("redteam_*.py")):
        functions += len(re.findall(r"^def (test_\w+)", path.read_text(encoding="utf-8"), re.MULTILINE))
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"of which ([\d,]+) are adversarial cases from the ([\d,]+) attack tests", readme)
    assert match, "the README should state the adversarial case and attack-test counts"
    assert int(match.group(1).replace(",", "")) == cases, f"README says {match.group(1)} cases, pytest collects {cases}"
    assert int(match.group(2).replace(",", "")) == functions, f"README says {match.group(2)} attack tests, the files define {functions}"


def review_text() -> str:
    return (REPO_ROOT / "docs" / "security-review.md").read_text(encoding="utf-8")


def test_every_test_the_security_review_names_as_proof_actually_exists():
    """The review's proof column is the load-bearing claim of this repository. A row naming a test
    that pytest does not collect is worse than a row with no proof at all, so no name may be a
    truncation or a leftover from a rename."""
    named = {name for name in re.findall(r"(?<![\w])(test_[a-z0-9_]+)", review_text())}
    named -= {"test_safety"}  # the module, referenced as a path elsewhere in the same sentence
    missing = sorted(name for name in named if name not in collected_test_names())
    assert missing == [], f"the security review names tests pytest does not collect: {missing}"
    assert len(named) >= 25, f"only {len(named)} proving tests named; the table should cite far more"


def test_the_review_table_holds_the_number_of_findings_it_claims():
    text = review_text()
    rows = [line for line in text.splitlines() if re.match(r"\|\s*[A-Z]\d+\s*\|", line)]
    assert len(rows) == 27, f"the table has {len(rows)} finding rows"
    ids = [re.match(r"\|\s*([A-Z]\d+)\s*\|", line).group(1) for line in rows]
    assert len(set(ids)) == len(ids), "two findings share an id"
    assert "Twenty-seven findings" in text
    assert "Twenty-five are fixed" in text


def test_the_open_findings_the_review_declares_match_the_markers_in_the_tests():
    """Two findings stay open by choice. That honesty is the point, so the count in the prose and the
    number of strict expected-failure markers in the red-team files have to agree."""
    markers = 0
    for path in sorted((REPO_ROOT / "tests").glob("redteam_*.py")):
        markers += len(re.findall(r"pytest\.mark\.xfail\(\s*\n?\s*strict=True", path.read_text(encoding="utf-8")))
    assert markers == 2, f"{markers} strict xfail markers in the red-team files, the review declares 2"
    assert "Two stay open" in review_text()


def test_no_page_claims_every_medium_finding_was_fixed():
    """One medium stays open on purpose, so a page saying otherwise contradicts the review two
    paragraphs later. That is exactly the contradiction a reviewer of this repository would find."""
    pages = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "SECURITY.md",
        REPO_ROOT / "keel" / "web" / "templates" / "landing.html",
    ]
    overclaim = re.compile(r"every (?:finding of )?medium[- ](?:or[- ]higher|severity or above)", re.IGNORECASE)
    offenders = [page.name for page in pages if overclaim.search(page.read_text(encoding="utf-8"))]
    assert offenders == [], f"these pages claim every medium was fixed: {offenders}"


def test_pages_that_list_the_agent_tools_name_all_of_them():
    """`http_get` is the only tool that reaches outside the appliance, so a list that omits it
    understates the attack surface to the reader who most needs it."""
    import keel.agent.tools as agent_tools

    registered = set(re.findall(r'name="([a-z_]+)"', Path(agent_tools.__file__).read_text(encoding="utf-8")))
    assert {"search_docs", "calculator", "sql_readonly", "http_get", "create_ticket"} <= registered
    for path in (
        REPO_ROOT / "keel" / "web" / "templates" / "landing.html",
        REPO_ROOT / "keel" / "web" / "templates" / "chat.html",
        REPO_ROOT / "docs" / "tutorial.md",
    ):
        text = path.read_text(encoding="utf-8").lower()
        if "typed tool" not in text:
            continue
        assert "http" in text, f"{path.name} lists the agent tools without naming http_get"


def test_no_page_claims_a_number_of_human_reviewers():
    """The published pages describe the review method, which the repository evidences, rather than a
    headcount of people, which it does not.

    An earlier draft said three independent reviewers had been pointed at the security-bearing paths.
    Nothing here supports that, and on a repository whose argument is that its claims are checkable,
    an unverifiable one about provenance costs more than it earns.
    """
    claim = re.compile(
        r"\b(one|two|three|four|five|\d+)\s+(independent\s+|external\s+|third[- ]party\s+)?"
        r"(reviewers?|auditors?|assessors?|pen[- ]?testers?)\b",
        re.IGNORECASE,
    )
    pages = [REPO_ROOT / "README.md", REPO_ROOT / "SECURITY.md", REPO_ROOT / "CHANGELOG.md"]
    pages += sorted((REPO_ROOT / "docs").glob("*.md"))
    pages += sorted((REPO_ROOT / "keel" / "web" / "templates").glob("*.html"))
    offenders = [
        f"{page.name}: {claim.search(page.read_text(encoding='utf-8')).group(0)}"
        for page in pages
        if claim.search(page.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"unverifiable reviewer headcount on: {offenders}"
