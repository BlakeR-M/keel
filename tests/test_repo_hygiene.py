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


def tracked_files() -> list[Path]:
    """Every file git tracks, which is exactly what a reader of the repository receives."""
    listed = subprocess.run(  # noqa: S603 (fixed argv, no shell)
        [_git(), "ls-files"], capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120
    )
    return [REPO_ROOT / name for name in listed.stdout.split() if (REPO_ROOT / name).is_file()]


def _git() -> str:
    from shutil import which

    return which("git") or "git"


#: Characters that carry no visible mark. A run of them can encode a signature into prose that reads
#: normally, so a repository whose argument is that its claims are checkable should carry none.
INVISIBLE = {
    0x00AD: "SOFT HYPHEN",
    0x061C: "ARABIC LETTER MARK",
    0x180E: "MONGOLIAN VOWEL SEPARATOR",
    0x200B: "ZERO WIDTH SPACE",
    0x200C: "ZERO WIDTH NON-JOINER",
    0x200D: "ZERO WIDTH JOINER",
    0x200E: "LEFT-TO-RIGHT MARK",
    0x200F: "RIGHT-TO-LEFT MARK",
    0x2028: "LINE SEPARATOR",
    0x2029: "PARAGRAPH SEPARATOR",
    0x202A: "LEFT-TO-RIGHT EMBEDDING",
    0x202B: "RIGHT-TO-LEFT EMBEDDING",
    0x202C: "POP DIRECTIONAL FORMATTING",
    0x202D: "LEFT-TO-RIGHT OVERRIDE",
    0x202E: "RIGHT-TO-LEFT OVERRIDE",
    0x2060: "WORD JOINER",
    0x2066: "LEFT-TO-RIGHT ISOLATE",
    0x2067: "RIGHT-TO-LEFT ISOLATE",
    0x2068: "FIRST STRONG ISOLATE",
    0x2069: "POP DIRECTIONAL ISOLATE",
    0xFEFF: "ZERO WIDTH NO-BREAK SPACE",
}
INVISIBLE.update({code: "VARIATION SELECTOR" for code in range(0xFE00, 0xFE10)})
INVISIBLE.update({code: "TAG CHARACTER" for code in range(0xE0000, 0xE0080)})


def test_no_tracked_file_carries_an_invisible_character():
    """Zero-width and bidirectional characters read as nothing and can encode a mark into prose.

    Nothing here needs them, so their absence is worth asserting rather than assuming. Tag characters
    and variation selectors are covered too: both are the usual vehicle for hiding text in plain sight.
    """
    offenders = []
    for path in tracked_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # a binary file carries no prose to hide anything in
        for index, character in enumerate(text):
            if ord(character) in INVISIBLE:
                line = text.count("\n", 0, index) + 1
                offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{line} {INVISIBLE[ord(character)]}")
    assert offenders == [], f"invisible characters found: {offenders[:10]}"


#: Built from its codepoint, so this file, which every check here reads, carries none itself.
EM_DASH = chr(0x2014)


def test_no_tracked_file_carries_an_em_dash():
    """Em dashes are the loudest punctuation tell in generated prose, so the house style drops them.

    En dashes in number ranges stay welcome; this is about U+2014 alone.
    """
    offenders = []
    for path in tracked_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if EM_DASH in text:
            line = text.count("\n", 0, text.index(EM_DASH)) + 1
            offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{line}")
    assert offenders == [], f"em dashes found: {offenders}"


def test_no_local_assistant_or_editor_configuration_is_published():
    """A cloned repository should carry the project, not whoever happened to build it.

    Assistant and editor directories are local working state. They are ignored in
    `.git/info/exclude`, which does the same job without naming anybody's tooling in a file every
    reader receives.
    """
    private = (".claude", "claude.md", ".cursor", ".aider", ".continue", ".windsurf", ".github/copilot")
    tracked = {path.relative_to(REPO_ROOT).as_posix().lower() for path in tracked_files()}
    offenders = sorted(
        name for name in tracked if any(name == p or name.startswith(f"{p}/") for p in private)
    )
    assert offenders == [], f"local tooling configuration is tracked: {offenders}"
    ignore_file = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").lower()
    assert "claude" not in ignore_file, ".gitignore names a specific assistant; keep that local"


def test_every_command_the_cli_offers_is_documented():
    """A command nobody wrote down is a command nobody finds.

    `docs/cli.md` is the reference a reader reaches for, and four commands went undocumented for a
    day before a staleness scan caught them. This closes that gap by asking typer what exists rather
    than trusting a list.
    """
    from keel.cli import app as cli

    def leaf_names(typer_app) -> set[str]:  # noqa: ANN001
        return {
            command.name or (command.callback.__name__ if command.callback else "")
            for command in getattr(typer_app, "registered_commands", [])
        }

    names = leaf_names(cli)
    for group in cli.registered_groups:
        group_name = group.name or ""
        names.add(group_name)
        # Subcommands too: `keel documents clear` shipped a day after its siblings and would have
        # slipped through a check that only asked for the group.
        names.update(f"{group_name} {leaf}" for leaf in leaf_names(group.typer_instance) if leaf)
    names = {name.replace("_", "-") for name in names if name}
    assert names, "typer reported no commands, so this check is not reading what it thinks"

    documented = (REPO_ROOT / "docs" / "cli.md").read_text(encoding="utf-8")
    missing = sorted(name for name in names if f"keel {name}" not in documented)
    assert missing == [], f"these commands are absent from docs/cli.md: {missing}"


def test_every_route_the_web_app_serves_is_documented():
    """The same for the web app: `docs/web.md` lists the routes, so a new one belongs there too."""
    from keel.web.app import app as web

    rows = []
    for route in web.routes:
        included = getattr(route, "original_router", None)
        rows.extend(included.routes if included is not None else [route])
    paths = {
        getattr(route, "path", "")
        for route in rows
        if getattr(route, "path", "").startswith(("/admin", "/api", "/docs", "/chat", "/about", "/ask"))
    }
    documented = (REPO_ROOT / "docs" / "web.md").read_text(encoding="utf-8")
    missing = sorted(path for path in paths if path and path not in documented)
    assert missing == [], f"these routes are absent from docs/web.md: {missing}"
