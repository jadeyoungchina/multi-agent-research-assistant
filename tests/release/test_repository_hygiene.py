"""Exercise the release verifier against real temporary Git indexes."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/verify_repository.py"
PIN = "4c95ae14cc2462c442b5c064cccd74430d02bc46"


def git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md", ".env.example"):
        shutil.copy2(ROOT / name, tmp_path / name)
    for name in ("THIRD_PARTY_LICENSES", "benchmarks", "app"):
        shutil.copytree(ROOT / name, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__"))
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    return tmp_path


def verify(repository: Path) -> subprocess.CompletedProcess[str]:
    assert SCRIPT.is_file(), "Missing tracked-file repository verifier"
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(repository)],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )


def test_valid_repository_passes_and_untracked_local_data_is_not_scanned(repository: Path) -> None:
    (repository / ".env").write_text("DASHSCOPE_API_KEY=" + "sk-" + "A7b9" * 8, encoding="utf-8")
    (repository / "notes-untracked.pdf").write_bytes(b"private")
    result = verify(repository)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "30 benchmark cases" in result.stdout


@pytest.mark.parametrize("name", [
    ".env", ".env.production", "nested/.env.local", "data/.gitkeep", "data/source.txt",
    "artifacts/evaluation/fake/results.json", "nested/uploads/paper.md", "uploads/paper.txt",
    "research.db", "backup.sqlite3", "backup.sqlite-wal", "private.pdf", "private.docx",
    "evaluation/results/report.md", "results.json", "results.csv", "credential.key",
])
def test_verifier_rejects_tracked_runtime_files(repository: Path, name: str) -> None:
    path = repository / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("local runtime data", encoding="utf-8")
    git(repository, "add", "--", name)
    result = verify(repository)
    assert result.returncode == 1
    assert name in result.stdout + result.stderr


@pytest.mark.parametrize("prefix", ["sk-", "sk-proj-", "sk-svcacct-"])
def test_verifier_rejects_key_patterns_without_printing_credentials(repository: Path, prefix: str) -> None:
    secret = prefix + "A7b9Q2x8" * 5
    (repository / "notes.md").write_text("accidental token: " + secret, encoding="utf-8")
    git(repository, "add", "notes.md")
    result = verify(repository)
    assert result.returncode == 1
    assert "notes.md" in result.stdout + result.stderr
    assert secret not in result.stdout + result.stderr


@pytest.mark.parametrize("value", ["replace-me", '""', "abc123", "${DASHSCOPE_API_KEY}"])
def test_example_requires_literally_blank_key_assignments(repository: Path, value: str) -> None:
    path = repository / ".env.example"
    path.write_text("DASHSCOPE_API_KEY=" + value + "\nOPENAI_API_KEY=\n", encoding="utf-8")
    result = verify(repository)
    assert result.returncode == 1
    assert ".env.example" in result.stdout + result.stderr


def test_example_requires_both_provider_key_assignments(repository: Path) -> None:
    (repository / ".env.example").write_text("CHAT_PROVIDER=fake\n", encoding="utf-8")
    assert verify(repository).returncode == 1


@pytest.mark.parametrize("name", ["LICENSE", "THIRD_PARTY_LICENSES/GenAI_Agents-LICENSE.txt"])
def test_verifier_rejects_changed_license(repository: Path, name: str) -> None:
    with (repository / name).open("a", encoding="utf-8") as handle:
        handle.write("\nAltered terms\n")
    result = verify(repository)
    assert result.returncode == 1
    assert name in result.stdout + result.stderr


def test_verifier_accepts_crlf_license_normalization(repository: Path) -> None:
    for name in ("LICENSE", "THIRD_PARTY_LICENSES/GenAI_Agents-LICENSE.txt"):
        path = repository / name
        path.write_bytes(path.read_text(encoding="utf-8").replace("\n", "\r\n").encode())
    result = verify(repository)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("mutation", ["missing_case", "duplicate_id", "absent_phrase"])
def test_verifier_validates_all_cases_and_evidence_phrases(repository: Path, mutation: str) -> None:
    path = repository / "benchmarks/cases.jsonl"
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if mutation == "missing_case":
        cases.pop()
    elif mutation == "duplicate_id":
        cases[-1]["id"] = cases[0]["id"]
    else:
        cases[-1]["expected_evidence"][0]["contains"] = "a phrase absent from this corpus"
    path.write_text("\n".join(json.dumps(case) for case in cases), encoding="utf-8")
    result = verify(repository)
    assert result.returncode == 1
    assert "benchmark" in (result.stdout + result.stderr).lower()


@pytest.mark.parametrize("mutation", ["missing", "unpinned", "untracked"])
def test_verifier_requires_each_adapted_target_to_exist_be_tracked_and_pin_source(repository: Path, mutation: str) -> None:
    path = repository / "app/agents/planner.py"
    if mutation == "missing":
        path.unlink()
    elif mutation == "untracked":
        git(repository, "rm", "--cached", "app/agents/planner.py")
    else:
        path.write_text(path.read_text(encoding="utf-8").replace(PIN, "wrong-commit"), encoding="utf-8")
    result = verify(repository)
    assert result.returncode == 1
    assert "app/agents/planner.py" in result.stdout + result.stderr


def test_concept_only_targets_are_not_misclassified_as_adapted(repository: Path) -> None:
    path = repository / "THIRD_PARTY_NOTICES.md"
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n| `concept.ipynb` | 1 | 2024-01-01 | `future/concept.py` | concept-only |\n")
    result = verify(repository)
    assert result.returncode == 0, result.stdout + result.stderr


def test_runtime_data_is_not_tracked_in_real_repository() -> None:
    result = verify(ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


def test_example_environment_contains_no_secret() -> None:
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    for name in ("DASHSCOPE_API_KEY", "OPENAI_API_KEY"):
        assert [line for line in lines if line.startswith(name + "=")] == [name + "="]
