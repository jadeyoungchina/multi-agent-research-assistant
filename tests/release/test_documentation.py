"""Release acceptance requirements for the public documentation and CI."""

from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
DISCLOSURE = (
    "This repository was reconstructed on 2026-09-08 after the original local project was accidentally deleted. "
    "Commit dates restore the documented development milestones; they are not the original Git objects "
    "or an unreconstructed historical record."
)


@pytest.mark.parametrize("name", ["README.md", "README.zh-CN.md"])
def test_readmes_disclose_reconstruction_license_and_runnable_workflows(name: str) -> None:
    path = ROOT / name
    assert path.is_file(), f"Missing release documentation: {name}"
    text = path.read_text(encoding="utf-8")
    for required in (
        "qwen3.7-flash", "--provider fake", "--enforce-gate", "CHAT_PROVIDER=fake",
        "EMBEDDING_PROVIDER=fake", "/api/documents", "/api/research", "/events",
        "Last-Event-ID", "pytest", "THIRD_PARTY_NOTICES.md", "LICENSE",
        "NirDiamant/GenAI_Agents", "127.0.0.1",
    ):
        assert required in text, (name, required)
    if name == "README.md":
        assert DISCLOSURE in text
        assert "non-commercial" in text and "not affiliated" in text
    else:
        assert "重建" in text and "非商业" in text and "原始 Git 对象" in text


@pytest.mark.parametrize("name", [
    "CHANGELOG.md", "docs/architecture.md", "docs/experiments.md",
    "docs/reconstruction-history.md", "docs/technical-report.md",
    "docs/release-checklist.md", "THIRD_PARTY_NOTICES.md",
])
def test_required_documents_exist(name: str) -> None:
    assert (ROOT / name).is_file(), name


def test_changelog_starts_with_release_and_repeats_exact_disclosure() -> None:
    path = ROOT / "CHANGELOG.md"
    assert path.is_file(), "Missing changelog"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# 1.0.0 - 2026-09-08")
    assert DISCLOSURE in text


def test_ci_runs_offline_release_checks_and_uploads_benchmark_even_on_failure() -> None:
    path = ROOT / ".github/workflows/ci.yml"
    assert path.is_file(), "Missing offline CI workflow"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["test"]
    # Global overrides would invalidate tests of the application's real defaults.
    assert "CHAT_PROVIDER" not in job["env"] and "EMBEDDING_PROVIDER" not in job["env"]
    assert job["env"]["DASHSCOPE_API_KEY"] == job["env"]["OPENAI_API_KEY"] == ""
    setup = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/setup-python@"))
    assert setup["with"]["python-version"] == "3.11"
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert 'python -m pytest -m "not live" -q' in commands
    assert "python scripts/verify_repository.py" in commands
    assert "--provider fake --variants all" in commands and "--enforce-gate" in commands
    assert "--provider dashscope" not in commands and "--provider openai" not in commands
    benchmark = next(step for step in job["steps"] if "--provider fake" in step.get("run", ""))
    assert benchmark["env"]["CHAT_PROVIDER"] == benchmark["env"]["EMBEDDING_PROVIDER"] == "fake"
    upload = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/upload-artifact@"))
    assert upload["if"] == "always()"
    assert upload["with"]["path"] == "artifacts/evaluation/fake"
