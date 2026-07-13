import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _is_ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", "--", path],
        cwd=REPO_ROOT,
        check=False,
    )
    return result.returncode == 0


def test_generated_data_is_ignored_but_gitkeep_is_allowed() -> None:
    assert _is_ignored("data/generated.txt")
    assert _is_ignored("logs/generated.log")
    assert not _is_ignored("data/.gitkeep")
    assert not _is_ignored("logs/.gitkeep")
    assert not _is_ignored("evaluation/results/.gitkeep")
