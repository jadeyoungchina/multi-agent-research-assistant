"""Release-history contracts, using disposable repositories for all ref writes."""

import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
ANCHORS = [
    "2026-07-06T20:18:00+08:00", "2026-07-13T21:07:00+08:00",
    "2026-07-22T22:16:00+08:00", "2026-07-30T20:43:00+08:00",
    "2026-08-08T21:26:00+08:00", "2026-08-17T22:09:00+08:00",
    "2026-08-25T20:51:00+08:00", "2026-09-01T21:34:00+08:00",
    "2026-09-05T22:12:00+08:00", "2026-09-08T21:40:00+08:00",
]
TOOLING_SUBJECT = "chore: add release verification tooling"
TARGET = "refs/heads/codex/rebuilt-main"
DISCLOSURE = (
    "This repository was reconstructed on 2026-09-08 after the original local project was accidentally deleted. "
    "Commit dates restore the documented development milestones; they are not the original Git objects "
    "or an unreconstructed historical record.\n\n"
    "本仓库于 2026-09-08 在原本地项目意外删除后重建。提交日期用于恢复已记录的开发里程碑；"
    "这些提交不是原始 Git 对象，也不是未经重建的历史记录。\n"
)


def sample_schedule() -> dict:
    return {
        "timezone": "Asia/Shanghai",
        "disclosure": "Transparent reconstruction after accidental deletion",
        "commits": [
            {"subject": f"milestone {index}", "timestamp": timestamp}
            for index, timestamp in enumerate(ANCHORS)
        ],
    }


def verifier_module():
    assert (ROOT / "scripts/verify_reconstructed_history.py").is_file(), "Missing history verifier"
    from scripts import verify_reconstructed_history
    return verify_reconstructed_history


def test_schedule_contains_approved_anchor_times_and_maps_actual_history() -> None:
    path = ROOT / "docs/reconstruction-schedule.json"
    assert path.is_file(), "Missing explicit reconstruction schedule"
    schedule = json.loads(path.read_text(encoding="utf-8"))
    assert set(ANCHORS) <= {entry["timestamp"] for entry in schedule["commits"]}
    result = subprocess.run(
        ["git", "-C", str(ROOT), "log", "--reverse", "--format=%s"],
        check=True, capture_output=True, text=True, encoding="utf-8",
    )
    subjects = result.stdout.splitlines()
    # The tooling commit is the sole approved pending entry while these tests
    # run before its initial commit; after commit the mapping must be exact.
    if TOOLING_SUBJECT not in subjects:
        subjects.append(TOOLING_SUBJECT)
    assert [entry["subject"] for entry in schedule["commits"]] == subjects
    assert list(verifier_module().validate_schedule(schedule)) == subjects


def test_schedule_validator_accepts_complete_ordered_anchors() -> None:
    assert list(verifier_module().validate_schedule(sample_schedule()).values()) == ANCHORS


@pytest.mark.parametrize("mutation, reason", [
    ("timezone", "timezone"), ("disclosure", "disclosure"),
    ("duplicate_subject", "duplicate subject"), ("duplicate_timestamp", "duplicate timestamp"),
    ("missing_anchor", "anchor"), ("reversed", "order"),
    ("bad_offset", "+08:00"), ("daytime", "20:00"),
    ("naive", "+08:00"), ("bad_date", "timestamp"),
    ("fraction", "timestamp"), ("empty_subject", "subject"),
    ("subject_newline", "subject"), ("wrong_entry", "entry"),
    ("wrong_type", "commits"), ("unknown_field", "fields"),
])
def test_schedule_validator_rejects_invalid_data(mutation: str, reason: str) -> None:
    data = copy.deepcopy(sample_schedule())
    if mutation in {"timezone", "disclosure"}:
        data[mutation] = "wrong"
    elif mutation == "duplicate_subject":
        data["commits"][1]["subject"] = data["commits"][0]["subject"]
    elif mutation == "duplicate_timestamp":
        data["commits"][1]["timestamp"] = ANCHORS[0]
    elif mutation == "missing_anchor":
        data["commits"].pop()
    elif mutation == "reversed":
        data["commits"].reverse()
    elif mutation in {"bad_offset", "daytime", "naive", "bad_date", "fraction"}:
        data["commits"][0]["timestamp"] = {
            "bad_offset": "2026-07-06T20:18:00+09:00",
            "daytime": "2026-07-06T19:18:00+08:00",
            "naive": "2026-07-06T20:18:00",
            "bad_date": "2026-02-30T20:18:00+08:00",
            "fraction": "2026-07-06T20:18:00.001+08:00",
        }[mutation]
    elif mutation in {"empty_subject", "subject_newline"}:
        data["commits"][0]["subject"] = "" if mutation == "empty_subject" else "one\ntwo"
    elif mutation == "wrong_entry":
        data["commits"][0] = None
    elif mutation == "wrong_type":
        data["commits"] = {}
    else:
        data["commits"][0]["unexpected"] = True
    with pytest.raises(ValueError, match=re.escape(reason)):
        verifier_module().validate_schedule(data)


def test_schedule_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "schedule.json"
    path.write_text('{"commits": [], "commits": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        verifier_module().load_schedule(path)


def git(root: Path, *args: str, input: bytes | None = None, env: dict | None = None) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True,
        input=input, env=env,
    ).stdout


def commit(root: Path, message: str, parents: list[str], date: str) -> str:
    tree = git(root, "write-tree").decode().strip()
    env = dict(os.environ, GIT_AUTHOR_NAME="Original 作者", GIT_AUTHOR_EMAIL="author@example.test",
               GIT_COMMITTER_NAME="Original Integrator", GIT_COMMITTER_EMAIL="integrator@example.test",
               GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
    args = ["commit-tree", tree]
    for parent in parents:
        args.extend(["-p", parent])
    return git(root, *args, input=message.encode("utf-8"), env=env).decode().strip()


@pytest.fixture
def history_repo(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repository with spaces"
    root.mkdir()
    git(root, "init", "-q", "--initial-branch=main")
    git(root, "config", "core.autocrlf", "false")
    git(root, "config", "commit.gpgsign", "false")
    for name in ("README.md", "CHANGELOG.md"):
        (root / name).write_text(DISCLOSURE, encoding="utf-8")
    parent: list[str] = []
    for index in range(10):
        (root / "payload.txt").write_text(str(index), encoding="utf-8")
        if index in {3, 7}:
            path = root / ("app/retrieval/loaders.py" if index == 3 else "app/evaluation/trace.py")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# adapted source\n", encoding="utf-8")
        git(root, "add", ".")
        oid = commit(root, f"milestone {index}\n\nFull body — 正文 {index}\n\nTrailer: preserved\n",
                     parent, "2026-09-10T20:00:00+08:00")
        parent = [oid]
    git(root, "update-ref", "refs/heads/main", oid)
    schedule = tmp_path / "schedule.json"
    schedule.write_text(json.dumps(sample_schedule()), encoding="utf-8")
    assert git(root, "status", "--porcelain") == b""
    return root, schedule


def cli(script: str, root: Path, schedule: Path, *args: str) -> subprocess.CompletedProcess[str]:
    path = ROOT / "scripts" / script
    assert path.is_file(), f"Missing {script}"
    return subprocess.run(
        [sys.executable, str(path), "--root", str(root), "--schedule", str(schedule), *args],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )


def rebuild(root: Path, schedule: Path, target: str = TARGET) -> subprocess.CompletedProcess[str]:
    return cli("rebuild_history.py", root, schedule, "--source-ref", "main", "--target-ref", target)


def assert_rejected_without_ref_changes(root: Path, schedule: Path, reason: str, target: str = TARGET) -> None:
    before = git(root, "show-ref")
    status = git(root, "status", "--porcelain=v1", "--untracked-files=all")
    result = rebuild(root, schedule, target)
    assert result.returncode == 1, result.stdout + result.stderr
    assert reason in (result.stdout + result.stderr).lower()
    assert git(root, "show-ref") == before
    assert git(root, "status", "--porcelain=v1", "--untracked-files=all") == status
    assert git(root, "symbolic-ref", "HEAD").strip() == b"refs/heads/main"


def test_rebuild_preserves_trees_full_messages_authors_and_sets_both_dates(history_repo) -> None:
    root, schedule = history_repo
    before = git(root, "rev-parse", "main")
    original = git(root, "rev-list", "--reverse", "main").decode().splitlines()
    result = rebuild(root, schedule)
    assert result.returncode == 0, result.stdout + result.stderr
    rebuilt = git(root, "rev-list", "--reverse", TARGET).decode().splitlines()
    assert len(rebuilt) == len(original) == 10
    for old, new, timestamp in zip(original, rebuilt, ANCHORS, strict=True):
        assert git(root, "rev-parse", f"{old}^{{tree}}") == git(root, "rev-parse", f"{new}^{{tree}}")
        assert git(root, "cat-file", "commit", old).split(b"\n\n", 1)[1] == git(
            root, "cat-file", "commit", new).split(b"\n\n", 1)[1]
        assert git(root, "show", "-s", "--format=%an <%ae>|%cn <%ce>", new) == (
            "Original 作者 <author@example.test>|Original Integrator <integrator@example.test>\n".encode()
        )
        assert git(root, "show", "-s", "--format=%aI%n%cI", new).decode().splitlines() == [timestamp, timestamp]
    assert git(root, "rev-parse", "main") == before
    assert git(root, "symbolic-ref", "HEAD").strip() == b"refs/heads/main"
    assert git(root, "status", "--porcelain") == b""
    verified = cli("verify_reconstructed_history.py", root, schedule, "--ref", TARGET, "--source-ref", "main")
    assert verified.returncode == 0, verified.stdout + verified.stderr
    assert "10 commits" in verified.stdout


@pytest.mark.parametrize("kind", ["untracked", "unstaged", "staged"])
def test_rebuild_refuses_dirty_worktree_without_changing_refs(history_repo, kind: str) -> None:
    root, schedule = history_repo
    path = root / ("untracked.txt" if kind == "untracked" else "payload.txt")
    path.write_text("user data", encoding="utf-8")
    if kind == "staged":
        git(root, "add", "payload.txt")
    assert_rejected_without_ref_changes(root, schedule, "dirty")


@pytest.mark.parametrize("target", ["refs/heads/main", "refs/heads/MAIN", "main", "refs/tags/release"])
def test_rebuild_refuses_unsafe_target_refs(history_repo, target: str) -> None:
    assert_rejected_without_ref_changes(*history_repo, "target", target)


@pytest.mark.parametrize("symbolic", [False, True])
def test_rebuild_refuses_existing_or_dangling_symbolic_target(history_repo, symbolic: bool) -> None:
    root, schedule = history_repo
    if symbolic:
        git(root, "symbolic-ref", TARGET, "refs/heads/absent")
    else:
        git(root, "update-ref", TARGET, "main")
    assert_rejected_without_ref_changes(root, schedule, "target")


@pytest.mark.parametrize("mutation, reason", [
    ("unmapped", "unmapped"), ("duplicate", "duplicate subject"),
    ("extra", "unused"), ("order", "order"), ("invalid", "timestamp"),
])
def test_rebuild_rejects_schedule_mismatch_before_ref_writes(history_repo, mutation: str, reason: str) -> None:
    root, schedule = history_repo
    data = sample_schedule()
    if mutation == "unmapped":
        data["commits"][0]["subject"] = "not in source"
    elif mutation == "duplicate":
        data["commits"][1]["subject"] = "milestone 0"
    elif mutation == "extra":
        data["commits"].append({"subject": "unused commit", "timestamp": "2026-09-08T21:41:00+08:00"})
    elif mutation == "order":
        data["commits"][0]["subject"], data["commits"][1]["subject"] = "milestone 1", "milestone 0"
    else:
        data["commits"][-1]["timestamp"] = "invalid"
    schedule.write_text(json.dumps(data), encoding="utf-8")
    assert_rejected_without_ref_changes(root, schedule, reason)


@pytest.mark.parametrize("kind", ["merge", "duplicate"])
def test_rebuild_refuses_non_linear_or_duplicate_subject_history(history_repo, kind: str) -> None:
    root, schedule = history_repo
    parent = git(root, "rev-parse", "main").decode().strip()
    parents = [parent]
    if kind == "merge":
        parents.append(git(root, "rev-parse", "main~2").decode().strip())
    oid = commit(root, "milestone 9\n", parents, "2026-09-10T20:00:00+08:00")
    git(root, "update-ref", "refs/heads/main", oid)
    assert_rejected_without_ref_changes(root, schedule, "merge" if kind == "merge" else "duplicate subject")


@pytest.mark.parametrize("path, timestamp", [
    ("app/retrieval/loaders.py", "2026-07-03T22:00:00+08:00"),
    ("app/evaluation/trace.py", "2026-08-27T22:00:00+08:00"),
    ("app/evaluation/metrics.py", "2026-08-27T22:00:00+08:00"),
])
def test_rebuild_enforces_upstream_cutoffs_by_changed_file_even_with_generic_subject(
    tmp_path: Path, path: str, timestamp: str,
) -> None:
    root = tmp_path / "early"
    root.mkdir()
    git(root, "init", "-q", "--initial-branch=main")
    for name in ("README.md", "CHANGELOG.md"):
        (root / name).write_text(DISCLOSURE, encoding="utf-8")
    data = sample_schedule()
    data["commits"].append({"subject": "generic adaptation", "timestamp": timestamp})
    data["commits"].sort(key=lambda entry: entry["timestamp"])
    parents = []
    for entry in data["commits"]:
        if entry["subject"] == "generic adaptation":
            adapted = root / path
            adapted.parent.mkdir(parents=True, exist_ok=True)
            adapted.write_text("# adapted\n", encoding="utf-8")
        git(root, "add", ".")
        oid = commit(root, entry["subject"] + "\n", parents, "2026-09-10T20:00:00+08:00")
        parents = [oid]
    git(root, "update-ref", "refs/heads/main", oid)
    schedule = tmp_path / "early.json"
    schedule.write_text(json.dumps(data), encoding="utf-8")
    assert_rejected_without_ref_changes(root, schedule, "upstream")


@pytest.mark.parametrize("name", ["README.md", "CHANGELOG.md"])
def test_verifier_checks_disclosures_in_requested_ref_not_worktree(history_repo, name: str) -> None:
    root, schedule = history_repo
    (root / name).write_text("Undisclosed history\n", encoding="utf-8")
    git(root, "add", name)
    parent = git(root, "rev-parse", "main~1").decode().strip()
    oid = commit(root, "milestone 9\n", [parent], ANCHORS[-1])
    git(root, "update-ref", "refs/heads/main", oid)
    # A truthful working copy cannot excuse a false committed disclosure.
    (root / name).write_text(DISCLOSURE, encoding="utf-8")
    result = cli("verify_reconstructed_history.py", root, schedule, "--ref", "main")
    assert result.returncode == 1
    assert "disclosure" in (result.stdout + result.stderr).lower()


def test_verifier_rejects_original_unscheduled_dates(history_repo) -> None:
    root, schedule = history_repo
    result = cli("verify_reconstructed_history.py", root, schedule, "--ref", "main")
    assert result.returncode == 1
    assert "date" in (result.stdout + result.stderr).lower()


@pytest.mark.parametrize("failure", ["git_failure", "wrong_final_tree", "dirty_during_build", "source_changed"])
def test_rebuild_aborts_before_target_creation_on_midflight_failure(history_repo, monkeypatch, failure: str) -> None:
    from scripts import rebuild_history as tool

    root, schedule = history_repo
    before = git(root, "show-ref")
    old_tip = git(root, "rev-parse", "main").decode().strip()
    previous_tip = git(root, "rev-parse", "main~1").decode().strip()
    previous_tree = git(root, "rev-parse", "main~1^{tree}").decode().strip()
    real_git = tool.git
    count = 0

    def with_failure(repo, *args, **kwargs):
        nonlocal count
        if "commit-tree" in args:
            count += 1
            if failure == "git_failure" and count == 2:
                raise ValueError("injected commit-tree failure")
            if count == 10:
                if failure == "wrong_final_tree":
                    args = list(args)
                    args[args.index("commit-tree") + 1] = previous_tree
                elif failure == "dirty_during_build":
                    (root / "payload.txt").write_text("concurrent user data", encoding="utf-8")
                elif failure == "source_changed":
                    git(root, "update-ref", "refs/heads/main", previous_tip, old_tip)
        return real_git(repo, *args, **kwargs)

    monkeypatch.setattr(tool, "git", with_failure)
    with pytest.raises(ValueError):
        tool.rebuild_history(root, "main", TARGET, schedule)
    assert not git(root, "for-each-ref", "--format=%(refname)", TARGET)
    if failure == "source_changed":
        assert git(root, "rev-parse", "main").decode().strip() == previous_tip
    else:
        assert git(root, "show-ref") == before
    if failure == "dirty_during_build":
        assert (root / "payload.txt").read_text(encoding="utf-8") == "concurrent user data"


@pytest.mark.parametrize("symbolic", [False, True])
def test_atomic_ref_creation_preserves_a_racing_target(history_repo, monkeypatch, symbolic: bool) -> None:
    from scripts import rebuild_history as tool

    root, schedule = history_repo
    original = git(root, "rev-parse", "main")
    real_git = tool.git

    def with_competing_writer(repo, *args, **kwargs):
        if args[0] == "update-ref":
            if symbolic:
                git(root, "symbolic-ref", TARGET, "refs/heads/missing-racing-target")
            else:
                git(root, "update-ref", TARGET, "main")
        return real_git(repo, *args, **kwargs)

    monkeypatch.setattr(tool, "git", with_competing_writer)
    with pytest.raises(ValueError):
        tool.rebuild_history(root, "main", TARGET, schedule)
    assert git(root, "rev-parse", "main") == original
    if symbolic:
        assert git(root, "symbolic-ref", TARGET).strip() == b"refs/heads/missing-racing-target"
    else:
        assert git(root, "rev-parse", TARGET) == original


def test_verifier_detects_a_different_source_tree(history_repo) -> None:
    root, schedule = history_repo
    result = rebuild(root, schedule)
    assert result.returncode == 0, result.stdout + result.stderr
    (root / "payload.txt").write_text("different source tree", encoding="utf-8")
    git(root, "add", "payload.txt")
    parent = git(root, "rev-parse", "main~1").decode().strip()
    oid = commit(root, "milestone 9\n", [parent], "2026-09-10T20:00:00+08:00")
    git(root, "update-ref", "refs/heads/main", oid)
    result = cli("verify_reconstructed_history.py", root, schedule, "--ref", TARGET, "--source-ref", "main")
    assert result.returncode == 1
    assert "final tree" in (result.stdout + result.stderr).lower()


def test_rebuild_accepts_english_readme_with_separate_chinese_translation(history_repo) -> None:
    root, schedule = history_repo
    (root / "README.md").write_text(DISCLOSURE.split("\n\n")[0] + "\n", encoding="utf-8")
    (root / "README.zh-CN.md").write_text(DISCLOSURE.split("\n\n")[1], encoding="utf-8")
    git(root, "add", ".")
    parent = git(root, "rev-parse", "main~1").decode().strip()
    oid = commit(root, "milestone 9\n", [parent], "2026-09-10T20:00:00+08:00")
    git(root, "update-ref", "refs/heads/main", oid)
    result = rebuild(root, schedule)
    assert result.returncode == 0, result.stdout + result.stderr
