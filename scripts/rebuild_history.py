"""Create a new branch with disclosed scheduled dates; never switch or alter main.

All validation and commit creation complete before one create-only ref update.
Failures can leave unreachable commit objects, but no existing ref is changed.
Create a verified backup before using this separately authorized release tool.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_reconstructed_history import (
    compare_histories, git, load_schedule, read_history, resolve_commit, validate_history,
)


def require_clean_worktree(root: Path) -> None:
    if git(root, "rev-parse", "--is-inside-work-tree").stdout.strip() != b"true":
        raise ValueError("a non-bare worktree is required")
    if git(root, "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none").stdout:
        raise ValueError("dirty worktree: commit or preserve changes before reconstruction")


def require_new_target(root: Path, target_ref: str) -> None:
    if not target_ref.startswith("refs/heads/") or target_ref.casefold() == "refs/heads/main":
        raise ValueError("target must be a full branch ref other than main")
    if git(root, "check-ref-format", target_ref, allowed_codes=(0, 1)).returncode:
        raise ValueError("invalid target ref")
    if git(root, "symbolic-ref", "-q", target_ref, allowed_codes=(0, 1)).returncode == 0:
        raise ValueError("target already exists as a symbolic ref")
    if git(root, "show-ref", "--verify", "--quiet", target_ref, allowed_codes=(0, 1)).returncode == 0:
        raise ValueError("target already exists")


def rebuild_history(root: Path, source_ref: str, target_ref: str, schedule_path: Path) -> str:
    require_clean_worktree(root)
    require_new_target(root, target_ref)
    schedule = load_schedule(schedule_path)
    source_tip = resolve_commit(root, source_ref)
    source = read_history(root, source_tip)
    validate_history(root, source, schedule, check_dates=False)
    parent: str | None = None
    for commit in source:
        timestamp = schedule[commit.subject]
        env = dict(os.environ, GIT_AUTHOR_NAME=commit.author_name, GIT_AUTHOR_EMAIL=commit.author_email,
                   GIT_COMMITTER_NAME=commit.committer_name, GIT_COMMITTER_EMAIL=commit.committer_email,
                   GIT_AUTHOR_DATE=timestamp, GIT_COMMITTER_DATE=timestamp)
        args = ["-c", "commit.gpgsign=false", "-c", f"i18n.commitEncoding={commit.encoding}", "commit-tree", commit.tree]
        if parent is not None:
            args.extend(["-p", parent])
        parent = git(root, *args, input=commit.message, env=env).stdout.decode("ascii").strip()
    assert parent is not None
    rebuilt = read_history(root, parent)
    compare_histories(source, rebuilt)
    validate_history(root, rebuilt, schedule)
    require_clean_worktree(root)
    require_new_target(root, target_ref)
    if resolve_commit(root, source_ref) != source_tip:
        raise ValueError("source ref changed during reconstruction")
    # The all-zero old OID makes this create-only, including a competing writer
    # after the preflight checks. --no-deref also protects symbolic-ref targets.
    git(root, "update-ref", "--no-deref", target_ref, parent, "0" * len(parent))
    return parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Git worktree (must be clean)")
    parser.add_argument("--source-ref", required=True, help="Original backed-up history")
    parser.add_argument("--target-ref", required=True, help="New full refs/heads/... ref; main is forbidden")
    parser.add_argument("--schedule", type=Path, default=Path("docs/reconstruction-schedule.json"))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    schedule = args.schedule if args.schedule.is_absolute() else root / args.schedule
    try:
        oid = rebuild_history(root, args.source_ref, args.target_ref, schedule)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"History reconstruction FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"History reconstruction PASS: created {args.target_ref} at {oid}; source and worktree preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
