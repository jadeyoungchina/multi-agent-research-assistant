"""Verify the disclosed reconstruction timeline against a Git commit history."""

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANCHORS = frozenset({
    "2026-07-06T20:18:00+08:00", "2026-07-13T21:07:00+08:00",
    "2026-07-22T22:16:00+08:00", "2026-07-30T20:43:00+08:00",
    "2026-08-08T21:26:00+08:00", "2026-08-17T22:09:00+08:00",
    "2026-08-25T20:51:00+08:00", "2026-09-01T21:34:00+08:00",
    "2026-09-05T22:12:00+08:00", "2026-09-08T21:40:00+08:00",
})
SCHEDULE_DISCLOSURE = "Transparent reconstruction after accidental deletion"
DISCLOSURE = (
    "This repository was reconstructed on 2026-09-08 after the original local project was accidentally deleted. "
    "Commit dates restore the documented development milestones; they are not the original Git objects "
    "or an unreconstructed historical record."
)
UPSTREAM_CUTOFFS = {
    "app/retrieval/loaders.py": "2026-07-04T03:36:00+08:00",
    "app/evaluation/trace.py": "2026-08-28T19:39:40+08:00",
    "app/evaluation/metrics.py": "2026-08-28T19:39:40+08:00",
}


@dataclass(frozen=True)
class Commit:
    oid: str
    tree: str
    subject: str
    message: bytes
    author_name: str
    author_email: str
    committer_name: str
    committer_email: str
    author_date: str
    committer_date: str
    encoding: str
    changed_paths: frozenset[str]


def git(root: Path, *args: str, input: bytes | None = None, env: dict | None = None,
        allowed_codes: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess[bytes]:
    """Run Git without a shell, replacement objects, or ambient repository overrides."""
    git_env = dict(os.environ if env is None else env)
    # -C must always select the requested repository, even inside a Git hook.
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE",
                "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
                "GIT_SHALLOW_FILE", "GIT_REPLACE_REF_BASE"):
        git_env.pop(key, None)
    git_env["GIT_OPTIONAL_LOCKS"] = "0"
    result = subprocess.run(
        ["git", "--no-replace-objects", "-C", str(root), *args],
        input=input, capture_output=True, env=git_env, check=False,
    )
    if result.returncode not in allowed_codes:
        # Do not print raw object/message content returned by Git.
        raise ValueError(f"Git {args[0]} failed (exit {result.returncode})")
    return result


def resolve_commit(root: Path, ref: str) -> str:
    return git(root, "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}").stdout.decode("ascii").strip()


def _identity(headers: list[bytes], kind: bytes) -> tuple[str, str]:
    identities = [line[len(kind) + 1:] for line in headers if line.startswith(kind + b" ")]
    if len(identities) != 1:
        raise ValueError("commit must have exactly one author and committer")
    match = re.fullmatch(rb"(.+) <([^<>]+)> -?\d+ [+-]\d{4}", identities[0])
    if match is None:
        raise ValueError("invalid commit identity")
    name, email = (part.decode("utf-8") for part in match.groups())
    if any(ord(char) < 32 for char in name + email):
        raise ValueError("invalid control character in commit identity")
    return name, email


def read_history(root: Path, ref: str) -> list[Commit]:
    tip = resolve_commit(root, ref)
    oids = git(root, "rev-list", "--reverse", "--topo-order", tip).stdout.decode("ascii").splitlines()
    commits: list[Commit] = []
    subjects: set[str] = set()
    for oid in oids:
        raw = git(root, "cat-file", "commit", oid).stdout
        header, separator, message = raw.partition(b"\n\n")
        if not separator:
            raise ValueError("malformed commit object")
        headers = header.split(b"\n")
        parents = [line[7:].decode("ascii") for line in headers if line.startswith(b"parent ")]
        if len(parents) > 1:
            raise ValueError("merge commits cannot be reconstructed as linear history")
        if parents != ([commits[-1].oid] if commits else []):
            raise ValueError("history is not a complete linear chain (possibly shallow)")
        trees = [line[5:].decode("ascii") for line in headers if line.startswith(b"tree ")]
        if len(trees) != 1:
            raise ValueError("commit must have exactly one tree")
        encoding_headers = [line[9:].decode("ascii") for line in headers if line.startswith(b"encoding ")]
        if len(encoding_headers) > 1:
            raise ValueError("commit has duplicate encoding headers")
        subject, author_date, committer_date = git(
            root, "-c", "i18n.logOutputEncoding=UTF-8", "show", "-s", "--format=%s%x00%aI%x00%cI", oid,
        ).stdout.decode("utf-8").rstrip("\n").split("\0")
        if subject in subjects:
            raise ValueError(f"duplicate subject in history: {subject}")
        subjects.add(subject)
        changed_paths = frozenset(name.decode("utf-8") for name in git(
            root, "diff-tree", "--root", "--no-commit-id", "--name-only", "--no-renames", "-r", "-z", oid,
        ).stdout.split(b"\0") if name)
        commits.append(Commit(
            oid, trees[0], subject, message, *_identity(headers, b"author"),
            *_identity(headers, b"committer"), author_date, committer_date,
            encoding_headers[0] if encoding_headers else "UTF-8", changed_paths,
        ))
    if not commits:
        raise ValueError("history contains no commits")
    return commits


def validate_schedule(data: object) -> dict[str, str]:
    """Return an ordered subject/date mapping, rejecting ambiguous input."""
    if not isinstance(data, dict) or set(data) != {"timezone", "disclosure", "commits"}:
        raise ValueError("schedule fields must be timezone, disclosure, and commits")
    if data["timezone"] != "Asia/Shanghai":
        raise ValueError("schedule timezone must be Asia/Shanghai")
    if data["disclosure"] != SCHEDULE_DISCLOSURE:
        raise ValueError("schedule disclosure is missing or changed")
    entries = data["commits"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("schedule commits must be a nonempty list")
    schedule: dict[str, str] = {}
    timestamps: set[str] = set()
    previous: datetime | None = None
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("schedule entry must be an object")
        if set(entry) != {"subject", "timestamp"}:
            raise ValueError("schedule entry fields must be subject and timestamp")
        subject, timestamp = entry["subject"], entry["timestamp"]
        if (not isinstance(subject, str) or not subject.strip()
                or subject != subject.strip() or any(ord(char) < 32 for char in subject)):
            raise ValueError("schedule subject must be a nonempty full single-line subject")
        if subject in schedule:
            raise ValueError(f"duplicate subject: {subject}")
        if not isinstance(timestamp, str) or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00", timestamp
        ):
            raise ValueError("timestamp must use YYYY-MM-DDTHH:MM:SS+08:00")
        try:
            date = datetime.fromisoformat(timestamp)
        except ValueError as exc:
            raise ValueError("invalid calendar timestamp") from exc
        if date.hour < 20:
            raise ValueError("timestamp local time must be at least 20:00")
        if timestamp in timestamps:
            raise ValueError(f"duplicate timestamp: {timestamp}")
        if previous is not None and date < previous:
            raise ValueError("schedule timestamps must follow commit order")
        schedule[subject] = timestamp
        timestamps.add(timestamp)
        previous = date
    if not ANCHORS <= timestamps:
        raise ValueError("schedule is missing approved anchor timestamps")
    return schedule


def _unique_json_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_schedule(path: Path) -> dict[str, str]:
    return validate_schedule(json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_json_keys))


def validate_history(root: Path, commits: list[Commit], schedule: dict[str, str], *,
                     check_dates: bool = True) -> None:
    subjects = [commit.subject for commit in commits]
    unmapped = set(subjects) - schedule.keys()
    if unmapped:
        raise ValueError("unmapped commit subjects: " + ", ".join(sorted(unmapped)))
    unused = schedule.keys() - set(subjects)
    if unused:
        raise ValueError("unused schedule subjects: " + ", ".join(sorted(unused)))
    if subjects != list(schedule):
        raise ValueError("schedule subject order differs from commit order")
    for name in ("README.md", "CHANGELOG.md"):
        try:
            contents = git(root, "show", f"{commits[-1].oid}:{name}").stdout.decode("utf-8")
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f"{name}: committed reconstruction disclosure missing") from exc
        if DISCLOSURE not in contents:
            raise ValueError(f"{name}: committed reconstruction disclosure missing or changed")
    for commit in commits:
        timestamp = schedule[commit.subject]
        for path, cutoff in UPSTREAM_CUTOFFS.items():
            if path in commit.changed_paths and datetime.fromisoformat(timestamp) < datetime.fromisoformat(cutoff):
                raise ValueError(f"upstream availability violation: {path} in {commit.subject}")
        if check_dates and (commit.author_date != timestamp or commit.committer_date != timestamp):
            raise ValueError(f"author/committer dates do not match schedule: {commit.subject}")


def compare_histories(source: list[Commit], rebuilt: list[Commit]) -> None:
    if len(source) != len(rebuilt):
        raise ValueError("source and reconstructed commit counts differ")
    if source[-1].tree != rebuilt[-1].tree:
        raise ValueError("final tree differs from source")
    for old, new in zip(source, rebuilt, strict=True):
        for field in ("tree", "subject", "message", "author_name", "author_email", "committer_name", "committer_email", "encoding"):
            if getattr(old, field) != getattr(new, field):
                raise ValueError(f"reconstructed {field} differs from source: {old.subject}")


def verify_history(root: Path, ref: str, schedule_path: Path, source_ref: str | None = None) -> int:
    schedule = load_schedule(schedule_path)
    commits = read_history(root, ref)
    validate_history(root, commits, schedule)
    if source_ref is not None:
        compare_histories(read_history(root, source_ref), commits)
    return len(commits)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Git worktree to inspect")
    parser.add_argument("--ref", default="HEAD", help="Reconstructed ref or commit (default: HEAD)")
    parser.add_argument("--source-ref", help="Optional original ref for tree/message/identity comparison")
    parser.add_argument("--schedule", type=Path, default=Path("docs/reconstruction-schedule.json"))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    schedule = args.schedule if args.schedule.is_absolute() else root / args.schedule
    try:
        count = verify_history(root, args.ref, schedule, args.source_ref)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"History verification FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"History verification PASS: {count} commits; schedule, disclosures, and upstream cutoffs verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
