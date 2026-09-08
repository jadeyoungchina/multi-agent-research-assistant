"""Offline release checks over Git-tracked working-tree files; never prints secrets."""

import argparse
from hashlib import sha256
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.dataset import load_benchmark_cases, validate_benchmark_corpus


LICENSE_SHA256 = "c9877e4d8788a0bb97502348dca8fbd78a6eaa73db0638221b3cb67422d30177"
UPSTREAM_COMMIT = "4c95ae14cc2462c442b5c064cccd74430d02bc46"
KEY_NAMES = ("DASHSCOPE_API_KEY", "OPENAI_API_KEY")
# Both providers use sk-prefixed credentials; OpenAI also has project/service keys.
KEY_PATTERN = re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b")
ASSIGNMENT = re.compile(r'''^\s*(?:export\s+)?["']?(DASHSCOPE_API_KEY|OPENAI_API_KEY)["']?\s*[:=]\s*(.*?)\s*$''')
PLACEHOLDERS = {"", "replace-me", "your-api-key", "your_api_key", "<your-api-key>", "placeholder"}


def tracked_files(root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True, capture_output=True,
    )
    return {name for name in result.stdout.decode("utf-8").split("\0") if name}


def forbidden_path(name: str) -> bool:
    path = PurePosixPath(name.casefold())
    parts = path.parts
    basename = path.name
    if basename != ".env.example" and (basename == ".env" or basename.startswith(".env.")):
        return True
    if any(part in {"data", "artifacts", "uploads", "uploaded_documents"} for part in parts[:-1]):
        return True
    if "evaluation/results" in "/".join(parts) or basename in {"results.json", "results.csv"}:
        return True
    if re.search(r"\.(?:db(?:-.*)?|sqlite[^.]*|key|pem)$", basename):
        return True
    fixture = name.startswith(("benchmarks/corpus/", "examples/demo-corpus/", "tests/fixtures/"))
    return not fixture and path.suffix in {".pdf", ".doc", ".docx", ".odt", ".rtf"}


def scan_text(name: str, text: str) -> list[str]:
    problems: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if KEY_PATTERN.search(line):
            problems.append(f"{name}:{number}: possible provider credential")
        match = ASSIGNMENT.match(line)
        if match and name != ".env.example":
            value = match[2].strip().strip("\"'").rstrip(",").strip("\"'")
            if value.casefold() not in PLACEHOLDERS and not re.fullmatch(r"\$\{[A-Z_]+\}", value):
                problems.append(f"{name}:{number}: non-placeholder provider key assignment")
    return problems


def verify_repository(root: Path) -> tuple[list[str], int]:
    tracked = tracked_files(root)
    problems: list[str] = []
    texts: dict[str, str] = {}
    for name in sorted(tracked):
        if forbidden_path(name):
            problems.append(f"{name}: tracked runtime data, upload, or credential file")
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            problems.append(f"{name}: tracked symbolic link or path outside repository")
            continue
        try:
            content = path.read_bytes()
        except OSError:
            problems.append(f"{name}: tracked file is missing or unreadable")
            continue
        # UTF-16 text can contain credentials too; binary documents are path-checked above.
        try:
            encoding = "utf-16" if content.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
            text = content.decode(encoding)
        except UnicodeDecodeError:
            continue
        texts[name] = text
        problems.extend(scan_text(name, text))

    example = texts.get(".env.example", "")
    for key in KEY_NAMES:
        assignments = [
            line for line in example.splitlines()
            if (match := ASSIGNMENT.match(line)) and match[1] == key
        ]
        if assignments != [key + "="]:
            problems.append(f".env.example: require exactly one blank {key} assignment")

    for name in ("LICENSE", "THIRD_PARTY_LICENSES/GenAI_Agents-LICENSE.txt"):
        # Read exact UTF-8 separately: only newline normalization is licensed here.
        try:
            normalized = (root / name).read_text(encoding="utf-8").replace("\r\n", "\n")
            valid = name in tracked and sha256(normalized.encode("utf-8")).hexdigest() == LICENSE_SHA256
        except (OSError, UnicodeError):
            valid = False
        if not valid:
            problems.append(f"{name}: missing, untracked, or upstream license SHA-256 mismatch")

    try:
        cases = load_benchmark_cases(root / "benchmarks/cases.jsonl")
        if len(cases) != 30:
            raise ValueError("exactly 30 benchmark cases required")
        required_sources = {"benchmarks/cases.jsonl"} | {
            "benchmarks/corpus/" + filename for case in cases for filename in case.source_files
        }
        if not required_sources <= tracked:
            raise ValueError("benchmark cases and corpus must be tracked")
        validate_benchmark_corpus(cases, root / "benchmarks/corpus")
    except (OSError, UnicodeError, ValueError):
        problems.append("benchmarks: require 30 valid sequential cases and matching tracked corpus evidence phrases")

    notice = texts.get("THIRD_PARTY_NOTICES.md", "")
    if UPSTREAM_COMMIT not in notice:
        problems.append("THIRD_PARTY_NOTICES.md: missing pinned upstream commit")
    adapted: set[str] = set()
    for line in notice.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5 or cells[-1].strip("`").casefold() != "adapted":
            continue
        targets = re.findall(r"`([^`]+)`", cells[3])
        if not targets:
            problems.append("THIRD_PARTY_NOTICES.md: adapted row requires explicit backticked local file paths")
        adapted.update(targets)
    if not adapted:
        problems.append("THIRD_PARTY_NOTICES.md: no explicit adapted local files")
    for name in sorted(adapted):
        if name not in tracked or name not in texts:
            problems.append(f"{name}: adapted target is missing, untracked, or unreadable")
        elif UPSTREAM_COMMIT not in "\n".join(texts[name].splitlines()[:20]):
            problems.append(f"{name}: adapted file header lacks pinned upstream commit")
    return problems, len(tracked)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    try:
        problems, count = verify_repository(args.root.resolve())
    except (OSError, subprocess.SubprocessError, UnicodeError):
        print("Repository verification FAIL: cannot inspect Git-tracked files.", file=sys.stderr)
        return 1
    if problems:
        print("Repository verification FAIL:\n" + "\n".join(problems))
        return 1
    print(f"Repository verification PASS: {count} tracked files; licenses, attribution, and 30 benchmark cases verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
