"""Supervise frozen execution and publish checkpoints; never change scientific inputs.

Default/check mode is read-only. --execute makes the already authorized paid calls
and ordinary commits/pushes on the existing feature branch. It never merges main.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from app.evaluation.frozen import FREEZE_PATH, exclusive_run, validate_freeze, write_json
from app.evaluation.test_judging import load_repeat
from publish_checkpoint import checkpoint, receipts, OUT, RUN

BRANCH = "codex/real-evaluation-foundation"
REPOSITORY = "jadeyoungchina/multi-agent-research-assistant"
CONTROL = ROOT / "artifacts/evaluation/frozen-run-control"
PYTHON = Path(sys.executable)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def check():
    record = validate_freeze(FREEZE_PATH)
    if git("branch", "--show-current") != BRANCH:
        raise ValueError("supervisor requires the existing feature branch")
    if git("remote", "get-url", "origin") != f"https://github.com/{REPOSITORY}.git":
        raise ValueError("origin differs from the authorized repository")
    for path in Path(__file__).parent.glob("*.py"):
        compile(path.read_text("utf-8"), str(path), "exec")
    return record


def state(stage, **details):
    write_json(CONTROL / "status.json", {"stage": stage, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
               "freeze_id": json.loads(FREEZE_PATH.read_text("utf-8"))["freeze_id"], **details})


def command(args, log, *, accepted=(0,)):
    with (CONTROL / log).open("a", encoding="utf-8", newline="\n") as stream:
        result = subprocess.run([str(x) for x in args], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode not in accepted:
        raise RuntimeError(f"command failed; see {log}")
    return result.returncode


def wait_existing_generation():
    # A currently attached first-repeat process owns this OS lock. Never kill it
    # or create a second paid process. A crash releases the lock; the frozen
    # runner then stops on an uncertain unit instead of silently retrying it.
    while True:
        try:
            with exclusive_run(RUN / ".generation.lock"):
                return
        except ValueError as error:
            if str(error) != "experiment is already running":
                raise
            state("waiting_for_existing_generation", wait_policy="no interruption or duplicate paid execution")
            time.sleep(10)


def wait_ci(sha):
    deadline = time.monotonic() + 600
    url = f"https://api.github.com/repos/{REPOSITORY}/actions/runs?per_page=30"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "frozen-experiment-checkpoint"}), timeout=20) as response:
                runs = json.load(response)["workflow_runs"]
        except (OSError, ValueError):
            time.sleep(15)
            continue
        current = [r for r in runs if r["head_sha"] == sha and r["event"] in {"push", "pull_request"} and r["name"] == "ci"]
        latest = {event: max((r for r in current if r["event"] == event), key=lambda r: r["id"], default=None)
                  for event in ("push", "pull_request")}
        if any(r and r["status"] == "completed" and r["conclusion"] != "success" for r in latest.values()):
            raise RuntimeError("checkpoint CI failed; preserve logs and stop further paid work")
        if all(r and r["status"] == "completed" and r["conclusion"] == "success" for r in latest.values()):
            return [{"event": event, "run_id": r["id"], "url": r["html_url"], "conclusion": r["conclusion"]}
                    for event, r in latest.items()]
        time.sleep(15)
    raise TimeoutError("checkpoint CI not confirmed; preserve results and stop")


def publish_commit(message):
    # Stage only this publication directory. Other user changes are never swept
    # into an automatic commit. The clean-tree guard runs before packet writes.
    command(["git", "add", "--", "docs/evaluation/heldout-test"], "publication.log")
    if git("diff", "--cached", "--name-only"):
        staged = git("diff", "--cached", "--name-only").splitlines()
        if any(not name.startswith("docs/evaluation/heldout-test/") for name in staged):
            raise ValueError("unrelated staged files; automatic commit refused")
        command(["git", "commit", "-m", message], "publication.log")
    sha = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        raise ValueError("working tree changed; stop before push or paid work")
    command(["git", "push", "origin", BRANCH], "publication.log")
    remote = git("ls-remote", "origin", f"refs/heads/{BRANCH}").split()[0]
    if remote != sha:
        raise ValueError("remote does not contain the checkpoint head")
    state("checking_ci", checkpoint_commit=sha)
    verified = wait_ci(sha)
    write_json(CONTROL / f"ci-{sha[:7]}.json", verified)
    return sha


def publish_initial_analysis(record):
    validate_freeze(FREEZE_PATH)
    analysis = json.loads((RUN / "primary-analysis.json").read_text("utf-8"))
    if analysis["freeze_id"] != record["freeze_id"] or analysis["executed_units"] != 480:
        raise ValueError("analysis is incomplete or belongs to another freeze")
    # This preserves original raw judgments inside unit_scores. It is deliberately
    # a preliminary packet until the prespecified source-review sample is checked.
    write_json(OUT / "initial-analysis.json", analysis)
    rows = [row for repeat in range(3) for row in receipts(RUN / "judgments" / f"repeat-{repeat}.sqlite", "blinded_judging", repeat)]
    (OUT / "judge-request-usage.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
    primary = analysis["primary"]
    source = analysis["review_sample_completion"]
    write_json(OUT / "initial-analysis-summary.json", {k: v for k, v in analysis.items() if k != "unit_scores"})
    text = ["# Preliminary held-out analysis: source review pending", "",
            "This packet contains the original blinded AI judgments and the frozen paired analysis. It is **preliminary**: the prespecified 80-answer source review is not yet complete. No architecture winner or independent human validation is claimed.", "",
            f"Generation: {analysis['executed_units']}/480; execution failures: {analysis['execution_failures']}; successful answers missing valid judgments: {len(analysis['successful_answers_missing_judgment'])}.", "",
            f"Primary multi-agent minus single-agent paired correctness: {primary['mean_paired_correctness_difference']}; complete question pairs: {primary['complete_pairs']}/40; family-bootstrap 95% interval: {primary['family_bootstrap_95_interval']}.", "",
            f"All-question missing-judgment sensitivity bounds: {primary['all_question_missing_value_bounds']}. Execution failures are zero; judge errors remain missing. Three repeats are averaged within each question before pairing; trace rows are not treated as independent questions.", "",
            f"Prespecified AI source review: {source['completed']}/{source['required_answers']}. Human scoring: **0**. Eight document families give an exploratory interval. AI references, judge alias drift and possible public-source pretraining exposure remain limitations.", "",
            "[Full original judgments and analysis](initial-analysis.json) · [Compact summary and phase costs](initial-analysis-summary.json) · [Safe judge receipts](judge-request-usage.jsonl) · [Generation packets](README.md).", "",
            "Judge account prices remain unknown; null monetary amounts are not zero. Index, answer and judge requests remain separately attributed. The next review must preserve these original judgments and record each accepted or corrected decision against its exact trace hash.", ""]
    (OUT / "INITIAL-ANALYSIS.md").write_text("\n".join(text), encoding="utf-8", newline="\n")
    path = OUT / "README.md"
    readme = path.read_text("utf-8")
    readme += "\n[Preliminary blinded analysis](INITIAL-ANALYSIS.md) is available; the 80-answer source review remains pending.\n"
    path.write_text(readme, encoding="utf-8", newline="\n")


def execute():
    CONTROL.mkdir(parents=True, exist_ok=True)
    with exclusive_run(CONTROL / ".supervisor.lock"):
        record = check()
        state("starting", supervisor_sha256=sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest())
        wait_existing_generation()
        for repeat in range(3):
            check()
            state("generation", repeat_index=repeat)
            command([PYTHON, "-u", "scripts/frozen_experiment.py", "generate", "--execute", "--repeat", str(repeat)], f"generation-repeat-{repeat}.log")
            load_repeat(record, RUN, repeat)
            state("publishing_generation_checkpoint", repeat_index=repeat)
            checkpoint(repeat)
            publish_commit(f"docs: publish frozen held-out generation repeat {repeat + 1} of 3")
        check()
        state("blinded_judging")
        judge_exit = command([PYTHON, "-u", "scripts/frozen_experiment.py", "judge", "--execute"], "judging.log", accepted=(0, 3))
        state("paired_initial_analysis", judge_errors_retained=judge_exit == 3)
        command([PYTHON, "scripts/frozen_experiment.py", "analyze"], "analysis.log")
        check()
        publish_initial_analysis(record)
        sha = publish_commit("docs: retain initial held-out judgments and paired analysis pending source review")
        state("generation_and_initial_judging_complete_source_review_pending", checkpoint_commit=sha,
              next_step="prespecified 80-answer AI source review and separate additional error reviews; originals remain unchanged")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        path = CONTROL / "status.json"
        print(path.read_text("utf-8") if path.exists() else "Supervisor not started.")
        return 0
    try:
        if not args.execute:
            record = check()
            print(f"Supervisor preflight passed: {record['freeze_id']}; zero paid calls; no Git mutations.")
        else:
            execute()
        return 0
    except Exception as error:
        if args.execute:
            state("stopped", error_type=type(error).__name__, detail="Inspect local phase logs; preserve all completed and uncertain units; no automatic paid retries.")
        print(f"Supervisor stopped ({type(error).__name__}); no frozen-input override.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
