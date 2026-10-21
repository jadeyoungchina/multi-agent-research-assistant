"""Validate the test judge schema on 20 already source-reviewed dev answers."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.budget import BudgetLedger
from app.evaluation.deepseek_cli import run_task
from app.evaluation.development import block_judging_prompt, load_scoring_cases, score_blocks
from app.evaluation.frozen import calibration_identity, digest_json, write_json
from app.evaluation.test_judging import TestJudgment, decode
from app.evaluation.trace import EvaluationTrace


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/evaluation/frozen-judge-calibration")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    packet = ROOT / "docs/evaluation/final-dev"
    cases = load_scoring_cases(ROOT / "benchmarks/real-v1")
    traces = [EvaluationTrace.model_validate_json(x) for x in (packet / "traces.jsonl").read_text("utf-8").splitlines()]
    approved = {x["unit"]: x for x in json.loads((packet / "ai-judgments.json").read_text("utf-8"))}
    selected = {"REAL-DEV-001", "REAL-DEV-007", "REAL-DEV-013", "REAL-DEV-019", "REAL-DEV-020"}
    jobs = [(f"answer-{i+1:03d}", t) for i, t in enumerate(traces) if t.case_id in selected]
    if len(jobs) != 20 or any("codex_adjudication" not in approved[u] for u, t in jobs):
        raise ValueError("calibration requires the prespecified 20 source-reviewed dev answers")
    identity = calibration_identity()
    if not args.execute:
        print("Validated 20 development-only calibration answers; no test answer or paid call.")
        return 0
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("commit implementation before calibration")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text("utf-8")) != identity:
        raise ValueError("calibration identity changed")
    write_json(manifest, identity)
    ledger = BudgetLedger(args.output_dir / "ledger.sqlite")
    ledger.set_enforcement(False)

    def judge(job):
        unit, trace = job
        path = args.output_dir / f"{unit}.json"
        if path.exists():
            saved = json.loads(path.read_text("utf-8"))
            if saved["identity"] != identity:
                raise ValueError("saved calibration differs")
            return saved
        prompt = block_judging_prompt(cases[trace.case_id], trace, ROOT / "benchmarks/real-v1", TestJudgment)
        base = prompt
        entry = {"unit": unit, "identity": identity, "case_id": trace.case_id, "variant": trace.variant.value,
                 "trace_sha256": sha256(trace.model_dump_json().encode()).hexdigest(),
                 "status": "judge_error", "attempts": [], "source_reviewed_judgment": approved[unit]["judgment"]}
        for repair in range(2):
            try:
                result = run_task(prompt, task_id="test-schema-calib-" + digest_json({"identity": identity, "unit": unit, "prompt": prompt})[:40],
                                  ledger=ledger, max_output=8192, timeout=180, max_prompt_bytes=131072)
                entry["attempts"].append({"raw_response": result, "format_repair": repair})
                judgment = decode(result["text"])
                score_blocks(cases[trace.case_id], trace, judgment)
                entry.update(status="valid_ai_judgment", judgment=judgment.model_dump())
                break
            except Exception as error:
                entry["error_type"] = type(error).__name__
                if repair or not entry["attempts"]:
                    break
                prompt = (base + "\nPrevious output failed strict validation. Return every point and block once. "
                          "Uncited factual blocks are uncited; nonfactual blocks N/A. Preserve factual judgments.\nPrevious:\n"
                          + entry["attempts"][-1]["raw_response"]["text"])
        write_json(path, entry)
        print(f"{unit}: {entry['status']}", flush=True)
        return entry

    with ThreadPoolExecutor(max_workers=3) as pool:
        entries = list(pool.map(judge, jobs))
    point_match = point_count = block_match = block_count = 0
    for entry in entries:
        if entry["status"] != "valid_ai_judgment":
            continue
        expected, actual = entry["source_reviewed_judgment"], entry["judgment"]
        a = {x["index"]: x for x in actual["key_points"]}
        for x in expected["key_points"]:
            point_count += 1
            point_match += a[x["index"]]["score"] == x["score"]
        a = {x["index"]: x for x in actual["blocks"]}
        for x in expected["blocks"]:
            block_count += 1
            block_match += (a[x["index"]]["requires_citation"], a[x["index"]]["support"]) == (x["requires_citation"], x["support"])
    summary = {"identity": identity, "planned_answers": 20,
        "valid_ai_judgments": sum(e["status"] == "valid_ai_judgment" for e in entries),
        "judge_errors": sum(e["status"] == "judge_error" for e in entries),
        "key_point_agreement": {"matched": point_match, "valid_denominator": point_count,
            "planned_denominator": sum(len(approved[u]["judgment"]["key_points"]) for u, t in jobs)},
        "block_verdict_agreement": {"matched": block_match, "valid_denominator": block_count,
            "planned_denominator": sum(len(approved[u]["judgment"]["blocks"]) for u, t in jobs)},
        "comparison_reference": "Codex source-reviewed AI labels, not human truth",
        "human_reviewed": False, "no_heldout_answers": True, "entries": entries}
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "usage.json", ledger.summary())
    print(f"Calibration valid={summary['valid_ai_judgments']}/20; errors={summary['judge_errors']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
