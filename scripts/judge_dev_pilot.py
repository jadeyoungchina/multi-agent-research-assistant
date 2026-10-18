"""Explicit paid, blinded DeepSeek CLI scoring of a saved dev pilot.

Never generates application answers, runs held-out questions, or labels AI scores
as human judgments. Raw responses and validation errors are retained per unit.
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.budget import BudgetLedger
from app.evaluation.deepseek_cli import run_task
from app.evaluation.real_dataset import load_real_suite
from app.evaluation.runner import EvaluationReport
from app.evaluation.semantic import SemanticJudgment, judge_prompt, score_semantic_judgment


def decode_judgment(text):
    """Allow only a whole JSON response or a single whole fenced JSON block."""
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[len("```json\n"):-len("\n```")]
    elif text.startswith("```\n") and text.endswith("\n```"):
        text = text[len("```\n"):-len("\n```")]
    return SemanticJudgment.model_validate_json(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--suite", type=Path, default=ROOT / "benchmarks/real-v1")
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="explicitly allow paid DeepSeek CLI requests")
    parser.add_argument("--max-output-tokens", type=int, default=6144)
    args = parser.parse_args(argv)
    try:
        report_path = args.run_dir / "results.json"
        report = EvaluationReport.model_validate_json(report_path.read_text("utf-8"))
        suite = load_real_suite(args.suite / "corpus-manifest.json", args.suite / "dev.jsonl", "dev")
        by_id = {case.id: case for case in suite.cases}
        if report.metadata.synthetic or not report.execution_complete:
            raise ValueError("only completed real development runs may be judged")
        if report.metadata.dataset_sha256 != sha256((args.suite / "dev.jsonl").read_bytes()).hexdigest():
            raise ValueError("reference dataset differs from generation dataset")
        if any(trace.case_id not in by_id for trace in report.traces):
            raise ValueError("held-out or unknown cases cannot enter pilot scoring")
        if len({(trace.case_id, trace.variant) for trace in report.traces}) != len(report.traces):
            raise ValueError("duplicate trace unit")
        if not 1 <= args.max_output_tokens <= 8192:
            raise ValueError("invalid judge output cap")
        if not args.execute:
            print(f"Validated {len(report.traces)} development traces; no requests sent. Use --execute to score.")
            return 0
        ledger = BudgetLedger(args.ledger)
        ledger.set_enforcement(False)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for index, trace in enumerate(report.traces):
            unit = f"answer-{index + 1:03d}"
            case = by_id[trace.case_id]
            entry = {"unit": unit, "case_id": trace.case_id, "variant": trace.variant.value,
                     "status": "not_scored_execution_failed", "judge_kind": "ai", "human_reviewed": False}
            if trace.status == "success":
                prompt = judge_prompt(case, trace) + (
                    "\nKeep reasons concise (at most 12 words each). Extract every factual claim, "
                    "including code examples and uncited claims. Preserve Markdown in verbatim spans. "
                    "Use citation IDs exactly as supplied; only attach a citation when its exact "
                    "[[cite:ID]] token immediately follows the claim in the answer (within 160 characters). "
                    "Never borrow citations from a different sentence. Do not include the citation token "
                    "inside claim_text. For a bare code example include surrounding newlines exactly."
                )
                identity = json.dumps({"prompt": prompt, "cap": args.max_output_tokens,
                                       "script": sha256(Path(__file__).read_bytes()).hexdigest()}, sort_keys=True)
                task_id = "dev-pilot-judge-" + sha256(identity.encode()).hexdigest()[:24]
                entry["task_id"] = task_id
                try:
                    result = run_task(prompt, task_id=task_id, ledger=ledger, mode="account",
                                      max_output=args.max_output_tokens, timeout=180, max_prompt_bytes=131072)
                    entry["raw_response"] = result
                    judgment = decode_judgment(result["text"])
                    score = score_semantic_judgment(case, trace, judgment)
                    entry.update(status="valid_ai_judgment", judgment=judgment.model_dump(), score=score.model_dump())
                except Exception as error:
                    # Values are allowlisted. Raw exception/CLI diagnostics can include credentials.
                    entry.update(status="judge_error", error_type=type(error).__name__)
            (args.output_dir / f"{unit}.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n", "utf-8")
            entries.append(entry)
            print(f"{unit}: {entry['status']}", flush=True)
        summary = {"created_at": datetime.now(timezone.utc).isoformat(), "judge_kind": "ai",
                   "human_reviewed": False, "report_sha256": sha256(report_path.read_bytes()).hexdigest(),
                   "valid_count": sum(item["status"] == "valid_ai_judgment" for item in entries),
                   "judge_error_count": sum(item["status"] == "judge_error" for item in entries),
                   "execution_failure_count": sum(item["status"] == "not_scored_execution_failed" for item in entries),
                   "units": [{key: value for key, value in entry.items() if key != "raw_response"} for entry in entries]}
        (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", "utf-8")
        print(f"Valid={summary['valid_count']}; judge errors={summary['judge_error_count']}; execution failures={summary['execution_failure_count']}.")
        return 0 if summary["judge_error_count"] == 0 else 3
    except Exception as error:
        print(f"Pilot scoring stopped ({type(error).__name__}).", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
