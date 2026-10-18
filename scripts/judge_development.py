"""Blinded, resumable DeepSeek CLI auditing of a completed real dev run."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.budget import BudgetLedger
from app.evaluation.deepseek_cli import run_task
from app.evaluation.development import DevelopmentJudgment, development_prompt, load_scoring_cases, score_development
from app.evaluation.runner import EvaluationReport


def decode(text):
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4]
    elif text.startswith("```\n") and text.endswith("\n```"):
        text = text[4:-4]
    return DevelopmentJudgment.model_validate_json(text)


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--suite", type=Path, default=ROOT / "benchmarks/real-v1")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        path = args.run_dir / "results.json"
        report = EvaluationReport.model_validate_json(path.read_text("utf-8"))
        cases = load_scoring_cases(args.suite)
        if report.metadata.synthetic or not report.execution_complete or any(t.case_id not in cases for t in report.traces):
            raise ValueError("only completed real development runs may be scored")
        if report.metadata.dataset_sha256 != sha256((args.suite / "dev.jsonl").read_bytes()).hexdigest():
            raise ValueError("generation dataset has changed")
        if len({(t.case_id, t.variant) for t in report.traces}) != len(report.traces):
            raise ValueError("duplicate generation unit")
        identity = {"report_sha256": sha256(path.read_bytes()).hexdigest(),
                    "policy_sha256": sha256((args.suite / "dev-scoring-v2.json").read_bytes()).hexdigest(),
                    "judge_source_sha256": sha256((ROOT / "app/evaluation/development.py").read_bytes()).hexdigest(),
                    "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                    "max_output_tokens": 8192, "max_format_repairs": 1}
        if not args.execute:
            print(f"Validated {len(report.traces)} development units; no paid calls.")
            return 0
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = args.output_dir / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text("utf-8")) != identity:
            raise ValueError("judge run differs; use a new directory")
        write(manifest, identity)
        ledger = BudgetLedger(args.ledger)
        ledger.set_enforcement(False)

        def judge(index_trace):
            index, trace = index_trace
            unit = f"answer-{index + 1:03d}"
            target = args.output_dir / f"{unit}.json"
            if target.exists():
                entry = json.loads(target.read_text("utf-8"))
                if entry.get("identity") != identity or entry["trace_sha256"] != sha256(trace.model_dump_json().encode()).hexdigest():
                    raise ValueError("saved judgment identity differs")
                print(f"{unit}: {entry['status']} (resumed)", flush=True)
                return entry
            entry = {"unit": unit, "case_id": trace.case_id, "variant": trace.variant.value,
                     "trace_sha256": sha256(trace.model_dump_json().encode()).hexdigest(), "identity": identity,
                     "judge_kind": "ai", "human_reviewed": False, "attempts": [],
                     "status": "not_scored_execution_failed"}
            if trace.status == "success":
                prompt = development_prompt(cases[trace.case_id], trace, args.suite)
                base_prompt = prompt
                for repair in range(2):
                    task = "dev-block-v2-" + sha256((json.dumps(identity, sort_keys=True) + unit + prompt).encode()).hexdigest()[:32]
                    try:
                        result = run_task(prompt, task_id=task, ledger=ledger, max_output=8192,
                                          timeout=180, max_prompt_bytes=131072)
                        attempt = {"raw_response": result, "format_repair": repair}
                        entry["attempts"].append(attempt)
                        judgment = decode(result["text"])
                        score = score_development(cases[trace.case_id], trace, judgment)
                        entry.update(status="valid_ai_judgment", judgment=judgment.model_dump(), score=score.model_dump())
                        break
                    except Exception as error:
                        entry.update(status="judge_error", error_type=type(error).__name__)
                        if entry["attempts"]:
                            entry["attempts"][-1]["validation_error_type"] = type(error).__name__
                        # Only a returned invalid judgment can receive one explicit
                        # format repair. Transport failures are not silently retried.
                        if repair or not entry["attempts"]:
                            break
                        prompt = (base_prompt + "\nYour previous output failed strict validation. "
                                  "Return a complete replacement: every key point and block once, "
                                  "factual blocks without citations must be uncited, nonfactual blocks N/A. "
                                  "Do not change factual judgments merely to satisfy formatting.\nPrevious:\n"
                                  + entry["attempts"][-1]["raw_response"]["text"])
            write(target, entry)
            print(f"{unit}: {entry['status']}", flush=True)
            return entry

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            entries = list(pool.map(judge, enumerate(report.traces)))
        summary = {"identity": identity, "judge_kind": "ai", "human_reviewed": False,
                   "units": [{key: value for key, value in entry.items() if key != "attempts"} for entry in entries],
                   "valid_count": sum(item["status"] == "valid_ai_judgment" for item in entries),
                   "judge_error_count": sum(item["status"] == "judge_error" for item in entries),
                   "execution_failure_count": sum(item["status"] == "not_scored_execution_failed" for item in entries)}
        write(args.output_dir / "summary.json", summary)
        print(f"Valid={summary['valid_count']}; judge errors={summary['judge_error_count']}; execution failures={summary['execution_failure_count']}.")
        return 0 if not summary["judge_error_count"] else 3
    except Exception as error:
        print(f"Development scoring stopped ({type(error).__name__}).", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
