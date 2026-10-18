"""Recompute and publish public-corpus pilot evidence without paid calls.

The runtime ledger is not published. Only allowlisted physical-attempt usage and
public question/answer/evidence traces leave artifacts/. Raw judge feedback and
explicit Codex adjudications remain distinguishable from final AI scores.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.budget import BudgetLedger, cny
from app.evaluation.models import WorkflowVariant
from app.evaluation.real_dataset import load_real_suite
from app.evaluation.runner import EvaluationReport
from app.evaluation.semantic import SemanticJudgment, score_semantic_judgment


def usage(rows):
    complete = all(row["input_tokens"] is not None and row["output_tokens"] is not None for row in rows)
    priced = complete and all(row["price_json"] != "null" and row["state"] == "settled" for row in rows)
    return {"physical_attempts": len(rows), "usage_complete": complete,
            "input_tokens": sum(row["input_tokens"] or 0 for row in rows),
            "output_tokens": sum(row["output_tokens"] or 0 for row in rows),
            "cached_input_tokens": sum(row["cached_tokens"] or 0 for row in rows),
            "estimated_cost_cny": cny(sum(row["charged"] for row in rows)) if priced else None,
            "unknown_usage_attempts": sum(row["state"] in {"reserved", "unknown"} for row in rows)}


def average(values):
    applicable = [value for value in values if value is not None]
    return statistics.mean(applicable) if applicable else None


def publish(root: Path, output: Path, corrections: Path | None = None):
    output.mkdir(parents=True, exist_ok=True)
    by_id = {case.id: case for case in load_real_suite(
        ROOT / "benchmarks/real-v1/corpus-manifest.json", ROOT / "benchmarks/real-v1/dev.jsonl", "dev").cases}
    overrides = json.loads(corrections.read_text("utf-8")) if corrections else {}
    corrected = overrides.get("adjudications", {})
    previous = set()
    summaries, publications = {}, []
    for name in ["initial", "diagnostic", "followup", "calibrated", "final"]:
        run_dir = root / name
        report = EvaluationReport.model_validate_json((run_dir / "results.json").read_text("utf-8"))
        manifest = json.loads((run_dir / "run-manifest.json").read_text("utf-8"))
        budget = json.loads((run_dir / "budget-summary.json").read_text("utf-8"))
        all_rows = budget["attempts"]
        rows = [row for row in all_rows if row["id"] not in previous]
        if not previous <= {row["id"] for row in all_rows}:
            raise ValueError("run ledgers are not consecutive snapshots")
        previous = {row["id"] for row in all_rows}
        by_attempt = {row["id"]: row for row in rows}
        assigned = set()
        variant_summaries = {}
        for variant in WorkflowVariant:
            traces = [trace for trace in report.traces if trace.variant == variant]
            if not traces:
                continue
            identifiers = [identifier for trace in traces for identifier in trace.attempt_ids]
            if len(identifiers) != len(set(identifiers)) or set(identifiers) & assigned or not set(identifiers) <= by_attempt.keys():
                raise ValueError("trace attempt IDs overlap or leave this run scope")
            assigned.update(identifiers)
            variant_rows = [by_attempt[identifier] for identifier in identifiers]
            metrics = report.by_variant[variant]
            entry = {"trace_count": len(traces), "successes": sum(trace.status == "success" for trace in traces),
                     "median_latency_seconds": statistics.median(trace.latency_ms for trace in traces) / 1000,
                     "mean_latency_seconds": statistics.mean(trace.latency_ms for trace in traces) / 1000,
                     "first_round_recall_at_5": metrics.mean_retrieval_recall_at_5,
                     "recall_denominator": metrics.metric_denominators.get("retrieval_recall_at_5"),
                     "chat": usage([row for row in variant_rows if row["kind"] == "chat"]),
                     "query_embedding": usage([row for row in variant_rows if row["kind"] == "embedding"]),
                     "all_requests": usage(variant_rows)}
            variant_summaries[variant.value] = entry
        unassigned = [row for row in rows if row["id"] not in assigned]
        summaries[name] = {"metadata": report.metadata.model_dump(mode="json"), "by_variant": variant_summaries,
                           "generation_manifest": {key: value for key, value in manifest.items() if key != "ledger"},
                           "shared_index_embedding": usage([row for row in unassigned if row["kind"] == "embedding"]),
                           "unassigned_non_embedding": usage([row for row in unassigned if row["kind"] != "embedding"]),
                           "total": usage(rows),
                           "failures": [{"case_id": trace.case_id, "variant": trace.variant.value, "error": trace.error_code}
                                        for trace in report.traces if trace.status != "success"]}
        # This is evaluator-owned public data, not raw provider HTTP responses.
        path = output / f"{name}-traces.jsonl"
        path.write_text("".join(trace.model_dump_json() + "\n" for trace in report.traces), "utf-8")
        publications.append({"path": path.name, "sha256": sha256(path.read_bytes()).hexdigest()})

    followup = EvaluationReport.model_validate_json((root / "final/results.json").read_text("utf-8"))
    if len(followup.traces) != 20 or len({trace.case_id for trace in followup.traces}) != 5:
        raise ValueError("pilot must contain all twenty prespecified units")
    # Judge packets are order-blinded before requests; labels are joined here.
    judgments = []
    for index, trace in enumerate(followup.traces):
        unit = f"answer-{index + 1:03d}"
        raw = json.loads((root / f"judge-final/{unit}.json").read_text("utf-8"))
        if raw["case_id"] != trace.case_id or raw["variant"] != trace.variant.value:
            raise ValueError("judge unit mismatch")
        entry = {"unit": unit, "case_id": trace.case_id, "variant": trace.variant.value,
                 "original_status": raw["status"], "raw_response": raw.get("raw_response"),
                 "judge_kind": "ai", "human_reviewed": False}
        decision = corrected.get(unit)
        if decision:
            if not decision.get("reason") or decision.get("trace_sha256") != sha256(trace.model_dump_json().encode()).hexdigest():
                raise ValueError("adjudication requires a reason and exact trace identity")
            judgment = SemanticJudgment.model_validate(decision["judgment"])
            entry["codex_adjudication"] = decision
        elif raw["status"] == "valid_ai_judgment":
            judgment = SemanticJudgment.model_validate(raw["judgment"])
        else:
            entry["status"] = "missing_judge_score"
            judgments.append(entry)
            continue
        entry.update(status="valid_ai_judgment", judgment=judgment.model_dump(),
                     score=score_semantic_judgment(by_id[trace.case_id], trace, judgment).model_dump())
        judgments.append(entry)
    if set(corrected) - {item["unit"] for item in judgments}:
        raise ValueError("unknown adjudication unit")
    for variant in WorkflowVariant:
        entries = [item for item in judgments if item["variant"] == variant.value and item["status"] == "valid_ai_judgment"]
        target = summaries["final"]["by_variant"][variant.value]
        target["valid_ai_score_count"] = len(entries)
        for metric in ["answer_correctness", "answer_key_point_coverage", "citation_support_rate", "citation_completeness"]:
            values = [item["score"][metric] for item in entries]
            target[metric] = average(values)
            target[f"{metric}_denominator"] = sum(value is not None for value in values)
    ledger_rows = BudgetLedger(root / "ledger.sqlite").summary()["attempts"]
    judge_ids = {identifier for item in judgments for identifier in (item.get("raw_response") or {}).get("attempt_ids", [])}
    summaries["judge_usage"] = usage([row for row in ledger_rows if row["id"] in judge_ids])
    summaries["all_pilot_usage"] = usage(ledger_rows)
    summaries["publications"] = publications
    summaries["limitations"] = ["Development tuning on five questions; no inferential quality claim.",
                                 "AI reference review, AI judge and Codex adjudication; no human scoring.",
                                 "Closed-book model may have seen these public docs in training.",
                                 "Prices are dated estimates, not invoices; account-mode judge price is unknown.",
                                 "Per-variant model_calls includes query embeddings; chat requests are separated here."]
    (output / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", "utf-8")
    (output / "ai-judgments.json").write_text(json.dumps(judgments, ensure_ascii=False, indent=2) + "\n", "utf-8")
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, default=ROOT / "artifacts/evaluation/real-v1-pilot")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/evaluation/dev-pilot")
    parser.add_argument("--adjudications", type=Path)
    args = parser.parse_args(argv)
    try:
        result = publish(args.pilot_root, args.output_dir, args.adjudications)
        print(json.dumps(result["final"]["by_variant"], ensure_ascii=True, indent=2))
        return 0
    except Exception as error:
        print(f"Pilot publication stopped ({type(error).__name__}).", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
