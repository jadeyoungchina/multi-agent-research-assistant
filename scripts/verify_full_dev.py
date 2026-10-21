"""Recompute the published 80-unit dev matrix without credentials or API calls."""

from hashlib import sha256
from datetime import datetime
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.budget import Price, amount, cny
from app.evaluation.development import DevelopmentJudgment, load_scoring_cases, score_development
from app.evaluation.trace import EvaluationTrace
from scripts.publish_full_dev import citation_diagnostic


def check_usage(aggregate, rows):
    complete = all(row["input_tokens"] is not None and row["output_tokens"] is not None for row in rows)
    priced = complete and all(row["state"] == "settled" and row["price"] is not None for row in rows)
    expected = {"physical_attempts":len(rows),"usage_complete":complete,
                "input_tokens":sum(row["input_tokens"] or 0 for row in rows),
                "output_tokens":sum(row["output_tokens"] or 0 for row in rows),
                "cached_input_tokens":sum(row["cached_input_tokens"] or 0 for row in rows),
                "estimated_cost_cny":cny(sum(amount(row["estimated_cost_cny"]) for row in rows)) if priced else None,
                "unknown_usage_attempts":sum(row["state"] in {"reserved","unknown"} for row in rows)}
    if aggregate != expected:
        raise ValueError("usage aggregate differs")


def verify(root=ROOT / "docs/evaluation/full-dev"):
    summary = json.loads((root / "summary.json").read_text("utf-8"))
    for item in summary["publications"]:
        path = root / item["path"]
        if path.parent.resolve() != root.resolve() or sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("published file identity differs")
    suite = ROOT / "benchmarks/real-v1"
    cases = load_scoring_cases(suite)
    if summary["policy_sha256"] != sha256((suite / "dev-scoring-v2.json").read_bytes()).hexdigest():
        raise ValueError("scoring policy differs")
    traces = [EvaluationTrace.model_validate_json(line) for line in (root / "traces.jsonl").read_text("utf-8").splitlines()]
    units = {(trace.case_id, trace.variant.value):trace for trace in traces}
    expected = {(case, variant) for case in cases for variant in ("baseline_llm","llm_rag","single_agent_rag","multi_agent_rag")}
    if len(traces) != 80 or units.keys() != expected:
        raise ValueError("missing or duplicate development unit")
    judgments = json.loads((root / "ai-judgments.json").read_text("utf-8"))
    if len(judgments) != 80 or {(j["case_id"],j["variant"]) for j in judgments} != expected:
        raise ValueError("judge matrix differs")
    for entry in judgments:
        trace = units[entry["case_id"],entry["variant"]]
        if entry["human_reviewed"] is not False or entry["original"]["trace_sha256"] != sha256(trace.model_dump_json().encode()).hexdigest():
            raise ValueError("judge identity or AI attribution differs")
        if entry["status"] == "valid_ai_judgment":
            score = score_development(cases[trace.case_id],trace,DevelopmentJudgment.model_validate(entry["judgment"]))
            if score.model_dump() != entry["score"]:
                raise ValueError("semantic score differs")
    valid_count = sum(entry["status"] == "valid_ai_judgment" for entry in judgments)
    if summary["valid_ai_scores"] != valid_count or summary["successful_units"] != sum(t.status=="success" for t in traces):
        raise ValueError("execution or score denominator differs")
    for variant, aggregate in summary["by_variant"].items():
        selected = [j for j in judgments if j["variant"] == variant and j["status"] == "valid_ai_judgment"]
        if aggregate["valid_ai_score_count"] != len(selected):
            raise ValueError("variant score denominator differs")
        for metric in ("answer_correctness","answer_key_point_coverage","citation_support_rate","citation_completeness",
                       "correct_abstention","false_abstention","unsupported_assertion"):
            values = [j["score"][metric] for j in selected if j["score"][metric] is not None]
            if aggregate[metric] != (statistics.mean(values) if values else None) or aggregate[metric+"_denominator"] != len(values):
                raise ValueError("semantic aggregate differs")
        times = [t.latency_ms/1000 for t in traces if t.variant.value==variant and t.status=="success"]
        if aggregate["successes"] != len(times) or aggregate["median_success_latency_seconds"] != (statistics.median(times) if times else None):
            raise ValueError("success latency denominator differs")
        diagnostics = [citation_diagnostic(t) for t in traces if t.variant.value==variant]
        applicable = [item for item in diagnostics if item["applicable"]]
        if (aggregate["citation_diagnostic_denominator"] != len(applicable)
                or aggregate["citation_format_valid_rate"] != (statistics.mean(item["syntax_valid"] for item in applicable) if applicable else None)
                or aggregate["citation_reference_consistency_rate"] != (statistics.mean(item["passed"] for item in applicable) if applicable else None)):
            raise ValueError("supplementary citation diagnostic differs")
    receipts = [json.loads(line) for line in (root / "request-usage.jsonl").read_text("utf-8").splitlines()]
    by_id = {row["id"]:row for row in receipts}
    if len(by_id) != len(receipts) or summary["all_usage"]["physical_attempts"] != len(receipts):
        raise ValueError("physical attempt count differs")
    for row in receipts:
        if row["state"]=="settled" and row["price"] is not None:
            if cny(Price(**row["price"]).cost(row["input_tokens"],row["output_tokens"],row["cached_input_tokens"])) != row["estimated_cost_cny"]:
                raise ValueError("cost receipt differs")
        elif row["estimated_cost_cny"] is not None:
            raise ValueError("unknown price or usage presented as known")
    assigned = set()
    for trace in traces:
        if len(set(trace.attempt_ids)) != len(trace.attempt_ids) or assigned & set(trace.attempt_ids):
            raise ValueError("physical generation attempt reused")
        assigned.update(trace.attempt_ids)
        rows = [by_id[identifier] for identifier in trace.attempt_ids]
        if any(row["phase"]!="generation" for row in rows) or len(rows)!=trace.model_calls:
            raise ValueError("generation call attribution differs")
        if sum(row["input_tokens"] or 0 for row in rows)!=trace.prompt_tokens or sum(row["output_tokens"] or 0 for row in rows)!=trace.completion_tokens:
            raise ValueError("generation token attribution differs")
    adjudications = json.loads((root / "adjudications.json").read_text("utf-8"))
    decisions = adjudications["adjudications"]
    if (adjudications["human_reviewed"] is not False
            or summary["codex_source_checks"] != len(decisions)
            or summary["corrected_judgments"] != sum(d["action"] == "corrected" for d in decisions.values())):
        raise ValueError("source review attribution or count differs")
    selected = {"REAL-DEV-001", "REAL-DEV-007", "REAL-DEV-013", "REAL-DEV-019", "REAL-DEV-020"}
    required = {j["unit"] for j in judgments if units[j["case_id"],j["variant"]].status == "success"
                and (j["case_id"] in selected or j["original"]["status"] == "judge_error")}
    if not required <= decisions.keys():
        raise ValueError("prespecified source reviews are missing")
    for entry in judgments:
        decision = decisions.get(entry["unit"])
        if decision is None:
            if "codex_adjudication" in entry:
                raise ValueError("undocumented source review")
            continue
        if (entry.get("codex_adjudication") != decision or decision["reviewer_kind"] != "ai"
                or not decision["reason"] or decision["trace_sha256"] != entry["original"]["trace_sha256"]
                or decision["judgment"] != entry["judgment"]):
            raise ValueError("source review identity differs")
        if decision["action"] == "accepted" and decision["judgment"] != entry["original"].get("judgment"):
            raise ValueError("accepted review changes the original judgment")
        if decision["action"] not in {"accepted","corrected"}:
            raise ValueError("unknown source review action")
    failures = {j["unit"] for j in judgments if units[j["case_id"],j["variant"]].status != "success"}
    audits = adjudications.get("execution_failure_audits", {})
    if set(audits) != failures or summary["codex_execution_failure_audits"] != len(audits):
        raise ValueError("execution failure audits are missing")
    if summary["metadata"]["prompt_version"] == "evaluation-v8":
        if summary["metadata"]["git_dirty"] or any(citation_diagnostic(t).get("passed") is False for t in traces):
            raise ValueError("final-rule generation or citation contract differs")
    check_usage(summary["all_usage"],receipts)
    check_usage(summary["generation_usage"],[row for row in receipts if row["phase"]=="generation"])
    check_usage(summary["judge_usage"],[row for row in receipts if row["phase"]=="judge"])
    setup_ids = {row["id"] for row in receipts if row["phase"]=="generation" and row["kind"]=="embedding"
                 and datetime.fromisoformat(row["created_at"]) < datetime.fromisoformat(summary["metadata"]["run_at_utc"])}
    if setup_ids & assigned or any(row["shared_index_setup"] != (row["id"] in setup_ids) for row in receipts):
        raise ValueError("shared index attribution differs")
    check_usage(summary["shared_index_embedding"],[row for row in receipts if row["id"] in setup_ids])
    check_usage(summary["unassigned_query_embedding"],[row for row in receipts if row["phase"]=="generation" and row["kind"]=="embedding" and row["id"] not in assigned | setup_ids])
    check_usage(summary["unassigned_non_embedding"],[row for row in receipts if row["phase"]=="generation" and row["id"] not in assigned and row["kind"]!="embedding"])
    for variant, aggregate in summary["by_variant"].items():
        ids = {identifier for trace in traces if trace.variant.value==variant for identifier in trace.attempt_ids}
        rows = [by_id[identifier] for identifier in ids]
        check_usage(aggregate["all_requests"],rows)
        check_usage(aggregate["chat"],[row for row in rows if row["kind"]=="chat"])
        check_usage(aggregate["query_embedding"],[row for row in rows if row["kind"]=="embedding"])
    return len(traces),valid_count,len(receipts)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", choices=("all", "full-dev", "final-dev"), default="all")
    args = parser.parse_args()
    try:
        packets = ("full-dev", "final-dev") if args.packet == "all" else (args.packet,)
        for packet in packets:
            count,valid,requests=verify(ROOT / "docs/evaluation" / packet)
            print(f"{packet} verified offline: {count} traces; {valid} AI scores; {requests} physical receipts.")
    except Exception as error:
        print(f"Full development verification failed ({type(error).__name__}).",file=sys.stderr)
        raise SystemExit(1)
