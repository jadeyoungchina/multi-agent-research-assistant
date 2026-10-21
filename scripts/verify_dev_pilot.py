"""Verify published pilot scores, trace identities and cost receipts offline."""

from hashlib import sha256
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.budget import Price, cny
from app.evaluation.real_dataset import load_real_suite
from app.evaluation.semantic import SemanticJudgment, score_semantic_judgment
from app.evaluation.trace import EvaluationTrace


def verify(root=ROOT / "docs/evaluation/dev-pilot"):
    summary = json.loads((root / "summary.json").read_text("utf-8"))
    for publication in summary["publications"]:
        path = root / publication["path"]
        if path.parent.resolve() != root.resolve() or sha256(path.read_bytes()).hexdigest() != publication["sha256"]:
            raise ValueError("publication hash mismatch")
    cases = {case.id: case for case in load_real_suite(
        ROOT / "benchmarks/real-v1/corpus-manifest.json", ROOT / "benchmarks/real-v1/dev.jsonl", "dev").cases}
    traces = [EvaluationTrace.model_validate_json(line) for line in (root / "final-traces.jsonl").read_text("utf-8").splitlines()]
    units = {(trace.case_id, trace.variant.value): trace for trace in traces}
    if len(traces) != len(units) or len(traces) != 20:
        raise ValueError("missing or duplicate final unit")
    if any(trace.case_id not in cases for trace in traces):
        raise ValueError("non-development trace")
    judgments = json.loads((root / "ai-judgments.json").read_text("utf-8"))
    if {(item["case_id"], item["variant"]) for item in judgments} != units.keys() or len(judgments) != 20:
        raise ValueError("judge matrix differs from generation matrix")
    for item in judgments:
        if item["status"] != "valid_ai_judgment":
            continue
        trace = units[item["case_id"], item["variant"]]
        score = score_semantic_judgment(cases[trace.case_id], trace, SemanticJudgment.model_validate(item["judgment"]))
        if score.model_dump() != item["score"] or item["human_reviewed"] is not False:
            raise ValueError("semantic score or AI label mismatch")
    for variant, aggregate in summary["final"]["by_variant"].items():
        selected = [item for item in judgments if item["variant"] == variant and item["status"] == "valid_ai_judgment"]
        if len(selected) != aggregate["valid_ai_score_count"]:
            raise ValueError("semantic denominator mismatch")
        for metric in ["answer_correctness", "answer_key_point_coverage", "citation_support_rate", "citation_completeness"]:
            values = [item["score"][metric] for item in selected if item["score"][metric] is not None]
            expected = statistics.mean(values) if values else None
            if aggregate[metric] != expected or aggregate[f"{metric}_denominator"] != len(values):
                raise ValueError("semantic aggregate mismatch")
    receipts = [json.loads(line) for line in (root / "request-usage.jsonl").read_text("utf-8").splitlines()]
    by_id = {row["id"]: row for row in receipts}
    if len(by_id) != len(receipts):
        raise ValueError("duplicate physical request")
    for row in receipts:
        if row["state"] == "settled" and row["price"] is not None:
            expected = cny(Price(**row["price"]).cost(row["input_tokens"], row["output_tokens"], row["cached_input_tokens"]))
            if expected != row["estimated_cost_cny"]:
                raise ValueError("cost receipt mismatch")
        elif row["estimated_cost_cny"] is not None:
            raise ValueError("unknown cost presented as known")
    for trace in traces:
        rows = [by_id[identifier] for identifier in trace.attempt_ids]
        if any(row["phase"] != "final" for row in rows):
            raise ValueError("trace references a request from another run")
        if len(rows) != trace.model_calls or sum(row["input_tokens"] or 0 for row in rows) != trace.prompt_tokens or sum(row["output_tokens"] or 0 for row in rows) != trace.completion_tokens:
            raise ValueError("final trace usage differs from physical receipts")
    return len(traces), len(receipts)


if __name__ == "__main__":
    try:
        traces, attempts = verify()
        print(f"Published pilot verified offline: {traces} final traces, {attempts} physical receipts; scores, hashes and costs match.")
    except Exception as error:
        print(f"Pilot verification failed ({type(error).__name__}).", file=sys.stderr)
        raise SystemExit(1)
