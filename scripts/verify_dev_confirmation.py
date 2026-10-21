"""Verify targeted engineering records separately from the 80-unit experiment."""

import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from app.evaluation.budget import Price,cny
from app.evaluation.trace import EvaluationTrace
from app.evaluation.workflows import _validate_citations
from scripts.verify_full_dev import check_usage


def verify(path):
    packet=json.loads(path.read_text("utf-8"))
    assert packet["human_reviewed"] is False and packet["semantic_scoring_performed"] is False
    traces=[EvaluationTrace.model_validate_json(json.dumps(t)) for t in packet["traces"]]
    expected=({("REAL-DEV-019","multi_agent_rag")} if path.name=="enum-confirmation.json" else
              {("REAL-DEV-019","multi_agent_rag"),("REAL-DEV-020","multi_agent_rag"),
               ("REAL-DEV-007","llm_rag"),("REAL-DEV-007","single_agent_rag")})
    assert len(traces)==len(expected) and {(t.case_id,t.variant.value) for t in traces}==expected
    assert all(m["git_commit"]==packet["generation_commit"] and not m["git_dirty"] for m in packet["metadata"].values())
    rows=packet["request_receipts"]
    by_id={row["id"]:row for row in rows}
    assert len(rows)==len(by_id)
    for row in rows:
        if row["state"]=="settled" and row["price"] is not None:
            assert row["estimated_cost_cny"]==cny(Price(**row["price"]).cost(row["input_tokens"],row["output_tokens"],row["cached_input_tokens"]))
        else:
            assert row["estimated_cost_cny"] is None
    check_usage(packet["usage"],rows)
    assigned=set()
    for trace in traces:
        assert not assigned & set(trace.attempt_ids)
        assigned.update(trace.attempt_ids)
        selected=[by_id[identifier] for identifier in trace.attempt_ids]
        assert len(selected)==trace.model_calls
        assert sum(row["input_tokens"] or 0 for row in selected)==trace.prompt_tokens
        assert sum(row["output_tokens"] or 0 for row in selected)==trace.completion_tokens
        if trace.status=="success": _validate_citations(trace)
    assert sum(t.status=="success" for t in traces)==packet["successful_units"]
    return len(traces),packet["successful_units"],len(rows)


if __name__=="__main__":
    for name in ("post-dev-confirmation.json","enum-confirmation.json"):
        path=ROOT/"docs/evaluation"/name
        if path.exists():
            count,success,requests=verify(path)
            print(f"Verified {name}: {success}/{count} successful units, {requests} physical receipts.")
