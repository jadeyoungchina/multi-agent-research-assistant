"""Publish four targeted engineering confirmations without replacing v6 results."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from app.evaluation.budget import BudgetLedger, cny
from app.evaluation.runner import EvaluationReport
from scripts.publish_dev_pilot import usage
from scripts.publish_full_dev import citation_diagnostic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enum",action="store_true",help="publish the separate v8 single-unit confirmation")
    args=parser.parse_args()
    root=ROOT/("artifacts/evaluation/real-v1-dev-enum" if args.enum else "artifacts/evaluation/real-v1-dev-followup")
    reports={name:EvaluationReport.model_validate_json((root/name/"results.json").read_text("utf-8")) for name in (("generation",) if args.enum else ("multi","baselines"))}
    traces=[trace for report in reports.values() for trace in report.traces]
    expected={("REAL-DEV-019","multi_agent_rag"),("REAL-DEV-020","multi_agent_rag"),
              ("REAL-DEV-007","llm_rag"),("REAL-DEV-007","single_agent_rag")}
    if args.enum: expected={("REAL-DEV-019","multi_agent_rag")}
    if len(traces)!=len(expected) or {(t.case_id,t.variant.value) for t in traces}!=expected:
        raise ValueError("targeted matrix differs")
    version="evaluation-v8" if args.enum else "evaluation-v7"
    if any(report.metadata.git_dirty or report.metadata.prompt_version!=version for report in reports.values()):
        raise ValueError("confirmation must use the selected clean source")
    if len({report.metadata.git_commit for report in reports.values()})!=1:
        raise ValueError("confirmation source commits differ")
    rows=BudgetLedger(root/"ledger.sqlite").summary()["attempts"]
    receipts=[{"id":row["id"],"provider":row["provider"],"model":row["model"],"kind":row["kind"],
               "state":row["state"],"outcome":row["outcome"],"request_sha256":row["request_hash"],
               "input_tokens":row["input_tokens"],"output_tokens":row["output_tokens"],
               "cached_input_tokens":row["cached_tokens"],"price":json.loads(row["price_json"]),
               "estimated_cost_cny":cny(row["charged"]) if row["state"]=="settled" and row["price_json"]!="null" else None}
              for row in rows]
    checks=[{"case_id":trace.case_id,"variant":trace.variant.value,"status":trace.status,
             "error_code":trace.error_code,"trace_sha256":sha256(trace.model_dump_json().encode()).hexdigest(),
             "citation_diagnostic":citation_diagnostic(trace)} for trace in traces]
    packet={"schema_version":"targeted-dev-confirmation-v1","purpose":"Engineering confirmation selected after v6 failures; not independent accuracy repeats.",
            "generation_commit":next(iter(reports.values())).metadata.git_commit,"human_reviewed":False,
            "semantic_scoring_performed":False,"successful_units":sum(t.status=="success" for t in traces),
            "checks":checks,"metadata":{name:report.metadata.model_dump(mode="json") for name,report in reports.items()},
            "traces":[trace.model_dump(mode="json") for trace in traces],"request_receipts":receipts,"usage":usage(rows),
            "limits":["The original 80-unit v6 report, three failures and citation-format diagnostics are unchanged.",
                      "Selected engineering units cannot establish general reliability, semantic superiority or held-out quality.",
                      "The 4096-token cap remains; truncation is still an explicit failure mode."]}
    path=ROOT/"docs/evaluation"/("enum-confirmation.json" if args.enum else "post-dev-confirmation.json")
    path.write_text(json.dumps(packet,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(f"Published {packet['successful_units']}/{len(expected)} successful targeted units; {len(receipts)} physical receipts.")


if __name__=="__main__":
    main()
