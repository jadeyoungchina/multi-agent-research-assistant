"""Publish a completed 80-unit development run from retained local receipts."""

import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.budget import BudgetLedger, cny
from app.evaluation.development import DevelopmentJudgment, load_scoring_cases, score_development
from app.evaluation.runner import EvaluationReport
from scripts.publish_dev_pilot import usage
from app.workflow.citations import _CITATION_TOKEN, _CITATION_FRAGMENT


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def aggregate(rows):
    result = {"valid_ai_score_count": len(rows)}
    for metric in ("answer_correctness", "answer_key_point_coverage", "citation_support_rate",
                   "citation_completeness", "correct_abstention", "false_abstention", "unsupported_assertion"):
        values = [item["score"][metric] for item in rows if item["score"][metric] is not None]
        result[metric] = statistics.mean(values) if values else None
        result[f"{metric}_denominator"] = len(values)
    return result


def citation_diagnostic(trace):
    if trace.status != "success" or trace.variant.value == "baseline_llm":
        return {"applicable":False}
    ids = set(_CITATION_TOKEN.findall(trace.answer))
    known = {item.id for item in trace.retrieved_evidence}
    malformed = bool(_CITATION_FRAGMENT.search(_CITATION_TOKEN.sub("",trace.answer)))
    unknown = sorted((ids | set(trace.cited_evidence_ids)) - known)
    missing = sorted(set(trace.cited_evidence_ids) - ids)
    return {"applicable":True,"syntax_valid":not malformed,"unknown_ids":unknown,
            "declared_ids_missing_inline":missing,"passed":not (malformed or unknown or missing)}


def report_markdown(summary):
    fmt = lambda value: "N/A" if value is None else f"{value:.3f}"
    rows = ["# Complete 20-question development calibration", "",
            f"**{summary['successful_units']}/80 successful executions**; "
            f"**{summary['valid_ai_scores']}/80 valid AI scores**. "
            f"Codex source checks: {summary['codex_source_checks']}/80; human scoring: **0**.", "",
            "This is a single development run after five-question tuning, not a held-out comparison. "
            "The 40 held-out questions have not been generated or judged. No significance or general "
            "architecture-superiority claim follows from these results.", "",
            "[Prespecified protocol](../full-dev-protocol.md) · [Summary](summary.json) · "
            "[Answers and exact evidence](traces.jsonl) · [Original AI judgments and adjudications](ai-judgments.json) · "
            "[Physical receipts](request-usage.jsonl)", "",
            "## Generation and semantic quality", "",
            "| Workflow | Executions | AI scores | Key-point correctness | Full key points | Citation block support | Citation block completeness | Median successful seconds | Chat attempts | Query embeddings | API estimate CNY |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, row in summary["by_variant"].items():
        rows.append(f"| {name} | {row['successes']}/{row['trace_count']} | {row['valid_ai_score_count']} | "
                    f"{fmt(row['answer_correctness'])} | {fmt(row['answer_key_point_coverage'])} | "
                    f"{fmt(row['citation_support_rate'])} | {fmt(row['citation_completeness'])} | "
                    f"{fmt(row['median_success_latency_seconds'])} | {row['chat']['physical_attempts']} | "
                    f"{row['query_embedding']['physical_attempts']} | "
                    f"{row['all_requests']['estimated_cost_cny'] if row['all_requests']['estimated_cost_cny'] is not None else 'unknown'} |")
    rows += ["", "Correctness is the per-answer mean of required key-point scores (0/1/2), normalized to 0–1. "
             "Full key points counts points receiving 2. Metrics are macro means over valid, applicable judgments; "
             "all denominators and missing judgments are explicit in summary.json. Execution success means "
             "the pipeline returned under its then-current checks. Those checks did not enforce uniform "
             "inline citation syntax across workflows; execution success is not correctness or citation validity.", "",
             "Citation metrics audit every deterministic Markdown block, including uncited prose and code. "
             "Full support requires all domain assertions in a cited block to follow from its attached exact "
             "passages. Scoped missing-evidence limitations are N/A. Baseline citation metrics are N/A. "
             "These block denominators differ from the five-question pilot's model-selected atomic claims; "
             "do not compare their citation rates directly. ID validity alone is not entailment.", "",
             "## Abstention and retrieval diagnostics", "",
             "| Workflow | Correct abstention | False abstention | Answers with unsupported assertions | First-round recall@5 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, row in summary["by_variant"].items():
        rows.append(f"| {name} | {fmt(row['correct_abstention'])} (n={row['correct_abstention_denominator']}) | "
                    f"{fmt(row['false_abstention'])} (n={row['false_abstention_denominator']}) | "
                    f"{fmt(row['unsupported_assertion'])} (n={row['unsupported_assertion_denominator']}) | "
                    f"{fmt(row['first_round_recall_at_5'])} (n={row['recall_denominator']}) |")
    rows += ["", "Retrieval uses the original global first-round top five and original gold spans. "
             "A reference can prescribe one of several valid implementations; recall is a diagnostic rather "
             "than semantic answer correctness. Correct abstention means declining an unanswerable measurement "
             "without an invented conclusion; false abstention applies only to answerable questions.", "",
             "## Supplementary citation-format audit", "",
             "This deterministic diagnostic was added after inspecting full-run failures. It is reported "
             "separately from the prespecified semantic metrics and does not relabel historical executions. "
             "All RAG answers are checked with the same exact token syntax, retrieved-ID and declared-inline "
             "consistency rules; failed executions and closed-book citations are N/A.", "",
             "| Workflow | Exact syntax | Syntax and reference consistency | Applicable successful answers |",
             "| --- | ---: | ---: | ---: |"]
    for name,row in summary["by_variant"].items():
        rows.append(f"| {name} | {fmt(row['citation_format_valid_rate'])} | {fmt(row['citation_reference_consistency_rate'])} | {row['citation_diagnostic_denominator']} |")
    rows += ["", "## Cost, failures and provenance", "",
             f"Generation commit: `{summary['generation_manifest']['code_commit']}`; prompt "
             f"`{summary['metadata']['prompt_version']}`. All chat roles use "
             f"`{summary['metadata']['configuration']['model']}`, strict field output, temperature 0, disabled "
             "thinking, cap 4096, two provider retries, at most two critic loops and one writer citation repair. "
             "Real text-embedding-v4 vectors have 1024 dimensions, chunks 1200/200 and top-k 5. All RAG variants "
             "search the same complete eight-document dev pool. Generation is serial with the prespecified "
             "shuffled/rotated order and seed 20261018.", "",
             f"Shared index setup: {summary['shared_index_embedding']['physical_attempts']} embedding attempts; "
             f"{summary['shared_index_embedding']['estimated_cost_cny']} CNY estimated; "
             f"{summary['metadata']['setup_latency_ms']} ms. Setup is excluded from answer latency and variant "
             "costs. Variant costs include all recorded answer attempts, retries and repairs. Failed requests "
             "with unknown usage remain unknown; no zero charge is invented. Dated API price estimates are not invoices.", "",
             f"The combined ledger records {summary['all_usage']['physical_attempts']} physical requests. "
             f"Judge requests: {summary['judge_usage']['physical_attempts']}; separate preparation requests: "
             f"{summary['preparation_usage']['physical_attempts']}. DeepSeek official CLI account-mode pricing "
             "is unverified, so judge and combined monetary totals remain unknown. Raw responses and any "
             "single bounded format repair remain separate. Model aliases can drift for the judge; generation "
             "uses the fixed snapshot above.", "",
             f"Execution failures: {80-summary['successful_units']}; unresolved judge errors: "
             f"{summary['judge_error_count']}; Codex-corrected judgments: {summary['corrected_judgments']}. "
             "Failures remain missing for semantic averages, with their execution denominator visible. "
             "The original judge output is never overwritten by a source adjudication.", "",
             "[Failure diagnostics](failure-diagnostics.json) preserve only allowlisted generated response "
             "captures associated with failed traces. They contain public-corpus model text and validation "
             "locations, not provider HTTP headers or credentials.", "",
             "The two scoring-only reference clarifications, source review and local runtime check were "
             "recorded before full generation. The original dataset and historical pilot references remain "
             "unchanged. Codex source checks cover successful answers for the prespecified five pilot IDs "
             "across all workflows; execution failures are audited separately. This is AI review, not human "
             "validation. Remaining "
             "judge errors and correlated AI biases may remain even with schema-complete output. Public "
             "documentation may have appeared in model pretraining. Latency is exploratory on one shared machine.", "",
             "Recompute published hashes, scores, denominators and request costs without API calls:", "", "```powershell",
             ".venv/Scripts/python.exe scripts/verify_full_dev.py", "```", "",
             "This records the complete development run, including its unresolved limitations. "
             "Fixing remaining development failures, experiment freeze, repeated held-out testing, "
             "independent human validation and any final comparison remain separate future work.", ""]
    markdown = "\n".join(rows)
    if summary["metadata"]["prompt_version"] == "evaluation-v8":
        markdown = markdown.replace("# Complete 20-question development calibration", "# Final-rule 20-question development regression")
        markdown = markdown.replace("../full-dev-protocol.md", "../final-dev-protocol.md")
        markdown = markdown.replace("one writer citation repair", "one researcher and one answer/writer citation correction")
        markdown = markdown.replace(
            "the pipeline returned under its then-current checks. Those checks did not enforce uniform "
            "inline citation syntax across workflows; execution success is not correctness or citation validity.",
            "the pipeline returned under shared strict inline citation checks. Structural validity does not "
            "establish semantic correctness or citation entailment.")
        markdown = markdown.replace(
            "This deterministic diagnostic was added after inspecting full-run failures. It is reported "
            "separately from the prespecified semantic metrics and does not relabel historical executions.",
            "This shared structural check was fixed before the final-rule regression. "
            "Its results are separate from semantic support and do not relabel earlier executions.")
        markdown = markdown.replace(
            "This is a single development run after five-question tuning, not a held-out comparison.",
            "This is a separate final-rule development regression after tuning on the same questions. "
            "It is not pooled with earlier runs or treated as a held-out comparison.")
        markdown = markdown.replace(
            "Fixing remaining development failures, experiment freeze, repeated held-out testing, "
            "independent human validation and any final comparison remain separate future work.",
            "The final-rule development regression is complete. Experiment freeze, repeated held-out "
            "testing, independent human validation and any final comparison remain separate future work.")
    return markdown


def publish(root, output, adjudications_path):
    output.mkdir(parents=True, exist_ok=True)
    suite = ROOT / "benchmarks/real-v1"
    cases = load_scoring_cases(suite)
    report = EvaluationReport.model_validate_json((root / "generation/results.json").read_text("utf-8"))
    if report.metadata.synthetic or not report.execution_complete or len(report.traces) != 80:
        raise ValueError("requires a completed 80-unit real development run")
    if {(t.case_id,t.variant.value) for t in report.traces} != {(c,v) for c in cases for v in ("baseline_llm","llm_rag","single_agent_rag","multi_agent_rag")}:
        raise ValueError("development matrix is incomplete")
    overrides = json.loads(adjudications_path.read_text("utf-8"))
    policy_hash = sha256((suite / "dev-scoring-v2.json").read_bytes()).hexdigest()
    if overrides["policy_sha256"] != policy_hash or overrides["human_reviewed"] is not False:
        raise ValueError("adjudications use a different scoring policy")
    decisions = overrides["adjudications"]
    judgments = []
    for index, trace in enumerate(report.traces, 1):
        unit = f"answer-{index:03d}"
        entry = json.loads((root / f"judgments/{unit}.json").read_text("utf-8"))
        if (entry["case_id"],entry["variant"]) != (trace.case_id,trace.variant.value) or entry["trace_sha256"] != sha256(trace.model_dump_json().encode()).hexdigest() or entry["identity"]["policy_sha256"] != policy_hash:
            raise ValueError("original judgment identity differs")
        original = dict(entry)
        decision = decisions.get(unit)
        if decision:
            if (decision["action"] not in {"accepted","corrected"} or decision["reviewer_kind"] != "ai"
                    or not decision["reason"] or decision["trace_sha256"] != entry["trace_sha256"]):
                raise ValueError("source adjudication lacks an exact trace and reason")
            judgment = DevelopmentJudgment.model_validate(decision["judgment"])
            entry.update(status="valid_ai_judgment", judgment=judgment.model_dump(), codex_adjudication=decision)
        if entry["status"] == "valid_ai_judgment":
            entry["score"] = score_development(cases[trace.case_id], trace, DevelopmentJudgment.model_validate(entry["judgment"])).model_dump()
        judgments.append({"unit":unit,"case_id":trace.case_id,"variant":trace.variant.value,
                          "status":entry["status"],"human_reviewed":False,"original":original,
                          **{key:entry[key] for key in ("judgment","score","codex_adjudication") if key in entry}})
    if set(decisions) - {item["unit"] for item in judgments}:
        raise ValueError("unknown adjudication unit")
    all_rows = BudgetLedger(root / "ledger.sqlite").summary()["attempts"]
    generation_snapshot = json.loads((root / "generation/budget-summary.json").read_text("utf-8"))["attempts"]
    gen_ids = {row["id"] for row in generation_snapshot}
    gen_rows = [row for row in all_rows if row["id"] in gen_ids]
    judge_rows = [row for row in all_rows if row["id"] not in gen_ids]
    by_id = {row["id"]:row for row in gen_rows}
    assigned = set()
    by_variant = {}
    for variant in report.by_variant:
        traces = [trace for trace in report.traces if trace.variant == variant]
        identifiers = [identifier for trace in traces for identifier in trace.attempt_ids]
        if len(identifiers) != len(set(identifiers)) or assigned & set(identifiers) or not set(identifiers) <= by_id.keys():
            raise ValueError("generation attempt assignment differs")
        assigned.update(identifiers)
        receipts = [by_id[identifier] for identifier in identifiers]
        valid = [item for item in judgments if item["variant"] == variant.value and item["status"] == "valid_ai_judgment"]
        success_times = [trace.latency_ms/1000 for trace in traces if trace.status == "success"]
        by_variant[variant.value] = dict(aggregate(valid), trace_count=len(traces), successes=len(success_times),
            median_success_latency_seconds=statistics.median(success_times) if success_times else None,
            first_round_recall_at_5=report.by_variant[variant].mean_retrieval_recall_at_5,
            recall_denominator=report.by_variant[variant].metric_denominators.get("retrieval_recall_at_5"),
            chat=usage([row for row in receipts if row["kind"]=="chat"]),
            query_embedding=usage([row for row in receipts if row["kind"]=="embedding"]), all_requests=usage(receipts))
        diagnostics = [citation_diagnostic(trace) for trace in traces]
        applicable = [item for item in diagnostics if item["applicable"]]
        by_variant[variant.value].update(citation_diagnostic_denominator=len(applicable),
            citation_format_valid_rate=statistics.mean(item["syntax_valid"] for item in applicable) if applicable else None,
            citation_reference_consistency_rate=statistics.mean(item["passed"] for item in applicable) if applicable else None)
    manifest = json.loads((root / "generation/run-manifest.json").read_text("utf-8"))
    prep_path = root / "preparation/ledger.sqlite"
    prep_rows = BudgetLedger(prep_path).summary()["attempts"] if prep_path.exists() else []
    setup_ids = {row["id"] for row in gen_rows if row["kind"] == "embedding"
                 and datetime.fromisoformat(row["created_at"]) < report.metadata.run_at_utc}
    if setup_ids & assigned:
        raise ValueError("index construction overlaps answer requests")
    summary = {"schema_version":"full-dev-publication-v1", "semantic_rubric":"dev-block-v2",
        "policy_sha256":policy_hash,"metadata":report.metadata.model_dump(mode="json"),
        "generation_manifest":{key:value for key,value in manifest.items() if key!="ledger"},
        "successful_units":sum(t.status=="success" for t in report.traces),
        "valid_ai_scores":sum(item["status"]=="valid_ai_judgment" for item in judgments),
        "codex_source_checks":len(decisions),"corrected_judgments":sum(d["action"]=="corrected" for d in decisions.values()),
        "codex_execution_failure_audits":len(overrides.get("execution_failure_audits",{})),
        "judge_error_count":sum(item["status"]=="judge_error" for item in judgments),
        "by_variant":by_variant,"generation_usage":usage(gen_rows),"judge_usage":usage(judge_rows),
        "preparation_usage":usage(prep_rows),"all_usage":usage(all_rows),
        "shared_index_embedding":usage([row for row in gen_rows if row["id"] in setup_ids]),
        "unassigned_query_embedding":usage([row for row in gen_rows if row["id"] not in assigned | setup_ids and row["kind"]=="embedding"]),
        "unassigned_non_embedding":usage([row for row in gen_rows if row["id"] not in assigned and row["kind"]!="embedding"]),
        "failures":[{"case_id":t.case_id,"variant":t.variant.value,"error":t.error_code} for t in report.traces if t.status!="success"],
        "by_question_type":{kind:{variant:aggregate([item for item in judgments if item["variant"]==variant and cases[item["case_id"]].question_type==kind and item["status"]=="valid_ai_judgment"]) for variant in by_variant} for kind in sorted({c.question_type for c in cases.values()})}}
    receipts = [{"id":row["id"],"phase":"generation" if row["id"] in gen_ids else "judge",
        "provider":row["provider"],"model":row["model"],"kind":row["kind"],"role":row["role"],
        "shared_index_setup":row["id"] in setup_ids,
        "created_at":row["created_at"],"request_sha256":row["request_hash"],"state":row["state"],"outcome":row["outcome"],
        "input_tokens":row["input_tokens"],"output_tokens":row["output_tokens"],"cached_input_tokens":row["cached_tokens"],
        "price":json.loads(row["price_json"]),"estimated_cost_cny":cny(row["charged"]) if row["state"]=="settled" and row["price_json"]!="null" else None}
        for row in all_rows]
    (output / "traces.jsonl").write_text("".join(t.model_dump_json()+"\n" for t in report.traces),encoding="utf-8",newline="\n")
    (output / "request-usage.jsonl").write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in receipts),encoding="utf-8",newline="\n")
    write(output / "ai-judgments.json", judgments)
    write(output / "adjudications.json", overrides)
    captures = [json.loads(line) for line in (root / "generation/provider-responses.jsonl").read_text("utf-8").splitlines()]
    failed = [{"case_id":trace.case_id,"variant":trace.variant.value,"trace_sha256":sha256(trace.model_dump_json().encode()).hexdigest(),
               "error_code":trace.error_code,
               "captures":[{key:item[key] for key in ("attempt_id","schema","content","finish_reason","validation_issues")}
                           for item in captures if item["attempt_id"] in trace.attempt_ids and (item["schema"]=="DraftReport" or item["validation_issues"])]}
              for trace in report.traces if trace.status!="success"]
    write(output / "failure-diagnostics.json",failed)
    summary["publications"] = [{"path":name,"sha256":sha256((output/name).read_bytes()).hexdigest()} for name in ("traces.jsonl","request-usage.jsonl","ai-judgments.json","adjudications.json","failure-diagnostics.json")]
    write(output / "summary.json",summary)
    (output / "README.md").write_text(report_markdown(summary),encoding="utf-8",newline="\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT / "artifacts/evaluation/real-v1-dev-full")
    parser.add_argument("--output-dir",type=Path,default=ROOT / "docs/evaluation/full-dev")
    parser.add_argument("--adjudications",type=Path,required=True)
    args = parser.parse_args(argv)
    try:
        result=publish(args.root,args.output_dir,args.adjudications)
        print(json.dumps(result["by_variant"],ensure_ascii=False,indent=2))
        return 0
    except Exception as error:
        print(f"Development publication stopped ({type(error).__name__}).",file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
