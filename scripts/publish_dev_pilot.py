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


def markdown_report(summary, judgments):
    final = summary["final"]
    config = final["metadata"]["configuration"]
    valid = sum(item["status"] == "valid_ai_judgment" for item in judgments)
    reviewed = sum("codex_adjudication" in item for item in judgments)
    corrected = sum(item.get("codex_adjudication", {}).get("action") == "corrected" for item in judgments)
    fmt = lambda value: "N/A" if value is None else f"{value:.3f}"
    lines = [
        "# Five-question live development pilot", "",
        f"Final matrix: **{sum(row['successes'] for row in final['by_variant'].values())}/20 successful executions**, "
        f"**{valid}/20 valid AI scores**. Codex source checks: {reviewed}/20; "
        f"{corrected} judge outputs corrected with explicit reasons. Human scoring: **0**.", "",
        "This is development calibration on the same five prespecified questions, not a held-out "
        "estimate or evidence that one architecture wins. Earlier failures are retained separately. "
        "The 40 held-out questions have not been run.", "",
        "[Selection and tuning decisions](../dev-pilot-selection.md) · [Machine-readable summary](summary.json) · "
        "[Final answers and evidence](final-traces.jsonl) · [AI judgments and adjudications](ai-judgments.json) · "
        "[Physical request receipts](request-usage.jsonl)", "",
        "## Final development matrix", "",
        "| Variant | Executions | AI scores | Key-point correctness | Citation support | Citation completeness | First-round recall@5 | Median seconds | Chat attempts | Query embedding attempts | API estimate CNY |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, row in final["by_variant"].items():
        lines.append(f"| {variant} | {row['successes']}/{row['trace_count']} | {row['valid_ai_score_count']} | "
                     f"{fmt(row['answer_correctness'])} | {fmt(row['citation_support_rate'])} | "
                     f"{fmt(row['citation_completeness'])} | {fmt(row['first_round_recall_at_5'])} | "
                     f"{row['median_latency_seconds']:.3f} | {row['chat']['physical_attempts']} | "
                     f"{row['query_embedding']['physical_attempts']} | {row['all_requests']['estimated_cost_cny'] or 'unknown'} |")
    lines += ["", "Correctness is a per-answer mean of reference-key-point scores (0/1/2), normalized to 0–1. "
              "Citation support checks entailment of attached factual claims; completeness includes uncited factual claims. "
              "They are macro means over applicable, valid judgments, not citation-ID validity. Applicable counts are "
              "explicit in summary.json; baseline citations and empty-gold recall are N/A. Failed execution and judge "
              "errors remain missing for semantic quality, with execution failures separately visible. A valid ID alone "
              "never establishes claim support. First-round recall uses the original global top five.", "",
              f"Shared index setup: {final['shared_index_embedding']['physical_attempts']} embedding requests, "
              f"{final['shared_index_embedding']['estimated_cost_cny']} CNY estimated; "
              f"{final['metadata']['setup_latency_ms']} ms. This setup is excluded from answer latency and per-variant cost. "
              "Request counts include failed attempts and bounded repairs. Trace model_calls counts both chat and "
              "query embeddings; the table separates them.", "",
              "## Retained tuning history", "",
              "| Phase | Successful traces | Failed traces | Chat attempts | Embedding attempts including setup | API estimate CNY |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for phase in ["initial", "diagnostic", "followup", "calibrated", "final"]:
        block = summary[phase]
        successes = sum(row["successes"] for row in block["by_variant"].values())
        count = sum(row["trace_count"] for row in block["by_variant"].values())
        chats = sum(row["chat"]["physical_attempts"] for row in block["by_variant"].values()) + block["unassigned_non_embedding"]["physical_attempts"]
        embeddings = block["total"]["physical_attempts"] - chats
        lines.append(f"| [{phase}]({phase}-traces.jsonl) | {successes} | {count-successes} | {chats} | {embeddings} | {block['total']['estimated_cost_cny'] or 'unknown'} |")
    lines += ["", "Initial failures exposed misspelled JSON fields, schema confusion, output truncation and missing "
              "inline citations. Follow-ups exposed long-ID transcription errors and a bounded citation repair that "
              "could repeat the same uncited body. The final code uses strict field output, a Markdown-first report "
              "schema, reversible request-local evidence labels for every RAG variant, and deterministic validation. "
              "Unknown labels and unsupported claims are not repaired by guessing. Invalid researcher/writer responses "
              "retain their paid usage even when the draft is rejected. Early missing trace counters remain visible "
              "as unassigned non-embedding requests in the physical ledger; they are not charged to index setup.", "",
              "One embedding timeout during the historical followup phase has unknown usage and charge. Its reservation "
              "remains open; it is not treated as free. Thus that phase's total is unknown. The final generation run's "
              "request usage is complete.", "",
              "## Answer and judge calibration findings", "",
              "The closed-book final answers invented a generator-only cache-bypass requirement, incorrectly preferred "
              "a subclass exception handler, and fabricated deployment latency/benchmark figures. RAG answers avoided "
              "those particular errors, but some still incorrectly claimed that serialization_alias controls editor "
              "initializer signatures. The multi-agent insufficiency answer exposed internal critic commentary and "
              "used two extra retrieval loops without finding measurements. These are development findings to address, "
              "not claims of general superiority.", "",
              "The initial judge returned 9 valid and 11 invalid judgments. Citation status was often confused with "
              "factual correctness, several claim spans changed Markdown/indentation, and long adjacent citation groups "
              "triggered a faulty character-window check. The binding validator now accepts a contiguous group after "
              "its first nearby token and rejects borrowing across paragraphs; entailment still requires source review. "
              "Chinese error-message translations and a documented alias+populate_by_name alternative were wrongly "
              "penalized. Every raw output and Codex decision is preserved; judge errors were never application-quality zeros.", "",
              "Reference caveat: question 019 asks which exception classes to raise/handle, but its key also expects "
              "JSON-detail and an example import alias. The original additional criteria are retained in these scores "
              "and can penalize an otherwise relevant answer. Question 007 admits valid alternatives to its reference "
              "implementation. These pilot findings need reference clarification before full development-set scoring; "
              "the test set has not been tuned against model results.", "",
              "## Reproduction and limits", "",
              f"Generation commit: `{final['generation_manifest']['code_commit']}`; prompt `{final['metadata']['prompt_version']}`. "
              f"Chat snapshot: `{config['model']}` for all four variants and all chat roles, disabled thinking, temperature 0, "
              f"cap {config['max_output_tokens']}, {config['max_retries']} provider retries, at most "
              f"{config['max_critic_loops']} critic loops and {config['max_writer_citation_repairs']} writer citation repair. "
              f"Embedding: `{config['embedding_model']}`, 1024 dimensions; chunks {config['chunk_size']}/{config['chunk_overlap']}, "
              "top-k 5, whole eight-document development pool. Selection, exact serial execution order, seed, source/data "
              "hashes, raw judge responses and corrections are published. No query/answer cache supplies independent repeats.", "",
              "DeepSeek Flash non-thinking scoring uses the official dsh CLI in account mode. System/model labels, "
              "costs and timings are excluded from judge input; answer style can still reveal architecture. AI reference "
              "review, AI scoring and Codex source checks can share biases and do not constitute human validation. "
              "The judge uses a rolling model alias, not a pinned snapshot. Sample size five does not support significance "
              "tests, admissions predictions, or general RAG-quality claims. Public-document pretraining contamination "
              "is possible. Latency is exploratory on a shared development machine.", "",
              f"Judge usage: {summary['judge_usage']['physical_attempts']} physical requests; "
              f"{summary['judge_usage']['input_tokens']} input and {summary['judge_usage']['output_tokens']} output tokens "
              f"({summary['judge_usage']['cached_input_tokens']} cached input tokens, already included in input). "
              "Account pricing is unknown, so judge cost and the combined monetary total are unknown. DashScope costs "
              "are dated list-price estimates, not invoices. Six separate connectivity/constraint probes and earlier "
              "dataset-authoring/review calls are outside this dedicated pilot ledger.", "",
              "Recompute published evidence without API calls:", "", "```powershell",
              ".venv/Scripts/python.exe scripts/verify_dev_pilot.py", "```", "",
              "Regenerate publication from retained local runtime records:", "", "```powershell",
              ".venv/Scripts/python.exe scripts/publish_dev_pilot.py --adjudications docs/evaluation/dev-pilot/adjudications.json",
              "```", "", "Next: complete the full 20-question development set, then freeze the experiment before "
              "running the repeated held-out comparison. This pilot does not freeze or complete that comparison.", ""]
    return "\n".join(lines)


def publish(root: Path, output: Path, corrections: Path | None = None):
    output.mkdir(parents=True, exist_ok=True)
    by_id = {case.id: case for case in load_real_suite(
        ROOT / "benchmarks/real-v1/corpus-manifest.json", ROOT / "benchmarks/real-v1/dev.jsonl", "dev").cases}
    overrides = json.loads(corrections.read_text("utf-8")) if corrections else {}
    corrected = overrides.get("adjudications", {})
    previous = set()
    summaries, publications, phases = {}, [], {}
    for name in ["initial", "diagnostic", "followup", "calibrated", "final"]:
        run_dir = root / name
        report = EvaluationReport.model_validate_json((run_dir / "results.json").read_text("utf-8"))
        manifest = json.loads((run_dir / "run-manifest.json").read_text("utf-8"))
        budget = json.loads((run_dir / "budget-summary.json").read_text("utf-8"))
        all_rows = budget["attempts"]
        rows = [row for row in all_rows if row["id"] not in previous]
        phases.update({row["id"]: name for row in rows})
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
        path.write_text("".join(trace.model_dump_json() + "\n" for trace in report.traces), "utf-8", newline="\n")
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
    receipts = [{
        "id": row["id"], "phase": phases.get(row["id"], "judge" if row["id"] in judge_ids else "unassigned"),
        "created_at": row["created_at"], "provider": row["provider"], "model": row["model"],
        "kind": row["kind"], "role": row["role"], "request_sha256": row["request_hash"],
        "state": row["state"], "outcome": row["outcome"],
        "input_tokens": row["input_tokens"], "output_tokens": row["output_tokens"],
        "cached_input_tokens": row["cached_tokens"], "price": json.loads(row["price_json"]),
        "reserved_estimate_cny": cny(row["reserved"]) if row["price_json"] != "null" else None,
        "estimated_cost_cny": cny(row["charged"]) if row["state"] == "settled" and row["price_json"] != "null" else None,
    } for row in ledger_rows]
    path = output / "request-usage.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in receipts), "utf-8", newline="\n")
    publications.append({"path": path.name, "sha256": sha256(path.read_bytes()).hexdigest()})
    if corrections is not None:
        path = output / "adjudications.json"
        path.write_text(corrections.read_text("utf-8"), "utf-8", newline="\n")
        publications.append({"path": path.name, "sha256": sha256(path.read_bytes()).hexdigest()})
    summaries["limitations"] = ["Development tuning on five questions; no inferential quality claim.",
                                 "AI reference review, AI judge and Codex adjudication; no human scoring.",
                                 "Closed-book model may have seen these public docs in training.",
                                 "Prices are dated estimates, not invoices; account-mode judge price is unknown.",
                                 "Per-variant model_calls includes query embeddings; chat requests are separated here."]
    path = output / "ai-judgments.json"
    path.write_text(json.dumps(judgments, ensure_ascii=False, indent=2) + "\n", "utf-8", newline="\n")
    publications.append({"path": path.name, "sha256": sha256(path.read_bytes()).hexdigest()})
    summaries["publications"] = publications
    (output / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", "utf-8", newline="\n")
    (output / "README.md").write_text(markdown_report(summaries, judgments), "utf-8", newline="\n")
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
