"""Publish source-reviewed results without replacing the original initial packet."""
from hashlib import sha256
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from app.evaluation.budget import BudgetLedger
from app.evaluation.frozen import validate_freeze, write_json
from app.evaluation.test_judging import TestJudgment, load_repeat, score_test
from app.evaluation.real_dataset import load_real_suite
from scripts.publish_dev_pilot import usage
from scripts.publish_full_dev import citation_diagnostic
from publish_checkpoint import receipts
import source_review as review

RUN, OUT = review.RUN, ROOT / "docs/evaluation/heldout-test"


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def ledger_rows(path):
    if not path.is_file():
        raise ValueError("missing ledger; do not create a zero replacement")
    return BudgetLedger(path).summary()["attempts"]


def publish(*, require_committed=True):
    # Offline packet construction may run before a documentation commit. This
    # never changes the clean-tree guard on paid generation or source review.
    record = validate_freeze(require_committed=require_committed)
    analysis = json.loads((RUN / "primary-analysis.json").read_text("utf-8"))
    adjudications = json.loads((RUN / "adjudications.json").read_text("utf-8"))
    if (analysis["freeze_id"] != record["freeze_id"] or analysis["executed_units"] != 480
            or analysis["review_sample_completion"] != {"required_answers": 80, "completed": 80}
            or analysis["human_scores"] != 0):
        raise ValueError("review sample incomplete or identity differs")
    selected = review.jobs(record)
    identity = review.build_manifest(record, selected)
    if json.loads((review.OUTPUT / "manifest.json").read_text("utf-8"))["identity"] != identity:
        raise ValueError("source-review manifest changed")
    audits = [json.loads((review.OUTPUT / f"{j['unit']}.json").read_text("utf-8")) for j in selected]
    for job, entry in zip(selected, audits):
        target = next(t for t in identity["targets"] if t["unit"] == job["unit"])
        if entry["target"] != target or entry["review_identity"] != review.digest_json(identity):
            raise ValueError("source review input identity differs")
    cases = {c.id: c for c in load_real_suite(review.SUITE / "corpus-manifest.json", review.SUITE / "test.jsonl", "test").cases}
    scored = {row["unit"]: row for row in analysis["unit_scores"]}
    by_variant, all_receipts, all_rows, phase_usage = {}, [], [], {}
    for repeat in range(3):
        report = load_repeat(record, RUN, repeat)
        for phase, path in (("answer_generation", RUN / f"repeat-{repeat}/generation.sqlite"),
                            ("index_setup", RUN / f"repeat-{repeat}/index.sqlite"),
                            ("blinded_judging", RUN / f"judgments/repeat-{repeat}.sqlite")):
            rows = ledger_rows(path)
            all_rows.extend(rows)
            phase_usage[f"repeat-{repeat}/{phase}"] = usage(rows)
            all_receipts.extend(receipts(path, phase, repeat))
        for index, trace in enumerate(report.traces, 1):
            unit = f"r{repeat}-answer-{index:03d}"
            item = scored[unit]
            row = by_variant.setdefault(trace.variant.value, {"traces": [], "scores": [], "qualities": {}, "diagnostics": []})
            row["traces"].append(trace)
            row["qualities"].setdefault(trace.case_id, []).append(item["primary_quality"])
            row["diagnostics"].append(citation_diagnostic(trace))
            if trace.status == "success":
                judgment = (item["source_adjudication"] or {}).get("judgment") or (item["original"] or {}).get("judgment")
                if item["primary_quality"] is not None:
                    if judgment is None:
                        raise ValueError("known successful quality lacks a judgment")
                    row["scores"].append(score_test(cases[trace.case_id], trace, TestJudgment.model_validate(judgment)).model_dump())
    source_rows = ledger_rows(review.OUTPUT / "ledger.sqlite")
    all_rows.extend(source_rows)
    phase_usage["source_review"] = usage(source_rows)
    all_receipts.extend(receipts(review.OUTPUT / "ledger.sqlite", "source_review", None))
    if len({r["id"] for r in all_rows}) != len(all_rows):
        raise ValueError("phase ledgers contain duplicate physical attempts")
    rows_by_id = {r["id"]: r for r in all_rows}
    table = {}
    for variant, row in by_variant.items():
        times = [t.latency_ms / 1000 for t in row["traces"] if t.status == "success"]
        groups = row["qualities"]
        complete = [statistics.mean(values) for values in groups.values() if len(values) == 3 and all(v is not None for v in values)]
        identifiers = [i for t in row["traces"] for i in t.attempt_ids]
        metrics = {name: {"mean": mean(s[name] for s in row["scores"]), "denominator": sum(s[name] is not None for s in row["scores"])}
                   for name in ("answer_correctness", "answer_key_point_coverage", "citation_support_rate", "citation_completeness",
                                "correct_abstention", "false_abstention", "unsupported_assertion")}
        structural = [d for d in row["diagnostics"] if d["applicable"]]
        table[variant] = {"planned_executions": 120, "successful_executions": len(times), "execution_failures": 120-len(times),
            "known_primary_units": sum(v is not None for values in groups.values() for v in values),
            "fully_scored_question_groups": len(complete), "mean_question_primary_quality": mean(complete),
            "valid_successful_semantic_scores": len(row["scores"]), "conditional_semantic_metrics": metrics,
            "median_success_latency_seconds": statistics.median(times) if times else None, "latency_denominator": len(times),
            "answer_usage": usage([rows_by_id[i] for i in identifiers]),
            "strict_citation_structure_passes": sum(d["passed"] for d in structural), "structure_denominator": len(structural)}
    initial = json.loads((OUT / "initial-analysis.json").read_text("utf-8"))
    raw_counts = {status: sum(e["status"] == status for e in audits) for status in {e["status"] for e in audits}}
    summary = {"schema_version": "source-reviewed-heldout-publication-v1", "freeze_id": record["freeze_id"],
        "planned_units": 480, "executed_units": 480, "successful_executions": 480-analysis["execution_failures"],
        "execution_failures": analysis["execution_failures"], "missing_successful_judgments": analysis["successful_answers_missing_judgment"],
        "human_reference_reviews": 0, "human_answer_scores": 0, "by_variant": table, "primary": analysis["primary"],
        "initial_primary": initial["primary"], "incremental_answer_cost": analysis["incremental_answer_cost"],
        "source_review_sample": analysis["review_sample_completion"], "source_review_targets": len(selected),
        "source_review_extras": sum(not j["in_sample"] for j in selected), "raw_source_review_statuses": raw_counts,
        "source_adjudications": len(adjudications["adjudications"]), "source_failure_audits": len(adjudications["execution_failure_audits"]),
        "accepted": sum(d["action"] == "accepted" for d in adjudications["adjudications"].values()),
        "corrected": sum(d["action"] == "corrected" for d in adjudications["adjudications"].values()),
        "codex_followups": sum(d["reviewer"] == "Codex" for d in list(adjudications["adjudications"].values())+list(adjudications["execution_failure_audits"].values())),
        "phase_usage": phase_usage, "all_recorded_usage": usage(all_rows),
        "same_judge_and_source_reviewer_alias": True, "independent_human_validation": False,
        "limitations": ["AI-authored references and AI scoring; no human validation", "same DeepSeek alias can induce correlated errors",
                        "only eight document families; exploratory family-bootstrap interval", "public docs may be in model pretraining",
                        "serial runs on one machine; latency is exploratory", "unknown API usage/charges stay unknown; estimates are not invoices"]}
    write_json(OUT / "reviewed-analysis.json", analysis)
    write_json(OUT / "source-adjudications.json", adjudications)
    write_json(OUT / "source-review-records.json", {"manifest": json.loads((review.OUTPUT / "manifest.json").read_text("utf-8")), "records": audits})
    (OUT / "all-request-usage.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in all_receipts), encoding="utf-8", newline="\n")
    names = ("reviewed-analysis.json", "source-adjudications.json", "source-review-records.json", "all-request-usage.jsonl")
    summary["publications"] = [{"path": n, "sha256": sha256((OUT/n).read_bytes()).hexdigest()} for n in names]
    write_json(OUT / "reviewed-summary.json", summary)
    fmt = lambda x: "N/A" if x is None else f"{x:.3f}"
    p = analysis["primary"]
    lines = ["# Frozen held-out results with AI source review", "",
        f"**480/480 executions recorded: {summary['successful_executions']} successful, {summary['execution_failures']} failed.** "
        f"Prespecified source review: **80/80**, plus {summary['source_review_extras']} additional error/failure reviews. Human review/scoring: **0**.", "",
        "These are exploratory, AI-assisted results on 40 documentation questions across eight document families, with three fresh repeats. "
        "Source checks use the same DeepSeek alias in separate contexts, with explicit Codex follow-ups; they are not independent human validation.", "",
        "[Frozen protocol](../frozen-experiment.md) · [Initial analysis before source review](INITIAL-ANALYSIS.md) · [Audit method](SOURCE-REVIEW-METHOD.md) · "
        "[Recomputed analysis](reviewed-analysis.json) · [Summary](reviewed-summary.json) · [Trace-bound decisions](source-adjudications.json) · "
        "[Raw source reviews and literal proofs](source-review-records.json) · [All safe receipts](all-request-usage.jsonl)", "",
        "| Workflow | Successful / 120 | Known three-repeat questions / 40 | Primary quality | Conditional citation support | Conditional citation completeness | Median successful seconds | Answer attempts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for variant in record['design']['matrix']['variants']:
        row = table[variant]
        m = row["conditional_semantic_metrics"]
        lines.append(f"| {variant} | {row['successful_executions']} | {row['fully_scored_question_groups']} | {fmt(row['mean_question_primary_quality'])} | "
                     f"{fmt(m['citation_support_rate']['mean'])} | {fmt(m['citation_completeness']['mean'])} | {fmt(row['median_success_latency_seconds'])} | {row['answer_usage']['physical_attempts']} |")
    ratio = table['multi_agent_rag']['median_success_latency_seconds'] / table['single_agent_rag']['median_success_latency_seconds']
    lines += ["", f"Plain RAG has the highest observed primary quality in this small question set ({fmt(table['llm_rag']['mean_question_primary_quality'])}); "
        f"multi-agent RAG is {ratio:.1f} times slower than single-agent RAG by median successful latency. The prespecified primary comparison is multi-agent versus single-agent, not a claim that multi-agent beats every baseline.", "",
        "Primary quality averages the three repeats within a question, including executed-failure zeros. A question with a missing successful-answer judgment is excluded and disclosed. "
        "Citation metrics condition on applicable, successfully scored answers; baseline citations are N/A. Exact denominators and missingness are in the summary. Structure validity is separate from entailment.", "",
        f"The prespecified multi-agent minus single-agent difference is **{fmt(p['mean_paired_correctness_difference'])}**, with {p['complete_pairs']}/40 complete question pairs "
        f"and exploratory family-bootstrap 95% interval **{p['family_bootstrap_95_interval']}**. Whole-matrix missing-value bounds: {p['all_question_missing_value_bounds']}.", "",
        "The primary comparison, sample, seeds and eight-family bootstrap were fixed before generation. The initial result and every original judge error remain available. "
        "An interval containing zero does not establish a reliable improvement; a positive estimate on this dataset alone does not establish general superiority.", "",
        f"Source decisions: {summary['accepted']} accepted, {summary['corrected']} corrected; {summary['codex_followups']} explicit Codex follow-ups. "
        f"Successful answers still missing judgments: {len(summary['missing_successful_judgments'])}. Original review errors and any bounded format repairs remain in the raw records.", "",
        f"All phases retain **{summary['all_recorded_usage']['physical_attempts']} physical requests**. Index construction, answer generation, original judging and source review are separate in the summary. "
        "Unknown usage and unverified account prices remain null. Shared index setup and judge/review costs are excluded from the paired incremental answer-cost estimate; dated API estimates are not invoices.", "",
        "Generation packets retain exact answers/evidence: [repeat 0](repeat-0/summary.json), [repeat 1](repeat-1/summary.json), [repeat 2](repeat-2/summary.json). "
        "Failed executions are preserved with safe diagnostics. This completes the delegated AI experiment stage; independent human checking and broader benchmarks remain separate work.", ""]
    lines += ["Recompute identities, original-judge preservation, all scores, request costs and the frozen paired statistic without paid calls:", "", "```powershell", ".venv/Scripts/python.exe docs/evaluation/heldout-test/tools/verify_reviewed.py", "```", ""]
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    path = OUT / "README.md"
    text = path.read_text("utf-8")
    banner = "**Current results:** [source-reviewed held-out report](RESULTS.md). The dated generation snapshot below and initial analysis remain historical records.\n\n"
    if banner not in text:
        path.write_text(banner+text, encoding="utf-8", newline="\n")
    print(json.dumps({k: summary[k] for k in ("successful_executions", "execution_failures", "source_review_sample", "source_adjudications", "corrected", "primary")}, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    try:
        publish()
    except Exception as error:
        print(f"Reviewed publication stopped ({type(error).__name__}); original packets remain intact.", file=sys.stderr)
        raise SystemExit(2)
