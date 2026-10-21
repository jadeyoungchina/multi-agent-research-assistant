"""Prespecified paired question analysis; failures and missingness stay distinct."""

from hashlib import sha256
import json
from pathlib import Path
import statistics

import numpy as np

from .budget import BudgetLedger, cny
from .frozen import SUITE, validate_freeze, write_json
from .real_dataset import load_real_suite
from .test_judging import TestJudgment, load_repeat, score_test


def _family_interval(cases, differences, replicates, seed):
    families = sorted({c.family for c in cases})
    groups = {f: [differences[c.id] for c in cases if c.family == f and c.id in differences] for f in families}
    generator = np.random.default_rng(seed)
    samples = []
    informative = sum(bool(row) for row in groups.values())
    if informative >= 2:
        for sample in generator.integers(0, len(families), size=(replicates, len(families))):
            rows = [value for i in sample for value in groups[families[int(i)]]]
            if rows:
                samples.append(statistics.mean(rows))
    return {
        "family_bootstrap_95_interval": [float(x) for x in np.quantile(samples, [0.025, 0.975])] if samples else None,
        "bootstrap_replicates": replicates, "valid_bootstrap_replicates": len(samples), "bootstrap_seed": seed,
        "family_count": len(families), "informative_family_count": informative,
        "interval_interpretation": "exploratory; skip and disclose empty draws; no interval below two informative families",
        "by_family": {f: {"complete_pairs": len(row), "mean_difference": statistics.mean(row) if row else None}
                      for f, row in groups.items()},
    }


def primary_comparison(cases, values, *, replicates=10000, seed=20261023):
    """values[(case_id, variant, repeat)] is a score or missing (None)."""
    treatment, control = "multi_agent_rag", "single_agent_rag"
    differences, bounds = {}, []
    for case in cases:
        scores = {v: [values.get((case.id, v, r)) for r in range(3)] for v in (treatment, control)}
        if all(x is not None for row in scores.values() for x in row):
            differences[case.id] = statistics.mean(scores[treatment]) - statistics.mean(scores[control])
        low = {v: statistics.mean(x if x is not None else 0 for x in row) for v, row in scores.items()}
        high = {v: statistics.mean(x if x is not None else 1 for x in row) for v, row in scores.items()}
        bounds.append((low[treatment] - high[control], high[treatment] - low[control]))
    return {"complete_pairs": len(differences), "missing_pairs": len(cases) - len(differences),
        "mean_paired_correctness_difference": statistics.mean(differences.values()) if differences else None,
        "all_question_missing_value_bounds": [statistics.mean(x[i] for x in bounds) for i in (0, 1)],
        **_family_interval(cases, differences, replicates, seed)}


def _cost(rows):
    if any(r["state"] != "settled" or r["price_json"] == "null" for r in rows):
        return None
    return cny(sum(r["charged"] for r in rows))


def _receipts(path):
    if not path.is_file():
        raise ValueError("required request ledger is missing; never replace it with an empty ledger")
    return BudgetLedger(path).summary()["attempts"]


def analyze(freeze_path, root):
    record = validate_freeze(freeze_path)
    root = Path(root)
    cases = load_real_suite(SUITE / "corpus-manifest.json", SUITE / "test.jsonl", "test").cases
    by_case = {c.id: c for c in cases}
    values, costs, execution, judge_missing, phase_costs = {}, {}, [], [], {}
    all_judge_rows, missing_judge_ledgers = [], []
    source_path = root / "adjudications.json"
    source = json.loads(source_path.read_text("utf-8")) if source_path.exists() else {
        "freeze_id": record["freeze_id"], "adjudications": {}}
    if source["freeze_id"] != record["freeze_id"] or source.get("human_reviewed", False) is not False:
        raise ValueError("source adjudications belong to another experiment")
    original_and_final = []
    seen_reviews, seen_failures = set(), set()
    failure_audits = source.get("execution_failure_audits", {})
    for repeat in range(3):
        report = load_repeat(record, root, repeat)
        directory = root / f"repeat-{repeat}"
        receipts = _receipts(directory / "generation.sqlite")
        by_id = {r["id"]: r for r in receipts}
        index_rows = _receipts(directory / "index.sqlite")
        judge_path = root / "judgments" / f"repeat-{repeat}.sqlite"
        judge_rows = _receipts(judge_path) if judge_path.exists() else None
        if judge_rows is None:
            missing_judge_ledgers.append(repeat)
        else:
            all_judge_rows.extend(judge_rows)
        phase_costs[f"repeat-{repeat}"] = {"answer_attempts": len(receipts), "answer_cost_cny": _cost(receipts),
            "index_attempts": len(index_rows), "index_cost_cny": _cost(index_rows),
            "judge_attempts": len(judge_rows) if judge_rows is not None else None,
            "judge_cost_cny": _cost(judge_rows) if judge_rows is not None else None}
        assigned = set()
        for index, trace in enumerate(report.traces):
            execution.append(trace)
            key = (trace.case_id, trace.variant.value, repeat)
            if len(set(trace.attempt_ids)) != len(trace.attempt_ids) or assigned & set(trace.attempt_ids):
                raise ValueError("answer attempts are duplicated")
            assigned.update(trace.attempt_ids)
            rows = [by_id[i] for i in trace.attempt_ids]
            if len(rows) != trace.model_calls:
                raise ValueError("physical answer attempts disagree with trace")
            if (sum(r["input_tokens"] or 0 for r in rows) != trace.prompt_tokens
                    or sum(r["output_tokens"] or 0 for r in rows) != trace.completion_tokens):
                raise ValueError("physical token counts disagree with trace")
            costs[key] = _cost(rows)
            unit = f"r{repeat}-answer-{index + 1:03d}"
            path = root / "judgments" / f"{unit}.json"
            entry = json.loads(path.read_text("utf-8")) if path.exists() else None
            if entry is not None and (entry["identity"]["freeze_id"] != record["freeze_id"]
                    or entry["identity"]["trace_sha256"] != sha256(trace.model_dump_json().encode()).hexdigest()
                    or entry["human_reviewed"] is not False or entry["case_id"] != trace.case_id
                    or entry["variant"] != trace.variant.value or entry["unit"] != unit):
                raise ValueError("judgment identity or attribution differs")
            decision = source["adjudications"].get(unit)
            if decision:
                seen_reviews.add(unit)
                if (entry is None or trace.status != "success" or decision["reviewer_kind"] != "ai"
                        or not decision["reason"] or decision["trace_sha256"] != entry["identity"]["trace_sha256"]
                        or decision["action"] not in {"accepted", "corrected"}):
                    raise ValueError("invalid source adjudication")
                if decision["action"] == "accepted" and decision["judgment"] != entry.get("judgment"):
                    raise ValueError("accepted source review changes the original judgment")
            if trace.status != "success":
                values[key] = 0.0
                audit = failure_audits.get(unit)
                if audit:
                    if (audit["reviewer_kind"] != "ai" or not audit["reason"]
                            or audit["trace_sha256"] != sha256(trace.model_dump_json().encode()).hexdigest()):
                        raise ValueError("failure audit differs from its execution trace")
                    seen_failures.add(unit)
            elif decision or (entry and entry["status"] == "valid_ai_judgment"):
                judgment = TestJudgment.model_validate(decision["judgment"] if decision else entry["judgment"])
                values[key] = score_test(by_case[trace.case_id], trace, judgment).answer_correctness
            else:
                values[key] = None
                judge_missing.append(unit)
            original_and_final.append({"unit": unit, "original": entry, "source_adjudication": decision,
                                       "primary_quality": values[key]})
        if assigned != by_id.keys():
            raise ValueError("unattributed answer requests; reconcile before comparison")
    if seen_reviews != source["adjudications"].keys() or seen_failures != failure_audits.keys():
        raise ValueError("source review names unknown units")
    paired_costs = {}
    from decimal import Decimal
    for case in cases:
        t = [costs.get((case.id, "multi_agent_rag", r)) for r in range(3)]
        c = [costs.get((case.id, "single_agent_rag", r)) for r in range(3)]
        if all(value is not None for value in t + c):
            paired_costs[case.id] = sum(Decimal(v) for v in t) / 3 - sum(Decimal(v) for v in c) / 3
    options = record["design"]["statistics"]
    result = {"schema_version": "frozen-primary-analysis-v1", "freeze_id": record["freeze_id"],
        "planned_units": 480, "executed_units": len(execution), "execution_failures": sum(t.status != "success" for t in execution),
        "successful_answers_missing_judgment": judge_missing, "human_scores": 0,
        "primary": primary_comparison(cases, values, replicates=options["bootstrap_replicates"], seed=options["bootstrap_seed"]),
        "incremental_answer_cost": {"complete_pairs": len(paired_costs), "missing_pairs": len(cases)-len(paired_costs),
            "mean_difference_cny": str(sum(paired_costs.values()) / len(paired_costs)) if paired_costs else None,
            **_family_interval(cases, {k: float(v) for k, v in paired_costs.items()}, options["bootstrap_replicates"], options["bootstrap_seed"])},
        "phase_costs": phase_costs, "known_judge_attempts": len(all_judge_rows),
        "judge_attempts": len(all_judge_rows) if not missing_judge_ledgers else None,
        "judge_cost_cny": _cost(all_judge_rows) if not missing_judge_ledgers else None,
        "missing_judge_ledgers": missing_judge_ledgers,
        "source_reviews": len(seen_reviews), "execution_failure_audits": len(seen_failures), "review_sample_completion": {
            "required_answers": 80, "completed": sum(
                trace.case_id in {c["id"] for c in record["design"]["source_review_sample"]["metadata"]}
                and f"r0-answer-{index+1:03d}" in seen_reviews | seen_failures
                for index, trace in enumerate(load_repeat(record, root, 0).traces))},
        "unit_scores": original_and_final}
    write_json(root / "primary-analysis.json", result)
    return result
