"""Reproduce pre-test sampling and validate the explicitly unfrozen design."""

from hashlib import sha256
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]


def verify():
    suite = ROOT / "benchmarks/real-v1"
    design = json.loads((suite / "test-preparation.json").read_text("utf-8"))
    summary = json.loads((ROOT / "docs/evaluation/final-dev/summary.json").read_text("utf-8"))
    if design["status"] != "not_frozen_no_test_execution" or design["no_heldout_answers_generated"] is not True:
        raise ValueError("preparation must not assert completed freeze or execution")
    if design["dataset_sha256"] != sha256((suite / "test.jsonl").read_bytes()).hexdigest():
        raise ValueError("test dataset changed")
    if design["reference_review_sha256"] != sha256((suite / "review/final-audit.json").read_bytes()).hexdigest():
        raise ValueError("reference audit changed")
    candidate = design["generation_candidate"]
    if (candidate["configuration"] != summary["metadata"]["configuration"]
            or candidate["code_commit"] != summary["metadata"]["git_commit"]
            or candidate["source_sha256"] != summary["generation_manifest"]["source_sha256"]
            or candidate["prompt_version"] != summary["metadata"]["prompt_version"]):
        raise ValueError("generation candidate differs from final development")
    cases = [json.loads(line) for line in (suite / "test.jsonl").read_text("utf-8").splitlines()]
    matrix = design["matrix"]
    if (len(cases) != matrix["test_questions"] or matrix["planned_units"] != len(cases) * 4 * 3
            or matrix["independent_repeats"] != 3 or len(set(matrix["repeat_order_seeds"])) != 3
            or set(matrix["variants"]) != {"baseline_llm", "llm_rag", "single_agent_rag", "multi_agent_rag"}):
        raise ValueError("test matrix differs")
    sample = design["source_review_sample"]
    strata = {}
    for case in cases:
        strata.setdefault((case["question_type"], case["language"]), []).append(case)
    generator = random.Random(sample["seed"])
    families = {case["family"] for case in cases}
    for draw in range(1, 10001):
        selected = [case for key in sorted(strata)
                    for case in generator.sample(sorted(strata[key], key=lambda c: c["id"]), 2)]
        if {case["family"] for case in selected} == families:
            break
    else:
        raise ValueError("covering review sample not found")
    expected = [{key: case[key] for key in ("id", "question_group", "family", "question_type", "language")}
                for case in sorted(selected, key=lambda c: c["id"])]
    if (sample["metadata"] != expected or sample["accepted_draw"] != draw
            or sample["repeat_index"] != 0 or sample["question_groups"] != 20 or sample["answers"] != 80
            or len({case["question_group"] for case in selected}) != 20):
        raise ValueError("source review sample differs")
    return len(cases), len(selected), len(families)


if __name__ == "__main__":
    cases, sample, families = verify()
    print(f"Preparation verified: {cases} test questions; {sample} prespecified groups; {families} families. Not frozen; no API calls.")
