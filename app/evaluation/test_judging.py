"""Frozen held-out judging using calibrated block rules and original test keys."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal

from .budget import BudgetLedger
from .deepseek_cli import _entry_path, run_task
from .development import DevelopmentJudgment, block_judging_prompt, score_blocks
from .frozen import SUITE, digest_json, exclusive_run, validate_freeze, write_json
from .real_dataset import load_real_suite
from .runner import EvaluationReport, corpus_sha256, execution_order
from .models import WorkflowVariant


class TestJudgment(DevelopmentJudgment):
    __test__ = False
    rubric_version: Literal["test-block-v2"]


def score_test(case, trace, judgment):
    if case.split != "test":
        raise ValueError("held-out judge requires an original test case")
    return score_blocks(case, trace, judgment)


def test_prompt(case, trace):
    if case.split != "test":
        raise ValueError("held-out judge requires an original test case")
    return block_judging_prompt(case, trace, SUITE, TestJudgment)


test_prompt.__test__ = False


def decode(text):
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4]
    elif text.startswith("```\n") and text.endswith("\n```"):
        text = text[4:-4]
    return TestJudgment.model_validate_json(text)


def load_repeat(record, root, repeat):
    directory = Path(root) / f"repeat-{repeat}"
    identity = json.loads((directory / "repeat-manifest.json").read_text("utf-8"))
    report = EvaluationReport.model_validate_json((directory / "results.json").read_text("utf-8"))
    actual = [f"{t.case_id}/{t.variant.value}" for t in report.traces]
    matrix = record["design"]["matrix"]
    cases = load_real_suite(SUITE / "corpus-manifest.json", SUITE / "test.jsonl", "test").cases
    expected = [f"{c}/{v.value}" for c, v in execution_order(cases,
        [WorkflowVariant(v) for v in matrix["variants"]], matrix["repeat_order_seeds"][repeat])]
    if (identity["freeze_id"] != record["freeze_id"] or identity["repeat_index"] != repeat
            or identity["execution_order"] != actual or actual != expected or len(set(actual)) != 160
            or report.metadata.synthetic or not report.execution_complete
            or report.metadata.configuration.model_dump() != record["design"]["generation_candidate"]["configuration"]
            or report.metadata.dataset_sha256 != record["design"]["dataset_sha256"]
            or report.metadata.ai_reference_review_sha256 != record["design"]["reference_review_sha256"]
            or report.metadata.order_seed != matrix["repeat_order_seeds"][repeat]
            or report.metadata.execution_order != expected or report.metadata.git_dirty
            or report.metadata.prompt_version != record["prompt_version"]
            or report.metadata.corpus_manifest_sha256 != sha256((SUITE / "corpus-manifest.json").read_bytes()).hexdigest()
            or report.metadata.corpus_sha256 != corpus_sha256(SUITE / "corpus")
            or not report.metadata.references_ai_reviewed or report.metadata.references_human_reviewed
            or any(t.provider != report.metadata.configuration.provider or t.model != report.metadata.configuration.model for t in report.traces)):
        raise ValueError("repeat is incomplete or does not match the freeze")
    return report


def judge_repeats(freeze_path, root, repeats):
    repeats = list(repeats)
    if not repeats or len(set(repeats)) != len(repeats) or any(type(r) is not int or not 0 <= r < 3 for r in repeats):
        raise ValueError("judge repeats must be unique indices from the frozen matrix")
    record = validate_freeze(freeze_path)
    if sha256(_entry_path().read_bytes()).hexdigest() != record["judge_runtime"]["cli_sha256"]:
        raise ValueError("DeepSeek CLI changed after freeze")
    root = Path(root)
    cases = {c.id: c for c in load_real_suite(SUITE / "corpus-manifest.json", SUITE / "test.jsonl", "test").cases}
    jobs = [(repeat, index, trace) for repeat in repeats
            for index, trace in enumerate(load_repeat(record, root, repeat).traces)]
    with exclusive_run(root / ".judging.lock"):
        output = root / "judgments"
        output.mkdir(parents=True, exist_ok=True)
        saved_paths = sorted(output.glob("r[012]-answer-[0-9][0-9][0-9].json"))
        saved_repeats = {int(path.name[1]) for path in saved_paths}
        known_jobs = {f"r{r}-answer-{i+1:03d}": (r, i, t) for r, i, t in jobs}
        for repeat in saved_repeats - set(repeats):
            known_jobs.update({f"r{repeat}-answer-{i+1:03d}": (repeat, i, trace)
                               for i, trace in enumerate(load_repeat(record, root, repeat).traces)})
        cached = {}
        for path in saved_paths:
            if path.stem not in known_jobs:
                raise ValueError("saved judgment names an unknown unit")
            repeat, _, trace = known_jobs[path.stem]
            entry = json.loads(path.read_text("utf-8"))
            expected = {"freeze_id": record["freeze_id"], "repeat_index": repeat,
                        "trace_sha256": sha256(trace.model_dump_json().encode()).hexdigest()}
            if (entry["identity"] != expected or entry["unit"] != path.stem
                    or entry["case_id"] != trace.case_id or entry["variant"] != trace.variant.value
                    or entry["human_reviewed"] is not False
                    or entry["status"] not in {"valid_ai_judgment", "judge_error", "not_scored_execution_failed"}):
                raise ValueError("saved judgment identity differs")
            if entry["status"] == "valid_ai_judgment":
                judgment = TestJudgment.model_validate(entry["judgment"])
                if (entry["score"] != score_test(cases[trace.case_id], trace, judgment).model_dump()
                        or judgment != decode(entry["attempts"][-1]["raw_response"]["text"])):
                    raise ValueError("saved judgment differs from its raw response")
            cached[path.stem] = entry
        ledgers = {repeat: BudgetLedger(output / f"repeat-{repeat}.sqlite") for repeat in repeats}
        for ledger in ledgers.values():
            ledger.set_enforcement(False)

        def judge(job):
            repeat, index, trace = job
            unit = f"r{repeat}-answer-{index + 1:03d}"
            path = output / f"{unit}.json"
            identity = {"freeze_id": record["freeze_id"], "repeat_index": repeat,
                        "trace_sha256": sha256(trace.model_dump_json().encode()).hexdigest()}
            if unit in cached:
                return cached[unit]
            entry = {"unit": unit, "identity": identity, "case_id": trace.case_id, "variant": trace.variant.value,
                     "judge_kind": "ai", "human_reviewed": False, "attempts": [],
                     "status": "not_scored_execution_failed"}
            if trace.status == "success":
                prompt = test_prompt(cases[trace.case_id], trace)
                base = prompt
                for repair in range(2):
                    task = "test-block-v2-" + digest_json({"identity": identity, "prompt": prompt})[:40]
                    try:
                        result = run_task(prompt, task_id=task, ledger=ledgers[repeat], max_output=8192,
                                          timeout=180, max_prompt_bytes=131072)
                        entry["attempts"].append({"raw_response": result, "format_repair": repair})
                        judgment = decode(result["text"])
                        score = score_test(cases[trace.case_id], trace, judgment)
                        entry.update(status="valid_ai_judgment", judgment=judgment.model_dump(), score=score.model_dump())
                        break
                    except Exception as error:
                        entry.update(status="judge_error", error_type=type(error).__name__)
                        if entry["attempts"]:
                            entry["attempts"][-1]["validation_error_type"] = type(error).__name__
                        if repair or not entry["attempts"]:
                            break
                        prompt = (base + "\nPrevious output failed strict validation. Return every point and block once. "
                                  "Uncited factual blocks are uncited; nonfactual blocks N/A. Preserve factual judgments.\nPrevious:\n"
                                  + entry["attempts"][-1]["raw_response"]["text"])
            write_json(path, entry)
            print(f"{unit}: {entry['status']}", flush=True)
            return entry

        try:
            with ThreadPoolExecutor(max_workers=3) as pool:
                entries = list(pool.map(judge, jobs))
            retained = [json.loads(path.read_text("utf-8")) for path in sorted(output.glob("r[012]-answer-[0-9][0-9][0-9].json"))]
            if any(e["identity"]["freeze_id"] != record["freeze_id"] for e in retained):
                raise ValueError("retained judgments belong to another freeze")
            write_json(output / "summary.json", {"freeze_id": record["freeze_id"],
                "repeats": sorted({e["identity"]["repeat_index"] for e in retained}),
                "units": [{k: v for k, v in entry.items() if k != "attempts"} for entry in retained],
                "valid_ai_scores": sum(e["status"] == "valid_ai_judgment" for e in retained),
                "judge_errors": sum(e["status"] == "judge_error" for e in retained), "human_scores": 0})
            return entries
        finally:
            for repeat, ledger in ledgers.items():
                write_json(output / f"usage-repeat-{repeat}.json", ledger.summary())
