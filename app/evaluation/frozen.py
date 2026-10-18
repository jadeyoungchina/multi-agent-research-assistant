"""Committed experiment identities and crash-aware, repeat-specific execution.

Validation is read-only. Paid execution requires a committed, matching freeze.
Index requests have a separate ledger, including every rebuilt index on resume.
An unfinished started unit never silently receives another paid generation.
"""

from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import statistics
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

from app.config import Settings
from .budget import BudgetLedger
from .cli import _build_workflows
from .models import WorkflowVariant
from .metrics import percentile
from .real_dataset import load_real_suite
from .reference_review import validate_ai_reference_review
from .reporters import write_reports
from .runner import PROJECT_ROOT, PROMPT_VERSION, ProviderConfiguration, EvaluationReport, evaluate_benchmark, execution_order
from .trace import EvaluationTrace

FREEZE_PATH = PROJECT_ROOT / "benchmarks/real-v1/frozen-experiment.json"
SUITE = PROJECT_ROOT / "benchmarks/real-v1"
CALIBRATION_PATH = PROJECT_ROOT / "docs/evaluation/frozen-judge-calibration.json"


def digest_json(data):
    return sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_json(path, data):
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(data, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def exclusive_run(path):
    """OS-released advisory lock; a crashed process cannot leave a stale lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError("experiment is already running") from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()


def _inputs():
    paths = _git("ls-files").splitlines()
    selected = [p for p in paths if p.startswith(("app/", "scripts/", "tests/", "benchmarks/", ".github/"))
                or p.startswith("requirements") or p in {".gitattributes", ".gitignore", "pyproject.toml", "package.json", "package-lock.json"}]
    calibration = "docs/evaluation/frozen-judge-calibration.json"
    if calibration in paths:
        selected.append(calibration)
    # The freeze binds its prerequisites, not itself (which would be circular).
    selected = [p for p in selected if p != "benchmarks/real-v1/frozen-experiment.json"]
    records = []
    for name in sorted(selected):
        data = (PROJECT_ROOT / name).read_bytes()
        mode = "bytes" if name.startswith("benchmarks/real-v1/") or name == calibration else "lf"
        if mode == "lf":
            data = data.replace(b"\r\n", b"\n")
        records.append({"path": name, "mode": mode, "sha256": sha256(data).hexdigest()})
    return records


def _runtime():
    return {"python_major_minor": list(sys.version_info[:2]),
            "packages": {name: version(name) for name in ("pydantic", "fastapi", "openai", "langgraph", "numpy")}}


def generation_compatibility():
    """Prove calibrated generation code stayed unchanged outside orchestration."""
    candidate = json.loads((SUITE / "test-preparation.json").read_text("utf-8"))["generation_candidate"]["code_commit"]
    old = set(_git("ls-tree", "-r", "--name-only", candidate, "app").splitlines())
    current = set(_git("ls-files", "app").splitlines())
    changed = {"app/evaluation/cli.py", "app/evaluation/development.py"}
    added = {"app/evaluation/frozen.py", "app/evaluation/test_judging.py", "app/evaluation/test_analysis.py"}
    if old - current or current - old - added:
        raise ValueError("unexpected generation source addition or deletion")
    checked = []
    for name in sorted(old - changed):
        previous = subprocess.check_output(["git", "show", f"{candidate}:{name}"], cwd=PROJECT_ROOT)
        actual = (PROJECT_ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        if previous.replace(b"\r\n", b"\n") != actual:
            raise ValueError("calibrated generation code changed; new development validation required")
        checked.append(name)
    return {"candidate_commit": candidate, "unchanged_files": checked,
            "reviewed_orchestration_changes": sorted(changed), "new_evaluation_modules": sorted(added)}


def calibration_identity():
    from .deepseek_cli import _entry_path
    sources = ["app/evaluation/development.py", "app/evaluation/test_judging.py",
               "app/evaluation/deepseek_cli.py", "app/evaluation/dsh_guard.mjs", "scripts/calibrate_frozen_judge.py"]
    return {"kind": "development_only_test_schema_calibration", "human_reviewed": False,
        "packet_summary_sha256": sha256((PROJECT_ROOT / "docs/evaluation/final-dev/summary.json").read_bytes()).hexdigest(),
        "sources": {p: sha256((PROJECT_ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in sources},
        "cli_sha256": sha256(_entry_path().read_bytes()).hexdigest(),
        "max_output_tokens": 8192, "max_format_repairs": 1, "minimum_structurally_valid_answers": 16}


def validate_calibration():
    report = json.loads(CALIBRATION_PATH.read_text("utf-8"))
    if (report["identity"] != calibration_identity() or report["planned_answers"] != 20
            or report["valid_ai_judgments"] < 16 or report["human_reviewed"] is not False
            or report["no_heldout_answers"] is not True or len(report["entries"]) != 20
            or report["valid_ai_judgments"] + report["judge_errors"] != 20
            or len({e["unit"] for e in report["entries"]}) != 20
            or any(not e["case_id"].startswith("REAL-DEV-") for e in report["entries"])):
        raise ValueError("judge calibration is missing, stale or below the prespecified structural gate")
    return {"path": CALIBRATION_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256(CALIBRATION_PATH.read_bytes()).hexdigest(),
            "valid_ai_judgments": report["valid_ai_judgments"], "judge_errors": report["judge_errors"]}


def create_freeze(path=FREEZE_PATH):
    if _git("status", "--porcelain"):
        raise ValueError("commit implementation before creating the freeze")
    path = Path(path)
    if path.exists():
        raise ValueError("freeze already exists; never overwrite an experiment identity")
    from scripts.verify_test_preparation import verify
    verify()
    calibration = validate_calibration()
    preparation = json.loads((SUITE / "test-preparation.json").read_text("utf-8"))
    validate_ai_reference_review(SUITE)
    inputs = _inputs()
    record = {"schema_version": "frozen-real-experiment-v1", "status": "frozen_before_heldout_execution",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "implementation_commit": _git("rev-parse", "HEAD"), "prompt_version": PROMPT_VERSION,
              "runtime": _runtime(), "inputs": inputs, "design": preparation,
              "judgment_version": "test-block-v2", "embedding_batch_size": 10,
              "unit_interruption_policy": "stop; explicit resolution may retain failure, never regenerate the uncertain unit",
              "setup_policy": "separate index ledger; every resumed setup retained",
              "human_scoring_count": 0}
    record["judge_calibration"] = calibration
    record["generation_compatibility"] = generation_compatibility()
    from .deepseek_cli import _entry_path
    record["judge_runtime"] = {"requested_model": "deepseek-flash", "immutable_model_snapshot": None,
        "cli_sha256": sha256(_entry_path().read_bytes()).hexdigest(), "alias_drift_limitation": True}
    record["freeze_id"] = digest_json(record)
    write_json(path, record)
    return record


def validate_freeze(path=FREEZE_PATH, *, require_committed=True):
    path = Path(path).resolve()
    record = json.loads(path.read_text("utf-8"))
    if record["schema_version"] != "frozen-real-experiment-v1" or record["status"] != "frozen_before_heldout_execution":
        raise ValueError("not a frozen experiment")
    if record["freeze_id"] != digest_json({k: v for k, v in record.items() if k != "freeze_id"}):
        raise ValueError("freeze identity differs")
    if record["inputs"] != _inputs() or record["runtime"] != _runtime() or record["prompt_version"] != PROMPT_VERSION:
        raise ValueError("frozen code, data, runtime or prompt differs")
    if (record["design"] != json.loads((SUITE / "test-preparation.json").read_text("utf-8"))
            or record["embedding_batch_size"] != 10):
        raise ValueError("freeze differs from the prespecified design")
    if require_committed:
        if _git("status", "--porcelain"):
            raise ValueError("paid execution requires a clean committed freeze")
        name = path.relative_to(PROJECT_ROOT).as_posix()
        committed = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=PROJECT_ROOT)
        if json.loads(committed) != record:
            raise ValueError("freeze has not been committed")
    return record


def settings_for(record):
    config = ProviderConfiguration.model_validate(record["design"]["generation_candidate"]["configuration"])
    settings = Settings(
        _env_file=".env", chat_provider=config.provider, embedding_provider=config.embedding_provider,
        **{f"{config.provider}_chat_model": config.model, f"{config.embedding_provider}_embedding_model": config.embedding_model},
        provider_timeout_seconds=config.timeout_seconds, provider_max_retries=config.max_retries,
        provider_max_output_tokens=config.max_output_tokens, provider_structured_output=config.structured_output,
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap,
        embedding_batch_size=record["embedding_batch_size"], retrieval_top_k=config.retrieval_top_k,
        retrieval_candidate_k=config.retrieval_candidate_k, retrieval_rrf_k=config.retrieval_rrf_k,
        retrieval_min_similarity=config.retrieval_min_similarity, max_query_expansions=config.max_query_expansions,
        max_revision_iterations=config.max_critic_loops, max_workflow_steps=config.max_workflow_steps,
        max_researcher_citation_repairs=config.max_researcher_citation_repairs,
        max_writer_citation_repairs=config.max_writer_citation_repairs,
        max_upload_file_bytes=20 * 1024 * 1024, max_upload_total_bytes=50 * 1024 * 1024,
    )
    if ProviderConfiguration.from_settings(settings) != config:
        raise ValueError("effective settings differ from freeze")
    return settings


class RepeatJournal:
    def __init__(self, root, identity, ledger):
        self.root, self.ledger = Path(root), ledger
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = self.root / "repeat-manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text("utf-8")) != identity:
                raise ValueError("repeat identity differs")
        else:
            if any(self.root.iterdir()):
                raise ValueError("repeat directory is not new")
            write_json(manifest, identity)
        self.expected = set(identity["execution_order"])
        self.traces = {}
        path = self.root / "traces.jsonl"
        lines = path.read_bytes().splitlines() if path.exists() else []
        valid = []
        for index, line in enumerate(lines):
            try:
                text = line.decode("utf-8")
                trace = EvaluationTrace.model_validate_json(text)
            except (ValueError, UnicodeError):
                if index != len(lines) - 1:
                    raise
                # A torn tail is not a completed unit. Its started record below
                # forces explicit interruption resolution before any new calls.
                append_json(self.root / "torn-traces.jsonl", {"at_utc": datetime.now(timezone.utc).isoformat(),
                    "partial_bytes_hex": line.hex(), "sha256": sha256(line).hexdigest()})
                with path.open("w", encoding="utf-8", newline="\n") as stream:
                    stream.write("".join(x + "\n" for x in valid))
                    stream.flush()
                    os.fsync(stream.fileno())
                break
            key = f"{trace.case_id}/{trace.variant.value}"
            if key not in self.expected or key in self.traces:
                raise ValueError("unknown or duplicate completed unit")
            self.traces[key] = trace
            valid.append(text)
        starts = self.root / "started.jsonl"
        self.started = {}
        for line in starts.read_text("utf-8").splitlines() if starts.exists() else []:
            entry = json.loads(line)
            if entry["unit"] not in self.expected or entry["unit"] in self.started:
                raise ValueError("unknown or duplicate started unit")
            self.started[entry["unit"]] = entry
        if not self.traces.keys() <= self.started.keys():
            raise ValueError("completed unit lacks a durable start record")

    @property
    def pending(self):
        return self.started.keys() - self.traces.keys()

    def begin(self, case, variant):
        key = f"{case}/{variant.value}"
        if key in self.started or self.pending:
            raise ValueError("unfinished or duplicate unit; reconcile before generation")
        entry = {"unit": key, "at_utc": datetime.now(timezone.utc).isoformat(),
                 "prior_attempt_ids": [r["id"] for r in self.ledger.summary()["attempts"]]}
        append_json(self.root / "started.jsonl", entry)
        self.started[key] = entry

    def append(self, trace):
        key = f"{trace.case_id}/{trace.variant.value}"
        if key not in self.started or key in self.traces:
            raise ValueError("unit completion has no unique start")
        append_json(self.root / "traces.jsonl", trace.model_dump(mode="json"))
        self.traces[key] = trace

    def resolve(self, configuration):
        if len(self.pending) > 1:
            raise ValueError("serial run has multiple uncertain units")
        for key in self.pending:
            prior = set(self.started[key]["prior_attempt_ids"])
            rows = [r for r in self.ledger.summary()["attempts"] if r["id"] not in prior]
            case, variant = key.split("/")
            trace = EvaluationTrace.failed(case, WorkflowVariant(variant), "workflow_failed")
            trace.provider, trace.model = configuration.provider, configuration.model
            trace.attempt_ids, trace.model_calls = [r["id"] for r in rows], len(rows)
            trace.prompt_tokens = sum(r["input_tokens"] or 0 for r in rows)
            trace.completion_tokens = sum(r["output_tokens"] or 0 for r in rows)
            trace.total_tokens = trace.prompt_tokens + trace.completion_tokens
            trace.usage_complete = all(r["input_tokens"] is not None and r["output_tokens"] is not None for r in rows)
            append_json(self.root / "resolutions.jsonl", {"unit": key, "action": "retain_interrupted_failure",
                "at_utc": datetime.now(timezone.utc).isoformat(), "attempt_ids": trace.attempt_ids,
                "latency_and_retry_count": "unknown; trace placeholders must not be used as measured zeros"})
            self.append(trace)


class _JournaledWorkflow:
    def __init__(self, workflow, variant, journal):
        self.workflow, self.variant, self.journal = workflow, variant, journal

    def run(self, case_id, question, source_files):
        self.journal.begin(case_id, self.variant)
        return self.workflow.run(case_id, question, source_files)


def _performance(report, directory):
    path = directory / "resolutions.jsonl"
    unknown = {json.loads(line)["unit"] for line in path.read_text("utf-8").splitlines()} if path.exists() else set()
    result = {"unknown_latency_retry_units": sorted(unknown), "by_variant": {}}
    groups = [("all", report.aggregate, report.traces)] + [
        (v.value, report.by_variant[v], [t for t in report.traces if t.variant == v]) for v in report.by_variant]
    for name, metrics, traces in groups:
        measured = [t for t in traces if f"{t.case_id}/{t.variant.value}" not in unknown]
        times = [t.latency_ms for t in measured]
        values = {"latency_retry_denominator": len(measured), "unknown_measurements": len(traces)-len(measured),
            "mean_latency_ms": statistics.mean(times) if times else None,
            "p50_latency_ms": percentile(times, .5) if times else None,
            "p95_latency_ms": percentile(times, .95) if times else None,
            "mean_retry_count": statistics.mean(t.retry_count for t in measured) if measured else None}
        metrics.mean_latency_ms = values["mean_latency_ms"] or 0.0
        metrics.p50_latency_ms = values["p50_latency_ms"] or 0
        metrics.p95_latency_ms = values["p95_latency_ms"] or 0
        metrics.mean_retry_count = values["mean_retry_count"] or 0.0
        metrics.metric_denominators.update(latency_ms=len(measured), retry_count=len(measured))
        result["by_variant"][name] = values
    write_json(directory / "performance.json", result)
    return result


def run_repeat(freeze_path, root, repeat, *, resolve_interrupted=False):
    """Validate the committed freeze, then run one immutable 160-unit repeat."""
    record = validate_freeze(freeze_path)
    matrix = record["design"]["matrix"]
    if type(repeat) is not int or not 0 <= repeat < matrix["independent_repeats"]:
        raise ValueError("repeat is outside the frozen matrix")
    suite = load_real_suite(SUITE / "corpus-manifest.json", SUITE / "test.jsonl", "test", require_design_counts=True)
    validate_ai_reference_review(SUITE)
    variants = [WorkflowVariant(v) for v in matrix["variants"]]
    seed = matrix["repeat_order_seeds"][repeat]
    identity = {"freeze_id": record["freeze_id"], "repeat_index": repeat, "order_seed": seed,
                "execution_order": [f"{c}/{v.value}" for c, v in execution_order(suite.cases, variants, seed)]}
    root = Path(root)
    with exclusive_run(root / ".generation.lock"):
        directory = root / f"repeat-{repeat}"
        # Journal identity must precede ledger creation in a new directory.
        journal = RepeatJournal(directory, identity, None)
        configuration = ProviderConfiguration.model_validate(record["design"]["generation_candidate"]["configuration"])
        if any(t.provider != configuration.provider or t.model != configuration.model for t in journal.traces.values()):
            raise ValueError("returned provider or model differs from the frozen snapshot")
        result_path = directory / "results.json"
        if len(journal.traces) == len(identity["execution_order"]) and result_path.exists():
            report = EvaluationReport.model_validate_json(result_path.read_text("utf-8"))
            if ({f"{t.case_id}/{t.variant.value}": t for t in report.traces} != journal.traces
                    or not report.execution_complete):
                raise ValueError("completed report differs from durable journal")
            from .test_judging import load_repeat
            return load_repeat(record, root, repeat)  # No provider construction or index requests on resume.
        ledger = BudgetLedger(directory / "generation.sqlite")
        ledger.set_enforcement(False)
        journal.ledger = ledger
        if journal.pending:
            if not resolve_interrupted:
                raise ValueError("uncertain paid unit; use explicit interrupted-failure resolution")
            journal.resolve(ProviderConfiguration.model_validate(record["design"]["generation_candidate"]["configuration"]))
        settings = settings_for(record)
        cached = {(t.case_id, t.variant): t for t in journal.traces.values()}
        with TemporaryDirectory(prefix="frozen-research-") as temporary, ExitStack() as resources:
            temporary = Path(temporary)
            settings = settings.model_copy(update={"data_dir": temporary, "upload_dir": temporary / "uploads",
                                                   "database_path": temporary / "evaluation.db"})
            setup_ledger = BudgetLedger(directory / "index.sqlite")
            setup_ledger.set_enforcement(False)
            started = perf_counter()
            prior = {r["id"] for r in setup_ledger.summary()["attempts"]}
            if len(cached) == len(identity["execution_order"]):
                # The final trace was durable but the final report was interrupted.
                workflows = {v: None for v in variants}
            else:
                def capture(item):
                    append_json(directory / "provider-responses.jsonl", item)
                try:
                    workflows = _build_workflows(settings, suite.cases, SUITE / "corpus", variants, resources, ledger,
                        candidate_source_files=suite.candidate_source_files, response_observer=capture,
                        setup_ledger=setup_ledger)
                except BaseException:
                    append_json(directory / "sessions.jsonl", {"id": uuid4().hex, "event": "setup_failed",
                        "at_utc": datetime.now(timezone.utc).isoformat(), "setup_latency_ms": int((perf_counter()-started)*1000),
                        "index_attempt_ids": [r["id"] for r in setup_ledger.summary()["attempts"] if r["id"] not in prior]})
                    write_json(directory / "generation-usage.json", ledger.summary())
                    write_json(directory / "index-usage.json", setup_ledger.summary())
                    raise
                workflows = {v: _JournaledWorkflow(w, v, journal) for v, w in workflows.items()}
            setup_ms = int((perf_counter() - started) * 1000)
            append_json(directory / "sessions.jsonl", {"id": uuid4().hex, "at_utc": datetime.now(timezone.utc).isoformat(),
                "setup_latency_ms": setup_ms, "index_attempt_ids": [r["id"] for r in setup_ledger.summary()["attempts"] if r["id"] not in prior]})
            try:
                def completed(trace):
                    journal.append(trace)
                    if trace.provider != configuration.provider or trace.model != configuration.model:
                        raise ValueError("returned provider or model differs from the frozen snapshot")
                    print(f"Repeat {repeat}: {trace.case_id}/{trace.variant.value}: {trace.status}", flush=True)
                report = evaluate_benchmark(suite.cases, workflows, corpus_dir=SUITE / "corpus", settings=settings,
                    dataset_path=SUITE / "test.jsonl", on_trace=completed, resume_traces=cached,
                    candidate_source_files=suite.candidate_source_files, corpus_manifest_path=SUITE / "corpus-manifest.json",
                    ai_reference_review_path=SUITE / "review/final-audit.json", order_seed=seed, setup_latency_ms=setup_ms)
                if {f"{t.case_id}/{t.variant.value}": t for t in report.traces} != journal.traces:
                    raise ValueError("evaluated report diverges from the durable journal")
                performance = _performance(report, directory)
                write_reports(report, directory)
                if performance["unknown_latency_retry_units"]:
                    path = directory / "report.md"
                    warning = ("Interrupted units have unknown latency/retries. Performance aggregates exclude them; "
                               "zero-denominator numeric fields are schema placeholders. "
                               "See performance.json for authoritative nulls and valid denominators.\n\n")
                    path.write_text(warning + path.read_text("utf-8"), encoding="utf-8", newline="\n")
                return report
            finally:
                write_json(directory / "generation-usage.json", ledger.summary())
                write_json(directory / "index-usage.json", setup_ledger.summary())
