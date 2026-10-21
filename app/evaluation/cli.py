"""Run the synthetic benchmark: python -m app.evaluation.cli."""

import argparse
from hashlib import sha256
import json
from contextlib import ExitStack
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

from app.agents.critic import CriticAgent
from app.agents.planner import PlannerAgent
from app.agents.researcher import ResearcherAgent
from app.agents.retriever import RetrieverAgent
from app.agents.writer import WriterAgent
from app.bootstrap import _close_providers
from app.config import Settings
from app.providers.factory import build_chat_provider, build_embedding_provider
from app.providers.fake import build_demo_fake_provider
from app.retrieval.index import LocalVectorIndex
from app.retrieval.retriever import EvidenceRetriever
from app.services.documents import DocumentService
from app.services.research import ResearchService
from app.storage.database import Database
from app.storage.documents import DocumentRepository
from app.storage.files import LocalDocumentStore
from app.storage.runs import RunRepository
from app.workflow.checkpoints import checkpointed_node
from app.workflow.citations import CitationValidatorNode
from app.workflow.graph import WorkflowDependencies, build_research_graph

from .dataset import load_benchmark_cases, validate_benchmark_corpus
from .budget import BudgetError, BudgetLedger
from .models import BenchmarkCase, WorkflowVariant
from .real_dataset import load_real_suite
from .reference_review import validate_ai_reference_review
from .quality_gate import QualityGateError, enforce_quality_gate
from .reporters import write_reports
from .runner import PROJECT_ROOT, PROMPT_VERSION, ProviderConfiguration, corpus_sha256, evaluate_benchmark, execution_order
from .trace import EvaluationTrace
from .workflows import (
    BaselineLlmWorkflow, EvaluationWorkflow, LlmRagWorkflow, MultiAgentRagWorkflow,
    SingleAgentRagWorkflow, prepare_benchmark_documents,
)


def _variants(value: str) -> list[WorkflowVariant]:
    if value == "all":
        return list(WorkflowVariant)
    try:
        selected = {WorkflowVariant(item.strip()) for item in value.split(",")}
    except ValueError:
        raise argparse.ArgumentTypeError("variants must be all or comma-separated workflow names") from None
    return [variant for variant in WorkflowVariant if variant in selected]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate four workflows on the local synthetic benchmark.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "benchmarks/cases.jsonl")
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "benchmarks/corpus")
    parser.add_argument("--real-manifest", type=Path, help="public-document corpus manifest; selects real-v1 cases")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--allow-unreviewed", action="store_true", help="preliminary dev runs only; never mark AI review as human review")
    parser.add_argument("--allow-ai-reviewed", action="store_true", help="dev pilot using a complete, current AI reference audit; human review remains separate")
    parser.add_argument("--provider", choices=("fake", "dashscope", "openai", "deepseek"), default="fake")
    parser.add_argument("--embedding-provider", choices=("fake", "dashscope", "openai"))
    parser.add_argument("--chat-model")
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--structured-output", choices=("json_object", "json_schema"), default="json_object")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, help="first N cases; paid runs default to 5")
    selection.add_argument("--case-ids", help="explicit comma-separated case IDs, selected before any paid requests")
    parser.add_argument("--order-seed", type=int, help="reproducible serial order with balanced variant positions")
    parser.add_argument("--capture-responses", action="store_true", help="save generated text/schema diagnostics locally; public real corpus only")
    parser.add_argument("--ledger", type=Path, default=Path("artifacts/evaluation/budget/ledger.sqlite"))
    parser.add_argument("--budget-cny", default="100")
    parser.add_argument("--stop-cny", default="90")
    parser.add_argument("--enforce-budget", action="store_true", help="opt in to monetary limits; otherwise record usage only")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--variants", type=_variants, default=list(WorkflowVariant))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation/fake"))
    parser.add_argument("--quality-gates", type=Path, default=PROJECT_ROOT / "benchmarks/quality-gates.json")
    parser.add_argument("--enforce-gate", action="store_true")
    return parser


def _build_workflows(
    settings: Settings, cases: list[BenchmarkCase], corpus: Path,
    variants: list[WorkflowVariant], resources: ExitStack,
    ledger: BudgetLedger | None = None,
    candidate_source_files: list[str] | None = None,
    response_observer=None,
    setup_ledger: BudgetLedger | None = None,
) -> dict[WorkflowVariant, EvaluationWorkflow]:
    chat = build_demo_fake_provider() if settings.chat_provider == "fake" else build_chat_provider(settings, ledger=ledger, role="evaluation")
    if response_observer is not None:
        chat.response_observer = response_observer
    resources.callback(_close_providers, chat)
    embeddings = build_embedding_provider(settings, ledger=ledger, role="corpus_and_query")
    resources.callback(_close_providers, embeddings)
    ingestion_embeddings = embeddings
    if setup_ledger is not None:
        ingestion_embeddings = build_embedding_provider(settings, ledger=setup_ledger, role="corpus_index")
        resources.callback(_close_providers, ingestion_embeddings)
    database = Database(settings.database_path)
    database.initialize()
    documents = DocumentRepository(database)
    runs = RunRepository(database)
    document_service = DocumentService(documents, LocalDocumentStore(settings.upload_dir), ingestion_embeddings, settings)
    mapping = prepare_benchmark_documents(cases, corpus, document_service, candidate_source_files=candidate_source_files)
    retriever = EvidenceRetriever(
        embeddings, LocalVectorIndex(documents), candidate_k=settings.retrieval_candidate_k,
        rrf_k=settings.retrieval_rrf_k, min_similarity=settings.retrieval_min_similarity,
        max_query_expansions=settings.max_query_expansions,
    )
    configuration = ProviderConfiguration.from_settings(settings)
    graph = build_research_graph(WorkflowDependencies(
        planner=PlannerAgent(chat), retriever=RetrieverAgent(retriever, top_k=5),
        researcher=ResearcherAgent(chat, max_citation_repairs=settings.max_researcher_citation_repairs), critic=CriticAgent(chat, max_iterations=settings.max_revision_iterations),
        writer=WriterAgent(chat, max_citation_repairs=settings.max_writer_citation_repairs), citation_validator=CitationValidatorNode(),
        max_iterations=settings.max_revision_iterations,
        node_wrapper=lambda stage, handler: checkpointed_node(stage, handler, runs),
    ))
    research = ResearchService(
        documents, runs, graph, configuration.provider, configuration.model,
        recursion_limit=settings.max_workflow_steps,
    )
    workflows = {
        WorkflowVariant.BASELINE_LLM: BaselineLlmWorkflow(chat),
        WorkflowVariant.LLM_RAG: LlmRagWorkflow(chat, retriever, mapping, top_k=5, max_citation_repairs=settings.max_writer_citation_repairs),
        WorkflowVariant.SINGLE_AGENT_RAG: SingleAgentRagWorkflow(chat, retriever, mapping, top_k=5, max_citation_repairs=settings.max_writer_citation_repairs),
        WorkflowVariant.MULTI_AGENT_RAG: MultiAgentRagWorkflow(research, mapping),
    }
    return {variant: workflows[variant] for variant in variants}


def _journal(args, settings, cases):
    """Resume only the exact recorded configuration; never rerun completed units."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, journal_path = args.output_dir / "run-manifest.json", args.output_dir / "traces.jsonl"
    import subprocess
    manifest = {
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "configuration": ProviderConfiguration.from_settings(settings).model_dump(),
        "dataset_sha256": sha256(args.dataset.read_bytes()).hexdigest(), "corpus_sha256": corpus_sha256(args.corpus),
        "cases": [case.id for case in cases], "variants": [variant.value for variant in args.variants],
        "prompt_version": PROMPT_VERSION, "ledger": str(args.ledger.resolve()),
        "candidate_source_files": getattr(args, "candidate_source_files", None),
        "corpus_manifest_sha256": sha256(args.real_manifest.read_bytes()).hexdigest() if getattr(args, "real_manifest", None) else None,
        "ai_reference_review_sha256": sha256(args.ai_reference_review.read_bytes()).hexdigest() if getattr(args, "ai_reference_review", None) else None,
        "order_seed": args.order_seed,
        "capture_responses": args.capture_responses,
        "execution_order": [f"{case_id}/{variant.value}" for case_id, variant in execution_order(cases, args.variants, args.order_seed)],
    }
    # Changes in tracked or untracked source are part of the resume identity.
    source_digest = sha256()
    for source in sorted((PROJECT_ROOT / "app").rglob("*.py")):
        source_digest.update(source.relative_to(PROJECT_ROOT).as_posix().encode())
        source_digest.update(source.read_bytes())
    manifest["source_sha256"] = source_digest.hexdigest()
    cached = {}
    if args.resume:
        if not manifest_path.is_file() or json.loads(manifest_path.read_text("utf-8")) != manifest:
            raise ValueError("resume configuration mismatch")
        valid_lines = []
        lines = journal_path.read_text("utf-8").splitlines() if journal_path.exists() else []
        for index, line in enumerate(lines):
            try:
                trace = EvaluationTrace.model_validate_json(line)
            except ValueError:
                if index != len(lines) - 1:
                    raise
                break  # repair only a torn final append
            key = (trace.case_id, trace.variant)
            if key in cached:
                raise ValueError("duplicate journal unit")
            # A denied request has not executed and may be retried after reconciliation.
            if not (trace.error_code or "").startswith("budget_"):
                cached[key] = trace
                valid_lines.append(line)
        journal_path.write_text("".join(line + "\n" for line in valid_lines), encoding="utf-8")
    else:
        if manifest_path.exists() or journal_path.exists():
            raise ValueError("run exists; select a new output directory or resume")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    def append(trace):
        with journal_path.open("a", encoding="utf-8") as output:
            output.write(trace.model_dump_json() + "\n")
            output.flush()
        print(f"Completed {trace.case_id}/{trace.variant.value}: {trace.status}; "
              f"calls={trace.model_calls}; latency_ms={trace.latency_ms}; "
              f"error={trace.error_code or 'none'}", flush=True)
    return cached, append


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    ledger = None
    try:
        embedding = args.embedding_provider or ("dashscope" if args.provider == "deepseek" else args.provider)
        paid = args.provider != "fake" or embedding != "fake"
        args.candidate_source_files = None
        args.ai_reference_review = None
        if args.capture_responses and (not paid or not args.real_manifest):
            raise ValueError("response capture requires a paid public-corpus evaluation")
        if args.limit is not None and args.limit <= 0:
            raise ValueError("limit must be positive")
        if args.chat_model and args.provider == "fake":
            raise ValueError("fake model is fixed")
        settings = Settings(
            _env_file=".env" if paid else None,
            chat_provider=args.provider, embedding_provider=embedding,
            provider_max_output_tokens=args.max_output_tokens,
            provider_structured_output=args.structured_output,
            retrieval_top_k=5, max_revision_iterations=2,
            **({f"{args.provider}_chat_model": args.chat_model} if args.chat_model else {}),
            **({"retrieval_min_similarity": -1.0} if args.provider == "fake" else {}),
        )
        try:
            if args.real_manifest:
                if args.enforce_gate:
                    raise ValueError("synthetic release thresholds cannot gate real-document quality")
                if paid and args.split == "test":
                    raise ValueError("paid test execution requires scripts/frozen_experiment.py and a committed freeze")
                args.dataset = args.real_manifest.parent / f"{args.split}.jsonl"
                args.corpus = args.real_manifest.parent / "corpus"
                suite = load_real_suite(
                    args.real_manifest, args.dataset, args.split,
                    require_human_review=paid and not (args.allow_unreviewed or args.allow_ai_reviewed), require_design_counts=True,
                )
                if args.allow_ai_reviewed:
                    validate_ai_reference_review(args.real_manifest.parent)
                    args.ai_reference_review = args.real_manifest.parent / "review/final-audit.json"
                cases = suite.cases
                args.candidate_source_files = suite.candidate_source_files
            else:
                cases = load_benchmark_cases(args.dataset)
                validate_benchmark_corpus(cases, args.corpus, max_upload_file_bytes=settings.max_upload_file_bytes)
            if not cases:
                raise ValueError("benchmark dataset is empty")
            if args.case_ids is not None:
                selected = [item.strip() for item in args.case_ids.split(",")]
                if len(set(selected)) != len(selected) or not set(selected) <= {case.id for case in cases}:
                    raise ValueError("case selection must contain unique IDs from the selected split")
                cases = [case for case in cases if case.id in selected]
        except (OSError, ValueError):
            print("Invalid dataset: check cases, corpus hashes, split, and required human review. Paid test runs require a frozen experiment.", file=sys.stderr)
            return 2
        if args.case_ids is None:
            cases = cases[:args.limit if args.limit is not None else 5 if paid else len(cases)]
        if paid:
            ledger = BudgetLedger(args.ledger, total_cny=args.budget_cny, stop_cny=args.stop_cny)
            ledger.set_enforcement(args.enforce_budget)
        cached, on_trace = _journal(args, settings, cases) if paid or args.resume else ({}, None)
        def capture_response(receipt):
            with (args.output_dir / "provider-responses.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps(receipt, ensure_ascii=False) + "\n")
                output.flush()
        # Disposable database/uploads prevent benchmark runs from touching application data.
        with TemporaryDirectory(prefix="research-evaluation-") as temporary, ExitStack() as resources:
            root = Path(temporary)
            settings = settings.model_copy(update={
                "data_dir": root, "upload_dir": root / "uploads", "database_path": root / "evaluation.db",
            })
            setup_started = perf_counter()
            workflows = _build_workflows(settings, cases, args.corpus, args.variants, resources, ledger,
                                         candidate_source_files=args.candidate_source_files,
                                         response_observer=capture_response if args.capture_responses else None)
            report = evaluate_benchmark(
                cases, workflows, corpus_dir=args.corpus, settings=settings, dataset_path=args.dataset,
                on_trace=on_trace, resume_traces=cached,
                candidate_source_files=args.candidate_source_files, corpus_manifest_path=args.real_manifest,
                ai_reference_review_path=args.ai_reference_review,
                order_seed=args.order_seed,
                setup_latency_ms=max(0, int((perf_counter() - setup_started) * 1000)),
            )
            paths = write_reports(report, args.output_dir)
        print(f"Wrote {len(report.traces)} traces to {len(paths)} output files in {args.output_dir}.")
        if not report.execution_complete:
            print("Budget stopped the run; partial reports and journal were saved.", file=sys.stderr)
            return 5
        if args.enforce_gate:
            enforce_quality_gate(report, args.quality_gates)
            print("Quality gate PASS (multi_agent_rag).")
        return 0
    except QualityGateError as error:
        print(str(error), file=sys.stderr)
        return 3
    except BudgetError as error:
        print(f"Budget stopped execution ({error.code}).", file=sys.stderr)
        return 5
    except Exception as error:
        # Exception text and provider response bodies may contain credentials.
        print(f"Execution error ({type(error).__name__}); evaluation could not complete.", file=sys.stderr)
        return 4
    finally:
        if ledger is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "budget-summary.json").write_text(json.dumps(ledger.summary(), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
