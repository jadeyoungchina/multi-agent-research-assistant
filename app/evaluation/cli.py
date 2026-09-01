"""Run the synthetic benchmark: python -m app.evaluation.cli."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

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
from .models import BenchmarkCase, WorkflowVariant
from .quality_gate import QualityGateError, enforce_quality_gate
from .reporters import write_reports
from .runner import PROJECT_ROOT, ProviderConfiguration, evaluate_benchmark
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
    parser.add_argument("--provider", choices=("fake", "dashscope", "openai"), default="fake")
    parser.add_argument("--variants", type=_variants, default=list(WorkflowVariant))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation/fake"))
    parser.add_argument("--quality-gates", type=Path, default=PROJECT_ROOT / "benchmarks/quality-gates.json")
    parser.add_argument("--enforce-gate", action="store_true")
    return parser


def _build_workflows(
    settings: Settings, cases: list[BenchmarkCase], corpus: Path,
    variants: list[WorkflowVariant], resources: ExitStack,
) -> dict[WorkflowVariant, EvaluationWorkflow]:
    chat = build_demo_fake_provider() if settings.chat_provider == "fake" else build_chat_provider(settings)
    resources.callback(_close_providers, chat)
    embeddings = build_embedding_provider(settings)
    resources.callback(_close_providers, embeddings)
    database = Database(settings.database_path)
    database.initialize()
    documents = DocumentRepository(database)
    runs = RunRepository(database)
    document_service = DocumentService(documents, LocalDocumentStore(settings.upload_dir), embeddings, settings)
    mapping = prepare_benchmark_documents(cases, corpus, document_service)
    retriever = EvidenceRetriever(
        embeddings, LocalVectorIndex(documents), candidate_k=settings.retrieval_candidate_k,
        rrf_k=settings.retrieval_rrf_k, min_similarity=settings.retrieval_min_similarity,
        max_query_expansions=settings.max_query_expansions,
    )
    configuration = ProviderConfiguration.from_settings(settings)
    graph = build_research_graph(WorkflowDependencies(
        planner=PlannerAgent(chat), retriever=RetrieverAgent(retriever, top_k=5),
        researcher=ResearcherAgent(chat), critic=CriticAgent(chat, max_iterations=settings.max_revision_iterations),
        writer=WriterAgent(chat), citation_validator=CitationValidatorNode(),
        max_iterations=settings.max_revision_iterations,
        node_wrapper=lambda stage, handler: checkpointed_node(stage, handler, runs),
    ))
    research = ResearchService(
        documents, runs, graph, configuration.provider, configuration.model,
        recursion_limit=settings.max_workflow_steps,
    )
    workflows = {
        WorkflowVariant.BASELINE_LLM: BaselineLlmWorkflow(chat),
        WorkflowVariant.LLM_RAG: LlmRagWorkflow(chat, retriever, mapping, top_k=5),
        WorkflowVariant.SINGLE_AGENT_RAG: SingleAgentRagWorkflow(chat, retriever, mapping, top_k=5),
        WorkflowVariant.MULTI_AGENT_RAG: MultiAgentRagWorkflow(research, mapping),
    }
    return {variant: workflows[variant] for variant in variants}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        cases = load_benchmark_cases(args.dataset)
        if not cases:
            raise ValueError("benchmark dataset is empty")
        validate_benchmark_corpus(cases, args.corpus)
    except (OSError, ValueError):
        print("Invalid dataset: check JSONL cases and corpus evidence files.", file=sys.stderr)
        return 2
    try:
        # Disposable database/uploads prevent benchmark runs from touching application data.
        with TemporaryDirectory(prefix="research-evaluation-") as temporary, ExitStack() as resources:
            root = Path(temporary)
            settings = Settings(
                _env_file=None if args.provider == "fake" else ".env",
                chat_provider=args.provider, embedding_provider=args.provider,
                data_dir=root, upload_dir=root / "uploads", database_path=root / "evaluation.db",
                retrieval_top_k=5, max_revision_iterations=2,
                **({"retrieval_min_similarity": -1.0} if args.provider == "fake" else {}),
            )
            workflows = _build_workflows(settings, cases, args.corpus, args.variants, resources)
            report = evaluate_benchmark(
                cases, workflows, corpus_dir=args.corpus, settings=settings, dataset_path=args.dataset,
            )
            paths = write_reports(report, args.output_dir)
        print(f"Wrote {len(report.traces)} traces to {len(paths)} output files in {args.output_dir}.")
        if args.enforce_gate:
            enforce_quality_gate(report, args.quality_gates)
            print("Quality gate PASS (multi_agent_rag).")
        return 0
    except QualityGateError as error:
        print(str(error), file=sys.stderr)
        return 3
    except Exception as error:
        # Exception text and provider response bodies may contain credentials.
        print(f"Execution error ({type(error).__name__}); evaluation could not complete.", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
