"""Comparable workflow adapters; benchmark gold targets never enter a run."""

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from time import perf_counter
from typing import Protocol

from pydantic import BaseModel, Field

from app.workflow.citations import validate_citation_tokens
from app.agents.common import EvidenceLabels, format_evidence
from app.domain.documents import EvidenceChunk
from app.domain.errors import CitationError, DomainError, WorkflowError
from app.domain.providers import ChatMessage, ChatProvider, ProviderMetadata
from app.domain.runs import ResearchRun
from app.retrieval.contracts import EvidenceRetrieverProtocol
from app.services.documents import DocumentService
from app.services.research import ResearchService

from .dataset import preflight_benchmark_sources, preflight_source_files
from .models import BenchmarkCase, WorkflowVariant
from .trace import EvaluationTrace, EvidenceSnapshot, RetrievalRound


class QueryPlan(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=5)


class GroundedAnswer(BaseModel):
    answer_markdown: str
    cited_evidence_ids: list[str]


class EvaluationWorkflow(Protocol):
    def run(self, case_id: str, question: str, source_files: list[str]) -> EvaluationTrace: ...


def prepare_benchmark_documents(
    cases: Sequence[BenchmarkCase], corpus_dir: Path, document_service: DocumentService,
    *, candidate_source_files: Sequence[str] | None = None,
) -> dict[str, str]:
    """Preflight confined sources, then ingest each resolved file once.

    This setup step reads source_files only. Gold expectations are evaluator-owned.
    Ingestion costs are shared setup costs, excluded from per-workflow traces.
    Distinct benchmark filenames sharing a document ID are ambiguous and rejected.
    """
    sources = preflight_source_files(
        candidate_source_files, corpus_dir, max_upload_file_bytes=document_service.settings.max_upload_file_bytes,
    ) if candidate_source_files is not None else preflight_benchmark_sources(
        cases, corpus_dir, max_upload_file_bytes=document_service.settings.max_upload_file_bytes,
    )
    sources_by_document: dict[str, str] = {}
    mapping: dict[str, str] = {}
    for filename, source in sources.items():
        document = document_service.ingest(filename, source.media_type, source.content)
        if document.status != "ready":
            raise WorkflowError("document_not_ready", "benchmark document is not ready")
        document_id = document.id
        if document_id in sources_by_document:
            raise ValueError("ambiguous benchmark source filenames share one document ID")
        sources_by_document[document_id] = filename
        mapping[filename] = document_id
    return mapping


_SAFE_CODES = frozenset({
    "provider_timeout", "provider_request_failed", "provider_invalid_response",
    "document_not_found", "document_not_ready", "empty_question",
    "no_documents_selected", "invalid_result_limit", "invalid_query_embedding",
    "invalid_stored_embedding", "embedding_dimension_mismatch",
    "run_not_found", "run_failed", "report_not_persisted",
    "workflow_failed", "workflow_stage_failed", "invalid_citation",
    "provider_embedding_failed", "provider_output_truncated",
    "budget_exhausted", "budget_overrun", "budget_input_limit", "budget_unpriced_model",
})


def _error_code(error: Exception) -> str:
    if isinstance(error, CitationError):
        return "invalid_citation"
    if isinstance(error, DomainError) and error.code in _SAFE_CODES:
        return error.code
    return "workflow_failed"


def _safe_identifier(value: str) -> str:
    """Keep model/provider labels, excluding URLs, credentials, and free-form text."""
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}", value)
        or "://" in value or value.casefold().startswith(("sk-", "bearer"))
    ):
        return ""
    return value


def _add_metadata(trace: EvaluationTrace, metadata: ProviderMetadata) -> None:
    trace.prompt_tokens += max(0, metadata.usage.prompt_tokens)
    trace.completion_tokens += max(0, metadata.usage.completion_tokens)
    trace.total_tokens += max(0, metadata.usage.total_tokens)
    trace.retry_count += max(0, metadata.retries)
    trace.model_calls += metadata.attempt_count
    trace.attempt_ids.extend(metadata.attempt_ids)
    trace.usage_complete = trace.usage_complete and metadata.usage_complete
    trace.provider = _safe_identifier(metadata.provider)
    trace.model = _safe_identifier(metadata.model)


def _snapshots(
    evidence: list[EvidenceChunk], sources_by_document: Mapping[str, str],
) -> list[EvidenceSnapshot]:
    snapshots: dict[str, EvidenceSnapshot] = {}
    for item in evidence:
        if item.id not in snapshots:
            snapshots[item.id] = EvidenceSnapshot(
                id=item.id, source_file=sources_by_document[item.document_id], page_number=item.page_number,
                chunk_index=item.chunk_index, text=item.content,
            )
    return list(snapshots.values())


def _validate_citations(trace: EvaluationTrace) -> None:
    inline = validate_citation_tokens(trace.answer, set(trace.cited_evidence_ids), {item.id for item in trace.retrieved_evidence})
    if set(inline) - set(trace.cited_evidence_ids):
        raise CitationError("invalid_citation", "inline citation was not declared")


def _grounded_messages(question: str, evidence: list[EvidenceChunk]) -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content=(
            "Answer using only the supplied evidence. Return a JSON object with "
            "answer_markdown (string) and cited_evidence_ids (array of strings) "
            "containing only supplied Evidence IDs. State limitations "
            "when evidence is absent or conflicting. Cite each evidence-backed statement "
            "in answer_markdown with exact tokens such as [[cite:ref_0001]], without curly braces, using only "
            "the declared cited_evidence_ids. Never invent source details."
        )),
        ChatMessage(role="user", content=(
            f"Question:\n{question}\n\nSupplied evidence:\n{format_evidence(evidence)}"
        )),
    ]


class _Workflow:
    variant: WorkflowVariant

    def run(self, case_id: str, question: str, source_files: list[str]) -> EvaluationTrace:
        started = perf_counter()
        trace = EvaluationTrace.failed(case_id, self.variant, "workflow_failed")
        try:
            self._execute(trace, question, list(source_files))
            trace.status = "success"
            trace.error_code = None
        except Exception as error:
            if self.variant != WorkflowVariant.MULTI_AGENT_RAG and isinstance(getattr(error, "metadata", None), ProviderMetadata):
                _add_metadata(trace, error.metadata)
            trace.status = "failed"
            trace.error_code = _error_code(error)
        trace.latency_ms = max(0, int((perf_counter() - started) * 1000))
        return trace

    def _execute(self, trace: EvaluationTrace, question: str, source_files: list[str]) -> None:
        raise NotImplementedError


class BaselineLlmWorkflow(_Workflow):
    variant = WorkflowVariant.BASELINE_LLM

    def __init__(self, provider: ChatProvider) -> None:
        self.provider = provider

    def _execute(self, trace: EvaluationTrace, question: str, source_files: list[str]) -> None:
        answer, metadata = self.provider.generate([
            ChatMessage(role="system", content="Answer the question directly. Do not invent citations."),
            ChatMessage(role="user", content=question),
        ])
        _add_metadata(trace, metadata)
        trace.answer = answer


class _DocumentWorkflow(_Workflow):
    def __init__(self, document_mapping: Mapping[str, str]) -> None:
        self._document_mapping = dict(document_mapping)
        self._sources_by_document = {
            document_id: filename for filename, document_id in self._document_mapping.items()
        }

    def _document_ids(self, source_files: list[str]) -> list[str]:
        try:
            ids = list(dict.fromkeys(self._document_mapping[filename] for filename in source_files))
        except KeyError:
            raise WorkflowError("document_not_found", "benchmark source has no document mapping") from None
        if not ids:
            raise WorkflowError("no_documents_selected", "at least one benchmark source is required")
        return ids


class LlmRagWorkflow(_DocumentWorkflow):
    variant = WorkflowVariant.LLM_RAG

    def __init__(
        self, provider: ChatProvider, retriever: EvidenceRetrieverProtocol,
        document_mapping: Mapping[str, str], top_k: int = 5, *, max_citation_repairs: int = 0,
    ) -> None:
        super().__init__(document_mapping)
        self.provider = provider
        self.retriever = retriever
        self.top_k = top_k
        if max_citation_repairs not in (0, 1):
            raise ValueError("answer citation repairs must be zero or one")
        self._max_citation_repairs = max_citation_repairs

    def _queries(self, trace: EvaluationTrace, question: str) -> list[str]:
        return []

    def _execute(self, trace: EvaluationTrace, question: str, source_files: list[str]) -> None:
        document_ids = self._document_ids(source_files)
        expansions = self._queries(trace, question)
        batch = self.retriever.retrieve(question, expansions, document_ids, self.top_k)
        for metadata in batch.provider_metrics:
            _add_metadata(trace, metadata)
        trace.retrieved_evidence = _snapshots(batch.evidence, self._sources_by_document)
        trace.retrieval_rounds = [RetrievalRound(
            round_index=0, queries=list(batch.queries), evidence=list(trace.retrieved_evidence),
        )]
        labels = EvidenceLabels(batch.evidence)
        original = _grounded_messages(question, labels.evidence)
        messages = original
        for repair in range(self._max_citation_repairs + 1):
            answer, metadata = self.provider.generate_structured(messages, GroundedAnswer)
            metadata = metadata.model_copy(deep=True)
            metadata.retries += int(repair > 0)
            _add_metadata(trace, metadata)
            trace.answer = labels.text(answer.answer_markdown)
            trace.cited_evidence_ids = [labels.original(identifier) for identifier in answer.cited_evidence_ids]
            try:
                _validate_citations(trace)
                break
            except CitationError:
                if repair == self._max_citation_repairs:
                    raise
                messages = [*original, ChatMessage(role="assistant", content=answer.model_dump_json()),
                            ChatMessage(role="user", content=(
                                "Correct the citation formatting once. Use exact tokens such as [[cite:ref_0001]], "
                                "without curly braces, and only supplied IDs. Every declared ID must appear "
                                "in answer_markdown. Check support; do not insert unrelated citations or invent facts."
                            ))]


class SingleAgentRagWorkflow(LlmRagWorkflow):
    variant = WorkflowVariant.SINGLE_AGENT_RAG

    def _queries(self, trace: EvaluationTrace, question: str) -> list[str]:
        plan, metadata = self.provider.generate_structured([
            ChatMessage(role="system", content=(
                "Expand the question into one to five focused search queries. "
                "Return a JSON object with a queries array containing one to five strings."
            )),
            ChatMessage(role="user", content=question),
        ], QueryPlan)
        _add_metadata(trace, metadata)
        return list(plan.queries)


def _copy_run_metadata(trace: EvaluationTrace, run: ResearchRun) -> None:
    trace.prompt_tokens = max(0, run.prompt_tokens)
    trace.completion_tokens = max(0, run.completion_tokens)
    trace.total_tokens = max(0, run.total_tokens)
    trace.model_calls = max(0, run.model_call_count)
    trace.retry_count = max(0, run.retry_count)
    trace.critic_loops = max(0, run.iteration)
    trace.evidence_sufficient = run.evidence_sufficient
    trace.provider = _safe_identifier(run.provider)
    trace.model = _safe_identifier(run.model)


class MultiAgentRagWorkflow(_DocumentWorkflow):
    variant = WorkflowVariant.MULTI_AGENT_RAG

    def __init__(self, research_service: ResearchService, document_mapping: Mapping[str, str]) -> None:
        super().__init__(document_mapping)
        self.research_service = research_service

    def _execute(self, trace: EvaluationTrace, question: str, source_files: list[str]) -> None:
        service = self.research_service
        run = service.create_run(question, self._document_ids(source_files))
        try:
            service.execute(run.id)
        finally:
            stored = service.get_run(run.id)
            _copy_run_metadata(trace, stored)
            state = service.runs.get_state(run.id)
            metrics = [ProviderMetadata.model_validate(item) for item in state.get("provider_metrics", [])]
            trace.attempt_ids = [identifier for metric in metrics for identifier in metric.attempt_ids]
            trace.usage_complete = all(metric.usage_complete for metric in metrics)
            trace.retrieved_evidence = _snapshots(
                [EvidenceChunk.model_validate(item) for item in state.get("evidence", [])],
                self._sources_by_document,
            )
            trace.retrieval_rounds = [RetrievalRound(
                round_index=index, queries=list(item["queries"]),
                evidence=_snapshots([EvidenceChunk.model_validate(value) for value in item["evidence"]], self._sources_by_document),
            ) for index, item in enumerate(state.get("retrieval_rounds", []))]
        report = service.get_report(run.id)
        if report is None:
            raise WorkflowError("report_not_persisted", "research report was not persisted")
        # Score the authored answer, excluding the renderer's copied source
        # excerpts and bibliography, which can otherwise inflate answer coverage.
        trace.answer = state.get("draft", {}).get("markdown", report.markdown)
        if report.limitations:
            trace.answer += "\n\nLimitations:\n" + "\n".join(report.limitations)
        trace.cited_evidence_ids = [citation.evidence_id for citation in report.citations]
        _validate_citations(trace)
        if stored.status != "completed":
            code = stored.error_code if stored.error_code in _SAFE_CODES else "workflow_failed"
            raise WorkflowError(code, "research run did not complete")
