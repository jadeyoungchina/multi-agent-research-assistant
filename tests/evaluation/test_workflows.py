from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import httpx
from openai import BadRequestError
import pytest

from app.agents.critic import CriticAgent
from app.agents.planner import PlannerAgent
from app.agents.researcher import ResearcherAgent
from app.agents.retriever import RetrieverAgent
from app.agents.writer import WriterAgent
from app.config import Settings
from app.domain.documents import EvidenceChunk
from app.domain.errors import ProviderError
from app.domain.providers import ProviderMetadata, TokenUsage
from app.domain.research import Critique, DraftReport, Finding, ReportFinding, ResearchPlan, ResearchSynthesis
from app.evaluation.metrics import score_case
from app.evaluation.models import BenchmarkCase, WorkflowVariant
from app.evaluation.workflows import (
    BaselineLlmWorkflow,
    GroundedAnswer,
    LlmRagWorkflow,
    MultiAgentRagWorkflow,
    QueryPlan,
    SingleAgentRagWorkflow,
    prepare_benchmark_documents,
)
from app.providers.fake import DeterministicEmbeddingProvider, FakeChatProvider
from app.providers.openai_compatible import OpenAICompatibleChatProvider
from app.retrieval.contracts import RetrievalBatch
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


def chat_metadata():
    return ProviderMetadata(
        provider="fake", model="fake-chat", latency_ms=999999, retries=1,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=3, total_tokens=13),
    )


def embedding_metadata():
    return ProviderMetadata(
        provider="fake", model="fake-embedding", latency_ms=999999, retries=2,
        usage=TokenUsage(prompt_tokens=4, total_tokens=4),
    )


class RecordingChat(FakeChatProvider):
    def __init__(self, *, text_responses=(), structured_responses=(), failure=None):
        super().__init__(text_responses, structured_responses)
        self.calls = []
        self.failure = failure

    def generate(self, messages):
        self.calls.append((None, deepcopy(messages)))
        if self.failure is not None:
            raise self.failure
        response, _ = super().generate(messages)
        return response, chat_metadata()

    def generate_structured(self, messages, schema):
        self.calls.append((schema, deepcopy(messages)))
        if self.failure is not None:
            raise self.failure
        response, _ = super().generate_structured(messages, schema)
        return response, chat_metadata()


class RecordingRetriever:
    def __init__(self, evidence):
        self.evidence = evidence
        self.calls = []

    def retrieve(self, question, expansions, document_ids, top_k):
        self.calls.append((question, list(expansions), list(document_ids), top_k))
        return RetrievalBatch(evidence=self.evidence, provider_metrics=[embedding_metadata()])


class CountingDocuments(DocumentService):
    def __init__(self, *args):
        super().__init__(*args)
        self.ingestions = []

    def ingest(self, filename, media_type, content):
        self.ingestions.append((filename, media_type, content))
        return super().ingest(filename, media_type, content)


@pytest.fixture
def case():
    return BenchmarkCase(
        id="BENCH-001", question="What is the solar capacity?",
        source_files=["solar-storage.md"],
        expected_evidence=[{"source_file": "solar-storage.md", "contains": "PRIVATE-GOLD-EVIDENCE"}],
        answer_key_points=["PRIVATE-GOLD-ANSWER"],
    )


@pytest.fixture
def environment(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "solar-storage.md").write_text("The solar capacity is 12 MW.", encoding="utf-8")
    database = Database(tmp_path / "test.db")
    database.initialize()
    documents = DocumentRepository(database)
    service = CountingDocuments(
        documents, LocalDocumentStore(tmp_path / "uploads"), DeterministicEmbeddingProvider(),
        Settings(_env_file=None, chunk_size=100, chunk_overlap=10),
    )
    document = service.ingest("solar-storage.md", "text/markdown", b"The solar capacity is 12 MW.")
    evidence = documents.list_chunks([document.id])[0][0]
    service.ingestions.clear()
    return SimpleNamespace(
        corpus=corpus, documents=documents, document_service=service,
        runs=RunRepository(database), evidence=evidence,
        mapping={"solar-storage.md": document.id},
    )


def multi_service(environment, *, insufficient=False, invalid_citation=False):
    evidence_id = environment.evidence.id
    plan = ResearchPlan(
        objective="Find capacity", subquestions=["Solar capacity?"],
        search_queries=["capacity"], completion_criteria=["12 MW"],
    )
    synthesis = ResearchSynthesis(findings=[Finding(
        claim="12 MW", supporting_evidence_ids=[evidence_id], confidence="high",
    )])
    critique = Critique(sufficient=not insufficient, reason="Capacity evidence", follow_up_queries=["capacity"])
    draft = DraftReport(
        title="Capacity", summary="12 MW", findings=[ReportFinding(
            heading="Capacity", narrative="12 MW", evidence_ids=[evidence_id],
        )], markdown=f"12 MW [[cite:{'unknown' if invalid_citation else evidence_id}]]",
    )
    rounds = [synthesis, critique] * (3 if insufficient else 1)
    provider = RecordingChat(structured_responses=[plan, *rounds, draft])
    retriever = RecordingRetriever([environment.evidence])
    graph = build_research_graph(WorkflowDependencies(
        planner=PlannerAgent(provider), retriever=RetrieverAgent(retriever, top_k=5),
        researcher=ResearcherAgent(provider), critic=CriticAgent(provider),
        writer=WriterAgent(provider), citation_validator=CitationValidatorNode(),
        node_wrapper=lambda stage, handler: checkpointed_node(stage, handler, environment.runs),
    ))
    return ResearchService(environment.documents, environment.runs, graph, "fake", "fake-chat"), provider, retriever


@pytest.fixture
def workflow_factory(environment):
    def factory(variant):
        answer = GroundedAnswer(answer_markdown="The solar capacity is 12 MW.", cited_evidence_ids=[environment.evidence.id])
        if variant == WorkflowVariant.BASELINE_LLM:
            provider = RecordingChat(text_responses=["The solar capacity is 12 MW."])
            return BaselineLlmWorkflow(provider), provider, None
        if variant == WorkflowVariant.MULTI_AGENT_RAG:
            service, provider, retriever = multi_service(environment)
            return MultiAgentRagWorkflow(service, environment.mapping), provider, retriever
        responses = [answer] if variant == WorkflowVariant.LLM_RAG else [QueryPlan(queries=["capacity"]), answer]
        provider = RecordingChat(structured_responses=responses)
        retriever = RecordingRetriever([environment.evidence])
        cls = LlmRagWorkflow if variant == WorkflowVariant.LLM_RAG else SingleAgentRagWorkflow
        return cls(provider, retriever, environment.mapping), provider, retriever
    return factory


def test_all_variants_return_normalized_traces(workflow_factory, case):
    traces = [workflow_factory(variant)[0].run(case.id, case.question, case.source_files) for variant in WorkflowVariant]
    assert [trace.variant for trace in traces] == list(WorkflowVariant)
    assert all(trace.case_id == case.id and trace.status == "success" for trace in traces)
    assert all("12 MW" in trace.answer for trace in traces)
    assert traces[0].retrieved_evidence == traces[0].cited_evidence_ids == []
    assert [trace.model_calls for trace in traces] == [1, 2, 3, 5]
    assert all(trace.provider == "fake" and trace.model == "fake-chat" for trace in traces)
    assert all(0 <= trace.latency_ms < 999999 for trace in traces)
    assert traces[3].critic_loops <= 2


@pytest.mark.parametrize("variant", list(WorkflowVariant))
def test_gold_never_crosses_workflow_boundary(variant, workflow_factory, case):
    workflow, provider, retriever = workflow_factory(variant)
    trace = workflow.run(case.id, case.question, case.source_files)
    assert trace.status == "success"
    request_text = "\n".join(message.content for _, messages in provider.calls for message in messages)
    assert "PRIVATE-GOLD" not in request_text
    assert case.question in request_text
    assert "expected_evidence" not in request_text and "answer_key_points" not in request_text
    if retriever is not None:
        assert all("PRIVATE-GOLD" not in str(call) for call in retriever.calls)


def test_corpus_sources_are_ingested_once_and_mapping_contains_real_ids(environment, case):
    cases = [case, case.model_copy(update={"id": "BENCH-002", "source_files": ["solar-storage.md", "solar-storage.md"]})]
    mapping = prepare_benchmark_documents(cases, environment.corpus, environment.document_service)
    assert mapping == environment.mapping
    assert environment.document_service.ingestions == [("solar-storage.md", "text/markdown", b"The solar capacity is 12 MW.")]
    assert environment.documents.get_document(mapping["solar-storage.md"]).status == "ready"


@pytest.mark.parametrize("source", ["../outside.md", "sub/../../outside.md", "missing.md", "bad.exe"])
def test_prepare_rejects_invalid_source_before_ingesting_any_document(environment, case, source):
    (environment.corpus.parent / "outside.md").write_text("outside", encoding="utf-8")
    (environment.corpus / "bad.exe").write_bytes(b"bad")
    invalid = case.model_copy(update={"source_files": ["solar-storage.md", source]})
    with pytest.raises(ValueError):
        prepare_benchmark_documents([invalid], environment.corpus, environment.document_service)
    assert environment.document_service.ingestions == []


def test_prepare_rejects_absolute_source(environment, case):
    invalid = case.model_copy(update={"source_files": [str(environment.corpus / "solar-storage.md")]})
    with pytest.raises(ValueError):
        prepare_benchmark_documents([invalid], environment.corpus, environment.document_service)
    assert environment.document_service.ingestions == []


def test_prepare_rejects_oversized_source_before_ingestion(environment, case):
    environment.document_service.settings.max_upload_file_bytes = 8
    with pytest.raises(ValueError, match="size limit"):
        prepare_benchmark_documents([case], environment.corpus, environment.document_service)
    assert environment.document_service.ingestions == []


def test_llm_rag_retrieves_only_original_question_and_counts_embedding_metadata(workflow_factory, case, environment):
    workflow, provider, retriever = workflow_factory(WorkflowVariant.LLM_RAG)
    trace = workflow.run(case.id, case.question, case.source_files)
    assert retriever.calls == [(case.question, [], [environment.mapping["solar-storage.md"]], 5)]
    assert [schema for schema, _ in provider.calls] == [GroundedAnswer]
    assert (trace.prompt_tokens, trace.completion_tokens, trace.total_tokens, trace.retry_count) == (14, 3, 17, 3)
    assert trace.cited_evidence_ids == [environment.evidence.id]
    assert trace.retrieved_evidence[0].source_file == "solar-storage.md"
    assert trace.retrieved_evidence[0].text == "The solar capacity is 12 MW."
    assert environment.evidence.id in provider.calls[0][1][-1].content


def test_single_agent_expands_then_fuses_and_answers_once(environment, case):
    class RecordingEmbedding(DeterministicEmbeddingProvider):
        def __init__(self):
            super().__init__()
            self.queries = []

        def embed_query(self, text):
            self.queries.append(text)
            vector, _ = super().embed_query(text)
            return vector, embedding_metadata()

    embedding = RecordingEmbedding()
    retriever = EvidenceRetriever(embedding, LocalVectorIndex(environment.documents), min_similarity=-1)
    provider = RecordingChat(structured_responses=[
        QueryPlan(queries=["solar output", "capacity", "solar output"]),
        GroundedAnswer(answer_markdown="12 MW", cited_evidence_ids=[environment.evidence.id]),
    ])
    workflow = SingleAgentRagWorkflow(provider, retriever, environment.mapping)
    trace = workflow.run(case.id, case.question, case.source_files)
    assert trace.status == "success"
    assert embedding.queries == [case.question, "solar output", "capacity"]
    assert [schema for schema, _ in provider.calls] == [QueryPlan, GroundedAnswer]
    assert len(trace.retrieved_evidence) == 1
    assert trace.model_calls == 5
    assert (trace.prompt_tokens, trace.completion_tokens, trace.total_tokens, trace.retry_count) == (32, 6, 38, 8)


@pytest.mark.parametrize("insufficient, loops, calls, prompt, completion, total, retries", [
    (False, 0, 5, 44, 12, 56, 6), (True, 2, 11, 92, 24, 116, 14),
])
def test_multi_agent_copies_persisted_run_and_report_metadata(
    environment, case, insufficient, loops, calls, prompt, completion, total, retries,
):
    service, _, retriever = multi_service(environment, insufficient=insufficient)
    trace = MultiAgentRagWorkflow(service, environment.mapping).run(case.id, case.question, case.source_files)
    assert trace.status == "success"
    assert trace.critic_loops == loops
    assert trace.model_calls == calls
    assert (trace.prompt_tokens, trace.completion_tokens, trace.total_tokens, trace.retry_count) == (prompt, completion, total, retries)
    assert trace.evidence_sufficient is (not insufficient)
    assert trace.cited_evidence_ids == [environment.evidence.id]
    assert f"#citation-{environment.evidence.id}" in trace.answer
    assert all(call[2] == [environment.mapping["solar-storage.md"]] for call in retriever.calls)
    if insufficient:
        assert "Capacity evidence" in trace.answer


@pytest.mark.parametrize("cls", [LlmRagWorkflow, SingleAgentRagWorkflow])
def test_unknown_citation_fails_trace_without_discarding_known_costs(cls, environment, case):
    responses = [GroundedAnswer(answer_markdown="Unsupported", cited_evidence_ids=["unknown"])]
    if cls is SingleAgentRagWorkflow:
        responses.insert(0, QueryPlan(queries=["capacity"]))
    provider = RecordingChat(structured_responses=responses)
    workflow = cls(provider, RecordingRetriever([environment.evidence]), environment.mapping)
    trace = workflow.run(case.id, case.question, case.source_files)
    assert trace.status == "failed"
    assert trace.error_code == "invalid_citation"
    assert trace.cited_evidence_ids == ["unknown"]
    assert len(trace.retrieved_evidence) == 1 and trace.prompt_tokens > 0


def test_multi_agent_maps_citation_failure_and_preserves_checkpoint_costs(environment, case):
    service, _, _ = multi_service(environment, invalid_citation=True)
    trace = MultiAgentRagWorkflow(service, environment.mapping).run(case.id, case.question, case.source_files)
    assert trace.status == "failed" and trace.error_code == "invalid_citation"
    assert (trace.prompt_tokens, trace.completion_tokens, trace.total_tokens, trace.retry_count) == (34, 9, 43, 5)
    assert trace.model_calls == 4
    assert trace.retrieved_evidence[0].id == environment.evidence.id


@pytest.mark.parametrize("failure, code", [
    (ProviderError("provider_timeout", "Authorization: Bearer sk-secret"), "provider_timeout"),
    (ProviderError("sk-secret", "Authorization: Bearer sk-secret"), "workflow_failed"),
    (RuntimeError("Authorization: Bearer sk-secret"), "workflow_failed"),
])
def test_failure_is_normalized_without_secrets_and_other_adapter_stays_usable(
    workflow_factory, environment, case, failure, code,
):
    healthy, _, _ = workflow_factory(WorkflowVariant.LLM_RAG)
    failing = BaselineLlmWorkflow(RecordingChat(failure=failure))
    mapping_before = dict(environment.mapping)
    trace = failing.run(case.id, case.question, case.source_files)
    assert trace.status == "failed" and trace.error_code == code
    assert "secret" not in trace.model_dump_json() and "Authorization" not in trace.model_dump_json()
    assert healthy.run(case.id, case.question, case.source_files).status == "success"
    assert environment.mapping == mapping_before


@pytest.mark.parametrize("variant", [WorkflowVariant.LLM_RAG, WorkflowVariant.SINGLE_AGENT_RAG, WorkflowVariant.MULTI_AGENT_RAG])
def test_unmapped_source_fails_before_provider_calls(variant, workflow_factory, case):
    workflow, provider, retriever = workflow_factory(variant)
    trace = workflow.run(case.id, case.question, ["unmapped.md"])
    assert trace.status == "failed" and trace.error_code == "document_not_found"
    assert provider.calls == [] and retriever.calls == []


def test_mapping_and_trace_are_owned_by_each_adapter(environment, case):
    mapping = dict(environment.mapping)
    provider = RecordingChat(structured_responses=[
        GroundedAnswer(answer_markdown="12 MW", cited_evidence_ids=[environment.evidence.id]),
        GroundedAnswer(answer_markdown="12 MW", cited_evidence_ids=[environment.evidence.id]),
    ])
    retriever = RecordingRetriever([environment.evidence])
    workflow = LlmRagWorkflow(provider, retriever, mapping)
    mapping.clear()
    first = workflow.run(case.id, case.question, case.source_files)
    first.retrieved_evidence[0].text = "changed by caller"
    first.cited_evidence_ids.clear()
    second = workflow.run(case.id, case.question, case.source_files)
    assert second.status == "success"
    assert second.retrieved_evidence[0].text == "The solar capacity is 12 MW."
    assert second.cited_evidence_ids == [environment.evidence.id]
    assert second.model_calls == 2 and second.prompt_tokens == 14


def test_metadata_identifiers_reject_secrets_and_negative_counts(case):
    class UnsafeMetadataChat(RecordingChat):
        def generate(self, messages):
            return "answer", ProviderMetadata(
                provider="https://user:sk-secret@example.test", model="Bearer sk-secret",
                latency_ms=-4, retries=-3,
                usage=TokenUsage(prompt_tokens=-10, completion_tokens=-2, total_tokens=-12),
            )

    trace = BaselineLlmWorkflow(UnsafeMetadataChat()).run(case.id, case.question, case.source_files)
    assert trace.status == "success"
    assert trace.provider == trace.model == ""
    assert trace.prompt_tokens == trace.completion_tokens == trace.total_tokens == trace.retry_count == 0
    assert "secret" not in trace.model_dump_json()


@pytest.mark.parametrize("cls, calls", [(LlmRagWorkflow, 2), (SingleAgentRagWorkflow, 3)])
def test_structured_adapters_satisfy_real_provider_json_mode_boundary(cls, calls, environment, case):
    responses = [GroundedAnswer(answer_markdown="12 MW", cited_evidence_ids=[environment.evidence.id])]
    if cls is SingleAgentRagWorkflow:
        responses.insert(0, QueryPlan(queries=["capacity"]))
    responses = iter(responses)

    class JsonRequiredCompletions:
        def create(self, **kwargs):
            if kwargs.get("response_format") != {"type": "json_object"} or not any(
                "json" in message["content"].casefold() for message in kwargs["messages"]
            ):
                raise BadRequestError(
                    "messages must mention JSON when requesting JSON mode",
                    response=httpx.Response(400, request=httpx.Request("POST", "https://provider.invalid/chat")),
                    body={"error": "JSON instruction required"},
                )
            return SimpleNamespace(
                model="boundary-chat",
                choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses).model_dump_json()))],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3, total_tokens=13),
            )

    provider = OpenAICompatibleChatProvider(
        client=SimpleNamespace(chat=SimpleNamespace(completions=JsonRequiredCompletions())),
        provider_name="fake", model="boundary-chat", max_retries=0,
    )
    trace = cls(provider, RecordingRetriever([environment.evidence]), environment.mapping).run(
        case.id, case.question, case.source_files,
    )
    assert trace.status == "success"
    assert trace.answer == "12 MW" and trace.cited_evidence_ids == [environment.evidence.id]
    assert trace.model_calls == calls and trace.retry_count == 2


@pytest.mark.parametrize("variant", [WorkflowVariant.LLM_RAG, WorkflowVariant.SINGLE_AGENT_RAG, WorkflowVariant.MULTI_AGENT_RAG])
def test_deduplicated_document_uses_benchmark_filename_for_recall_and_coverage(
    variant, environment, workflow_factory, case,
):
    benchmark_filename = "benchmark-solar.md"
    (environment.corpus / benchmark_filename).write_bytes(b"The solar capacity is 12 MW.")
    benchmark = BenchmarkCase(
        id=case.id, question=case.question, source_files=[benchmark_filename],
        expected_evidence=[{"source_file": benchmark_filename, "contains": "12 MW"}],
        answer_key_points=["The solar capacity is 12 MW."],
    )
    environment.mapping = prepare_benchmark_documents(
        [benchmark], environment.corpus, environment.document_service,
    )
    document_id = environment.mapping[benchmark_filename]
    assert environment.documents.get_document(document_id).filename == "solar-storage.md"
    assert document_id == environment.evidence.document_id
    workflow, _, _ = workflow_factory(variant)
    trace = workflow.run(benchmark.id, benchmark.question, benchmark.source_files)
    score = score_case(benchmark, trace)
    assert score.retrieval_recall_at_5 == 1.0
    assert score.evidence_coverage == 1.0
    assert trace.retrieved_evidence[0].source_file == benchmark_filename
    assert environment.evidence.filename == "solar-storage.md"


@pytest.mark.parametrize("alias", ["same-content.md", "./solar-storage.md"])
def test_prepare_rejects_ambiguous_benchmark_aliases_for_one_document(environment, case, alias):
    if alias == "same-content.md":
        (environment.corpus / alias).write_bytes(b"The solar capacity is 12 MW.")
    ambiguous = case.model_copy(update={"source_files": ["solar-storage.md", alias]})
    with pytest.raises(ValueError, match="ambiguous"):
        prepare_benchmark_documents([ambiguous], environment.corpus, environment.document_service)
