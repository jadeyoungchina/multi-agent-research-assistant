from app.evaluation.metrics import aggregate_scores, score_case
from app.evaluation.models import BenchmarkCase, WorkflowVariant
from app.evaluation.trace import EvaluationTrace, EvidenceSnapshot


def _case() -> BenchmarkCase:
    return BenchmarkCase(
        id="BENCH-001",
        question="Which pilot facts are supported?",
        source_files=["solar.md", "water.md"],
        expected_evidence=[
            {"source_file": "solar.md", "contains": "12 MW solar array"},
            {"source_file": "water.md", "contains": "24 MWh battery"},
        ],
        answer_key_points=[
            "The pilot used a 12 MW solar array.",
            "The pilot used a 24 MWh battery.",
        ],
    )


def _trace() -> EvaluationTrace:
    return EvaluationTrace(
        case_id="BENCH-001",
        variant=WorkflowVariant.LLM_RAG,
        status="success",
        answer="The pilot used a 12 MW solar array. It has a battery.",
        retrieved_evidence=[
            EvidenceSnapshot(
                id="evidence-1",
                source_file="solar.md",
                page_number=1,
                chunk_index=0,
                text="The pilot used a 12 MW solar array.",
            ),
            EvidenceSnapshot(
                id="evidence-2",
                source_file="solar.md",
                page_number=1,
                chunk_index=1,
                text="An unrelated observation.",
            ),
            EvidenceSnapshot(
                id="evidence-3",
                source_file="other.md",
                page_number=1,
                chunk_index=0,
                text="The pilot used a 24 MWh battery.",
            ),
        ],
        cited_evidence_ids=["evidence-1", "missing-id", "missing-id"],
        latency_ms=40,
        prompt_tokens=11,
        completion_tokens=7,
        model_calls=2,
        critic_loops=1,
        provider="fake",
        model="fake-1",
    )


def test_grounding_metrics_have_exact_values() -> None:
    """Wrong-source retrieval and unknown citations must not count as grounded evidence."""
    score = score_case(_case(), _trace())

    assert score.retrieval_recall_at_5 == 0.5
    assert score.citation_precision == 0.5
    assert score.evidence_coverage == 0.5


def test_answer_key_point_metrics_use_sentence_f1_and_cjk_tokens() -> None:
    """Scoring must split CJK characters and require the 0.60 coverage threshold."""
    cjk_case = BenchmarkCase(
        id="BENCH-002",
        question="What did the pilot report?",
        source_files=["source.md"],
        expected_evidence=[{"source_file": "source.md", "contains": "报告"}],
        answer_key_points=["中文报告", "24 MWh battery"],
    )
    trace = EvaluationTrace(
        case_id="BENCH-002",
        variant=WorkflowVariant.BASELINE_LLM,
        status="success",
        answer="中文报告。unrelated latin span.",
        latency_ms=10,
        prompt_tokens=1,
        completion_tokens=1,
        model_calls=1,
        critic_loops=0,
        provider="fake",
        model="fake-1",
    )

    score = score_case(cjk_case, trace)

    assert score.answer_key_point_f1 == 0.5
    assert score.answer_key_point_coverage == 0.5


def test_cjk_characters_contribute_individually_to_token_f1() -> None:
    """Treating a CJK phrase as one token would incorrectly score this answer as zero."""
    case = BenchmarkCase(
        id="BENCH-003",
        question="What was reported?",
        source_files=["source.md"],
        expected_evidence=[{"source_file": "source.md", "contains": "报告"}],
        answer_key_points=["中文报"],
    )
    trace = EvaluationTrace(
        case_id="BENCH-003",
        variant=WorkflowVariant.BASELINE_LLM,
        status="success",
        answer="中文。",
        latency_ms=10,
        prompt_tokens=1,
        completion_tokens=1,
        model_calls=1,
        critic_loops=0,
        provider="fake",
        model="fake-1",
    )

    score = score_case(case, trace)

    assert score.answer_key_point_f1 == 0.8
    assert score.answer_key_point_coverage == 1.0


def test_aggregate_scores_keeps_failed_traces_and_uses_nearest_rank_latency() -> None:
    """A failed trace must remain in success-rate and latency aggregates."""
    successful = score_case(_case(), _trace())
    failed = score_case(
        _case(),
        EvaluationTrace.failed("BENCH-001", WorkflowVariant.LLM_RAG, "provider_timeout"),
    )

    aggregate = aggregate_scores([successful, failed])

    assert aggregate.score_count == 2
    assert aggregate.success_count == 1
    assert aggregate.failure_count == 1
    assert aggregate.success_rate == 0.5
    assert aggregate.failure_rate == 0.5
    assert aggregate.mean_retrieval_recall_at_5 == 0.25
    assert aggregate.p50_latency_ms == 0
    assert aggregate.p95_latency_ms == 40
    assert aggregate.mean_total_tokens == 9.0
