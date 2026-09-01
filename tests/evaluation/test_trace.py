from app.evaluation.metrics import percentile, score_case
from app.evaluation.models import BenchmarkCase, WorkflowVariant
from app.evaluation.trace import EvaluationTrace


def _case() -> BenchmarkCase:
    return BenchmarkCase(
        id="BENCH-001",
        question="Which facts are supported?",
        source_files=["solar.md", "water.md"],
        expected_evidence=[
            {"source_file": "solar.md", "contains": "12 MW solar array"},
            {"source_file": "water.md", "contains": "24 MWh battery"},
        ],
        answer_key_points=["The pilot used a 12 MW solar array."],
    )


def test_failed_trace_scores_zero_without_aborting() -> None:
    """A provider failure must contribute a zero-quality score instead of raising."""
    score = score_case(
        _case(),
        EvaluationTrace.failed("BENCH-001", WorkflowVariant.MULTI_AGENT_RAG, "provider_timeout"),
    )

    assert score.success is False
    assert score.retrieval_recall_at_5 == 0.0
    assert score.citation_precision == 0.0
    assert score.evidence_coverage == 0.0
    assert score.answer_key_point_f1 == 0.0
    assert score.answer_key_point_coverage == 0.0


def test_percentile_uses_nearest_rank() -> None:
    """Changing percentile interpolation must not change published latency percentiles."""
    assert percentile([10, 20, 30, 40], 0.50) == 20
    assert percentile([10, 20, 30, 40], 0.95) == 40
