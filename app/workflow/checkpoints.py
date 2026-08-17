import logging
from collections.abc import Callable
from time import perf_counter
from typing import Any

from app.domain.providers import ProviderMetadata
from app.domain.research import ResearchReport
from app.storage.runs import RunRepository
from app.workflow.state import WorkflowState, merge_state


logger = logging.getLogger(__name__)

Node = Callable[[WorkflowState], dict[str, Any]]


def _safe_provider_metrics(values: object) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []

    safe: list[dict[str, Any]] = []
    for value in values:
        try:
            metric = ProviderMetadata.model_validate(value)
        except (TypeError, ValueError):
            continue
        safe.append(metric.model_dump(mode="json"))
    return safe


def _log_metadata(
    run_id: str,
    stage: str,
    provider_metrics: object,
    started_at: float,
    exception_type: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "stage": stage,
        "provider_metrics": _safe_provider_metrics(provider_metrics),
        "elapsed_ms": max(0, int((perf_counter() - started_at) * 1000)),
    }
    if exception_type is not None:
        metadata["exception_type"] = exception_type
    return metadata


def checkpointed_node(
    stage: str,
    handler: Node,
    repository: RunRepository,
) -> Node:
    def run(state: WorkflowState) -> dict[str, Any]:
        started_at = perf_counter()
        run_id = state["run_id"]
        repository.start_stage(run_id, dict(state), stage)
        logger.info(
            "workflow_stage_started",
            extra=_log_metadata(
                run_id,
                stage,
                state.get("provider_metrics", []),
                started_at,
            ),
        )
        try:
            updates = handler(state)
            merged = merge_state(state, updates)
            merged["last_completed_stage"] = stage
            repository.complete_stage(
                run_id=run_id,
                state=merged,
                current_stage=merged["next_stage"],
                last_completed_stage=stage,
                iteration=int(merged.get("iteration", 0)),
                provider_metrics=list(merged.get("provider_metrics", [])),
                event_payload={},
                report=(
                    ResearchReport.model_validate(merged["report"])
                    if stage == "citation_validator"
                    else None
                ),
            )
            logger.info(
                "workflow_stage_completed",
                extra=_log_metadata(
                    run_id,
                    stage,
                    merged.get("provider_metrics", []),
                    started_at,
                ),
            )
            return {**updates, "last_completed_stage": stage}
        except Exception as exc:
            exception_type = type(exc).__name__
            repository.fail_stage(
                run_id,
                dict(state),
                stage,
                "workflow_stage_failed",
                exception_type,
            )
            logger.error(
                "workflow_stage_failed",
                extra=_log_metadata(
                    run_id,
                    stage,
                    state.get("provider_metrics", []),
                    started_at,
                    exception_type,
                ),
            )
            raise

    return run
