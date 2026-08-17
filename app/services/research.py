from typing import Any

from app.domain.errors import WorkflowError
from app.domain.research import ResearchReport
from app.domain.runs import ResearchRun, RunEvent
from app.storage.documents import DocumentRepository
from app.storage.runs import RunRepository
from app.workflow.state import initial_state


class ResearchService:
    def __init__(
        self,
        document_repository: DocumentRepository,
        run_repository: RunRepository,
        graph: Any,
        provider_name: str,
        model: str,
        recursion_limit: int = 24,
    ) -> None:
        self.documents = document_repository
        self.runs = run_repository
        self.graph = graph
        self.provider_name = provider_name
        self.model = model
        self.recursion_limit = recursion_limit

    def create_run(self, question: str, document_ids: list[str]) -> ResearchRun:
        cleaned = question.strip()
        if not cleaned:
            raise WorkflowError("empty_question", "question must not be blank")

        selected_ids = list(document_ids)
        documents = [self.documents.get_document(value) for value in selected_ids]
        if not documents or any(document is None for document in documents):
            raise WorkflowError(
                "document_not_found", "one or more documents do not exist"
            )
        if any(document.status != "ready" for document in documents if document):
            raise WorkflowError("document_not_ready", "all documents must be ready")

        run = ResearchRun.new(
            cleaned,
            selected_ids,
            self.provider_name,
            self.model,
        )
        self.runs.create(run, initial_state(run.id, cleaned, selected_ids))
        return run

    def execute(self, run_id: str) -> ResearchReport:
        run = self.runs.get(run_id)
        if run is None:
            raise WorkflowError("run_not_found", "research run does not exist")
        if run.status == "failed":
            raise WorkflowError(
                "run_failed", "failed runs are not retried automatically"
            )

        existing = self.runs.get_report(run_id)
        if existing is not None:
            return existing

        state = self.runs.get_state(run_id)
        try:
            self.graph.invoke(
                state,
                {"recursion_limit": self.recursion_limit},
            )
        except Exception as exc:
            current = self.runs.get(run_id)
            if current is not None and current.status != "failed":
                failed_state = self.runs.get_state(run_id)
                self.runs.fail_stage(
                    run_id,
                    failed_state,
                    failed_state["next_stage"],
                    "workflow_failed",
                    type(exc).__name__,
                )
            raise

        report = self.runs.get_report(run_id)
        if report is None:
            raise WorkflowError(
                "report_not_persisted",
                "workflow completed without a persisted report",
            )
        return report

    def get_run(self, run_id: str) -> ResearchRun:
        run = self.runs.get(run_id)
        if run is None:
            raise WorkflowError("run_not_found", "research run does not exist")
        return run

    def get_report(self, run_id: str) -> ResearchReport | None:
        self.get_run(run_id)
        return self.runs.get_report(run_id)

    def list_events(
        self,
        run_id: str,
        after_sequence: int = 0,
    ) -> list[RunEvent]:
        self.get_run(run_id)
        return self.runs.list_events(run_id, after_sequence)

    def list_incomplete_ids(self) -> list[str]:
        return [run.id for run in self.runs.list_incomplete()]
