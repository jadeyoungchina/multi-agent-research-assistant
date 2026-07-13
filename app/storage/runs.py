import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime

from app.domain.research import ResearchReport
from app.domain.runs import ResearchRun, RunEvent
from app.storage.database import Database


def _now() -> str:
    return datetime.now(UTC).isoformat()


class RunRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> ResearchRun:
        value = dict(row)
        value["document_ids"] = json.loads(value.pop("document_ids_json"))
        value.pop("state_json")
        if value["evidence_sufficient"] is not None:
            value["evidence_sufficient"] = bool(value["evidence_sufficient"])
        return ResearchRun.model_validate_json(json.dumps(value, ensure_ascii=False))

    @staticmethod
    def _run_values(run: ResearchRun, state: dict) -> tuple:
        value = run.model_dump(mode="json")
        return (
            value["id"],
            value["question"],
            json.dumps(value["document_ids"], ensure_ascii=False),
            value["status"],
            value["current_stage"],
            value["last_completed_stage"],
            value["iteration"],
            value["provider"],
            value["model"],
            value["prompt_tokens"],
            value["completion_tokens"],
            value["total_tokens"],
            value["model_call_count"],
            value["retry_count"],
            value["evidence_sufficient"],
            json.dumps(state, ensure_ascii=False),
            value["error_code"],
            value["error_message"],
            value["created_at"],
            value["updated_at"],
            value["completed_at"],
        )

    def _insert_event(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        stage: str,
        event_type: str,
        payload: dict,
    ) -> RunEvent:
        sequence = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM run_events WHERE run_id = ?",
            (run_id,),
        ).fetchone()[0]
        created_at = _now()
        connection.execute(
            "INSERT INTO run_events "
            "(run_id, sequence, stage, event_type, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                run_id,
                sequence,
                stage,
                event_type,
                json.dumps(payload, ensure_ascii=False),
                created_at,
            ),
        )
        return RunEvent(
            sequence=sequence,
            run_id=run_id,
            stage=stage,
            event_type=event_type,
            payload=payload,
            created_at=created_at,
        )

    def create(self, run: ResearchRun, state: dict) -> None:
        with self._database.transaction() as connection:
            connection.execute(
                "INSERT INTO research_runs "
                "(id, question, document_ids_json, status, current_stage, "
                "last_completed_stage, iteration, provider, model, prompt_tokens, "
                "completion_tokens, total_tokens, model_call_count, retry_count, "
                "evidence_sufficient, state_json, error_code, error_message, created_at, "
                "updated_at, completed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self._run_values(run, state),
            )
            self._insert_event(
                connection, run.id, run.current_stage, "created", {"status": run.status}
            )

    def get(self, run_id: str) -> ResearchRun | None:
        with closing(self._database.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM research_runs WHERE id = ?", (run_id,)
            ).fetchone()
        return None if row is None else self._run_from_row(row)

    def get_state(self, run_id: str) -> dict:
        with closing(self._database.connect()) as connection:
            row = connection.execute(
                "SELECT state_json FROM research_runs WHERE id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return json.loads(row["state_json"])

    def start_stage(self, run_id: str, state: dict, stage: str) -> RunEvent:
        now = _now()
        with self._database.transaction() as connection:
            connection.execute(
                "UPDATE research_runs SET status = 'running', current_stage = ?, "
                "state_json = ?, error_code = NULL, error_message = NULL, updated_at = ? "
                "WHERE id = ?",
                (stage, json.dumps(state, ensure_ascii=False), now, run_id),
            )
            return self._insert_event(connection, run_id, stage, "started", {})

    def get_report(self, run_id: str) -> ResearchReport | None:
        with closing(self._database.connect()) as connection:
            row = connection.execute(
                "SELECT report_json FROM reports WHERE run_id = ?", (run_id,)
            ).fetchone()
        return None if row is None else ResearchReport.model_validate_json(row["report_json"])

    def append_event(
        self, run_id: str, stage: str, event_type: str, payload: dict
    ) -> RunEvent:
        with self._database.transaction() as connection:
            return self._insert_event(connection, run_id, stage, event_type, payload)

    def list_events(self, run_id: str, after_sequence: int) -> list[RunEvent]:
        with closing(self._database.connect()) as connection:
            rows = connection.execute(
                "SELECT sequence, run_id, stage, event_type, payload_json, created_at "
                "FROM run_events WHERE run_id = ? AND sequence > ? ORDER BY sequence",
                (run_id, after_sequence),
            ).fetchall()
        return [
            RunEvent.model_validate_json(
                json.dumps(
                    {
                        "sequence": row["sequence"],
                        "run_id": row["run_id"],
                        "stage": row["stage"],
                        "event_type": row["event_type"],
                        "payload": json.loads(row["payload_json"]),
                        "created_at": row["created_at"],
                    },
                    ensure_ascii=False,
                )
            )
            for row in rows
        ]

    def list_incomplete(self) -> list[ResearchRun]:
        with closing(self._database.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM research_runs WHERE status IN ('queued', 'running') "
                "ORDER BY created_at, id"
            ).fetchall()
        return [self._run_from_row(row) for row in rows]

    @staticmethod
    def _aggregate_metrics(provider_metrics: list[dict]) -> tuple[int, int, int, int]:
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        retry_count = 0
        for metric in provider_metrics:
            usage = metric.get("usage") or {}
            prompt_tokens += int(usage.get("prompt_tokens", 0))
            completion_tokens += int(usage.get("completion_tokens", 0))
            total_tokens += int(usage.get("total_tokens", 0))
            retry_count += int(metric.get("retries", 0))
        return prompt_tokens, completion_tokens, total_tokens, retry_count

    def complete_stage(
        self,
        run_id: str,
        state: dict,
        current_stage: str,
        last_completed_stage: str,
        iteration: int,
        provider_metrics: list[dict],
        event_payload: dict,
        report: ResearchReport | None = None,
    ) -> RunEvent:
        if state.get("next_stage") != current_stage:
            raise ValueError("current_stage must match state['next_stage']")
        if current_stage == "completed" and report is None:
            raise ValueError("a report is required when completing a run")

        prompt, completion, total, retries = self._aggregate_metrics(provider_metrics)
        now = _now()
        final = current_stage == "completed"
        status = "completed" if final else "running"
        evidence_sufficient = report.evidence_sufficient if report is not None else None
        with self._database.transaction() as connection:
            connection.execute(
                "UPDATE research_runs SET status = ?, current_stage = ?, "
                "last_completed_stage = ?, iteration = ?, prompt_tokens = ?, "
                "completion_tokens = ?, total_tokens = ?, model_call_count = ?, "
                "retry_count = ?, evidence_sufficient = ?, state_json = ?, "
                "error_code = NULL, error_message = NULL, updated_at = ?, completed_at = ? "
                "WHERE id = ?",
                (
                    status,
                    current_stage,
                    last_completed_stage,
                    iteration,
                    prompt,
                    completion,
                    total,
                    len(provider_metrics),
                    retries,
                    evidence_sufficient,
                    json.dumps(state, ensure_ascii=False),
                    now,
                    now if final else None,
                    run_id,
                ),
            )
            event = self._insert_event(
                connection,
                run_id,
                last_completed_stage,
                "completed",
                event_payload,
            )
            if final:
                report_json = json.dumps(report.model_dump(mode="json"), ensure_ascii=False)
                connection.execute(
                    "INSERT INTO reports (run_id, report_json, markdown, created_at) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT(run_id) DO UPDATE SET "
                    "report_json = excluded.report_json, markdown = excluded.markdown, "
                    "created_at = excluded.created_at",
                    (run_id, report_json, report.markdown, now),
                )
                self._insert_event(
                    connection,
                    run_id,
                    last_completed_stage,
                    "finished",
                    {
                        "status": "completed",
                        "evidence_sufficient": report.evidence_sufficient,
                    },
                )
            return event

    def fail_stage(
        self,
        run_id: str,
        state: dict,
        stage: str,
        code: str,
        message: str,
    ) -> RunEvent:
        del state
        now = _now()
        with self._database.transaction() as connection:
            connection.execute(
                "UPDATE research_runs SET status = 'failed', error_code = ?, "
                "error_message = ?, updated_at = ? WHERE id = ?",
                (code, message, now, run_id),
            )
            return self._insert_event(
                connection,
                run_id,
                stage,
                "failed",
                {"code": code, "message": message},
            )
