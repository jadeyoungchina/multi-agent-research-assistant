import json
import sqlite3
from contextlib import closing

from app.domain.documents import DocumentRecord, EvidenceChunk
from app.storage.database import Database
from app.storage.serialization import blob_to_vector, vector_to_blob


class DocumentRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    @staticmethod
    def _document_from_row(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord.model_validate_json(
            json.dumps(dict(row), ensure_ascii=False)
        )

    @staticmethod
    def _document_values(document: DocumentRecord) -> tuple:
        value = document.model_dump(mode="json")
        return (
            value["id"],
            value["filename"],
            value["media_type"],
            value["sha256"],
            value["storage_path"],
            value["status"],
            value["page_count"],
            value["error_message"],
            value["created_at"],
        )

    def add_document(self, document: DocumentRecord) -> None:
        with self._database.transaction() as connection:
            connection.execute(
                "INSERT INTO documents "
                "(id, filename, media_type, sha256, storage_path, status, page_count, "
                "error_message, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self._document_values(document),
            )

    def add_document_if_absent(
        self, document: DocumentRecord
    ) -> tuple[DocumentRecord, bool]:
        with self._database.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO documents "
                "(id, filename, media_type, sha256, storage_path, status, page_count, "
                "error_message, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(sha256) DO NOTHING",
                self._document_values(document),
            )
            if cursor.rowcount == 1:
                return document, True
            row = connection.execute(
                "SELECT * FROM documents WHERE sha256 = ?", (document.sha256,)
            ).fetchone()
        if row is None:
            raise RuntimeError("document insert conflict did not retain a record")
        return self._document_from_row(row), False

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with closing(self._database.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return None if row is None else self._document_from_row(row)

    def get_document_by_sha256(self, sha256: str) -> DocumentRecord | None:
        with closing(self._database.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE sha256 = ?", (sha256,)
            ).fetchone()
        return None if row is None else self._document_from_row(row)

    def list_documents(self) -> list[DocumentRecord]:
        with closing(self._database.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY created_at, id"
            ).fetchall()
        return [self._document_from_row(row) for row in rows]

    def update_document(self, document: DocumentRecord) -> None:
        value = document.model_dump(mode="json")
        with self._database.transaction() as connection:
            connection.execute(
                "UPDATE documents SET filename = ?, media_type = ?, sha256 = ?, "
                "storage_path = ?, status = ?, page_count = ?, error_message = ?, "
                "created_at = ? WHERE id = ?",
                (
                    value["filename"],
                    value["media_type"],
                    value["sha256"],
                    value["storage_path"],
                    value["status"],
                    value["page_count"],
                    value["error_message"],
                    value["created_at"],
                    value["id"],
                ),
            )

    def update_status(
        self,
        document_id: str,
        status: str,
        page_count: int,
        error_message: str | None = None,
    ) -> None:
        with self._database.transaction() as connection:
            connection.execute(
                "UPDATE documents SET status = ?, page_count = ?, error_message = ? "
                "WHERE id = ?",
                (status, page_count, error_message, document_id),
            )

    def _insert_chunks(
        self,
        connection: sqlite3.Connection,
        chunks: list[tuple[EvidenceChunk, list[float]]],
    ) -> None:
        connection.executemany(
            "INSERT INTO chunks "
            "(id, document_id, filename, page_number, chunk_index, content, "
            "content_sha256, embedding_model, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    value["id"],
                    value["document_id"],
                    value["filename"],
                    value["page_number"],
                    value["chunk_index"],
                    value["content"],
                    value["content_sha256"],
                    value["embedding_model"],
                    vector_to_blob(vector),
                )
                for chunk, vector in chunks
                for value in [chunk.model_dump(mode="json")]
            ],
        )

    def replace_chunks(
        self,
        document_id: str,
        chunks: list[tuple[EvidenceChunk, list[float]]],
    ) -> None:
        with self._database.transaction() as connection:
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            self._insert_chunks(connection, chunks)

    def complete_ingestion(
        self,
        document: DocumentRecord,
        chunks: list[tuple[EvidenceChunk, list[float]]],
    ) -> None:
        ready = document.model_copy(
            update={"status": "ready", "error_message": None}
        ).model_dump(mode="json")
        with self._database.transaction() as connection:
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document.id,))
            self._insert_chunks(connection, chunks)
            connection.execute(
                "UPDATE documents SET filename = ?, media_type = ?, sha256 = ?, "
                "storage_path = ?, status = ?, page_count = ?, error_message = ?, "
                "created_at = ? WHERE id = ?",
                (
                    ready["filename"],
                    ready["media_type"],
                    ready["sha256"],
                    ready["storage_path"],
                    ready["status"],
                    ready["page_count"],
                    ready["error_message"],
                    ready["created_at"],
                    ready["id"],
                ),
            )

    def fail_ingestion(self, document_id: str, error_message: str) -> None:
        with self._database.transaction() as connection:
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            connection.execute(
                "UPDATE documents SET status = 'failed', error_message = ? WHERE id = ?",
                (error_message, document_id),
            )

    def list_chunks(
        self, document_ids: list[str]
    ) -> list[tuple[EvidenceChunk, list[float]]]:
        if not document_ids:
            return []
        placeholders = ", ".join("?" for _ in document_ids)
        order = " ".join(f"WHEN ? THEN {index}" for index, _ in enumerate(document_ids))
        parameters = [*document_ids, *document_ids]
        with closing(self._database.connect()) as connection:
            rows = connection.execute(
                f"SELECT * FROM chunks WHERE document_id IN ({placeholders}) "
                f"ORDER BY CASE document_id {order} END, chunk_index",
                parameters,
            ).fetchall()
        results = []
        for row in rows:
            chunk_data = {
                key: row[key]
                for key in (
                    "id",
                    "document_id",
                    "filename",
                    "page_number",
                    "chunk_index",
                    "content",
                    "content_sha256",
                    "embedding_model",
                )
            }
            chunk = EvidenceChunk.model_validate_json(
                json.dumps(chunk_data, ensure_ascii=False)
            )
            results.append((chunk, blob_to_vector(row["embedding"])))
        return results

    def delete_document(self, document_id: str) -> None:
        with self._database.transaction() as connection:
            connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
