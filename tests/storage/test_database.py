from pathlib import Path

from app.storage.database import Database


def test_initialize_creates_expected_tables(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    database.initialize()
    with database.connect() as connection:
        names = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"documents", "chunks", "research_runs", "run_events", "reports"} <= names


def test_connection_enables_foreign_keys_and_wal(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    database.initialize()
    with database.connect() as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_transaction_commits_changes(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    database.initialize()

    with database.transaction() as connection:
        connection.execute(
            "INSERT INTO documents "
            "(id, filename, media_type, sha256, storage_path, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("doc-1", "paper.pdf", "application/pdf", "hash-1", "/tmp/paper.pdf", "ready", "2026-01-01T00:00:00Z"),
        )

    with database.connect() as connection:
        assert connection.execute("SELECT id FROM documents").fetchone()[0] == "doc-1"
