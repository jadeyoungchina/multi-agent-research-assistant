CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    media_type TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    storage_path TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('processing', 'ready', 'failed')),
    page_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    page_number INTEGER,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    embedding_model TEXT,
    embedding BLOB,
    UNIQUE(document_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS research_runs (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    document_ids_json TEXT NOT NULL,
    status TEXT NOT NULL,
    current_stage TEXT NOT NULL,
    last_completed_stage TEXT,
    iteration INTEGER NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    model_call_count INTEGER NOT NULL DEFAULT 0,
    retry_count INTEGER NOT NULL DEFAULT 0,
    evidence_sufficient INTEGER CHECK(evidence_sufficient IN (0, 1)),
    state_json TEXT NOT NULL,
    error_code TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS run_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    stage TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, sequence)
);

CREATE TABLE IF NOT EXISTS reports (
    run_id TEXT PRIMARY KEY REFERENCES research_runs(id) ON DELETE CASCADE,
    report_json TEXT NOT NULL,
    markdown TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_events_run_sequence ON run_events(run_id, sequence);
CREATE INDEX IF NOT EXISTS idx_runs_status ON research_runs(status);
