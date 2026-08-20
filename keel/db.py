"""SQLite store. One file holds documents, chunks (with embeddings and ACL tags), the FTS5 index,
the inference log, the approval queue and the hash-chained ledger.

Schema is applied idempotently on connect. Migrations are additive: append to SCHEMA, never edit
an existing statement.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS documents (
  id            INTEGER PRIMARY KEY,
  source        TEXT NOT NULL,           -- path or URL as given
  title         TEXT,
  checksum      TEXT NOT NULL UNIQUE,    -- sha256 of the raw bytes; idempotent re-ingest
  mime          TEXT,
  acl_tags      TEXT NOT NULL DEFAULT '["public"]',  -- JSON list; a user needs one of these
  ingested_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  meta          TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS chunks (
  id            INTEGER PRIMARY KEY,
  document_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  ordinal       INTEGER NOT NULL,        -- position within document
  text          TEXT NOT NULL,
  heading       TEXT,                    -- nearest section heading
  page          INTEGER,                 -- 1-based page for paginated sources
  char_start    INTEGER,
  char_end      INTEGER,
  checksum      TEXT NOT NULL,           -- sha256 of text
  acl_tags      TEXT NOT NULL DEFAULT '["public"]',
  quarantined   INTEGER NOT NULL DEFAULT 0,   -- 1 when injection screening flagged it
  quarantine_reason TEXT,
  embedding     BLOB,                    -- float32 little-endian, dim from embed model
  embed_model   TEXT,
  UNIQUE(document_id, ordinal)
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);

-- Full-text index over chunk text (BM25 via FTS5 rank).
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text, heading, content='chunks', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, text, heading) VALUES (new.id, new.text, new.heading);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text, heading) VALUES('delete', old.id, old.text, old.heading);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text, heading) VALUES('delete', old.id, old.text, old.heading);
  INSERT INTO chunks_fts(rowid, text, heading) VALUES (new.id, new.text, new.heading);
END;

-- Every request the appliance answers.
CREATE TABLE IF NOT EXISTS inference_log (
  id            INTEGER PRIMARY KEY,
  ts            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  request_id    TEXT NOT NULL UNIQUE,
  user_id       TEXT,
  user_tags     TEXT,                    -- JSON list of ACL tags the user carried
  mode          TEXT NOT NULL,           -- answer | agent | eval
  question      TEXT NOT NULL,
  retrieved_ids TEXT,                    -- JSON list of chunk ids after ACL filtering
  tool_calls    TEXT,                    -- JSON list
  answer        TEXT,
  refused       INTEGER NOT NULL DEFAULT 0,
  citations     TEXT,                    -- JSON list
  latency_ms    INTEGER,
  prompt_tokens INTEGER,
  output_tokens INTEGER,
  judge         TEXT,                    -- JSON scores when evaluated
  provider      TEXT,
  model         TEXT
);
CREATE INDEX IF NOT EXISTS idx_log_ts ON inference_log(ts);

-- Write-capable tool calls wait here for a human.
CREATE TABLE IF NOT EXISTS approvals (
  id            INTEGER PRIMARY KEY,
  ts            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  request_id    TEXT NOT NULL,
  tool          TEXT NOT NULL,
  arguments     TEXT NOT NULL,           -- JSON
  status        TEXT NOT NULL DEFAULT 'pending',   -- pending | approved | rejected | executed
  decided_at    TEXT,
  decided_by    TEXT,
  result        TEXT
);

-- Hash-chained audit ledger. Each row's hash covers its payload and the previous hash.
CREATE TABLE IF NOT EXISTS ledger (
  seq           INTEGER PRIMARY KEY,
  ts            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  kind          TEXT NOT NULL,           -- request | retrieval | tool_call | tool_result | answer | approval | ingest | quarantine
  request_id    TEXT,
  payload       TEXT NOT NULL,           -- canonical JSON
  prev_hash     TEXT NOT NULL,
  hash          TEXT NOT NULL UNIQUE
);
"""


def connect(path: Path | str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN")
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
