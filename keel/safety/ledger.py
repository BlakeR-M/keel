"""Hash-chained, tamper-evident audit ledger.

Every request, retrieval set, tool call, approval and answer is appended as one row whose hash covers
its own content and the hash of the row before it. Changing, removing or reordering any row breaks the
chain from that point on, and `verify()` names the first broken link.

Hash recipe (the contract an auditor recomputes offline):

    canonical_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    hash = sha256(prev_hash + "|" + kind + "|" + (request_id or "") + "|" + canonical_payload)

The first row chains from GENESIS_HASH (sixty-four zeros). Timestamps and sequence numbers sit outside
the hash; order is bound by the chain itself. A chain proves nothing about rows removed from its tail,
so anchor `head()` somewhere external (a log line, a signed release note) when that matters.

Payloads are strict JSON: NaN and Infinity are refused at append time (allow_nan=False), so an export
verifies with any RFC 8259 parser, not only Python's.

Standard library only: an auditor runs `verify_file()` on an exported JSONL file with nothing but Python.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GENESIS_HASH = "0" * 64

# Kinds the rest of Keel writes. Informational; the ledger accepts any non-empty kind.
LEDGER_KINDS = frozenset(
    {"request", "retrieval", "tool_call", "tool_result", "answer", "approval", "ingest", "quarantine"}
)

# (seq, kind, request_id, canonical_payload, prev_hash, hash)
_Record = tuple[int, str, str | None, str, str, str]


@dataclass(frozen=True)
class LedgerEntry:
    """One committed ledger row."""

    seq: int
    ts: str
    kind: str
    request_id: str | None
    prev_hash: str
    hash: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of a chain verification. `checked` counts rows examined, including a failing row."""

    ok: bool
    checked: int
    first_bad_seq: int | None
    reason: str | None


def canonical_json(payload: dict[str, Any]) -> str:
    """Serialise a payload in the one byte-stable form the hash covers. Raises ValueError for NaN or
    Infinity, which strict JSON parsers refuse."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def hash_material(prev_hash: str, kind: str, request_id: str | None, canonical_payload: str) -> str:
    """The exact bytes a row hash covers: one canonical JSON array of the four chained fields.

    A JSON array keeps every field boundary explicit, so text cannot move between `kind`, `request_id`
    and the payload without changing the material, and a NULL request_id stays distinct from ''.
    The payload is embedded as its canonical string (already sorted and compact), so an auditor can
    recompute the material from an exported row with the standard library alone.
    """
    return json.dumps(
        [prev_hash, kind, request_id, canonical_payload],
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def entry_hash(prev_hash: str, kind: str, request_id: str | None, canonical_payload: str) -> str:
    """Compute a row hash from its chained fields."""
    return hashlib.sha256(
        hash_material(prev_hash, kind, request_id, canonical_payload).encode("utf-8")
    ).hexdigest()


_HEAD_SQL = "SELECT hash FROM ledger ORDER BY seq DESC LIMIT 1"
_INSERT_SQL = (
    "INSERT INTO ledger (kind, request_id, payload, prev_hash, hash) VALUES (?, ?, ?, ?, ?) RETURNING seq, ts"
)
_ROWS_SQL = "SELECT seq, ts, kind, request_id, payload, prev_hash, hash FROM ledger ORDER BY seq"

# One lock per SQLite connection, shared by every Ledger built on it. The web app hands one connection to
# every request thread; the lock keeps the head read and the row insert of one append together, and keeps
# two Ledger instances on the same connection (the ingest pipeline builds its own) from interleaving.
_LOCKS: dict[int, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(conn: sqlite3.Connection) -> threading.RLock:
    key = id(conn)
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _LOCKS[key] = lock
        return lock


class Ledger:
    """Append-only view over the `ledger` table of one SQLite connection.

    Appends are serialised with a per-connection lock and run inside `BEGIN IMMEDIATE`, so the head hash
    is read and the next row written under one write lock and two writers, on this connection or on
    another connection to the same file, can never both chain from the same predecessor. An append
    issued inside a caller's open transaction joins that transaction. Hashing happens in Python, never
    inside a SQLite user function: a Python callback running inside `sqlite3_step` while another thread
    uses the same connection froze the whole process.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._lock = _lock_for(conn)

    def append(self, kind: str, request_id: str | None, payload: dict[str, Any]) -> LedgerEntry:
        """Append one row and return it as committed."""
        if not isinstance(kind, str) or not kind:
            raise ValueError("kind must be a non-empty string")
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")
        canonical = canonical_json(payload)
        with self._lock:
            if self._conn.in_transaction:
                seq, ts, prev_hash, row_hash = self._append_row(kind, request_id, canonical)
            else:
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    seq, ts, prev_hash, row_hash = self._append_row(kind, request_id, canonical)
                    self._conn.execute("COMMIT")
                except BaseException:
                    self._conn.execute("ROLLBACK")
                    raise
        return LedgerEntry(
            seq=seq,
            ts=ts,
            kind=kind,
            request_id=request_id,
            prev_hash=prev_hash,
            hash=row_hash,
            payload=payload,
        )

    def _append_row(self, kind: str, request_id: str | None, canonical: str) -> tuple[int, str, str, str]:
        """Read the head and insert the next row. Runs under the lock inside an open transaction."""
        head = self._conn.execute(_HEAD_SQL).fetchone()
        prev_hash = head["hash"] if head else GENESIS_HASH
        row_hash = entry_hash(prev_hash, kind, request_id, canonical)
        row = self._conn.execute(_INSERT_SQL, (kind, request_id, canonical, prev_hash, row_hash)).fetchone()
        return int(row["seq"]), row["ts"], prev_hash, row_hash

    def head(self) -> str:
        """Return the hash of the newest row, or GENESIS_HASH when the ledger is empty."""
        with self._lock:
            row = self._conn.execute(_HEAD_SQL).fetchone()
        return row["hash"] if row else GENESIS_HASH

    def count(self) -> int:
        """Return the number of rows."""
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM ledger").fetchone()[0])

    def entries(self) -> Iterator[LedgerEntry]:
        """Yield every row in sequence order."""
        with self._lock:
            rows = self._conn.execute(_ROWS_SQL).fetchall()
        for row in rows:
            yield LedgerEntry(
                seq=row["seq"],
                ts=row["ts"],
                kind=row["kind"],
                request_id=row["request_id"],
                prev_hash=row["prev_hash"],
                hash=row["hash"],
                payload=json.loads(row["payload"]),
            )

    def verify(self) -> VerifyResult:
        """Recompute every hash in sequence order and report the first broken link."""
        with self._lock:
            rows = self._conn.execute(_ROWS_SQL).fetchall()
        return _verify_records(
            (row["seq"], row["kind"], row["request_id"], row["payload"], row["prev_hash"], row["hash"])
            for row in rows
        )

    def export(self, path: Path | str) -> int:
        """Write the ledger as JSONL (one row per line, payload as an object) and return the row count."""
        written = 0
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for entry in self.entries():
                record = {
                    "seq": entry.seq,
                    "ts": entry.ts,
                    "kind": entry.kind,
                    "request_id": entry.request_id,
                    "prev_hash": entry.prev_hash,
                    "hash": entry.hash,
                    "payload": entry.payload,
                }
                fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                written += 1
        return written


def verify_file(path: Path | str) -> VerifyResult:
    """Verify an exported JSONL ledger with nothing but the file. This is what an auditor runs.

    A leading UTF-8 byte-order mark (an editor re-save) is skipped; any other decoding failure is
    reported as unreadable rather than raised."""
    records: list[_Record] = []
    try:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for line_no, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(
                        (
                            int(record["seq"]),
                            record["kind"],
                            record.get("request_id"),
                            canonical_json(record["payload"]),
                            record["prev_hash"],
                            record["hash"],
                        )
                    )
                except (ValueError, KeyError, TypeError) as exc:
                    return VerifyResult(
                        ok=False,
                        checked=len(records),
                        first_bad_seq=None,
                        reason=f"line {line_no}: unreadable ({exc})",
                    )
    except UnicodeDecodeError as exc:
        return VerifyResult(
            ok=False, checked=len(records), first_bad_seq=None, reason=f"file is unreadable as UTF-8 ({exc})"
        )
    return _verify_records(records)


def _verify_records(records: Iterable[_Record]) -> VerifyResult:
    """Walk records in order, recomputing each hash against the previous one."""
    expected_prev = GENESIS_HASH
    last_seq: int | None = None
    checked = 0
    for seq, kind, request_id, canonical_payload, prev_hash, stored_hash in records:
        checked += 1
        if last_seq is not None and seq <= last_seq:
            return VerifyResult(
                False, checked, seq, f"seq {seq} follows seq {last_seq}; rows are out of order"
            )
        if prev_hash != expected_prev:
            return VerifyResult(False, checked, seq, f"seq {seq}: prev_hash breaks the chain")
        if entry_hash(prev_hash, kind, request_id, canonical_payload) != stored_hash:
            return VerifyResult(False, checked, seq, f"seq {seq}: hash mismatch (row content changed)")
        expected_prev = stored_hash
        last_seq = seq
    return VerifyResult(True, checked, None, None)
