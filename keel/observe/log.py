"""Inference log: one row per request the appliance answered, plus the read queries the admin page
and the eval runner use.

`record(**fields)` is the duck-typed hook the answer engine and agent loop call. Unknown fields are
ignored; JSON-able fields are serialised.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

_COLUMNS = (
    "request_id",
    "user_id",
    "user_tags",
    "mode",
    "question",
    "retrieved_ids",
    "tool_calls",
    "answer",
    "refused",
    "citations",
    "latency_ms",
    "prompt_tokens",
    "output_tokens",
    "judge",
    "provider",
    "model",
)
_JSON_COLUMNS = {"user_tags", "retrieved_ids", "tool_calls", "citations", "judge"}


def _dump(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


class InferenceLog:
    """Writes to and reads from the `inference_log` table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # ------------------------------------------------------------------ write

    def record(self, **fields: Any) -> int:
        row = {}
        for col in _COLUMNS:
            if col not in fields:
                continue
            value = fields[col]
            if col in _JSON_COLUMNS:
                value = _dump(value)
            elif col == "refused":
                value = 1 if value else 0
            row[col] = value
        if "request_id" not in row or "mode" not in row or "question" not in row:
            raise ValueError("record() needs request_id, mode and question")
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        cur = self.conn.execute(
            f"INSERT INTO inference_log ({cols}) VALUES ({marks}) "
            "ON CONFLICT(request_id) DO UPDATE SET "
            + ", ".join(f"{c}=excluded.{c}" for c in row if c != "request_id"),
            tuple(row.values()),
        )
        return int(cur.lastrowid or 0)

    def attach_judge(self, request_id: str, judge: dict[str, Any]) -> None:
        self.conn.execute(
            "UPDATE inference_log SET judge=? WHERE request_id=?",
            (json.dumps(judge, ensure_ascii=False, default=str), request_id),
        )

    # ------------------------------------------------------------------- read

    def recent(self, limit: int = 50, mode: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM inference_log"
        args: tuple[Any, ...] = ()
        if mode:
            sql += " WHERE mode=?"
            args = (mode,)
        sql += " ORDER BY id DESC LIMIT ?"
        rows = self.conn.execute(sql, (*args, int(limit))).fetchall()
        return [self._inflate(r) for r in rows]

    def get(self, request_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM inference_log WHERE request_id=?", (request_id,)).fetchone()
        return self._inflate(row) if row else None

    def daily_summary(self, days: int = 14) -> list[dict[str, Any]]:
        """Per-day counts, refusal rate, median latency and mean judge groundedness when present."""
        rows = self.conn.execute(
            """
            SELECT substr(ts,1,10) AS day,
                   COUNT(*) AS requests,
                   SUM(refused) AS refused,
                   AVG(latency_ms) AS avg_latency_ms,
                   AVG(CASE WHEN judge IS NOT NULL THEN json_extract(judge,'$.groundedness') END) AS groundedness,
                   AVG(CASE WHEN judge IS NOT NULL THEN json_extract(judge,'$.relevance') END) AS relevance
            FROM inference_log
            WHERE ts >= datetime('now', ?)
            GROUP BY day ORDER BY day
            """,
            (f"-{int(days)} days",),
        ).fetchall()
        return [dict(r) for r in rows]

    def totals(self) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT COUNT(*) AS requests, COALESCE(SUM(refused),0) AS refused, "
            "COALESCE(AVG(latency_ms),0) AS avg_latency_ms, COALESCE(SUM(prompt_tokens),0) AS prompt_tokens, "
            "COALESCE(SUM(output_tokens),0) AS output_tokens FROM inference_log"
        ).fetchone()
        return dict(row) if row else {}

    @staticmethod
    def _inflate(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        for col in _JSON_COLUMNS:
            v = d.get(col)
            if isinstance(v, str):
                try:
                    d[col] = json.loads(v)
                except json.JSONDecodeError:
                    pass
        d["refused"] = bool(d.get("refused"))
        return d
