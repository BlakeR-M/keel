"""The approval queue. Write tool calls wait here as `pending` rows; a person approves or rejects,
and an approved call is executed on request with its result stored on the row.

Execution re-checks the live policy (the one passed to `execute()`, or the one bound to the registry)
before the tool runs: an approval may sit for hours, and a tool dropped from the allowlist, an argument
rule tightened or air-gap switched on in the meantime refuses the call, which is recorded as its result.

Backed by the `approvals` table in keel/db.py. Ledger hook is duck-typed like the answer engine's:
`ledger.append(kind, request_id, payload)`.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from keel.agent.tools import ToolContext, ToolError, ToolRegistry

STATUSES = ("pending", "approved", "rejected", "executed")


class ApprovalQueue:
    """Queue of write tool calls awaiting a human decision."""

    def __init__(self, conn: sqlite3.Connection, ledger: Any = None) -> None:
        self.conn = conn
        self.ledger = ledger

    def enqueue(self, request_id: str, tool: str, arguments: dict[str, Any]) -> int:
        """Add a pending call and return its id."""
        cursor = self.conn.execute(
            "INSERT INTO approvals (request_id, tool, arguments) VALUES (?, ?, ?)",
            (request_id, tool, json.dumps(arguments, sort_keys=True, default=str)),
        )
        approval_id = int(cursor.lastrowid)
        self._ledger("approval", request_id, {"id": approval_id, "tool": tool, "arguments": arguments, "status": "pending"})
        return approval_id

    def get(self, approval_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,)).fetchone()
        return _row_to_dict(row) if row else None

    def list(self, status: str | None = None) -> list[dict[str, Any]]:
        """Rows oldest first, optionally filtered by status."""
        if status is None:
            rows = self.conn.execute("SELECT * FROM approvals ORDER BY id").fetchall()
        else:
            if status not in STATUSES:
                raise ValueError(f"status must be one of {', '.join(STATUSES)}")
            rows = self.conn.execute("SELECT * FROM approvals WHERE status = ? ORDER BY id", (status,)).fetchall()
        return [_row_to_dict(r) for r in rows]

    def decide(self, approval_id: int, approved: bool, by: str) -> dict[str, Any]:
        """Approve or reject a pending call. Returns the updated row."""
        row = self._require(approval_id, "pending")
        status = "approved" if approved else "rejected"
        self.conn.execute(
            "UPDATE approvals SET status = ?, decided_at = strftime('%Y-%m-%dT%H:%M:%fZ','now'), decided_by = ? "
            "WHERE id = ?",
            (status, by, approval_id),
        )
        self._ledger(
            "approval",
            row["request_id"],
            {"id": approval_id, "tool": row["tool"], "status": status, "decided_by": by},
        )
        return self._require(approval_id, status)

    def execute(
        self,
        approval_id: int,
        registry: ToolRegistry,
        context: ToolContext | None = None,
        *,
        policy: Any = None,
    ) -> str:
        """Run an approved call through the registry, store the result and mark it executed. The call
        is checked against `policy` (or the policy bound to `registry`) first; a refusal is stored as
        the result and nothing runs."""
        row = self._require(approval_id, "approved")
        live_policy = policy if policy is not None else getattr(registry, "policy", None)
        decision = live_policy.check(row["tool"], row["arguments"]) if live_policy is not None else None
        if decision is not None and not decision.allowed:
            result = f"refused: {decision.reason}"
        else:
            try:
                result = registry.call(row["tool"], row["arguments"], context)
            except ToolError as exc:
                result = f"error: {exc}"
        self.conn.execute("UPDATE approvals SET status = 'executed', result = ? WHERE id = ?", (result, approval_id))
        self._ledger(
            "approval",
            row["request_id"],
            {"id": approval_id, "tool": row["tool"], "status": "executed", "result": result},
        )
        return result

    # ------------------------------------------------------------------ internals

    def _require(self, approval_id: int, status: str) -> dict[str, Any]:
        row = self.get(approval_id)
        if row is None:
            raise ValueError(f"approval {approval_id} does not exist")
        if row["status"] != status:
            raise ValueError(f"approval {approval_id} is {row['status']}; this step needs it to be {status}")
        return row

    def _ledger(self, kind: str, request_id: str, payload: dict[str, Any]) -> None:
        if self.ledger is not None:
            self.ledger.append(kind, request_id, payload)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    out = dict(row)
    try:
        out["arguments"] = json.loads(out["arguments"]) if out.get("arguments") else {}
    except ValueError:
        out["arguments"] = {}
    return out
