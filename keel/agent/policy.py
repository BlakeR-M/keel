"""Policy at the tool boundary. Every tool call passes through Policy.check before it runs:
a per-deployment allowlist, argument rules per tool, and an approval requirement for write tools.

Config shape (dict or YAML):

    allowed_tools: [search_docs, calculator, sql_readonly, http_get, create_ticket]
    write_tools_require_approval: true
    tool_arg_rules:
      sql_readonly: {tables: [users, orders], max_rows: 50}
      http_get: {hosts: [example.com]}
    max_tool_calls_per_request: 6
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from keel.agent.tools import AIRGAP_REFUSAL, ToolRegistry, check_select_query, host_problem

log = logging.getLogger("keel.policy")

DEFAULT_MAX_TOOL_CALLS = 6


@dataclass
class Decision:
    """The outcome of a policy check. `needs_approval` marks a call that may run only after a
    person approves it."""

    allowed: bool
    reason: str
    needs_approval: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Policy:
    """Per-deployment tool policy."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        registry: ToolRegistry | None = None,
        airgap: bool = False,
    ) -> None:
        cfg = config or {}
        self.allowed_tools: list[str] = list(cfg.get("allowed_tools") or [])
        self.write_tools_require_approval: bool = bool(cfg.get("write_tools_require_approval", True))
        self.tool_arg_rules: dict[str, dict[str, Any]] = dict(cfg.get("tool_arg_rules") or {})
        self.max_tool_calls_per_request: int = int(
            cfg.get("max_tool_calls_per_request", DEFAULT_MAX_TOOL_CALLS)
        )
        self.registry = registry
        self.airgap = airgap
        if registry is not None:
            # The registry remembers its live policy so an approved call executed later (from the
            # queue, the admin page or the CLI) is checked against the policy in force at that moment.
            registry.policy = self

    @classmethod
    def from_yaml(cls, path: str | Path, **kwargs: Any) -> Policy:
        """Load the policy from a YAML file. Keyword arguments go to the constructor."""
        with open(path, encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
        return cls(config, **kwargs)

    def rules_for(self, tool_name: str) -> dict[str, Any]:
        return dict(self.tool_arg_rules.get(tool_name) or {})

    def check(self, tool_name: str, arguments: dict[str, Any], *, write: bool | None = None) -> Decision:
        """Decide whether `tool_name(arguments)` may run now, must wait for approval, or is refused.
        `write` overrides the registry's write flag when given."""
        if tool_name not in self.allowed_tools:
            return self._deny(tool_name, f"tool '{tool_name}' is outside this deployment's allowlist")

        problem = self._argument_problem(tool_name, arguments)
        if problem:
            return self._deny(tool_name, problem)

        if write is None:
            spec = self.registry.get(tool_name) if self.registry is not None else None
            write = bool(spec and spec.write)
        if write and self.write_tools_require_approval:
            return Decision(True, "write tool: waits for a person to approve", needs_approval=True)
        return Decision(True, "allowed")

    # ------------------------------------------------------------------ internals

    def _argument_problem(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        rules = self.rules_for(tool_name)
        if tool_name == "sql_readonly":
            return check_select_query(str(arguments.get("query", "")), rules.get("tables"))
        if tool_name == "http_get":
            if self.airgap:
                return AIRGAP_REFUSAL
            return host_problem(str(arguments.get("url", "")), rules.get("hosts"))
        return None

    def _deny(self, tool_name: str, reason: str) -> Decision:
        log.warning("policy refused %s: %s", tool_name, reason)
        return Decision(False, reason)
