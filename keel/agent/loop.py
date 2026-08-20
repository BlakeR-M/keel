"""The tool-calling agent loop. The model proposes tool calls; every call meets the policy at the
boundary: refused calls go back to the model as `refused: <reason>`, write calls are queued for
approval and reported as `queued for approval (id N)`, and the rest run through the registry.

Ledger and log hooks are duck-typed: `ledger.append(kind, request_id, payload)` and
`log.record(**fields)`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from keel.agent.policy import Decision
from keel.agent.tools import ToolContext, ToolError
from keel.answer.types import User
from keel.providers.base import ChatMessage, ToolSpec

SYSTEM_PROMPT_TEMPLATE = """You are Keel, an assistant that answers questions with the help of tools.

Tools available:
{tool_lines}

Rules:
1. Call a tool when it gives a better answer than memory. Read the result, then answer the user in plain sentences.
2. Tools marked (write) are queued for a person to approve. When a call is queued, tell the user it awaits approval; the action has yet to happen.
3. A tool result that starts with "refused:" or "error:" means that call is unavailable. Say so and answer with what you have.
4. When search_docs returns passages, cite them as [n]."""

STEP_LIMIT_TEXT = "The step limit for this request is reached before a final answer. The tool results so far are recorded in the steps."
NO_QUEUE_REASON = "write tools stay off: this deployment has no approval queue"
TOOL_RESULT_CHARS = 6000


@dataclass
class AgentResult:
    """What one agent run produced. Each step records the tool, its arguments, the policy
    decision, and either the result text or the approval queue id."""

    text: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    refused_tools: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    request_id: str = ""


class AgentLoop:
    """Runs the model against the registry under the policy, up to `max_steps` model turns."""

    def __init__(
        self,
        llm: Any,
        registry: Any,
        policy: Any,
        approvals: Any,
        settings: Any,
        ledger: Any = None,
        log: Any = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.policy = policy
        self.approvals = approvals
        self.settings = settings
        self.ledger = ledger
        self.log = log

    # ------------------------------------------------------------------ public

    def run(self, question: str, user: User, max_steps: int = 6) -> AgentResult:
        """Answer `question` for `user`, calling tools as the model asks, until the model replies
        with plain content or `max_steps` model turns are spent."""
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        context = ToolContext(user=user, request_id=request_id)
        specs: list[ToolSpec] = self.registry.specs()
        self._ledger(
            "request",
            request_id,
            {"mode": "agent", "question": question, "user_id": user.user_id, "user_tags": list(user.tags)},
        )

        result = AgentResult(text="", request_id=request_id)
        messages = [
            ChatMessage("system", build_system_prompt(specs)),
            ChatMessage("user", question),
        ]
        model = ""
        calls_made = 0
        for _ in range(max_steps):
            reply = self.llm.chat(
                messages,
                tools=specs or None,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_output_tokens,
            )
            result.prompt_tokens += reply.prompt_tokens
            result.output_tokens += reply.output_tokens
            model = reply.model or model
            if not reply.tool_calls:
                result.text = (reply.content or "").strip()
                break
            messages.append(ChatMessage("assistant", reply.content or "", tool_calls=reply.tool_calls))
            for index, call in enumerate(reply.tool_calls):
                calls_made += 1
                name = str(call.get("name") or "")
                arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
                content = self._handle_call(name, arguments, calls_made, context, result)
                call_id = str(call.get("id") or f"call_{calls_made}_{index}")
                messages.append(ChatMessage("tool", content, name=name, tool_call_id=call_id))
        else:
            result.text = STEP_LIMIT_TEXT

        result.latency_ms = int((time.perf_counter() - started) * 1000)
        self._ledger(
            "answer",
            request_id,
            {
                "text": result.text,
                "steps": len(result.steps),
                "refused_tools": list(result.refused_tools),
                "prompt_tokens": result.prompt_tokens,
                "output_tokens": result.output_tokens,
                "latency_ms": result.latency_ms,
                "model": model,
            },
        )
        self._record(
            request_id=request_id,
            user_id=user.user_id,
            user_tags=list(user.tags),
            mode="agent",
            question=question,
            retrieved_ids=None,
            tool_calls=list(result.steps),
            answer=result.text,
            refused=False,
            citations=None,
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens,
            output_tokens=result.output_tokens,
            provider=getattr(self.llm, "name", ""),
            model=model,
        )
        return result

    # ------------------------------------------------------------------ one tool call

    def _handle_call(
        self, name: str, arguments: dict[str, Any], calls_made: int, context: ToolContext, result: AgentResult
    ) -> str:
        budget = self.policy.max_tool_calls_per_request
        if calls_made > budget:
            decision = Decision(False, f"the tool call budget of {budget} for this request is used up")
        else:
            decision = self.policy.check(name, arguments)
        if decision.allowed and name not in self.registry:
            decision = Decision(False, f"tool '{name}' is unknown")
        if decision.allowed and decision.needs_approval and self.approvals is None:
            decision = Decision(False, NO_QUEUE_REASON)

        self._ledger(
            "tool_call",
            context.request_id,
            {"tool": name, "arguments": arguments, "decision": decision.to_dict()},
        )
        step: dict[str, Any] = {"tool": name, "arguments": arguments, "decision": decision.to_dict()}

        if not decision.allowed:
            content = f"refused: {decision.reason}"
            result.refused_tools.append(name)
            step["result"] = content
        elif decision.needs_approval:
            queued_id = self.approvals.enqueue(context.request_id, name, arguments)
            content = f"queued for approval (id {queued_id})"
            step["queued_id"] = queued_id
            self._ledger(
                "approval",
                context.request_id,
                {"id": queued_id, "tool": name, "arguments": arguments, "status": "pending"},
            )
        else:
            try:
                content = self.registry.call(name, arguments, context)
            except ToolError as exc:
                content = f"error: {exc}"
            content = content[:TOOL_RESULT_CHARS]
            step["result"] = content
            self._ledger("tool_result", context.request_id, {"tool": name, "result": content})

        result.steps.append(step)
        return content

    # ------------------------------------------------------------------ hooks

    def _ledger(self, kind: str, request_id: str, payload: dict[str, Any]) -> None:
        if self.ledger is not None:
            self.ledger.append(kind, request_id, payload)

    def _record(self, **fields: Any) -> None:
        if self.log is not None:
            self.log.record(**fields)


def build_system_prompt(specs: list[ToolSpec]) -> str:
    """The agent system prompt listing each tool with its description; write tools are marked."""
    lines = [f"- {s.name}{' (write)' if s.write else ''}: {s.description}" for s in specs]
    tool_lines = "\n".join(lines) if lines else "- (none)"
    return SYSTEM_PROMPT_TEMPLATE.format(tool_lines=tool_lines)
