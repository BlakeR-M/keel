"""Agent loop, approval queue and the OpenAI-compatible provider (against a fake client), plus one
integration test against a running llama-server."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest

from keel.agent import AgentLoop, ApprovalQueue, Policy, TicketBook, default_registry
from keel.agent.loop import STEP_LIMIT_TEXT, build_system_prompt
from keel.answer.types import User
from keel.config import Settings
from keel.db import connect
from keel.providers.base import ChatMessage, ToolSpec
from keel.providers.local_llm import OpenAICompatibleLLM
from tests.fakes import FakeLedger, FakeLLM, FakeLog, FakeRetriever, make_hit, tool_call_reply

POLICY_CONFIG = {
    "allowed_tools": ["search_docs", "calculator", "sql_readonly", "http_get", "create_ticket"],
    "write_tools_require_approval": True,
    "tool_arg_rules": {"http_get": {"hosts": ["example.com"]}},
    "max_tool_calls_per_request": 6,
}


@pytest.fixture
def settings() -> Settings:
    return Settings(temperature=0.0, max_output_tokens=200, airgap=False)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = connect(tmp_path / "keel.db")
    yield connection
    connection.close()


@pytest.fixture
def tickets() -> TicketBook:
    return TicketBook()


@pytest.fixture
def registry(settings, tickets):
    retriever = FakeRetriever([make_hit(1, "Band 2 needs three written quotes.", score=0.9)])
    return default_registry(settings, retriever=retriever, http_hosts=["example.com"], tickets=tickets)


def build_loop(llm: FakeLLM, registry, conn, settings, policy_config: dict[str, Any] | None = None):
    policy = Policy(policy_config or POLICY_CONFIG, registry=registry, airgap=settings.airgap)
    ledger, log = FakeLedger(), FakeLog()
    approvals = ApprovalQueue(conn, ledger=ledger)
    loop = AgentLoop(llm, registry, policy, approvals, settings, ledger=ledger, log=log)
    return loop, approvals, ledger, log


# ---------------------------------------------------------------------- agent loop


def test_loop_executes_calculator_then_returns_final_text(registry, conn, settings):
    llm = FakeLLM([tool_call_reply("calculator", {"expression": "2*(3+4)"}, call_id="c1"), "The result is 14."])
    loop, _, ledger, log = build_loop(llm, registry, conn, settings)

    result = loop.run("What is 2*(3+4)?", User("u1", ["public"]))

    assert result.text == "The result is 14."
    assert len(result.steps) == 1
    step = result.steps[0]
    assert step["tool"] == "calculator" and step["arguments"] == {"expression": "2*(3+4)"}
    assert step["decision"] == {"allowed": True, "reason": "allowed", "needs_approval": False}
    assert step["result"] == "14"
    assert result.refused_tools == []
    assert result.prompt_tokens == 20 and result.output_tokens == 10
    assert result.request_id and result.latency_ms >= 0
    assert llm.call_count == 2

    # the model saw the tool result on its second turn, tied to the call id
    second = llm.calls[1]["messages"]
    assert [m.role for m in second] == ["system", "user", "assistant", "tool"]
    assert second[2].tool_calls[0]["name"] == "calculator"
    assert second[3].tool_call_id == "c1" and second[3].content == "14"
    assert [t.name for t in llm.calls[0]["tools"]] == registry.names()

    assert ledger.kinds() == ["request", "tool_call", "tool_result", "answer"]
    assert log.records[0]["mode"] == "agent"
    assert log.records[0]["tool_calls"][0]["tool"] == "calculator"
    assert log.records[0]["answer"] == "The result is 14."


def test_write_tool_is_queued_and_never_executed(registry, conn, settings, tickets):
    llm = FakeLLM(
        [
            tool_call_reply("create_ticket", {"title": "Printer", "body": "Toner is out"}),
            "I have queued a ticket for approval.",
        ]
    )
    loop, approvals, ledger, _ = build_loop(llm, registry, conn, settings)

    result = loop.run("Raise a ticket: the printer needs toner.", User("u1", ["public"]))

    assert result.text == "I have queued a ticket for approval."
    step = result.steps[0]
    assert step["tool"] == "create_ticket"
    assert step["decision"]["allowed"] is True and step["decision"]["needs_approval"] is True
    assert "queued_id" in step and "result" not in step
    pending = approvals.list("pending")
    assert len(pending) == 1
    assert pending[0]["id"] == step["queued_id"]
    assert pending[0]["tool"] == "create_ticket"
    assert pending[0]["arguments"] == {"title": "Printer", "body": "Toner is out"}
    assert pending[0]["request_id"] == result.request_id
    assert tickets.created == []
    assert llm.calls[1]["messages"][-1].content == f"queued for approval (id {step['queued_id']})"
    assert "approval" in ledger.kinds() and "tool_result" not in ledger.kinds()


def test_unknown_tool_is_refused_and_the_loop_continues(registry, conn, settings):
    llm = FakeLLM([tool_call_reply("teleport", {"to": "Mars"}), "Teleporting is unavailable; here is a plain answer."])
    loop, _, ledger, _ = build_loop(llm, registry, conn, settings)

    result = loop.run("Teleport me to Mars", User("u1", ["public"]))

    assert result.text == "Teleporting is unavailable; here is a plain answer."
    assert result.refused_tools == ["teleport"]
    step = result.steps[0]
    assert step["decision"]["allowed"] is False
    assert step["result"] == "refused: tool 'teleport' is outside this deployment's allowlist"
    assert llm.calls[1]["messages"][-1].role == "tool"
    assert llm.calls[1]["messages"][-1].content.startswith("refused:")
    tool_call_entries = [p for k, _, p in ledger.entries if k == "tool_call"]
    assert tool_call_entries[0]["decision"]["allowed"] is False


def test_allowlisted_but_unregistered_tool_is_refused(registry, conn, settings):
    llm = FakeLLM([tool_call_reply("ghost", {}), "done"])
    config = {**POLICY_CONFIG, "allowed_tools": ["calculator", "ghost"]}
    loop, _, _, _ = build_loop(llm, registry, conn, settings, config)
    result = loop.run("q", User("u1", ["public"]))
    assert result.steps[0]["result"] == "refused: tool 'ghost' is unknown"


def test_tool_errors_go_back_to_the_model_as_error_messages(registry, conn, settings):
    llm = FakeLLM([tool_call_reply("calculator", {"expression": "__import__('os')"}), "That expression is out of scope."])
    loop, _, _, _ = build_loop(llm, registry, conn, settings)
    result = loop.run("q", User("u1", ["public"]))
    assert result.steps[0]["result"].startswith("error: ")
    assert result.refused_tools == []
    assert result.text == "That expression is out of scope."


def test_tool_call_budget_is_enforced(registry, conn, settings):
    reply = tool_call_reply("calculator", {"expression": "1+1"}, call_id="a")
    reply.tool_calls.append({"id": "b", "name": "calculator", "arguments": {"expression": "2+2"}})
    llm = FakeLLM([reply, "final"])
    loop, _, _, _ = build_loop(llm, registry, conn, settings, {**POLICY_CONFIG, "max_tool_calls_per_request": 1})
    result = loop.run("q", User("u1", ["public"]))
    assert result.steps[0]["result"] == "2"
    assert result.steps[1]["result"] == "refused: the tool call budget of 1 for this request is used up"
    assert result.refused_tools == ["calculator"]


def test_step_limit_returns_a_plain_message(registry, conn, settings):
    llm = FakeLLM([tool_call_reply("calculator", {"expression": "1+1"})])
    loop, _, _, _ = build_loop(llm, registry, conn, settings)
    result = loop.run("q", User("u1", ["public"]), max_steps=1)
    assert result.text == STEP_LIMIT_TEXT
    assert len(result.steps) == 1


def test_search_docs_runs_under_the_users_tags(registry, conn, settings):
    llm = FakeLLM([tool_call_reply("search_docs", {"query": "quotes"}), "Three quotes [1]."])
    loop, _, _, _ = build_loop(llm, registry, conn, settings)
    result = loop.run("How many quotes?", User("u1", ["public"]))
    assert "[1]" in result.steps[0]["result"] and "three written quotes" in result.steps[0]["result"]


def test_http_get_under_airgap_is_refused_by_policy(registry, conn):
    sealed = Settings(airgap=True)
    llm = FakeLLM([tool_call_reply("http_get", {"url": "https://example.com/"}), "The web is out of reach here."])
    loop, _, _, _ = build_loop(llm, registry, conn, sealed)
    result = loop.run("fetch example.com", User("u1", ["public"]))
    assert result.steps[0]["result"] == "refused: air-gap mode is on: outbound HTTP stays off"


def test_write_tool_without_a_queue_is_refused(registry, settings):
    llm = FakeLLM([tool_call_reply("create_ticket", {"title": "t", "body": "b"}), "no queue"])
    policy = Policy(POLICY_CONFIG, registry=registry)
    loop = AgentLoop(llm, registry, policy, None, settings)
    result = loop.run("q", User("u1", ["public"]))
    assert result.steps[0]["result"].startswith("refused: write tools stay off")


def test_system_prompt_lists_tools_and_marks_write_tools(registry):
    prompt = build_system_prompt(registry.specs())
    assert "- calculator:" in prompt
    assert "- create_ticket (write):" in prompt
    assert "queued for a person to approve" in prompt


# ---------------------------------------------------------------------- approval queue


def test_approval_queue_decide_and_execute(conn, registry, tickets):
    ledger = FakeLedger()
    queue = ApprovalQueue(conn, ledger=ledger)
    approval_id = queue.enqueue("req-1", "create_ticket", {"title": "Printer", "body": "Toner"})
    assert queue.get(approval_id)["status"] == "pending"

    with pytest.raises(ValueError, match="pending"):
        queue.execute(approval_id, registry)

    row = queue.decide(approval_id, approved=True, by="blake")
    assert row["status"] == "approved" and row["decided_by"] == "blake" and row["decided_at"]
    assert queue.list("approved")[0]["id"] == approval_id

    result = queue.execute(approval_id, registry)
    assert result.startswith("ticket created: Printer")
    assert tickets.created == [{"title": "Printer", "body": "Toner"}]
    stored = queue.get(approval_id)
    assert stored["status"] == "executed" and stored["result"] == result
    assert [p["status"] for k, _, p in ledger.entries if k == "approval"] == ["pending", "approved", "executed"]

    with pytest.raises(ValueError):
        queue.decide(approval_id, approved=True, by="blake")
    with pytest.raises(ValueError):
        queue.list("bogus")


def test_approval_queue_reject_path(conn, registry, tickets):
    queue = ApprovalQueue(conn)
    approval_id = queue.enqueue("req-2", "create_ticket", {"title": "t", "body": "b"})
    row = queue.decide(approval_id, approved=False, by="blake")
    assert row["status"] == "rejected"
    with pytest.raises(ValueError, match="rejected"):
        queue.execute(approval_id, registry)
    assert tickets.created == []
    assert queue.get(999) is None
    assert [r["id"] for r in queue.list()] == [approval_id]


# ---------------------------------------------------------------------- OpenAICompatibleLLM against a fake client


def _completion(content: str | None = "", tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "model": "qwen-fake",
        "choices": [{"message": {"role": "assistant", "content": content, "tool_calls": tool_calls}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 42, "completion_tokens": 7},
    }


class _FakeCompletions:
    def __init__(self, responses: list[dict[str, Any]], reject_json_schema: bool = False) -> None:
        self.responses = list(responses)
        self.reject_json_schema = reject_json_schema
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.reject_json_schema and "response_format" in kwargs:
            request = httpx.Request("POST", "http://127.0.0.1:8081/v1/chat/completions")
            raise openai.BadRequestError(
                "response_format is unsupported", response=httpx.Response(400, request=request), body=None
            )
        return self.responses.pop(0)


class _FakeModels:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok

    def list(self) -> list[str]:
        if not self.ok:
            raise openai.APIConnectionError(request=httpx.Request("GET", "http://127.0.0.1:8081/v1/models"))
        return ["qwen-fake"]


class _FakeOpenAI:
    def __init__(self, responses: list[dict[str, Any]], *, reject_json_schema: bool = False, healthy: bool = True) -> None:
        self.completions = _FakeCompletions(responses, reject_json_schema)
        self.chat = type("Chat", (), {"completions": self.completions})()
        self.models = _FakeModels(healthy)


CALC_SPEC = ToolSpec(
    "calculator", "Arithmetic.", {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
)


def test_provider_maps_messages_and_tools_and_parses_tool_calls():
    client = _FakeOpenAI(
        [
            _completion(
                None,
                [
                    {"id": "x1", "type": "function", "function": {"name": "calculator", "arguments": '{"expression": "1+1"}'}},
                    {"id": "x2", "type": "function", "function": {"name": "calculator", "arguments": "not json"}},
                ],
            )
        ]
    )
    llm = OpenAICompatibleLLM(model="qwen2.5-3b-instruct", client=client, name="local")
    messages = [
        ChatMessage("system", "sys"),
        ChatMessage("user", "hi"),
        ChatMessage("assistant", "", tool_calls=[{"id": "p1", "name": "calculator", "arguments": {"expression": "2"}}]),
        ChatMessage("tool", "2", name="calculator", tool_call_id="p1"),
    ]
    result = llm.chat(messages, tools=[CALC_SPEC], temperature=0.2, max_tokens=50)

    sent = client.completions.calls[0]
    assert sent["model"] == "qwen2.5-3b-instruct"
    assert sent["temperature"] == 0.2 and sent["max_tokens"] == 50
    assert sent["messages"][0] == {"role": "system", "content": "sys"}
    assert sent["messages"][2]["tool_calls"] == [
        {"id": "p1", "type": "function", "function": {"name": "calculator", "arguments": '{"expression": "2"}'}}
    ]
    assert sent["messages"][3] == {"role": "tool", "content": "2", "tool_call_id": "p1"}
    assert sent["tools"][0]["type"] == "function" and sent["tools"][0]["function"]["name"] == "calculator"
    assert "response_format" not in sent

    assert result.content == ""
    assert result.tool_calls[0] == {"id": "x1", "name": "calculator", "arguments": {"expression": "1+1"}}
    assert result.tool_calls[1]["arguments"] == {} and result.tool_calls[1]["arguments_raw"] == "not json"
    assert result.prompt_tokens == 42 and result.output_tokens == 7
    assert result.model == "qwen-fake"
    assert llm.name == "local"


def test_provider_requests_json_schema_and_falls_back_when_rejected():
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    strict = _FakeOpenAI([_completion('{"n": 1}')])
    llm = OpenAICompatibleLLM(model="m", client=strict)
    llm.chat([ChatMessage("system", "s"), ChatMessage("user", "u")], json_schema=schema)
    assert strict.completions.calls[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "out", "schema": schema},
    }
    assert llm.json_schema_fallback is False

    lenient = _FakeOpenAI([_completion('{"n": 2}'), _completion('{"n": 3}')], reject_json_schema=True)
    llm = OpenAICompatibleLLM(model="m", client=lenient)
    result = llm.chat([ChatMessage("system", "s"), ChatMessage("user", "u")], json_schema=schema)
    assert result.content == '{"n": 2}'
    assert llm.json_schema_fallback is True
    calls = lenient.completions.calls
    assert "response_format" in calls[0] and "response_format" not in calls[1]
    assert json.dumps(schema) in calls[1]["messages"][0]["content"]
    # later calls skip the rejected response_format straight away
    llm.chat([ChatMessage("user", "u")], json_schema=schema)
    assert "response_format" not in calls[2] and calls[2]["messages"][0]["role"] == "system"


def test_provider_health_reflects_the_models_endpoint():
    assert OpenAICompatibleLLM(model="m", client=_FakeOpenAI([], healthy=True)).healthy() is True
    assert OpenAICompatibleLLM(model="m", client=_FakeOpenAI([], healthy=False)).healthy() is False


def test_provider_builds_a_client_from_base_url():
    llm = OpenAICompatibleLLM("http://127.0.0.1:9/v1", "m", timeout=1)
    assert llm.model == "m" and llm.base_url == "http://127.0.0.1:9/v1"
    assert llm.healthy() is False  # nothing listens on port 9


# ---------------------------------------------------------------------- integration


def _server_up(url: str = "http://127.0.0.1:8081/v1/models") -> bool:
    try:
        return httpx.get(url, timeout=2).status_code == 200
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _server_up(), reason="llama-server is not reachable at http://127.0.0.1:8081/v1")
def test_agent_loop_against_local_llama_server(conn):
    settings = Settings(temperature=0.0, max_output_tokens=200)
    llm = OpenAICompatibleLLM(settings.local_llm_base_url, settings.local_llm_model, timeout=180)
    assert llm.healthy() is True
    registry = default_registry(settings)
    policy = Policy({"allowed_tools": ["calculator"], "max_tool_calls_per_request": 4}, registry=registry)
    loop = AgentLoop(llm, registry, policy, ApprovalQueue(conn), settings)

    result = loop.run("What is 1234*5678? Use the calculator tool.", User("blake", ["public"]))

    assert any(step["tool"] == "calculator" for step in result.steps)
    assert "7006652" in result.text.replace(",", "")
