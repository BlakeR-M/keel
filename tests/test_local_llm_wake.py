"""A model host that sleeps when idle: calls wait out the wake-up, and page loads start it early."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from keel.providers.base import ChatMessage
from keel.providers.local_llm import OpenAICompatibleLLM

REQUEST = httpx.Request("POST", "http://keel-llm.internal:8081/v1/chat/completions")
REPLY = {
    "model": "qwen2.5-3b-instruct",
    "choices": [{"message": {"role": "assistant", "content": "three written quotes [1]"}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
}


def unreachable() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=REQUEST)


def bad_gateway() -> openai.InternalServerError:
    return openai.InternalServerError("502", response=httpx.Response(502, request=REQUEST), body=None)


def bad_request() -> openai.BadRequestError:
    return openai.BadRequestError("400", response=httpx.Response(400, request=REQUEST), body=None)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class ScriptedClient:
    """Raises each queued exception in turn, then answers."""

    def __init__(self, failures: list[Exception]) -> None:
        self.failures = list(failures)
        self.calls = 0
        self.probes = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.models = SimpleNamespace(list=self._list)

    def _create(self, **_: Any) -> dict[str, Any]:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return REPLY

    def _list(self) -> list[Any]:
        self.probes += 1
        return []


def make(client: ScriptedClient, wake_seconds: float, clock: FakeClock) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        model="qwen2.5-3b-instruct", client=client, wake_seconds=wake_seconds, sleep=clock.sleep, clock=clock
    )


ASK = [ChatMessage(role="user", content="How many quotes?")]


def test_a_waking_server_is_waited_for_and_answers() -> None:
    clock = FakeClock()
    client = ScriptedClient([unreachable(), bad_gateway(), bad_gateway()])
    result = make(client, 90, clock).chat(ASK)
    assert result.content == "three written quotes [1]"
    assert client.calls == 4
    assert clock.sleeps == [1.0, 2.0, 4.0]


def test_the_wait_stops_at_the_wake_window_and_raises() -> None:
    clock = FakeClock()
    client = ScriptedClient([unreachable()] * 50)
    with pytest.raises(openai.APIConnectionError):
        make(client, 20, clock).chat(ASK)
    assert clock.now == pytest.approx(20)
    assert max(clock.sleeps) <= 8.0


def test_no_wake_window_fails_on_the_first_error_as_before() -> None:
    clock = FakeClock()
    client = ScriptedClient([unreachable()])
    with pytest.raises(openai.APIConnectionError):
        make(client, 0, clock).chat(ASK)
    assert client.calls == 1
    assert clock.sleeps == []


def test_a_rejected_request_is_never_retried() -> None:
    clock = FakeClock()
    client = ScriptedClient([bad_request()])
    with pytest.raises(openai.BadRequestError):
        make(client, 90, clock).chat(ASK)
    assert client.calls == 1


def test_nudge_probes_at_most_once_a_minute() -> None:
    clock = FakeClock()
    client = ScriptedClient([])
    llm = make(client, 90, clock)
    assert llm.nudge() is True
    assert llm.nudge() is False
    clock.now += 61
    assert llm.nudge() is True


def test_nudge_does_nothing_without_a_wake_window() -> None:
    clock = FakeClock()
    client = ScriptedClient([])
    assert make(client, 0, clock).nudge() is False
    assert client.probes == 0
