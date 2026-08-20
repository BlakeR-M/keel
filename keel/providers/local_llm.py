"""OpenAI-compatible chat provider. Serves llama-server locally and, given a prebuilt AzureOpenAI
client, the Azure profile as well.

Messages, tools and JSON-schema output are mapped onto the chat completions API. Tool call
arguments come back JSON-decoded; usage counts and the model name ride along on ChatResult.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import openai

from keel.providers.base import ChatMessage, ChatResult, ToolSpec

log = logging.getLogger("keel.providers.local_llm")

JSON_FALLBACK_INSTRUCTION = (
    "Reply with a single JSON object and nothing else. The object must match this JSON schema:\n{schema}"
)


class OpenAICompatibleLLM:
    """LLMProvider over any OpenAI-compatible chat completions endpoint.

    Pass `base_url` and `model` for llama-server, or pass a prebuilt `client` (for example an
    `openai.AzureOpenAI` built with DefaultAzureCredential) and the deployment name as `model`.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str = "",
        api_key: str = "local",
        timeout: float = 120,
        *,
        client: Any | None = None,
        name: str = "openai-compatible",
    ) -> None:
        if client is None:
            client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._client = client
        self.model = model
        self.name = name
        self.base_url = base_url
        self.json_schema_fallback = False
        """True once the server rejected `response_format=json_schema` and the prompt carries the schema."""

    # ------------------------------------------------------------------ public

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> ChatResult:
        """Send one chat turn and return the assistant reply with any tool calls parsed."""
        api_messages = [to_api_message(m) for m in messages]
        kwargs: dict[str, Any] = {"model": self.model, "messages": api_messages}
        if tools:
            kwargs["tools"] = [to_api_tool(t) for t in tools]
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if json_schema is not None and not self.json_schema_fallback:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "out", "schema": json_schema},
            }
            try:
                response = self._client.chat.completions.create(**kwargs)
            except (openai.BadRequestError, openai.UnprocessableEntityError) as exc:
                log.warning("server rejected response_format=json_schema (%s); using prompt fallback", exc)
                self.json_schema_fallback = True
                kwargs.pop("response_format")
                kwargs["messages"] = with_json_instruction(api_messages, json_schema)
                response = self._client.chat.completions.create(**kwargs)
        else:
            if json_schema is not None:
                kwargs["messages"] = with_json_instruction(api_messages, json_schema)
            response = self._client.chat.completions.create(**kwargs)

        return parse_response(response, default_model=self.model)

    def healthy(self) -> bool:
        """True when GET /models succeeds."""
        try:
            self._client.models.list()
            return True
        except Exception as exc:  # any transport or auth failure counts as unhealthy
            log.debug("health check failed: %s", exc)
            return False


# ---------------------------------------------------------------------- mapping helpers


def to_api_message(message: ChatMessage) -> dict[str, Any]:
    """Convert a ChatMessage into the chat completions message dict."""
    out: dict[str, Any] = {"role": message.role, "content": message.content or ""}
    if message.name and message.role != "tool":
        out["name"] = message.name
    if message.role == "tool":
        out["tool_call_id"] = message.tool_call_id or ""
    if message.role == "assistant" and message.tool_calls:
        out["tool_calls"] = [to_api_tool_call(tc) for tc in message.tool_calls]
    return out


def to_api_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    """Convert {id, name, arguments} into the API's function tool call shape. Dicts already in
    API shape (carrying a `function` key) pass through unchanged."""
    if "function" in call:
        return call
    arguments = call.get("arguments")
    if not arguments and call.get("arguments_raw") is not None:
        arguments_str = str(call["arguments_raw"])
    else:
        arguments_str = json.dumps(arguments or {})
    return {
        "id": call.get("id") or "",
        "type": "function",
        "function": {"name": call.get("name", ""), "arguments": arguments_str},
    }


def to_api_tool(spec: ToolSpec) -> dict[str, Any]:
    """Convert a ToolSpec into an OpenAI function tool definition."""
    return {
        "type": "function",
        "function": {"name": spec.name, "description": spec.description, "parameters": spec.parameters},
    }


def with_json_instruction(api_messages: list[dict[str, Any]], schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a copy of the messages whose system prompt carries the JSON schema instruction."""
    instruction = JSON_FALLBACK_INSTRUCTION.format(schema=json.dumps(schema))
    messages = [dict(m) for m in api_messages]
    for m in messages:
        if m["role"] == "system":
            m["content"] = f"{m['content']}\n\n{instruction}".strip()
            return messages
    return [{"role": "system", "content": instruction}, *messages]


def parse_response(response: Any, *, default_model: str = "") -> ChatResult:
    """Turn a chat completion response (SDK object or plain dict) into a ChatResult."""
    data = response.model_dump() if hasattr(response, "model_dump") else response
    choices = data.get("choices") or []
    message: dict[str, Any] = (choices[0].get("message") or {}) if choices else {}
    content = message.get("content") or ""
    tool_calls = [parse_tool_call(tc, i) for i, tc in enumerate(message.get("tool_calls") or [])]
    usage = data.get("usage") or {}
    return ChatResult(
        content=content,
        tool_calls=tool_calls,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
        model=data.get("model") or default_model,
        raw=response,
    )


def parse_tool_call(tc: dict[str, Any], index: int) -> dict[str, Any]:
    """Return {id, name, arguments} with arguments JSON-decoded. Undecodable arguments are kept
    verbatim under `arguments_raw` with `arguments` set to an empty dict."""
    function = tc.get("function") or {}
    raw = function.get("arguments")
    out: dict[str, Any] = {"id": tc.get("id") or f"call_{index}", "name": function.get("name") or ""}
    if isinstance(raw, dict):
        out["arguments"] = raw
        return out
    try:
        parsed = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        out["arguments"] = parsed
    else:
        out["arguments"] = {}
        out["arguments_raw"] = raw
    return out
