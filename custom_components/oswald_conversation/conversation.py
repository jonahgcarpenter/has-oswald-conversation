from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from aiohttp import ClientError
from homeassistant.components import conversation
from homeassistant.components.conversation import (
    ChatLog,
    ConversationEntity,
    ConversationEntityFeature,
    ConversationInput,
    ConversationResult,
)
from homeassistant.components.conversation.chat_log import (
    AssistantContent,
    SystemContent,
    ToolResultContent,
    UserContent,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent, llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback

try:  # Home Assistant >= 2026.9 exposes probatio instead of voluptuous_openapi.
    from probatio import to_openapi as _schema_to_openapi_impl

    def _schema_to_openapi(schema: Any, custom_serializer: Any = None) -> Any:
        return _schema_to_openapi_impl(schema, custom_serializer=custom_serializer)

except ImportError:  # Home Assistant <= 2026.8 uses voluptuous_openapi.
    from voluptuous_openapi import convert as _schema_to_openapi_impl_legacy

    def _schema_to_openapi(schema: Any, custom_serializer: Any = None) -> Any:
        return _schema_to_openapi_impl_legacy(
            schema, custom_serializer=custom_serializer
        )


from .client import (
    ChatCompletionError,
    ConnectionFailedError,
    InvalidAuthError,
    InvalidResponseError,
    async_stream_chat_completion,
)
from .const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_LLM_HASS_API,
    CONF_MAX_HISTORY,
    CONF_MODEL,
    CONF_OSWALD_TOOLS,
    CONF_PROMPT,
    DEFAULT_LLM_HASS_API,
    DEFAULT_MAX_HISTORY,
    DEFAULT_PROMPT,
    DOMAIN,
)
from .tools import DEFAULT_OSWALD_TOOLS, offerable_builtin_names

_LOGGER = logging.getLogger(__name__)
_FAILURE_RESPONSE = "I could not reach Oswald."
_INTERRUPTED_RESPONSE = (
    "The connection to Oswald was interrupted before it finished answering."
)
_AUTH_RESPONSE = "Oswald rejected the API key. Reauthenticate the integration."
_MISSING_CONVERSATION_RESPONSE = "Oswald could not identify this conversation."

MAX_TOOL_TURNS = 10


def _fix_invalid_argument(value: Any) -> Any:
    """Attempt to repair a tool argument value mangled by a small model.

    Same idea as the Ollama integration: values that look like JSON
    objects or arrays get one decode attempt, anything else passes
    through untouched.
    """
    if not isinstance(value, str):
        return value
    if (value.startswith("[") and value.endswith("]")) or (
        value.startswith("{") and value.endswith("}")
    ):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def _parse_tool_args(raw_arguments: str) -> dict[str, Any]:
    """Parse streamed tool arguments, tolerating small-model mangling.

    Mirrors the Ollama integration's _parse_tool_args: surrounding
    whitespace is ignored, arguments wrapped in an extra string layer
    are unwrapped, per-value JSON is repaired, and empty values are
    omitted. Only truly invalid arguments are rejected.
    """
    try:
        parsed: Any = json.loads(raw_arguments or "{}")
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed.strip())
        except json.JSONDecodeError:
            parsed = None
    if not isinstance(parsed, dict):
        raise InvalidResponseError("Invalid tool arguments")
    return {
        key: _fix_invalid_argument(value)
        for key, value in parsed.items()
        if value is not None and value != ""
    }


_OSWALD_TOOL_NAME_RE = __import__("re").compile(r"^[A-Za-z0-9_.-]{1,128}$")
_OSWALD_TOOL_TEXT_LIMIT = 2000


def _format_oswald_tool_entry(entry: Any) -> str | None:
    """Format one server-side tool timeline entry as details markdown.

    The server streams builtin/MCP activity via the ``oswald_tool_calls``
    delta extension; it must never become HA ``ToolInput`` (HA cannot
    execute server-side tools). Rendering as ``thinking_content`` feeds the
    frontend "Show details" dropdown, mirroring the Ollama integration.
    """
    if not isinstance(entry, dict):
        return None
    name = entry.get("name")
    phase = entry.get("phase")
    if (
        not isinstance(name, str)
        or not _OSWALD_TOOL_NAME_RE.match(name)
        or phase not in ("call", "result")
    ):
        return None
    if phase == "call":
        arguments = entry.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return None
        try:
            encoded = json.dumps(arguments, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return None
        if len(encoded) > _OSWALD_TOOL_TEXT_LIMIT:
            encoded = encoded[:_OSWALD_TOOL_TEXT_LIMIT] + "\u2026"
        return f"\n\n`oswald__{name}`\n```json\n{encoded}\n```\n"
    result = entry.get("result_text", "")
    if not isinstance(result, str):
        return None
    if len(result) > _OSWALD_TOOL_TEXT_LIMIT:
        result = result[:_OSWALD_TOOL_TEXT_LIMIT] + "\u2026"
    truncated = entry.get("result_truncated") is True and result.endswith("\u2026")
    _ = truncated
    if entry.get("is_error") is True:
        return f"\n\n`oswald__{name}` failed\n```\n{result}\n```\n"
    web_search = entry.get("web_search")
    if isinstance(web_search, dict):
        query = web_search.get("query")
        if isinstance(query, str) and query and query not in result:
            return f"\n\n`oswald__{name}`\n```\n{result}\n```\n"
    return f"\n\n`oswald__{name}`\n```\n{result}\n```\n"


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    async_add_entities([OswaldConversationEntity(hass, entry)])


class OswaldConversationEntity(ConversationEntity):
    _attr_has_entity_name = True
    _attr_name = "Oswald"
    _attr_supported_features = ConversationEntityFeature.CONTROL
    _attr_supports_streaming = True

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self.entry = entry
        self.base_url = entry.data[CONF_BASE_URL]
        self.api_key = entry.data[CONF_API_KEY]
        self.model = entry.options.get(CONF_MODEL) or entry.data.get(CONF_MODEL)
        self.prompt = entry.options.get(CONF_PROMPT, DEFAULT_PROMPT)
        configured_apis = entry.options.get(CONF_LLM_HASS_API, DEFAULT_LLM_HASS_API)
        if isinstance(configured_apis, str):
            configured_apis = [configured_apis]
        self.llm_hass_apis = [
            api_id for api_id in configured_apis if isinstance(api_id, str) and api_id
        ]
        try:
            self.max_history = int(
                entry.options.get(CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY)
            )
        except (TypeError, ValueError):
            self.max_history = DEFAULT_MAX_HISTORY
        if self.max_history < 1:
            self.max_history = DEFAULT_MAX_HISTORY
        configured_builtins = entry.options.get(CONF_OSWALD_TOOLS, DEFAULT_OSWALD_TOOLS)
        if isinstance(configured_builtins, str):
            configured_builtins = [configured_builtins]
        # Only names the server offers may be listed; anything else
        # (including image_generate) would make the server reject the turn.
        self.oswald_tools = offerable_builtin_names(
            [name for name in configured_builtins if isinstance(name, str)]
        )
        self._attr_unique_id = f"{entry.entry_id}_conversation"

    @property
    def supported_languages(self) -> list[str] | str:
        return conversation.MATCH_ALL

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
    ) -> ConversationResult:
        conversation_id = chat_log.conversation_id
        if not conversation_id:
            return self._result(
                user_input, conversation_id, _MISSING_CONVERSATION_RESPONSE
            )

        try:
            llm_context = user_input.as_llm_context(DOMAIN)
            await chat_log.async_provide_llm_data(
                llm_context,
                user_llm_hass_api=self.llm_hass_apis or None,
                user_llm_prompt=self.prompt or None,
            )
        except HomeAssistantError as err:
            _LOGGER.warning("Oswald could not prepare conversation tools: %s", err)
            return self._result(user_input, conversation_id, _FAILURE_RESPONSE)

        initial_length = len(chat_log.content)
        received_any = False
        progress_any = False
        response_text = _FAILURE_RESPONSE

        try:
            for _ in range(MAX_TOOL_TURNS):
                messages = self._trim_history(
                    self._messages_from_log(chat_log), self.max_history
                )
                tools_param = self._openai_tools(chat_log)
                payload: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                }
                if tools_param:
                    payload["tools"] = tools_param
                if self.oswald_tools:
                    # Opt the server builtins in; name-only refs, mode auto.
                    # Rebuilt every turn (including tool-result continuations).
                    payload["tool_choice"] = {
                        "type": "allowed_tools",
                        "allowed_tools": {
                            "mode": "auto",
                            "tools": [
                                {"type": "function", "function": {"name": name}}
                                for name in [t["function"]["name"] for t in tools_param]
                                + self.oswald_tools
                            ],
                        },
                    }

                turn_state = {"received": False, "progress": False}

                try:
                    async for _content in chat_log.async_add_delta_content_stream(
                        user_input.agent_id,
                        self._stream_turn(payload, turn_state),
                    ):
                        pass
                finally:
                    if turn_state["received"]:
                        received_any = True
                    if turn_state["progress"]:
                        progress_any = True

                last = chat_log.content[-1]
                if isinstance(last, ToolResultContent):
                    # Tools ran; ask Oswald for a follow-up answer.
                    continue
                if isinstance(last, AssistantContent) and last.tool_calls:
                    # Tool calls without results; do not loop forever.
                    response_text = last.content or _FAILURE_RESPONSE
                    break
                if isinstance(last, AssistantContent):
                    if last.content:
                        response_text = last.content
                    elif last.thinking_content:
                        # Thinking-only turn: the model reasoned without
                        # answering out loud. Keep the reasoning in the log
                        # (it feeds the "Show details" dropdown) and answer
                        # empty, matching HA core result handling.
                        response_text = ""
                    else:
                        response_text = _FAILURE_RESPONSE
                    break
                response_text = _FAILURE_RESPONSE
                break
            else:
                _LOGGER.warning("Oswald exceeded the tool calling limit")
                last = chat_log.content[-1]
                if isinstance(last, AssistantContent) and last.content:
                    response_text = last.content
        except InvalidAuthError:
            self.entry.async_start_reauth(self.hass)
            response_text = _AUTH_RESPONSE
        except ChatCompletionError as err:
            _LOGGER.warning(
                "Oswald conversation request failed: error_type=%s",
                type(err).__name__,
            )
            if err.status is not None:
                # The server answered with an error; report its message.
                response_text = err.message
            elif received_any or len(chat_log.content) > initial_length:
                response_text = _INTERRUPTED_RESPONSE
            else:
                response_text = _FAILURE_RESPONSE
        except (
            ConnectionFailedError,
            ClientError,
            asyncio.TimeoutError,
            OSError,
        ) as err:
            _LOGGER.warning(
                "Oswald conversation request failed: error_type=%s",
                type(err).__name__,
            )
            if received_any or len(chat_log.content) > initial_length:
                response_text = _INTERRUPTED_RESPONSE
            else:
                response_text = _FAILURE_RESPONSE
        except InvalidResponseError as err:
            _LOGGER.warning(
                "Oswald conversation request failed: error_type=%s",
                type(err).__name__,
            )
            if progress_any or len(chat_log.content) > initial_length:
                response_text = _INTERRUPTED_RESPONSE
            else:
                response_text = _FAILURE_RESPONSE

        if len(chat_log.content) > initial_length:
            tail = chat_log.content[-1]
            if (
                isinstance(tail, AssistantContent)
                and not tail.tool_calls
                and tail.content == response_text
            ):
                return self._result(user_input, conversation_id, response_text)

        async for _content in chat_log.async_add_delta_content_stream(
            user_input.agent_id,
            self._single_delta(response_text),
        ):
            pass

        return self._result(user_input, conversation_id, response_text)

    @staticmethod
    def _result(
        user_input: ConversationInput,
        conversation_id: str | None,
        response_text: str,
    ) -> ConversationResult:
        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response_text)
        return ConversationResult(
            conversation_id=conversation_id,
            response=intent_response,
            continue_conversation=False,
        )

    @staticmethod
    async def _single_delta(
        response_text: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        yield {"content": response_text}

    async def _stream_turn(
        self,
        payload: dict[str, Any],
        state: dict[str, bool],
    ) -> AsyncGenerator[dict[str, Any], None]:
        session = async_get_clientsession(self.hass)
        fragments: dict[int, dict[str, Any]] = {}
        yielded_any = False

        async for chunk in async_stream_chat_completion(
            session, self.base_url, self.api_key, payload
        ):
            state["received"] = True
            choices = chunk.get("choices")
            if not isinstance(choices, list):
                raise InvalidResponseError("Missing choices")
            for choice in choices:
                if not isinstance(choice, dict):
                    raise InvalidResponseError("Invalid choice")
                delta = choice.get("delta", {})
                if not isinstance(delta, dict):
                    raise InvalidResponseError("Invalid delta")
                content = delta.get("content")
                if isinstance(content, str) and content:
                    yielded_any = True
                    state["progress"] = True
                    yield {"content": content}
                reasoning = delta.get("reasoning_content")
                if isinstance(reasoning, str) and reasoning:
                    yielded_any = True
                    state["progress"] = True
                    yield {"thinking_content": reasoning}
                oswald_calls = delta.get("oswald_tool_calls", [])
                if oswald_calls:
                    if not isinstance(oswald_calls, list):
                        raise InvalidResponseError("Invalid oswald tool calls")
                    for oswald_entry in oswald_calls:
                        if not isinstance(oswald_entry, dict):
                            raise InvalidResponseError("Invalid oswald tool entry")
                        formatted = _format_oswald_tool_entry(oswald_entry)
                        if formatted:
                            yielded_any = True
                            state["progress"] = True
                            yield {"thinking_content": formatted}
                tool_calls = delta.get("tool_calls", [])
                if not isinstance(tool_calls, list):
                    raise InvalidResponseError("Invalid tool calls")
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        raise InvalidResponseError("Invalid tool call")
                    index = tool_call.get("index", 0)
                    if not isinstance(index, int):
                        raise InvalidResponseError("Invalid tool call index")
                    fragment = fragments.setdefault(
                        index, {"id": None, "name": None, "arguments": []}
                    )
                    if isinstance(tool_call.get("id"), str) and tool_call["id"]:
                        fragment["id"] = tool_call["id"]
                    function = tool_call.get("function", {})
                    if not isinstance(function, dict):
                        raise InvalidResponseError("Invalid tool function")
                    if isinstance(function.get("name"), str) and function["name"]:
                        fragment["name"] = function["name"]
                    arguments = function.get("arguments", "")
                    if not isinstance(arguments, str):
                        raise InvalidResponseError("Invalid tool arguments")
                    if arguments:
                        fragment["arguments"].append(arguments)

        if fragments:
            tool_inputs: list[llm.ToolInput] = []
            for index in sorted(fragments):
                fragment = fragments[index]
                name = fragment["name"]
                if not isinstance(name, str) or not name:
                    raise InvalidResponseError("Missing tool name")
                args = _parse_tool_args("".join(fragment["arguments"]))
                if isinstance(fragment["id"], str) and fragment["id"]:
                    tool_inputs.append(
                        llm.ToolInput(tool_name=name, tool_args=args, id=fragment["id"])
                    )
                else:
                    tool_inputs.append(llm.ToolInput(tool_name=name, tool_args=args))
            yielded_any = True
            state["progress"] = True
            yield {"tool_calls": tool_inputs}

        if not yielded_any:
            raise InvalidResponseError("Empty response")

    def _messages_from_log(self, chat_log: ChatLog) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for item in chat_log.content:
            if isinstance(item, SystemContent):
                if item.content:
                    messages.append({"role": "system", "content": item.content})
            elif isinstance(item, UserContent):
                messages.append({"role": "user", "content": item.content})
            elif isinstance(item, AssistantContent):
                message: dict[str, Any] = {"role": "assistant"}
                message["content"] = item.content or None
                if item.thinking_content:
                    message["reasoning_content"] = item.thinking_content
                if item.tool_calls:
                    message["tool_calls"] = [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.tool_name,
                                "arguments": json.dumps(
                                    tool_call.tool_args, ensure_ascii=False
                                ),
                            },
                        }
                        for tool_call in item.tool_calls
                    ]
                if (
                    message.get("content")
                    or "tool_calls" in message
                    or "reasoning_content" in message
                ):
                    messages.append(message)
            elif isinstance(item, ToolResultContent):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.tool_call_id,
                        "content": json.dumps(
                            item.tool_result, ensure_ascii=False, default=str
                        ),
                    }
                )
        return messages

    @staticmethod
    def _trim_history(
        messages: list[dict[str, Any]], max_history: int
    ) -> list[dict[str, Any]]:
        """Keep system messages plus the last max_history other messages."""
        system = [msg for msg in messages if msg.get("role") == "system"]
        rest = [msg for msg in messages if msg.get("role") != "system"]
        return system + rest[-max_history:]

    @staticmethod
    def _openai_tools(chat_log: ChatLog) -> list[dict[str, Any]]:
        if chat_log.llm_api is None:
            return []
        tools: list[dict[str, Any]] = []
        for tool in chat_log.llm_api.tools:
            try:
                parameters = _schema_to_openapi(
                    tool.parameters,
                    custom_serializer=chat_log.llm_api.custom_serializer,
                )
            except Exception:
                _LOGGER.warning(
                    "Oswald skipping tool with invalid parameters: tool_name=%s",
                    tool.name,
                )
                continue
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": parameters,
                    },
                }
            )
        return tools
