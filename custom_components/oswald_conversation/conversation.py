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
from .tools import BUILTIN_TOOL_NAMES, DEFAULT_OSWALD_TOOLS, builtin_definitions

_LOGGER = logging.getLogger(__name__)
_FAILURE_RESPONSE = "I could not reach Oswald."
_INTERRUPTED_RESPONSE = (
    "The connection to Oswald was interrupted before it finished answering."
)
_AUTH_RESPONSE = "Oswald rejected the API key. Reauthenticate the integration."
_MISSING_CONVERSATION_RESPONSE = "Oswald could not identify this conversation."

MAX_TOOL_TURNS = 10


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
        self.oswald_tools = [
            name
            for name in configured_builtins
            if isinstance(name, str) and name in BUILTIN_TOOL_NAMES
        ]
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
            llm_context = llm.LLMContext(
                platform=DOMAIN,
                context=user_input.context,
                language=user_input.language,
                assistant=user_input.agent_id,
                device_id=user_input.device_id,
            )
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
                tools_param = self._openai_tools(chat_log) + builtin_definitions(
                    self.oswald_tools
                )
                payload: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                }
                if tools_param:
                    payload["tools"] = tools_param

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
                    response_text = last.content or _FAILURE_RESPONSE
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
                    yield {"thinking_content": reasoning}
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
                try:
                    args: Any = json.loads("".join(fragment["arguments"]) or "{}")
                except (json.JSONDecodeError, TypeError) as err:
                    raise InvalidResponseError("Invalid tool arguments") from err
                if not isinstance(args, dict):
                    raise InvalidResponseError("Invalid tool arguments")
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
                if item.content:
                    message["content"] = item.content
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
                if "content" in message or "tool_calls" in message:
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
