from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from typing import Any
from uuid import uuid4

from aiohttp import ClientError, ClientWSTimeout, WSMsgType, WSServerHandshakeError
from homeassistant.components import conversation
from homeassistant.components.conversation import (
    ChatLog,
    ConversationEntity,
    ConversationInput,
    ConversationResult,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent, llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_AUTH_TOKEN, CONF_DEFAULT_USER_ID, CONF_WS_URL
from .protocol import CONNECTION_TIMEOUT_SECONDS, UnsupportedProtocolError, expect_ready

_LOGGER = logging.getLogger(__name__)
_FAILURE_RESPONSE = "I could not reach Oswald."
_INTERRUPTED_RESPONSE = (
    "The connection to Oswald was interrupted. The request may still be running "
    "and actions may have completed. Send /stop in this conversation to stop "
    "active work; do not repeat the request unless you have checked its outcome."
)
_AUTH_RESPONSE = (
    "Oswald rejected the authentication token. Reauthenticate the integration."
)
_MISSING_USER_RESPONSE = "Oswald requires an authenticated Home Assistant user."
_MISSING_CONVERSATION_RESPONSE = "Oswald could not identify this conversation."


class ProtocolFrameError(Exception):
    """Raised when an Oswald response frame violates protocol v1."""


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    async_add_entities([OswaldConversationEntity(hass, entry)])


class OswaldConversationEntity(ConversationEntity):
    _attr_has_entity_name = True
    _attr_name = "Oswald"
    _attr_supports_streaming = True

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self.entry = entry
        self.ws_url = entry.data[CONF_WS_URL]
        self.auth_token = entry.data[CONF_AUTH_TOKEN]
        self.default_user_id = entry.data.get(CONF_DEFAULT_USER_ID)
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
        if not user_input.context.user_id and not self.default_user_id:
            return self._result(user_input, conversation_id, _MISSING_USER_RESPONSE)
        if not conversation_id:
            return self._result(
                user_input, conversation_id, _MISSING_CONVERSATION_RESPONSE
            )

        state: dict[str, Any] = {
            "final_response": None,
            "done": False,
            "needs_assistant_role": False,
            "pending_tools": [],
        }

        async for _content in chat_log.async_add_delta_content_stream(
            user_input.agent_id,
            self._async_oswald_delta_stream(user_input, conversation_id, state),
        ):
            pass

        response_text = state["final_response"] or _FAILURE_RESPONSE
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

    async def _async_oswald_delta_stream(
        self,
        user_input: ConversationInput,
        conversation_id: str,
        state: dict[str, Any],
    ) -> AsyncGenerator[dict[str, Any], None]:
        context_user_id = user_input.context.user_id
        user_id = context_user_id or self.default_user_id
        if user_id is None:
            raise RuntimeError("user identity was not validated")

        ha_user = await self.hass.auth.async_get_user(user_id)
        if ha_user is None or (
            context_user_id is None
            and (not ha_user.is_active or ha_user.system_generated)
        ):
            yield {"role": "assistant", "content": _MISSING_USER_RESPONSE}
            state["final_response"] = _MISSING_USER_RESPONSE
            return
        display_name = ha_user.name or "Home Assistant User"

        request_id = str(uuid4())
        payload = {
            "type": "conversation",
            "request_id": request_id,
            "user_id": user_id,
            "display_name": display_name,
            "conversation_id": conversation_id,
            "text": user_input.text,
        }

        yield {"role": "assistant"}

        submitted = False
        try:
            async with AsyncExitStack() as stack:
                async with asyncio.timeout(CONNECTION_TIMEOUT_SECONDS):
                    ws = await stack.enter_async_context(
                        async_get_clientsession(self.hass).ws_connect(
                            self.ws_url,
                            headers={"Authorization": f"Bearer {self.auth_token}"},
                            autoping=True,
                            timeout=ClientWSTimeout(ws_receive=None, ws_close=10),
                        )
                    )
                    await expect_ready(ws)
                    # Even a failed send can have reached the server. Never retry it.
                    submitted = True
                    await ws.send_json(payload)

                async for msg in ws:
                    if msg.type != WSMsgType.TEXT:
                        raise ProtocolFrameError
                    delta = self._parse_ws_message(msg.data, request_id, state)
                    if delta is not None:
                        yield delta
                    if state["done"]:
                        break

                if not state["done"]:
                    raise ProtocolFrameError
        except (
            ClientError,
            asyncio.TimeoutError,
            OSError,
            ProtocolFrameError,
            UnsupportedProtocolError,
        ) as err:
            _LOGGER.warning(
                "Oswald conversation request failed: error_type=%s",
                type(err).__name__,
            )
            if isinstance(err, WSServerHandshakeError) and err.status == 401:
                self.entry.async_start_reauth(self.hass)
                state["final_response"] = _AUTH_RESPONSE
            else:
                state["final_response"] = (
                    _INTERRUPTED_RESPONSE if submitted else _FAILURE_RESPONSE
                )

        yield self._assistant_delta(
            state, "content", state["final_response"] or _FAILURE_RESPONSE
        )

    def _parse_ws_message(
        self,
        raw: str,
        request_id: str,
        state: dict[str, Any],
    ) -> dict[str, Any] | None:
        try:
            data: Any = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            raise ProtocolFrameError from None
        if not isinstance(data, dict):
            raise ProtocolFrameError

        msg_type = data.get("type")
        if data.get("request_id") != request_id and not (
            "request_id" not in data
            and msg_type == "error"
            and data.get("code") == "invalid_request"
        ):
            raise ProtocolFrameError
        if msg_type in {"content", "thinking"}:
            text = data.get("text")
            if not isinstance(text, str) or not text:
                raise ProtocolFrameError
            if msg_type == "content":
                # HA deltas are append-only; previews can differ from result.response.
                # Only the terminal answer is safe to publish to streaming TTS.
                return None
            return self._assistant_delta(state, "thinking_content", text)

        if msg_type == "tool_call":
            return self._parse_tool_call(data, state)
        if msg_type == "tool_result":
            return self._parse_tool_result(data, state)
        if msg_type == "result":
            response = data.get("response", "")
            if not isinstance(response, str):
                raise ProtocolFrameError
            state["final_response"] = response
            state["done"] = True
            return None
        if msg_type == "error":
            code = data.get("code")
            message = data.get("message")
            if (
                not isinstance(code, str)
                or not code
                or not isinstance(message, str)
                or not message
            ):
                raise ProtocolFrameError
            state["final_response"] = message
            state["done"] = True
            return None

        raise ProtocolFrameError

    @staticmethod
    def _assistant_delta(state: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
        delta: dict[str, Any] = {key: value}
        if state["needs_assistant_role"]:
            state["needs_assistant_role"] = False
            delta["role"] = "assistant"
        return delta

    def _parse_tool_call(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        tool = data.get("tool")
        if not isinstance(tool, dict):
            raise ProtocolFrameError
        tool_name = tool.get("name")
        arguments = tool.get("arguments") or {}
        if (
            not isinstance(tool_name, str)
            or not tool_name
            or not isinstance(arguments, dict)
        ):
            raise ProtocolFrameError

        tool_input = llm.ToolInput(
            tool_name=tool_name,
            tool_args=arguments,
            external=True,
        )
        state["pending_tools"].append((tool_input.id, tool_name))
        return self._assistant_delta(state, "tool_calls", [tool_input])

    def _parse_tool_result(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        tool = data.get("tool")
        if not isinstance(tool, dict):
            raise ProtocolFrameError
        tool_name = tool.get("name")
        arguments = tool.get("arguments") or {}
        if (
            not isinstance(tool_name, str)
            or not tool_name
            or not isinstance(arguments, dict)
        ):
            raise ProtocolFrameError

        pending_tools = state["pending_tools"]
        if not pending_tools:
            raise ProtocolFrameError
        tool_call_id, expected_tool_name = pending_tools.pop(0)
        if tool_name != expected_tool_name:
            raise ProtocolFrameError
        tool_result = {
            "name": tool_name,
            "arguments": arguments,
            "result_text": tool.get("result_text"),
            "duration_ms": tool.get("duration_ms"),
            "is_error": tool.get("is_error", False),
        }
        for key in ("web.search", "web.fetch", "user_memory", "global_memory"):
            if key in tool:
                if not isinstance(tool[key], dict):
                    raise ProtocolFrameError
                tool_result[key] = tool[key]
        state["needs_assistant_role"] = True
        return {
            "role": "tool_result",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "tool_result": tool_result,
        }
