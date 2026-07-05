from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from aiohttp import ClientError, WSMsgType
from homeassistant.components import conversation
from homeassistant.components.conversation import (
    ChatLog,
    ConversationEntity,
    ConversationInput,
    ConversationResult,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import intent
from homeassistant.helpers import llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_WS_URL

_LOGGER = logging.getLogger(__name__)


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
        self._attr_unique_id = f"{entry.entry_id}_conversation"

    @property
    def supported_languages(self) -> list[str] | str:
        return conversation.MATCH_ALL

    async def _async_home_assistant_identity(
        self,
        user_input: ConversationInput,
    ) -> tuple[str, str]:
        if user_input.context.user_id:
            display_name = "Home Assistant"
            ha_user = await self.hass.auth.async_get_user(user_input.context.user_id)
            if ha_user is not None and ha_user.name:
                display_name = ha_user.name

            _LOGGER.debug(
                "Resolved Oswald identity from Home Assistant user: "
                "user_id=%s display_name=%s found=%s",
                user_input.context.user_id,
                display_name,
                ha_user is not None,
            )
            return f"homeassistant:{user_input.context.user_id}", display_name

        if user_input.device_id:
            device_reg = dr.async_get(self.hass)
            device = device_reg.async_get(user_input.device_id)
            display_name = "Home Assistant Device"
            if device is not None:
                display_name = (
                    device.name_by_user
                    or device.name
                    or device.model
                    or display_name
                )

            _LOGGER.debug(
                "Resolved Oswald identity from Home Assistant device: "
                "device_id=%s display_name=%s found=%s",
                user_input.device_id,
                display_name,
                device is not None,
            )
            return (
                f"homeassistant:device:{user_input.device_id}",
                display_name,
            )

        if user_input.satellite_id:
            entity_reg = er.async_get(self.hass)
            entity = entity_reg.async_get(user_input.satellite_id)
            display_name = "Home Assistant Satellite"
            if entity is not None:
                display_name = (
                    entity.name
                    or entity.original_name
                    or entity.entity_id
                    or display_name
                )

            _LOGGER.debug(
                "Resolved Oswald identity from Home Assistant satellite: "
                "satellite_id=%s display_name=%s found=%s",
                user_input.satellite_id,
                display_name,
                entity is not None,
            )
            return (
                f"homeassistant:satellite:{user_input.satellite_id}",
                display_name,
            )

        _LOGGER.debug(
            "Resolved Oswald identity from config entry fallback: entry_id=%s",
            self.entry.entry_id,
        )
        return f"homeassistant:entry:{self.entry.entry_id}", "Home Assistant"

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
    ) -> ConversationResult:
        state = {
            "final_response": None,
            "streamed_text": "",
            "streamed_content": False,
            "done": False,
            "needs_assistant_role": False,
            "pending_tool_call_ids": [],
        }

        async for _content in chat_log.async_add_delta_content_stream(
            user_input.agent_id,
            self._async_oswald_delta_stream(user_input, state),
        ):
            pass

        response_text = (
            state["final_response"]
            or state["streamed_text"]
            or "I could not reach Oswald."
        )

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response_text)

        return ConversationResult(
            conversation_id=user_input.conversation_id,
            response=intent_response,
            continue_conversation=False,
        )

    async def _async_oswald_delta_stream(
        self,
        user_input: ConversationInput,
        state: dict[str, Any],
    ) -> AsyncGenerator[dict[str, Any], None]:
        session = async_get_clientsession(self.hass)

        oswald_user_id, display_name = await self._async_home_assistant_identity(
            user_input
        )
        payload = {
            "user_id": oswald_user_id,
            "display_name": display_name,
            "prompt": user_input.text,
        }

        yield {"role": "assistant"}

        try:
            _LOGGER.debug(
                "Connecting to Oswald websocket: url=%s user_id=%s display_name=%s",
                self.ws_url,
                oswald_user_id,
                display_name,
            )
            async with session.ws_connect(self.ws_url) as ws:
                await ws.send_json(payload)
                _LOGGER.debug(
                    "Sent Oswald websocket request: user_id=%s display_name=%s "
                    "prompt_length=%s",
                    oswald_user_id,
                    display_name,
                    len(user_input.text),
                )

                async for msg in ws:
                    if msg.type == WSMsgType.TEXT:
                        delta = self._parse_ws_message(msg.data, state)
                        if delta is not None:
                            _LOGGER.debug("Yielding Home Assistant delta: %s", delta)
                            yield delta
                        if state["done"]:
                            _LOGGER.debug(
                                "Stopping Oswald websocket stream: terminal frame received"
                            )
                            break

                    if msg.type in (WSMsgType.CLOSED, WSMsgType.ERROR):
                        _LOGGER.debug(
                            "Oswald websocket closed or errored: msg_type=%s",
                            msg.type,
                        )
                        break

        except ClientError as err:
            _LOGGER.warning("Failed to contact Oswald websocket: %s", err)
            fallback = "I could not reach Oswald."
            state["final_response"] = fallback
            state["streamed_content"] = True
            if state["needs_assistant_role"]:
                state["needs_assistant_role"] = False
                yield {"role": "assistant", "content": fallback}
                return

            yield {"content": fallback}
            return

        if state["streamed_content"]:
            _LOGGER.debug(
                "Oswald stream completed with streamed content: final_response=%s "
                "streamed_length=%s",
                state["final_response"] is not None,
                len(state["streamed_text"]),
            )
            return

        if state["final_response"]:
            _LOGGER.debug(
                "Oswald stream had final response without streamed content; "
                "yielding fallback delta: length=%s",
                len(state["final_response"]),
            )
            state["streamed_content"] = True
            if state["needs_assistant_role"]:
                state["needs_assistant_role"] = False
                yield {"role": "assistant", "content": state["final_response"]}
                return

            yield {"content": state["final_response"]}
            return

        fallback = "I could not reach Oswald."
        _LOGGER.debug(
            "Oswald stream ended without content or final response; yielding fallback"
        )
        state["final_response"] = fallback
        state["streamed_content"] = True
        if state["needs_assistant_role"]:
            state["needs_assistant_role"] = False
            yield {"role": "assistant", "content": fallback}
            return

        yield {"content": fallback}

    def _parse_ws_message(
        self,
        raw: str,
        state: dict[str, Any],
    ) -> dict[str, Any] | None:
        try:
            data: Any = json.loads(raw)
        except json.JSONDecodeError:
            _LOGGER.debug("Ignoring non-JSON Oswald websocket frame: %r", raw)
            return None

        if not isinstance(data, dict):
            _LOGGER.debug("Ignoring non-object Oswald websocket frame: %r", data)
            return None

        if _LOGGER.isEnabledFor(logging.DEBUG):
            # Deep diagnostics can include prompts, responses, thinking text,
            # tool arguments, tool results, user IDs, and device/entity IDs.
            _LOGGER.debug("Received Oswald websocket frame: %s", data)

        msg_type = data.get("type")

        if msg_type in {"content", "thinking"}:
            text = data.get("text")
            if not isinstance(text, str) or not text:
                _LOGGER.debug(
                    "Ignoring Oswald %s frame without text content: %s",
                    msg_type,
                    data,
                )
                return None

            if msg_type == "content":
                state["streamed_text"] += text
                state["streamed_content"] = True
                _LOGGER.debug(
                    "Received Oswald content chunk: length=%s streamed_length=%s",
                    len(text),
                    len(state["streamed_text"]),
                )
                if state["needs_assistant_role"]:
                    state["needs_assistant_role"] = False
                    return {"role": "assistant", "content": text}

                return {"content": text}

            _LOGGER.debug("Received Oswald thinking chunk: length=%s", len(text))
            if state["needs_assistant_role"]:
                state["needs_assistant_role"] = False
                return {"role": "assistant", "thinking_content": text}

            return {"thinking_content": text}

        if msg_type == "status":
            _LOGGER.debug("Ignoring Oswald status frame")
            return None

        if msg_type == "tool_call":
            return self._parse_tool_call(data, state)

        if msg_type == "tool_result":
            return self._parse_tool_result(data, state)

        error = data.get("error")
        if isinstance(error, str) and error.strip():
            state["final_response"] = error.strip()
            state["streamed_content"] = True
            state["done"] = True
            _LOGGER.debug(
                "Received terminal Oswald error frame: length=%s",
                len(error.strip()),
            )
            if state["needs_assistant_role"]:
                state["needs_assistant_role"] = False
                return {"role": "assistant", "content": error.strip()}

            return {"content": error.strip()}

        response = data.get("response")
        if isinstance(response, str) and response.strip():
            state["final_response"] = response.strip()
            state["done"] = True
            _LOGGER.debug(
                "Received terminal Oswald final response: length=%s",
                len(response.strip()),
            )
            return None

        _LOGGER.debug(
            "Ignoring unrecognized Oswald websocket frame: type=%s keys=%s",
            msg_type,
            sorted(data),
        )
        return None

    def _parse_tool_call(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any] | None:
        tool = data.get("tool")
        if not isinstance(tool, dict):
            _LOGGER.debug(
                "Ignoring Oswald tool call frame without tool object: %s",
                data,
            )
            return None

        tool_name = tool.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            _LOGGER.debug("Ignoring Oswald tool call frame without tool name: %s", data)
            return None

        arguments = tool.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}

        tool_input = llm.ToolInput(
            tool_name=tool_name,
            tool_args=arguments,
            external=True,
        )
        state["pending_tool_call_ids"].append(tool_input.id)
        _LOGGER.debug(
            "Received Oswald tool call: name=%s id=%s args=%s",
            tool_name,
            tool_input.id,
            arguments,
        )
        return {"tool_calls": [tool_input]}

    def _parse_tool_result(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any] | None:
        tool = data.get("tool")
        if not isinstance(tool, dict):
            _LOGGER.debug(
                "Ignoring Oswald tool result frame without tool object: %s",
                data,
            )
            return None

        tool_name = tool.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            _LOGGER.debug(
                "Ignoring Oswald tool result frame without tool name: %s",
                data,
            )
            return None

        pending_tool_call_ids = state["pending_tool_call_ids"]
        if pending_tool_call_ids:
            tool_call_id = pending_tool_call_ids.pop(0)
        else:
            tool_call_id = llm.ToolInput(
                tool_name=tool_name,
                tool_args={},
                external=True,
            ).id

        tool_result = {
            "name": tool_name,
            "arguments": tool.get("arguments") or {},
            "result_text": tool.get("result_text"),
            "duration_ms": tool.get("duration_ms"),
            "is_error": tool.get("is_error", False),
            "soul": tool.get("soul"),
        }
        _LOGGER.debug(
            "Received Oswald tool result: name=%s id=%s is_error=%s duration_ms=%s",
            tool_name,
            tool_call_id,
            tool_result["is_error"],
            tool_result["duration_ms"],
        )
        state["needs_assistant_role"] = True
        return {
            "role": "tool_result",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "tool_result": tool_result,
        }
