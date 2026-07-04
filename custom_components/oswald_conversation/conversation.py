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

            return (
                f"homeassistant:satellite:{user_input.satellite_id}",
                display_name,
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

        try:
            async with session.ws_connect(self.ws_url) as ws:
                await ws.send_json(payload)

                async for msg in ws:
                    if msg.type == WSMsgType.TEXT:
                        delta = self._parse_ws_message(msg.data, state)
                        if delta is not None:
                            yield delta
                        if state["done"]:
                            break

                    if msg.type in (WSMsgType.CLOSED, WSMsgType.ERROR):
                        break

        except ClientError as err:
            _LOGGER.warning("Failed to contact Oswald websocket: %s", err)
            fallback = "I could not reach Oswald."
            state["final_response"] = fallback
            state["streamed_content"] = True
            yield {"content": fallback}
            return

        if state["streamed_content"]:
            return

        if state["final_response"]:
            state["streamed_content"] = True
            yield {"content": state["final_response"]}
            return

        fallback = "I could not reach Oswald."
        state["final_response"] = fallback
        state["streamed_content"] = True
        yield {"content": fallback}

    def _parse_ws_message(
        self,
        raw: str,
        state: dict[str, Any],
    ) -> dict[str, Any] | None:
        try:
            data: Any = json.loads(raw)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        msg_type = data.get("type")

        if msg_type in {"content", "thinking"}:
            text = data.get("text")
            if not isinstance(text, str) or not text:
                return None

            if msg_type == "content":
                state["streamed_text"] += text
                state["streamed_content"] = True
                return {"content": text}

            return {"thinking_content": text}

        if msg_type in {"status", "tool_call", "tool_result"}:
            return None

        error = data.get("error")
        if isinstance(error, str) and error.strip():
            state["final_response"] = error.strip()
            state["streamed_content"] = True
            state["done"] = True
            return {"content": error.strip()}

        response = data.get("response")
        if isinstance(response, str) and response.strip():
            state["final_response"] = response.strip()
            state["done"] = True
            return None

        return None
