from __future__ import annotations

import json
import logging
from typing import Any

from aiohttp import ClientError, WSMsgType
from homeassistant.components import conversation
from homeassistant.components.conversation import (
    AssistantContent,
    ChatLog,
    ConversationEntity,
    ConversationInput,
    ConversationResult,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
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

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
    ) -> ConversationResult:
        response_text = await self._send_to_oswald(user_input)

        chat_log.async_add_assistant_content_without_tools(
            AssistantContent(
                agent_id=user_input.agent_id,
                content=response_text,
            )
        )

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response_text)

        return ConversationResult(
            conversation_id=user_input.conversation_id,
            response=intent_response,
            continue_conversation=False,
        )

    async def _send_to_oswald(self, user_input: ConversationInput) -> str:
        session = async_get_clientsession(self.hass)

        ha_user_id = user_input.context.user_id or "unknown"
        payload = {
            "user_id": f"homeassistant:{ha_user_id}",
            "display_name": "Home Assistant",
            "prompt": user_input.text,
        }

        try:
            async with session.ws_connect(self.ws_url) as ws:
                await ws.send_json(payload)

                async for msg in ws:
                    if msg.type == WSMsgType.TEXT:
                        parsed = self._parse_ws_message(msg.data)
                        if parsed is not None:
                            return parsed

                    if msg.type in (WSMsgType.CLOSED, WSMsgType.ERROR):
                        break

        except ClientError as err:
            _LOGGER.warning("Failed to contact Oswald websocket: %s", err)

        return "I could not reach Oswald."

    def _parse_ws_message(self, raw: str) -> str | None:
        try:
            data: Any = json.loads(raw)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        msg_type = data.get("type")

        # Oswald sends these while generating. Home Assistant needs the final response.
        if msg_type in {"thinking", "content", "status", "tool_call", "tool_result"}:
            return None

        error = data.get("error")
        if isinstance(error, str) and error.strip():
            return error.strip()

        response = data.get("response")
        if isinstance(response, str) and response.strip():
            return response.strip()

        return None
