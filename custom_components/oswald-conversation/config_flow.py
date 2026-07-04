from __future__ import annotations

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResult

from .const import CONF_WS_URL, DEFAULT_WS_URL, DOMAIN


class OswaldConversationConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input: dict | None = None) -> FlowResult:
        errors: dict[str, str] = {}

        if user_input is not None:
            ws_url = user_input[CONF_WS_URL].strip()

            if not ws_url.startswith(("ws://", "wss://")):
                errors[CONF_WS_URL] = "invalid_ws_url"
            else:
                await self.async_set_unique_id("oswald_conversation")
                self._abort_if_unique_id_configured()

                return self.async_create_entry(
                    title="Oswald Conversation",
                    data={CONF_WS_URL: ws_url},
                )

        schema = vol.Schema(
            {
                vol.Required(CONF_WS_URL, default=DEFAULT_WS_URL): str,
            }
        )

        return self.async_show_form(
            step_id="user",
            data_schema=schema,
            errors=errors,
        )
