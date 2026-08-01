from __future__ import annotations

import asyncio

import voluptuous as vol
from aiohttp import ClientError, WSServerHandshakeError
from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import CONF_AUTH_TOKEN, CONF_WS_URL, DEFAULT_WS_URL, DOMAIN
from .protocol import (
    CONNECTION_TIMEOUT_SECONDS,
    UnsupportedProtocolError,
    expect_ready,
    is_valid_ws_url,
)

AUTH_TOKEN_SELECTOR = selector.TextSelector(
    selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
)


class OswaldConversationConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input: dict | None = None) -> FlowResult:
        errors: dict[str, str] = {}

        if user_input is not None:
            ws_url = user_input[CONF_WS_URL].strip()
            auth_token = user_input[CONF_AUTH_TOKEN].strip()

            if not is_valid_ws_url(ws_url):
                errors[CONF_WS_URL] = "invalid_ws_url"
            else:
                error = await self._async_validate_connection(ws_url, auth_token)
                if error is not None:
                    errors["base"] = error
                else:
                    await self.async_set_unique_id("oswald_conversation")
                    self._abort_if_unique_id_configured()

                    return self.async_create_entry(
                        title="Oswald Conversation",
                        data={
                            CONF_WS_URL: ws_url,
                            CONF_AUTH_TOKEN: auth_token,
                        },
                    )

        schema = vol.Schema(
            {
                vol.Required(CONF_WS_URL, default=DEFAULT_WS_URL): str,
                vol.Required(CONF_AUTH_TOKEN): AUTH_TOKEN_SELECTOR,
            }
        )

        return self.async_show_form(
            step_id="user",
            data_schema=schema,
            errors=errors,
        )

    async def async_step_reconfigure(
        self, user_input: dict | None = None
    ) -> FlowResult:
        entry = self._get_reconfigure_entry()
        errors: dict[str, str] = {}

        if user_input is not None:
            ws_url = user_input[CONF_WS_URL].strip()
            auth_token = user_input[CONF_AUTH_TOKEN].strip()

            if not is_valid_ws_url(ws_url):
                errors[CONF_WS_URL] = "invalid_ws_url"
            else:
                error = await self._async_validate_connection(ws_url, auth_token)
                if error is not None:
                    errors["base"] = error
                else:
                    return self.async_update_reload_and_abort(
                        entry,
                        data_updates={
                            CONF_WS_URL: ws_url,
                            CONF_AUTH_TOKEN: auth_token,
                        },
                    )

        schema = vol.Schema(
            {
                vol.Required(
                    CONF_WS_URL,
                    default=entry.data.get(CONF_WS_URL, DEFAULT_WS_URL),
                ): str,
                vol.Required(CONF_AUTH_TOKEN): AUTH_TOKEN_SELECTOR,
            }
        )

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=schema,
            errors=errors,
        )

    async def _async_validate_connection(
        self, ws_url: str, auth_token: str
    ) -> str | None:
        if not auth_token:
            return "invalid_auth"

        try:
            async with asyncio.timeout(CONNECTION_TIMEOUT_SECONDS):
                async with async_get_clientsession(self.hass).ws_connect(
                    ws_url,
                    headers={"Authorization": f"Bearer {auth_token}"},
                ) as ws:
                    await expect_ready(ws)
        except WSServerHandshakeError as err:
            if err.status == 401:
                return "invalid_auth"
            return "cannot_connect"
        except UnsupportedProtocolError:
            return "unsupported_protocol"
        except (ClientError, asyncio.TimeoutError, OSError):
            return "cannot_connect"

        return None
