from __future__ import annotations

import asyncio

import voluptuous as vol
from aiohttp import ClientError, WSServerHandshakeError
from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    CONF_AUTH_TOKEN,
    CONF_DEFAULT_USER_ID,
    CONF_WS_URL,
    DEFAULT_WS_URL,
    DOMAIN,
)
from .protocol import (
    CONNECTION_TIMEOUT_SECONDS,
    UnsupportedProtocolError,
    expect_ready,
    is_valid_ws_url,
)

AUTH_TOKEN_SELECTOR = selector.TextSelector(
    selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
)
_MASKED_AUTH_TOKEN = "******"


class OswaldConversationConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input: dict | None = None) -> FlowResult:
        errors: dict[str, str] = {}

        if user_input is not None:
            ws_url = user_input[CONF_WS_URL].strip()
            auth_token = user_input.get(CONF_AUTH_TOKEN, "").strip()
            default_user_id = user_input.get(CONF_DEFAULT_USER_ID, "")

            if not is_valid_ws_url(ws_url):
                errors[CONF_WS_URL] = "invalid_ws_url"
            if default_user_id and not await self._async_valid_default_user(
                default_user_id
            ):
                errors[CONF_DEFAULT_USER_ID] = "invalid_user"
            if not errors:
                error = await self._async_validate_connection(ws_url, auth_token)
                if error is not None:
                    errors["base"] = error
                else:
                    return self.async_create_entry(
                        title="Oswald Conversation",
                        data={
                            CONF_WS_URL: ws_url,
                            CONF_AUTH_TOKEN: auth_token,
                            CONF_DEFAULT_USER_ID: default_user_id,
                        },
                    )

        schema = await self._async_schema(
            ws_url=DEFAULT_WS_URL,
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
        existing_auth_token = entry.data.get(CONF_AUTH_TOKEN, "").strip()
        errors: dict[str, str] = {}

        if user_input is not None:
            ws_url = user_input[CONF_WS_URL].strip()
            submitted_auth_token = user_input.get(CONF_AUTH_TOKEN, "").strip()
            auth_token = (
                existing_auth_token
                if not submitted_auth_token
                or submitted_auth_token == _MASKED_AUTH_TOKEN
                else submitted_auth_token
            )
            default_user_id = user_input.get(CONF_DEFAULT_USER_ID, "")

            if not is_valid_ws_url(ws_url):
                errors[CONF_WS_URL] = "invalid_ws_url"
            if default_user_id and not await self._async_valid_default_user(
                default_user_id
            ):
                errors[CONF_DEFAULT_USER_ID] = "invalid_user"
            if not errors:
                error = await self._async_validate_connection(ws_url, auth_token)
                if error is not None:
                    errors["base"] = error
                else:
                    return self.async_update_reload_and_abort(
                        entry,
                        data_updates={
                            CONF_WS_URL: ws_url,
                            CONF_AUTH_TOKEN: auth_token,
                            CONF_DEFAULT_USER_ID: default_user_id,
                        },
                    )

        schema = await self._async_schema(
            ws_url=entry.data.get(CONF_WS_URL, DEFAULT_WS_URL),
            default_user_id=entry.data.get(CONF_DEFAULT_USER_ID),
            mask_auth_token=bool(existing_auth_token),
        )

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=schema,
            errors=errors,
        )

    async def async_step_reauth(self, entry_data: dict) -> FlowResult:
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict | None = None
    ) -> FlowResult:
        entry = self._get_reauth_entry()
        errors: dict[str, str] = {}

        if user_input is not None:
            auth_token = user_input.get(CONF_AUTH_TOKEN, "").strip()
            ws_url = entry.data.get(CONF_WS_URL, DEFAULT_WS_URL)
            if not is_valid_ws_url(ws_url):
                errors["base"] = "invalid_ws_url"
            else:
                error = await self._async_validate_connection(ws_url, auth_token)
                if error is not None:
                    errors["base"] = error
                else:
                    return self.async_update_reload_and_abort(
                        entry,
                        data_updates={CONF_AUTH_TOKEN: auth_token},
                        reason="reauth_successful",
                    )

        return self.async_show_form(
            step_id="reauth_confirm",
            data_schema=vol.Schema(
                {vol.Required(CONF_AUTH_TOKEN): AUTH_TOKEN_SELECTOR}
            ),
            errors=errors,
        )

    async def _async_schema(
        self,
        *,
        ws_url: str,
        default_user_id: str | None = None,
        mask_auth_token: bool = False,
    ) -> vol.Schema:
        users = [
            user
            for user in await self.hass.auth.async_get_users()
            if user.is_active and not user.system_generated
        ]
        user_selector = selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=[
                    selector.SelectOptionDict(
                        value=user.id,
                        label=user.name or "Unnamed user",
                    )
                    for user in users
                ],
                mode=selector.SelectSelectorMode.DROPDOWN,
            )
        )
        default_user_key = (
            vol.Optional(
                CONF_DEFAULT_USER_ID,
                description={"suggested_value": default_user_id},
            )
            if default_user_id and any(user.id == default_user_id for user in users)
            else vol.Optional(CONF_DEFAULT_USER_ID)
        )
        auth_token_key = (
            vol.Optional(CONF_AUTH_TOKEN, default=_MASKED_AUTH_TOKEN)
            if mask_auth_token
            else vol.Required(CONF_AUTH_TOKEN)
        )

        return vol.Schema(
            {
                vol.Required(CONF_WS_URL, default=ws_url): str,
                auth_token_key: AUTH_TOKEN_SELECTOR,
                default_user_key: user_selector,
            }
        )

    async def _async_valid_default_user(self, user_id: str) -> bool:
        user = await self.hass.auth.async_get_user(user_id)
        return bool(user and user.is_active and not user.system_generated)

    async def _async_validate_connection(
        self, ws_url: str, auth_token: str
    ) -> str | None:
        if not auth_token or auth_token == _MASKED_AUTH_TOKEN:
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
