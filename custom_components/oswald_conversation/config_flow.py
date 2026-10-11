from __future__ import annotations

import asyncio

import voluptuous as vol
from aiohttp import ClientError
from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .client import (
    ConnectionFailedError,
    InvalidAuthError,
    InvalidResponseError,
    async_list_models,
    is_valid_base_url,
    normalize_base_url,
)
from .const import CONF_API_KEY, CONF_BASE_URL, CONF_MODEL, DEFAULT_BASE_URL, DOMAIN

API_KEY_SELECTOR = selector.TextSelector(
    selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
)
_MASKED_API_KEY = "******"


class OswaldConversationConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 2

    def __init__(self) -> None:
        self._base_url: str | None = None
        self._api_key: str | None = None
        self._models: list[str] = []
        self._is_reconfigure = False

    async def async_step_user(self, user_input: dict | None = None) -> FlowResult:
        errors: dict[str, str] = {}
        base_url = DEFAULT_BASE_URL
        api_key: str | None = None

        if user_input is not None:
            base_url = normalize_base_url(user_input.get(CONF_BASE_URL, ""))
            api_key = user_input.get(CONF_API_KEY, "").strip()
            models, error = await self._async_fetch_models(base_url, api_key)
            if error is not None:
                if error == "invalid_url":
                    errors[CONF_BASE_URL] = error
                else:
                    errors["base"] = error
            else:
                self._base_url = base_url
                self._api_key = api_key
                self._models = models
                return await self.async_step_model()

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_BASE_URL, default=base_url): str,
                    vol.Required(CONF_API_KEY, default=api_key or ""): API_KEY_SELECTOR,
                }
            ),
            errors=errors,
        )

    async def async_step_model(self, user_input: dict | None = None) -> FlowResult:
        if self._base_url is None or self._api_key is None or not self._models:
            return await self.async_step_user()

        errors: dict[str, str] = {}
        if user_input is not None:
            model = user_input.get(CONF_MODEL, "")
            if self._is_reconfigure:
                entry = self._get_reconfigure_entry()
                return self.async_update_reload_and_abort(
                    entry,
                    data_updates={
                        CONF_BASE_URL: self._base_url,
                        CONF_API_KEY: self._api_key,
                        CONF_MODEL: model,
                    },
                )
            else:
                return self.async_create_entry(
                    title="Oswald Conversation",
                    data={
                        CONF_BASE_URL: self._base_url,
                        CONF_API_KEY: self._api_key,
                        CONF_MODEL: model,
                    },
                )

        return self.async_show_form(
            step_id="model",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_MODEL): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=self._models,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    )
                }
            ),
            errors=errors,
            description_placeholders={"base_url": self._base_url},
        )

    async def async_step_reconfigure(
        self, user_input: dict | None = None
    ) -> FlowResult:
        entry = self._get_reconfigure_entry()
        existing_api_key = entry.data.get(CONF_API_KEY, "").strip()
        self._is_reconfigure = True
        errors: dict[str, str] = {}
        base_url = entry.data.get(CONF_BASE_URL, DEFAULT_BASE_URL)

        if user_input is not None:
            base_url = normalize_base_url(user_input.get(CONF_BASE_URL, ""))
            submitted_api_key = user_input.get(CONF_API_KEY, "").strip()
            api_key = (
                existing_api_key
                if not submitted_api_key or submitted_api_key == _MASKED_API_KEY
                else submitted_api_key
            )
            models, error = await self._async_fetch_models(base_url, api_key)
            if error is not None:
                if error == "invalid_url":
                    errors[CONF_BASE_URL] = error
                else:
                    errors["base"] = error
            else:
                self._base_url = base_url
                self._api_key = api_key
                self._models = models
                return await self.async_step_model()

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_BASE_URL, default=base_url): str,
                    vol.Optional(
                        CONF_API_KEY,
                        default=_MASKED_API_KEY if existing_api_key else "",
                    ): API_KEY_SELECTOR,
                }
            ),
            errors=errors,
        )

    async def async_step_reauth(self, entry_data: dict) -> FlowResult:
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict | None = None
    ) -> FlowResult:
        entry = self._get_reauth_entry()
        errors: dict[str, str] = {}
        base_url = normalize_base_url(entry.data.get(CONF_BASE_URL, ""))

        if user_input is not None:
            api_key = user_input.get(CONF_API_KEY, "").strip()
            if not is_valid_base_url(base_url):
                errors["base"] = "invalid_url"
            else:
                _, error = await self._async_fetch_models(base_url, api_key)
                if error is not None:
                    errors["base"] = error
                else:
                    return self.async_update_reload_and_abort(
                        entry,
                        data_updates={CONF_API_KEY: api_key},
                        reason="reauth_successful",
                    )

        return self.async_show_form(
            step_id="reauth_confirm",
            data_schema=vol.Schema({vol.Required(CONF_API_KEY): API_KEY_SELECTOR}),
            errors=errors,
            description_placeholders={"base_url": base_url},
        )

    async def _async_fetch_models(
        self, base_url: str, api_key: str
    ) -> tuple[list[str], str | None]:
        if not is_valid_base_url(base_url):
            return [], "invalid_url"
        if not api_key or api_key == _MASKED_API_KEY:
            return [], "invalid_auth"

        try:
            async with asyncio.timeout(15):
                models = await async_list_models(
                    async_get_clientsession(self.hass), base_url, api_key
                )
        except InvalidAuthError:
            return [], "invalid_auth"
        except (
            ConnectionFailedError,
            InvalidResponseError,
            ClientError,
            asyncio.TimeoutError,
            OSError,
        ):
            return [], "cannot_connect"

        if not models:
            return [], "no_models"
        return models, None
