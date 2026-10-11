from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed

from .const import CONF_API_KEY

PLATFORMS: list[Platform] = [Platform.CONVERSATION]

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    if not entry.data.get(CONF_API_KEY, "").strip():
        raise ConfigEntryAuthFailed("An Oswald API key is required")

    entry.async_on_unload(entry.add_update_listener(_async_update_options))
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def _async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    _ = hass
    _LOGGER.warning(
        "Oswald entry %s cannot be migrated from the WebSocket version; "
        "remove and re-add the integration",
        entry.entry_id,
    )
    return False


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
