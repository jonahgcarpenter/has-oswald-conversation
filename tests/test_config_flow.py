"""Exercise config flows with real Home Assistant and a mocked WebSocket peer."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import voluptuous as vol
from aiohttp import ClientError, WSMessage, WSMsgType, WSServerHandshakeError
from homeassistant import config_entries
from homeassistant.const import Platform
from homeassistant.data_entry_flow import FlowResultType, InvalidData
from homeassistant.exceptions import ConfigEntryAuthFailed
from homeassistant.helpers.translation import async_get_translations
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.oswald_conversation import async_setup_entry, async_unload_entry
from custom_components.oswald_conversation.const import (
    CONF_AUTH_TOKEN,
    CONF_DEFAULT_USER_ID,
    CONF_WS_URL,
    DEFAULT_WS_URL,
    DOMAIN,
)

pytestmark = pytest.mark.asyncio

DATA = {CONF_WS_URL: DEFAULT_WS_URL, CONF_AUTH_TOKEN: "secret"}
FLOW_MODULE = "custom_components.oswald_conversation.config_flow"


@pytest.fixture(autouse=True)
async def setup_core(hass):
    """Initialize exposed entities before the conversation dependency starts."""
    assert await async_setup_component(hass, "homeassistant", {})


@pytest.fixture
def websocket():
    """Mock only the transport, keeping HA flow and validation code real."""
    ws = AsyncMock()
    ws.receive.return_value = WSMessage(
        WSMsgType.TEXT, '{"type":"ready","protocol_version":1}', ""
    )
    connection = MagicMock()
    connection.__aenter__ = AsyncMock(return_value=ws)
    connection.__aexit__ = AsyncMock(return_value=False)
    with patch(f"{FLOW_MODULE}.async_get_clientsession") as get_session:
        get_session.return_value.ws_connect.return_value = connection
        yield get_session.return_value.ws_connect, connection, ws


@pytest.mark.parametrize(
    "token_data", [{}, {CONF_AUTH_TOKEN: ""}, {CONF_AUTH_TOKEN: "  "}]
)
async def test_legacy_setup_requires_auth(hass, token_data):
    entry = MockConfigEntry(
        domain=DOMAIN, data={CONF_WS_URL: DEFAULT_WS_URL, **token_data}
    )
    entry.add_to_hass(hass)
    with patch.object(hass.config_entries, "async_forward_entry_setups") as forward:
        with pytest.raises(ConfigEntryAuthFailed):
            await async_setup_entry(hass, entry)
        forward.assert_not_called()


async def test_legacy_setup_starts_reauth(hass):
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_WS_URL: DEFAULT_WS_URL})
    entry.add_to_hass(hass)
    assert not await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    flows = hass.config_entries.flow.async_progress()
    assert len(flows) == 1
    assert flows[0]["context"]["source"] == config_entries.SOURCE_REAUTH
    assert flows[0]["context"]["entry_id"] == entry.entry_id
    assert flows[0]["step_id"] == "reauth_confirm"
    assert entry.version == 1


async def test_user_success(hass, websocket):
    connect, _, _ = websocket
    with patch(
        "custom_components.oswald_conversation.async_setup_entry", return_value=True
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN,
            context={"source": config_entries.SOURCE_USER},
            data={CONF_WS_URL: f" {DEFAULT_WS_URL} ", CONF_AUTH_TOKEN: " secret "},
        )
        await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {**DATA, CONF_DEFAULT_USER_ID: ""}
    assert result["result"].unique_id is None
    connect.assert_called_once_with(
        DEFAULT_WS_URL, headers={"Authorization": "Bearer secret"}
    )


@pytest.mark.parametrize("unique_id", [None, DOMAIN])
async def test_multiple_entries_allowed(hass, websocket, unique_id):
    entry = MockConfigEntry(
        domain=DOMAIN, data={CONF_WS_URL: DEFAULT_WS_URL}, unique_id=unique_id
    )
    entry.add_to_hass(hass)
    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM
    websocket[0].assert_not_called()
    with patch(
        "custom_components.oswald_conversation.async_setup_entry", return_value=True
    ):
        result = await hass.config_entries.flow.async_configure(result["flow_id"], DATA)
        await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["result"].entry_id != entry.entry_id
    assert result["result"].unique_id is None
    assert len(hass.config_entries.async_entries(DOMAIN)) == 2
    assert entry.data == {CONF_WS_URL: DEFAULT_WS_URL}


@pytest.mark.parametrize("stored_token", [None, "old-token"])
@pytest.mark.parametrize("submitted_token", [None, "", "  ", "******", " replacement "])
async def test_reconfigure_tokens(hass, websocket, stored_token, submitted_token):
    data = {CONF_WS_URL: DEFAULT_WS_URL}
    if stored_token is not None:
        data[CONF_AUTH_TOKEN] = stored_token
    entry = MockConfigEntry(domain=DOMAIN, data=data)
    entry.add_to_hass(hass)
    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "entry_id": entry.entry_id,
        },
    )
    schema = result["data_schema"]
    if stored_token:
        assert schema({CONF_WS_URL: DEFAULT_WS_URL})[CONF_AUTH_TOKEN] == "******"
    else:
        with pytest.raises(vol.Invalid):
            schema({CONF_WS_URL: DEFAULT_WS_URL})
    submitted = {CONF_WS_URL: DEFAULT_WS_URL}
    if submitted_token is not None:
        submitted[CONF_AUTH_TOKEN] = submitted_token
    if stored_token is None and submitted_token is None:
        with pytest.raises(InvalidData):
            await hass.config_entries.flow.async_configure(result["flow_id"], submitted)
        assert dict(entry.data) == data
        websocket[0].assert_not_called()
        return
    with patch.object(hass.config_entries, "async_reload", return_value=True) as reload:
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            submitted,
        )
        await hass.async_block_till_done()
    expected = (
        "replacement"
        if (submitted_token or "").strip() == "replacement"
        else stored_token
    )
    if expected is None:
        assert result["errors"] == {"base": "invalid_auth"}
        assert dict(entry.data) == data
        websocket[0].assert_not_called()
        reload.assert_not_called()
    else:
        assert result["type"] is FlowResultType.ABORT
        assert result["reason"] == "reconfigure_successful"
        assert entry.data[CONF_AUTH_TOKEN] == expected
        websocket[0].assert_called_once_with(
            DEFAULT_WS_URL, headers={"Authorization": f"Bearer {expected}"}
        )
        reload.assert_awaited_once_with(entry.entry_id)


@pytest.mark.parametrize("stored_token", [None, "old-token"])
async def test_reauth_requires_replacement_and_preserves_data(
    hass, websocket, stored_token
):
    user = await hass.auth.async_create_user("Fallback user")
    data = {CONF_WS_URL: DEFAULT_WS_URL, CONF_DEFAULT_USER_ID: user.id}
    if stored_token is not None:
        data[CONF_AUTH_TOKEN] = stored_token
    entry = MockConfigEntry(domain=DOMAIN, data=data)
    entry.add_to_hass(hass)
    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": config_entries.SOURCE_REAUTH, "entry_id": entry.entry_id},
        data=data,
    )
    assert result["step_id"] == "reauth_confirm"
    assert "old-token" not in str(result["data_schema"])
    for token in ("", "  ", "******"):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {CONF_AUTH_TOKEN: token},
        )
        assert result["errors"] == {"base": "invalid_auth"}
        assert dict(entry.data) == data
    websocket[0].assert_not_called()
    with patch.object(hass.config_entries, "async_reload", return_value=True) as reload:
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {CONF_AUTH_TOKEN: " replacement "},
        )
        await hass.async_block_till_done()
    assert result["reason"] == "reauth_successful"
    assert dict(entry.data) == {**data, CONF_AUTH_TOKEN: "replacement"}
    assert len(hass.config_entries.async_entries(DOMAIN)) == 1
    reload.assert_awaited_once_with(entry.entry_id)


@pytest.mark.parametrize(
    ("updates", "errors"),
    [
        ({CONF_WS_URL: "https://example.com"}, {CONF_WS_URL: "invalid_ws_url"}),
        ({CONF_WS_URL: "ws://user:pass@example.com"}, {CONF_WS_URL: "invalid_ws_url"}),
        ({CONF_AUTH_TOKEN: " "}, {"base": "invalid_auth"}),
        ({CONF_AUTH_TOKEN: "******"}, {"base": "invalid_auth"}),
        (
            {CONF_DEFAULT_USER_ID: "missing-user"},
            {CONF_DEFAULT_USER_ID: "invalid_user"},
        ),
    ],
)
async def test_input_validation(hass, websocket, updates, errors):
    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": config_entries.SOURCE_USER},
        data={**DATA, **updates},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == errors
    websocket[0].assert_not_called()


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (WSServerHandshakeError(MagicMock(), (), status=401), "invalid_auth"),
        (WSServerHandshakeError(MagicMock(), (), status=503), "cannot_connect"),
        (ClientError(), "cannot_connect"),
        (TimeoutError(), "cannot_connect"),
        (OSError(), "cannot_connect"),
    ],
)
async def test_connection_errors(hass, websocket, failure, expected):
    websocket[1].__aenter__.side_effect = failure
    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": config_entries.SOURCE_USER},
        data=DATA,
    )
    assert result["errors"] == {"base": expected}


async def test_unsupported_protocol(hass, websocket):
    websocket[2].receive.return_value = WSMessage(
        WSMsgType.TEXT, '{"type":"ready","protocol_version":2}', ""
    )
    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": config_entries.SOURCE_USER},
        data=DATA,
    )
    assert result["errors"] == {"base": "unsupported_protocol"}


@pytest.mark.parametrize(
    "source", [config_entries.SOURCE_REAUTH, config_entries.SOURCE_RECONFIGURE]
)
async def test_failed_update_preserves_entry(hass, websocket, source):
    entry = MockConfigEntry(domain=DOMAIN, data=DATA)
    entry.add_to_hass(hass)
    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": source, "entry_id": entry.entry_id},
        data=DATA if source == config_entries.SOURCE_REAUTH else None,
    )
    websocket[1].__aenter__.side_effect = WSServerHandshakeError(
        MagicMock(), (), status=401
    )
    submitted = {CONF_AUTH_TOKEN: "rejected"}
    if source == config_entries.SOURCE_RECONFIGURE:
        submitted[CONF_WS_URL] = "wss://example.com/ws"
    with patch.object(hass.config_entries, "async_reload") as reload:
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], submitted
        )
        await hass.async_block_till_done()
    assert result["errors"] == {"base": "invalid_auth"}
    assert dict(entry.data) == DATA
    reload.assert_not_called()


async def test_reconfigure_removes_fallback(hass, websocket):
    user = await hass.auth.async_create_user("Fallback")
    entry = MockConfigEntry(domain=DOMAIN, data={**DATA, CONF_DEFAULT_USER_ID: user.id})
    entry.add_to_hass(hass)
    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "entry_id": entry.entry_id,
        },
    )
    user_key = next(
        key for key in result["data_schema"].schema if key == CONF_DEFAULT_USER_ID
    )
    assert user_key.description == {"suggested_value": user.id}
    assert CONF_DEFAULT_USER_ID not in result["data_schema"](DATA)
    with patch.object(hass.config_entries, "async_reload", return_value=True):
        result = await hass.config_entries.flow.async_configure(result["flow_id"], DATA)
        await hass.async_block_till_done()
    assert result["reason"] == "reconfigure_successful"
    assert entry.data[CONF_DEFAULT_USER_ID] == ""


@pytest.mark.parametrize("user_kind", ["active", "inactive", "system"])
async def test_fallback_user_validation(hass, websocket, user_kind):
    await hass.auth.async_create_user("Owner")
    if user_kind == "system":
        user = await hass.auth.async_create_system_user("Fallback")
    else:
        user = await hass.auth.async_create_user("Fallback")
    if user_kind == "inactive":
        await hass.auth.async_deactivate_user(user)
    with patch(
        "custom_components.oswald_conversation.async_setup_entry", return_value=True
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN,
            context={"source": config_entries.SOURCE_USER},
            data={**DATA, CONF_DEFAULT_USER_ID: user.id},
        )
        await hass.async_block_till_done()
    if user_kind == "active":
        assert result["type"] is FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_DEFAULT_USER_ID] == user.id
    else:
        assert result["errors"] == {CONF_DEFAULT_USER_ID: "invalid_user"}
        websocket[0].assert_not_called()


async def test_english_translations_load(hass):
    translations = await async_get_translations(hass, "en", "config", {DOMAIN})
    prefix = f"component.{DOMAIN}.config"
    for step in ("user", "reconfigure", "reauth_confirm"):
        assert (
            translations[f"{prefix}.step.{step}.data.auth_token"]
            == "Authentication token"
        )
        assert (
            "without encryption"
            in translations[f"{prefix}.step.{step}.data_description.auth_token"]
        )
    for reason in (
        "reauth_successful",
        "reconfigure_successful",
    ):
        assert translations[f"{prefix}.abort.{reason}"]
    for error in (
        "invalid_ws_url",
        "invalid_auth",
        "invalid_user",
        "cannot_connect",
        "unsupported_protocol",
    ):
        assert translations[f"{prefix}.error.{error}"]


async def test_setup_forwards_platform(hass):
    entry = MockConfigEntry(domain=DOMAIN, data=DATA)
    entry.add_to_hass(hass)
    with patch.object(hass.config_entries, "async_forward_entry_setups") as forward:
        assert await async_setup_entry(hass, entry)
    forward.assert_awaited_once_with(entry, [Platform.CONVERSATION])


async def test_setup_propagates_platform_failure(hass):
    entry = MockConfigEntry(domain=DOMAIN, data=DATA)
    entry.add_to_hass(hass)
    with (
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            side_effect=RuntimeError("Platform failed"),
        ),
        pytest.raises(RuntimeError, match="Platform failed"),
    ):
        await async_setup_entry(hass, entry)


@pytest.mark.parametrize("success", [True, False])
async def test_unload_returns_platform_result(hass, success):
    entry = MockConfigEntry(domain=DOMAIN, data=DATA)
    entry.add_to_hass(hass)
    with patch.object(
        hass.config_entries, "async_unload_platforms", return_value=success
    ) as unload:
        assert await async_unload_entry(hass, entry) is success
    unload.assert_awaited_once_with(entry, [Platform.CONVERSATION])
