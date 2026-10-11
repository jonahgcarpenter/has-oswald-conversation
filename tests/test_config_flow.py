"""Exercise config flows with real Home Assistant and a mocked OpenAI peer."""

from unittest.mock import patch

import pytest
from homeassistant import config_entries
from homeassistant.const import Platform
from homeassistant.data_entry_flow import FlowResultType, InvalidData
from homeassistant.exceptions import ConfigEntryAuthFailed
from homeassistant.helpers import selector
from homeassistant.helpers.translation import async_get_translations
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.oswald_conversation import (
    async_migrate_entry,
    async_setup_entry,
    async_unload_entry,
)
from custom_components.oswald_conversation.client import (
    ConnectionFailedError,
    InvalidAuthError,
)
from custom_components.oswald_conversation.const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_LLM_HASS_API,
    CONF_MAX_HISTORY,
    CONF_MODEL,
    CONF_OSWALD_TOOLS,
    CONF_PROMPT,
    DEFAULT_BASE_URL,
    DEFAULT_MAX_HISTORY,
    DEFAULT_PROMPT,
    DOMAIN,
)
from custom_components.oswald_conversation.tools import DEFAULT_OSWALD_TOOLS

pytestmark = pytest.mark.asyncio

BASE_URL = DEFAULT_BASE_URL
DATA = {CONF_BASE_URL: BASE_URL, CONF_API_KEY: "secret"}
MODELS = ["oswald-large", "oswald-small"]
FLOW_MODULE = "custom_components.oswald_conversation.config_flow"


@pytest.fixture(autouse=True)
async def setup_core(hass):
    """Initialize exposed entities before the conversation dependency starts."""
    assert await async_setup_component(hass, "homeassistant", {})


@pytest.fixture
def list_models():
    """Mock only the /models transport, keeping HA flow code real."""
    with patch(f"{FLOW_MODULE}.async_list_models", return_value=list(MODELS)) as mock:
        yield mock


async def start_user_flow(hass):
    return await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )


async def submit_user_form(hass, flow_id, updates=None):
    return await hass.config_entries.flow.async_configure(
        flow_id, {**DATA, **(updates or {})}
    )


async def test_user_success(hass, list_models):
    with patch(
        "custom_components.oswald_conversation.async_setup_entry", return_value=True
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN,
            context={"source": config_entries.SOURCE_USER},
            data={CONF_BASE_URL: f" {BASE_URL}// ", CONF_API_KEY: " secret "},
        )
        await hass.async_block_till_done()
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "model"
    list_models.assert_awaited_once()
    assert list_models.call_args.args[1:] == (BASE_URL, "secret")

    with patch(
        "custom_components.oswald_conversation.async_setup_entry", return_value=True
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_MODEL: MODELS[0]}
        )
        await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {**DATA, CONF_MODEL: MODELS[0]}


async def test_model_options_match_server_list(hass, list_models):
    result = await start_user_flow(hass)
    result = await submit_user_form(hass, result["flow_id"])
    assert result["step_id"] == "model"
    schema = result["data_schema"].schema
    model_key = next(key for key in schema if str(key) == CONF_MODEL)
    model_selector = schema[model_key]
    assert isinstance(model_selector, selector.SelectSelector)
    assert model_selector.config["options"] == MODELS


@pytest.mark.parametrize(
    ("updates", "errors", "called"),
    [
        (
            {CONF_BASE_URL: "ws://127.0.0.1:8080/v1"},
            {CONF_BASE_URL: "invalid_url"},
            False,
        ),
        (
            {CONF_BASE_URL: "http://user:pass@127.0.0.1:8080/v1"},
            {CONF_BASE_URL: "invalid_url"},
            False,
        ),
        ({CONF_API_KEY: " "}, {"base": "invalid_auth"}, False),
        ({CONF_API_KEY: "******"}, {"base": "invalid_auth"}, False),
    ],
)
async def test_input_validation(hass, list_models, updates, errors, called):
    result = await start_user_flow(hass)
    result = await submit_user_form(hass, result["flow_id"], updates)
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == errors
    assert list_models.called is called


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (InvalidAuthError(), "invalid_auth"),
        (ConnectionFailedError(), "cannot_connect"),
        (TimeoutError(), "cannot_connect"),
        (OSError(), "cannot_connect"),
    ],
)
async def test_connection_errors(hass, list_models, failure, expected):
    list_models.side_effect = failure
    result = await start_user_flow(hass)
    result = await submit_user_form(hass, result["flow_id"])
    assert result["errors"] == {"base": expected}


async def test_no_models(hass, list_models):
    list_models.return_value = []
    result = await start_user_flow(hass)
    result = await submit_user_form(hass, result["flow_id"])
    assert result["errors"] == {"base": "no_models"}


@pytest.mark.parametrize("stored_key", [None, "old-key"])
@pytest.mark.parametrize("submitted_key", [None, "", "  ", "******", " replacement "])
async def test_reconfigure(hass, list_models, stored_key, submitted_key):
    data = {CONF_BASE_URL: BASE_URL, CONF_MODEL: MODELS[0]}
    if stored_key is not None:
        data[CONF_API_KEY] = stored_key
    entry = MockConfigEntry(domain=DOMAIN, data=data)
    entry.add_to_hass(hass)
    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "entry_id": entry.entry_id,
        },
    )
    assert result["step_id"] == "reconfigure"
    submitted = {CONF_BASE_URL: f" {BASE_URL}/ "}
    if submitted_key is not None:
        submitted[CONF_API_KEY] = submitted_key
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], submitted
    )
    expected = (
        "replacement" if (submitted_key or "").strip() == "replacement" else stored_key
    )
    if expected is None:
        assert result["errors"] == {"base": "invalid_auth"}
        assert dict(entry.data) == data
        list_models.assert_not_called()
        return
    assert result["step_id"] == "model"
    list_models.assert_awaited_once_with(
        list_models.call_args.args[0], BASE_URL, expected
    )
    with patch.object(hass.config_entries, "async_reload", return_value=True) as reload:
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_MODEL: MODELS[1]}
        )
        await hass.async_block_till_done()
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert entry.data[CONF_API_KEY] == expected
    assert entry.data[CONF_MODEL] == MODELS[1]
    reload.assert_awaited_once_with(entry.entry_id)


@pytest.mark.parametrize("stored_key", [None, "old-key"])
async def test_reauth(hass, list_models, stored_key):
    data = {CONF_BASE_URL: BASE_URL, CONF_MODEL: MODELS[0]}
    if stored_key is not None:
        data[CONF_API_KEY] = stored_key
    entry = MockConfigEntry(domain=DOMAIN, data=data)
    entry.add_to_hass(hass)
    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": config_entries.SOURCE_REAUTH, "entry_id": entry.entry_id},
        data=data,
    )
    assert result["step_id"] == "reauth_confirm"
    assert "old-key" not in str(result["data_schema"])
    for key in ("", "  ", "******"):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_API_KEY: key}
        )
        assert result["errors"] == {"base": "invalid_auth"}
        assert dict(entry.data) == data
    list_models.assert_not_called()
    with patch.object(hass.config_entries, "async_reload", return_value=True) as reload:
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_API_KEY: " replacement "}
        )
        await hass.async_block_till_done()
    assert result["reason"] == "reauth_successful"
    assert dict(entry.data) == {**data, CONF_API_KEY: "replacement"}
    list_models.assert_awaited_once()
    reload.assert_awaited_once_with(entry.entry_id)


async def test_reauth_rejected_key_preserves_entry(hass, list_models):
    entry = MockConfigEntry(domain=DOMAIN, data={**DATA, CONF_MODEL: MODELS[0]})
    entry.add_to_hass(hass)
    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": config_entries.SOURCE_REAUTH, "entry_id": entry.entry_id},
        data=dict(entry.data),
    )
    list_models.side_effect = InvalidAuthError()
    with patch.object(hass.config_entries, "async_reload") as reload:
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_API_KEY: "rejected"}
        )
        await hass.async_block_till_done()
    assert result["errors"] == {"base": "invalid_auth"}
    assert dict(entry.data) == {**DATA, CONF_MODEL: MODELS[0]}
    reload.assert_not_called()


async def test_english_translations_load(hass):
    translations = await async_get_translations(hass, "en", "config", {DOMAIN})
    prefix = f"component.{DOMAIN}.config"
    for step in ("user", "reconfigure", "reauth_confirm"):
        assert translations[f"{prefix}.step.{step}.data.api_key"] == "API key"
    assert translations[f"{prefix}.step.model.data.model"] == "Model"
    for error in (
        "invalid_url",
        "invalid_auth",
        "cannot_connect",
        "no_models",
    ):
        assert translations[f"{prefix}.error.{error}"]
    for reason in ("reauth_successful", "reconfigure_successful"):
        assert translations[f"{prefix}.abort.{reason}"]


@pytest.mark.parametrize("key_data", [{}, {CONF_API_KEY: ""}, {CONF_API_KEY: "  "}])
async def test_setup_requires_api_key(hass, key_data):
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_BASE_URL: BASE_URL, **key_data})
    entry.add_to_hass(hass)
    with patch.object(hass.config_entries, "async_forward_entry_setups") as forward:
        with pytest.raises(ConfigEntryAuthFailed):
            await async_setup_entry(hass, entry)
        forward.assert_not_called()


async def test_migrate_legacy_entry_fails(hass):
    entry = MockConfigEntry(domain=DOMAIN, data={"ws_url": "ws://x"})
    assert await async_migrate_entry(hass, entry) is False


async def test_setup_forwards_platform(hass):
    entry = MockConfigEntry(domain=DOMAIN, data={**DATA, CONF_MODEL: MODELS[0]})
    entry.add_to_hass(hass)
    with patch.object(hass.config_entries, "async_forward_entry_setups") as forward:
        assert await async_setup_entry(hass, entry)
    forward.assert_awaited_once_with(entry, [Platform.CONVERSATION])


async def test_setup_propagates_platform_failure(hass):
    entry = MockConfigEntry(domain=DOMAIN, data={**DATA, CONF_MODEL: MODELS[0]})
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
    entry = MockConfigEntry(domain=DOMAIN, data={**DATA, CONF_MODEL: MODELS[0]})
    entry.add_to_hass(hass)
    with patch.object(
        hass.config_entries, "async_unload_platforms", return_value=success
    ) as unload:
        assert await async_unload_entry(hass, entry) is success
    unload.assert_awaited_once_with(entry, [Platform.CONVERSATION])


OPTIONS_DATA = {
    CONF_MODEL: MODELS[1],
    CONF_PROMPT: "Custom instructions.",
    CONF_LLM_HASS_API: ["assist"],
    CONF_MAX_HISTORY: 5,
    CONF_OSWALD_TOOLS: ["web_search", "memory"],
}


def _schema_default(data_schema, key_name):
    key = next(key for key in data_schema.schema if str(key) == key_name)
    default = key.default
    return default() if callable(default) else default


async def test_options_flow(hass, list_models):
    entry = MockConfigEntry(domain=DOMAIN, data={**DATA, CONF_MODEL: MODELS[0]})
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"
    schema = result["data_schema"].schema
    assert {str(key) for key in schema} == {
        CONF_MODEL,
        CONF_PROMPT,
        CONF_LLM_HASS_API,
        CONF_MAX_HISTORY,
        CONF_OSWALD_TOOLS,
    }
    model_key = next(key for key in schema if str(key) == CONF_MODEL)
    assert schema[model_key].config["options"] == MODELS
    assert _schema_default(result["data_schema"], CONF_PROMPT) == DEFAULT_PROMPT
    assert (
        _schema_default(result["data_schema"], CONF_OSWALD_TOOLS)
        == DEFAULT_OSWALD_TOOLS
    )

    result = await hass.config_entries.options.async_configure(
        result["flow_id"], OPTIONS_DATA
    )
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options[CONF_MODEL] == MODELS[1]
    assert entry.options[CONF_PROMPT] == "Custom instructions."
    assert entry.options[CONF_LLM_HASS_API] == ["assist"]
    assert entry.options[CONF_MAX_HISTORY] == 5
    assert entry.options[CONF_OSWALD_TOOLS] == ["web_search", "memory"]


async def test_options_flow_defaults_from_entry(hass, list_models):
    entry = MockConfigEntry(
        domain=DOMAIN, data={**DATA, CONF_MODEL: MODELS[0]}, options=OPTIONS_DATA
    )
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    schema = result["data_schema"].schema
    defaults = {
        str(key): _schema_default(result["data_schema"], str(key)) for key in schema
    }
    assert defaults[CONF_MODEL] == MODELS[1]
    assert defaults[CONF_PROMPT] == "Custom instructions."
    assert defaults[CONF_LLM_HASS_API] == ["assist"]
    assert defaults[CONF_MAX_HISTORY] == 5
    assert defaults[CONF_OSWALD_TOOLS] == ["web_search", "memory"]


async def test_options_flow_rejects_unknown_apis(hass, list_models):
    entry = MockConfigEntry(domain=DOMAIN, data={**DATA, CONF_MODEL: MODELS[0]})
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    # Select validation rejects unknown APIs before the handler runs.
    with pytest.raises(InvalidData):
        await hass.config_entries.options.async_configure(
            result["flow_id"],
            {**OPTIONS_DATA, CONF_LLM_HASS_API: ["assist", "no-such-api"]},
        )
    assert entry.options == {}


async def test_options_flow_model_fallback_when_server_down(hass, list_models):
    list_models.side_effect = ConnectionFailedError()
    entry = MockConfigEntry(domain=DOMAIN, data={**DATA, CONF_MODEL: MODELS[0]})
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    schema = result["data_schema"].schema
    model_key = next(key for key in schema if str(key) == CONF_MODEL)
    assert isinstance(schema[model_key], selector.TextSelector)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {**OPTIONS_DATA, CONF_MODEL: "typed-model"}
    )
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options[CONF_MODEL] == "typed-model"


async def test_options_flow_rejects_empty_model(hass, list_models):
    # With the server down the model field falls back to free text,
    # so blank input reaches the handler instead of schema validation.
    list_models.side_effect = ConnectionFailedError()
    entry = MockConfigEntry(domain=DOMAIN, data={**DATA, CONF_MODEL: MODELS[0]})
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {**OPTIONS_DATA, CONF_MODEL: "  "}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {CONF_MODEL: "empty_model"}
    assert entry.options == {}


async def test_options_update_reloads_entry(hass):
    entry = MockConfigEntry(domain=DOMAIN, data={**DATA, CONF_MODEL: MODELS[0]})
    entry.add_to_hass(hass)
    with patch.object(hass.config_entries, "async_forward_entry_setups"):
        assert await async_setup_entry(hass, entry)
    with patch.object(hass.config_entries, "async_reload", return_value=True) as reload:
        hass.config_entries.async_update_entry(
            entry, options={**OPTIONS_DATA, CONF_MAX_HISTORY: DEFAULT_MAX_HISTORY}
        )
        await hass.async_block_till_done()
    reload.assert_awaited_once_with(entry.entry_id)


async def test_options_translations_load(hass):
    translations = await async_get_translations(hass, "en", "options", {DOMAIN})
    prefix = f"component.{DOMAIN}.options"
    assert translations[f"{prefix}.step.init.data.model"] == "Model"
    assert translations[f"{prefix}.step.init.data.prompt"] == "Instructions"
    assert (
        translations[f"{prefix}.step.init.data.llm_hass_api"]
        == "Control Home Assistant"
    )
    assert (
        translations[f"{prefix}.step.init.data.max_history_messages"]
        == "Max history messages"
    )
    assert translations[f"{prefix}.step.init.data.oswald_tools"] == "Oswald tools"
