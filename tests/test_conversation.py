"""Replay OpenAI gateway chunks through Home Assistant's real ChatLog."""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import ClientError
from homeassistant.components.conversation import ChatLog, ConversationInput
from homeassistant.components.conversation.chat_log import (
    AssistantContent,
    UserContent,
)
from homeassistant.core import Context
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.oswald_conversation.client import (
    ChatCompletionError,
    ConnectionFailedError,
    InvalidAuthError,
)
from custom_components.oswald_conversation.const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_MODEL,
    DEFAULT_BASE_URL,
    DOMAIN,
)
from custom_components.oswald_conversation.conversation import (
    _AUTH_RESPONSE,
    _FAILURE_RESPONSE,
    _INTERRUPTED_RESPONSE,
    _MISSING_CONVERSATION_RESPONSE,
    OswaldConversationEntity,
)

MODULE = "custom_components.oswald_conversation.conversation"
BASE_URL = DEFAULT_BASE_URL
MODEL = "oswald-large"


@pytest.fixture(autouse=True)
async def setup_core(hass):
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "conversation", {})


@pytest.fixture
async def user_input(hass):
    user = await hass.auth.async_create_user("Alice")
    return ConversationInput(
        text="Hello",
        context=Context(user_id=user.id),
        conversation_id="conversation:room.1",
        device_id=None,
        satellite_id=None,
        language="en",
        agent_id="conversation.oswald",
    )


@pytest.fixture
def entity(hass):
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_BASE_URL: BASE_URL, CONF_API_KEY: "test-key", CONF_MODEL: MODEL},
    )
    entry.add_to_hass(hass)
    return OswaldConversationEntity(hass, entry)


def content_chunk(text):
    return {"choices": [{"delta": {"content": text}, "finish_reason": None}]}


def stop_chunk():
    return {"choices": [{"delta": {}, "finish_reason": "stop"}]}


def tool_call_chunk(name, arguments, call_id="call_1", index=0):
    return {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": call_id,
                            "function": {"name": name, "arguments": arguments},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ]
    }


@pytest.fixture
def chat_peer():
    """Fake the SSE transport, keeping entity, ChatLog and tools real."""
    calls: list[dict] = []
    behaviors: list = []

    async def fake_stream(session, base_url, api_key, payload):
        calls.append(
            {
                "session": session,
                "base_url": base_url,
                "api_key": api_key,
                "payload": payload,
            }
        )
        behavior = behaviors[min(len(calls) - 1, len(behaviors) - 1)]
        if isinstance(behavior, BaseException):
            raise behavior
        for item in behavior:
            if isinstance(item, BaseException):
                raise item
            yield item

    with patch(f"{MODULE}.async_stream_chat_completion", new=fake_stream):
        yield behaviors, calls


async def converse(hass, entity, user_input):
    deltas = []
    chat_log = ChatLog(
        hass,
        user_input.conversation_id,
        delta_listener=lambda log, delta: deltas.append(delta),
    )
    # Mirror production: ConversationEntity.async_process adds the user message.
    chat_log.async_add_user_content(UserContent(content=user_input.text))
    result = await entity._async_handle_message(user_input, chat_log)
    return result, chat_log, deltas


def speech(result):
    return result.response.speech["plain"]["speech"]


async def test_simple_answer(hass, entity, user_input, chat_peer):
    behaviors, calls = chat_peer
    behaviors.append([content_chunk("Hello "), content_chunk("there."), stop_chunk()])
    result, chat_log, deltas = await converse(hass, entity, user_input)
    assert speech(result) == "Hello there."
    assert result.conversation_id == user_input.conversation_id
    # The answer is recorded exactly once; no duplicate final message.
    answers = [item for item in chat_log.content if isinstance(item, AssistantContent)]
    assert len(answers) == 1
    assert answers[0].content == "Hello there."
    assert "".join(delta.get("content", "") for delta in deltas) == "Hello there."

    assert len(calls) == 1
    assert calls[0]["base_url"] == BASE_URL
    assert calls[0]["api_key"] == "test-key"
    payload = calls[0]["payload"]
    assert payload["model"] == MODEL
    assert payload["stream"] is True
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][-1] == {"role": "user", "content": "Hello"}
    tool_names = [tool["function"]["name"] for tool in payload["tools"]]
    assert any(name.endswith("GetDateTime") for name in tool_names)


async def assist_tool_name(hass, user_input, suffix):
    """Return the Assist tool name ending with suffix (names vary by HA)."""
    from homeassistant.helpers import llm

    api = await llm.async_get_api(
        hass,
        "assist",
        llm.LLMContext(
            platform=DOMAIN,
            context=user_input.context,
            language=user_input.language,
            assistant=user_input.agent_id,
            device_id=user_input.device_id,
        ),
    )
    return next(tool.name for tool in api.tools if tool.name.endswith(suffix))


async def test_tool_loop(hass, entity, user_input, chat_peer):
    tool_name = await assist_tool_name(hass, user_input, "GetDateTime")
    behaviors, calls = chat_peer
    behaviors.append(
        [
            tool_call_chunk(tool_name, '{"format":'),
            tool_call_chunk(tool_name, ' "iso"}', call_id=""),
            stop_chunk(),
        ]
    )
    behaviors.append([content_chunk("Done."), stop_chunk()])
    result, chat_log, _ = await converse(hass, entity, user_input)
    assert speech(result) == "Done."
    assert len(calls) == 2

    roles = [item.role for item in chat_log.content]
    assert roles == ["system", "user", "assistant", "tool_result", "assistant"]

    follow_up = calls[1]["payload"]["messages"]
    assistant_msg = next(msg for msg in follow_up if msg.get("tool_calls"))
    assert assistant_msg["tool_calls"][0]["function"]["name"] == tool_name
    call_id = assistant_msg["tool_calls"][0]["id"]
    tool_msg = next(msg for msg in follow_up if msg.get("role") == "tool")
    assert tool_msg["tool_call_id"] == call_id
    assert json.loads(tool_msg["content"])


async def test_rejected_key_starts_reauth(hass, entity, user_input, chat_peer):
    behaviors, _ = chat_peer
    behaviors.append(InvalidAuthError())
    with patch.object(entity.entry, "async_start_reauth") as reauth:
        result, chat_log, _ = await converse(hass, entity, user_input)
    assert speech(result) == _AUTH_RESPONSE
    assert chat_log.content[-1].content == _AUTH_RESPONSE
    reauth.assert_called_once_with(hass)


async def test_connection_failure(hass, entity, user_input, chat_peer):
    behaviors, _ = chat_peer
    behaviors.append(ConnectionFailedError("down"))
    result, chat_log, _ = await converse(hass, entity, user_input)
    assert speech(result) == _FAILURE_RESPONSE
    assert chat_log.content[-1].content == _FAILURE_RESPONSE


async def test_interrupted_after_partial_content(hass, entity, user_input, chat_peer):
    behaviors, _ = chat_peer
    behaviors.append([content_chunk("Partial"), ClientError("boom")])
    result, chat_log, _ = await converse(hass, entity, user_input)
    assert speech(result) == _INTERRUPTED_RESPONSE
    assert chat_log.content[-1].content == _INTERRUPTED_RESPONSE


async def test_server_error_message_is_authoritative(
    hass, entity, user_input, chat_peer
):
    behaviors, _ = chat_peer
    behaviors.append(ChatCompletionError("Unknown model.", status=400))
    result, _, _ = await converse(hass, entity, user_input)
    assert speech(result) == "Unknown model."


async def test_empty_response_is_failure(hass, entity, user_input, chat_peer):
    behaviors, _ = chat_peer
    behaviors.append([stop_chunk()])
    result, _, _ = await converse(hass, entity, user_input)
    assert speech(result) == _FAILURE_RESPONSE


async def test_missing_conversation(hass, entity, user_input, chat_peer):
    behaviors, calls = chat_peer
    behaviors.append([content_chunk("Hi"), stop_chunk()])
    user_input.conversation_id = ""
    result, _, _ = await converse(hass, entity, user_input)
    assert speech(result) == _MISSING_CONVERSATION_RESPONSE
    assert calls == []


async def test_caller_cancellation_propagates(hass, entity, user_input, chat_peer):
    behaviors, calls = chat_peer
    behaviors.append(asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await converse(hass, entity, user_input)
    assert len(calls) == 1


async def test_tool_limit(hass, entity, user_input, chat_peer, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.MAX_TOOL_TURNS", 2)
    tool_name = await assist_tool_name(hass, user_input, "GetDateTime")
    behaviors, calls = chat_peer
    behaviors.append([tool_call_chunk(tool_name, "{}"), stop_chunk()])
    result, _, _ = await converse(hass, entity, user_input)
    assert speech(result) == _FAILURE_RESPONSE
    assert len(calls) == 2


async def test_error_logs_do_not_include_secrets(
    hass, entity, user_input, chat_peer, caplog
):
    behaviors, _ = chat_peer
    behaviors.append(ClientError("marker-secret-boom"))
    await converse(hass, entity, user_input)
    assert "marker-secret-boom" not in caplog.text
    assert "test-key" not in caplog.text
    assert "error_type=ClientError" in caplog.text


async def test_handshake_error_is_failure(hass, entity, user_input, chat_peer):
    from aiohttp import WSServerHandshakeError

    behaviors, _ = chat_peer
    behaviors.append(WSServerHandshakeError(MagicMock(), (), status=401))
    result, _, _ = await converse(hass, entity, user_input)
    assert speech(result) == _FAILURE_RESPONSE
