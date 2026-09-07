"""Replay gateway frames through Home Assistant's real append-only ChatLog."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientError, WSMessage, WSMsgType, WSServerHandshakeError
from homeassistant.components.conversation import ChatLog, ConversationInput
from homeassistant.core import Context
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.oswald_conversation.const import (
    CONF_AUTH_TOKEN,
    CONF_WS_URL,
    DEFAULT_WS_URL,
    DOMAIN,
)
from custom_components.oswald_conversation.conversation import (
    _AUTH_RESPONSE,
    _FAILURE_RESPONSE,
    _INTERRUPTED_RESPONSE,
    _MISSING_USER_RESPONSE,
    OswaldConversationEntity,
)

MODULE = "custom_components.oswald_conversation.conversation"


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
        data={CONF_WS_URL: DEFAULT_WS_URL, CONF_AUTH_TOKEN: "test-token"},
    )
    entry.add_to_hass(hass)
    return OswaldConversationEntity(hass, entry)


@pytest.fixture
def peer():
    ws = AsyncMock()
    ws.receive.return_value = WSMessage(
        WSMsgType.TEXT, '{"type":"ready","protocol_version":1}', ""
    )
    frames = []

    async def messages():
        for frame in frames:
            if isinstance(frame, BaseException):
                raise frame
            if callable(frame):
                await frame()
                continue
            if isinstance(frame, WSMessage):
                yield frame
                continue
            data = {"request_id": ws.send_json.call_args.args[0]["request_id"], **frame}
            if data["request_id"] is None:
                del data["request_id"]
            yield WSMessage(WSMsgType.TEXT, json.dumps(data), "")

    ws.__aiter__.side_effect = messages
    connection = MagicMock()
    connection.__aenter__ = AsyncMock(return_value=ws)
    connection.__aexit__ = AsyncMock(return_value=False)
    with patch(f"{MODULE}.async_get_clientsession") as session:
        session.return_value.ws_connect.return_value = connection
        yield frames, ws, connection, session.return_value.ws_connect


async def converse(hass, entity, user_input):
    deltas = []
    chat_log = ChatLog(
        hass,
        user_input.conversation_id,
        delta_listener=lambda log, delta: deltas.append(delta),
    )
    result = await entity._async_handle_message(user_input, chat_log)
    return result, chat_log, deltas


def speech(result):
    return result.response.speech["plain"]["speech"]


@pytest.mark.parametrize(
    "preview", [[], ["The answer."], ["Wrong preview"], ["The ", "answer."]]
)
async def test_authoritative_answer_once(hass, entity, user_input, peer, preview):
    frames, ws, connection, connect = peer
    frames.extend({"type": "content", "text": text} for text in preview)
    frames.append({"type": "result", "response": "The answer."})
    result, chat_log, deltas = await converse(hass, entity, user_input)
    assert speech(result) == "The answer."
    assert chat_log.content[-1].content == "The answer."
    assert "".join(delta.get("content", "") for delta in deltas) == "The answer."
    assert result.conversation_id == user_input.conversation_id
    payload = ws.send_json.call_args.args[0]
    assert payload == {
        "type": "conversation",
        "request_id": payload["request_id"],
        "user_id": user_input.context.user_id,
        "display_name": "Alice",
        "conversation_id": user_input.conversation_id,
        "text": user_input.text,
    }
    assert connect.call_args.kwargs["headers"] == {"Authorization": "Bearer test-token"}
    assert connect.call_args.kwargs["timeout"].ws_receive is None
    connection.__aexit__.assert_awaited_once()


async def test_tool_rounds_and_final_only_answer(hass, entity, user_input, peer):
    frames = peer[0]
    frames.append({"type": "content", "text": "Let me check."})
    for name, metadata in [
        ("web.fetch", {"web.fetch": {"title": "Page", "is_degraded": True}}),
        ("web.search", {"web.search": {"results": [{"title": "Result"}]}}),
        ("user_memory_save", {"user_memory": {"action": "save"}}),
        ("global_memory_search", {"global_memory": {"action": "search"}}),
    ]:
        frames.extend(
            [
                {"type": "tool_call", "tool": {"name": name}},
                {"type": "tool_result", "tool": {"name": name, **metadata}},
            ]
        )
    frames.append({"type": "result", "response": "Finished."})
    result, chat_log, deltas = await converse(hass, entity, user_input)
    assert speech(result) == "Finished."
    assert [item.role for item in chat_log.content] == ["system"] + [
        "assistant",
        "tool_result",
    ] * 4 + ["assistant"]
    for index in range(1, 9, 2):
        call = chat_log.content[index].tool_calls[0]
        tool_result = chat_log.content[index + 1]
        assert call.external is True
        assert call.id == tool_result.tool_call_id
        metadata = frames[index + 1]["tool"]
        key = next(key for key in metadata if key != "name")
        assert tool_result.tool_result[key] == metadata[key]
    assert "".join(delta.get("content", "") for delta in deltas) == "Finished."
    assert deltas[-1] == {"role": "assistant", "content": "Finished."}


@pytest.mark.parametrize(
    "code", ["request_canceled", "request_failed", "invalid_request"]
)
async def test_terminal_error_replaces_preview(hass, entity, user_input, peer, code):
    peer[0].extend(
        [
            {"type": "content", "text": "Partial answer"},
            {"type": "tool_call", "tool": {"name": "web.search"}},
            {"type": "error", "code": code, "message": "Stopped."},
        ]
    )
    result, _, deltas = await converse(hass, entity, user_input)
    assert speech(result) == "Stopped."
    assert "".join(delta.get("content", "") for delta in deltas) == "Stopped."


async def test_uncorrelated_validation_error(hass, entity, user_input, peer):
    peer[0].append(
        {
            "type": "error",
            "request_id": None,
            "code": "invalid_request",
            "message": "The request was invalid.",
        }
    )
    result, _, _ = await converse(hass, entity, user_input)
    assert speech(result) == "The request was invalid."


@pytest.mark.parametrize(
    "frame",
    [
        {"type": "result", "request_id": "different", "response": "Wrong user"},
        {"type": "result", "request_id": None, "response": "Missing ID"},
        {
            "type": "error",
            "request_id": None,
            "code": "request_failed",
            "message": "No ID",
        },
        {"type": "error", "request_id": None, "code": "invalid_request", "message": ""},
        {
            "type": "error",
            "request_id": "different",
            "code": "invalid_request",
            "message": "Mismatch",
        },
        {"type": "content", "text": ""},
        {"type": "result", "response": 42},
        {"type": "unknown"},
        WSMessage(WSMsgType.TEXT, "not json", ""),
        WSMessage(WSMsgType.TEXT, "[]", ""),
        WSMessage(WSMsgType.BINARY, b"binary", ""),
        {"type": "tool_result", "tool": {"name": "unmatched"}},
    ],
)
async def test_invalid_frame_is_uncertain_not_retried(
    hass, entity, user_input, peer, frame
):
    peer[0].append(frame)
    result, _, _ = await converse(hass, entity, user_input)
    assert speech(result) == _INTERRUPTED_RESPONSE
    peer[1].send_json.assert_awaited_once()
    peer[3].assert_called_once()


@pytest.mark.parametrize("failure", [None, ClientError(), TimeoutError(), OSError()])
async def test_interrupted_request(hass, entity, user_input, peer, failure):
    peer[0].append({"type": "content", "text": "Not a final answer"})
    if failure is not None:
        peer[0].append(failure)
    result, _, deltas = await converse(hass, entity, user_input)
    assert speech(result) == _INTERRUPTED_RESPONSE
    assert (
        "".join(delta.get("content", "") for delta in deltas) == _INTERRUPTED_RESPONSE
    )
    peer[2].__aexit__.assert_awaited_once()


@pytest.mark.parametrize("failure", [ClientError(), TimeoutError(), OSError()])
async def test_connection_failure(hass, entity, user_input, peer, failure):
    peer[2].__aenter__.side_effect = failure
    result, _, _ = await converse(hass, entity, user_input)
    assert speech(result) == _FAILURE_RESPONSE
    peer[1].send_json.assert_not_awaited()


@pytest.mark.parametrize("failure", [TimeoutError(), OSError()])
async def test_failed_send_has_unknown_outcome(hass, entity, user_input, peer, failure):
    peer[1].send_json.side_effect = failure
    result, _, _ = await converse(hass, entity, user_input)
    assert speech(result) == _INTERRUPTED_RESPONSE
    peer[1].send_json.assert_awaited_once()
    peer[2].__aexit__.assert_awaited_once()


async def test_ready_timeout_closes_without_submission(hass, entity, user_input, peer):
    peer[1].receive.side_effect = TimeoutError()
    result, _, _ = await converse(hass, entity, user_input)
    assert speech(result) == _FAILURE_RESPONSE
    peer[1].send_json.assert_not_awaited()
    peer[2].__aexit__.assert_awaited_once()


async def test_error_logs_do_not_include_transport_secrets(
    hass, entity, user_input, peer, caplog
):
    peer[2].__aenter__.side_effect = ClientError("private-token-and-url")
    await converse(hass, entity, user_input)
    assert "private-token-and-url" not in caplog.text
    assert "error_type=ClientError" in caplog.text


async def test_rejected_token_starts_reauth(hass, entity, user_input, peer):
    peer[2].__aenter__.side_effect = WSServerHandshakeError(MagicMock(), (), status=401)
    with patch.object(entity.entry, "async_start_reauth") as reauth:
        result, _, _ = await converse(hass, entity, user_input)
    assert speech(result) == _AUTH_RESPONSE
    reauth.assert_called_once_with(hass)


async def test_long_request_has_no_total_deadline(
    hass, entity, user_input, peer, freezer
):
    async def long_wait():
        freezer.tick(300)
        await asyncio.sleep(0)

    peer[0].extend(
        [long_wait, {"type": "result", "response": "Finished after five minutes."}]
    )
    result, _, _ = await converse(hass, entity, user_input)
    assert speech(result) == "Finished after five minutes."


async def test_caller_cancellation_closes_socket_without_retry(
    hass, entity, user_input, peer
):
    peer[0].append(asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await converse(hass, entity, user_input)
    peer[2].__aexit__.assert_awaited_once()
    peer[1].send_json.assert_awaited_once()


async def test_thinking_not_spoken(hass, entity, user_input, peer):
    peer[0].extend(
        [
            {"type": "thinking", "text": "Reasoning"},
            {"type": "result", "response": "Answer"},
        ]
    )
    result, chat_log, _ = await converse(hass, entity, user_input)
    assert chat_log.content[-1].thinking_content == "Reasoning"
    assert speech(result) == "Answer"


@pytest.mark.parametrize("fallback", [False, True])
async def test_missing_context_identity(hass, entity, user_input, peer, fallback):
    entity.default_user_id = user_input.context.user_id if fallback else None
    user_input.context = Context()
    peer[0].append({"type": "result", "response": "Answer"})
    result, _, _ = await converse(hass, entity, user_input)
    assert speech(result) == ("Answer" if fallback else _MISSING_USER_RESPONSE)
    assert peer[3].call_count == int(fallback)


async def test_disabled_fallback_not_used(hass, entity, user_input, peer):
    # HA's owner cannot be deactivated; create a separate fallback account.
    user = await hass.auth.async_create_user("Fallback")
    await hass.auth.async_deactivate_user(user)
    entity.default_user_id = user.id
    user_input.context = Context()
    result, _, _ = await converse(hass, entity, user_input)
    assert speech(result) == _MISSING_USER_RESPONSE
    peer[3].assert_not_called()


async def test_command_inline_export(hass, entity, user_input, peer):
    user_input.text = "/memories list"
    response = "Memory export\n" * 10000
    peer[0].append({"type": "result", "response": response})
    result, chat_log, _ = await converse(hass, entity, user_input)
    assert speech(result) == chat_log.content[-1].content == response
