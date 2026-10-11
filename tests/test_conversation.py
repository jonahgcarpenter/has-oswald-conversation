"""Replay OpenAI gateway chunks through Home Assistant's real ChatLog."""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import ClientError
from homeassistant.components.conversation import (
    ChatLog,
    ConversationEntityFeature,
    ConversationInput,
)
from homeassistant.components.conversation.chat_log import (
    AssistantContent,
    ToolResultContent,
    UserContent,
)
from homeassistant.core import Context
from homeassistant.helpers.llm import ToolInput
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.oswald_conversation.client import (
    ChatCompletionError,
    ConnectionFailedError,
    InvalidAuthError,
    InvalidResponseError,
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
    DOMAIN,
)
from custom_components.oswald_conversation.conversation import (
    _AUTH_RESPONSE,
    _FAILURE_RESPONSE,
    _INTERRUPTED_RESPONSE,
    _MISSING_CONVERSATION_RESPONSE,
    OswaldConversationEntity,
)
from custom_components.oswald_conversation.tools import BUILTIN_TOOL_NAMES

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


def test_entity_advertises_home_control(entity):
    assert (
        entity.supported_features & ConversationEntityFeature.CONTROL
    ) == ConversationEntityFeature.CONTROL


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
    # OpenAI standard: tool-call turn carries explicit null content.
    assert "content" in assistant_msg
    assert assistant_msg["content"] is None
    assert '"content": null' in json.dumps(assistant_msg)
    call_id = assistant_msg["tool_calls"][0]["id"]
    tool_msg = next(msg for msg in follow_up if msg.get("role") == "tool")
    assert tool_msg["tool_call_id"] == call_id
    assert json.loads(tool_msg["content"])


async def test_llm_context_uses_shared_assistant_key(
    hass, entity, user_input, chat_peer
):
    behaviors, _ = chat_peer
    behaviors.append([content_chunk("Hi."), stop_chunk()])
    # Fixtures use the entity id as agent; exposure is keyed by the
    # shared conversation assistant, like every core integration.
    assert user_input.agent_id == "conversation.oswald"
    _, chat_log, _ = await converse(hass, entity, user_input)
    assert chat_log.llm_api is not None
    assert chat_log.llm_api.llm_context.platform == DOMAIN
    assert chat_log.llm_api.llm_context.assistant == "conversation"


async def test_exposed_entity_visible_to_get_live_context(
    hass, entity, user_input, chat_peer
):
    from homeassistant.components.homeassistant.exposed_entities import (
        async_set_assistant_option,
        async_should_expose,
    )
    from homeassistant.helpers import entity_registry as er

    entry = er.async_get(hass).async_get_or_create(
        "light", "test", "kitchen_uid", suggested_object_id="kitchen"
    )
    hass.states.async_set(entry.entity_id, "on", {"friendly_name": "Kitchen Light"})
    async_set_assistant_option(
        hass, "conversation", entry.entity_id, "should_expose", True
    )

    # The old entity-id key never matches exposure configuration.
    assert not async_should_expose(hass, "conversation.oswald", entry.entity_id)
    assert async_should_expose(hass, "conversation", entry.entity_id)

    # Tool naming differs by HA version (namespaced on newer releases),
    # so discover the live-context tool instead of hardcoding it.
    from homeassistant.helpers import llm

    api = await llm.async_get_api(
        hass,
        "assist",
        llm.LLMContext(
            platform=DOMAIN,
            context=user_input.context,
            language=user_input.language,
            assistant="conversation",
            device_id=user_input.device_id,
        ),
    )
    tool_name = next(
        tool.name for tool in api.tools if tool.name.endswith("GetLiveContext")
    )

    behaviors, _ = chat_peer
    behaviors.append([tool_call_chunk(tool_name, "{}"), stop_chunk()])
    behaviors.append([content_chunk("The kitchen light is on."), stop_chunk()])
    result, chat_log, _ = await converse(hass, entity, user_input)
    assert speech(result) == "The kitchen light is on."
    tool_results = [
        item for item in chat_log.content if isinstance(item, ToolResultContent)
    ]
    assert len(tool_results) == 1
    assert tool_results[0].tool_result["success"] is True
    assert "error" not in tool_results[0].tool_result
    # The exposed light is what the tool saw.
    assert "domain: light" in tool_results[0].tool_result["result"]
    assert "state: 'on'" in tool_results[0].tool_result["result"]


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


def options_entity(hass, options):
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_BASE_URL: BASE_URL, CONF_API_KEY: "test-key", CONF_MODEL: MODEL},
        options=options,
    )
    entry.add_to_hass(hass)
    return OswaldConversationEntity(hass, entry)


async def test_options_override_model_prompt_apis_and_history(
    hass, user_input, chat_peer
):
    entity = options_entity(
        hass,
        {
            CONF_MODEL: "custom-model",
            CONF_PROMPT: "Custom instructions XYZ.",
            CONF_LLM_HASS_API: [],
            CONF_MAX_HISTORY: 2,
        },
    )
    behaviors, calls = chat_peer
    behaviors.append([content_chunk("Hi."), stop_chunk()])

    chat_log = ChatLog(hass, user_input.conversation_id)
    for index in range(3):
        chat_log.async_add_user_content(UserContent(content=f"old question {index}"))
        chat_log.async_add_assistant_content_without_tools(
            AssistantContent(
                agent_id="conversation.oswald", content=f"old answer {index}"
            )
        )
    chat_log.async_add_user_content(UserContent(content=user_input.text))
    result = await entity._async_handle_message(user_input, chat_log)

    assert speech(result) == "Hi."
    assert len(calls) == 1
    payload = calls[0]["payload"]
    assert payload["model"] == "custom-model"
    tools = {tool["function"]["name"] for tool in payload.get("tools", [])}
    assert tools.isdisjoint(BUILTIN_TOOL_NAMES | {"image_generate"})
    choice = payload["tool_choice"]
    assert choice["type"] == "allowed_tools"
    assert choice["allowed_tools"]["mode"] == "auto"
    allowed = [item["function"]["name"] for item in choice["allowed_tools"]["tools"]]
    assert allowed == [
        tool["function"]["name"] for tool in payload.get("tools", [])
    ] + [
        "memory",
        "session_search",
        "web_search",
        "vision_analyze",
    ]
    assert payload["messages"][0]["role"] == "system"
    assert "Custom instructions XYZ." in payload["messages"][0]["content"]
    rest = payload["messages"][1:]
    assert len(rest) == 2
    assert rest[-1] == {"role": "user", "content": "Hello"}
    assert rest[0] == {"role": "assistant", "content": "old answer 2"}


async def test_options_invalid_values_fall_back_to_defaults(
    hass, user_input, chat_peer
):
    entity = options_entity(
        hass,
        {CONF_PROMPT: "", CONF_LLM_HASS_API: "assist", CONF_MAX_HISTORY: 0},
    )
    assert entity.model == MODEL
    assert entity.llm_hass_apis == ["assist"]
    assert entity.max_history == 20

    behaviors, calls = chat_peer
    behaviors.append([content_chunk("Hi."), stop_chunk()])
    result, _, _ = await converse(hass, entity, user_input)
    assert speech(result) == "Hi."
    assert "tools" in calls[0]["payload"]


def test_trim_history_keeps_system_and_tail():
    messages = [
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]
    trimmed = OswaldConversationEntity._trim_history(messages, 2)
    assert trimmed == [messages[0], messages[2], messages[3]]
    assert OswaldConversationEntity._trim_history(messages, 10) == messages


def reasoning_chunk(text):
    return {"choices": [{"delta": {"reasoning_content": text}, "finish_reason": None}]}


async def test_builtin_tools_advertised_by_default(hass, entity, user_input, chat_peer):
    behaviors, calls = chat_peer
    behaviors.append([content_chunk("Hi."), stop_chunk()])
    result, _, _ = await converse(hass, entity, user_input)
    assert speech(result) == "Hi."
    payload = calls[0]["payload"]
    # tools carries HA tool schemas only — no builtin schemas are duplicated.
    tools = {tool["function"]["name"] for tool in payload["tools"]}
    assert tools.isdisjoint(BUILTIN_TOOL_NAMES | {"image_generate"})
    assert any(name.endswith("GetDateTime") for name in tools)
    # Opt-in travels via the tool_choice allow-list instead.
    choice = payload["tool_choice"]
    assert choice["type"] == "allowed_tools"
    assert choice["allowed_tools"]["mode"] == "auto"
    allowed = {item["function"]["name"] for item in choice["allowed_tools"]["tools"]}
    assert BUILTIN_TOOL_NAMES <= allowed
    assert "image_generate" not in allowed
    assert tools <= allowed


async def test_builtin_tools_subset_honored(hass, user_input, chat_peer):
    entity = options_entity(hass, {CONF_OSWALD_TOOLS: ["memory"]})
    behaviors, calls = chat_peer
    behaviors.append([content_chunk("Hi."), stop_chunk()])
    await converse(hass, entity, user_input)
    payload = calls[0]["payload"]
    tools = {tool["function"]["name"] for tool in payload["tools"]}
    assert "memory" not in tools
    allowed = {
        item["function"]["name"]
        for item in payload["tool_choice"]["allowed_tools"]["tools"]
    }
    assert "memory" in allowed
    assert "web_search" not in allowed
    assert "image_generate" not in allowed
    assert tools <= allowed


async def test_unofferable_names_never_listed(hass, user_input, chat_peer):
    entity = options_entity(
        hass, {CONF_OSWALD_TOOLS: ["memory", "image_generate", "no-such-tool"]}
    )
    assert entity.oswald_tools == ["memory"]
    behaviors, calls = chat_peer
    behaviors.append([content_chunk("Hi."), stop_chunk()])
    await converse(hass, entity, user_input)
    allowed = [
        item["function"]["name"]
        for item in calls[0]["payload"]["tool_choice"]["allowed_tools"]["tools"]
    ]
    assert "memory" in allowed
    assert "image_generate" not in allowed
    assert "no-such-tool" not in allowed


async def test_no_builtin_opt_in_omits_tool_choice(hass, user_input, chat_peer):
    entity = options_entity(hass, {CONF_OSWALD_TOOLS: []})
    behaviors, calls = chat_peer
    behaviors.append([content_chunk("Hi."), stop_chunk()])
    await converse(hass, entity, user_input)
    assert "tool_choice" not in calls[0]["payload"]
    assert "tools" in calls[0]["payload"]


async def test_tool_choice_repeated_on_continuation(
    hass, entity, user_input, chat_peer
):
    tool_name = await assist_tool_name(hass, user_input, "GetDateTime")
    behaviors, calls = chat_peer
    behaviors.append([tool_call_chunk(tool_name, "{}"), stop_chunk()])
    behaviors.append([content_chunk("Done."), stop_chunk()])
    await converse(hass, entity, user_input)
    assert len(calls) == 2
    first = [
        item["function"]["name"]
        for item in calls[0]["payload"]["tool_choice"]["allowed_tools"]["tools"]
    ]
    second = [
        item["function"]["name"]
        for item in calls[1]["payload"]["tool_choice"]["allowed_tools"]["tools"]
    ]
    assert first == second
    assert "memory" in first


async def test_reasoning_content_becomes_thinking(hass, entity, user_input, chat_peer):
    behaviors, calls = chat_peer
    behaviors.append(
        [reasoning_chunk("Considering."), content_chunk("Answer."), stop_chunk()]
    )
    result, chat_log, deltas = await converse(hass, entity, user_input)
    assert speech(result) == "Answer."
    assert chat_log.content[-1].thinking_content == "Considering."
    assert chat_log.content[-1].content == "Answer."
    assert "".join(delta.get("content", "") for delta in deltas) == "Answer."


async def test_reasoning_only_response_preserves_thinking(
    hass, entity, user_input, chat_peer
):
    behaviors, _ = chat_peer
    behaviors.append([reasoning_chunk("Considering."), stop_chunk()])
    result, chat_log, deltas = await converse(hass, entity, user_input)
    # A thinking-only turn succeeds with empty speech (HA core behavior),
    # not the failure message; the reasoning stays in the log for the
    # "Show details" dropdown.
    assert speech(result) == ""
    assert speech(result) != _FAILURE_RESPONSE
    thinking = [
        item
        for item in chat_log.content
        if isinstance(item, AssistantContent) and item.thinking_content
    ]
    assert len(thinking) == 1
    assert thinking[0].thinking_content == "Considering."
    assert any(delta.get("thinking_content") == "Considering." for delta in deltas)


async def test_thinking_echoed_to_server_on_continuation(
    hass, entity, user_input, chat_peer
):
    tool_name = await assist_tool_name(hass, user_input, "GetDateTime")
    behaviors, calls = chat_peer
    behaviors.append(
        [
            reasoning_chunk("Let me check. "),
            tool_call_chunk(tool_name, "{}"),
            stop_chunk(),
        ]
    )
    behaviors.append([content_chunk("Done."), stop_chunk()])
    result, chat_log, _ = await converse(hass, entity, user_input)
    assert speech(result) == "Done."
    assert len(calls) == 2
    assistant_msg = next(
        msg for msg in calls[1]["payload"]["messages"] if msg.get("tool_calls")
    )
    # OpenAI standard: tool-call turn carries explicit null content,
    # with the reasoning echoed back alongside the tool calls.
    assert assistant_msg["content"] is None
    assert assistant_msg["reasoning_content"] == "Let me check. "


def test_assistant_text_serializes_verbatim(entity):
    from types import SimpleNamespace

    chat_log = SimpleNamespace(
        content=[
            AssistantContent(agent_id="conversation.oswald", content="Hello there.")
        ]
    )
    assert entity._messages_from_log(chat_log) == [
        {"role": "assistant", "content": "Hello there."}
    ]


def test_assistant_tool_call_serializes_null_content(entity):
    from types import SimpleNamespace

    chat_log = SimpleNamespace(
        content=[
            AssistantContent(
                agent_id="conversation.oswald",
                content=None,
                tool_calls=[
                    ToolInput(tool_name="test_tool", tool_args={"a": 1}, id="call_1")
                ],
            )
        ]
    )
    messages = entity._messages_from_log(chat_log)
    assert len(messages) == 1
    message = messages[0]
    assert message["role"] == "assistant"
    assert "content" in message
    assert message["content"] is None
    assert '"content": null' in json.dumps(message)
    assert message["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "test_tool", "arguments": '{"a": 1}'},
        }
    ]


def test_assistant_empty_string_tool_call_serializes_null_content(entity):
    from types import SimpleNamespace

    chat_log = SimpleNamespace(
        content=[
            AssistantContent(
                agent_id="conversation.oswald",
                content="",
                tool_calls=[
                    ToolInput(tool_name="test_tool", tool_args={}, id="call_1")
                ],
            )
        ]
    )
    messages = entity._messages_from_log(chat_log)
    assert len(messages) == 1
    assert messages[0]["content"] is None


def test_assistant_without_content_or_calls_is_dropped(entity):
    from types import SimpleNamespace

    for content in (None, ""):
        chat_log = SimpleNamespace(
            content=[AssistantContent(agent_id="conversation.oswald", content=content)]
        )
        assert entity._messages_from_log(chat_log) == []


def test_assistant_thinking_serializes_reasoning_content(entity):
    from types import SimpleNamespace

    chat_log = SimpleNamespace(
        content=[
            AssistantContent(
                agent_id="conversation.oswald",
                content="Answer.",
                thinking_content="Considering.",
            )
        ]
    )
    assert entity._messages_from_log(chat_log) == [
        {
            "role": "assistant",
            "content": "Answer.",
            "reasoning_content": "Considering.",
        }
    ]


def test_assistant_thinking_only_serializes_reasoning_content(entity):
    from types import SimpleNamespace

    chat_log = SimpleNamespace(
        content=[
            AssistantContent(
                agent_id="conversation.oswald",
                content=None,
                thinking_content="Considering.",
            )
        ]
    )
    assert entity._messages_from_log(chat_log) == [
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Considering.",
        }
    ]


def test_parse_tool_args_repairs_mangled_arguments():
    from custom_components.oswald_conversation.conversation import _parse_tool_args

    assert _parse_tool_args("") == {}
    assert _parse_tool_args("{}") == {}
    assert _parse_tool_args('{"format": "iso"}') == {"format": "iso"}
    # Arguments wrapped in an extra string layer by a small model.
    assert _parse_tool_args('"{\\"format\\": \\"iso\\"}"') == {"format": "iso"}
    # Stringified per-value JSON is repaired; empty values are omitted.
    assert _parse_tool_args('{"a": "{\\"b\\": 1}", "c": "", "d": null}') == {
        "a": {"b": 1}
    }
    # Truly invalid arguments are still rejected.
    for raw in ("not-json{{{", "[1, 2]", "null", '"just a string"'):
        with pytest.raises(InvalidResponseError):
            _parse_tool_args(raw)


async def test_mangled_tool_args_still_call_tool(hass, entity, user_input, chat_peer):
    tool_name = await assist_tool_name(hass, user_input, "GetDateTime")
    behaviors, calls = chat_peer
    behaviors.append(
        [tool_call_chunk(tool_name, '"{\\"format\\": \\"iso\\"}"'), stop_chunk()]
    )
    behaviors.append([content_chunk("Done."), stop_chunk()])
    result, _, _ = await converse(hass, entity, user_input)
    assert speech(result) == "Done."
    assert len(calls) == 2


async def test_garbage_tool_args_are_failure(hass, entity, user_input, chat_peer):
    tool_name = await assist_tool_name(hass, user_input, "GetDateTime")
    behaviors, _ = chat_peer
    behaviors.append([tool_call_chunk(tool_name, "not-json{{{"), stop_chunk()])
    result, _, _ = await converse(hass, entity, user_input)
    assert speech(result) == _FAILURE_RESPONSE
