"""Unit tests for the OpenAI-compatible HTTP client (no Home Assistant)."""

import json
from typing import Any

import pytest

from custom_components.oswald_conversation.client import (
    ChatCompletionError,
    ConnectionFailedError,
    InvalidAuthError,
    InvalidResponseError,
    async_list_models,
    async_stream_chat_completion,
    is_valid_base_url,
    normalize_base_url,
)

BASE_URL = "http://127.0.0.1:8080/v1"


@pytest.mark.parametrize(
    ("url", "valid"),
    [
        ("http://127.0.0.1:8080/v1", True),
        ("http://127.0.0.1:8080/v1/", True),
        ("  https://oswald.example/v1  ", True),
        ("https://oswald.example", True),
        ("http://localhost:8080/openai/v1", True),
        ("ws://127.0.0.1:8080/v1", False),
        ("http://user:pass@127.0.0.1:8080/v1", False),
        ("http://127.0.0.1:8080/v1#frag", False),
        ("http://127.0.0.1:notaport/v1", False),
        ("http://", False),
        ("not a url", False),
        ("", False),
    ],
)
def test_base_url_validation(url, valid):
    assert is_valid_base_url(url) is valid


def test_normalize_base_url():
    assert (
        normalize_base_url("  http://127.0.0.1:8080/v1// ")
        == "http://127.0.0.1:8080/v1"
    )


class FakeContent:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    async def __aiter__(self):
        for line in self._lines:
            yield line


class FakeResponse:
    def __init__(
        self,
        status: int = 200,
        json_data: Any = None,
        json_error: Exception | None = None,
        sse_lines: list[bytes] | None = None,
    ) -> None:
        self.status = status
        self._json_data = json_data
        self._json_error = json_error
        self.content = FakeContent(sse_lines or [])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._json_data


class FakeSession:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def _call(self, method: str, url: str, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        if self.error is not None:
            raise self.error
        return self.response

    def get(self, url: str, **kwargs):
        return self._call("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        return self._call("POST", url, **kwargs)


async def test_list_models_success():
    session = FakeSession(
        FakeResponse(
            json_data={
                "object": "list",
                "data": [
                    {"id": "oswald-large"},
                    {"id": "oswald-small"},
                    {"id": "oswald-large"},
                    {"id": "  "},
                    {"nope": True},
                ],
            }
        )
    )
    models = await async_list_models(session, f"{BASE_URL}/", "key")
    assert models == ["oswald-large", "oswald-small"]
    call = session.calls[0]
    assert call["method"] == "GET"
    assert call["url"] == f"{BASE_URL}/models"
    assert call["headers"] == {"Authorization": "Bearer key"}


@pytest.mark.parametrize("status", [401, 403])
async def test_list_models_rejects_bad_key(status):
    session = FakeSession(FakeResponse(status=status))
    with pytest.raises(InvalidAuthError):
        await async_list_models(session, BASE_URL, "bad")


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(status=500),
        FakeResponse(status=404),
        FakeResponse(json_data={"unexpected": True}),
        FakeResponse(json_data={"data": "not-a-list"}),
        FakeResponse(json_data=None, json_error=ValueError("bad json")),
    ],
)
async def test_list_models_failures(response):
    session = FakeSession(response)
    with pytest.raises((ConnectionFailedError, InvalidResponseError)):
        await async_list_models(session, BASE_URL, "key")


async def test_list_models_network_error():
    session = FakeSession(error=OSError("down"))
    with pytest.raises(ConnectionFailedError):
        await async_list_models(session, BASE_URL, "key")


def sse(*lines: str) -> list[bytes]:
    return [(line + "\n").encode() for line in lines]


async def test_stream_content_and_tool_fragments():
    tool_first = json.dumps(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {
                                    "name": "GetDateTime",
                                    "arguments": '{"format":',
                                },
                            }
                        ]
                    }
                }
            ]
        }
    )
    tool_second = json.dumps(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": ' "iso"}'},
                            }
                        ]
                    }
                }
            ]
        }
    )
    session = FakeSession(
        FakeResponse(
            sse_lines=sse(
                ": comment line is ignored",
                f"data: {json.dumps({'choices': [{'delta': {'content': 'Hello'}}]})}",
                "",
                f"data: {tool_first}",
                "",
                f"data: {tool_second}",
                "",
                "data: [DONE]",
                "",
            )
        )
    )
    chunks = [
        chunk
        async for chunk in async_stream_chat_completion(
            session, BASE_URL, "key", {"model": "m", "stream": True}
        )
    ]
    assert len(chunks) == 3
    assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
    call = session.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == f"{BASE_URL}/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer key"


@pytest.mark.parametrize("status", [401, 403])
async def test_stream_rejects_bad_key(status):
    session = FakeSession(FakeResponse(status=status))
    with pytest.raises(InvalidAuthError):
        async for _ in async_stream_chat_completion(
            session, BASE_URL, "bad", {"model": "m"}
        ):
            pass


async def test_stream_surfaces_server_error_message():
    session = FakeSession(
        FakeResponse(status=400, json_data={"error": {"message": "Unknown model."}})
    )
    with pytest.raises(ChatCompletionError) as exc_info:
        async for _ in async_stream_chat_completion(
            session, BASE_URL, "key", {"model": "nope"}
        ):
            pass
    assert exc_info.value.message == "Unknown model."
    assert exc_info.value.status == 400


async def test_stream_invalid_chunk():
    session = FakeSession(FakeResponse(sse_lines=sse("data: not json", "")))
    with pytest.raises(ChatCompletionError):
        async for _ in async_stream_chat_completion(
            session, BASE_URL, "key", {"model": "m"}
        ):
            pass
