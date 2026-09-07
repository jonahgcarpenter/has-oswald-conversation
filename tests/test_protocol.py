"""Handshake and URL contracts shared by setup and conversations."""

from unittest.mock import AsyncMock

import pytest
from aiohttp import WSMessage, WSMsgType

from custom_components.oswald_conversation.protocol import (
    UnsupportedProtocolError,
    expect_ready,
    is_valid_ws_url,
)


@pytest.mark.parametrize(
    "url,valid",
    [
        ("ws://localhost:8000/homeassistant/ws", True),
        ("wss://oswald.example/homeassistant/ws", True),
        ("http://oswald.example", False),
        ("ws://user:password@oswald.example", False),
        ("ws://oswald.example/#token", False),
        ("ws://oswald.example:invalid", False),
        ("ws://", False),
    ],
)
def test_url(url, valid):
    assert is_valid_ws_url(url) is valid


@pytest.mark.parametrize(
    "raw",
    [
        '{"type":"ready","protocol_version":2}',
        '{"type":"result"}',
        "[]",
        "null",
        "bad json",
    ],
)
async def test_invalid_ready(raw):
    ws = AsyncMock()
    ws.receive.return_value = WSMessage(WSMsgType.TEXT, raw, "")
    with pytest.raises(UnsupportedProtocolError):
        await expect_ready(ws)


async def test_ready_timeout():
    ws = AsyncMock()
    ws.receive.side_effect = TimeoutError
    with pytest.raises(TimeoutError):
        await expect_ready(ws)
    ws.receive.assert_awaited_once_with(timeout=10)
