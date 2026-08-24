from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlsplit

from aiohttp import ClientWebSocketResponse, WSMsgType

from .const import PROTOCOL_VERSION

READY_TIMEOUT_SECONDS = 10
CONNECTION_TIMEOUT_SECONDS = 10


class UnsupportedProtocolError(Exception):
    """Raised when the peer does not implement the expected protocol."""


def is_valid_ws_url(value: str) -> bool:
    """Return whether value is a structurally valid WebSocket URL."""
    try:
        parsed = urlsplit(value)
        parsed.port
    except ValueError:
        return False

    return (
        parsed.scheme in {"ws", "wss"}
        and parsed.hostname is not None
        and parsed.username is None
        and parsed.password is None
        and not parsed.fragment
    )


async def expect_ready(ws: ClientWebSocketResponse) -> None:
    """Require the protocol v1 ready frame as the first server message."""
    msg = await ws.receive(timeout=READY_TIMEOUT_SECONDS)
    if msg.type != WSMsgType.TEXT:
        raise UnsupportedProtocolError

    try:
        data: Any = json.loads(msg.data)
    except (json.JSONDecodeError, TypeError):
        raise UnsupportedProtocolError from None

    if data != {"type": "ready", "protocol_version": PROTOCOL_VERSION}:
        raise UnsupportedProtocolError
