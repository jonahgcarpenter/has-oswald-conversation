"""Minimal by-the-book OpenAI-compatible HTTP client for Oswald."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any
from urllib.parse import urlsplit

from aiohttp import ClientSession, ClientTimeout

MODELS_TIMEOUT_SECONDS = 10
CONNECT_TIMEOUT_SECONDS = 10


class InvalidAuthError(Exception):
    """Raised when the server rejects the API key (HTTP 401/403)."""


class ConnectionFailedError(Exception):
    """Raised when the server cannot be reached or returns an HTTP error."""


class ChatCompletionError(Exception):
    """Raised when a chat completion fails after the request was sent."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


class InvalidResponseError(Exception):
    """Raised when the server response is not OpenAI-shaped."""


def normalize_base_url(value: str) -> str:
    """Strip whitespace and trailing slashes from a base URL."""
    return value.strip().rstrip("/")


def is_valid_base_url(value: str) -> bool:
    """Return whether value is a structurally valid http(s) base URL."""
    try:
        parsed = urlsplit(value.strip())
        _ = parsed.port  # Access validates malformed or out-of-range ports.
    except ValueError:
        return False

    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname is not None
        and parsed.username is None
        and parsed.password is None
        and not parsed.fragment
    )


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _error_message(payload: Any) -> str | None:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return None


async def async_list_models(
    session: ClientSession, base_url: str, api_key: str
) -> list[str]:
    """Return the model ids reported by GET {base_url}/models."""
    url = f"{normalize_base_url(base_url)}/models"
    try:
        async with session.get(
            url,
            headers=_headers(api_key),
            timeout=ClientTimeout(total=MODELS_TIMEOUT_SECONDS),
        ) as resp:
            if resp.status in (401, 403):
                raise InvalidAuthError
            if resp.status >= 400:
                raise ConnectionFailedError(f"HTTP {resp.status}")
            try:
                payload: Any = await resp.json()
            except (ValueError, TypeError) as err:
                raise InvalidResponseError("Invalid JSON") from err
    except InvalidAuthError:
        raise
    except InvalidResponseError:
        raise
    except ConnectionFailedError:
        raise
    except Exception as err:
        raise ConnectionFailedError(type(err).__name__) from err

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise InvalidResponseError("Missing data list")
    models = [
        item["id"]
        for item in data
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    models = [model for model in models if isinstance(model, str) and model.strip()]
    return sorted(set(models))


async def async_stream_chat_completion(
    session: ClientSession, base_url: str, api_key: str, payload: dict[str, Any]
) -> AsyncGenerator[dict[str, Any], None]:
    """Yield parsed SSE data chunks from POST {base_url}/chat/completions."""
    url = f"{normalize_base_url(base_url)}/chat/completions"
    timeout = ClientTimeout(
        total=None,
        connect=CONNECT_TIMEOUT_SECONDS,
        sock_connect=CONNECT_TIMEOUT_SECONDS,
    )
    try:
        resp_cm = session.post(
            url,
            json=payload,
            headers={**_headers(api_key), "Accept": "text/event-stream"},
            timeout=timeout,
        )
    except Exception as err:
        raise ConnectionFailedError(type(err).__name__) from err

    try:
        async with resp_cm as resp:
            if resp.status in (401, 403):
                raise InvalidAuthError
            if resp.status >= 400:
                message: str | None = None
                try:
                    message = _error_message(await resp.json())
                except (ValueError, TypeError):
                    message = None
                raise ChatCompletionError(
                    message or f"Oswald returned HTTP {resp.status}.",
                    status=resp.status,
                )
            try:
                async for chunk in _aiter_sse_data(resp):
                    yield chunk
            except (ChatCompletionError, InvalidAuthError):
                raise
            except Exception as err:
                raise ChatCompletionError(type(err).__name__) from err
    except (ChatCompletionError, InvalidAuthError):
        raise
    except Exception as err:
        raise ConnectionFailedError(type(err).__name__) from err


async def _aiter_sse_data(resp: Any) -> AsyncGenerator[dict[str, Any], None]:
    """Yield parsed JSON payloads from an SSE byte stream."""
    pending: list[str] = []
    async for raw_line in resp.content:
        try:
            line = raw_line.decode("utf-8", errors="replace").strip()
        except (UnicodeError, AttributeError) as err:
            raise InvalidResponseError("Invalid SSE encoding") from err
        if not line:
            if pending:
                data = "\n".join(pending)
                pending.clear()
                chunk = _parse_sse_data(data)
                if chunk is not None:
                    yield chunk
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            pending.append(line[5:].strip())
            continue
        # Ignore other SSE fields such as "event:" or "id:".
    if pending:
        chunk = _parse_sse_data("\n".join(pending))
        if chunk is not None:
            yield chunk


def _parse_sse_data(data: str) -> dict[str, Any] | None:
    if data == "[DONE]":
        return None
    try:
        chunk: Any = json.loads(data)
    except (json.JSONDecodeError, TypeError) as err:
        raise InvalidResponseError("Invalid SSE JSON") from err
    if not isinstance(chunk, dict):
        raise InvalidResponseError("Invalid SSE chunk")
    return chunk
