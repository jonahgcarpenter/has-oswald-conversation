"""Oswald server-side built-in tools.

These tools execute inside the Oswald agent loop; their calls and results
stream via the `oswald_tool_calls` delta extension and render as
thinking details (never as HA tool calls). They are opt-in per
request via the ``tool_choice`` allow-list — never default. Only the names
in ``SERVER_OFFERABLE_TOOLS`` may be listed; anything else (including
``image_generate`` and unknown names) makes the server reject the request
with 400 ``invalid_tool_choice``.

No schemas are duplicated here: the server owns the definitions. This
module only tracks which names exist for the settings UI and allow-list.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass


@dataclass(frozen=True)
class OswaldBuiltinTool:
    name: str
    label: str
    default_enabled: bool = True


OSWALD_BUILTIN_TOOLS: tuple[OswaldBuiltinTool, ...] = (
    OswaldBuiltinTool(name="memory", label="Memory"),
    OswaldBuiltinTool(name="session_search", label="Session search"),
    OswaldBuiltinTool(name="web_search", label="Web search"),
    OswaldBuiltinTool(name="vision_analyze", label="Vision"),
    OswaldBuiltinTool(name="tool_search", label="Tool search"),
    OswaldBuiltinTool(name="tool_describe", label="Tool describe"),
    OswaldBuiltinTool(name="tool_call", label="Tool call"),
)

SERVER_OFFERABLE_TOOLS: frozenset[str] = frozenset(
    tool.name for tool in OSWALD_BUILTIN_TOOLS
)

BUILTIN_TOOL_NAMES: frozenset[str] = SERVER_OFFERABLE_TOOLS

DEFAULT_OSWALD_TOOLS: list[str] = [
    tool.name for tool in OSWALD_BUILTIN_TOOLS if tool.default_enabled
]


def offerable_builtin_names(enabled: Collection[str] | None) -> list[str]:
    """Return the enabled names the server accepts, in catalog order."""
    wanted = set(enabled or [])
    return [tool.name for tool in OSWALD_BUILTIN_TOOLS if tool.name in wanted]
