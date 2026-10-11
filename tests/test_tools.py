"""Unit tests for the Oswald built-in tool registry (no Home Assistant)."""

from custom_components.oswald_conversation.tools import (
    BUILTIN_TOOL_NAMES,
    DEFAULT_OSWALD_TOOLS,
    OSWALD_BUILTIN_TOOLS,
    SERVER_OFFERABLE_TOOLS,
    offerable_builtin_names,
)


def test_registry_names():
    assert [tool.name for tool in OSWALD_BUILTIN_TOOLS] == [
        "memory",
        "session_search",
        "web_search",
        "vision_analyze",
    ]
    assert "image_generate" not in BUILTIN_TOOL_NAMES
    assert BUILTIN_TOOL_NAMES == SERVER_OFFERABLE_TOOLS
    assert set(DEFAULT_OSWALD_TOOLS) == SERVER_OFFERABLE_TOOLS
    assert len(DEFAULT_OSWALD_TOOLS) == len(OSWALD_BUILTIN_TOOLS)
    for tool in OSWALD_BUILTIN_TOOLS:
        assert tool.name
        assert tool.label
        assert tool.default_enabled is True


def test_offerable_builtin_names():
    assert offerable_builtin_names(["memory", "web_search"]) == [
        "memory",
        "web_search",
    ]
    # Unknown names and image_generate are never offerable.
    assert offerable_builtin_names(["memory", "image_generate", "no-such-tool"]) == [
        "memory"
    ]
    assert offerable_builtin_names([]) == []
    assert offerable_builtin_names(None) == []
