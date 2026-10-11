"""Unit tests for the Oswald built-in tool catalog (no Home Assistant)."""

from custom_components.oswald_conversation.tools import (
    BUILTIN_TOOL_NAMES,
    DEFAULT_OSWALD_TOOLS,
    OSWALD_BUILTIN_TOOLS,
    builtin_definitions,
)


def test_catalog_names():
    assert [tool.name for tool in OSWALD_BUILTIN_TOOLS] == [
        "web_search",
        "vision_analyze",
        "session_search",
        "memory",
        "image_generate",
    ]
    assert BUILTIN_TOOL_NAMES == {
        "web_search",
        "vision_analyze",
        "session_search",
        "memory",
        "image_generate",
    }
    assert set(DEFAULT_OSWALD_TOOLS) == BUILTIN_TOOL_NAMES
    assert len(DEFAULT_OSWALD_TOOLS) == len(OSWALD_BUILTIN_TOOLS)


def test_catalog_required_fields():
    by_name = {tool.name: tool for tool in OSWALD_BUILTIN_TOOLS}
    assert by_name["web_search"].parameters["required"] == ["query"]
    assert by_name["vision_analyze"].parameters["required"] == [
        "image_url",
        "question",
    ]
    assert by_name["memory"].parameters["required"] == ["target"]
    assert by_name["image_generate"].parameters["required"] == ["prompt"]
    assert "required" not in by_name["session_search"].parameters


def test_catalog_spot_checks():
    by_name = {tool.name: tool for tool in OSWALD_BUILTIN_TOOLS}
    limit = by_name["web_search"].parameters["properties"]["limit"]
    assert (limit["default"], limit["minimum"], limit["maximum"]) == (5, 1, 100)
    assert by_name["web_search"].parameters["additionalProperties"] is False
    region = by_name["vision_analyze"].parameters["properties"]["region"]
    assert region["type"] == "array"
    assert (region["minItems"], region["maxItems"]) == (4, 4)
    detail = by_name["session_search"].parameters["properties"]["detail"]
    assert detail["enum"] == ["adaptive", "full"]
    assert detail["default"] == "adaptive"
    aspect = by_name["image_generate"].parameters["properties"]["aspect_ratio"]
    assert aspect["enum"] == ["landscape", "square", "portrait"]
    assert aspect["default"] == "landscape"
    for tool in OSWALD_BUILTIN_TOOLS:
        assert tool.name
        assert tool.label
        assert tool.description
        assert tool.parameters["type"] == "object"
        assert isinstance(tool.parameters["properties"], dict)


def test_builtin_definitions_filters_to_enabled():
    definitions = builtin_definitions(["web_search", "memory", "no-such-tool"])
    assert [d["function"]["name"] for d in definitions] == ["web_search", "memory"]
    for definition in definitions:
        assert definition["type"] == "function"
        assert definition["function"]["description"]
        assert definition["function"]["parameters"]["type"] == "object"
    assert builtin_definitions([]) == []
    assert len(builtin_definitions(DEFAULT_OSWALD_TOOLS)) == len(OSWALD_BUILTIN_TOOLS)
