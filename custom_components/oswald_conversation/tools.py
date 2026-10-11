"""Oswald server-side built-in tools.

These tools execute inside the Oswald agent loop; their calls and results
never appear on the wire (content-only streaming). The definitions below
are transcribed 1:1 from the Oswald server and are sent in the OpenAI
``tools`` array purely to advertise availability (opt-in per tool).
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OswaldBuiltinTool:
    name: str
    label: str
    description: str
    parameters: dict[str, Any]
    default_enabled: bool = True


OSWALD_BUILTIN_TOOLS: tuple[OswaldBuiltinTool, ...] = (
    OswaldBuiltinTool(
        name="web_search",
        label="Web search",
        description=(
            "Search the web for information. Returns up to 5 results by "
            "default with titles, URLs, and descriptions. The query is passed "
            "through to the configured backend, so operators such as "
            'site:domain, filetype:pdf, intitle:word, -term, and "exact '
            'phrase" may work when the backend supports them.'
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query to look up on the web. You may include "
                        "backend-supported operators such as site:example.com, "
                        'filetype:pdf, intitle:word, -term, or "exact phrase".'
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Maximum number of results to return. Defaults to 5."
                    ),
                    "default": 5,
                    "minimum": 1,
                    "maximum": 100,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    OswaldBuiltinTool(
        name="vision_analyze",
        label="Vision",
        description=(
            "Load an image into the conversation so you can see it. Call it when "
            "the user references a previous image that is not currently visible, "
            "then answer from what you see. Current attached images are already "
            "visible. Loads a still image (the first frame of a GIF), not a "
            "separate model's analysis."
        ),
        parameters={
            "type": "object",
            "properties": {
                "image_url": {
                    "type": "string",
                    "description": (
                        "Public HTTP/HTTPS image URL, an absolute path to an "
                        "unexpired image in the current profile's managed image "
                        "cache, or a base64 data URL for PNG/JPEG/GIF/WebP. "
                        "Other local files are not allowed."
                    ),
                },
                "question": {
                    "type": "string",
                    "description": (
                        "Your question or request about the image (1–2000 characters)."
                    ),
                },
                "region": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": (
                        "Optional [x1, y1, x2, y2] crop in source-image pixel "
                        "coordinates; x2/y2 are exclusive. Applied before further "
                        "downscaling. Load the full image first to get source "
                        "dimensions, then re-call to zoom into detail. Cached "
                        "attachments may already have been resized; original "
                        "detail cannot be recovered."
                    ),
                },
            },
            "required": ["image_url", "question"],
        },
    ),
    OswaldBuiltinTool(
        name="session_search",
        label="Session search",
        description=(
            "Recall past conversations: search or read old Oswald sessions (FTS5), "
            "or scroll inside one. Four shapes, picked by args: `query` = discovery "
            "(top-N matching sessions, top result fully hydrated); `session_id` + "
            "`around_message_id` = scroll (window of messages around an anchor); "
            "`session_id` alone = read a whole session — how you resolve an "
            "`@session:<profile>/<id>` link (split on '/' into profile + id); no "
            "args = browse recent sessions. Results are actual DB messages, no LLM. "
            "Searches conversation history ONLY — when the user gave a direct source "
            "(URL, file, contact, live system), inspect that first; never conclude "
            "'not found' from history alone. Use for questions about past "
            "conversations: 'what did we do about X', 'where did we leave Y'. When "
            "referring the user to a session, write its `link` value verbatim inline "
            "(it renders as a titled link)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query (discovery shape). Keywords, phrases, or "
                        "boolean expressions to find in past sessions. Omit to "
                        "browse recent sessions. Ignored when session_id + "
                        "around_message_id are set (scroll shape)."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Discovery shape only. Max sessions to return (default 3, "
                        "max 10). Bump to 5–10 when the topic likely spans several "
                        "sessions and you want to pick the right one to scroll into."
                    ),
                    "default": 3,
                },
                "sort": {
                    "type": "string",
                    "enum": ["newest", "oldest"],
                    "description": (
                        "Discovery shape only. Temporal bias on top of FTS5 ranking: "
                        "omit for relevance-only (exploratory recall), 'newest' for "
                        '"where did we leave X", \'oldest\' for "how did X start".'
                    ),
                },
                "detail": {
                    "type": "string",
                    "enum": ["adaptive", "full"],
                    "description": (
                        "Discovery shape only. 'adaptive' (default) fully hydrates "
                        "the top-ranked result and returns only the exact anchor "
                        "message for lower-ranked results. 'full' returns bookends "
                        "and the complete anchored window for every result."
                    ),
                    "default": "adaptive",
                },
                "after": {
                    "type": "string",
                    "description": (
                        "Discovery shape only. Inclusive lower bound on session start "
                        "time. ISO date/datetime (e.g. 2026-06-01) or relative "
                        "duration (7d, 24h, 2w = within the last N). Use only when "
                        "the user names a time frame. sort is a ranking bias, not a "
                        "bound."
                    ),
                },
                "before": {
                    "type": "string",
                    "description": (
                        "Discovery shape only. Exclusive upper bound on session start "
                        "time. ISO date/datetime (a date-only value is midnight UTC "
                        "that day) or relative duration (7d = older than a week). "
                        "Use only when the user names a time frame."
                    ),
                },
                "exclude_session_ids": {
                    "type": "array",
                    "description": (
                        "Discovery shape only. Session ids already inspected this "
                        "task. Those sessions and their lineage are omitted so a "
                        "later query explores instead of repeating the same hit. "
                        "Cap 20."
                    ),
                    "items": {"type": "string"},
                },
                "session_id": {
                    "type": "string",
                    "description": (
                        "Scroll shape. Session to read inside. Use the session_id "
                        "returned from a prior discovery call. Must be paired with "
                        "around_message_id."
                    ),
                },
                "around_message_id": {
                    "type": "integer",
                    "description": (
                        "Scroll shape. Message id to center the window on — use "
                        "match_message_id from a discovery result, or any id from a "
                        "prior window."
                    ),
                },
                "window": {
                    "type": "integer",
                    "description": (
                        "Scroll shape only. Messages to return on each side of the "
                        "anchor (anchor itself always included). Clamped to [1, 20]. "
                        "Default 5."
                    ),
                    "default": 5,
                },
                "role_filter": {
                    "type": "string",
                    "description": (
                        "Optional. Comma-separated roles to include. Discovery "
                        "defaults to 'user,assistant' (tool output is usually "
                        "noise). Pass 'user,assistant,tool' to include tool output "
                        "(debugging tool behaviour) or 'tool' to search tool output "
                        "only."
                    ),
                },
                "profile": {
                    "type": "string",
                    "description": (
                        "Optional. Read sessions from another Oswald profile's "
                        "database (read-only). Use when resolving an "
                        "`@session:<profile>/<id>` link: pass the profile segment "
                        "here with session_id as the id segment. Omit to use the "
                        "current profile."
                    ),
                },
            },
        },
    ),
    OswaldBuiltinTool(
        name="memory",
        label="Memory",
        description=(
            "Save durable facts to persistent memory that survive across sessions. "
            "Memory is captured in the system prompt once per session; edits take "
            "effect in new sessions, not in the current session's frozen prompt. "
            "Keep entries compact and high-signal.\n"
            "\n"
            "HOW: make ALL your changes in ONE call via an 'operations' array "
            "(each item: {action, content?, old_text?}). The batch applies "
            "atomically and the char limit is checked only on the FINAL result — "
            "so a single call can remove/replace stale entries to free room AND "
            "add new ones, even when an add alone would overflow. The response "
            "reports current/limit chars and confirms completion; one batch call "
            "finishes the update, so don't repeat it. Use the bare "
            "action/content/old_text fields only for a single lone change.\n"
            "\n"
            "WHEN: only for facts that apply to EVERY session regardless of task: "
            "who the user is, stable environment facts, standing conventions with "
            "no task home. Anything learned while doing a task (procedures, "
            "pitfalls, and the user's preferences and corrections for that kind of "
            "work) belongs in the task's skill via skill_manage, where it loads "
            "only when relevant; memory is injected into every turn and must stay "
            "small.\n"
            "\n"
            "IF FULL: an add is rejected with the current entries shown. Reissue as "
            "ONE batch that removes or shortens enough stale entries and adds the "
            "new one together.\n"
            "\n"
            "TARGETS: 'user' = who the user is (name, role, preferences, style). "
            "'memory' = your notes (environment, conventions, tool quirks, lessons).\n"
            "\n"
            "SKIP: trivial/obvious info, easily re-discovered facts, raw data dumps, "
            "task progress, completed-work logs, temporary TODO state (use "
            "session_search for those). Reusable procedures belong in a skill, not "
            "memory."
        ),
        parameters={
            "type": "object",
            "required": ["target"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "replace", "remove"],
                    "description": (
                        "The action to perform (single-op shape). Omit when using "
                        "'operations'."
                    ),
                },
                "target": {
                    "type": "string",
                    "enum": ["memory", "user"],
                    "description": (
                        "Which memory store: 'memory' for personal notes, 'user' "
                        "for user profile."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": (
                        "The entry content. Required for 'add' and 'replace'. For "
                        "'replace' it is the COMPLETE new entry text: the whole "
                        "matched entry is overwritten, so include everything you "
                        "want to keep. Alias: 'new_text' is also accepted (same "
                        "full-entry meaning)."
                    ),
                },
                "old_text": {
                    "type": "string",
                    "description": (
                        "REQUIRED for 'replace' and 'remove' (single-op shape): a "
                        "short unique substring IDENTIFYING the existing entry to "
                        "modify -- it locates the entry, it is not spliced out. "
                        "Omit only for 'add'."
                    ),
                },
                "new_text": {
                    "type": "string",
                    "description": (
                        "Alias for 'content' (single-op shape): the COMPLETE new "
                        "entry for 'replace', not a patch of old_text. If both are "
                        "set, 'content' wins."
                    ),
                },
                "operations": {
                    "type": "array",
                    "description": (
                        "Batch shape: a list of operations applied atomically in one "
                        "call against the final char budget. Preferred when making "
                        "multiple changes or consolidating to make room. Each item "
                        "is {action, content?, old_text?}."
                    ),
                    "items": {
                        "type": "object",
                        "required": ["action"],
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["add", "replace", "remove"],
                            },
                            "content": {
                                "type": "string",
                                "description": (
                                    "Entry content for add/replace. For replace, the "
                                    "COMPLETE new entry (whole entry is overwritten). "
                                    "Alias: 'new_text'."
                                ),
                            },
                            "new_text": {
                                "type": "string",
                                "description": "Alias for 'content' in a batch op.",
                            },
                            "old_text": {
                                "type": "string",
                                "description": (
                                    "Substring identifying the entry for "
                                    "replace/remove."
                                ),
                            },
                        },
                    },
                },
            },
        },
    ),
    OswaldBuiltinTool(
        name="image_generate",
        label="Image generation",
        description=(
            "Generate an image from a text description, or transform an existing "
            "image by supplying image_url. For transformations, describe the "
            "complete desired final image and the details to retain. Editing affects "
            "the whole image and may change faces, geometry, or other details; "
            "exact preservation and localized edits are not guaranteed. Returns the "
            "image directly in the tool result with its private absolute cache path "
            "and source dimensions. The model preview may be downscaled; the latest "
            "successful original version is delivered as an attachment with the "
            "final response."
        ),
        parameters={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Describe the desired final image, including subject, "
                        "composition, style, lighting, and colors. When editing, "
                        "clearly describe the change and the existing details to "
                        "retain."
                    ),
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": (
                        "Output aspect ratio: landscape is 1280x720, square is "
                        "1024x1024, and portrait is 720x1280. When editing, match "
                        "the source ratio where possible to avoid center cropping."
                    ),
                    "enum": ["landscape", "square", "portrait"],
                    "default": "landscape",
                },
                "image_url": {
                    "type": "string",
                    "description": (
                        "Optional source image: a public HTTPS image URL or an "
                        "absolute path to an unexpired image in the current user's "
                        "managed image cache. Omit to generate a new image from text."
                    ),
                },
            },
            "required": ["prompt"],
        },
    ),
)

BUILTIN_TOOL_NAMES: frozenset[str] = frozenset(
    tool.name for tool in OSWALD_BUILTIN_TOOLS
)

DEFAULT_OSWALD_TOOLS: list[str] = [
    tool.name for tool in OSWALD_BUILTIN_TOOLS if tool.default_enabled
]


def builtin_definitions(enabled: Collection[str]) -> list[dict[str, Any]]:
    """Return OpenAI function definitions for the enabled built-in tools."""
    wanted = set(enabled or [])
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in OSWALD_BUILTIN_TOOLS
        if tool.name in wanted
    ]
