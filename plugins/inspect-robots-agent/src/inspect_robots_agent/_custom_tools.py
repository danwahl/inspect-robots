"""Tools a caller adds beside the built-in motion tools.

A custom tool answers the model in text and never moves the robot. It is
validated when the policy is constructed and dispatched by the toolset.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from inspect_robots.errors import ConfigError
from inspect_robots.types import Observation

#: Every name a built-in tool may take, reserved whether or not it is bound.
BUILTIN_TOOL_NAMES = frozenset({"move_to", "move_joints", "move_by", "done", "give_up", "take_pic"})

_TOOL_NAME = re.compile(r"[a-zA-Z0-9_-]{1,64}")


class ToolError(Exception):
    """Raised by a custom tool to return its message to the model as a correctable error."""


@dataclass(frozen=True)
class CustomTool:
    """A tool that answers the model in text without moving the robot.

    ``parameters`` is the JSON Schema object for the arguments, in the OpenAI
    function format the built-in tools use; it is shown to the model and not
    enforced. ``run`` receives the arguments as the model sent them and the
    observation the model is looking at, and returns the text the model reads.
    It raises [`ToolError`][inspect_robots_agent.ToolError] to reject its
    arguments. Any other exception propagates, and a result that is not a
    string raises ``TypeError``.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[[dict[str, Any], Observation], str]


def validated_custom_tools(tools: object) -> tuple[CustomTool, ...]:
    """Return the custom tools as a tuple, else raise a guided ``ConfigError``."""
    if isinstance(tools, (str, bytes)) or not isinstance(tools, Sequence):
        raise ConfigError(
            f"tools must be a sequence of CustomTool, got {tools!r}.\n"
            "fix: -P CLI flags cannot carry tools; pass them programmatically"
        )
    seen: set[str] = set()
    for tool in tools:
        if not isinstance(tool, CustomTool):
            raise ConfigError(f"tools must contain CustomTool instances, got {tool!r}")
        if not isinstance(tool.name, str) or not _TOOL_NAME.fullmatch(tool.name):
            raise ConfigError(
                f"custom tool name {tool.name!r} must match [a-zA-Z0-9_-]{{1,64}}, "
                "the strictest provider rule"
            )
        if tool.name in BUILTIN_TOOL_NAMES:
            raise ConfigError(
                f"custom tool name {tool.name!r} is reserved for a built-in tool; "
                f"reserved: {', '.join(sorted(BUILTIN_TOOL_NAMES))}"
            )
        if tool.name in seen:
            raise ConfigError(f"custom tool name {tool.name!r} is given twice")
        seen.add(tool.name)
        if not isinstance(tool.parameters, dict) or tool.parameters.get("type") != "object":
            raise ConfigError(
                f"custom tool {tool.name!r} parameters must be a JSON Schema object "
                '({"type": "object", ...})'
            )
        if not isinstance(tool.description, str):
            raise ConfigError(f"custom tool {tool.name!r} description must be a string")
        if not callable(tool.run):
            raise ConfigError(f"custom tool {tool.name!r} run must be callable")
    return tuple(tools)


def custom_tool_schema(tool: CustomTool) -> dict[str, Any]:
    """Return the tool's schema in the OpenAI function format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def run_custom_tool(tool: CustomTool, arguments: dict[str, Any], observation: Observation) -> str:
    """Return the tool's text; ``ToolError`` propagates and a non-string raises ``TypeError``."""
    text = tool.run(arguments, observation)
    if not isinstance(text, str):
        raise TypeError(f"custom tool {tool.name!r} run returned {type(text).__name__}, not str")
    return text
