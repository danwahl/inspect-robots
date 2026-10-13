"""Custom tools: read-only answers beside the motion tool (plan 0086)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from inspect_robots import eval as ir_eval
from inspect_robots.errors import ConfigError
from inspect_robots.logging.sink import NullSink
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.rollout import TrialRecord
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task
from inspect_robots.types import Observation
from inspect_robots_agent import CustomTool, LLMAgentPolicy, ToolError
from inspect_robots_agent._anthropic import _translate_tools as anthropic_tools
from inspect_robots_agent._interactions import _translate_tool as interactions_tool
from inspect_robots_agent._responses import _translate_tools as responses_tools
from inspect_robots_agent.policy import (
    _ON_DEMAND_SYSTEM_TEMPLATE,
    _SYSTEM_TEMPLATE,
    AgentPolicyConfig,
)

_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {"x": {"type": "number", "description": "A value to double."}},
    "required": ["x"],
}


class _Double:
    """Doubles ``x`` and remembers what it was called with."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], Observation]] = []

    def __call__(self, arguments: dict[str, Any], observation: Observation) -> str:
        self.calls.append((arguments, observation))
        if not isinstance(arguments.get("x"), (int, float)):
            raise ToolError("x must be a number")
        return f"2x = {2 * arguments['x']}"


def _double_tool(run: Any = None) -> CustomTool:
    return CustomTool(
        name="double",
        description="Double a number.",
        parameters=_PARAMETERS,
        run=run if run is not None else _Double(),
    )


def _call(name: str, arguments: dict[str, Any], index: int = 0) -> dict[str, Any]:
    if name == "move_by":
        arguments = {"note": "moving toward the cube", **arguments}
    return {
        "id": f"call_{index}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _turn(*calls: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    tool_calls = [_call(name, arguments, i) for i, (name, arguments) in enumerate(calls)]
    return {
        "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": tool_calls}}]
    }


class _Script:
    """Serves queued responses; repeats the last one when the queue runs dry."""

    def __init__(self, responses: list[dict[str, Any]]):
        self.queue = list(responses)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        payload = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        return httpx.Response(200, json=payload)


def _policy(script: _Script, **kwargs: Any) -> LLMAgentPolicy:
    return LLMAgentPolicy(
        model="test/model",
        base_url="http://llm.test/v1",
        transport=httpx.MockTransport(script),
        env={},
        **kwargs,
    )


def _task() -> Task:
    return Task(
        name="t",
        scenes=[Scene(id="s0", instruction="reach the cube", init_seed=0)],
        scorer=success_at_end(),
        max_steps=40,
    )


class _Records(NullSink):
    def __init__(self) -> None:
        self.records: list[TrialRecord] = []

    def on_trial_end(self, record: TrialRecord) -> None:
        self.records.append(record)


def _tool_messages(request: dict[str, Any]) -> list[str]:
    return [m["content"] for m in request["messages"] if m["role"] == "tool"]


def _first_act(
    script: _Script, tool: CustomTool, **kwargs: Any
) -> tuple[Any, Observation, LLMAgentPolicy]:
    """Bind to the cube world and run one act() on its opening observation."""
    embodiment = CubePickEmbodiment()
    policy = _policy(script, tools=[tool], **kwargs)
    policy.bind(embodiment.info)
    scene = Scene(id="s0", instruction="reach the cube", init_seed=0)
    policy.reset(scene)
    observation = embodiment.reset(scene, seed=0)
    return policy.act(observation), observation, policy


def _transcript_tool_messages(policy: LLMAgentPolicy) -> list[str]:
    transcript = policy.transcript()
    assert transcript is not None
    return [m["content"] for m in transcript if m["role"] == "tool"]


def test_schemas_follow_the_builtins_and_are_absent_when_unset(tmp_path: Path) -> None:
    done = _turn(("done", {"summary": "ok", "hindsight": "none"}))
    without, with_tool = _Script([done]), _Script([done])
    ir_eval(_task(), _policy(without), CubePickEmbodiment(), log_dir=str(tmp_path))
    ir_eval(
        _task(),
        _policy(with_tool, tools=[_double_tool()]),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
    )

    builtins = without.requests[0]["tools"]
    tools = with_tool.requests[0]["tools"]
    assert tools[: len(builtins)] == builtins
    assert tools[len(builtins) :] == [
        {
            "type": "function",
            "function": {
                "name": "double",
                "description": "Double a number.",
                "parameters": _PARAMETERS,
            },
        }
    ]


def test_a_custom_call_before_the_motion_answers_and_the_motion_runs(tmp_path: Path) -> None:
    double = _Double()
    script = _Script(
        [
            _turn(("double", {"x": 1.5}), ("move_by", {"deltas": {"dx": 0.01}})),
            _turn(("done", {"summary": "ok", "hindsight": "none"})),
        ]
    )
    records = _Records()
    ir_eval(
        _task(),
        _policy(script, tools=[_double_tool(double)]),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        sinks=[records],
    )

    ((arguments, observation),) = double.calls
    assert arguments == {"x": 1.5}
    assert "eef_pos" in observation.state
    answer, motion = _tool_messages(script.requests[1])
    assert answer == "2x = 3.0"
    assert motion.startswith("executing move_by")
    assert len(records.records[0].steps) > 0


def test_a_custom_only_turn_returns_to_the_model_with_the_answer() -> None:
    double = _Double()
    script = _Script([_turn(("double", {"x": 2})), _turn(("move_by", {"deltas": {"dx": 0.01}}))])
    chunk, observation, _ = _first_act(script, _double_tool(double))

    assert double.calls[0][1] is observation
    assert len(script.requests) == 2
    assert _tool_messages(script.requests[1]) == ["2x = 4"]
    assert len(chunk) > 0


def test_a_custom_call_after_the_motion_is_ignored() -> None:
    double = _Double()
    script = _Script([_turn(("move_by", {"deltas": {"dx": 0.01}}), ("double", {"x": 2}))])
    _, _, policy = _first_act(script, _double_tool(double))

    assert double.calls == []
    assert _transcript_tool_messages(policy)[-1] == ("ignored: the turn closed before double")


def test_custom_calls_chain_before_take_pic_but_not_after() -> None:
    double = _Double()
    move = _turn(("move_by", {"deltas": {"dx": 0.01}}))
    before = _Script([_turn(("double", {"x": 2}), ("take_pic", {"note": "look"})), move])
    _first_act(before, _double_tool(double), images="on_demand")
    after = _Script([_turn(("take_pic", {"note": "look"}), ("double", {"x": 2})), move])
    _first_act(after, _double_tool(double), images="on_demand")

    assert len(double.calls) == 1
    answer, captured = _tool_messages(before.requests[1])
    assert answer == "2x = 4"
    assert captured.startswith("captured")
    assert _tool_messages(after.requests[1])[1] == ("ignored: the turn closed before double")


def test_tool_errors_count_toward_the_failure_budget_and_successes_do_not_reset_it() -> None:
    bad = _turn(("double", {"x": "two"}))
    script = _Script(
        [bad, _turn(("double", {"x": 2})), bad, _turn(("move_by", {"deltas": {"dx": 0.01}}))]
    )
    chunk, _, _ = _first_act(script, _double_tool())

    assert _tool_messages(script.requests[1]) == ["double: x must be a number"]
    assert len(chunk) > 0
    with pytest.raises(RuntimeError, match="double: x must be a number"):
        _first_act(_Script([bad, _turn(("double", {"x": 2})), bad, bad]), _double_tool())


def test_a_custom_success_cannot_mask_a_failing_motion() -> None:
    turn = _turn(("double", {"x": 2}), ("move_by", {"deltas": {"nope": 1.0}}))
    with pytest.raises(RuntimeError, match="kept failing"):
        _first_act(_Script([turn]), _double_tool())


def test_three_tool_errors_in_a_row_fail_the_turn() -> None:
    with pytest.raises(RuntimeError, match="double: x must be a number"):
        _first_act(_Script([_turn(("double", {"x": "two"}))]), _double_tool())


def test_a_result_that_is_not_text_raises() -> None:
    def silent(arguments: dict[str, Any], observation: Observation) -> str:
        return cast(str, None)

    with pytest.raises(TypeError, match="returned NoneType, not str"):
        _first_act(_Script([_turn(("double", {"x": 1}))]), _double_tool(silent))


def test_a_crashing_tool_propagates() -> None:
    def crash(arguments: dict[str, Any], observation: Observation) -> str:
        raise ValueError("calibration missing")

    with pytest.raises(ValueError, match="calibration missing"):
        _first_act(_Script([_turn(("double", {"x": 1}))]), _double_tool(crash))


def test_an_unknown_tool_lists_the_custom_names() -> None:
    script = _Script([_turn(("triple", {"x": 1})), _turn(("move_by", {"deltas": {"dx": 0.01}}))])
    _first_act(script, _double_tool())

    (message,) = _tool_messages(script.requests[1])
    assert message == "unknown tool 'triple'; available: move_by, done, give_up, double"


@pytest.mark.parametrize(
    ("tools", "match"),
    [
        ("double", "(?s)tools must be a sequence of CustomTool.*-P CLI flags"),
        ([object()], "tools must contain CustomTool instances"),
        ([CustomTool("two words", "", _PARAMETERS, _Double())], "must match"),
        ([CustomTool("take_pic", "", _PARAMETERS, _Double())], "reserved for a built-in"),
        ([_double_tool(), _double_tool()], "given twice"),
        ([CustomTool("double", "", {"type": "array"}, _Double())], "JSON Schema object"),
        ([CustomTool("double", cast(Any, None), _PARAMETERS, _Double())], "description must"),
        ([CustomTool("double", "", _PARAMETERS, cast(Any, "run"))], "run must be callable"),
    ],
    ids=[
        "string",
        "not-a-tool",
        "bad-name",
        "builtin",
        "duplicate",
        "schema",
        "description",
        "run",
    ],
)
def test_invalid_tools_are_refused_at_construction(tools: Any, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        _policy(_Script([_turn()]), tools=tools)


def test_config_records_each_tool_name_and_run_identity() -> None:
    def run(arguments: dict[str, Any], observation: Observation) -> str:
        return ""

    unset = _policy(_Script([_turn()]))
    named = _policy(_Script([_turn()]), tools=[_double_tool(run)])
    instance = _policy(_Script([_turn()]), tools=[_double_tool()])

    assert isinstance(unset.config, AgentPolicyConfig)
    assert isinstance(named.config, AgentPolicyConfig)
    assert isinstance(instance.config, AgentPolicyConfig)
    assert unset.config.tools == ()
    assert named.config.tools == (f"double={run.__module__}.{run.__qualname__}",)
    assert instance.config.tools == (f"double={_Double.__module__}.{_Double.__qualname__}",)


@pytest.mark.parametrize(
    ("images", "template"),
    [("always", _SYSTEM_TEMPLATE), ("on_demand", _ON_DEMAND_SYSTEM_TEMPLATE)],
)
def test_prompt_clause_is_present_only_with_tools(images: str, template: str) -> None:
    expected = template.format(name="(unbound)", budget=100)
    without = _policy(_Script([_turn()]), images=images)
    without.reset(Scene(id="s0", instruction="reach"))
    with_tools = _policy(_Script([_turn()]), images=images, tools=[_double_tool()])
    with_tools.reset(Scene(id="s0", instruction="reach"))

    without_transcript = without.transcript()
    with_transcript = with_tools.transcript()
    assert without_transcript is not None and with_transcript is not None
    assert without_transcript[0]["content"] == expected
    assert with_transcript[0]["content"] == (
        expected + " The tools double answer in text without moving the robot; an error "
        "from one fails the turn."
    )


def test_every_translating_wire_carries_the_custom_schema() -> None:
    schema = {
        "type": "function",
        "function": {
            "name": "double",
            "description": "Double a number.",
            "parameters": _PARAMETERS,
        },
    }
    assert anthropic_tools([schema])[0]["input_schema"] == _PARAMETERS
    assert responses_tools([schema])[0]["parameters"] == _PARAMETERS
    assert interactions_tool(schema)["parameters"] == _PARAMETERS
