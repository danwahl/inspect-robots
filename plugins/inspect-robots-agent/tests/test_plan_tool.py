"""The plan tool: dry runs of the motion tool that report without moving (plan 0087)."""

from __future__ import annotations

import json
from typing import Any, cast

import httpx
import numpy as np
import numpy.typing as npt
import pytest

from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.errors import ConfigError
from inspect_robots.scene import Scene
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.types import Observation
from inspect_robots_agent import CustomTool, LLMAgentPolicy
from inspect_robots_agent._llm import ToolCall
from inspect_robots_agent._tools import build_toolset
from inspect_robots_agent.policy import (
    _ON_DEMAND_SYSTEM_TEMPLATE,
    _SYSTEM_TEMPLATE,
    AgentPolicyConfig,
)

_NOTE = "checking the reach"

_SPACE = Box(
    shape=(1,),
    low=np.array([-1.0]),
    high=np.array([1.0]),
    semantics=ActionSemantics("joint_pos", dim_labels=("joint",)),
)
_OBS_SPACE = ObservationSpace(state=StateSpec(fields=(StateField(key="q", shape=(1,)),)))
_INFO = EmbodimentInfo(
    name="plan-test",
    action_space=_SPACE,
    observation_space=_OBS_SPACE,
    control_hz=10.0,
    is_simulated=True,
)


def _reject_past_half(waypoints: npt.NDArray[np.float64]) -> str | None:
    return "joint past 0.5" if bool(np.any(waypoints[:, 0] > 0.5)) else None


def _toolset(**kwargs: Any) -> Any:
    return build_toolset(_SPACE, _OBS_SPACE, control_hz=10.0, **kwargs)


def _call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id="call_0", name=name, arguments=json.dumps({"note": _NOTE, **arguments}))


def _obs(q: float = 0.0) -> Observation:
    return Observation(state={"q": np.array([q])}, instruction="reach")


def _schema(toolset: Any, name: str) -> dict[str, Any]:
    (schema,) = [s for s in toolset.schemas() if s["function"]["name"] == name]
    return cast(dict[str, Any], schema["function"])


# --- toolset -------------------------------------------------------------------


def test_the_plan_tool_follows_the_motion_tool_only_when_enabled() -> None:
    assert _toolset().schemas() == _toolset(plan_tool=False).schemas()
    assert _toolset().plan_tool is None
    toolset = _toolset(plan_tool=True)
    names = [s["function"]["name"] for s in toolset.schemas()]
    assert names[:3] == ["move_joints", "plan_joints", "done"]
    plan, move = _schema(toolset, "plan_joints"), _schema(toolset, "move_joints")
    assert plan["parameters"] == move["parameters"]
    assert move == _schema(_toolset(), "move_joints")


def test_a_plan_reports_the_motion_and_sends_nothing() -> None:
    seen: list[npt.NDArray[np.float64]] = []

    def record(waypoints: npt.NDArray[np.float64]) -> str | None:
        seen.append(waypoints.copy())
        return None

    toolset = _toolset(plan_tool=True, pre_check=record)
    plan = toolset.execute(_call("plan_joints", targets={"joint": 0.4}), _obs())
    run = toolset.execute(_call("move_joints", targets={"joint": 0.4}), _obs())
    assert plan.chunk is None and plan.error is None
    assert run.chunk is not None
    steps = len(run.chunk)
    assert plan.note == (
        f"move_joints would run over {steps} steps ({steps / 10:.1f}s), ending at joint=0.4"
    )
    assert len(seen) == 2 and np.array_equal(seen[0], seen[1])


def test_a_refused_plan_is_a_note_not_an_error() -> None:
    toolset = _toolset(plan_tool=True, pre_check=_reject_past_half)
    result = toolset.execute(_call("plan_joints", targets={"joint": 0.9}), _obs())
    assert result.error is None and result.chunk is None
    assert result.note == (
        "this motion would be refused: pre-check rejected this motion: joint past 0.5"
    )
    bounds = toolset.execute(_call("plan_joints", targets={"joint": 2.0}), _obs())
    assert bounds.error is None
    assert bounds.note.startswith("this motion would be refused: target for joint")


def test_a_refused_motion_names_the_plan_tool_when_offered() -> None:
    call = _call("move_joints", targets={"joint": 0.9})
    offered = _toolset(plan_tool=True, pre_check=_reject_past_half).execute(call, _obs())
    assert offered.error == (
        "pre-check rejected this motion: joint past 0.5. "
        "plan_joints checks a motion without moving the robot."
    )
    toolset = _toolset(plan_tool=True, pre_check=lambda waypoints: "joint past 0.5.\n")
    assert toolset.execute(call, _obs()).error == offered.error
    plain = _toolset(pre_check=_reject_past_half).execute(call, _obs())
    assert plain.error == "pre-check rejected this motion: joint past 0.5"
    malformed = _toolset(plan_tool=True).execute(
        _call("move_joints", targets={"elbow": 0.1}), _obs()
    )
    assert malformed.error is not None and "plan_joints" not in malformed.error


def test_malformed_plans_stay_errors() -> None:
    toolset = _toolset(plan_tool=True)
    assert toolset.execute(_call("plan_joints", targets={"elbow": 0.1}), _obs()).error
    no_note = ToolCall(id="c", name="plan_joints", arguments='{"targets": {"joint": 0.1}}')
    assert toolset.execute(no_note, _obs()).error


def test_the_plan_tool_is_unknown_when_not_offered() -> None:
    result = _toolset().execute(_call("plan_joints", targets={"joint": 0.1}), _obs())
    assert result.chunk is None
    assert result.error == "unknown tool 'plan_joints'; available: move_joints, done, give_up"
    offered = _toolset(plan_tool=True).execute(_call("plan_to", targets={}), _obs())
    assert offered.error is not None
    assert "available: move_joints, plan_joints, done, give_up" in offered.error


def test_plan_tool_names_are_reserved() -> None:
    for name in ("plan_to", "plan_joints", "plan_by"):
        tool = CustomTool(
            name=name,
            description="d",
            parameters={"type": "object", "properties": {}},
            run=lambda arguments, observation: "",
        )
        with pytest.raises(ConfigError):
            _policy(_Script([_turn()]), tools=[tool])


def test_a_displacement_plan_reports_steps_without_an_ending() -> None:
    space = Box(
        shape=(1,),
        low=np.array([-0.1]),
        high=np.array([0.1]),
        semantics=ActionSemantics("eef_delta_pos", frame="world", dim_labels=("x",)),
    )
    toolset = build_toolset(space, ObservationSpace(), control_hz=10.0, plan_tool=True)
    assert toolset.plan_tool == "plan_by"
    run = toolset.execute(_call("move_by", deltas={"x": 0.03}), Observation())
    assert run.chunk is not None
    plan = toolset.execute(_call("plan_by", deltas={"x": 0.03}), Observation())
    steps = len(run.chunk)
    assert plan.note == f"move_by would run over {steps} steps ({steps / 10:.1f}s)"


# --- turn loop -----------------------------------------------------------------


def _turn(*calls: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    tool_calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps({"note": _NOTE, **arguments})},
        }
        for i, (name, arguments) in enumerate(calls)
    ]
    return {
        "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": tool_calls}}]
    }


class _Script:
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


def _act(script: _Script, **kwargs: Any) -> tuple[Any, LLMAgentPolicy]:
    policy = _policy(script, **kwargs)
    policy.bind(_INFO)
    policy.reset(Scene(id="s0", instruction="reach"))
    return policy.act(_obs()), policy


def _tool_messages(policy: LLMAgentPolicy) -> list[str]:
    transcript = policy.transcript()
    assert transcript is not None
    return [m["content"] for m in transcript if m["role"] == "tool"]


_PLAN = ("plan_joints", {"targets": {"joint": 0.3}})
_MOVE = ("move_joints", {"targets": {"joint": 0.3}})


def test_a_plan_before_the_motion_runs_both() -> None:
    chunk, policy = _act(_Script([_turn(_PLAN, _MOVE)]), plan_tool=True)
    assert chunk is not None and float(chunk.actions[-1].data[0]) == pytest.approx(0.3)
    plan, run = _tool_messages(policy)
    assert plan.startswith("move_joints would run over")
    assert run.startswith("executing move_joints")


def test_refused_plans_do_not_spend_the_failure_budget() -> None:
    probe = ("plan_joints", {"targets": {"joint": 0.9}})
    script = _Script([_turn(probe)] * 4 + [_turn(_MOVE)])
    chunk, policy = _act(script, plan_tool=True, pre_check=_reject_past_half)
    assert chunk is not None
    messages = _tool_messages(policy)
    assert len(messages) == 5
    assert all("would be refused" in m for m in messages[:4])


def test_a_plan_does_not_reset_the_failure_count() -> None:
    bad = ("move_joints", {"targets": {"elbow": 0.1}})
    script = _Script([_turn(bad), _turn(_PLAN), _turn(bad), _turn(_PLAN), _turn(bad)])
    with pytest.raises(RuntimeError, match="kept failing"):
        _act(script, plan_tool=True)


def test_a_plan_after_the_motion_is_ignored() -> None:
    chunk, policy = _act(_Script([_turn(_MOVE, _PLAN)]), plan_tool=True)
    assert chunk is not None
    assert _tool_messages(policy)[1] == "ignored: call plan_joints before the motion or take_pic"


@pytest.mark.parametrize(
    ("images", "template"),
    [("always", _SYSTEM_TEMPLATE), ("on_demand", _ON_DEMAND_SYSTEM_TEMPLATE)],
)
def test_the_prompt_clause_is_present_only_when_enabled(images: str, template: str) -> None:
    info = _INFO
    if images == "on_demand":
        info = EmbodimentInfo(
            name="plan-test",
            action_space=_SPACE,
            observation_space=ObservationSpace(
                cameras=(CameraSpec(name="top", height=2, width=2),),
                state=_OBS_SPACE.state,
            ),
            control_hz=10.0,
        )
    prompts = []
    for enabled in (False, True):
        policy = _policy(_Script([_turn()]), images=images, plan_tool=enabled)
        policy.bind(info)
        policy.reset(Scene(id="s0", instruction="reach"))
        transcript = policy.transcript()
        assert transcript is not None
        prompts.append(transcript[0]["content"])
    prefix = template.format(name="plan-test", budget=100)
    assert prompts[0].startswith(prefix) and prompts[1].startswith(prefix)
    clause = (
        " plan_joints checks a move_joints motion as if to run it and reports the result "
        "without moving the robot; call it alone or before the motion call in the same turn."
    )
    assert clause not in prompts[0]
    assert prompts[1] == prompts[0].replace(prefix, prefix + clause, 1)


def test_the_config_records_plan_tool_and_rejects_non_bools() -> None:
    for enabled in (False, True):
        config = cast(AgentPolicyConfig, _policy(_Script([_turn()]), plan_tool=enabled).config)
        assert config.plan_tool is enabled
    assert cast(AgentPolicyConfig, _policy(_Script([_turn()])).config).plan_tool is False
    with pytest.raises(ConfigError, match="plan_tool must be true or false"):
        _policy(_Script([_turn()]), plan_tool="true")
