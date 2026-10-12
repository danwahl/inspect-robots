"""plan_only: dry runs of the motion tool that report without moving (plan 0087)."""

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
from inspect_robots.spaces import ActionSemantics, Box, ObservationSpace, StateField, StateSpec
from inspect_robots.types import Observation
from inspect_robots_agent import LLMAgentPolicy
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


def _move(**arguments: Any) -> ToolCall:
    return ToolCall(
        id="call_0", name="move_joints", arguments=json.dumps({"note": _NOTE, **arguments})
    )


def _obs(q: float = 0.0) -> Observation:
    return Observation(state={"q": np.array([q])}, instruction="reach")


def _move_properties(toolset: Any) -> dict[str, Any]:
    (move,) = [s for s in toolset.schemas() if s["function"]["name"] == "move_joints"]
    properties: dict[str, Any] = move["function"]["parameters"]["properties"]
    return properties


# --- toolset -------------------------------------------------------------------


def test_the_schema_offers_plan_only_only_when_enabled() -> None:
    assert "plan_only" not in _move_properties(_toolset())
    assert _move_properties(_toolset(plan_only=True))["plan_only"]["type"] == "boolean"
    assert _toolset().schemas() == _toolset(plan_only=False).schemas()


def test_a_plan_reports_the_motion_and_sends_nothing() -> None:
    seen: list[npt.NDArray[np.float64]] = []

    def record(waypoints: npt.NDArray[np.float64]) -> str | None:
        seen.append(waypoints.copy())
        return None

    toolset = _toolset(plan_only=True, pre_check=record)
    plan = toolset.execute(_move(targets={"joint": 0.4}, plan_only=True), _obs())
    run = toolset.execute(_move(targets={"joint": 0.4}), _obs())
    assert plan.chunk is None and plan.error is None
    assert run.chunk is not None
    steps = len(run.chunk)
    assert plan.note == (
        f"plan_only: move_joints would run over {steps} steps ({steps / 10:.1f}s), "
        "ending at joint=0.4"
    )
    assert len(seen) == 2 and np.array_equal(seen[0], seen[1])


def test_a_refused_plan_is_a_note_not_an_error() -> None:
    toolset = _toolset(plan_only=True, pre_check=_reject_past_half)
    result = toolset.execute(_move(targets={"joint": 0.9}, plan_only=True), _obs())
    assert result.error is None and result.chunk is None
    assert result.note == (
        "plan_only: this motion would be refused: pre-check rejected this motion: joint past 0.5"
    )
    bounds = toolset.execute(_move(targets={"joint": 2.0}, plan_only=True), _obs())
    assert bounds.error is None
    assert bounds.note.startswith("plan_only: this motion would be refused: target for joint")


def test_a_refused_motion_mentions_plan_only_when_offered() -> None:
    call = _move(targets={"joint": 0.9})
    offered = _toolset(plan_only=True, pre_check=_reject_past_half).execute(call, _obs())
    assert offered.error == (
        "pre-check rejected this motion: joint past 0.5. "
        "plan_only checks a motion without moving the robot."
    )
    plain = _toolset(pre_check=_reject_past_half).execute(call, _obs())
    assert plain.error == "pre-check rejected this motion: joint past 0.5"
    toolset = _toolset(plan_only=True, pre_check=lambda waypoints: "joint past 0.5.\n")
    assert toolset.execute(call, _obs()).error == (
        "pre-check rejected this motion: joint past 0.5. "
        "plan_only checks a motion without moving the robot."
    )
    malformed = _toolset(plan_only=True).execute(_move(targets={"elbow": 0.1}), _obs())
    assert malformed.error is not None and "plan_only" not in malformed.error


def test_malformed_calls_stay_errors_under_plan_only() -> None:
    toolset = _toolset(plan_only=True)
    assert toolset.execute(_move(targets={"elbow": 0.1}, plan_only=True), _obs()).error
    not_bool = toolset.execute(_move(targets={"joint": 0.1}, plan_only="yes"), _obs())
    assert not_bool.error == "plan_only must be true or false, got 'yes'"
    explicit_false = toolset.execute(_move(targets={"joint": 0.1}, plan_only=False), _obs())
    assert explicit_false.chunk is not None


def test_plan_only_is_refused_when_not_offered() -> None:
    result = _toolset().execute(_move(targets={"joint": 0.1}, plan_only=True), _obs())
    assert result.chunk is None
    assert result.error == "plan_only is not available on move_joints here"


def test_a_displacement_plan_reports_steps_without_an_ending() -> None:
    space = Box(
        shape=(1,),
        low=np.array([-0.1]),
        high=np.array([0.1]),
        semantics=ActionSemantics("eef_delta_pos", frame="world", dim_labels=("x",)),
    )
    toolset = build_toolset(space, ObservationSpace(), control_hz=10.0, plan_only=True)
    call = ToolCall(
        id="c", name="move_by", arguments=json.dumps({"note": _NOTE, "deltas": {"x": 0.03}})
    )
    run = toolset.execute(call, Observation())
    assert run.chunk is not None
    call = ToolCall(
        id="c",
        name="move_by",
        arguments=json.dumps({"note": _NOTE, "deltas": {"x": 0.03}, "plan_only": True}),
    )
    plan = toolset.execute(call, Observation())
    steps = len(run.chunk)
    assert plan.note == f"plan_only: move_by would run over {steps} steps ({steps / 10:.1f}s)"


def test_is_plan_reads_the_flag_on_the_motion_tool_only() -> None:
    toolset = _toolset(plan_only=True)
    assert toolset.is_plan(_move(targets={"joint": 0.1}, plan_only=True))
    assert not toolset.is_plan(_move(targets={"joint": 0.1}))
    assert not toolset.is_plan(ToolCall(id="c", name="done", arguments='{"plan_only": true}'))
    assert not toolset.is_plan(ToolCall(id="c", name="move_joints", arguments="not json"))
    assert not _toolset().is_plan(_move(targets={"joint": 0.1}, plan_only=True))


# --- turn loop -----------------------------------------------------------------


def _turn(*calls: dict[str, Any]) -> dict[str, Any]:
    tool_calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": "move_joints", "arguments": json.dumps({"note": _NOTE, **c})},
        }
        for i, c in enumerate(calls)
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


def test_a_plan_before_the_motion_runs_both() -> None:
    script = _Script(
        [_turn({"targets": {"joint": 0.3}, "plan_only": True}, {"targets": {"joint": 0.3}})]
    )
    chunk, policy = _act(script, plan_only=True)
    assert chunk is not None and float(chunk.actions[-1].data[0]) == pytest.approx(0.3)
    plan, run = _tool_messages(policy)
    assert plan.startswith("plan_only: move_joints would run over")
    assert run.startswith("executing move_joints")


def test_refused_plans_do_not_spend_the_failure_budget() -> None:
    probe = {"targets": {"joint": 0.9}, "plan_only": True}
    script = _Script(
        [_turn(probe), _turn(probe), _turn(probe), _turn(probe), _turn({"targets": {"joint": 0.4}})]
    )
    chunk, policy = _act(script, plan_only=True, pre_check=_reject_past_half)
    assert chunk is not None
    messages = _tool_messages(policy)
    assert len(messages) == 5
    assert all("would be refused" in m for m in messages[:4])


def test_a_plan_does_not_reset_the_failure_count() -> None:
    bad = {"targets": {"elbow": 0.1}}
    plan = {"targets": {"joint": 0.3}, "plan_only": True}
    script = _Script([_turn(bad), _turn(plan), _turn(bad), _turn(plan), _turn(bad)])
    with pytest.raises(RuntimeError, match="kept failing"):
        _act(script, plan_only=True)


def test_a_plan_after_the_motion_is_ignored() -> None:
    script = _Script(
        [_turn({"targets": {"joint": 0.3}}, {"targets": {"joint": 0.6}, "plan_only": True})]
    )
    chunk, policy = _act(script, plan_only=True)
    assert chunk is not None
    assert _tool_messages(policy)[1] == "ignored: call plan_only before the motion or take_pic"


@pytest.mark.parametrize(
    ("images", "template"),
    [("always", _SYSTEM_TEMPLATE), ("on_demand", _ON_DEMAND_SYSTEM_TEMPLATE)],
)
def test_the_prompt_clause_is_present_only_when_enabled(images: str, template: str) -> None:
    info = _INFO
    if images == "on_demand":
        from inspect_robots.spaces import CameraSpec

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
        policy = _policy(_Script([_turn()]), images=images, plan_only=enabled)
        policy.bind(info)
        policy.reset(Scene(id="s0", instruction="reach"))
        transcript = policy.transcript()
        assert transcript is not None
        prompts.append(transcript[0]["content"])
    prefix = template.format(name="plan-test", budget=100)
    assert prompts[0].startswith(prefix) and prompts[1].startswith(prefix)
    clause = (
        " With plan_only set, move_joints checks the motion as if to run it and reports the "
        "result without moving the robot; call it alone or before the motion call in the same turn."
    )
    assert clause not in prompts[0]
    assert prompts[1] == prompts[0].replace(prefix, prefix + clause, 1)


def test_the_config_records_plan_only_and_rejects_non_bools() -> None:
    for enabled in (False, True):
        config = cast(AgentPolicyConfig, _policy(_Script([_turn()]), plan_only=enabled).config)
        assert config.plan_only is enabled
    assert cast(AgentPolicyConfig, _policy(_Script([_turn()])).config).plan_only is False
    with pytest.raises(ConfigError, match="plan_only must be true or false"):
        _policy(_Script([_turn()]), plan_only="true")
