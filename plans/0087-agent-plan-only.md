# Plan 0087: agent plugin plan_only (dry runs of the motion tool)

Status: proposed.

## 1. Problem

The agent learns where the robot can go by moving it. A motion the bounds,
the per-step cap or `pre_check` refuses comes back as a tool error, and three
errors in a row end the trial. The model has no way to ask whether a target is
reachable without spending one of those three.

On an SO-101 block-stack embodiment whose `pre_check` reports the first
unreachable waypoint, GPT-6 Luna at low effort stacked 1 of 8 layouts, and 2
of 10 trials errored. Both of those died on their first three motion calls, each
refused because the target itself was out of reach. The model learns the reach
only by failing against it, and three failures end the trial.

Retrying in the plugin is not an option: a motion the model did not ask for
would move the robot unasked.

## 2. Goals

- A motion call can ask only to be checked. It runs every check a real call
  runs (bounds, per-step cap, `pre_check`) and reports the outcome in text.
- A checked motion sends nothing to the robot and does not end the turn.
- A check that would be refused does not count toward the
  consecutive-failure budget.
- Opt-in, with byte-identical prompts and schemas when off.

## 3. Non-goals

- Path planning. The check covers the motion the tool would emit, as emitted.
- A separate `plan_to` tool. A flag on the existing tool keeps one schema per
  motion.
- Checks the plugin cannot run, such as the approver chain. A plan that passes
  can still be refused there, as a `pre_check` pass can.

## 4. Design

### 4.1 Opt-in

`LLMAgentPolicy(plan_only=False)` is a scalar, so `-P plan_only=true` works on
the CLI. A non-bool raises `ConfigError`. `AgentPolicyConfig.plan_only` records
it. `build_toolset` and `Toolset` take it.

### 4.2 Schema

When enabled, the motion tool's parameters gain an optional boolean:

> plan_only: If true, check this motion as if to run it and report the result
> without moving the robot.

### 4.3 Execution

In `Toolset._move`, after the call is parsed:

- A `plan_only` that is not a bool is an error.
- `plan_only: true` on a toolset that does not offer it is an error. Running
  it would move the robot when the model asked only for a check.
- Malformed calls (missing note, unknown dimension, non-number) stay errors
  under `plan_only`: they are mistakes in the call, not facts about the motion.
- Otherwise the motion is built as usual and `_plan` turns the result into a
  note without a chunk:
  - refused: `plan_only: this motion would be refused: <the error a real call
    returns>`;
  - accepted: `plan_only: {tool} would run over N steps (Xs), ending at
    label=value, ...`. The ending names every dimension, unnamed ones filled
    from the current state. Displacement modes have no absolute target and
    omit it.

### 4.4 Turn loop

A plan is a result with no chunk and no error, so it leaves the turn open, as
a custom tool's answer does (plan 0086): `[plan, move]` runs both. A plan
after the turn has closed is answered "ignored: call plan_only before the
motion or take_pic". `Toolset.is_plan(call)` recognises one. A plan does not
reset the failure count, for the reason 0086 gives for custom tools. A run of
plans alone is bounded by `max_llm_calls`.

### 4.5 Prompt

When enabled, both templates gain one clause, appended after the pre-check
and custom-tool clauses:

> With plan_only set, {move} checks the motion as if to run it and reports the
> result without moving the robot; call it alone or before the motion call in
> the same turn.

## 5. Testing

`tests/test_plan_only.py`, with scripted transports:

- Schema: the property is present only when enabled; schemas are identical
  when off.
- A plan reports steps, duration and ending, calls `pre_check` with the same
  waypoints a real call would, and returns no chunk.
- A refused plan (pre-check and bounds) is a note, not an error.
- Malformed calls and a non-bool flag stay errors; `plan_only: false` runs.
- `plan_only: true` is an error when not offered.
- A displacement plan reports steps and duration without an ending.
- `is_plan` reads the flag on the motion tool only.
- Turn loop: `[plan, move]` runs both; four refused plans then a move do not
  raise; plans between failing moves do not reset the count; `[move, plan]`
  ignores the plan.
- Prompt clause present only when enabled, on both templates.
- Config records the flag; a non-bool raises `ConfigError`.

## 6. Docs and release

- Plugin README: a "Planning a motion" section, and `plan_only` in the
  configuration knobs.
- A changelog fragment.

## 7. File tree

```
inspect-robots/
├── plans/0087-agent-plan-only.md              (this document)
├── changelog.d/+agent-plan-only.added.md
└── plugins/inspect-robots-agent/
    ├── README.md                              (planning a motion)
    ├── src/inspect_robots_agent/
    │   ├── _tools.py                          (flag, schema, _plan, is_plan, move_tool)
    │   └── policy.py                          (kwarg, config field, prompt clause,
    │                                           ignored-call message)
    └── tests/test_plan_only.py
```

## 8. Prior art

- This plugin: custom tools (plan 0086), whose no-chunk, no-error result
  leaves the turn open.
