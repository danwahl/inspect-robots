# Plan 0087: agent plugin plan tool (dry runs of the motion tool)

Status: proposed.

## 1. Problem

The agent learns where the robot can go by moving it. A motion that the bounds,
the per-step cap or `pre_check` refuses comes back as a tool error, and three
errors in a row end the trial. The model has no way to ask whether a target is
reachable without spending one of those three.

On an SO-101 block-stack embodiment whose `pre_check` reports where a motion
leaves the arm's reach, GPT-6 Luna at low effort stacked 1 of 8 layouts, and 2
of 10 trials errored. Both died on their first three motion calls, each refused
because the target itself was out of reach. The notes call these moves tests
("I am testing a slightly more central point before committing"), but the
model ran them as real motions.

Retrying in the plugin is not an option: a motion the model did not ask for
would move the robot unasked.

## 2. Goals

- The model can check a motion without running it. The check runs every
  check a real call runs (bounds, per-step cap, `pre_check`) and reports the
  outcome in text.
- A checked motion sends nothing to the robot and does not end the turn.
- A check that would be refused does not count toward the
  consecutive-failure budget.
- Opt-in, with byte-identical prompts and schemas when off.

## 3. Non-goals

- Path planning. The check covers the motion the tool would emit, as emitted.
- Checks the plugin cannot run, such as the approver chain. A plan that passes
  can still be refused there, as a `pre_check` pass can.

## 4. Design

### 4.1 Opt-in

`LLMAgentPolicy(plan_tool=False)` is a scalar, so `-P plan_tool=true` works on
the CLI. A non-bool raises `ConfigError`. `AgentPolicyConfig.plan_tool`
records it. `build_toolset` and `Toolset` take it.

### 4.2 The tool

When enabled, a second tool follows the motion tool: `plan_to` beside
`move_to`, `plan_joints` beside `move_joints`, `plan_by` beside `move_by`. It
takes the motion tool's parameters, unchanged, and its description is:

> Check a {move} motion as if to run it and report the result without moving
> the robot. Takes the same arguments.

The three names join the reserved built-in names, so a custom tool cannot
take one.

A first version put a `plan_only` flag on the motion tool.
Over 30 trials across two effort levels, with the flag in the schema, a
prompt clause, and finally a sentence naming it on every refusal, Luna never
set it once. A tool of its own sits in the tool list under a name that says
what it does.

### 4.3 Execution

`execute` sends a plan call through `_move`, the motion tool's own path:

- Malformed calls (missing note, unknown dimension, non-number) stay errors:
  they are mistakes in the call, not facts about the motion.
- Otherwise the motion is built as usual and `_plan` turns the result into a
  note without a chunk:
  - refused: `this motion would be refused: <the refusal a real call
    returns, without the sentence below>`;
  - accepted: `{move} would run over N steps (Xs), ending at label=value,
    ...`. The ending names every dimension, unnamed ones filled from the
    current state. Displacement modes have no absolute target and omit it.
- A real motion refused by bounds, cap, `pre_check`, or a delta too small to
  step appends one sentence to its error when the option is on: "{plan}
  checks a motion without moving the robot." It states that the tool exists,
  without suggesting a target. Malformed calls fail earlier and do not get it.

### 4.4 Turn loop

A plan is a result with no chunk and no error, so it leaves the turn open, as
a custom tool's answer does (plan 0086): `[plan, move]` runs both. A plan
after the turn has closed is answered "ignored: call {plan} before the motion
or take_pic". A plan does not reset the failure count, for the reason 0086
gives for custom tools. A run of plans alone is bounded by `max_llm_calls`.

### 4.5 Prompt

When enabled, both templates gain one clause, appended after the pre-check
and custom-tool clauses:

> {plan} checks a {move} motion as if to run it and reports the result without
> moving the robot; call it alone or before the motion call in the same turn.

## 5. Testing

`tests/test_plan_tool.py`, with scripted transports:

- Schema: the plan tool follows the motion tool only when enabled, with the
  same parameters; the motion schema and the whole list are unchanged when off.
- A plan reports steps, duration and ending, calls `pre_check` with the same
  waypoints a real call would, and returns no chunk.
- A refused plan (pre-check and bounds) is a note, not an error.
- A refused real motion names the plan tool only when offered, joins cleanly
  after a trailing period, and a malformed call does not get the sentence.
- Malformed plans stay errors.
- The plan tool is unknown when not offered, and the unknown-tool error lists
  it when it is.
- The three plan names are reserved.
- A displacement plan (`plan_by`) reports steps and duration without an
  ending.
- Turn loop: `[plan, move]` runs both; four refused plans then a move do not
  raise; plans between failing moves do not reset the count; `[move, plan]`
  ignores the plan.
- Prompt clause present only when enabled, on both templates.
- Config records the option; a non-bool raises `ConfigError`.

## 6. Docs and release

- Plugin README: a "Planning a motion" section, and `plan_tool` in the
  configuration knobs.
- A changelog fragment.

## 7. File tree

```
inspect-robots/
├── plans/0087-agent-plan-tool.md              (this document)
├── changelog.d/+agent-plan-tool.added.md
└── plugins/inspect-robots-agent/
    ├── README.md                              (planning a motion)
    ├── src/inspect_robots_agent/
    │   ├── _tools.py                          (plan tool names, schema, dispatch,
    │   │                                       _plan, refusal sentence)
    │   └── policy.py                          (kwarg, config field, prompt clause,
    │                                           ignored-call message)
    └── tests/test_plan_tool.py
```

## 8. Prior art

- This plugin: custom tools (plan 0086), whose no-chunk, no-error result
  leaves the turn open.
