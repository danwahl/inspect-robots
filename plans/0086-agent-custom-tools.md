# Plan 0086: agent plugin custom tools (text answers beside the motion tool)

Status: proposed. Revised after review round 1 (12 findings, addressed below).

## 1. Problem

The agent's tool surface is generated from the embodiment's spaces: one motion
tool (`move_to`, `move_joints` or `move_by`), `done`, `give_up`, and `take_pic`
under `images="on_demand"`. Everything else the model needs has to arrive as
prose in `EmbodimentInfo.docs`.

A calibrated overhead camera offers a computation, which `docs` can only
describe. The model sees a block in the image and needs its position in the
base frame. `docs` can state the intrinsics and the camera-to-base transform,
but the model then has to carry out a 4x4 transform and a ray-plane
intersection unaided. On an SO-101 block-stack embodiment, GPT-6 Luna at low
effort stacked 3 of 5 layouts when `docs` held a worked pixel-to-table
formula, 0 of 30 with the pose and focal length in words, and 0 of 10 with K
and the transform as matrices. Its notes show it estimating positions by eye
and missing the grasp. The calibration is the same in all three; only how it
reaches the model changes. With the matrices in `docs` and the computation as
a tool, it stacked 1 of 8 (2 trials errored) on the same 10 layouts.

- A formula in `docs` is a computation written as prose, with no precedent
  and no record of when it was used.
- Raising reasoning effort turns the eval into an arithmetic test.
- `capx` exposes perception as Python helpers, but it is a different policy
  with external GPU servers and a joint-space-only profile.

## 2. Goals

- A caller can hand `LLMAgentPolicy` extra tools: a name, a description, a
  JSON Schema for the arguments, and a callable that answers them.
- A custom tool does not move the robot. It answers in text, does not end the
  turn, and may precede the motion call in the same turn.
- The callable sees the observation the model is looking at, so an answer can
  be about the current frame and state.
- A tool rejects its arguments by raising `ToolError`; the model reads the
  message and retries, under the existing consecutive-failure budget.
- Zero new dependencies, and byte-identical prompts and schemas when no tools
  are given.

## 3. Non-goals

- Motion. A tool that moved the robot would be a second execution path beside
  the chunk the rollout approves. Custom tools return text only and never an
  `ActionChunk`.
- Images in results. `take_pic` already owns image delivery and its eviction
  rules. A follow-up can extend `CustomTool` once a consumer needs it.
- Argument validation against `parameters`. The schema is shown to the model
  and not enforced; `run` checks its own arguments, as the built-in tools do.
- Schema inference from type hints and docstrings, as Inspect AI's `@tool`
  does. It needs a docstring parser, which the httpx-only plugin does not
  carry. The schema is written out, in the same OpenAI function format the
  built-in tools use.
- CLI wiring. `-P` flags carry scalars, so tools are programmatic only, as
  `pre_check` is. An entry-point group (`inspect_robots_agent.tools`) that the
  CLI could name tools from is a follow-up.
- Embodiment-declared tools. Putting tools on `EmbodimentInfo` would let
  `capx` and other policies share them, but it changes the core contract;
  this plan keeps the change inside the plugin.

## 4. Design

### 4.1 The type

```python
@dataclass(frozen=True)
class CustomTool:
    name: str
    description: str
    parameters: dict[str, Any]       # JSON Schema object for the arguments
    run: Callable[[dict[str, Any], Observation], str]

class ToolError(Exception):
    """Raised by a custom tool to send its message back as a correctable error."""
```

- `run` receives the arguments as the model sent them and the observation of
  the current turn, and returns the text the model reads.
- Raising `ToolError` returns its message, prefixed with the tool's name, as a
  tool error. Any other exception propagates, as from `pre_check`: a crashing
  tool must not look like an answer. A result that is not a string raises
  `TypeError` for the same reason.
- Both are exported from `inspect_robots_agent`. The name `ToolError` follows
  Inspect AI, where it carries the same meaning.

### 4.2 Plumbing

- `LLMAgentPolicy.__init__` gains `tools: Sequence[CustomTool] = ()`. It
  raises `ConfigError` for a non-sequence, a non-`CustomTool` element, a name
  outside `^[a-zA-Z0-9_-]{1,64}$` (the strictest provider rule), a duplicate,
  a name used by a built-in tool (`move_to`, `move_joints`, `move_by`, `done`,
  `give_up`, `take_pic`), a `parameters` value that is not a dict with
  `"type": "object"`, a non-string description, and a `run` that is not
  callable. The built-in names are reserved even when that tool is not bound,
  so one tool list works across embodiments.
- `AgentPolicyConfig` gains `tools: tuple[str, ...]`, each entry
  `name=module.qualname` of `run`, derived as `pre_check`'s identity is (the
  two share `_callable_identity`).
- `build_toolset(...)` and `Toolset` take the tools. `schemas()` appends one
  function definition per tool after the built-ins. `execute()` checks that
  the arguments are a JSON object, as for every call, and dispatches a custom
  name to `_custom`, which calls `run` and returns `ToolResult(note=text)`, or
  `ToolResult(error=...)` on `ToolError`. The "unknown tool" error lists the
  custom names among the available tools.
- In the turn loop, a successful result with no chunk leaves the turn open: a
  later motion call in the same message runs, and a turn of custom calls alone
  returns to the model. A custom call after the turn has closed, behind a
  motion or `take_pic`, is answered "ignored: call {name} before the motion or
  take_pic".
- A successful custom call does not reset the consecutive-failure count. If
  it did, a model repeating `[custom, failing move]` would never reach the
  limit and would spend the whole call budget. A turn of custom calls alone
  is bounded by `max_llm_calls`, which ends in a forced `give_up`.

### 4.3 Model-facing text

When tools are given, both system prompt templates gain one clause, appended
after the pre-check clause so the template text before it stays
byte-identical:

> The tools {names} answer in text without moving the robot; call them alone
> or before the motion call in the same turn.

No change when none are given. Each tool's own description carries its
semantics, as with the built-ins.

### 4.4 Wires

Every wire already translates the OpenAI function format (`_translate_tools`
in `_anthropic.py` and `_responses.py`, `_translate_tool` in
`_interactions.py`, the declarations in `_gemini_live.py`). Custom schemas
take the same path. Gemini rejects some JSON Schema keywords; the README says
to keep to the subset the built-ins use (`type`, `properties`, `items`,
`required`, `description`, `enum`).

## 5. Testing

Scripted transports, in `tests/test_custom_tools.py`:

- Schemas: custom definitions follow the built-ins in order; none when unset
  (byte-identical to today).
- Dispatch: a custom call returns its text as the tool message; `run` gets
  the decoded arguments and the same observation `act()` received.
- Chaining: `[custom, move]` in one message runs both and returns the chunk;
  `[move, custom]` and, under `images="on_demand"`, `[take_pic, custom]`
  answer the custom call as ignored; `[custom, take_pic]` runs both; a
  custom-only turn goes back to the model and its answer is in the next
  request.
- Errors: `ToolError` comes back as a tool error and counts toward the
  failure budget; a custom success does not reset it; three in a row raise
  as other tool errors do; `[custom, failing move]` repeated raises; a
  generic exception and a non-string result escape `act()`.
- Validation: each `ConfigError` in 4.2.
- Config: `AgentPolicyConfig.tools` records names and identities.
- Prompt: the clause is present iff tools are given, on both templates.
- Wires: one custom schema through the Messages, Responses and Interactions
  translations. Gemini Live passes `parameters` through unchanged in the
  same way and has no separate translation function to test.

## 6. Docs and release

- Plugin README: a "Custom tools" section with the contract and a worked
  `locate_points` example for a calibrated overhead camera, which intersects
  each pixel's ray with a plane of given height.
- `tests/test_package.py` lists the two new exports. The version bump is left
  to the release, per AGENTS.md.
- A changelog fragment.

## 7. File tree

```
inspect-robots/
├── plans/0086-agent-custom-tools.md           (this document)
├── changelog.d/+agent-custom-tools.added.md
└── plugins/inspect-robots-agent/
    ├── README.md                              (custom tools section)
    ├── src/inspect_robots_agent/
    │   ├── __init__.py                        (CustomTool, ToolError exports)
    │   ├── _tools.py                          (CustomTool, ToolError, BUILTIN_TOOL_NAMES,
    │   │                                       schemas, dispatch, custom_names)
    │   └── policy.py                          (tools kwarg and validation, config field,
    │                                           prompt clause, ignored-call message)
    └── tests/
        ├── test_custom_tools.py               (new tests per §5)
        └── test_package.py                    (exports)
```

## 8. Resolved questions

- Why a callable and not a schema-checked handler: the plugin carries no JSON
  Schema validator, and a tool knows its own argument rules better than a
  generic check would; `ToolError` gives it the same correctable path.
- Why a custom success does not reset the failure count: see 4.2.
- Why the clause is appended rather than placed beside the one-call sentence:
  the pre-check clause set the precedent, and appending keeps the template
  text a prefix of every prompt, which the prompt tests rely on.

## 9. Prior art

- Inspect AI: `@tool` functions with a typed signature and docstring, attached
  through the solver, reporting `ToolError` to the model.
- innate-os: `go_to_point_in_view` projects a pixel onto the floor.
- This plugin: `pre_check` (an injected callable, programmatic only,
  identity recorded) and `take_pic` (a call that does not move the robot and
  can be chained in a turn).
