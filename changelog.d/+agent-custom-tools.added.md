**Agent plugin:** `LLMAgentPolicy(tools=...)` takes `CustomTool`s, which answer the model in
text without moving the robot. A custom call may precede the motion call in the same turn,
and a tool raises `ToolError` to return a correctable error
([plan 0086](plans/0086-agent-custom-tools.md)).
