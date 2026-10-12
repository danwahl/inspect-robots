**Agent plugin:** `LLMAgentPolicy(plan_tool=True)` adds `plan_to` (or `plan_joints`, `plan_by`)
beside the motion tool. It runs the motion's checks and reports the outcome without moving the
robot, and a refused plan does not count toward the failure budget
([plan 0087](plans/0087-agent-plan-tool.md)).
