**Agent plugin:** `LLMAgentPolicy(plan_only=True)` gives the motion tool a `plan_only` flag,
which runs the motion's checks and reports the outcome without moving the robot. A refused
plan does not count toward the failure budget ([plan 0087](plans/0087-agent-plan-only.md)).
