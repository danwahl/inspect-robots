**Agent plugin:** a motion refused by the pre-check, the action bounds or the playout cap no
longer counts toward the three-turn failure limit, since it never reaches the robot. The system
prompt now states the limit and what fails a turn: a malformed tool call or a reply with no tool
call, and under `images="on_demand"` a second `take_pic` refusal for the same observation.
