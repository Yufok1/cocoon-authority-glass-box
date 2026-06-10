# Cocoon Skills Cookbook

Use these as operator skills. Each skill has a trigger, route, expected output, and persistence move.

## Skill: Diagnose

Trigger:
You need to know what is real on this phone.

Command:
```sh
./cocoon doctor
./cocoon agent brief
./cocoon capabilities run health
```

Look for:
Core runtime ready, active cocoon path, and any missing requisites for a lane.

Persist:
No persistence needed.

## Skill: Talk And Learn

Trigger:
You want conversational improvement.

Command:
```sh
./cocoon capabilities run chat --payload '{"prompt":"state what you know and what you need next","learn":true,"steps":1}'
```

Look for:
`response`, `teacher_note`, `state`, `receipt`.

Persist:
Run `save` after useful turns.

## Skill: Teach A Concept

Trigger:
You have a lesson, definition, rule, or association.

Command:
```sh
./cocoon capabilities run teach --payload '{"text":"verification means testing before trusting","reward":0.8,"steps":2}'
```

Look for:
New vocabulary, relations, logs.

Persist:
Run `snapshot` then `save`.

## Skill: Build Associations

Trigger:
Two ideas should be linked.

Command:
```sh
./cocoon capabilities run relate --payload '{"source":"signal","target":"meaning","steps":3}'
```

Look for:
Relation count and teacher note.

Persist:
Run `save`.

## Skill: Govern A Claim

Trigger:
You need routing, arbitration, and verification.

Command:
```sh
./cocoon capabilities run govern --payload '{"text":"verify cause effect before response","steps":2}'
```

Look for:
Chosen faculty, ranked faculties, verification notes.

Persist:
Save if the routing was useful.

## Skill: Drill Language

Trigger:
You want simple repeatable language formation.

Command:
```sh
./cocoon capabilities run follow --payload '{"mode":"cue","seed":"signal meaning reason verify","rounds":6,"steps":2}'
```

Look for:
Rounds, pairs, losses, receipt.

Persist:
Run `logs` then `save`.

## Skill: Forge Syntax

Trigger:
You want safe fixed-point / structure practice without executing generated commands.

Command:
```sh
./cocoon capabilities run quine --payload '{"limit":8,"steps":2}'
```

Look for:
Drill outputs and receipts.

Persist:
Save after a good batch.

## Skill: Run A Game Grounding Pass

Trigger:
You want action/reward grounding.

Command:
```sh
./cocoon capabilities run sphere --payload '{"frames":120,"balls":2,"misses":4,"train":true}'
```

Look for:
Catches, misses, best streak, losses.

Persist:
Run `save`.

## Skill: Use The Real Pygame Surface

Trigger:
You want the phone-local graphical operating surface.

Command:
```sh
export DISPLAY=:0
./cocoon glass
```

Controls:
Left/Right selects capability. Enter runs selected. `S` sphere. `A` autopilot. `D` harden. `R` refresh.

Persist:
Use `save` or `export` from the capability fabric.

## Skill: Deepen Broadly

Trigger:
You want a bounded multi-facility improvement cycle.

Command:
```sh
./cocoon capabilities run autopilot --payload '{"mission":"improve chat, reasoning, verification, games, and semantic memory","depth":1,"include_games":true,"save_after":false}'
```

Look for:
Stages, teacher card, receipt, state.

Persist:
Run `save`; export only after the run looks good.

## Skill: Persist

Trigger:
You want the live runtime state durable.

Command:
```sh
./cocoon capabilities run save --payload '{"output_dir":"facility_save"}'
```

Look for:
Written output paths.

Persist:
This is persistence.

## Skill: Export Checkpoint

Trigger:
You want a standalone checkpoint after useful training.

Command:
```sh
./cocoon capabilities run export --payload '{"name":"facility_checkpoint.py"}'
```

Look for:
Checkpoint path.

Persist:
Keep the checkpoint and reload it when you need to validate it.

## Skill: Parallel Agency Gate

Trigger:
You want another process/agent to interact with cocoon.

Command:
```sh
./cocoon mcp
```

Use MCP tools:
`capabilities`, `run_capability`, `start_job`, `job_status`, `job_logs`, `stop_job`.

Gate:
Do not grant autonomous loops until health, logs, stop controls, and route timeouts are proven.

TMRL note:
Do not pursue TMRL setup from this package. The route is retained only as disabled historical context.

## Skill: Agent Navigator

Trigger:
An AI or operator needs an immediate route map without reading source.

Command:
```sh
./cocoon agent status --json
./cocoon agent next
./cocoon agent inspect sphere
./cocoon agent manifest
```

Look for:
`health`, compact `state`, and `next` route suggestions with payloads.

Persist:
No persistence. This skill decides what to run next.

## Skill: Audit The Interface

Trigger:
You changed routes, docs, CLI, MCP, or pygame affordances.

Command:
```sh
./cocoon audit
```

Look for:
`ok: true` and `failures: []`.

Persist:
No persistence. Fix failures before autonomous use.
