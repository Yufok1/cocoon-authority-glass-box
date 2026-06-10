# Cocoon Operator Playbook

This package has four operating surfaces:

1. `./cocoon app`
   Local Authority HTTP server and browser UI.

2. `./cocoon glass`
   Real pygame Glass Box phone window. Uses Termux:X11 for a visible window.

3. `./cocoon capabilities`
   CLI route fabric for every named Authority capability.

4. `./cocoon agent`
   Agent-facing navigator: health, compact state, recommended next routes, manifest.

5. `./cocoon mcp`
   Stdio MCP bridge for tools and agencies. This is MCP-over-stdio, not SSE.

The system is intentionally local-first. SSE/HTTP MCP should only be added when a real client needs network MCP and the phone can keep the server alive reliably.

## First Commands

```sh
cd /data/data/com.termux/files/home/downloads/cocoon_cognition_agency_package
./cocoon doctor
./cocoon agent brief
./cocoon agent status
./cocoon agent inspect autopilot
./cocoon capabilities list
./cocoon capabilities inspect sphere
./cocoon audit
./cocoon app --host 127.0.0.1 --port 8765 --no-open
./cocoon glass
```

## Operating Loop

1. Diagnose:
   `./cocoon doctor`

2. Start Authority:
   `./cocoon app --host 127.0.0.1 --port 8765 --no-open`

3. Inspect capability fabric:
   `./cocoon capabilities list`

4. Ask the agent navigator for the next useful routes:
   `./cocoon agent next`

5. Run one safe action:
   `./cocoon capabilities run health`

6. Train lightly:
   `./cocoon capabilities run chat --payload '{"prompt":"explain your current state","learn":true,"steps":1}'`

7. Deepen:
   `./cocoon capabilities run autopilot --payload '{"mission":"improve verification and semantic memory","depth":1,"include_games":true,"save_after":false}'`

8. Persist:
   `./cocoon capabilities run save`

9. Export after a good batch:
   `./cocoon capabilities run export --payload '{"name":"facility_checkpoint.py"}'`

## Agent-First Ergonomics

Use this before touching mutating routes:

```sh
./cocoon agent brief
./cocoon agent next
./cocoon agent status --json
```

The agent console returns:

- authority health,
- compact active state,
- fabric counts,
- suggested next routes with payloads,
- graceful errors with concrete next commands when the Authority server is down.

Use this to inspect one route without running it:

```sh
./cocoon agent inspect sphere
./cocoon capabilities inspect autopilot
```

Use this after edits:

```sh
./cocoon audit
```

The audit checks that every capability has schema, expected outputs, risk class, timeout, followups, agent affordance, docs, CLI hooks, MCP bridge, and Glass Box files.

## Capability Groups

- diagnostics: health, state, capabilities, facilities, manual, map, events, lessons, cocoons.
- memory: curriculum, vocabulary, logs, snapshot.
- persistence: save, export, reload.
- language: chat, teach, follow, quine.
- reasoning: signal, govern, relate, dreamer observe/propose, perturb.
- compound: autopilot, tour, deepen, feed.
- action: act, learn, curriculum score.
- games: sphere, gym, game recipe.
- external: drone, relay link. `tmrl_doctor` is retained only as an operator-disabled historical diagnostic.
- overclock: cheat.

## Integration Gates

Only integrate an external/parallel agency when these gates pass:

- `./cocoon doctor` says the required capability is ready.
- `./cocoon capabilities run health` returns `status: ok`.
- The target agency has a bounded job lifecycle: start, status, logs, stop.
- The route has an explicit payload schema and timeout.
- The route can produce a receipt, state delta, or log artifact.
- The phone has enough resources for the loop without thermal or memory collapse.

If any gate fails, treat it as a real blocker for that lane. Do not call requisites optional.

TMRL is outside the pursuit path by operator decision. Treat any TMRL route as disabled documentation, not a dependency to repair.

## MCP

`./cocoon mcp` is a stdio MCP server. It exposes:

- `doctor`
- `list_cocoons`
- `persistence`
- `start_job`
- `stop_job`
- `job_status`
- `job_logs`
- `eval`
- `capabilities`
- `run_capability`

That is enough for local agencies to discover resources, start bounded jobs, inspect logs, run evals, and call the full Authority capability fabric.

## SSE/HTTP MCP Position

Do not add SSE just because it sounds bigger. Add it when:

- an MCP client on another process/device needs network transport,
- auth and LAN exposure are understood,
- a watchdog restarts the server,
- logs and stop controls are exposed,
- the stdio MCP path is already passing.

Until then, stdio MCP plus local HTTP Authority is the simpler reliable core.
