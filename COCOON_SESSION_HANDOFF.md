# Cocoon Session Handoff

Last updated: 2026-06-10

This file is the durable resume point for the current Cocoon/Glass Box work. Use it when a chat resets.

## Active Package

```sh
cd /data/data/com.termux/files/home/downloads/cocoon_cognition_agency_package
```

This is the actual package recovered from the prior transcript, not a duplicate workspace.

## Primary Commands

```sh
./cocoon
./cocoon app --host 127.0.0.1 --port 8765 --no-open
export DISPLAY=:0
./cocoon glass
./cocoon doctor --json
./cocoon audit
./cocoon agent brief
./cocoon capabilities list
./cocoon capabilities run health
./cocoon mcp
```

`./cocoon` defaults to launching the local Authority app. `./cocoon glass` launches the pygame phone operations window and auto-starts Authority if needed.

## Current Running Service

The patched Authority was last restarted on `http://127.0.0.1:8765/`.

Pid file:

```sh
cat mira_kite_authority_runtime/authority_launch.pid
```

Last known pid: `27511`.

## GUI State

`cocoon_glass_phone.py` now drives a real pygame operations surface using actual Authority API data.

Controls:

- `Left` / `Right`: select capability.
- `Enter`: run selected capability.
- `P`: prime cache from live endpoints: state, health, facilities, manual, map, events, lessons, curriculum, vocab, logs, snapshot, game recipe, capability manifest.
- `I`: study selected route contract and load followups.
- `N`: run next followup.
- `E`: run current nested drill sequence.
- `L`: refresh training logs.
- `M`: refresh facility map.
- `S` or `Space`: run sphere burst.
- `A` or `T`: run bounded autopilot.
- `D`: run hardening route.
- `R`: refresh state.
- `Esc` or `Q`: quit.

The overlay reports real route, endpoint, teaches text, cached datapools, result keys, sequence, receipt, and blockers.

The browser Authority app now has a utility-grade operations layer:

- `Capability Contract Matrix`: generated from `/api/capability_manifest`.
- Contract cards show key, group, risk, method/path, purpose, requisites, expected outputs, followups, default payload, and schema.
- Contract cards support `Inspect` and `Run`.
- `Live Operation Relay`: every browser control records button, status, route, method, payload, latency, output keys, receipt, state delta, errors/blockers, and the last 8 operations.
- The canvas overlays the active operation route/status/keys/delta so the visual surface reflects actual IO.

## Truth Standard

No fake readiness:

- Ready means importable/runnable in this Termux Python and verified by live route or smoke test.
- Blocked means exact import/build/runtime reason is recorded.
- TMRL is intentionally not pursued unless explicitly re-enabled by the operator.
- External lanes are requisites, not optional, when they are part of the intended lane.

## Ready

- Core agent: `numpy`, `torch`
- Provenance: `cascade`, `quinesmith`
- Authority API and app: `flask`, local HTTP routes
- Link mode dependency: `websockets`
- Gym lane: `gymnasium`
- Sphere visual/headless: `pygame`, `OpenGL`, `numpy`, `torch`
- Drone files: `cocoon_drone_adapter.py`, `cocoon_drone_arena.py`, `jsbsim_quadcopter.py`
- Drone plots: `matplotlib`, `scipy`
- PyFlyt package shell: `PyFlyt`
- PyFlyt pure Python companion: `pettingzoo`

## Blocked

- `model_export`: blocked because `onnx` imports fail with native symbol error:
  `ImportError: dlopen failed: cannot locate symbol "PyExc_ImportError" referenced by onnx_cpp2py_export.abi3.so`.
- Complete drone 3D visualization: blocked because `pybullet` and `numba` are unavailable.
  - `pybullet` source wheel build failed on Android/Termux.
  - `numba` depends on `llvmlite`; `llvmlite` build fails with `RuntimeError: unsupported platform: 'android'`.
- `tmrl`: intentionally not pursued.
- Termux package manager has a pre-existing dirty post-install state around `xdg-utils`, causing Qt/OpenCV packages to remain unconfigured even when target packages install.

## Files Changed In This Work

- `cocoon_glass_phone.py`
- `glass_box_phone/display.py`
- `cocoon_capabilities.py`
- `cocoon_ops.py`
- `mira_kite_authority.py`
- `cocoon_drone_adapter.py`
- `GLASS_BOX_PHONE.md`
- `COCOON_OPERATOR_PLAYBOOK.md`
- `COCOON_SKILLS_COOKBOOK.md`
- `COCOON_REQUISITE_STATUS.md`
- `COCOON_SESSION_HANDOFF.md`

## Last Known Good Verification

```sh
python -m py_compile cocoon_ops.py cocoon_capabilities.py mira_kite_authority.py cocoon_glass_phone.py glass_box_phone/display.py cocoon_drone_adapter.py
./cocoon doctor --json
./cocoon audit
./cocoon capabilities run facilities
SDL_VIDEODRIVER=dummy ./cocoon glass --max-seconds 2 --fps 5 --poll-seconds 1
```

Expected summary:

- `./cocoon audit` returns `ok: true`.
- `doctor` reports `core_ready: true`.
- `doctor` reports `model_export` blocked on `onnx`.
- `doctor` reports `drone_visual` blocked on `pybullet` and `numba`.
- `facilities` reports drone files present and dependency errors for missing visual requisites.
- dummy `glass` starts and closes cleanly.
- served browser HTML contains `Capability Contract Matrix`, `Live Operation Relay`, and `runCapabilityContract`.

## MCP Evaluation

MCP transport: stdio JSON-RPC, launched with:

```sh
./cocoon mcp
```

Verified tools:

- `doctor`: passed; returns real dependency blockers and cocoon inventory.
- `list_cocoons`: passed; returns runnable Python cocoons and zip packages.
- `persistence`: passed; returns schema/log/memory artifacts.
- `capabilities`: passed; returns full capability fabric.
- `run_capability`: passed with `health`.
- `start_job`: passed; starts a real subprocess and returns job id/log path.
- `job_status`: passed.
- `job_logs`: passed.
- `stop_job`: passed.
- `eval`: initially failed because the default cocoon path pointed at `/data/data/com.termux/files/home/downloads/cocoon_cognition_agency.py`; fixed to use `PACKAGE_DIR / "cocoon_cognition_agency.py"`. Retest passed: 4 cases run, 4 passed.

MCP warnings from `onnxruntime`/`pygame.pkgdata` appear on stderr during some tools. The JSON-RPC stdout responses remain structured.

## Next Evaluation Pass

1. Confirm `./cocoon`/Authority launch behavior.
2. Confirm `./cocoon glass` on real Termux:X11 with `DISPLAY=:0`.
3. Exercise MCP stdio tools:
   - `doctor`
   - `capabilities`
   - `run_capability`
   - `start_job`
   - `job_status`
   - `job_logs`
   - `stop_job`
4. Debug errors through real outputs only.
