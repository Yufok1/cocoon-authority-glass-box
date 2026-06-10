# Glass Box Phone UI

This package now has two surfaces:

- `./cocoon app` starts the local Cocoon Authority HTTP API and browser UI.
- `./cocoon glass` starts a real `pygame` Glass Box window that talks to that API.

The Glass Box path is isolated from Twitch, OBS, StreamElements, WebRTC, and cloud streaming. It reuses the visual pygame dashboard from `Yufok1/Glass-Box-Games`, but drives it from the local cocoon state and sphere/autopilot endpoints.

## Start

Recommended, one command after Termux:X11 is running:

```sh
export DISPLAY=:0
cd /data/data/com.termux/files/home/downloads/cocoon_cognition_agency_package
./cocoon glass
```

If the Authority API is not already running, `./cocoon glass` starts it in the background at `http://127.0.0.1:8765/`.

Manual two-session mode:

```sh
cd /data/data/com.termux/files/home/downloads/cocoon_cognition_agency_package
./cocoon app --host 127.0.0.1 --port 8765 --no-open
```

In a Termux:X11 session:

```sh
export DISPLAY=:0
cd /data/data/com.termux/files/home/downloads/cocoon_cognition_agency_package
./cocoon glass
```

Use `./cocoon glass --no-start-authority` when you want it to fail instead of launching the backend automatically.

## Controls

- `Left` / `Right`: move through the complete capability fabric.
- `Enter`: run the selected capability.
- `P`: prime the local study cache from real Authority endpoints: state, health, facilities, manual, map, events, lessons, curriculum, vocab, logs, snapshot, game recipe, and capability manifest.
- `I`: study the selected route contract and load its metadata/followups into the overlay.
- `N`: run the selected route's next followup, when one is defined.
- `E`: run the current nested drill sequence.
- `L`: refresh training logs.
- `M`: refresh the facility map.
- `S` or `Space`: run a real sphere burst through `/api/native/sphere_burst`.
- `A` or `T`: run a bounded autopilot cycle through `/api/autopilot`.
- `D`: run hardening checks through `/api/harden_facilities`.
- `R`: refresh live state from `/api/state`.
- `Esc` or `Q`: exit the pygame window.

The overlay reports only live or cached Authority data: selected route, endpoint, teaches text, result keys, cache age, sequence, receipt, and blockers. Blocked requisites are shown as blockers instead of being treated as optional.

## Capability Fabric

The full route table lives in `cocoon_capabilities.py`.

```sh
./cocoon capabilities list
./cocoon capabilities list --json
./cocoon capabilities run chat --payload '{"prompt":"hello","learn":true,"steps":1}'
./cocoon capabilities tour
```

The table includes diagnostics, memory, persistence, language, reasoning, dreamer, compound training, action heads, game lanes, overclocking, and dependency-gated external adapters.

## Smoke Test

This verifies the pygame surface without opening a real window:

```sh
SDL_VIDEODRIVER=dummy ./cocoon glass --max-seconds 2 --fps 5
```

For a real phone window, `SDL_VIDEODRIVER` must not be `dummy`, and `DISPLAY` must point at Termux:X11.
