# Termux and Hugging Face Space Quickstart

## Termux

Global command:

```sh
cocoon help
cocoon doctor
cocoon gui
```

Open the GUI on the phone:

```sh
cocoon gui --host 127.0.0.1 --port 8765
```

Open from another device on the same network:

```sh
cocoon gui --host 0.0.0.0 --port 8765
```

Then browse to:

```text
http://PHONE_IP:8765/
```

Useful commands:

```sh
cocoon info
cocoon readme
cocoon chat
cocoon sphere --headless --train
cocoon mcp
```

## Current Termux Runtime

Core installed and verified:

- `torch`
- `onnxruntime`
- `flask`
- `websockets`
- `gymnasium`
- `PyOpenGL`
- `cascade-lattice`
- `quinesmith`

Extension faculties still need platform work on this phone:

- `pygame`: needs SDL build path or a Termux-compatible package.
- `matplotlib`: source build is slow/heavy on Android.
- `PyFlyt`: depends on simulator/graphics stack.
- `tmrl`: pulls pandas/native build stack.
- `onnx`: source build wants native CMake/protobuf tooling; ONNX Runtime is already installed.

## Hugging Face Space Plan

Use the existing GUI as the Space app. Do not add another server.

Space runtime:

- SDK: Docker.
- Public port: `7860`.
- Command: `python mira_kite_authority.py --cocoon cocoon_cognition_agency.py --host 0.0.0.0 --port 7860`.
- Hardware: CPU first. Upgrade to GPU only if training latency or PyTorch load requires it.

Minimum Space files:

- `README.md` with Space YAML front matter.
- `Dockerfile`.
- `requirements.txt`.
- Cocoon package files.

Before push:

1. Ensure `cocoon doctor` core runtime is ready.
2. Ensure GUI launches locally.
3. Ensure no redundant host/API surface is introduced.
4. Keep heavyweight downloaded cocoons out unless intentionally tracked with Git LFS or copied into the Space.
5. Install and authenticate Hugging Face CLI.
6. Create the Space repo as Docker SDK.
7. Push only after local priorities and docs are finalized.
