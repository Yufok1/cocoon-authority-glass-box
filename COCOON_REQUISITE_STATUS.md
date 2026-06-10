# Cocoon Requisite Status

Generated from live Termux install/import attempts on 2026-06-10.

## Ready

- Core agent: `numpy`, `torch`
- Provenance: `cascade`, `quinesmith`
- Authority app: `flask`, local HTTP routes
- Link mode: `websockets`
- Gym games: `gymnasium`
- Sphere visual/headless: `pygame`, `OpenGL`, `numpy`, `torch`
- Drone plots: `matplotlib`, `scipy`
- PyFlyt package shell: `PyFlyt`
- PyFlyt pure Python companion: `pettingzoo`

## Blocked Requisites

- `onnx`: installed wheel imports fail with `ImportError: dlopen failed: cannot locate symbol "PyExc_ImportError" referenced by onnx_cpp2py_export.abi3.so`.
- `pybullet`: source build fails on Android/Termux while building the wheel.
- `numba`: source build depends on `llvmlite`; `llvmlite` build fails with `RuntimeError: unsupported platform: 'android'`.
- `tmrl`: intentionally not pursued by operator decision.

## Consequences

- `model_export` remains blocked until `onnx` imports cleanly.
- Complete drone 3D visualization remains blocked until `pybullet` and `numba/llvmlite` are available for this Android Python.
- Drone non-visual adapter/files and matplotlib plot support are present, but the complete PyFlyt visual lane is not ready.
