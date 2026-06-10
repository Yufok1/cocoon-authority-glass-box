#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import importlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = PACKAGE_DIR.parent
MODULE_AVAILABILITY_CACHE: dict[str, bool] = {}
MODULE_ERROR_CACHE: dict[str, str] = {}


REQUIRED_MODULES: dict[str, list[str]] = {
    "core_agent": ["numpy", "torch"],
    "provenance": ["cascade", "quinesmith"],
    "model_export": ["onnx", "onnxruntime"],
    "native_serve": ["flask"],
    "authority_app": ["numpy", "torch"],
    "link_mode": ["websockets"],
    "gym_games": ["gymnasium"],
    "sphere_headless": ["numpy", "torch"],
    "pygame_visual": ["pygame"],
    "sphere_visual": ["numpy", "torch", "pygame", "OpenGL"],
    "drone_plots": ["matplotlib"],
    "drone_visual": ["PyFlyt", "pybullet", "pettingzoo", "numba"],
    "tmrl": ["torch", "tmrl"],
}


CORE_CAPABILITIES = {"core_agent", "provenance", "native_serve", "authority_app", "link_mode", "sphere_headless"}


COCOON_MARKERS = (
    "class CocoonAgent",
    "def cocoon_capability_contract",
    "class SphereArena",
    "_ARCHITECTURE_B64",
)


MODE_REQUIREMENTS: dict[str, list[str]] = {
    "info": [],
    "readme": [],
    "chat": ["core_agent"],
    "serve": ["core_agent", "native_serve"],
    "authority": ["authority_app"],
    "sphere": ["sphere_headless"],
    "sphere_visual": ["sphere_visual"],
    "gym": ["core_agent", "gym_games"],
    "link": ["core_agent", "link_mode"],
    "tmrl_doctor": ["tmrl"],
}


COMMAND_PRESETS: dict[str, dict[str, Any]] = {
    "info": {"cocoon": "cocoon_cognition_agency.py", "mode": "info"},
    "readme": {"cocoon": "cocoon_cognition_agency.py", "mode": "readme"},
    "chat": {"cocoon": "cocoon_cognition_agency.py", "mode": "chat"},
    "serve": {"cocoon": "cocoon_cognition_agency.py", "mode": "serve", "port": 8765},
    "authority": {"cocoon": "cocoon_cognition_agency.py", "mode": "authority", "port": 8080},
    "sphere_headless": {"cocoon": "cocoon_cognition_agency.py", "mode": "sphere", "headless": True, "train": True},
    "gym_cartpole": {"cocoon": "cocoon_cognition_agency.py", "mode": "gym", "env": "CartPole-v1", "episodes": 1},
    "tmrl_doctor": {"cocoon": "cocoon_cognition_agency.py", "mode": "tmrl_doctor"},
}


def module_available(name: str) -> bool:
    if name in MODULE_AVAILABILITY_CACHE:
        return MODULE_AVAILABILITY_CACHE[name]
    try:
        if importlib.util.find_spec(name) is None:
            MODULE_AVAILABILITY_CACHE[name] = False
            MODULE_ERROR_CACHE[name] = "module spec not found"
            return False
        importlib.import_module(name)
        MODULE_AVAILABILITY_CACHE[name] = True
        MODULE_ERROR_CACHE.pop(name, None)
        return True
    except Exception as exc:
        MODULE_AVAILABILITY_CACHE[name] = False
        MODULE_ERROR_CACHE[name] = f"{exc.__class__.__name__}: {str(exc)[:240]}"
        return False


def dependency_report() -> dict[str, Any]:
    modules = sorted({module for group in REQUIRED_MODULES.values() for module in group})
    installed = {module: module_available(module) for module in modules}
    module_errors = {module: MODULE_ERROR_CACHE.get(module, "") for module, ok in installed.items() if not ok}
    capabilities = {}
    for capability, required in REQUIRED_MODULES.items():
        missing = [module for module in required if not installed.get(module)]
        capabilities[capability] = {
            "ready": not missing,
            "required_for_core": capability in CORE_CAPABILITIES,
            "required": required,
            "missing": missing,
        }
    core_missing = sorted(
        {module for capability, data in capabilities.items() if data["required_for_core"] for module in data["missing"]}
    )
    return {
        "modules": installed,
        "module_errors": module_errors,
        "capabilities": capabilities,
        "core_ready": not core_missing,
        "core_missing": core_missing,
    }


def file_state(path: Path) -> dict[str, Any]:
    path = path.resolve()
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists and path.is_file() else 0,
    }


def persistence_report() -> dict[str, Any]:
    schema_path = PACKAGE_DIR / "training_logs" / "schema.json"
    schema: dict[str, Any] = {}
    if schema_path.exists():
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except Exception as exc:
            schema = {"error": f"schema_read_failed: {exc}"}
    default_log = schema.get("default_path", "training_logs/live_learning_trace.jsonl")
    return {
        "schema": file_state(schema_path),
        "default_log": file_state(PACKAGE_DIR / default_log),
        "environment_memory": file_state(PACKAGE_DIR / "GLOBAL_ENVIRONMENT_MEMORY.md"),
        "metadata": file_state(PACKAGE_DIR / "metadata.json"),
        "pet_manifest": file_state(PACKAGE_DIR / "pet_manifest.json"),
        "schema_version": schema.get("version"),
        "event_types": schema.get("event_types", []),
        "required_fields": schema.get("required_fields", []),
    }


def command_matrix(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    matrix = {}
    for name, preset in COMMAND_PRESETS.items():
        cocoon = PACKAGE_DIR / preset["cocoon"]
        mode = preset["mode"]
        kwargs = {key: value for key, value in preset.items() if key not in {"cocoon", "mode"}}
        missing = missing_for_mode(mode)
        entry: dict[str, Any] = {
            "ready": not missing,
            "missing": missing,
            "cocoon_path": str(resolve_within_root(cocoon, root)),
            "mode": mode,
        }
        if not missing:
            entry["command"] = build_command(resolve_within_root(cocoon, root), mode, **kwargs)
        else:
            entry["command"] = None
        matrix[name] = entry
    return matrix


def missing_for_mode(mode: str) -> list[str]:
    report = dependency_report()["capabilities"]
    missing: set[str] = set()
    for capability in MODE_REQUIREMENTS.get(mode, []):
        missing.update(report.get(capability, {}).get("missing", []))
    return sorted(missing)


def looks_like_cocoon(path: Path) -> bool:
    if path.name in {Path(__file__).name, "cocoon_doctor.py", "cocoon_mcp_stdio.py", "cocoon"}:
        return False
    if path.suffix == ".zip":
        return "cocoon" in path.name.lower()
    if path.suffix != ".py":
        return False
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    return False
                if any(marker in chunk for marker in COCOON_MARKERS):
                    return True
    except Exception:
        return False


def scan_cocoons(root: Path = DEFAULT_ROOT, max_depth: int = 5) -> list[dict[str, Any]]:
    root = root.resolve()
    found = []
    for path in root.rglob("*"):
        try:
            rel_depth = len(path.relative_to(root).parts)
        except ValueError:
            continue
        if rel_depth > max_depth or not path.is_file():
            continue
        if path.suffix not in {".py", ".zip"}:
            continue
        if not looks_like_cocoon(path):
            continue
        kind = "python_cocoon" if path.suffix == ".py" else "zip_package"
        found.append(
            {
                "path": str(path),
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                "kind": kind,
                "runnable": kind == "python_cocoon",
            }
        )
    found.sort(key=lambda item: (not item["runnable"], item["path"]))
    return found


def resolve_within_root(path: str | os.PathLike[str], root: Path = DEFAULT_ROOT) -> Path:
    root = root.resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path is outside managed root: {resolved}") from exc
    return resolved


def build_command(
    cocoon_path: Path,
    mode: str,
    *,
    port: int | None = None,
    max_organisms: int | None = None,
    train: bool = False,
    headless: bool = True,
    balls: int = 1,
    misses: int = 3,
    env: str = "CartPole-v1",
    episodes: int = 1,
) -> list[str]:
    if cocoon_path.suffix != ".py":
        raise ValueError("only extracted Python cocoon files are directly runnable")
    if not cocoon_path.is_file():
        raise FileNotFoundError(str(cocoon_path))

    py = sys.executable
    if mode == "authority":
        cmd = [py, str(PACKAGE_DIR / "mira_kite_authority.py"), "--cocoon", str(cocoon_path)]
        if port is not None:
            cmd.extend(["--port", str(int(port))])
        return cmd
    if mode == "readme":
        return [py, str(cocoon_path), "--readme"]
    if mode == "tmrl_doctor":
        return [py, str(PACKAGE_DIR / "cocoon_tmrl_adapter.py"), "--doctor", "--cocoon", str(cocoon_path)]

    cmd = [py, str(cocoon_path), "--mode", mode]
    if max_organisms is not None:
        cmd.extend(["--max-organisms", str(max(1, int(max_organisms)))])
    if mode == "serve" and port is not None:
        cmd.extend(["--port", str(int(port))])
    elif mode == "sphere":
        cmd.extend(["--balls", str(max(1, min(5, int(balls))))])
        cmd.extend(["--misses", str(max(1, min(20, int(misses))))])
        if headless:
            cmd.append("--headless")
        if train:
            cmd.append("--train")
    elif mode == "gym":
        cmd.extend(["--env", str(env), "--episodes", str(max(1, int(episodes)))])
        if not train:
            cmd.append("--no-learn")
    return cmd


@dataclass
class Job:
    id: str
    command: list[str]
    cwd: str
    mode: str
    cocoon_path: str
    started_at: float
    log_path: str
    process: subprocess.Popen[Any] = field(repr=False)

    def status(self) -> dict[str, Any]:
        code = self.process.poll()
        return {
            "id": self.id,
            "mode": self.mode,
            "cocoon_path": self.cocoon_path,
            "command": self.command,
            "cwd": self.cwd,
            "started_at": self.started_at,
            "runtime_seconds": round(time.time() - self.started_at, 3),
            "running": code is None,
            "returncode": code,
            "log_path": self.log_path,
        }


class JobManager:
    def __init__(self, root: Path = DEFAULT_ROOT, runtime_dir: Path | None = None):
        self.root = root.resolve()
        self.runtime_dir = (runtime_dir or PACKAGE_DIR / "cocoon_host_runtime").resolve()
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}

    def start(
        self,
        cocoon_path: str,
        mode: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        mode = str(mode)
        missing = missing_for_mode(mode)
        if missing:
            return {"started": False, "error": "missing_dependencies", "missing": missing, "mode": mode}
        cocoon = resolve_within_root(cocoon_path, self.root)
        command = build_command(cocoon, mode, **kwargs)
        job_id = uuid.uuid4().hex[:12]
        log_path = self.runtime_dir / f"{job_id}_{mode}.log"
        log_file = log_path.open("ab")
        process = subprocess.Popen(
            command,
            cwd=str(cocoon.parent),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        log_file.close()
        job = Job(
            id=job_id,
            command=command,
            cwd=str(cocoon.parent),
            mode=mode,
            cocoon_path=str(cocoon),
            started_at=time.time(),
            log_path=str(log_path),
            process=process,
        )
        self.jobs[job_id] = job
        return {"started": True, "job": job.status()}

    def list(self) -> list[dict[str, Any]]:
        return [job.status() for job in self.jobs.values()]

    def get(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if not job:
            return {"error": "job_not_found", "id": job_id}
        return job.status()

    def stop(self, job_id: str, timeout: float = 5.0) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if not job:
            return {"stopped": False, "error": "job_not_found", "id": job_id}
        if job.process.poll() is not None:
            return {"stopped": True, "job": job.status()}
        try:
            os.killpg(job.process.pid, signal.SIGTERM)
        except Exception:
            job.process.terminate()
        end = time.time() + timeout
        while time.time() < end:
            if job.process.poll() is not None:
                return {"stopped": True, "job": job.status()}
            time.sleep(0.1)
        try:
            os.killpg(job.process.pid, signal.SIGKILL)
        except Exception:
            job.process.kill()
        return {"stopped": True, "forced": True, "job": job.status()}

    def logs(self, job_id: str, tail_bytes: int = 8192) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if not job:
            return {"error": "job_not_found", "id": job_id}
        path = Path(job.log_path)
        if not path.exists():
            return {"id": job_id, "log": ""}
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > tail_bytes:
                handle.seek(-tail_bytes, os.SEEK_END)
            data = handle.read()
        return {"id": job_id, "size_bytes": size, "log": data.decode("utf-8", errors="replace")}


def doctor(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    return {
        "root": str(root.resolve()),
        "package_dir": str(PACKAGE_DIR),
        "dependencies": dependency_report(),
        "persistence": persistence_report(),
        "cocoons": scan_cocoons(root),
        "commands": command_matrix(root),
    }


def as_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True)
