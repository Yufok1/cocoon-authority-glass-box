#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pygame

from cocoon_capabilities import CAPABILITIES, Capability, enriched_capability, request_json
from glass_box_phone.display import GameDisplay


DEFAULT_AUTHORITY = "http://127.0.0.1:8765"
PACKAGE_DIR = Path(__file__).resolve().parent
PRIME_ENDPOINTS = (
    ("state", "/api/state"),
    ("health", "/api/native/health"),
    ("facilities", "/api/native/facilities"),
    ("manual", "/api/facility_manual"),
    ("map", "/api/facility_map"),
    ("events", "/api/events"),
    ("lessons", "/api/lessons"),
    ("curriculum", "/api/native/curriculum"),
    ("vocab", "/api/native/vocab?limit=80"),
    ("logs", "/api/native/training/logs?limit=40"),
    ("snapshot", "/api/native/snapshot?full=false"),
    ("game_recipe", "/api/native/game_recipe?lane=all"),
    ("capability_manifest", "/api/capability_manifest"),
    ("wealth", "/api/wealth_surface"),
)

AFK_WORKER_SEQUENCE = (
    "health",
    "state",
    "dreamer_observe",
    "dreamer_propose",
    "chat",
    "teach",
    "follow",
    "govern",
    "relate",
    "council",
    "quine",
    "sphere",
    "snapshot",
    "logs",
    "save",
)


def authority_json(base_url: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 8.0) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    if payload is None:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    raw = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=raw,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def require_display() -> None:
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        return
    if not os.environ.get("DISPLAY"):
        raise SystemExit(
            "No DISPLAY is set, so pygame cannot open a real graphical window.\n"
            "Start Termux:X11 or another X server, then run:\n"
            "  export DISPLAY=:0\n"
            "  ./cocoon glass\n"
        )


def synth_frame(width: int, height: int, state: dict[str, Any], sphere: dict[str, Any], tick: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    cx = width / 2.0
    cy = height / 2.0
    pulse = tick / 18.0
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rings = (np.sin(dist / 18.0 - pulse) + 1.0) * 0.5
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = (rings * 18).astype(np.uint8)
    frame[..., 1] = (rings * 70 + 14).astype(np.uint8)
    frame[..., 2] = (rings * 96 + 20).astype(np.uint8)

    catches = int(sphere.get("collective_catches", 0) or 0)
    misses = int(sphere.get("collective_misses", 0) or 0)
    balls = max(1, int(sphere.get("balls", 1) or 1))
    org_count = max(2, len(state.get("organism_names", []) or []) + 2)

    def dot(px: float, py: float, radius: int, color: tuple[int, int, int]) -> None:
        mask = (x - px) ** 2 + (y - py) ** 2 <= radius ** 2
        frame[mask] = color

    orbit = min(width, height) * 0.28
    for idx in range(org_count):
        angle = pulse * 0.8 + idx * (math.tau / org_count)
        dot(cx + math.cos(angle) * orbit * 0.55, cy + math.sin(angle) * orbit * 0.55, 7, (60, 226, 138))
    for idx in range(balls):
        angle = -pulse * 1.4 + idx * (math.tau / balls)
        dot(cx + math.cos(angle) * orbit, cy + math.sin(angle) * orbit, 11 + min(catches, 8), (230, 184, 75))
    if misses:
        frame[-18:, : min(width, misses * 38), :] = (140, 42, 44)
    return frame


def action_probs(state: dict[str, Any], sphere: dict[str, Any]) -> np.ndarray:
    cycles = max(1, int(state.get("cycles", 1) or 1))
    catches = float(sphere.get("collective_catches", 0) or 0)
    misses = float(sphere.get("collective_misses", 0) or 0)
    base = np.array([1.0 + catches, 1.0 + misses, 1.0 + cycles % 7, 2.0], dtype=float)
    return base / base.sum()


def safe_capabilities() -> list[Capability]:
    return list(CAPABILITIES)


def compact_keys(data: Any, limit: int = 9) -> str:
    if isinstance(data, dict):
        keys = sorted(str(key) for key in data.keys())
        return ", ".join(keys[:limit]) + ("..." if len(keys) > limit else "")
    if isinstance(data, list):
        return f"list[{len(data)}]"
    return type(data).__name__


def short_receipt(data: Any, fallback: str = "") -> str:
    if not isinstance(data, dict):
        return fallback[:18] if fallback else "none"
    for key in ("receipt", "last_receipt", "merkle_root", "checkpoint"):
        value = data.get(key)
        if value:
            return str(value)[:24]
    state = data.get("state") or data.get("state_after")
    if isinstance(state, dict):
        return short_receipt(state, fallback)
    return fallback[:18] if fallback else "none"


def extract_state(current: dict[str, Any], data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return current
    for key in ("state", "state_after"):
        value = data.get(key)
        if isinstance(value, dict):
            return value
    if "vocabulary_size" in data or "active_cocoon_name" in data:
        return data
    return current


def dependency_probe(cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    facilities = cache.get("facilities", {}).get("data")
    if isinstance(facilities, dict):
        probe = facilities.get("dependency_probe")
        if isinstance(probe, dict):
            return probe
    health = cache.get("health", {}).get("data")
    if isinstance(health, dict):
        deps = health.get("dependencies") or health.get("dependency_probe")
        if isinstance(deps, dict):
            return deps
    return {}


def dependency_errors(cache: dict[str, dict[str, Any]]) -> dict[str, str]:
    facilities = cache.get("facilities", {}).get("data")
    if isinstance(facilities, dict):
        errors = facilities.get("dependency_errors")
        if isinstance(errors, dict):
            return {str(key): str(value) for key, value in errors.items()}
    return {}


def capability_blocker(cap: Capability, cache: dict[str, dict[str, Any]]) -> str:
    if "operator_disabled" in cap.prerequisites:
        return "operator disabled"
    if not cap.prerequisites:
        return "none"
    deps = dependency_probe(cache)
    errors = dependency_errors(cache)
    missing = [item for item in cap.prerequisites if not deps.get(item)]
    if not missing:
        return "none"
    detailed = []
    for item in missing:
        detail = errors.get(item)
        detailed.append(f"{item}: {detail}" if detail else item)
    return "; ".join(detailed)


def cache_summary(cache: dict[str, dict[str, Any]]) -> str:
    if not cache:
        return "empty"
    items = []
    now = time.time()
    for name, entry in sorted(cache.items())[-5:]:
        age = max(0, int(now - float(entry.get("time", now))))
        status = entry.get("status", "ok")
        items.append(f"{name}:{status}:{age}s")
    return " | ".join(items)


def overlay_for(
    cap: Capability,
    state: dict[str, Any],
    cache: dict[str, dict[str, Any]],
    last_result: dict[str, Any] | None,
    last_status: str,
    sequence: list[str],
) -> dict[str, Any]:
    blocker = capability_blocker(cap, cache)
    status = "blocked" if blocker != "none" else last_status
    border = (230, 105, 80) if status == "blocked" else (230, 190, 70) if status == "running" else (50, 210, 150)
    result = last_result or {}
    route = f"{cap.method} {cap.path} [{cap.group}/{cap.key}]"
    if cap.prerequisites:
        route += " req=" + ",".join(cap.prerequisites)
    return {
        "title": f"{state.get('active_cocoon_name') or 'Cocoon'} -> {cap.label}",
        "route": route,
        "teaches": cap.teaches,
        "status": status,
        "last_keys": compact_keys(result) if result else "none",
        "cache": cache_summary(cache),
        "sequence": " > ".join(sequence[:5]) if sequence else "none",
        "receipt": short_receipt(result, str(state.get("last_receipt") or "")),
        "blocker": blocker,
        "border": border,
    }


def game_lanes(cache: dict[str, dict[str, Any]], wealth_surface: dict[str, Any], sphere: dict[str, Any]) -> list[dict[str, Any]]:
    facility_map = wealth_surface.get("facility_map") if isinstance(wealth_surface.get("facility_map"), dict) else {}
    nodes = facility_map.get("nodes") if isinstance(facility_map.get("nodes"), list) else []
    node_status = {
        str(node.get("id")): str(node.get("status", "unknown"))
        for node in nodes
        if isinstance(node, dict) and node.get("id")
    }
    recipe_data = cache.get("game_recipe", {}).get("data")
    recipes = recipe_data.get("recipes", {}) if isinstance(recipe_data, dict) else {}
    readiness = recipe_data.get("readiness", {}) if isinstance(recipe_data, dict) else {}
    lanes = []
    labels = {
        "sphere": "Sphere",
        "gym": "Gym",
        "drone": "Drone",
        "tmrl": "TMRL",
        "link": "Relay Link",
    }
    for key in ("sphere", "gym", "drone", "tmrl", "link"):
        recipe = recipes.get(key, {}) if isinstance(recipes, dict) else {}
        ready = readiness.get(key, {}) if isinstance(readiness, dict) else {}
        status = node_status.get(key)
        if not status:
            if isinstance(ready, dict) and ready.get("ready"):
                status = "ready"
            elif key == "sphere":
                status = "ready"
            elif key == "tmrl":
                status = "disabled"
            else:
                status = "needs dependency"
        details = []
        if key == "sphere":
            details.append(f"frames {sphere.get('frames_run', 0)} catches {sphere.get('collective_catches', 0)} misses {sphere.get('collective_misses', 0)}")
            details.append("headless arena geometry relayed to pygame")
        if isinstance(recipe, dict):
            for field in ("app_endpoint", "dependency", "bounded", "cli"):
                if recipe.get(field):
                    details.append(f"{field}: {recipe.get(field)}")
        lanes.append(
            {
                "key": key,
                "label": labels.get(key, key.title()),
                "status": status,
                "active": key == "sphere" and bool(sphere.get("frames_run")),
                "details": [str(item) for item in details[:4]],
            }
        )
    return lanes


def build_operational_surface(
    selected_overlay: dict[str, Any],
    selected_key: str,
    state: dict[str, Any],
    sphere: dict[str, Any],
    cache: dict[str, dict[str, Any]],
    history: deque[dict[str, Any]],
    wealth_surface: dict[str, Any],
    afk: dict[str, Any],
) -> dict[str, Any]:
    wealth = wealth_surface if isinstance(wealth_surface, dict) else {}
    capability_manifest = wealth.get("capability_manifest") if isinstance(wealth.get("capability_manifest"), dict) else {}
    capabilities = capability_manifest.get("capabilities") if isinstance(capability_manifest.get("capabilities"), list) else []
    cascade = wealth.get("cascade") if isinstance(wealth.get("cascade"), dict) else {}
    receipts = cascade.get("latest_receipts") if isinstance(cascade.get("latest_receipts"), list) else []
    return {
        "updated": time.time(),
        "selected": selected_overlay,
        "selected_key": selected_key,
        "state": state,
        "sphere": sphere,
        "history": list(history)[:30],
        "events": wealth.get("events", []),
        "lessons": wealth.get("lessons", []),
        "receipts": receipts,
        "diagnostics": wealth.get("diagnostics", {}),
        "facility_map": wealth.get("facility_map", {}),
        "capabilities": capabilities,
        "games": game_lanes(cache, wealth, sphere),
        "feed_reports": wealth.get("feed_reports", []),
        "informational_wealth": wealth.get("informational_wealth", {}),
        "afk": dict(afk),
    }


def capability_obs(state: dict[str, Any], sphere: dict[str, Any], selected: int, count: int) -> np.ndarray:
    return np.array([
        float(state.get("vocabulary_size", 0) or 0) / 100000.0,
        float(state.get("knowledge_relations", 0) or 0) / 10000.0,
        float(state.get("cycles", 0) or 0) / 1000.0,
        float(sphere.get("collective_catches", 0) or 0) / 20.0,
        float(sphere.get("collective_misses", 0) or 0) / 20.0,
        float(selected) / max(1, count - 1),
    ], dtype=np.float32)


def wait_for_authority(base_url: str, timeout: float = 12.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            return authority_json(base_url, "/api/state", timeout=2.0)
        except Exception as exc:
            last_error = exc
            time.sleep(0.35)
    raise SystemExit(f"Authority API is not reachable at {base_url}: {last_error}")


def pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def existing_authority_pid() -> int | None:
    pid_path = PACKAGE_DIR / "mira_kite_authority_runtime" / "authority_launch.pid"
    if not pid_path.exists():
        return None
    try:
        raw = pid_path.read_text(encoding="utf-8").strip()
        digits = "".join(ch for ch in raw if ch.isdigit())
        if not digits:
            return None
        pid = int(digits)
    except Exception:
        return None
    return pid if pid_is_running(pid) else None


def launch_authority(base_url: str) -> subprocess.Popen[bytes]:
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8765
    cmd = [
        sys.executable,
        str(PACKAGE_DIR / "cocoon_launch.py"),
        "--cocoon",
        str(PACKAGE_DIR / "cocoon_cognition_agency.py"),
        "--host",
        host,
        "--port",
        str(port),
        "--no-open",
    ]
    log_dir = PACKAGE_DIR / "mira_kite_authority_runtime"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "glass_box_authority_launch.log"
    log_file = log_path.open("ab")
    proc = subprocess.Popen(
        cmd,
        cwd=str(PACKAGE_DIR),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    log_file.close()
    print(f"[GLASS] Authority was not reachable. Launching it in background at {base_url} (pid {proc.pid}).")
    print(f"[GLASS] Authority launch log: {log_path}")
    return proc


def run(args: argparse.Namespace) -> int:
    require_display()
    try:
        state = wait_for_authority(args.authority, args.timeout)
    except SystemExit:
        if not args.start_authority:
            raise
        pid = existing_authority_pid()
        if pid:
            print(f"[GLASS] Authority process {pid} exists but API is not ready yet; waiting before fallback launch.")
            state = wait_for_authority(args.authority, args.timeout)
        else:
            launch_authority(args.authority)
            state = wait_for_authority(args.authority, args.timeout)
    display = GameDisplay(width=args.width, height=args.height, title="Cocoon Glass Authority")
    display.start()
    if args.fullscreen:
        display.toggle_fullscreen()
    caps = safe_capabilities()
    labels = [cap.label[:16] for cap in caps]
    display.set_game("CocoonCapabilityFabric", "Cocoon Capability Fabric", len(labels), (30,))
    display._action_labels = labels
    display._state_labels = ["vocab", "relations", "cycles", "catches", "misses", "selected"]
    display.set_genesis(str(state.get("last_receipt") or "authority"))

    sphere: dict[str, Any] = {"collective_catches": 0, "collective_misses": 0, "balls": 1, "frames_run": 0}
    cache: dict[str, dict[str, Any]] = {}
    history: deque[dict[str, Any]] = deque(maxlen=24)
    wealth_surface: dict[str, Any] = {}
    sequence: list[str] = ["health", "state", "snapshot", "logs"]
    last_result: dict[str, Any] | None = None
    last_status = "watching"
    selected_cap = 0
    view_mode = args.view
    afk_mode = bool(args.afk)
    last_afk_step = 0.0
    afk_index = 0
    afk_last_worker = "idle"
    afk_last_status = "off" if not afk_mode else "armed"
    afk_cycles = 0
    last_poll = 0.0
    last_wealth_poll = 0.0
    last_burst = 0.0
    started = time.time()
    tick = 0
    running = True

    def remember(name: str, data: Any, status: str = "ok") -> None:
        nonlocal last_result, last_status, wealth_surface
        entry = {"name": name, "status": status, "time": time.time(), "keys": compact_keys(data), "data": data}
        cache[name] = entry
        history.appendleft({k: v for k, v in entry.items() if k != "data"})
        if isinstance(data, dict):
            last_result = data
        else:
            last_result = {"result": data}
        last_status = status
        if name == "wealth" and isinstance(data, dict):
            wealth_surface = data
        display.update_queue([{"viewer": item["name"], "game": item["status"], "pos": idx + 1} for idx, item in enumerate(list(history)[:8])])

    def fetch_named(name: str, path: str, timeout: float = 6.0) -> Any:
        data = authority_json(args.authority, path, timeout=timeout)
        remember(name, data, "ok")
        return data

    def prime_cache() -> None:
        nonlocal state, last_status
        last_status = "running"
        display.show_action_feedback("Prime cache", "Cocoon")
        for name, path in PRIME_ENDPOINTS:
            try:
                data = fetch_named(name, path, timeout=8.0)
                if name == "state" and isinstance(data, dict):
                    state = data
            except Exception as exc:
                remember(name, {"error": str(exc), "path": path}, "blocked")

    def study_capability(cap: Capability) -> None:
        nonlocal sequence
        try:
            data = authority_json(args.authority, f"/api/capability?key={cap.key}", timeout=5.0)
        except Exception:
            data = enriched_capability(cap)
        remember(f"study:{cap.key}", data, "ok")
        followups = data.get("followups") if isinstance(data, dict) else None
        if isinstance(followups, list) and followups:
            sequence = [cap.key] + [str(item) for item in followups if str(item) in {c.key for c in CAPABILITIES}]
        display.show_action_feedback(f"Study {cap.key}", "Cocoon")

    def run_capability(cap: Capability, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        nonlocal state, sphere, last_status
        blocker = capability_blocker(cap, cache)
        if blocker != "none":
            data = {
                "ok": False,
                "capability": cap.key,
                "blocked": blocker,
                "next": ["Prime cache with P", "Run doctor", "Install requisites or choose another route"],
            }
            remember(cap.key, data, "blocked")
            display.show_action_feedback(f"Blocked {cap.key}: {blocker}", "Cocoon")
            return data
        last_status = "running"
        display.show_action_feedback(f"Run {cap.key}", "Cocoon")
        data = request_json(args.authority, cap, payload, timeout=args.capability_timeout)
        state = extract_state(state, data)
        if isinstance(data, dict) and "collective_catches" in data:
            sphere = data
        remember(cap.key, data, "ok")
        return data

    def run_followup() -> None:
        current = caps[selected_cap]
        candidates = list(current.followups or ())
        if not candidates:
            meta = enriched_capability(current)
            candidates = [str(item) for item in meta.get("followups", [])]
        lookup = {cap.key: cap for cap in CAPABILITIES}
        for key_name in candidates:
            cap = lookup.get(key_name)
            if cap:
                run_capability(cap)
                return
        remember("followup", {"ok": False, "message": "no runnable followup", "source": current.key}, "blocked")

    def select_capability(key_name: str) -> None:
        nonlocal selected_cap
        for idx, cap in enumerate(caps):
            if cap.key == key_name:
                selected_cap = idx
                display.show_action_feedback(cap.label, "Selected")
                return
        remember("select", {"ok": False, "key": key_name, "message": "unknown capability"}, "blocked")

    def toggle_afk() -> None:
        nonlocal afk_mode, afk_last_status, last_afk_step
        afk_mode = not afk_mode
        afk_last_status = "armed" if afk_mode else "off"
        last_afk_step = 0.0
        display.show_action_feedback("AFK workers on" if afk_mode else "AFK workers off", "Glass")

    def run_button_command(command: str) -> None:
        nonlocal view_mode, sphere, state
        if command == "run_selected":
            run_capability(caps[selected_cap])
        elif command == "study_selected":
            study_capability(caps[selected_cap])
        elif command == "followup":
            run_followup()
        elif command == "sequence":
            run_sequence()
        elif command == "prime":
            prime_cache()
        elif command == "sphere":
            sphere = run_capability(next(cap for cap in CAPABILITIES if cap.key == "sphere"), {"frames": args.sphere_frames})
            state = sphere.get("state") or authority_json(args.authority, "/api/state")
        elif command == "autopilot":
            data = authority_json(
                args.authority,
                "/api/autopilot",
                {"mission": args.mission, "depth": 1, "include_games": True, "save_after": False},
                timeout=45,
            )
            state = data.get("state") or authority_json(args.authority, "/api/state")
            for stage in data.get("stages", []):
                result = stage.get("result") if isinstance(stage, dict) else None
                if isinstance(result, dict) and "collective_catches" in result:
                    sphere = result
            remember("autopilot", data, "ok")
        elif command == "harden":
            data = authority_json(args.authority, "/api/harden_facilities", timeout=20)
            state = data.get("state") or authority_json(args.authority, "/api/state")
            remember("harden", data, "ok")
        elif command == "refresh":
            state = fetch_named("state", "/api/state", timeout=5.0)
        elif command == "logs":
            fetch_named("logs", "/api/native/training/logs?limit=80", timeout=5.0)
        elif command == "map":
            fetch_named("map", "/api/facility_map", timeout=5.0)
        elif command == "fullscreen":
            display.toggle_fullscreen()
        elif command == "afk":
            toggle_afk()
        elif command in {"ops", "games", "feed"}:
            view_mode = command
        display.show_action_feedback(command.replace("_", " ").title(), "Glass")

    def run_sequence() -> None:
        lookup = {cap.key: cap for cap in CAPABILITIES}
        for key_name in list(sequence):
            cap = lookup.get(key_name)
            if cap:
                try:
                    run_capability(cap)
                except Exception as exc:
                    remember(f"sequence:{key_name}", {"error": str(exc)}, "blocked")
                    break

    def afk_step() -> None:
        nonlocal state, sphere, afk_index, afk_last_worker, afk_last_status, afk_cycles
        lookup = {cap.key: cap for cap in CAPABILITIES}
        key_name = AFK_WORKER_SEQUENCE[afk_index % len(AFK_WORKER_SEQUENCE)]
        afk_index += 1
        afk_last_worker = key_name
        afk_last_status = "running"
        cap = lookup.get(key_name)
        if not cap:
            afk_last_status = "missing"
            return
        if cap.destructive or not cap.safe_on_phone or cap.group == "external":
            afk_last_status = "skipped"
            remember("afk_skip", {"key": key_name, "reason": "unsafe for autonomous afk"}, "blocked")
            return
        payload = None
        if key_name == "dreamer_observe":
            payload = {"game": "glass_afk", "observation": {"mission": args.mission, "worker_index": afk_index}, "reward": 0.1, "done": False}
        elif key_name == "chat":
            payload = {"prompt": f"AFK worker interrogates mission: {args.mission}. Reply with one useful next learning cue.", "learn": True, "steps": 1}
        elif key_name == "teach":
            payload = {"text": f"afk worker cycle means bounded autonomous practice toward {args.mission}", "reward": 0.7, "steps": 1}
        elif key_name == "follow":
            payload = {"mode": "chain", "seed": args.mission, "rounds": 4, "steps": 1}
        elif key_name == "govern":
            payload = {"text": args.mission, "steps": 1}
        elif key_name == "relate":
            payload = {"source": "signal", "target": "reason", "steps": 1}
        elif key_name == "council":
            payload = {"prompt": f"AFK council: {args.mission}. Compare next safe practice step.", "learn": True, "steps": 1, "max_cocoons": args.afk_max_cocoons, "export_after": False}
        elif key_name == "quine":
            payload = {"limit": 6, "steps": 1}
        elif key_name == "sphere":
            payload = {"frames": args.afk_sphere_frames, "balls": 2, "misses": 4, "train": True}
        elif key_name == "save":
            payload = {"output_dir": "afk_glass_state"}
        try:
            data = run_capability(cap, payload)
            if key_name == "sphere":
                sphere = data
                state = data.get("state") or state
            afk_cycles += 1
            afk_last_status = "ok"
            display.show_action_feedback(f"AFK {key_name}", "Cocoon")
        except Exception as exc:
            afk_last_status = "blocked"
            remember(f"afk:{key_name}", {"error": str(exc), "key": key_name}, "blocked")

    prime_cache()
    print("[GLASS] Real pygame window running.")
    print("[GLASS] O ops, G games, F feed, V cycle views, K AFK workers, F11 fullscreen, Left/Right select, Enter runs, P prime, I study, N followup, E sequence, L logs, M map, S sphere, A autopilot, D harden, R refresh, Esc quit.")
    while running and display.is_running:
        tick += 1
        events = display.handle_events()
        key = events.get("hold_key")
        raw_key = events.get("key")
        if events.get("fullscreen"):
            display.toggle_fullscreen()
        button = events.get("button")
        if events.get("quit") or key == "ESC" or raw_key == pygame.K_q:
            break
        if args.max_seconds and time.time() - started >= args.max_seconds:
            break
        try:
            if isinstance(button, dict):
                if button.get("type") == "select_cap":
                    select_capability(str(button.get("key") or ""))
                elif button.get("type") == "mode":
                    view_mode = str(button.get("mode") or "ops")
                elif button.get("type") == "command":
                    run_button_command(str(button.get("command") or ""))
            elif raw_key in {pygame.K_RIGHT, pygame.K_DOWN, pygame.K_TAB}:
                selected_cap = (selected_cap + 1) % len(caps)
                display.show_action_feedback(caps[selected_cap].label, "Selected")
            elif raw_key in {pygame.K_LEFT, pygame.K_UP}:
                selected_cap = (selected_cap - 1) % len(caps)
                display.show_action_feedback(caps[selected_cap].label, "Selected")
            elif raw_key in {pygame.K_RETURN, pygame.K_KP_ENTER}:
                run_capability(caps[selected_cap])
            elif raw_key == pygame.K_p:
                prime_cache()
            elif raw_key == pygame.K_i:
                study_capability(caps[selected_cap])
            elif raw_key == pygame.K_n:
                run_followup()
            elif raw_key == pygame.K_e:
                run_sequence()
            elif raw_key == pygame.K_l:
                fetch_named("logs", "/api/native/training/logs?limit=80", timeout=5.0)
                display.show_action_feedback("Training logs", "Cocoon")
            elif raw_key == pygame.K_m:
                fetch_named("map", "/api/facility_map", timeout=5.0)
                display.show_action_feedback("Facility map", "Cocoon")
            elif raw_key == pygame.K_o:
                view_mode = "ops"
                display.show_action_feedback("Operations view", "Glass")
            elif raw_key == pygame.K_g:
                view_mode = "games"
                display.show_action_feedback("Games view", "Glass")
            elif raw_key == pygame.K_f:
                view_mode = "feed"
                display.show_action_feedback("Feed view", "Glass")
            elif raw_key == pygame.K_v:
                view_mode = {"ops": "games", "games": "feed", "feed": "ops"}.get(view_mode, "ops")
                display.show_action_feedback(f"{view_mode.title()} view", "Glass")
            elif raw_key == pygame.K_k:
                toggle_afk()
            if key == "FIRE" or raw_key == pygame.K_s:
                sphere = run_capability(next(cap for cap in CAPABILITIES if cap.key == "sphere"), {"frames": args.sphere_frames})
                state = sphere.get("state") or authority_json(args.authority, "/api/state")
                display.show_action_feedback("Sphere burst", "Cocoon")
            elif key == "H" or raw_key == pygame.K_r:
                state = fetch_named("state", "/api/state", timeout=5.0)
                display.show_action_feedback("Refresh", "Cocoon")
            elif raw_key == pygame.K_d:
                data = authority_json(args.authority, "/api/harden_facilities", timeout=20)
                state = data.get("state") or authority_json(args.authority, "/api/state")
                remember("harden", data, "ok")
                display.show_action_feedback("Harden", "Cocoon")
            elif key == "T" or raw_key == pygame.K_a:
                data = authority_json(
                    args.authority,
                    "/api/autopilot",
                    {"mission": args.mission, "depth": 1, "include_games": True, "save_after": False},
                    timeout=45,
                )
                state = data.get("state") or authority_json(args.authority, "/api/state")
                for stage in data.get("stages", []):
                    result = stage.get("result") if isinstance(stage, dict) else None
                    if isinstance(result, dict) and "collective_catches" in result:
                        sphere = result
                remember("autopilot", data, "ok")
                display.show_action_feedback("Autopilot", "Cocoon")
        except Exception as exc:
            remember("api_error", {"error": str(exc)}, "blocked")
            display.show_action_feedback(f"API error: {exc}", "Glass")

        now = time.time()
        if afk_mode and now - last_afk_step > max(3.0, args.afk_interval):
            try:
                afk_step()
            except Exception as exc:
                afk_last_status = "blocked"
                remember("afk_supervisor", {"error": str(exc), "worker": afk_last_worker}, "blocked")
            last_afk_step = now
        if now - last_poll > args.poll_seconds:
            try:
                state = authority_json(args.authority, "/api/state", timeout=3.0)
                cache["state"] = {"name": "state", "status": "ok", "time": time.time(), "keys": compact_keys(state), "data": state}
            except Exception:
                pass
            last_poll = now
        if now - last_wealth_poll > max(2.0, args.poll_seconds):
            try:
                wealth_surface = authority_json(args.authority, "/api/wealth_surface", timeout=5.0)
                cache["wealth"] = {"name": "wealth", "status": "ok", "time": time.time(), "keys": compact_keys(wealth_surface), "data": wealth_surface}
            except Exception:
                pass
            last_wealth_poll = now
        if args.auto_sphere and now - last_burst > args.auto_sphere:
            try:
                sphere = authority_json(
                    args.authority,
                    "/api/native/sphere_burst",
                    {"frames": args.sphere_frames, "balls": 2, "misses": 4, "train": False},
                    timeout=max(8.0, args.sphere_frames / 10),
                )
                state = sphere.get("state") or state
                remember("auto_sphere", sphere, "ok")
            except Exception:
                pass
            last_burst = now

        selected_overlay = overlay_for(caps[selected_cap], state, cache, last_result, last_status, sequence)
        display.set_cocoon_overlay(selected_overlay)
        display.set_glass_view_mode(view_mode)
        display.set_operational_surface(
            build_operational_surface(
                selected_overlay,
                caps[selected_cap].key,
                state,
                sphere,
                cache,
                history,
                wealth_surface,
                {
                    "enabled": afk_mode,
                    "worker": afk_last_worker,
                    "status": afk_last_status,
                    "cycles": afk_cycles,
                    "interval": args.afk_interval,
                    "sequence": list(AFK_WORKER_SEQUENCE),
                },
            )
        )
        frame = synth_frame(display.GAME_W, display.GAME_H, state, sphere, tick)
        obs = capability_obs(state, sphere, selected_cap, len(caps))
        probs = np.ones(len(caps), dtype=float) * 0.01
        probs[selected_cap] = 1.0
        probs = probs / probs.sum()
        chosen = selected_cap
        display.update_frame(frame, obs)
        display.update_info(
            game_name=f"{state.get('active_cocoon_name') or 'Cocoon'} :: {caps[selected_cap].group}",
            ai_action=caps[selected_cap].label,
            control_mode="PHONE",
        )
        display.update_decision(
            action_probs=probs,
            chosen_action=chosen,
            value=float(probs[chosen]),
            merkle_root=str(state.get("last_receipt") or sphere.get("receipt") or ""),
            reward=float(sphere.get("collective_catches", 0) or 0) - float(sphere.get("collective_misses", 0) or 0),
        )
        display.update_control(hold_mode="WATCHING", viewer_name="Phone", viewer_timer=999.0, viewer_actions=tick)
        if not display.render():
            break
        time.sleep(max(0.0, 1.0 / max(1, args.fps)))
    display.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Glass Box pygame phone UI against Cocoon Authority.")
    parser.add_argument("--authority", default=DEFAULT_AUTHORITY)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=780)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--poll-seconds", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--no-start-authority", dest="start_authority", action="store_false")
    parser.add_argument("--sphere-frames", type=int, default=90)
    parser.add_argument("--auto-sphere", type=float, default=0.0, help="Run a passive sphere burst every N seconds. 0 disables.")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="Exit after N seconds. Useful for smoke tests.")
    parser.add_argument("--mission", default="improve chat, games, verification, and semantic memory")
    parser.add_argument("--view", choices=("ops", "games", "feed"), default="ops", help="Initial pygame glass view.")
    parser.add_argument("--fullscreen", action="store_true", help="Start the pygame rendering surface fullscreen.")
    parser.add_argument("--afk", action="store_true", help="Start bounded Kleene-star AFK workers immediately.")
    parser.add_argument("--afk-interval", type=float, default=20.0, help="Seconds between terminating AFK worker cycles.")
    parser.add_argument("--afk-sphere-frames", type=int, default=60, help="Sphere frames per AFK game-grounding worker.")
    parser.add_argument("--afk-max-cocoons", type=int, default=6, help="Max cocoons consulted by AFK council worker.")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
