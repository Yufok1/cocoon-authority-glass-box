#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any

from cocoon_capabilities import (
    CAPABILITIES,
    CAPABILITY_INDEX,
    CapabilityRouteError,
    capability_manifest,
    enriched_capability,
    guided_error,
    request_json,
)


DEFAULT_AUTHORITY = "http://127.0.0.1:8765"


def try_route(authority: str, key: str, payload: dict[str, Any] | None = None, timeout: float = 10.0) -> dict[str, Any]:
    cap = CAPABILITY_INDEX[key]
    try:
        return {"ok": True, "capability": key, "data": request_json(authority, cap, payload, timeout=timeout)}
    except CapabilityRouteError as exc:
        return {"ok": False, "capability": key, "error": exc.error}
    except Exception as exc:
        return {"ok": False, "capability": key, "error": guided_error(authority, cap, exc)}


def compact_state(state: dict[str, Any]) -> dict[str, Any]:
    if not state:
        return {}
    return {
        "active_cocoon_name": state.get("active_cocoon_name") or state.get("pet_name"),
        "active_cocoon_path": state.get("active_cocoon_path") or state.get("source_cocoon"),
        "mode": state.get("mode"),
        "cycles": state.get("cycles"),
        "vocabulary_size": state.get("vocabulary_size"),
        "knowledge_relations": state.get("knowledge_relations"),
        "conversation_turns": state.get("conversation_turns"),
        "training_logs": state.get("training_logs"),
        "last_receipt": state.get("last_receipt"),
        "last_checkpoint": state.get("last_checkpoint"),
    }


def suggested_routes(state: dict[str, Any], health_ok: bool) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    if not health_ok:
        return [
            {
                "capability": "health",
                "reason": "Authority is not healthy or not reachable.",
                "payload": {},
                "command": "./cocoon app --host 127.0.0.1 --port 8765 --no-open",
            }
        ]
    if not state.get("last_receipt"):
        suggestions.append({"capability": "snapshot", "reason": "Establish a readable baseline before mutating.", "payload": {}})
    suggestions.extend(
        [
            {
                "capability": "chat",
                "reason": "Low-cost conversational state probe with optional learning.",
                "payload": {"prompt": "summarize your current state and one useful next move", "learn": True, "steps": 1},
            },
            {
                "capability": "govern",
                "reason": "Exercises faculty routing and verification without a broad run.",
                "payload": {"text": "verify cause effect before response", "steps": 2},
            },
            {
                "capability": "sphere",
                "reason": "Ground action/reward feedback in a bounded game loop.",
                "payload": {"frames": 90, "balls": 2, "misses": 4, "train": True},
            },
            {
                "capability": "save",
                "reason": "Persist after useful learning, before larger experiments.",
                "payload": {"output_dir": "agent_console_save"},
            },
        ]
    )
    if int(state.get("cycles") or 0) >= 2:
        suggestions.append(
            {
                "capability": "autopilot",
                "reason": "System has enough baseline to run a bounded compound cycle.",
                "payload": {
                    "mission": "improve chat, reasoning, verification, games, and semantic memory",
                    "depth": 1,
                    "include_games": True,
                    "save_after": False,
                },
            }
        )
    return suggestions


def status(authority: str) -> dict[str, Any]:
    health = try_route(authority, "health")
    state_result = try_route(authority, "state")
    state = state_result.get("data") if state_result.get("ok") else {}
    return {
        "authority": authority,
        "health": health,
        "state": compact_state(state if isinstance(state, dict) else {}),
        "next": suggested_routes(state if isinstance(state, dict) else {}, bool(health.get("ok"))),
        "fabric": {
            "capability_count": len(CAPABILITIES),
            "groups": sorted({cap.group for cap in CAPABILITIES}),
            "safe_phone_count": sum(1 for cap in CAPABILITIES if cap.safe_on_phone),
            "mutating_count": sum(1 for cap in CAPABILITIES if cap.mutates),
            "external_count": sum(1 for cap in CAPABILITIES if not cap.safe_on_phone),
        },
    }


def brief(authority: str) -> str:
    data = status(authority)
    if not data["health"]["ok"]:
        err = data["health"].get("error", {})
        return "\n".join([
            "COCOON AGENT BRIEF",
            "Authority: offline or unhealthy",
            f"Reason: {err.get('message', 'unknown')}",
            "Next: ./cocoon app --host 127.0.0.1 --port 8765 --no-open",
        ])
    state = data["state"]
    lines = [
        "COCOON AGENT BRIEF",
        f"Active: {state.get('active_cocoon_name')} ({state.get('mode')})",
        f"Vocab: {state.get('vocabulary_size')} | Relations: {state.get('knowledge_relations')} | Cycles: {state.get('cycles')}",
        f"Receipt: {state.get('last_receipt') or 'none'}",
        "",
        "Suggested next routes:",
    ]
    for idx, item in enumerate(data["next"][:5], 1):
        lines.append(f"{idx}. {item['capability']}: {item['reason']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agent-facing Cocoon navigator.")
    parser.add_argument("command", choices=["status", "brief", "manifest", "next", "inspect"], nargs="?", default="brief")
    parser.add_argument("capability", nargs="?")
    parser.add_argument("--authority", default=DEFAULT_AUTHORITY)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "manifest":
        print(json.dumps(capability_manifest(), indent=2))
        return 0
    if args.command == "inspect":
        if not args.capability:
            print(json.dumps(capability_manifest(), indent=2))
            return 0
        cap = CAPABILITY_INDEX.get(args.capability)
        if not cap:
            print(json.dumps({"ok": False, "message": f"unknown capability: {args.capability}", "known": sorted(CAPABILITY_INDEX)}, indent=2))
            return 1
        print(json.dumps(enriched_capability(cap), indent=2))
        return 0
    if args.command in {"status", "next"}:
        data = status(args.authority)
        if args.command == "next":
            data = {"next": data["next"], "state": data["state"], "health_ok": data["health"]["ok"]}
        print(json.dumps(data, indent=2))
        return 0 if data.get("health", {}).get("ok", True) else 1
    output = brief(args.authority)
    if args.json:
        print(json.dumps(status(args.authority), indent=2))
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
