#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_AUTHORITY = "http://127.0.0.1:8765"


@dataclass(frozen=True)
class Capability:
    key: str
    group: str
    label: str
    method: str
    path: str
    payload: dict[str, Any]
    teaches: str
    safe_on_phone: bool = True
    deepen: bool = False
    mutates: bool = False
    destructive: bool = False
    timeout_class: str = "short"
    prerequisites: tuple[str, ...] = ()
    expected: tuple[str, ...] = ()
    followups: tuple[str, ...] = ()


CAPABILITIES: tuple[Capability, ...] = (
    Capability("health", "diagnostics", "Health", "GET", "/api/native/health", {}, "Runtime health and native readiness."),
    Capability("state", "diagnostics", "Live State", "GET", "/api/state", {}, "Current vocabulary, relations, cycles, receipts, brains."),
    Capability("capabilities", "diagnostics", "Capability Contract", "GET", "/capabilities", {}, "Machine-readable native capability contract."),
    Capability("facilities", "diagnostics", "Facility Inventory", "GET", "/api/native/facilities", {}, "Native methods, subsystems, endpoints, dependencies."),
    Capability("facility_manual", "diagnostics", "Facility Manual", "GET", "/api/facility_manual", {}, "All user-facing facility cards and protocols."),
    Capability("facility_map", "diagnostics", "Facility Map", "GET", "/api/facility_map", {}, "Graph of stations, routes, receipts, and events."),
    Capability("events", "diagnostics", "Event Ledger", "GET", "/api/events", {}, "Recent authority events."),
    Capability("lessons", "diagnostics", "Lesson Ledger", "GET", "/api/lessons", {}, "Recent training/teacher notes."),
    Capability("cocoons", "diagnostics", "Cocoon Scan", "GET", "/api/cocoons", {}, "Find local runnable cocoon files."),
    Capability("curriculum", "memory", "Curriculum", "GET", "/api/native/curriculum", {}, "Language curriculum and training sequence."),
    Capability("vocab", "memory", "Vocabulary", "GET", "/api/native/vocab?limit=160", {}, "Vocabulary awareness and word IDs."),
    Capability("logs", "memory", "Training Logs", "GET", "/api/native/training/logs?limit=80", {}, "Native learning audit trail."),
    Capability("snapshot", "memory", "Snapshot", "GET", "/api/native/snapshot?full=false", {}, "Symbolic/runtime memory snapshot."),
    Capability("save", "persistence", "Save Runtime", "POST", "/api/native/save", {"output_dir": "ui_native_state"}, "Durable runtime state files.", mutates=True),
    Capability("export", "persistence", "Export Cocoon", "POST", "/api/export", {"name": "facility_checkpoint.py"}, "Bake live learning into a checkpoint file.", mutates=True),
    Capability("reload", "persistence", "Reload Active", "POST", "/api/reload", {}, "Reload active or selected cocoon.", mutates=True),
    Capability("chat", "language", "Chat", "POST", "/api/native/chat", {"prompt": "hello cocoon", "learn": True, "steps": 1}, "Conversation, response selection, semantic reward, optional learning.", deepen=True, mutates=True),
    Capability("teach", "language", "Teach", "POST", "/api/native/teach", {"text": "signal means input that carries meaning", "reward": 0.8, "steps": 1}, "New words, token links, knowledge web, training logs.", deepen=True, mutates=True),
    Capability("follow", "language", "Follow Along", "POST", "/api/follow_along", {"mode": "cue", "seed": "signal meaning reason verify", "rounds": 6, "steps": 2}, "Echo/cue/chain/role/chant practice.", deepen=True, mutates=True),
    Capability("quine", "language", "Syntax Forge", "POST", "/api/quine_drill", {"limit": 8, "steps": 2}, "Fixed-point syntax and structural verification.", deepen=True, mutates=True),
    Capability("signal", "reasoning", "Signal Route", "POST", "/api/signal", {"text": "verify cause effect before response", "steps": 4}, "Faculty routing, anchor bridging, informational wealth.", deepen=True, mutates=True),
    Capability("govern", "reasoning", "Govern", "POST", "/api/govern", {"text": "verify cause effect before response", "steps": 2}, "Arbitration and verification habits.", deepen=True, mutates=True),
    Capability("relate", "reasoning", "Relate", "POST", "/api/relate", {"source": "signal", "target": "meaning", "steps": 3}, "Manual semantic relations and supervised pairs.", deepen=True, mutates=True),
    Capability("dreamer_observe", "reasoning", "Dreamer Observe", "POST", "/api/native/dreamer/observe", {"reward": 0.1, "note": "phone observation"}, "Record reward observations for proposal-making.", mutates=True),
    Capability("dreamer_propose", "reasoning", "Dreamer Propose", "POST", "/api/native/dreamer/propose", {}, "Generate next training proposals from recent observations."),
    Capability("perturb", "reasoning", "Perturb", "POST", "/api/perturb", {"mode": "stabilize", "intensity": 0.2, "steps": 4}, "Controlled stabilization/adaptation pressure.", mutates=True),
    Capability("autopilot", "compound", "Autopilot", "POST", "/api/autopilot", {"mission": "improve chat, tool reasoning, games, verification, and semantic memory", "depth": 1, "include_games": True, "save_after": False}, "Scout, drill, associate, verify, receipt, summarize.", deepen=True, mutates=True),
    Capability("tour", "compound", "Facility Tour", "POST", "/api/facility_tour", {}, "Tour across core facilities and records a receipt.", mutates=True),
    Capability("deepen", "compound", "Deepen Autopilot", "POST", "/api/deepen_facility", {"facility": "autopilot", "depth": 2}, "Staged deepening ladder for a facility.", deepen=True, mutates=True),
    Capability("feed", "compound", "Internet Scout", "POST", "/api/feed", {"url": "https://export.arxiv.org/rss/cs.AI", "max_items": 2, "steps": 2}, "Public feed scoring and lesson conversion.", safe_on_phone=False, deepen=True, mutates=True),
    Capability("council", "compound", "Butterfly Council", "POST", "/api/council", {"prompt": "hello cocoon council", "learn": True, "steps": 1, "max_cocoons": 8, "export_after": False}, "Fan one prompt out across discovered runnable cocoons, aggregate their replies, and optionally train each one.", deepen=True, mutates=True),
    Capability("act", "action", "Action Head", "POST", "/api/native/act", {"state": [], "explore": False}, "Action selection and exploration."),
    Capability("learn", "action", "Reward Learn", "POST", "/api/native/learn", {"state": [], "action": 0, "reward": 0.1, "next_state": [], "done": False}, "ExperienceBuffer and train_step.", mutates=True),
    Capability("score", "action", "Curriculum Score", "POST", "/api/native/curriculum/score", {"event_type": "phone_score", "score": {"reward": 0.5}}, "Outside coach scoring without using coach as speaker.", mutates=True),
    Capability("sphere", "games", "Sphere Burst", "POST", "/api/native/sphere_burst", {"frames": 120, "balls": 2, "misses": 4, "train": True}, "Action grounding, reward, catches/misses, replay buffer.", deepen=True, mutates=True),
    Capability("gym", "games", "Gym Burst", "POST", "/api/native/gym_burst", {"env": "CartPole-v1", "episodes": 1, "learn": True}, "Classic RL transitions and rewards.", deepen=True, mutates=True),
    Capability("game_recipe", "games", "Game Recipe", "GET", "/api/native/game_recipe?lane=all", {}, "How game lanes can be run."),
    Capability("drone", "external", "Drone Arena", "POST", "/api/run_facility", {"facility": "drone", "mode": "free_fly", "time": 5, "drones": 2, "timeout": 25}, "Robotics-style adapter and bounded run. Matplotlib, PyFlyt, pybullet, pettingzoo, and numba are requisites for the complete visual lane.", safe_on_phone=False, prerequisites=("matplotlib", "PyFlyt", "pybullet", "pettingzoo", "numba")),
    Capability(
        "tmrl_doctor",
        "external",
        "TMRL Doctor",
        "POST",
        "/api/run_facility",
        {"facility": "tmrl_doctor", "timeout": 25},
        "Operator-disabled historical TrackMania diagnostic. Do not pursue installation or autonomous use.",
        safe_on_phone=False,
        prerequisites=("operator_disabled",),
        expected=("disabled_by_operator", "recipe", "dependency_probe", "status", "error"),
        followups=("doctor", "facilities"),
    ),
    Capability("link", "external", "Relay Link", "POST", "/api/run_facility", {"facility": "link"}, "Networking recipe for relay-backed cocoon links.", safe_on_phone=False),
    Capability("cheat", "overclock", "Overclock", "POST", "/api/cheat", {"lr_boost": 5.0, "multiplier": 5, "steps": 128}, "High-pressure super-experience training.", deepen=True, mutates=True, destructive=True),
)


CAPABILITY_INDEX = {cap.key: cap for cap in CAPABILITIES}


TYPE_BY_VALUE = {
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
    list: "array",
    dict: "object",
}


DEFAULT_EXPECTED = {
    "diagnostics": ("status", "state", "keys", "dependency_probe"),
    "memory": ("vocab_size", "entries", "snapshot_time", "training_logs_tail"),
    "persistence": ("path", "checkpoint", "state", "receipt"),
    "language": ("response", "teacher_note", "losses", "state", "receipt"),
    "reasoning": ("arbitration", "teacher_note", "informational_wealth", "state", "receipt"),
    "compound": ("stages", "trainer_card", "teacher_note", "state", "receipt"),
    "action": ("action", "reward", "losses", "state_dim", "receipt"),
    "games": ("frames_run", "collective_catches", "collective_misses", "runner_result", "state", "receipt"),
    "external": ("recipe", "dependency_probe", "status", "error"),
    "overclock": ("losses", "state", "receipt"),
}


DEFAULT_FOLLOWUPS = {
    "diagnostics": ("agent", "snapshot"),
    "memory": ("chat", "govern"),
    "persistence": ("state", "cocoons"),
    "language": ("govern", "save"),
    "reasoning": ("relate", "save"),
    "compound": ("logs", "save", "export"),
    "action": ("sphere", "logs"),
    "games": ("snapshot", "save"),
    "external": ("doctor", "facilities"),
    "overclock": ("snapshot", "save", "export"),
}


def payload_schema(payload: dict[str, Any]) -> dict[str, Any]:
    properties = {}
    for key, value in payload.items():
        value_type = TYPE_BY_VALUE.get(type(value), "string")
        entry: dict[str, Any] = {"type": value_type, "default": value}
        if key in {"steps", "frames", "rounds", "episodes", "balls", "misses", "limit", "depth", "timeout"}:
            entry["minimum"] = 0
        if key in {"mode"}:
            entry["examples"] = ["cue", "echo", "chain", "role", "chant"]
        properties[key] = entry
    return {
        "type": "object",
        "additionalProperties": True,
        "properties": properties,
    }


def risk_class(cap: Capability) -> str:
    if cap.destructive:
        return "overclock"
    if not cap.safe_on_phone:
        return "external"
    if cap.mutates:
        return "mutating"
    return "inspect"


def timeout_for(cap: Capability) -> float:
    if cap.timeout_class == "long" or cap.key in {"autopilot", "deepen", "feed", "gym", "drone", "tmrl_doctor"}:
        return 90.0
    if cap.timeout_class == "medium" or cap.key in {"sphere", "tour", "cheat", "export"}:
        return 45.0
    return 15.0


def enriched_capability(cap: Capability) -> dict[str, Any]:
    data = asdict(cap)
    data["risk_class"] = risk_class(cap)
    data["payload_schema"] = payload_schema(cap.payload)
    data["expected_outputs"] = list(cap.expected or DEFAULT_EXPECTED.get(cap.group, ()))
    data["followups"] = list(cap.followups or DEFAULT_FOLLOWUPS.get(cap.group, ()))
    data["timeout_seconds"] = timeout_for(cap)
    data["command"] = f"./cocoon capabilities run {cap.key}"
    if cap.payload:
        data["command_with_payload"] = f"./cocoon capabilities run {cap.key} --payload '{json.dumps(cap.payload)}'"
    else:
        data["command_with_payload"] = data["command"]
    data["agent_affordance"] = {
        "when_to_use": cap.teaches,
        "before": ["health"] if cap.mutates else [],
        "after": data["followups"],
        "guard": "requires explicit operator intent" if cap.destructive else ("dependency-gated" if not cap.safe_on_phone else "phone-safe"),
    }
    return data


class CapabilityRouteError(RuntimeError):
    def __init__(self, error: dict[str, Any]):
        self.error = error
        super().__init__(error["message"])


def guided_error(base_url: str, cap: Capability | None, exc: Exception) -> dict[str, Any]:
    if isinstance(exc, urllib.error.HTTPError):
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:1200]
        except Exception:
            pass
        return {
            "ok": False,
            "kind": "http_error",
            "status": exc.code,
            "capability": cap.key if cap else None,
            "message": f"Authority returned HTTP {exc.code}.",
            "detail": body,
            "next": ["Check payload shape with ./cocoon capabilities list --json", "Run ./cocoon capabilities run health"],
        }
    if isinstance(exc, urllib.error.URLError) or isinstance(exc, ConnectionError):
        return {
            "ok": False,
            "kind": "authority_unreachable",
            "capability": cap.key if cap else None,
            "message": f"Authority API is not reachable at {base_url}.",
            "detail": str(exc),
            "next": ["Start it with ./cocoon app --host 127.0.0.1 --port 8765 --no-open", "Then retry the capability command."],
        }
    if isinstance(exc, socket.timeout) or isinstance(exc, TimeoutError):
        return {
            "ok": False,
            "kind": "timeout",
            "capability": cap.key if cap else None,
            "message": "The route timed out before returning.",
            "detail": str(exc),
            "next": ["Retry with a smaller depth/steps/frames payload.", "Inspect logs with tail -80 mira_kite_authority_runtime/authority_launch.log"],
        }
    return {
        "ok": False,
        "kind": "client_error",
        "capability": cap.key if cap else None,
        "message": str(exc),
        "detail": exc.__class__.__name__,
        "next": ["Run ./cocoon doctor", "Run ./cocoon capabilities run health"],
    }


def request_json(base_url: str, cap: Capability, payload_override: dict[str, Any] | None = None, timeout: float = 45.0) -> dict[str, Any]:
    url = base_url.rstrip("/") + cap.path
    payload = dict(cap.payload)
    if payload_override:
        payload.update(payload_override)
    try:
        if cap.method == "GET":
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        raw = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=raw, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise CapabilityRouteError(guided_error(base_url, cap, exc)) from exc


def capability_manifest() -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for cap in CAPABILITIES:
        groups.setdefault(cap.group, []).append(enriched_capability(cap))
    return {
        "count": len(CAPABILITIES),
        "groups": groups,
        "capabilities": [enriched_capability(cap) for cap in CAPABILITIES],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List and run Cocoon Authority capabilities.")
    parser.add_argument("command", choices=["list", "inspect", "run", "tour"], nargs="?", default="list")
    parser.add_argument("capability", nargs="?")
    parser.add_argument("--authority", default=DEFAULT_AUTHORITY)
    parser.add_argument("--payload", default="{}", help="JSON payload override for run.")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--include-mutating", action="store_true", help="Allow tour to execute mutating routes.")
    parser.add_argument("--include-external", action="store_true", help="Allow tour to try dependency-gated external routes.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "list":
        manifest = capability_manifest()
        if args.json:
            print(json.dumps(manifest, indent=2))
        else:
            for group, items in manifest["groups"].items():
                print(f"\n[{group}]")
                for item in items:
                    flag = "" if item["safe_on_phone"] else " (external/dependency-gated)"
                    print(f"  {item['key']:<18} {item['label']}{flag}")
        return 0

    if args.command == "inspect":
        manifest = capability_manifest()
        if not args.capability:
            print(json.dumps(manifest, indent=2))
            return 0
        cap = CAPABILITY_INDEX.get(args.capability)
        if not cap:
            parser.error(f"unknown capability: {args.capability}")
        print(json.dumps(enriched_capability(cap), indent=2))
        return 0

    if args.command == "tour":
        results = []
        for cap in CAPABILITIES:
            if not args.include_external and not cap.safe_on_phone:
                results.append({"capability": cap.key, "skipped": True, "reason": "external/dependency-gated"})
                continue
            if not args.include_mutating and cap.mutates:
                results.append({"capability": cap.key, "skipped": True, "reason": "mutating route; pass --include-mutating to execute"})
                continue
            try:
                data = request_json(args.authority, cap, timeout=min(args.timeout, 20.0))
                results.append({"capability": cap.key, "ok": True, "keys": sorted(data.keys()) if isinstance(data, dict) else []})
            except CapabilityRouteError as exc:
                results.append({"capability": cap.key, **exc.error})
            except Exception as exc:
                results.append({"capability": cap.key, **guided_error(args.authority, cap, exc)})
        print(json.dumps({"results": results}, indent=2))
        return 0

    if not args.capability:
        parser.error("run requires a capability key")
    cap = CAPABILITY_INDEX.get(args.capability)
    if not cap:
        parser.error(f"unknown capability: {args.capability}")
    payload = json.loads(args.payload or "{}")
    try:
        print(json.dumps(request_json(args.authority, cap, payload, timeout=args.timeout), indent=2))
        return 0
    except CapabilityRouteError as exc:
        print(json.dumps(exc.error, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
