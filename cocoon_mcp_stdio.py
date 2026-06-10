#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

from cocoon_capabilities import CAPABILITY_INDEX, capability_manifest, request_json
from cocoon_eval import run_eval
from cocoon_ops import DEFAULT_ROOT, PACKAGE_DIR, JobManager, doctor, persistence_report, scan_cocoons


ROOT = DEFAULT_ROOT.resolve()
JOBS = JobManager(root=ROOT)


def content(data: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(data, indent=2, sort_keys=True)}]}


def tool_doctor(args: dict[str, Any]) -> Any:
    return doctor(Path(args.get("root", ROOT)))


def tool_list_cocoons(args: dict[str, Any]) -> Any:
    return scan_cocoons(Path(args.get("root", ROOT)))


def tool_persistence(args: dict[str, Any]) -> Any:
    return persistence_report()


def tool_start_job(args: dict[str, Any]) -> Any:
    return JOBS.start(**args)


def tool_stop_job(args: dict[str, Any]) -> Any:
    return JOBS.stop(str(args["job_id"]), timeout=float(args.get("timeout", 5.0)))


def tool_job_status(args: dict[str, Any]) -> Any:
    return JOBS.get(str(args["job_id"]))


def tool_job_logs(args: dict[str, Any]) -> Any:
    return JOBS.logs(str(args["job_id"]), tail_bytes=int(args.get("tail_bytes", 8192)))


def tool_eval(args: dict[str, Any]) -> Any:
    cocoon_path = args.get("cocoon_path", str(PACKAGE_DIR / "cocoon_cognition_agency.py"))
    return run_eval(
        Path(cocoon_path),
        root=Path(args.get("root", ROOT)),
        delay_ms=int(args.get("delay_ms", 50)),
        max_failures=int(args.get("max_failures", 1)),
        no_receipt=bool(args.get("no_receipt", False)),
    )


def tool_eval_all(args: dict[str, Any]) -> Any:
    root = Path(args.get("root", ROOT))
    max_cocoons = max(1, min(int(args.get("max_cocoons", 8)), 32))
    discovered = [item for item in scan_cocoons(root) if item.get("runnable")]
    results: list[dict[str, Any]] = []
    for item in discovered[:max_cocoons]:
        report = run_eval(
            Path(item["path"]),
            root=root,
            delay_ms=int(args.get("delay_ms", 50)),
            max_failures=int(args.get("max_failures", 1)),
            no_receipt=bool(args.get("no_receipt", False)),
        )
        results.append({"path": item["path"], "name": item.get("name"), "report": report})
    return {
        "root": str(root),
        "max_cocoons": max_cocoons,
        "discovered": len(discovered),
        "results": results,
        "passed": all(result["report"].get("passed", False) for result in results) if results else False,
    }


def tool_council(args: dict[str, Any]) -> Any:
    payload = args.get("payload") if isinstance(args.get("payload"), dict) else None
    return request_json(
        str(args.get("authority", "http://127.0.0.1:8765")),
        CAPABILITY_INDEX["council"],
        payload,
        timeout=float(args.get("timeout", 120.0)),
    )


def tool_capabilities(args: dict[str, Any]) -> Any:
    return capability_manifest()


def tool_run_capability(args: dict[str, Any]) -> Any:
    key = str(args["key"])
    if key not in CAPABILITY_INDEX:
        raise ValueError(f"unknown capability: {key}")
    try:
        return request_json(
            str(args.get("authority", "http://127.0.0.1:8765")),
            CAPABILITY_INDEX[key],
            args.get("payload") if isinstance(args.get("payload"), dict) else None,
            timeout=float(args.get("timeout", 45.0)),
        )
    except Exception as exc:
        error = getattr(exc, "error", None)
        return error if isinstance(error, dict) else {"ok": False, "message": str(exc), "capability": key}


TOOLS: dict[str, tuple[str, dict[str, Any], Callable[[dict[str, Any]], Any]]] = {
    "doctor": (
        "Report real dependencies, discovered cocoons, and command readiness.",
        {"type": "object", "properties": {"root": {"type": "string"}}},
        tool_doctor,
    ),
    "list_cocoons": (
        "List discovered local cocoon resources under the managed root.",
        {"type": "object", "properties": {"root": {"type": "string"}}},
        tool_list_cocoons,
    ),
    "persistence": (
        "Report real training-log schema, live-log path, and memory artifact state.",
        {"type": "object", "properties": {}},
        tool_persistence,
    ),
    "start_job": (
        "Start a real cocoon subprocess after dependency checks pass.",
        {
            "type": "object",
            "required": ["cocoon_path", "mode"],
            "properties": {
                "cocoon_path": {"type": "string"},
                "mode": {"type": "string"},
                "port": {"type": "integer"},
                "max_organisms": {"type": "integer"},
                "train": {"type": "boolean"},
                "headless": {"type": "boolean"},
                "balls": {"type": "integer"},
                "misses": {"type": "integer"},
                "env": {"type": "string"},
                "episodes": {"type": "integer"},
            },
        },
        tool_start_job,
    ),
    "stop_job": (
        "Stop a running cocoon subprocess.",
        {"type": "object", "required": ["job_id"], "properties": {"job_id": {"type": "string"}, "timeout": {"type": "number"}}},
        tool_stop_job,
    ),
    "job_status": (
        "Return status for one cocoon subprocess.",
        {"type": "object", "required": ["job_id"], "properties": {"job_id": {"type": "string"}}},
        tool_job_status,
    ),
    "job_logs": (
        "Return tail logs for one cocoon subprocess.",
        {
            "type": "object",
            "required": ["job_id"],
            "properties": {"job_id": {"type": "string"}, "tail_bytes": {"type": "integer"}},
        },
        tool_job_logs,
    ),
    "eval": (
        "Run real Cocoon smoke evals and return the receipt-capable report.",
        {
            "type": "object",
            "properties": {
                "cocoon_path": {"type": "string"},
                "root": {"type": "string"},
                "delay_ms": {"type": "integer"},
                "max_failures": {"type": "integer"},
                "no_receipt": {"type": "boolean"},
            },
        },
        tool_eval,
    ),
    "eval_all": (
        "Run smoke evals across discovered runnable cocoons under the managed root.",
        {
            "type": "object",
            "properties": {
                "root": {"type": "string"},
                "max_cocoons": {"type": "integer"},
                "delay_ms": {"type": "integer"},
                "max_failures": {"type": "integer"},
                "no_receipt": {"type": "boolean"},
            },
        },
        tool_eval_all,
    ),
    "council": (
        "Fan one prompt out across runnable cocoons and return the aggregate council reply.",
        {
            "type": "object",
            "properties": {
                "authority": {"type": "string"},
                "timeout": {"type": "number"},
                "payload": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "learn": {"type": "boolean"},
                        "steps": {"type": "integer"},
                        "max_cocoons": {"type": "integer"},
                        "export_after": {"type": "boolean"},
                    },
                },
            },
        },
        tool_council,
    ),
    "capabilities": (
        "List the full Cocoon Authority capability fabric grouped by purpose.",
        {"type": "object", "properties": {}},
        tool_capabilities,
    ),
    "run_capability": (
        "Run one named Cocoon Authority capability through the local HTTP API.",
        {
            "type": "object",
            "required": ["key"],
            "properties": {
                "key": {"type": "string"},
                "authority": {"type": "string"},
                "payload": {"type": "object"},
                "timeout": {"type": "number"},
            },
        },
        tool_run_capability,
    ),
}


def tools_list() -> list[dict[str, Any]]:
    return [
        {"name": name, "description": description, "inputSchema": schema}
        for name, (description, schema, _handler) in TOOLS.items()
    ]


def handle(request: dict[str, Any]) -> dict[str, Any] | None:
    method = request.get("method")
    request_id = request.get("id")
    try:
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "cocoon-mcp-stdio", "version": "0.1.0"},
                "capabilities": {"tools": {}},
            }
        elif method == "notifications/initialized":
            return None
        elif method == "tools/list":
            result = {"tools": tools_list()}
        elif method == "tools/call":
            params = request.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            if name not in TOOLS:
                raise ValueError(f"unknown tool: {name}")
            result = content(TOOLS[name][2](args))
        else:
            raise ValueError(f"unknown method: {method}")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except Exception as exc:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32000, "message": str(exc), "data": {"method": method}},
        }


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        response = handle(json.loads(line))
        if response is not None:
            print(json.dumps(response), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
