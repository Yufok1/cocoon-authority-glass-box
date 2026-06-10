#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from cocoon_capabilities import CAPABILITIES, capability_manifest


PACKAGE_DIR = Path(__file__).resolve().parent


def main() -> int:
    manifest = capability_manifest()
    failures: list[dict[str, str]] = []
    for item in manifest["capabilities"]:
        for field in ("key", "group", "label", "payload_schema", "expected_outputs", "followups", "risk_class", "timeout_seconds", "agent_affordance"):
            if item.get(field) in (None, "", [], {}):
                failures.append({"capability": item.get("key", "unknown"), "missing": field})
        if item["key"] not in item.get("command", ""):
            failures.append({"capability": item["key"], "missing": "command_contains_key"})

    required_files = [
        "COCOON_OPERATOR_PLAYBOOK.md",
        "COCOON_SKILLS_COOKBOOK.md",
        "GLASS_BOX_PHONE.md",
        "cocoon_capabilities.py",
        "cocoon_agent_console.py",
        "cocoon_mcp_stdio.py",
        "cocoon_glass_phone.py",
        "glass_box_phone/display.py",
    ]
    missing_files = [name for name in required_files if not (PACKAGE_DIR / name).exists()]
    for name in missing_files:
        failures.append({"capability": "package", "missing": name})

    cocoon_text = (PACKAGE_DIR / "cocoon").read_text(encoding="utf-8")
    for command in ("agent", "capabilities", "playbook", "skills", "mcp", "glass"):
        if command not in cocoon_text:
            failures.append({"capability": "cli", "missing": command})

    result = {
        "ok": not failures,
        "capability_count": len(CAPABILITIES),
        "safe_phone_count": sum(1 for cap in CAPABILITIES if cap.safe_on_phone),
        "mutating_count": sum(1 for cap in CAPABILITIES if cap.mutates),
        "external_count": sum(1 for cap in CAPABILITIES if not cap.safe_on_phone),
        "failures": failures,
    }
    print(json.dumps(result, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
