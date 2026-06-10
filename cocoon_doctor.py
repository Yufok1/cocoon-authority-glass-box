#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from cocoon_ops import doctor


def main() -> int:
    parser = argparse.ArgumentParser(description="Report real Cocoon prerequisites and local resources.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = doctor(Path(args.root))
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("COCOON DOCTOR")
    print("=" * 60)
    for capability, data in report["dependencies"]["capabilities"].items():
        status = "READY" if data["ready"] else "MISSING " + ", ".join(data["missing"])
        tier = "core" if data.get("required_for_core") else "faculty"
        print(f"{capability:16s} {tier:7s} {status}")
    print()
    core_status = "READY" if report["dependencies"]["core_ready"] else "MISSING " + ", ".join(report["dependencies"]["core_missing"])
    print(f"Core runtime: {core_status}")
    print()
    print("Persistence:")
    persistence = report["persistence"]
    print(f"  schema           {'READY' if persistence['schema']['exists'] else 'MISSING'}")
    print(f"  live_log         {'READY' if persistence['default_log']['exists'] else 'MISSING'}  {persistence['default_log']['path']}")
    print(f"  memory           {'READY' if persistence['environment_memory']['exists'] else 'MISSING'}")
    print(f"  event_types      {len(persistence['event_types'])}")
    print()
    print("Discovered cocoons:")
    for item in report["cocoons"]:
        print(f"  {item['kind']:14s} {item['size_mb']:8.1f} MB  {item['path']}")
    print()
    print("Real commands:")
    for name, data in report["commands"].items():
        status = "READY" if data["ready"] else "BLOCKED missing " + ", ".join(data["missing"])
        print(f"  {name:16s} {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
