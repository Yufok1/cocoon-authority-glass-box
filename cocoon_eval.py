#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cocoon_ops import PACKAGE_DIR, build_command, doctor, missing_for_mode, resolve_within_root


DEFAULT_ROOT = PACKAGE_DIR.parent
DEFAULT_REPORT_DIR = PACKAGE_DIR / "eval_reports"
DEFAULT_COCOON = PACKAGE_DIR / "cocoon_cognition_agency.py"


@dataclass(frozen=True)
class EvalCase:
    name: str
    mode: str
    kwargs: dict[str, Any]
    timeout_seconds: float


CORE_CASES = (
    EvalCase("info", "info", {}, 30.0),
    EvalCase("readme", "readme", {}, 30.0),
    EvalCase("gym_cartpole_no_learn", "gym", {"env": "CartPole-v1", "episodes": 1, "train": False}, 120.0),
    EvalCase("sphere_headless_smoke", "sphere", {"headless": True, "train": True, "balls": 1, "misses": 1, "max_organisms": 1}, 120.0),
)


def short_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def tail(text: str, limit: int = 3000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def run_case(cocoon_path: Path, case: EvalCase) -> dict[str, Any]:
    missing = missing_for_mode(case.mode)
    result: dict[str, Any] = {
        "name": case.name,
        "mode": case.mode,
        "timeout_seconds": case.timeout_seconds,
        "missing": missing,
        "ready": not missing,
        "skipped": bool(missing),
    }
    if missing:
        result["passed"] = False
        result["reason"] = "missing_dependencies"
        return result

    command = build_command(cocoon_path, case.mode, **case.kwargs)
    started = time.time()
    result["command"] = command
    try:
        completed = subprocess.run(
            command,
            cwd=str(cocoon_path.parent),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=case.timeout_seconds,
            check=False,
        )
        output = completed.stdout or ""
        result.update(
            {
                "returncode": completed.returncode,
                "duration_seconds": round(time.time() - started, 3),
                "passed": completed.returncode == 0,
                "output_sha256_16": short_sha256(output),
                "output_tail": tail(output),
            }
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        result.update(
            {
                "returncode": None,
                "duration_seconds": round(time.time() - started, 3),
                "passed": False,
                "timed_out": True,
                "output_sha256_16": short_sha256(output),
                "output_tail": tail(output),
            }
        )
    return result


def observe_with_amalgam(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        from amalgam_agency.receipts import observe
    except Exception as exc:
        return {"recorded": False, "reason": f"amalgam_import_failed: {exc}"}
    try:
        receipt = observe("cocoon-eval", payload, sync=False)
    except Exception as exc:
        return {"recorded": False, "reason": f"amalgam_observe_failed: {exc}"}
    return {"recorded": True, "receipt": receipt}


def run_eval(
    cocoon_path: Path,
    *,
    root: Path = DEFAULT_ROOT,
    delay_ms: int = 50,
    max_failures: int = 1,
    no_receipt: bool = False,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> dict[str, Any]:
    cocoon_path = resolve_within_root(cocoon_path, root)
    report_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    report: dict[str, Any] = {
        "event_type": "cocoon_eval",
        "started_at": started,
        "cocoon_path": str(cocoon_path),
        "root": str(root.resolve()),
        "delay_ms": delay_ms,
        "max_failures": max_failures,
        "doctor": doctor(root),
        "cases": [],
    }

    failures = 0
    for case in CORE_CASES:
        case_result = run_case(cocoon_path, case)
        report["cases"].append(case_result)
        if not case_result.get("passed"):
            failures += 1
        if failures >= max_failures:
            break
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

    passed = [case for case in report["cases"] if case.get("passed")]
    skipped = [case for case in report["cases"] if case.get("skipped")]
    failed = [case for case in report["cases"] if not case.get("passed") and not case.get("skipped")]
    report.update(
        {
            "finished_at": time.time(),
            "duration_seconds": round(time.time() - started, 3),
            "passed": len(failed) == 0,
            "passed_cases": len(passed),
            "failed_cases": len(failed),
            "skipped_cases": len(skipped),
            "total_cases_run": len(report["cases"]),
        }
    )

    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(started))
    report_path = report_dir / f"cocoon_eval_{stamp}.json"
    report["report_path"] = str(report_path)
    if not no_receipt:
        receipt_payload = {
            "event_type": "cocoon_eval",
            "report_path": str(report_path),
            "passed": report["passed"],
            "passed_cases": report["passed_cases"],
            "failed_cases": report["failed_cases"],
            "skipped_cases": report["skipped_cases"],
            "cocoon_path": str(cocoon_path),
            "case_names": [case["name"] for case in report["cases"]],
        }
        report["amalgam"] = observe_with_amalgam(receipt_payload)

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run real Cocoon smoke evals and record receipt-capable reports.")
    parser.add_argument("--cocoon", default=str(DEFAULT_COCOON))
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--delay-ms", type=int, default=50)
    parser.add_argument("--max-failures", type=int, default=1)
    parser.add_argument("--no-receipt", action="store_true")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    args = parser.parse_args(argv)

    report = run_eval(
        Path(args.cocoon),
        root=Path(args.root),
        delay_ms=max(0, args.delay_ms),
        max_failures=max(1, args.max_failures),
        no_receipt=args.no_receipt,
        report_dir=Path(args.report_dir),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
