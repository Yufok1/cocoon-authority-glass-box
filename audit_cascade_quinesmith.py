#!/usr/bin/env python3
"""Generate a durable Cascade + Quinesmith environment playbook."""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import inspect
import json
import pkgutil
import subprocess
from pathlib import Path


PACKAGES = {
    "cascade-lattice": "cascade",
    "quinesmith": "quinesmith",
}


def package_meta(dist_name: str) -> dict:
    md = metadata.metadata(dist_name)
    return {
        "distribution": dist_name,
        "version": metadata.version(dist_name),
        "summary": md.get("Summary"),
        "author": md.get("Author"),
        "license_expression": md.get("License-Expression") or md.get("License"),
        "requires_python": md.get("Requires-Python"),
        "project_urls": md.get_all("Project-URL") or [],
    }


def public_members(module_name: str, limit: int = 80) -> list[dict]:
    module = importlib.import_module(module_name)
    members = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        kind = "object"
        signature = None
        if inspect.isfunction(obj):
            kind = "function"
        elif inspect.isclass(obj):
            kind = "class"
        elif inspect.ismodule(obj):
            kind = "module"
        if kind in {"function", "class"}:
            try:
                signature = str(inspect.signature(obj))
            except Exception:
                signature = None
        members.append({"name": name, "kind": kind, "signature": signature})
        if len(members) >= limit:
            break
    return members


def module_files(package_name: str) -> list[str]:
    package = importlib.import_module(package_name)
    root = Path(package.__file__).parent
    return [str(path) for path in sorted(root.rglob("*.py"))]


def submodules(package_name: str) -> list[str]:
    package = importlib.import_module(package_name)
    root = Path(package.__file__).parent
    names = []
    for mod in pkgutil.walk_packages([str(root)], prefix=f"{package_name}."):
        names.append(mod.name)
    return names


def command_output(command: list[str]) -> str:
    try:
        proc = subprocess.run(command, text=True, capture_output=True, check=False, timeout=20)
        return (proc.stdout or proc.stderr).strip()
    except Exception as exc:
        return f"COMMAND_FAILED: {exc}"


def build_audit() -> dict:
    audit = {
        "purpose": "Global environment playbook for Mira-Kite cocoon training.",
        "packages": {},
        "integration_recipe": {
            "cascade_lattice": [
                "Use cascade.store.observe(model_id, data, sync=False) for every training cycle.",
                "Use cascade query/list/stats to inspect receipts and chain continuity.",
                "Use HOLD semantics to label the 6 action heads as observable decision faculties.",
                "Do not require HOLD blocking for autonomous runtime; store the decision matrix as receipt data.",
            ],
            "quinesmith": [
                "Use Level and braces_required as fixed-point syntax drills.",
                "Use Transform/template_write_for as safe template formulation drills.",
                "Use verify.prepare_commands only to prepare operator commands; do not auto-run quine verification.",
                "Map quine nesting to Mira-Kite fixed-point ladders: fixed -> template -> emit -> verify.",
            ],
            "mira_kite": [
                "Train from stable known anchors before introducing novel terms.",
                "Bridge unknown terms into atomic language and knowledge_web relations.",
                "Use predictable low-ID words for neural language targets; keep high-ID terms symbolic.",
                "Action head labels: 0 observe, 1 associate, 2 recall, 3 reason, 4 respond, 5 verify.",
                "Export checkpoint on interval and on interrupt.",
            ],
        },
    }
    for dist_name, import_name in PACKAGES.items():
        package = importlib.import_module(import_name)
        info = package_meta(dist_name)
        info["import_name"] = import_name
        info["package_file"] = package.__file__
        info["module_files"] = module_files(import_name)
        info["submodules"] = submodules(import_name)
        info["public_members"] = public_members(import_name)
        audit["packages"][dist_name] = info

    audit["cascade_store_api"] = public_members("cascade.store", limit=80)
    audit["cascade_hold_api"] = public_members("cascade.hold", limit=80)
    audit["quinesmith_levels_api"] = public_members("quinesmith.levels", limit=80)
    audit["quinesmith_transforms_api"] = public_members("quinesmith.transforms", limit=80)
    audit["quinesmith_verify_api"] = public_members("quinesmith.verify", limit=80)
    audit["cascade_cli_help"] = command_output(["cascade", "--help"])
    return audit


def render_markdown(audit: dict) -> str:
    lines = [
        "# Cascade + Quinesmith Global Playbook",
        "",
        "This file is the environment-level memory for using `cascade-lattice` and `quinesmith` with Mira-Kite.",
        "",
        "## Package Facts",
    ]
    for dist, info in audit["packages"].items():
        lines.extend([
            f"### {dist}",
            f"- Import: `{info['import_name']}`",
            f"- Version: `{info['version']}`",
            f"- Summary: {info.get('summary')}",
            f"- Python: `{info.get('requires_python')}`",
            f"- License: `{info.get('license_expression')}`",
            f"- Source path: `{info.get('package_file')}`",
            f"- Python files audited: `{len(info.get('module_files', []))}`",
            "",
        ])

    lines.extend([
        "## Cascade-Lattice Recipe",
        "",
        "- Treat Cascade as the provenance and observability lattice.",
        "- Receipt primitive: `cascade.store.observe(model_id, data, parent_cid=None, sync=True)`.",
        "- Query primitive: `cascade.store.query(model_id=None, since=None, limit=100, include_remote=False)`.",
        "- CLI primitives: `cascade stats`, `cascade list`, `cascade chains`, `cascade inspect`, `cascade ingest`, `cascade analyze`, `cascade hold-status`.",
        "- HOLD is a decision freeze-frame: action probabilities, value, observation, brain id, labels, latent/attention/features/reasoning can be serialized.",
        "- For Mira-Kite, store each training cycle as a receipt with cycle number, sources, anchors, novel terms, losses, relation count, and checkpoint path.",
        "- Do not make indefinite runs invisible; every cycle should write `runtime_status.json` and a Cascade receipt when available.",
        "",
        "## Quinesmith Recipe",
        "",
        "- Treat Quinesmith as syntax formulation and fixed-point discipline.",
        "- `braces_required(level)` implements the doubling rule: host=1, template=2, nested=4, level n=2**n.",
        "- `template_write_for(transform)` gives canonical Level-1 template strings for transforms such as braces, variable references, dict literals, newline escapes, and f-string nesting.",
        "- `verify.prepare_commands(path)` only prepares commands for a human operator. The package intentionally has no execution verifier.",
        "- For Mira-Kite, turn syntax primitives into drills: `fixed -> level -> brace -> count -> stable`, `template -> level -> reason -> brace`, `nested -> template -> fixed -> point`.",
        "- Preserve the safety boundary: never silently compile, emit, diff, or run quine verification as an autonomous background step.",
        "",
        "## Mira-Kite Runtime Recipe",
        "",
        "1. Start from a trained cocoon checkpoint, currently `cocoon_cognition_agency.py`.",
        "2. Fetch a small bounded feed slice or use local text.",
        "3. Extract words and choose known fixed-point anchors: `know`, `remember`, `understand`, `reason`, `cause`, `effect`, `trace`, `sequence`, `signal`, `meaning`, `request`, `respond`, `provide`.",
        "4. Add novel words as symbolic bridge nodes, not forced language-head targets.",
        "5. Train predictable low-ID anchor words as neural targets.",
        "6. Add atomic associations and `knowledge_web` relations for every bridge.",
        "7. Route the 6 action heads as cognition faculties:",
        "   - `0 observe`",
        "   - `1 associate`",
        "   - `2 recall`",
        "   - `3 reason`",
        "   - `4 respond`",
        "   - `5 verify`",
        "8. Record Cascade receipts.",
        "9. Export on interval and on interrupt.",
        "",
        "## Files",
        "",
        "- Indefinite trainer: `mira_kite_infinite_trainer.py`",
        "- Warm-start trainer: `semantic_cocoon_trainer.py`",
        "- Trained cocoon: `cocoon_cognition_agency.py`",
        "- Package: `cocoon_cognition_agency_package/`",
        "- This memory: `.codex/global_memory/CASCADE_QUINESMITH_PLAYBOOK.md`",
        "- JSON audit: `.codex/global_memory/cascade_quinesmith_audit.json`",
        "",
        "## Public API Snapshot",
        "",
        "### cascade.store",
    ])
    for item in audit["cascade_store_api"]:
        sig = item.get("signature") or ""
        lines.append(f"- `{item['name']}{sig}` ({item['kind']})")
    lines.append("")
    lines.append("### cascade.hold")
    for item in audit["cascade_hold_api"]:
        sig = item.get("signature") or ""
        lines.append(f"- `{item['name']}{sig}` ({item['kind']})")
    lines.append("")
    lines.append("### quinesmith")
    for section in ["quinesmith_levels_api", "quinesmith_transforms_api", "quinesmith_verify_api"]:
        lines.append(f"#### {section}")
        for item in audit[section]:
            sig = item.get("signature") or ""
            lines.append(f"- `{item['name']}{sig}` ({item['kind']})")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    root = Path("/data/data/com.termux/files/home/downloads")
    memory_dir = root / ".codex" / "global_memory"
    package_dir = root / "cocoon_cognition_agency_package"
    memory_dir.mkdir(parents=True, exist_ok=True)
    package_dir.mkdir(exist_ok=True)

    audit = build_audit()
    audit_json = json.dumps(audit, indent=2, default=str)
    markdown = render_markdown(audit)

    targets = [
        memory_dir / "cascade_quinesmith_audit.json",
        package_dir / "cascade_quinesmith_audit.json",
    ]
    for target in targets:
        target.write_text(audit_json, encoding="utf-8")

    md_targets = [
        memory_dir / "CASCADE_QUINESMITH_PLAYBOOK.md",
        package_dir / "CASCADE_QUINESMITH_PLAYBOOK.md",
        root / "GLOBAL_ENVIRONMENT_MEMORY.md",
    ]
    for target in md_targets:
        target.write_text(markdown, encoding="utf-8")

    print(json.dumps({
        "memory_dir": str(memory_dir),
        "markdown": [str(p) for p in md_targets],
        "json": [str(p) for p in targets],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
