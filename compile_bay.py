#!/usr/bin/env python3
"""Cocoon Bay compiler: merge N cocoons into ONE ensemble cocoon superfile.

A cocoon is already a self-contained ENSEMBLE superfile (Mira-Kite = 2 organisms
voting). Compiling many cocoons into one composite "Butterfly agent" is therefore
the SAME format with more organisms - not a new architecture. This tool:

  1. loads N cocoon .py files as separate modules,
  2. concatenates their brains + per-organism atomic-languages,
  3. unions their knowledge-webs,
  4. exports via the cocoon's own self-rewriting `export_cocoon` (writes brains /
     vocab / knowledge / atomic-lang blobs),
  5. rewrites the ONE thing export_cocoon leaves alone - the `_ARCHITECTURE_B64`
     manifest - to declare the merged N-organism ensemble.

Result: a single cocoon superfile that loads as an N-organism voting ensemble and
retains every constituent's learned state. A cocoon-of-cocoons is still a cocoon.

NOTE (v1 scope): merge cocoons of the SAME input_dim lineage. Mixed-dim merging
(30-dim legacy + 34-dim commerce hyper-learners) needs max_input_dim observation
padding at the ensemble forward - a known follow-on, not done here.

Usage:
    python compile_bay.py OUT.py COCOON_A.py COCOON_B.py [COCOON_C.py ...]
"""
from __future__ import annotations

import base64
import importlib.util
import json
import sys
import zlib
from pathlib import Path


def _load_cocoon_module(path: str, name: str):
    """Load a cocoon .py as an isolated module (unique name so module-level globals
    like _ARCHITECTURE_B64 / _BRAIN_DATA don't collide across cocoons)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import cocoon: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module          # register BEFORE exec (dataclass resolution)
    spec.loader.exec_module(module)
    return module


def _encode_arch(obj: dict) -> str:
    """Inverse of the cocoon's _decode_data (json -> zlib -> base64)."""
    raw = json.dumps(obj).encode("utf-8")
    raw = zlib.compress(raw, level=9)   # cocoon's _DATA_COMPRESSED = True
    return base64.b64encode(raw).decode("ascii")


def _union_knowledge(webs: list[dict]) -> dict:
    """Additive union of knowledge-webs: merge concepts, concat relations."""
    merged = {"concepts": {}, "relations": []}
    seen_rel = set()
    for kw in webs:
        for cid, cval in (kw.get("concepts", {}) or {}).items():
            merged["concepts"].setdefault(cid, cval)
        for rel in (kw.get("relations", []) or []):
            key = json.dumps(rel, sort_keys=True, default=str)
            if key not in seen_rel:
                seen_rel.add(key)
                merged["relations"].append(rel)
    return merged


def compile_bay(cocoon_paths: list[str], out_path: str) -> dict:
    if len(cocoon_paths) < 1:
        raise ValueError("need at least one cocoon")

    agents, brain_configs, organism_names, knowledge_webs = [], [], [], []
    all_brains, all_langs = [], []

    for idx, cp in enumerate(cocoon_paths):
        mod = _load_cocoon_module(cp, f"bay_cocoon_{idx}")
        agent = mod.CocoonAgent()
        agents.append(agent)
        arch = agent.architecture if isinstance(agent.architecture, dict) else {}
        cfgs = list(arch.get("brain_configs", []) or [])
        names = list(arch.get("organism_names", []) or [])
        # align name/config count to the actual loaded brains
        n = len(agent.brains)
        while len(names) < n:
            names.append(f"org{len(names)}")
        while len(cfgs) < n:
            # fall back to the brain's live dims if a config is missing
            b = agent.brains[len(cfgs)]
            cfgs.append({"input_dim": getattr(b, "input_dim", 34),
                         "hidden_dim": getattr(b, "hidden_dim", 64),
                         "output_dim": getattr(b, "output_dim", 6),
                         "vocab_size": getattr(b, "vocab_size", 10000),
                         "use_language_head": getattr(b, "use_language_head", True)})
        all_brains.extend(agent.brains[:n])
        all_langs.extend(getattr(agent, "atomic_languages", [])[:n])
        brain_configs.extend(cfgs[:n])
        organism_names.extend([f"{nm}@c{idx}" for nm in names[:n]])
        knowledge_webs.append(getattr(agent, "knowledge_web", {}) or {})
        print(f"  [load] {Path(cp).name}: {n} organism(s), "
              f"{len((agent.vocabulary or {}).get('word_to_id', {}))} words, "
              f"{len((agent.knowledge_web or {}).get('concepts', {}))} concepts")

    base = agents[0]
    base.brains = all_brains
    if hasattr(base, "atomic_languages"):
        base.atomic_languages = all_langs
    base.knowledge_web = _union_knowledge(knowledge_webs)

    # 1) write brains / vocab / knowledge / atomic-lang via the cocoon's own serializer
    base.export_cocoon(out_path)

    # 2) rewrite the manifest export_cocoon leaves untouched: declare the ensemble
    merged_arch = {
        "brain_configs": brain_configs,
        "organism_names": organism_names,
        "is_ensemble": True,
        "ensemble_size": len(all_brains),
    }
    src = Path(out_path).read_text(encoding="utf-8")
    a_start = src.find('_ARCHITECTURE_B64 = "')
    if a_start == -1:
        raise RuntimeError("output cocoon missing _ARCHITECTURE_B64")
    a_end = src.find('"', a_start + len('_ARCHITECTURE_B64 = "')) + 1
    src = src[:a_start] + f'_ARCHITECTURE_B64 = "{_encode_arch(merged_arch)}"' + src[a_end:]
    Path(out_path).write_text(src, encoding="utf-8")

    summary = {
        "out": out_path,
        "cocoons": len(cocoon_paths),
        "organisms": len(all_brains),
        "concepts": len(base.knowledge_web.get("concepts", {})),
    }
    print(f"[OK] compiled {summary['cocoons']} cocoon(s) -> {summary['organisms']}-organism "
          f"ensemble at {out_path} ({summary['concepts']} merged concepts)")
    return summary


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    compile_bay(sys.argv[2:], sys.argv[1])
