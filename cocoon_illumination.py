#!/usr/bin/env python3
"""Cocoon Illumination Engine - grant a compiled cocoon tiered access to its OWN
causal history, ported from the Butterfly alliance_warfare illumination engine.

Lineage: causation engine -> cascade-lattice (distilled from it) -> the cocoon's
embedded cascade receipts. So a cocoon's record is already causation-grade data;
illumination just reads it, gated by civilization tier. And the tier EMERGES from
compilation - the more distinct source cocoons a bay-compiled megafile fuses, the
higher its civilization, so the bay manufactures hegemonies that earn omniscience.

Tiers (the real ladder, from alliance_warfare.check_and_grant_illumination):
    isolated -> basic -> alliance -> confederation -> empire -> hegemony
Capabilities scale with tier:
    can_see_self (basic+), can_see_alliance (alliance+), can_see_confederation,
    can_see_root_causes + can_predict_impact (empire+), can_see_all (hegemony).

The causation explorer the engine queries is adapted over the cocoon's cascade
record: training_logs (events / receipts) + knowledge_web relations (causal links).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# ── capability ladder (verbatim semantics from the Butterfly illumination engine) ──
_TIER_ORDER = ["none", "basic", "alliance", "confederation", "empire", "hegemony"]
_TIER_NAMES = {
    "none": "Isolated", "basic": "Basic Causation", "alliance": "Alliance Insight",
    "confederation": "Confederation Vision", "empire": "Imperial Foresight",
    "hegemony": "Hegemonic Omniscience",
}


def _capabilities_for(level: str) -> Dict[str, bool]:
    rank = _TIER_ORDER.index(level)
    return {
        "can_see_self": rank >= _TIER_ORDER.index("basic"),
        "can_see_alliance": rank >= _TIER_ORDER.index("alliance"),
        "can_see_confederation": rank >= _TIER_ORDER.index("confederation"),
        "can_see_root_causes": rank >= _TIER_ORDER.index("empire"),
        "can_predict_impact": rank >= _TIER_ORDER.index("empire"),
        "can_see_all": rank >= _TIER_ORDER.index("hegemony"),
    }


class CocoonCausationRecord:
    """Adapts a compiled cocoon's cascade record into the causation-explorer surface
    the illumination engine expects (get_timeline / get_events_by_component /
    get_most_consequential / find_root_causes / analyze_impact)."""

    def __init__(self, agent: Any):
        self.logs: List[dict] = list(getattr(agent, "training_logs", []) or [])
        kw = getattr(agent, "knowledge_web", {}) or {}
        self.relations: List[dict] = list(kw.get("relations", []) or [])
        self.concepts: Dict[str, Any] = dict(kw.get("concepts", {}) or {})

    def get_timeline(self) -> List[dict]:
        return self.logs

    def get_events_by_component(self, component: str) -> List[dict]:
        c = component.replace("organism_", "")
        return [e for e in self.logs
                if c in str(e.get("organism", "")) or c in str(e.get("stage", ""))]

    def get_most_consequential(self, limit: int = 20) -> List[dict]:
        # most consequential = highest-reward receipts + highest-degree concepts
        ranked = sorted(self.logs, key=lambda e: float(e.get("reward", 0) or 0), reverse=True)
        degree: Dict[str, int] = {}
        for r in self.relations:
            for k in ("source", "target"):
                degree[str(r.get(k))] = degree.get(str(r.get(k)), 0) + 1
        top_concepts = sorted(degree.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        return {"top_receipts": ranked[:limit],
                "top_concepts": [{"concept": c, "degree": d} for c, d in top_concepts]}

    def find_root_causes(self, target: str, depth: int = 3) -> dict:
        # trace relations backward (target <- source chains)
        frontier, seen, chain = {target}, set(), []
        for _ in range(depth):
            nxt = set()
            for r in self.relations:
                if str(r.get("target")) in frontier and str(r.get("source")) not in seen:
                    chain.append({"cause": r.get("source"), "effect": r.get("target"),
                                  "strength": r.get("strength")})
                    nxt.add(str(r.get("source"))); seen.add(str(r.get("source")))
            frontier = nxt
            if not frontier:
                break
        return {"target": target, "root_cause_chain": chain, "roots": list(seen)}

    def analyze_impact(self, source: str, depth: int = 3) -> dict:
        frontier, seen, chain = {source}, set(), []
        for _ in range(depth):
            nxt = set()
            for r in self.relations:
                if str(r.get("source")) in frontier and str(r.get("target")) not in seen:
                    chain.append({"cause": r.get("source"), "effect": r.get("target"),
                                  "strength": r.get("strength")})
                    nxt.add(str(r.get("target"))); seen.add(str(r.get("target")))
            frontier = nxt
            if not frontier:
                break
        return {"source": source, "impact_chain": chain, "affected": list(seen)}


def cocoon_civilization_tier(agent: Any) -> str:
    """The tier a compiled cocoon earns - EMERGES from compilation. Distinct source
    cocoons (the @cN provenance tags compile_bay stamps) raise the civilization:
    a bay megafile fused from many runs IS a hegemony by construction."""
    arch = getattr(agent, "architecture", {}) or {}
    names = arch.get("organism_names", []) or getattr(agent, "organism_names", []) or []
    sources = {m.group(1) for n in names if (m := re.search(r"@c(\d+)$", str(n)))}
    n_sources = len(sources)
    n_orgs = len(getattr(agent, "brains", []) or names)
    if n_sources >= 2:                 # fused from 2+ runs = a hegemony of hegemonies
        return "hegemony"
    if n_orgs >= 4:
        return "empire"
    if n_orgs >= 3:
        return "confederation"
    if n_orgs >= 2:
        return "alliance"
    if n_orgs >= 1:
        return "basic"
    return "none"


class CocoonIllumination:
    """Tiered causal self-knowledge for a compiled cocoon - the illumination engine,
    native to the cocoon's own cascade record."""

    def __init__(self, agent: Any, force_tier: Optional[str] = None):
        self.agent = agent
        self.record = CocoonCausationRecord(agent)
        self.level = force_tier or cocoon_civilization_tier(agent)
        self.caps = _capabilities_for(self.level)

    def status(self) -> Dict[str, Any]:
        return {"level": self.level, "level_name": _TIER_NAMES[self.level],
                "organisms": len(getattr(self.agent, "brains", []) or []),
                "receipts": len(self.record.logs), "relations": len(self.record.relations),
                **self.caps}

    def query(self, query_type: str, target: Optional[str] = None) -> Dict[str, Any]:
        # access control - identical ladder to the Butterfly engine
        gate = {"self": "can_see_self", "alliance": "can_see_alliance",
                "root_causes": "can_see_root_causes", "impact": "can_see_root_causes",
                "all": "can_see_all"}
        need = gate.get(query_type)
        if need and not self.caps.get(need):
            return {"error": "Insufficient Illumination", "required": need, "have": self.level}
        if query_type == "self":
            comp = target or (getattr(self.agent, "organism_names", []) or ["cocoon"])[0]
            ev = self.record.get_events_by_component(comp)
            return {"authorized": True, "level": self.level, "events": ev[:50], "total": len(ev)}
        if query_type == "root_causes":
            return {"authorized": True, "level": self.level, **self.record.find_root_causes(target or "")}
        if query_type == "impact":
            return {"authorized": True, "level": self.level, **self.record.analyze_impact(target or "")}
        if query_type == "all":
            return {"authorized": True, "level": "hegemony", "level_name": "Hegemonic Omniscience",
                    "timeline_events": len(self.record.get_timeline()),
                    "most_consequential": self.record.get_most_consequential(20),
                    "message": "Full causation access - the megafile sees its whole history"}
        return {"error": f"unknown query_type: {query_type}"}


def _load_cocoon(path: str, name: str = "illum_cocoon"):
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("usage: python cocoon_illumination.py COCOON.py [query_type] [target]")
        raise SystemExit(2)
    mod = _load_cocoon(sys.argv[1])
    agent = mod.CocoonAgent()
    illum = CocoonIllumination(agent)
    print("=== ILLUMINATION STATUS ===")
    print(json.dumps(illum.status(), indent=2, default=str))
    qt = sys.argv[2] if len(sys.argv) > 2 else "all"
    tgt = sys.argv[3] if len(sys.argv) > 3 else None
    print(f"=== QUERY: {qt} ===")
    res = illum.query(qt, tgt)
    # trim the most_consequential for readable demo output
    if isinstance(res.get("most_consequential"), dict):
        mc = res["most_consequential"]
        res["most_consequential"] = {"top_receipts": len(mc.get("top_receipts", [])),
                                     "top_concepts": mc.get("top_concepts", [])[:5]}
    print(json.dumps(res, indent=2, default=str)[:1500])
