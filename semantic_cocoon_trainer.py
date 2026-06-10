#!/usr/bin/env python3
"""Focused semantic warm-start trainer for a standalone cocoon export."""

from __future__ import annotations

import argparse
import importlib.util
import os
import time
from typing import Iterable

import numpy as np


CURRICULUM = [
    ("greeting hello hi", "respond help now"),
    ("can you help", "provide help"),
    ("are you running", "say live here now"),
    ("what mode are you in", "summarize request"),
    ("unknown fact question", "say not know ask"),
    ("missing detail", "ask missing"),
    ("answer clearly", "reply precise"),
    ("avoid nonsense", "respond coherent"),
    ("repeat problem", "reply focused"),
    ("user request", "understand request"),
    ("please summarize", "summarize request"),
    ("explain what you know", "tell what know"),
    ("train semantic chat", "learn remember understand"),
    ("be useful", "assist provide support"),
    ("be calm", "calm steady"),
    ("make a plan", "advise sequence"),
    ("check status", "say current state"),
    ("direct answer", "reply direct"),
    ("need context", "remember request"),
    ("uncertain answer", "say not know"),
    ("normal chat", "respond help now"),
    ("good response", "accurate focused reply"),
    ("semantic meaning", "understand meaning"),
    ("save learning", "remember learn"),
]


SEMANTIC_BRIDGES = {
    "greeting": ["respond", "help", "here", "now", "social"],
    "hello": ["respond", "social", "connected"],
    "status": ["current", "state", "live", "here", "now"],
    "chat": ["respond", "reply", "tell", "listen", "social"],
    "normal": ["coherent", "stable", "balanced", "directed"],
    "semantic": ["meaning", "understand", "knowledge", "context"],
    "association": ["linked", "connected", "networked", "relation"],
    "associate": ["link", "connect", "unite", "integrated"],
    "word": ["signal", "concept", "represent", "meaning"],
    "words": ["signal", "concept", "represent", "meaning"],
    "known": ["know", "remember", "understand", "knowledgeable"],
    "learn": ["adapt", "grow", "remember", "understand"],
    "training": ["learn", "adapt", "strengthen", "stabilize"],
    "horizon": ["expand", "explore", "discover", "beyond"],
    "horizons": ["expand", "explore", "discover", "beyond"],
    "hopfield": ["remember", "pattern", "stabilize", "consolidate"],
    "registry": ["memory", "trace", "track", "knowledge"],
    "registries": ["memory", "trace", "track", "knowledge"],
    "cocoon": ["agent", "learn", "respond", "adapt"],
    "agent": ["act", "respond", "learn", "assist"],
    "agents": ["act", "respond", "learn", "assist"],
    "experiment": ["explore", "test", "learn", "discover"],
    "save": ["remember", "preserve", "sustain", "maintain"],
    "export": ["provide", "present", "preserve", "package"],
    "packet": ["message", "manifest", "trace", "receipt"],
    "resonance": ["signal", "rhythm", "connected", "coherent"],
    "meaning": ["understand", "represent", "context", "knowledge"],
    "context": ["remember", "surrounding", "relation", "request"],
    "request": ["respond", "provide", "assist", "reply"],
    "uncertain": ["not", "know", "ask", "careful"],
    "missing": ["ask", "find", "provide", "detail"],
}


REASONING_BRIDGES = {
    "reason": ["cause", "effect", "infer", "explain"],
    "reasoning": ["reason", "sequence", "trace", "explain"],
    "cause": ["effect", "trace", "explain"],
    "effect": ["cause", "reason", "infer"],
    "infer": ["reason", "understand", "explain"],
    "fixed": ["stable", "focused", "coherent"],
    "point": ["center", "stable", "focused"],
    "points": ["center", "sequence", "stable"],
    "chain": ["sequence", "trace", "reason"],
    "evidence": ["trace", "signal", "reason"],
    "verify": ["check", "trace", "precise"],
    "check": ["scan", "trace", "current"],
}


FIXED_POINT_LADDERS = [
    ["know", "remember", "understand", "explain"],
    ["signal", "meaning", "understand", "respond"],
    ["request", "respond", "provide", "support"],
    ["missing", "ask", "find", "provide"],
    ["cause", "effect", "reason", "explain"],
    ["trace", "sequence", "reason", "infer"],
    ["stable", "focused", "coherent", "precise"],
    ["help", "support", "together", "connected"],
    ["learn", "adapt", "strengthen", "remember"],
    ["current", "trace", "summarize", "reply"],
]


REASONING_FIXED_POINTS = [
    ("know", "know"),
    ("remember", "remember"),
    ("understand", "understand"),
    ("reason", "reason"),
    ("cause effect", "cause effect"),
    ("request respond", "request respond"),
    ("missing ask", "missing ask"),
    ("signal meaning", "signal meaning"),
    ("trace sequence", "trace sequence"),
    ("stable coherent", "stable coherent"),
    ("know remember", "understand"),
    ("remember understand", "explain"),
    ("cause effect", "reason"),
    ("effect reason", "explain"),
    ("request", "respond provide"),
    ("missing", "ask provide"),
    ("signal", "meaning understand"),
    ("trace", "sequence reason"),
    ("fixed", "stable focused"),
    ("chain", "sequence reason"),
]


COGNITIVE_FACULTIES = {
    0: {
        "name": "observe",
        "triggers": ["current", "state", "signal", "scan", "trace"],
        "target": "scan signal",
    },
    1: {
        "name": "associate",
        "triggers": ["associate", "relation", "connected", "meaning", "context"],
        "target": "connect meaning",
    },
    2: {
        "name": "recall",
        "triggers": ["remember", "know", "history", "trace", "registry"],
        "target": "remember know",
    },
    3: {
        "name": "reason",
        "triggers": ["reason", "cause", "effect", "infer", "chain"],
        "target": "cause effect reason",
    },
    4: {
        "name": "respond",
        "triggers": ["request", "help", "reply", "respond", "provide"],
        "target": "respond provide",
    },
    5: {
        "name": "verify",
        "triggers": ["check", "verify", "evidence", "precise", "uncertain"],
        "target": "trace precise",
    },
}


def load_cocoon(path: str):
    spec = importlib.util.spec_from_file_location("loaded_cocoon", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import cocoon from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean_words(text: str) -> list[str]:
    words = []
    for word in text.lower().split():
        clean = "".join(ch for ch in word if ch.isalnum())
        if clean:
            words.append(clean)
    return words


def ensure_words(agent, text: str) -> list[int]:
    return [agent.add_word(word) for word in clean_words(text)]


def encode_state(agent, prompt_tokens: Iterable[int], prefix_tokens: Iterable[int]) -> np.ndarray:
    brain = agent.brains[0]
    state = np.zeros(brain.input_dim, dtype=np.float32)
    context = list(prompt_tokens) + list(prefix_tokens)
    for idx, token in enumerate(context[-brain.input_dim :]):
        state[idx] = float(token) / 1000.0
    return state


def add_supervised_pair(agent, prompt: str, target: str, reward: float) -> int:
    prompt_tokens = ensure_words(agent, prompt)
    target_tokens = ensure_words(agent, target)
    max_vocab = min(brain.vocab_size for brain in agent.brains)
    usable = [token for token in target_tokens if 0 < token < max_vocab]
    if not usable:
        return 0

    for text in (prompt, target):
        for word in clean_words(text):
            if "concepts" not in agent.knowledge_web:
                agent.knowledge_web["concepts"] = {}
            agent.knowledge_web["concepts"].setdefault(
                word,
                {"category": "semantic_warmstart", "confidence": 0.8, "source": "semantic_trainer"},
            )
            if hasattr(agent, "enhanced_kb"):
                agent.enhanced_kb.concepts.setdefault(
                    word,
                    {"category": "semantic_warmstart", "confidence": 0.8, "source": "semantic_trainer"},
                )
            if hasattr(agent, "atomic_languages"):
                for als in agent.atomic_languages:
                    als.acquire_concept(word, source="semantic_trainer", initial_strength=0.45)

    added = 0
    prefix: list[int] = []
    for token in usable:
        state = encode_state(agent, prompt_tokens, prefix)
        agent.add_experience(
            state=state,
            action=0,
            reward=reward,
            next_state=state,
            done=False,
            input_tokens=prompt_tokens + prefix,
            target_tokens=[token],
            vp_value=0.05,
        )
        prefix.append(token)
        added += 1

    agent.conversation.add_message("user", prompt)
    agent.conversation.add_message("assistant", target, {"semantic_reward": reward})
    agent.record_training_log(
        "semantic_pair",
        stage="semantic_warmstart",
        input_text=prompt,
        target=target,
        output=target,
        reward=reward,
        score={"usable_target_tokens": added},
    )
    return added


def concept_category(agent, word: str) -> str:
    concepts = agent.knowledge_web.get("concepts", {}) if isinstance(agent.knowledge_web, dict) else {}
    info = concepts.get(word, {})
    return info.get("category") or f"bridge_{word}"


def add_relation(agent, source: str, target: str, relation_type: str, strength: float) -> None:
    if "relations" not in agent.knowledge_web or not isinstance(agent.knowledge_web["relations"], list):
        agent.knowledge_web["relations"] = []
    relation = {
        "source": source,
        "target": target,
        "type": relation_type,
        "strength": strength,
    }
    agent.knowledge_web["relations"].append(relation)

    if hasattr(agent, "enhanced_kb"):
        rel = type(agent.enhanced_kb.relations[0]) if agent.enhanced_kb.relations else None
        if rel is not None:
            obj = rel(source=source, target=target, relation_type=relation_type, strength=strength)
            agent.enhanced_kb.relations.append(obj)
            agent.enhanced_kb.relation_index.setdefault(source, []).append(obj)


def strengthen_association(agent, source: str, target: str, strength: float, reason: str) -> None:
    if hasattr(agent, "atomic_languages"):
        for als in agent.atomic_languages:
            als.acquire_concept(source, source="semantic_bridge", semantic_frame="relationship", initial_strength=0.55)
            als.acquire_concept(target, source="semantic_bridge", semantic_frame="relationship", initial_strength=0.55)
            als.form_association(source, target, strength, reason)
            als.form_association(target, source, strength * 0.75, reason)


def add_semantic_bridge(agent, source: str, targets: list[str]) -> int:
    source = source.lower()
    targets = [target.lower() for target in targets]
    ensure_words(agent, " ".join([source] + targets))

    if "concepts" not in agent.knowledge_web:
        agent.knowledge_web["concepts"] = {}

    category = concept_category(agent, targets[0])
    agent.knowledge_web["concepts"][source] = {
        "category": category,
        "confidence": 0.92,
        "source": "semantic_bridge",
        "bridges_to": targets,
    }
    if hasattr(agent, "enhanced_kb"):
        agent.enhanced_kb.concepts[source] = agent.knowledge_web["concepts"][source]

    made = 0
    for target in targets:
        agent.knowledge_web["concepts"].setdefault(
            target,
            {"category": category, "confidence": 0.85, "source": "semantic_bridge_core"},
        )
        strengthen_association(agent, source, target, 0.9, "semantic_bridge")
        add_relation(agent, source, target, "semantic_bridge", 0.9)
        add_relation(agent, target, source, "semantic_alias", 0.65)
        made += add_supervised_pair(agent, source, " ".join(targets[:4]), 1.0)
    agent.record_training_log(
        "semantic_bridge",
        stage="association_warmstart",
        input_text=source,
        target=" ".join(targets),
        output="atomic association + knowledge relation",
        reward=1.0,
        score={"targets": len(targets)},
    )
    return made


def add_reasoning_ladder(agent, ladder: list[str], reward: float) -> int:
    """Train fixed-point growth: a stable word expands into a short reasoning chain."""
    made = 0
    for idx, source in enumerate(ladder):
        prefix = ladder[: idx + 1]
        target = ladder[: min(len(ladder), idx + 2)]
        target_text = " ".join(target)
        prompt_text = " ".join(prefix)
        made += add_supervised_pair(agent, prompt_text, target_text, reward)
        for other in target:
            if other != source:
                strengthen_association(agent, source, other, 0.82, "fixed_point_ladder")
                add_relation(agent, source, other, "fixed_point_step", 0.82)
    agent.record_training_log(
        "fixed_point_ladder",
        stage="reasoning_warmstart",
        input_text=ladder[0],
        target=" ".join(ladder),
        output="small-to-complex association ladder",
        reward=reward,
        score={"length": len(ladder)},
    )
    return made


def add_cognitive_faculty(agent, action_id: int, name: str, triggers: list[str], target: str) -> int:
    """Use the robotics action head as a cognition-router head."""
    made = 0
    ensure_words(agent, " ".join([name] + triggers + target.split()))
    target_tokens = ensure_words(agent, target)
    max_vocab = min(brain.vocab_size for brain in agent.brains)
    usable_targets = [token for token in target_tokens if 0 < token < max_vocab]
    for trigger in triggers:
        prompt_tokens = ensure_words(agent, f"{name} {trigger}")
        state = encode_state(agent, prompt_tokens, [])
        agent.add_experience(
            state=state,
            action=action_id,
            reward=1.0,
            next_state=state,
            done=False,
            input_tokens=prompt_tokens,
            target_tokens=[usable_targets[0]] if usable_targets else None,
            vp_value=0.04,
        )
        strengthen_association(agent, name, trigger, 0.88, "cognitive_faculty")
        add_relation(agent, name, trigger, "faculty_trigger", 0.88)
        made += 1
    agent.record_training_log(
        "cognitive_faculty",
        stage="action_head_faculty_router",
        input_text=name,
        target=target,
        output=f"action_head_{action_id}",
        reward=1.0,
        score={"action_id": action_id, "triggers": triggers},
    )
    return made


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cocoon", help="Path to the source cocoon .py file")
    parser.add_argument("--output", default="cocoon_semantic_trained.py")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--max-organisms", type=int, default=None)
    args = parser.parse_args()

    module = load_cocoon(os.path.abspath(args.cocoon))
    agent = module.CocoonAgent(max_organisms=args.max_organisms)

    total_pairs = 0
    total_experiences = 0
    total_bridges = 0
    for source, targets in SEMANTIC_BRIDGES.items():
        total_experiences += add_semantic_bridge(agent, source, targets)
        total_bridges += 1
    for source, targets in REASONING_BRIDGES.items():
        total_experiences += add_semantic_bridge(agent, source, targets)
        total_bridges += 1
    for ladder in FIXED_POINT_LADDERS:
        total_experiences += add_reasoning_ladder(agent, ladder, 1.0)
    for prompt, target in REASONING_FIXED_POINTS:
        total_experiences += add_supervised_pair(agent, prompt, target, 1.0)
    for action_id, spec in COGNITIVE_FACULTIES.items():
        total_experiences += add_cognitive_faculty(
            agent,
            action_id,
            spec["name"],
            spec["triggers"],
            spec["target"],
        )

    for epoch in range(args.epochs):
        reward = 1.0 - min(epoch, 10) * 0.01
        for prompt, target in CURRICULUM:
            total_experiences += add_supervised_pair(agent, prompt, target, reward)
            total_pairs += 1

    losses = []
    started = time.time()
    for step in range(args.steps):
        loss = agent.train_step()
        if loss > 0:
            losses.append(loss)
        if (step + 1) % 8 == 0:
            tail = f"{losses[-1]:.4f}" if losses else "n/a"
            print(f"[TRAIN] step {step + 1}/{args.steps}, latest_loss={tail}", flush=True)

    agent.record_training_log(
        "semantic_warmstart_complete",
        stage="semantic_warmstart",
        input_text=f"{total_pairs} pairs",
        target="normal chat behavior plus semantic bridges, fixed-point reasoning, and action faculties",
        output=f"{total_experiences} supervised token experiences; {total_bridges} bridges",
        reward=1.0,
        score={
            "train_steps": len(losses),
            "first_loss": losses[0] if losses else None,
            "last_loss": losses[-1] if losses else None,
            "seconds": time.time() - started,
        },
    )

    agent.export_cocoon(args.output)
    print(f"[DONE] Exported {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
