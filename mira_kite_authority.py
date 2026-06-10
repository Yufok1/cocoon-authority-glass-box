#!/usr/bin/env python3
"""Cocoon Authority: local Android/Termux trainer app.

This app wraps a standalone cocoon export with a browser UI and JSON API.
It is designed for Termux on Android: no Flask dependency, no paid service,
and all training/checkpointing happens on the local filesystem.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import io
import json
import os
import re
import contextlib
import subprocess
import sys
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from cocoon_capabilities import CAPABILITY_INDEX, capability_manifest, enriched_capability
from cocoon_ops import DEFAULT_ROOT, scan_cocoons

MODULE_AVAILABILITY_CACHE: dict[str, bool] = {}
MODULE_ERROR_CACHE: dict[str, str] = {}

try:
    from cascade import (
        CausationLink,
        HoldPoint,
        MetricsEngine,
        Monitor,
        SymbioticAdapter,
        Tracer,
    )
except Exception:
    CausationLink = None
    HoldPoint = None
    MetricsEngine = None
    Monitor = None
    SymbioticAdapter = None
    Tracer = None

try:
    from cascade.viz import create_tape_path, write_tape_event
except Exception:
    create_tape_path = None
    write_tape_event = None

from semantic_cocoon_trainer import (
    COGNITIVE_FACULTIES,
    FIXED_POINT_LADDERS,
    add_cognitive_faculty,
    add_reasoning_ladder,
    add_relation,
    add_semantic_bridge,
    add_supervised_pair,
    clean_words,
    strengthen_association,
)

from mira_kite_infinite_trainer import (
    CORE_ANCHORS,
    choose_anchors,
    classify_faculty,
    fetch_url_text,
    observe_receipt,
    quinesmith_ladders,
    train_item,
    vocabulary_sets,
)


ACTION_LABELS = {
    0: "observe",
    1: "associate",
    2: "recall",
    3: "reason",
    4: "respond",
    5: "verify",
}

DEFAULT_CARETAKER_NAMES = {
    "16525dfc6c33419b": "Mira",
    "0c019d24a56b0086": "Kite",
}

FACULTY_ORDER = ["observe", "associate", "recall", "reason", "respond", "verify"]

FACULTY_HINTS = {
    "observe": {"notice", "sense", "input", "data", "feed", "signal", "watch", "measure"},
    "associate": {"relate", "connect", "bridge", "semantic", "meaning", "network", "map"},
    "recall": {"remember", "memory", "ancestry", "lineage", "history", "restore", "list"},
    "reason": {"why", "cause", "effect", "logic", "sequence", "fixed", "point", "arbitrate"},
    "respond": {"chat", "answer", "speak", "explain", "inform", "teach", "dialogue"},
    "verify": {"audit", "receipt", "prove", "check", "harden", "safe", "trace", "validate"},
}

SOURCE_WEIGHT = {
    "manual_signal": 1.0,
    "govern": 1.15,
    "manual_relate": 1.05,
    "feed": 0.92,
    "quine_drill": 1.08,
    "autopilot": 1.2,
}

DEFAULT_SCOUT_FEEDS = [
    "https://export.arxiv.org/rss/cs.AI",
    "https://export.arxiv.org/rss/cs.LG",
    "https://hnrss.org/newest?q=machine%20learning",
]


def load_cocoon(path: str):
    module_key = hashlib.sha1(str(Path(path).resolve()).encode("utf-8")).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(f"mira_kite_authority_cocoon_{module_key}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import cocoon from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def response_quality(prompt: str, response: str) -> dict[str, Any]:
    response_words = clean_words(response)
    prompt_words = set(clean_words(prompt))
    if not response_words:
        return {
            "word_count": 0,
            "unique_count": 0,
            "unique_ratio": 0.0,
            "prompt_overlap": 0.0,
            "longest_run": 0,
            "repetition_penalty": 1.0,
            "primitive_signal": True,
            "score": 0.05,
        }
    longest_run = 1
    run = 1
    for prev, word in zip(response_words, response_words[1:]):
        if word == prev:
            run += 1
            longest_run = max(longest_run, run)
        else:
            run = 1
    unique = set(response_words)
    unique_ratio = len(unique) / max(1, len(response_words))
    prompt_overlap = len(unique & prompt_words) / max(1, min(len(prompt_words), len(unique)))
    repetition_penalty = min(0.75, max(0.0, (longest_run - 2) * 0.18 + max(0.0, 0.35 - unique_ratio)))
    primitive_signal = longest_run >= 3 or unique_ratio < 0.35
    score = clamp(
        0.12
        + (0.38 * unique_ratio)
        + (0.28 * prompt_overlap)
        + (0.14 * min(1.0, len(response_words) / 10.0))
        - repetition_penalty,
        0.05,
        1.0,
    )
    return {
        "word_count": len(response_words),
        "unique_count": len(unique),
        "unique_ratio": round(unique_ratio, 4),
        "prompt_overlap": round(prompt_overlap, 4),
        "longest_run": longest_run,
        "repetition_penalty": round(repetition_penalty, 4),
        "primitive_signal": primitive_signal,
        "score": round(score, 4),
    }


def acclimation_lesson(prompt: str, response: str, anchors: list[str], faculty: str) -> str:
    prompt_atoms = top_words(prompt, 16)
    response_atoms = top_words(response, 16)
    anchor_text = " ".join(anchors[:6] or ["understand", "reason", "respond", "verify"])
    prompt_text = " ".join(prompt_atoms[:12])
    response_text = " ".join(response_atoms[:12]) if response_atoms else "silent primitive output"
    return (
        "council acclimation lesson. "
        f"prompt atoms: {prompt_text}. "
        f"known anchors: {anchor_text}. "
        f"faculty route: {faculty}. "
        f"primitive response observed: {response_text}. "
        "better response uses known anchors, preserves the request meaning, and answers with varied words."
    )


def summarize_agent_dialogue(agent: Any, prompt: str, learn: bool = True, steps: int = 1) -> dict[str, Any]:
    input_dim = agent.brains[0].input_dim if getattr(agent, "brains", None) else 30
    vp_info = (
        agent.vp_runtime.compute_from_state(np.zeros(input_dim, dtype=np.float32), agent.reward_history)
        if hasattr(agent, "vp_runtime")
        else {"violation_pressure": 0.0}
    )
    current_vp = float(vp_info.get("violation_pressure", 0.0) or 0.0)
    names = list(getattr(agent, "organism_names", []) or [])
    if not names and getattr(agent, "brains", None):
        names = [f"org_{idx}" for idx, _brain in enumerate(agent.brains)]
    _known, predictable = vocabulary_sets(agent)
    prompt_words = top_words(prompt, 64)
    anchors = choose_anchors(prompt_words, predictable)
    _action_id, faculty = classify_faculty(prompt_words + anchors)
    responses = []
    for i, name in enumerate(names):
        response, confidence = agent.generate_response(prompt, organism_idx=i, vp_value=current_vp)
        fitness = agent.organism_fitness[i] if i < len(getattr(agent, "organism_fitness", []) or []) else 1.0
        semantic_reward = 0.25
        if hasattr(agent, "_calculate_semantic_reward"):
            semantic_reward = float(agent._calculate_semantic_reward(prompt, response, confidence, current_vp))
        quality = response_quality(prompt, response)
        base_weight = float(fitness) * float(confidence) * (0.5 + semantic_reward)
        responses.append(
            {
                "organism": name,
                "response": response,
                "confidence": float(confidence),
                "fitness": float(fitness),
                "semantic_reward": semantic_reward,
                "quality": quality,
                "base_weight": base_weight,
                "weight": base_weight * float(quality["score"]),
            }
        )
    valid = [
        r
        for r in responses
        if r["response"].strip()
        and not r["response"].startswith("[")
        and not r.get("quality", {}).get("primitive_signal")
    ]
    best = max(valid or responses, key=lambda item: item["weight"]) if responses else {}
    best_quality = best.get("quality", {}) if isinstance(best.get("quality"), dict) else {}
    reward = clamp(float(best.get("semantic_reward", 0.1)) * (0.55 + 0.45 * float(best_quality.get("score", 0.1))), 0.05, 0.95)
    losses = []
    lesson = acclimation_lesson(prompt, str(best.get("response", "")), anchors, faculty)
    if learn and hasattr(agent, "learn_from_text"):
        agent.learn_from_text(prompt, reward=reward, vp_value=current_vp)
        agent.learn_from_text(lesson, reward=max(0.7, reward), vp_value=current_vp)
        if hasattr(agent, "train_step"):
            for _ in range(max(0, int(steps))):
                loss = agent.train_step()
                if loss and loss > 0 and not np.isnan(loss):
                    losses.append(float(loss))
        if hasattr(agent, "record_training_log"):
            try:
                agent.record_training_log(
                    "council_chat_turn",
                    stage="butterfly_council",
                    input_text=prompt[:500],
                    target=lesson[:500],
                    output=str(best.get("response", ""))[:500],
                    reward=reward,
                    score={
                        "semantic_reward": reward,
                        "vp_value": current_vp,
                        "losses": losses,
                        "quality": best_quality,
                        "anchors": anchors,
                        "faculty": faculty,
                    },
                )
            except Exception:
                pass
    if hasattr(agent, "conversation"):
        try:
            agent.conversation.add_message("user", prompt)
            agent.conversation.add_message("assistant", str(best.get("response", "")), {"semantic_reward": reward})
        except Exception:
            pass
    return {
        "response": best.get("response", ""),
        "all_responses": responses,
        "selected_index": int(responses.index(best)) if best in responses else None,
        "selected_organism": best.get("organism"),
        "selected_weight": float(best.get("weight", 0.0) or 0.0),
        "selected_quality": best_quality,
        "semantic_reward": reward,
        "vp_value": current_vp,
        "anchors": anchors,
        "faculty": faculty,
        "acclimation_lesson": lesson,
        "primitive_signals": sum(1 for r in responses if r.get("quality", {}).get("primitive_signal")),
        "losses": losses,
        "trained": bool(learn),
    }


def json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def top_words(text: str, limit: int = 32) -> list[str]:
    words = [w for w in clean_words(text) if len(w) > 2]
    out: list[str] = []
    for word in words:
        if word not in out:
            out.append(word)
        if len(out) >= limit:
            break
    return out


def stable_id(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def cocoon_display_name(path: Path) -> str:
    return path.stem


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def module_available(name: str) -> bool:
    if name in MODULE_AVAILABILITY_CACHE:
        return MODULE_AVAILABILITY_CACHE[name]
    try:
        if importlib.util.find_spec(name) is None:
            MODULE_AVAILABILITY_CACHE[name] = False
            MODULE_ERROR_CACHE[name] = "module spec not found"
            return False
        importlib.import_module(name)
        MODULE_AVAILABILITY_CACHE[name] = True
        MODULE_ERROR_CACHE.pop(name, None)
        return True
    except Exception as exc:
        MODULE_AVAILABILITY_CACHE[name] = False
        MODULE_ERROR_CACHE[name] = f"{exc.__class__.__name__}: {str(exc)[:240]}"
        return False


class AuthorityState:
    def __init__(self, cocoon_path: Path, output_dir: Path):
        self.root = DEFAULT_ROOT.resolve()
        requested = Path(cocoon_path).expanduser()
        if not requested.is_absolute():
            requested = self.root / requested
        requested = requested.resolve()
        try:
            requested.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"cocoon path is outside managed root: {requested}") from exc
        if requested.suffix != ".py" or not requested.is_file():
            raise ValueError(f"expected a runnable Python cocoon file: {requested}")
        self.cocoon_path = requested
        self.cocoon_name = cocoon_display_name(self.cocoon_path)
        self.loaded_at = time.time()
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.module = load_cocoon(str(self.cocoon_path))
        self.agent = self.module.CocoonAgent()
        self.started_at = time.time()
        self.cycles = 0
        self.last_receipt: str | None = None
        self.last_checkpoint: str | None = None
        self.last_wealth: dict[str, Any] | None = None
        self.last_arbitration: dict[str, Any] | None = None
        self.event_log: list[dict[str, Any]] = []
        self.education_log: list[dict[str, Any]] = []
        self.feed_reports: list[dict[str, Any]] = []
        self.capability_cache: dict[str, Any] = {}
        self.adapter = SymbioticAdapter() if SymbioticAdapter is not None else None
        self.monitor = Monitor("mira_kite_authority") if Monitor is not None else None
        self.graph = getattr(self.monitor, "graph", None)
        self.metrics = getattr(self.monitor, "metrics", None)
        if self.metrics is None and MetricsEngine is not None:
            self.metrics = MetricsEngine(self.graph)
        self.tracer = getattr(self.monitor, "tracer", None)
        if self.tracer is None and Tracer is not None and self.graph is not None:
            self.tracer = Tracer(self.graph)
        self.cascade_errors: list[str] = []
        self.tape_path = None
        if create_tape_path is not None:
            try:
                self.tape_path = create_tape_path(str(self.output_dir / "cascade_tapes"), "mira_kite")
            except Exception as exc:
                self.cascade_errors.append(str(exc))
        self._record_event("authority_started", {"cocoon": str(self.cocoon_path)})

    def cocoon_inventory(self) -> dict[str, Any]:
        discovered = []
        for item in scan_cocoons(self.root):
            if item.get("kind") != "python_cocoon":
                continue
            item = dict(item)
            item["active"] = item.get("path") == str(self.cocoon_path)
            item["display_name"] = cocoon_display_name(Path(item["path"]))
            discovered.append(item)
        return {
            "root": str(self.root),
            "active": {
                "path": str(self.cocoon_path),
                "name": self.cocoon_name,
                "loaded_at": self.loaded_at,
                "mode": "ENSEMBLE" if getattr(self.agent, "is_ensemble", False) else "SOLO",
                "organisms": len(getattr(self.agent, "brains", []) or []),
            },
            "cocoons": discovered,
        }

    def reload_cocoon(self, path: str | None = None) -> dict[str, Any]:
        with self.lock:
            requested = Path(path).expanduser() if path else self.cocoon_path
            if not requested.is_absolute():
                requested = self.root / requested
            requested = requested.resolve()
            try:
                requested.relative_to(self.root)
            except ValueError as exc:
                raise ValueError(f"cocoon path is outside managed root: {requested}") from exc
            if requested.suffix != ".py" or not requested.is_file():
                raise ValueError(f"expected a runnable Python cocoon file: {requested}")

            old_path = self.cocoon_path
            old_name = self.cocoon_name
            module = load_cocoon(str(requested))
            agent = module.CocoonAgent()
            self.module = module
            self.agent = agent
            self.cocoon_path = requested
            self.cocoon_name = cocoon_display_name(requested)
            self.loaded_at = time.time()
            self.cycles = 0
            self.last_receipt = None
            self.last_checkpoint = None
            self.last_wealth = None
            self.last_arbitration = None
            self.education_log = []
            self.feed_reports = []
            data = {
                "old_cocoon": str(old_path),
                "old_name": old_name,
                "new_cocoon": str(self.cocoon_path),
                "new_name": self.cocoon_name,
                "inventory": self.cocoon_inventory(),
                "state": self.capability_snapshot(),
            }
            self.last_receipt = observe_receipt("amalgam_cocoon_reload", data)
            data["receipt"] = self.last_receipt
            self._record_event("cocoon_reloaded", data)
            self._write_status(data)
            return data

    def _compact_payload(self, value: Any, depth: int = 0) -> Any:
        if depth > 3:
            return f"<{type(value).__name__}>"
        if isinstance(value, dict):
            skip = {
                "state",
                "state_before",
                "state_after",
                "signal_result",
                "mission_result",
                "drill_results",
                "quine_result",
                "feed_results",
                "game_result",
                "checkpoint",
                "all_responses",
                "raw",
            }
            compact: dict[str, Any] = {}
            for key, item in value.items():
                if key in skip:
                    compact[key] = self._summarize_nested(item)
                else:
                    compact[key] = self._compact_payload(item, depth + 1)
                if len(compact) >= 32:
                    compact["_truncated_keys"] = len(value) - len(compact)
                    break
            return compact
        if isinstance(value, list):
            return [self._compact_payload(item, depth + 1) for item in value[:12]]
        if isinstance(value, tuple):
            return [self._compact_payload(item, depth + 1) for item in value[:12]]
        if isinstance(value, str):
            return value[:700] + ("..." if len(value) > 700 else "")
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, np.ndarray):
            return {"array_shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        return str(value)[:700]

    def _summarize_nested(self, value: Any) -> Any:
        if isinstance(value, dict):
            summary: dict[str, Any] = {}
            for key in [
                "mode",
                "cycle",
                "facility",
                "receipt",
                "response",
                "plain_summary",
                "simple_summary",
                "bridges",
                "pairs",
                "losses",
                "anchors",
                "novel",
                "vocabulary_size",
                "knowledge_relations",
                "conversation_turns",
            ]:
                if key in value:
                    summary[key] = self._compact_payload(value[key], 1)
            if "teacher_note" in value and isinstance(value["teacher_note"], dict):
                summary["teacher_note"] = value["teacher_note"].get("summary", "")
            if "state" in value and isinstance(value["state"], dict):
                state = value["state"]
                summary["state"] = {
                    "vocabulary_size": state.get("vocabulary_size"),
                    "knowledge_relations": state.get("knowledge_relations"),
                    "conversation_turns": state.get("conversation_turns"),
                    "cycles": state.get("cycles"),
                }
            return summary or {"keys": list(value.keys())[:12]}
        if isinstance(value, list):
            return {"items": len(value), "head": [self._summarize_nested(item) for item in value[:3]]}
        return self._compact_payload(value, 1)

    def _record_event(self, kind: str, data: dict[str, Any]) -> None:
        entry = {"time": time.time(), "kind": kind, "data": self._compact_payload(data)}
        self.event_log.append(entry)
        self.event_log = self.event_log[-200:]
        try:
            path = self.output_dir / "authority_events.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=json_default) + "\n")
        except Exception:
            pass
        if write_tape_event is not None and self.tape_path is not None:
            try:
                write_tape_event(self.tape_path, entry)
            except Exception as exc:
                self.cascade_errors.append(str(exc))
                self.cascade_errors = self.cascade_errors[-12:]

    def _record_lesson(self, title: str, data: dict[str, Any]) -> dict[str, Any]:
        lesson = {"time": time.time(), "title": title, **data}
        self.education_log.append(lesson)
        self.education_log = self.education_log[-80:]
        try:
            path = self.output_dir / "education_log.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(lesson, default=json_default) + "\n")
        except Exception:
            pass
        return lesson

    def _cascade_observe(self, kind: str, signal: dict[str, Any]) -> dict[str, Any]:
        report: dict[str, Any] = {
            "kind": kind,
            "available": self.monitor is not None or self.adapter is not None,
            "event_id": None,
            "component": None,
            "event_type": None,
            "metrics": {},
            "health": {},
            "roots": [],
        }
        try:
            event = None
            if self.monitor is not None:
                event = self.monitor.observe(signal)
            elif self.adapter is not None:
                event = self.adapter.interpret(signal)
                if self.graph is not None:
                    self.graph.add_event(event)

            if event is not None:
                report["event_id"] = getattr(event, "event_id", None)
                report["component"] = getattr(event, "component", None)
                report["event_type"] = getattr(event, "event_type", None)
                if self.metrics is not None:
                    metric_report = self.metrics.ingest(event)
                    report["metrics"] = {
                        name: len(series) if hasattr(series, "__len__") else 1
                        for name, series in (metric_report or {}).items()
                    }
                    try:
                        report["health"] = self.metrics.health_summary()
                    except Exception as exc:
                        report["health_error"] = str(exc)
                if self.tracer is not None and report["event_id"]:
                    try:
                        roots = self.tracer.find_root_causes(report["event_id"])
                        report["roots"] = [str(root)[:200] for root in roots[:5]]
                    except Exception as exc:
                        report["trace_error"] = str(exc)
        except Exception as exc:
            report["error"] = str(exc)
            self.cascade_errors.append(str(exc))
            self.cascade_errors = self.cascade_errors[-12:]
        return report

    def _add_causation(self, from_id: str | None, to_id: str | None, kind: str, strength: float, explanation: str) -> None:
        if not from_id or not to_id or self.graph is None or CausationLink is None:
            return
        try:
            self.graph.add_link(
                CausationLink(
                    from_event=from_id,
                    to_event=to_id,
                    causation_type=kind,
                    strength=clamp(strength),
                    explanation=explanation,
                    metrics_involved=["novelty", "anchors", "faculty", "training_loss"],
                )
            )
        except Exception as exc:
            self.cascade_errors.append(str(exc))
            self.cascade_errors = self.cascade_errors[-12:]

    def _score_faculties(self, words: list[str], anchors: list[str], source: str) -> dict[str, float]:
        bag = set(words + anchors)
        scores: dict[str, float] = {}
        for faculty in FACULTY_ORDER:
            hints = FACULTY_HINTS[faculty]
            overlap = len(bag & hints)
            anchor_bonus = 0.08 * len([a for a in anchors if a in hints])
            scores[faculty] = 0.15 + overlap * 0.18 + anchor_bonus
        scores["associate"] += min(0.25, len(words) / 120)
        scores["reason"] += 0.12 if {"cause", "effect", "fixed", "point"} & bag else 0
        scores["verify"] += 0.18 if {"harden", "receipt", "audit", "safe"} & bag else 0
        scores["respond"] += 0.14 if {"chat", "inform", "teach", "explain"} & bag else 0
        weight = SOURCE_WEIGHT.get(source, 1.0)
        return {name: round(clamp(value * weight), 4) for name, value in scores.items()}

    def _arbitrate(self, words: list[str], anchors: list[str], source: str) -> dict[str, Any]:
        action_id, classified = classify_faculty(words + anchors)
        scores = self._score_faculties(words, anchors, source)
        if classified in scores:
            scores[classified] = round(clamp(scores[classified] + 0.18), 4)
        chosen = max(scores, key=scores.get)
        for idx, label in ACTION_LABELS.items():
            if label == chosen:
                action_id = idx
                break
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        verification = {
            "passes": bool(anchors and words),
            "checks": [
                "has_known_anchor" if anchors else "missing_known_anchor",
                "has_trainable_words" if words else "missing_words",
                "uses_safe_quine_boundary",
            ],
        }
        arbitration = {
            "chosen_faculty": chosen,
            "action_id": action_id,
            "classified_faculty": classified,
            "ranked_faculties": ranked,
            "verification": verification,
            "reason": f"{chosen} ranked highest from source={source}, anchors={','.join(anchors[:4]) or 'none'}",
        }
        self.last_arbitration = arbitration
        return arbitration

    def _informational_wealth(
        self,
        text: str,
        words: list[str],
        anchors: list[str],
        novel: list[str],
        arbitration: dict[str, Any],
        losses: list[float],
        source: str,
    ) -> dict[str, Any]:
        scores = dict(arbitration.get("ranked_faculties", []))
        total = sum(scores.values()) or 1.0
        probs = [scores.get(label, 0.0) / total for label in FACULTY_ORDER]
        features = {
            "known_anchor_count": float(len(anchors)),
            "novel_symbol_count": float(len(novel)),
            "word_count": float(len(words)),
            "loss_latest": float(losses[-1]) if losses else 0.0,
            "source_weight": SOURCE_WEIGHT.get(source, 1.0),
        }
        attention = {word: round(1.0 / (idx + 1), 4) for idx, word in enumerate((anchors + novel)[:8])}
        reasoning = [
            f"Route through {arbitration['chosen_faculty']} because it best fits the signal.",
            f"Bridge novel symbols {', '.join(novel[:4]) or 'none'} to known anchors {', '.join(anchors[:4]) or 'none'}.",
            "Verify by recording Cascade receipt, causation, and training log.",
        ]
        world_prediction = {
            "expected_effect": "more stable association recall and better chat routing",
            "capacity_note": "words above the language-head limit are preserved as symbolic web nodes",
            "next_best_action": "export checkpoint after meaningful training batches",
        }
        wealth = {
            "id": stable_id(source, text[:120], time.time(), self.cycles),
            "action_labels": FACULTY_ORDER,
            "action_probs": [round(p, 5) for p in probs],
            "value": round(clamp(max(scores.values()) if scores else 0.0), 4),
            "observation": {"source": source, "text": text[:500], "anchors": anchors, "novel": novel},
            "attention": attention,
            "features": features,
            "reasoning": reasoning,
            "world_prediction": world_prediction,
        }
        if HoldPoint is not None:
            try:
                hold = HoldPoint(
                    action_probs=np.array(probs, dtype=float),
                    value=wealth["value"],
                    observation=wealth["observation"],
                    brain_id="mira_kite_authority",
                    action_labels=FACULTY_ORDER,
                    attention=attention,
                    features=features,
                    reasoning=reasoning,
                    world_prediction=world_prediction,
                )
                wealth["hold_point_id"] = getattr(hold, "id", None)
                wealth["hold_state"] = str(getattr(hold, "state", "pending"))
            except Exception as exc:
                wealth["hold_error"] = str(exc)
        self.last_wealth = wealth
        return wealth

    def _teacher_note(
        self,
        text: str,
        words: list[str],
        anchors: list[str],
        novel: list[str],
        arbitration: dict[str, Any],
        bridges: int,
        pairs: int,
        losses: list[float],
        receipt: str | None,
    ) -> dict[str, Any]:
        latest_loss = round(losses[-1], 5) if losses else None
        note = {
            "summary": (
                f"Signal routed to {arbitration['chosen_faculty']}; "
                f"{len(novel)} novel words were bridged to {len(anchors)} known anchors."
            ),
            "you_learn": [
                "A cocoon learns safest when new words are tied to words already inside its predictable vocabulary.",
                "The authority loop separates routing, association, reasoning, and verification so each lesson has a trace.",
                "Quine dynamics are used as syntax/fixed-point drills only; generated commands are not executed by this app.",
            ],
            "cocoon_learns": {
                "anchors": anchors,
                "novel": novel,
                "faculty": arbitration["chosen_faculty"],
                "bridges": bridges,
                "supervised_pairs": pairs,
                "latest_loss": latest_loss,
            },
            "receipt": receipt,
            "input_preview": text[:220],
            "top_words": words[:16],
        }
        return self._record_lesson("cycle_teacher_note", note)

    def native_facilities(self) -> dict[str, Any]:
        package_dir = Path(__file__).resolve().parent
        curriculum_dir = package_dir / "curriculum"
        curriculum_files = sorted(p.name for p in curriculum_dir.glob("*.json")) if curriculum_dir.exists() else []
        contract = {}
        if hasattr(self.module, "cocoon_capability_contract"):
            try:
                contract = self.module.cocoon_capability_contract()
            except Exception as exc:
                contract = {"error": str(exc)}
        methods = [
            "generate_response",
            "learn_from_text",
            "get_action",
            "get_continuous_action",
            "add_experience",
            "train_step",
            "record_training_log",
            "add_word",
            "add_concept",
            "tokenize",
            "detokenize",
            "export_cocoon",
            "export_onnx",
            "export_torchscript",
            "export_package",
        ]
        facilities = {
            "native_cli_modes": {
                "info": "metadata, architecture, vocabulary, alliance summary",
                "chat": "interactive conversation with optional learning",
                "council": "fan one prompt out across discovered runnable cocoons",
                "serve": "Flask HTTP API in original cocoon; mirrored here without Flask",
                "gym": "Gymnasium environments, training or inference",
                "sphere": "headless or visual 3D swarm defense arena",
                "link": "websocket relay for cocoon battles/chat",
            },
            "native_agent_methods": [name for name in methods if hasattr(self.agent, name)],
            "native_subsystems": [
                "OrganismBrain action/language heads",
                "HopfieldLayer iterative pattern memory",
                "MultiHeadAttention",
                "AtomicLanguageSystem",
                "ConversationHistory",
                "EnhancedKnowledgeWeb",
                "VPRuntime",
                "ExperienceBuffer",
                "SphereArena",
                "GymRunner",
            ],
            "curriculum_files": curriculum_files,
            "curriculum_sequence": (contract.get("language_development", {}) or {}).get("sequence", []),
            "http_endpoints_mirrored": [
                "/health",
                "/act",
                "/learn",
                "/chat",
                "/council",
                "/teach",
                "/vocab",
                "/curriculum",
                "/training/logs",
                "/curriculum/score",
                "/snapshot",
                "/save",
                "/export",
                "/capabilities",
                "/dreamer/observe",
                "/dreamer/propose",
                "/api/native/facilities",
                "/api/native/health",
                "/api/native/chat",
                "/api/council",
                "/api/native/teach",
                "/api/native/act",
                "/api/native/learn",
                "/api/native/vocab",
                "/api/native/curriculum",
                "/api/native/training/logs",
                "/api/native/curriculum/score",
                "/api/native/snapshot",
                "/api/native/save",
                "/api/native/export",
                "/api/native/capabilities",
                "/api/native/dreamer/observe",
                "/api/native/dreamer/propose",
                "/api/native/sphere_burst",
                "/api/native/gym_burst",
                "/api/follow_along",
                "/api/autopilot",
            ],
            "game_lanes": {
                "sphere": {
                    "ready": hasattr(self.module, "SphereArena"),
                    "safe_app_hook": "bounded headless burst",
                    "supports_training": True,
                },
                "gym": {
                    "ready": hasattr(self.module, "GymRunner") and (module_available("gymnasium") or module_available("gym")),
                    "missing_if_false": "pip install gymnasium",
                    "supports_training": True,
                },
                "drone": {
                    "ready_files": {
                        "adapter": (package_dir / "cocoon_drone_adapter.py").exists(),
                        "arena": (package_dir / "cocoon_drone_arena.py").exists(),
                        "physics": (package_dir / "jsbsim_quadcopter.py").exists(),
                    },
                    "modes": [
                        "free_fly",
                        "formation",
                        "pursuit",
                        "tag_battle",
                        "zone_control",
                        "capture_flag",
                        "survival",
                        "escort",
                    ],
                    "needs": ["matplotlib for plots", "PyFlyt", "pybullet", "pettingzoo", "numba"],
                    "app_status": "bounded job runner is available; complete lane requires dependency probe success",
                },
                "tmrl": {
                    "ready_file": (package_dir / "cocoon_tmrl_adapter.py").exists(),
                    "needs": ["operator disabled; do not pursue"],
                    "app_status": "disabled by operator decision; retained as historical diagnostic only",
                },
                "link": {
                    "ready": module_available("websockets"),
                    "needs": ["websockets package", "CocoonHatch relay URL"],
                    "app_status": "requires relay target before launch",
                },
            },
            "dependency_probe": {
                "torch": module_available("torch"),
                "numpy": module_available("numpy"),
                "flask": module_available("flask"),
                "gymnasium": module_available("gymnasium"),
                "gym": module_available("gym"),
                "pygame": module_available("pygame"),
                "OpenGL": module_available("OpenGL"),
                "matplotlib": module_available("matplotlib"),
                "PyFlyt": module_available("PyFlyt"),
                "pybullet": module_available("pybullet"),
                "pettingzoo": module_available("pettingzoo"),
                "numba": module_available("numba"),
                "websockets": module_available("websockets"),
                "tmrl": module_available("tmrl"),
            },
            "dependency_errors": dict(MODULE_ERROR_CACHE),
            "contract": contract,
        }
        return facilities

    def follow_along_game(self, mode: str = "cue", seed: str = "", rounds: int = 6, steps: int = 2) -> dict[str, Any]:
        with self.lock:
            mode = (mode or "cue").lower()
            rounds = max(1, min(int(rounds), 24))
            words = top_words(seed or "signal meaning reason verify", 48)
            known, predictable = vocabulary_sets(self.agent)
            anchors = choose_anchors(words, predictable)
            base = words or anchors
            novel = [w for w in base if w not in anchors][:8]
            pairs: list[dict[str, Any]] = []

            def add_pair(inp: str, target: str, reward: float = 0.92) -> None:
                add_supervised_pair(self.agent, inp, target, reward)
                self.agent.learn_from_text(inp, reward=reward, allow_tool_language=False)
                self.agent.record_training_log(
                    "follow_along_pair",
                    stage=f"follow_along_{mode}",
                    input_text=inp[:250],
                    target=target[:250],
                    reward=reward,
                    score={"mode": mode, "round": len(pairs) + 1},
                )
                pairs.append({"input": inp, "target": target, "reward": reward})

            if mode == "echo":
                source = base[:rounds] or anchors
                for word in source:
                    phrase = f"{word} {word}" if len(word) > 2 else word
                    add_pair(phrase, phrase, 0.94)
            elif mode == "chain":
                chain = (anchors + novel + base)[: max(2, rounds + 1)]
                if len(chain) < 2:
                    chain = ["signal", "meaning", "reason", "verify"]
                for idx in range(min(rounds, len(chain) - 1)):
                    inp = " ".join(chain[: idx + 1])
                    target = " ".join(chain[: idx + 2])
                    add_pair(inp, target, 0.9)
                add_reasoning_ladder(self.agent, chain[:6], 0.92)
            elif mode == "role":
                transforms = [
                    ("I hear signal", "you hear signal"),
                    ("you cue me", "I follow cue"),
                    ("I remember meaning", "you remember meaning"),
                    ("we move then verify", "we move then verify"),
                ]
                for inp, target in transforms[:rounds]:
                    add_pair(inp, target, 0.93)
            elif mode == "chant":
                chant_words = (anchors + novel + ["signal", "meaning", "reason", "verify"])[:8]
                for idx, word in enumerate(chant_words[:rounds]):
                    nxt = chant_words[(idx + 1) % len(chant_words)]
                    # Original call-response chant; do not use copyrighted song lyrics.
                    add_pair(f"call {word}", f"response {word} links to {nxt}", 0.91)
                add_reasoning_ladder(self.agent, chant_words[:6], 0.91)
            else:
                cues = (novel + anchors + base)[:rounds]
                for idx, cue in enumerate(cues):
                    target = anchors[idx % len(anchors)] if anchors else "meaning"
                    add_pair(f"cue {cue}", f"trigger {target}", 0.92)
                    strengthen_association(self.agent, cue, target, 0.88, "follow_along_cue")
                    add_relation(self.agent, cue, target, "follow_along_trigger", 0.88)

            losses = self._train_steps(steps)
            self.cycles += 1
            text = " ".join([p["input"] + " " + p["target"] for p in pairs])
            game_words = top_words(text, 64)
            game_anchors = choose_anchors(game_words, predictable)
            game_novel = [w for w in game_words if w not in game_anchors][:8]
            arbitration = self._arbitrate(game_words, game_anchors, "manual_signal")
            wealth = self._informational_wealth(text, game_words, game_anchors, game_novel, arbitration, losses, "manual_signal")
            cascade = self._cascade_observe(
                "follow_along_game",
                {"mode": mode, "seed": seed[:400], "pairs": pairs, "wealth": wealth},
            )
            data = {
                "cycle": self.cycles,
                "mode": mode,
                "seed": seed[:600],
                "rounds": len(pairs),
                "pairs": pairs,
                "anchors": game_anchors,
                "novel": game_novel,
                "losses": losses,
                "arbitration": arbitration,
                "informational_wealth": wealth,
                "cascade": cascade,
                "state": self.capability_snapshot(),
            }
            self.last_receipt = observe_receipt("mira_kite_authority_follow_along", data)
            data["receipt"] = self.last_receipt
            data["teacher_note"] = self._teacher_note(
                text, game_words, game_anchors, game_novel, arbitration, 0, len(pairs), losses, self.last_receipt
            )
            self._record_event("follow_along", data)
            self._write_status(data)
            return data

    def native_health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "organisms": len(getattr(self.agent, "brains", []) or []),
            "pet_name": self.cocoon_name,
            "active_cocoon": str(self.cocoon_path),
            "vocab_size": len(self.agent.vocabulary.get("word_to_id", {})),
            "training_step": getattr(self.agent, "training_step", 0),
            "authority_cycles": self.cycles,
        }

    def native_curriculum(self) -> dict[str, Any]:
        if hasattr(self.agent, "language_curriculum"):
            return self.agent.language_curriculum
        if hasattr(self.module, "cocoon_language_curriculum"):
            return self.module.cocoon_language_curriculum()
        return {}

    def native_vocab(self, limit: int | None = None) -> dict[str, Any]:
        words = list(self.agent.vocabulary.get("word_to_id", {}).keys())
        if limit is not None:
            words = words[: max(1, min(int(limit), len(words)))]
        return {
            "vocab_size": len(self.agent.vocabulary.get("word_to_id", {})),
            "words": words,
            "limited": limit is not None,
        }

    def native_training_logs(self, limit: int = 100) -> dict[str, Any]:
        limit = max(1, min(int(limit), 1000))
        schema = self.module.cocoon_training_log_schema() if hasattr(self.module, "cocoon_training_log_schema") else {}
        logs = list(getattr(self.agent, "training_logs", []) or [])
        return {"schema": schema, "count": len(logs), "entries": logs[-limit:]}

    def native_curriculum_score(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            entry = self.agent.record_training_log(
                str(payload.get("event_type", "curriculum_score")),
                stage=str(payload.get("stage", "curriculum")),
                input_text=str(payload.get("input", ""))[:500],
                target=str(payload.get("target", ""))[:500],
                output=str(payload.get("output", ""))[:500],
                reward=float(payload.get("reward", 0.0)),
                score=payload.get("score") if isinstance(payload.get("score"), dict) else {},
                extra={"coach": payload.get("coach", "authority_reward_judge")},
            )
            data = {"accepted": True, "entry": entry, "state": self.capability_snapshot()}
            self.last_receipt = observe_receipt("mira_kite_authority_curriculum_score", data)
            data["receipt"] = self.last_receipt
            self._record_event("native_curriculum_score", data)
            self._write_status(data)
            return data

    def native_capabilities(self) -> dict[str, Any]:
        contract = {}
        if hasattr(self.module, "cocoon_capability_contract"):
            contract = self.module.cocoon_capability_contract()
        contract["live"] = {
            "active_cocoon": str(self.cocoon_path),
            "active_cocoon_name": self.cocoon_name,
            "organisms": len(self.agent.brains),
            "vocab_size": len(self.agent.vocabulary.get("word_to_id", {})),
            "training_step": getattr(self.agent, "training_step", 0),
            "authority_cycles": self.cycles,
        }
        contract["authority_facilities"] = self.native_facilities()
        contract["cocoon_inventory"] = self.cocoon_inventory()
        return contract

    def native_chat(self, prompt: str, learn: bool = True, steps: int = 1) -> dict[str, Any]:
        with self.lock:
            if not prompt.strip():
                raise ValueError("prompt is required")
            input_dim = self.agent.brains[0].input_dim if self.agent.brains else 30
            vp_info = self.agent.vp_runtime.compute_from_state(
                np.zeros(input_dim, dtype=np.float32),
                self.agent.reward_history,
            ) if hasattr(self.agent, "vp_runtime") else {"violation_pressure": 0.0}
            current_vp = float(vp_info.get("violation_pressure", 0.0) or 0.0)
            responses = []
            for i, name in enumerate(self.agent.organism_names):
                response, confidence = self.agent.generate_response(prompt, organism_idx=i, vp_value=current_vp)
                fitness = self.agent.organism_fitness[i] if i < len(self.agent.organism_fitness) else 1.0
                semantic_reward = 0.25
                if hasattr(self.agent, "_calculate_semantic_reward"):
                    semantic_reward = float(
                        self.agent._calculate_semantic_reward(prompt, response, confidence, current_vp)
                    )
                responses.append(
                    {
                        "organism": name,
                        "caretaker_name": DEFAULT_CARETAKER_NAMES.get(name),
                        "response": response,
                        "confidence": float(confidence),
                        "fitness": float(fitness),
                        "semantic_reward": semantic_reward,
                        "weight": float(fitness) * float(confidence) * (0.5 + semantic_reward),
                    }
                )
            valid = [r for r in responses if r["response"].strip() and not r["response"].startswith("[")]
            best = max(valid or responses, key=lambda item: item["weight"]) if responses else {}
            reward = float(best.get("semantic_reward", 0.1))
            losses = []
            if learn:
                self.agent.learn_from_text(prompt, reward=reward, vp_value=current_vp)
                losses = self._train_steps(steps)
                self.agent.record_training_log(
                    "native_chat_turn",
                    stage="authority_native_chat",
                    input_text=prompt[:500],
                    output=str(best.get("response", ""))[:500],
                    reward=reward,
                    score={"semantic_reward": reward, "vp_value": current_vp, "losses": losses},
                )
            if hasattr(self.agent, "conversation"):
                self.agent.conversation.add_message("user", prompt)
                self.agent.conversation.add_message("assistant", str(best.get("response", "")), {"semantic_reward": reward})
            self.cycles += 1
            words = top_words(prompt, 64)
            known, predictable = vocabulary_sets(self.agent)
            anchors = choose_anchors(words, predictable)
            novel = [w for w in words if w not in anchors][:8]
            arbitration = self._arbitrate(words, anchors, "manual_signal")
            wealth = self._informational_wealth(prompt, words, anchors, novel, arbitration, losses, "manual_signal")
            data = {
                "cycle": self.cycles,
                "response": best.get("response", ""),
                "all_responses": responses,
                "semantic_reward": reward,
                "vp_value": current_vp,
                "losses": losses,
                "informational_wealth": wealth,
                "state": self.capability_snapshot(),
            }
            self.last_receipt = observe_receipt("mira_kite_authority_native_chat", data)
            data["receipt"] = self.last_receipt
            data["teacher_note"] = self._teacher_note(
                prompt, words, anchors, novel, arbitration, 0, 0, losses, self.last_receipt
            )
            self._record_event("native_chat", data)
            self._write_status(data)
            return data

    def council_chat(
        self,
        prompt: str,
        learn: bool = True,
        steps: int = 1,
        max_cocoons: int = 8,
        export_after: bool = False,
    ) -> dict[str, Any]:
        with self.lock:
            if not prompt.strip():
                raise ValueError("prompt is required")
            max_cocoons = max(1, min(int(max_cocoons), 24))
            inventory = self.cocoon_inventory()
            roster = []
            for item in inventory.get("cocoons", []) or []:
                if not item.get("runnable"):
                    continue
                path = Path(item.get("path", ""))
                if path == self.cocoon_path:
                    roster.insert(0, item)
                else:
                    roster.append(item)
            roster = roster[:max_cocoons]
            member_results = []
            export_dir = self.output_dir / "butterfly_council"
            if export_after:
                export_dir.mkdir(parents=True, exist_ok=True)
            for item in roster:
                path = Path(item["path"])
                try:
                    module = load_cocoon(str(path))
                    if not hasattr(module, "CocoonAgent"):
                        raise RuntimeError("CocoonAgent missing")
                    agent = module.CocoonAgent()
                    dialogue = summarize_agent_dialogue(agent, prompt, learn=learn, steps=steps)
                    export_path = None
                    if export_after and hasattr(agent, "export_cocoon"):
                        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem or "cocoon")
                        export_path = export_dir / f"{safe_name}_council.py"
                        agent.export_cocoon(str(export_path))
                    member_results.append(
                        {
                            "path": str(path),
                            "name": item.get("display_name") or path.name,
                            "dialogue": dialogue,
                            "export_path": str(export_path) if export_path else None,
                        }
                    )
                except Exception as exc:
                    member_results.append(
                        {
                            "path": str(path),
                            "name": item.get("display_name") or path.name,
                            "error": str(exc),
                        }
                    )

            successful = [member for member in member_results if member.get("dialogue")]
            failed = [member for member in member_results if member.get("error")]
            selected = None
            for member in successful:
                dialogue = member["dialogue"]
                if not selected or float(dialogue.get("selected_weight", 0.0) or 0.0) > float(selected["dialogue"].get("selected_weight", 0.0) or 0.0):
                    selected = member

            central_losses = []
            selected_dialogue = (selected or {}).get("dialogue", {})
            central_lesson = str(selected_dialogue.get("acclimation_lesson") or "")
            if learn and hasattr(self.agent, "learn_from_text"):
                central_reward = float(selected_dialogue.get("semantic_reward", 0.1) or 0.1)
                self.agent.learn_from_text(prompt, reward=central_reward, vp_value=0.0)
                if central_lesson:
                    self.agent.learn_from_text(central_lesson, reward=max(0.72, central_reward), vp_value=0.0)
                central_losses = self._train_steps(steps)
                if hasattr(self.agent, "record_training_log"):
                    try:
                        self.agent.record_training_log(
                            "council_chat_turn",
                            stage="authority_butterfly_council",
                            input_text=prompt[:500],
                            target=central_lesson[:500],
                            output=str(selected_dialogue.get("response", ""))[:500],
                            reward=central_reward,
                            score={
                                "members": len(successful),
                                "failed_members": len(failed),
                                "losses": central_losses,
                                "selected_quality": selected_dialogue.get("selected_quality", {}),
                                "primitive_signals": selected_dialogue.get("primitive_signals", 0),
                                "anchors": selected_dialogue.get("anchors", []),
                                "faculty": selected_dialogue.get("faculty"),
                            },
                        )
                    except Exception:
                        pass

            consensus = selected_dialogue.get("response", "")
            quality_summary = {
                "primitive_signals": sum(int(member.get("dialogue", {}).get("primitive_signals", 0) or 0) for member in successful),
                "member_quality": [
                    {
                        "name": member.get("name"),
                        "selected_weight": member.get("dialogue", {}).get("selected_weight"),
                        "selected_quality": member.get("dialogue", {}).get("selected_quality", {}),
                        "anchors": member.get("dialogue", {}).get("anchors", []),
                        "faculty": member.get("dialogue", {}).get("faculty"),
                        "primitive_signals": member.get("dialogue", {}).get("primitive_signals", 0),
                    }
                    for member in successful
                ],
            }
            data = {
                "prompt": prompt[:1200],
                "learn": bool(learn),
                "steps": int(steps),
                "max_cocoons": max_cocoons,
                "member_count": len(member_results),
                "successful_members": len(successful),
                "failed_members": len(failed),
                "members": member_results,
                "selected_member": selected,
                "consensus": consensus,
                "response": consensus,
                "quality_summary": quality_summary,
                "acclimation_lesson": central_lesson,
                "central_losses": central_losses,
                "state": self.capability_snapshot(),
            }
            self.last_receipt = observe_receipt("mira_kite_authority_council_chat", data)
            data["receipt"] = self.last_receipt
            data["teacher_note"] = self._teacher_note(
                prompt,
                top_words(prompt, 64),
                choose_anchors(top_words(prompt, 64), vocabulary_sets(self.agent)[1]),
                [],
                self._arbitrate(top_words(prompt, 64), choose_anchors(top_words(prompt, 64), vocabulary_sets(self.agent)[1]), "butterfly_council"),
                0,
                0,
                central_losses,
                self.last_receipt,
            )
            self._record_event("council_chat", data)
            self._write_status(data)
            return data

    def native_teach(
        self,
        text: str,
        reward: float = 0.5,
        stage: str = "authority_native_teach",
        target: str = "",
        allow_tool_language: bool = False,
        steps: int = 1,
    ) -> dict[str, Any]:
        with self.lock:
            if not text.strip():
                raise ValueError("text is required")
            tokens = self.agent.learn_from_text(
                text,
                reward=float(reward),
                allow_tool_language=bool(allow_tool_language),
            )
            losses = self._train_steps(steps)
            entry = self.agent.record_training_log(
                "native_teach_text",
                stage=stage,
                input_text=text[:500],
                target=target[:500],
                reward=float(reward),
                score={"losses": losses, "token_count": len(tokens)},
                extra={"allow_tool_language": bool(allow_tool_language)},
            )
            result = self.process_signal(text, steps=0, source="manual_signal")
            data = {
                "tokens": tokens,
                "token_count": len(tokens),
                "losses": losses,
                "training_entry": entry,
                "semantic_result": result,
                "state": self.capability_snapshot(),
            }
            self.last_receipt = observe_receipt("mira_kite_authority_native_teach", data)
            data["receipt"] = self.last_receipt
            self._record_event("native_teach", data)
            self._write_status(data)
            return data

    def native_act(self, state: list[Any], explore: bool = False, action_space_size: int | None = None) -> dict[str, Any]:
        with self.lock:
            arr = np.array(state or [], dtype=np.float32).flatten()
            if arr.size == 0:
                input_dim = self.agent.brains[0].input_dim if self.agent.brains else 30
                arr = np.zeros(input_dim, dtype=np.float32)
            action = self.agent.get_action(arr, explore=bool(explore), action_space_size=action_space_size)
            data = {"action": action, "state_dim": int(arr.size), "explore": bool(explore)}
            self.last_receipt = observe_receipt("mira_kite_authority_native_act", data)
            data["receipt"] = self.last_receipt
            self._record_event("native_act", data)
            return data

    def cheat(self, lr_boost: float = 5.0, multiplier: int = 5, steps: int = 256) -> dict[str, Any]:
        """🦋 COCOON OVERCLOCK - Cheat mode for hyper-acceleration."""
        with self.lock:
            print(f"🚀 [CHEAT] Overclocking Agent (LR Boost: {lr_boost}x)")
            # 1. Boost Learning Rate
            self.agent.learning_rate *= lr_boost
            for opt in self.agent.optimizers:
                for param_group in opt.param_groups:
                    param_group['lr'] = self.agent.learning_rate
            
            # 2. Direct Knowledge Injection
            cheats = {
                "overclock": ["accelerate", "fast", "optimal", "power"],
                "cheat": ["shortcut", "win", "top", "mastery"],
                "victory": ["success", "gain", "fitness", "evolved"]
            }
            for source, targets in cheats.items():
                add_semantic_bridge(self.agent, source, targets)
                for t in targets:
                    strengthen_association(self.agent, source, t, 1.0, "direct_injection")
            
            # 3. Super-Experience Injection
            print(f"💉 [CHEAT] Injecting Super-Experiences (x{multiplier})")
            for action_id, spec in COGNITIVE_FACULTIES.items():
                for _ in range(multiplier):
                    add_cognitive_faculty(self.agent, action_id, spec["name"], spec["triggers"], spec["target"])
            
            for ladder in FIXED_POINT_LADDERS:
                for _ in range(multiplier):
                    add_reasoning_ladder(self.agent, ladder, 5.0)
            
            # 4. Turbo Training
            print(f"🔥 [CHEAT] Starting Turbo-Train ({steps} steps)")
            losses = []
            for _ in range(steps):
                l = self.agent.train_step()
                if l > 0:
                    losses.append(float(l))
            
            self.cycles += 1
            data = {
                "cycle": self.cycles,
                "lr_boost": lr_boost,
                "multiplier": multiplier,
                "steps": steps,
                "avg_loss": sum(losses)/len(losses) if losses else 0.0,
                "state": self.capability_snapshot(),
            }
            self.last_receipt = observe_receipt("mira_kite_authority_cheat", data)
            data["receipt"] = self.last_receipt
            self._record_lesson("cocoon_overclock_victory", {
                "summary": "Hyper-acceleration active. Neural pathways forced into optimal alignment.",
                "cheat_metrics": {"lr": self.agent.learning_rate, "steps": steps, "loss": data["avg_loss"]}
            })
            self._record_event("cheat_overclock", data)
            self._write_status(data)
            return data

    def native_learn(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            state = np.array(payload.get("state", []), dtype=np.float32).flatten()
            next_state = np.array(payload.get("next_state", payload.get("nextState", [])), dtype=np.float32).flatten()
            input_dim = self.agent.brains[0].input_dim if self.agent.brains else 30
            if state.size == 0:
                state = np.zeros(input_dim, dtype=np.float32)
            if next_state.size == 0:
                next_state = np.zeros_like(state)
            action = int(payload.get("action", 0))
            reward = float(payload.get("reward", 0.0))
            done = bool(payload.get("done", False))
            input_tokens = payload.get("input_tokens")
            target_tokens = payload.get("target_tokens")
            vp_value = payload.get("vp_value")
            self.agent.add_experience(
                state,
                action,
                reward,
                next_state,
                done,
                input_tokens=input_tokens,
                target_tokens=target_tokens,
                vp_value=vp_value,
            )
            loss = self.agent.train_step()
            self.cycles += 1
            entry = self.agent.record_training_log(
                "native_rl_transition",
                stage=str(payload.get("stage", "authority_native_learn")),
                reward=reward,
                score={"loss": float(loss or 0.0), "action": action, "done": done},
                extra={"state_dim": int(state.size), "next_state_dim": int(next_state.size)},
            )
            data = {
                "cycle": self.cycles,
                "loss": float(loss or 0.0),
                "step": getattr(self.agent, "training_step", None),
                "entry": entry,
                "state_dim": int(state.size),
                "next_state_dim": int(next_state.size),
                "receipt": None,
                "state": self.capability_snapshot(),
            }
            self.last_receipt = observe_receipt("mira_kite_authority_native_learn", data)
            data["receipt"] = self.last_receipt
            self._record_event("native_learn", data)
            self._write_status(data)
            return data

    def native_snapshot(self, full: bool = False) -> dict[str, Any]:
        with self.lock:
            if hasattr(self.module, "build_live_snapshot"):
                snapshot = self.module.build_live_snapshot(self.agent)
            else:
                snapshot = self.capability_snapshot()
            if full:
                return snapshot
            return {
                "snapshot_time": snapshot.get("snapshot_time"),
                "mode": snapshot.get("mode"),
                "num_organisms": snapshot.get("num_organisms"),
                "vocab_size": snapshot.get("vocab_size"),
                "knowledge_web_concepts": snapshot.get("knowledge_web_concepts"),
                "training_step": snapshot.get("training_step"),
                "reward_history_tail": snapshot.get("reward_history", [])[-12:],
                "training_logs_tail": snapshot.get("training_logs", [])[-12:],
                "capabilities": snapshot.get("capabilities", {}),
            }

    def native_save(self, output_dir: str = "mira_kite_native_state") -> dict[str, Any]:
        with self.lock:
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", output_dir or "mira_kite_native_state")
            path = self.output_dir / safe
            if hasattr(self.module, "save_runtime_state"):
                data = self.module.save_runtime_state(self.agent, str(path))
            else:
                path.mkdir(parents=True, exist_ok=True)
                data = {"saved": False, "output_dir": str(path), "error": "save_runtime_state unavailable"}
            self.last_receipt = observe_receipt("mira_kite_authority_native_save", data)
            data["receipt"] = self.last_receipt
            self._record_event("native_save", data)
            self._write_status(data)
            return data

    def native_dreamer_observe(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            observation = {
                "timestamp": time.time(),
                "game": payload.get("game", "unknown"),
                "observation": payload.get("observation", payload),
                "reward": payload.get("reward"),
                "done": bool(payload.get("done", False)),
            }
            if not hasattr(self.agent, "dreamer_observations"):
                self.agent.dreamer_observations = []
            self.agent.dreamer_observations.append(observation)
            self.agent.dreamer_observations = self.agent.dreamer_observations[-200:]
            data = {
                "accepted": True,
                "observation_count": len(self.agent.dreamer_observations),
                "proposal_hint": "Call /api/native/dreamer/propose or /dreamer/propose.",
                "observation": observation,
            }
            self.last_receipt = observe_receipt("mira_kite_authority_dreamer_observe", data)
            data["receipt"] = self.last_receipt
            self._record_event("native_dreamer_observe", data)
            return data

    def native_dreamer_propose(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            observations = getattr(self.agent, "dreamer_observations", [])[-20:]
            rewards = [o.get("reward") for o in observations if isinstance(o.get("reward"), (int, float))]
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            facilities = self.native_facilities()
            proposals = []
            if avg_reward < 0.2:
                proposals.append(
                    {
                        "lane": "sphere",
                        "endpoint": "/api/native/sphere_burst",
                        "args": {"frames": 180, "headless": True, "train": True, "balls": 1, "misses": 3},
                        "reason": "Low recent reward; use short headless sphere training for dense feedback.",
                    }
                )
            else:
                proposals.append(
                    {
                        "lane": "gym",
                        "endpoint": "/api/native/gym_burst",
                        "args": {"env": payload.get("env", "CartPole-v1"), "episodes": 1, "learn": True},
                        "reason": "Reward is nonzero; try a tiny Gym loop if dependencies are available.",
                    }
                )
            proposals.append(
                {
                    "lane": "save",
                    "endpoint": "/api/native/save",
                    "args": {"output_dir": "dreamer_proposed_save"},
                    "reason": "Persist live vocabulary/knowledge after runtime learning.",
                }
            )
            proposals.append(
                {
                    "lane": "quine_drill",
                    "endpoint": "/api/quine_drill",
                    "args": {"limit": 8, "steps": 2},
                    "reason": "Reinforce safe syntax and fixed-point nesting concepts.",
                }
            )
            return {
                "avg_reward": avg_reward,
                "observation_count": len(observations),
                "proposals": proposals,
                "readiness": facilities.get("game_lanes", {}),
            }

    def native_sphere_burst(
        self,
        frames: int = 120,
        balls: int = 1,
        misses: int = 3,
        train: bool = False,
        seed: int | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            if not hasattr(self.module, "SphereArena"):
                raise RuntimeError("SphereArena is not available in this cocoon")
            frames = max(1, min(int(frames), 1200))
            balls = max(1, min(int(balls), 5))
            misses = max(1, min(int(misses), 20))
            arena = self.module.SphereArena(
                self.agent,
                max_misses=misses,
                headless=True,
                seed=seed,
                num_balls=balls,
                enable_training=bool(train),
                verbose=False,
            )
            last_info: dict[str, Any] = {}
            observations = {}
            rewards = {}
            done = False
            render_tail: list[dict[str, Any]] = []

            def clean_vec(value: Any) -> list[float]:
                try:
                    return [float(item) for item in value]
                except Exception:
                    return []

            def sphere_render_state() -> dict[str, Any]:
                organisms = []
                for org in getattr(arena, "organisms", []) or []:
                    organisms.append(
                        {
                            "idx": int(getattr(org, "idx", len(organisms))),
                            "theta": float(getattr(org, "theta", 0.0)),
                            "phi": float(getattr(org, "phi", 0.0)),
                            "position": clean_vec(getattr(org, "position", ())),
                            "alive": bool(getattr(org, "alive", True)),
                            "catches": int(getattr(org, "catches", 0)),
                            "is_commander": bool(getattr(org, "is_commander", False)),
                            "color": clean_vec(getattr(org, "color", ())),
                        }
                    )
                balls_state = []
                for ball in getattr(arena, "balls", []) or []:
                    balls_state.append(
                        {
                            "position": clean_vec(getattr(ball, "position", ())),
                            "velocity": clean_vec(getattr(ball, "velocity", ())),
                            "active": bool(getattr(ball, "active", True)),
                            "bounces": int(getattr(ball, "bounces", 0)),
                            "last_catcher": getattr(ball, "last_catcher", None),
                        }
                    )
                return {
                    "frame": int(getattr(arena, "total_frames", 0)),
                    "organisms": organisms,
                    "balls": balls_state,
                    "commander": getattr(arena, "current_commander", None),
                    "active_command": clean_vec(getattr(arena, "active_command", ())),
                    "catches": int(getattr(arena, "collective_catches", 0)),
                    "misses": int(getattr(arena, "collective_misses", 0)),
                    "streak": int(getattr(arena, "current_streak", 0)),
                    "best_streak": int(getattr(arena, "best_streak", 0)),
                }

            sample_every = max(1, frames // 48)
            for frame_idx in range(frames):
                observations, rewards, done, last_info = arena.step()
                if frame_idx % sample_every == 0 or done:
                    render_tail.append(sphere_render_state())
                if train:
                    for org in arena.organisms:
                        if org.idx in arena.prev_observations and org.idx in rewards:
                            arena._add_experience(
                                org.idx,
                                arena.prev_observations[org.idx],
                                arena.prev_actions.get(org.idx, 0),
                                rewards[org.idx],
                                observations.get(org.idx, np.zeros(30)),
                                done,
                            )
                if done:
                    break
            losses = self._train_steps(1 if train else 0)
            self.cycles += 1
            data = {
                "cycle": self.cycles,
                "frames_requested": frames,
                "frames_run": int(arena.total_frames),
                "balls": balls,
                "misses_limit": misses,
                "train": bool(train),
                "collective_catches": int(arena.collective_catches),
                "collective_misses": int(arena.collective_misses),
                "best_streak": int(arena.best_streak),
                "rewards": {str(k): float(v) for k, v in rewards.items()},
                "done": bool(done),
                "info": last_info,
                "render_state": sphere_render_state(),
                "render_tail": render_tail[-48:],
                "losses": losses,
                "state": self.capability_snapshot(),
            }
            self.last_receipt = observe_receipt("mira_kite_authority_sphere_burst", data)
            data["receipt"] = self.last_receipt
            self._record_event("native_sphere_burst", data)
            self._write_status(data)
            return data

    def native_gym_burst(self, env: str = "CartPole-v1", episodes: int = 1, learn: bool = True) -> dict[str, Any]:
        with self.lock:
            if not hasattr(self.module, "GymRunner"):
                raise RuntimeError("GymRunner is not available in this cocoon")
            if not (module_available("gymnasium") or module_available("gym")):
                raise RuntimeError("Gym/Gymnasium is not installed. Install with: pip install gymnasium")
            episodes = max(1, min(int(episodes), 10))
            runner = self.module.GymRunner(self.agent)
            buf = io.StringIO()
            started = time.time()
            with contextlib.redirect_stdout(buf):
                result = runner.run(str(env or "CartPole-v1"), episodes=episodes, render=False, learn=bool(learn))
            self.cycles += 1
            losses = self._train_steps(1 if learn else 0)
            data = {
                "cycle": self.cycles,
                "env": str(env or "CartPole-v1"),
                "episodes": episodes,
                "learn": bool(learn),
                "duration_seconds": time.time() - started,
                "runner_result": result,
                "stdout": buf.getvalue()[-4000:],
                "losses": losses,
                "state": self.capability_snapshot(),
            }
            self.last_receipt = observe_receipt("mira_kite_authority_gym_burst", data)
            data["receipt"] = self.last_receipt
            self._record_event("native_gym_burst", data)
            self._write_status(data)
            return data

    def native_game_recipe(self, lane: str = "all") -> dict[str, Any]:
        lane = (lane or "all").lower()
        package_dir = Path(__file__).resolve().parent
        recipes = {
            "sphere": {
                "app_endpoint": "/api/native/sphere_burst",
                "cli": "python cocoon_cognition_agency.py --mode sphere --headless --balls 1 --misses 3 --train",
                "bounded": True,
                "notes": "Native in-process app hook is safest on Termux.",
            },
            "gym": {
                "app_endpoint": "/api/native/gym_burst",
                "cli": "python cocoon_cognition_agency.py --mode gym --env CartPole-v1 --episodes 1",
                "bounded": True,
                "dependency": "gymnasium or gym",
            },
            "drone": {
                "cli": f"cd {package_dir} && python cocoon_drone_adapter.py --mode free_fly --time 30",
                "modes": [
                    "free_fly",
                    "formation",
                    "pursuit",
                    "tag_battle",
                    "zone_control",
                    "capture_flag",
                    "survival",
                    "escort",
                ],
                "bounded": "adapter supports --time; run manually or after dependency approval",
                "dependency": "matplotlib, PyFlyt, pybullet, pettingzoo, numba",
            },
            "tmrl": {
                "cli": f"cd {package_dir} && python cocoon_tmrl_adapter.py --cocoon ../cocoon_cognition_agency.py --drive --episodes 1",
                "bounded": "external TrackMania/OpenPlanet runtime required",
                "dependency": "operator disabled; do not pursue",
            },
            "link": {
                "cli": "python cocoon_cognition_agency.py --mode link --hatch ws://localhost:9000 --name cocoon",
                "bounded": False,
                "dependency": "websockets and relay server",
            },
            "serve": {
                "native_mirror": "this authority app mirrors the useful server endpoints without Flask",
                "original_cli": "python cocoon_cognition_agency.py --mode serve --port 8080",
                "dependency": "flask",
            },
        }
        selected = recipes if lane == "all" else {lane: recipes.get(lane, {"error": "unknown lane"})}
        return {"lane": lane, "recipes": selected, "readiness": self.native_facilities().get("game_lanes", {})}

    def facility_map(self) -> dict[str, Any]:
        facilities = self.native_facilities()
        readiness = facilities.get("game_lanes", {})
        nodes = [
            {"id": "trainer", "label": "Trainer", "type": "user", "status": "ready"},
            {"id": "authority", "label": "Authority", "type": "router", "status": "ready"},
            {"id": "mira", "label": "Mira", "type": "organism", "status": "ready"},
            {"id": "kite", "label": "Kite", "type": "organism", "status": "ready"},
            {"id": "chat", "label": "Chat", "type": "facility", "status": "ready"},
            {"id": "council", "label": "Council", "type": "facility", "status": "ready"},
            {"id": "teach", "label": "Teach", "type": "facility", "status": "ready"},
            {"id": "drills", "label": "Drills", "type": "facility", "status": "ready"},
            {"id": "sphere", "label": "Sphere", "type": "game", "status": "ready" if readiness.get("sphere", {}).get("ready") else "blocked"},
            {"id": "gym", "label": "Gym", "type": "game", "status": "ready" if readiness.get("gym", {}).get("ready") else "needs dependency"},
            {"id": "drone", "label": "Drone", "type": "game", "status": "ready" if all(readiness.get("drone", {}).get("ready_files", {}).values()) else "needs package"},
            {"id": "tmrl", "label": "TMRL", "type": "external", "status": "disabled by operator"},
            {"id": "link", "label": "Link", "type": "external", "status": "ready" if readiness.get("link", {}).get("ready") else "needs relay"},
            {"id": "cascade", "label": "Cascade", "type": "ledger", "status": "ready" if self.monitor is not None else "limited"},
            {"id": "quine", "label": "Quine", "type": "syntax", "status": "ready"},
        ]
        edges = [
            {"from": "trainer", "to": "authority", "label": "input"},
            {"from": "authority", "to": "chat", "label": "camp"},
            {"from": "authority", "to": "council", "label": "fan-out"},
            {"from": "authority", "to": "teach", "label": "academy"},
            {"from": "authority", "to": "drills", "label": "moves"},
            {"from": "authority", "to": "sphere", "label": "game"},
            {"from": "authority", "to": "gym", "label": "game"},
            {"from": "authority", "to": "drone", "label": "arena"},
            {"from": "authority", "to": "tmrl", "label": "external"},
            {"from": "authority", "to": "link", "label": "relay"},
            {"from": "authority", "to": "cascade", "label": "receipt"},
            {"from": "authority", "to": "quine", "label": "syntax"},
            {"from": "chat", "to": "mira", "label": "response"},
            {"from": "chat", "to": "kite", "label": "response"},
            {"from": "council", "to": "mira", "label": "consult"},
            {"from": "council", "to": "kite", "label": "consult"},
            {"from": "teach", "to": "mira", "label": "association"},
            {"from": "teach", "to": "kite", "label": "action"},
            {"from": "drills", "to": "mira", "label": "anchors"},
            {"from": "drills", "to": "kite", "label": "sequence"},
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "tape_path": self.tape_path,
            "graph_stats": self.graph.get_stats() if self.graph is not None else {},
            "recent_events": self.event_log[-20:],
        }

    def wealth_surface(self) -> dict[str, Any]:
        state = self.capability_snapshot()
        facility_map = self.facility_map()
        facilities = self.native_facilities()
        manifest = capability_manifest()
        recent_events = self.event_log[-24:]
        recent_lessons = self.education_log[-12:]

        receipts: list[dict[str, Any]] = []
        for event in recent_events:
            data = event.get("data", {}) if isinstance(event, dict) else {}
            receipt = data.get("receipt") or data.get("last_receipt")
            if not receipt and isinstance(data.get("state"), dict):
                receipt = data["state"].get("last_receipt")
            if receipt:
                receipts.append(
                    {
                        "time": event.get("time"),
                        "kind": event.get("kind"),
                        "receipt": str(receipt),
                    }
                )

        cascade = {
            "available": self.monitor is not None or self.adapter is not None,
            "tape_path": str(self.tape_path) if self.tape_path else None,
            "graph_stats": self.graph.get_stats() if self.graph is not None else {},
            "errors": self.cascade_errors[-8:],
            "latest_receipts": receipts[-12:],
        }
        if self.metrics is not None:
            try:
                cascade["metric_health"] = self.metrics.health_summary()
            except Exception as exc:
                cascade["metric_health_error"] = str(exc)

        quine_ladders = quinesmith_ladders()
        diagnostics = {
            "capability_count": len(manifest.get("capabilities", [])),
            "safe_phone_count": len([cap for cap in manifest.get("capabilities", []) if cap.get("safe_on_phone")]),
            "mutating_count": len([cap for cap in manifest.get("capabilities", []) if cap.get("mutates")]),
            "external_count": len([cap for cap in manifest.get("capabilities", []) if cap.get("risk_class") == "external"]),
            "dependency_probe": facilities.get("dependency_probe", {}),
            "dependency_errors": facilities.get("dependency_errors", {}),
        }
        mcp_stdio = {
            "transport": "stdio JSON-RPC",
            "command": "./cocoon mcp",
            "tools": [
                "doctor",
                "list_cocoons",
                "persistence",
                "capabilities",
                "run_capability",
                "start_job",
                "job_status",
                "job_logs",
                "stop_job",
                "eval",
                "eval_all",
                "council",
            ],
            "note": "Authority HTTP routes and MCP stdio tools share the same capability fabric; the browser polls read-only telemetry here.",
        }
        wealth = state.get("last_informational_wealth") or self.last_wealth or {}
        return {
            "updated": time.time(),
            "state": state,
            "diagnostics": diagnostics,
            "cascade": cascade,
            "quine": {
                "available": bool(quine_ladders),
                "ladder_count": len(quine_ladders),
                "sample_ladders": quine_ladders[:8],
                "boundary": "syntax/fixed-point drills only; generated commands are never executed by Authority",
            },
            "mcp_stdio": mcp_stdio,
            "facility_map": facility_map,
            "capability_manifest": {
                "groups": sorted({cap.get("group", "") for cap in manifest.get("capabilities", []) if cap.get("group")}),
                "capabilities": manifest.get("capabilities", [])[:80],
            },
            "events": recent_events,
            "lessons": recent_lessons,
            "informational_wealth": wealth,
            "feed_reports": self.feed_reports[-8:],
            "summary": {
                "active": state.get("active_cocoon_name"),
                "vocabulary_size": state.get("vocabulary_size"),
                "relations": state.get("knowledge_relations"),
                "cycles": state.get("cycles"),
                "last_receipt": state.get("last_receipt"),
                "events": len(self.event_log),
                "lessons": len(self.education_log),
            },
        }

    def complete_facility_manual(self) -> dict[str, Any]:
        facilities = self.native_facilities()
        deps = facilities.get("dependency_probe", {})

        def card(
            key: str,
            name: str,
            plain: str,
            teaches: str,
            endpoint: str,
            run_facility: str,
            payload: dict[str, Any] | None = None,
            status: str = "ready",
            needs: list[str] | None = None,
            deep: str = "",
        ) -> dict[str, Any]:
            return {
                "key": key,
                "name": name,
                "plain": plain,
                "teaches": teaches,
                "endpoint": endpoint,
                "run_facility": run_facility,
                "payload": payload or {},
                "status": status,
                "needs": needs or [],
                "deep": deep,
            }

        cards = [
            card("autopilot", "Whole Facility", "Press one button. It scouts, drills, associates, verifies, receipts, and summarizes.", "Broad chat and reasoning growth from public signals plus known anchors.", "/api/autopilot", "autopilot", {"depth": 1}),
            card("chat", "Camp Chat", "Talk normally to the active cocoon.", "Conversation turns, semantic reward, response selection, and memory.", "/api/native/chat", "chat", {"prompt": "hello cocoon", "learn": True, "steps": 1}, deep="generate_response + semantic reward + optional learn_from_text"),
            card("council", "Butterfly Council", "Fan one prompt out across every runnable cocoon and compare the answers.", "Parallel consultation, optional batch learning, and central response selection.", "/api/council", "council", {"prompt": "hello cocoon council", "learn": True, "steps": 1, "max_cocoons": 8, "export_after": False}, deep="Loads each runnable cocoon, aggregates the replies, and can export trained checkpoints."),
            card("teach", "Teach Lesson", "Give it one lesson or idea.", "New words, token links, knowledge web growth, and training logs.", "/api/native/teach", "teach", {"text": "signal means input that carries meaning", "reward": 0.8, "steps": 1}, deep="learn_from_text + record_training_log"),
            card("follow", "Follow Along", "Echo, cue, chain, role, or chant with it.", "Short word sequencing and cue-trigger association.", "/api/follow_along", "follow", {"mode": "cue", "seed": "signal meaning reason verify", "rounds": 6, "steps": 2}),
            card("relate", "Association Move", "Connect one phrase to another.", "Manual semantic relation and supervised pair links.", "/api/relate", "relate", {"source": "signal", "target": "meaning"}),
            card("govern", "Judge Move", "Route a signal, then verify it.", "Faculty arbitration and verification habits.", "/api/govern", "govern", {"text": "verify cause effect before response", "steps": 2}),
            card("feed", "Internet Scout", "Pull public feeds and convert them into lessons.", "Novel terms bridged to known anchors; feed items scored before training.", "/api/feed", "feed", {"url": DEFAULT_SCOUT_FEEDS[0], "max_items": 2, "steps": 2}, needs=["network access from Termux/Python"]),
            card("quine", "Syntax Forge", "Practice safe fixed-point syntax patterns.", "Template, brace, quote, nesting, and verification concepts.", "/api/quine_drill", "quine", {"limit": 8, "steps": 2}, deep="Quinesmith drills only; no generated commands are executed."),
            card("sphere", "Sphere Arena", "Tiny headless action game.", "Action grounding, reward, catches/misses, experience buffer.", "/api/native/sphere_burst", "sphere", {"frames": 120, "balls": 1, "misses": 3, "train": False}, deep="SphereArena.step in-process"),
            card("gym", "Gym Arena", "Run a small ML game if Gym is installed.", "Classic RL transitions and rewards.", "/api/native/gym_burst", "gym", {"env": "CartPole-v1", "episodes": 1, "learn": True}, status="ready" if deps.get("gymnasium") or deps.get("gym") else "needs dependency", needs=[] if deps.get("gymnasium") or deps.get("gym") else ["pip install gymnasium"]),
            card("drone", "Drone Arena", "Run a bounded drone adapter session when requisites are present.", "Robotics-style action mapping and arena rewards.", "/api/run_facility", "drone", {"mode": "free_fly", "time": 5, "drones": 2, "timeout": 25}, status="ready/files present" if all(deps.get(item) for item in ["matplotlib", "PyFlyt", "pybullet", "pettingzoo", "numba"]) else "needs requisite", needs=[] if all(deps.get(item) for item in ["matplotlib", "PyFlyt", "pybullet", "pettingzoo", "numba"]) else ["matplotlib", "PyFlyt", "pybullet", "pettingzoo", "numba"], deep="cocoon_drone_adapter.py; external subprocess with timeout"),
            card("tmrl", "Track Racer Doctor", "Disabled historical diagnostic.", "No active pursuit by operator decision.", "/api/run_facility", "tmrl_doctor", {"timeout": 25}, status="disabled by operator", needs=["do not pursue"]),
            card("link", "Relay Link", "Connect to other cocoons when a hatch relay exists.", "Networking, challenges, remote chat.", "manual recipe", "link", {}, status="needs relay", needs=["websockets", "CocoonHatch URL"]),
            card("dreamer", "Dreamer Scout", "Record observations and ask what to do next.", "Proposal-making from recent reward observations.", "/api/native/dreamer/propose", "dreamer_propose", {}),
            card("act", "Action Head", "Ask the cocoon for an action from numbers.", "Action selection and exploration.", "/api/native/act", "act", {"state": [], "explore": False}),
            card("learn", "Reward Lesson", "Give state, action, reward, next state.", "ExperienceBuffer and train_step.", "/api/native/learn", "learn", {"state": [], "action": 0, "reward": 0.1, "next_state": [], "done": False}),
            card("vocab", "Word Bag", "See known words.", "Vocabulary awareness.", "/api/native/vocab", "vocab", {"limit": 120}),
            card("logs", "Training Scroll", "See recent learning events.", "Audit trail and coach scoring.", "/api/native/training/logs", "logs", {"limit": 50}),
            card("snapshot", "Memory Snapshot", "See live symbolic state.", "Vocabulary, knowledge web, conversation, atomic language, logs.", "/api/native/snapshot", "snapshot", {}),
            card("save", "Save State", "Write live runtime JSON files.", "Durable clone/snapshot workflow.", "/api/native/save", "save", {"output_dir": "facility_save"}),
            card("export", "Export Cocoon", "Bake current learning into a new Python cocoon.", "Durable standalone checkpoint.", "/api/export", "export", {"name": "facility_checkpoint.py"}),
            card("capabilities", "Capability Sheet", "Show the machine-readable contract.", "What the cocoon can do and what is mirrored.", "/capabilities", "capabilities", {}),
            card("cascade", "Receipt Ledger", "Proof cards for what happened.", "Receipts, causation graph, metrics, event tape.", "/api/facility_map", "map", {}, deep="Cascade Monitor + SymbioticAdapter + CausationGraph + viz tape"),
            card("map", "Facility Map", "See every station and route.", "How trainer input flows into facilities and organisms.", "/api/facility_map", "map", {}),
        ]

        blueprints = {
            "brain": "Two organisms with neural action heads, language heads, attention, and Hopfield-style pattern memory.",
            "language": "Words become tokens; known low-ID words are easiest for the language head; high-ID words can still live as symbolic concepts.",
            "atomic_language": "Concept atoms can be created or reinforced and linked by association strength.",
            "knowledge_web": "Concepts and relations are the main semantic map.",
            "conversation": "Chat turns accumulate context and training traces.",
            "vp_runtime": "Vigilance/plasticity regulation watches pressure and adaptation.",
            "experience_buffer": "Game/tool rewards become replayable state-action-reward lessons.",
            "cascade_lattice": "Events become receipts, graph links, metrics, and playback tape entries.",
            "quine_dynamics": "Safe syntax and fixed-point drills help it learn structure without running generated code.",
        }
        protocol_defaults = {
            "use": "Press Run for a small safe pass. Press Deepen for a staged training ladder.",
            "hardening": [
                "Keep steps bounded on phone.",
                "Prefer known-anchor bridging before new-word expansion.",
                "Receipt each meaningful cycle.",
                "Export after good training batches.",
            ],
            "telemetry": ["receipt", "teacher_note", "losses", "anchors", "novel", "state"],
        }
        facility_protocols = {}
        for item in cards:
            key = item["key"]
            facility_protocols[key] = {
                **protocol_defaults,
                "ladder": [
                    f"Wake {item['name']} with a tiny safe run.",
                    "Bind new signal words to known anchors.",
                    "Run a repetition/cue drill so the pattern sticks.",
                    "Verify with Cascade receipt and teacher note.",
                    "Save/export if the result is useful.",
                ],
                "schema": {
                    "input": item["payload"],
                    "output": {
                        "state": "live cocoon state after the run",
                        "teacher_note": "plain-language summary",
                        "receipt": "Cascade receipt CID when available",
                    },
                },
            }
        return {
            "motto": "Press big button, watch battle card, chat, repeat.",
            "simple_loop": [
                "Autopilot scouts.",
                "Known words anchor new words.",
                "Drills make the pattern repeatable.",
                "Games/tools add reward signals.",
                "Cascade receipts prove the run.",
                "Export saves the creature.",
            ],
            "cards": cards,
            "blueprints": blueprints,
            "facility_protocols": facility_protocols,
            "dependency_probe": deps,
            "facility_map": self.facility_map(),
        }

    def harden_facilities(self) -> dict[str, Any]:
        with self.lock:
            manual = self.complete_facility_manual()
            state = self.capability_snapshot()
            deps = manual.get("dependency_probe", {})
            checks = []

            def add_check(name: str, ok: bool, plain: str, fix: str = "") -> None:
                checks.append({"name": name, "ok": bool(ok), "plain": plain, "fix": fix})

            add_check("cocoon_loaded", bool(self.agent.brains), f"{self.cocoon_name} has loaded brains.", "Reload the cocoon file.")
            add_check("chat_ready", hasattr(self.agent, "generate_response"), "Basic chat can run.", "Use the cocoon Python export with CocoonAgent.")
            add_check("learning_ready", hasattr(self.agent, "learn_from_text") and hasattr(self.agent, "train_step"), "Text learning and neural train steps exist.", "Use a full Python cocoon, not ONNX-only.")
            add_check("knowledge_web", isinstance(self.agent.knowledge_web, dict), "Semantic relation web exists.", "Re-export from a full cocoon package.")
            add_check("cascade_ready", self.monitor is not None or self.adapter is not None, "Cascade observe/receipt support is available.", "Install cascade-lattice.")
            add_check("tape_ready", self.tape_path is not None, "Cascade tape visualization log is writable.", "Check output directory permissions.")
            add_check("gym_ready", bool(deps.get("gymnasium") or deps.get("gym")), "Gym facility can run if true.", "Install gymnasium for classic ML games.")
            add_check("drone_files", all(manual["facility_map"]["nodes"][9].get("status", "") != "needs package" for _ in [0]), "Drone adapter files are present.", "Unpack the cocoon package.")
            add_check("language_capacity", state.get("language_head_limit", 0) > 1000, "Language head has predictable-token capacity.", "Use symbolic bridging for high-ID words.")

            issues = [c for c in checks if not c["ok"]]
            data = {
                "mode": "facility_hardening",
                "checks": checks,
                "issues": issues,
                "simple_summary": "Core trainer is usable." if not issues else f"{len(issues)} things need attention.",
                "next_moves": [
                    "Use Autopilot depth 1 for phone-friendly training.",
                    "Use Follow Along after chat gets repetitive.",
                    "Export checkpoints after useful sessions.",
                ],
                "state": state,
            }
            self.last_receipt = observe_receipt("mira_kite_authority_hardening", data)
            data["receipt"] = self.last_receipt
            self._record_event("harden_facilities", data)
            self._write_status(data)
            return data

    def deepen_facility(self, facility: str = "autopilot", depth: int = 2) -> dict[str, Any]:
        with self.lock:
            facility = (facility or "autopilot").lower()
            depth = max(1, min(int(depth), 4))
            mission = f"deepen {facility} for {self.cocoon_name} using known anchors, simple words, verification, and useful chat"
            stages = []

            stages.append({"stage": "wake", "result": self.process_signal(mission, steps=depth, source="autopilot")})
            stages.append({"stage": "cue_drill", "result": self.follow_along_game("cue", mission, rounds=4 + depth, steps=max(0, depth - 1))})
            stages.append({"stage": "chain_drill", "result": self.follow_along_game("chain", mission, rounds=4 + depth, steps=max(0, depth - 1))})

            if facility in {"autopilot", "feed", "internet", "machine_learning", "robotics"}:
                for feed in DEFAULT_SCOUT_FEEDS[: min(2, depth)]:
                    try:
                        stages.append({"stage": f"scout:{feed}", "result": self.train_feed(feed, max_items=1, steps=depth)})
                    except Exception as exc:
                        stages.append({"stage": f"scout:{feed}", "error": str(exc)})
            if facility in {"quine", "autopilot", "tools", "syntax"}:
                stages.append({"stage": "quine", "result": self.quine_drill(limit=4 + depth, steps=max(0, depth - 1))})
            if facility in {"sphere", "gym", "drone", "robotics", "tools"}:
                try:
                    stages.append({"stage": "sphere_grounding", "result": self.native_sphere_burst(frames=90, balls=1, misses=3, train=True)})
                except Exception as exc:
                    stages.append({"stage": "sphere_grounding", "error": str(exc)})

            losses = []
            bridges = 0
            pairs = 0
            for stage in stages:
                result = stage.get("result")
                if isinstance(result, dict):
                    losses.extend([float(x) for x in result.get("losses", []) if isinstance(x, (int, float))])
                    bridges += int(result.get("bridges", 0) or 0)
                    pair_value = result.get("pairs", 0) or 0
                    pairs += len(pair_value) if isinstance(pair_value, list) else int(pair_value)
            words = top_words(mission, 64)
            known, predictable = vocabulary_sets(self.agent)
            anchors = choose_anchors(words, predictable)
            novel = [w for w in words if w not in anchors][:8]
            arbitration = self._arbitrate(words, anchors, "autopilot")
            wealth = self._informational_wealth(mission, words, anchors, novel, arbitration, losses, "autopilot")
            data = {
                "mode": "facility_deepening",
                "facility": facility,
                "depth": depth,
                "plain_summary": f"Deepened {facility} through signal, cue, chain, scout, and verification stages.",
                "stages": [
                    {
                        "stage": stage.get("stage"),
                        "ok": "error" not in stage,
                        "error": stage.get("error"),
                        "result": self._summarize_nested(stage.get("result")),
                    }
                    for stage in stages
                ],
                "bridges": bridges,
                "pairs": pairs,
                "losses": losses[-12:],
                "anchors": anchors,
                "novel": novel,
                "informational_wealth": wealth,
                "teacher_note": self._teacher_note(mission, words, anchors, novel, arbitration, bridges, pairs, losses, self.last_receipt),
                "state": self.capability_snapshot(),
            }
            self.last_receipt = observe_receipt("mira_kite_authority_deepen_facility", data)
            data["receipt"] = self.last_receipt
            data["teacher_note"]["receipt"] = self.last_receipt
            self._record_event("deepen_facility", data)
            self._write_status(data)
            return data

    def run_external_facility(self, facility: str, payload: dict[str, Any]) -> dict[str, Any]:
        package_dir = Path(__file__).resolve().parent
        timeout = max(5, min(int(payload.get("timeout", 25)), 60))
        if facility == "drone":
            mode = str(payload.get("mode", "free_fly"))
            seconds = max(1, min(float(payload.get("time", 5)), 20.0))
            drones = max(1, min(int(payload.get("drones", 2)), 8))
            cmd = [
                sys.executable,
                "cocoon_drone_adapter.py",
                "--mode",
                mode,
                "--time",
                str(seconds),
                "--drones",
                str(drones),
                "--no-plot",
            ]
        elif facility == "tmrl_doctor":
            cmd = [sys.executable, "cocoon_tmrl_adapter.py", "--doctor", "--cocoon", "../cocoon_cognition_agency.py"]
        else:
            raise ValueError(f"external facility not supported: {facility}")
        started = time.time()
        proc = subprocess.run(
            cmd,
            cwd=str(package_dir),
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        data = {
            "facility": facility,
            "cmd": cmd,
            "returncode": proc.returncode,
            "duration_seconds": time.time() - started,
            "stdout": proc.stdout[-5000:],
            "stderr": proc.stderr[-5000:],
        }
        self.last_receipt = observe_receipt("mira_kite_authority_external_facility", data)
        data["receipt"] = self.last_receipt
        self._record_event("external_facility", data)
        return data

    def run_facility(self, facility: str, payload: dict[str, Any]) -> dict[str, Any]:
        facility = (facility or "health").lower()
        if facility == "autopilot":
            feeds = payload.get("feeds")
            if isinstance(feeds, str):
                feeds = [line.strip() for line in feeds.splitlines() if line.strip()]
            return self.autopilot_train(
                str(payload.get("mission", payload.get("text", ""))),
                feeds if isinstance(feeds, list) else None,
                int(payload.get("depth", 1)),
                bool(payload.get("include_games", False)),
                bool(payload.get("save_after", False)),
            )
        if facility == "health":
            return self.native_health()
        if facility in {"info", "state"}:
            return self.capability_snapshot()
        if facility == "chat":
            return self.native_chat(str(payload.get("prompt", payload.get("message", "hello"))), bool(payload.get("learn", True)), int(payload.get("steps", 1)))
        if facility == "council":
            return self.council_chat(
                str(payload.get("prompt", payload.get("message", "hello cocoon council"))),
                bool(payload.get("learn", True)),
                int(payload.get("steps", 1)),
                int(payload.get("max_cocoons", 8)),
                bool(payload.get("export_after", False)),
            )
        if facility == "teach":
            return self.native_teach(str(payload.get("text", "")), float(payload.get("reward", 0.5)), str(payload.get("stage", "facility_teach")), str(payload.get("target", "")), bool(payload.get("allow_tool_language", False)), int(payload.get("steps", 1)))
        if facility == "relate":
            return self.relate(str(payload.get("source", "signal")), str(payload.get("target", "meaning")), int(payload.get("steps", 2)))
        if facility == "govern":
            return self.govern(str(payload.get("text", "verify signal meaning before response")), int(payload.get("steps", 2)))
        if facility == "act":
            size = payload.get("action_space_size")
            return self.native_act(payload.get("state", []), bool(payload.get("explore", False)), int(size) if size is not None else None)
        if facility == "learn":
            return self.native_learn(payload)
        if facility == "vocab":
            return self.native_vocab(int(payload.get("limit", 120)))
        if facility == "curriculum":
            return self.native_curriculum()
        if facility == "logs":
            return self.native_training_logs(int(payload.get("limit", 50)))
        if facility == "score":
            return self.native_curriculum_score(payload)
        if facility == "snapshot":
            return self.native_snapshot(full=bool(payload.get("full", False)))
        if facility == "save":
            return self.native_save(str(payload.get("output_dir", "facility_save")))
        if facility == "export":
            return self.export(payload.get("name", "facility_checkpoint.py"))
        if facility == "capabilities":
            return self.native_capabilities()
        if facility == "dreamer_observe":
            return self.native_dreamer_observe(payload)
        if facility == "dreamer_propose":
            return self.native_dreamer_propose(payload)
        if facility == "sphere":
            return self.native_sphere_burst(int(payload.get("frames", 120)), int(payload.get("balls", 1)), int(payload.get("misses", 3)), bool(payload.get("train", False)), payload.get("seed"))
        if facility == "gym":
            return self.native_gym_burst(str(payload.get("env", "CartPole-v1")), int(payload.get("episodes", 1)), bool(payload.get("learn", True)))
        if facility == "follow":
            return self.follow_along_game(str(payload.get("mode", "cue")), str(payload.get("seed", "")), int(payload.get("rounds", 6)), int(payload.get("steps", 2)))
        if facility == "quine":
            return self.quine_drill(int(payload.get("limit", 8)), int(payload.get("steps", 2)))
        if facility == "feed":
            return self.train_feed(str(payload.get("url", "https://export.arxiv.org/rss/cs.AI")), int(payload.get("max_items", 2)), int(payload.get("steps", 2)))
        if facility in {"drone", "tmrl_doctor"}:
            return self.run_external_facility(facility, payload)
        if facility == "map":
            return self.facility_map()
        raise ValueError(f"unknown facility: {facility}")

    def facility_tour(self) -> dict[str, Any]:
        tour = ["health", "capabilities", "vocab", "curriculum", "snapshot", "dreamer_propose", "map"]
        results = []
        for facility in tour:
            try:
                results.append({"facility": facility, "ok": True, "result": self.run_facility(facility, {})})
            except Exception as exc:
                results.append({"facility": facility, "ok": False, "error": str(exc)})
        data = {"tour": tour, "results": results, "map": self.facility_map()}
        self.last_receipt = observe_receipt("mira_kite_authority_facility_tour", data)
        data["receipt"] = self.last_receipt
        self._record_event("facility_tour", data)
        return data

    def capability_snapshot(self) -> dict[str, Any]:
        agent = self.agent
        brains = []
        for idx, brain in enumerate(agent.brains):
            cfg = {
                "index": idx,
                "organism_id": agent.organism_names[idx] if idx < len(agent.organism_names) else f"org_{idx}",
                "caretaker_name": DEFAULT_CARETAKER_NAMES.get(agent.organism_names[idx], None)
                if idx < len(agent.organism_names)
                else None,
                "input_dim": getattr(brain, "input_dim", None),
                "hidden_dim": getattr(brain, "hidden_dim", None),
                "output_dim": getattr(brain, "output_dim", None),
                "vocab_size": getattr(brain, "vocab_size", None),
                "use_attention": bool(getattr(brain, "use_attention", False)),
                "use_hopfield": bool(getattr(brain, "use_hopfield", False)),
                "use_language_head": bool(getattr(brain, "use_language_head", False)),
                "use_concept_head": bool(getattr(brain, "use_concept_head", False)),
                "hopfield_patterns": getattr(brain, "hopfield_patterns", None),
                "hopfield_iterations": getattr(brain, "hopfield_iterations", None),
            }
            brains.append(cfg)

        atomic_counts = []
        assoc_counts = []
        for als in getattr(agent, "atomic_languages", []) or []:
            atoms = getattr(als, "atoms", {})
            atomic_counts.append(len(atoms))
            assoc_counts.append(sum(len(getattr(atom, "associations", {}) or {}) for atom in atoms.values()))

        kw = agent.knowledge_web if isinstance(agent.knowledge_web, dict) else {}
        alliance = agent.alliance_system.to_dict() if hasattr(agent, "alliance_system") else {}
        snapshot = {
            "pet_name": self.cocoon_name,
            "source_cocoon": str(self.cocoon_path),
            "active_cocoon_name": self.cocoon_name,
            "active_cocoon_path": str(self.cocoon_path),
            "loaded_at": self.loaded_at,
            "uptime_seconds": time.time() - self.started_at,
            "cycles": self.cycles,
            "last_receipt": self.last_receipt,
            "last_checkpoint": self.last_checkpoint,
            "mode": "ENSEMBLE" if agent.is_ensemble else "SOLO",
            "organism_names": agent.organism_names,
            "caretaker_names": DEFAULT_CARETAKER_NAMES,
            "fitness": agent.organism_fitness,
            "brains": brains,
            "vocabulary_size": len(agent.vocabulary.get("word_to_id", {})),
            "language_head_limit": min(getattr(b, "vocab_size", 0) for b in agent.brains) if agent.brains else 0,
            "knowledge_concepts": len(kw.get("concepts", {})),
            "knowledge_relations": len(kw.get("relations", [])),
            "conversation_turns": getattr(agent.conversation, "turn_count", 0),
            "training_logs": len(getattr(agent, "training_logs", [])),
            "atomic_concepts": atomic_counts,
            "atomic_associations": assoc_counts,
            "action_faculties": {str(key): value for key, value in ACTION_LABELS.items()},
            "authority_policy": {
                "faculty_order": FACULTY_ORDER,
                "routing": "score signal words against faculty hints, boost classifier result, verify anchor coverage",
                "feed_scoring": "rank items by novelty, anchor coverage, quine/fixed-point content, verification value, and source fitness",
                "quine_boundary": "formulate fixed-point syntax drills; do not execute generated commands",
            },
            "cascade": {
                "available": self.monitor is not None or self.adapter is not None,
                "adapter_signals": getattr(self.adapter, "signal_count", None),
                "monitor_name": getattr(self.monitor, "name", None),
                "graph_stats": self.graph.get_stats() if self.graph is not None else {},
                "metric_health": self.metrics.health_summary() if self.metrics is not None else {},
                "errors": self.cascade_errors[-5:],
            },
            "last_arbitration": self.last_arbitration,
            "last_informational_wealth": self.last_wealth,
            "recent_lessons": self.education_log[-8:],
            "recent_feed_reports": self.feed_reports[-5:],
            "training_config": getattr(agent, "config", {}),
            "alliances": {
                "count": len(alliance.get("alliances", {})),
                "trust_records": len(alliance.get("organism_trust", {})),
                "memberships": alliance.get("organism_to_alliance", {}),
            },
            "recent_events": self.event_log[-12:],
        }
        self.capability_cache = snapshot
        return snapshot

    def _train_steps(self, steps: int) -> list[float]:
        losses = []
        for _ in range(max(0, int(steps))):
            loss = self.agent.train_step()
            if loss and loss > 0 and not np.isnan(loss):
                losses.append(float(loss))
        return losses

    def quine_drill(self, limit: int = 8, steps: int = 2) -> dict[str, Any]:
        with self.lock:
            ladders = quinesmith_ladders()[: max(1, int(limit))]
            pairs = 0
            for ladder in ladders:
                pairs += add_reasoning_ladder(self.agent, ladder[:5], 0.91)
                add_relation(self.agent, "quine", ladder[-1], "safe_fixed_point_syntax", 0.82)
            losses = self._train_steps(steps)
            self.cycles += 1
            text = " ".join(" ".join(ladder) for ladder in ladders)
            words = top_words(text, 64)
            known, predictable = vocabulary_sets(self.agent)
            anchors = choose_anchors(words, predictable)
            novel = [w for w in words if w not in anchors][:8]
            arbitration = self._arbitrate(words, anchors, "quine_drill")
            wealth = self._informational_wealth(text, words, anchors, novel, arbitration, losses, "quine_drill")
            cascade = self._cascade_observe(
                "quine_drill",
                {"text": text[:1000], "ladders": ladders, "arbitration": arbitration, "wealth": wealth},
            )
            data = {
                "cycle": self.cycles,
                "ladders": ladders,
                "pairs": pairs,
                "losses": losses,
                "arbitration": arbitration,
                "informational_wealth": wealth,
                "cascade": cascade,
                "state": self.capability_snapshot(),
            }
            self.last_receipt = observe_receipt("mira_kite_authority_quine_drill", data)
            data["receipt"] = self.last_receipt
            data["teacher_note"] = self._teacher_note(
                text, words, anchors, novel, arbitration, 0, pairs, losses, self.last_receipt
            )
            self._record_event("quine_drill", data)
            self._write_status(data)
            return data

    def process_signal(self, text: str, steps: int = 4, source: str = "manual_signal") -> dict[str, Any]:
        with self.lock:
            words = top_words(text, 64)
            known, predictable = vocabulary_sets(self.agent)
            anchors = choose_anchors(words, predictable)
            arbitration = self._arbitrate(words, anchors, source)
            action_id = int(arbitration["action_id"])
            faculty = str(arbitration["chosen_faculty"])
            novel = [w for w in words if w not in anchors][:8]
            input_event = self._cascade_observe(
                "input_signal",
                {"source": source, "text": text[:1200], "words": words, "anchors": anchors, "novel": novel},
            )

            bridge_count = 0
            for word in novel[:6]:
                bridge_count += add_semantic_bridge(self.agent, word, anchors)
                for anchor in anchors:
                    strengthen_association(self.agent, word, anchor, 0.74, "authority_signal")
                    add_relation(self.agent, word, anchor, "authority_signal_anchor", 0.74)

            pair_count = 0
            if anchors:
                pair_count += add_reasoning_ladder(self.agent, anchors[:4], 0.96)
                pair_count += add_supervised_pair(self.agent, " ".join(words[:6] or anchors), " ".join(anchors[:4]), 0.96)

            spec = COGNITIVE_FACULTIES[action_id]
            add_cognitive_faculty(self.agent, action_id, spec["name"], spec["triggers"], spec["target"])
            if source in {"govern", "manual_signal"} and {"quine", "template", "syntax", "fixed", "point"} & set(words):
                for ladder in quinesmith_ladders()[:4]:
                    pair_count += add_reasoning_ladder(self.agent, ladder[:5], 0.9)
            losses = self._train_steps(steps)
            wealth = self._informational_wealth(text, words, anchors, novel, arbitration, losses, source)
            output_event = self._cascade_observe(
                "trained_signal",
                {
                    "source": source,
                    "faculty": faculty,
                    "action_id": action_id,
                    "bridges": bridge_count,
                    "pairs": pair_count,
                    "losses": losses,
                    "wealth": wealth,
                },
            )
            self._add_causation(
                input_event.get("event_id"),
                output_event.get("event_id"),
                "authority_training_route",
                wealth["value"],
                arbitration["reason"],
            )

            self.cycles += 1
            data = {
                "cycle": self.cycles,
                "source": source,
                "text": text[:1200],
                "words": words,
                "anchors": anchors,
                "novel": novel,
                "faculty": faculty,
                "action_id": action_id,
                "arbitration": arbitration,
                "bridges": bridge_count,
                "pairs": pair_count,
                "losses": losses,
                "informational_wealth": wealth,
                "cascade": {"input": input_event, "output": output_event},
                "state": self.capability_snapshot(),
            }
            self.last_receipt = observe_receipt("mira_kite_authority_signal", data)
            data["receipt"] = self.last_receipt
            data["teacher_note"] = self._teacher_note(
                text, words, anchors, novel, arbitration, bridge_count, pair_count, losses, self.last_receipt
            )
            self._record_event("signal", data)
            self._write_status(data)
            return data

    def relate(self, source: str, target: str, steps: int = 3) -> dict[str, Any]:
        with self.lock:
            source_words = top_words(source, 12)
            target_words = top_words(target, 12)
            if not source_words or not target_words:
                raise ValueError("source and target must contain words")
            known, predictable = vocabulary_sets(self.agent)
            anchors = choose_anchors(source_words + target_words, predictable)
            bridge_count = 0
            for word in source_words:
                bridge_count += add_semantic_bridge(self.agent, word, target_words[:4] + anchors[:2])
                for tw in target_words:
                    strengthen_association(self.agent, word, tw, 0.86, "manual_relate")
                    add_relation(self.agent, word, tw, "manual_relate", 0.86)
            pair_count = add_supervised_pair(self.agent, " ".join(source_words), " ".join(target_words[:4]), 1.0)
            losses = self._train_steps(steps)
            self.cycles += 1
            data = {
                "cycle": self.cycles,
                "source_words": source_words,
                "target_words": target_words,
                "anchors": anchors,
                "bridges": bridge_count,
                "pairs": pair_count,
                "losses": losses,
                "state": self.capability_snapshot(),
            }
            self.last_receipt = observe_receipt("mira_kite_authority_relate", data)
            data["receipt"] = self.last_receipt
            self._record_event("relate", data)
            self._write_status(data)
            return data

    def govern(self, text: str, steps: int = 2) -> dict[str, Any]:
        with self.lock:
            state_before = self.capability_snapshot()
            result = self.process_signal(text, steps=steps, source="govern")
            # Add a verification pass as civic arbitration.
            verify = COGNITIVE_FACULTIES[5]
            add_cognitive_faculty(self.agent, 5, verify["name"], verify["triggers"], verify["target"])
            losses = self._train_steps(1)
            data = {
                "decision": {
                    "primary_faculty": result["faculty"],
                    "primary_action": result["action_id"],
                    "verification_action": 5,
                    "equilibrium": "observe -> associate -> recall -> reason -> respond -> verify",
                },
                "signal_result": result,
                "verification_losses": losses,
                "state_before": state_before,
                "state_after": self.capability_snapshot(),
            }
            self.last_receipt = observe_receipt("mira_kite_authority_govern", data)
            data["receipt"] = self.last_receipt
            self._record_event("govern", data)
            self._write_status(data)
            return data

    def perturb(self, mode: str, intensity: float = 0.2, steps: int = 4) -> dict[str, Any]:
        with self.lock:
            mode = (mode or "stabilize").lower()
            intensity = max(0.0, min(1.0, float(intensity)))
            if mode == "explore":
                ladder = ["explore", "discover", "meaning", "reason"]
                reward = 0.75 + intensity * 0.2
            elif mode == "stabilize":
                ladder = ["stable", "focused", "coherent", "precise"]
                reward = 0.95
            elif mode == "verify":
                ladder = ["trace", "sequence", "reason", "precise"]
                reward = 0.92
            elif mode == "associate":
                ladder = ["signal", "meaning", "connected", "understand"]
                reward = 0.9
            else:
                ladder = ["learn", "adapt", "remember", "understand"]
                reward = 0.85
            pair_count = add_reasoning_ladder(self.agent, ladder, reward)
            for idx, spec in COGNITIVE_FACULTIES.items():
                if spec["name"] == mode or mode in spec["triggers"]:
                    add_cognitive_faculty(self.agent, idx, spec["name"], spec["triggers"], spec["target"])
            losses = self._train_steps(steps)
            self.cycles += 1
            data = {
                "cycle": self.cycles,
                "mode": mode,
                "intensity": intensity,
                "ladder": ladder,
                "pairs": pair_count,
                "losses": losses,
                "state": self.capability_snapshot(),
            }
            self.last_receipt = observe_receipt("mira_kite_authority_perturb", data)
            data["receipt"] = self.last_receipt
            self._record_event("perturb", data)
            self._write_status(data)
            return data

    def _score_feed_item(self, item: dict[str, Any], predictable: set[str]) -> dict[str, Any]:
        text = str(item.get("text", ""))[:3000]
        words = top_words(text, 96)
        anchors = choose_anchors(words, predictable)
        novel = [w for w in words if w not in anchors and len(w) > 4][:10]
        bag = set(words)
        quine_terms = {"quine", "template", "syntax", "brace", "fixed", "point", "nested"}
        verify_terms = {"audit", "receipt", "trace", "verify", "safe", "causal", "provenance"}
        route = self._arbitrate(words, anchors, "feed")
        novelty = min(1.0, len(novel) / 10)
        anchor_coverage = min(1.0, len(anchors) / 4)
        quine_value = min(1.0, len(bag & quine_terms) / 3)
        verify_value = min(1.0, len(bag & verify_terms) / 3)
        faculty_value = dict(route["ranked_faculties"]).get(route["chosen_faculty"], 0.0)
        score = round(0.30 * novelty + 0.25 * anchor_coverage + 0.18 * quine_value + 0.17 * verify_value + 0.10 * faculty_value, 4)
        return {
            "score": score,
            "title": item.get("title", ""),
            "source": item.get("source", ""),
            "link": item.get("link", ""),
            "words": words[:24],
            "anchors": anchors,
            "novel": novel,
            "routing": route,
            "components": {
                "novelty": round(novelty, 4),
                "anchor_coverage": round(anchor_coverage, 4),
                "quine_value": round(quine_value, 4),
                "verify_value": round(verify_value, 4),
                "faculty_value": round(faculty_value, 4),
            },
        }

    def train_feed(self, url: str, max_items: int = 4, steps: int = 4) -> dict[str, Any]:
        with self.lock:
            raw_items = fetch_url_text(url)[: max(1, int(max_items)) * 3]
            known, predictable = vocabulary_sets(self.agent)
            scored = [self._score_feed_item(item, predictable) for item in raw_items]
            ranked = sorted(zip(scored, raw_items), key=lambda pair: pair[0]["score"], reverse=True)
            items = [item for _, item in ranked[: max(1, int(max_items))]]
            selected_scores = [score for score, _ in ranked[: max(1, int(max_items))]]
            summaries = []
            for item in items:
                summaries.append(train_item(self.agent, item, self.cycles + 1))
            losses = self._train_steps(steps)
            text = " ".join(str(item.get("text", ""))[:1000] for item in items)
            words = top_words(text, 64)
            anchors = choose_anchors(words, predictable)
            novel = [w for w in words if w not in anchors][:8]
            arbitration = self._arbitrate(words, anchors, "feed")
            wealth = self._informational_wealth(text, words, anchors, novel, arbitration, losses, "feed")
            cascade = self._cascade_observe(
                "feed_training",
                {"url": url, "selected_scores": selected_scores, "summaries": summaries, "wealth": wealth},
            )
            self.cycles += 1
            data = {
                "cycle": self.cycles,
                "url": url,
                "items_seen": len(raw_items),
                "items_trained": len(items),
                "feed_scores": selected_scores,
                "summaries": summaries,
                "losses": losses,
                "arbitration": arbitration,
                "informational_wealth": wealth,
                "cascade": cascade,
                "state": self.capability_snapshot(),
            }
            self.last_receipt = observe_receipt("mira_kite_authority_feed", data)
            data["receipt"] = self.last_receipt
            data["teacher_note"] = self._teacher_note(
                text, words, anchors, novel, arbitration, sum(s.get("bridges", 0) for s in summaries), sum(s.get("pairs", 0) for s in summaries), losses, self.last_receipt
            )
            self.feed_reports.append({"time": time.time(), "url": url, "scores": selected_scores, "receipt": self.last_receipt})
            self.feed_reports = self.feed_reports[-20:]
            self._record_event("feed", data)
            self._write_status(data)
            return data

    def autopilot_train(
        self,
        mission: str = "",
        feeds: list[str] | None = None,
        depth: int = 1,
        include_games: bool = False,
        save_after: bool = False,
    ) -> dict[str, Any]:
        with self.lock:
            depth = max(1, min(int(depth), 4))
            feeds = [str(feed).strip() for feed in (feeds or DEFAULT_SCOUT_FEEDS) if str(feed).strip()]
            feeds = feeds[: max(1, min(len(feeds), 4))]
            mission = mission.strip() or f"teach {self.cocoon_name} to chat, associate, reason, verify, and learn from public machine learning signals"

            plan = [
                "Scout public feeds.",
                "Score items by known anchors, novelty, verification value, and fixed-point value.",
                "Bridge new words to predictable known words.",
                "Run follow-along cue and chain drills.",
                "Run safe Quinesmith fixed-point syntax drills.",
                "Receipt everything through Cascade and write a trainer note.",
            ]
            if include_games:
                plan.append("Run a tiny headless sphere burst for action grounding.")
            if save_after:
                plan.append("Export a checkpoint after the cycle.")

            started = time.time()
            words = top_words(mission, 64)
            known, predictable = vocabulary_sets(self.agent)
            anchors = choose_anchors(words, predictable)
            arbitration = self._arbitrate(words, anchors, "autopilot")
            mission_result = self.process_signal(mission, steps=depth, source="autopilot")

            drill_results = [
                self.follow_along_game("cue", mission, rounds=4 + depth, steps=max(0, depth - 1)),
                self.follow_along_game("chain", mission, rounds=4 + depth, steps=max(0, depth - 1)),
            ]
            quine_result = self.quine_drill(limit=4 + depth * 2, steps=max(0, depth - 1))

            feed_results = []
            for feed in feeds:
                try:
                    feed_results.append(self.train_feed(feed, max_items=1 + depth, steps=depth))
                except Exception as exc:
                    feed_results.append({"url": feed, "error": str(exc)})

            game_result = None
            if include_games:
                try:
                    game_result = self.native_sphere_burst(frames=90 + depth * 30, balls=1, misses=3, train=True)
                except Exception as exc:
                    game_result = {"error": str(exc)}

            checkpoint = None
            if save_after:
                checkpoint = self.export("mira_kite_autopilot_checkpoint.py")

            losses: list[float] = []
            for result in [mission_result, quine_result, *(drill_results or []), *(feed_results or [])]:
                if isinstance(result, dict):
                    losses.extend([float(x) for x in result.get("losses", []) if isinstance(x, (int, float))])
            
            def count_val(res, key):
                if not isinstance(res, dict): return 0
                val = res.get(key, 0)
                if isinstance(val, list): return len(val)
                try: return int(val)
                except (ValueError, TypeError): return 0

            bridge_count = sum(count_val(r, "bridges") for r in [mission_result, *feed_results])
            pair_count = sum(count_val(r, "pairs") for r in [mission_result, quine_result, *drill_results])
            for result in feed_results:
                if isinstance(result, dict):
                    for s in result.get("summaries", []):
                        pair_count += count_val(s, "pairs")

            novel = [w for w in top_words(mission, 64) if w not in anchors][:8]
            wealth = self._informational_wealth(mission, words, anchors, novel, arbitration, losses, "autopilot")
            teacher_note = self._teacher_note(
                mission,
                words,
                anchors,
                novel,
                arbitration,
                bridge_count,
                pair_count,
                losses,
                self.last_receipt,
            )
            data = {
                "mode": "caveman_scientist_autopilot",
                "mission": mission,
                "depth": depth,
                "plan": plan,
                "duration_seconds": time.time() - started,
                "feeds": feeds,
                "mission_result": self._summarize_nested(mission_result),
                "drill_results": [self._summarize_nested(result) for result in drill_results],
                "quine_result": self._summarize_nested(quine_result),
                "feed_results": [self._summarize_nested(result) for result in feed_results],
                "game_result": self._summarize_nested(game_result),
                "checkpoint": self._summarize_nested(checkpoint),
                "anchors": anchors,
                "novel": novel,
                "bridges": bridge_count,
                "pairs": pair_count,
                "losses": losses[-12:],
                "arbitration": arbitration,
                "informational_wealth": wealth,
                "teacher_note": teacher_note,
                "trainer_card": {
                    "simple_summary": "Autopilot scouted, drilled, associated, verified, and receipted the cycle.",
                    "you_do": "Read the battle card, chat with the cocoon, then press Autopilot again when ready.",
                    "cocoon_got": {
                        "known_anchors": anchors,
                        "new_terms": novel,
                        "bridge_count": bridge_count,
                        "pair_count": pair_count,
                    },
                },
                "state": self.capability_snapshot(),
            }
            self.last_receipt = observe_receipt("mira_kite_authority_autopilot", data)
            data["receipt"] = self.last_receipt
            data["teacher_note"]["receipt"] = self.last_receipt
            self._record_event("autopilot", data)
            self._write_status(data)
            return data

    def export(self, name: str | None = None) -> dict[str, Any]:
        with self.lock:
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name or f"mira_kite_authority_{int(time.time())}.py")
            if not safe.endswith(".py"):
                safe += ".py"
            path = self.output_dir / safe
            self.agent.export_cocoon(str(path))
            self.last_checkpoint = str(path)
            data = {"checkpoint": str(path), "state": self.capability_snapshot()}
            self.last_receipt = observe_receipt("mira_kite_authority_export", data)
            data["receipt"] = self.last_receipt
            self._record_event("export", data)
            self._write_status(data)
            return data

    def _write_status(self, data: dict[str, Any]) -> None:
        status_path = self.output_dir / "authority_status.json"
        payload = {
            "updated": time.time(),
            "last": self._compact_payload(data),
            "state": self.capability_snapshot(),
        }
        status_path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cocoon Authority</title>
  <style>
    :root {
      --bg: #050808;
      --panel: #0a1212;
      --line: #182222;
      --text: #a0b0b0;
      --muted: #5a6a6a;
      --accent: #00ff9d; /* Phosphor Green */
      --accent2: #00d1ff; /* Deep Cyan */
      --gold: #ffcc00; /* Pirate Gold */
      --warn: #ff3e3e; /* Danger Red */
      --amber: #ffb300; /* CRT Amber */
      --font: 'Courier New', Courier, monospace;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      font-size: 14px;
      line-height: 1.4;
      overflow-x: hidden;
      background-image: 
        linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%),
        linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 112, 0.06));
      background-size: 100% 2px, 3px 100%;
    }
    a { color: var(--accent); text-decoration: none; }
    button {
      background: var(--line);
      color: var(--text);
      border: 1px solid var(--line);
      padding: 8px 14px;
      border-radius: 4px;
      cursor: pointer;
      font-family: var(--font);
      font-weight: bold;
      text-transform: uppercase;
      letter-spacing: 1px;
      transition: all 0.2s;
    }
    button:hover { border-color: var(--accent); color: var(--accent); background: #0c1a1a; }
    button.primary { background: #004d30; border-color: var(--accent); color: var(--accent); }
    button.warn { background: #4d0000; border-color: var(--warn); color: var(--warn); }
    button.ultimate { 
      background: linear-gradient(135deg, #4d3d00, #004d30); 
      border-color: var(--gold); 
      color: var(--gold);
      box-shadow: 0 0 10px rgba(255, 204, 0, 0.2);
    }
    input, textarea, select {
      background: #000;
      border: 1px solid var(--line);
      color: var(--accent);
      padding: 8px;
      border-radius: 4px;
      width: 100%;
      font-family: var(--font);
    }
    textarea { resize: vertical; min-height: 60px; }
    header {
      background: var(--panel);
      border-bottom: 2px solid var(--accent);
      padding: 12px 16px;
      position: sticky;
      top: 0;
      z-index: 100;
      box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    .titlebar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
    h1 { margin: 0; font-size: 20px; color: var(--accent); text-transform: uppercase; letter-spacing: 2px; }
    .league { font-size: 11px; background: #004d30; color: var(--accent); padding: 2px 8px; border-radius: 99px; font-weight: bold; border: 1px solid var(--accent); }
    .tabs { display: grid; grid-template-columns: repeat(6, 1fr); gap: 4px; }
    .tabs button { border-bottom: 3px solid transparent; border-radius: 4px 4px 0 0; font-size: 11px; padding: 10px 4px; }
    .tabs button.active { border-color: var(--accent); color: var(--accent); background: #0c1a1a; }
    main { padding: 14px; max-width: 1400px; margin: 0 auto; }
    section { margin-bottom: 16px; background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 14px; }
    h2 { margin: 0 0 10px; font-size: 15px; color: var(--accent2); text-transform: uppercase; border-left: 4px solid var(--accent); padding-left: 10px; }
    .grid { display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(290px, 0.8fr); gap: 14px; align-items: start; }
    .stack { display: flex; flex-direction: column; gap: 14px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px; }
    .row3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin-bottom: 8px; }
    .view { display: none; }
    .view.active { display: block; }
    .muted { color: var(--muted); font-size: 12px; }
    .status { font-size: 12px; margin-top: 8px; color: var(--gold); border-top: 1px dashed var(--line); padding-top: 8px; }
    .hero { background: linear-gradient(180deg, #0a1414, #050808); border: 1px solid var(--accent); }
    .mission { background: rgba(0, 255, 157, 0.05); padding: 12px; border-radius: 6px; }
    .controlGrid { display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 12px; align-items: start; }
    .canvasCard { background: #000; border: 1px solid var(--line); border-radius: 6px; padding: 10px; }
    .canvasShell { position: relative; aspect-ratio: 16 / 10; background:
      radial-gradient(circle at 20% 20%, rgba(0,255,157,0.08), transparent 28%),
      radial-gradient(circle at 80% 20%, rgba(0,209,255,0.10), transparent 25%),
      radial-gradient(circle at 50% 80%, rgba(255,204,0,0.08), transparent 30%),
      #020404;
      border: 1px solid var(--line);
      overflow: hidden;
    }
    .canvasShell canvas {
      width: 100%;
      height: 100%;
      display: block;
    }
    .canvasHud {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px;
      margin-top: 8px;
    }
    .hudTile {
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 8px;
      background: #000;
      min-height: 52px;
    }
    .hudTile b { display: block; color: var(--gold); font-size: 11px; text-transform: uppercase; }
    .hudTile span { color: var(--text); font-size: 12px; }
    .cavemanSteps {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
      font-size: 12px;
    }
    .cavemanSteps div {
      background: #000;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      min-height: 58px;
    }
    .cavemanSteps b { color: var(--gold); display: block; }
    .bigtext { font-size: 14px; min-height: 80px; color: var(--gold); border-color: var(--gold); }
    .partnerName { font-size: 20px; color: var(--gold); font-weight: 900; letter-spacing: 1px; }
    .xpbar { height: 10px; border: 1px solid var(--line); border-radius: 0; background: #000; overflow: hidden; margin: 12px 0; }
    .xpfill { height: 100%; width: 0%; background: linear-gradient(90deg, var(--accent2), var(--accent)); }
    .duel { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; }
    .duelBox { border: 1px solid var(--line); border-radius: 4px; padding: 10px; background: #000; min-height: 82px; font-size: 12px; }
    .duelBox b { color: var(--gold); text-transform: uppercase; font-size: 11px; display: block; margin-bottom: 4px; }
    .map { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 8px; margin-top: 8px; }
    .node { border: 1px solid var(--line); border-radius: 4px; padding: 8px; background: #000; min-height: 74px; }
    .node.ready { border-color: var(--accent); box-shadow: 0 0 5px rgba(0, 255, 157, 0.3); }
    .node.blocked, .node.needs { border-color: var(--warn); }
    .node b { display: block; color: var(--accent); font-size: 12px; }
    .edgeList { margin-top: 8px; color: var(--muted); font-size: 11px; font-family: monospace; }
    .manualGrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 8px; margin-top: 8px; }
    .contractToolbar {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin: 8px 0;
    }
    .contractGrid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 8px;
      margin-top: 8px;
      max-height: 520px;
      overflow: auto;
      padding-right: 4px;
    }
    .contractCard {
      border: 1px solid var(--line);
      background: #000;
      border-radius: 4px;
      padding: 10px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-height: 230px;
    }
    .contractCard.inspect { border-color: var(--accent2); }
    .contractCard.mutating { border-color: var(--gold); }
    .contractCard.external, .contractCard.overclock { border-color: var(--warn); }
    .contractCard h3 {
      margin: 0;
      color: var(--gold);
      font-size: 14px;
      text-transform: uppercase;
    }
    .contractMeta {
      color: var(--muted);
      font-size: 11px;
      overflow-wrap: anywhere;
    }
    .contractMeta b { color: var(--accent); }
    .contractActions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      margin-top: auto;
    }
    .toolCard {
      border: 1px solid var(--line);
      border-radius: 4px;
      background: #000;
      padding: 12px;
      min-height: 168px;
      display: flex;
      flex-direction: column;
      gap: 7px;
    }
    .toolCard.ready { border-color: var(--accent); border-width: 2px; }
    .toolCard.needs { border-color: var(--amber); }
    .toolCard b { color: var(--gold); font-size: 15px; }
    .toolCard p { margin: 0; color: var(--text); font-size: 12px; }
    .toolCard small { color: var(--muted); font-size: 11px; }
    details { background: #000; border: 1px solid var(--line); border-radius: 4px; padding: 8px; margin-bottom: 8px; }
    summary { cursor: pointer; color: var(--gold); font-weight: bold; font-size: 12px; }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      margin: 8px 0 0;
      max-height: 400px;
      overflow: auto;
      font-size: 11px;
      color: var(--accent);
      background: rgba(0,0,0,0.5);
      padding: 8px;
    }
    .procession { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 6px; margin-top: 6px; }
    .step { border: 1px solid var(--line); border-radius: 4px; padding: 8px; background: #000; min-height: 58px; text-align: center; }
    .step.active { border-color: var(--accent); background: #001a11; box-shadow: inset 0 0 10px rgba(0,255,157,0.2); }
    .step b { display: block; font-size: 11px; color: var(--accent); text-transform: uppercase; }
    .step span { display: block; color: var(--gold); font-size: 12px; font-weight: bold; }
    .chatlog { 
      height: 320px; 
      overflow-y: auto; 
      border: 1px solid var(--line); 
      background: #000; 
      padding: 12px; 
      margin-bottom: 12px;
      border-radius: 4px;
      display: flex;
      flex-direction: column; gap: 12px;
    }
    .msg { max-width: 85%; padding: 10px; border-radius: 4px; position: relative; }
    .msg.user { align-self: flex-end; background: #001a11; border: 1px solid var(--accent); color: var(--accent); }
    .msg.assistant { align-self: flex-start; background: #0a1212; border: 1px solid var(--accent2); color: var(--accent2); }
    .msg b { display: block; font-size: 10px; margin-bottom: 4px; text-transform: uppercase; opacity: 0.7; }
    .badge { background: var(--gold); color: #000; font-size: 9px; padding: 1px 4px; border-radius: 2px; font-weight: bold; margin-right: 4px; text-transform: uppercase; }
    .stat { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px dashed var(--line); }
    .stat span { font-size: 11px; color: var(--muted); text-transform: uppercase; }
    .stat b { font-size: 12px; color: var(--gold); }
    .pill { display: inline-block; background: #182222; color: var(--accent); font-size: 10px; padding: 2px 6px; border-radius: 4px; margin: 2px; border: 1px solid var(--line); }
    
    /* Pirate Overlay Effects */
    .scanlines::after {
      content: " ";
      display: block;
      position: fixed;
      top: 0; left: 0; bottom: 0; right: 0;
      background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.1) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.03), rgba(0, 255, 0, 0.01), rgba(0, 0, 255, 0.03));
      z-index: 200;
      background-size: 100% 2px, 3px 100%;
      pointer-events: none;
    }
    
    .facilityStrip { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
    .liveDot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; background: var(--warn); box-shadow: 0 0 0 0 rgba(255,62,62,0.5); vertical-align: middle; }
    .liveDot.live { background: var(--accent); box-shadow: 0 0 12px rgba(0,255,157,0.5); }
    .liveDot.busy { background: var(--gold); box-shadow: 0 0 12px rgba(255,204,0,0.5); }
    .liveDot.fail { background: var(--warn); box-shadow: 0 0 12px rgba(255,62,62,0.5); }
    .mini { font-size: 11px; color: var(--muted); }
    .reactionDock {
      position: fixed;
      right: 12px;
      bottom: 12px;
      width: min(360px, calc(100vw - 24px));
      display: flex;
      flex-direction: column;
      gap: 8px;
      z-index: 220;
      pointer-events: none;
    }
    .reaction {
      border: 1px solid var(--line);
      border-left: 4px solid var(--accent);
      background: rgba(0, 0, 0, 0.92);
      color: var(--text);
      padding: 10px;
      border-radius: 6px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.45);
      animation: reactionIn 0.18s ease-out;
      font-size: 12px;
    }
    .reaction b { color: var(--gold); display: block; margin-bottom: 3px; text-transform: uppercase; }
    .reaction.ok { border-left-color: var(--accent); }
    .reaction.bad { border-left-color: var(--warn); }
    .reaction.exec { border-left-color: var(--gold); }
    .opRelay {
      display: grid;
      gap: 8px;
      font-size: 12px;
    }
    .opTile {
      border: 1px solid var(--line);
      background: #000;
      border-radius: 4px;
      padding: 8px;
      min-height: 54px;
    }
    .opTile b {
      display: block;
      color: var(--gold);
      font-size: 10px;
      text-transform: uppercase;
      margin-bottom: 4px;
    }
    .opTile code {
      color: var(--accent2);
      font-size: 11px;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .opTile pre {
      max-height: 150px;
      margin: 0;
      color: var(--text);
    }
    .opStatusLine {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 6px;
    }
    .opMetric {
      border: 1px solid var(--line);
      background: #020606;
      padding: 6px;
      min-height: 42px;
    }
    .opMetric span {
      display: block;
      color: var(--muted);
      font-size: 9px;
      text-transform: uppercase;
    }
    .opMetric strong {
      display: block;
      color: var(--accent);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .opHistory {
      display: flex;
      flex-direction: column;
      gap: 4px;
      max-height: 180px;
      overflow: auto;
    }
    .opHistory div {
      border-left: 3px solid var(--line);
      background: #020606;
      padding: 5px 7px;
      font-size: 11px;
    }
    .opHistory .ok { border-left-color: var(--accent); }
    .opHistory .bad { border-left-color: var(--warn); }
    .buttonPulse {
      border-color: var(--gold) !important;
      color: var(--gold) !important;
      box-shadow: 0 0 14px rgba(255,204,0,0.45);
    }
    .resultFlash { animation: resultFlash 0.5s ease-out; }
    @keyframes reactionIn {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @keyframes resultFlash {
      from { outline: 2px solid var(--accent); outline-offset: 4px; }
      to { outline: 0 solid transparent; outline-offset: 0; }
    }
    @media (max-width: 980px) {
      .grid, .controlGrid { grid-template-columns: 1fr; }
      .facilityStrip { grid-template-columns: 1fr 1fr; }
      .opStatusLine { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 780px) {
      .tabs { grid-template-columns: repeat(3, 1fr); }
      .procession { grid-template-columns: repeat(2, 1fr); }
      header { position: static; }
    }
    @media (max-width: 520px) {
      .row, .row3 { grid-template-columns: 1fr; }
      .facilityStrip { grid-template-columns: 1fr; }
      .contractToolbar { grid-template-columns: 1fr; }
      main { padding: 8px; }
    }
  </style>
</head>
<body>
<header>
  <div class="titlebar">
    <h1>Cocoon Authority</h1>
    <div class="league" title="Local Termux-backed cocoon control surface.">Active Entity</div>
  </div>
  <div class="tabs" role="tablist">
    <button data-tab="chat" class="active" title="Partner quarters. The cocoon can learn from each turn.">Quarters</button>
    <button data-tab="drills" title="Tactical drills: echo, cue, chain, role, chant, fixed-point practice.">The Brig</button>
    <button data-tab="teach" title="Cognitive spoils: signals, relations, feeds, and arbitration.">The Vault</button>
    <button data-tab="facilities" title="Native game lanes, snapshots, and facility readiness.">Systems</button>
    <button data-tab="overclock" title="🦋 HYPER-ACCELERATION: Overclock learning and inject super-experiences." style="color: var(--warn); font-weight: bold;">Overclock</button>
    <button data-tab="wealth" title="The Black Ledger: Cascade receipts, routes, and teacher notes.">Black Ledger</button>
  </div>
</header>
<main>
  <section class="hero">
    <h2>Active Entity</h2>
    <div class="controlGrid">
      <div class="mission">
        <div class="facilityStrip">
          <button id="autopilotRun" class="ultimate" title="Run a real bounded training cycle and record the result.">Autopilot</button>
          <button id="tourRunTop" title="Run the safe built-in tour across core systems.">Tour</button>
          <button id="hardenRun" title="Run health, dependency, receipt, tape, and facility readiness checks.">Harden</button>
        </div>
        <div class="row3" style="margin-top:8px;">
          <select id="autoDepth" title="Low is phone-friendly. Higher does more feed and drill work.">
            <option value="1">small hunt</option>
            <option value="2">deep hunt</option>
            <option value="3">boss hunt</option>
            <option value="4">overkill hunt</option>
          </select>
          <select id="autoGames" title="Include a visible sphere/gym pass where available.">
            <option value="false">games off</option>
            <option value="true">games on</option>
          </select>
          <select id="autoSave" title="Export a checkpoint after the cycle.">
            <option value="false">save later</option>
            <option value="true">save after</option>
          </select>
        </div>
        <textarea id="autoMission" class="bigtext" title="Say the mission in normal words. The app handles scouting, anchors, drills, feeds, receipts, and summary." placeholder="Optimize chat, tool reasoning, machine learning, robotics, semantic association, and verification.">make the active cocoon better at chat, tool reasoning, machine learning, robotics, semantic association, and verification</textarea>
        <textarea id="autoFeeds" title="Optional public feeds. One URL per line. Leave as-is for default AI/ML scouting.">https://export.arxiv.org/rss/cs.AI
https://export.arxiv.org/rss/cs.LG
https://hnrss.org/newest?q=machine%20learning</textarea>
      </div>
      <div class="canvasCard">
        <div class="canvasShell">
          <canvas id="sphereCanvas" width="960" height="600"></canvas>
        </div>
        <div class="canvasHud">
          <div class="hudTile"><b>Sphere</b><span id="sphereHud">idle</span></div>
          <div class="hudTile"><b>Gym</b><span id="gymHud">idle</span></div>
          <div class="hudTile"><b>Active</b><span id="activeHud">local cocoon</span></div>
          <div class="hudTile"><b>Status</b><span id="surfaceHud">ready</span></div>
        </div>
      </div>
    </div>
    <div class="cavemanSteps">
      <div><b>1. Press</b><span class="muted">Launch the local app.</span></div>
      <div><b>2. Watch</b><span class="muted">Inspect the active entity and ledger.</span></div>
      <div><b>3. Chat</b><span class="muted">Converse with the loaded cocoon.</span></div>
    </div>
  </section>
  <section>
    <h2>Capability Contract Matrix</h2>
    <div class="muted">Every control below is generated from the real backend capability manifest: route, payload schema, expected outputs, risk, requisites, and followups.</div>
    <div class="contractToolbar">
      <select id="contractFilter" title="Filter contracts by risk/group.">
        <option value="all">all capabilities</option>
        <option value="inspect">inspect/read-only</option>
        <option value="mutating">mutating</option>
        <option value="external">external/requisite gated</option>
        <option value="overclock">overclock</option>
      </select>
      <button id="contractReload" title="Reload the machine-readable capability manifest from /api/capability_manifest.">Reload Contracts</button>
    </div>
    <div id="contractGrid" class="contractGrid"></div>
  </section>
  <section>
    <h2>System Cards</h2>
    <div class="muted">Plain-language facility cards with direct execution and deepening controls.</div>
    <div class="row">
      <button id="manualRun" title="Reload the complete plain-language facility manual.">Reload Systems</button>
      <button id="deepenAllRun" title="Run a staged deepening pass for the whole facility.">Deepen Autopilot</button>
    </div>
    <div id="manualGrid" class="manualGrid"></div>
  </section>
  <div class="grid">
    <div class="stack">
      <div id="view-chat" class="view active">
        <section>
          <h2>Captain's Quarters <small style='color:var(--warn); font-size:9px;'>[128 TOKEN MAX PAYLOAD]</small></h2>
          <div id="chatLog" class="chatlog"></div>
          <textarea id="chatText" placeholder="Talk to the active cocoon..."></textarea>
          <div class="row3">
            <select id="chatLearn" title="When enabled, the cocoon learns from this chat turn.">
              <option value="true">learn on</option>
              <option value="false">learn off</option>
            </select>
            <input id="chatSteps" title="Extra neural train steps after the chat turn." type="number" min="0" max="20" value="1">
            <button id="chatSend" class="primary" title="Send a normal chat message to the cocoon.">Send</button>
          </div>
          <div id="chatStatus" class="status"></div>
        </section>
        <section>
          <h2>Butterfly Council</h2>
          <div id="councilLog" class="chatlog"></div>
          <textarea id="councilText" placeholder="Speak once to all runnable cocoons..."></textarea>
          <div class="row3">
            <select id="councilLearn" title="When enabled, each runnable cocoon trains on this prompt before the central response is chosen.">
              <option value="true">learn on</option>
              <option value="false">learn off</option>
            </select>
            <input id="councilSteps" title="Extra train steps for each runnable cocoon after the prompt." type="number" min="0" max="20" value="1">
            <input id="councilLimit" title="Maximum runnable cocoons to consult in one pass." type="number" min="1" max="24" value="8">
          </div>
          <div class="row">
            <select id="councilExport" title="Write trained council checkpoints to the local runtime directory.">
              <option value="false">export off</option>
              <option value="true">export on</option>
            </select>
            <button id="councilRun" class="primary" title="Fan one prompt out across every runnable cocoon, compare replies, and optionally train each one.">Council</button>
          </div>
          <div id="councilStatus" class="status"></div>
        </section>
      </div>

      <div id="view-drills" class="view">
        <section>
          <h2>Follow-Along Moves</h2>
          <select id="followMode" title="Echo repeats, cue maps cue to trigger, chain grows phrases, role flips perspective, chant uses original call-response.">
            <option value="cue">cue trigger</option>
            <option value="echo">echo repeat</option>
            <option value="chain">word chain</option>
            <option value="role">role turn</option>
            <option value="chant">call response</option>
          </select>
          <textarea id="followSeed" placeholder="Seed words: signal meaning reason verify"></textarea>
          <div class="row3">
            <input id="followRounds" title="How many follow-along pairs to create." type="number" value="6" min="1" max="24">
            <input id="followSteps" title="Train steps after building the drill." type="number" value="2" min="0" max="20">
            <button id="followRun" class="primary" title="Build cue/response pairs, train them, and record a receipt.">Run Drill</button>
          </div>
        </section>
        <section>
          <h2>Tactical Maneuvers</h2>
          <div class="row">
            <button id="quineRun" title="Run Quinesmith syntax and nesting ladders without executing generated commands.">Quine Drill</button>
            <button id="sphereRun" title="Run a small bounded headless sphere-arena burst.">Sphere Burst</button>
          </div>
        </section>
      </div>

      <div id="view-teach" class="view">
        <section>
          <h2>The Cognitive Vault</h2>
          <textarea id="signal" placeholder="Concept, lesson, problem, association, or sentence..."></textarea>
          <div class="row3">
            <input id="signalSteps" title="Train steps after routing this signal." type="number" value="4" min="0" max="40">
            <button id="signalTrain" class="primary" title="Route through faculties, bridge novel words to known anchors, train, and receipt.">Train Signal</button>
            <button id="signalGovern" title="Run the same signal with an explicit verification/arbitration pass.">Arbitrate</button>
          </div>
        </section>
        <section>
          <h2>Association Move</h2>
          <input id="relSource" placeholder="source term or phrase">
          <input id="relTarget" placeholder="target term or phrase">
          <div class="row">
            <input id="relSteps" title="Train steps after relation building." type="number" value="3" min="0" max="40">
            <button id="relRun" title="Create direct semantic relations and supervised pairs.">Relate</button>
          </div>
        </section>
        <section>
          <h2>The Crow's Nest</h2>
          <input id="feedUrl" value="https://export.arxiv.org/rss/cs.AI" title="RSS, Atom, or simple public URL. Items are scored before training.">
          <div class="row3">
            <input id="feedItems" title="Max selected items after feed scoring." type="number" value="3" min="1" max="12">
            <input id="feedSteps" title="Train steps after selected feed items." type="number" value="4" min="0" max="40">
            <button id="feedRun" title="Score feed items for novelty, anchors, verification, and fixed-point value, then train selected items.">Study Feed</button>
          </div>
        </section>
      </div>

      <div id="view-facilities" class="view">
        <section>
          <h2>The Ship's Systems</h2>
          <select id="facilitySelect" title="Run any mirrored cocoon facility from one control.">
            <option value="health">health</option>
            <option value="chat">chat</option>
            <option value="council">council</option>
            <option value="teach">teach</option>
            <option value="act">act</option>
            <option value="learn">learn</option>
            <option value="vocab">vocab</option>
            <option value="curriculum">curriculum</option>
            <option value="logs">logs</option>
            <option value="snapshot">snapshot</option>
            <option value="save">save</option>
            <option value="export">export</option>
            <option value="capabilities">capabilities</option>
            <option value="dreamer_propose">dreamer propose</option>
            <option value="sphere">sphere</option>
            <option value="gym">gym</option>
            <option value="follow">follow</option>
            <option value="quine">quine</option>
            <option value="feed">feed</option>
            <option value="drone">drone bounded</option>
            <option value="tmrl_doctor">tmrl doctor</option>
            <option value="map">map</option>
          </select>
          <textarea id="facilityPayload" title="Optional JSON payload for the selected facility.">{"prompt":"hello cocoon","text":"signal meaning reason verify","seed":"signal meaning reason verify","steps":0}</textarea>
          <div class="row">
            <button id="facilityRun" class="primary" title="Run the selected facility with the JSON payload.">Run Facility</button>
            <button id="tourRun" title="Run a safe tour across core facilities and show the map.">Run Tour</button>
          </div>
          <div class="row3">
            <button id="facilitiesRun" title="Inventory native cocoon methods, games, dependencies, and mirrored endpoints.">Facilities</button>
            <button id="snapshotRun" title="Read the cocoon live symbolic/training snapshot.">Snapshot</button>
            <button id="saveRun" title="Save native runtime state files under the authority runtime directory.">Save State</button>
          </div>
          <div class="row3">
            <button id="vocabRun" title="Show the first vocabulary slice and vocabulary size.">Vocab</button>
            <button id="logsRun" title="Show recent native training logs.">Logs</button>
            <button id="exportRun" class="warn" title="Export a checkpoint with the current learned state embedded.">Export</button>
          </div>
        </section>
        <section>
          <h2>Facility Map</h2>
          <div id="facilityMap" class="map"></div>
          <div id="facilityEdges" class="edgeList"></div>
        </section>
      </div>

      <div id="view-overclock" class="view">
        <section style="border: 2px solid #ff3e3e; padding: 16px; border-radius: 8px; background: rgba(255, 62, 62, 0.05);">
          <h2 style="color: #ff3e3e;">🦋 Hyper-Acceleration Dashboard</h2>
          <p class="muted">Warning: High intensity learning may cause neural drift. Use sparingly for rapid evolution.</p>

          <div class="row3" style="margin-bottom: 16px;">
            <div>
              <label>LR Boost</label>
              <input id="cheatLR" type="number" min="1" max="50" value="5" title="Multiplier for the learning rate.">
            </div>
            <div>
              <label>Injection</label>
              <input id="cheatMult" type="number" min="1" max="20" value="5" title="Repeat factor for super-experiences.">
            </div>
            <div>
              <label>Turbo Steps</label>
              <input id="cheatSteps" type="number" min="1" max="2000" value="256" title="Number of training steps to run.">
            </div>
          </div>

          <button id="cheatRun" class="ultimate" style="background: linear-gradient(135deg, #ff3e3e, #d62828); color: white;" 
                  title="Butterfly Overclock: inject high-pressure training. Requires explicit operator intent.">RUN OVERCLOCK TRAINING</button>

          <div id="cheatStatus" class="status" style="margin-top: 16px;"></div>
        </section>
      </div>
      <div id="view-wealth" class="view">
        <section>
          <h2>Battle Ledger</h2>
          <div id="wealthStatus" class="status">Live wealth relay pending.</div>
          <div id="wealthGrid" class="manualGrid"></div>
          <div id="procession" class="procession"></div>
          <div id="duelLesson" class="duel"></div>
          <div id="wealthReceipts" class="edgeList"></div>
          <div id="wealthEvents" class="edgeList"></div>
          <div id="ledgerMap" class="map"></div>
          <div id="lessonBox"></div>
        </section>
      </div>
    </div>

    <aside class="stack">
      <section>
        <h2>Trainer Card</h2>
        <div id="partnerCard" class="partner"></div>
        <div id="stats" class="stats"></div>
        <div id="faculties"></div>
        <div class="row">
          <button id="refreshRun" title="Refresh live state and recent lessons.">Refresh</button>
          <button id="clearOutput" title="Clear the visible output panel only.">Clear</button>
        </div>
        <div id="globalStatus" class="status"></div>
      </section>
      <section>
        <h2>Live Operation Relay</h2>
        <div class="opRelay">
          <div class="opStatusLine">
            <div class="opMetric"><span>Button</span><strong id="opButton">none</strong></div>
            <div class="opMetric"><span>Status</span><strong id="opState">idle</strong></div>
            <div class="opMetric"><span>Latency</span><strong id="opLatency">0 ms</strong></div>
            <div class="opMetric"><span>Receipt</span><strong id="opReceipt">none</strong></div>
          </div>
          <div class="opTile"><b>Route</b><code id="opRoute">No route fired yet.</code></div>
          <div class="opTile"><b>Purpose</b><code id="opPurpose">Press any button to see what it does before and after execution.</code></div>
          <div class="opTile"><b>Input Payload</b><pre id="opPayload">{}</pre></div>
          <div class="opTile"><b>Output Keys</b><code id="opKeys">none</code></div>
          <div class="opTile"><b>State Delta</b><code id="opDelta">none</code></div>
          <div class="opTile"><b>Errors / Blockers</b><code id="opError">none</code></div>
          <div class="opTile"><b>Last 8 Operations</b><div id="opHistory" class="opHistory"></div></div>
        </div>
      </section>
      <section>
        <h2>🏴‍☠️ Tactical Terminal</h2><div id='terminal' style='background:#000; color:var(--accent); padding:8px; border:1px solid var(--line); height:180px; overflow-y:auto; font-size:10px; font-family:monospace;'><div>[SYSTEM] Terminal initialized.</div></div></section><section><h2>Last Result</h2>
        <details open>
          <summary>Readable</summary>
          <div id="readable" class="muted">Ready.</div>
        </details>
        <details>
          <summary>Raw JSON</summary>
          <pre id="out">{}</pre>
        </details>
      </section>
    </aside>
  </div>
</main>
<div id="reactionDock" class="reactionDock" aria-live="polite"></div>
<script>
const $ = (id) => document.getElementById(id);
let lastState = null;
let busy = false;
let activeButton = null;
let currentOp = null;
let opHistory = [];
let capabilityManifest = null;
let surfaceState = {
  tick: 0,
  mode: 'idle',
  sphere: {catches: 0, misses: 0, streak: 0, frames: 0, balls: 1, rewards: {}},
  gym: {env: 'idle', episodes: 0, reward: null},
  active: 'local cocoon',
  status: 'ready'
};

async function api(path, payload=null) {
  const started = performance.now();
  if (currentOp) {
    currentOp.path = path;
    currentOp.method = payload === null ? 'GET' : 'POST';
    currentOp.payload = payload === null ? {} : payload;
    renderOperation();
  }
  const opts = payload === null ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)};
  const res = await fetch(path, opts);
  const data = await res.json();
  if (currentOp) {
    currentOp.latency = Math.round(performance.now() - started);
    currentOp.httpStatus = res.status;
    currentOp.outputKeys = outputKeys(data);
    currentOp.receipt = receiptOf(data);
    currentOp.error = res.ok ? '' : (data.error || JSON.stringify(data).slice(0, 400));
    renderOperation();
  }
  if (!res.ok) throw new Error(data.error || JSON.stringify(data));
  return data;
}

function outputKeys(data) {
  if (!data || typeof data !== 'object') return [typeof data];
  return Object.keys(data).sort().slice(0, 18);
}

function receiptOf(data) {
  if (!data || typeof data !== 'object') return 'none';
  if (data.receipt) return String(data.receipt).slice(0, 24);
  if (data.last_receipt) return String(data.last_receipt).slice(0, 24);
  if (data.state && data.state.last_receipt) return String(data.state.last_receipt).slice(0, 24);
  if (Array.isArray(data.stages)) {
    const stage = data.stages.find(s => s && s.result && (s.result.receipt || (s.result.state && s.result.state.last_receipt)));
    if (stage) return receiptOf(stage.result);
  }
  return 'none';
}

function stateDelta(before, after) {
  if (!after || typeof after !== 'object') return 'none';
  const b = before || {};
  const fields = [
    ['vocabulary_size', 'vocab'],
    ['knowledge_relations', 'relations'],
    ['conversation_turns', 'turns'],
    ['training_logs', 'logs'],
    ['cycles', 'cycles'],
    ['training_step', 'train_step']
  ];
  const parts = [];
  for (const [field, label] of fields) {
    if (typeof after[field] === 'undefined') continue;
    const oldVal = Number(b[field] || 0);
    const newVal = Number(after[field] || 0);
    const diff = newVal - oldVal;
    parts.push(`${label}: ${newVal}${diff ? ' (' + (diff > 0 ? '+' : '') + diff + ')' : ''}`);
  }
  if (after.last_receipt && after.last_receipt !== b.last_receipt) parts.push(`receipt: ${String(after.last_receipt).slice(0, 24)}`);
  return parts.join(' | ') || 'no tracked state change';
}

function operationPurpose(label) {
  const text = String(label || '').toLowerCase();
  if (text.includes('chat')) return 'Send a prompt to /api/native/chat. Optional learning updates conversation state and training logs.';
  if (text.includes('signal')) return 'Route text through faculties, bridge anchors, train selected associations, and return a teacher note.';
  if (text.includes('arbitrate')) return 'Run verification/governance over the signal and report faculty routing.';
  if (text.includes('drill')) return 'Generate follow-along pairs and train cue/echo/chain/role/chant behavior.';
  if (text.includes('sphere')) return 'Run a bounded action/reward game burst and report catches, misses, frames, and state.';
  if (text.includes('council') || text.includes('consult')) return 'Fan one prompt across discovered runnable cocoons, compare replies, and optionally train each one.';
  if (text.includes('autopilot') || text.includes('armada')) return 'Run a multi-stage training pass: scout, drill, associate, verify, game, summarize.';
  if (text.includes('facility')) return 'Run the selected system route with the JSON payload shown here.';
  if (text.includes('harden')) return 'Audit health, dependencies, files, receipts, and route readiness.';
  if (text.includes('save')) return 'Write current runtime state files to local storage.';
  if (text.includes('export')) return 'Bake current learned state into a checkpoint file.';
  if (text.includes('vocab')) return 'Read vocabulary slice and vocabulary size.';
  if (text.includes('logs')) return 'Read recent training/audit logs.';
  if (text.includes('snapshot')) return 'Read live symbolic/runtime memory.';
  if (text.includes('systems') || text.includes('manual')) return 'Load the full route manual and capability cards.';
  if (text.includes('map') || text.includes('tour')) return 'Read or execute facility map/tour routes and show system graph.';
  if (text.includes('overclock')) return 'Explicit high-pressure training route. This is mutating and intentionally gated by operator intent.';
  return 'Run the selected Authority route and display exact input/output telemetry.';
}

function capabilityByKey(key) {
  if (!capabilityManifest || !Array.isArray(capabilityManifest.capabilities)) return null;
  return capabilityManifest.capabilities.find(cap => cap.key === key) || null;
}

function capabilityPurpose(cap) {
  if (!cap) return 'No capability selected.';
  const parts = [
    cap.teaches || 'No description supplied.',
    `Route: ${cap.method} ${cap.path}.`,
    `Risk: ${cap.risk_class || 'unknown'}.`,
    `Expected: ${(cap.expected_outputs || []).join(', ') || 'unspecified'}.`
  ];
  if (cap.prerequisites && cap.prerequisites.length) parts.push(`Requisites: ${cap.prerequisites.join(', ')}.`);
  if (cap.followups && cap.followups.length) parts.push(`Followups: ${cap.followups.join(', ')}.`);
  return parts.join(' ');
}

function beginOperation(label) {
  const cap = capabilityByKey(label);
  currentOp = {
    label,
    purpose: cap ? capabilityPurpose(cap) : operationPurpose(label),
    state: 'running',
    method: 'pending',
    path: 'pending',
    payload: {},
    outputKeys: [],
    latency: 0,
    receipt: 'none',
    delta: 'waiting for response',
    error: '',
    beforeState: lastState ? JSON.parse(JSON.stringify(lastState)) : null
  };
  renderOperation();
}

function finishOperation(ok, data, err) {
  if (!currentOp) return;
  currentOp.state = ok ? 'ok' : 'error';
  currentOp.outputKeys = outputKeys(data || {});
  currentOp.receipt = receiptOf(data || {});
  const after = (data && (data.state || data.state_after || (data.vocabulary_size ? data : null))) || lastState;
  currentOp.delta = ok ? stateDelta(currentOp.beforeState, after) : 'failed before state update';
  currentOp.error = ok ? 'none' : String((err && (err.message || err)) || 'unknown error');
  opHistory.unshift({
    label: currentOp.label,
    state: currentOp.state,
    route: `${currentOp.method} ${currentOp.path}`,
    latency: currentOp.latency,
    keys: currentOp.outputKeys,
    delta: currentOp.delta
  });
  opHistory = opHistory.slice(0, 8);
  renderOperation();
}

function renderOperation() {
  if (!currentOp) return;
  const set = (id, value) => { const el = document.getElementById(id); if (el) el.textContent = value; };
  set('opButton', currentOp.label || 'none');
  set('opState', currentOp.state || 'idle');
  set('opLatency', `${currentOp.latency || 0} ms`);
  set('opReceipt', currentOp.receipt || 'none');
  set('opRoute', `${currentOp.method || 'pending'} ${currentOp.path || 'pending'}${currentOp.httpStatus ? ' -> HTTP ' + currentOp.httpStatus : ''}`);
  set('opPurpose', currentOp.purpose || operationPurpose(currentOp.label));
  set('opPayload', JSON.stringify(currentOp.payload || {}, null, 2));
  set('opKeys', (currentOp.outputKeys || []).join(', ') || 'none');
  set('opDelta', currentOp.delta || 'none');
  set('opError', currentOp.error || 'none');
  const hist = document.getElementById('opHistory');
  if (hist) {
    hist.innerHTML = opHistory.map(item => `<div class="${item.state === 'ok' ? 'ok' : 'bad'}"><b>${escapeHtml(item.label)}</b> ${escapeHtml(item.route)} | ${item.latency || 0} ms<br>${escapeHtml(item.delta || '')}</div>`).join('');
  }
}

function renderContractMatrix() {
  const grid = document.getElementById('contractGrid');
  if (!grid) return;
  if (!capabilityManifest || !Array.isArray(capabilityManifest.capabilities)) {
    grid.innerHTML = '<div class="muted">No capability manifest loaded.</div>';
    return;
  }
  const filter = (document.getElementById('contractFilter') || {}).value || 'all';
  const caps = capabilityManifest.capabilities.filter(cap => filter === 'all' || cap.risk_class === filter || cap.group === filter);
  grid.innerHTML = caps.map(cap => {
    const reqs = (cap.prerequisites || []).length ? cap.prerequisites.join(', ') : 'none';
    const expected = (cap.expected_outputs || []).join(', ') || 'unspecified';
    const followups = (cap.followups || []).join(', ') || 'none';
    const payload = escapeHtml(JSON.stringify(cap.payload || {}, null, 2));
    const schema = escapeHtml(JSON.stringify(cap.payload_schema || {}, null, 2));
    return `<article class="contractCard ${escapeHtml(cap.risk_class || '')}">
      <h3>${escapeHtml(cap.label || cap.key)}</h3>
      <div class="contractMeta"><b>key</b> ${escapeHtml(cap.key)} | <b>group</b> ${escapeHtml(cap.group)} | <b>risk</b> ${escapeHtml(cap.risk_class || '')}</div>
      <div class="contractMeta"><b>route</b> ${escapeHtml(cap.method)} ${escapeHtml(cap.path)}</div>
      <div class="contractMeta"><b>purpose</b> ${escapeHtml(cap.teaches || '')}</div>
      <div class="contractMeta"><b>requisites</b> ${escapeHtml(reqs)}</div>
      <div class="contractMeta"><b>expected</b> ${escapeHtml(expected)}</div>
      <div class="contractMeta"><b>followups</b> ${escapeHtml(followups)}</div>
      <details><summary>Payload</summary><pre>${payload}</pre></details>
      <details><summary>Schema</summary><pre>${schema}</pre></details>
      <div class="contractActions">
        <button data-cap-inspect="${escapeHtml(cap.key)}" title="Load this route contract into the operation relay without executing it.">Inspect</button>
        <button data-cap-run="${escapeHtml(cap.key)}" title="Execute ${escapeHtml(cap.method)} ${escapeHtml(cap.path)} with the default payload shown in this card.">Run</button>
      </div>
    </article>`;
  }).join('');
  bindContractButtons();
}

function inspectCapability(key) {
  const cap = capabilityByKey(key);
  if (!cap) return;
  currentOp = {
    label: cap.key,
    purpose: capabilityPurpose(cap),
    state: 'contract',
    method: cap.method,
    path: cap.path,
    payload: cap.payload || {},
    outputKeys: cap.expected_outputs || [],
    latency: 0,
    receipt: 'none',
    delta: cap.mutates ? 'will mutate if executed' : 'read-only if executed',
    error: (cap.prerequisites || []).length ? `requisites: ${cap.prerequisites.join(', ')}` : 'none'
  };
  renderOperation();
  $('facilityPayload').value = JSON.stringify(cap.payload || {}, null, 2);
  if ($('facilitySelect')) $('facilitySelect').value = cap.key;
}

async function runCapabilityContract(key) {
  const cap = capabilityByKey(key);
  if (!cap) throw new Error('Unknown capability: ' + key);
  return run(cap.key, () => {
    const payload = cap.payload || {};
    if (cap.method === 'GET') return api(cap.path);
    return api(cap.path, payload);
  });
}

function bindContractButtons() {
  document.querySelectorAll('[data-cap-inspect]').forEach(btn => {
    btn.addEventListener('click', () => inspectCapability(btn.dataset.capInspect));
  });
  document.querySelectorAll('[data-cap-run]').forEach(btn => {
    btn.addEventListener('click', () => runCapabilityContract(btn.dataset.capRun));
  });
}

function previewOperation(btn) {
  if (!btn || busy) return;
  const label = (btn.textContent || btn.title || btn.id || 'control').trim();
  currentOp = {
    label,
    purpose: btn.title || operationPurpose(label),
    state: 'preview',
    method: 'not fired',
    path: 'press to execute',
    payload: {},
    outputKeys: [],
    latency: 0,
    receipt: 'none',
    delta: 'no state change yet',
    error: 'none'
  };
  renderOperation();
}

function setStatus(text, kind='') {
  $('globalStatus').textContent = text;
  $('globalStatus').className = 'status ' + kind;
  setSurfaceStatus(text || 'ready');
}

function setHud(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function setSurfaceStatus(text) {
  surfaceState.status = String(text || 'ready').slice(0, 120);
  setHud('surfaceHud', surfaceState.status);
}

function pushReaction(title, body, kind='exec') {
  const dock = document.getElementById('reactionDock');
  if (!dock) return;
  const div = document.createElement('div');
  div.className = 'reaction ' + kind;
  div.innerHTML = `<b>${escapeHtml(title)}</b><span>${escapeHtml(body || '')}</span>`;
  dock.prepend(div);
  while (dock.children.length > 5) dock.removeChild(dock.lastChild);
  setTimeout(() => {
    if (div.parentNode) div.style.opacity = '0.78';
  }, 3200);
}

function summarizeResult(data) {
  if (!data || typeof data !== 'object') return 'Action completed.';
  if (data.response) return data.response.slice(0, 180);
  if (typeof data.collective_catches !== 'undefined') {
    return `Sphere ran ${data.frames_run || 0} frames: ${data.collective_catches || 0} catches, ${data.collective_misses || 0} misses, receipt ${String(data.receipt || '').slice(0, 18)}...`;
  }
  if (data.trainer_card) return data.trainer_card.simple_summary || 'Training cycle complete.';
  if (data.simple_summary) return data.simple_summary;
  if (data.plain_summary) return data.plain_summary;
  if (data.checks) return `${data.checks.filter(x => x.ok).length}/${data.checks.length} checks passed.`;
  if (data.cards) return `${data.cards.length} system cards loaded.`;
  if (data.nodes && data.edges) return `${data.nodes.length} nodes, ${data.edges.length} routes mapped.`;
  if (data.receipt) return `Receipt ${String(data.receipt).slice(0, 24)}...`;
  if (data.vocabulary_size) return `Live state: ${data.vocabulary_size} words, ${data.knowledge_relations || 0} relations.`;
  return 'Action completed and returned data.';
}

function updateSurfaceFromState(s) {
  if (!s) return;
  surfaceState.active = s.active_cocoon_name || s.pet_name || 'local cocoon';
  setHud('activeHud', surfaceState.active);
}

function updateSurfaceFromResult(data) {
  if (!data || typeof data !== 'object') return;
  if (data.state) updateSurfaceFromState(data.state);
  if (typeof data.collective_catches !== 'undefined' || typeof data.collective_misses !== 'undefined') {
    surfaceState.mode = 'sphere';
    surfaceState.sphere = {
      catches: Number(data.collective_catches || 0),
      misses: Number(data.collective_misses || 0),
      misses_limit: Number(data.misses_limit || 3),
      streak: Number(data.best_streak || 0),
      frames: Number(data.frames_run || data.frames_requested || 0),
      balls: Number(data.balls || 1),
      rewards: data.rewards || {}
    };
    setHud('sphereHud', `catch ${surfaceState.sphere.catches} / miss ${surfaceState.sphere.misses} / streak ${surfaceState.sphere.streak}`);
  }
  if (data.env || data.runner_result) {
    surfaceState.mode = 'gym';
    const rr = data.runner_result || {};
    const reward = rr.total_reward ?? rr.reward ?? rr.mean_reward ?? null;
    surfaceState.gym = {
      env: data.env || rr.env || 'gym',
      episodes: Number(data.episodes || rr.episodes || 0),
      reward
    };
    setHud('gymHud', `${surfaceState.gym.env} x${surfaceState.gym.episodes || 1}${reward == null ? '' : ' reward ' + Number(reward).toFixed(2)}`);
  }
  if (Array.isArray(data.stages)) {
    const sphereStage = data.stages.find(s => s && s.result && typeof s.result.collective_catches !== 'undefined');
    const gymStage = data.stages.find(s => s && s.result && (s.result.env || s.result.runner_result));
    if (sphereStage) updateSurfaceFromResult(sphereStage.result);
    if (gymStage) updateSurfaceFromResult(gymStage.result);
  }
}

function drawSurface() {
  const canvas = document.getElementById('sphereCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  surfaceState.tick += 1;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#020404';
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = 'rgba(0,255,157,0.12)';
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 48) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }
  for (let y = 0; y < h; y += 48) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  const cx = w * 0.5;
  const cy = h * 0.52;
  const radius = Math.min(w, h) * 0.31;
  ctx.strokeStyle = 'rgba(0,209,255,0.55)';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.stroke();
  ctx.strokeStyle = 'rgba(255,204,0,0.3)';
  ctx.beginPath();
  ctx.arc(cx, cy, radius * 0.62, 0, Math.PI * 2);
  ctx.stroke();

  const catches = surfaceState.sphere.catches;
  const misses = surfaceState.sphere.misses;
  const balls = Math.max(1, surfaceState.sphere.balls || 1);
  for (let i = 0; i < balls; i += 1) {
    const angle = (surfaceState.tick * 0.022) + i * (Math.PI * 2 / balls) + catches * 0.09;
    const r = radius * (0.68 + 0.18 * Math.sin(surfaceState.tick * 0.017 + i));
    const x = cx + Math.cos(angle) * r;
    const y = cy + Math.sin(angle) * r;
    ctx.fillStyle = '#ffcc00';
    ctx.beginPath();
    ctx.arc(x, y, 9 + Math.min(7, catches), 0, Math.PI * 2);
    ctx.fill();
  }

  const agents = Math.max(2, Math.min(8, Object.keys(surfaceState.sphere.rewards || {}).length || 4));
  for (let i = 0; i < agents; i += 1) {
    const angle = -surfaceState.tick * 0.012 + i * (Math.PI * 2 / agents);
    const x = cx + Math.cos(angle) * radius * 0.43;
    const y = cy + Math.sin(angle) * radius * 0.43;
    ctx.fillStyle = i % 2 ? '#00d1ff' : '#00ff9d';
    ctx.fillRect(x - 8, y - 8, 16, 16);
  }

  const missPct = Math.min(1, misses / Math.max(1, surfaceState.sphere.misses_limit || misses || 3));
  ctx.fillStyle = 'rgba(255,62,62,0.18)';
  ctx.fillRect(0, h - 18, w * missPct, 18);
  ctx.fillStyle = '#00ff9d';
  ctx.font = '18px Courier New, monospace';
  ctx.fillText(surfaceState.active, 24, 36);
  ctx.fillStyle = '#a0b0b0';
  ctx.font = '14px Courier New, monospace';
  ctx.fillText(`mode ${surfaceState.mode} | ${surfaceState.status}`, 24, 60);
  ctx.fillText(`sphere catches ${catches} misses ${misses} frames ${surfaceState.sphere.frames}`, 24, h - 34);

  if (currentOp) {
    const panelX = w - 390;
    const panelY = 28;
    ctx.fillStyle = 'rgba(0,0,0,0.76)';
    ctx.fillRect(panelX, panelY, 360, 150);
    ctx.strokeStyle = currentOp.state === 'error' ? '#ff3e3e' : (currentOp.state === 'ok' ? '#00ff9d' : '#ffcc00');
    ctx.lineWidth = 2;
    ctx.strokeRect(panelX, panelY, 360, 150);
    ctx.fillStyle = '#ffcc00';
    ctx.font = '14px Courier New, monospace';
    ctx.fillText('LIVE OPERATION', panelX + 14, panelY + 24);
    ctx.fillStyle = '#00d1ff';
    ctx.font = '13px Courier New, monospace';
    const route = `${currentOp.method || 'pending'} ${currentOp.path || 'pending'}`.slice(0, 42);
    ctx.fillText(route, panelX + 14, panelY + 48);
    ctx.fillStyle = '#dce8e8';
    ctx.fillText(`button: ${String(currentOp.label || 'none').slice(0, 32)}`, panelX + 14, panelY + 72);
    ctx.fillText(`status: ${currentOp.state || 'idle'}  ${currentOp.latency || 0}ms`, panelX + 14, panelY + 96);
    ctx.fillText(`keys: ${(currentOp.outputKeys || []).slice(0, 4).join(', ') || 'waiting'}`.slice(0, 44), panelX + 14, panelY + 120);
    ctx.fillText(`delta: ${String(currentOp.delta || 'waiting').slice(0, 38)}`, panelX + 14, panelY + 144);
  }
  requestAnimationFrame(drawSurface);
}


function logTerm(msg, kind) {
  kind = kind || 'INFO';
  var term = document.getElementById('terminal');
  if (!term) return;
  var div = document.createElement('div');
  var time = new Date().toLocaleTimeString().split(' ')[0];
  var color = kind === 'DONE' ? 'var(--accent)' : (kind === 'ERROR' ? 'var(--warn)' : 'var(--gold)');
  div.innerHTML = '<span style="color:var(--muted)">[' + time + ']</span> <span style="color:' + color + '">[' + kind + ']</span> ' + escapeHtml(msg);
  term.appendChild(div);
  term.scrollTop = term.scrollHeight;
  if (term.childNodes.length > 64) term.removeChild(term.firstChild);
}

async function run(label, fn) {
  if (busy) {
    pushReaction('Still Running', 'The current action is still working. Watch the terminal/status for completion.', 'exec');
    return;
  }
  busy = true;
  beginOperation(label);
  setStatus(label + '...', '');
  logTerm('Initiating ' + label, 'EXEC');
  pushReaction(label, 'Command accepted. Waiting for the cocoon response.', 'exec');
  if (activeButton) {
    activeButton.disabled = true;
    activeButton.classList.add('buttonPulse');
  }
  surfaceState.mode = 'working';
  try {
    const data = await fn();
    show(data);
    finishOperation(true, data, null);
    setStatus(label + ' complete', 'ok');
    logTerm(label + ' success', 'DONE');
    pushReaction(label + ' Complete', summarizeResult(data), 'ok');
  } catch (err) {
    finishOperation(false, {}, err);
    setStatus(String(err.message || err), 'bad');
    $('readable').textContent = String(err.stack || err.message || err);
    logTerm(label + ' failed: ' + err.message, 'ERROR');
    pushReaction(label + ' Failed', String(err.message || err), 'bad');
  } finally {
    busy = false;
    if (activeButton) {
      activeButton.disabled = false;
      activeButton.classList.remove('buttonPulse');
      activeButton = null;
    }
  }
}

function show(data) {
  $('out').textContent = JSON.stringify(data, null, 2);
  $('readable').classList.remove('resultFlash');
  void $('readable').offsetWidth;
  $('readable').classList.add('resultFlash');
  const state = data.state || data.state_after || data;
  if (state && state.vocabulary_size) renderState(state);
  updateSurfaceFromResult(data);
  if (data.teacher_note) renderTeacher(data.teacher_note);
  renderReadable(data);
}

function renderReadable(data) {
  const lines = [];
  if (data.response) lines.push('Cocoon: ' + data.response);
  if (data.trainer_card) {
    lines.push('Battle Card: ' + data.trainer_card.simple_summary);
    lines.push('Do Next: ' + data.trainer_card.you_do);
    if (data.trainer_card.cocoon_got) {
      lines.push('Cocoon Got: ' + JSON.stringify(data.trainer_card.cocoon_got));
    }
  }
  if (data.plain_summary) lines.push('Deepening: ' + data.plain_summary);
  if (data.simple_summary) lines.push('Hardening: ' + data.simple_summary);
  if (data.checks) {
    const ok = data.checks.filter(x => x.ok).length;
    lines.push('Checks: ' + ok + '/' + data.checks.length + ' passed');
  }
  if (data.stages) lines.push('Stages: ' + data.stages.map(s => s.stage).join(' -> '));
  if (data.teacher_note) lines.push('Note: ' + data.teacher_note.summary);
  if (data.receipt) lines.push('Receipt: ' + data.receipt);
  if (data.faculty) lines.push('Faculty: ' + data.faculty);
  if (data.mode) lines.push('Mode: ' + data.mode);
  if (data.rounds) lines.push('Rounds: ' + data.rounds);
  if (data.losses && data.losses.length) lines.push('Latest loss: ' + data.losses[data.losses.length - 1].toFixed(5));
  if (data.informational_wealth) lines.push('Value: ' + data.informational_wealth.value);
  $('readable').textContent = lines.join('\n') || JSON.stringify(data, null, 2).slice(0, 1400);
  if (data.informational_wealth) renderWealth(data.informational_wealth);
  if (data.state && data.state.last_informational_wealth) renderWealth(data.state.last_informational_wealth);
  if (data.nodes && data.edges) renderMap(data);
  if (data.map && data.map.nodes) renderMap(data.map);
  if (data.cards) renderManual(data);
}

function renderState(s) {
  lastState = s;
  updateSurfaceFromState(s);
  const xp = Math.max(0, Number(s.cycles || 0) * 12 + Number(s.training_logs || 0));
  const level = Math.max(1, Math.floor(xp / 250) + 1);
  const xpPct = Math.min(100, Math.round((xp % 250) / 250 * 100));
  const badgeNames = [];
  if ((s.conversation_turns || 0) > 0) badgeNames.push('Chat Badge');
  if ((s.knowledge_relations || 0) > 0) badgeNames.push('Link Badge');
  if ((s.last_receipt || '').length > 8) badgeNames.push('Receipt Badge');
  if ((s.recent_lessons || []).length > 0) badgeNames.push('Drill Badge');
  $('partnerCard').innerHTML = `
    <div class="partnerName">${escapeHtml(s.active_cocoon_name || s.pet_name || 'Cocoon')} Lv.${level}</div>
    <div class="muted">Active entity: ${escapeHtml(s.active_cocoon_path || s.source_cocoon || 'local cocoon')}</div>
    <div class="xpbar" title="XP is estimated from cycles and training logs."><div class="xpfill" style="width:${xpPct}%"></div></div>
    <div>${badgeNames.map(b => `<span class="badge" title="Earned from live cocoon state">${b}</span>`).join('')}</div>
  `;
  const stats = [
    ['Active Cocoon', s.active_cocoon_name || s.pet_name || 'unknown'],
    ['Vocabulary', s.vocabulary_size],
    ['Relations', s.knowledge_relations],
    ['Turns', s.conversation_turns],
    ['Logs', s.training_logs],
    ['Cycles', s.cycles],
    ['Receipt', s.last_receipt || 'none']
  ];
  $('stats').innerHTML = stats.map(([k,v]) => `<div class="stat" title="${k}"><span>${k}</span><b>${v}</b></div>`).join('');
  $('faculties').innerHTML = Object.entries(s.action_faculties || {}).map(([k,v]) => `<span class="pill" title="action head ${k}">${k}: ${v}</span>`).join('');
  if (s.last_informational_wealth) renderWealth(s.last_informational_wealth);
  if (s.recent_lessons && s.recent_lessons.length) renderTeacher(s.recent_lessons[s.recent_lessons.length - 1]);
}

function renderWealth(w) {
  const labels = w.action_labels || ['observe','associate','recall','reason','respond','verify'];
  const probs = w.action_probs || [];
  const active = probs.indexOf(Math.max(...probs));
  $('procession').innerHTML = labels.map((label, i) => {
    const p = probs[i] == null ? 0 : Math.round(probs[i] * 100);
    return `<div class="step ${i === active ? 'active' : ''}" title="${label} routing probability"><b>${label}</b><span>${p}%</span></div>`;
  }).join('');
}

function renderTeacher(note) {
  if (!note) return;
  const cocoon = note.cocoon_learns || {};
  const you = (note.you_learn || []).slice(0, 3).map(x => `<div>${escapeHtml(x)}</div>`).join('');
  const them = [
    `Faculty: ${escapeHtml(cocoon.faculty || 'unknown')}`,
    `Anchors: ${escapeHtml((cocoon.anchors || []).join(', ') || 'none')}`,
    `Novel: ${escapeHtml((cocoon.novel || []).join(', ') || 'none')}`,
  ].map(x => `<div>${x}</div>`).join('');
  $('duelLesson').innerHTML = `
    <div class="duelBox" title="What the trainer should understand from this cycle."><b>Your XP</b>${you}</div>
    <div class="duelBox" title="What the active cocoon received as trainable structure."><b>Cocoon XP</b>${them}</div>
  `;
  $('lessonBox').innerHTML = `
    <details open><summary>Teacher note</summary><pre>${escapeHtml(JSON.stringify(note, null, 2))}</pre></details>
    <details><summary>Anchors</summary><pre>${escapeHtml(JSON.stringify(cocoon.anchors || [], null, 2))}</pre></details>
    <details><summary>Novel terms</summary><pre>${escapeHtml(JSON.stringify(cocoon.novel || [], null, 2))}</pre></details>
  `;
}

function renderWealthSurface(data) {
  if (!data || typeof data !== 'object') return;
  if (data.state) renderState(data.state);
  if (data.facility_map) renderMap(data.facility_map);
  if (data.informational_wealth && Object.keys(data.informational_wealth).length) renderWealth(data.informational_wealth);
  const summary = data.summary || {};
  const diagnostics = data.diagnostics || {};
  const cascade = data.cascade || {};
  const quine = data.quine || {};
  const mcp = data.mcp_stdio || {};
  const status = document.getElementById('wealthStatus');
  if (status) {
    const stamp = data.updated ? new Date(data.updated * 1000).toLocaleTimeString() : 'unknown';
    status.textContent = `live ${stamp} | ${summary.active || 'cocoon'} | vocab ${summary.vocabulary_size || 0} | relations ${summary.relations || 0} | receipt ${String(summary.last_receipt || 'none').slice(0, 24)}`;
  }
  const grid = document.getElementById('wealthGrid');
  if (grid) {
    const depErrors = Object.keys(diagnostics.dependency_errors || {});
    const groups = ((data.capability_manifest || {}).groups || []).join(', ');
    grid.innerHTML = [
      ['Capability Fabric', `${diagnostics.capability_count || 0} routes | safe ${diagnostics.safe_phone_count || 0} | mutating ${diagnostics.mutating_count || 0}`, groups || 'no groups'],
      ['Cascade Lattice', cascade.available ? 'available' : 'limited', `receipts ${(cascade.latest_receipts || []).length} | tape ${cascade.tape_path || 'none'}`],
      ['Quinesmith', quine.available ? `${quine.ladder_count || 0} ladders` : 'not available', quine.boundary || 'syntax boundary'],
      ['MCP Stdio', mcp.command || './cocoon mcp', (mcp.tools || []).join(', ')],
      ['Dependencies', depErrors.length ? `blocked: ${depErrors.join(', ')}` : 'ready dependencies only', JSON.stringify(diagnostics.dependency_probe || {})],
      ['Events', `${summary.events || 0} recorded`, `${summary.lessons || 0} lessons | cycles ${summary.cycles || 0}`],
    ].map(([title, body, detail]) => `<div class="toolCard ready">
      <b>${escapeHtml(title)}</b>
      <p>${escapeHtml(body)}</p>
      <small>${escapeHtml(detail)}</small>
    </div>`).join('');
  }
  const receipts = document.getElementById('wealthReceipts');
  if (receipts) {
    const rows = (cascade.latest_receipts || []).slice(-10).reverse();
    receipts.innerHTML = '<b>Receipt Relay</b><br>' + (rows.length ? rows.map(r => `${escapeHtml(r.kind || 'event')} -> ${escapeHtml(String(r.receipt || '').slice(0, 48))}`).join('<br>') : 'no receipts yet');
  }
  const events = document.getElementById('wealthEvents');
  if (events) {
    const rows = (data.events || []).slice(-10).reverse();
    events.innerHTML = '<b>Event Relay</b><br>' + (rows.length ? rows.map(e => {
      const keys = e.data && typeof e.data === 'object' ? Object.keys(e.data).slice(0, 8).join(',') : '';
      return `${escapeHtml(e.kind || 'event')} | ${escapeHtml(keys)}`;
    }).join('<br>') : 'no events yet');
  }
  if (data.lessons && data.lessons.length) renderTeacher(data.lessons[data.lessons.length - 1]);
}

async function refreshWealthSurface(silent=false) {
  try {
    const res = await fetch('/api/wealth_surface');
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || JSON.stringify(data));
    renderWealthSurface(data);
    if (!silent) {
      show(data);
      logTerm('Wealth surface refreshed', 'DONE');
    }
    return data;
  } catch (err) {
    const status = document.getElementById('wealthStatus');
    if (status) status.textContent = 'wealth relay error: ' + String(err.message || err);
    if (!silent) throw err;
  }
}

function renderMap(map) {
  const nodes = map.nodes || [];
  const edges = map.edges || [];
  const html = nodes.map(n => {
    const status = String(n.status || '').toLowerCase();
    const cls = status.includes('ready') ? 'ready' : (status.includes('need') || status.includes('block') ? 'needs' : '');
    return `<div class="node ${cls}" title="${escapeHtml(n.type || '')}: ${escapeHtml(n.status || '')}">
      <b>${escapeHtml(n.label || n.id)}</b>
      <span>${escapeHtml(n.type || '')}</span><br>
      <span>${escapeHtml(n.status || '')}</span>
    </div>`;
  }).join('');
  const edgeHtml = edges.slice(0, 24).map(e => `${escapeHtml(e.from)} -> ${escapeHtml(e.to)} (${escapeHtml(e.label || '')})`).join('<br>');
  if ($('facilityMap')) $('facilityMap').innerHTML = html;
  if ($('ledgerMap')) $('ledgerMap').innerHTML = html;
  if ($('facilityEdges')) $('facilityEdges').innerHTML = edgeHtml;
}

function renderManual(manual) {
  const cards = manual.cards || [];
  const protocols = manual.facility_protocols || {};
  $('manualGrid').innerHTML = cards.map(card => {
    const needs = (card.needs || []).length ? `Needs: ${(card.needs || []).map(escapeHtml).join(', ')}` : 'Ready now or already mirrored.';
    const cls = String(card.status || '').includes('need') ? 'needs' : 'ready';
    const payload = escapeHtml(JSON.stringify(card.payload || {}));
    const protocol = protocols[card.key] || {};
    const ladder = (protocol.ladder || []).map(x => `<li>${escapeHtml(x)}</li>`).join('');
    const schema = escapeHtml(JSON.stringify(protocol.schema || {}, null, 2));
    return `<div class="toolCard ${cls}">
      <b>${escapeHtml(card.name)}</b>
      <p>${escapeHtml(card.plain)}</p>
      <small><b>Trains:</b> ${escapeHtml(card.teaches)}</small>
      <small><b>Status:</b> ${escapeHtml(card.status || 'ready')}</small>
      <small>${needs}</small>
      <details><summary>Deep stuff</summary><small>${escapeHtml(protocol.use || '')}</small><ul>${ladder}</ul><pre>${schema}</pre></details>
      <button data-facility-card="${escapeHtml(card.run_facility)}" data-payload="${payload}" title="Run ${escapeHtml(card.name)} through the facility runner.">Run</button>
      <button data-deepen-card="${escapeHtml(card.run_facility)}" title="Run a staged deepening ladder for ${escapeHtml(card.name)}.">Deepen</button>
    </div>`;
  }).join('');
  document.querySelectorAll('[data-facility-card]').forEach(btn => {
    btn.addEventListener('click', () => run('system card', () => {
      let payload = {};
      try { payload = JSON.parse(btn.dataset.payload || '{}'); } catch (err) { payload = {}; }
      payload.facility = btn.dataset.facilityCard;
      return api('/api/run_facility', payload);
    }));
  });
  document.querySelectorAll('[data-deepen-card]').forEach(btn => {
    btn.addEventListener('click', () => run('deepen system', () => api('/api/deepen_facility', {
      facility: btn.dataset.deepenCard,
      depth: 2
    })));
  });
  if (manual.facility_map) renderMap(manual.facility_map);
}

function renderCocoons(data) {
  const select = $('cocoonSelect');
  if (!select) return;
  const active = data && data.active ? data.active : null;
  const cocoons = Array.isArray(data && data.cocoons) ? data.cocoons : [];
  select.innerHTML = cocoons.map(item => {
    const selected = item.active ? ' selected' : '';
    return `<option value="${escapeHtml(item.path || '')}"${selected}>${escapeHtml(item.display_name || item.name || item.path || 'cocoon')}</option>`;
  }).join('');
  if (active && active.path) select.value = active.path;
  const status = $('cocoonStatus');
  if (status) {
    status.textContent = active
      ? `Active: ${active.name} at ${active.path} | ${active.organisms} organism(s) | ${cocoons.length} local cocoon(s) found`
      : 'No active cocoon selected.';
  }
}

function addChat(who, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + (String(who).toLowerCase() === 'you' ? 'user' : 'assistant');
  div.innerHTML = `<b>${escapeHtml(who)}</b><br>${escapeHtml(text)}`;
  $('chatLog').appendChild(div);
  $('chatLog').scrollTop = $('chatLog').scrollHeight;
}

function addCouncilChat(who, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + (String(who).toLowerCase() === 'you' ? 'user' : 'assistant');
  const body = escapeHtml(String(text || '')).replace(/\n/g, '<br>');
  div.innerHTML = `<b>${escapeHtml(who)}</b><br>${body}`;
  const log = $('councilLog');
  if (log) {
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function bind() {
  const b = (id, evt, fn) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener(evt, (event) => {
      activeButton = event.currentTarget;
      fn(event);
    });
  };

  document.addEventListener('click', (event) => {
    const btn = event.target && event.target.closest ? event.target.closest('button') : null;
    if (!btn) return;
    activeButton = btn;
    btn.classList.add('buttonPulse');
    setTimeout(() => btn.classList.remove('buttonPulse'), 450);
    const label = (btn.textContent || btn.title || 'button').trim();
    if (label) pushReaction('Pressed', label, 'exec');
  }, true);

  document.querySelectorAll('button').forEach(btn => {
    btn.addEventListener('mouseenter', () => previewOperation(btn));
    btn.addEventListener('focus', () => previewOperation(btn));
  });

  const contractFilter = document.getElementById('contractFilter');
  if (contractFilter) contractFilter.addEventListener('change', renderContractMatrix);
  b('contractReload', 'click', () => run('capability manifest', async () => {
    const data = await api('/api/capability_manifest');
    capabilityManifest = data;
    renderContractMatrix();
    return data;
  }));

  document.querySelectorAll('[data-tab]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-tab]').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
      btn.classList.add('active');
      const view = document.getElementById('view-' + btn.dataset.tab);
      if (view) view.classList.add('active');
      logTerm('Switched to ' + btn.dataset.tab, 'NAV');
    });
  });

  b('chatSend', 'click', () => run('chat', async () => {
    const text = $('chatText').value.trim();
    if (!text) throw new Error('Type a chat message first.');
    addChat('You', text);
    const data = await api('/api/native/chat', {prompt:text, learn:$('chatLearn').value === 'true', steps:Number($('chatSteps').value || 0)});
    addChat(lastState && lastState.active_cocoon_name ? lastState.active_cocoon_name : 'Cocoon', data.response || '');
    $('chatText').value = '';
    return data;
  }));

  b('councilRun', 'click', () => run('butterfly council', async () => {
    const text = $('councilText').value.trim();
    if (!text) throw new Error('Type a council prompt first.');
    const learn = $('councilLearn').value === 'true';
    const steps = Number($('councilSteps').value || 0);
    const maxCocoons = Number($('councilLimit').value || 8);
    const exportAfter = $('councilExport').value === 'true';
    addCouncilChat('You', text);
    const data = await api('/api/council', {
      prompt: text,
      learn,
      steps,
      max_cocoons: maxCocoons,
      export_after: exportAfter
    });
    const selected = data.selected_member && data.selected_member.name ? data.selected_member.name : 'none';
    addCouncilChat('Council', `${data.consensus || data.response || ''}\n\nmembers: ${data.successful_members || 0}/${data.member_count || 0}\nselected: ${selected}`);
    $('councilStatus').textContent = `members ${data.successful_members || 0}/${data.member_count || 0} | selected ${selected} | receipt ${String(data.receipt || 'none').slice(0, 24)}`;
    $('councilText').value = '';
    return data;
  }));

  b('followRun', 'click', () => run('drill', () => api('/api/follow_along', {
    mode:$('followMode').value, seed:$('followSeed').value, rounds:Number($('followRounds').value || 1), steps:Number($('followSteps').value || 0)
  })));

  b('quineRun', 'click', () => run('quine', () => api('/api/quine_drill', {limit:8, steps:2})));
  b('sphereRun', 'click', () => run('sphere burst', () => api('/api/native/sphere_burst', {frames:120, balls:1, misses:3, train:false})));
  b('signalTrain', 'click', () => run('pillage', () => api('/api/signal', {text:$('signal').value, steps:Number($('signalSteps').value || 0)})));
  b('signalGovern', 'click', () => run('arbitrate', () => api('/api/govern', {text:$('signal').value, steps:Number($('signalSteps').value || 0)})));
  b('relRun', 'click', () => run('relate', () => api('/api/relate', {source:$('relSource').value, target:$('relTarget').value, steps:Number($('relSteps').value || 0)})));
  b('feedRun', 'click', () => run('scout & raid', () => api('/api/feed', {url:$('feedUrl').value, max_items:Number($('feedItems').value || 1), steps:Number($('feedSteps').value || 0)})));
  
  b('facilitiesRun', 'click', () => run('facilities', () => api('/api/native/facilities')));
  b('manualRun', 'click', () => run('systems manual', () => api('/api/facility_manual')));
  b('tourRunTop', 'click', () => run('safe tour', () => api('/api/facility_tour', {})));
  b('hardenRun', 'click', () => run('harden', () => api('/api/harden_facilities')));
  b('deepenAllRun', 'click', () => run('deepen', () => api('/api/deepen_facility', {facility:'autopilot', depth:2})));
  b('cocoonRefresh', 'click', () => run('refresh inventory', async () => {
    const data = await api('/api/cocoons');
    renderCocoons(data);
    return data;
  }));
  b('cocoonReload', 'click', () => run('reload cocoon', async () => {
    const path = $('cocoonSelect').value;
    if (!path) throw new Error('No cocoon selected.');
    const data = await api('/api/reload', {path});
    const inventory = await api('/api/cocoons');
    renderCocoons(inventory);
    return data;
  }));
  
  b('autopilotRun', 'click', () => run('armada launch', () => api('/api/autopilot', {
    mission:$('autoMission').value,
    feeds:$('autoFeeds').value,
    depth:Number($('autoDepth').value || 1),
    include_games:$('autoGames').value === 'true',
    save_after:$('autoSave').value === 'true'
  })));

  b('facilityRun', 'click', () => run('facility run', () => {
    let payload = {};
    try { payload = JSON.parse($('facilityPayload').value || '{}'); }
    catch (err) { throw new Error('Invalid JSON payload.'); }
    payload.facility = $('facilitySelect').value;
    return api('/api/run_facility', payload);
  }));

  b('tourRun', 'click', () => run('tour', () => api('/api/facility_tour', {})));
  b('snapshotRun', 'click', () => run('snapshot', () => api('/api/native/snapshot')));
  b('saveRun', 'click', () => run('save', () => api('/api/native/save', {output_dir:'ui_native_state'})));
  b('vocabRun', 'click', () => run('vocab', () => api('/api/native/vocab?limit=120')));
  b('logsRun', 'click', () => run('logs', () => api('/api/native/training/logs?limit=30')));
  b('exportRun', 'click', () => run('export', () => api('/api/export', {name:'mira_kite_authority_checkpoint.py'})));
  b('refreshRun', 'click', () => run('refresh', () => api('/api/state')));
  b('clearOutput', 'click', () => { $('out').textContent = '{}'; $('readable').textContent = 'Cleared.'; logTerm('Output cleared', 'SYS'); });
  
  b('cheatRun', 'click', () => run('overclock', () => api('/api/cheat', {
    lr_boost: Number($('cheatLR').value),
    multiplier: Number($('cheatMult').value),
    steps: Number($('cheatSteps').value)
  })));

  b('chatText', 'keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      const send = document.getElementById('chatSend');
      if (send) send.click();
    }
  });
  b('councilText', 'keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      const send = document.getElementById('councilRun');
      if (send) send.click();
    }
  });
}

bind();
drawSurface();
run('initial refresh', async () => {
  const state = await api('/api/state');
  capabilityManifest = await api('/api/capability_manifest');
  const map = await api('/api/facility_map');
  const manual = await api('/api/facility_manual');
  const cocoons = await api('/api/cocoons');
  const wealth = await refreshWealthSurface(true);
  renderContractMatrix();
  renderMap(map);
  renderManual(manual);
  renderCocoons(cocoons);
  return state;
});
setInterval(() => refreshWealthSurface(true), 4000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    authority: AuthorityState

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[HTTP] {self.address_string()} {fmt % args}")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, indent=2, default=json_default).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path in {"/health", "/api/native/health"}:
                self.send_json(self.authority.native_health())
            elif parsed.path in {"/api/state", "/info", "/api/native/info"}:
                with self.authority.lock:
                    self.send_json(self.authority.capability_snapshot())
            elif parsed.path == "/api/events":
                with self.authority.lock:
                    self.send_json(self.authority.event_log)
            elif parsed.path == "/api/lessons":
                with self.authority.lock:
                    self.send_json(self.authority.education_log)
            elif parsed.path in {"/api/native/facilities", "/api/native/capabilities", "/capabilities"}:
                with self.authority.lock:
                    if parsed.path == "/api/native/facilities":
                        self.send_json(self.authority.native_facilities())
                    else:
                        self.send_json(self.authority.native_capabilities())
            elif parsed.path in {"/api/cocoons", "/api/native/cocoons"}:
                with self.authority.lock:
                    self.send_json(self.authority.cocoon_inventory())
            elif parsed.path in {"/api/capability_manifest", "/api/capabilities/fabric"}:
                self.send_json(capability_manifest())
            elif parsed.path == "/api/capability":
                key = (query.get("key") or [""])[0]
                cap = CAPABILITY_INDEX.get(key)
                if not cap:
                    self.send_json({"error": f"unknown capability: {key}", "known": sorted(CAPABILITY_INDEX)}, HTTPStatus.NOT_FOUND)
                else:
                    self.send_json(enriched_capability(cap))
            elif parsed.path in {"/api/native/curriculum", "/curriculum"}:
                with self.authority.lock:
                    self.send_json(self.authority.native_curriculum())
            elif parsed.path in {"/api/native/vocab", "/vocab"}:
                limit = query.get("limit", [None])[0]
                with self.authority.lock:
                    self.send_json(self.authority.native_vocab(int(limit) if limit else None))
            elif parsed.path in {"/api/native/training/logs", "/training/logs"}:
                limit = int(query.get("limit", [100])[0])
                with self.authority.lock:
                    self.send_json(self.authority.native_training_logs(limit))
            elif parsed.path in {"/api/native/snapshot", "/snapshot"}:
                full = query.get("full", ["0"])[0] in {"1", "true", "yes"}
                self.send_json(self.authority.native_snapshot(full=full))
            elif parsed.path == "/api/native/game_recipe":
                lane = query.get("lane", ["all"])[0]
                self.send_json(self.authority.native_game_recipe(lane))
            elif parsed.path == "/api/facility_map":
                self.send_json(self.authority.facility_map())
            elif parsed.path == "/api/wealth_surface":
                with self.authority.lock:
                    self.send_json(self.authority.wealth_surface())
            elif parsed.path == "/api/facility_manual":
                self.send_json(self.authority.complete_facility_manual())
            elif parsed.path == "/api/harden_facilities":
                self.send_json(self.authority.harden_facilities())
            else:
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_json({"error": str(exc), "trace": traceback.format_exc()}, 500)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self.read_json()
            if parsed.path == "/api/signal":
                self.send_json(self.authority.process_signal(str(payload.get("text", "")), int(payload.get("steps", 4))))
            elif parsed.path == "/api/govern":
                self.send_json(self.authority.govern(str(payload.get("text", "")), int(payload.get("steps", 2))))
            elif parsed.path == "/api/relate":
                self.send_json(self.authority.relate(str(payload.get("source", "")), str(payload.get("target", "")), int(payload.get("steps", 3))))
            elif parsed.path == "/api/perturb":
                self.send_json(self.authority.perturb(str(payload.get("mode", "stabilize")), float(payload.get("intensity", 0.2)), int(payload.get("steps", 4))))
            elif parsed.path == "/api/feed":
                self.send_json(self.authority.train_feed(str(payload.get("url", "")), int(payload.get("max_items", 4)), int(payload.get("steps", 4))))
            elif parsed.path == "/api/quine_drill":
                self.send_json(self.authority.quine_drill(int(payload.get("limit", 8)), int(payload.get("steps", 2))))
            elif parsed.path == "/api/follow_along":
                self.send_json(
                    self.authority.follow_along_game(
                        str(payload.get("mode", "cue")),
                        str(payload.get("seed", payload.get("text", ""))),
                        int(payload.get("rounds", 6)),
                        int(payload.get("steps", 2)),
                    )
                )
            elif parsed.path == "/api/autopilot":
                feeds = payload.get("feeds")
                if isinstance(feeds, str):
                    feeds = [line.strip() for line in feeds.splitlines() if line.strip()]
                self.send_json(
                    self.authority.autopilot_train(
                        str(payload.get("mission", payload.get("text", ""))),
                        feeds if isinstance(feeds, list) else None,
                        int(payload.get("depth", 1)),
                        bool(payload.get("include_games", False)),
                        bool(payload.get("save_after", False)),
                    )
                )
            elif parsed.path == "/api/run_facility":
                self.send_json(self.authority.run_facility(str(payload.get("facility", "health")), payload))
            elif parsed.path == "/api/facility_tour":
                self.send_json(self.authority.facility_tour())
            elif parsed.path == "/api/deepen_facility":
                self.send_json(
                    self.authority.deepen_facility(
                        str(payload.get("facility", "autopilot")),
                        int(payload.get("depth", 2)),
                    )
                )
            elif parsed.path == "/api/export":
                self.send_json(self.authority.export(payload.get("name")))
            elif parsed.path == "/api/reload":
                self.send_json(self.authority.reload_cocoon(str(payload.get("path", "")) or None))
            elif parsed.path in {"/api/native/chat", "/chat"}:
                prompt = payload.get("prompt", payload.get("message", ""))
                self.send_json(self.authority.native_chat(str(prompt), bool(payload.get("learn", True)), int(payload.get("steps", 1))))
            elif parsed.path == "/api/council":
                prompt = payload.get("prompt", payload.get("message", ""))
                self.send_json(
                    self.authority.council_chat(
                        str(prompt),
                        bool(payload.get("learn", True)),
                        int(payload.get("steps", 1)),
                        int(payload.get("max_cocoons", 8)),
                        bool(payload.get("export_after", False)),
                    )
                )
            elif parsed.path in {"/api/native/teach", "/teach"}:
                self.send_json(
                    self.authority.native_teach(
                        str(payload.get("text", "")),
                        float(payload.get("reward", 0.5)),
                        str(payload.get("stage", "authority_native_teach")),
                        str(payload.get("target", "")),
                        bool(payload.get("allow_tool_language", False)),
                        int(payload.get("steps", 1)),
                    )
                )
            elif parsed.path in {"/api/native/act", "/act", "/infer"}:
                action_space_size = payload.get("action_space_size", payload.get("actionSpaceSize"))
                self.send_json(
                    self.authority.native_act(
                        payload.get("state", []),
                        bool(payload.get("explore", False)),
                        int(action_space_size) if action_space_size is not None else None,
                    )
                )
            elif parsed.path in {"/api/native/learn", "/learn"}:
                self.send_json(self.authority.native_learn(payload))
            elif parsed.path in {"/api/native/curriculum/score", "/curriculum/score"}:
                self.send_json(self.authority.native_curriculum_score(payload))
            elif parsed.path in {"/api/native/save", "/save"}:
                self.send_json(self.authority.native_save(str(payload.get("output_dir", payload.get("dir", "mira_kite_native_state")))))
            elif parsed.path in {"/api/native/export", "/export"}:
                self.send_json(self.authority.export(payload.get("path", payload.get("output", payload.get("name")))))
            elif parsed.path in {"/api/native/dreamer/observe", "/dreamer/observe"}:
                self.send_json(self.authority.native_dreamer_observe(payload))
            elif parsed.path in {"/api/native/dreamer/propose", "/dreamer/propose"}:
                self.send_json(self.authority.native_dreamer_propose(payload))
            elif parsed.path == "/api/native/sphere_burst":
                seed = payload.get("seed")
                self.send_json(
                    self.authority.native_sphere_burst(
                        int(payload.get("frames", 120)),
                        int(payload.get("balls", 1)),
                        int(payload.get("misses", 3)),
                        bool(payload.get("train", False)),
                        int(seed) if seed is not None else None,
                    )
                )
            elif parsed.path == "/api/native/gym_burst":
                self.send_json(
                    self.authority.native_gym_burst(
                        str(payload.get("env", "CartPole-v1")),
                        int(payload.get("episodes", 1)),
                        bool(payload.get("learn", True)),
                    )
                )
            elif parsed.path == "/api/cheat":
                self.send_json(
                    self.authority.cheat(
                        float(payload.get("lr_boost", 5.0)),
                        int(payload.get("multiplier", 5)),
                        int(payload.get("steps", 256)),
                    )
                )
            else:
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_json({"error": str(exc), "trace": traceback.format_exc()}, 500)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local cocoon trainer app")
    parser.add_argument("--cocoon", default="cocoon_cognition_agency.py")
    parser.add_argument("--output-dir", default="mira_kite_authority_runtime")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    authority = AuthorityState(Path(args.cocoon).resolve(), Path(args.output_dir))
    Handler.authority = authority
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"[MIRA-KITE] Authority app running at {url}", flush=True)
    print("[MIRA-KITE] Press Ctrl+C to stop. Use Export Checkpoint before closing if you trained.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[MIRA-KITE] stopping, exporting final checkpoint...", flush=True)
        authority.export("mira_kite_authority_final.py")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
