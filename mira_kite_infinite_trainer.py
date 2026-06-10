#!/usr/bin/env python3
"""Indefinite processive trainer for the Mira-Kite cocoon.

This is a controlled long-running trainer:
- web/RSS feed -> semantic anchor extraction
- unknown terms -> known fixed-point bridges
- short reasoning ladders -> supervised replay
- action heads -> cognition faculties
- cascade-lattice receipts for each cycle
- quinesmith syntax primitives as safe fixed-point exercises
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import os
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import numpy as np

from semantic_cocoon_trainer import (
    COGNITIVE_FACULTIES,
    add_cognitive_faculty,
    add_reasoning_ladder,
    add_semantic_bridge,
    add_supervised_pair,
    clean_words,
    strengthen_association,
    add_relation,
)

try:
    from cascade.store import observe as cascade_observe
except Exception:  # pragma: no cover - optional runtime dependency
    cascade_observe = None

try:
    from quinesmith import Level, Transform, braces_required, template_write_for
    from quinesmith.verify import explain_commands as quine_explain_commands
except Exception:  # pragma: no cover - optional runtime dependency
    Level = Transform = None
    braces_required = template_write_for = quine_explain_commands = None


DEFAULT_FEEDS = [
    "https://export.arxiv.org/rss/cs.AI",
    "https://export.arxiv.org/rss/cs.LG",
    "https://news.ycombinator.com/rss",
]

CORE_ANCHORS = [
    "know", "remember", "understand", "explain", "reason", "cause", "effect",
    "infer", "trace", "sequence", "signal", "meaning", "request", "respond",
    "provide", "support", "ask", "missing", "stable", "focused", "coherent",
    "precise", "learn", "adapt", "strengthen", "current", "scan", "reply",
]

FACULTY_KEYWORDS = {
    "observe": {"current", "state", "signal", "scan", "trace", "detect", "see"},
    "associate": {"meaning", "relation", "context", "connect", "link", "map"},
    "recall": {"remember", "history", "memory", "registry", "past", "trace"},
    "reason": {"reason", "cause", "effect", "infer", "proof", "algorithm"},
    "respond": {"request", "answer", "reply", "provide", "help", "support"},
    "verify": {"check", "verify", "evidence", "test", "precise", "audit"},
}


def load_cocoon(path: str):
    spec = importlib.util.spec_from_file_location("mira_kite_cocoon", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import cocoon from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def strip_markup(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_url_text(url: str, timeout: float = 12.0) -> list[dict]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "MiraKiteTrainer/0.1 (+local termux trainer)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read(600_000)
    body = raw.decode("utf-8", errors="replace")

    items: list[dict] = []
    try:
        root = ET.fromstring(body)
        for item in root.findall(".//item")[:12]:
            title = strip_markup(item.findtext("title", ""))
            desc = strip_markup(item.findtext("description", ""))
            link = item.findtext("link", "")
            text = f"{title}. {desc}".strip()
            if text:
                items.append({"source": url, "title": title, "link": link, "text": text})
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry")[:12]:
            title = strip_markup(entry.findtext("{http://www.w3.org/2005/Atom}title", ""))
            summary = strip_markup(entry.findtext("{http://www.w3.org/2005/Atom}summary", ""))
            text = f"{title}. {summary}".strip()
            if text:
                items.append({"source": url, "title": title, "link": "", "text": text})
    except Exception:
        text = strip_markup(body)
        if text:
            items.append({"source": url, "title": url, "link": url, "text": text[:3000]})
    return items


def load_local_texts(paths: Iterable[str]) -> list[dict]:
    items: list[dict] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        items.append({"source": str(path), "title": path.name, "link": "", "text": text[:5000]})
    return items


def vocabulary_sets(agent):
    word_to_id = agent.vocabulary.get("word_to_id", {})
    max_vocab = min(brain.vocab_size for brain in agent.brains)
    predictable = {word for word, token in word_to_id.items() if isinstance(token, int) and 0 < token < max_vocab}
    known = set(word_to_id)
    return known, predictable


def choose_anchors(words: list[str], predictable: set[str], limit: int = 4) -> list[str]:
    ranked = []
    present = set(words)
    for anchor in CORE_ANCHORS:
        score = 2 if anchor in present else 1
        if anchor in predictable:
            ranked.append((score, anchor))
    for word in words:
        if word in predictable and word not in CORE_ANCHORS:
            ranked.append((1, word))
    out = []
    for _, word in sorted(ranked, reverse=True):
        if word not in out:
            out.append(word)
        if len(out) >= limit:
            break
    return out or ["know", "remember", "understand", "reason"]


def classify_faculty(words: list[str]) -> tuple[int, str]:
    bag = set(words)
    best_name = "associate"
    best_score = -1
    for name, keys in FACULTY_KEYWORDS.items():
        score = len(bag & keys)
        if score > best_score:
            best_name = name
            best_score = score
    for action_id, spec in COGNITIVE_FACULTIES.items():
        if spec["name"] == best_name:
            return action_id, best_name
    return 1, "associate"


def quinesmith_ladders() -> list[list[str]]:
    ladders = []
    if braces_required is not None:
        for level in [0, 1, 2, 3]:
            count = braces_required(level)
            ladders.append(["fixed", "level", "brace", "count", "stable"])
            ladders.append(["template", "level", "reason", "brace"])
            ladders.append(["brace", "count", "sequence", "reason"])
            if count >= 4:
                ladders.append(["nested", "template", "fixed", "point"])
    if template_write_for is not None and Transform is not None:
        for transform in list(Transform)[:6]:
            text = template_write_for(transform)
            atoms = clean_words(transform.value.replace("_", " "))
            if atoms:
                ladders.append(atoms[:3] + ["template", "stable"])
            if "brace" in transform.value:
                ladders.append(["brace", "literal", "fixed", "point"])
            if "string" in text or "quote" in transform.value:
                ladders.append(["syntax", "quote", "template", "verify"])
    return ladders


def train_item(agent, item: dict, cycle: int) -> dict:
    text = item["text"][:3000]
    words = [w for w in clean_words(text) if len(w) > 2]
    if not words:
        return {"bridges": 0, "pairs": 0, "faculty": "none"}

    known, predictable = vocabulary_sets(agent)
    anchors = choose_anchors(words, predictable)
    novel = []
    for word in words:
        if word not in novel and word not in anchors and len(word) > 4:
            novel.append(word)
        if len(novel) >= 8:
            break

    bridges = 0
    for word in novel[:6]:
        # New words may exceed the language head, so they become symbolic bridge nodes.
        bridges += add_semantic_bridge(agent, word, anchors)
        for anchor in anchors:
            strengthen_association(agent, word, anchor, 0.72, "internet_processive_learning")
            add_relation(agent, word, anchor, "internet_anchor", 0.72)

    pairs = 0
    if len(anchors) >= 2:
        pairs += add_reasoning_ladder(agent, anchors[:4], 0.95)
    pairs += add_supervised_pair(agent, " ".join(anchors[:3]), " ".join(anchors[:4]), 0.95)

    action_id, faculty = classify_faculty(words + anchors)
    spec = COGNITIVE_FACULTIES[action_id]
    add_cognitive_faculty(agent, action_id, spec["name"], spec["triggers"], spec["target"])

    agent.record_training_log(
        "internet_processive_item",
        stage="indefinite_runtime_faculty_trainer",
        input_text=text[:700],
        target=" ".join(anchors),
        output=f"faculty={faculty}; novel={','.join(novel[:6])}",
        reward=0.95,
        score={"cycle": cycle, "bridges": bridges, "pairs": pairs, "action_id": action_id},
        extra={"source": item.get("source"), "title": item.get("title"), "link": item.get("link")},
    )
    return {"bridges": bridges, "pairs": pairs, "faculty": faculty, "anchors": anchors, "novel": novel[:6]}


def observe_receipt(model_id: str, data: dict) -> str | None:
    if cascade_observe is None:
        return None
    try:
        receipt = cascade_observe(model_id, data, sync=False)
        return getattr(receipt, "cid", None)
    except Exception as exc:
        return f"cascade_error:{exc}"


def export_checkpoint(agent, output_dir: Path, cycle: int, base_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{base_name}_cycle_{cycle:06d}.py"
    agent.export_cocoon(str(path))
    latest = output_dir / f"{base_name}_latest.py"
    latest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Mira-Kite indefinite processive trainer")
    parser.add_argument("--cocoon", default="cocoon_cognition_agency.py")
    parser.add_argument("--output-dir", default="mira_kite_runtime")
    parser.add_argument("--base-name", default="mira_kite_runtime")
    parser.add_argument("--feed", action="append", default=[], help="RSS/Atom/HTML URL. Can repeat.")
    parser.add_argument("--local", action="append", default=[], help="Local text file feed. Can repeat.")
    parser.add_argument("--max-cycles", type=int, default=0, help="0 means run indefinitely")
    parser.add_argument("--sleep", type=float, default=300.0)
    parser.add_argument("--steps-per-cycle", type=int, default=8)
    parser.add_argument("--export-every", type=int, default=5)
    parser.add_argument("--items-per-cycle", type=int, default=6)
    parser.add_argument("--no-web", action="store_true")
    args = parser.parse_args()

    module = load_cocoon(os.path.abspath(args.cocoon))
    agent = module.CocoonAgent()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "runtime_config.json"
    config_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    memory_path = output_dir / "MIRA_KITE_MEMORY.md"
    memory_path.write_text(
        "Mira-Kite runtime trainer: fixed points -> bridges -> reasoning ladders -> "
        "action-head faculties; Cascade receipts; Quinesmith syntax fixed-point drills.\n",
        encoding="utf-8",
    )

    feeds = args.feed or DEFAULT_FEEDS
    cycle = 0
    last_checkpoint = None
    try:
        while args.max_cycles <= 0 or cycle < args.max_cycles:
            cycle += 1
            items: list[dict] = []
            if not args.no_web:
                for feed in feeds:
                    try:
                        items.extend(fetch_url_text(feed))
                    except Exception as exc:
                        items.append({"source": feed, "title": "fetch_error", "link": "", "text": f"fetch error {exc}"})
            items.extend(load_local_texts(args.local))
            if not items:
                items.append({"source": "offline", "title": "offline fixed points", "link": "", "text": "reason cause effect trace sequence meaning request respond provide"})

            # Quinesmith stays as safe syntax formulation, not execution.
            for ladder in quinesmith_ladders()[:8]:
                add_reasoning_ladder(agent, ladder, 0.9)
            if quine_explain_commands is not None:
                add_semantic_bridge(agent, "quine", ["fixed", "point", "template", "verify"])
                agent.record_training_log(
                    "quinesmith_boundary",
                    stage="syntax_formulation",
                    input_text="quinesmith verification commands are prepared for operator, not executed",
                    target="fixed point template verify",
                    output=quine_explain_commands(),
                    reward=0.9,
                )

            summaries = []
            for item in items[: args.items_per_cycle]:
                summaries.append(train_item(agent, item, cycle))

            losses = []
            for _ in range(max(0, args.steps_per_cycle)):
                loss = agent.train_step()
                if loss and loss > 0 and not np.isnan(loss):
                    losses.append(float(loss))

            receipt = observe_receipt(
                "mira_kite_indefinite_trainer",
                {
                    "cycle": cycle,
                    "items": len(items),
                    "summaries": summaries,
                    "losses": losses[-3:],
                    "vocab_size": len(agent.vocabulary.get("word_to_id", {})),
                    "knowledge_relations": len(agent.knowledge_web.get("relations", [])),
                    "conversation_turns": agent.conversation.turn_count,
                },
            )

            status = {
                "cycle": cycle,
                "time": time.time(),
                "receipt": receipt,
                "losses": losses,
                "summaries": summaries,
                "vocab_size": len(agent.vocabulary.get("word_to_id", {})),
                "knowledge_relations": len(agent.knowledge_web.get("relations", [])),
                "conversation_turns": agent.conversation.turn_count,
            }
            (output_dir / "runtime_status.json").write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
            print(json.dumps(status, indent=2, default=str), flush=True)

            if cycle % max(1, args.export_every) == 0:
                last_checkpoint = export_checkpoint(agent, output_dir, cycle, args.base_name)
                observe_receipt("mira_kite_checkpoint", {"cycle": cycle, "path": str(last_checkpoint)})

            if args.max_cycles > 0 and cycle >= args.max_cycles:
                break
            time.sleep(max(0.0, args.sleep))
    except KeyboardInterrupt:
        print("[INTERRUPT] exporting checkpoint before exit", flush=True)
    finally:
        last_checkpoint = export_checkpoint(agent, output_dir, cycle, args.base_name)
        print(f"[DONE] checkpoint={last_checkpoint}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
