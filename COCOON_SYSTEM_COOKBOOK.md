# Cocoon System Cookbook

This package is centered on the Python cocoon quine files. PyTorch is the living runtime. ONNX is an export faculty, not the organism.

## Current Core

- `cocoon_cognition_agency.py`: wholesale agent/quinesmith organism file.
- `mira_kite_authority.py`: primary GUI, training, teaching, receipt, and checkpoint surface.
- `cocoon`: global Termux launcher installed at `/data/data/com.termux/files/usr/bin/cocoon`.
- `cocoon_doctor.py`: real readiness check; separates core runtime from optional faculties.
- `cocoon_ops.py`: shared inventory/dependency/persistence primitives.
- `cocoon_mcp_stdio.py`: stdio MCP bridge for controlled tool access to doctor, cocoon inventory, jobs, logs, and persistence.

## Core Loop

1. User gives language, examples, corrections, goals, and judgments through the GUI or CLI.
2. Cocoon routes signal through action heads: observe, associate, recall, reason, respond, verify.
3. Training events are written into cocoon memory and training logs.
4. Cascade records receipts for interaction, training, checkpoint, and authority events.
5. Quinesmith governs syntax and nested quine construction so generated code stays structurally valid.
6. Export checkpoint only after priorities are complete and the doctor report is acceptable.

## Bilateral Learning Contract

The system must not be one-way prompting. Every session should leave artifacts both sides can inspect.

- User to cocoon: examples, reward scores, corrections, semantic relations, faculty targets, game outcomes.
- Cocoon to user: response, selected action/faculty, confidence or score, trace, receipt CID when available, checkpoint path when saved.
- System to future agents: doctor report, training log tail, Cascade receipts, Quinesmith audit notes, README/quickstart updates.

## Cascade-Lattice Role

Installed version: `cascade-lattice 0.8.3`.

Use Cascade as the provenance lattice and debug memory:

- `cascade.store.observe(model_id, data, parent_cid=None, sync=True)` records a receipt.
- `cascade.store.query(model_id=None, since=None, limit=100, include_remote=False)` retrieves receipts.
- `cascade.store.stats()` reports store state.
- `cascade.hold.HoldPoint(...)` can freeze an inference decision with probabilities, value, observation, brain id, labels, latent features, reasoning, and world prediction.

Relevant module families discovered locally:

- `cascade.store`, `cascade.observe`, `cascade.observation`: receipt and observation entry points.
- `cascade.hold.primitives`, `cascade.hold.session`: HOLD freeze-frame and session primitives.
- `cascade.core.*`, `cascade.system.*`: event, graph, provenance, repo/folder/system analyzers.
- `cascade.diagnostics.*`, `cascade.forensics.*`: execution/code diagnostics and artifact analysis.
- `cascade.patches.*`: integration hooks for OpenAI, Hugging Face, LiteLLM, Ollama, Anthropic.
- `cascade.viz.tape`, `cascade.tui`: visualization and terminal review surfaces.

## Quinesmith Role

Installed version: `quinesmith 0.1.1`.

Use Quinesmith as the syntax discipline for nested quines:

- `quinesmith.braces_required(level)` returns brace expansion count, where level 3 returns 8.
- `quinesmith.template_write_for(transform)` returns canonical template forms.
- `quinesmith.verify.prepare_commands(path)` prepares verification commands; it does not silently execute them.

Module families discovered locally:

- `quinesmith.levels`: nesting level and brace-count rules.
- `quinesmith.transforms`: canonical template transform strings.
- `quinesmith.verify`: verification command preparation.
- `quinesmith.audit`, `quinesmith.failure_modes`: audit/failure vocabulary.

## Faculty Readiness

Core runtime is ready when `cocoon doctor` reports:

- `core_agent`: numpy + torch.
- `provenance`: cascade + quinesmith.
- `native_serve`: flask.
- `authority_app`: numpy + torch.
- `link_mode`: websockets.
- `sphere_headless`: numpy + torch.

Optional faculties are valuable but must not block the living cocoon:

- `model_export`: onnx + onnxruntime.
- `gym_games` and `sphere_visual`: pygame.
- `drone_plots`: matplotlib.
- `drone_visual`: PyFlyt.
- `tmrl`: tmrl.

## Operating Rules

- Do not create placeholder facilities.
- Do not split the GUI into redundant control planes.
- Prefer extending `mira_kite_authority.py` for interactive features.
- Use `cocoon_ops.py` for shared readiness and inventory logic.
- Before checkpoint/push, finish active priorities, run `cocoon doctor`, run syntax checks, and document any optional faculties that remain blocked by platform constraints.
