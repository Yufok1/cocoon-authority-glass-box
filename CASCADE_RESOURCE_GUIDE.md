# Cascade-Lattice Resource Guide for Cocoon Work

Audience: operators, builders, researchers, and data scientists who need to understand how Cascade-Lattice turns agent activity into durable, inspectable provenance.

Installed package inspected: `cascade-lattice 0.8.3`.

## Why Cascade Matters Here

Cocoons should not train invisibly. Every meaningful teaching, game, correction, export, checkpoint, and tool decision should leave a trace that can be inspected later. Cascade gives us that trace as local receipts and, when configured, Hugging Face dataset sync.

The practical model:

```text
event -> receipt -> content address -> local index -> optional HF sync -> later query/debug/eval
```

For bilateral learning, a receipt is how the cocoon shows its work back to the user. The user gives signal; the cocoon returns not only text, but a durable claim about what happened.

## Core Storage Model

Primary module: `cascade.store`.

Important objects and functions:

- `Receipt(cid, model_id, merkle_root, timestamp, data, parent_cid=None)`: content-addressed record of an observation.
- `LocalStore(lattice_dir=None)`: local SQLite index plus CBOR blob store.
- `observe(model_id, data, parent_cid=None, sync=True)`: write an observation and return a receipt.
- `query(model_id=None, since=None, limit=100, include_remote=False)`: retrieve receipts.
- `stats()`: summarize local receipt store.
- `sync_all()`: upload local observations to Hugging Face datasets when configured.
- `pull_from_hf(dataset_id=None)`: pull remote observations into the local lattice.

Local layout:

- Default root: `~/.cascade/lattice`
- Full data: `cbor/<cid>.cbor`
- Query index: `index.db`

The source implements a dual-write idea:

- Central dataset: `tostido/cascade-observations`
- User dataset from `CASCADE_USER_DATASET`

This means a private/local run can stay local, while a configured research run can sync receipts to a dataset the user owns.

## Minimal Receipt Pattern

```python
from cascade.store import observe

receipt = observe(
    model_id="mira_kite_authority_teach",
    data={
        "event_type": "teacher_correction",
        "input": "user sentence or task",
        "output": "cocoon response",
        "reward": 0.82,
        "faculty": "reason",
        "checkpoint": "mira_kite_authority_runtime/checkpoint.py",
    },
    sync=False,
)

print(receipt.cid)
```

Use `sync=False` for phone-local speed and privacy. Use `sync=True` only when Hugging Face login and dataset policy are intentionally configured.

## HOLD: Human-in-the-Loop Decision Freeze

Primary modules:

- `cascade.hold.primitives`
- `cascade.hold.session`

Core objects:

- `HoldPoint`: frozen inference moment before an action is committed.
- `HoldResolution`: accepted, overridden, timed out, or cancelled result.
- `Hold`: singleton controller that creates hold points and resolves them.

`HoldPoint` captures:

- action probabilities
- value estimate
- observation
- brain id
- action labels
- latent activations
- attention and features
- imagination/predicted outcomes
- logits
- reasoning
- world prediction

For cocoons, this maps directly onto action heads:

```text
0 observe
1 associate
2 recall
3 reason
4 respond
5 verify
```

A good training UI should show the hold point, let the user accept or override, and record the resolution. This makes the relationship bilateral: the model proposes, the user teaches, the system remembers.

## Diagnostic and Research Surfaces

Discovered module families:

- `cascade.analysis.*`: metrics and tracing.
- `cascade.core.*`: event, causation graph, adapter, provenance, web3 bridge.
- `cascade.data.*`: dataset/hub/live/schema/pii/provenance helpers.
- `cascade.diagnostics.*`: static bug detection, code tracing, execution monitoring, reports.
- `cascade.forensics.*`: artifact and fingerprint analysis.
- `cascade.logging.*`: interpretive and Kleene-style log managers.
- `cascade.patches.*`: integration hooks for model/provider libraries.
- `cascade.system.*`: repo/folder extraction and system analyzers.
- `cascade.torch_hook`: PyTorch observation hooks.
- `cascade.viz.tape`: visualization tape.

For data scientists, the interesting path is:

```text
training/session events
-> receipts with normalized fields
-> local SQLite query
-> exported dataframe/tableau/dataset
-> evals: reward, override rate, faculty choice, error recurrence, checkpoint deltas
```

Suggested receipt schema for cocoon evals:

```json
{
  "event_type": "turn_exchange_trial",
  "session_id": "string",
  "cocoon_id": "string",
  "faculty": "observe|associate|recall|reason|respond|verify",
  "input": "string",
  "output": "string",
  "target": "string|null",
  "reward": 0.0,
  "score": 0.0,
  "action_probs": [0.0],
  "was_override": false,
  "teacher_note": "string",
  "checkpoint_path": "string|null"
}
```

## Cocoon Integration Rules

- Record receipts at teaching turns, game transitions, manual association edits, checkpoint exports, and authority-mode saves.
- Keep receipt payloads JSON/CBOR serializable.
- Use `parent_cid` or model-local latest chaining for session continuity.
- Do not hide indefinite runs. Long processes should write status plus receipts.
- Do not dump raw private prompts into public sync destinations unless the user explicitly chooses that policy.

## Practical Commands

```sh
python - <<'PY'
from cascade.store import stats, query
print(stats())
for r in query(limit=5):
    print(r.cid, r.model_id, r.timestamp)
PY
```

For Cocoon:

```sh
cocoon doctor
cocoon gui
```

Then use the GUI’s Black Ledger / receipt surfaces where available.
