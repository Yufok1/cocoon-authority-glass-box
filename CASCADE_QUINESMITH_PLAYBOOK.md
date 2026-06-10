# Cascade + Quinesmith Global Playbook

This file is the environment-level memory for using `cascade-lattice` and `quinesmith` with Mira-Kite.

## Package Facts
### cascade-lattice
- Import: `cascade`
- Version: `0.8.3`
- Summary: Universal AI provenance layer — cryptographic receipts for every call, HOLD inference halt protocol, and code diagnostics
- Python: `>=3.9`
- License: `MIT`
- Source path: `/data/data/com.termux/files/usr/lib/python3.13/site-packages/cascade/__init__.py`
- Python files audited: `74`

### quinesmith
- Import: `quinesmith`
- Version: `0.1.1`
- Summary: Safe primitives and protocol for AI-assisted editing of templated code-generation in quine/capsule systems.
- Python: `>=3.10`
- License: `Apache-2.0`
- Source path: `/data/data/com.termux/files/usr/lib/python3.13/site-packages/quinesmith/__init__.py`
- Python files audited: `6`

## Cascade-Lattice Recipe

- Treat Cascade as the provenance and observability lattice.
- Receipt primitive: `cascade.store.observe(model_id, data, parent_cid=None, sync=True)`.
- Query primitive: `cascade.store.query(model_id=None, since=None, limit=100, include_remote=False)`.
- CLI primitives: `cascade stats`, `cascade list`, `cascade chains`, `cascade inspect`, `cascade ingest`, `cascade analyze`, `cascade hold-status`.
- HOLD is a decision freeze-frame: action probabilities, value, observation, brain id, labels, latent/attention/features/reasoning can be serialized.
- For Mira-Kite, store each training cycle as a receipt with cycle number, sources, anchors, novel terms, losses, relation count, and checkpoint path.
- Do not make indefinite runs invisible; every cycle should write `runtime_status.json` and a Cascade receipt when available.

## Quinesmith Recipe

- Treat Quinesmith as syntax formulation and fixed-point discipline.
- `braces_required(level)` implements the doubling rule: host=1, template=2, nested=4, level n=2**n.
- `template_write_for(transform)` gives canonical Level-1 template strings for transforms such as braces, variable references, dict literals, newline escapes, and f-string nesting.
- `verify.prepare_commands(path)` only prepares commands for a human operator. The package intentionally has no execution verifier.
- For Mira-Kite, turn syntax primitives into drills: `fixed -> level -> brace -> count -> stable`, `template -> level -> reason -> brace`, `nested -> template -> fixed -> point`.
- Preserve the safety boundary: never silently compile, emit, diff, or run quine verification as an autonomous background step.

## Mira-Kite Runtime Recipe

1. Start from a trained cocoon checkpoint, currently `cocoon_cognition_agency.py`.
2. Fetch a small bounded feed slice or use local text.
3. Extract words and choose known fixed-point anchors: `know`, `remember`, `understand`, `reason`, `cause`, `effect`, `trace`, `sequence`, `signal`, `meaning`, `request`, `respond`, `provide`.
4. Add novel words as symbolic bridge nodes, not forced language-head targets.
5. Train predictable low-ID anchor words as neural targets.
6. Add atomic associations and `knowledge_web` relations for every bridge.
7. Route the 6 action heads as cognition faculties:
   - `0 observe`
   - `1 associate`
   - `2 recall`
   - `3 reason`
   - `4 respond`
   - `5 verify`
8. Record Cascade receipts.
9. Export on interval and on interrupt.

## Files

- Indefinite trainer: `mira_kite_infinite_trainer.py`
- Warm-start trainer: `semantic_cocoon_trainer.py`
- Trained cocoon: `cocoon_cognition_agency.py`
- Package: `cocoon_cognition_agency_package/`
- This memory: `.codex/global_memory/CASCADE_QUINESMITH_PLAYBOOK.md`
- JSON audit: `.codex/global_memory/cascade_quinesmith_audit.json`

## Public API Snapshot

### cascade.store
- `Any(*args, **kwargs)` (class)
- `CENTRAL_DATASET` (object)
- `CID(base: 'Union[str, Multibase]', version: 'int', codec: 'Union[str, int, Multicodec]', digest: 'Union[str, BytesLike, Tuple[Union[str, int, Multihash], Union[str, BytesLike]]]') -> '_CIDSubclass'` (class)
- `DEFAULT_LATTICE_DIR` (object)
- `Dict` (object)
- `IPFS_GATEWAYS` (object)
- `List` (object)
- `LocalStore(lattice_dir: pathlib._local.Path = None)` (class)
- `Optional` (object)
- `Path(*args, **kwargs)` (class)
- `Receipt(cid: str, model_id: str, merkle_root: str, timestamp: float, data: Dict[str, Any], parent_cid: Optional[str] = None) -> None` (class)
- `USER_DATASET` (object)
- `compute_cid(data: bytes) -> str` (function)
- `dag_cbor` (module)
- `data_to_cid(data: Dict[str, Any]) -> tuple[str, bytes]` (function)
- `dataclass(cls=None, /, *, init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False, weakref_slot=False)` (function)
- `dataset_info(dataset_id: str = None) -> Dict[str, Any]` (function)
- `discover_datasets(query_str: str = 'cascade') -> List[Dict[str, Any]]` (function)
- `discover_live(dataset_id: str = None) -> Dict[str, Any]` (function)
- `discover_models(dataset_id: str = None) -> Dict[str, int]` (function)
- `fetch_from_gateway(cid: str, timeout: float = 10.0) -> Optional[bytes]` (function)
- `fetch_receipt(cid: str, local_store: cascade.store.LocalStore = None) -> Optional[cascade.store.Receipt]` (function)
- `field(*, default=<dataclasses._MISSING_TYPE object at 0x7808b78d70>, default_factory=<dataclasses._MISSING_TYPE object at 0x7808b78d70>, init=True, repr=True, hash=None, compare=True, metadata=None, kw_only=<dataclasses._MISSING_TYPE object at 0x7808b78d70>)` (function)
- `get(cid: str) -> Optional[cascade.store.Receipt]` (function)
- `get_genesis_root() -> str` (function)
- `hashlib` (module)
- `json` (module)
- `multihash` (module)
- `observe(model_id: str, data: Dict[str, Any], parent_cid: Optional[str] = None, sync: bool = True) -> cascade.store.Receipt` (function)
- `os` (module)
- `pull_from_hf(dataset_id: str = None) -> int` (function)
- `query(model_id: Optional[str] = None, since: Optional[float] = None, limit: int = 100, include_remote: bool = False) -> List[cascade.store.Receipt]` (function)
- `sqlite3` (module)
- `stats() -> Dict[str, Any]` (function)
- `sync_all() -> dict` (function)
- `sync_observation(cid: str, filepath: pathlib._local.Path) -> dict` (function)
- `time` (module)

### cascade.hold
- `ArcadeFeedback(message: str, intensity: float, sound_cue: str, color: Tuple[int, int, int] = (255, 255, 255)) -> None` (class)
- `CausationHold(cascade_bus=None)` (class)
- `Hold()` (class)
- `HoldAwareMixin(*args, **kwargs)` (class)
- `HoldPoint(action_probs: numpy.ndarray, value: float, observation: Dict[str, Any], brain_id: str, action_labels: Optional[List[str]] = None, latent: Optional[numpy.ndarray] = None, attention: Optional[Dict[str, float]] = None, features: Optional[Dict[str, float]] = None, imagination: Optional[Dict[int, Dict]] = None, logits: Optional[numpy.ndarray] = None, reasoning: Optional[List[str]] = None, world_prediction: Optional[Dict[str, Any]] = None, id: str = <factory>, timestamp: float = <factory>, parent_merkle: Optional[str] = None, merkle_root: Optional[str] = None, state: cascade.hold.primitives.HoldState = <HoldState.PENDING: 'pending'>) -> None` (class)
- `HoldResolution(hold_point: cascade.hold.primitives.HoldPoint, action: int, was_override: bool, override_source: Optional[str] = None, hold_duration: float = 0.0, timestamp: float = <factory>, merkle_root: Optional[str] = None) -> None` (class)
- `HoldSession(session_id: str, agent_id: str, started_at: float, steps: List[cascade.hold.session.InferenceStep] = <factory>, current_index: int = 0, total_steps: int = 0, human_overrides: int = 0, correct_predictions: int = 0, combo: int = 0, max_combo: int = 0, speed_level: int = 0, speed_map: Dict[int, float] = <factory>, state: cascade.hold.session.SessionState = <SessionState.IDLE: 'idle'>) -> None` (class)
- `HoldState(*values)` (class)
- `InferenceStep(step_id: str, step_index: int, timestamp: float, input_context: Dict[str, Any], candidates: List[Dict[str, Any]], top_choice: Any, top_probability: float, hidden_state: Optional[numpy.ndarray] = None, attention_weights: Optional[Dict[str, float]] = None, chosen_value: Any = None, was_override: bool = False, override_by: str = 'model', cascade_hash: Optional[str] = None, _state_snapshot: Optional[Dict[str, Any]] = None) -> None` (class)
- `primitives` (module)
- `session` (module)

### quinesmith
#### quinesmith_levels_api
- `IntEnum(new_class_name, /, names, *, module=None, qualname=None, type=None, start=1, boundary=None)` (class)
- `Level(*values)` (class)
- `annotations` (object)
- `braces_required(level: 'Level | int') -> 'int'` (function)

#### quinesmith_transforms_api
- `Enum(new_class_name, /, names, *, module=None, qualname=None, type=None, start=1, boundary=None)` (class)
- `Transform(*values)` (class)
- `annotations` (object)
- `template_write_for(transform: 'Transform | str') -> 'str'` (function)

#### quinesmith_verify_api
- `Path(*args, **kwargs)` (class)
- `annotations` (object)
- `explain_commands() -> 'str'` (function)
- `prepare_commands(quine_file: 'str | Path') -> 'str'` (function)

