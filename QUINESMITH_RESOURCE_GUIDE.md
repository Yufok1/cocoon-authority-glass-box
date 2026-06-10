# Quinesmith Resource Guide for Cocoon Work

Audience: operators, builders, researchers, and data scientists who need to understand why nested quine editing is fragile and how to preserve self-reproducing Python cocoon files.

Installed package inspected: `quinesmith 0.1.1`.

## Why Quinesmith Matters Here

The cocoon files are not ordinary scripts. They are large Python organisms with embedded data and quine-like self-continuity. When code writes code, tiny syntax edits can destroy the fixed point. Quinesmith supplies the syntax discipline and failure vocabulary for safe nested templates.

The rule of thumb:

```text
host code writes template -> template writes generated code -> generated code must still be valid and self-identical when required
```

## Level Model

Primary module: `quinesmith.levels`.

Levels:

- `Level.HOST = 0`: compiler/source Python. `{x}` interpolates from host scope.
- `Level.TEMPLATE = 1`: inside a template body. `{{x}}` produces literal `{x}` in generated code.
- `Level.NESTED = 2`: template inside template. `{{{{x}}}}` produces literal `{x}` after another generation layer.
- Level `n`: brace count doubles as `2 ** n`.

Verified locally:

```python
from quinesmith import braces_required

assert braces_required(0) == 1
assert braces_required(1) == 2
assert braces_required(2) == 4
assert braces_required(3) == 8
```

## Canonical Transform Writes

Primary module: `quinesmith.transforms`.

`template_write_for(transform)` returns what to write inside a Level-1 template.

Known transforms:

- `open_brace`
- `close_brace`
- `variable_reference`
- `dict_literal`
- `triple_quote`
- `newline_escape`
- `backslash`
- `fstring_level_1`
- `fstring_level_2`

Use this when editing code-generation surfaces, training quine syntax drills, or explaining why a generated cocoon broke.

## Failure Mode Taxonomy

Primary module: `quinesmith.failure_modes`.

Important failure classes:

- `host_syntax_error`: host Python does not compile.
- `emission_keyerror`: host compiles, but emission crashes on a wrongly interpreted template variable.
- `over_doubled_braces`: generated file contains a literal where a value was wanted.
- `level_2_escaping_lost`: regenerated file no longer compiles after brace loss.
- `fixed_point_broken`: source and regenerated output differ.
- `hash_verify_failure`: syntax/fixed point may pass, but provenance hash does not.
- `collapsed_brace`: braces were eaten by the wrong f-string level.
- `unescaped_triple_quote`: template string closes early.
- `level_confusion`: nested f-string evaluated at the wrong level.
- `injection_mistake`: literal braces were produced where host interpolation was intended.

These are not severity labels. Each names a different boundary in the producer, contract, transport, renderer, or verification chain. Classify first, patch second.

## Verification Boundary

Primary module: `quinesmith.verify`.

Quinesmith prepares verification commands for a human operator. It intentionally does not run them.

The five checks:

1. Compile source.
2. Verify quine/self-hash.
3. Emit regenerated source.
4. Compile regenerated source.
5. Diff source vs regenerated source.

Example:

```python
from quinesmith.verify import prepare_commands
print(prepare_commands("cocoon_cognition_agency.py"))
```

This returns commands. The operator decides when to execute them.

## Cocoon Integration Rules

- Treat the Python cocoon as the living artifact, not an ONNX export.
- Use Quinesmith before editing embedded templates, self-export routines, or generated checkpoint writers.
- Preserve brace levels explicitly in comments or training examples.
- Keep verification human-visible. Do not silently run destructive or self-modifying quine steps.
- Convert common failures into training tasks so cocoons learn syntax stability as a faculty.

## Training Drill Ideas

For user-to-cocoon teaching:

- Ask the cocoon to classify a quine failure from an error message.
- Ask it to compute braces for a level.
- Ask it to choose whether a string should interpolate now or later.
- Reward correct distinction between literal and injected values.

For cocoon-to-user teaching:

- Cocoon explains the failure class.
- Cocoon provides the smallest safe edit.
- Cocoon prepares verification commands.
- User runs the commands and reports the result.
- Cascade records the whole exchange.

## Data Science View

Quine training can be evaluated as structured classification:

```json
{
  "task": "classify_quine_failure",
  "level": 2,
  "symptom": "regen does not compile",
  "expected_failure_mode": "level_2_escaping_lost",
  "model_answer": "level_2_escaping_lost",
  "reward": 1.0
}
```

Useful metrics:

- failure-mode accuracy
- brace-count accuracy by level
- fixed-point verification pass rate
- regression frequency after edits
- teacher override rate

Connect these metrics to Cascade receipts so each improvement is inspectable.
