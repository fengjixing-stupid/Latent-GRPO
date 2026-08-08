# Agent G — Stage 1/2 Metrics Core

## Scope

Implemented a pure-Python, dependency-free Stage 1/2 metrics contract in
`latent_grpo_runner/metrics/`.  The surface is deliberately scalar/list based,
so it is importable on macOS without torch, NumPy, Ray, or PyArrow.

## TDD evidence

RED command (17 expected import failures before the package existed):

```bash
python -m unittest tests.unit.test_events tests.unit.test_aggregators \
  tests.unit.test_masks tests.unit.test_identity tests.unit.test_stage1 \
  tests.unit.test_stage2 tests.unit.test_schemas -v
```

The later schema-coverage regression was also observed RED with a missing
`gumbel_diagnostics` table, before that manifest entry was added.

GREEN command:

```bash
python -m unittest tests.unit.test_events tests.unit.test_aggregators \
  tests.unit.test_masks tests.unit.test_identity tests.unit.test_stage1 \
  tests.unit.test_stage2 tests.unit.test_schemas -v
```

The suite contains 18 unit tests, all using hand-written literal fixtures.

## Covered contract fields and behavior

- Immutable `StepContext`, with ordinary training events rejecting checkpoint-only fields.
- Mergeable `SufficientStats`: `sum`, `sum_sq`, `count`, `nan_count`,
  `masked_count`, `min`, `max`, and `numerator_count`; driver aggregation is
  sum-based, never a mean of worker means.
- Null/unavailable representation for empty effective domains, including a
  stable reason and count preservation.
- Response/attention/loss/sentinel latent masks; zero-advantage numerator and
  denominator share exactly the eligible latent domain.
- Deterministic `group_id` plus post-repeat/pre-reorder `trajectory_id`; binary
  `correct`/`non_correct` classification with independently overlapping
  overlong status.
- All ten Stage 1 core metrics, their availability/count fields, definitions,
  and exact final-training trajectory-length integer sum for
  `train/generated_token_count`.
- Stage 2 noisy-mixture effective-K/top-1, zero-advantage support interface,
  reward/advantage standard deviations, group raw counts, and explicitly
  gated Gumbel diagnostics with separate raw and post-transform denominators.
- Scalar-only Stage 2 surrogate/FlipGrad reduction interface.
- Schema manifest fields for Stage 1/2 metrics, counts, availability families,
  group facts, and Gumbel diagnostic rows.  It rejects full tensors/gradient
  fields and exposes Stage 3 as `deferred` and Stage 4 as `disabled`.

## Deferred intentionally

- No Parquet/JSON writer, atomic commit, resume, or upstream training patch.
- No Stage 3 top-K alignment or Stage 4 forward/backward/probe implementation.
- No runtime tensor shape/device/EOS semantics claim; those require the target
  Linux GPU runtime probe.

## Review fix round 1

Added ten literal regression tests for review findings (the Task 2 suite now
contains 28 tests).  The corresponding RED run produced six assertion failures
and five missing-API errors before implementation.  Fixes include:

- `Stage2SufficientStats`: workers return only mergeable sufficient
  statistics; the driver merges them and appends the six Stage 2 metrics,
  shared counts, and availability to authoritative `train_step_metrics`.
- Context-aware `build_train_group_metrics` and in-memory
  `select_optimal_correct_path`, including the stable winner ID/old-log-prob
  raw facts and all required group context, classification/version, and
  availability fields.
- Driver-only whole-step timing, empty final-trajectory unavailability,
  finite-domain handling for Gumbel, zero-advantage, and mechanism counts,
  plus stable-string/hash-only prompt identities.
- Schema/record alignment: no synthetic Stage 2 table, no fabricated
  per-metric count for generated tokens/step time, and `int64` generated-token
  storage.

Verification after the fixes:

```bash
python -m unittest tests.unit.test_events tests.unit.test_aggregators \
  tests.unit.test_masks tests.unit.test_identity tests.unit.test_stage1 \
  tests.unit.test_stage2 tests.unit.test_schemas -v
# Ran 28 tests — OK
```

`python -m unittest discover -s tests/unit -v` ran 47 tests: all 28 Task 2
tests pass; the remaining 12 failures/errors are pre-existing config/launcher
tests blocked by `ModuleNotFoundError: yaml` / `PyYAML is required to parse
runner profiles`.  No dependency was installed, as required.

## Review fix round 2 — manifest/emit agreement

Two new literal regression tests first failed: a train-step row emitted
undeclared `train_core_*` fields, and a train-group row emitted contextual
fields absent from its schema.  The schema now declares every emitted Stage
1/2 field, and both row builders emit every declared field (using explicit
`null` for current runtime gaps).  This includes dynamic context and quality
fields such as resume state, worker count, record version, compute/write
times, group definitions, and OCP availability.

`python3 -m unittest discover -s tests/unit -v` completed successfully:

```text
Ran 49 tests — OK
```

No Stage 3/4 schema fields were added.  The helpers establish a pure-Python
contract only; attachment of stable IDs and the in-memory OCP result to the
real trainer repeat/reorder path remains **integration deferred to Task 4
instrumentation** and was not implemented in the author repository.
