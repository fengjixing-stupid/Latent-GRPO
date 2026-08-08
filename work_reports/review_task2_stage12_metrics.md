# Task 2 Review — Stage 1/2 metrics (round 3)

## Task2 scope verdict: PASS

The final round resolves the former P1 manifest/emit mismatch.  Within the
Task2 pure-Python metrics scope, all reviewed Stage 1/2 reductions, record
builders, availability/count behavior, and authoritative schema contracts are
now accepted.  The stable-ID / Optimal Correct Path connection to a real
trainer remains **deferred to Task4 instrumentation** and is not a Task2
blocker.

## Original findings — final status

| Finding | Status | Evidence |
|---|---|---|
| Stage 2 needed a globally mergeable authoritative record | **ADDRESSED** | `Stage2SufficientStats` carries mergeable packages only, merges fieldwise, and is emitted into `train_step_metrics`. |
| Group raw facts and Optimal Correct Path were incomplete | **ADDRESSED** | Context-aware group builder emits the required group facts, availability, versions, and OCP winner fields. |
| Gumbel NaN/Inf rates used mismatched denominators | **ADDRESSED** | Rates use the same finite input domain as their reported counts; literal non-finite regression passes. |
| Mechanism auxiliary counts could exceed effective count | **ADDRESSED** | A common finite effective-component mask gates all auxiliary counts. |
| Zero-advantage denominator included non-finites | **ADDRESSED** | Real advantage values are passed into `SufficientStats`; non-finites increment `nan_count` and leave the effective denominator. |
| Step time was reduced as a worker mean | **ADDRESSED** | The builder accepts a driver-only complete-step elapsed time, separately from worker sufficient statistics. |
| Empty generated-token domain was incorrectly available | **ADDRESSED** | It emits null with `empty_effective_mask`. |
| P1: manifest and emitted rows differed | **ADDRESSED** | `schemas.py:25-99` now declares train-core fields, full group dynamic context, resume and record-quality fields. `StepContext` supplies resume fields (`events.py:17-40`); both builders emit quality metadata (`stage1.py:51-96`, `stage2.py:136-165`). Exact set comparisons have no missing or undeclared fields for either table. |

## Manifest/emit re-verification

Live builder-to-manifest comparisons yielded:

```text
train_step_metrics:  undeclared=[]; missing=[]
train_group_metrics: undeclared=[]; missing=[]
```

The new literal tests enforce the same bidirectional invariant:

- `test_train_step_emits_no_undeclared_fields_and_includes_quality_fields`
- `test_train_group_emits_no_undeclared_fields_and_all_quality_context`

The manifest now includes `train_core_available` /
`train_core_unavailable_reason`, all ordinary training context for group rows,
`is_resume_run` / `resume_from_step`, and per-record aggregation/version/
compute/write-time fields.  Both group and step builder rows exactly equal the
respective declared field sets under normal and unavailable inputs.

## Deferred (Task4, non-blocking here)

`attach_stable_trajectory_ids()` and the in-memory OCP helper are pure-Python
interfaces and tests only.  They are not yet invoked at the actual
repeat-before-reorder rollout boundary.  Task4 instrumentation must wire them
into that runtime path and prove the same selected winner is passed forward;
this is deliberately outside the Task2 module-only scope.

## Verification

Executed from the project root with the required default interpreter:

```text
python3 --version
Python 3.11.9

python3 -m unittest -v
Ran 49 tests in 0.160s
OK
```
