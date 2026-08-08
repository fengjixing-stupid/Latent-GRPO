# Agent i1 report: identity and OCP hook

## Scope

Implemented only the post-repeat stable identity adapter and optional OCP winner facts. Optimizer, sampler, component, evaluation, and runtime/GPU behavior were not modified.

## TDD evidence

- RED: targeted tests reported `EEEF` because `latent_grpo_runner.upstream_adapter` did not exist, the repeat site had no stable-ID hook, and both OCP functions lacked opt-in observer arguments.
- GREEN: nine targeted identity/OCP adapter and AST contract tests passed.

## Changes

- Added `latent_grpo_runner/upstream_adapter.py` with `NoOpObserver`, a bounded synthetic `BufferedObserver`, deterministic ID attachment for DataProto-like objects, and `OCPSelectionFacts`/plain event conversion.
- Added the optional `ray_trainer.py` hook after `batch.repeat(...)` and before `batch.union(...)`. Missing adapter and disabled observer paths preserve the input batch.
- Updated the outer launcher to prepend the workspace root to `PYTHONPATH` only when observation is enabled, so the upstream working directory can resolve the adapter; a synthetic launch test also verifies the disabled environment is unchanged.
- Extended both latent OCP include/exclude functions with default-off `trajectory_ids` and `return_observer_data`. Opt-in returns a fourth list of detached winner facts; default callers retain the previous three-value return.
- Added identity/OCP tests in `tests/unit/test_upstream_adapter.py` and `tests/unit/test_upstream_patch_contract.py`.

## Verification limits

macOS verification uses standard-library synthetic objects plus AST/text checks. It does not import torch or Ray. Linux/CUDA/runtime integration remains pending.

## 2026-08-03 independent-review repair

The review found that list-backed identity columns violated the actual `DataProto` contract and that the real trainer never opted into OCP facts. The repair was again test-first:

- RED: four executable failures demonstrated missing NumPy fail-closed behavior, missing stable-group fact validation, absent trainer observer wiring, and absent same-winner masked-mean fact construction. Two real dependency tests were skipped with explicit target-machine reasons.
- Identity columns are now lazy-imported NumPy arrays and remain index-aligned through the same repeat/select/reorder operations used by `DataProto` when NumPy is available.
- The trainer's persistent observer now controls the OCP opt-in. Enabled calls pass stable `group_id` and `trajectory_id`, consume the fourth return, and emit detached `ocp_selection` facts. Disabled calls use the unchanged three-return core path.
- A dependency-free core helper validates one stable group per upstream UID and emits the stable group identity. Both include/exclude functions pass the selected winner's response-masked mean old log-probability into that helper.
- This Mac has no NumPy, Torch, Ray, or TensorDict. Therefore the real ndarray indexing test and real Torch numerical default/opt-in equivalence test are included but reported as `target_machine_test_deferred`, not passed.
- Final Task 4a scope verification: 20 tests ran successfully with 3 explicit dependency skips; both upstream files parsed with `ast`, local Python files compiled, and the focused upstream `git diff --check` returned clean. A full shared patch-contract run additionally exposed one still-failing dynamic-batch/Flash guard test owned by Task 4c; it is not counted as a Task 4a pass and was not modified here.
