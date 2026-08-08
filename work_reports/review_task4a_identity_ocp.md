# Task 4a independent review: stable identity and OCP hook

## Verdict

**FAIL**

The patch places the identity hook at the required post-repeat/pre-reorder point and preserves the default three-value OCP return, but the enabled runtime path is not yet usable with the real upstream `DataProto`, and the real trainer never opts into or exports the new OCP facts.

## High-priority findings

### P0 — Attached identity columns violate the upstream `DataProto` contract

`attach_stable_ids_to_batch()` assigns Python lists to `batch.non_tensor_batch["group_id"]` and `batch.non_tensor_batch["trajectory_id"]` (`latent_grpo_runner/upstream_adapter.py:182-185`). Upstream requires every `non_tensor_batch` value to be an `np.ndarray` (`verl-0.4.x/verl/protocol.py:316-327`) and later indexes every value with NumPy arrays during `select_idxs()` and `_balance_batch()`/`reorder()` (`protocol.py:464-466`, `protocol.py:712-718`).

Consequences when observation is enabled:

- ordinary balance/reorder attempts `python_list[numpy_array]` and fails;
- dynamic filtering/select has the same failure;
- constructing/concatenating a `DataProto` fails its consistency checks.

The synthetic test uses list-backed fake data and therefore misses the real container invariant. The adapter must attach one-dimensional NumPy arrays with batch-aligned length and add a DataProto-compatible synthetic test (or a faithful array/indexing double).

### P0 — OCP observer data is not connected to the real trainer call path

Both real GRPO call sites in `ray_trainer.compute_advantage()` still unpack three values and pass neither `trajectory_ids` nor `return_observer_data=True` (`ray_trainer.py:285-301`). No downstream code consumes or emits `observer_data` from `core_algos.py`. Therefore, even with `LATENT_GRPO_OBSERVER_ENABLED=1`, the actual training path produces no `optimal_correct_trajectory_id` or `optimal_correct_mean_old_log_prob` fact.

This fails the required “OCP winner directly returned” runtime interface. The opt-in must be selected from the already-created observer state, must pass the post-repeat trajectory identities, and must emit/store the detached returned facts without changing the disabled return path.

### P1 — The OCP fact's `group_id` is not the stable group identity

The OCP functions currently store `"group_id": idx` (`core_algos.py:243-251`, `core_algos.py:379-387`). Here `idx` comes from the grouping input `data.non_tensor_batch["uid"]`; that UID is freshly generated with `uuid.uuid4()` each outer batch (`ray_trainer.py:1095`) and is not the stable `group_id` attached by the adapter.

If the opt-in call were wired as-is, OCP records could not join deterministically to `train_group_metrics` and would confuse upstream grouping identity with the contract's stable identity. Keep `uid` for unchanged training grouping if needed, but pass a separate stable group-ID array for emitted facts and verify every candidate in one upstream UID maps to exactly one stable group ID.

### P1 — Current tests prove source shape, not actual OCP semantics or DataProto survival

The OCP tests only assert that argument names and dictionary key strings exist. They do not verify:

- disabled opt-in gives numerically identical outputs and the exact original return arity;
- include and exclude variants select the actual positive-advantage winner;
- `mean_old_log_prob` is the response-mask mean for that same winner;
- stable group/trajectory identity survives actual select/filter/reorder;
- enabled and disabled behavior against a DataProto-compatible container.

Add CPU/synthetic behavioral tests with ties, nontrivial masks, multiple groups, reorder/filter, and a default-off equivalence assertion. CUDA is not required for these contracts.

## Checks that passed

- The hook is textually after `batch.repeat(..., interleave=True)` and before union, balance, dynamic filtering, selection, and reorder.
- Stable trajectory ordinals are assigned per stable group in post-repeat encounter order and are deterministic for identical `(global_step, prompt_identity)` input.
- Official preprocessing supplies `extra_info.index`; `data_source:index` is therefore a plausible stable prompt identity for the repository's intended datasets. Missing identity fails closed instead of silently inventing one.
- Disabled observation returns the original batch object without mutation.
- Missing external adapter is handled by the upstream import guard.
- `distributed.launch()` prepends the workspace root to `PYTHONPATH` for an explicitly enabled observer and preserves the incoming path while disabled, so the outer adapter is reachable from the upstream working directory.
- Both OCP functions retain the original three-value return by default and only allocate detached Python scalar/dictionary facts when explicitly requested. Winner selection and advantage mutation are not changed by the observer branch.
- The observer buffer is bounded and facts contain scalars/strings only; no full logits, hidden states, gradients, tensors, or computation graphs are persisted by Task 4a.

## Verification evidence

Executed on macOS with `python3` and without importing upstream Torch/Ray/SGLang runtime:

```text
python3 -m unittest -v tests.unit.test_upstream_adapter.UpstreamAdapterTests
8 tests passed

python3 -m unittest -v \
  tests.unit.test_upstream_patch_contract.UpstreamPatchContractTests.test_ocp_default_return_stays_compatible_and_observer_return_is_opt_in \
  tests.unit.test_upstream_patch_contract.UpstreamPatchContractTests.test_ocp_opt_in_facts_capture_the_selected_candidate_without_changing_formula
2 tests passed

python3 -m compileall -q \
  latent_grpo_runner/upstream_adapter.py \
  Latent-GRPO/verl-0.4.x/verl/trainer/ppo/core_algos.py \
  Latent-GRPO/verl-0.4.x/verl/trainer/ppo/ray_trainer.py
passed

git -C Latent-GRPO diff --check
passed
```

The whole `test_upstream_patch_contract` module was also run during concurrent Task 4b/4c edits and reported three failures: two belonged to incomplete concurrent optimizer/sampler guards, while the Task 4a ordering test had become stale because it requires `load_observer_from_env()` adjacent to the hook even though the trainer now loads the observer once in `__init__`. That assertion should be updated to verify semantic initialization plus reuse; it is not itself an implementation defect.

## Required re-review gate

Re-review after all of the following are present:

1. real `DataProto`-compatible NumPy identity columns;
2. real trainer opt-in wiring for both OCP variants;
3. stable group identity supplied separately from random upstream UID;
4. detached OCP events delivered to the authoritative downstream metrics interface;
5. behavioral tests covering default equivalence, winner identity, masked mean old log-probability, and identity survival through reorder/filter/select.

Linux/CUDA runtime execution remains `target_machine_test_deferred`; this review makes no GPU validation claim.
