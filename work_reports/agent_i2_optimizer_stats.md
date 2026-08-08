# Agent i2 work report: optimizer outcomes and component statistics

## Scope

Implemented only the Task 4b compatibility patch: real optimizer-step outcome exposure, skipped-update scheduler guard, detached component sufficient statistics, and the confirmed top-K logits/list bug. No dependency was installed and no training, CUDA, Ray, SGLang, or FSDP execution was started.

## Evidence-led bug finding

`DataParallelPPOActor.compute_log_prob()` accumulated `cur_topk_logits` into `topk_logits_lst`, but returned `torch.concat(topk_ids_lst, dim=0)` for `topk_logits`. That is a direct type/semantic mismatch in the checked-in author source. The patch changes only that concat input.

## Changed files

- `Latent-GRPO/verl-0.4.x/verl/workers/actor/dp_actor.py`
  - keeps `_optimizer_step()`'s original `grad_norm` return;
  - records `did_step` only after the optimizer call returns;
  - counts successful calls per `update_policy()` invocation;
  - exposes an opt-in, plain-dict observer payload;
  - reduces component facts immediately to detached scalars;
  - fixes `topk_logits_lst` concatenation.
- `Latent-GRPO/verl-0.4.x/verl/workers/fsdp_workers.py`
  - advances the actor scheduler only for `update_count > 0`;
  - adds observer metadata only when the opt-in payload is non-empty.
- `latent_grpo_runner/upstream_adapter.py`
  - adds a standard-library component-statistics reference reducer for Mac synthetic verification.
- `tests/unit/test_upstream_optimizer_patch.py`
  - adds AST/static and dependency-free synthetic tests.
- `docs/upstream_changes.md`
  - records compatibility, data minimization, rollback, and deferred validation.

## TDD evidence

RED command:

```text
python3 -m unittest tests.unit.test_upstream_optimizer_patch -v
```

Initial result: 5 tests ran; 4 failed because the observer methods, actual optimizer outcome, worker export, and correct logits concat were absent. The parse-only test passed.

GREEN command:

```text
python3 -m unittest tests.unit.test_upstream_optimizer_patch tests.unit.test_upstream_adapter tests.unit.test_upstream_patch_contract.UpstreamPatchContractTests.test_optimizer_reports_real_success_and_scheduler_skips_nonfinite_update tests.unit.test_upstream_patch_contract.UpstreamPatchContractTests.test_topk_logits_are_concatenated_from_logits_not_ids -v
```

Result: 17 tests passed.

Static compile command:

```text
python3 -m compileall -q latent_grpo_runner/upstream_adapter.py tests/unit/test_upstream_optimizer_patch.py Latent-GRPO/verl-0.4.x/verl/workers/actor/dp_actor.py Latent-GRPO/verl-0.4.x/verl/workers/fsdp_workers.py
```

Result: exit code 0.

## Compatibility and safety notes

- Observer default is off (`LATENT_GRPO_OBSERVER_ENABLED=0`).
- `_optimizer_step()` and `update_policy()` retain their historical return types.
- Disabled mode adds no observer field to the worker `DataProto` output.
- No optimizer `state_dict`, parameter, `.grad`, full logits, hidden state, or embedding is included in the observer payload.
- The component reducer adds no model forward/backward or random operation.
- `successful_step_count` is deliberately not reported as a global counter; the driver/checkpoint layer must derive and restore the contract-level cumulative `optimizer_step` from successful events.

## Target-machine deferred validation

- Verify all FSDP ranks agree on finite/non-finite gradient outcome before driver counting.
- Induce or safely simulate a non-finite gradient and confirm both `optimizer.step()` and scheduler advancement are skipped.
- Compare CUDA-reduced component sufficient statistics with an offline reference on the same captured bounded fixture.
- Confirm Ray `DataProto.meta_info` carries the enabled plain observer payload after sharding postprocessing.
- Verify no measurable training-semantic change with observer disabled and quantify overhead when enabled.

## 2026-08-03 review-fix round

The independent review correctly found that `DP_COMPUTE_PROTO` used `DataProto.concat()`, which retained only rank 0 metadata, and that the component hook ran before final overlong advantage handling without the response/loss mask. The first implementation was therefore not a real three-worker Stage 2 chain.

Fixes implemented after new RED tests:

- custom DP collector preserves all rank packets and worker identity;
- standard-library coordinator merge checks the complete rank set, exact optimizer-outcome consensus, and sufficient-statistic definition/schema;
- component stats merge by `sum`, `sum_sq`, integer counts, and global `min`, never worker means;
- `ray_trainer.fit()` consumes the packet list and emits one immutable post-update driver event to the same injected durable observer used by OCP/eval facts;
- final overlong advantage zeroing now precedes the forward and component capture;
- component eligibility includes the real response mask or multi-turn loss mask, hard-token sentinel, valid component IDs, shape/K agreement, and finite-value filtering;
- all profiles declare `metrics_enabled=true`; CLI and launcher provide an authoritative enable/disable mapping, including resume;
- scheduler all-skipped behavior is recorded as an observer-off algorithmic fix;
- the shared latent guard contract now checks the actual helper and also rejects `use_remove_padding=false`.

Review-round RED command:

```text
python3 -m unittest tests.unit.test_upstream_optimizer_patch tests.unit.test_launcher tests.unit.test_config -v
```

Result before fixes: 26 tests ran with 3 failures and 4 errors, covering absent rank merge/transport, wrong component timing/mask, and missing config mapping.

Review-round GREEN command:

```text
python3 -m unittest tests.unit.test_upstream_optimizer_patch tests.unit.test_launcher tests.unit.test_config tests.unit.test_upstream_patch_contract -q
```

Result: 39 tests passed, 1 dependency-gated OCP numerical test skipped.

Durable-sink status is explicitly **not implemented**: the current driver observer interface is not wired to `AppendOnlyPartWriter`, `schema_manifest()`, `build_train_step_metrics()`, or `build_stage2_metrics()`. Accordingly, a real `metrics_enabled=true` launch is blocked before target probing/model loading. No in-memory buffer or JSONL is presented as contract storage. The worker→coordinator interface is `synthetic_test_passed`; the Parquet/sidecar runtime chain is `unavailable_with_reason=durable_parquet_sink_not_wired`.
