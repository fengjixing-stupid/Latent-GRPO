# Independent review: Task 4c sampler/runtime guards

## Verdict

**FAIL.** The configurable latent-end ID is propagated through the real SGLang object/process path and the ordinary non-latent sampler branch no longer reads an undefined latent tensor. However, the implementation is not ready for target smoke because required fail-closed actor-path guards are absent, and the eval hook terminates in a bounded in-memory buffer rather than the authoritative metrics writer.

Review status: `static_check_passed` for syntax/diff hygiene, `synthetic_test_passed` only for the narrow new sampler tests, and `target_machine_test_deferred`. No CUDA, Ray, SGLang server, FlashAttention, FlashInfer, sgl-kernel, or GPU result is claimed.

## High-priority findings

### P0 — latent Gumbel training still silently falls back to the wrong objective when FlashAttention cross entropy is unavailable

`verl-0.4.x/verl/utils/torch_functional.py:32-37` catches the FlashAttention cross-entropy import failure. Its latent-specific `logprobs_from_logits_topk_gumbel()` then takes the `else` branch and returns ordinary label log-probability (`torch_functional.py:133-195`), dropping the Top-K/Gumbel/FlipGrad calculation instead of failing closed. No new guard in `dp_actor.py` prevents this path.

This is not merely a performance fallback: it changes the latent policy objective. It was already identified as a P0 in `docs/repo_audit.md`, and the existing contract test requires the explicit error text `FlashAttention cross entropy is required for latent Gumbel log-prob`. That test fails.

Required correction: before any latent actor forward/update, reject the configuration/runtime unless the FlashAttention cross-entropy implementation is importable. The check must be conditional on the latent path so unrelated non-latent upstream use remains compatible.

### P1 — unsupported dynamic-batch and fused latent paths are not rejected

`RayPPOTrainer._validate_config()` contains no observer/latent guards for:

- `actor_rollout_ref.actor.use_dynamic_bsz=true`;
- `actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true`;
- `actor_rollout_ref.actor.use_fused_kernels=true`.

The repository audit records that dynamic batching does not restore all latent Top-K/component alignment, while the new component observer only covers the ordinary actor path. Allowing those configurations can therefore produce incomplete or misaligned Stage 1/2 facts. `test_dynamic_batch_and_fused_latent_paths_fail_safe_and_flash_is_required` fails on the first missing guard.

Required correction: add explicit pre-training validation for all three unsupported combinations (using the contract-test messages), scoped to the latent instrumentation path. Add synthetic config tests for each rejection and an observer-off/non-latent compatibility case.

### P1 — eval raw facts have no end-to-end delivery or authoritative persistence

The capture point in `_validate()` is correctly after decode/reward and before `process_validation_metrics()`, and correctness is not guessed from reward: only `reward_extra_info.acc` is used, otherwise `is_correct=null` with a reason. But the real path is incomplete:

- `load_observer_from_env()` creates only `BufferedObserver(max_events=1024)`.
- `BufferedObserver.emit()` silently drops the oldest event after 1,024 entries.
- No production code drains that buffer, forwards it to the single driver writer, or appends `eval_question_results` Parquet rows.
- The emitted event lacks contract-required checkpoint/profile/seed/version/hash/length/validity/availability fields; there is no downstream enrichment stage.
- `generation_id` ordinals restart on every `emit_eval_question_facts()` call, so uniqueness is only incidental if each question appears in exactly one dataloader batch.
- `is_correct` is passed through without enforcing `bool | null`; reward implementations may return numeric accuracy values.

Thus enabling `LATENT_GRPO_OBSERVER_ENABLED=1` does not produce durable, complete `eval_question_results`, and sufficiently large validation runs lose facts in memory. The current tests only drain a synthetic buffer immediately and cannot establish the actual runtime chain.

Required correction: introduce an explicit production observer transport to the single authoritative writer (or a documented driver-owned append interface), make overflow/failure observable rather than lossy, enrich and validate the complete schema before commit, and keep checkpoint-wide generation identity stable. Until then, report eval raw-fact persistence as `target_machine_test_deferred`/`unavailable_with_reason`, not implemented end to end.

## Verified properties

| Area | Result | Evidence |
|---|---|---|
| Outer token-string validation | PASS for the main entry design | `train_latent_grpo.py` loads tokenizer/model metadata before launch; `validate_latent_end_token()` compares the configured string with `convert_ids_to_tokens(id)` and bounds-checks both vocabularies. SGLang correctly retains only runtime integer validation. |
| Hydra to engine | PASS statically | `ResolvedConfig.author_hydra_overrides()` emits `actor_rollout_ref.rollout.latent_end_token_id`; `SGLangRollout` validates it and passes the same value to `AsyncEngine`. |
| ServerArgs/CLI naming | PASS statically | Canonical `--latent-end-token-id` and legacy `--latent-end-str-id` both target `latent_end_token_id`; programmatic engine construction uses the canonical dataclass field. |
| Real SGLang process/object propagation | PASS statically | `ServerArgs -> ModelConfig/Scheduler -> Req.sampling_params`, and `ModelRunner` updates the process-local shared dict consumed by `Sampler`. `ModelRunner.get_worker_info()` also returns the actual dict to Scheduler, covering the worker/scheduler copy. |
| Configured sampler comparison | PASS | Hard-coded `524` comparison is removed; enabled latent mode requires a non-bool, non-negative integer. Outer/model validation supplies the upper-bound check before launch. |
| Non-latent undefined local | PASS | Both the final `torch.where(... latent_batch_next_token_ids ...)` and the Gumbel log-prob branch are guarded by `enable_latent`; Python short-circuiting prevents latent-only local access in the ordinary branch. |
| Eval capture time / correctness inference | PARTIAL PASS | Capture is before aggregate validation metrics and does not infer correctness from reward. Completeness/type/persistence fail as described above. |
| Observer-off training mathematics | PASS statically for Task 4c changes | Disabled observer skips eval emission; ID adapter returns the original batch. Latent sampler changes are configuration/guard changes. Target seeded first-step equivalence remains deferred. |
| Stage 3/4 scope | PASS | No Support implementation, backward probe, credit attribution, full-vocabulary retention, extra full forward, or training gradient hook was introduced by Task 4c. |

## Commands and observed results

```text
python3 -m unittest tests.unit.test_sampler_guards tests.unit.test_upstream_adapter -v
17 tests: PASS

python3 -m unittest tests.unit.test_sampler_guards tests.unit.test_upstream_patch_contract -v
16 tests: 14 PASS, 2 FAIL
```

The Task 4c-relevant failure is `test_dynamic_batch_and_fused_latent_paths_fail_safe_and_flash_is_required`: the required dynamic/fused validation and FlashAttention fail-fast text are absent. The other failure is the previously documented stale Task 4a source-window assertion: it expects `load_observer_from_env()` adjacent to the stable-ID call even though the observer is now initialized once in `__init__`; it is not counted as a Task 4c defect.

```text
python3 -m py_compile \
  latent_grpo_runner/upstream_adapter.py \
  tests/unit/test_sampler_guards.py \
  Latent-GRPO/verl-0.4.x/verl/trainer/ppo/ray_trainer.py \
  Latent-GRPO/verl-0.4.x/verl/workers/rollout/sglang_rollout/sglang_rollout.py \
  Latent-GRPO/sglang_latent_reasoning_pkg/python/sglang/srt/layers/sampler.py \
  Latent-GRPO/sglang_latent_reasoning_pkg/python/sglang/srt/managers/schedule_batch.py \
  Latent-GRPO/sglang_latent_reasoning_pkg/python/sglang/srt/model_executor/model_runner.py \
  Latent-GRPO/sglang_latent_reasoning_pkg/python/sglang/srt/server_args.py
PASS

git -C Latent-GRPO diff --check
PASS
```

## Test-quality gaps

The new sampler tests are useful dependency-free source and helper checks, but they do not prove the central failure boundaries. They should additionally cover:

1. all unsupported dynamic/fused config combinations;
2. latent FlashAttention cross-entropy absence raising before update;
3. production observer transport to writer, overflow/failure semantics, and complete eval schema;
4. checkpoint-wide generation-ID uniqueness across multiple validation batches;
5. non-bool accuracy rejection/normalization;
6. target-machine SGLang startup and worker-local `global_server_args_dict` inspection.

GPU/runtime checks remain `target_machine_test_deferred`.
