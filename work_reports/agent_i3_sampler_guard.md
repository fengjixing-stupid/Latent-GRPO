# Agent i3 report: sampler and target-runtime guards

## Scope and result

Implemented the bounded Task 4c patch without installing dependencies or importing/running CUDA, Ray, SGLang, FlashAttention, or FlashInfer. Status is `implemented`, `static_check_passed`, `synthetic_test_passed`, and `target_machine_test_deferred`.

## RED evidence

Command:

```bash
python3 -m unittest tests.unit.test_sampler_guards -v
```

Initial result: 6 tests ran; 3 failures and 2 errors. The failures proved that the sampler still contained hard-coded 524, the rollout still supplied a 524 fallback, the SGLang CLI did not map to `latent_end_token_id`, the shared server-args dictionary omitted the field, and no dependency-free runtime guard existed. A second RED cycle showed the eval raw-fact adapter and trainer call were absent.

## Implemented changes

- `sglang_rollout.py`: validates enabled latent mode has a non-bool integer ID inside `model_hf_config.vocab_size`; passes `enable_latent=False` and `latent_end_token_id=None` defaults instead of enabling latent/using 524 implicitly.
- `server_args.py`: adds canonical `--latent-end-token-id` with legacy `--latent-end-str-id` alias, both targeting `latent_end_token_id`.
- `schedule_batch.py` and `model_runner.py`: include `latent_end_token_id` in the process-local shared server argument mapping consumed by sampler.
- `sampler.py`: reads the runtime ID, fails closed when missing/invalid, removes hard-coded 524, and guards latent-only final selection so ordinary sampling does not reference an undefined latent tensor.
- `upstream_adapter.py` and `ray_trainer.py`: add a small detached per-generation eval fact interface, keep unknown correctness as null with a reason, emit before aggregation, and reuse one optional observer instance. No Stage 3 Support or Stage 4 credit logic was added.

The full auditable propagation is:

```text
runner profile
  -> Hydra actor_rollout_ref.rollout.latent_end_token_id
  -> SGLangRollout config validation
  -> AsyncEngine(ServerArgs.latent_end_token_id)
  -> ModelConfig / Scheduler / Req sampling params
  -> model_runner global_server_args_dict
  -> sampler runtime guard and configured comparison
```

## GREEN evidence

Command:

```bash
python3 -m unittest tests.unit.test_sampler_guards tests.unit.test_upstream_adapter -v
```

Result: 17 tests ran, all passed. These are CPU/standard-library synthetic and AST tests only.

## Target-machine deferred checks

- Real tokenizer special-token string and ID agreement for every profile/model.
- SGLang process startup and confirmation that worker-local shared args contain the selected ID.
- LLaMA candidate 524 and Qwen candidate 522 termination behavior in real latent rollouts.
- FlashInfer, sgl-kernel, FlashAttention, PyTorch and CUDA ABI/import/tiny-forward checks.
- Ray/FSDP observer payload transport and authoritative-writer integration.
- Observer-off first-step equivalence under identical seeds.

No GPU/CUDA/runtime validation is claimed.

## 2026-08-03 review-fix round

Independent review found three real gaps: latent Gumbel log-prob could still silently fall back to the ordinary label objective when FlashAttention cross entropy was missing; unsupported dynamic/fused paths were not rejected; and production observation selected a bounded, lossy synthetic buffer without a writer.

RED evidence:

```text
python3 -m unittest tests.unit.test_sampler_guards -v
13 tests: 9 passed, 3 errors, 1 failure
```

The errors/failure were the missing FlashAttention guard, missing latent-instrumentation config validator, missing checkpoint-wide eval ordinal state, and enabled production observation returning `BufferedObserver` instead of requiring an authoritative sink. A separate RED assertion proved the latent-specific `torch_functional` fallback was still present.

Fixes:

- Added an actor pre-forward capability gate and a defense-in-depth exception in the latent-only Top-K/Gumbel log-prob function. Ordinary non-latent log-prob fallback is unchanged.
- Added pre-worker validation for actor dynamic batch, rollout log-prob dynamic batch, and fused kernels whenever latent mode or observation is enabled.
- Made `BufferedObserver` synthetic-only. Production enablement now requires an explicitly injected durable/enabled sink and otherwise fails closed with `eval raw-fact persistence is interface-only`.
- Kept generation ordinals across validation dataloader batches and normalized only exact 0/1 accuracy values; other numeric values fail validation.

Eval raw capture remains an interface, not an end-to-end implementation. No authoritative sink is injected by the current runner and the emitted facts are not yet enriched/committed as complete `eval_question_results` rows. Its honest status is `unavailable_with_reason=authoritative_eval_sink_not_integrated` plus `target_machine_test_deferred` for real transport validation.

GREEN evidence:

```text
python3 -m unittest \
  tests.unit.test_sampler_guards \
  tests.unit.test_upstream_adapter \
  tests.unit.test_upstream_patch_contract -v

34 tests: PASS, 3 dependency-gated skips
```

The skips are NumPy/DataProto or upstream Torch numerical checks unavailable in the base Mac interpreter; they are not GPU pass claims. Stage 3 Support and Stage 4 credit were not implemented.

Final fresh verification also ran `compileall`, all Mac unit tests, author-repository `git diff --check`, and AST parsing of the four changed Python boundaries: 124 tests passed with the same 3 dependency-gated skips; compile, AST, and diff checks exited successfully.
