# Task 4a round-2 independent review: stable identity and OCP hook

## Verdict

**FAIL**

The repair resolves the ndarray, stable-identity, real `compute_advantage`, default-return, and winner/old-log-probability defects from the first review. It does **not** satisfy that review's fourth re-review gate: the official `verl.trainer.main_ppo` construction path never creates or injects the authoritative durable observer sink. Consequently the standard launcher cannot run the enabled OCP emission path: with `LATENT_GRPO_OBSERVER_ENABLED=1`, trainer construction receives `observer_sink=None` and fails closed before `fit()`.

This is an honest, documented unavailable state rather than silent data loss, but it is not an implemented persistent metrics delivery path.

## Re-review matrix

| Gate | Result | Evidence |
|---|---|---|
| DataProto-compatible identity columns | PASS (static/synthetic evidence) | `attach_stable_ids_to_batch()` creates one-dimensional `np.asarray(..., dtype=object)` and `np.asarray(..., dtype=np.int64)` columns at `latent_grpo_runner/upstream_adapter.py:208-216`. This matches `DataProto.check_consistency()` and its NumPy select/concat/reorder operations. The executable ndarray survival test is present but skipped because this Mac interpreter has no NumPy. |
| Stable group and trajectory identity | PASS | IDs derive from `(global_step, data_source:index)` and per-group post-repeat ordinals. The core helper validates that every row in one upstream random `uid` maps to exactly one stable group and emits the stable group plus the selected row's trajectory ID (`core_algos.py:30-57`). |
| Ordering through repeat/filter/balance | PASS | The hook remains immediately after `batch.repeat(...)` and before union, balance, dynamic filtering, accumulation, and reorder (`ray_trainer.py:1133-1153`). The concurrent Task 4c edits did not move or bypass it; the source-order contract test passes. |
| Real `compute_advantage` opt-in and emit | PASS inside an injected trainer | The real GRPO branch reads the trainer observer, passes stable ID arrays, requests the four-value return, and emits every detached fact (`ray_trainer.py:285-310`). `fit()` passes the same trainer-owned observer (`ray_trainer.py:1363-1376`). |
| Persistent authoritative downstream sink | **FAIL** | `RayPPOTrainer.__init__` requires an injected sink when the environment enables observation (`ray_trainer.py:449-451`; `upstream_adapter.py:140-162`), but the official constructor call in `verl/trainer/main_ppo.py:138-152` supplies no `observer_sink`. The outer launcher only adjusts `PYTHONPATH`; it neither constructs nor transports the metrics writer. The repository itself records `unavailable_with_reason=authoritative_eval_sink_not_integrated` in `docs/upstream_changes.md:7`. |
| Disabled three-value compatibility | PASS (static/synthetic evidence) | Both OCP functions retain `return_observer_data=False` and return exactly three values by default; the trainer uses that path when disabled. The real Torch numerical equivalence test is present but explicitly skipped on this host because the upstream CPU dependency set is unavailable. |
| Winner identity and response-masked mean old log-probability | PASS (static/synthetic evidence) | Both variants select the winner from positive-advantage candidates, compute `(old_log_probs * response_mask).sum / response_mask.sum`, and pass the same `winner_idx` to the fact helper. The dependency-free helper test verifies stable group/trajectory identity. The real Torch test covering both include/exclude variants is deferred by its explicit skip. |

## Blocking finding

### P0 — The enabled observer path is unreachable from the official training entry

`load_observer_from_env()` deliberately rejects enabled observation without a durable, enabled sink. That fail-closed contract is correct. However, `main_ppo.py` constructs `RayPPOTrainer` without the newly added `observer_sink` argument, and no runner-side code constructs an adapter from the append-only metrics writer and passes it into that constructor.

Observed standard-path behavior is therefore:

```text
train_latent_grpo.py
  -> python -m verl.trainer.main_ppo
  -> RayPPOTrainer(..., observer_sink omitted)
  -> load_observer_from_env(sink=None)
  -> RuntimeError when LATENT_GRPO_OBSERVER_ENABLED=1
```

The in-trainer OCP call and `observer.emit("ocp_selection", fact)` are correct once a sink is manually injected, but the project has no official route that performs that injection. The first review required detached OCP events to be delivered to the authoritative downstream metrics interface; an injectable but unbound constructor parameter does not meet that gate.

Required closure: create the driver-owned durable sink/writer adapter in the official driver path, inject it into `RayPPOTrainer`, and add a Mac-safe integration test that invokes the same construction route (with runtime dependencies mocked) and proves one OCP fact reaches the authoritative sink. Keep the current fail-closed behavior for an invalid/missing sink.

## Checks executed

```text
python3 -m unittest -v tests.unit.test_upstream_adapter tests.unit.test_upstream_patch_contract
21 tests run: 18 passed, 3 explicitly skipped

python3 -m compileall -q \
  latent_grpo_runner/upstream_adapter.py \
  Latent-GRPO/verl-0.4.x/verl/trainer/ppo/core_algos.py \
  Latent-GRPO/verl-0.4.x/verl/trainer/ppo/ray_trainer.py \
  tests/unit/test_upstream_adapter.py \
  tests/unit/test_upstream_patch_contract.py
passed

git -C Latent-GRPO diff --check
passed
```

The three skips were explicit and appropriate for this host:

- enabled ndarray attachment behavior because NumPy is unavailable;
- NumPy identity survival through repeat/select/reorder;
- real upstream Torch numerical default/opt-in equivalence because the full upstream CPU import set is unavailable.

The ndarray and formula chains remain statically demonstrable, so these skips do not create a second blocking finding. They do not constitute target-machine or GPU validation.

## Status boundaries

- Identity/OCP patch internals: `implemented`, `static_check_passed`, `synthetic_test_passed`.
- Real NumPy/Torch upstream execution on this Mac: `target_machine_test_deferred` for the skipped tests.
- Authoritative persistent OCP storage path: `unavailable_with_reason=authoritative_observer_sink_not_integrated`.
- CUDA, Ray GPU, FSDP, SGLang, and three-GPU behavior: `target_machine_test_deferred`.

