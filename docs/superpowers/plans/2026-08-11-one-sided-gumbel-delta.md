# One-sided Gumbel Delta Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the paper's one-sided Gumbel positive offset as a reproducible `rollout.one_sided_gumbel_delta` control and apply it exactly in the rollout sampler.

**Architecture:** Add one typed scalar to the outer runner and pass it through the existing Hydra -> verl rollout -> SGLang request -> SGLang batch pipeline. Keep the sampler math isolated in a small pure tensor helper so CPU tests can verify the one-sided and two-sided behavior without importing the CUDA runtime.

**Tech Stack:** Python 3.11, dataclasses, PyYAML/Hydra overrides, PyTorch tensors, `unittest`, AST-based dependency isolation.

## Global Constraints

- The user-facing field is exactly `rollout.one_sided_gumbel_delta`.
- The baseline/default value is exactly `0.0`.
- Accepted values are finite floats greater than or equal to zero.
- The offset affects only requests with `use_one_sided_gumbel_noise=true`.
- Existing `noise_scale` ordering is unchanged; scaling occurs after the offset.
- Existing uncommitted user changes in `latent_grpo_runner/config.py` and `tests/unit/test_config.py` must be preserved.

---

## File Structure

- `latent_grpo_runner/config.py`: typed profile field, validation, hash inclusion, and Hydra mapping.
- `configs/*.yaml`, `configs/author/*.yaml`: explicit baseline value in every repository profile.
- `Latent-GRPO/verl-0.4.x/verl/trainer/config/ppo_trainer.yaml`: upstream default.
- `Latent-GRPO/verl-0.4.x/verl/workers/rollout/sglang_rollout/sglang_rollout.py`: request construction.
- `Latent-GRPO/sglang_latent_reasoning_pkg/python/sglang/srt/sampling/sampling_params.py`: request-level scalar and validation.
- `Latent-GRPO/sglang_latent_reasoning_pkg/python/sglang/srt/sampling/sampling_batch_info.py`: per-request tensor, filtering, and merging.
- `Latent-GRPO/sglang_latent_reasoning_pkg/python/sglang/srt/layers/sampler.py`: one-sided transform helper and call site.
- `tests/unit/test_config.py`: outer interface and validation tests.
- `tests/unit/test_t4_runtime_semantics.py`: mathematical behavior test.
- `tests/unit/test_sampler_guards.py`: end-to-end source-contract test for the upstream parameter path.

### Task 1: Typed runner interface

**Files:**
- Modify: `tests/unit/test_config.py`
- Modify: `latent_grpo_runner/config.py`
- Modify: `configs/smoke.yaml`
- Modify: `configs/3gpu-low.yaml`
- Modify: `configs/3gpu-high-smoke.yaml`
- Modify: all remaining `configs/*.yaml` and `configs/author/*.yaml` rollout sections

**Interfaces:**
- Consumes: profile YAML `rollout` mapping.
- Produces: `RolloutConfig.one_sided_gumbel_delta: float` and Hydra override `actor_rollout_ref.rollout.one_sided_gumbel_delta=<value>`.

- [ ] **Step 1: Write the failing runner tests**

Add tests that load `configs/smoke.yaml`, assert the typed value and Hydra override, mutate the YAML to `-0.001` and `.nan`, and assert `ConfigError` with `one_sided_gumbel_delta must be finite and non-negative`. Also load every YAML below `configs/` and assert the field is present.

```python
def test_one_sided_gumbel_delta_is_typed_hash_bound_and_validated(self) -> None:
    contents = (ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8")
    config = load_config(ROOT / "configs" / "smoke.yaml", workspace_root=ROOT)
    self.assertEqual(config.rollout.one_sided_gumbel_delta, 0.0)
    self.assertIn(
        "actor_rollout_ref.rollout.one_sided_gumbel_delta=0.0",
        config.author_hydra_overrides(),
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "delta.yaml"
        for invalid in ("-0.001", ".nan"):
            path.write_text(
                contents.replace("one_sided_gumbel_delta: 0.0", f"one_sided_gumbel_delta: {invalid}"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "must be finite and non-negative"):
                load_config(path, workspace_root=ROOT)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m unittest tests.unit.test_config.ConfigTests.test_one_sided_gumbel_delta_is_typed_hash_bound_and_validated -v`

Expected: FAIL because `RolloutConfig` has no `one_sided_gumbel_delta` field or the YAML key is rejected.

- [ ] **Step 3: Implement the minimal runner field**

Add the allowed key, dataclass field, YAML parser, Hydra mapping, and semantic check:

```python
if not math.isfinite(config.rollout.one_sided_gumbel_delta) or config.rollout.one_sided_gumbel_delta < 0:
    raise ConfigError("one_sided_gumbel_delta must be finite and non-negative")
```

Add `one_sided_gumbel_delta: 0.0` beside `use_one_sided_gumbel_noise` in every profile.

- [ ] **Step 4: Run runner tests and verify GREEN**

Run: `python -m unittest tests.unit.test_config -v`

Expected: PASS with zero failures.

### Task 2: Upstream request and batch propagation

**Files:**
- Modify: `tests/unit/test_sampler_guards.py`
- Modify: `Latent-GRPO/verl-0.4.x/verl/trainer/config/ppo_trainer.yaml`
- Modify: `Latent-GRPO/verl-0.4.x/verl/workers/rollout/sglang_rollout/sglang_rollout.py`
- Modify: `Latent-GRPO/sglang_latent_reasoning_pkg/python/sglang/srt/sampling/sampling_params.py`
- Modify: `Latent-GRPO/sglang_latent_reasoning_pkg/python/sglang/srt/sampling/sampling_batch_info.py`

**Interfaces:**
- Consumes: Hydra key `actor_rollout_ref.rollout.one_sided_gumbel_delta`.
- Produces: `SamplingParams.one_sided_gumbel_delta: float` and `SamplingBatchInfo.one_sided_gumbel_deltas: Tensor[batch, 1]`.

- [ ] **Step 1: Write the failing propagation contract test**

Add one source-contract test asserting the exact field name appears in the upstream default, rollout request kwargs, sampling parameter constructor/validation, and batch tensor/filter/merge lists.

```python
def test_one_sided_gumbel_delta_reaches_sampler_batch(self) -> None:
    rollout = _source("verl-0.4.x/verl/workers/rollout/sglang_rollout/sglang_rollout.py")
    params = _source("sglang_latent_reasoning_pkg/python/sglang/srt/sampling/sampling_params.py")
    batch = _source("sglang_latent_reasoning_pkg/python/sglang/srt/sampling/sampling_batch_info.py")
    self.assertIn('one_sided_gumbel_delta=self.config.get("one_sided_gumbel_delta", 0.0)', rollout)
    self.assertIn("self.one_sided_gumbel_delta = one_sided_gumbel_delta", params)
    self.assertIn("one_sided_gumbel_deltas=one_sided_gumbel_deltas", batch)
    self.assertIn('filter_list.append("one_sided_gumbel_deltas")', batch)
    self.assertIn('merge_list.append("one_sided_gumbel_deltas")', batch)
```

- [ ] **Step 2: Run the propagation test and verify RED**

Run: `python -m unittest tests.unit.test_sampler_guards.SamplerGuardTests.test_one_sided_gumbel_delta_reaches_sampler_batch -v`

Expected: FAIL because no upstream layer knows the new field.

- [ ] **Step 3: Implement request and batch propagation**

Add a `0.0` upstream default, pass it in the rollout kwargs, store it in `SamplingParams`, reject negative/non-finite direct requests in `verify()`, and create a float `[batch, 1]` tensor in `SamplingBatchInfo.from_schedule_batch`. Include that tensor in both latent filter and merge lists.

- [ ] **Step 4: Run the propagation tests and verify GREEN**

Run: `python -m unittest tests.unit.test_sampler_guards -v`

Expected: PASS with zero failures.

### Task 3: Sampler mathematics

**Files:**
- Modify: `tests/unit/test_t4_runtime_semantics.py`
- Modify: `Latent-GRPO/sglang_latent_reasoning_pkg/python/sglang/srt/layers/sampler.py`

**Interfaces:**
- Consumes: clipped Gumbel tensor, one-sided row mask, and per-request delta tensor.
- Produces: `_apply_one_sided_gumbel_delta(gumbels, use_one_sided_gumbel_noise, one_sided_gumbel_deltas)` tensor with two-sided rows unchanged.

- [ ] **Step 1: Write the failing mathematical behavior test**

```python
def test_one_sided_delta_shifts_only_enabled_rows_before_noise_scaling(self):
    torch = self.torch
    fn = _extract_function(
        SGLANG_SAMPLER,
        "_apply_one_sided_gumbel_delta",
        {"torch": torch},
    )
    clipped = torch.tensor([[-1.5, 0.0, 3.0], [-1.5, 0.0, 3.0]])
    enabled = torch.tensor([[True], [False]])
    deltas = torch.tensor([[0.001], [0.25]])
    actual = fn(clipped, enabled, deltas)
    expected = torch.tensor([[0.001, 1.501, 4.501], [-1.5, 0.0, 3.0]])
    self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=0))
```

- [ ] **Step 2: Run the behavior test and verify RED**

Run: `python -m unittest tests.unit.test_t4_runtime_semantics.T4RuntimeSemanticTests.test_one_sided_delta_shifts_only_enabled_rows_before_noise_scaling -v`

Expected: FAIL because `_apply_one_sided_gumbel_delta` does not exist.

- [ ] **Step 3: Implement the pure transform and integrate it**

```python
def _apply_one_sided_gumbel_delta(gumbels, use_one_sided_gumbel_noise, one_sided_gumbel_deltas):
    one_sided_gumbels = gumbels + 1.5 + one_sided_gumbel_deltas
    return torch.where(use_one_sided_gumbel_noise, one_sided_gumbels, gumbels)
```

Call the helper after clipping and before the existing `noise_scales` multiplication.

- [ ] **Step 4: Run focused and regression verification**

Run:

```bash
python -m unittest tests.unit.test_t4_runtime_semantics tests.unit.test_sampler_guards tests.unit.test_config -v
python -m compileall -q latent_grpo_runner Latent-GRPO/verl-0.4.x/verl Latent-GRPO/sglang_latent_reasoning_pkg/python/sglang/srt tests
python train_latent_grpo.py --config configs/smoke.yaml --dry-run --validate-config
```

Expected: all tests pass, compileall exits 0, and dry-run output contains `actor_rollout_ref.rollout.one_sided_gumbel_delta=0.0`.

- [ ] **Step 5: Review the final diff without committing unrelated work**

Run: `git diff --check` and inspect `git diff` for every listed file. Because `latent_grpo_runner/config.py` and `tests/unit/test_config.py` already contain user changes, do not stage or commit implementation files unless the new hunks can be isolated without staging any pre-existing change.
