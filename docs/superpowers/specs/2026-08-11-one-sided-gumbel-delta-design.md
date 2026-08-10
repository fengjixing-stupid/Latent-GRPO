# One-sided Gumbel Delta Interface Design

## Goal

Expose the paper's small positive offset `delta` in the existing Latent-GRPO
configuration chain so controlled runs can measure its effect without changing
the current baseline implicitly.

The paper defines the transformed perturbation as:

```text
xi_positive = clip(xi, -a, b) + a + delta
```

with `a = 1.5`, `b = 3.0`, and `delta > 0`. The vendored sampler currently
implements `clip(xi, -1.5, 3.0) + 1.5`, which is equivalent to `delta = 0`.

## Considered Approaches

1. Add a typed end-to-end field named `one_sided_gumbel_delta` from the runner
   YAML through Hydra, verl rollout, SGLang request/batch state, and the sampler.
   This is the selected approach because it is explicit, hash-bound, testable,
   and visible in saved run configuration.
2. Put the value only in `upstream_overrides`. This is smaller in the outer
   runner but weakens validation and makes the experimental control less
   discoverable.
3. Read an environment variable directly in the sampler. This avoids plumbing
   changes but produces hidden, poorly reproducible run semantics and is
   therefore rejected.

## Interface and Defaults

- Add `rollout.one_sided_gumbel_delta` to every repository-owned profile.
- Use `0.0` as the default and baseline value to preserve the current released
  implementation exactly.
- Accept finite values greater than or equal to zero.
- Reject negative and non-finite values during configuration validation.
- Apply the offset only where `use_one_sided_gumbel_noise` is true. Standard
  two-sided Gumbel sampling remains unchanged.
- Apply `noise_scale` after the one-sided transform, preserving the existing
  order. The effective minimum perturbation is therefore
  `noise_scale * one_sided_gumbel_delta`.

The primary user-facing YAML example is:

```yaml
rollout:
  use_one_sided_gumbel_noise: true
  one_sided_gumbel_delta: 0.001
```

## Data Flow

`rollout.one_sided_gumbel_delta` flows through these existing boundaries:

1. `latent_grpo_runner.config.RolloutConfig` validates and maps it to
   `actor_rollout_ref.rollout.one_sided_gumbel_delta`.
2. The vendored verl PPO rollout config supplies a backward-compatible default
   of `0.0`, and `SGLangRollout` includes it in request sampling parameters.
3. SGLang `SamplingParams` stores the scalar; `SamplingBatchInfo` materializes
   a per-request tensor and preserves it across filter and merge operations.
4. `Sampler.forward` adds the per-request delta to the clipped-and-shifted
   noise only for rows using one-sided noise.

All run hashes and configuration snapshots include the value through the
existing `RolloutConfig` serialization.

## Tests

- Runner config tests prove the value is parsed, included in the Hydra
  overrides, hash-bound, and rejects negative/non-finite input.
- SGLang parameter and batch tests prove the value survives request batching.
- A sampler-level unit test uses fixed noise to prove:
  - `delta = 0` preserves the current result;
  - positive delta raises the one-sided perturbation margin exactly;
  - two-sided rows are not shifted.
- Existing configuration and sampler tests must remain green.

## Experiment Contract

Use identical model, data, seed, rollout count, noise scale, and training budget
for each run. Compare at minimum `delta = 0`, `1e-4`, `1e-3`, and `1e-2`.
Record the configured delta beside the existing one-sided Delta/FlipGrad probe
metrics so observed optimization-margin changes can be attributed to the
offset rather than another sampling change.

This change exposes the control and verifies its mathematical effect; it does
not claim a paper-author default because the paper calls delta small and
positive but does not publish a numeric value.
