# Agent F: config, launcher, and environment probe report

## Status

- implemented: strict profile parsing, Hydra override preview, Ray-direct launcher plan, explicit torchrun compatibility gate, Mac-safe CLI validation, and development/target probe scripts.
- static_check_passed: compileall and forbidden eager-runtime import scan completed.
- synthetic_test_passed: 15 standard-library unit tests passed.
- mac_development_check_passed: smoke dry-run and development probe completed.
- target_machine_test_deferred: CUDA, NCCL, Ray placement, FSDP, SGLang, GPU memory, and real training were not run on this Mac.

## RED

Command:

```text
python3 -m unittest tests.unit.test_config tests.unit.test_launcher tests.unit.test_environment
```

Observed result before implementation: 3 expected module import errors for the absent `latent_grpo_runner` package.

## GREEN and verification

Commands:

```text
python3 -m unittest tests.unit.test_config tests.unit.test_launcher tests.unit.test_environment
python3 -m compileall -q train_latent_grpo.py latent_grpo_runner scripts tests
python3 train_latent_grpo.py --config configs/smoke.yaml --dry-run --validate-config
python3 scripts/check_environment.py --mode development --output /private/tmp/latent-grpo-development-probe.json
```

Observed results:

- `Ran 15 tests ... OK`.
- `compileall` exited 0.
- dry-run exited 0 and emitted `launcher_plan` with `mode=ray_direct` plus `target_machine_test_deferred`.
- development probe exited 0 with `host_platform=macos_arm64`, `cuda_available=false`, and `training_runtime_validation=deferred_to_target_machine`.

## Files

- `train_latent_grpo.py`
- `latent_grpo_runner/__init__.py`
- `latent_grpo_runner/config.py`
- `latent_grpo_runner/distributed.py`
- `latent_grpo_runner/environment.py`
- `configs/smoke.yaml`
- `configs/3gpu-low.yaml`
- `configs/3gpu-high-smoke.yaml`
- `scripts/check_environment.py`
- `scripts/probe_target_machine.py`
- `scripts/probe_ray_distributed.py`
- `tests/__init__.py`
- `tests/unit/__init__.py`
- `tests/unit/test_config.py`
- `tests/unit/test_launcher.py`
- `tests/unit/test_environment.py`

## Deferred target checks

- Target `--mode target --require-gpus 3 --min-vram-gb 40` behavior is unit-tested with synthetic reports, but was not run on this Mac because it is a Linux/NVIDIA probe.
- No CUDA runtime, BF16, NCCL, Ray GPU placement, FSDP, SGLang, FlashAttention, FlashInfer, target dependency lock, model load, or training smoke was executed.
- The 3-GPU YAML profiles are arithmetic-validated candidate profiles, not validated memory-feasible or paper-reproduction configurations.

## Fix round 1

The independent Task 1 review requested four acceptance fixes. All changes remain in the assigned Task 1 file range.

### RED

Command:

```text
python3 -m unittest tests.unit.test_config tests.unit.test_launcher tests.unit.test_environment
```

Observed result before the fix: 3 expected import errors for the absent `validate_latent_end_token`, `collect_ray_placement_evidence`, and `build_report_envelope` interfaces.

### GREEN

Commands:

```text
python3 -m unittest tests.unit.test_config tests.unit.test_launcher tests.unit.test_environment
python3 -m compileall -q train_latent_grpo.py latent_grpo_runner scripts tests
python3 train_latent_grpo.py --config configs/smoke.yaml --dry-run --validate-config
python3 scripts/check_environment.py --mode development --output /private/tmp/latent-grpo-development-probe-fix1.json
```

Observed result: `Ran 19 tests ... OK`; all remaining commands exited 0.

Changes covered by the new tests:

- The Ray target probe submits three distinct one-GPU tasks, records Ray-assigned IDs and CUDA visibility, rejects non-unique bindings or driver GPU ownership, and runs an intentional one-GPU failing task to prove exception propagation. The fake-Ray test runs this behavior without Ray/CUDA.
- `validate_latent_end_token()` is a target-launch fail-closed interface. It accepts injected tokenizer/model-config objects, checks both vocabulary bounds and the decoded token string, emits validation metadata, and rejects missing/mismatched data without a fallback. The real target launcher imports Transformers only after the target hardware gate.
- Target environment and Ray probe JSON artifacts now use the documented report envelope fields.
- Hydra override testing now reads the vendored `ppo_trainer.yaml` and verifies every emitted dotted key exists there. Schema version 1 and bool-versus-int rejection are also covered.

Still deferred: actual Ray allocation, tokenizer download/validation, CUDA/NCCL/FSDP/SGLang, and all target-machine runtime behavior.
