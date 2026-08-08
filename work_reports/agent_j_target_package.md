# Agent J — target-machine execution package

## Scope

Implemented the Linux three-GPU execution package, machine-readable deferred templates, and operator/teammate validation documentation. No dependency installation, CUDA command, Ray process, SGLang server, GPU probe, or training command was executed.

## Files

- `scripts/target_machine/00_probe_environment.sh` through `11_collect_reports.sh`
- standard-library helpers: `_common.sh`, `run_reported.py`, `create_venv.py`, `install_runtime.py`, `import_check.py`, `run_with_gpu_telemetry.py`, `build_report_manifest.py`
- ten deferred templates in `artifacts/target_machine/`
- `docs/teammate_target_machine_runbook.md`
- `docs/operator_runbook.md`
- `docs/target_machine_validation_plan.md`
- `tests/unit/test_target_machine_package.py`

## Design decisions

- Every external step has a fail-closed JSON envelope and separate stdout/stderr logs; failed commands still produce reports where the system Python can run.
- PyTorch 2.6.0/torchvision 0.21.0 use the official cu124 index before runtime requirements.
- Runtime groups install in audited order; vendored verl/SGLang install editable with `--no-deps`.
- `SGL_KERNEL_VERSION` is mandatory because 0.1.0 vs 0.1.1 is unresolved; the package does not silently guess.
- FlashAttention 2.7.3 is installed only after torch with `--no-build-isolation`; FlashInfer and all CUDA extensions face a strict target import/ABI gate.
- The three-GPU path is one direct Python `ray_direct` driver and never torchrun.
- Single- and three-GPU smokes are capped at 2 steps and collect memory telemetry; high-smoke/long training is not automated.
- Resume is fail-closed until the operator supplies a real validated `global_step_<N>` checkpoint. The current outer profile does not silently override the author's `save_freq=-1`.
- All checked-in target JSON remains `target_machine_test_deferred`; no target status was claimed on Mac.

## Mac-safe evidence

Static/synthetic tests cover the exact 12 filenames/order, `bash -n`, ray-direct command templates, max-step caps, complete/deferred JSON fields, wrapper success/failure behavior, and runbook content. GPU-dependent scripts were intentionally not executed.

Executed on Mac:

```text
bash -n scripts/target_machine/[0-9][0-9]_*.sh scripts/target_machine/_common.sh  -> passed
python3 -m compileall -q scripts/target_machine tests/unit/test_target_machine_package.py -> passed
direct invocation of all five test functions, including temporary success/failure wrapper commands -> target_package_static_and_synthetic_passed
python3 -m pytest tests/unit/test_target_machine_package.py -q -> unavailable because pytest was not installed in the starting system Python
```

The pytest file itself is ready for the parent task's isolated Mac development venv. No target script was invoked during these checks; `bash -n` parses shell syntax only.
