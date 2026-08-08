# Task 1 scoped re-review — Fix round 1

## Scope

This is a regression-only re-review of the four prior Important findings and their directly related changes. It does not re-review Stage 1/2/storage work and does not claim target-GPU runtime validation.

## Verification run on the Mac development machine

    python3 -m unittest tests.unit.test_config tests.unit.test_launcher tests.unit.test_environment
    # Ran 19 tests: OK
    python3 -m compileall -q train_latent_grpo.py latent_grpo_runner scripts tests
    # exit 0
    python3 train_latent_grpo.py --config configs/smoke.yaml --dry-run --validate-config
    # exit 0
    python3 scripts/check_environment.py --mode target --require-gpus 3 --min-vram-gb 40 --output /private/tmp/task1-fix1-target.json
    # exit 1 on this GPU-less Mac; emitted standard envelope
    python3 scripts/probe_ray_distributed.py --num-gpus 3 --output /private/tmp/task1-fix1-ray.json
    # exit 1 because Ray is absent; emitted standard envelope

## Verdict

- **Specification verdict: PASS for this scoped Fix round 1.** All four previously Important acceptance gaps are addressed in static/synthetic Mac checks. Actual CUDA/Ray/SGLang/FSDP execution remains correctly target_machine_test_deferred.
- **Quality verdict: PASS for the reviewed changes.** The new interfaces remain lazy-import/Mac-safe, have focused unit coverage, and preserve the default ray_direct path.

## Prior Important findings

1. **Ray worker placement, binding, driver ownership, and exception propagation — ADDRESSED (synthetic/static; target runtime deferred).**
   - Evidence: scripts/probe_ray_distributed.py lines 19-60 creates three Ray tasks with num_gpus=1, records each task's Ray GPU ID and CUDA_VISIBLE_DEVICES, requires unique single-GPU IDs and no driver Ray GPU assignment (lines 32-39), and verifies an intentional worker error reaches the driver (lines 41-59). The target main gate fails on either failed binding or missing propagation (lines 80-91).
   - Regression test: tests/unit/test_launcher.py lines 74-121 uses FakeRay to execute the task submission, unique binding validation, driver check, and propagated failure path; all 19 unit tests passed.

2. **Launch-time latent-end token ID/string/vocabulary validation — ADDRESSED (synthetic/static; target tokenizer resolution deferred).**
   - Evidence: latent_grpo_runner/config.py lines 529-552 fails closed unless both tokenizer and model config expose integer vocabulary sizes, the ID is in both ranges, and convert_ids_to_tokens(ID) exactly equals the configured token. It returns the required ID/token/source/status metadata. train_latent_grpo.py lines 76-95 performs this check only after the target hardware gate and imports Transformers only in the non-dry-run target path.
   - Regression test: tests/unit/test_config.py lines 91-118 covers success, token mismatch, and out-of-range rejection. The existing dry-run import guard remains green.

3. **Machine-readable target-probe report envelope — ADDRESSED.**
   - Evidence: latent_grpo_runner/environment.py lines 135-160 defines the required command, timestamps, exit code, status, environment summary, stdout/stderr paths, artifact list, and failure-reason envelope. scripts/check_environment.py lines 37-51 writes it for target mode; scripts/probe_ray_distributed.py lines 99-114 writes it for the Ray probe.
   - Regression evidence: both Mac-generated target JSON files contain exactly the ten required envelope keys; tests/unit/test_environment.py lines 82-108 asserts the contract.

4. **Hydra mapping checked only against self-generated strings — ADDRESSED (static schema contract).**
   - Evidence: tests/unit/test_config.py lines 64-78 loads the vendored ppo_trainer.yaml and walks every dotted key emitted by author_hydra_overrides(), failing when any key is absent. This validates the actual upstream config hierarchy rather than merely checking selected output strings.
   - Related regression: config.py now validates schema_version and scalar field types before command construction (lines 343-433), with coverage at tests/unit/test_config.py lines 80-89.

## Deferred target-machine checks

The following are not represented as passed: real Ray resource allocation and CUDA_VISIBLE_DEVICES behavior, Transformers tokenizer download, CUDA/NCCL/FSDP/SGLang startup, and three-GPU training. They require the specified Linux target machine.
