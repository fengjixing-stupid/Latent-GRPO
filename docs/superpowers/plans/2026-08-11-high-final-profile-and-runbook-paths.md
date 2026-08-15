# High Final Profile And Runbook Paths Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add author-faithful formal/validation High profiles, make 3GPU operator tooling config-driven, enforce local model directories, and document symmetric Low/High path examples.

**Architecture:** Keep one set of 3GPU wrappers and parameterize them with a validation or formal YAML path. The final validator derives durable-row identity from the loaded validation config, and the training wrapper requires a matching-family PASS report before launch. High uses its own author values and explicit 3GPU batch adaptation; Low remains backward compatible as the wrapper default.

**Tech Stack:** Python 3.9+, PyYAML, Bash, pytest, existing `latent_grpo_runner` and append-only metrics stack.

## Global Constraints

- Runtime `--model-path` must be an existing local model directory; Hugging Face IDs/pages are provenance/download sources only.
- `--train-data` and `--val-data` must be existing local parquet files.
- High formal preserves author LR, lengths, KL, offload, Top-K/Gumbel, latent-end ID 522, rollout `n=8`, and 5 epochs.
- High formal prompt/mini batch is `12/12`, preserving 32 normalized trajectories per rank: `12×8÷3 = 32×8÷8`.
- High validation prompt/mini batch is `3/3`, performs two real optimizer updates, and saves every step.
- Low remains the default wrapper family and its existing parameter semantics must not change.
- Low acceptance cannot authorize High training, and High acceptance cannot authorize Low training.
- High target status remains `TARGET_RUNTIME_EXECUTION_REQUIRED` until real Linux 3GPU execution passes.

---

### Task 1: Add Author-Faithful High Final Profiles

**Files:**
- Create: `configs/3gpu-final-high.yaml`
- Create: `configs/3gpu-final-high-validation.yaml`
- Modify: `latent_grpo_runner/config.py`
- Modify: `tests/unit/test_3gpu_final_package.py`

**Interfaces:**
- Consumes: `load_config(path, workspace_root=ROOT)` and `configs/author/latent_grpo_math_qwen.yaml`.
- Produces: supported profiles `3gpu-final-high` and `3gpu-final-high-validation`.

- [ ] **Step 1: Add failing High profile tests**

Add a test that loads both files and asserts:

```python
formal = load_config(ROOT / "configs/3gpu-final-high.yaml", workspace_root=ROOT)
validation = load_config(ROOT / "configs/3gpu-final-high-validation.yaml", workspace_root=ROOT)

assert formal.profile_kind == "formal_training"
assert formal.batch.prompt_batch == 12
assert formal.batch.mini_prompt_batch == 12
assert formal.batch_arithmetic()[:2] == (32, 32)
assert formal.batch.rollout_n == 8
assert formal.batch.actor_micro_batch_per_gpu == 1
assert formal.model.latent_end_token_id == 522
assert formal.model.use_kl_loss is True
assert formal.model.enable_gradient_checkpointing is True
assert formal.model.actor_param_offload is True
assert formal.model.actor_optimizer_offload is True
assert formal.model.ref_param_offload is False
assert formal.data.max_prompt_length == 1024
assert formal.data.max_response_length == 4096
assert formal.rollout.max_model_len == 12000
assert formal.rollout.max_num_batched_tokens == 12000
assert formal.rollout.gpu_memory_utilization == 0.8
assert "trainer.total_epochs=5" in formal.author_hydra_overrides()

assert validation.profile_kind == "final_runtime_validation"
assert validation.batch.prompt_batch == 3
assert validation.batch.mini_prompt_batch == 3
assert validation.training.max_steps == 2
assert validation.model.latent_end_token_id == 522
assert "trainer.total_training_steps=2" in validation.author_hydra_overrides()
assert "trainer.save_freq=1" in validation.author_hydra_overrides()
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m pytest \
  tests/unit/test_3gpu_final_package.py -q
```

Expected: FAIL because the High final profiles are missing or unsupported.

- [ ] **Step 3: Implement both complete YAML profiles**

Copy every author-frozen High field from `configs/author/latent_grpo_math_qwen.yaml`. Use formal `12/12`, validation `3/3`, required GPUs 3, BF16, all metric feature flags true, local-path-overridable model/data values, and the existing allowlisted upstream overrides.

- [ ] **Step 4: Register both profile names**

Add to `SUPPORTED_PROFILES`:

```python
"3gpu-final-high",
"3gpu-final-high-validation",
```

- [ ] **Step 5: Run focused tests and both dry-runs**

```bash
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m pytest \
  tests/unit/test_3gpu_final_package.py tests/unit/test_config.py -q
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 train_latent_grpo.py \
  --config configs/3gpu-final-high.yaml --dry-run --validate-config \
  --output-root /tmp/latent-grpo-high-formal-dry-run
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 train_latent_grpo.py \
  --config configs/3gpu-final-high-validation.yaml --dry-run --validate-config \
  --output-root /tmp/latent-grpo-high-validation-dry-run
```

Expected: tests PASS; both dry-runs report `mac_development_check_passed` and target runtime deferred.

### Task 2: Make Final Evidence Profile-Aware

**Files:**
- Modify: `tools/validate_3gpu_final.py`
- Modify: `tests/unit/test_3gpu_final_package.py`

**Interfaces:**
- Consumes: validation config path supplied through `args.config`.
- Produces: `acceptance.json.profile_name` and profile-filtered durable evidence for either family.

- [ ] **Step 1: Add failing pure gate identity tests**

Extend the existing evidence fixture with:

```python
evidence["profile_name"] = "3gpu-final-high-validation"
report = evaluate_final_gate(evidence)
assert report["profile_name"] == "3gpu-final-high-validation"
```

Add a source/runtime test that `_load_table` accepts `profile_name` as an argument and does not reference a module-level low-only `PROFILE` constant.

- [ ] **Step 2: Run the tests and verify RED**

```bash
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m pytest \
  tests/unit/test_3gpu_final_package.py -q
```

Expected: FAIL because the report omits profile identity and table filtering is hard-coded.

- [ ] **Step 3: Load config before durable tables and filter dynamically**

Change the helper contract to:

```python
def _load_table(root: Path, name: str, *, profile_name: str) -> list[dict[str, object]]:
    ...
    return [row for row in rows if row.get("profile_name") == profile_name]
```

In `collect_evidence`, load/override `config` before constructing `tables`, pass `config.profile_name` to every `_load_table`, and add `profile_name` to normalized evidence and report output. Include the profile in `ACCEPTANCE_SUMMARY.md`.

- [ ] **Step 4: Run focused validator tests**

```bash
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m pytest \
  tests/unit/test_3gpu_final_package.py tests/unit/test_kaggle_t4_30_runtime.py -q
```

Expected: PASS.

### Task 3: Generalize Wrappers And Enforce Local Assets

**Files:**
- Modify: `tools/run_3gpu_preflight.sh`
- Modify: `tools/run_3gpu_final_validation.sh`
- Modify: `tools/run_3gpu_training.sh`
- Modify: `tests/unit/test_3gpu_final_package.py`

**Interfaces:**
- Consumes: `--config PATH`, local `--model-path DIR`, local parquet paths, and matching acceptance JSON.
- Produces: one Low-default wrapper set capable of executing either family without low profile hard-coding.

- [ ] **Step 1: Add failing wrapper contract tests**

Assert all three scripts parse `--config`, final passes `${CONFIG}` to preflight/train/validator, and training contains no forced `--profile-name 3gpu-final-low`. Assert preflight and training reject a missing local model directory. Assert the training acceptance gate contains both exact mappings:

```text
3gpu-final-low -> 3gpu-final-validation
3gpu-final-high -> 3gpu-final-high-validation
```

- [ ] **Step 2: Run wrapper tests and verify RED**

```bash
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m pytest \
  tests/unit/test_3gpu_final_package.py::test_final_wrappers_exist_are_syntax_checked_and_fail_closed -q
```

Expected: FAIL on low-only hard-coded config/profile strings.

- [ ] **Step 3: Parameterize preflight**

Add `CONFIG="configs/3gpu-final-validation.yaml"`, parse `--config`, require `[[ -f "${CONFIG}" ]]`, require `[[ -d "${MODEL_PATH}" ]]`, pass `${CONFIG}` to dry-run, and pass it into the tokenizer-gate Python block so `load_config(sys.argv[1], ...)` validates ID 524 or 522 from the selected family.

- [ ] **Step 4: Parameterize final validation**

Add the same default/config parser and thread `${CONFIG}` through preflight, both `train_latent_grpo.py` invocations, and `tools/validate_3gpu_final.py`. Omit explicit `--profile-name`; the strict YAML declares the authoritative name.

- [ ] **Step 5: Parameterize formal training and cross-family acceptance**

Keep default `configs/3gpu-final-low.yaml`, require local model directory and local data files, omit the forced profile-name, and replace the acceptance check with logic equivalent to:

```python
mapping = {
    "3gpu-final-low": "3gpu-final-validation",
    "3gpu-final-high": "3gpu-final-high-validation",
}
formal = yaml.safe_load(Path(config_path).read_text())["profile_name"]
accepted = json.loads(Path(report_path).read_text())
ok = accepted.get("final_gate") == "PASS" and accepted.get("profile_name") == mapping.get(formal)
raise SystemExit(0 if ok else 1)
```

- [ ] **Step 6: Run shell syntax and wrapper tests**

```bash
bash -n tools/run_3gpu_preflight.sh tools/run_3gpu_final_validation.sh tools/run_3gpu_training.sh
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m pytest \
  tests/unit/test_3gpu_final_package.py -q
```

Expected: PASS.

### Task 4: Document Symmetric Local Paths And Verify Everything

**Files:**
- Modify: `docs/3GPU_RUNBOOK.md`
- Modify: `docs/3GPU_HYPERPARAMETER_DEVIATIONS.md`
- Modify: `docs/PROJECT_TECHNICAL_HANDOFF.md`
- Modify: `Latent-GRPO/cairn/LOG.md`
- Modify: `tests/unit/test_3gpu_final_package.py`

**Interfaces:**
- Consumes: both final profile families and generic wrapper CLI.
- Produces: copy/paste Low/High local-path validation and formal-training instructions.

- [ ] **Step 1: Add failing documentation tests**

Require the runbook to contain both official model source URLs, all four profile paths, `LOW_MODEL_PATH`, `HIGH_MODEL_PATH`, local train/validation parquet variables, distinct validation/training outputs, and `--config` in both family commands. Require examples to assign `MODEL_PATH` from a filesystem path such as `/data/models/...`, never from an HF ID.

- [ ] **Step 2: Run docs tests and verify RED**

```bash
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m pytest \
  tests/unit/test_3gpu_final_package.py -q
```

Expected: FAIL because the High final paths/commands are absent.

- [ ] **Step 3: Update the runbook path contract**

Add a variable-to-CLI table and two full local-path blocks. Use separate roots:

```bash
export LOW_MODEL_PATH=/data/models/LLaMA3.2-1B-Instruct-Latent-SFT-Top10
export LOW_TRAIN_DATA=/data/latent-grpo/GSM8k-Aug-oss-dup-all.parquet
export LOW_VAL_DATA=/data/latent-grpo/GSM8k-Aug-test.parquet
export LOW_VALIDATION_ROOT="$PWD/artifacts/validation/3gpu-final-low"
export LOW_TRAIN_OUTPUT="$PWD/artifacts/runs/latent-grpo-gsm8k-seed17"

export HIGH_MODEL_PATH=/data/models/Qwen2.5-Math-7B-Latent-SFT-4k-Top10
export HIGH_TRAIN_DATA=/data/latent-grpo/DAPO-Math-17k-en-train.parquet
export HIGH_VAL_DATA=/data/latent-grpo/Math-500-test.parquet
export HIGH_VALIDATION_ROOT="$PWD/artifacts/validation/3gpu-final-high"
export HIGH_TRAIN_OUTPUT="$PWD/artifacts/runs/latent-grpo-math-seed17"
```

Explain that model source URLs are used before runtime to populate the local directories. Validation/training always receive the local directory variables.

- [ ] **Step 4: Record High deviations and update the model handoff**

Document High `8→3 GPUs`, batch `32→12`, mini `32→12`, validation-only controls, and target runtime deferral. Update the technical handoff repository map and command section with both High profiles and local model path rule.

- [ ] **Step 5: Add a short Cairn log entry**

Record that Low/High now share config-driven wrappers while retaining separate author semantics and matching-family acceptance.

- [ ] **Step 6: Run complete verification**

```bash
git diff --check
bash -n tools/prepare_3gpu_assets.sh tools/run_3gpu_preflight.sh \
  tools/run_3gpu_final_validation.sh tools/run_3gpu_training.sh
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m pytest tests/unit -q
```

Expected: all commands PASS; runtime status remains target-machine deferred.
