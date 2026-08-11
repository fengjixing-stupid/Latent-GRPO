# 3GPU Final Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package Latent-GRPO so a teammate can run fail-closed 3-GPU final validation and then launch formal low-difficulty training without Docker or Codex.

**Architecture:** Keep the existing `ray_direct` runner as the single driver. Add author truth YAML, 3-GPU final profiles, shell wrappers, and tests that compare immutable author fields against the 3-GPU profiles. Runtime CUDA claims remain target-machine gated and produce PASS/BLOCKED reports only on the teammate machine.

**Tech Stack:** Python 3, PyYAML, Bash, existing `train_latent_grpo.py`, vendored `verl.trainer.main_ppo`.

## Global Constraints

- Source of truth baseline commit: `53438ec07b804ebd1b670d6fe118199798350505`.
- Target runtime: Linux, exactly 3 selected GPUs, at least 40 GB VRAM per GPU, CUDA/NCCL, BF16.
- Launcher: `ray_direct`; do not use `torchrun` for final 3-GPU commands.
- Author-published hyperparameters are authoritative unless a field is documented as a 3-GPU topology adaptation.
- Validation must not auto-start formal training.
- Local Mac work may only claim static, syntax, and unit validation; target runtime remains `TARGET_RUNTIME_EXECUTION_REQUIRED`.

---

### Task 1: Author Truth And Drift Tests

**Files:**
- Create: `configs/author/latent_grpo_gsm8k_llama3.yaml`
- Create: `configs/author/latent_grpo_math_qwen.yaml`
- Modify: `tests/unit/test_3gpu_final_package.py`

**Interfaces:**
- Consumes: `Latent-GRPO/Latent-GRPO-gsm8k-llama3.sh`, `Latent-GRPO/Latent-GRPO-math500-qwen.sh`
- Produces: machine-readable author YAML with `provenance.source_file`, `hydra_overrides`, and grouped values.

- [x] Step 1: Add failing test that loads both author YAML files and checks low/high values against the vendored shell strings.
- [x] Step 2: Run `python -m pytest tests/unit/test_3gpu_final_package.py::test_author_truth_files_capture_vendored_shell_values -q`; expected failure is missing author YAML.
- [x] Step 3: Create both author YAML files from the current vendored shell values.
- [x] Step 4: Re-run the test; expected PASS.

### Task 2: 3GPU Final Profiles

**Files:**
- Create: `configs/3gpu-final-low.yaml`
- Create: `configs/3gpu-final-validation.yaml`
- Modify: `latent_grpo_runner/config.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_3gpu_final_package.py`

**Interfaces:**
- Consumes: `load_config(path, workspace_root=ROOT)`
- Produces: supported profiles named `3gpu-final-low` and `3gpu-final-validation`.

- [x] Step 1: Add failing tests that both final profiles parse and keep author-frozen sampling/model/algorithm values.
- [x] Step 2: Run the focused tests; expected failure is unsupported/missing profiles.
- [x] Step 3: Add profile names to `SUPPORTED_PROFILES` and create YAML profiles.
- [x] Step 4: Re-run focused tests; expected PASS.

### Task 3: Target Shell Wrappers

**Files:**
- Create: `tools/prepare_3gpu_assets.sh`
- Create: `tools/run_3gpu_preflight.sh`
- Create: `tools/run_3gpu_final_validation.sh`
- Create: `tools/run_3gpu_training.sh`
- Modify: `tests/unit/test_3gpu_final_package.py`

**Interfaces:**
- Consumes: `train_latent_grpo.py --config ...`
- Produces: target commands that write logs under `artifacts/validation/3gpu-final/logs/` and fail closed with `BLOCKED_REASON`, `LOG_PATH`, and `NEXT_ACTION`.

- [x] Step 1: Add failing test that all shell wrappers exist, pass `bash -n`, avoid `torchrun`, and mention required gate labels.
- [x] Step 2: Run focused test; expected failure is missing wrappers.
- [x] Step 3: Create wrappers with strict argument parsing, explicit `CUDA_VISIBLE_DEVICES`, dry config gates, target runtime gates, and report writing.
- [x] Step 4: Re-run focused test; expected PASS.

### Task 4: Runbooks And Acceptance Docs

**Files:**
- Create: `docs/3GPU_RUNBOOK.md`
- Create: `docs/3GPU_ACCEPTANCE_CHECKLIST.md`
- Create: `docs/AUTHOR_HYPERPARAMETER_AUDIT.md`
- Create: `docs/3GPU_HYPERPARAMETER_DEVIATIONS.md`
- Modify: `tests/unit/test_3gpu_final_package.py`

**Interfaces:**
- Consumes: final commands from Task 3 and author/profile YAML from Tasks 1-2.
- Produces: teammate-ready copy/paste commands and 0-silent-deviation audit docs.

- [x] Step 1: Add failing test for required docs, baseline commit, `ray_direct`, `TARGET_RUNTIME_EXECUTION_REQUIRED`, validation command, and training command.
- [x] Step 2: Run focused test; expected failure is missing docs.
- [x] Step 3: Create docs with exact commands and audit/deviation tables.
- [x] Step 4: Re-run focused test; expected PASS.

### Task 5: Verification And Cairn Log

**Files:**
- Modify: `Latent-GRPO/cairn/LOG.md`

**Interfaces:**
- Consumes: all changed files.
- Produces: final static verification evidence and short project log entry.

- [x] Step 1: Run `git diff --check`.
- [x] Step 2: Run shell syntax checks for new tools.
- [x] Step 3: Run the focused package tests and the complete `tests/unit` suite.
- [x] Step 4: Run dry config checks for final profiles.
- [x] Step 5: Add a top LOG entry pointing to the 3-GPU final packaging docs.
