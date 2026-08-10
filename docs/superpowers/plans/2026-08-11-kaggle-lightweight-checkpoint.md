# Kaggle Lightweight Checkpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Kaggle dual-T4 metric checkpoint omit optimizer serialization while preserving default resumable checkpoints.

**Architecture:** The runner selects checkpoint contents through existing Hydra configuration. The vendored FSDP manager validates and honors that component list for sequential save/load operations.

**Tech Stack:** Python 3.9, PyTorch FSDP, Hydra, unittest/pytest.

## Global Constraints

- Kaggle saves `model` and `extra` only and disables resume.
- Other profiles retain `model`, `optimizer`, and `extra` defaults.
- Stage 4 scheduling and probe computation remain unchanged.

---

### Task 1: Lock Configuration Behavior

**Files:**
- Modify: `tests/unit/test_kaggle_t4_30_runtime.py`
- Modify: `tests/unit/test_config.py`
- Modify: `latent_grpo_runner/config.py`

**Interfaces:**
- Consumes: `ResolvedConfig.author_hydra_overrides()`
- Produces: Kaggle-only `actor_rollout_ref.actor.checkpoint.contents=[model,extra]` and `trainer.resume_mode=disable`

- [ ] Add failing assertions for Kaggle lightweight contents, disabled resume, and unchanged ordinary profiles.
- [ ] Run the focused tests and confirm they fail on missing overrides.
- [ ] Add the two Kaggle-only Hydra overrides.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Honor FSDP Checkpoint Contents

**Files:**
- Modify: `tests/unit/test_upstream_patch_contract.py`
- Modify: `Latent-GRPO/verl-0.4.x/verl/utils/checkpoint/fsdp_checkpoint_manager.py`

**Interfaces:**
- Consumes: `checkpoint_contents: list[str]`
- Produces: conditional sequential save/load of `model`, `optimizer`, and `extra`

- [ ] Add failing static contracts for optional components and sequential release.
- [ ] Run the focused contract and confirm it fails on the unconditional manager.
- [ ] Validate contents and conditionally save/load each selected component.
- [ ] Run focused and full unit tests.

### Task 3: Publish the Pinned Notebook

**Files:**
- Modify: `Latent-GRPO/cairn/LOG.md`
- Modify: `tests/unit/test_kaggle_29_metric_notebook.py`
- Regenerate: `Latent_GRPO_Kaggle_2xT4_30_Metric_Runtime_Validation.ipynb`

**Interfaces:**
- Consumes: committed runtime SHA
- Produces: clean notebook pinned to the lightweight-checkpoint runtime

- [ ] Commit the runtime implementation and obtain its SHA.
- [ ] Update the notebook expected SHA and project log.
- [ ] Regenerate the notebook and run full verification.
- [ ] Commit and push both commits to `origin/main`.
