# Project Technical Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create one self-contained Markdown technical handoff that a teammate can give directly to another large language model, including a copy/paste formal-training wrapper example.

**Architecture:** Add a content-contract test first, then create `docs/PROJECT_TECHNICAL_HANDOFF.md` from current source-of-truth files. The handoff remains concise by summarizing architecture and status while linking to the detailed 3GPU runbook, author truth, deviation register, acceptance checklist, and Cairn record.

**Tech Stack:** Markdown, Python 3.9+ unit tests, pytest, Git path checks.

## Global Constraints

- The primary reader is a coding or research model with no conversation history.
- Keep the document self-contained but do not duplicate long installation instructions or full hyperparameter tables.
- State `TARGET_RUNTIME_EXECUTION_REQUIRED`; do not claim Mac tests prove CUDA, NCCL, BF16, FSDP, or three-device RNG.
- Distinguish the outer runner and packaging from the vendored author tree under `Latent-GRPO/`.
- Name the formal low profile “3-GPU target-runtime / engineering adaptation,” not strict paper reproduction.
- Include no `/Users/...` path, secret, placeholder, or unsupported performance claim.
- End with a one-wrapper formal training example using `tools/run_3gpu_training.sh` and an acceptance report produced by final validation.

---

### Task 1: Freeze The Handoff Content Contract

**Files:**
- Modify: `tests/unit/test_3gpu_final_package.py`
- Test: `tests/unit/test_3gpu_final_package.py`

**Interfaces:**
- Consumes: repository root constant `ROOT` and the intended path `docs/PROJECT_TECHNICAL_HANDOFF.md`.
- Produces: a regression test that rejects missing architecture/status/navigation/training-handoff content.

- [x] **Step 1: Write the failing document contract test**

Append this test:

```python
def test_project_technical_handoff_is_model_ready() -> None:
    path = ROOT / "docs/PROJECT_TECHNICAL_HANDOFF.md"
    content = path.read_text(encoding="utf-8")
    for required in (
        "TARGET_RUNTIME_EXECUTION_REQUIRED",
        "ray_direct",
        "train_latent_grpo.py",
        "configs/3gpu-final-low.yaml",
        "configs/3gpu-final-validation.yaml",
        "29 个核心指标",
        "train/raw_generated_token_count",
        "3-GPU target-runtime / engineering adaptation",
        "tools/run_3gpu_final_validation.sh",
        "tools/run_3gpu_training.sh",
        "--acceptance-report",
        "给接手大模型的推荐提示词",
    ):
        assert required in content
    assert "/Users/" not in content
    for marker in ("T" + "BD", "T" + "ODO"):
        assert marker not in content
```

- [x] **Step 2: Run the test and verify the missing-file failure**

Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m pytest \
  tests/unit/test_3gpu_final_package.py::test_project_technical_handoff_is_model_ready -q
```

Expected: FAIL with `FileNotFoundError` for `docs/PROJECT_TECHNICAL_HANDOFF.md`.

- [x] **Step 3: Review the test boundary**

Confirm that the test checks stable concepts and executable entrypoints, not prose formatting or exact paragraph wording.

### Task 2: Write The Model-Ready Technical Handoff

**Files:**
- Create: `docs/PROJECT_TECHNICAL_HANDOFF.md`
- Test: `tests/unit/test_3gpu_final_package.py`

**Interfaces:**
- Consumes: `Latent-GRPO/README.md`, `train_latent_grpo.py`, `configs/author/*.yaml`, `configs/3gpu-final-*.yaml`, `docs/3GPU_*.md`, `docs/AUTHOR_HYPERPARAMETER_AUDIT.md`, and `Latent-GRPO/cairn/3gpu-runtime-packaging.md`.
- Produces: the single file a teammate passes to another model before assigning repository work.

- [x] **Step 1: Create the document with the approved section order**

Use these exact top-level sections:

```markdown
# Latent-GRPO 项目技术交接
## 1. 给接手大模型的事实边界
## 2. 项目目标与算法定位
## 3. 系统架构与训练数据流
## 4. 仓库地图与权威入口
## 5. 配置、作者真值与 3GPU 适配
## 6. 指标、checkpoint 与验收证据
## 7. 当前完成状态与未验证边界
## 8. 接手后的推荐阅读顺序
## 9. 常见误判与禁止事项
## 10. 给接手大模型的推荐提示词
## 11. 一键验证与一键正式训练
```

The architecture section must describe this exact control flow:

```text
train_latent_grpo.py
→ strict YAML/config validation
→ one ray_direct Python driver
→ one Ray runtime/job
→ 3 FSDP actor workers + customized SGLang rollout
→ driver-owned worker aggregation and append-only metric output
```

- [x] **Step 2: Include the exact formal-training example**

The final command section must include:

```bash
export MODEL_PATH=/path/to/LLaMA3.2-1B-Instruct-Latent-SFT-Top10
export TRAIN_DATA=/path/to/GSM8k-Aug-oss-dup-all.parquet
export VAL_DATA=/path/to/GSM8k-Aug-test.parquet
export VALIDATION_ROOT="$PWD/artifacts/validation/3gpu-final"
export TRAIN_OUTPUT="$PWD/artifacts/runs/latent-grpo-gsm8k-seed17"

bash tools/run_3gpu_training.sh \
  --config configs/3gpu-final-low.yaml \
  --model-path "$MODEL_PATH" \
  --train-data "$TRAIN_DATA" \
  --val-data "$VAL_DATA" \
  --output-root "$TRAIN_OUTPUT" \
  --gpus 0,1,2 \
  --seed 17 \
  --acceptance-report "$VALIDATION_ROOT/acceptance.json"
```

State directly above the block that the wrapper refuses to start unless the acceptance report has `final_gate == PASS`, and refuses to overwrite an existing training output directory.

- [x] **Step 3: Run the focused contract test**

Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m pytest \
  tests/unit/test_3gpu_final_package.py::test_project_technical_handoff_is_model_ready -q
```

Expected: PASS.

- [x] **Step 4: Preserve the tested handoff without an unsafe mixed commit**

The commit was intentionally deferred because `tests/unit/test_3gpu_final_package.py`
already contains uncommitted work from the preceding 3GPU packaging task. Staging the
whole file here would create a partial commit whose tests depend on unstaged implementation.

```bash
git status --short
```

### Task 3: Validate Navigation, Accuracy, And Repository Hygiene

**Files:**
- Modify: `Latent-GRPO/cairn/LOG.md`
- Verify: `docs/PROJECT_TECHNICAL_HANDOFF.md`
- Verify: `tests/unit/test_3gpu_final_package.py`

**Interfaces:**
- Consumes: the completed document and all local relative Markdown links.
- Produces: verified handoff documentation and a short reverse-chronological Cairn pointer.

- [x] **Step 1: Check all repository-relative paths named in the handoff**

Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 - <<'PY'
from pathlib import Path

root = Path.cwd()
required = [
    "train_latent_grpo.py",
    "configs/3gpu-final-low.yaml",
    "configs/3gpu-final-validation.yaml",
    "tools/run_3gpu_final_validation.sh",
    "tools/run_3gpu_training.sh",
    "tools/validate_3gpu_final.py",
    "docs/3GPU_RUNBOOK.md",
    "docs/3GPU_ACCEPTANCE_CHECKLIST.md",
    "docs/AUTHOR_HYPERPARAMETER_AUDIT.md",
    "docs/3GPU_HYPERPARAMETER_DEVIATIONS.md",
    "Latent-GRPO/cairn/3gpu-runtime-packaging.md",
]
missing = [path for path in required if not (root / path).exists()]
raise SystemExit(f"missing paths: {missing}" if missing else 0)
PY
```

Expected: exit 0 with no missing paths.

- [x] **Step 2: Add a Cairn log pointer**

Add a top entry dated `2026-08-11` stating that `docs/PROJECT_TECHNICAL_HANDOFF.md` is the preferred single-file model context, while the 3GPU runtime remains target-machine gated.

- [x] **Step 3: Run verification**

Run:

```bash
git diff --check
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m pytest \
  tests/unit/test_3gpu_final_package.py -q
```

Expected: both commands PASS.

- [x] **Step 4: Review the final diff without committing unrelated files**

Run:

```bash
git status --short
git diff -- docs/PROJECT_TECHNICAL_HANDOFF.md \
  tests/unit/test_3gpu_final_package.py Latent-GRPO/cairn/LOG.md
```

Expected: only the intended handoff/test/log content is reviewed; pre-existing 3GPU packaging changes remain preserved.

- [x] **Step 5: Preserve the Cairn update without an unsafe mixed commit**

The commit was intentionally deferred because `Latent-GRPO/cairn/LOG.md` already contains
the preceding task's uncommitted 3GPU packaging entry.

```bash
git status --short
```
