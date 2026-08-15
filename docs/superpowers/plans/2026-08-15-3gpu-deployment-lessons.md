# Three-GPU Deployment Lessons Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a durable Chinese deployment-experience note and make its five-category three-GPU design check mandatory for future Codex-authored target-machine scripts.

**Architecture:** Keep training topology and runtime-gate truth in the existing `3gpu-runtime-packaging.md`; place target-machine installation, environment, proxy, GPU-mapping, and acceptance lessons in a new focused Cairn topic note. Keep `AGENTS.md` as a short always-read trigger and `cairn/LOG.md` as a reverse-chronological pointer rather than duplicating the conclusions.

**Tech Stack:** Markdown, YAML frontmatter, Project Cairn, Git diff/static text checks

## Global Constraints

- Write the Cairn topic note, `AGENTS.md`, and `LOG.md` in Chinese; keep frontmatter keys and file names unchanged.
- Do not add a `contributors` field because the teammate's stable identity is not available.
- Do not change `Latent-GRPO/cairn/3gpu-runtime-packaging.md`; it remains authoritative for topology, author-parameter semantics, and runtime-gate decisions.
- Keep `Latent-GRPO/AGENTS.md` at no more than 60 physical lines.
- Preserve every pre-existing working-tree change; `Latent-GRPO/cairn/LOG.md` is already modified, so do not discard, overwrite, or commit unrelated lines.
- Treat local package/static evidence separately from L20 target-runtime evidence; only `3GPU_FINAL_GATE: PASS` closes the latter.
- Keep destructive cleanup, process termination, TLS disabling, and silent formal-hyperparameter changes out of the recommended procedures.

---

### Task 1: Create the three-GPU deployment knowledge topic

**Files:**
- Create: `Latent-GRPO/cairn/3gpu-deployment-lessons.md`
- Read: `Latent-GRPO/cairn/3gpu-runtime-packaging.md`
- Read: `docs/superpowers/specs/2026-08-15-3gpu-deployment-lessons-design.md`

**Interfaces:**
- Consumes: the approved design specification, shared-conversation URL, and existing runtime-packaging conclusions.
- Produces: one `project_topic` note containing current conclusions, experience, lessons, procedure, reference, and the five-category design check.

- [ ] **Step 1: Verify the topic does not already exist**

Run:

```bash
test ! -e Latent-GRPO/cairn/3gpu-deployment-lessons.md
```

Expected: exit code 0. If the file exists, inspect and update it in place instead of replacing it.

- [ ] **Step 2: Create exact Project Cairn frontmatter and scope**

Start the file with:

```yaml
---
type: project_topic
status: active
summary: "三卡 L20 目标机部署必须隔离 CUDA/Python 工具链、固定物理/逻辑 GPU 映射，并以真机门禁闭合。"
tags: [3gpu, deployment, cuda, conda, proxy, runtime-gate]
contains: [decision, experience, lesson, procedure, reference]
created: "2026-08-15"
updated: "2026-08-15"
related:
  - "3gpu-runtime-packaging.md"
  - "../../docs/3GPU_RUNBOOK.md"
  - "../../docs/superpowers/specs/2026-08-15-3gpu-deployment-lessons-design.md"
authoring_mode: ai_generated
---
```

Use the title `# 三卡目标机部署经验与脚本设计检查`. State that the shared conversation is the source, the observed target was 8×L20 with physical GPUs 4/5/6 selected, and time-varying hardware occupancy must always be re-probed.

- [ ] **Step 3: Write the current conclusions and evidence boundary**

Record these exact conclusions:

```text
Driver 550.90.07 reporting CUDA 12.4 capability does not prove nvcc 12.4 is installed.
The target Python/pip must come from .venv-target; nvcc/CUDA_HOME/CUDACXX must come from the CUDA 12.4 bootstrap environment.
Physical GPUs 4/5/6 become logical cuda:0/1/2, and telemetry/preflight/acceptance must share that mapping.
One Python Driver owns Ray and three FSDP workers; no outer torchrun layer is added.
Local/static validation remains TARGET_RUNTIME_EXECUTION_REQUIRED until the two-step target validation emits 3GPU_FINAL_GATE: PASS.
```

Write the prose in Chinese and keep the literal command, variable, and gate names unchanged.

- [ ] **Step 4: Write the failure matrix**

For every row, include `现象`, `原因`, `修复`, `验证`, and `防复发检查`:

| Failure | Required resolution |
|---|---|
| Package paths/version conflicts, all-machine telemetry, hard-coded Top-K, dirty tree | Move docs to the tested path, unify FlashInfer/sgl-kernel pins, filter selected physical GPUs, derive Top-K width, and separate package validation from target validation. |
| `nvidia-smi` says CUDA 12.4 while `nvcc --version` says 11.4 | Install/use user-space CUDA Toolkit 12.4 without changing the shared host driver or `/usr/local/cuda`. |
| Conda `.zst` metadata returns proxy 502 | Use `--no-repodata-use-zst`, `--repodata-fn repodata.json`, and disable shards when supported. |
| `conda create` still visits failing default/legacy channels | Add `--override-channels` and only explicit, connectivity-tested NVIDIA and `pkgs/main` channels. |
| Bare `pip` shows CUDA 13/Torch 2.12 while target Python has no torch | Put `.venv-target/bin` before the CUDA bootstrap `bin`, pin `PYTHON_BIN`, and use only `"$PYTHON_BIN" -m pip`. |
| PyTorch cu124 index cannot resolve `nvidia-cudnn-cu12==9.1.0.70` | Download the exact PyPI wheel with `--no-deps`, verify SHA-256 `165764f44ef8c61fcdfdfdbe769d687e06374059fbb388b6c89ecb0e28793a6f`, install locally, then rerun the pinned PyTorch installer. |
| Process search appears to show unrelated pip installs | Avoid regexes that match `pipe_handle`; identify exact user, interpreter, and install command before any process action. |
| `nvidia-smi Processes` is empty while GPUs are busy | Account for PID namespace/container isolation and use memory, utilization, power, P-State, and in-process CUDA visibility together. |
| GPU7 has memory in use while 4/5/6 are idle | Keep the synchronous job on 4/5/6; re-probe immediately before launch and fail closed on occupancy thresholds. |
| Local tests pass | Do not call the L20 run accepted; require matching two-step acceptance and `3GPU_FINAL_GATE: PASS`. |

- [ ] **Step 5: Write the five-category script design check**

Add checkboxes under these exact categories:

```text
GPU 与编号映射
启动拓扑
解释器、CUDA 与依赖
网络、代理与存储
算法语义与验收证据
```

Each category must require a concrete result or an explicit `不适用：原因`; unknown target facts must cause probe-first or fail-closed behavior.

- [ ] **Step 6: Add the recommended procedure and concise appendix**

Document the order:

```text
资源/磁盘/工作树探测
→ 固定 bootstrap CUDA 12.4 与 .venv-target 路径
→ 版本、pip check、ABI/import gate
→ 模型/数据/输出路径检查
→ GPU4/5/6 映射检查
→ Low 两步 validation
→ 3GPU_FINAL_GATE: PASS
→ 正式 Low
→ High validation 与正式 High
```

Add a short appendix for the conversation's ancillary lessons: `top/free/nvidia-smi` interpretation, pressure-test concurrency coming from `--concurrency` with default 2, GitLab author-email rejection, and `cannot edit in read-only editor`. Do not reproduce long terminal dumps.

- [ ] **Step 7: Validate the topic note**

Run:

```bash
rg -n '^type: project_topic$|^status: active$|^contains:.*lesson.*procedure.*reference|^# 三卡目标机部署经验与脚本设计检查$|3GPU_FINAL_GATE: PASS|nvidia-cudnn-cu12==9.1.0.70|--override-channels|PYTHON_BIN|物理 GPU' Latent-GRPO/cairn/3gpu-deployment-lessons.md
git diff --check -- Latent-GRPO/cairn/3gpu-deployment-lessons.md
```

Expected: every required concept is found and `git diff --check` produces no output.

---

### Task 2: Add the always-read trigger and Cairn log pointer

**Files:**
- Modify: `Latent-GRPO/AGENTS.md:54-57`
- Modify: `Latent-GRPO/cairn/LOG.md:5`

**Interfaces:**
- Consumes: `Latent-GRPO/cairn/3gpu-deployment-lessons.md` and `Latent-GRPO/cairn/3gpu-runtime-packaging.md`.
- Produces: a mandatory Codex trigger discoverable from `AGENTS.md` and a reverse-chronological LOG entry pointing to the current truth.

- [ ] **Step 1: Capture the pre-existing overlapping diff**

Run:

```bash
git diff -- Latent-GRPO/AGENTS.md Latent-GRPO/cairn/LOG.md
```

Expected: preserve all existing content. Do not restore, reformat, or reorder unrelated LOG entries.

- [ ] **Step 2: Add one mandatory rule without exceeding 60 lines**

Append this single bullet under `## 知识沉淀规则`:

```markdown
- 生成或修改三卡目标机的训练、部署、环境、监控或验收脚本前，必须阅读 `cairn/3gpu-runtime-packaging.md` 与 `cairn/3gpu-deployment-lessons.md`；交付前逐项报告 GPU/映射、拓扑、环境/依赖、网络/存储、语义/验收五类检查，未知项必须先探测或 fail-closed。
```

Expected: `wc -l Latent-GRPO/AGENTS.md` reports 58 or fewer lines.

- [ ] **Step 3: Add the newest LOG entry**

Insert immediately before the current first dated entry:

```markdown
## 2026-08-15 · 沉淀三卡目标机部署经验

- 从 teammate 的 L20 部署排障中沉淀 CUDA/Conda/代理、双层 Python 环境、精确 cuDNN wheel、GPU 映射和证据分层经验。
- 后续生成或修改三卡训练、部署、环境、监控或验收脚本，必须执行五类设计检查并对未知项 fail-closed。
- 当前真相见 `3gpu-deployment-lessons.md`；训练拓扑与 runtime gate 仍见 `3gpu-runtime-packaging.md`。
```

Expected: the entry is below the LOG introduction, above every `2026-08-11` entry, and contains fewer than 20 lines.

- [ ] **Step 4: Validate the trigger and LOG ordering**

Run:

```bash
wc -l Latent-GRPO/AGENTS.md
rg -n '三卡目标机|3gpu-runtime-packaging.md|3gpu-deployment-lessons.md|GPU/映射|fail-closed' Latent-GRPO/AGENTS.md
sed -n '1,18p' Latent-GRPO/cairn/LOG.md
git diff --check -- Latent-GRPO/AGENTS.md Latent-GRPO/cairn/LOG.md
```

Expected: AGENTS has at most 60 lines; both topic paths and five-category requirement are present; the 2026-08-15 entry is first; whitespace check is clean.

---

### Task 3: Cross-file verification and handoff

**Files:**
- Verify: `Latent-GRPO/cairn/3gpu-deployment-lessons.md`
- Verify: `Latent-GRPO/AGENTS.md`
- Verify: `Latent-GRPO/cairn/LOG.md`
- Verify: `Latent-GRPO/cairn/3gpu-runtime-packaging.md`

**Interfaces:**
- Consumes: all Task 1 and Task 2 outputs.
- Produces: evidence that the rule is discoverable, responsibilities do not overlap, and unrelated dirty-worktree content remains preserved.

- [ ] **Step 1: Run cross-file discovery checks**

Run:

```bash
rg -n '3gpu-(runtime-packaging|deployment-lessons)\.md' Latent-GRPO/AGENTS.md Latent-GRPO/cairn/LOG.md Latent-GRPO/cairn/3gpu-deployment-lessons.md
rg -n 'GPU 与编号映射|启动拓扑|解释器、CUDA 与依赖|网络、代理与存储|算法语义与验收证据' Latent-GRPO/cairn/3gpu-deployment-lessons.md
```

Expected: AGENTS points to both notes, LOG points to the new note, and all five categories exist exactly once as primary checklist headings.

- [ ] **Step 2: Confirm scope separation**

Run:

```bash
git diff --exit-code -- Latent-GRPO/cairn/3gpu-runtime-packaging.md
rg -n 'CUDA 12\.4|repodata\.json|--override-channels|nvidia-cudnn-cu12|\.venv-target' Latent-GRPO/cairn/3gpu-deployment-lessons.md
```

Expected: the existing runtime-packaging note has no new diff; all deployment-specific facts live in the new note.

- [ ] **Step 3: Review only task-owned diffs**

Run:

```bash
git diff -- Latent-GRPO/cairn/3gpu-deployment-lessons.md Latent-GRPO/AGENTS.md Latent-GRPO/cairn/LOG.md
git status --short
```

Expected: the three intended files contain the new work; all other pre-existing modified/untracked files remain untouched. Do not stage or commit `Latent-GRPO/cairn/LOG.md` while its unrelated pre-existing diff is mixed into the same file.

- [ ] **Step 4: Report what now works**

Report these outcomes without upgrading evidence:

```text
Future Codex runs can discover the mandatory three-GPU check from AGENTS.md.
The complete failure/solution history is in one focused Cairn topic note.
Runtime-packaging decisions remain separate and unchanged.
Documentation/static checks passed; no new L20 runtime execution was performed.
```
