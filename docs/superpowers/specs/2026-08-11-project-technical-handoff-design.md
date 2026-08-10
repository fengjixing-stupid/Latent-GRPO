# Project Technical Handoff Document Design

Date: 2026-08-11

## Goal

Create `docs/PROJECT_TECHNICAL_HANDOFF.md` as a self-contained technical introduction that a teammate can provide directly to another large language model. The document must let that model understand the project before inspecting source code, while preventing it from confusing local static evidence with completed three-GPU runtime validation.

## Audience And Scope

The primary reader is a capable coding or research model receiving the repository without prior conversation history. A human teammate is the secondary reader. The document covers the outer runner, vendored author implementation, metric instrumentation, validation/training workflow, current evidence state, and navigation pointers. It does not duplicate installation commands or full hyperparameter tables already owned by dedicated runbooks and audit documents.

## Document Structure

1. Identity and current status: project purpose, Git baseline, author repository relationship, and `TARGET_RUNTIME_EXECUTION_REQUIRED`.
2. System model: one `ray_direct` driver, Ray/FSDP actors, customized VERL and SGLang, and the driver-owned durable metrics path.
3. Repository map: authoritative entrypoints, profiles, runtime patches, tests, reports, and knowledge files.
4. Training and evidence flow: config validation, asset/preflight gates, rollout, real optimizer update, 29 metrics, Stage 3/4, checkpoint/resume, and final acceptance.
5. Semantic guardrails: author-frozen parameters, documented 3GPU adaptations, forbidden silent changes, and evidence labels.
6. Current state and next work: what passes locally, what remains target-machine gated, known operational assumptions, and exact first files/commands another model should inspect.
7. Reusable handoff prompt: a short prompt the teammate can paste with the document when assigning work to another model.

## Source Hierarchy

Facts are resolved in this order:

1. Current source and tests in the Git worktree.
2. Author experiment shells and machine-readable author truth YAML.
3. `docs/3GPU_RUNBOOK.md`, acceptance checklist, hyperparameter audit, and deviation register.
4. Project Cairn records for stable decisions and lessons.
5. Historical audit/progress documents, which may describe older incomplete states and must be labeled as historical when referenced.

## Accuracy And Safety Rules

- State the task-start baseline commit separately from uncommitted packaging changes.
- Never claim CUDA, NCCL, BF16, FSDP, three-device RNG, or final 3GPU PASS from Mac tests.
- Distinguish the vendored author implementation under `Latent-GRPO/` from the outer packaging and runner.
- Describe the formal low profile as a “3-GPU target-runtime / engineering adaptation,” not strict paper reproduction.
- Link to detailed documents instead of copying large command blocks or parameter tables.
- Mark older documents such as `docs/progress.md` as historical when their status conflicts with current code.
- Contain no placeholders, secrets, machine-specific private paths, or unsupported performance claims.

## Acceptance Checks

The finished document must:

- Name the real launcher and single-Ray invariant.
- Identify `train_latent_grpo.py`, both final profiles, all four operator wrappers, and the final validator.
- Explain all 29 metric stages plus `train/raw_generated_token_count` at summary level.
- State local verification evidence and target runtime deferral exactly.
- Point to author truth, deviation, acceptance, runbook, and Cairn documents.
- Include a concise recommended prompt for the teammate’s model.
- Pass a placeholder scan, Markdown link/path existence checks, and `git diff --check`.
