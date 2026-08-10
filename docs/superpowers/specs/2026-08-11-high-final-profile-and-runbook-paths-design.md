# High Final Profile And Runbook Paths Design

Date: 2026-08-11

## Goal

Give the low- and high-difficulty experiments symmetric, fail-closed 3GPU configuration and operator workflows. Extend the runbook so teammates pass local model directories, local parquet files, and separate validation/training output directories without editing source code.

## Supported Experiment Families

The low family remains:

- formal: `configs/3gpu-final-low.yaml`
- validation: `configs/3gpu-final-validation.yaml`
- model source: `DJCheng/LLaMA3.2-1B-Instruct-Latent-SFT-Top10`
- local data: `GSM8k-Aug-oss-dup-all.parquet`, `GSM8k-Aug-test.parquet`

The high family adds:

- formal: `configs/3gpu-final-high.yaml`
- validation: `configs/3gpu-final-high-validation.yaml`
- model source: `DJCheng/Qwen2.5-Math-7B-Latent-SFT-4k-Top10`
- local data: `DAPO-Math-17k-en-train.parquet`, `Math-500-test.parquet`

Hugging Face IDs/pages are download provenance only. Every preflight, validation, and formal-training example passes `--model-path` as an existing target-machine directory. Train and validation data arguments are existing local parquet files, never dataset IDs.

## High Author Semantics And Topology Adaptation

`configs/author/latent_grpo_math_qwen.yaml` remains immutable source truth. The formal high profile keeps the author's LR `1e-6`, rollout `n=8`, prompt/response lengths `1024/4096`, PPO micro batch `1`, KL enabled, gradient checkpointing enabled, actor parameter/optimizer offload enabled, max model/batched-token length `12000`, BF16, latent-end token ID `522`, Gumbel/Top-K/noise settings, dynamic filtering, and 5 epochs.

The author used 8 GPUs with train/mini batch `32/32`. The 3GPU formal profile uses `12/12`: both yield 32 normalized trajectories per actor rank (`32×8÷8` and `12×8÷3`). This is an explicit target-topology adaptation, not strict paper reproduction. The high validation profile uses prompt/mini batch `3/3`, two real optimizer updates, and checkpoint frequency 1; its results are gate evidence only.

## Configuration-Driven Operator Flow

The operator scripts become experiment-family neutral:

1. `tools/run_3gpu_preflight.sh --config <validation-profile>` loads the supplied profile for dry-run and tokenizer latent-end validation.
2. `tools/run_3gpu_final_validation.sh --config <validation-profile>` passes the same profile through preflight, both training/resume invocations, and the final validator.
3. `tools/validate_3gpu_final.py` filters durable rows using the loaded validation profile instead of a hard-coded low profile.
4. `tools/run_3gpu_training.sh --config <formal-profile>` obtains the declared profile name from the YAML/config loader rather than forcing `3gpu-final-low`.

Low remains the default config for backward-compatible commands. High commands provide the high config explicitly.

## Acceptance Identity

`acceptance.json` records the validation `profile_name`. Formal training maps:

- `3gpu-final-low` → `3gpu-final-validation`
- `3gpu-final-high` → `3gpu-final-high-validation`

The training wrapper blocks if the acceptance report is missing, `final_gate` is not `PASS`, or its profile does not match the selected formal family. Thus a low PASS cannot authorize high training.

## Runbook Path Contract

`docs/3GPU_RUNBOOK.md` gains a Low/High table and copy/paste blocks with these variables:

- `MODEL_PATH`: existing local model directory containing model/tokenizer metadata.
- `TRAIN_DATA`: existing local training parquet file.
- `VAL_DATA`: existing local validation parquet file.
- `CACHE_ROOT`: local download/cache root; examples export `HF_HOME` under it for asset preparation.
- `VALIDATION_ROOT`: fresh or reusable parent dedicated to one validation family; the script rejects an existing run/acceptance result.
- `TRAIN_OUTPUT`: new, nonexistent directory dedicated to one experiment family and seed.

Low and high examples use distinct validation and training output directories. The official Hugging Face model page is shown next to each family so the teammate knows what to download, but runtime commands always use local `MODEL_PATH`.

## Verification

Tests must prove:

- both new high profiles parse and preserve all high author-frozen values;
- formal high batch arithmetic preserves 32 normalized trajectories per rank;
- generic wrappers propagate the supplied config/profile and contain no low-only launch hard-coding;
- final reports contain validation profile identity and formal training rejects a cross-family report;
- the runbook includes Low/High local model/data/output examples and does not pass a Hugging Face ID directly to runtime commands;
- both low and high profiles pass dry-run/schema checks;
- existing low and complete `tests/unit` regression suites remain green.

No target CUDA claim is upgraded by these changes. High remains `TARGET_RUNTIME_EXECUTION_REQUIRED` until the high validation profile passes on the actual Linux 3GPU machine.
