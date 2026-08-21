# Latent-GRPO 3GPU 运行手册

状态：Low `16/32/0.45` 参数集合已在运行时等价的父提交 `8c9ce49` 完成目标机 validation；任何新的当前 HEAD 仍须重新生成 commit-bound acceptance。High 仍为 `TARGET_RUNTIME_EXECUTION_REQUIRED`。

基线提交：`53438ec07b804ebd1b670d6fe118199798350505`

本手册面向 Linux 三卡机器上的 VSCode + shell；不使用 Docker，不要求 Codex，不修改 Python/Hydra 源码。唯一正式 launcher 是 `ray_direct`：一个 Python driver 启动一套 Ray runtime 和一份训练 job，不使用外层多进程控制面。

## 1. 解压最终实验包并确认代码身份

```bash
unzip Latent-GRPO-final-experiment-package.zip
cd Latent-GRPO-final
git status --short
git rev-parse HEAD
```

`git status --short` 必须无输出。压缩包已经包含一个干净的 Git 提交；不要重新 checkout 历史基线，否则会丢失三卡封装和修复。上面的基线提交用于作者代码溯源，最终实验应记录当前包内 `git rev-parse HEAD`。

## 2. 创建并安装无 Docker 环境

要求 Linux、Python 3.11、validation output 所在文件系统至少 20 GiB 可用空间、CUDA 12.4 兼容驱动、恰好选择三张至少 40 GB 显存且支持 BF16 的 GPU。按仓库已审计顺序执行：

```bash
# SYSTEM_PYTHON 必须是 Python 3.11；conda 用户可先创建一个 3.11 bootstrap 环境。
# 先把 pip/HF/Ray cache 指向大容量盘，避免安装阶段占满系统盘。
export STORAGE_ROOT=/data/$USER/latent-grpo
export CACHE_ROOT="$STORAGE_ROOT/cache"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export HF_HOME="$CACHE_ROOT/huggingface"
export RAY_TMPDIR="$CACHE_ROOT/ray"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$RAY_TMPDIR"
export SYSTEM_PYTHON="$(command -v python)"
export TARGET_VENV="$PWD/.venv-target"
# 安装/ABI 报告写入 Git 忽略目录，避免覆盖仓库中的历史占位报告。
export TARGET_REPORT_DIR="$PWD/artifacts/runtime_setup"
python --version
nvidia-smi
# flash-attn 2.7.3 需要 nvcc，除非环境中已经预装了完全匹配的 wheel。
nvcc --version

bash scripts/target_machine/01_create_venv.sh
source .venv-target/bin/activate
export PYTHON_BIN="$PWD/.venv-target/bin/python"
bash scripts/target_machine/02_install_pytorch.sh
export SGL_KERNEL_VERSION=0.1.1
bash scripts/target_machine/03_install_runtime.sh
bash scripts/target_machine/04_import_check.sh
```

最终包已将活动依赖统一为 `flashinfer-python==0.2.5`、`sgl-kernel==0.1.1`，并在安装后运行 `pip check`。包内 `release_validation/LOCAL_RELEASE_ACCEPTANCE.json` 记录静态、单元测试和四个配置 dry-run 的交付验收；只有其 `git_commit` 等于当前 `git rev-parse HEAD` 时才有效。L20/CUDA/NCCL 真机验收仍以第 5 节生成的 `acceptance.json` 为准。不要凭经验更换 torch、CUDA、SGLang、FlashInfer 或 kernel 版本。Low 正式配置为 console-only；High 正式配置仍含 W&B logger，wrapper 默认设置 `WANDB_MODE=offline`，无需登录。

## 3. 准备本地模型、数据和输出路径

Hugging Face 页面只用于确认模型来源或预先下载；训练 wrapper 的 `--model-path` **必须传现存本地目录**，不能传仓库 ID 或 URL。

| profile | 模型来源（仅下载/溯源） | 本地训练集 | 本地验证集 |
|---|---|---|---|
| Low | [DJCheng/LLaMA3.2-1B-Instruct-Latent-SFT-Top10](https://huggingface.co/DJCheng/LLaMA3.2-1B-Instruct-Latent-SFT-Top10) | `GSM8k-Aug-oss-dup-all.parquet` | `GSM8k-Aug-test.parquet` |
| High | [DJCheng/Qwen2.5-Math-7B-Latent-SFT-4k-Top10](https://huggingface.co/DJCheng/Qwen2.5-Math-7B-Latent-SFT-4k-Top10) | `DAPO-Math-17k-en-train.parquet` | `Math-500-test.parquet` |

一次性设置 Low/High 的本地资产和独立输出目录：

```bash
# 请替换为服务器上的大容量持久化盘；不要把模型、pip cache 和正式 checkpoint 放在小系统盘。
export STORAGE_ROOT=/data/$USER/latent-grpo
export CACHE_ROOT="$STORAGE_ROOT/cache"
export EXPERIMENT_ROOT="$STORAGE_ROOT/experiments"
export HF_HOME="$CACHE_ROOT/huggingface"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export RAY_TMPDIR="$CACHE_ROOT/ray"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$RAY_TMPDIR" "$EXPERIMENT_ROOT"

export LOW_MODEL_PATH="$STORAGE_ROOT/models/LLaMA3.2-1B-Instruct-Latent-SFT-Top10"
export LOW_TRAIN_DATA="$STORAGE_ROOT/data/GSM8k-Aug-oss-dup-all.parquet"
export LOW_VAL_DATA="$STORAGE_ROOT/data/GSM8k-Aug-test.parquet"
export LOW_VALIDATION_ROOT="$EXPERIMENT_ROOT/validation/3gpu-final-low-seed17"
export LOW_TRAIN_OUTPUT="$EXPERIMENT_ROOT/runs/latent-grpo-gsm8k-seed17"

export HIGH_MODEL_PATH="$STORAGE_ROOT/models/Qwen2.5-Math-7B-Latent-SFT-4k-Top10"
export HIGH_TRAIN_DATA="$STORAGE_ROOT/data/DAPO-Math-17k-en-train.parquet"
export HIGH_VAL_DATA="$STORAGE_ROOT/data/Math-500-test.parquet"
export HIGH_VALIDATION_ROOT="$EXPERIMENT_ROOT/validation/3gpu-final-high-seed17"
export HIGH_TRAIN_OUTPUT="$EXPERIMENT_ROOT/runs/latent-grpo-math500-seed17"

# 对当前 8×L20 服务器，使用空闲的物理 GPU 4、5、6；不要加入已占用的 GPU7。
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export RUN_GPUS=4,5,6
```

设置 `CUDA_VISIBLE_DEVICES=4,5,6` 后，程序内部的 `cuda:0/1/2` 分别映射到物理 GPU 4/5/6。preflight 会核对物理编号、空闲显存和瞬时利用率；默认要求已有显存占用不超过 2048 MiB、利用率不超过 20%。

模型尚未落盘时，可用来源 ID 下载到上面的本地目录；数据仍使用本地文件：

```bash
bash tools/prepare_3gpu_assets.sh \
  --model-source DJCheng/LLaMA3.2-1B-Instruct-Latent-SFT-Top10 \
  --model-path "$LOW_MODEL_PATH" \
  --train-data "$LOW_TRAIN_DATA" \
  --val-data "$LOW_VAL_DATA"

bash tools/prepare_3gpu_assets.sh \
  --model-source DJCheng/Qwen2.5-Math-7B-Latent-SFT-4k-Top10 \
  --model-path "$HIGH_MODEL_PATH" \
  --train-data "$HIGH_TRAIN_DATA" \
  --val-data "$HIGH_VAL_DATA"
```

数据缺失时应先从团队共享盘复制到上述路径，也可向准备脚本显式传团队确认的本地源，例如 `--train-source /mnt/share/DAPO-Math-17k-en-train.parquet`。wrapper 不会擅自下载数据。若误把 `DJCheng/...` 传给运行时 `--model-path`，preflight 会以 `local_model_directory_missing` 阻断。

## 4. 单独执行 preflight

Low preflight：

```bash
bash tools/run_3gpu_preflight.sh \
  --config configs/3gpu-final-validation.yaml \
  --model-path "$LOW_MODEL_PATH" \
  --train-data "$LOW_TRAIN_DATA" \
  --val-data "$LOW_VAL_DATA" \
  --output-root "$LOW_VALIDATION_ROOT" \
  --gpus "$RUN_GPUS"
```

High preflight 只切换 profile 与对应本地资产：

```bash
bash tools/run_3gpu_preflight.sh \
  --config configs/3gpu-final-high-validation.yaml \
  --model-path "$HIGH_MODEL_PATH" \
  --train-data "$HIGH_TRAIN_DATA" \
  --val-data "$HIGH_VAL_DATA" \
  --output-root "$HIGH_VALIDATION_ROOT" \
  --gpus "$RUN_GPUS"
```

命令检查 Linux、Python 3.11、Git/工作区、三张选中 GPU、显存、CUDA、BF16、NCCL、Ray、依赖、parquet、本地模型/tokenizer 和对应 latent-end marker。成功末尾是 `3GPU_PREFLIGHT_GATE: PASS`。

## 5. 一键执行最终三卡 validation

每个 validation 独立输出到其 root 下的 `run`，执行两个真实 BF16 optimizer updates、29 指标、三 worker 聚合、Stage 3、Stage 4、CUDA RNG、checkpoint 和最小 resume load；它不会自动开始正式训练。

Low：

```bash
bash tools/run_3gpu_final_validation.sh \
  --config configs/3gpu-final-validation.yaml \
  --model-path "$LOW_MODEL_PATH" \
  --train-data "$LOW_TRAIN_DATA" \
  --val-data "$LOW_VAL_DATA" \
  --output-root "$LOW_VALIDATION_ROOT" \
  --gpus "$RUN_GPUS" \
  --seed 17
```

High：

```bash
bash tools/run_3gpu_final_validation.sh \
  --config configs/3gpu-final-high-validation.yaml \
  --model-path "$HIGH_MODEL_PATH" \
  --train-data "$HIGH_TRAIN_DATA" \
  --val-data "$HIGH_VAL_DATA" \
  --output-root "$HIGH_VALIDATION_ROOT" \
  --gpus "$RUN_GPUS" \
  --seed 17
```

成功时必须同时看到：

```text
3GPU_PREFLIGHT_GATE: PASS
3GPU_DISTRIBUTED_RUNTIME_GATE: PASS
CORE_METRICS: 29/29
CUDA_RNG_ALL_DEVICES: PASS
CHECKPOINT_GATE: PASS
3GPU_FINAL_GATE: PASS
```

机器报告在对应 `$*_VALIDATION_ROOT/acceptance.json`，且 `profile_name` 必须分别为 `3gpu-final-validation` 或 `3gpu-final-high-validation`。当前 SGLang 配置占用率为 Low `0.45`、High `0.8`。任何一项未证实都会非零退出并打印 `BLOCKED_REASON`、`LOG_PATH`、`NEXT_ACTION`。

## 6. 对应 validation PASS 后一键启动正式训练

正式训练前先确认输出盘空间。High 7B 的 optimizer/offload/checkpoint 组合可能产生很大的主存与磁盘峰值，不能只满足 validation 的 20 GiB 最低门槛：

```bash
df -h "$EXPERIMENT_ROOT" "$CACHE_ROOT"
free -h
```

正式配置是 `configs/3gpu-final-low.yaml`。它保留作者算法/采样语义，并显式包含 8→3 GPU、`train_batch_size` 64→48、`ppo_mini_batch_size` 16→12 的拓扑适配，以及已通过 validation 的 `actor_micro=16`、`logprob_micro=32`、SGLang `0.45` 吞吐/显存适配，因此不是 strict paper reproduction。Low formal 还关闭 metrics/support/checkpoint/credit probes，使用 `save_freq=5000`、`test_freq=-1`、`val_before_train=false` 和 console-only；完整观测证据由绑定同一 HEAD 的 Low validation 提供。

Low 一键训练：

```bash
bash tools/run_3gpu_training.sh \
  --config configs/3gpu-final-low.yaml \
  --model-path "$LOW_MODEL_PATH" \
  --train-data "$LOW_TRAIN_DATA" \
  --val-data "$LOW_VAL_DATA" \
  --output-root "$LOW_TRAIN_OUTPUT" \
  --gpus "$RUN_GPUS" \
  --seed 17 \
  --acceptance-report "$LOW_VALIDATION_ROOT/acceptance.json"
```

High 一键训练：

```bash
bash tools/run_3gpu_training.sh \
  --config configs/3gpu-final-high.yaml \
  --model-path "$HIGH_MODEL_PATH" \
  --train-data "$HIGH_TRAIN_DATA" \
  --val-data "$HIGH_VAL_DATA" \
  --output-root "$HIGH_TRAIN_OUTPUT" \
  --gpus "$RUN_GPUS" \
  --seed 17 \
  --acceptance-report "$HIGH_VALIDATION_ROOT/acceptance.json"
```

脚本拒绝已有 output，并拒绝将 Low acceptance 用于 High 或反向混用；acceptance 还必须来自当前 Git 提交。正式启动前会再次执行 preflight，防止 validation 后 GPU 被其他任务占用或环境发生漂移。启动前写入 `run_manifest.json` 和 `resolved_config.yaml`；每个 profile/seed 使用新的输出目录。

正式 profile 的训练长度由 `trainer.total_epochs` 控制：Low 为 10 epochs，High 为 5 epochs。YAML 中的 `training.max_steps` 只服务于有界 validation/profile 基础结构，不会把正式训练截成两步；两步验收必须使用对应的 `*-validation.yaml`。

## 7. 查看指标、checkpoint、停止与 resume

29 个核心指标和 `train/raw_generated_token_count` 位于启用 metrics 的 validation/High 训练目录下的 Parquet 表。当前 Low formal 为 observer-off，不保证生成这些 Parquet 指标或 probe sidecar；Low formal 应主要检查 `training.log`、`run_manifest.json`、`resolved_config.yaml`、global step 和 checkpoint。检查启用 metrics 的输出结构：

```bash
python scripts/validate_outputs.py --input "$LOW_TRAIN_OUTPUT"
cat "$LOW_TRAIN_OUTPUT/run_manifest.json"
cat "$LOW_TRAIN_OUTPUT/latest_checkpointed_iteration.txt"
find "$LOW_TRAIN_OUTPUT" -maxdepth 2 -name latent_grpo_metrics_sidecar.json -print
```

Stage 4 在启用 probe 的 profile 中是 checkpoint-only；当前 Low formal 已关闭 probe，不要为了每 step 出值而临时打开或改变频率。正常停止先在训练终端按一次 `Ctrl-C`，等待 Ray worker 退出；若仍有本 job 的进程，再检查 `ray status`，不要杀死其他用户的 Ray 集群。

从已验证 checkpoint resume 使用同一 output、同一 seed、同一配置和资产：

```bash
export RESUME_STEP="$(cat "$LOW_TRAIN_OUTPUT/latest_checkpointed_iteration.txt")"
export RESUME_FROM="$LOW_TRAIN_OUTPUT/global_step_$RESUME_STEP"
CUDA_VISIBLE_DEVICES="$RUN_GPUS" "$PYTHON_BIN" train_latent_grpo.py \
  --config configs/3gpu-final-low.yaml \
  --model-path "$LOW_MODEL_PATH" \
  --train-files "$LOW_TRAIN_DATA" \
  --val-files "$LOW_VAL_DATA" \
  --output-root "$LOW_TRAIN_OUTPUT" \
  --seed 17 \
  --resume-from "$RESUME_FROM"
```

## 8. 常见 BLOCKED

| blocker | 原因 | 修复 |
|---|---|---|
| `git_working_tree_not_clean` | 源码或配置未记录 | 提交/移走变更后重跑；不要 reset 用户代码 |
| `selected_gpu_count_must_equal_3` | `--gpus` 不是三个 ID | 使用例如 `--gpus "$RUN_GPUS"` |
| `selected_gpu_not_idle_or_mapping_failed` | 指定物理 GPU 不匹配、已有显存占用或利用率过高 | 重新运行 `nvidia-smi`，只选择空闲卡；当前机器使用 4、5、6 |
| `pip_dependency_consistency_gate_failed` | 安装元数据或包版本不一致 | 使用最终包的 0.2.5/0.1.1 固定组合重新建环境 |
| `runtime_import_or_abi_gate_failed` | CUDA wheel/fork/kernel ABI 不一致 | 回到第 2 节按固定顺序重装 |
| `local_model_directory_missing` | `--model-path` 是仓库 ID、URL 或不存在路径 | 先把来源模型下载到本地目录，再传目录绝对路径 |
| `asset_tokenizer_gate_failed` | parquet 无效或 token 524（Low）/522（High）不匹配 | 核对对应作者模型和数据，不改 token ID 绕过 |
| `metrics_probe_rng_or_checkpoint_acceptance_failed` | 29 指标、聚合、RNG、污染或 checkpoint 之一未证实 | 查看 `$OUTPUT_ROOT/logs/probe.log` 和 `acceptance.json.blockers` |
| `gpu_memory_telemetry` | 三卡利用率、两步 allocator 或三 worker probe 开销不完整/异常增长 | 查看 `gpu_telemetry.json`、`run/gpu_runtime_metrics.json`、`run/probe_worker_runtime.json` |

作者真值见 `configs/author/latent_grpo_gsm8k_llama3.yaml` 与 `configs/author/latent_grpo_math_qwen.yaml`；所有差异见 `docs/3GPU_HYPERPARAMETER_DEVIATIONS.md`，目标是 `0 silent deviations`。
