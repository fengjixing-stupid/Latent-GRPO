# Latent-GRPO 项目技术交接

本文是给接手本项目的工程师及其大模型使用的单文件上下文。它优先说明当前代码的事实边界、权威入口和不可擅改项；具体安装、逐字段超参数和验收操作请继续阅读文中链接的专项文档。

## 1. 给接手大模型的事实边界

- 项目在本轮 3GPU 打包开始时的 Git 基线是 `53438ec07b804ebd1b670d6fe118199798350505`。接手时必须重新执行 `git rev-parse HEAD` 和 `git status --short`，以实际工作树为准。
- 源码归档或无 GPU 本地环境只能证明配置解析、纯逻辑、shell 语法和 CPU 单元测试。Low `16/32/0.45` 参数集合已在运行时等价的父提交 `8c9ce49` 完成目标三卡全门禁 validation；任何新的当前 HEAD 仍必须在服务器重新生成 commit-bound `acceptance.json`。High 仍是 `TARGET_RUNTIME_EXECUTION_REQUIRED`。
- `Latent-GRPO/` 是 vendored 作者实现，外层目录是本项目增加的严格配置、launcher、metrics、验证和 teammate 打包。不要把两层代码混为一个 upstream。
- 较早的 `docs/progress.md`、审计报告和 decision log 记录的是历史阶段，可能仍写着“尚未实现”。判断当前状态时，以当前源码、测试、final profiles、3GPU 文档和 Cairn 最新记录为准。
- 目标机 preflight 要求 Git 工作树干净。开发目录中的未提交改动必须先由负责人审阅并形成可复现提交，不能用 reset 或忽略检查绕过。

## 2. 项目目标与算法定位

Latent-GRPO 是 Latent-SFT 之后的强化学习阶段：模型不是只生成离散思维链，而是在词表空间用 Top-K token 分布表示 latent reasoning state，再用 GRPO 优化这些轨迹。训练依赖已经学会 latent chain 的 SFT checkpoint；直接从普通 base/instruct 模型启动 latent RL 容易失稳，不属于支持路径。

核心语义来自作者实现：

- low-difficulty：LLaMA 1B + GSM8K 风格数据，latent-end token ID 为 524。
- high-difficulty：Qwen Math 7B + Math 数据，latent-end token ID 为 522。
- rollout 使用定制 SGLang、Top-K latent mixture、Gumbel-Softmax、one-sided Gumbel noise 和动态 group filtering。
- 优化使用 GRPO、作者定义的 advantage/reward/KL/FlipGrad 路径；工程观测代码不得增加训练 forward、额外 `loss.backward()` 或额外 `optimizer.step()`。

作者原始说明见 [vendored README](../Latent-GRPO/README.md)。作者训练 shell 是公开超参数第一来源：

- `Latent-GRPO/Latent-GRPO-gsm8k-llama3.sh`
- `Latent-GRPO/Latent-GRPO-math500-qwen.sh`

## 3. 系统架构与训练数据流

正式控制面只有一套 job：

```text
train_latent_grpo.py
→ strict YAML/config validation
→ one ray_direct Python driver
→ one Ray runtime/job
→ 3 FSDP actor workers + customized SGLang rollout
→ validation/metrics-enabled profile 执行 driver-owned worker aggregation 和 append-only metric output
```

不得在外层再套三进程 launcher。三张 GPU 是 Ray/FSDP worker 的资源，不是三个各自启动 trainer 的 outer ranks。

一次训练迭代的主要数据流是：

```text
prompt parquet
→ SGLang latent rollout（每 prompt 生成 n 条轨迹）
→ stable group/trajectory identity
→ reward、dynamic filtering、GRPO advantage
→ FSDP actor forward/backward/optimizer step
→ 若 observer 开启：三 worker scalar sufficient statistics
→ 若 observer 开启：driver authoritative merge 与 append-only Parquet metrics
→ 若 probe 开启：checkpoint 时执行有界 Stage 4 probe 并写 sidecar
```

当前 Low formal 关闭 observer/probe 以减少长跑开销；Low validation 保持全部开启。probe 可以用 `torch.autograd.grad()` 获取 credit，但必须保持 parameter、optimizer state、训练 `.grad`、Python/NumPy/CPU/CUDA RNG 和 module mode。packed actor path 通过可微 `index_copy` 恢复 probe tensor 到 batch/sequence 域，不能为了省事把作者的 `use_remove_padding=true` 改成 false。

## 4. 仓库地图与权威入口

| 路径 | 职责 |
|---|---|
| [`train_latent_grpo.py`](../train_latent_grpo.py) | 唯一外层 Python 训练入口；负责配置、环境门禁、tokenizer latent-end 校验和 launch |
| [`latent_grpo_runner/config.py`](../latent_grpo_runner/config.py) | 严格 profile schema、batch arithmetic、author override 白名单和 resume compatibility hash |
| `latent_grpo_runner/metrics/` | Stage 1–4 指标构造、worker/driver 聚合、append-only storage、checkpoint sidecar |
| `Latent-GRPO/verl-0.4.x/` | 修改后的 VERL actor、FSDP worker 和 Ray trainer |
| `Latent-GRPO/sglang_latent_reasoning_pkg/` | 支持 latent rollout 的定制 SGLang |
| [`configs/author/`](../configs/author/) | low/high 作者机器真值；不得做 3GPU 改值 |
| [`configs/3gpu-final-low.yaml`](../configs/3gpu-final-low.yaml) | 正式 low-difficulty 三卡训练 profile |
| [`configs/3gpu-final-validation.yaml`](../configs/3gpu-final-validation.yaml) | 两个真实 optimizer updates 的低成本三卡 gate，不作为实验结果 |
| [`configs/3gpu-final-high.yaml`](../configs/3gpu-final-high.yaml) | 正式 high-difficulty 三卡训练 profile |
| [`configs/3gpu-final-high-validation.yaml`](../configs/3gpu-final-high-validation.yaml) | 保持 High 作者语义的两个真实 optimizer updates gate |
| [`tools/run_3gpu_preflight.sh`](../tools/run_3gpu_preflight.sh) | Linux/Python/Git/GPU/CUDA/NCCL/BF16/依赖/资产/Ray 门禁 |
| [`tools/run_3gpu_final_validation.sh`](../tools/run_3gpu_final_validation.sh) | 串联 preflight、真实更新、29 指标、probe、checkpoint/resume 和最终报告 |
| [`tools/run_3gpu_training.sh`](../tools/run_3gpu_training.sh) | acceptance PASS 后的正式训练 wrapper，写 manifest 和 resolved config |
| [`tools/validate_3gpu_final.py`](../tools/validate_3gpu_final.py) | 生成 `acceptance.json` 与 `ACCEPTANCE_SUMMARY.md` 的 fail-closed 验收器 |
| [`tests/unit/`](../tests/unit/) | Mac 可运行的配置、数学、聚合、状态保护、脚本和打包回归测试 |

## 5. 配置、作者真值与 3GPU 适配

完整作者字段及 README/论文冲突见 [作者超参数审计](AUTHOR_HYPERPARAMETER_AUDIT.md)。所有差异见 [3GPU 偏差清单](3GPU_HYPERPARAMETER_DEVIATIONS.md)，目标是 `0 silent deviations`。

正式 low profile 是 “3-GPU target-runtime / engineering adaptation”，不是 strict paper reproduction。主要拓扑差异只有：

| 字段 | 作者 low | 正式 3GPU low | 原因 |
|---|---:|---:|---|
| `trainer.n_gpus_per_node` | 8 | 3 | 目标机器恰好三卡 |
| `data.train_batch_size` | 64 | 48 | 保持 rollout `n=8` 时满足三 rank 整除 |
| `actor.ppo_mini_batch_size` | 16 | 12 | 每 rank 32 trajectories，可被当前 actor micro 16 整除 |

LR `1e-6`、prompt/response length、rollout `n=8`、Top-K、Gumbel、noise、KL、reward/advantage、FlipGrad、BF16 和 10 epochs 保持作者值。Low 当前还显式采用 `actor_micro=16`、`logprob_micro=32`、SGLang `gpu_memory_utilization=0.45`，并将 formal 的 metrics/support/checkpoint/credit probes 关闭、`save_freq` 改为 5000、周期 eval 关闭。这些是吞吐、显存和运行开销适配，不得写成作者原始值。路径、GPU ID、seed、experiment name 是显式运行身份，不伪装成作者参数。

High 正式 profile 也只做显式拓扑适配：作者 prompt/mini `32/32` 改为 `12/12`，使 `12×8÷3=32` trajectories/rank，恰好保持作者 `32×8÷8=32` 的每 rank 归一化负载。Qwen Math 7B、1024/4096 长度、KL、gradient checkpointing、actor parameter/optimizer offload、SGLang 12000 token 限制、0.8 memory utilization 和 5 epochs 保持作者值。

Low validation 保持 formal 的 prompt/mini `48/12`、actor/logprob micro `16/32` 和 SGLang `0.45`，只将总步数限制为 2、每步 checkpoint，并重新打开全部 metrics/support/probes。High validation 仍将 prompt/mini `12/12` 缩到 `3/3`。两者都只证明 target runtime 链路，不产生可汇报的正式实验结果。

## 6. 指标、checkpoint 与验收证据

启用 metrics/probe 的 validation 与 High profile 可持久化 29 个核心指标，并额外记录 `train/raw_generated_token_count`。当前 Low formal 为 observer-off，不保证生成这些表和 sidecar：

| 阶段 | 数量 | 目的 |
|---|---:|---|
| Stage 1 | 10 | policy/KL/ratio、长度、step time 等训练核心事实 |
| Stage 2 | 6 | latent surrogate、符号/近零/FlipGrad 等分量统计 |
| Stage 3 | 2 | rollout Top-K 与 old-policy Top-K 的 support/top1 retention；有效 position 必须大于 0 |
| Stage 4 One-sided | 7 | checkpoint-only one-sided delta 分布与状态证据 |
| Stage 4 Credit | 4 | autograd credit concentration、rank/alignment 等 probe 指标 |

三卡验收要求 worker 0/1/2 都产生 rank-local packet，再由 driver 按 sum、sum-of-squares、count、min 和正确 denominator 合并成一个 authoritative row/set。`aggregation_worker_count` 必须为 3，global step 不得重复。

validation checkpoint 至少包含 actor、`data.pt`、metrics sidecar、optimizer/global step identity 和 compatibility hash，并会启动新进程做最小 resume load。显存/性能证据包括每卡 `nvidia-smi` memory/utilization、每 worker PyTorch allocated/reserved 当前值与峰值、两个 step time，以及每 worker probe extra time/peak memory。Low formal 因 probe 关闭，不应被要求生成 validation sidecar。

机器验收看 `artifacts/validation/3gpu-final/acceptance.json`，人类摘要看 `ACCEPTANCE_SUMMARY.md`。详细字段与失败日志位置见 [3GPU 验收清单](3GPU_ACCEPTANCE_CHECKLIST.md)。

## 7. 当前完成状态与未验证边界

当前代码已具备：

- 作者 low/high 真值和 SHA256 漂移检查。
- Low/High 各一套 formal/validation final profile 及严格 topology/override 校验。
- teammate 可直接使用的资产、preflight、final validation 和 formal training wrapper。
- packed/padded Stage 4、三 worker 聚合、allocator/utilization、checkpoint/resume 的实现与 CPU/纯逻辑测试。
- shell syntax、Python compile、profile dry-run 和完整 `tests/unit` 本地回归。

源码包本身仍不能替代目标机证据。Low `16/32/0.45` 在父提交 `8c9ce49` 的三卡 validation 已报告 `29/29`、CUDA RNG、checkpoint/resume 和 `3GPU_FINAL_GATE: PASS`；新提交即使只改测试或文档，也必须现场重新满足 `acceptance.json.git_commit == git rev-parse HEAD`。High 尚未完成同等级目标机验收，因此 High 状态保持 `TARGET_RUNTIME_EXECUTION_REQUIRED`。Kaggle 2×T4 的既有指标验证不能替代任一 final 三卡 gate。

## 8. 接手后的推荐阅读顺序

1. 本文：建立事实边界和目录地图。
2. [3GPU 运行手册](3GPU_RUNBOOK.md)：目标机环境、资产、validation、训练、停止和 resume。
3. [作者超参数审计](AUTHOR_HYPERPARAMETER_AUDIT.md)与[偏差清单](3GPU_HYPERPARAMETER_DEVIATIONS.md)：判断一次参数修改是否合法。
4. [3GPU 验收清单](3GPU_ACCEPTANCE_CHECKLIST.md)：理解 PASS/BLOCKED 证据。
5. [Cairn 三卡专题](../Latent-GRPO/cairn/3gpu-runtime-packaging.md)和最新 [Cairn LOG](../Latent-GRPO/cairn/LOG.md)：了解稳定决策与已解决陷阱。

需要修改代码时，再从 `tests/unit/test_3gpu_final_package.py`、`latent_grpo_runner/config.py`、`latent_grpo_runner/metrics/` 和对应 upstream hook 向内阅读。不要从历史 progress 文档反推当前实现。

## 9. 常见误判与禁止事项

- 不要加入外层三进程 launcher；它会导致三套 Ray/trainer 重复训练。
- 不要把旧 `configs/3gpu-low.yaml` 或 T4 profile 当作者参数真值。
- 不要为了显存或“稳定性”静默修改 rollout `n`、Top-K、Gumbel、reward、advantage、FlipGrad、长度、LR 或 KL。
- 不要把 validation profile 的两步结果当正式实验。
- 不要在 validation profile 中关闭 metrics/support/probe、减少期望 worker 数或放宽 unavailable 字段让 gate 通过；Low formal 在 validation PASS 后按已记录配置关闭 observer/probe，不属于绕过 gate。
- 不要把 worker-local row 当 global row，也不要平均 worker mean；必须合并 sufficient statistics。
- 不要声称本地 Mac 测试证明 CUDA runtime PASS。
- 不要覆盖已有 training output；每个 seed 使用独立目录，并保留 manifest、resolved config 和 Git identity。

## 10. 给接手大模型的推荐提示词

可把本文与下面的提示词一起交给另一个模型：

```text
请先完整阅读 docs/PROJECT_TECHNICAL_HANDOFF.md，再按其中的权威阅读顺序检查相关源码和测试。
当前任务必须保持作者公开算法/采样参数；任何 3GPU topology、吞吐、显存或运行开销变更都必须显式记录并重新绑定 validation acceptance。
不要把本地静态/单元测试当作 CUDA、NCCL、BF16 或三卡 runtime PASS；Low 运行结论以当前 HEAD 对应 acceptance 为准。
开始修改前先报告：当前 Git HEAD/工作树、你认定的 source of truth、受影响的训练语义、计划运行的测试，以及是否需要目标机证据。
```

## 11. 本地资产传参、一键验证与一键正式训练

目标机应先按照 [3GPU 运行手册](3GPU_RUNBOOK.md)安装 Python 3.11/CUDA 依赖并准备模型和 parquet。一次 final validation 调用如下：

```bash
export LOW_MODEL_PATH=/data/models/LLaMA3.2-1B-Instruct-Latent-SFT-Top10
export LOW_TRAIN_DATA=/data/latent-grpo/GSM8k-Aug-oss-dup-all.parquet
export LOW_VAL_DATA=/data/latent-grpo/GSM8k-Aug-test.parquet
export LOW_VALIDATION_ROOT="$PWD/artifacts/validation/3gpu-final-low-seed17"

bash tools/run_3gpu_final_validation.sh \
  --config configs/3gpu-final-validation.yaml \
  --model-path "$LOW_MODEL_PATH" \
  --train-data "$LOW_TRAIN_DATA" \
  --val-data "$LOW_VAL_DATA" \
  --output-root "$LOW_VALIDATION_ROOT" \
  --gpus 0,1,2 \
  --seed 17
```

只有上述命令生成的 `acceptance.json` 满足 `final_gate == PASS` 后才能启动正式训练。下面是一个可复制的一键训练示例；wrapper 会拒绝非 PASS acceptance，也会拒绝覆盖已经存在的 `TRAIN_OUTPUT`：

```bash
export LOW_TRAIN_OUTPUT="$PWD/artifacts/runs/latent-grpo-gsm8k-seed17"

bash tools/run_3gpu_training.sh \
  --config configs/3gpu-final-low.yaml \
  --model-path "$LOW_MODEL_PATH" \
  --train-data "$LOW_TRAIN_DATA" \
  --val-data "$LOW_VAL_DATA" \
  --output-root "$LOW_TRAIN_OUTPUT" \
  --gpus 0,1,2 \
  --seed 17 \
  --acceptance-report "$LOW_VALIDATION_ROOT/acceptance.json"
```

High 使用完全对称的本地传参；Hugging Face 页面 `https://huggingface.co/DJCheng/Qwen2.5-Math-7B-Latent-SFT-4k-Top10` 仅是下载来源，不能直接传给 `--model-path`：

```bash
export HIGH_MODEL_PATH=/data/models/Qwen2.5-Math-7B-Latent-SFT-4k-Top10
export HIGH_TRAIN_DATA=/data/latent-grpo/DAPO-Math-17k-en-train.parquet
export HIGH_VAL_DATA=/data/latent-grpo/Math-500-test.parquet
export HIGH_VALIDATION_ROOT="$PWD/artifacts/validation/3gpu-final-high-seed17"
export HIGH_TRAIN_OUTPUT="$PWD/artifacts/runs/latent-grpo-math500-seed17"

bash tools/run_3gpu_final_validation.sh \
  --config configs/3gpu-final-high-validation.yaml \
  --model-path "$HIGH_MODEL_PATH" \
  --train-data "$HIGH_TRAIN_DATA" \
  --val-data "$HIGH_VAL_DATA" \
  --output-root "$HIGH_VALIDATION_ROOT" \
  --gpus 0,1,2 \
  --seed 17

bash tools/run_3gpu_training.sh \
  --config configs/3gpu-final-high.yaml \
  --model-path "$HIGH_MODEL_PATH" \
  --train-data "$HIGH_TRAIN_DATA" \
  --val-data "$HIGH_VAL_DATA" \
  --output-root "$HIGH_TRAIN_OUTPUT" \
  --gpus 0,1,2 \
  --seed 17 \
  --acceptance-report "$HIGH_VALIDATION_ROOT/acceptance.json"
```
