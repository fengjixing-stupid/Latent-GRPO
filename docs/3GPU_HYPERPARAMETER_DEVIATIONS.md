# 3GPU 超参数偏差

基线：`53438ec07b804ebd1b670d6fe118199798350505`。本文件保证 `0 silent deviations`。

正式 profile `configs/3gpu-final-low.yaml` 与 `configs/3gpu-final-high.yaml` 都是 “3-GPU target-runtime / engineering adaptation”，不是 strict paper reproduction。所有已发布的 AUTHOR-FROZEN 算法、Gumbel、sampling、reward/advantage、FlipGrad、长度、LR、KL、offload、epochs 和 probe 语义保持对应作者值。

## 正式训练偏差

| parameter | author value | 3GPU value | change required? | reason | semantic impact |
|---|---:|---:|---|---|---|
| `trainer.n_gpus_per_node` | 8 | 3 | yes | target 恰好三卡 | FSDP world size 改变；工程适配 |
| `data.train_batch_size` | 64 | 48 | yes | 64×8 trajectories 不能被 world size 3 整除；48 可整除且保持 rollout.n=8 | 每次 update 的 prompt/trajectory 总量下降 |
| `actor.ppo_mini_batch_size` | 16 | 12 | yes | 16 不能在 3 DP ranks 上均分；12 可按每 rank 4、micro 2 执行 | mini-batch 统计量改变 |
| `CUDA_VISIBLE_DEVICES` | 0..7 default | operator 传入三个 ID | yes | 显式选择目标三卡 | 只改变 placement |
| model/data/output/cache paths | shell defaults/placeholders | CLI 实际路径 | yes | 机器资产位置不同 | 无算法影响；identity 写入 manifest |
| `trainer.experiment_name` | `latent-grpo-gsm8k-llama3` | 后缀 `-3gpu` | yes | 防止跟 8GPU run 混淆 | 仅跟踪标识 |
| `training.seed` | AUTHOR_NOT_PUBLISHED | 17 或 CLI seed | yes | 多 seed run identity | 影响随机轨迹，必须逐 run 记录 |
| `actor.ppo_epochs` | AUTHOR_NOT_PUBLISHED | 1 | no topology need | vendored upstream default 1；runner schema显式化 | 不声称作者值 |
| actor sequence parallel size | AUTHOR_NOT_PUBLISHED | 1 | no topology need | vendored upstream default；TP/SP=1 target baseline | 不声称作者值 |
| ref param offload | AUTHOR_NOT_PUBLISHED | false | no topology need | vendored upstream default；runner显式化 | 不声称作者值 |
| latent marker string | AUTHOR_NOT_PUBLISHED | `</think>`，首 ID 必须 524 | validation need | 运行时验证作者 ID 与目标 tokenizer | 不改作者 token ID |
| target min VRAM | AUTHOR_NOT_PUBLISHED | 40 GB/GPU | gate only | 目标机器约束 | 不进入训练 objective |
| target Python / free disk | AUTHOR_NOT_PUBLISHED | Python 3.11 / >=20 GiB validation filesystem | gate only | 固定运行时 ABI 并保证短 gate checkpoint 空间 | 不进入训练 objective |
| formal `training.max_steps` schema field | AUTHOR_NOT_PUBLISHED | 2（不发给 upstream） | schema only | parser 要求正整数；formal profile 只发 `total_epochs=10` | 无 runtime 影响 |

`train_batch_size=64` 和 `ppo_mini_batch_size=16` 不能在当前 runner 的三 rank arithmetic 下严格保持，因此此运行不得称为论文严格复现。

## High 正式训练偏差

High 作者真值来自 `configs/author/latent_grpo_math_qwen.yaml`，三卡正式配置为 `configs/3gpu-final-high.yaml`。

| parameter | author value | 3GPU value | change required? | reason | semantic impact |
|---|---:|---:|---|---|---|
| `trainer.n_gpus_per_node` | 8 | 3 | yes | target 恰好三卡 | FSDP world size 改变；工程适配 |
| `data.train_batch_size` | 32 | 12 | yes | 保持 `n=8` 并满足三 rank 整除 | 全局 prompt 数下降；每 rank trajectories 仍为 32 |
| `actor.ppo_mini_batch_size` | 32 | 12 | yes | mini batch 与三卡 formal prompt batch 对齐 | 全局 mini batch 下降；每 rank trajectories 仍为 32 |
| model/data/output paths | shell placeholder | CLI 本地绝对路径 | yes | 机器资产位置不同 | 无算法影响；identity 写入 manifest |
| `trainer.experiment_name` | `latent-grpo-math500-qwen` | 后缀 `-3gpu` | yes | 防止与 8GPU run 混淆 | 仅跟踪标识 |
| `training.seed` | AUTHOR_NOT_PUBLISHED | 17 或 CLI seed | yes | run identity | 影响随机轨迹，逐 run 记录 |
| ref param offload | AUTHOR_NOT_PUBLISHED | false | no topology need | 作者 shell 未设置；runner 显式采用 upstream 默认 | 不声称作者值 |
| formal `training.max_steps` schema field | AUTHOR_NOT_PUBLISHED | 2（不发给 upstream） | schema only | formal profile 只发 `total_epochs=5` | 无 runtime 影响 |

High 保持作者的 Qwen Math 7B、1024/4096 长度、`n=8`、actor micro batch 1、gradient checkpointing、actor parameter/optimizer offload、KL loss、20480 token budget、SGLang 12000 长度和 memory utilization 0.8。

## Final runtime validation-only 偏差

这些值用于 `configs/3gpu-final-validation.yaml`（Low）和 `configs/3gpu-final-high-validation.yaml`（High）的两步 gate，不产生正式实验结果；算法语义、rollout.n=8、Top-K/Gumbel、reward、advantage、FlipGrad 和 sampling 值保持各自正式 profile。

| parameter | formal 3GPU value | validation value | reason | semantic impact |
|---|---:|---:|---|---|
| `data.train_batch_size` | 48 | 3 | 最小三 rank 真实 update | 仅 gate 成本；结果不可报告 |
| `actor.ppo_mini_batch_size` | 12 | 3 | 与 validation batch 对齐 | 仅 gate 成本 |
| `trainer.total_training_steps` | omitted | 2 | 强制两个真实 update 后停止 | validation 控制 |
| `trainer.total_epochs` | 10 | 1 | max-step 为主，防止长跑 | validation 控制 |
| `trainer.val_before_train` | true | false | 缩短 runtime gate | validation 控制 |
| `trainer.save_freq` | 40 | 1 | 每步触发 Stage 4/checkpoint probe | 提高 probe 频率，不改 probe 算法 |
| `trainer.test_freq` | 10 | -1 | gate 不做完整 eval | validation 控制 |
| `trainer.logger` | console+wandb | console | 不上传短 gate | 仅日志后端 |
| output root | per-seed formal run | `artifacts/validation/3gpu-final/run` | 隔离验证 | 不污染正式输出 |

High validation 使用同一类控制偏差：formal prompt/mini `12/12` 缩到 `3/3`，`total_training_steps=2`、`total_epochs=1`、`val_before_train=false`、`save_freq=1`、`test_freq=-1`、console-only；其模型、数据、1024/4096 长度、KL/offload 和 0.8 memory utilization 不变。Low 与 High acceptance 不可交叉作为正式训练前置证据。

正式 profile 始终启用 `metrics_enabled=true`、`support_enabled=true`、`checkpoint_probe_enabled=true`、`credit_probe_enabled=true`。Stage 4 仍是 checkpoint-only。
