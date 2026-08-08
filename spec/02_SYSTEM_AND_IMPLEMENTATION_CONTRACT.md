# Latent-GRPO 训练系统与硬件实现契约


## 0. 路径约定

Codex 必须从项目根目录工作。固定路径为：

```text
./Latent-GRPO       # 作者原始仓库
./spec              # 全部任务规范 Markdown
```

引用任何规范文件时都使用 `./spec/<filename>.md`，不得到 `./Latent-GRPO/spec` 或其他目录中查找。

---

## 1. 目标系统

本任务要产生的是一套**专用于 Latent-GRPO** 的可执行 Python 训练系统，而不只是一个把作者 shell 命令包起来的薄脚本。

建议结构如下，Codex 可根据上游仓库调整，但必须保持职责分离：

```text
latent_grpo_runner/
├── __init__.py
├── config.py
├── environment.py
├── upstream_adapter.py
├── trainer.py
├── distributed.py
├── checkpointing.py
├── evaluation.py
├── metrics/
│   ├── events.py
│   ├── aggregators.py
│   ├── masks.py
│   ├── schemas.py
│   ├── storage.py
│   ├── stage1.py
│   ├── stage2.py
│   ├── support.py
│   └── probe.py
└── validation/
    ├── runtime_probe.py
    └── output_validator.py
```

用户入口：

```text
train_latent_grpo.py
```

作者仓库：

```text
./Latent-GRPO
```

新代码优先放在作者仓库之外的独立目录，通过 adapter 使用上游代码。若上游运行机制要求代码位于仓库内，可建立清晰命名的子包并保留最小 patch。

---

## 2. 目标硬件与运行表面

锁定的主要环境：

```text
OS: Linux
IDE/remote surface: VSCode Remote
Container: none
CUDA GPUs: 3
Approximate VRAM per GPU: 46 GB
Distributed launcher: torchrun
```

启动示例：

```bash
torchrun --standalone --nproc_per_node=3 \
  train_latent_grpo.py \
  --config configs/3gpu-low.yaml
```

不得把 Docker、Kubernetes、Slurm 或 8-GPU 环境作为三卡训练的前置条件。

### 2.1 启动前 runtime probe

必须探测并保存：

```text
hostname_redacted
os
python_version
python_executable_fingerprint
torch_version
cuda_runtime_version
cuda_driver_version
cudnn_version
nccl_version
gpu_count
gpu_names
gpu_total_memory_bytes
gpu_compute_capabilities
bf16_supported
distributed_backend
world_size
local_world_size
rank
local_rank
visible_devices
disk_free_bytes
upstream_repo_commit
workspace_fingerprint
```

保存到：

```text
platform_config_snapshot.json
```

敏感路径、用户名、主机名和密钥必须脱敏或哈希。

若检测到不是 3 GPU 或单卡显存明显低于目标配置要求：

- 默认在 `3gpu-low`/`3gpu-high-smoke` 下拒绝启动；
- `--allow-hardware-mismatch` 可允许工程 smoke；
- mismatch 必须进入 `run_config.json` 和日志；
- 不得因此降低配置后仍宣称运行的是原 profile。

---

## 3. 配置 Profile

## 3.1 `smoke`

目的：

- 验证 import、数据、模型、rollout、reward、advantage、update、checkpoint、日志写盘；
- 尽量短；
- 可使用 1 GPU；
- 不代表算法效果。

最低要求：

```text
max_steps: 1-3
rollout_n: 小
prompt_count: 小
max_response_length: 小
max_latent_length: 小
support: 可开启一次
checkpoint_probe: 可开启一次
credit_probe: 默认关闭
```

## 3.2 `3gpu-low`

目的：

- 在 3×约 46 GB GPU 上运行低难度 Latent-GRPO；
- 是主要目标设备配置；
- 显式适配 batch、mini-batch、micro-batch 和并行参数；
- 不标为论文严格复现。

低难度模型的默认候选为：

```text
DJCheng/LLaMA3.2-1B-Instruct-Latent-SFT-Top10
```

配置中应使用 Hugging Face model ID，不把下载逻辑散落在训练代码中。启动时先检查配置指定的本地模型目录和 cache；只有缺失时才由 rank 0 下载，其他 rank 在 barrier 后读取。

必须支持：

```text
BF16（硬件支持时）
gradient checkpointing（配置化）
remove padding（上游支持时）
合理的 micro-batch
3-rank 分布式聚合
断点续训
```

## 3.3 `3gpu-high-smoke`

目的：

- 在三卡设备上验证高难度/7B 模型的完整代码链；
- 缩小 batch、rollout、最大 response、最大 latent 和 step；
- 不用于性能或论文结果。

高难度模型的默认候选为：

```text
DJCheng/Qwen2.5-Math-7B-Latent-SFT-4k-Top10
```

同样必须先检查本地路径和 cache，避免每次启动重复下载。

若显存仍不足，应：

1. 输出可复现的 OOM 诊断；
2. 记录实际峰值显存；
3. 提供一组更小但不改变算法类型的 smoke 参数；
4. 不静默切换到 CPU offload、量化或不同模型；
5. 若采用 offload/量化，必须作为新 profile 明确命名。

---

## 4. 上游仓库集成

## 4.1 导入与路径

入口必须可靠找到：

```text
./Latent-GRPO
```

不要依赖用户当前 shell 已经把仓库加入 `PYTHONPATH`。可以：

- 以 editable package 安装；
- 使用明确的 adapter/import bootstrap；
- 或在配置中指定 `upstream_repo_path`。

不得硬编码用户绝对路径。

启动时必须验证：

- 路径存在；
- 关键 Python 包/入口存在；
- Git 状态和 commit 可记录；
- 上游依赖满足；
- 配置与模型兼容。

## 4.2 上游代码修改

默认不修改上游算法文件。

必须修改时：

```text
patches/
docs/upstream_changes.md
```

每个修改写明：

```text
patch_id
upstream_file
reason
algorithmic_effect
logging_only_or_training_effect
test
rollback
```

任何声称“logging-only”的 patch 都必须通过关闭日志时的等价性测试。

---

## 5. 训练步语义

必须维护两个独立计数：

```text
global_step
optimizer_step
```

定义：

- `global_step`：一次完整训练外层迭代所属 step；
- `optimizer_step`：成功执行参数更新的累计次数；
- 梯度累积、跳过更新、overflow 或失败更新不能被错误计数；
- resume 后必须从 checkpoint 元数据恢复；
- checkpoint eval/probe 的 `checkpoint_step` 与执行上下文 `global_step` 分开。

所有 metric event 在创建时绑定不可变的 step context，写盘线程不得事后读取“当前 step”补字段。

---

## 6. 分布式要求

必须支持：

```text
torchrun --nproc_per_node=3
```

若作者仓库实际使用 Ray、FSDP、Megatron、vLLM、SGLang 或其他运行时，Codex 应通过上游审计决定适配方式，但用户入口仍需清晰。

### 6.1 聚合

worker/rank 只返回充分统计量：

```text
sum
sum_sq
count
nan_count
masked_count
min
max
numerator_count
```

只有 driver/rank 0 写权威输出。

禁止：

- 每个 rank 写同一个文件；
- 简单平均 worker mean；
- all-gather 完整 logits、hidden states 或梯度；
- 为日志额外同步全量 Tensor；
- 无界队列累积日志事件。

### 6.2 错误传播

任一 rank 发生致命训练错误时：

- 其他 rank 应停止；
- driver 尝试 flush 已完成分片；
- 写 `run_status.json`；
- 保留 traceback；
- 不把失败 run 标为 completed。

日志 family 的非致命不可用，不应导致训练中断，除非该字段是算法本身必需。

---

## 7. 模型、数据与缓存

模型和数据路径必须配置化。

推荐：

```text
cache_root: <workspace_or_user_configured_cache>
model_cache_dir: <cache_root>/models
dataset_cache_dir: <cache_root>/datasets
```

要求：

- 已存在完整模型时不重复下载；
- 不在每个 rank 重复下载；
- rank 0 完成缓存准备后 barrier；
- 下载失败给出准确错误；
- 离线模式可使用已有 cache；
- 下载 URL/模型 ID 写入配置或文档；
- 不把大模型保存到输出指标目录；
- 不记录访问 token。

---

## 8. checkpoint 与 resume

checkpoint 至少包含或可恢复：

```text
model state
optimizer state
scheduler state
global_step
optimizer_step
RNG states
sampler/generator states（上游需要时）
profile/config hash
schema version
upstream commit
```

resume 要求：

- 检查 profile 与关键算法配置兼容；
- 已提交 Parquet part 不重复写；
- 主键冲突可检测；
- `is_resume_run`、`resume_from_step` 正确；
- 不因重启改变 trajectory ID 的局部规则；
- probe/eval 可针对旧 checkpoint 重跑并保留 `checkpoint_step`。

---

## 9. 性能边界

日志默认不得：

- 每训练 step 做额外 full forward；
- 每训练 step执行 credit autograd；
- 保存 full vocabulary Tensor；
- 在 Python 中逐 token 同步 GPU；
- 每条 metric 都立即 fsync；
- 长期保留计算图。

应使用：

- GPU 局部 reduce；
- 批量搬运小型 detached 统计量；
- 有界队列；
- 批量 flush；
- checkpoint-only probe；
- Support 的低频触发；
- definition JSON 避免动态行重复大字段。

必须分别测量：

```text
metrics_compute_time
metrics_write_time
support_extra_time_seconds
probe_extra_time_seconds
support_cache_peak_bytes
probe_peak_memory_bytes
```

---

## 10. 训练语义不变性

至少验证：

1. 相同 seed、相同配置、关闭日志与开启 Stage 1/2 日志时，首个可比 step 的 loss/reward/参数更新在允许误差内一致；
2. Support 关闭与开启的非 Support step 训练语义一致；
3. checkpoint probe 前后参数、optimizer state、正常 `.grad` 和 RNG 恢复；
4. 写盘失败的非关键日志可降级而不篡改训练数据；
5. 指标采集不插入隐式 optimizer step；
6. 不因 logger 触发额外随机采样。

无法证明完全 bitwise 相等时，必须解释随机源与数值误差，不能直接声称“无影响”。
