# Latent-GRPO 可观测训练系统实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: 下一阶段使用 `superpowers:subagent-driven-development` 或 `superpowers:executing-plans` 按任务逐项实现；每个任务先写失败测试，再写最小实现。本文件只做设计，不代表任何训练代码已实现或验证。

**Goal:** 在不改变作者 Latent-GRPO 算法语义的前提下，提供单卡 smoke、3×约 46 GB GPU 运行、完整 target-variable 观测、append-only 存储、checkpoint/probe 与可恢复验证链。

**Architecture:** 新代码位于作者仓库外的 `latent_grpo_runner/`，由一个稳定的 Python 入口完成配置、环境探测和启动协调；算法仍由 vendored verl + 定制 SGLang 执行。由于上游训练循环缺少稳定 observer 接口，下一阶段只对作者仓库施加可审查的最小 patch，把稳定 ID、统计包、step/update 结果和 checkpoint/eval 事件送入新指标层；权威写盘发生在 Ray coordinator/rank 0。

**Tech Stack:** Python 3.11（作者基线，目标机复核）、PyTorch/FSDP、Ray、vendored verl 0.4.x、vendored SGLang 0.4.6.post1 派生版、Hydra/OmegaConf、PyArrow/Parquet、pytest；CUDA/NCCL/驱动作为系统运行时，不写入 `requirements.txt`。

## Global Constraints

- 固定路径是 `./Latent-GRPO` 与 `./spec`，不得假设当前目录就是作者仓库。
- Linux + VSCode Remote、非 Docker、3×约 46 GB NVIDIA GPU；三卡默认由 `python train_latent_grpo.py --config configs/3gpu-low.yaml` 启动单一 Ray driver，`torchrun_control` 只是显式兼容模式。
- `./spec/target_variables.md` 对字段、mask、count、时间点和禁止项具有最高优先级。
- 不实现 Classic-GRPO；latent token、noisy top-K mixture、one-sided noise、FlipGrad 与 Optimal Correct Path 必须复用真实链路。
- driver/rank 0 是唯一权威 writer；worker 只返回 sufficient statistics 或严格受限的 probe 样本。
- 不额外执行每 step full-vocabulary forward，不持久化 full logits/hidden states/gradient/长期计算图。
- Stage 4 credit 默认关闭；Gumbel diagnostics 仅独立 diagnostic/smoke；不记录 `train/gradient_norm`，不创建 checkpoint 级 `eval_metrics`。
- JSON 仅用于小型静态/状态数据；动态表是 append-only Parquet dataset；原子提交、主键去重、resume 安全。
- 本计划中的版本约束只有在作者文件或官方兼容证据支持时才能进入安装文件；其余必须由目标机 runtime probe 决定。
- `smoke`、`3gpu-low`、`3gpu-high-smoke` 都不得声明为论文严格复现；硬件不匹配默认拒绝，显式 override 只允许工程 smoke。

---

## 1. 文件与职责边界

计划新增：

```text
train_latent_grpo.py                    # 唯一用户入口；参数解析与 launcher 协调
latent_grpo_runner/
  config.py                             # profile 合并、强类型校验、config hash
  environment.py                        # 脱敏 runtime/dependency/tensor probe
  upstream_adapter.py                   # Hydra override 与上游 observer 接口
  trainer.py                            # run 生命周期、状态、失败传播
  distributed.py                        # 默认单 Ray driver 与可选 torchrun 控制兼容
  checkpointing.py                      # sidecar 元数据、resume 兼容检查
  evaluation.py                         # eval raw facts 与 checkpoint 调度
  metrics/
    events.py                           # immutable StepContext 与 typed events
    aggregators.py                      # sum/sum_sq/count 等充分统计
    masks.py                            # response/latent/component 对齐规则
    schemas.py                          # 表 schema、版本、主键
    storage.py                          # JSON 原子写与 Parquet part commit
    stage1.py                           # 基础训练、group/eval raw facts
    stage2.py                           # mixture/zero-adv/signal 与 diagnostic
    support.py                          # rollout↔pre-update Support
    probe.py                            # checkpoint one-sided/credit probe
  validation/
    runtime_probe.py                    # 目标机 probe 与 profile gate
    output_validator.py                 # schema/PK/completeness validator
configs/{smoke,3gpu-low,3gpu-high-smoke}.yaml
scripts/{run_smoke,run_3gpu_low,run_3gpu_high_smoke}.sh
scripts/{inspect_environment,validate_outputs}.py
requirements.txt
requirements/
  runtime-core.txt
  runtime-sglang.txt
  metrics.txt
  tracking.txt
  reward-math.txt
  vllm-optional.txt
  test.txt
  docs.txt
constraints/linux-cu124-py311.txt         # runtime probe 后才固化的目标机组合
tests/{unit,integration,fixtures}/...
patches/latent_grpo_observer.patch
docs/upstream_changes.md
```

计划最小修改的作者文件（精确行号在实施时以 commit `c0994fb781a2d180662bb522d8ff3e8638dcf56d` 重新定位）：

```text
Latent-GRPO/verl-0.4.x/verl/trainer/ppo/ray_trainer.py
Latent-GRPO/verl-0.4.x/verl/trainer/ppo/core_algos.py
Latent-GRPO/verl-0.4.x/verl/workers/actor/dp_actor.py
Latent-GRPO/verl-0.4.x/verl/workers/fsdp_workers.py
Latent-GRPO/sglang_latent_reasoning_pkg/python/sglang/srt/layers/sampler.py
```

修改原则：只增加稳定 observer/hook、返回小型统计包、修复与观测直接相关且已有证据的上游缺陷；不复制整段 `fit()`，不 monkey patch 私有方法。每项改动同时生成 `patches/` 补丁、`docs/upstream_changes.md` 条目和 logging-off 等价性测试。

## 2. 核心接口

```python
@dataclass(frozen=True)
class StepContext:
    profile_name: str
    seed: int
    global_step: int
    optimizer_step: int
    observation_phase: str
    is_resume_run: bool
    resume_from_step: int | None

@dataclass(frozen=True)
class SufficientStats:
    sum: float
    sum_sq: float
    count: int
    nan_count: int
    masked_count: int
    minimum: float | None
    maximum: float | None
    numerator_count: int | None

class TrainingObserver(Protocol):
    def on_post_rollout(self, context: StepContext, batch_view: "RolloutView") -> None: ...
    def on_post_advantage_pre_update(self, context: StepContext, batch_view: "AdvantageView") -> None: ...
    def on_post_update(self, context: StepContext, update: "UpdateResult") -> None: ...
    def on_checkpoint(self, context: StepContext, checkpoint: "CheckpointEvent") -> None: ...

@dataclass(frozen=True)
class UpdateResult:
    successful_optimizer_steps: int
    skipped_nonfinite_steps: int
    worker_stats: Mapping[str, SufficientStats]
```

`StepContext` 在事件创建时冻结；writer 不得读取训练器的可变“当前 step”。`RolloutView`/`AdvantageView` 只暴露计算所需字段，不把 full-vocabulary Tensor 或计算图放入队列。

## 3. 任务分解

### Task 1：配置、依赖分层与 dry-run

**Files:** `train_latent_grpo.py`, `latent_grpo_runner/config.py`, `configs/*.yaml`, `requirements.txt`, `requirements/*.txt`, `constraints/*.txt`, `tests/unit/test_config.py`

**Produces:** `ResolvedConfig`, config hash、profile/hardware gate、作者 Hydra overrides。

- [ ] 写失败测试：三个 profile 可解析；未知字段、非 3-GPU target、batch/DP 不整除、paper profile 冒充均拒绝。
- [ ] 实现 CLI 规定的全部参数，并保证 `--dry-run --validate-config` 不加载模型、不初始化 CUDA/Ray。
- [ ] 生成 author-Hydra override 预览与脱敏 resolved command。
- [ ] 运行 `python -m pytest tests/unit/test_config.py -q`，预期全部通过。

### Task 2：环境与依赖 runtime probe

**Files:** `latent_grpo_runner/environment.py`, `latent_grpo_runner/validation/runtime_probe.py`, `scripts/inspect_environment.py`, `tests/unit/test_environment.py`

**Produces:** `platform_config_snapshot.json` 的完整字段与 `dependency_check_status`。

- [ ] 写失败测试：主机名/用户名/绝对私有路径被哈希；缺 GPU、缺 BF16、驱动与 wheel 不兼容时输出稳定 reason。
- [ ] 只用只读 API 探测 Python、torch build/runtime CUDA、driver、cuDNN、NCCL、GPU memory/compute capability、磁盘和仓库 commit。
- [ ] 增加一个最小 tensor/NCCL probe 子命令；默认 dry-run 不执行分布式 collective。
- [ ] 运行 `python -m pytest tests/unit/test_environment.py -q`。

### Task 3：Ray direct 上游启动与可选 torchrun 兼容封装

**Files:** `latent_grpo_runner/distributed.py`, `latent_grpo_runner/trainer.py`, `latent_grpo_runner/upstream_adapter.py`, `tests/integration/test_launcher.py`

**Consumes:** `ResolvedConfig`、runtime probe。

**Produces:** 单一 Ray coordinator、3 GPU resource pool、跨 wrapper rank 的退出状态。

- [ ] 写失败测试：在模拟 `WORLD_SIZE=3` 下只有 rank 0 启动上游；rank 1/2 不初始化 CUDA、不写输出；rank 0 失败传播非零状态。
- [ ] wrapper 用 CPU/Gloo 控制面同步；rank 0 子进程移除 torchrun 的 `RANK/LOCAL_RANK/WORLD_SIZE` 后启动单一 Ray driver，避免上游误建三套 Ray 作业。
- [ ] 上游配置固定 `trainer.n_gpus_per_node=3`, `nnodes=1`, SGLang TP=1；Ray/FSDP 是实际训练后端。
- [ ] 运行 CPU 模拟 launcher 测试；目标机再运行三 rank 初始化测试。

### Task 4：最小上游 observer patch 与稳定 ID

**Files:** 上述 5 个作者文件、`patches/latent_grpo_observer.patch`, `docs/upstream_changes.md`, `tests/unit/test_upstream_observer.py`

**Produces:** `TrainingObserver` 回调、stable `trajectory_id`、真实 `UpdateResult`。

- [ ] 写失败测试：`trajectory_id` 在 repeat 后且 balance/filter/select 前创建，重排后保持不变，同 group 唯一。
- [ ] 给 `RayPPOTrainer` 增加显式 observer 工厂与阶段事件，不复制或重写算法主体。
- [ ] 令 actor optimizer 明确返回是否真正执行 `optimizer.step()`；scheduler 与 `optimizer_step` 只按已证实语义推进，跳过非有限梯度不误计数。
- [ ] 修复 `dp_actor.py` 中 `topk_logits` 错误从 ID 列表拼接的问题，并用回归测试证明返回内容/shape。
- [ ] 将 latent-end token 判断改为配置值并覆盖 LLaMA 524/Qwen 522。
- [ ] 关闭 observer 后运行同 seed 首步等价性测试；保存补丁和回滚说明。

### Task 5：Schema、事件与充分统计

**Files:** `metrics/events.py`, `metrics/aggregators.py`, `metrics/schemas.py`, `tests/unit/test_{events,aggregators,schemas}.py`

**Produces:** 全部 target fields 的 schema manifest 和 worker merge API。

- [ ] 从 RTM 生成/校验 schema；禁止字段扫描必须失败。
- [ ] 测试 sum/sum_sq/count、NaN、empty mask、rate、不同 mask 独立 count，禁止 worker-mean 平均。
- [ ] availability=false 时值为 null、reason 稳定；credit/diagnostic 默认关闭但 schema 仍存在。
- [ ] 运行对应 unit tests，并生成 `validation/target_variable_coverage.json`，要求 `missing_fields=[]`。

### Task 6：Stage 1/2 训练指标与 group raw facts

**Files:** `metrics/masks.py`, `metrics/stage1.py`, `metrics/stage2.py`, `tests/unit/test_stage{1,2}.py`

**Produces:** `train_step_metrics`, `train_group_metrics`, 独立 `gumbel_diagnostics`。

- [ ] 测试 10 个 Stage 1 指标、6 个 Stage 2 指标、所有 count/definition/availability 字段。
- [ ] generated-token count 从最终训练 rollout trajectory 的整数长度求和；保留 overlong、排除 discarded retry/probe。
- [ ] correct/non_correct 严格二分类；overlong 是可重叠布尔；Optimal Correct winner 复用正 first-step advantage + mean old log-prob。
- [ ] noisy mixture 由实际 perturb-score/temperature 得到与 latent embedding 相同的权重；不得计算 clean mixture。
- [ ] Gumbel diagnostic 仅独立模式在真实采样/one-sided transform 处局部 reduce，正式训练路径不启用。

### Task 7：Support

**Files:** `metrics/support.py`, `tests/unit/test_support.py`

**Consumes:** rollout `rollout_topk_ids`、同一次 old-log-prob forward 已返回的 `old_topk_indices`、内存 winner。

**Produces:** `support_definition.json`, `support_metrics`, `support_benchmark_metrics`。

- [ ] 测试 same group/trajectory/position/K 的严格对齐；任何 shape/K/order 错误整 family unavailable，禁止截断或广播。
- [ ] top1 retention 测试为 rollout top1 是否属于 pre-update set，而非 top1 相等。
- [ ] 按 position count 加权；trajectory 与 benchmark count 分开。
- [ ] 证明 Support 不新增 forward、不改变参数/RNG/optimizer/.grad，且不计入训练 token/length。

### Task 8：Checkpoint eval raw facts

**Files:** `evaluation.py`, `metrics/stage1.py`, `tests/integration/test_checkpoint_eval.py`

**Produces:** dataset manifest、`eval_question_results`, `eval_clean_topk`。

- [ ] 每 checkpoint×question×generation 保存一行，包括失败；generation IDs 完整。
- [ ] clean top-K 只转存同一 eval forward 已有的 IDs/probs；不可用时 null+reason，不做额外 softmax/top-k/forward。
- [ ] 不创建 checkpoint 汇总 `eval_metrics`；验证器从 raw facts 可恢复 accuracy/length/valid rate。

### Task 9：Checkpoint one-sided/credit probe

**Files:** `metrics/probe.py`, `checkpointing.py`, `tests/unit/test_probe.py`, `tests/integration/test_probe_no_mutation.py`

**Produces:** `probe_definition.json`, `probe_metrics`, `probe_benchmark_metrics`。

- [ ] 固定 prompts、顺序、generation/Gumbel seed 与上限；使用独立 RNG 并恢复 Python/NumPy/CPU/CUDA RNG。
- [ ] one-sided 复用训练 reward/group/advantage/mask/Optimal Correct Path 与真实 `perturbed_score - component_log_prob`。
- [ ] 测试 7 个 one-sided 指标及 Delta/FlipGrad 独立 count/p05 算法。
- [ ] credit 默认写 unavailable/disabled_by_config 且不执行 autograd；启用路径只允许一次局部 `autograd.grad`，先通过符号、tie/constant/zero-gradient、显存和 no-.grad-pollution 测试。
- [ ] probe 前后比较参数、optimizer、scheduler、`.grad`、RNG、steps、cumulative tokens。

### Task 10：原子存储、resume 与 checkpoint sidecar

**Files:** `metrics/storage.py`, `checkpointing.py`, `tests/unit/test_storage.py`, `tests/integration/test_resume.py`

**Produces:** JSON 原子文件、Parquet parts/manifests、writer checkpoint、resume sidecar。

- [ ] part 临时写→关闭→可读/schema/row-count 校验→rename→manifest→writer checkpoint。
- [ ] 测试 crash tmp、manifest rebuild、duplicate PK、future records、schema mismatch、resume append 与 quarantine。
- [ ] checkpoint sidecar 保存 global/optimizer step、config/schema/upstream hash、writer manifests；兼容上游 model/optimizer/scheduler/RNG shards。
- [ ] 执行 2-step→resume 到 4-step 与连续 4-step 对照。

### Task 11：输出验证器与安全检查

**Files:** `validation/output_validator.py`, `scripts/validate_outputs.py`, `tests/unit/test_output_validator.py`

- [ ] 覆盖 `03_METRICS_STORAGE_CONTRACT` 第 12 节全部检查和禁止字段。
- [ ] 每种损坏 fixture 断言非零退出；成功 fixture 生成三份 validation JSON。
- [ ] 静态扫描硬编码私有绝对路径、secret 样式、CSV 权威表与 full-tensor 字段。

### Task 12：按风险递增验收

**Files:** `scripts/run_*.sh`, `docs/operator_runbook.md`, `docs/progress.md`, `FINAL_IMPLEMENTATION_REPORT.md`

- [ ] `python -m compileall train_latent_grpo.py latent_grpo_runner scripts tests`。
- [ ] `python -m pytest tests/unit -q`，再运行 CPU/synthetic integration。
- [ ] `python train_latent_grpo.py --config configs/smoke.yaml --dry-run --validate-config`。
- [ ] 单卡 1–2 step smoke、validator、checkpoint/probe no-mutation。
- [ ] 三卡 rank-init only，再运行 `3gpu-low --max-steps 2`；记录峰值显存/时间/写盘开销。
- [ ] 仅在 low 通过后运行 `3gpu-high-smoke --max-steps 1`；OOM 必须保留诊断，不静默换模型/量化/offload profile。
- [ ] 独立 reviewer 对照 RTM，blocker 清零后才可写 completed。

## 4. 三卡 profile 初始设计（必须由目标机 probe 校准）

| Profile | 低风险初值 | 设计理由 |
|---|---|---|
| `smoke` | 1 GPU；1B；prompt batch 1；rollout n=2；mini-batch 1；micro-batch 1；response/latent 取最小可闭环值；1–2 steps | 验证算法/日志闭环，不代表效果 |
| `3gpu-low` | 3 GPU FSDP DP；SGLang TP=1/SP=1；1B；prompt batch `P=6`；rollout `n=4`；mini prompt batch `M=3`；per-GPU actor/old-log-prob micro-batch `2/2`；BF16；remove-padding；无 offload 起步 | `T=P×n=24`，每 rank local batch 8、local mini 4、gradient accumulation 2、每 global step 2 次 optimizer attempt；保留真实 latent/noise/FlipGrad |
| `3gpu-high-smoke` | 3 GPU FSDP DP；SGLang TP=1/SP=1；7B；`P=3,n=4,M=3`；actor/old/ref micro-batch 1；BF16；gradient checkpointing；actor param+optimizer offload，ref param offload；prompt/response 512/256；1 step | `T=12`，每 rank local batch/mini 4、gradient accumulation 4、每 global step 1 次 optimizer attempt；只验证高难链路，禁止标为论文复现 |

其中 low 的 prompt/response 候选为 192/128，high 为 512/256；SGLang static memory fraction 均以 0.50 作为待测起点。作者链路当前没有独立、已生效的 `max_latent_length` 旋钮，YAML 中声明目标 cap 不能被当成已实现。以每卡恰为 46 GiB 为示例，至少保留 `max(2 GiB, 10%)=4.6 GiB` headroom，admitted peak 不超过 41.4 GiB；实际阈值必须按目标卡实测容量计算。目标机依次执行空闲显存 probe、rank-init、model-load、weight-sync/rollout、old/ref forward、backward、checkpoint probe 后再固化；任何自动调小都生成新 profile/hash，不覆盖原 profile 身份。完整算术、回滚和 OOM 分阶段策略见 `work_reports/agent_d_3gpu_plan.md`。

## 5. 阶段门禁与回滚

1. 依赖/runtime probe 未闭合：不得导入训练栈。
2. config/schema/ID/count 单测未通过：不得做 GPU smoke。
3. 单卡闭环、validator、resume、probe no-mutation 任一失败：不得做三卡最小训练。
4. 三卡 rank-init 或 writer ownership 失败：不得做 7B smoke。
5. reviewer 有 blocker：不得启动长训练或声称完成。

所有上游改动由单一 patch 可逆；新 runner 可通过 `metrics_enabled=false` 关闭 observer。回滚验证是应用 patch 前后/observer off 的首个可比 step 对照，而不是删除测试或放宽断言。

## 6. 本计划自审结果

- 规范覆盖：训练、分布式、29 core、5 diagnostic、raw facts、Support/probe、存储、resume、性能和交付均有任务归属。
- 模糊占位扫描：本文件不使用 `TBD/TODO/implement later`；尚未能由静态证据确定的版本与显存参数明确转为 runtime probe gate。
- 接口一致性：`StepContext`、`SufficientStats`、`TrainingObserver`、`UpdateResult` 在任务间使用同一名称和职责。
- 本阶段状态：仅设计；所有实现和测试状态仍为 `planned`。
