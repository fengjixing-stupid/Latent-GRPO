# Codex 主任务：实现可记录完整中间变量的 Latent-GRPO Python 训练系统

你是本任务的总协调智能体和最终代码负责人。请从同时包含 `./Latent-GRPO` 与 `./spec` 的**项目根目录**启动，并在该项目根目录中完成全部工作。

## 一、必须先确认的事实

作者原始 Git 仓库位于：

```text
./Latent-GRPO
```

所有任务规范位于：

```text
./spec
```

指标契约位于：

```text
./spec/target_variables.md
```

运行环境是：

```text
Linux + VSCode Remote
不使用 Docker
3 张约 46 GB 的 NVIDIA CUDA GPU
主要分布式启动方式：torchrun --nproc_per_node=3
```

你不得把当前工作目录误认为 `./Latent-GRPO` 仓库根目录，也不得要求用户另外下载作者仓库。

---

## 二、任务目标

基于作者仓库的真实 Latent-GRPO 算法语义，编写一套专用 Python 训练系统，使其能够：

1. 启动 Latent-GRPO 的训练、checkpoint、评估、Support 和 checkpoint probe；
2. 支持单卡 smoke 与三卡目标运行；
3. 在训练过程中记录 `./spec/target_variables.md` 中要求的全部字段；
4. 对默认关闭、当前不可用或退化的指标写明确 availability 和 reason，而不是静默省略；
5. 使用可靠、可追加、可恢复、便于分析的存储格式；
6. 不改变 Latent-GRPO 的训练结果语义；
7. 不把三卡适配配置声明为论文严格复现；
8. 提供测试、运行脚本、配置、schema 校验器、实现报告和操作说明。

本任务只实现 Latent-GRPO，不实现 Classic-GRPO。

---

## 三、必须阅读的文件

在修改代码前，从项目根目录完整阅读：

```text
./spec/00_START_HERE.md
./spec/02_SYSTEM_AND_IMPLEMENTATION_CONTRACT.md
./spec/03_METRICS_STORAGE_CONTRACT.md
./spec/04_AGENT_ORCHESTRATION.md
./spec/05_VALIDATION_AND_DELIVERABLES.md
./spec/target_variables.md
```

同时审计 `./Latent-GRPO`：

- README、安装说明和依赖；
- 所有训练入口、配置系统和启动脚本；
- rollout/sampling 路径；
- latent token 构造；
- noisy top-K mixture；
- Gumbel 与 one-sided noise；
- reward manager/scorer；
- group 构造；
- advantage；
- Optimal Correct Path；
- PPO/GRPO actor update；
- old/current policy log-probability；
- FlipGrad；
- checkpoint/eval；
- 分布式 worker、DP/TP、Ray 或其他运行时；
- 模型与数据加载；
- resume 与 RNG 处理。

不得只根据变量名猜测算法语义。

---

## 四、先审计，后设计，再编码

### Phase A：只读审计

首先不修改作者仓库，生成：

```text
docs/repo_audit.md
```

至少包含：

- 仓库 commit/hash（可获得时）；
- 实际 Python 入口和启动链；
- 配置解析链；
- 训练 global step 与 optimizer step 定义；
- rollout 到 actor update 的真实数据流；
- latent/noisy mixture/one-sided/FlipGrad/Optimal Correct Path 的真实实现；
- reward、advantage、mask、overlong、EOS 和长度语义；
- checkpoint 和 resume 语义；
- 分布式拓扑；
- 可直接复用的公共接口；
- 需要 adapter 的私有接口；
- 不能安全复用或需要补充实现的部分；
- 每项结论的文件路径与行号；
- 尚未确认、必须通过 runtime probe 解决的问题。

### Phase B：需求追踪

建立：

```text
docs/requirements_traceability_matrix.md
```

为 `./spec/target_variables.md` 中每个目标字段记录：

```text
field_name
stage
record_type
semantic_definition
observation_phase
effective_mask
aggregation_method
storage_table
schema_type
implementation_module
test_id
default_enabled
availability_behavior
status
```

状态只能使用：

```text
planned
implemented
verified
unavailable_with_reason
blocked
```

不得用“基本完成”“应该可用”等模糊描述。

### Phase C：架构设计

生成：

```text
docs/implementation_plan.md
docs/decision_log.md
```

明确：

- 新脚本目录结构；
- 对作者仓库的复用方式；
- 是否需要 patch；
- 分布式训练方式；
- 指标事件如何从 worker 汇总到 driver；
- 写盘线程/进程与训练线程的边界；
- checkpoint、resume、去重和原子写入；
- Support 与 probe 的低频调度；
- 测试顺序；
- 风险与回滚方式。

设计完成后即可继续实现，不需要等待用户确认。遇到非关键歧义时做保守决定并记录到 `decision_log.md`。

### Phase D：实现与测试

按照小步可验证顺序实现：

1. 配置、环境探测和启动入口；
2. 单卡最小训练闭环；
3. 三卡分布式训练闭环；
4. Stage 1 基础日志；
5. Stage 2 低成本机制变量；
6. Stage 3 Support；
7. Stage 4 checkpoint probe；
8. checkpoint eval raw facts；
9. resume、原子写盘和 schema validator；
10. 性能与正确性验收。

每完成一阶段立即运行对应测试，不要把所有测试推迟到最后。

---

## 五、对作者仓库的使用原则

优先顺序：

1. 通过 adapter 复用作者仓库已验证的算法与运行时；
2. 在新目录中实现独立的训练入口、配置、指标和存储层；
3. 若必须修改作者仓库，优先提供最小 patch；
4. 不得大规模复制作者代码并造成两套难以同步的实现；
5. 不得直接覆盖作者文件且不保留 diff；
6. 不得通过脆弱 monkey patch 隐藏关键算法修改；
7. 所有上游修改写入 `patches/` 并在报告中逐项解释。

作者仓库默认视为算法参考和可复用依赖，而不是让你忽略审计直接重写算法。

---

## 六、必须交付的训练入口

至少提供：

```text
train_latent_grpo.py
```

该入口必须支持：

```text
--config
--profile-name
--seed
--output-root
--resume-from
--max-steps
--log-level
--dry-run
--validate-config
--enable-support / --disable-support
--enable-checkpoint-probe / --disable-checkpoint-probe
--enable-credit-probe
--allow-hardware-mismatch
```

可以使用 Hydra、argparse 或作者仓库现有配置系统，但用户必须能够用一个明确命令启动。

提供配置：

```text
configs/smoke.yaml
configs/3gpu-low.yaml
configs/3gpu-high-smoke.yaml
```

语义：

- `smoke`：最小端到端工程验证，不代表实验结果；
- `3gpu-low`：3×约 46 GB GPU 的低难度主要运行配置，属于设备适配配置，不是论文严格复现；
- `3gpu-high-smoke`：三卡 7B/高难度链路验证，缩小 batch、rollout、response length 和 step，不是性能实验或论文结果；
- 若保留 `paper-low`/`paper-high`，在少于论文要求 GPU 数量时必须明确拒绝或要求显式 override，不得伪装成严格复现。

---

## 七、指标完整性要求

`./spec/target_variables.md` 是唯一指标契约。

必须满足：

1. 每个字段都有明确 schema；
2. 每个统计族都有 count；
3. 每个可失败 family 都有 availability 与 reason；
4. 分布式聚合使用 sum/sum_sq/count 等充分统计量；
5. driver/rank 0 负责最终写盘；
6. 禁止 worker mean 的简单平均；
7. 不同 mask 不共享 count；
8. 所有时间点遵循 `observation_phase`；
9. 不得增加额外 full-vocabulary forward 来伪造“已有变量”；
10. 不得保存完整 logits、hidden states、完整梯度或长期计算图；
11. Stage 4 credit 默认关闭；
12. Gumbel diagnostic 仅独立 diagnostic/smoke；
13. 没有 checkpoint 级 `eval_metrics` 汇总表；
14. 不记录 `train/gradient_norm`；
15. `train/generated_token_count` 必须遵循 target contract；
16. probe 生成不计入训练 token/length；
17. Support 和 probe 不得改变 optimizer、参数、RNG 或正常 `.grad` 状态；
18. 字段不能安全得到时，写 unavailable_with_reason，不得猜测、静默 reshape、截断、广播或换口径。

---

## 八、存储要求

严格执行 `./spec/03_METRICS_STORAGE_CONTRACT.md`。

核心原则：

- 小型静态配置：JSON；
- 训练、group、Support、probe、eval 等表：分区/分片 Parquet dataset；
- 可选 JSONL 仅用于 debug，不是权威数据源；
- 原子写入；
- append-only；
- resume 安全；
- schema version；
- 主键去重；
- 输出验证器；
- 训练失败时尽量保留已提交分片和状态文件；
- 不使用 CSV 作为主要存储格式；
- 不把不同粒度的数据塞进一张宽表。

---

## 九、智能体分配

按照 `./spec/04_AGENT_ORCHESTRATION.md` 自行分配智能体。至少覆盖：

- 上游仓库审计；
- 训练与分布式运行；
- 指标语义与存储；
- Support/probe；
- 测试与验收；
- 最终集成审查。

你是唯一总协调者。不得让多个智能体无协调地同时编辑同一文件。每个智能体必须输出可审查报告，主智能体负责合并、运行测试和承担最终正确性。

若当前 Codex 环境不支持真实子智能体，则按这些角色顺序执行，并保留相同的工作报告与交叉审查过程。

---

## 十、安全和操作限制

未经用户明确允许，不得：

- 删除作者仓库或用户已有数据；
- 重写 Git 历史；
- 执行 `git reset --hard`、大范围 `rm -rf`；
- 自动上传代码、日志、模型或数据；
- 扫描工作区之外的目录；
- 使用 sudo；
- 修改系统 CUDA/驱动；
- 自动安装系统包；
- 把 token、密钥、用户名、绝对私有路径写入日志；
- 启动长时间正式训练来代替 smoke 验证。

Python 依赖可以写入 requirements/lock 文件。实际安装前先检查环境，并把需要用户执行的命令写入 runbook。若当前环境允许且任务上下文明确授权执行，可安装项目局部 Python 依赖，但不得改系统驱动。

---

## 十一、最终交付

至少交付 `./spec/05_VALIDATION_AND_DELIVERABLES.md` 规定的全部内容，并生成：

```text
FINAL_IMPLEMENTATION_REPORT.md
```

报告必须包含：

- 完成项；
- 未完成项；
- 真实运行过的命令；
- 测试结果；
- 目标 GPU 上的 smoke 结果；
- 三卡结果；
- 指标覆盖率；
- unavailable 字段及原因；
- 性能开销；
- 峰值显存；
- 对作者仓库的所有修改；
- 已知限制；
- 用户下一步的准确运行命令。

只有代码存在但未运行测试，不得声称“完成”。
