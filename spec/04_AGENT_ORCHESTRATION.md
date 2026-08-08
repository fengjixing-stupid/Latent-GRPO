# Codex 多智能体分工协议


## 0. 路径约定

Codex 必须从项目根目录工作。固定路径为：

```text
./Latent-GRPO       # 作者原始仓库
./spec              # 全部任务规范 Markdown
```

引用任何规范文件时都使用 `./spec/<filename>.md`，不得到 `./Latent-GRPO/spec` 或其他目录中查找。

---

## 1. 总原则

主 Codex 是唯一总协调者、合并者和最终责任人。

可以自行创建或调用多个智能体，但必须遵守：

- 先定义任务边界再分配；
- 每个文件同一时间只有一个 owner；
- 只读审计可以并行；
- 核心训练文件、schema 和入口的最终修改由主智能体合并；
- 子智能体不能自行改变指标契约；
- 子智能体的结论必须带证据；
- 所有报告落到 `work_reports/`；
- 主智能体必须运行测试，不得只相信子智能体口头结论。

若平台不支持真实子智能体，按以下角色依次工作，并保留相同报告。

---

## 2. 推荐角色

## Agent A：Upstream Repository Auditor

只读范围：

```text
./Latent-GRPO
```

任务：

- 找出真实训练入口、配置链和依赖；
- 绘制 rollout→reward→advantage→update 数据流；
- 定位 Latent-GRPO 关键算法语义；
- 确认 step、mask、length、overlong、checkpoint、resume；
- 列出可复用接口与潜在 patch；
- 不修改代码。

输出：

```text
work_reports/agent_a_repo_audit.md
```

必须带文件路径和行号。

---

## Agent B：Runtime and Distributed Training Engineer

owner：

```text
train_latent_grpo.py
latent_grpo_runner/config.py
latent_grpo_runner/environment.py
latent_grpo_runner/distributed.py
latent_grpo_runner/trainer.py
configs/
scripts/run_*.sh
```

任务：

- 环境探测；
- 上游 adapter 的训练启动；
- 单卡 smoke；
- 三卡 torchrun；
- step/checkpoint/resume；
- profile 与显存适配；
- 错误传播。

输出：

```text
work_reports/agent_b_runtime.md
```

不得独立定义指标语义。

---

## Agent C：Metrics and Storage Engineer

owner：

```text
latent_grpo_runner/metrics/events.py
latent_grpo_runner/metrics/aggregators.py
latent_grpo_runner/metrics/schemas.py
latent_grpo_runner/metrics/storage.py
scripts/validate_outputs.py
```

任务：

- 把 `./spec/target_variables.md` 转成 schema；
- 设计 event 与 sufficient statistics；
- Parquet/JSON 写盘；
- 原子提交；
- resume/去重；
- validator；
- 性能计时。

输出：

```text
work_reports/agent_c_metrics_storage.md
```

不得根据旧代码位置改变变量定义。

---

## Agent D：Latent Mechanism, Support and Probe Engineer

owner：

```text
latent_grpo_runner/metrics/masks.py
latent_grpo_runner/metrics/stage1.py
latent_grpo_runner/metrics/stage2.py
latent_grpo_runner/metrics/support.py
latent_grpo_runner/metrics/probe.py
latent_grpo_runner/evaluation.py
```

任务：

- valid latent mask；
- noisy mixture；
- zero-advantage；
- stable group/trajectory；
- Optimal Correct Path；
- Support；
- surrogate margin/FlipGrad；
- checkpoint probe；
- credit 可选路径；
- eval raw facts。

输出：

```text
work_reports/agent_d_latent_metrics.md
```

所有公式和时间点必须引用 target contract。

---

## Agent E：Testing and Verification Engineer

优先只新增测试，不修改生产实现，除非主智能体分配修复。

owner：

```text
tests/
```

任务：

- 单元测试；
- schema 测试；
- synthetic distributed aggregation；
- resume；
- RNG restore；
- no-mutation；
- output validator；
- smoke integration；
- 三卡测试计划。

输出：

```text
work_reports/agent_e_validation.md
```

失败测试必须给最小复现，不得降低断言绕过问题。

---

## Agent F：Independent Reviewer

只读审查所有实现与报告。

任务：

- 对照 `./spec/target_variables.md` 检查漏项；
- 查找 Classic-GRPO 逻辑误用；
- 查找时间点、mask、count 和分布式聚合错误；
- 查找 probe 改变训练状态的风险；
- 查找写盘/resume 数据损坏风险；
- 检查三卡 profile 是否被错误标为严格复现；
- 检查未测试却声称完成的内容。

输出：

```text
work_reports/agent_f_review.md
```

问题按：

```text
blocker
major
minor
note
```

分级。

---

## 3. 分工顺序

推荐并行与串行关系：

```text
A 仓库审计 ─────────────┐
                         ├─> 主智能体形成 implementation_plan
C schema 初稿 ──────────┘

B runtime 骨架 ─────────┐
C storage 实现 ─────────┼─> 主智能体集成单卡 smoke
D 指标接口 ─────────────┘

主智能体单卡闭环
        ↓
E 单元/集成测试
        ↓
主智能体三卡闭环
        ↓
D Support/probe 集成
        ↓
E 性能与 no-mutation
        ↓
F 独立审查
        ↓
主智能体修复、复测、最终报告
```

Agent D 在 Agent A 的审计结论和主智能体的数据接口确定前，不应大规模修改训练路径。

---

## 4. 子智能体任务模板

主智能体分配任务时至少包含：

```text
Role:
Goal:
Read-only inputs:
Owned files:
Files that must not be edited:
Required evidence:
Required tests:
Output report:
Definition of done:
```

示例：

```text
Role: Metrics and Storage Engineer
Goal: 实现 Stage 1/2 的 schema、聚合和 append-only Parquet writer。
Read-only inputs: ./spec/target_variables.md, ./spec/03_METRICS_STORAGE_CONTRACT.md
Owned files: latent_grpo_runner/metrics/{events,aggregators,schemas,storage}.py
Files that must not be edited: trainer.py, upstream repository
Required evidence: 每个字段到 schema 的追踪表
Required tests: schema、atomic write、resume、duplicate key
Output report: work_reports/agent_c_metrics_storage.md
Definition of done: 测试通过且无遗漏字段
```

---

## 5. 合并协议

每次合并前主智能体检查：

1. owner 范围；
2. 是否修改了契约；
3. 是否有测试；
4. 是否有运行证据；
5. 是否引入额外 forward/backward；
6. 是否在 worker 写权威文件；
7. 是否保存禁止的全量 Tensor；
8. 是否改变训练 RNG/参数；
9. 是否有 schema/version 变更；
10. 是否更新 traceability matrix。

冲突解决：

- 指标语义冲突：以 target contract 为准；
- 上游实现事实冲突：复现实验或 runtime probe；
- 两个 agent 修改同一文件：由主智能体重新手工合并，不直接选择一方覆盖；
- 测试与实现冲突：先确认测试是否对应契约，禁止仅删测试。

---

## 6. 进度与停止条件

每阶段结束主智能体更新：

```text
docs/progress.md
```

包含：

```text
completed
in_progress
blocked
tests_run
test_results
next_step
```

以下情况不得进入长训练：

- 单卡 smoke 未闭环；
- schema validator 未通过；
- 三卡 rank 初始化未通过；
- generated token count 语义未验证；
- trajectory ID 不稳定；
- Support 对齐测试失败；
- probe no-mutation/RNG restore 失败；
- 输出目录 resume/去重失败；
- reviewer 存在 blocker。

不要用长训练掩盖工程错误。
