# Latent-GRPO Codex 任务包：使用说明

## 1. 本任务包的目的

本任务包用于指导 Codex 在当前工作目录中：

1. 阅读作者原始仓库 `./Latent-GRPO`；
2. 审计 Latent-GRPO 的真实训练、采样、奖励、advantage、更新与 checkpoint 语义；
3. 编写一套**专属于 Latent-GRPO**、可直接运行的 Python 训练入口与配套模块；
4. 在训练、评估、Support 和 checkpoint probe 过程中，按 `./spec/target_variables.md` 记录全部目标变量；
5. 在 Linux + VSCode Remote、非 Docker、3×约 46 GB NVIDIA GPU 环境中通过 smoke 和目标配置验证；
6. 输出可复现的配置、运行命令、测试、日志 schema 和实现报告。

本任务不是 Classic-GRPO 实现任务，不得把 Classic-GRPO 的 rollout、显式 CoT 或 token-level 逻辑替代 Latent-GRPO 的 latent token、noisy top-K mixture、one-sided noise、FlipGrad 或 Optimal Correct Path 机制。

---

## 2. 已知目录与环境约束

Codex 必须知道：

```text
当前工作目录/
├── Latent-GRPO/                  # 作者原始 Git 仓库
└── spec/                         # 所有任务规范与指标契约
    ├── 00_START_HERE.md
    ├── 01_CODEX_MASTER_PROMPT.md
    ├── 02_SYSTEM_AND_IMPLEMENTATION_CONTRACT.md
    ├── 03_METRICS_STORAGE_CONTRACT.md
    ├── 04_AGENT_ORCHESTRATION.md
    ├── 05_VALIDATION_AND_DELIVERABLES.md
    ├── target_variables.md
    └── package_manifest.json
```

作者原始仓库的相对路径固定为：

```text
./Latent-GRPO
```

不得假设当前工作目录就是作者仓库根目录。

目标运行环境按此前约定为：

```text
操作系统：Linux
交互方式：VSCode Remote / 可视化终端
容器：不使用 Docker
GPU：3 张 NVIDIA CUDA GPU
单卡显存：约 46 GB
主要启动方式：torchrun --nproc_per_node=3
```

实际 GPU 型号、显存、驱动、CUDA、PyTorch、NCCL、BF16 支持与可用磁盘空间必须在目标机器上运行时探测，不能写死。

---

## 3. 推荐喂给 Codex 的顺序

所有规范文件均位于 `./spec`。请从项目根目录启动 Codex，并按顺序阅读：

1. `./spec/00_START_HERE.md`
2. `./spec/01_CODEX_MASTER_PROMPT.md`
3. `./spec/02_SYSTEM_AND_IMPLEMENTATION_CONTRACT.md`
4. `./spec/03_METRICS_STORAGE_CONTRACT.md`
5. `./spec/04_AGENT_ORCHESTRATION.md`
6. `./spec/05_VALIDATION_AND_DELIVERABLES.md`
7. `./spec/target_variables.md`

然后从项目根目录将 `./spec/01_CODEX_MASTER_PROMPT.md` 的全文作为主任务交给 Codex。

不要把七份文件拆成七个互相独立的任务。它们共同组成一次实现任务：

- `01` 决定总目标和工作顺序；
- `02` 决定训练脚本与硬件约束；
- `03` 决定指标如何保存；
- `04` 决定 Codex 如何分配智能体；
- `05` 决定如何验收；
- `./spec/target_variables.md` 决定必须记录什么。

---

## 4. 冲突优先级

若文件之间出现冲突，按以下优先级处理：

1. `./spec/target_variables.md` 中的变量定义、mask、count、时间语义和禁止项；
2. `./spec/05_VALIDATION_AND_DELIVERABLES.md` 中的安全与正确性验收条件；
3. `./spec/03_METRICS_STORAGE_CONTRACT.md` 中的 schema 与写盘规则；
4. `./spec/02_SYSTEM_AND_IMPLEMENTATION_CONTRACT.md` 中的训练与硬件约束；
5. `./spec/04_AGENT_ORCHESTRATION.md` 中的工作组织方式；
6. `./spec/01_CODEX_MASTER_PROMPT.md` 中的一般性流程。

若作者仓库真实实现与文档中的非核心描述不一致，以运行时事实和作者仓库算法语义为准，但：

- 不得静默改变 `./spec/target_variables.md`；
- 必须在 `docs/decision_log.md` 记录差异、证据、影响和采用方案；
- 涉及指标语义的差异必须写入 definition version 或 availability reason；
- 不得为了“让测试通过”而伪造指标、mask、count 或时间点。

---

## 5. 完成后用户应获得的核心入口

Codex 最终至少应交付：

```text
train_latent_grpo.py
configs/smoke.yaml
configs/3gpu-low.yaml
configs/3gpu-high-smoke.yaml
scripts/run_smoke.sh
scripts/run_3gpu_low.sh
scripts/run_3gpu_high_smoke.sh
scripts/validate_outputs.py
docs/implementation_report.md
docs/repo_audit.md
docs/requirements_traceability_matrix.md
docs/operator_runbook.md
```

实际模块目录可以由 Codex根据仓库结构合理设计，但上述用户入口和报告不得缺失。
