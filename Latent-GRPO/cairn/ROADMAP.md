# Latent-GRPO 路线图

**当前重点**：等待用户确认 Phase A 审计结论；获授权后先完成目标机 runtime probe，再进入测试先行实现。

## 里程碑

- [x] 完成 Phase A 只读仓库/依赖/变量映射/三卡/验收审计设计。
- [ ] 在目标 Linux 3×约 46 GB GPU 环境闭合 Python/CUDA/NCCL/shape/launcher probe。
- [ ] 按 RTM 和门禁实现 runner、最小 observer patch、指标存储与验证链。

## 开放问题

1. 目标机实际 GPU 型号、driver、torch CUDA build、NCCL 与 kernel wheel 组合是什么？
2. 用户是否授权下一阶段创建 requirements/constraints、测试与最小上游 patch？
