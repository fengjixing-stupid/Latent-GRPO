# Latent-GRPO 路线图

**当前重点**：Low/High 三卡配置、门禁、遥测与运行手册已打包；等待目标 Linux 3GPU 机器分别执行对应 final validation，PASS 后再启动对应正式多 seed 训练。

## 里程碑

- [x] 完成 Phase A 只读仓库/依赖/变量映射/三卡/验收审计设计。
- [x] 按 RTM 和门禁实现 runner、最小 observer patch、指标存储、三卡配置与 teammate 运行手册。
- [ ] 在目标 Linux 3×约 46 GB GPU 环境闭合 Python/CUDA/NCCL/shape/launcher probe。
- [ ] Low/High 分别通过对应 `3GPU_FINAL_GATE` 后启动并跟踪正式多 seed 训练。

## 开放问题

1. 目标机实际 GPU 型号、driver、torch CUDA build、NCCL 与 kernel wheel 组合是什么？
2. 目标机第一次 validation 是否能在两个真实 update 内满足每卡 allocator 与 probe growth gate？
