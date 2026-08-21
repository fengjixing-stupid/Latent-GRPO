# Latent-GRPO 三卡最终实验包

本包用于 Linux、Python 3.11、CUDA 12.4、三张至少 40 GiB 且支持 BF16 的 NVIDIA GPU。当前服务器建议使用物理 GPU `4,5,6`；GPU7 已有任务时不要加入同步训练。

## 验收边界

- `release_validation/LOCAL_RELEASE_ACCEPTANCE.json`：代码语法、Shell 语法、单元测试、依赖固定值、Git 完整性和四个最终配置 dry-run 的交付验收；只有其中 `git_commit` 等于当前 `git rev-parse HEAD` 时才有效。
- `artifacts/validation/<run>/acceptance.json`：在目标 L20 服务器上真实执行 CUDA、NCCL、Ray、SGLang、FSDP backward、optimizer update、29 项指标、checkpoint/resume 后的最终运行验收。

本地交付验收通过不等于 GPU 实验已运行。正式训练必须先在目标机看到：

```text
3GPU_PREFLIGHT_GATE: PASS
3GPU_DISTRIBUTED_RUNTIME_GATE: PASS
CORE_METRICS: 29/29
CUDA_RNG_ALL_DEVICES: PASS
CHECKPOINT_GATE: PASS
3GPU_FINAL_GATE: PASS
```

## 入口

完整安装、资产准备、Low/High validation 与正式训练命令见：

```text
docs/3GPU_RUNBOOK.md
```

重新检查交付包：

```bash
python tools/validate_release_package.py
```
