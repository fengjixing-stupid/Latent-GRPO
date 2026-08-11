---
type: project_topic
status: active
summary: "三卡最终门禁采用单 ray_direct job、作者真值冻结与 fail-closed runtime evidence。"
tags: [3gpu, runtime-gate, packaging, stage4]
contains: [decision, lesson]
created: "2026-08-11"
updated: "2026-08-11"
related:
  - "../../docs/3GPU_RUNBOOK.md"
  - "../../docs/AUTHOR_HYPERPARAMETER_AUDIT.md"
  - "../../docs/3GPU_HYPERPARAMETER_DEVIATIONS.md"
authoring_mode: ai_generated
---
# 三卡 Runtime Gate 与训练打包

## Current Conclusions

- 正式控制面只有一个 `ray_direct` Python driver、一套 Ray runtime 和一份训练 job；三张 GPU 由 Ray/FSDP worker 使用，不增加外层多进程 launcher。
- 作者 low/high shell 是超参数第一来源。正式 low 三卡将 GPU 数 8→3、prompt batch 64→48、PPO mini batch 16→12；正式 high 三卡使用 prompt/mini 12/12，使 `12×8÷3=32` trajectories/rank，与作者 `32×8÷8=32` 相同。所有差异写入 deviation 文档。
- Low/High 各有独立的两步 final validation 和 acceptance identity；formal wrapper 拒绝跨 profile 复用验收报告。
- Hugging Face ID 只作为模型下载来源。preflight/formal runtime 的 `--model-path` 必须是现存本地目录，train/val 必须是现存本地 parquet；资产与输出路径均显式传入 wrapper。
- 本地 Mac 只验证配置、脚本和纯逻辑；BF16/CUDA/NCCL/FSDP 的最终状态必须保持 `TARGET_RUNTIME_EXECUTION_REQUIRED`，由目标机报告闭合。

## Lessons

- 作者 low profile 要求 `use_remove_padding=true`，而旧 checkpoint probe 只返回 padded-path tensor，因此三卡 Stage 4 会在真正运行前必然失败。正确修复是把 packed Top-K probe tensor 以可微 `index_copy` 恢复到 batch/sequence 域，再取 response predictor slice；不能把作者值静默改为 false 绕过。
- 仅有 `nvidia-smi memory.used` 不足以判断 allocator/probe 泄漏。最终证据同时保留两步、三个 actor rank 的 PyTorch allocated/reserved 当前值与峰值，以及每 worker probe time/peak memory；第二步 reserved 增量超过单卡总显存 25% 即 fail-closed。
- 把 Low 配置仅替换为 Qwen 模型和 Math 数据并不能得到作者 High 实验：High 同时要求 1024/4096 长度、KL loss、gradient checkpointing、actor offload、20480 actor token budget、12000 SGLang 长度和 0.8 memory utilization，必须使用独立 High profile。
