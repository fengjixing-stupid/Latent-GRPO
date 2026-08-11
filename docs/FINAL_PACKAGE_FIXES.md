# 最终实验包修复清单

日期：2026-08-11

## 已修复的确定性阻断

1. 将四份三卡文档移动到测试和链接约定的 `docs/` 目录。
2. 将活动 CUDA 依赖统一为 `flashinfer-python==0.2.5`、`sgl-kernel==0.1.1`，同步 requirements、constraint、SGLang metadata、CI 安装脚本和目标机安装器。
3. 目标机安装器补装 W&B 可选组、验证实际版本并执行 `pip check`；FlashAttention 未预装时明确检查 `nvcc`。
4. GPU telemetry 与环境探针只采集 `CUDA_VISIBLE_DEVICES` 选中的三张物理卡，并按选择顺序保留 rank 映射；不会再把 8 卡服务器上的 GPU0–3、GPU7 误计入三卡验收。
5. 预检增加物理 GPU 编号、重复 ID、已有显存、已有利用率、BF16、NCCL、Ray 三 worker 与依赖一致性门禁。
6. Parquet 资产检查改用 Arrow 顶层 schema，并验证 prompt、reward_model、extra_info 的样例结构。
7. 模型资产必须是本地目录并包含真实权重文件；tokenizer/config 必须离线可读，Low/High 的 `</think>` ID 必须匹配。
8. 正式训练 acceptance 与当前 Git commit、Low/High validation profile 绑定，并在正式启动前重新执行 preflight。
9. final validation wrapper 只接受 `profile_kind=final_runtime_validation` 且 `max_steps=2` 的配置，防止误把正式 10/5 epochs profile 当成短验收运行。
10. 正式 profile 显式拒绝命令行 `--max-steps`，消除“参数看似生效、实际仍按 epoch 运行”的静默行为。
11. Actor 的训练端 Top-K 宽度由 rollout payload 的最后一维决定，不再硬编码为 10；当前作者配置仍保持 Top-10。
12. 增加确定性的 `CUDA_DEVICE_ORDER=PCI_BUS_ID`、三卡运行报告、显存/利用率 telemetry、checkpoint/resume 与 29 指标 fail-closed 验收。
13. 安装报告默认可写入 Git 忽略目录，预检只允许生成的历史 target report 发生变化，源码和配置仍必须干净。
14. 清理 macOS metadata、Python cache，并移除一个 vendored Python 文件的 UTF-8 BOM，保证统一源代码编译检查。

## 已完成的交付验收

`python tools/validate_release_package.py` 覆盖：

- Git commit 与干净工作树；
- 必需文件和打包杂质；
- 所有 Python 源文件语法；
- 所有 Shell 脚本 `bash -n`；
- 完整 `pytest` 单元测试；
- 四个最终 Low/High、validation/formal profile 的 strict config dry-run；
- 活动 CUDA 依赖固定值一致性。

结果写入：

```text
release_validation/LOCAL_RELEASE_ACCEPTANCE.json
```

## 必须在 L20 服务器完成的运行验收

交付环境无法代替目标服务器执行 CUDA/NCCL/SGLang/FSDP。目标机只有在运行：

```bash
bash tools/run_3gpu_final_validation.sh ... --gpus 4,5,6
```

并生成以下结果后，才属于真实实验运行验收通过：

```text
CORE_METRICS: 29/29
CUDA_RNG_ALL_DEVICES: PASS
CHECKPOINT_GATE: PASS
3GPU_FINAL_GATE: PASS
```

机器证据为对应 validation 目录中的 `acceptance.json`。
