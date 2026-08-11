# 3GPU 最终验收清单

状态：`TARGET_RUNTIME_EXECUTION_REQUIRED`。基线：`53438ec07b804ebd1b670d6fe118199798350505`。launcher：`ray_direct`。

先运行：

```bash
bash tools/run_3gpu_final_validation.sh --model-path "$MODEL_PATH" --train-data "$TRAIN_DATA" --val-data "$VAL_DATA" --output-root "$OUTPUT_ROOT" --gpus 0,1,2 --seed 17
```

Low 需加 `--config configs/3gpu-final-validation.yaml`，High 需加 `--config configs/3gpu-final-high-validation.yaml`；`MODEL_PATH` 必须是本地目录，数据参数必须是本地 parquet 文件。

| 完成 | 检查项 | PASS 证据 | 失败日志 |
|---|---|---|---|
| [ ] | Git clean | preflight 前 `git status --short` 无输出 | `logs/preflight.log` |
| [ ] | exact commit recorded | `preflight.json.git_commit` 和 `acceptance.json.git_commit` 非空且相同 | `logs/preflight.log` |
| [ ] | 3 GPUs visible | `preflight.json.gpu_count == 3` | `runtime_probe.json` |
| [ ] | >= 40 GB each | `GPU_VRAM: PASS` | `runtime_probe.json` |
| [ ] | BF16 available | `BF16: PASS` | `runtime_probe.json` |
| [ ] | NCCL available | `NCCL: PASS` | `runtime_probe.json` |
| [ ] | model present | `MODEL_ASSET: PASS` | `logs/preflight.log` |
| [ ] | train parquet present | `DATA_ASSET: PASS` 且 parquet 行数大于 0 | `asset_probe.json` |
| [ ] | val parquet present | `DATA_ASSET: PASS` 且 parquet 行数大于 0 | `asset_probe.json` |
| [ ] | tokenizer gate | `LATENT_END_TOKEN: PASS`，首 token ID 为 Low 524 / High 522 | `asset_probe.json` |
| [ ] | config gate | 所选 Low/High validation config dry-run 退出 0 | `logs/preflight.log` |
| [ ] | profile identity | `acceptance.json.profile_name` 等于所选 validation profile | `acceptance.json` |
| [ ] | single Ray/job | `ray_gpu_probe.json.status == target_machine_probe_passed` 且 driver 无 GPU | `ray_gpu_probe.json` |
| [ ] | distributed runtime | `3GPU_DISTRIBUTED_RUNTIME_GATE: PASS` | `logs/runtime.log` |
| [ ] | real backward | `acceptance.json.real_backward == PASS` | `logs/probe.log` |
| [ ] | optimizer step | `acceptance.json.real_optimizer_step == PASS` 且最大 optimizer step >= 2 | `logs/probe.log` |
| [ ] | Stage 1 | `ACCEPTANCE_SUMMARY.md` 为 `10/10` | `logs/probe.log` |
| [ ] | Stage 2 | `ACCEPTANCE_SUMMARY.md` 为 `6/6` | `logs/probe.log` |
| [ ] | Stage 3 | `2/2` 且 effective position count > 0 | `logs/probe.log` |
| [ ] | Stage 4 One-sided | `7/7` | `logs/probe.log` |
| [ ] | Stage 4 Credit | `4/4` 且 `credit_autograd_executed=true` | `logs/probe.log` |
| [ ] | worker→driver aggregation | `aggregation_worker_count == 3` 且 `worker_driver_aggregation == PASS` | `acceptance.json.details` |
| [ ] | no duplicate global row | `scripts/validate_outputs.py` 退出 0 | `logs/metrics.log` |
| [ ] | CUDA RNG all devices | `CUDA_RNG_DEVICE_0/1/2` 和 `CUDA_RNG_ALL_DEVICES: PASS` | `logs/probe.log` |
| [ ] | probe does not pollute `.grad` | `grad_pollution == NONE` | `acceptance.json` |
| [ ] | probe does not mutate parameter | `parameter_pollution_by_probe == NONE` | `acceptance.json` |
| [ ] | probe does not mutate optimizer | `optimizer_state_pollution_by_probe == NONE` | `acceptance.json` |
| [ ] | checkpoint write | `CHECKPOINT_GATE: PASS`；actor、data.pt、metrics sidecar 均存在 | `logs/checkpoint.log` |
| [ ] | resume compatibility | `resume_compatibility == PASS` | `resume_gate.json`、`logs/checkpoint.log` |
| [ ] | per-GPU device telemetry | 三卡都有 peak memory.used、average/peak utilization | `gpu_telemetry.json`、`ACCEPTANCE_SUMMARY.md` |
| [ ] | PyTorch allocator | 两步、每步 rank 0/1/2 都有 allocated/reserved 当前值和峰值 | `run/gpu_runtime_metrics.json`、`acceptance.json.details.gpu_allocator` |
| [ ] | bounded memory growth | `bounded_second_step_growth == true`；第二步 reserved 增量不超过单卡总显存 25% | `acceptance.json.details.gpu_allocator` |
| [ ] | step/probe performance | 两个 `train/step_time`；三个 rank 都有 probe extra time/peak memory | `ACCEPTANCE_SUMMARY.md`、`run/probe_worker_runtime.json` |
| [ ] | SGLang utilization identity | configured memory utilization 为作者 Low `0.6` / High `0.8`，并同时报告 validation 窗口设备 utilization | `acceptance.json.details`、`gpu_telemetry.json` |
| [ ] | final gate | `acceptance.json.final_gate == PASS` 和 `3GPU_FINAL_GATE: PASS` | `acceptance.json.blockers` |

Stage 3 的 sufficient statistics 在 driver 持久化前跨三 worker 合并。Stage 4 在三个 actor worker 各生成一个有界 packet，driver 校验 rank 0/1/2 完整后合并为一个 authoritative checkpoint row/set；因此全局 CUDA RNG PASS 表示三个 packet 全部通过各自可见设备的 `get_rng_state_all()` 比较。

只有最后一项 PASS 后才运行 `bash tools/run_3gpu_training.sh`。Low formal 只接受 `3gpu-final-validation` 报告；High formal 只接受 `3gpu-final-high-validation` 报告。validation 不会自动启动正式训练。
