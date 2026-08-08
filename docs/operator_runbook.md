# Latent-GRPO 运行与验收手册

## 已知边界

当前交付状态是 `target_machine_test_deferred`。Mac 完成的是配置、静态/合成测试、报告生成和目标机脚本；没有 NVIDIA/CUDA 证据。`configs/3gpu-low.yaml` 是三卡低风险工程 profile，默认 `launcher.mode=ray_direct`；`configs/3gpu-high-smoke.yaml` 只在 low profile 通过且得到授权后人工运行，绝不由脚本自动触发。

## 权威执行入口

- Mac 配置验证：`python3 train_latent_grpo.py --config configs/smoke.yaml --dry-run --validate-config`
- 目标机全流程：依次运行 `scripts/target_machine/00_probe_environment.sh` 至 `11_collect_reports.sh`。
- 输出验证：`python scripts/validate_outputs.py --input <run-output-root>`。
- 三卡主路径：`CUDA_VISIBLE_DEVICES=0,1,2 python train_latent_grpo.py --config configs/3gpu-low.yaml --max-steps 2`，一个 Ray driver；`torchrun_control` 只作显式兼容模式。

完整逐步命令、PyTorch cu124 独立安装、requirements 顺序、本地 forks `--no-deps`、sgl-kernel 冲突处理和 CUDA 扩展 ABI gate 见 [teammate_target_machine_runbook.md](teammate_target_machine_runbook.md)。

## 启动 gate

按顺序满足才继续：

1. `runtime_probe.json`: Linux、3 GPU、每卡 ≥40 GiB、CUDA/BF16/NCCL；
2. 独立官方 cu124 PyTorch wheel 安装；
3. 分层 requirements 与 vendored verl/SGLang `--no-deps` 安装；
4. `import_check.json` 和 `requirements_validation.json`: CUDA 12.4、扩展 imports、`pip check`；
5. `ray_gpu_probe.json`: driver 不占卡、3 个唯一 1-GPU worker、异常传播；
6. 单卡 smoke + validator；
7. 三卡 smoke + validator + 10%/2 GiB 显存 headroom；
8. 有真实 checkpoint 后才做 resume。

任何 gate 失败都停止后续训练，不静默换模型、量化、offload、batch 或 token 长度。若要调参，复制为新 profile 并保留新 config hash。

## 报告语义

每份目标报告至少有 `command`、`started_at`、`finished_at`、`exit_code`、`status`、`environment_summary`、`stdout_log_path`、`stderr_log_path`、`artifacts`、`failure_reason`。仓库中的空模板均为 `target_machine_test_deferred`、`exit_code=null`；它们不是通过证据。

目标机真实成功后才允许使用 `target_machine_probe_passed`、`cuda_runtime_verified`、`requirements_lock_verified`、`single_gpu_tested`、`three_gpu_ray_tested`、`memory_feasibility_verified`。训练命令退出 0 不能替代输出 validator 或显存证据。

## 返还与诊断

执行 `11_collect_reports.sh` 后返还完整 `artifacts/target_machine/`。先看 `report_manifest.json` 的缺失/blocked 列表，再按每份报告指向的日志定位；扩展问题检查 FlashAttention、FlashInfer、sgl-kernel 与 torch/CUDA/CXX11 ABI，分布式问题检查 Ray placement、NCCL、GPU topology，数据问题检查 validator、Parquet part/manifest 和重复主键。

