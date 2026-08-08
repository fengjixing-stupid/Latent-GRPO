# Linux 三卡目标机验收计划

计划状态：`target_machine_test_deferred`。本文件定义证据门槛，不记录尚未发生的通过结论。

| Gate | 命令/报告 | 通过条件 | 可授予状态 | 失败诊断 |
|---|---|---|---|---|
| 环境 | `00_probe_environment.sh` / `runtime_probe.json` | Linux；3 卡；每卡 ≥40 GiB；CUDA、BF16、NCCL；路径/磁盘可用 | `target_machine_probe_passed` | `runtime_probe_details.json`, `logs/00_*`, `nvidia-smi`, driver/runtime |
| Python 隔离 | `01_create_venv.sh` | CPython 3.11 venv 创建成功 | 仅 gate evidence | `venv_creation.json`, `logs/01_*` |
| PyTorch cu124 | `02_install_pytorch.sh` | 从 `download.pytorch.org/whl/cu124` 独立安装 2.6.0/0.21.0 | 仅安装 evidence | `pytorch_install.json`, `logs/02_*` |
| Runtime | `03_install_runtime.sh` | 分层 requirements；显式 sgl-kernel 决策；FlashAttention 后装；local forks `--no-deps` | 仅安装 evidence | `runtime_install.json`, `logs/03_*` |
| Import/ABI/lock | `04_import_check.sh` | torch CUDA=12.4；3 GPU/BF16；verl/SGLang/sgl-kernel/FlashAttention/FlashInfer imports；`pip check` | `cuda_runtime_verified`, `requirements_lock_verified` | `import_check.json`, `requirements_validation.json`, `logs/04_*` |
| Ray placement | `05_probe_ray_gpus.sh` | Ray=3 GPU；driver 0 GPU；3 个唯一 1-GPU worker；worker exception propagated | `target_machine_probe_passed` | `ray_gpu_probe_details.json`, `logs/05_*`, Ray logs |
| 单卡 smoke | `06_*`, `07_*` | ≤2 step；rollout/reward/advantage/update；输出 validator 0 | `single_gpu_tested` | `single_gpu_smoke.json`, validation report, run dir |
| 三卡 smoke | `08_*`, `09_*` | 一个 `ray_direct` driver；FSDP DP=3；SGLang TP=1/SP=1；唯一 writer；validator 0 | `three_gpu_ray_tested` | `three_gpu_smoke.json`, Ray/NCCL logs, validator |
| 显存 | smoke telemetry | 所有阶段有采样；每卡保留 `max(2 GiB,10%)`；46 GiB 示例峰值 ≤41.4 GiB | `memory_feasibility_verified` | `single_gpu_memory.json`, `three_gpu_memory.json` |
| Resume | `10_resume_smoke.sh` | 真实 `global_step_<N>`；恢复 step/optimizer/RNG/writer；无重复 part/PK | `single_gpu_tested`（resume evidence） | `resume_smoke.json`, checkpoint sidecar, manifests |
| 汇总 | `11_collect_reports.sh` | 9 份核心报告存在且无 blocked/deferred | 汇总，不替代各 gate | `report_manifest.json`, `logs/11_*` |

## 兼容性判定

- PyTorch 2.6.0+cu124、Python 3.11 patch、driver、glibc、NCCL 和 GPU compute capability 以 runtime probe 为准。
- Transformers 4.51.1 与 Ray 精确版本、NumPy/PyArrow/TensorDict/TorchData 组合由 `pip check`、imports 和训练 smoke 共同验证。
- `sgl-kernel` 0.1.0/0.1.1 冲突必须先显式决策，禁止依赖 resolver 偶然结果。
- FlashAttention 2.7.3、FlashInfer 0.2.3、sgl-kernel 的 wheel/import 仅是 ABI 第一关；真实 kernel launch、SGLang rollout 和 FSDP update 由 smoke 提供最终证据。

## 分布式与存储验收

三卡日志需证明 3 个 Ray GPU worker、FSDP DP=3、SGLang TP=1/SP=1，driver 无训练 GPU、worker 异常传播。输出 validator 需证明唯一权威 writer、append-only parts、schema/nullability、availability/count、主键、manifest/commit 顺序和 resume 无重复；worker mean 不得未经 sufficient-statistics 合并。

## 返回物

返还整个 `artifacts/target_machine/`：10 份规范 JSON（含 manifest）、额外安装报告、detail reports、`logs/`、`runs/`、显存 JSON。只给截图或摘抄日志不能形成验收证据。失败也保留目录；`failure_reason` 与 stderr 是下一轮诊断入口。

