# Linux 三卡目标机执行手册

状态：`target_machine_test_deferred`。本手册和脚本已在 Mac 上做静态设计；没有执行 CUDA、Ray GPU、SGLang 或训练。目标机为 Linux、CPython 3.11、3 张约 46 GB NVIDIA GPU，不使用 Docker。以下命令均从项目根目录执行，默认 launcher 为 `ray_direct`，不是 `torchrun`。

## 0. 执行前约束

- 保留 `artifacts/target_machine/` 中的空模板；脚本会用真实报告原子替换对应模板。
- 不在共享生产环境直接安装。目标 venv 固定为 `.venv-target`，可用 `TARGET_VENV=/absolute/path` 改写。
- 不把 NVIDIA driver、CUDA toolkit、NCCL、cuDNN、GCC、glibc 或 apt 包写进 Python requirements。它们由只读 probe 和管理员管理。
- `constraints/linux-cu124-py311.txt` 仍是候选，不是已验证 lock；任何不满足 gate 的步骤立即停止后续训练。
- smoke 固定最多 2 step。脚本不会启动长训，也不会运行 `3gpu-high-smoke`。

先保存代码版本和干净状态：

```bash
git -C ./Latent-GRPO rev-parse HEAD
git -C ./Latent-GRPO status --short
uname -a
cat /etc/os-release
python3 --version
which python3
```

## 1. 只读环境 probe

```bash
bash scripts/target_machine/00_probe_environment.sh
```

它检查 Linux、Python、driver、CUDA runtime、NCCL、3 卡可见性、BF16、每卡至少 40 GiB、包元数据、磁盘和作者仓库路径。另行人工保留拓扑/编译器证据：

```bash
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version,compute_cap --format=csv
nvidia-smi topo -m
nvcc --version
gcc --version
```

只有 `artifacts/target_machine/runtime_probe.json` 的 `exit_code=0` 和 `status=target_machine_probe_passed` 才进入下一步。否则先看相邻 `logs/00_*` 和 `runtime_probe_details.json`。

## 2. 创建隔离 venv

```bash
bash scripts/target_machine/01_create_venv.sh
```

脚本 fail closed：只接受 CPython 3.11。不要用系统 Python 包冒充目标 lock。

## 3. 独立安装 PyTorch cu124

PyTorch CUDA wheel 不在普通 `requirements.txt`，必须先从官方 cu124 index 独立安装：

```bash
bash scripts/target_machine/02_install_pytorch.sh
```

等价核心命令是：

```bash
.venv-target/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0
```

安装成功还不等于 CUDA ABI 通过；第 5 步才执行严格 import/CUDA gate。

## 4. 安装 runtime、特殊扩展和本地 fork

先解决 `sgl-kernel` 冲突：vendored SGLang 元数据声明 0.1.0，而作者 README 给出 0.1.1。不要让 pip 静默替换。结合目标 wheel availability、上游 commit 和维护者结论显式选择：

```bash
export SGL_KERNEL_VERSION=0.1.0  # 示例；必须先完成上述证据核对
bash scripts/target_machine/03_install_runtime.sh
```

脚本顺序固定：

1. `runtime-core.txt`；
2. `metrics.txt`；
3. `reward-math.txt`；
4. `runtime-sglang.txt`（包含 FlashInfer 0.2.3 candidate）；
5. 显式 `sgl-kernel==$SGL_KERNEL_VERSION`；
6. PyTorch 已安装后用 `--no-build-isolation` 安装 FlashAttention 2.7.3 candidate；
7. `pip install --no-deps -e Latent-GRPO/verl-0.4.x`；
8. `pip install --no-deps -e Latent-GRPO/sglang_latent_reasoning_pkg`。

本地 forks 必须带 `--no-deps`，否则其元数据可能覆盖已审计组合。FlashAttention、FlashInfer、sgl-kernel 都是 CUDA/torch/Python/glibc/C++ ABI 敏感包；wheel 能安装不表示 kernel 能运行。

## 5. import、CUDA 与 requirements gate

```bash
bash scripts/target_machine/04_import_check.sh
```

该步骤加载 torch、Ray、Transformers、PyArrow、TensorDict、TorchData、vendored verl/SGLang、sgl-kernel、FlashAttention、FlashInfer，要求 torch 报告 CUDA 12.4、3 张可见 GPU、BF16，并执行 `pip check`。只有 import/ABI gate 与 `pip check` 同时成功，`requirements_validation.json` 才能写为 `requirements_lock_verified`。真实 CUDA kernel 仍由后续 smoke 证明；不能仅凭 import 给出训练结论。

## 6. Ray 三卡 placement probe

```bash
bash scripts/target_machine/05_probe_ray_gpus.sh
```

检查 Ray 发现 3 GPU、driver 的 `ray.get_gpu_ids()` 为空、3 个每卡 1 GPU worker 获得不重复绑定，并验证 worker 异常能传播回 driver。

## 7–8. 单卡两步 smoke 与输出验证

```bash
bash scripts/target_machine/06_single_gpu_smoke.sh
bash scripts/target_machine/07_validate_single_gpu_output.sh
```

训练命令固定为 `CUDA_VISIBLE_DEVICES=0 python train_latent_grpo.py --config configs/smoke.yaml --max-steps 2`（脚本使用 venv Python并覆盖输出目录）。显存采样写入 `single_gpu_memory.json`。训练退出 0 之后仍必须由 validator 证明 schema、主键、availability 和提交一致性。

## 9–10. 三卡 Ray 两步 smoke 与输出验证

```bash
bash scripts/target_machine/08_three_gpu_smoke.sh
bash scripts/target_machine/09_validate_three_gpu_output.sh
```

主路径是：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
.venv-target/bin/python train_latent_grpo.py \
  --config configs/3gpu-low.yaml \
  --max-steps 2
```

这是单个 `ray_direct` driver；不要改成 `torchrun --nproc_per_node=3`。核对 FSDP DP=3、SGLang TP=1/SP=1、3 个 worker 绑定、driver 不占训练 GPU、仅一个权威 writer、worker 错误传播。`three_gpu_memory.json` 必须显示每卡峰值不超过实测容量减 `max(2 GiB, 10%)`；以恰好 46 GiB 为例上限是 41.4 GiB。只有训练、validator 与显存证据都成立，才能分别提升 `three_gpu_ray_tested` 和 `memory_feasibility_verified`。

## 11. Resume smoke

当前 outer profile 不擅自覆盖作者 `trainer.save_freq=-1`，所以脚本不会伪造 checkpoint。先由已批准的 checkpoint-producing smoke 得到一个经校验的 `global_step_<N>`（必须 `N < 2`），再执行：

```bash
export RESUME_CHECKPOINT=/absolute/path/to/global_step_1
bash scripts/target_machine/10_resume_smoke.sh
```

未设置 `RESUME_CHECKPOINT` 时脚本有意失败并留下 `resume_smoke.json`。成功后检查 global/optimizer step、RNG、writer part 序号、manifest、无重复主键；未完成 checkpoint 前保持 `blocked` 或 `target_machine_test_deferred`，不可声称 resume 已验收。

## 12. 收集并返还完整目录

即使前一步失败，也执行：

```bash
bash scripts/target_machine/11_collect_reports.sh
tar -czf target_machine_artifacts.tgz artifacts/target_machine/
```

请返还整个 `artifacts/target_machine/`（或该 tarball），不要只返还终端截图。目录包含 JSON envelope、详细 probe、完整 stdout/stderr、训练输出、validator 结果和显存采样。

## 失败诊断顺序

1. 对应 JSON 的 `failure_reason`、`exit_code`、`command`；
2. 对应 `stdout_log_path` 和 `stderr_log_path`；
3. `runtime_probe_details.json` / `ray_gpu_probe_details.json`；
4. `python -m pip check` 输出及 import 报告中的精确包版本；
5. `nvidia-smi topo -m`、driver/runtime/NCCL；
6. FlashAttention/FlashInfer/sgl-kernel import 或 symbol 错误，结合 torch CUDA、CXX11 ABI、Python tag、glibc 和编译器排查；
7. smoke 训练目录、validator 错误、writer manifest/临时 part；
8. `single_gpu_memory.json` / `three_gpu_memory.json` 定位 OOM 阶段。

