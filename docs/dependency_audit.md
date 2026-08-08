# Latent-GRPO 依赖与 CUDA 兼容性审计

审计日期：2026-08-02。作者仓库 commit：`c0994fb781a2d180662bb522d8ff3e8638dcf56d`。本审计基于仓库静态证据、官方上游元数据和当前控制端只读 probe；未安装/升级/卸载任何包，未执行 `pip freeze`，未启动 CUDA 或训练。

## 1. 结论

- 作者真实训练栈是 **vendored verl 0.4.0 fork + vendored SGLang 0.4.6.post1 fork + Ray + PyTorch FSDP/FSDP2**。两个发布脚本都配置 `rollout.name=sglang`；vLLM/Megatron 不是默认 Latent-GRPO 运行依赖。
- `Latent-GRPO/requirements.txt` 是包含约 200 个直接/传递 pin 的环境快照式文件，内部已有硬冲突，**不能直接安装，不能作为新 requirements 的版本真相**。
- 有证据支持的候选锚点是 CPython 3.11.x、PyTorch 2.6.0 的官方 CUDA 12.4 wheel 家族、Transformers 4.51.1、作者本地 SGLang fork；这不是已验证 lock。
- NVIDIA driver、系统 CUDA toolkit、系统 NCCL/cuDNN、GCC/glibc、GPU 拓扑等不能写入 `requirements.txt`，必须由目标 Linux runtime probe 验证。
- 当前控制端是 Darwin arm64 / Python 3.11.9，无 NVIDIA runtime，且未安装 torch/verl/SGLang/Ray/PyArrow；不能替代目标三卡环境验证。

完整子审计证据见 `work_reports/agent_b_dependency_audit.md`。

## 2. 作者仓库已有依赖资产

### 2.1 Requirements、构建元数据与 conda

| 资产 | 已有内容 | 适用性 |
|---|---|---|
| `Latent-GRPO/requirements.txt:1-206` | 两个 editable 本地包、约 200 个精确 pin、CUDA wheel 与开发包 | 不可直接安装；存在互斥 pin，不能复制或用来反向猜 lock |
| `Latent-GRPO/verl-0.4.x/requirements.txt:1-24` | verl 所称 development 全集；含 GPU/dev 包，vLLM 被注释 | 未 pin，不能单独定义可复现环境 |
| `Latent-GRPO/verl-0.4.x/requirements_sglang.txt:1-22` | 通用 verl SGLang backend | 要求上游 SGLang post5，与作者 post1 fork 冲突 |
| `Latent-GRPO/verl-0.4.x/requirements-npu.txt:1-20` | Ascend/NPU | NVIDIA 目标不使用 |
| `Latent-GRPO/verl-0.4.x/pyproject.toml:4-24,68-87` | setuptools build、Python `>=3.8`、动态 dependencies | 实际依赖由 `setup.py` 给出 |
| `Latent-GRPO/verl-0.4.x/setup.py:26-67` | verl base 和 test/gpu/math/vllm/sglang extras | 重要直接依赖来源；base 要求 Ray `>=2.41`, PyArrow `>=19`, TensorDict `<=0.6.2` |
| `Latent-GRPO/sglang_latent_reasoning_pkg/python/pyproject.toml:1-114` | 作者 SGLang fork `0.4.6.post1` 和 runtime/srt/all extras | pin torch 2.6、vision .21、Transformers 4.51.1、FlashInfer .2.3、sgl-kernel .1.0 |
| `Latent-GRPO/sglang_latent_reasoning_pkg/sgl-kernel/pyproject.toml:1-39` | sgl-kernel 源构建元数据 | 源编译才需要 build tool；优先匹配 wheel |
| 两个 `docs/requirements*.txt` | Sphinx/Jupyter/主题 | docs-only |

全仓未发现 `environment.yml/.yaml`、`conda-lock.yml`、Pipfile、Poetry lock 或 uv lock。只有命令式 conda 说明：顶层 README `:109-112` 建议 Python 3.11.13；vendored verl 文档的 Python 3.10 示例是通用上游材料，不应覆盖顶层作者意图。

### 2.2 安装脚本与 Docker 参考

| 资产 | 证据 | 审计结论 |
|---|---|---|
| `Latent-GRPO/README.md:102-130` | torch 2.6.0、Transformers 4.51.1、sgl-kernel .1.1、FlashAttention 2.7.3、editable 安装两个 fork | 最接近作者意图，但缺 wheel index、driver/toolkit/NCCL/ABI 约束 |
| `verl-0.4.x/scripts/install_vllm_sglang_mcore.sh:3-51` | cu124/torch2.6/cp310/ABI 绑定 wheel，兼装 vLLM/Megatron | 大而全参考；CPython 3.11 不能照抄 cp310 wheel |
| `verl-0.4.x/docker/Dockerfile.sglang:39-55` | 上游 SGLang post5 + torch2.6 | 与作者 post1 fork 不是同一 lock |
| `verl-0.4.x/docker/Dockerfile.vllm.sglang.megatron:46-100` | CUDA toolkit 12.4、cuDNN、Apex、TE、Megatron | 证明系统/Python ABI 耦合；目标明确不用 Docker且默认不用 Megatron |
| SGLang `scripts/ci_install_dependency.sh:8-33` | CI 装 sgl-kernel .1.0、Transformers 4.51.0 | CI/dev 口径，与作者 README 的 .1.1/4.51.1 有差异 |

## 3. 已确认冲突

1. 根清单 `torch==2.6.0` 配 `torchaudio==2.8.0`（`:184,187`）；PyTorch 官方对应 trio 是 torch 2.6.0 / torchvision 0.21.0 / torchaudio 2.6.0。
2. 根清单 `tensordict==0.10.0`（`:179`）违反 vendored verl `tensordict<=0.6.2`（`verl/setup.py:40,51,53`）。
3. README/根清单使用 `sgl-kernel==0.1.1`（README `:112`；requirements `:168`），vendored SGLang `srt` extra 要求 `0.1.0`（SGLang pyproject `:47-53`）。
4. 根清单 `numpy==2.3.3`（`:95`），而 vendored verl CUDA 12.4 参考脚本/镜像采用 `numpy<2.0.0`。精确选择必须由 DataProto/datasets/PyArrow/Ray smoke 决定。
5. vendored SGLang 是 post1，而 `verl[sglang]` 会拉 post5（`verl/setup.py:52-57`）。不得安装该 extra；必须显式 editable 安装作者本地 fork。

## 4. 运行依赖与真实需要方

以下列直接依赖/上游组件；`certifi`、`urllib3`、`attrs` 等传递包由 package metadata/lock 工具解析，不手工逐个 pin。

| 依赖 | 分类 | import/模块/上游需要方 | 当前默认 profile |
|---|---|---|---|
| `-e ./Latent-GRPO/verl-0.4.x` | 本地核心源码 | shell 执行 `-m verl.trainer.main_ppo`；`main_ppo.py:18-154` | 必须，不可换 PyPI |
| `-e ./Latent-GRPO/sglang_latent_reasoning_pkg/python` | 本地核心 Python/CUDA 源码 | `verl/workers/rollout/sglang_rollout/sglang_rollout.py:29-55` import `sglang.srt.*` | 必须，不可换 post5 |
| `torch` | 核心 Python/CUDA | FSDP、distributed、DeviceMesh：`verl/workers/fsdp_workers.py:25-34`；SGLang exact 2.6.0 | 必须；CUDA wheel variant待 probe |
| `torchvision` | SGLang 配套 | SGLang `srt` extra exact 0.21.0 | 保留配套版本 |
| `torchaudio` | vLLM/音频配套 | 可选 vLLM metadata/PyTorch trio；文本 SGLang未直接需要 | 核心可省；若装只能匹配 2.6.0 |
| `transformers` | 模型/tokenizer核心 | `fsdp_workers.py:180-204`, `utils/dataset/rl_dataset.py:24-29`; SGLang exact 4.51.1 | 必须 exact 4.51.1 |
| `ray[default]` | 核心分布式 | `main_ppo.py:18-44,95-154`; verl setup `>=2.41.0` | 必须；精确版待 probe |
| `hydra-core`, `omegaconf` | 核心配置 | `main_ppo.py:18,24,50-55`, `ray_trainer.py:36` | 必须 |
| `numpy`, `pandas`, `tensordict` | 核心数据协议 | `verl/protocol.py:27-35`; rollout uses TensorDict | 必须；TensorDict须 `<=0.6.2` |
| `torchdata` | dataloader/resume | `ray_trainer.py:37-38` imports `StatefulDataLoader` | 必须；精确版待 resume smoke |
| `datasets` | 数据读取/预处理 | `utils/dataset/rl_dataset.py:24`; `data_preprocess_code/*.py:5` | 必须 |
| `pyarrow>=19` | 训练 parquet + 新指标 | verl setup `:35`; storage contract | 必须；精确版待 roundtrip/Ray smoke |
| `accelerate`, `peft`, `safetensors`, `packaging` | FSDP/model/checkpoint | `utils/model.py`, `fsdp_utils.py`, `fsdp_workers.py`, `protocol.py` | base metadata需要；具体版本由 resolver/probe |
| `flash-attn` | CUDA kernel/remove-padding | `utils/torch_functional.py:33`; actor forces flash-attention-2 | 实际核心；2.7.3 ABI待 probe |
| `flashinfer-python` | SGLang attention | SGLang srt exact 0.2.3 | 核心候选；GPU/kernel probe |
| `sgl-kernel` | SGLang CUDA ops | `sglang/srt/_custom_ops.py:14-32` | 核心；.1.0/.1.1待 probe |
| `triton` | SGLang/torch kernels | SGLang attention backend imports | 由 torch组合约束，不独立猜 pin |
| `cuda-python`, `nvidia-cuda-nvrtc-cu12` | Python binding/runtime wheel | SGLang srt extra/grammar编译路径 | 是 Python 包，但不等于系统 CUDA toolkit |
| `codetiming` | 训练计时 | `ray_trainer.py:35`, `fsdp_workers.py:29` | 必须 |
| `wandb`, `tensorboard` | 可选 tracking | 作者 shell `trainer.logger=['console','wandb']`; `utils/tracking.py`动态 import | 新 runner 默认本地 Parquet时移入 optional tracking |
| `math-verify`, `mathruler`, `pylatexenc`, `pyext` | 可选 reward | setup math/geo/prime extras及对应 scorer import | 按 high profile reward manager选择 |
| `fastapi`, `uvicorn`, `openai`, `aiohttp` | async/tool/HTTP可选 | `workers/rollout/async_server.py:26-34` | 同步默认非核心 |
| `vllm<=0.8.5` | 可选 rollout | `fsdp_workers.py:389-435` 条件 import | 作者 Latent profile不安装 |
| Megatron/Apex/TransformerEngine | 可选训练后端/扩展 | `main_ppo.py:84-90` 仅 strategy=megatron | 三卡 FSDP方案不安装 |

## 5. Python distribution 与系统依赖边界

可以出现在 requirements/constraints 中：torch/vision/audio、Transformers、Ray、Hydra、TensorDict、TorchData、Datasets、PyArrow、FlashAttention、FlashInfer、sgl-kernel、cuda-python、`nvidia-*-cu12` distributions、Triton、Ninja 等。即便如此，NVIDIA wheel、Triton、cuDNN/NCCL 应随选定的官方 torch/backend 组合解析，不复制环境快照逐一 pin。

不能写入 `requirements.txt`：

- NVIDIA kernel driver、`libcuda.so`；
- 实体 GPU、compute capability、NVLink/PCIe/P2P 拓扑；
- 系统 CUDA toolkit（nvcc/headers/static libs）与 `CUDA_HOME`；
- 系统 NCCL/cuDNN 和动态链接器路径；
- GCC/G++、libstdc++/GLIBC、CXX11 ABI；
- 系统 CMake/Ninja/Rust/pkg-config/headers（若走源码构建）；
- Linux kernel/distribution、磁盘、`/dev/shm`、ulimit、网络接口。

`cuda-python` 或 `nvidia-cuda-runtime-cu12` 是 Python distributions，不能替代 driver/系统 toolkit；这一点必须在 runbook 明示。

## 6. 测试/开发/文档专用依赖

- test/dev-only：`pytest`, `pre-commit`, `py-spy`（`verl/setup.py:46`）；`ruff`（安装脚本）；SGLang test extra 的 jsonlines/matplotlib/sentence-transformers 等。
- docs-only：两个 docs requirements 中的 Sphinx/Jupyter/主题。
- NPU/ROCm/HPU/XPU/router/多模态/音频/量化/Megatron/云 tracking：当前 NVIDIA 文本 FSDP profile不进入核心；相应路径实际启用后单独 extra。
- `datasets` 同时是 data-prep 和训练运行依赖；不能只放 dev。

## 7. 官方兼容关系与证据

外部来源访问日期均为 2026-08-02；表中“兼容”只表示元数据/官方矩阵闭合，不等于已在目标机验证。

| 关系 | 官方/上游证据 | 结论 |
|---|---|---|
| Python 3.11 | 作者 README 3.11.13；[vLLM 0.8.5.post1 pyproject](https://raw.githubusercontent.com/vllm-project/vllm/v0.8.5.post1/pyproject.toml)、[Transformers 4.51.1](https://pypi.org/project/transformers/4.51.1/)、[Ray 2.49.2](https://pypi.org/project/ray/2.49.2/) | 3.11元数据闭合；精确 patch/glibc/wheel架构待 probe |
| torch/vision/audio | [PyTorch previous versions](https://docs.pytorch.org/get-started/previous-versions/) | 2.6.0/0.21.0/2.6.0；官方有 cu118/cu124/cu126 wheels |
| PyTorch 2.6 ↔ CUDA | 同上；作者相关 wheel多为 cu124 | cu124是候选，不是目标机事实 |
| CUDA 12.x ↔ driver | [NVIDIA CUDA minor compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) | CUDA 12.x最低 driver family >=525，但 PTX/新特性可要求更高；实际 kernel probe必需 |
| NCCL 2.21.5 ↔ CUDA 12.4 | [NCCL 2.21.5 release notes](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2265/release-notes/rel_2-21-5.html) | 官方支持闭合；实际加载库/三卡 collective待 probe |
| Transformers | vendored SGLang exact 4.51.1；[vLLM common requirements](https://raw.githubusercontent.com/vllm-project/vllm/v0.8.5.post1/requirements/common.txt) | 4.51.1满足作者 SGLang与可选vLLM，禁止无上限升级 |
| verl ↔ Ray | vendored setup Ray `>=2.41`; root snapshot 2.49.2 | resolver可满足；没有可靠 tested声明，精确 Ray版本待 probe |
| vLLM ↔ torch/Ray | [vLLM pyproject](https://raw.githubusercontent.com/vllm-project/vllm/v0.8.5.post1/pyproject.toml)与[cuda requirements](https://raw.githubusercontent.com/vllm-project/vllm/v0.8.5.post1/requirements/cuda.txt) | torch2.6 trio；Ray `>=2.43,!=2.44.*`；仅可选profile |
| SGLang post1 ↔ torch/Transformers/kernels | [SGLang v0.4.6.post1 pyproject](https://raw.githubusercontent.com/sgl-project/sglang/v0.4.6.post1/python/pyproject.toml) | torch2.6、vision.21、Transformers4.51.1、FlashInfer.2.3、sgl-kernel.1.0；作者.1.1待 ABI probe |
| verl 0.4 ↔ SGLang | [verl v0.4.0 setup](https://raw.githubusercontent.com/volcengine/verl/v0.4.0/setup.py)要求 post5 | 与作者 post1 fork口径冲突；必须本地 fork + smoke，不能装 extra |
| FlashAttention/kernel ABI | 作者 README 2.7.3；上游 wheel名绑定 cp310/cu124/torch2.6/CXX11ABI=False | CPython3.11轮子/源码编译、nvcc/GCC/ABI未确认 |

## 8. 拟定 requirements 结构

本阶段不创建这些文件；下一阶段只有在 runtime probe 后才填入经证据支持的 constraints。

```text
requirements.txt                 # 只引用 runtime-core、runtime-sglang、metrics
requirements/
  runtime-core.txt               # verl base直接运行依赖
  runtime-sglang.txt             # 作者本地fork与直接SGLang依赖
  metrics.txt                    # pyarrow及runner新增存储依赖
  tracking.txt                   # wandb/tensorboard，可选
  reward-math.txt                # high profile按需
  vllm-optional.txt              # 只有新增vLLM profile时
  test.txt                       # pytest及测试直接依赖
  docs.txt                       # 文档构建按需
constraints/
  linux-cu124-py311.txt          # 仅保存目标机验证组合，不由pip freeze生成
```

规则：PyTorch官方 CUDA index先选定；kernel wheel随后匹配；最后 editable安装两个 fork；不使用 `verl[sglang]`；系统依赖写 runbook/snapshot；传递依赖由 metadata + 受控 lock/哈希工具解析，绝不运行 `pip freeze > requirements.txt`。

## 9. 必须由目标机 probe 确认

| ID | 未确认问题 | 验证条件 |
|---|---|---|
| DEP-U01 | GPU型号/约46GB/compute capability/P2P/BF16 | 三卡查询、拓扑和 BF16 probe |
| DEP-U02 | driver ↔ torch cu124 runtime/PTX | tensor/kernel/FlashAttention/SGLang实际运行 |
| DEP-U03 | Python patch/glibc/arch/wheel availability | 3.11.x及目标 wheels可解析 |
| DEP-U04 | torch local version、cuDNN、实际NCCL | API + 动态库路径记录 |
| DEP-U05 | TensorDict精确版本 | `<=0.6.2`且 DataProto/FSDP smoke |
| DEP-U06 | Ray精确版本 | `pip check`、Ray init、3 GPU分配/异常传播 |
| DEP-U07 | sgl-kernel .1.0/.1.1 | import/custom-op/latent rollout |
| DEP-U08 | FlashAttention 2.7.3 cp311/ABI | import + tiny forward；源码构建时记录nvcc/GCC/ABI |
| DEP-U09 | FlashInfer .2.3/GPU兼容 | import + SGLang最小生成 |
| DEP-U10 | NumPy/PyArrow精确版本 | datasets/Parquet/Ray serialization |
| DEP-U11 | TorchData精确版本 | StatefulDataLoader save/resume |
| DEP-U12 | NCCL三卡 | init/all_reduce/barrier无 hang |
| DEP-U13 | 默认SGLang是否可完全不装vLLM | 无vLLM import/engine/rollout |
| DEP-U14 | 根快照是否曾可运行 | 内部冲突已使其不能作证；只接受目标机实测 |

## 10. 下一阶段首先执行的只读命令

```bash
uname -a
cat /etc/os-release
python3 --version
python3 -c 'import sys,platform; print(sys.executable); print(platform.platform()); print(platform.libc_ver())'
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version,compute_cap --format=csv
nvidia-smi topo -m
nvcc --version
gcc --version
g++ --version
cmake --version
ninja --version
python3 -m pip check
python3 -c 'import importlib.metadata as m; print({n:m.version(n) for n in ["torch","transformers","ray","tensordict","torchdata","datasets","pyarrow"] if any(d.metadata.get("Name","").lower()==n.lower() for d in m.distributions())})'
```

已有候选环境时再执行 import/version probe；依赖未满足时先报告，不在未授权阶段安装。之后新增并运行短时 `probe_distributed.py` 与 `probe_sglang_runtime.py`，仍不启动正式训练。
