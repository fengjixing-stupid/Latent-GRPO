# Agent B：Python / CUDA 依赖审计

审计日期：2026-08-02（Asia/Shanghai）  
审计性质：只读静态审计 + 当前控制端只读 runtime probe；未安装依赖、未执行 `pip freeze`、未启动 CUDA 训练。  
作者仓库：`./Latent-GRPO`，commit `c0994fb781a2d180662bb522d8ff3e8638dcf56d`。工作树原有未跟踪文件未改动。

## 1. 结论先行

1. 作者发布的真实训练面是 `python3 -m verl.trainer.main_ppo`，由 Hydra 解析配置、Ray 创建 driver/worker、PyTorch FSDP/FSDP2 训练、自定义 SGLang rollout；两份作者训练脚本都明确配置 `rollout.name=sglang`，而不是 vLLM（`Latent-GRPO/Latent-GRPO-gsm8k-llama3.sh:8,35-37`；`Latent-GRPO/Latent-GRPO-math500-qwen.sh:8,35-37`）。Ray 在入口中是无条件运行时依赖（`Latent-GRPO/verl-0.4.x/verl/trainer/main_ppo.py:18-19,29-37,44`）。
2. 当前根 `Latent-GRPO/requirements.txt` **不能直接安装，也不能作为新 requirements 的版本真相**。至少存在四组硬冲突：
   - `torch==2.6.0` 配 `torchaudio==2.8.0`（根清单 `:184,187`），PyTorch 官方对应组合应为 `torch 2.6.0 / torchvision 0.21.0 / torchaudio 2.6.0`；
   - `tensordict==0.10.0`（根清单 `:179`）违反 vendored verl 的 `tensordict<=0.6.2`（`verl-0.4.x/setup.py:40,51,53`）；
   - 作者 README/根清单选择 `sgl-kernel==0.1.1`（`README.md:112`；根清单 `:168`），但 vendored SGLang `srt` extra 声明 `sgl-kernel==0.1.0`（`sglang_latent_reasoning_pkg/python/pyproject.toml:47-53`）；
   - 根清单 `numpy==2.3.3`（`:95`），而 verl 的 CUDA 12.4 参考安装脚本/镜像使用 `numpy<2.0.0`（`verl-0.4.x/scripts/install_vllm_sglang_mcore.sh:15-17`；`docker/Dockerfile.vllm.sglang.megatron:68-70`）。
3. vendored SGLang 是作者修改版 `0.4.6.post1`（`sglang_latent_reasoning_pkg/python/pyproject.toml:5-10`）；vendored verl 是修改版 `0.4.0`（`verl-0.4.x/verl/version/version:1`）。**不得以同名 PyPI 包替换这两个 editable 本地源码，也不得安装 `verl[sglang]`**：后者声明上游 `sglang==0.4.6.post5`（`verl-0.4.x/setup.py:52-57`），会绕开作者修改版。
4. vLLM 是 verl 的可选 rollout backend，不是作者两份 Latent-GRPO 配置的必装项。worker 仅当 `rollout_name == "vllm"` 时延迟 import vLLM；`sglang` 分支另行延迟 import 自定义 SGLang（`verl-0.4.x/verl/workers/fsdp_workers.py:389-420,438-464`）。SGLang 中的少量 vLLM import 是可选 custom-allreduce/GGUF 兼容路径并有 `try/except ImportError`（`sglang_latent_reasoning_pkg/python/sglang/srt/_custom_ops.py:10-25`；`.../srt/utils.py:623-640,715-723`）。
5. 作者仓库的共同兼容锚点是 **CPython 3.11.x + PyTorch 2.6.0 CUDA 12.4 wheel 家族 + Transformers 4.51.1 + 自定义 SGLang 0.4.6.post1**。但这只是“元数据可满足且与作者 README 一致”的候选基线，不是已经在目标 3-GPU 机器验证的 lock。GPU 型号/compute capability、驱动、实际 PyTorch wheel local tag、CUDA runtime/toolkit、cuDNN、NCCL、FlashAttention/FlashInfer/sgl-kernel ABI、Ray 进程与三卡 collective 均须 runtime probe 后才能确认。
6. 当前控制端是 macOS/CPython 3.11.9，未安装 torch/verl/sglang/ray 等，且无 `nvidia-smi`、`nvcc`；这里只能完成静态审计，不能为目标 Linux CUDA 机器背书。

## 2. 仓库依赖与安装资产清单

### 2.1 清单/构建元数据

| 文件 | 内容与用途 | 审计结论 |
|---|---|---|
| `Latent-GRPO/requirements.txt:1-206` | 两个 editable 本地包 + 约 200 个精确版本（含大量传递依赖和 CUDA wheel） | 很像一次环境快照；存在上述互斥精确 pin，禁止直接复用或由其反向生成新 requirements。不是 `pip freeze` 的授权替代品。 |
| `Latent-GRPO/verl-0.4.x/requirements.txt:1-24` | verl “full set of dependencies for development” | 主体未 pin；含 `pre-commit` dev-only、`flash-attn`/`liger-kernel` GPU optional、注释掉的 vLLM；不能单独定义可复现 CUDA 环境。 |
| `Latent-GRPO/verl-0.4.x/requirements_sglang.txt:1-22` | SGLang backend 参考 | 声明 `ray[default]>=2.10`、`tensordict<=0.6.2`、`sglang[all]==0.4.6.post5`；与作者 vendored post1 冲突。 |
| `Latent-GRPO/verl-0.4.x/requirements-npu.txt:1-20` | Ascend/NPU 路径 | NVIDIA 目标环境不使用；`torch_npu` 的实际 import 是可选探测（`verl/utils/device.py:18-29`）。 |
| `Latent-GRPO/verl-0.4.x/pyproject.toml:4-24,68-87` | verl 构建系统、Python 下界、包数据 | `setuptools>=61`, `wheel`; 声明 Python `>=3.8`，依赖由 setup 动态提供。 |
| `Latent-GRPO/verl-0.4.x/setup.py:15-67,73-92` | verl 实际 `install_requires` 与 extras | 最重要的 Python 依赖源；base 要求 Ray `>=2.41.0`、PyArrow `>=19`、TensorDict `<=0.6.2`；test/gpu/math/vllm/sglang 分层明确。 |
| `Latent-GRPO/sglang_latent_reasoning_pkg/python/pyproject.toml:1-114` | 作者 vendored SGLang 的包元数据与 extras | `runtime_common` pin Transformers 4.51.1；`srt` pin torch 2.6.0、torchvision 0.21.0、FlashInfer 0.2.3、sgl-kernel 0.1.0。作者 README/根清单对 sgl-kernel 的 0.1.1 修改必须实机核验。 |
| `Latent-GRPO/sglang_latent_reasoning_pkg/sgl-kernel/pyproject.toml:1-21,33-39` | sgl-kernel 源码构建元数据 | build-only：`scikit-build-core>=0.10`、`torch>=2.5.1`、`wheel`；要求 Python `>=3.9`、CUDA GPU。目标环境优先使用与 torch/CUDA/ABI 匹配的上游 wheel，不默认本地编译。 |
| `Latent-GRPO/sglang_latent_reasoning_pkg/sgl-router/pyproject.toml:1-29` | 独立 Rust router | 训练入口未见引用；不是本任务单机三卡核心依赖。源码构建才需要 Rust/setuptools-rust。 |
| `Latent-GRPO/sglang_latent_reasoning_pkg/docs/requirements.txt:1-20` | SGLang 文档构建 | docs-only，不进入训练 requirements。 |
| `Latent-GRPO/verl-0.4.x/docs/requirements-docs.txt:1-13` | verl 文档构建 | docs-only，不进入训练 requirements。 |

全仓未发现 `environment.yml` / `environment.yaml` / `conda-lock.yml` / Pipfile / Poetry lock / uv lock。存在的 conda 内容只是 README/安装文档中的命令，不是可解析、可锁定的环境清单：作者 README 指定 `conda create -n latent_grpo python=3.11.13`（`Latent-GRPO/README.md:102-117`）；vendored verl 文档另给旧的 Python 3.10 示例（`verl-0.4.x/docs/start/install.rst:164-180`）。本项目应以作者顶层 README 的 3.11.x 意图为先，再由目标机 wheel 可用性验证精确 patch。

### 2.2 安装脚本与 Docker 参考（只做证据，不作为目标执行面）

| 文件 | 关键证据 | 结论 |
|---|---|---|
| `Latent-GRPO/README.md:102-130` | Python 3.11.13；torch 2.6.0；Transformers 4.51.1；sgl-kernel 0.1.1；FlashAttention 2.7.3；editable `python[all]` 与 editable verl | 最接近作者项目的安装意图，但仍缺 CUDA index、driver/toolkit/NCCL/ABI 约束。 |
| `verl-0.4.x/scripts/install_vllm_sglang_mcore.sh:3-18` | 同时安装 SGLang post1、vLLM 0.8.5.post1、torch 2.6 stack、Ray；默认 Megatron | 面向“大而全”上游环境，不适合作者当前 SGLang+FSDP 最小面。 |
| 同上 `:23-39` | cp310/cu124/torch2.6/CXX11ABI=False 的 FlashAttention/FlashInfer wheel；可选 TransformerEngine/Megatron 源码 | 这些 wheel 是 Python 版本、CUDA 和 ABI 三重绑定，不能拿到 CPython 3.11 目标机直接照抄。 |
| 同上 `:43-51` | OpenCV fix 与 Megatron cuDNN wheel | 当前文本模型 + FSDP 路径不应默认引入；只在对应 backend/模型启用后安装。 |
| `verl-0.4.x/docker/Dockerfile.sglang:39-55` | SGLang post5 + torch 2.6 trio + FlashAttention 2.7.4.post1 + nvidia-ml-py | 是上游 SGLang/FSDP 参考，版本不是作者 post1 fork 的直接 lock。 |
| `verl-0.4.x/docker/Dockerfile.vllm.sglang.megatron:46-100` | CUDA toolkit 12.4、torch/cu124 ABI、vLLM/SGLang、cuDNN 9.8、Apex、TE、Megatron | 证明系统层和 Python 层必须协同，但目标明确“不用 Docker”，且当前 FSDP/SGLang 不需要整套 Megatron。 |
| `verl-0.4.x/docker/Dockerfile.ngc.vllm0.8:46-71` | vLLM 0.8.3 与 FlashInfer 0.2.2 组合 | 仅 vLLM 旧参考；不能拿来覆盖作者 SGLang post1 的 FlashInfer 0.2.3。 |
| `sglang_latent_reasoning_pkg/scripts/ci_install_dependency.sh:8-33` | CI 会卸载 vLLM、装 sgl-kernel 0.1.0、editable SGLang all、Transformers 4.51.0、cu12 NVRTC | CI/dev 路径，且与作者 README 的 4.51.1 / sgl-kernel 0.1.1 不一致。 |
| `sglang_latent_reasoning_pkg/docker/Dockerfile:1-46` | Python 3.10、CUDA 11.8/12.1/12.4/12.5 分支、torch wheel index 与 FlashInfer wheel index | 泛 SGLang 镜像，不是 Latent-GRPO 精确环境。 |

## 3. 直接运行时依赖到真实代码路径的映射

下表只列直接依赖或上游组件；根清单中的 `certifi`、`urllib3`、`attrs` 等传递包由上游包元数据解析，不应被手工逐个精确 pin。

| 包/组件 | 分类 | 谁需要、证据 | 当前 SGLang 训练面 |
|---|---|---|---|
| 本地 `verl==0.4.0` fork | 核心 Python 源码 | shell 入口 `python3 -m verl.trainer.main_ppo`（两训练脚本 `:8`）；版本文件 `verl/version/version:1` | 必须 editable 本地源码；不能替换为 PyPI。 |
| 本地 `sglang==0.4.6.post1` fork | 核心 Python/CUDA 源码 | rollout 直接 import `sglang.srt.*`（`verl/workers/rollout/sglang_rollout/sglang_rollout.py:29-55`） | 必须 editable 本地源码。 |
| `torch==2.6.0` + CUDA wheel variant | 核心 Python/CUDA | FSDP、distributed、device mesh（`verl/workers/fsdp_workers.py:25-34`）；SGLang package exact pin（SGLang pyproject `:47-53`） | 必须；wheel variant 待 probe。 |
| `torchvision==0.21.0` | SGLang/vLLM 配套，文本配置可能不直接用 | SGLang srt exact pin（pyproject `:51-52`）；vLLM 0.8.5 metadata 也同 pin | 保留配套版本，避免 resolver/ABI 不一致。 |
| `torchaudio==2.6.0` | vLLM/多模态配套；当前文本训练非直接 import | vLLM 0.8.5 CUDA metadata exact pin；PyTorch 官方 trio | 若不装 vLLM且不启用音频路径可不必装；绝不能用根清单 2.8.0。 |
| `transformers==4.51.1` | 核心 Python | model/tokenizer imports（`verl/workers/fsdp_workers.py:180-204`；`verl/utils/dataset/rl_dataset.py:24-29`）；SGLang exact pin（pyproject `:40-44`） | 必须 exact 4.51.1。 |
| `ray[default]` | 核心分布式 Python/C++ wheel | 训练入口与 actor orchestration（`main_ppo.py:18-19,29-44,95-104`）；verl setup `>=2.41.0`（`:38`） | 必须；精确版本需 runtime probe。 |
| `hydra-core` / `omegaconf` | 核心配置 | `hydra.main`（`main_ppo.py:18,24`）；OmegaConf（`main_ppo.py:50-55`、`ray_trainer.py:36`） | 必须。 |
| `numpy`, `pandas`, `tensordict` | 核心 batch/protocol | `verl/protocol.py:27-35`；TensorDict 还用于 SGLang rollout（`sglang_rollout.py:52`） | 必须；TensorDict 必须先按 verl `<=0.6.2` 收敛，不能用 0.10.0。 |
| `torchdata` | 核心 dataloader/resume | `StatefulDataLoader`（`verl/trainer/ppo/ray_trainer.py:37-38`） | 必须；版本与 torch 2.6 组合需 probe。 |
| `datasets` | 核心数据读取/预处理 | RL dataset import（`verl/utils/dataset/rl_dataset.py:24`）；预处理脚本（`data_preprocess_code/*.py:5`） | 必须。 |
| `pyarrow>=19.0.0` | 数据 parquet + 新指标存储 | verl setup `:35`；spec 要求权威 Parquet dataset | 必须；根清单 21.0.0 仅候选，需 Ray/datasets/Parquet smoke。 |
| `accelerate`, `peft`, `safetensors`, `packaging` | 模型/FSDP 工具 | accelerate 在 `verl/utils/model.py:279`、`fsdp_utils.py:52`；PEFT/safetensors 在 `fsdp_workers.py:31-32`；packaging 在 `protocol.py:33` | accelerate/packaging 核心；PEFT 可延迟但 setup base 需要；safetensors checkpoint 路径需要。 |
| `flash-attn` | GPU kernel / remove-padding | `verl/utils/torch_functional.py:33`；模型配置强制 `attn_implementation="flash_attention_2"`（`fsdp_workers.py:204`） | 实际核心；其 wheel/source build 必须与 Python、torch、CUDA、CXX11 ABI 对齐。 |
| `flashinfer-python` | SGLang attention kernel | SGLang srt extra exact 0.2.3（pyproject `:47-53`）；attention backend imports `flashinfer` | 核心候选；精确 0.2.3 与目标 GPU/runtime smoke 后确认。 |
| `sgl-kernel` | SGLang custom CUDA ops | custom all-reduce 默认 import（`sglang/srt/_custom_ops.py:14-32`） | 核心；0.1.0 vs 作者 0.1.1 冲突待 probe。 |
| `triton` | SGLang kernels；由 torch 组合约束 | attention 实际 import（`sglang/srt/layers/attention/utils.py:1-5`；`triton_backend.py:7-24`） | 必须但不应脱离 torch wheel单独猜 pin；根快照 3.2.0 可作为探测期望。 |
| `cuda-python`, `nvidia-cuda-nvrtc-cu12` | Python binding/runtime wheel | SGLang srt extra（pyproject `:53`）；CI 为 xgrammar 编译装 NVRTC（CI script `:28-29`） | Python 包，但不等于系统 CUDA toolkit/driver。是否必须显式列出由最终 SGLang install strategy 决定。 |
| `codetiming` | 训练 timing | `ray_trainer.py:35`、`fsdp_workers.py:29` | 必须。 |
| `wandb` / `tensorboard` | 可选观测后端 | 作者脚本启用 `['console','wandb']`（两 shell `:61`）；动态 import（`verl/utils/tracking.py:51-58,118-119,202-207`） | 当前作者脚本下 wandb 是运行时；新 runner 若默认 console/本地 Parquet，可移入 optional `tracking`。 |
| `math-verify`, `mathruler`, `pylatexenc`, `pyext` | reward family 可选 | `math_verify` reward import；`mathruler` geo scorer；`pylatexenc` prime math；setup extras `:47-50` | GSM8K regex scorer不需要；Math profile按真实 reward manager选择。不可一律当核心。 |
| `fastapi`, `uvicorn`, `starlette`, `openai`, `aiohttp` | async/HTTP/tool rollout optional | async server imports（`verl/workers/rollout/async_server.py:26-34`） | 同步作者 SGLang路径多为上游 extra；若新 runner不启 async/tool可不列核心。 |
| `vllm` | optional rollout backend | 条件分支（`fsdp_workers.py:406-435`）；verl extra `vllm<=0.8.5`（setup `:51`） | 作者配置不需要；单独 optional extra。 |
| Megatron-LM / Apex / TransformerEngine | optional training backend/system extension | 只有 strategy=megatron 才 import（`main_ppo.py:84-90`）；安装脚本 `:35-39` | FSDP/FSDP2 三卡方案不安装。 |

## 4. 测试、开发、文档与非当前后端依赖

- test/dev-only：`pytest`, `pre-commit`, `py-spy` 来自 `verl/setup.py:46`；`ruff` 只出现在安装脚本 `:18`；SGLang test extra 的 `jsonlines`, `matplotlib`, `sentence-transformers`, `accelerate`, `peft` 在 SGLang pyproject `:96-103`。`pytest` 的实际测试 import 见 `verl/tests/test_protocol.py:17-22` 等。
- docs-only：两个 docs requirements 文件中的 Sphinx/Jupyter/主题/tokenizers pin；不能进入训练 requirements。
- data-prep-only：直接依赖 `datasets`（两个 `data_preprocess_code/*.py:5`），但 `datasets` 同时也是训练依赖。
- eval-only：作者独立评估脚本直接使用 torch multiprocessing、tqdm、Transformers（`eval/eval_*_sglang.py:12-14`），这些包已被训练核心覆盖。
- conversion-only：`transformer_fsdp_to_safentensor.py:8-10` 需要 torch/Transformers。
- NPU/ROCm/HPU/XPU/router/多模态/音频/量化/云端 tracking 等均不进入 NVIDIA 文本 Latent-GRPO 最小面；只有 profile 或代码路径实际启用时再加对应 extra。

## 5. Python 包与系统/驱动/ABI 依赖的边界

### 5.1 可以出现在 requirements/constraints 中的 Python distribution

`torch`, `torchvision`, `torchaudio`, `transformers`, `ray[default]`, `hydra-core`, `omegaconf`, `tensordict`, `torchdata`, `datasets`, `pyarrow`, `accelerate`, `peft`, `flash-attn`, `flashinfer-python`, `sgl-kernel`, `cuda-python`, `nvidia-cuda-nvrtc-cu12`, `nvidia-nccl-cu12`, `nvidia-cudnn-cu12`, `triton`, `ninja` 等都是 Python package index 可见 distribution（其中若干携带本地动态库或命令行工具）。

但“可以由 pip 分发”不意味着应手工全部 pin：`nvidia-*-cu12`、Triton、NCCL/cuDNN wheel 应随已选择的官方 PyTorch/vLLM/SGLang wheel 约束解析；手动复制根快照会造成 ABI 漂移。`ninja`/`cmake`/`scikit-build-core` 是源码构建工具，只在没有匹配 wheel、明确选择源码构建时进入 build requirements。

### 5.2 不能写入 requirements.txt 的系统层

- NVIDIA kernel driver、`libcuda.so`；
- 实体 GPU、compute capability、NVLink/PCIe 拓扑、IOMMU/P2P 能力；
- 系统 CUDA toolkit（`nvcc`、headers、静态库）、`CUDA_HOME`；pip 的 `cuda-python`/`nvidia-cuda-runtime-cu12` 不能替代它；
- 系统 NCCL/cuDNN 安装及动态链接器搜索路径（即使也存在带 shared library 的 Python distribution）；
- GCC/G++ 及 libstdc++/GLIBC、CXX11 ABI；
- CMake/Ninja/Rust/pkg-config/libssl-dev 等系统构建工具（可有 Python/用户态替代，但必须明确工具来源）；
- 磁盘、`/dev/shm`、进程/文件句柄限制、网络接口；
- Linux 内核、发行版与 glibc 版本。

这些必须写入 platform probe / operator runbook，而不是伪装成 pip requirements。

## 6. 兼容矩阵（证据级别明确）

外部来源均为官方项目或官方包注册表，访问日期均为 **2026-08-02**。

| 关系 | 一手证据 | 审计判断 |
|---|---|---|
| Python 3.11 | 作者 README 精确写 3.11.13（`README.md:109-112`）；vLLM 0.8.5.post1 要求 Python `>=3.9,<3.13`（[vLLM pyproject](https://raw.githubusercontent.com/vllm-project/vllm/v0.8.5.post1/pyproject.toml)）；Transformers 4.51.1 要求 Python `>=3.9`（[PyPI 4.51.1](https://pypi.org/project/transformers/4.51.1/)）；Ray 2.49.2 发布 CPython 3.11 wheel（[PyPI Ray 2.49.2](https://pypi.org/project/ray/2.49.2/)） | 3.11.x 元数据兼容；优先作者 3.11.13，目标机仍检查 wheel 架构/glibc。 |
| torch/vision/audio | 作者/SGLang pin torch 2.6.0 + vision 0.21.0；[PyTorch 官方历史版本](https://docs.pytorch.org/get-started/previous-versions/)给出 2.6.0/0.21.0/2.6.0，且有 cu118/cu124/cu126 wheel | 根清单 torchaudio 2.8.0 是错误组合。CUDA 12.4 是作者相关 wheel最一致候选。 |
| PyTorch 2.6 ↔ CUDA | PyTorch 官方提供 cu124/cu126 wheel；作者/verl wheel URL多为 cu124；vendored verl custom env 要求 CUDA >=12.4（`docs/start/install.rst:98-112`） | 候选 `torch==2.6.0` 的官方 cu124 wheel；系统 toolkit并非运行 wheel必然需要，但 FlashAttention源码编译需要。 |
| CUDA 12.x ↔ driver | [NVIDIA CUDA minor compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)给出 CUDA 12.x 最低 driver family >=525，并警告 PTX/新特性可要求更新 driver | 不能只检查 `nvidia-smi` 顶部“CUDA Version”；必须实际 import/分配/编译 probe。作者 CUDA 12.4 installer示例包含 550.54.15（verl install doc `:116-123`），但不据此硬写目标 driver pin。 |
| NCCL 2.21.5 ↔ CUDA 12.4 | 根快照是 `nvidia-nccl-cu12==2.21.5`（requirements `:110`）；[NVIDIA NCCL 2.21.5 release notes](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2265/release-notes/rel_2-21-5.html)明确支持 CUDA 12.4 | 元数据/官方支持闭合；实际加载哪个 NCCL、三卡 collective/P2P 仍须 probe，不能只看 pip metadata。 |
| Transformers | vendored SGLang exact `4.51.1`（pyproject `:41`）；vLLM 0.8.5.post1要求 `>=4.51.1`（[vLLM official requirements](https://raw.githubusercontent.com/vllm-project/vllm/v0.8.5.post1/requirements/common.txt)） | `4.51.1` 同时满足作者 SGLang与可选 vLLM；不要无上限升级。 |
| verl 0.4.0 ↔ Ray | vendored setup要求 `ray[default]>=2.41.0`（setup `:38`）；root快照 2.49.2（requirements `:156`） | resolver 可满足，但未找到上游对 2.49.2 的明确 tested 声明；2.49.2 是作者快照线索，标记 **unconfirmed/runtime probe required**。 |
| vLLM 0.8.5.post1 ↔ torch/Ray | [vLLM 0.8.5.post1 pyproject](https://raw.githubusercontent.com/vllm-project/vllm/v0.8.5.post1/pyproject.toml) build pin torch 2.6.0；[CUDA requirements](https://raw.githubusercontent.com/vllm-project/vllm/v0.8.5.post1/requirements/cuda.txt) pin torch 2.6/audio 2.6/vision 0.21/xformers 0.0.29.post2，并要求 Ray cgraph `>=2.43,!=2.44.*` | root Ray 2.49.2在区间内；但 vLLM不是作者 SGLang运行面的必装包。需要 vLLM profile时单独 probe，不能污染核心环境。 |
| SGLang 0.4.6.post1 ↔ torch/Transformers/FlashInfer | [SGLang 官方 tag pyproject](https://raw.githubusercontent.com/sgl-project/sglang/v0.4.6.post1/python/pyproject.toml)与 vendored文件一致：torch 2.6、vision .21、Transformers 4.51.1、FlashInfer .2.3、sgl-kernel .1.0 | torch/Transformers/FlashInfer关系闭合；作者改用 sgl-kernel .1.1 的理由/ABI未在 metadata闭合，**runtime probe required**。 |
| verl 0.4.0 ↔ SGLang | [verl v0.4.0 setup](https://raw.githubusercontent.com/volcengine/verl/v0.4.0/setup.py) extra要求 SGLang post5；作者明确 vendored post1 fork | 通用上游 extra与作者算法 fork冲突。正确做法是安装作者本地 fork且不启 `verl[sglang]`；功能兼容必须以作者 smoke为准。 |
| FlashAttention/FlashInfer/sgl-kernel ABI | 作者 README给 FlashAttention 2.7.3（`:112-120`）；上游脚本的预编译文件名把 cp310+cu124+torch2.6+CXX11ABI=False写死（install script `:23-32`） | CPython 3.11目标不能直接用该 cp310 FlashAttention wheel；是否有匹配 cp311 wheel、是否源码编译、编译器/CUDA_HOME均 **unconfirmed/runtime probe required**。 |

注意：PyPI 当前把 vLLM 0.8.5.post1 列为包含已知安全问题的旧版本。这里讨论的是作者算法栈兼容，不代表建议把旧 vLLM 开放成外网服务；而作者当前 SGLang rollout 也无需安装它。

## 7. 拟定 requirements 结构（本阶段不创建、不安装）

建议不要再维护一个混合 200 个直接/传递/系统依赖的单文件。下一阶段在 runtime probe 后生成以下分层：

```text
requirements/
├── runtime-core.txt          # verl base直接运行依赖；不含GPU backend与tracking
├── runtime-sglang.txt        # 作者本地SGLang fork + SGLang direct Python deps
├── metrics.txt               # pyarrow及新runner真实新增的存储依赖
├── tracking.txt              # wandb/tensorboard，可选
├── reward-math.txt           # math-verify/pylatexenc等，仅高难profile
├── vllm-optional.txt         # 只有新增vLLM profile时
├── test.txt                  # pytest及测试直接依赖
└── docs.txt                  # 文档构建，若需要
constraints/
└── linux-cu124-py311.txt     # runtime probe通过后记录已验证组合，不由pip freeze生成
```

原则：

1. 顶层安装顺序明确选择 PyTorch 官方 cu124 index；随后安装匹配 kernel wheel；最后 editable 安装 `./Latent-GRPO/sglang_latent_reasoning_pkg/python` 与 `./Latent-GRPO/verl-0.4.x`。两个本地包路径只出现一次。
2. `verl` 不带 `[sglang]` extra；SGLang必须指向作者 vendored fork。
3. constraints 只收录经 `pip check`、import、单GPU kernel、Ray三卡 collective与最小 rollout验证的版本。传递依赖由 package metadata解析，再用受控 lock 工具/哈希锁定；绝不使用 `pip freeze > requirements.txt`。
4. Driver/CUDA toolkit/system NCCL/GCC等写入 `docs/operator_runbook.md` 与 `platform_config_snapshot.json`，不写进 requirements。
5. vLLM、Megatron、Apex、TransformerEngine、NPU/ROCm、多模态/音频、文档、开发工具不进入核心训练文件。

## 8. 尚未确认的问题（必须阻断安装/训练前闭合）

| ID | 未确认项 | 通过条件 |
|---|---|---|
| DEP-U01 | 目标 GPU型号、显存、compute capability、三卡拓扑/P2P/BF16 | 3卡均可见、内存约46GB、BF16支持、设备映射与P2P结果记录。 |
| DEP-U02 | driver与torch cu124 runtime/PTX兼容 | CUDA tensor分配、简单kernel、FlashAttention/SGLang kernel实际运行成功。 |
| DEP-U03 | Python精确patch与Linux/glibc/arch | Python 3.11.x；所有目标 wheel可用；记录glibc/平台tag。 |
| DEP-U04 | torch wheel local version、cuDNN、实际加载NCCL | `torch.__version__`, `torch.version.cuda`, cuDNN/NCCL probe与动态库路径一致。 |
| DEP-U05 | TensorDict精确版本 | 必须满足 `<=0.6.2` 并通过 DataProto/StatefulDataLoader/FSDP smoke；根0.10.0拒绝。 |
| DEP-U06 | Ray精确版本 | `pip check`、local Ray init、3 worker GPU分配、异常传播、resume dataloader smoke。2.49.2暂不声明 verified。 |
| DEP-U07 | sgl-kernel 0.1.0 vs 0.1.1 | import、custom ops、作者 latent rollout与三卡权重同步通过后确定。 |
| DEP-U08 | FlashAttention 2.7.3的cp311 wheel/源码构建与ABI | import + tiny forward；若源码构建，记录nvcc/GCC/CXX11 ABI。 |
| DEP-U09 | FlashInfer 0.2.3与作者修改SGLang、目标GPU | import + SGLang engine最小生成；不得用安装脚本的0.2.2覆盖后假称一致。 |
| DEP-U10 | NumPy/PyArrow精确版本 | 先解决 `<2`参考与根2.3.3冲突；Parquet roundtrip、datasets读取、Ray serialization通过。 |
| DEP-U11 | torchdata精确版本 | `StatefulDataLoader` import、checkpoint/resume smoke通过。 |
| DEP-U12 | NCCL三卡运行 | `torch.distributed` NCCL init/all_reduce/barrier通过，无hang；记录NCCL debug与拓扑。 |
| DEP-U13 | SGLang是否隐式需要vLLM | 用作者默认环境（无vLLM）完成 import/engine/rollout；若失败，定位具体条件路径，不能先盲装vLLM。 |
| DEP-U14 | root requirements是否来自某次可运行环境 | 它内部已有解析冲突，不能作为可运行证据；只接受目标机命令输出和smoke。 |

## 9. 下一阶段建议先执行的只读/小型 probe 命令

以下命令不安装依赖、不训练；应先在目标 Linux VSCode Remote 终端执行并保存脱敏结果：

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
```

候选环境已经存在依赖时（仍不安装）：

```bash
python3 -c 'import importlib.metadata as m; names=["torch","torchvision","torchaudio","transformers","ray","tensordict","torchdata","datasets","pyarrow","flash-attn","flashinfer-python","sgl-kernel"]; print({n:(m.version(n) if n in {d.metadata["Name"] for d in m.distributions()} else None) for n in names})'
python3 -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available(), torch.cuda.device_count()); print(torch.backends.cudnn.version()); print(torch.cuda.nccl.version()); print([torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]); print([torch.cuda.is_bf16_supported(i) for i in range(torch.cuda.device_count())])'
python3 -c 'import verl,sglang,ray,transformers,tensordict,torchdata,pyarrow,flash_attn,flashinfer,sgl_kernel; print("imports_ok")'
```

随后由下一阶段新增一个只做 init/all-reduce 的短脚本，再执行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 scripts/probe_distributed.py
python3 scripts/probe_sglang_runtime.py --model <local-small-latent-model> --max-new-tokens 2
```

只有这些 probe 通过后，才能写 `constraints/linux-cu124-py311.txt`、执行受控安装和单卡 smoke。正式训练仍须等用户下一阶段明确指示。

## 10. 本智能体实际执行的只读检查

- `rg --files` / `find`：枚举 requirements、pyproject、setup、conda/lock、shell、Docker 与文档安装资产；
- `nl -ba` / `sed` / `rg -n`：逐行读取声明、安装脚本、Docker 参考与训练条件分支；
- Python `ast`：扫描 vendored verl/SGLang 的全仓直接 import（未写文件）；
- `git rev-parse` / `git status --short`：记录 commit 与原有工作树状态；
- `python3 --version`、`importlib.metadata`、`command -v`、torch import probe：确认当前控制端不是目标 CUDA 环境；
- 官方网页/上游源码元数据核验：PyTorch、NVIDIA CUDA/NCCL、vLLM、SGLang、verl、Ray、Transformers；访问日期 2026-08-02。

未执行：依赖安装、卸载、升级、`pip freeze`、CUDA训练、模型下载、数据下载、写入作者仓库。
