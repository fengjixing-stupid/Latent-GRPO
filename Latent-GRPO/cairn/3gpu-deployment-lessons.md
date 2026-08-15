---
type: project_topic
status: active
summary: "三卡 L20 目标机部署必须隔离 CUDA/Python 工具链、固定物理/逻辑 GPU 映射，并以真机门禁闭合。"
tags: [3gpu, deployment, cuda, conda, proxy, runtime-gate]
contains: [decision, experience, lesson, procedure, reference]
created: "2026-08-15"
updated: "2026-08-15"
related:
  - "3gpu-runtime-packaging.md"
  - "../../docs/3GPU_RUNBOOK.md"
  - "../../docs/superpowers/specs/2026-08-15-3gpu-deployment-lessons-design.md"
authoring_mode: ai_generated
---
# 三卡目标机部署经验与脚本设计检查

## 背景

本文沉淀自 teammate 在公司 Linux GPU 服务器部署 Latent-GRPO 的完整排障对话：<https://chatgpt.com/share/6a7ea313-a9c8-83ec-b1f2-6e3c748c0a81>。

对话中观测到的目标机为 8 张 NVIDIA L20（单卡约 46 GiB）、Driver `550.90.07`；当时物理 GPU 4/5/6 空闲，GPU7 已有任务，因此正式方案选择物理 GPU 4/5/6。GPU 占用是时变事实，任何脚本都必须在启动前重新探测，不能把该快照写成永久假设。

本文不保存公司代理地址、内部下载地址、个人用户名或绝对工作路径；这些来源信息已脱敏。实际脚本应从环境变量或显式参数读取，不得硬编码内部基础设施信息。

职责边界：

- 本文负责目标机环境、安装、代理、GPU 映射、部署故障和防复发检查。
- `3gpu-runtime-packaging.md` 负责单 Driver/Ray/FSDP 拓扑、作者参数语义和最终 runtime gate。
- `../../docs/3GPU_RUNBOOK.md` 负责可执行的操作步骤。

## 当前结论

- `nvidia-smi` 中的 `CUDA Version: 12.4` 表示 Driver 可支持的最高 CUDA 运行时范围，不证明本机安装了 CUDA Toolkit 12.4；必须另外检查 `command -v nvcc` 和 `nvcc --version`。
- 本次目标机实际系统 Toolkit 为 CUDA 11.4。项目需要用户态 CUDA Toolkit 12.4，与 `/usr/local/cuda` 并存，不替换共享服务器 Driver，也不修改系统 CUDA 软链接。
- 目标 Python 和 pip 必须来自项目 `.venv-target`；`nvcc`、`CUDA_HOME`、`CUDA_PATH`、`CUDACXX` 必须来自 CUDA 12.4 bootstrap Conda 环境。
- 物理 GPU 4/5/6 在 `CUDA_VISIBLE_DEVICES=4,5,6` 下映射为逻辑 `cuda:0/1/2`；telemetry、preflight、训练进程和 acceptance 必须使用同一映射。
- 正式控制面是一个 Python Driver 启动 Ray 和三个 FSDP Worker；不要再套 `torchrun --nproc_per_node=3`，也不要手工启动三个训练 Python。
- 本地语法、单元、dry-run 和包级验收不能替代 L20 真机 CUDA/NCCL/SGLang/FSDP 验收。在两步 validation 产生 `3GPU_FINAL_GATE: PASS` 前，状态必须保持 `TARGET_RUNTIME_EXECUTION_REQUIRED`。

## 经验

### 1. 先区分资源快照与进程归属

对话最初通过 `top`、`free -h` 和 `nvidia-smi` 观察服务器。机器总内存约 936 GiB；一个 Python 进程曾占约 583 GiB RES 并处于 `D` 状态。8 张 L20 中 GPU0～3 高负载、GPU4～6 空闲、GPU7 已占约 18 GiB。

关键经验：

- `free` 很小不代表内存耗尽，应以 `available` 为主要判断。
- 单个进程 `100% CPU` 通常表示占满一个逻辑核，不表示整机 CPU 已满。
- `D` 状态表示不可中断睡眠，持续出现时再结合磁盘、网络存储或驱动 IO 排查。
- 容器或 PID namespace 隔离可能使 `nvidia-smi Processes` 为空；不能据此断言 GPU 无任务。
- GPU 是否可用应同时查看显存、利用率、功耗、P-State、温度和程序内 `torch.cuda` 可见性。
- 同步三卡任务不应混入已有负载的 GPU7；最慢或显存最紧的 rank 会拖慢或阻断整个作业。

### 2. 交付包级验收不等于目标机验收

共享对话中的交付包审计曾发现：

- 文档路径与测试期望不一致。
- `flashinfer-python` 存在互斥精确版本。
- runbook 与 requirements 的 `sgl-kernel` 版本不一致。
- telemetry 调用 `nvidia-smi` 后记录整机 8 张卡，没有过滤选中的三张物理卡。
- Actor Top-K 宽度写死为 10，而不是从 rollout tensor 推导。
- Git 工作树不干净会触发严格 preflight。
- 启用 W&B 但未安装 tracking 依赖会阻断运行。

当时的解决方向是：修正文档路径、统一 `flashinfer-python==0.2.5` 与 `sgl-kernel==0.1.1`、按 `CUDA_VISIBLE_DEVICES` 过滤 telemetry、动态推导 Top-K 宽度、增加资产/schema/版本检查，并区分 `LOCAL_RELEASE_GATE: PASS` 与目标机 `3GPU_FINAL_GATE: PASS`。

这些是来源对话中交付包的阶段性结论；后续判断当前仓库是否仍有同类问题时，必须重新读取实际文件和测试，不能只引用本文。

### 3. Driver 支持 CUDA 12.4，不代表 `nvcc` 是 12.4

目标机出现过：

```text
nvidia-smi  → Driver 550.90.07 / CUDA Version 12.4
nvcc --version → release 11.4
```

原因是前者来自 NVIDIA Driver 能力，后者来自 `/usr/local/cuda/bin/nvcc` 的实际 Toolkit。FlashAttention 等扩展需要真实编译工具链，因此仅检查 `nvidia-smi` 会产生错误放行。

修复原则：

- 在用户目录或独立 Conda 环境安装 CUDA Toolkit 12.4.1。
- 不安装 `cuda-drivers` 或 `nvidia-driver`。
- 不替换 `/usr/local/cuda`，不影响共享机器上的 CUDA 11.4 项目。
- 编译最小 CUDA 程序，并在 `CUDA_VISIBLE_DEVICES=4,5,6` 下确认设备数为 3。

### 4. 代理 502 必须拆分为元数据格式和频道问题

公司代理对普通 `repodata.json` 返回 200，但 Conda 请求 `repodata.json.zst` 时返回 `502 Bad Gateway`。这说明代理地址基本可用，失败点是压缩元数据路径，不应反复更换代理或关闭 TLS。

第一层修复：

```bash
conda config --set repodata_use_zst false
export CONDA_REPODATA_USE_ZST=false
export CONDA_REPODATA_USE_SHARDS=false
```

并在命令中显式使用：

```text
--no-repodata-use-zst
--repodata-fn repodata.json
```

随后 `conda search` 能找到 `cuda-toolkit=12.4.1`，但 `conda create` 仍访问 `conda-forge`、`defaults`、`pkgs/pro`、`pkgs/free`、`msys2` 等频道并再次 502。原因是 `--repodata-fn` 只选择元数据文件，不限制频道。

第二层修复是增加：

```text
--override-channels
```

并只声明已用 `wget --spider` 或 `conda search` 验证可达的 NVIDIA CUDA 12.4.1 channel 与 Anaconda `pkgs/main`。正式脚本不应读取未知 `.condarc` 频道。

### 5. 双层环境的 PATH 顺序曾让 Python 与 pip 分离

推荐结构为：

```text
bootstrap Conda 环境
  ├─ Python 3.11（只用于创建项目 venv）
  └─ CUDA Toolkit 12.4（nvcc、headers、libraries）

项目 .venv-target
  └─ torch、torchvision、SGLang、VERL 等 Python 包
```

故障现场中，同时激活 `.venv-target` 与 bootstrap Conda 环境后执行了：

```bash
export PATH="$CUDA_HOME/bin:$PATH"
```

这把 Conda 环境的 `python`/`pip` 放到 `.venv-target` 前面。结果是裸 `pip list` 显示 Torch 2.12.1 和 CUDA 13 包，而 `"$PYTHON_BIN"` 中甚至没有 torch。

正确顺序：

```bash
export PYTHON_BIN="$PROJECT_ROOT/.venv-target/bin/python"
export PATH="$PROJECT_ROOT/.venv-target/bin:$CUDA_HOME/bin:$PATH"
```

并统一使用：

```bash
"$PYTHON_BIN" -m pip ...
```

禁止在双层环境中用裸 `pip` 判断或修改目标环境。每次环境脚本都应打印并校验：

```text
sys.executable
python -m pip -V
command -v nvcc
nvcc --version
CUDA_HOME
CUDACXX
torch.__version__
torch.version.cuda
```

### 6. PyTorch cu124 索引缺失精确 cuDNN wheel

固定 PyTorch 2.6.0+cu124 安装失败于：

```text
No matching distribution found for nvidia-cudnn-cu12==9.1.0.70
```

原因不是 GPU 或 CUDA Driver，而是当时使用的 PyTorch cu124 索引未列出这一精确 wheel；PyTorch 元数据又要求精确版本，不能静默替换为 `9.1.1.17`。

修复流程：

1. 只从 PyPI 下载 `nvidia-cudnn-cu12==9.1.0.70`，带 `--only-binary=:all:` 和 `--no-deps`。
2. 校验 SHA-256：`165764f44ef8c61fcdfdfdbe769d687e06374059fbb388b6c89ecb0e28793a6f`。
3. 将校验通过的 wheel 路径保存为 `CUDNN_WHEEL`，用 `"$PYTHON_BIN" -m pip install --no-deps "$CUDNN_WHEEL"` 安装到 `.venv-target`。
4. 重新运行固定 PyTorch 安装脚本，让其补齐其余 cu124 依赖。
5. 运行 `pip check`、版本断言、CUDA available 和 BF16 gate。

不推荐用 `--extra-index-url` 作为正式修复，因为 pip 会汇总多个索引的候选，不保证额外索引只承担缺失依赖的后备角色。

### 7. 进程探测正则误匹配 `pipe_handle`

排查安装进程时使用过包含 `[p]ython.*pip` 的正则，意外匹配了其他用户 multiprocessing 命令行中的 `pipe_handle`。这些进程不是 pip 安装，也不应终止。

任何自动监控或清理脚本都必须同时限定：当前用户、目标解释器绝对路径、`-m pip`/安装脚本名称和父子进程关系。只看到模糊字符串匹配时，应报告而不是杀进程。

### 8. 代理只用于下载，本地分布式通信必须绕过

下载阶段可以使用企业代理，但 `no_proxy`/`NO_PROXY` 至少必须包含：

```text
localhost,127.0.0.1
```

否则 Ray、SGLang 或本地服务可能把 loopback 请求送入代理。内部域名规则应由运行环境注入，本文不保存其具体值。

正式训练启动前应明确 `unset RAY_ADDRESS`，避免误连历史或外部 Ray 集群。

### 9. 输出、缓存和所有权也是部署边界

- 模型、数据、Hugging Face/Pip 缓存、Ray 临时目录、checkpoint、日志和实验输出应放在大容量持久磁盘。
- 不要把大文件集中放入 100 GiB 系统盘或 Git 仓库。
- 训练应使用普通项目用户，避免 root 创建普通用户无法继续写入的缓存与结果。
- validation 与正式训练每次使用新的时间戳输出目录，不能覆盖或复用旧目录。
- 模型必须是匹配实验的 Latent-SFT checkpoint；普通 Base/Instruct checkpoint 不属于同一实验口径。

### 10. 三卡验收必须先于正式训练

推荐顺序：

```text
资源、磁盘和 Git 工作树探测
→ 固定 bootstrap CUDA 12.4 与 .venv-target 路径
→ 版本、pip check、ABI/import gate
→ 模型、数据和输出路径检查
→ GPU4/5/6 物理/逻辑映射检查
→ Low 两步 validation
→ 3GPU_FINAL_GATE: PASS
→ 正式 Low
→ High 两步 validation
→ 正式 High
```

正式训练必须引用匹配 profile、Git commit、seed、模型和数据身份的 acceptance。GPU 状态、模型路径或代码状态变化后，旧 acceptance 不应继续复用。

### 11. 算法语义不能用“部署降配”静默改变

出现 OOM 或运行慢时，不要直接修改以下参数后仍把结果称为正式实验：

```text
rollout_n
Top-K
response length
temperature/top_p
KL
learning rate
batch size
```

需要降配时新建明确标注的 smoke/debug profile，并记录与作者配置的偏差。

对话还澄清了 `</think>` 切换语义：

- `topk_idx[1:] = -100` 把其余槽位标记为无效 hard-token sentinel。
- `topk_prob[1:] = 0`、`topk_prob[0] = 1` 把输入表示规范化为单个确定 token，不表示模型原始 softmax 置信度为 100%。
- `</think>` 之后答案阶段仍按配置中的 `temperature=0.6`、`top_k=30`、`top_p=0.95` 随机采样，不是强制贪婪解码。

## 三卡脚本设计检查

任何生成或修改三卡目标机训练、部署、环境、监控或验收脚本的任务，都必须逐项填写结果。不能确认时先探测或 fail-closed；不适用时写明 `不适用：原因`，不得无声跳过。

### GPU 与编号映射

- [ ] 脚本是否显式要求恰好三张 GPU，并拒绝空值、重复编号和四卡输入？
- [ ] 是否区分物理编号与 `CUDA_VISIBLE_DEVICES` 后的逻辑 `cuda:0/1/2`？
- [ ] `CUDA_VISIBLE_DEVICES` 是否只由一个明确入口设置，避免调用方和子脚本二次映射？
- [ ] telemetry、preflight、训练和 acceptance 是否记录同一组三张物理卡？
- [ ] 启动前是否重新检查显存、利用率、P-State 和程序内可见设备数？

### 启动拓扑

- [ ] 是否保持一个 Python Driver → 一套 Ray runtime → 三个 FSDP Worker？
- [ ] 是否避免再套 `torchrun`、外层多进程 launcher 或手工三进程？
- [ ] 是否 `unset RAY_ADDRESS` 或显式验证目标 Ray 地址？
- [ ] 子进程失败是否能使 Driver 非零退出，而不是只在日志中打印错误？
- [ ] rank/world-size/资源申请是否与三卡配置一致？

### 解释器、CUDA 与依赖

- [ ] `PYTHON_BIN`、`python -m pip -V`、`nvcc`、`CUDA_HOME`、`CUDACXX` 是否解析为预期绝对路径？
- [ ] 是否禁止裸 `pip` 并固定 Python 3.11、PyTorch cu124 及项目精确依赖？
- [ ] 是否分别验证 Driver 能力、Toolkit `nvcc` 版本和 `torch.version.cuda`，而不是只看 `nvidia-smi`？
- [ ] 是否运行 `pip check`、关键包版本断言、ABI/import、CUDA available 和 BF16 gate？
- [ ] 多索引或离线 wheel 是否有精确版本、来源和 SHA-256 校验？

### 网络、代理与存储

- [ ] 代理是否只在下载阶段启用，`localhost,127.0.0.1` 是否进入 `no_proxy`/`NO_PROXY`？
- [ ] Conda 是否禁用失败的 `.zst`/shards，并使用 `--override-channels` 与已验证频道？
- [ ] 是否拒绝以关闭 TLS 校验作为正式解决方案？
- [ ] 模型、缓存、Ray 临时目录、checkpoint 和输出是否指向有容量的持久路径？
- [ ] 是否检查磁盘、CPU available memory、目录权限、输出目录唯一性和非 root 所有权？

### 算法语义与验收证据

- [ ] 模型是否为要求的 Latent-SFT checkpoint，数据是否为现存 parquet 且 schema/hash 可验证？
- [ ] 三卡适配是否只包含文档化的拓扑差异，未静默改动正式算法参数？
- [ ] Low/High validation 与正式 profile 是否严格匹配，acceptance 是否绑定 commit/config/seed/assets？
- [ ] 本地、静态、dry-run 与目标机 runtime 证据是否分层表述？
- [ ] 没有 `3GPU_FINAL_GATE: PASS` 时是否保持 `TARGET_RUNTIME_EXECUTION_REQUIRED`？

## 实践指南

### 环境入口应先打印真值

三卡环境脚本至少应在执行前输出并验证：

```bash
printf '%s\n' \
  "PYTHON_BIN=$PYTHON_BIN" \
  "CUDA_HOME=$CUDA_HOME" \
  "CUDACXX=$CUDACXX" \
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

"$PYTHON_BIN" -c 'import sys; print(sys.executable)'
"$PYTHON_BIN" -m pip -V
"$CUDACXX" --version
```

随后用目标 Python 断言 PyTorch、CUDA、BF16、关键包版本和逻辑 GPU 数。路径或版本不匹配时直接退出。

### GPU 探测应同时保留物理和逻辑证据

物理层通过 `nvidia-smi --query-gpu=index,... --format=csv` 记录选中卡；逻辑层在一次性 `CUDA_VISIBLE_DEVICES` 环境下通过 PyTorch 记录 `cuda:0/1/2`。不要让全机 telemetry 的 GPU0～7 被误当作当前三卡作业证据。

### 正式运行必须引用验收报告

两步 validation 与正式训练使用不同输出目录。正式 wrapper 应拒绝：缺失报告、失败报告、profile 不匹配、commit 不匹配、资产身份不匹配或来自旧输出目录的报告。

## 教训

- 版本号必须检查“来自哪里”，不能只检查“显示什么”：Driver、Toolkit、PyTorch wheel 和编译扩展分别有自己的 CUDA 身份。
- 多层环境中，提示符显示已激活并不构成证据；绝对解释器路径和 `python -m pip` 才是证据。
- 网络失败应沿“代理 → 元数据格式 → 频道 → 包索引 → 精确 wheel”分层定位，不应一次性放宽所有来源和 TLS 规则。
- GPU 编号、telemetry 和 acceptance 是同一条证据链；任何一处把物理编号误当逻辑编号都会使验收失真。
- 脚本中的模糊进程匹配可能伤及其他用户任务；诊断工具不能自动升级为破坏性清理工具。
- 真机未执行就是未执行。文档、单元测试和 dry-run 再完整，也不能写成 L20 CUDA/NCCL/FSDP 已通过。

## 补充经验

- 压测代码的并发由 `--concurrency` 控制，对话中默认值为 2；判断压测规模应追踪参数从 CLI 到 `asyncio.run()`/worker 的实际传递，而不是只看请求循环。
- GitLab pre-receive hook 曾因提交作者邮箱不是项目注册成员而拒绝 push；修复是将 Git `user.email` 设置为团队注册邮箱并重新生成提交，网络和分支名不是根因。
- `cannot edit in read-only editor` 通常表示光标位于 VS Code Output、Git 日志、diff 或只读预览；应切换到真正 Terminal，再执行需要的全局 `git config --global` 命令。

## 论据与开放边界

- 来源对话提供了故障现象、诊断过程和当时采用的解决方案；本文是经验总结，不是对当前目标机状态的实时证明。
- 对话中的本地包验收曾报告语法、shell、单元和 dry-run 通过，但没有替代目标机真实 CUDA/NCCL/SGLang/FSDP 运行。
- 当前服务器 GPU 占用、公司代理策略、外部包索引和内部镜像均可能变化，脚本必须重新探测。
- 若未来改变 Driver、CUDA、PyTorch、SGLang、Ray、GPU 型号或卡数，应重新验证本文中的兼容性结论，并更新本知识专题文档与 `cairn/LOG.md`。
