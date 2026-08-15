# 三卡部署经验与脚本设计检查设计

## 目标

将 teammate 在三卡 L20 目标机部署 Latent-GRPO 时的完整排障过程沉淀为可复用经验，并让后续 Codex 在生成或修改三卡机器上的训练、部署、环境、监控或验收脚本前，强制执行相关设计检查。

来源对话：<https://chatgpt.com/share/6a7ea313-a9c8-83ec-b1f2-6e3c748c0a81>

## 交付物

1. 新建 `Latent-GRPO/cairn/3gpu-deployment-lessons.md`，作为部署故障、修复方案和防复发检查的当前真相。
2. 更新 `Latent-GRPO/AGENTS.md`，加入三卡脚本的强制触发规则和知识专题文档入口。
3. 更新 `Latent-GRPO/cairn/LOG.md`，在顶部记录本次知识沉淀及文件指针。

`Latent-GRPO/cairn/3gpu-runtime-packaging.md` 保持职责不变：它继续描述训练拓扑、作者参数语义和最终 runtime gate；新文档专门描述目标机环境、安装、部署故障与脚本设计检查。

## 经验文档结构

经验文档按以下顺序组织：

1. 适用范围、服务器事实和证据边界。
2. 故障时间线与问题矩阵；每项采用“现象 → 原因 → 修复 → 验证 → 防复发检查”。
3. 三卡脚本统一设计检查表。
4. 推荐环境结构、命令约束和验收顺序。
5. 来源、已验证事实与仍需目标机闭合的事项。

共享对话中的无关内容不逐字复制。资源查看、Git 作者邮箱、只读编辑器、压测并发等内容仅在附录中简述；与三卡部署直接相关的代码包审计、CUDA、Conda、代理、双层环境、pip/cuDNN、GPU 映射和验收问题完整保留。

## 必须记录的故障与解决方案

- 代码包存在文档路径、FlashInfer/sgl-kernel 版本、全机 GPU telemetry、写死 Top-K、工作树不干净等缺陷；通过修复、版本统一、三卡过滤和本地包级测试解决。
- `nvidia-smi` 显示 Driver 可支持 CUDA 12.4，不代表系统 `nvcc` 是 12.4；目标机实际只有 CUDA Toolkit 11.4，需要用户态 CUDA 12.4，与系统 Toolkit 隔离。
- 公司代理可访问普通 `repodata.json`，但 `.zst` 请求返回 502；必须关闭 zstd/shards，并显式使用普通元数据。
- `conda create` 未使用 `--override-channels` 时会重新访问 `.condarc` 中不可达的默认或旧频道；必须只声明已验证频道。
- Conda CUDA 环境与项目 `.venv-target` 并存时，错误的 PATH 顺序会让裸 `pip`、Python 和目标解释器分离；必须固定 `.venv-target/bin/python -m pip`，同时让 `nvcc` 来自 CUDA bootstrap 环境。
- PyTorch cu124 索引缺少精确的 `nvidia-cudnn-cu12==9.1.0.70`；采用精确 wheel 下载、SHA-256 校验、`--no-deps` 本地预装，再运行固定 PyTorch 安装流程。
- 进程检测正则误把 `pipe_handle` 当成 pip；脚本应匹配明确的安装命令并避免根据模糊结果终止其他用户进程。
- `nvidia-smi Processes` 为空可能来自容器/PID namespace 隔离；资源判断必须结合显存、利用率、功耗、P-State 和程序内 CUDA 可见性。
- 物理 GPU 4/5/6 映射为逻辑 `cuda:0/1/2`；telemetry、preflight 和 acceptance 必须使用同一映射，且运行前重新检查三卡是否空闲。
- 本地静态验收不能表述为 L20 真机通过；最终结论必须由两步 validation 的 `3GPU_FINAL_GATE: PASS` 闭合。

## 三卡脚本强制设计检查

`AGENTS.md` 将规定：只要任务涉及生成或修改目标三卡机器上的训练脚本、部署脚本、环境安装脚本、资源监控脚本或验收脚本，Codex 必须先阅读两个知识专题文档，并在交付前逐项检查：

1. GPU：严格三卡、物理/逻辑编号、`CUDA_VISIBLE_DEVICES` 设置位置、共享卡占用与 telemetry 过滤。
2. 拓扑：单 Driver → Ray → 三个 FSDP Worker；不得额外套 `torchrun` 或重复多进程 launcher。
3. 环境：Python、pip、nvcc、CUDA_HOME、CUDACXX 的真实来源；禁止裸 `pip`；固定版本和 ABI/import gate。
4. 网络与存储：代理仅用于下载，`localhost/127.0.0.1` 必须进入 `no_proxy`；模型、缓存、Ray 临时目录、checkpoint 和输出使用大容量持久路径。
5. 语义与验收：不得静默修改正式超参数；资产、Git、输出目录和 acceptance identity 必须 fail-closed；本地证据与真机证据分层表述。

若检查项不适用，脚本设计说明必须写明理由；不得无声跳过。若目标机事实未知，脚本必须先探测或以明确错误退出，不得猜测。

## 数据流与使用方式

后续 Codex 的工作流为：

```text
识别三卡脚本任务
  → 阅读 AGENTS.md 指定的两个三卡知识专题文档
  → 收集目标机事实和现有配置
  → 完成五类设计检查
  → 生成或修改脚本
  → 运行静态检查、dry-run 或目标机门禁
  → 在交付中说明已通过、未执行和仍需闭合的检查
```

## 错误处理

- 对 GPU 数量、GPU 映射、解释器、CUDA Toolkit、依赖版本、资产路径、Git 状态或 acceptance identity 的不一致采用 fail-closed。
- 网络安装失败应区分代理、元数据格式、频道和 wheel 索引，不使用关闭 TLS 校验作为正式方案。
- 目标机无法执行时保留 `TARGET_RUNTIME_EXECUTION_REQUIRED`，不得把本地测试结果升级为真机结论。
- 不自动清理未知进程、工作树、环境或输出目录；任何破坏性处理都需单独确认。

## 验证

实施完成后验证：

1. 新知识专题文档 frontmatter、内部链接和 Project Cairn 术语符合规范。
2. `AGENTS.md` 保持精简，规则能够覆盖训练、部署、环境、监控和验收脚本。
3. `LOG.md` 新条目位于顶部且不超过约 20 行。
4. 搜索关键触发词，确认两个三卡知识专题文档可从 `AGENTS.md` 发现。
5. 检查 Git diff，确保没有改动用户现有代码或其他未提交文件。
