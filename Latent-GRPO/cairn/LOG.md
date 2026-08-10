# Project Cairn 日志

本文件按时间倒序记录实质进展——最新条目位于顶部、紧接在本说明之后。每个条目保持简短，只包含摘要与指针；结论沉淀到 `cairn/<topic>.md`。

## 2026-08-11 · Kaggle 指标验证使用轻量 checkpoint

- Kaggle 实证 FSDP `actor_rollout_save_checkpoint` 因同时构造 model/optimizer CPU state dict 触发 Ray `NODE_OUT_OF_MEMORY`；磁盘仍余 12 GB，输出目录仅 360 KB。
- `kaggle-t4-30-metric` 现保存 model shard、scheduler/RNG extra 和 observer sidecar，不序列化 optimizer，并明确禁止 resume；其他 profile 的完整 checkpoint 默认不变。
- FSDP manager 按 `checkpoint.contents` 条件保存/加载，保存时逐组件写入并立即释放；最终 gate 仍要求五张正式指标表和 `CORE METRICS: 29 / 29`。
- 修复运行时提交为 `aa26ea0a038b7e70fbf8add0f2d4fcd2c3c25651`；真实双 T4 指标完整性需 Kaggle 重新 Run All。

## 2026-08-11 · 撤回 Kaggle CUDA allocator 强制配置

- Kaggle 实证 `expandable_segments:True` 使 SGLang 权重同步进入 CUDA IPC 重建，并因容器拒绝 `pidfd_getfd` 而在首个 rollout 前失败；该错误不是 OOM。
- Launcher 不再注入 `PYTORCH_CUDA_ALLOC_CONF`，并原样保留调用方已有配置；FSDP root-only unshard、单次 embedding lookup 和 masked PPO/Gumbel 修复不变。
- 修复运行时提交为 `d1fb7b59b9bddfe62ad6cb0b78895f42f5eff066`；真实 3-GPU 行为仍需目标机器验证。

## 2026-08-11 · 限制双 T4 embedding unshard 峰值

- Kaggle `update_policy` 实证在 padded latent embedding lookup 中调用递归 `FSDP.summon_full_params`，GPU 仅余 30.69 MiB 时为 unshard 再申请 234 MiB 并 OOM。
- Embedding lookup 现仅 unshard root-owned 参数、在边界释放 allocator 缓存，并把 padded/packed 路径的重复 lookup 合并为一次；输出在上下文内立即 detach，保持原 latent mixture 语义。
- FSDP 峰值修复运行时提交为 `081d6de33b720a0f64a30d134f423f6993905e83`；其中曾加入的 Kaggle allocator 强制配置已由后续提交撤回。

## 2026-08-11 · 排除 masked token 的 PPO 数值污染

- Kaggle 双 T4 实证 195 个有效 response token 的 policy/KL/ratio count 均完整，但上游 `pg_loss`、`ppo_kl`、`grad_norm` 为 NaN；根因为 padding 非有限值经 `NaN * 0` 污染 masked reduction。
- Gumbel likelihood 现于非线性运算前清理 invalid 行，PPO 于 ratio/clipping 前清理 masked log-prob 与 advantage；masked 输出及梯度严格为零，有效 token 数值与梯度保持不变。
- 修复运行时提交为 `ebae7957cdb64e256212179b4177df22b0a3e5d6`；CPU 回归覆盖 masked `NaN/+Inf/-Inf`、Gumbel、PPO 和通用 masked mean。

## 2026-08-11 · 对齐 Stage 3 response Top-K 域

- Kaggle `support_benchmark_metrics` 实证第二个 Stage 3 阻塞为 `response_width_mismatch`：mask 仅覆盖 response，而 rollout/old Top-K 仍携带不同长度的 prompt 前缀。
- Support collector 现从 `response_mask` 动态取得宽度，对两份已有 Top-K 取 response 后缀；任一输入不足该宽度仍 fail-closed，不硬编码 32。
- 修复运行时提交为 `fffe2735887eda6a55688088ba3505088e2304e0`；本地测试覆盖不同前缀长度和不足宽度拒绝路径。

## 2026-08-11 · 修复 Stage 3 GSM8K 轨迹分类来源

- Kaggle `support_benchmark_metrics` 实证 `identity_vector_length_mismatch`：正式 batch 有 8 条轨迹，但 scalar GSM8K reward 不产生可选 `acc` extra-info，导致分类标签为空。
- Stage 3 现直接复用已计算的 `token_level_scores`，严格校验二值 sequence reward 后构造逐轨迹 `correct`/`non_correct` 标签；不增加 forward，也不修改 reward 或 dynamic filtering。
- 修复运行时提交为 `8f30db5feb99b77417672aefe4f0afdfeed37604`；本地测试覆盖 8 条标签和非二值 fail-closed。

## 2026-08-11 · 提高 Kaggle 2xT4 动态过滤采样成功率

- Kaggle 实跑确认旧 `rollout_n=2` 在 5 批生成内未产生奖励有方差的 prompt group，训练按设计在 dynamic filtering 阶段 fail-closed。
- 仅调整 30 指标验证 profile 为 `rollout_n=4`、`filter_groups_max_num_gen_batches=10`；reward、GRPO advantage 和 dynamic filtering 实现保持不变。
- 新运行时提交为 `1ab101c85ef1a75d1ed99011edbd0ca32ca68b87`，30 指标 notebook 已重新固定该 SHA。

## 2026-08-11 · 接通 Kaggle 2xT4 30 指标正式 runtime

- 新增 `kaggle-t4-30-metric` 一步真实更新 profile，启用 Stage 1-3 passive instrumentation、checkpoint-only Stage 4 和 Credit autograd。
- Gumbel likelihood 可选暴露真实图上的 Top-K log-prob/raw delta；worker 每 rank 仅调用一次 `torch.autograd.grad`，随后保持原训练 `loss.backward()`/optimizer 路径。
- Driver 在 checkpoint 保存前聚合有界 detached probe packet，写入正式 `probe_metrics`/`probe_benchmark_metrics`，并将状态保护证据纳入 schema。
- 新增双 T4 正式 runner 和 29 core + raw token validator；本地 dry-run 与 CPU 单元测试通过，真实 CUDA/Ray/SGLang 执行仍待 Kaggle。
- 运行时已发布为 `76c09fc8e45f57ad7c487ed0532e3994f38b53f2`；30 指标 notebook 固定该 SHA，并随 `bb2c1fd` 发布到 `main`。
- 本地单元测试、notebook 语法和静态契约已通过；真实 CUDA/Ray/SGLang 结果仍需 Kaggle 2xT4 Run All 生成。

## 2026-08-10 · 生成 Kaggle 2xT4 29 指标验证 Notebook

- 新增固定 commit `86aa1fca...` 的 Run All notebook、可复现生成器与静态验收测试。
- Notebook 逐项验证 29 个核心指标、raw generated token 扩展和 11 项训练污染检查；所有最终 PASS 路径均绑定正式 runtime/state evidence。
- 复用旧 notebook 的双 T4、隔离 runtime、模型/数据/tokenizer 和正式 compatibility 流程，删除全部源码 patch 与依赖 workaround。
- 当前 commit 缺少可启用 Stage 3/4 的 2xT4 正式 profile/Stage 4 runner，notebook 会 fail-closed 输出 BLOCKED，不在 notebook 内补实现。
- Kaggle 2xT4 Run All 尚未执行；3-GPU 最终 gate 仍为 TARGET_RUNTIME_REQUIRED。

## 2026-08-10 · 完成 Stage 3/4 指标采集本地实现

- 新增 Stage 3 Support collector，复用 `rollout_topk_ids` 与 pre-update `old_topk_indices`，严格 shape/K/identity fail-closed。
- 新增 Stage 4 checkpoint probe reducers：one-sided delta 统计、credit autograd concentration/Spearman/alignment、CPU state-preservation guard。
- Schema/sink 接通 `support_metrics`、`support_benchmark_metrics`、`probe_metrics`、`probe_benchmark_metrics`；Support trainer hook 由 `LATENT_GRPO_SUPPORT_ENABLED` gated。
- 本地 CPU 单元测试通过；Kaggle 2xT4 与 3-GPU Stage 3/4 runtime validation 仍为 TARGET_RUNTIME_REQUIRED。

## 2026-08-10 · 新增 raw generated token 训练指标

- 根据附件规范新增 `train/raw_generated_token_count`，保持 `train/generated_token_count` 的 final-training-rollout 语义不变。
- 采集点接在 PPO trainer 每次真实 `generate_sequences` 返回后、dynamic group filtering 之前；P1 observer payload 单独携带 raw generation lengths。
- 指标 schema 增加 raw scope/definition metadata；本地单元测试覆盖无过滤相等、过滤/重试 raw 更大、prompt/padding/Top-K 不误计、无额外 generation/forward/backward。
- 目标 Linux/Kaggle 3-GPU runtime 训练验证仍未执行。

## 2026-08-02 · 完成 Phase A 只读审计与实现设计

- 已完成训练链路、依赖/CUDA、target variables、三卡拓扑和验收方案的并行审计设计；未安装依赖、未运行训练、未改训练代码。
- 确定采用外部 runner + 作者仓库最小 observer patch；三卡与所有 GPU 兼容事实保持 runtime-gated。
- 详情见 `cairn/phase-a-audit-design.md` 及其指向的 `../../docs/`、`../../work_reports/` 资产。

## 2026-08-02 · 初始化 Project Cairn

- 已初始化 Project Cairn 核心结构。
- 历史迁移模式：`start_fresh`。
- 详情见 `AGENTS.md` 和 `.cairn/config.yaml`。
