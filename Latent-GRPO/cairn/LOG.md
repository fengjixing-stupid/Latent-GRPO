# Project Cairn 日志

本文件按时间倒序记录实质进展——最新条目位于顶部、紧接在本说明之后。每个条目保持简短，只包含摘要与指针；结论沉淀到 `cairn/<topic>.md`。

## 2026-08-11 · 接通 Kaggle 2xT4 30 指标正式 runtime

- 新增 `kaggle-t4-30-metric` 一步真实更新 profile，启用 Stage 1-3 passive instrumentation、checkpoint-only Stage 4 和 Credit autograd。
- Gumbel likelihood 可选暴露真实图上的 Top-K log-prob/raw delta；worker 每 rank 仅调用一次 `torch.autograd.grad`，随后保持原训练 `loss.backward()`/optimizer 路径。
- Driver 在 checkpoint 保存前聚合有界 detached probe packet，写入正式 `probe_metrics`/`probe_benchmark_metrics`，并将状态保护证据纳入 schema。
- 新增双 T4 正式 runner 和 29 core + raw token validator；本地 dry-run 与 CPU 单元测试通过，真实 CUDA/Ray/SGLang 执行仍待 Kaggle。
- Notebook 生成器已改用单一正式 runner，但最终 notebook 需要这些改动 commit/push 后才能绑定远端精确 SHA。

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
