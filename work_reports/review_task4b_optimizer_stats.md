# Task 4b 独立审查：optimizer outcome 与 component statistics

## 结论

**FAIL**。本地 optimizer 分支识别、单次 `update_policy()` 内的成功更新计数、全成功路径的 scheduler 兼容行为和 `topk_logits_lst` 修复方向正确；但当前 worker payload 既不能保留三卡各自的充分统计，也没有被 Ray coordinator 消费，因此还不能形成契约要求的全局 `optimizer_step` 或 Stage 2 worker→driver 聚合。component hook 还使用了不完整的有效 mask，并在 overlong 最终 advantage 置零之前计算 FlipGrad trigger，会得到与实际训练信号不同的计数。

当前 Mac 证据最多支持相关局部代码为 `static_check_passed` / 部分 helper 为 `synthetic_test_passed`；不能把 Task 4b 整体标为通过或 GPU/runtime 已验证。

## 高优先级问题

### P0：Ray collect 会丢掉 rank 1/2 的 observer payload，coordinator 也没有消费 rank 0 payload

- `fsdp_workers.py::update_actor()` 把每个 worker 的 `update_count`、optimizer attempts 和 component sufficient stats 放进本地 `DataProto.meta_info["latent_grpo_observer"]`。
- `verl/protocol.py::DataProto.concat()` 明确假设各 worker `meta_info` 相同，并只取 `data[0].meta_info`。`DP_COMPUTE_PROTO` 的 collector 正是调用该 concat。因此三卡 component stats 只会留下第一个 worker 的局部值；其余两卡被静默丢弃。这里的 meta_info 实际并不相同。
- `ray_trainer.py::fit()` 在 `actor_rollout_wg.update_actor(batch)` 后只读取 `actor_output.meta_info["metrics"]`，仓库中没有 coordinator 对 `latent_grpo_observer`、`optimizer_steps` 或 `component_sufficient_stats` 的消费、rank 一致性检查、充分统计合并或 writer 事件绑定。
- 结果是：契约级累计成功 `optimizer_step` 没有推进/恢复路径，`aggregation_worker_count` 无证据，三卡 Stage 2 统计不是全局统计，而且缺失不会报错。

修复门槛：使用不会被 `DataProto.concat` 首项覆盖的显式 worker payload/collector，保留 worker 身份；coordinator 必须验证 FSDP ranks 的 optimizer outcome 一致、对 component integer/sum/sum_sq/min 做正确合并、绑定不可变 step context，并把成功次数交给 checkpoint sidecar。必须增加一个三 worker synthetic collector 测试，断言 rank 1/2 的非零值确实进入合并结果。

### P1：FlipGrad 使用的不是训练最终 advantage，且有效 component mask 不符合 target contract

- `update_policy()` 先调用 `_forward_micro_batch(... collect_component_stats=True)`；之后才在 `exclude_overlong_samples_from_advantage=false` 路径对 max-length 样本执行 `data["advantages"][is_clipped] = 0`。hook 中的 `current_advantages` 因此是置零前值，而 policy loss 看到的是置零后的最终值。
- `spec/target_variables.md` 把 `advantage` 定义为训练实际使用的最终 advantage，并要求 `flipgrad_trigger_mask=(advantage<=0)&(surrogate_margin<0)`。当前 overlong 样本的 `flipgrad_trigger_count` 可与真实训练触发语义不一致。
- `valid_components` 只检查 top-K sentinel（`topk_ids_rmpad_rolled != -100` 及是否 hard token），没有应用 response domain、`response_mask`/multi-turn `loss_mask`。规范要求排除 prompt、padding、普通 hard token和 loss-mask 排除位置。padding 因 unpad 被排除，hard token由 sentinel 排除，但 prompt/response 边界和 loss-mask 没有被显式证明或应用。

修复门槛：先得到与 policy loss 完全相同的最终 advantage 和 response/loss mask，再构造 component mask；对 prompt、padding、hard token、multi-turn loss-mask、overlong final-zero、K 维和 non-finite 分别写 synthetic 测试。任何 shape/K/alignment 不匹配必须生成整 family unavailable，不能依赖运行时广播或异常中断。

### P1：observer facts 尚未由正常 runner 配置启用，文档中的关闭开关也不存在

- actor 与 trainer 都只读取环境变量 `LATENT_GRPO_OBSERVER_ENABLED`；三个 profile 和 runner config 当前没有 `metrics_enabled` 到该环境变量的权威映射。默认目标命令若没有额外手工 export，observer 是关闭的。
- `docs/decision_log.md` 写成可用 `metrics_enabled=false` 关闭 observer，但当前 config/launcher 没有这个已实现字段；RTM 中 `metrics_enabled` 仍是 `planned`。

修复门槛：由 resolved config 显式、可审计地设置 observer 环境，默认值与交付范围一致；dry-run 显示最终开关但不导入 CUDA 依赖；正常启动、显式关闭和 resume 都测试同一映射。文档不得引用不存在的开关。

## 其他问题与边界

### P2：scheduler 修复有意改变 observer-off 的全跳步训练行为，需独立记录

`update_count` 在每个 mini-batch optimizer attempt 后按真实 `did_step` 累加，这一局部语义正确。worker 对 `update_count > 0` 只执行一次 scheduler step：

- 全部 optimizer attempts 成功时，仍保持作者原有的“每次 outer actor update 前进一次 scheduler”行为；不会因多个 mini-batch 而把 scheduler 快进多次。
- 所有 attempts 都因 non-finite 跳过时，scheduler 不再前进。
- 成功/跳过混合时，因为至少一次参数更新成功而前进一步。

这与 scheduler 的 `total_training_steps` 按 outer steps 配置相符，是比“按 `update_count` 次 step”更保守的兼容方案。但它在 observer 关闭时仍生效，属于明确的训练语义修复，不是 logging-only。`docs/upstream_changes.md` 应提供规范要求的 patch ID、algorithmic effect、observer-off effect、test 和 rollback；`docs/decision_log.md` 的 D-008 仍写“是否修正需决策”，与当前实现不一致。

### P2：component reducer 有额外 full-vocabulary FP32 normalization 开销，只有“无持久化/无图”得到静态证明

hook 没有增加 model forward/backward、没有保存 full logits、没有保留 autograd graph，这部分符合数据最小化方向。但它重新执行 `logits_rmpad.detach().float()` + full-vocabulary `logsumexp`，会产生临时 FP32 full-vocabulary buffer/reduction；不是“只处理 selected logits”。这可能在 7B/长序列上带来显存和耗时风险，必须在目标机测峰值与 observer overhead。更理想的是从训练已有的同一 `full_log_probs/topk_log_probs/raw_diff` 计算点直接就地归约，避免第二次 full-vocabulary normalization。

### P2：当前测试主要是源码字符串断言，不能证明运行语义

现有定向测试能证明 AST 可解析、正确列表名出现、默认开关关闭以及标准库 reference reducer 的简单算例，但没有执行：

- fake optimizer 的 finite/non-finite/mixed multi-mini 行为；
- observer-off 前向/损失/返回数学等价；
- 最终 advantage、response/loss mask、K/shape 对齐；
- 三 worker collect/merge；
- coordinator 消费和 checkpoint `optimizer_step`；
- dynamic-batch top-K reorder 或其 fail-closed gate。

审查期间并行的 sampler/guard patch 已开始加入 `_validate_latent_instrumentation_config()` 和 FlashAttention fail-fast。最初相关 contract test 失败；随后 helper 已出现，但旧测试仍只在 `_validate_config()` 函数体内搜索错误字符串而继续失败。这属于并行 Task 4c 的集成状态，最终合并后必须重新运行完整 contract suite。当前 helper 已拒绝 actor/rollout dynamic batch 与 fused kernels，但没有拒绝 `use_remove_padding=false`；而 component hook 仅在 remove-padding 分支产出事实，非 remove-padding 时会静默没有 component 统计。

## 已确认的正确部分

- `_optimizer_step()` 仅在 `actor_optimizer.step()` 成功返回后把 `did_step` 置真；non-finite grad 不计数。
- `update_policy()` 对每个 mini-batch optimizer attempt 累加成功次数，不把 gradient accumulation micro-batch 当作 optimizer step。
- scheduler 在常规成功路径维持作者原有一次/outer-update 的节奏；全跳步时不前进。
- observer 关闭时 actor 的历史 public return type 仍为 metrics dict，`_optimizer_step()` 仍返回 `grad_norm`。
- `compute_log_prob()` 的 `topk_logits` 改为拼接 `topk_logits_lst` 是正确的局部 bug fix；但 dynamic-batch 下 top-K tensors 的 reorder 仍需 fail-closed 或完整修复。
- 导出的 observer payload 是普通 scalar/list/dict，不包含 optimizer state、parameter、gradient、hidden state、full logits 或持久计算图。

## 执行证据

通过：

```text
python3 -m unittest \
  tests.unit.test_upstream_optimizer_patch \
  tests.unit.test_upstream_patch_contract.UpstreamPatchContractTests.test_optimizer_reports_real_success_and_scheduler_skips_nonfinite_update \
  tests.unit.test_upstream_patch_contract.UpstreamPatchContractTests.test_topk_logits_are_concatenated_from_logits_not_ids -v
# 9 tests, OK

python3 -m compileall -q \
  latent_grpo_runner/upstream_adapter.py \
  tests/unit/test_upstream_optimizer_patch.py \
  Latent-GRPO/verl-0.4.x/verl/workers/actor/dp_actor.py \
  Latent-GRPO/verl-0.4.x/verl/workers/fsdp_workers.py
# exit 0

git -C Latent-GRPO diff --check -- \
  verl-0.4.x/verl/workers/actor/dp_actor.py \
  verl-0.4.x/verl/workers/fsdp_workers.py
# exit 0
```

额外静态检查确认 component capture 发生在 overlong advantage 最终置零之前。

未通过/未执行：

```text
python3 -m unittest \
  tests.unit.test_upstream_patch_contract.UpstreamPatchContractTests.test_dynamic_batch_and_fused_latent_paths_fail_safe_and_flash_is_required -v
# 审查中的最终快照：FAIL；guard 已被并行实现移动到 helper，测试仍检查旧函数体

python3 -m pytest ...
# 当前系统 python3 未安装 pytest；未安装依赖，符合本阶段限制
```

未执行任何 CUDA、Ray、FSDP、SGLang、FlashAttention 或正式训练；这些仍为 `target_machine_test_deferred`。

## 重新审查的最低条件

1. 修复三 worker payload transport 与 coordinator 消费，并用合成三 worker 测试证明 integer/sum/sum_sq/min 和 outcome consensus。
2. hook 使用最终 advantage 与完整有效 mask，覆盖 overlong、multi-turn loss-mask、prompt/padding/hard token、K/shape。
3. resolved config 权威启用 observer；删除不存在的 `metrics_enabled` 文档声明或真正实现它。
4. 明确记录 scheduler 的 observer-off 训练影响，并补 finite/non-finite/mixed 多 mini-batch fake optimizer 测试。
5. 完成 Task 4c guard 测试整合；至少对 dynamic batch、fused、remove-padding 与 FlashAttention fail-closed 做静态/合成覆盖。
6. 在安装隔离 Mac dev 依赖后重跑定向 pytest/full suite；在目标机再验证 FSDP rank consensus、Ray payload、CUDA 数值一致性和 observer overhead。
