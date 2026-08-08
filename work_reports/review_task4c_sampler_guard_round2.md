# Task 4c 修复轮独立复审（round 2）

## 结论

**PASS（限本修复轮范围）。** 上轮的三项阻断问题已被修复或被明确降级为不可用接口；未发现新的 Mac-safe 阻断缺陷。

本结论只支持 `static_check_passed`、`synthetic_test_passed`、`mac_development_check_passed` 与 `target_machine_test_deferred`。未执行或声称 CUDA、FlashAttention、SGLang server、Ray/FSDP、NCCL 或 GPU 训练验证。

## 逐项复核

| 检查项 | 结论 | 证据 |
|---|---|---|
| FlashAttention CE fail-fast | PASS | `dp_actor.py:54-60,153-156` 在 latent Gumbel micro-batch forward 前检查 `FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE`；`torch_functional.py:192-195` 在 latent 专用函数内提供 defense-in-depth 异常，不再回退普通 label log-prob。测试同时证明 `add_noise_gumbel_softmax=false` 不触发该 guard。 |
| dynamic/fused 配置守卫 | PASS | `ray_trainer.py:389-401,479-489` 在 worker 启动前拒绝 actor dynamic batch、rollout log-prob dynamic batch、fused kernels；条件仅在 `rollout.enable_latent` 或 observer enabled 时成立。observer-off 且非 latent 的相同配置保持兼容。 |
| production durable sink | PASS（fail-closed） | `upstream_adapter.py:140-162` 要求显式 sink 同时具备 `durable=true`、`enabled=true` 和 `emit`；缺 sink 立即报告 `eval raw-fact persistence is interface-only`。`BufferedObserver` 被明确拒绝用于生产，只保留为 synthetic 工具。 |
| runner 启用 observer 但未注入 sink | PASS（行为明确，不是功能完成） | 当前 runner 通过环境变量暴露 adapter，但没有向 `RayPPOTrainer(observer_sink=...)` 注入 writer；因此 `LATENT_GRPO_OBSERVER_ENABLED=1` 会在 trainer 构造的 `load_observer_from_env(sink=None)` 处立即失败，而不是运行后丢数据。`docs/upstream_changes.md:7-9` 明确标记 `unavailable_with_reason=authoritative_eval_sink_not_integrated`，没有宣称端到端 persistence。 |
| eval raw fact 字段与时间点 | PASS（接口级、非 schema 完成） | hook 位于 reward/decode 后、`process_validation_metrics()` 前；接口携带 source、question/generation identity、预测、参考答案、reward、correctness 与 reason。它仍缺完整 `eval_question_results` schema enrichment/write，因此只有接口证据，持久化按上述 reason 明确 unavailable。 |
| eval ordinal | PASS（单次 checkpoint validation） | `ray_trainer.py:723-733` 在一次 `_validate()` 外层创建 ordinal map，循环内复用；`upstream_adapter.py:103-112` 按 question 跨 dataloader batch 连续编号。新 checkpoint validation 重新从 0 开始，符合 checkpoint/question/generation 主键层级。 |
| eval correctness | PASS | 只使用 `reward_extra_info.acc`，缺失为 `null + reward_extra_info.acc_missing`；接受 bool 与精确数值 0/1，并拒绝 0.5 等非二值值，不从 reward 符号/大小推断正确性。 |
| 默认 observer-off / dry-run | PASS | 环境默认返回 `NoOpObserver`，identity/eval/OCP 发射均跳过；`smoke --dry-run --validate-config` 成功且未加载目标 CUDA runtime。 |
| 非 latent 数学边界 | PASS（静态） | `torch_functional.py` 的 diff 仅把 `logprobs_from_logits_topk_gumbel()` 的缺 FlashAttention fallback 改为异常；普通 `logprobs_from_logits()`、Top-K normal/dirichlet fallback 和 latent Gumbel 可用分支数学未改。未引入 Stage 3/4、额外 forward 或通用 gradient hook。 |

## 命令与结果

```text
python3 -m unittest \
  tests.unit.test_sampler_guards \
  tests.unit.test_upstream_adapter \
  tests.unit.test_upstream_patch_contract -v
34 tests: PASS, 3 dependency-gated skips

python3 -m unittest discover -s tests -v
124 tests: PASS, 3 dependency-gated skips

python3 -m compileall -q train_latent_grpo.py latent_grpo_runner scripts tests
PASS

AST parse: torch_functional.py, dp_actor.py, ray_trainer.py, upstream_adapter.py
AST_OK 4

git -C Latent-GRPO diff --check
PASS

python3 train_latent_grpo.py --config configs/smoke.yaml --dry-run --validate-config
exit 0; mac_development_check_passed; target_machine_test_deferred
```

三项 skip 分别是 NumPy/DataProto 合同两项和上游 Torch OCP 数值等价一项；本机缺依赖，不能把它们计为通过，也不影响本轮纯标准库/AST 守卫证据。

## 保留限制与目标机门槛

- authoritative eval sink 尚未集成，完整 `eval_question_results` enrichment、Parquet commit、overflow/error propagation 均不是 implemented；状态必须保持 `unavailable_with_reason=authoritative_eval_sink_not_integrated`。
- FlashAttention CE 的真实 import/ABI、latent forward 数值与梯度、SGLang worker 参数传播、Ray/FSDP transport 均为 `target_machine_test_deferred`。
- 本轮 ordinal 仅证明单次 `_validate()` 内跨 batch 连续；端到端主键唯一性须在 writer 集成后再次验收。
