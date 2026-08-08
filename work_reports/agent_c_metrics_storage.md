# Agent C — 指标可观测性与存储只读审计

## 0. 范围、证据等级与结论

- 本报告只读审计作者仓库 `Latent-GRPO`（commit `c0994fb781a2d180662bb522d8ff3e8638dcf56d`），未修改训练代码、未安装依赖、未启动训练、未联网。唯一写入是本报告。
- 指标语义的唯一权威是 `spec/target_variables.md`（`spec/03_METRICS_STORAGE_CONTRACT.md:17-27`）；作者变量名只能作为上游证据，不能反向改变目标定义（`spec/04_AGENT_ORCHESTRATION.md:99-127`）。
- 目标文档明确要求 shape/dtype/device/DP/TP 语义在目标 Linux/GPU runtime probe 中确认，禁止猜测或静默 reshape/cast/截断/广播（`spec/target_variables.md:23-25`）。因此本文使用三个等级：
  - **confirmed**：静态代码可直接证明。
  - **inferred**：由相邻路径强推得出，但没有实际 Tensor/runtime 证据。
  - **runtime_probe_required**：必须在目标环境检查 shape、顺序、device、并行复制或公式符号。
- 总结：Stage 1 的 response/reward/advantage/step timing、Stage 2 的 rollout top-K 与扰动分数、eval clean top-K 的原始来源均已存在；但作者代码没有契约化事件、count/sufficient-statistics、stable trajectory ID、winner 导出、Support/Probe 调度和可靠写盘。Stage 3 可复用现有 pre-update old-log-prob forward，不应再补 forward。Stage 4 one-sided/credit 没有可直接调用的 probe 接口，必须做 checkpoint-only controlled forward；credit 默认仍必须关闭。

为保持表格可读，后文 `path:line` 使用以下无歧义别名：

- `ray_trainer.py` = `Latent-GRPO/verl-0.4.x/verl/trainer/ppo/ray_trainer.py`
- `core_algos.py` = `Latent-GRPO/verl-0.4.x/verl/trainer/ppo/core_algos.py`
- `metric_utils.py` = `Latent-GRPO/verl-0.4.x/verl/trainer/ppo/metric_utils.py`
- `dp_actor.py` = `Latent-GRPO/verl-0.4.x/verl/workers/actor/dp_actor.py`
- `fsdp_workers.py` = `Latent-GRPO/verl-0.4.x/verl/workers/fsdp_workers.py`
- `sglang_rollout.py` = `Latent-GRPO/verl-0.4.x/verl/workers/rollout/sglang_rollout/sglang_rollout.py`
- `sampler.py` = `Latent-GRPO/sglang_latent_reasoning_pkg/python/sglang/srt/layers/sampler.py`
- `schedule_batch.py` = `Latent-GRPO/sglang_latent_reasoning_pkg/python/sglang/srt/managers/schedule_batch.py`
- `tokenizer_manager.py` = `Latent-GRPO/sglang_latent_reasoning_pkg/python/sglang/srt/managers/tokenizer_manager.py`
- `torch_functional.py` = `Latent-GRPO/verl-0.4.x/verl/utils/torch_functional.py`
- `target_variables.md` = `spec/target_variables.md`；`03_METRICS_STORAGE_CONTRACT.md` = `spec/03_METRICS_STORAGE_CONTRACT.md`

其余带目录的源码引用以 `Latent-GRPO/verl-0.4.x/verl/` 为基准；根级启动脚本和 `data_preprocess_code/` 引用以 `Latent-GRPO/` 为基准。

## 1. 真实数据流和采集边界

### 1.1 训练主链（confirmed）

1. driver 生成 rollout：`ray_trainer.py:1038-1046`。
2. prompt `uid` 在 repeat 前创建，随后 `repeat(..., interleave=True)`：`ray_trainer.py:1064-1069`。这里存在 group ID，但不存在 group 内 stable `trajectory_id`。
3. response mask 由生成结果构造，随后可能在 driver 上 balance/reorder：`ray_trainer.py:1071-1078`；balance 的确调用 `batch.reorder`：`ray_trainer.py:953-963`。
4. reward manager 产生 `token_level_scores`：`ray_trainer.py:1082-1103`；Naive manager 将 trajectory 标量 reward 放在最后一个有效 response token：`workers/reward_manager/naive.py:42-43,57-86`。
5. dynamic filtering 可丢弃 group，最终 batch 再次 balance：`ray_trainer.py:1105-1174,1207-1211`。因此任何“最终训练 rollout”统计都必须在这一边界之后计算。
6. actor old-log-prob 无梯度 forward 在 update 前执行：`ray_trainer.py:1217-1226`；reward/advantage 在 driver 上完成：`ray_trainer.py:1268-1303`；actor update 随后执行：`ray_trainer.py:1313-1322`。
7. step timer 包围 generation 到 update/eval/save 的主段，完成后才写 logger 并递增 global step：`ray_trainer.py:1038-1040,1351-1369`。这与 `train_step_metrics.observation_phase="post_update"` 的契约相符（`spec/target_variables.md:162-170`）。

### 1.2 latent/top-K 链（confirmed）

- SGLang 先在 full-vocabulary log-prob 上施加 top-p mask，采样 raw Gumbel、裁剪到 `[-1.5,3]`、可做 one-sided 平移和 scale，再形成 `logp + noise`，取 top-K，并以 softmax 得到实际 latent mixture weights：`sampler.py:73-128`。
- SGLang request 同时缓存 perturbed top-K scores、mixture weights、clean top-K probabilities/indices：`schedule_batch.py:730-771`。
- tokenizer response 已把 `output_topk_prob_list` 暴露在 meta-info：`tokenizer_manager.py:1062-1076`；但 verl `_post_process_outputs` 只读取 perturbed scores、IDs、clean probs/indices，没有读取 mixture weights：`sglang_rollout.py:191-224`。因此 noisy mixture **上游存在但当前 driver 接口缺失**。
- rollout 输出 shape 在 verl 侧先是 `B×O×K`，padding 后与 prompt top-K 拼成 `B×(I+O)×K`，并随 batch 返回：`sglang_rollout.py:699-720,734-766`。
- actor old-log-prob forward 使用 rollout top-K/perturbed scores重建 latent embedding，并以 `softmax(scores / gumbel_temperature)` 混合 embedding：`workers/actor/dp_actor.py:108-170`。这再次确认 `rollout_topk_gumbels` 实际是 **perturbed component score**，不是纯 Gumbel noise。
- actor 当前分布的 clean top-K 在同一次 old-log-prob forward 中由现有 logits 得到：`dp_actor.py:320-343`。该 forward 是 Stage 3 允许复用的 pre-update old-policy forward；不得再补 forward（契约 `spec/target_variables.md:844-870`）。

## 2. 按阶段/表的可观测性审计

下表中的“默认 availability”是对**尚未实施 adapter 的作者原仓库**的保守判断；完成 adapter 且通过 runtime probe 后方可变为 true。reason 必须用稳定枚举，空 mask 用 `empty_effective_mask`，shape/顺序分别用 `shape_mismatch`/`alignment_failed`（枚举见 `spec/03_METRICS_STORAGE_CONTRACT.md:198-223`）。

### 2.1 Stage 1 — `train_step_metrics`

| family / 目标字段 | upstream source（路径:行）与证据 | observation phase | effective mask / count | 现状与采集方式 | 默认 availability / reason | 额外 forward/backward 风险 |
|---|---|---|---|---|---|---|
| policy loss：`train/policy_loss` | `compute_policy_loss` 构造 dual-clip loss 并经真实 `agg_loss` 聚合：`core_algos.py:707-769`；actor 现仅返回 micro-batch 标量 `actor/pg_loss`：`dp_actor.py:563-608`。**confirmed** | actor update 内观察，row 在 `post_update` 提交 | 必须复用 `loss_agg_mode`。`token-mean` 计 masked token；seq modes 计有效 sequence/固定 normalizer，不能统一假装 token count；真实模式定义见 `core_algos.py:669-700` | 已有标量，但当前 `reduce_metrics` 默认对 micro-batch mean 再平均（`utils/metric/utils.py:23-53`），不满足充分统计。需在 loss adapter 返回 numerator/normalizer/count | false / `missing_runtime_interface` | 无额外 forward/backward；只 detach 局部统计。禁止改变反传 scalar |
| entropy：`train/entropy` + 4 个 definition 字段 | full categorical entropy 可由 actor logits计算：`dp_actor.py:283-306`；旧策略 forward 当前强制 `calculate_entropy=True`：`fsdp_workers.py:684-699`。update forward 在 `entropy_coeff=0` 时不算 entropy：`dp_actor.py:547-577`，发布脚本恰为 0：`Latent-GRPO-gsm8k-llama3.sh:27`。**confirmed** | 应从 actor-update 已有 forward 采集，最终 row `post_update`；不可把 rollout mixture entropy冒充 | update 使用的 response/loss mask；独立 entropy sum/count | 上游有算子但没有契约事件。adapter 应在已有 actor update forward 上开启/计算 entropy reduce，并写 `entropy_source=actor_update_logits`、概率空间、mask/version；不应把 pre-update old entropy无说明地冒充 | false / `missing_runtime_interface` | 不增加 forward/backward，但 full-vocab entropy reduction有显存/时间开销，需 benchmark；fused-kernel分支需 runtime probe |
| KL/clip：`train/kl`, `train/clip_fraction` | ratio、近似 KL、clip predicate 位于 `core_algos.py:749-767`，现导出 `actor/ppo_kl`、`actor/pg_clipfrac`：`dp_actor.py:603-607`。**confirmed** | actor update 内，row `post_update` | update 的真实 response/loss mask；KL sum/count与clip numerator/count分别保存 | 已有 micro-batch mean，需 loss adapter 输出 sufficient stats | false / `missing_runtime_interface` | 无额外 forward/backward |
| ratio：`train/importance_ratio_mean/std` | `ratio=exp(log_prob-old_log_prob)` 已存在：`core_algos.py:749-750`，但未返回 ratio统计。**confirmed** | actor update 内，row `post_update` | 与 ratio mean/std完全相同的 response/loss mask；`sum/sum_sq/count` | 需在 `compute_policy_loss` 邻接 adapter做 detached reduce | false / `missing_runtime_interface` | 无额外 forward/backward |
| response length：`train/response_length` | `attention_mask[:, -R:]` 求和：`metric_utils.py:50-77,109-119`；现有 mean：`metric_utils.py:158-162`。**confirmed** | final training batch，post-update提交 | trajectory count；长度由 response attention/loss counting rule决定 | 可直接在 driver final batch 做 integer sum/count；不得复用已有 scalar mean而丢 count | true（adapter后）/ 当前 false `missing_runtime_interface` | 无 forward/backward；EOS 是否计入需 runtime probe（契约 `target_variables.md:257-265`） |
| latent length：`train/latent_length` | hard token由 top-K 第 2..K 项均为 `-100` 标记：`dp_actor.py:137-160`；SGLang 在 latent 结束/普通 hard token处写 `-100`：`schedule_batch.py:746-761`。**inferred** | final training rollout；post-update提交 | response attention/loss mask AND latent sentinel；trajectory count用于 mean，另存 latent-position count | driver adapter 可由 response slice 的 `rollout_topk_ids` 构造，但必须先 probe off-by-one和 sentinel | false / `runtime_probe_failed`（probe前）；失败后 `shape_mismatch`/`alignment_failed` | 无 forward/backward |
| generated tokens：`train/generated_token_count`, cumulative | final filtered batch在 `ray_trainer.py:1173-1214` 固定；response token总数已由 mask可得 `metric_utils.py:195-198`。**confirmed** | dynamic filter/selection 后的 final training rollout；row post-update | integer sum of trajectory lengths；worker只求和，driver求和 | driver adapter在 final batch立即冻结总数，不能用 `perf/total_num_tokens`（它含 prompt，`metric_utils.py:235-243`） | true（adapter后）/ 当前 false `missing_runtime_interface` | 无 forward/backward。probe/support token绝不能进入该计数（`target_variables.md:267-292,1180-1187`） |
| step time：`train/step_time` | `_timer("step")`：`ray_trainer.py:1038-1040`；`perf/time_per_step`来源：`metric_utils.py:211-243`。**confirmed** | completed global step | 每 step 1；独立于 metrics compute/write time | driver直接采集；writer时间不能混入 step（存储契约 `spec/03_METRICS_STORAGE_CONTRACT.md:318-329`） | true（adapter后）/ 当前 false `missing_runtime_interface` | 无 forward/backward；异步 CUDA计时语义需固定 |
| `learning_rate`, `optimizer_step` | worker在 update后读取 LR 并 step scheduler：`fsdp_workers.py:602-604`；optimizer可能因非有限 grad 跳过：`dp_actor.py:379-393`，但 scheduler仍推进。**confirmed** | row post-update | optimizer success event count，不可按 rank相加；各 DP rank必须一致 | 需 worker显式返回 `optimizer_step_succeeded/count`，driver一致性检查后累计；当前仅 global step，无可靠累计 optimizer count | false / `missing_runtime_interface` | 无额外计算；若 ranks不一致必须整条记录失败，不可取平均 |

Stage 1 的 10 个指标、禁止 gradient norm、generated-token固定口径、无 `eval_metrics` 等验收清单见 `spec/target_variables.md:1657-1666`。当前 `actor/grad_norm` 虽在作者 logger 产生（`dp_actor.py:610-612`），新 schema **必须不映射、不落盘**。

### 2.2 Stage 1 — checkpoint eval / raw facts

| record/family | upstream source 与证据 | phase / mask/count | 采集与 shape 风险 | 默认 availability |
|---|---|---|---|---|
| `eval_dataset_manifest` | 数据预处理保存 `reward_model.ground_truth` 和 `extra_info.index`：`data_preprocess_code/gsm8k_aug.py:22-37`；dataset把 index带入 row：`utils/dataset/rl_dataset.py:278-286`。**confirmed** | dataset version × question，只写一次 | `question_id` 应由 dataset identity + stable index/hash构造并校验唯一；reference/hash在 eval前冻结 | manifest字段有源；`eval_dataset_version`/hash adapter缺失，false `missing_runtime_interface` |
| `eval_question_results` | validation repeat后生成、decode完整 outputs、调用同一 val reward函数：`ray_trainer.py:678-750`。GSM8K scorer正确为1/否则0：`utils/reward_score/gsm8k.py:44-63`；math scorer同为0/1：`utils/reward_score/math.py:17-28`。**confirmed（发布任务）** | `checkpoint_eval`；每 question × generation恰一行，包括失败；length用 eval response/latent mask | 现 `_validate` 只做汇总并可写简化 JSONL，未保留 question/generation ID、failure reason、generation seed：`ray_trainer.py:752-784`。需 eval adapter逐行构造；自定义 scorer的 `is_correct` 映射需 runtime probe | false / `missing_runtime_interface` |
| `eval_clean_topk` | validation显式关闭 Gumbel：`sglang_rollout.py:662-670`；同一 generation forward产生 clean probs/indices并传回 `rollout_topk_original_probs/indices`：`sampler.py:146-156`, `sglang_rollout.py:699-720,763-766`。**confirmed** | `checkpoint_eval`；仅严格有效 latent position，每 position一行；K/list长度相同 | `_validate` 当前忽略这些 batch字段。adapter只能转存已有值，不得补 softmax/top-k/forward；先验证 `B×R×K`、latent sentinel、K与list配对。Qwen配置 latent end=522（`Latent-GRPO-math500-qwen.sh:43-46`），sampler另有硬编码524分支（`sampler.py:132-145`），是高风险 runtime probe点 | false / `missing_runtime_interface`；shape失败 `shape_mismatch` |

契约要求 eval每次 generation完整保留、禁止 checkpoint级汇总、clean top-K只转存已有值（`spec/target_variables.md:361-467`）。作者 `_validate` 现会产生汇总 `val-core/val-aux`（`ray_trainer.py:770-784`）；这些可继续供 console，但不得进入权威 `eval_metrics` 表。

### 2.3 Stage 2 — 低成本机制与 group

| family / 目标字段 | upstream source（路径:行）与证据 | phase | effective mask/count | 现状/adapter | 默认 availability | 额外风险 |
|---|---|---|---|---|---|---|
| Gumbel diagnostic 5项 | raw Gumbel、clip、one-sided、scale的唯一真实执行点：`sampler.py:89-105`。**confirmed** | 独立 diagnostic/smoke，不属于训练 step | raw family：full-vocab实际采样元素，`raw_sum/sum_sq/count/lower/upper`；one-sided family：变换+scale后的 `zero/count` | 只允许在 sampler就地GPU reduce；绝不导出 raw tensor | false / `disabled_by_config` | 开启时会做 full-vocab reduce；不得在正式训练“每步算、低频写” |
| noisy mixture 2项 | 实际用于 latent state的 `topk_probs=softmax(perturbed_scores/T)`：`sampler.py:105-128`；request缓存该值：`schedule_batch.py:742-764`。**confirmed** | rollout后、最终训练 group filter后聚合 | 实际 noisy latent positions；effective-K/top1共享 `mixture/noisy_count` | 当前 verl漏接 `output_topk_prob_list`。最佳 adapter是在 SGLang端先压成每 trajectory/position两个标量，随 trajectory过 filter，再对 final batch聚合；不要保存完整 weights | false / `missing_runtime_interface` | 无 forward/backward；早期全局reduce会错误包含被filter丢弃的候选 |
| zero advantage | final advantage在 driver构造：`ray_trainer.py:1283-1300`；但 actor在 `exclude_overlong=false` 时又于 policy forward后原地把 max-length样本 advantage清零：`dp_actor.py:551-561`。**confirmed** | 最终参与 actor policy loss前 | `eligible_latent_mask = response/loss mask ∧ latent sentinel`；zero numerator同一mask，阈值版本化 | 必须在 actor mutation之后、首次有效 PPO观察处采集，或在driver严格复刻并验证语义；不能直接统计driver的 pre-mutation advantage | false / `missing_runtime_interface` | 无额外 forward/backward；多个 PPO epoch不能重复计同一训练语义，需定义只计一次 |
| reward mean/std | `token_level_rewards`来自真实 score/可含KL：`ray_trainer.py:1276-1282`；现有统计先按trajectory求和：`metric_utils.py:103-105,129-137`。**confirmed** | `post_advantage_pre_update`冻结，随 step row提交 | 发布任务是真实 trajectory scalar；`sum/sum_sq/trajectory_count` | driver可直接做 sufficient stats；若启用token级RM/不同manager，统计单位需 runtime/config审计 | true（adapter后）/ 当前 false `missing_runtime_interface` | 无 forward/backward |
| advantage std | final `advantages`由 Latent-GRPO advantage函数返回：`core_algos.py:113-245,247-359`；driver当前按response mask统计：`metric_utils.py:106-121,138-141`。**confirmed** | final actor-loss语义 | 独立 valid advantage mask/count；需包含actor后续overlong-zero语义 | 与 zero-adv同一 adapter边界做 `sum/sum_sq/count` | false / `missing_runtime_interface` | 无额外 forward/backward；不可用 group std均值替代全局std |
| stable `group_id/trajectory_id` | group uid在repeat前生成，repeat后轨迹已存在：`ray_trainer.py:1064-1069`；随后可能reorder：`ray_trainer.py:1071-1078`。**confirmed** | repeat后、任何 balance/filter/select前 | ID，不是统计mask | 需立刻分配 group内出现序 `trajectory_id` 并随DataProto传播；当前 UUID group uid不具备跨resume可复现性，持久化 `group_id` 应由 prompt stable identity/global step派生 | false / `missing_runtime_interface` | 不涉及模型计算；严禁用local batch index落盘 |
| group raw facts | final group由 uid聚合；response length/overlong从mask可得；reward score可分类。发布GSM8K/math scorer为0/1，证据见上。**confirmed for released profiles; inferred for custom** | `post_advantage_pre_update` | trajectory raw count；correct+non_correct=total；overlong独立重叠；zero variance基于同组真实 reward | driver group adapter在 final batch、balance后按stable IDs聚合；自定义reward必须提供明确 correctness adapter | false / `missing_runtime_interface` | 无 forward/backward；不能用 `invalid/incorrect` 第三类 |
| Optimal Correct Path | positive first-step advantage候选、mean old log-prob评分和winner选择都真实存在：`core_algos.py:191-240,307-356`。**confirmed** | advantage完成、update前 | response mask用于trajectory mean；候选是positive first-step advantage | winner只存在局部 `winner_idx`，函数未返回；需最小adapter同时导出stable trajectory ID与mean。若 old log prob缺失，作者会随机fallback（`core_algos.py:227-234,343-350`），契约运行应判 unavailable而非随机伪造 | false / `missing_runtime_interface` | 无额外 forward/backward；tie规则需固定并probe NumPy argmax顺序 |
| Stage 2 surrogate接口 | `raw_diff = perturbed score - current component log-prob`，Flip条件 `(adv<=0)&(raw_diff<0)`：`utils/torch_functional.py:149-175`。**confirmed** | 只暴露/测试，Stage 2不持久化 onesided | valid latent component；delta/flip count独立 | controlled instrumentation adapter在函数内局部reduce；Stage 2只测试接口，不写onesided | false / `missing_runtime_interface` | 无额外 forward/backward/hook；Flash-Attn分支不可用时会退化到普通logprob：`torch_functional.py:133-195`，必须probe |

Stage 2 的强约束来自 `spec/target_variables.md:544-586,590-675,679-838`；尤其 Stage 2 不写 `onesided/*`。

### 2.4 Stage 3 — `support_metrics` / `support_benchmark_metrics`

| family | upstream source与时点 | mask/count与shape | 建议 adapter | 默认 availability / 风险 |
|---|---|---|---|---|
| rollout noisy top-K side | final batch中的 `rollout_topk_ids`，来源 `sglang_rollout.py:702-720,735-766`。**confirmed** | response slice后应为 `B×R×K`；仅 stable trajectory的有效 latent position | 仅在Support step为选中trajectory短期缓存 IDs；不跨worker传全量top-K | false / `missing_runtime_interface` |
| pre-update clean top-K side | old-log-prob forward发生于 actor update前：`ray_trainer.py:1217-1226`；同一forward计算 clean top-K：`dp_actor.py:328-343`，worker返回 `old_topk_indices`：`fsdp_workers.py:695-700`。**confirmed** | 当前返回 `B×(I+R-1)×10` 而非明确 `B×R×K`，且 K硬编码10。需用同一 next-token切片严格对齐，K与rollout一致才可用 | 在 old-log-prob forward内部、Support step且选中trajectory时立即截取 response latent positions；不是额外forward | false / `shape_mismatch` 直到probe通过 |
| retention/top1 retention | 两端都已有；公式由契约固定（`target_variables.md:970-1000`） | 对齐键 `global_step/group_id/trajectory_id/latent_position`；共享有效position count；position加权 | driver只收每选中trajectory的 `intersection_fraction_sum/top1_retained_count/effective_position_count` | false / `missing_runtime_interface`；任何长度/K/顺序不一致 `alignment_failed`，禁止截断/广播 |
| trajectory selection | correct winner需复用同一Optimal结果；non-correct用old-log-prob masked mean，来源同上 | candidate count独立；overlong排除；tie取最小stable ID | selection必须在内存中完成，不能从Parquet回读 | false / `missing_runtime_interface` |
| benchmark | actor worker已有max allocated/reserved memory样例：`fsdp_workers.py:592-600`，driver已有timer机制：`ray_trainer.py:1038-1040`。**confirmed capability** | 每实际Support step一行；总position count与trajectory行count不同scope | Support前reset peak（若安全）/后同步读取，driver wall time，记录cache peak、selected/candidate/position totals | false / `missing_runtime_interface`；CUDA async计时和多rank peak取max需probe |

重要 shape 风险：`compute_log_prob` 在 dynamic batching时只恢复 `log_probs` 顺序，没有恢复 entropy/top-K顺序（`dp_actor.py:461-473`）；默认配置虽为 `use_dynamic_bsz=False`（`trainer/config/ppo_trainer.yaml:47,97`），任何开启该选项的profile都必须将 Support/entropy 标为 `alignment_failed`，直至 adapter 同步恢复所有字段。Support 时间字段、严格对齐和失败语义见 `spec/target_variables.md:874-966`。

### 2.5 Stage 4 — checkpoint-only `probe_metrics`

| family | 可观测来源 | mask/count | controlled probe设计输入 | 默认 availability / 计算风险 |
|---|---|---|---|---|
| common probe chain | 作者 checkpoint保存actor和dataloader/global step：`ray_trainer.py:869-900`，resume从folder step恢复：`ray_trainer.py:902-951`；训练reward/advantage链见§1。**confirmed partial** | 固定prompt/group、训练同reward/mask/advantage/Optimal语义 | 新probe必须显式保存/恢复Python/NumPy/CPU/CUDA RNG、参数/optimizer/scheduler/.grad哈希；checkpoint metadata当前没有显式累计successful optimizer step，需adapter | false / `missing_runtime_interface` |
| one-sided 7项 | `raw_diff`与flip trigger已在训练utility中存在：`torch_functional.py:149-175`。perturbed scores来自rollout；component log-prob由同一forward的global log-softmax gather得到：`torch_functional.py:143-153`。**confirmed formula source** | delta：valid latent component `sum/sum_sq/min/negative/near-zero/count`，小probe可detach有限样本算p05；flip独立 numerator/count | checkpoint固定小batch执行训练同构controlled forward；在utility边界输出detached stats，不注册module backward hook | false / `missing_runtime_interface`；onesided默认应开启但实现前不能声称available。额外 forward是Stage 4受控probe所需，不得混入训练；one-sided本身无需额外backward |
| credit concentration | component log-prob图当前在utility内部被聚合后丢失：`torch_functional.py:149-180`。**confirmed missing interface** | concentration count是可得有效 `u_i/q_i` 的position/component集合 | probe-only adapter暴露局部 differentiable component log-prob，单次 `torch.autograd.grad(L_PG, component_log_prob)`；立刻聚合/释放图 | false / `disabled_by_config`（默认）；开启后失败按 `zero_gradient`/`runtime_probe_failed` | 一次局部 backward；显存和符号高风险，不写parameter `.grad` |
| credit Spearman/alignment | noisy weights源在sampler但verl接口缺失；surrogate proxy源在 `torch_functional.py:165-175`。**confirmed partial** | Spearman只计长度一致且非退化vector；alignment只计非零且符号已验证元素，各自独立count | 同一次credit梯度结果完成所有trajectory/position分组；固定tie规则与alignment definition version | false / `disabled_by_config`；启用后常见 `degenerate_constant_vector`、`zero_gradient`, `alignment_failed` | 禁止每组重复backward；禁止module-level hook |
| probe benchmark | CUDA memory API/timer能力同Support | 每checkpoint一行，prompt/trajectory/latent count均为整数 | 比较关闭、forward-only、Delta、credit等模式；worker peak取max、driver时间单独记录 | false / `missing_runtime_interface` |

Probe契约默认 checkpoint/onesided开、credit关（`spec/target_variables.md:1099-1137`），credit只允许一次受控autograd且不得污染训练grad（`target_variables.md:1233-1282`），所有分组必须复用同一次梯度（`target_variables.md:1286-1319`）。因此：未实现probe时 `record_available=false, record_unavailable_reason="missing_runtime_interface"`；实现forward/onesided但credit默认关闭时，应严格按 `target_variables.md:1354-1363` 令整条record与onesided可用、credit families为 `disabled_by_config`。

## 3. 集中的 shape / mask / 并行风险清单

1. **pre-update top-K不是response-ready shape（confirmed）**：`full_topk_*`只去掉最后sequence位置，没有取response窗（`dp_actor.py:328-343`）。Support adapter必须在producer处按 next-token shift取 `[-R-1:-1]`，而不是driver静默裁剪。
2. **top-K logits字段拼接错误（confirmed）**：`topk_logits = torch.concat(topk_ids_lst, dim=0)`，不是 `topk_logits_lst`（`dp_actor.py:441-465`）。任何 Delta/credit设计都不可相信当前返回的 `topk_logits`；应在 `logprobs_from_logits_topk_gumbel` 内直接观测 component log-prob。
3. **dynamic-batch回序不完整（confirmed）**：只恢复log_probs（`dp_actor.py:468-473`），entropy/top-K仍可能是重排顺序。
4. **K硬编码（confirmed）**：actor clean top-K使用 `k=10`（`dp_actor.py:328-330`），而契约要求与runtime K严格一致。配置目前也是10（`Latent-GRPO-gsm8k-llama3.sh:39-42`），但不能依赖偶然相等。
5. **latent end hardcode冲突（confirmed）**：sampler分支写死524（`sampler.py:132-145`），Qwen profile配置522（`Latent-GRPO-math500-qwen.sh:43-46`）。这既影响latent mask，也可能影响 actual noisy mixture口径。
6. **TP复制（runtime_probe_required）**：rollout代码明确“all TP ranks contain same data”（`sglang_rollout.py:756-757`）。统计不能把TP复制当独立worker相加；只允许DP owner贡献，TP/SP先局部约简/去重。
7. **reward统计单位（confirmed for released profile, runtime elsewhere）**：Naive manager把trajectory标量放末token（`naive.py:42-86`），但其它reward manager可能不同。schema definition必须记录单位；adapter不能无条件按trajectory解释所有profile。
8. **advantage终态跨进程差异（confirmed）**：driver advantage完成后，worker还可能原地zero overlong（`dp_actor.py:556-561`）。Stage 2 zero-adv/advantage std与Stage 4 probe都必须使用真实actor-loss最终值。
9. **optimizer step不可由global step推断（confirmed）**：一次actor update含ppo epochs/minibatches并可能跳过非有限grad（`dp_actor.py:505-520,610-613,379-393`）；必须单独累计successful optimizer calls。
10. **top-K观测接口只在特定actor分支成立（confirmed）**：remove-padding分支构造 `full_topk_*`，non-remove-padding分支只构造logits/log-probs，但函数统一返回top-K变量（`dp_actor.py:328-378`）；fused-kernel分支也没有建立后续依赖的 `logits_rmpad`（`dp_actor.py:231-236,317-343`）。发布profile使用remove-padding且默认非fused，但其它profile必须先probe，失败标 `unsupported_upstream_version`/`missing_runtime_interface`。

Runtime probe最小输出应记录每个关键Tensor的 logical name、producer、shape、dtype、device、requires_grad、batch reorder token、DP/TP/SP ownership、mask sum、K、response slice和next-token shift；失败只改availability，不改shape。

## 4. 事件与 sufficient statistics 设计（Phase C输入，不实现）

### 4.1 内部 event envelope

事件只在内存/RPC中存在，writer按表schema投影；transport字段不应擅自落入目标表。

```text
MetricEventEnvelope {
  event_schema_version: string
  event_id: uuid                 # transport幂等ID，不代替表主键
  record_type: enum              # train_step/group/eval/... 
  target_table: enum
  producer: {role, dp_rank, tp_rank, sp_rank, worker_id}
  ownership: {is_dp_owner, replica_group_id}
  identity: {profile_name, seed, global_step?, optimizer_step?, checkpoint_step?}
  observation_phase: enum
  emitted_monotonic_ns: int64
  payload_kind: stats | raw_rows | availability | benchmark
  payload: detached CPU scalars/lists only
  availability: {record/family/metric -> available, reason}
  definition_versions: map<string,string>
}
```

约束：事件producer先验证shape/mask/K；失败发送availability event，不发送猜测值。任何Tensor进入队列前必须 `detach`、局部reduce、转CPU scalar/有限list并释放graph。rank0队列有界；core event队列满只能阻塞或明确fail，不能silent drop（`spec/03_METRICS_STORAGE_CONTRACT.md:305-316`）。

### 4.2 Worker sufficient-stat payload

统一使用契约建议的统计包（`spec/03_METRICS_STORAGE_CONTRACT.md:333-374`）：

```text
MomentsStats: sum, sum_sq, count, nan_count, masked_count, min, max
RateStats: numerator_count, count, nan_count, masked_count
IntegerSumStats: value_sum, count
LossStats: numerator_sum, normalizer_sum, aggregation_unit_count, loss_agg_mode
SmallQuantilePayload: detached finite values OR versioned sketch  # 仅小probe p05
RawRowsPayload: schema-typed rows                            # eval/group等raw facts
```

- mean/std在driver按全局 `sum/sum_sq/count` 计算；默认population std，算法进入definition version。不同mask永不共享count（`target_variables.md:1509-1518`）。
- loss需保留真实 `agg_loss` normalizer，不能简单平均worker/microbatch scalar。
- min/max做全局min/max；rate只合并numerator/count。
- TP/SP复制先去重；只有DP owner发包。`aggregation_worker_count`是实际独立贡献者数，不是CUDA world size。
- raw eval/group rows由driver已有全局batch产生最稳妥；若worker产生，必须附stable key且rank0检查重复。

### 4.3 rank0 writer边界

作者driver本来就是“single controller，轻量advantage在driver执行”（`ray_trainer.py:966-972`），因此保守方案是：worker只返回充分统计/有限raw facts；driver合并、计算最终float；**唯一rank0 writer**异步写 detached CPU Arrow batches。训练线程只enqueue并在checkpoint/结束/critical failure时做有界flush。

writer必须分别计 `metrics_compute_time` 与 `metrics_write_time`，不把写盘混入 `train/step_time`（`spec/03_METRICS_STORAGE_CONTRACT.md:318-329`）。磁盘失败默认写 `write_failure` 并按 `fail_training_on_logging_error` 决定训练是否失败；无论如何不得伪造已提交状态。

## 5. Append-only Parquet/JSON边界、schema和resume

### 5.1 文件/表边界

严格按不同粒度分表，禁止宽表混装（`spec/03_METRICS_STORAGE_CONTRACT.md:121-160`）：

| artifact | 粒度 / 主键 | 写法 |
|---|---|---|
| `run_config.json`, `platform_config_snapshot.json`, `schema_manifest.json`, `run_status.json` | run单例 | temp + optional fsync + atomic rename；标准JSON null |
| `support_definition.json`, `probe_definition.json` | run内定义单例 | 首次使用前原子写；恢复时hash必须一致 |
| `eval_dataset_manifest` | dataset version × question；PK=`eval_dataset_name+eval_dataset_version+question_id` | 契约目录写作 `eval_dataset_manifest.parquet`（`spec/03_METRICS_STORAGE_CONTRACT.md:147-151`，`spec/target_variables.md:1642`）；保守实现为一次性原子Parquet或append-only dataset，不能既JSON又Parquet产生双权威 |
| `train_step_metrics` | `profile+seed+global_step` | append-only parts |
| `train_group_metrics` | `profile+seed+global_step+group_id` | append-only parts |
| `eval_question_results` | `profile+seed+checkpoint_step+question_id+generation_id` | checkpoint partition可选 |
| `eval_clean_topk` | 上一PK + `latent_position` | checkpoint partition可选；list IDs/probs等长 |
| `gumbel_diagnostics` | `profile+seed+diagnostic_run_id+diagnostic_batch_index` | 仅diagnostic/smoke |
| `support_metrics` | `profile+seed+global_step+group_id+trajectory_id+trajectory_class` | append-only parts |
| `support_benchmark_metrics` | `profile+seed+global_step`（一个实际Support step一行） | append-only parts；若未来同step多attempt必须显式版本/新run，不能偷改PK |
| `probe_metrics` | `profile+seed+checkpoint_step+probe_batch_id+trajectory_group+latent_position_group` | append-only parts |
| `probe_benchmark_metrics` | `profile+seed+checkpoint_step+probe_batch_id` | append-only parts |

动态权威表清单由 `spec/03_METRICS_STORAGE_CONTRACT.md:55-92` 固定；JSONL只能是有上限的debug副本，不是权威源（`spec/03_METRICS_STORAGE_CONTRACT.md:94-108`）。

### 5.2 schema/version

- `_schema.json`/manifest每字段至少包含 name、logical/physical type、nullable、stage、record_type、unit、definition_version、PK member、availability family（`spec/03_METRICS_STORAGE_CONTRACT.md:163-196`）。保留契约中的斜杠字段原名，不擅自alias。
- int step/count/ID用int64，token id int32或int64，指标默认float64（源为float32时在manifest说明），list使用 `list<int32>`/`list<float32>` 并同时写K；布尔严格bool。
- unavailable值写null且reason非空；0绝不表示missing（`spec/03_METRICS_STORAGE_CONTRACT.md:198-207`）。
- schema version控制列集合/物理类型；definition version控制语义、mask、count、时点、聚合、EOS、分类、top-K空间、alignment符号（`spec/03_METRICS_STORAGE_CONTRACT.md:403-435`）。schema/definition不兼容时resume应拒绝，而不是自动cast。

### 5.3 part提交与幂等

每part流程固定为：同目录临时文件 → close → 可读/schema/row count/PK batch检查 → atomic rename最终文件 → 原子更新 `_SUCCESS_PARTS.json` → 推进writer checkpoint。崩溃后的tmp无效；已rename part有效；manifest可扫描重建（`spec/03_METRICS_STORAGE_CONTRACT.md:264-301`）。

`_SUCCESS_PARTS.json` 每part至少记录：part sequence、UUID、row count、schema/version hash、PK min/max（简单step表）、PK hash摘要、content checksum、committed_at。part filename sequence只增不复用。

重复处理：

1. enqueue batch内先查PK重复；
2. writer查本进程已提交PK索引；
3. resume加载manifest并对可能相交part做PK精查，不能只信min/max；
4. 相同PK+相同payload也默认停止/报告，不静默幂等覆盖；
5. eval/probe同checkpoint重跑：当前schema无 `evaluation_run_id`，所以同一run拒绝；需新输出run目录（`spec/03_METRICS_STORAGE_CONTRACT.md:441-456`）；
6. checkpoint之后的“未来记录”隔离到quarantine并报告，不删除、不混入新run；
7. validator最终全量检查PK与manifest一致。

## 6. availability状态机和保守默认

| 情况 | record/family处理 |
|---|---|
| family配置关闭（Gumbel diagnostic、credit） | 字段/schema仍存在；family false + `disabled_by_config`；不执行计算 |
| producer接口不存在 | null + `missing_runtime_interface` |
| upstream commit/schema不匹配白名单 | null + `unsupported_upstream_version` |
| mask count=0 | null，count=0，family/metric false + `empty_effective_mask` |
| shape/K/device不符 | null + `shape_mismatch`；绝不reshape/truncate/broadcast |
| stable key/顺序无法核对 | null + `alignment_failed` |
| Spearman常量 / credit零梯度 | 对应metric false + `degenerate_constant_vector` / `zero_gradient`；其它family不必整条失败 |
| controlled probe异常 | probe/family false + `runtime_probe_failed`；详细stack只入日志 |
| 写盘失败 | part不提交，run status记录 `write_failure` |

默认建议：Stage 1/2字段在adapter存在且probe通过后启用；Gumbel diagnostic false/disabled；Support按配置触发但在接口未实现时 false/missing；checkpoint probe/onesided配置默认true但实现未验证前 false/missing；credit families false/disabled。关闭credit不应让整条probe record失败（`target_variables.md:1323-1363`）。

## 7. 明确禁止字段和禁止持久化内容

### 7.1 schema中不得出现

`eval_metrics` checkpoint汇总、`train/gradient_norm`、可离线恢复的`train/throughput`、clean/noisy聚合对比、embedding shift、每step `onesided/*`/`credit/*`、`gumbel/one_sided_mean/std`、`group/incorrect_*`、`group/invalid_*`、`group/multi_correct`、被删除的signal/mask派生率等完整负面清单见 `spec/target_variables.md:50-84`。Support表还明确禁止 `incorrect`、`invalid`、`current_topk_ids`、`checkpoint_step`（`target_variables.md:1056-1064`）。

作者现有 `actor/grad_norm`（`dp_actor.py:610-612`）、`perf/throughput`（`metric_utils.py:240-243`）与validation汇总（`ray_trainer.py:770-784`）只能继续作为非权威console diagnostics，不得映射进目标Parquet schema。

### 7.2 只可短期内存，不得长期持久化或跨worker全量传输

raw full-vocabulary Gumbel、full logits/hidden/embeddings、完整训练梯度、长期计算图、rollout/pre-update完整top-K训练缓存、perturbed component scores、component log-prob、全量surrogate/flip mask、完整`u_i/q_i`、valid component mask、worker间完整top-K集合。唯一例外是eval已有clean top-K逐有效位置raw facts，以及小型probe内存中的有限detached margin样本；完整规则见 `spec/target_variables.md:1459-1485`。

## 8. `requirements_traceability_matrix.md` 覆盖方案

主智能体的matrix必须对 `target_variables.md` 中每个**字面目标字段**逐行建账，而不是只给family一行。最低列由主规范固定：`field_name, stage, record_type, semantic_definition, observation_phase, effective_mask, aggregation_method, storage_table, schema_type, implementation_module, test_id, default_enabled, availability_behavior, status`（`spec/01_CODEX_MASTER_PROMPT.md:119-154`）。建议额外加 `spec_path_line, upstream_path_line, evidence_level, count_field, definition_version_field, forbidden_persistence`。

覆盖必须包括四类：

1. **所有持久化动态字段**：逐表复制 `target_variables.md:294-334,340-467,513-533,717-747,1028-1087,1367-1444` 的每个字段；`<family>`和“必要的metric availability”模板必须展开为具体列。
2. **所有静态定义/config字段**：`run_config.json` 全字段（`target_variables.md:1522-1615`）、`support_definition.json`（`1004-1024`）、`probe_definition.json`（`1099-1128`），每项写source/config adapter和schema type。
3. **所有memory-only逻辑变量**：统一变量接口与 Stage 2/3/4 intermediates（`target_variables.md:92-115,618-625,804-838,844-966,1157-1201,1233-1245`），`storage_table="memory_only_not_persisted"`，并标注禁止持久化；不能因不落盘而从matrix遗漏。
4. **负面要求**：单独negative-requirement section覆盖 `target_variables.md:50-84,1459-1485`，test验证schema/part中不存在。它们不计“implemented metric”，但必须有test_id。

状态只可用 `planned/implemented/verified/unavailable_with_reason/blocked`。当前只读阶段所有待建adapter应为 `planned`；经静态证明但未实际运行也不能写verified。coverage生成器应从canonical field inventory与matrix做集合差，要求 `missing_fields=[]`；credit/diagnostic默认关闭仍需schema与reason，不能算missing（`spec/05_VALIDATION_AND_DELIVERABLES.md:399-434`）。

建议测试ID族：`T-SCHEMA-*`, `T-STATS-*`, `T-MASK-*`, `T-EVAL-*`, `T-GROUP-*`, `T-SUPPORT-*`, `T-PROBE-*`, `T-STORAGE-*`, `T-NEGATIVE-*`。统计/mask、stable ID、Support、credit、storage和resume的最低测试内容已由 `spec/05_VALIDATION_AND_DELIVERABLES.md:109-230,309-359` 固定。

## 9. Phase B/C 的优先 adapter 点与停止条件

1. **P0：stable identity adapter**，紧跟 `batch.repeat` 后、任何balance/filter前（`ray_trainer.py:1064-1078`）；否则group/winner/Support全部无法可靠对齐。
2. **P0：actor loss-stat adapter**，在 `compute_policy_loss` 邻接处输出ratio/KL/clip/loss充分统计，并在overlong zero之后构造eligible latent/zero-adv/final advantage统计（`dp_actor.py:551-575`）。
3. **P0：old-log-prob top-K adapter**，在producer处修正response next-token slice、K配置化、dynamic-batch回序；仅Support step选中trajectory输出有限统计。严禁消费当前错误的`topk_logits`返回。
4. **P0：SGLang mixture adapter**，消费已经存在的 `output_topk_prob_list`，先压缩为per-position scalar再随trajectory过filter；不导出full-vocab或长期weights。
5. **P0：winner return adapter**，从advantage函数同一次winner选择直接导出stable trajectory ID/mean，old-log-prob缺失时标 unavailable，不走随机fallback作为记录。
6. **P1：eval raw writer adapter**，在 `_validate` union生成batch后、汇总前构造question/generation rows和已有clean top-K；不补forward。
7. **P1：checkpoint probe adapter**，独立固定batch/RNG上下文，复用训练reward/advantage链；one-sided forward-only先行，credit默认关闭并在完整安全测试后才能开放。
8. **P1：optimizer-success clock adapter**，从实际 `_optimizer_step` 返回success flag并由driver一致性累计；checkpoint metadata必须保存。
9. **P1：rank0 event/writer adapter**，所有producer只发detached stats，Parquet原子append、PK/resume/validator按§5。

以下任一条件触发family unavailable而不是“继续凑数”：stable ID丢失、response/top-K next-token对齐无法证明、K不一致、TP复制ownership未知、custom reward无法给correctness、old/current policy时点不符、Flash-Attn latent logprob分支未执行、probe RNG/参数/.grad无法完整恢复。
