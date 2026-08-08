# target_variables.md

## 0. 文档目的与适用范围

本文档汇总以下四份阶段规范中**新训练脚本需要采集、聚合、记录或为计算而临时暴露的全部目标变量**：

1. `./spec/01_STAGE_LOGGING_INFRASTRUCTURE.md`
2. `./spec/02_STAGE_LOW_COST_MECHANISM_METRICS.md`
3. `./spec/03_STAGE_SUPPORT.md`
4. `./spec/04_STAGE_CHECKPOINT_PROBE.md`

本文档区分六类字段：

| 类别 | 含义 | 是否计入核心指标数 |
|---|---|---:|
| `core_metric` | 最终用于分析训练变化的观测指标 | 是 |
| `diagnostic` | 独立 smoke/diagnostic 的实现健康检查 | 否 |
| `raw_fact` | 逐题、逐 group 或逐位置的原始事实 | 否 |
| `support_count` | 均值、标准差和比例的有效分母 | 否 |
| `metadata` | 实验身份、时间点、版本、ID、availability | 否 |
| `benchmark` | 日志、Support、probe 的时间、显存和处理规模 | 否 |

> **与旧代码解耦。** 本文档不指定任何变量当前位于哪个文件、类、函数、batch 字段或 Tensor 名中。新脚本应按照本文给出的语义重新设计采集接口；不得把旧实现中的变量位置当作必须继承的约束。

> **运行时事实优先。** 四份规范没有固定所有 Tensor 的 shape、dtype、device、`requires_grad` 和 DP/TP 语义。这些信息必须在目标 GPU Linux/VSCode 环境中由 runtime probe 确认，并写入 `platform_config_snapshot.json`。不得根据本文档猜测或静默 reshape、cast、截断、广播或移动 device。

---

## 1. 总体变量清单

### 1.1 核心观测指标数量

| 阶段 | 核心指标数 | 说明 |
|---|---:|---|
| Stage 1 | 10 | 基础训练指标 |
| Stage 2 | 6 | noisy mixture、zero-advantage mask、reward/advantage signal |
| Stage 3 | 2 | rollout–pre-update top-K Support |
| Stage 4 | 11 | 7 个 one-sided/FlipGrad 指标与 4 个 credit 指标 |
| **合计** | **29** | Stage 4 的 4 个 credit 指标默认关闭，验证后才能开启 |

另有：

- 5 个 Gumbel diagnostic 指标，仅允许独立 diagnostic/smoke 运行；
- checkpoint eval 的逐题、逐 generation 原始结果；
- checkpoint eval clean top-K 的逐 latent-position 原始事实；
- 逐 group 原始事实；
- Support 和 probe benchmark；
- 各统计族的有效 count、availability 和版本字段。

### 1.2 当前明确不记录的内容

当前规范明确删除或禁止：

```text
checkpoint 级 eval_metrics 汇总表
train/gradient_norm
train/throughput（可由 token count / step time 离线恢复时）
clean/noisy top-K 聚合对比指标
clean/noisy latent embedding shift
完整 vocabulary logits
完整 hidden states
完整梯度
长期保存计算图 Tensor
训练期通用 gradient hook
每 step onesided/* 或 credit/*
gumbel/one_sided_mean
gumbel/one_sided_std
group/incorrect_trajectory_count
group/invalid_trajectory_count
group/invalid_generated_token_count
group/invalid_response_length_max
group/invalid_avg_generated_tokens
group/multi_correct
signal/advantage_mean
signal/positive_adv_rate
signal/negative_adv_rate
signal/nonzero_adv_rate
signal/valid_trajectory_rate
signal/correct_trajectory_rate
mask/effective_latent_token_count
mask/first_token_mask_rate
mask/selected_correct_path_rate
mask/valid_sample_rate
```

---

## 2. 统一变量接口与语义

本节只规定新训练脚本需要提供的逻辑变量及其语义，不指定这些变量应位于哪个文件、类、函数或已有 Tensor 中。

| 目标变量 | 保存方式 | 关键语义 |
|---|---|---|
| `global_step` | metadata | 一次完整 rollout、reward、advantage 与 actor update 所属的外层训练 step |
| `optimizer_step` | metadata | 全局累计成功参数更新次数 |
| `group_id` | raw fact / metadata | 同一 prompt 的 rollout group 标识 |
| `trajectory_id` | raw fact / metadata | 同一 group 内稳定且可复现的 trajectory 标识；必须在任何 reorder/filter/select 前确定 |
| `rollout_noisy_topk_token_ids` | Stage 3 临时缓存 | rollout 时实际用于 latent state 构造的 noisy top-K token IDs |
| `rollout_perturbed_topk_scores` | Stage 4 临时量 | rollout 时对应 top-K component 的扰动后分数 |
| `noisy_mixture_weights` | 聚合后保存 | 实际参与 latent state 构造的 noisy mixture 权重 |
| `clean_topk_token_ids` | Stage 1 eval raw fact | checkpoint eval 时未加噪分布的 top-K token IDs |
| `clean_topk_probs` | Stage 1 eval raw fact | checkpoint eval 时未加噪分布的 top-K 概率 |
| `pre_update_topk_token_ids` | Stage 3 临时缓存 | 当前 actor update 开始前，由同一模型状态得到的确定性 top-K token IDs |
| `optimal_correct_trajectory_id` | raw fact / metadata | Optimal Correct Path 选中的稳定 trajectory ID |
| `optimal_correct_mean_old_log_prob` | raw fact | Optimal Correct Path winner 的平均旧策略 log-probability |
| `trajectory_mean_old_log_prob` | 选中轨迹保存 | 有效 response token 上旧策略 log-probability 的掩码平均 |
| `reward` | 聚合或轨迹分类 | 训练实际使用的 reward 信号 |
| `advantage` | 聚合或临时量 | 训练实际使用的最终 advantage 信号 |
| `surrogate_margin` | Stage 2 接口；Stage 4 聚合 | rollout 扰动后 component score 与当前 component log-probability 的差值 |
| `flipgrad_trigger_mask` | Stage 2 接口；Stage 4 聚合 | 标记满足 FlipGrad 触发条件的有效 component |
| `component_log_prob` | Stage 4 临时量 | 用于 policy-gradient credit 和 surrogate margin 的可微 component log-probability |
| `u_i` | Stage 4 临时量 | 第 `i` 个 top-K component 的局部 policy-gradient credit |
| `q_i` | Stage 4 临时量 | 基于 `|u_i|` 归一化得到的 credit concentration 分布 |
| `valid_latent_position_mask` | 临时 mask | 标记有效 latent position，排除 prompt、padding、普通 hard token 与 loss-mask 排除位置 |
| `valid_latent_component_mask` | 临时 mask | 标记用于 Delta、FlipGrad 和 credit 统计的有效 latent component |

新脚本可以自行设计文件结构、模块边界和内部变量名，但写盘字段、统计定义、时间语义与有效 mask 必须符合本文档。

---

## 3. 公共实验身份、时间与记录质量字段

### 3.1 实验身份

```text
profile_name
seed
```

实验目录：

```text
<output_root>/<profile_name>/seed_<seed>/
```

### 3.2 公共动态上下文字段

普通训练记录使用：

```text
profile_name
seed
metric_scope
global_step
optimizer_step
observation_phase
learning_rate
wall_clock_seconds
cumulative_train_samples
cumulative_rollout_tokens
cumulative_gpu_hours
```

仅 checkpoint 派生记录额外使用：

```text
checkpoint_step
```

时间语义：

| 表/阶段 | `observation_phase` | optimizer 语义 |
|---|---|---|
| `train_step_metrics` | `post_update` | 当前 global step 的 actor update 已完成 |
| `train_group_metrics` | `post_advantage_pre_update` | 当前 step actor update 尚未开始 |
| `support_metrics` | `pre_update_old_log_prob` | old-log-prob forward 开始时的累计 optimizer step |
| `eval_question_results` | `checkpoint_eval` | checkpoint 对应状态 |
| `eval_clean_topk` | `checkpoint_eval` | checkpoint 对应状态 |
| `probe_metrics` | `checkpoint_probe` | checkpoint 元数据中的累计 optimizer step |

### 3.3 `checkpoint_step`

- checkpoint eval 和 checkpoint probe 始终保存；
- 普通 `train_step_metrics`、`train_group_metrics`、`support_metrics` 不保存；
- `global_step` 是执行上下文，`checkpoint_step` 是被评估或 probe 的模型 checkpoint 所属训练 step；
- 重跑旧 checkpoint 时两者允许不同。

### 3.4 每张动态表的记录质量字段

基础字段：

```text
record_available
record_unavailable_reason
aggregation_worker_count
record_version
is_resume_run
resume_from_step
metrics_compute_time
metrics_write_time
```

每个统计族：

```text
<family>_available
<family>_unavailable_reason
```

单指标可独立失败时：

```text
<metric_name>__available
<metric_name>__unavailable_reason
```

统计包内部至少支持：

```text
sum
sum_sq
count
nan_count
masked_count
min
max
```

规则：

- worker 返回 `sum/sum_sq/count` 等统计包；
- driver 合并后再算全局 mean/std/rate；
- 禁止先求 worker mean 再简单平均；
- 不同有效 mask 的指标不得共享 count；
- 只有 driver/rank 0 写最终文件。

---

# 4. Stage 1：日志基础设施与基础训练变量

## 4.1 `train_step_metrics` 的 10 个核心指标

一行对应一个完成后的 `global_step`。

| 字段 | 类型 | 定义 | 记录要求 |
|---|---|---|---|
| `train/policy_loss` | core metric | 现有 PPO/actor policy loss | 复用训练路径，不增加额外 forward |
| `train/entropy` | core metric | 现有 PPO/actor policy entropy | 必须记录 entropy 来源、概率空间、mask 与定义版本；不能把 noisy mixture entropy 冒充 policy entropy |
| `train/kl` | core metric | 现有 KL 路径 | 复用 loss 路径 |
| `train/clip_fraction` | core metric | 现有 PPO clipping 统计 | 记录真实定义版本，不另写近似公式 |
| `train/importance_ratio_mean` | core metric | PPO importance ratio | 与 std 使用同一有效 ratio mask |
| `train/importance_ratio_std` | core metric | PPO importance ratio | 由全局 `sum/sum_sq/count` 得到 |
| `train/response_length` | core metric | 当前代码真实 response-length mask/counting rule | 保存定义版本和 EOS/stop counting rule 版本 |
| `train/latent_length` | core metric | 当前代码真实 latent-position 规则 | 保存定义版本 |
| `train/generated_token_count` | core metric | 当前 global step 最终训练 rollout trajectory 的长度总和 | worker 求整数和，driver 求和；不得由均值×batch size 估算 |
| `train/step_time` | core metric | 一个训练 global step 的真实耗时 | 与日志计算/写盘时间分开 |

### 4.1.1 `train/entropy` 的配套定义字段

```text
entropy_source
entropy_probability_space
entropy_mask_definition
entropy_definition_version
```

### 4.1.2 长度字段的配套定义字段

```text
response_length_definition_version
latent_length_definition_version
length_counting_rule_version
```

EOS/stop 是否计入长度必须由目标 runtime 验证。日志不得改变训练原有 mask 语义。

### 4.1.3 `train/generated_token_count` 的固定语义

```text
generated_token_count_definition_version="paper_mixed_trajectory_sum_v1"
generated_token_count_scope="final_training_rollout_trajectories"
```

聚合：

```text
train/generated_token_count
= sum_workers(sum_local_trajectory_lengths)

cumulative_rollout_tokens
+= train/generated_token_count
```

约束：

- 使用进入当前 step 最终训练 rollout 集合的 trajectory；
- 进入 reward/advantage 集合的 overlong trajectory 不得因 actor update 后过滤而丢失；
- 某个 retry 结果只有在被明确保留为独立 trajectory 时才按自身长度计一次；
- 内部生成后丢弃、且不属于最终训练 rollout 集合的 retry/候选成本不能混入本指标；
- 若未来需要记录所有内部重试/丢弃计算成本，必须使用新的指标名；
- 不平均 worker token count；
- 不用 `mean_length * configured_batch_size` 估算。

### 4.1.4 `train_step_metrics` 完整字段框架

```text
profile_name
seed
metric_scope="train_step"
global_step
optimizer_step
observation_phase="post_update"
learning_rate
wall_clock_seconds
cumulative_train_samples
cumulative_rollout_tokens
cumulative_gpu_hours

train/policy_loss
train/entropy
train/kl
train/clip_fraction
train/importance_ratio_mean
train/importance_ratio_std
train/response_length
train/latent_length
train/generated_token_count
generated_token_count_definition_version
generated_token_count_scope
length_counting_rule_version
train/step_time

各统计族 count
record_available
record_unavailable_reason
train_core_available
train_core_unavailable_reason
必要的 <metric_name>__available
必要的 <metric_name>__unavailable_reason
aggregation_worker_count
record_version
metrics_compute_time
metrics_write_time
```

普通训练表不得保存 `checkpoint_step`。

---

## 4.2 Checkpoint eval 数据集 manifest

`eval_dataset_manifest.parquet` 每个数据集版本、每道题保存一次：

```text
eval_dataset_name
eval_dataset_version
question_id
prompt_hash
reference_answer
reference_answer_hash
```

规则：

- manifest 是 `reference_answer` 的权威来源；
- 逐题表可复制 reference answer 以便自包含，但必须核对 hash；
- 禁止在 manifest 和逐题表中独立修改 reference answer。

---

## 4.3 `eval_question_results`

每个 checkpoint × 每道题 × 每次 generation 恰好一行。

```text
profile_name
seed
global_step
checkpoint_step
observation_phase="checkpoint_eval"
question_id
generation_id
predicted_answer
reference_answer
reference_answer_hash
is_correct
response_length
latent_length
is_valid_response
generation_failure_reason
reward_or_score
prompt_hash
generation_seed
length_counting_rule_version
evaluation_rule_version
eval_dataset_version
clean_topk_available
clean_topk_unavailable_reason
record_available
record_unavailable_reason
```

主键：

```text
profile_name
+ seed
+ checkpoint_step
+ question_id
+ generation_id
```

完整性约束：

- `eval_generations_per_question >= 1`；
- `generation_id` 必须完整覆盖 `[0, eval_generations_per_question - 1]`；
- 单次 generation 使用 `generation_id=0`；
- 正常、答错、解析失败、超时、生成失败均保留一行；
- `predicted_answer` 保存完整文本，失败时可为 `null`，但必须写 `generation_failure_reason`；
- 不写 checkpoint 级 `eval_metrics`；
- accuracy、sample count、长度 mean/std、valid response rate 均从逐题 raw facts 离线恢复。

---

## 4.4 `eval_clean_topk`

新脚本必须在 checkpoint eval 生成过程中直接提供：

```text
clean_topk_token_ids
clean_topk_probs
```

它们表示同一次评估 forward 中未加噪分布的 top-K token IDs 与对应概率。

每个 checkpoint × question × generation × 有效 latent position 一行：

```text
profile_name
seed
global_step
checkpoint_step
observation_phase="checkpoint_eval"
question_id
generation_id
latent_position
clean_topk_token_ids
clean_topk_probs
clean_topk_k
clean_topk_available
clean_topk_unavailable_reason
clean_topk_source="checkpoint_eval_clean_distribution_topk"
record_available
record_unavailable_reason
```

主键：

```text
profile_name
+ seed
+ checkpoint_step
+ question_id
+ generation_id
+ latent_position
```

约束：

- 只转存已有 clean top-K；
- 不增加 forward；
- 不重新执行 full-vocabulary softmax 或 top-k；
- 不保存 full logits、hidden states 或 embedding；
- 位置筛选复用 latent-position sentinel/mask；
- 排除 prompt、padding 和普通 hard token；
- clean top-K 是 raw fact，不计算 clean/noisy 聚合对比；
- 字段缺失、shape 不符或无法对齐时，写 availability 和原因，不得回退重算。

---

# 5. Stage 2：低成本机制变量

## 5.1 Gumbel diagnostic：仅独立 diagnostic/smoke

正式训练默认：

```text
gumbel_diagnostics_enabled=false
gumbel_diagnostics_mode="disabled"
```

仅独立 diagnostic/smoke 允许开启。

### 5.1.1 指标

| 字段 | 定义 |
|---|---|
| `gumbel/raw_mean` | raw Gumbel 的均值 |
| `gumbel/raw_std` | raw Gumbel 的标准差 |
| `gumbel/lower_clip_rate` | raw Gumbel 小于等于下裁剪边界的比例 |
| `gumbel/upper_clip_rate` | raw Gumbel 大于等于上裁剪边界的比例 |
| `gumbel/zero_rate` | 单侧变换并应用 scale 后恰为零的比例 |

共享分母：

```text
gumbel/raw_count
gumbel/one_sided_count
```

统计中间量：

```text
raw_sum
raw_sum_sq
raw_count
lower_clip_count
upper_clip_count
one_sided_zero_count
one_sided_count
```

### 5.1.2 `gumbel_diagnostics` schema

```text
profile_name
seed
diagnostic_run_id
diagnostic_batch_index
gumbel_diagnostics_mode
gumbel/raw_mean
gumbel/raw_std
gumbel/lower_clip_rate
gumbel/upper_clip_rate
gumbel/zero_rate
gumbel/raw_count
gumbel/one_sided_count
gumbel_compute_time_seconds
record_available
record_unavailable_reason
gumbel_available
gumbel_unavailable_reason
```

约束：

- 在噪声采样与单侧变换实际执行处做 GPU 局部 reduce；
- 不保存或跨 worker 传输 raw full-vocabulary Gumbel Tensor；
- 正式训练不为 diagnostic 引入跨模块的训练 step 控制依赖；
- 不把“每步计算、低频写盘”描述成低频采集。

---

## 5.2 Noisy mixture

输入变量：

```text
noisy_mixture_weights
```

该变量必须是本次 rollout 中实际用于构造 latent state 的 noisy top-K mixture 权重。

记 noisy mixture weight 为 `α_i`。

核心指标：

```text
mixture/effective_k_noisy
mixture/top1_weight_noisy
```

定义：

```text
mixture/effective_k_noisy
= exp(-sum_i α_i log α_i)

mixture/top1_weight_noisy
= max_i α_i
```

共享：

```text
mixture/noisy_count
mixture_available
mixture_unavailable_reason
```

要求：

- 仅使用实际 noisy mixture weights；
- 不计算 clean mixture；
- 不增加完整 forward；
- 退化权重、NaN、空 mask 必须通过 count 和 availability 表达。

---

## 5.3 Zero-advantage mask

核心指标：

```text
mask/zero_advantage_rate
```

分母：

```text
mask/eligible_latent_token_count
```

分母定义：

> 原本有资格参与 latent policy loss 的有效 latent token 数量。

必须排除：

```text
padding
prompt token
普通 hard response token
loss-mask 排除位置
非 latent position
```

中间量：

```text
eligible_latent_mask
zero_advantage_mask
zero_advantage_count
eligible_latent_token_count
```

聚合：

```text
mask/zero_advantage_rate
= zero_advantage_count / mask/eligible_latent_token_count
```

zero 判定阈值必须固定并由配置或定义版本记录。eligible count 和 zero count 必须使用完全相同的 mask。

availability：

```text
mask_available
mask_unavailable_reason
```

---

## 5.4 训练信号

核心指标：

```text
signal/reward_mean
signal/reward_std
signal/advantage_std
```

分母：

```text
signal/reward_count
signal/advantage_count
```

availability：

```text
signal_available
signal_unavailable_reason
```

要求：

- 使用训练真实 reward/advantage Tensor；
- reward mean/std 的统计单位必须由代码审计明确，是 trajectory、token 还是其他真实单位；
- `signal/reward_mean` 与 `signal/reward_std` 共享完全相同的 reward mask 与 count；
- `signal/advantage_std` 使用最终参与训练语义的有效 advantage 与独立 count；
- 不重复记录 correct rate，正确性由 group raw facts 恢复。

---

## 5.5 稳定 group 与 trajectory 标识

### 5.5.1 `trajectory_id` 创建时机

必须在：

```text
batch.repeat(..., interleave=True) 完成之后
且同一 prompt 的 n 条 trajectory 已实际存在之后
且任何 _balance_batch / reorder / filter / select / 选样之前
```

按同一 `group_id` 在 repeat 后 batch 中的稳定出现顺序分配。

最小唯一键：

```text
global_step + group_id + trajectory_id
```

### 5.5.2 轨迹分类

只使用：

```text
trajectory_class ∈ {"correct", "non_correct"}
is_overlong_or_truncated_by_length ∈ {true, false}
```

注意：

- correct/non_correct 是互斥二分类；
- overlong 是可与上述二分类重叠的独立布尔属性；
- 禁止使用 `incorrect` 或 `invalid` 作为第三个互斥类别；
- 保存 `trajectory_classification_version` 和 `overlong_definition_version`。

---

## 5.6 `train_group_metrics` 原始事实

每个 `global_step × group_id` 一行：

```text
profile_name
seed
global_step
optimizer_step
observation_phase="post_advantage_pre_update"
group_id
prompt_id_or_hash

group/trajectory_count
group/correct_trajectory_count
group/non_correct_trajectory_count
group/overlong_trajectory_count
group/overlong_generated_token_count
group/overlong_response_length_max
group/zero_variance_reward

optimal_correct_trajectory_id
optimal_correct_mean_old_log_prob
group_definition_version
trajectory_classification_version
overlong_definition_version
record_available
record_unavailable_reason
group_available
group_unavailable_reason
```

一致性：

```text
group/correct_trajectory_count
+ group/non_correct_trajectory_count
= group/trajectory_count
```

`group/overlong_trajectory_count` 与 correct/non_correct 重叠，不能与二者相加求总数。

可离线派生但训练期不物化：

```text
derived_group/correct_rate
derived_group/non_correct_rate
derived_group/overlong_rate
derived_group/overlong_response_length_max
derived_group/zero_variance_reward_rate
derived_group/multi_correct_rate
```

---

## 5.7 Optimal Correct Path

候选集合：

- 使用本文规定的 positive first-step advantage 候选语义；
- 不使用 reward-only 近似选择器。

轨迹评分：

```text
trajectory_mean_old_log_prob
= masked_mean(old_policy_token_log_probs, valid_response_mask)
```

选择结果：

```text
optimal_correct_trajectory_id
optimal_correct_mean_old_log_prob
```

局部 batch 索引只能作为计算过程中的临时值，持久化前必须映射为稳定 `trajectory_id`。

同一份内存 winner 结果必须：

1. 写入 `train_group_metrics`；
2. 通过内存接口提供给 Stage 3；
3. 不允许 Stage 3 从 Parquet 回读；
4. 不允许根据后续被置零的 advantage 反推 winner。

---

## 5.8 `surrogate_margin` / FlipGrad 暴露接口

新脚本必须显式提供以下统一接口：

```text
surrogate_margin
flipgrad_trigger_mask
```

定义：

```text
surrogate_margin
= rollout_perturbed_topk_scores
  - component_log_prob

flipgrad_trigger_mask
= (advantage <= 0)
  & (surrogate_margin < 0)
```

Stage 2 仅允许实现并测试以下中间统计接口：

```text
sum
sum_sq
count
min
negative_count
near_zero_count
flipgrad_trigger_count
valid_latent_component_mask
```

Stage 2 不写 `onesided/*`，不执行额外 forward/backward，不注册训练期 gradient hook。真正的聚合和持久化仅在 Stage 4 checkpoint probe。

---

# 6. Stage 3：Rollout–Pre-update Top-K Support

## 6.1 两端变量

Rollout 端目标变量：

```text
rollout_noisy_topk_token_ids
```

Pre-update 端目标变量：

```text
pre_update_topk_token_ids
```

`pre_update_topk_token_ids` 必须来自：

> 本次 PPO actor update 开始前，对旧策略模型状态执行的一次受控、无梯度 forward。

禁止使用：

```text
update-policy micro-batch 中随 optimizer 更新变化的 top-K
未经过语义验证的 top-K 临时量
重新采样的 Gumbel
post-update forward
额外 forward
```

---

## 6.2 Support 时间字段

```text
global_step
optimizer_step_at_observation
observation_phase="pre_update_old_log_prob"
```

定义：

```text
optimizer_step_at_observation
= 旧策略无梯度 forward 开始时
  已成功完成的参数更新累计次数
```

不保存 `checkpoint_step`。

---

## 6.3 Support 轨迹选择中间量

Correct 轨迹复用：

```text
optimal_correct_trajectory_id
optimal_correct_mean_old_log_prob
```

Non-correct 候选条件：

```text
trajectory_class="non_correct"
is_overlong_or_truncated_by_length=false
```

评分：

```text
trajectory_mean_old_log_prob
= masked_mean(old_policy_token_log_probs, valid_response_mask)
```

选择结果：

```text
selected_non_correct_trajectory_id
selected_non_correct_mean_old_log_prob
```

并列时选择最小稳定 `trajectory_id`。

---

## 6.4 Latent position 对齐

新脚本必须显式构造：

```text
valid_latent_position_mask
```

该 mask 必须联合表达：

```text
latent position 身份
response 区间
实际 loss mask
实际 attention mask
runtime 验证后的 shape
```

不得依赖旧实现中的固定 sentinel 值或固定 Tensor 布局。

排除：

```text
prompt
padding
普通 hard token
无法严格对齐的位置
```

对齐键：

```text
global_step
group_id
trajectory_id
latent_position
```

若两侧长度、K、mask 或 trajectory 顺序不能严格核对，Support family 必须不可用；禁止静默截断、广播或补做 forward。

---

## 6.5 Support 指标

设 rollout top-K 集合为 `S_r`，pre-update top-K 集合为 `S_p`，大小均为 `K`。

```text
support/retention_rate
= |S_r ∩ S_p| / K
```

```text
support/top1_retention_rate
= 1[top1(S_r) ∈ S_p]
```

注意：top1 retention 是“rollout top1 是否仍在 pre-update top-K 集合中”，不是要求两侧 top1 完全相等。

共享分母：

```text
support/effective_position_count
```

中间统计量：

```text
intersection_fraction_sum
top1_retained_count
effective_position_count
```

对所有选中轨迹的有效 latent position 按 position count 加权。

---

## 6.6 `support_definition.json`

只写一次：

```text
support_definition_version
support_start_step
support_interval_steps
support_observation_phase="pre_update_old_log_prob"
support_current_policy_timepoint="pre_update_old_log_prob_forward"
rollout_topk_source="rollout_noisy_topk_token_ids"
pre_update_topk_source="pre_update_topk_token_ids"
trajectory_score_name="trajectory_mean_old_log_prob"
trajectory_score_formula="masked_mean(old_policy_token_log_probs, valid_response_mask)"
support_include_correct
support_include_non_correct
support_exclude_overlong
support_requires_remove_padding=true
latent_position_rule_version
platform_config_hash
```

---

## 6.7 `support_metrics`

每个 Support step × group × 选中 trajectory class 一行：

```text
profile_name
seed
global_step
optimizer_step_at_observation
observation_phase="pre_update_old_log_prob"
group_id
trajectory_id
trajectory_class
trajectory_mean_old_log_prob
trajectory_selection_rule_version
candidate_trajectory_count

support/retention_rate
support/top1_retention_rate
support/effective_position_count

record_available
record_unavailable_reason
support_available
support_unavailable_reason
record_version
```

不得出现：

```text
incorrect
invalid
current_topk_ids

checkpoint_step
```

---

## 6.8 `support_benchmark_metrics`

每个实际 Support step 一行：

```text
profile_name
seed
global_step
optimizer_step_at_observation
observation_phase="pre_update_old_log_prob"
support_extra_time_seconds
support_cache_peak_bytes
support_selected_trajectory_count
support_candidate_trajectory_count
support_benchmark/total_effective_position_count
record_available
record_unavailable_reason
support_available
support_unavailable_reason
```

注意：

- `support/effective_position_count` 是单个 trajectory 聚合行的分母；
- `support_benchmark/total_effective_position_count` 是整个 Support step 的 benchmark 总处理量；
- 两者 scope 不同，不得互换或重复写入同一行。

---

# 7. Stage 4：Checkpoint-only Probe

## 7.1 固定 probe 定义

`probe_definition.json` 只写一次：

```text
probe_definition_version
probe_batch_id
probe_prompt_ids
probe_generation_seed
probe_gumbel_seed
probe_sampling_parameters
probe_batch_size
probe_trajectories_per_prompt
probe_max_trajectories
probe_max_latent_positions
probe_use_independent_rng=true
probe_restore_training_rng=true
probe_performs_optimizer_step=false
probe_reward_manager_source="same_as_training_profile"
probe_reward_scorer_source="same_as_training_profile"
probe_group_construction_source="same_as_training"
probe_advantage_function_source="same_as_training"
probe_response_mask_source="same_as_training"
probe_first_token_selection_source="same_as_training"
probe_old_log_prob_timepoint="checkpoint_pre_probe_no_grad_forward"
onesided_probe_enabled
credit_probe_enabled
credit_alignment_probe_enabled
platform_config_hash
```

默认：

```text
checkpoint_probe_enabled=true
onesided_probe_enabled=true
credit_probe_enabled=false
credit_alignment_probe_enabled=false
```

---

## 7.2 Probe 动态时间字段

```text
global_step
optimizer_step
checkpoint_step
observation_phase="checkpoint_probe"
```

- `checkpoint_step`：被 probe checkpoint 的训练 step，始终保存；
- `global_step`：probe 执行上下文；
- `optimizer_step`：checkpoint 元数据中的累计成功 optimizer 更新数；
- 不得用当前进程临时计数替代 checkpoint 元数据。

---

## 7.3 Probe 必须复用的训练中间量

正式 probe 必须复用训练算法链：

```text
固定 prompt group
训练相同 reward manager
训练相同 reward scorer
训练相同 group construction
训练相同 response/loss mask
训练相同 overlong 处理
受控的旧策略无梯度 forward
训练相同 advantage function
训练相同 Optimal Correct Path
最终 advantage
valid latent-component mask
noisy mixture weights
rollout_perturbed_topk_scores
component_log_probs
surrogate_margin
flipgrad_trigger_mask
```

probe 产生的 trajectory/token 不得计入：

```text
train/response_length
train/latent_length
train/generated_token_count
cumulative_rollout_tokens
```

---

## 7.4 One-sided / Delta 指标

中间量：

```text
surrogate_margin
flipgrad_trigger_mask
valid_latent_component_mask
near_zero_mask
negative_mask
```

核心指标：

| 字段 | 定义 |
|---|---|
| `onesided/delta_mean` | valid `surrogate_margin` 均值 |
| `onesided/delta_std` | valid `surrogate_margin` 标准差 |
| `onesided/delta_p05` | valid `surrogate_margin` 的 5% 分位数 |
| `onesided/delta_min` | valid `surrogate_margin` 最小值 |
| `onesided/delta_negative_rate` | `surrogate_margin < 0` 的比例 |
| `onesided/delta_near_zero_rate` | 按固定 near-zero 阈值判定的比例 |
| `onesided/flipgrad_rate` | FlipGrad trigger 比例 |

分母：

```text
onesided/delta_count
onesided/flipgrad_count
```

要求：

- Delta mean/std/p05/min/negative/near-zero 默认共享同一 valid component mask；
- FlipGrad 的分母语义可能不同，必须使用独立 count；
- near-zero 阈值必须由固定定义版本或显式配置固定；
- `advantage == 0` 的语义必须通过 runtime 和代码审计确认；
- `one_sided_gumbel_noise_offset` 是静态配置，不能当作 Delta；
- 扰动后 component score 不能当作纯 Gumbel noise。

---

## 7.5 Credit 中间量与指标

局部 credit：

```text
u_i = -∂L_PG / ∂log p_i
```

归一化绝对 credit：

```text
q_i = |u_i| / (sum_j |u_j| + epsilon)
```

核心指标：

| 字段 | 定义 |
|---|---|
| `credit/top1_share` | 最大 `q_i` |
| `credit/effective_k` | `exp(-sum_i q_i log q_i)` |
| `credit/weight_credit_spearman` | noisy mixture weights 排序与 `|u_i|` 排序的 Spearman |
| `credit/surrogate_alignment_rate` | 经最小 autograd 符号测试确认后的 surrogate gradient 对齐比例 |

分母：

```text
credit/concentration_count
credit/spearman_count
credit/alignment_count
```

定义版本：

```text
credit_definition_version
surrogate_alignment_definition_version
```

要求：

- `credit/top1_share` 与 `credit/effective_k` 共享 concentration count；
- Spearman 与 alignment 各自使用独立 count；
- Spearman tie 规则必须固定；
- Spearman 退化时写 NaN、独立 count 和明确原因；
- zero-gradient alignment 不得伪造方向；
- 只允许一次受控 `torch.autograd.grad` 或等价局部 backward；
- 不使用 module-level backward hook；
- 不写入正常训练参数 `.grad`；
- 不保存完整梯度；
- credit 默认关闭，完成符号、显存、时间、RNG、参数不变性验证后才能开启。

---

## 7.6 Probe 分组字段

轨迹属性：

```text
trajectory_class ∈ {"correct", "non_correct"}
is_overlong_or_truncated_by_length
```

动态分组字段：

```text
trajectory_group
latent_position_group
```

推荐分组：

```text
trajectory_group:
  all
  correct
  non_correct
  overlong
  not_overlong

latent_position_group:
  all
  first_latent
  middle_latent
  last_latent
```

所有分组必须由同一次 probe 结果做 mask；禁止每个分组重新 backward。

---

## 7.7 Probe availability

每行：

```text
record_available
record_unavailable_reason
```

family 级：

```text
onesided_available
onesided_unavailable_reason
credit_concentration_available
credit_concentration_unavailable_reason
credit_spearman_available
credit_spearman_unavailable_reason
credit_alignment_available
credit_alignment_unavailable_reason
```

必要的单指标级：

```text
credit/weight_credit_spearman__available
credit/weight_credit_spearman__unavailable_reason
credit/surrogate_alignment_rate__available
credit/surrogate_alignment_rate__unavailable_reason
```

credit 默认关闭时：

```text
record_available=true
onesided_available=true
credit_*_available=false
credit_*_unavailable_reason="disabled_by_config"
```

关闭某个 family 不表示整条 probe 记录失败。

---

## 7.8 `probe_metrics`

每个 checkpoint × trajectory group × latent-position group 一行：

```text
profile_name
seed
global_step
optimizer_step
checkpoint_step
observation_phase="checkpoint_probe"
probe_batch_id
probe_definition_version
trajectory_group
latent_position_group

onesided/delta_mean
onesided/delta_std
onesided/delta_p05
onesided/delta_min
onesided/delta_negative_rate
onesided/delta_near_zero_rate
onesided/flipgrad_rate
onesided/delta_count
onesided/flipgrad_count

credit/top1_share
credit/effective_k
credit/weight_credit_spearman
credit/surrogate_alignment_rate
credit/concentration_count
credit/spearman_count
credit/alignment_count

record_available
record_unavailable_reason
onesided_available
onesided_unavailable_reason
credit_concentration_available
credit_concentration_unavailable_reason
credit_spearman_available
credit_spearman_unavailable_reason
credit_alignment_available
credit_alignment_unavailable_reason
必要的 <metric_name>__available
必要的 <metric_name>__unavailable_reason
probe_rng_restore_succeeded
record_version
```

固定 prompt、seed、采样参数和 probe 上限只写在 `probe_definition.json`，不在每个动态行重复。

---

## 7.9 `probe_benchmark_metrics`

每个 checkpoint 一行：

```text
profile_name
seed
global_step
optimizer_step
checkpoint_step
observation_phase="checkpoint_probe"
probe_extra_time_seconds
probe_peak_memory_bytes
probe_effective_prompt_count
probe_effective_trajectory_count
probe_effective_latent_position_count
credit_probe_enabled
credit_alignment_probe_enabled
probe_rng_restore_succeeded
record_available
record_unavailable_reason
probe_benchmark_available
probe_benchmark_unavailable_reason
```

至少比较：

```text
probe 完全关闭
checkpoint 触发但 probe 关闭
probe forward-only
probe forward + Delta/FlipGrad
probe forward + credit/autograd.grad
不同 batch/trajectory/latent-position 上限
```

---

# 8. 只在内存中临时保留、不得长期持久化的中间 Tensor

以下变量只用于就地聚合、短生命周期对齐或 checkpoint probe，不得长期写盘为完整 Tensor：

```text
raw full-vocabulary Gumbel Tensor
full-vocabulary logits
full hidden states
full embeddings
完整训练梯度
长期连接计算图的 component log-probability 或 surrogate margin
rollout_noisy_topk_token_ids（Stage 3 仅低频临时缓存）
pre_update_topk_token_ids（Stage 3 仅低频临时缓存）
rollout_perturbed_topk_scores
component_log_prob
surrogate_margin 的训练全量 Tensor
flipgrad_trigger_mask 的训练全量 Tensor
u_i 的完整跨训练数据
q_i 的完整跨训练数据
valid_latent_component_mask 的训练全量 Tensor
worker 间传输的完整 top-K 集合
```

允许持久化的例外：

- checkpoint eval 中已有的 clean top-K IDs/probs，以逐有效 latent position raw fact 保存；
- 固定小型 checkpoint probe 的有限 detached `surrogate_margin` 样本可仅在内存中交给 driver 计算 p05，但不得扩展到训练全量或持久化完整样本。

---

# 9. 统一 mask 与 count 规则

## 9.1 Mask 必须明确的统计族

| 统计族 | 有效元素 |
|---|---|
| 基础 PPO loss/entropy/KL/ratio | 现有训练实现的真实 token/sample mask |
| response length | 当前 runtime 的 response counting mask |
| latent length | latent sentinel + response/attention/loss 语义 |
| noisy mixture | 实际参与 latent state 构造的 noisy mixture position |
| zero advantage | eligible latent policy-loss token |
| reward | 真实 reward 统计单位和有效 mask |
| advantage | 最终训练 advantage 的有效 mask |
| Support | 选中 trajectory 的严格对齐有效 latent position |
| Delta | valid latent component |
| FlipGrad | 满足该 trigger 定义的有效 component 集合 |
| credit concentration | 可获得有效 `u_i/q_i` 的 component 集合 |
| Spearman | 非退化、长度一致的 weights-credit 向量 |
| alignment | 非零且符号定义已验证的有效 gradient 元素 |

## 9.2 Count 规则

- 同一统计族、完全相同的有效 mask 才能共享 count；
- 不同 mask 不得为了减少列数强行共享分母；
- count 为 0 时指标写 `NaN/null`，同时写 unavailable/degenerate reason；
- rate 使用 numerator/count；
- mean/std 使用全局 sum/sum_sq/count；
- 不简单平均各 worker、各 rank 或各 group 的平均值；
- Support 按有效 latent position 总数加权；
- group 比例从原始 count 离线构造，不平均 group-level rate。

---

# 10. 必须写一次的静态配置与定义字段

这些字段不属于训练中间指标，但用于解释和复现实验，必须写入 `run_config.json` 或对应 definition JSON。

## 10.1 `run_config.json`

```text
profile_name
seed
train_batch_size
rollout_batch_size
ppo_mini_batch_size
ppo_micro_batch_size_per_gpu
rollout_micro_batch_size_per_gpu
gradient_accumulation_steps
rollout_n
ppo_epochs
base_learning_rate
optimizer_name
lr_scheduler_name
warmup_steps
weight_decay
max_grad_norm
ppo_clip_ratio
entropy_coeff
kl_coeff
gamma
gae_lambda
top_k
top_p
sampling_temperature
gumbel_temperature
gumbel_noise_scale
gumbel_clip_lower
gumbel_clip_upper
one_sided_gumbel_noise_offset
use_one_sided_noise
use_flipgrad
use_first_token_selection
max_prompt_length
max_response_length
max_latent_length
overlong_sample_policy
model_name_or_path
dataset_name
eval_dataset_name
eval_generations_per_question
world_size
num_gpus
tensor_parallel_size
sequence_parallel_size
dtype
gradient_checkpointing
remove_padding
configured_total_steps
test_steps
metrics_enabled
metrics_schema_version
metrics_log_interval_steps
flush_interval_records
output_format
fail_training_on_logging_error
save_jsonl_debug_copy
platform_config_path
platform_config_hash
execution_surface
workspace_fingerprint
python_executable_fingerprint
python_environment_kind
runtime_environment_fingerprint
runtime_environment_fingerprint_available
runtime_environment_fingerprint_unavailable_reason
resolved_launch_command_redacted
platform_detector_version
platform_detection_mode
platform_detection_status
dependency_check_status
runtime_tensor_probe_status
runtime_provenance_verified
data_classification
safe_to_share
execution_scope
network_access_performed
workspace_external_scan_performed
privileged_operations_performed
automatic_installation_performed
secret_redaction_applied
platform_security_check_status
response_length_definition_version
latent_length_definition_version
generated_token_count_definition_version
generated_token_count_scope
length_counting_rule_version
```

注意：

- `base_learning_rate` 是静态初始学习率；
- 动态 scheduler 后学习率写入各动态表的 `learning_rate`；
- 禁止使用含义模糊的 `gumbel_delta`；
- 旧字段若必须兼容，只能显式 deprecated alias 到 `one_sided_gumbel_noise_offset`；
- `execution_surface` 在正式目标环境中记录为 `vscode_remote_workspace`；
- 所有环境与启动信息必须脱敏。

## 10.2 Stage 3 静态定义

见 `support_definition.json`，不得在每个 Support 动态行重复。

## 10.3 Stage 4 静态定义

见 `probe_definition.json`，不得在每个 probe 动态行重复。

---

# 11. 数据集与写盘频率总表

| 数据集 | 行粒度 | 频率 |
|---|---|---|
| `run_config.json` | 每次 run 一份 | 启动时一次 |
| `train_step_metrics` | 每个完成的 global step 一行 | 每 step 或配置日志间隔 |
| `eval_dataset_manifest.parquet` | 每个 dataset version × question 一行 | 数据集版本首次使用时 |
| `eval_question_results` | checkpoint × question × generation | 每次 checkpoint eval |
| `eval_clean_topk` | checkpoint × question × generation × latent position | 每次 checkpoint eval，字段可用时 |
| `train_group_metrics` | global step × group | 每个训练 step |
| `gumbel_diagnostics` | 独立 diagnostic run × diagnostic batch | 仅 diagnostic/smoke |
| `support_metrics` | Support step × group × selected trajectory class | 默认每 10 step，自 step 10 开始 |
| `support_benchmark_metrics` | 每个实际 Support step 一行 | Support step |
| `probe_definition.json` | 每个 run 一份固定定义 | 首次 probe 前 |
| `probe_metrics` | checkpoint × trajectory group × latent-position group | checkpoint-only |
| `probe_benchmark_metrics` | 每个 checkpoint probe 一行 | checkpoint-only |

---

# 12. Codex 实现时的最终变量检查表

## 12.1 Stage 1

- [ ] 10 个基础训练核心指标全部进入 `train_step_metrics`。
- [ ] `train/gradient_norm` 不进入 schema。
- [ ] `train/generated_token_count` 使用最终训练 rollout trajectory 总长度和。
- [ ] `generated_token_count_definition_version` 与 scope 固定。
- [ ] 不写 `eval_metrics`。
- [ ] 每题每 generation 完整保存，包括失败记录。
- [ ] clean top-K 只转存已有 indices/probs。
- [ ] 普通训练表不保存 `checkpoint_step`。

## 12.2 Stage 2

- [ ] 正式训练不执行 raw Gumbel diagnostic。
- [ ] noisy mixture 使用实际参与 latent state 构造的 `noisy_mixture_weights`。
- [ ] zero-advantage 分母只包含 eligible latent token。
- [ ] reward/advantage 使用真实 mask 与独立 count。
- [ ] correct/non_correct 二分类覆盖全部 trajectory。
- [ ] overlong 是独立可重叠属性。
- [ ] stable `trajectory_id` 在 repeat 后、reorder 前创建。
- [ ] winner 映射到 stable trajectory ID。
- [ ] `surrogate_margin`、noise offset 与 `onesided/delta_*` 不混淆。
- [ ] Stage 2 不持久化 `onesided/*`。

## 12.3 Stage 3

- [ ] rollout 与 pre-update top-K 来自规定时间点。
- [ ] `pre_update_topk_token_ids` 在规定的 pre-update 时间点立即捕获。
- [ ] 不补做 forward。
- [ ] 只比较相同 trajectory、相同 latent position。
- [ ] top1 retention 使用“rollout top1 是否仍在 pre-update top-K 集合中”。
- [ ] trajectory-level 与 benchmark position count 不混用。
- [ ] Support 不改变 Stage 1 长度/token count。

## 12.4 Stage 4

- [ ] probe 使用固定 prompt、顺序、seed 和采样参数。
- [ ] reward、advantage、mask、group、Optimal Correct Path 与训练一致。
- [ ] `surrogate_margin` 与 FlipGrad 公式按真实代码验证。
- [ ] onesided 7 个指标和 count 完整。
- [ ] credit 默认关闭。
- [ ] credit 开启前完成 autograd 符号、显存、时间、RNG 与参数不变性测试。
- [ ] 一次梯度结果通过 mask 完成所有分组，不重复 backward。
- [ ] probe 数据不计入训练长度与 token count。
- [ ] benchmark 每 checkpoint 只写一行。
- [ ] probe 结束后 RNG、参数、optimizer state 和正常 `.grad` 均未改变。
