# Agent A：作者仓库训练链路只读审计

审计日期：2026-08-02  
审计范围：`./Latent-GRPO`（只读）及任务契约（只读）  
唯一写入：本报告  

## 1. Executive summary

### 1.1 结论分级

- **Observed（O）**：由当前 checkout 的源码直接确认。
- **Inferred（I）**：由多个已确认调用点组成的保守推论，仍应在 smoke 中核对。
- **Unconfirmed（U）**：静态代码不足以确认，必须在目标 Linux/CUDA 机器 probe。

### 1.2 核心结论

1. **[O] 原生训练入口不是 `torchrun`，而是 `python3 -m verl.trainer.main_ppo` 启动一个 Ray driver。** 两个作者脚本均这样启动；Hydra 解析 `ppo_trainer.yaml` 与 CLI override，driver 再创建远程 `TaskRunner`、Ray resource pool、FSDP actor/rollout worker，并由 `RayPPOTrainer.fit()` 驱动训练。证据：`Latent-GRPO-gsm8k-llama3.sh:8-72`、`Latent-GRPO-math500-qwen.sh:8-72`、`verl-0.4.x/verl/trainer/main_ppo.py:18-37,44-58,95-154`。
2. **[O] Latent rollout 只能安全沿用仓库内定制 SGLang 路径。** 作者脚本显式选择 `rollout.name=sglang`、启用 latent、Top-K、Gumbel 和 one-sided noise；generic vLLM/HF 分支虽存在，但未显示提供同样的 latent tensors/mixture 语义。证据：两个脚本各自 `:35-47`；`verl-0.4.x/verl/workers/fsdp_workers.py:389-482`。
3. **[O] rollout 实际保存并送回训练侧的关键量是 noisy Top-K IDs、扰动后 Top-K scores、clean Top-K 概率/IDs。** `rollout_topk_gumbels` 这个名字具有误导性：SGLang 明确把它设为 `log p + transformed noise` 的 Top-K score，而不是纯 Gumbel noise。证据：`sglang_latent_reasoning_pkg/python/sglang/srt/layers/sampler.py:74-128`、`verl-0.4.x/verl/workers/rollout/sglang_rollout/sglang_rollout.py:191-224,699-776`。
4. **[O] noisy latent state 是 Top-K embedding 的加权和。** 权重为 perturb-score 经 Gumbel temperature 的 softmax；SGLang 下一 decode step 通过 `weighted_forward{,_tp}` 构造 embedding。证据：`sampler.py:105-128`、`models/llama.py:303-327`、`models/qwen2.py:285-309`、`layers/vocab_parallel_embedding.py:493-524,526-559`。
5. **[O] old policy log-prob 在 rollout、reward/filter 完成后、advantage 之前，以 actor 的无梯度 forward 重算；current log-prob 在每个 PPO micro-batch 的 actor update forward 内计算。** 这两个时间点可支撑 PPO ratio 和 pre-update Support，但当前接口没有把 current component-level log-prob 或充分统计量返回 driver。证据：`ray_trainer.py:1214-1250,1268-1322`、`fsdp_workers.py:669-713`、`dp_actor.py:475-610`。
6. **[O] Optimal Correct Path 是“positive first-step advantage 候选中，选 response-mask 上 mean old log-prob 最大者”，然后只把同 group 其他轨迹的首个 response token advantage 置零。** 它不是 reward-only winner，也不是把其他轨迹整条 advantage 清零。证据：`core_algos.py:188-245,304-359`。
7. **[O] FlipGrad 已嵌在 latent Gumbel likelihood 中。** `raw_diff = rollout perturbed score - current component log-prob`；trigger 为 `(advantage <= 0) & (raw_diff < 0)`；forward 数值保持标准 Gumbel log-density，backward 通过翻转后的 proxy（straight-through）路由。证据：`torch_functional.py:133-195`。
8. **[O] 当前实现不能满足 target contract 的稳定 ID、optimizer step、count/availability、checkpoint metadata、原子日志等要求。** 当前 group `uid` 是每轮 `uuid4()`，没有稳定 trajectory ID；只有 `global_steps`；optimizer 可在一个 outer step 内执行多次且 nonfinite 时跳过，但没有累计成功更新计数；现有 metric reduction 是均值列表，不能替代 `sum/sum_sq/count`。证据：`ray_trainer.py:1064-1069,1313-1322,1351-1369`、`dp_actor.py:379-393,496-613`、`trainer/ppo/metric_utils.py:29-47`。
9. **[O] checkpoint 可恢复 FSDP model/optimizer/scheduler、worker RNG 和 dataloader state，但缺少契约要求的 optimizer_step、config hash、schema、upstream commit、driver/SGLang generator state。** latest tracker 也是普通覆盖写，不是原子提交。证据：`ray_trainer.py:869-951`、`utils/checkpoint/fsdp_checkpoint_manager.py:76-128,129-252`、`utils/checkpoint/checkpoint_manager.py:108-132,135-171`。
10. **[O/U] 上游存在必须阻止静默退化的配置/runtime 敏感点。** latent 专用 Gumbel log-prob 只在 FlashAttention cross-entropy import 成功时使用，否则退回普通 token log-prob；Top-K 输出路径实际上要求 remove-padding；fused-kernel 分支与 Top-K 后处理存在静态可疑点；Qwen 脚本的 latent-end ID 为 522，而 sampler 中一处分支硬编码 524。必须通过 runtime probe 与最小数值测试确认。证据：`torch_functional.py:32-37,133-195`、`dp_actor.py:119-378`、`sampler.py:132-149`、`Latent-GRPO-math500-qwen.sh:43-46`。

### 1.3 对下一阶段的总判断

- **不应重写 Latent-GRPO 算法。** rollout、latent mixture、old/current likelihood、FlipGrad、GRPO/OCP、Ray/FSDP/SGLang 权重同步应复用上游。
- **需要新建外部 adapter/observer/storage 层；仅靠调用公开入口拿不到全部契约变量。** 至少需要受版本约束的私有 adapter；若无法通过 subclass/回调注入，则需要对 vendor 上游做小而显式的 instrumentation patch。
- **不能直接用 `torchrun --nproc_per_node=3 python -m verl.trainer.main_ppo`。** 那会启动三个相互独立的 Ray driver。三卡目标环境应优先采用“一份 adapter driver + Ray 分配 3 GPU”的上游原生拓扑；若用户入口必须为 torchrun，则 wrapper 必须有明确的 single-driver ownership 协议并经实测，不能简单包一层命令。

## 2. 仓库身份与工作树

### 2.1 Git 身份

- **[O] HEAD**：`c0994fb781a2d180662bb522d8ff3e8638dcf56d`
- **[O] branch**：`main`
- **[O] commit time**：`2026-05-12 19:38:56 +0800`
- **[O] subject**：`fixed`
- **[O] origin**：`https://github.com/DJC-GO-SOLO/Latent-GRPO.git`

### 2.2 Dirty 状态

`git status --porcelain=v1` 显示仓库 **dirty（仅 untracked）**：

```text
?? .DS_Store
?? .cairn/
?? AGENTS.md
?? CLAUDE.md
?? cairn/
```

`git diff --stat` 为空，说明当前 tracked 文件没有工作树 diff。本审计未修改 `./Latent-GRPO/**`。

## 3. 真实入口、Hydra/config 与启动链

### 3.1 作者发布脚本

两个脚本的共同链路：

```text
shell env / Hydra overrides
  -> python3 -m verl.trainer.main_ppo
  -> @hydra.main(config_path="config", config_name="ppo_trainer")
  -> run_ppo(config)
  -> ray.init(...)
  -> TaskRunner.remote().run(config)
  -> tokenizer / processor / RLHFDataset / sampler
  -> RayPPOTrainer(...).init_workers()
  -> RayPPOTrainer.fit()
```

证据：

- **[O] CLI**：`Latent-GRPO-gsm8k-llama3.sh:8-72`；`Latent-GRPO-math500-qwen.sh:8-72`。
- **[O] Hydra**：`verl-0.4.x/verl/trainer/main_ppo.py:18-26`。
- **[O] Ray local cluster / remote driver**：同文件 `:29-45`。
- **[O] resolved config、模型本地化、tokenizer**：同文件 `:46-65`。
- **[O] FSDP/worker class 与 resource pool**：同文件 `:75-108`。
- **[O] reward manager、dataset、trainer**：同文件 `:110-154`。
- **[O] 默认配置**：`verl-0.4.x/verl/trainer/config/ppo_trainer.yaml:1-291`；CLI 通过 Hydra 覆盖其中字段。

### 3.2 profile 事实

- **[O] GSM8K/1B 作者配置**：train batch 64、rollout `n=8`、response 128、TP=1、FSDP actor 无 offload、BF16、remove-padding、8 GPU。证据：`Latent-GRPO-gsm8k-llama3.sh:12-57,65-72`。
- **[O] Math/7B 作者配置**：train batch 32、rollout `n=8`、response 4096、TP=1、gradient checkpoint、actor param/optimizer offload、8 GPU。证据：`Latent-GRPO-math500-qwen.sh:12-57,65-72`。
- **[O] 两者都是 8-GPU 发布配置；三卡参数必须重新做整除与显存适配，不能称论文/作者严格复现。** trainer 校验 `train_batch_size*n` 对 GPU 数整除，并校验本地 mini/micro batch。证据：`ray_trainer.py:440-520`；worker 还会按 DP size 归一化 mini-batch：`fsdp_workers.py:143-164`。

### 3.3 torchrun 判断

- **[O] upstream main 自己创建 Ray，resource pool GPU 数来自 `trainer.n_gpus_per_node * nnodes`。** 证据：`main_ppo.py:29-37,95-108`。
- **[I] 直接让 torchrun 创建 3 个相同 main 进程会各自运行 `ray.init()` 和完整 `TaskRunner`，不是上游支持的单作业 3-rank 语义。** 代码中没有检查 torchrun rank 以只让 rank 0 建 driver。
- **设计决策建议**：新用户入口可以由普通 Python 或 torchrun rank-0 ownership wrapper 驱动，但底层仍应让 Ray 创建 3 个 actor workers；在完成 single-driver/rank lifecycle 测试前，`torchrun` 路径标为 `unconfirmed`。

## 4. rollout → reward → advantage → actor update 数据流

### 4.1 完整 outer-step 链

| 顺序 | 真实操作 | 关键 batch/interface | 证据 |
|---:|---|---|---|
| 1 | 从 `StatefulDataLoader` 取 prompt batch，弹出 generation keys | `input_ids/attention_mask/position_ids/raw_prompt_ids` | `ray_trainer.py:1010-1034` |
| 2 | actor-rollout worker 调 SGLang 生成 | `generate_sequences(gen_batch)` | `ray_trainer.py:1038-1047` |
| 3 | 为 prompt 创建随机 UUID group uid，再按 `n` interleave repeat | `non_tensor_batch["uid"]` | `ray_trainer.py:1064-1069` |
| 4 | union rollout 输出并取 response attention mask | response + latent Top-K tensors | `ray_trainer.py:1069-1078` |
| 5 | rule/model reward；可做 dynamic group filtering 和 retry | `token_level_scores` | `ray_trainer.py:1081-1212` |
| 6 | 在最终 batch 上重算 old log-prob/entropy/pre-update Top-K | `old_log_probs`, `old_topk_*` | `ray_trainer.py:1214-1250`; `fsdp_workers.py:669-713` |
| 7 | 可选 ref log-prob | `ref_log_prob` | `ray_trainer.py:1253-1260` |
| 8 | reward→token-level reward，driver 计算 GRPO advantage/OCP | `advantages`, `returns` | `ray_trainer.py:1268-1302` |
| 9 | actor worker 逐 PPO mini/micro-batch current forward、loss/backward/step | current `log_prob`, policy loss | `ray_trainer.py:1312-1322`; `dp_actor.py:475-613` |
| 10 | validation/checkpoint、聚合现有 metrics、log、global step++ | driver state | `ray_trainer.py:1339-1373` |

### 4.2 dynamic group filtering 与最终 rollout 集合

- **[O]** 过滤依据默认是每个 UUID group 的 sequence final reward 标准差；std=0 且 group size>1 的 group 被丢弃，继续从后续 dataloader batch 生成，直到累计够 `train_batch_size` 个完整 group。证据：`ray_trainer.py:1109-1174`。
- **[O]** 完整 group 的 `n` 条 trajectory 会被断言保留；之后重新 balance/reorder。证据：`ray_trainer.py:1175-1212`。
- **[O/I]** 契约的 `generated_token_count` 应从 `batch = acc_batch[kept_indices]` 之后的最终 batch 计算，不能把 `continue` 前被丢弃的内部生成计入。现有 `perf/total_num_tokens` 统计 prompt+response，不等价于目标字段。证据：`metric_utils.py:211-244`。
- **[O]** 现有 `uid=uuid4()` 不可跨 resume 复现，且不存在 trajectory ID。满足契约需要在 repeat 完成后、filter/reorder 前写入稳定 `group_id/trajectory_id`；不能沿用 local index 或 UUID。证据：`ray_trainer.py:1064-1078`。

## 5. Latent rollout、noisy Top-K 与长度/mask

### 5.1 SGLang noisy Top-K 语义

**[O] 公式（当前源码）**：

```text
full_log_probs = log_softmax(logits)
candidate mask = top-p mask，且至少保留 max_topk 个 token
raw_gumbel = -log(Exponential(1))
clipped_gumbel = clamp(raw_gumbel, -1.5, 3)
if one_sided:
    transformed_noise = clipped_gumbel - (-1.5)   # 即 +1.5，非负
scaled_noise = noise_scale * transformed_noise
perturbed_score = masked_log_prob + scaled_noise
selected = topk(perturbed_score, max_topk)
noisy_mixture_weights = softmax(selected_perturbed_score / gumbel_temperature)
```

证据：`sglang_latent_reasoning_pkg/python/sglang/srt/layers/sampler.py:74-128`。

重要命名校正：

- `logits_output.topk_gumbels` / training-side `rollout_topk_gumbels` = **selected perturbed scores**，不是 raw Gumbel，也不是 transformed pure noise（`sampler.py:105-126`）。
- 纯 selected noise `gumbel_noise_topk` 只在 sampler 局部用于 rollout Gumbel log-density，没有被写入 trainer batch（`sampler.py:116-124`）。
- SGLang 的 `output_topk_prob_list` 含实际 mixture weights，但 verl `_post_process_outputs()` 当前没有读取该 key；它读取 IDs、perturbed scores、clean probs/IDs（`sglang_rollout.py:191-224`）。

**复用决定**：

- `rollout_noisy_topk_token_ids`：可直接映射 `rollout_topk_ids`。
- `rollout_perturbed_topk_scores`：可直接映射 `rollout_topk_gumbels`，但 adapter 必须改正语义命名。
- `noisy_mixture_weights`：最佳方案是最小 adapter 把已有 `output_topk_prob_list` 带入 batch；备选是以同一 scores/temperature 重算并做逐元素等价测试。为避免 dtype/rounding 造成“非实际权重”，建议前者。
- raw Gumbel diagnostic：当前 trainer 收不到 raw tensor；如需要，只能在 sampler 就地做局部 reduce，不能把 full-vocab noise 传出。

### 5.2 latent embedding

- **[O]** SGLang request 将上一位置的 Top-K weights/IDs 放进下一 decode forward；LLaMA/Qwen 均调用 vocab-parallel embedding 的 weighted forward。证据：`schedule_batch.py:1669-1725`、`models/llama.py:303-327`、`models/qwen2.py:285-309`。
- **[O]** soft token embedding 是 `sum_i alpha_i * embedding(token_i)`；hard token 用首个 embedding。TP>1 时每 shard 局部算后 all-reduce。证据：`vocab_parallel_embedding.py:493-524,526-559`。
- **[O]** actor 训练重放同一 Top-K IDs/perturbed scores，并同样 softmax 构造 embedding；embedding 在送入 actor 时 `.detach()`，所以训练梯度不是穿过 latent embedding mixture 回去，而是由专用 likelihood/FlipGrad 路径承担。证据：`dp_actor.py:137-181,220-229,268-278`。

### 5.3 hard/latent position 与 mask

- **[O]** trainer replay 用 `topk_ids[...,1:] == -100` 判定 hard token；rollout 在退出 latent mode 后将后 K-1 IDs 设为 -100，首项为 hard token ID。证据：`dp_actor.py:137-176`、`schedule_batch.py:730-771`。
- **[O]** response mask 是 response 段 attention mask；`get_response_mask()` 包含第一个 EOS（先 cumsum 再减当前 EOS），之后为 0。证据：`sglang_rollout.py:743-754`、`torch_functional.py:333-353`。
- **[O]** 现有 response length 是 response attention mask 的和，因此按当前实现包含 EOS/第一个 stop token；metrics 也用同一规则。证据：`metric_utils.py:50-77,109-118,158-167`。
- **[I/U]** target `valid_latent_position_mask` 可候选为 response attention/loss mask 与 `~hard_token_mask` 联合，但契约禁止未经验证依赖固定 sentinel/layout。因此必须 probe：shape `(B,I+R,K)`、shift 后位置对齐、EOS/latent-end 边界、padding、multi-turn `loss_mask`、remove-padding revert order。
- **[O]** multi-turn actor loss 使用 `loss_mask`，但 outer `compute_response_mask()` 和普通 group logic 默认仍从 attention mask取 response 段。证据：`ray_trainer.py:198-213,269-273`、`dp_actor.py:529-535`。当前两个发布脚本未启用 multi-turn；新实现不应未经专项审计宣称支持其 latent mask。

### 5.4 overlong 语义

- **[O]** `exclude_overlong_samples_from_advantage=True` 时，以 `response_mask.sum == max_response_length` 判 truncated，从 group mean/std 排除，并给该 trajectory 全零 advantage。证据：`core_algos.py:145-186`。
- **[O]** `exclude_overlong_samples_from_advantage=False` 时，overlong 先参与 group stats 与 OCP；进入 actor update 后又按长度等于 response tensor width 把整条 advantage 原地置零。证据：`core_algos.py:278-359`、`dp_actor.py:556-562`。
- **[U]** “长度恰好到上限且末 token 是 EOS”是否被误标 truncated、SGLang finish reason 是否可直接提供更准确 overlong flag，需要 runtime probe。当前判定只看长度相等。
- **设计要求**：overlong 是可与 correct/non_correct 重叠的属性；必须在 actor-side过滤前保留 raw facts 与 generated-token count，且清楚区分“参与 group stats”和“真正参与 actor loss”的时间点。

## 6. Reward、advantage 与 Optimal Correct Path

### 6.1 reward manager/scorer

- **[O]** main 根据 config 选择 naive/prime/batch/dapo manager；无 custom reward 时调用 `default_compute_score`。证据：`trainer/ppo/reward.py:60-100`。
- **[O]** 发布脚本没有开启 model reward；默认 naive manager 解码有效 response，读取 `reward_model.ground_truth` 与 `data_source`，将标量 reward 写到最后一个有效 response token。证据：`workers/reward_manager/naive.py:32-108`。
- **[O]** 数据预处理输出 prompt、rule ground truth、data source、extra_info。证据：`data_preprocess_code/gsm8k_aug.py:22-52`、`math500_aug.py:22-50`。
- **[O]** `default_compute_score` 按 data source 分派不同 scorer。证据：`utils/reward_score/__init__.py:19-97`。
- **[U]** `correct` 的阈值不能只假设 `reward > 0`；需对目标 dataset/scorer 的真实返回域和 extra fields probe，并把 classification version 固定。

### 6.2 GRPO 与 OCP

- **[O] include 模式**：sequence score = token rewards sum；按 uid group 算 mean/std（PyTorch 默认 sample std），标准化后将同一标量扩到所有有效 response token。证据：`core_algos.py:278-302`。
- **[O] exclude 模式**：截断样本不进入 group stats且全零；其余同样标准化/扩展。证据：`core_algos.py:145-186`。
- **[O] OCP 候选**：在扩展后的 advantage 上取每条 trajectory 第一个有效 response token；只保留 `adv_val > 0` 候选。证据：`core_algos.py:188-223,304-338`。
- **[O] OCP score**：`masked_sum(old_log_probs)/response_length`；取最大者，`np.argmax` 因候选按 batch order 收集而隐式 first-tie。证据：`core_algos.py:201-235,317-350`。
- **[O] OCP effect**：只把同 group 非 winner 的第一个 response token advantage 置零，后续 token advantage 不变。证据：`core_algos.py:236-245,352-359`。
- **[O]** winner ID 和 mean old log-prob 当前未返回；函数只返回 `scores, scores, id2std`。所以 target `optimal_correct_trajectory_id` / score 需要扩展函数返回或旁路 observer，且必须基于同一内存 winner，不能事后反推。证据：`core_algos.py:245,359`。
- **[O]** 若 old log-prob 缺失会随机选 winner（NumPy RNG），但真实 trainer 总是先重算 old log-prob后传入。证据：`core_algos.py:203-234,319-350`、`ray_trainer.py:1217-1226,1276-1300`。

## 7. old/current log-prob、surrogate margin 与 FlipGrad

### 7.1 old/current 时间点

- **[O] rollout log-prob**：SGLang generation 同时返回，但 trainer 注释明确会由 actor 重算；它只用于 rollout-vs-actor prob diff diagnostic。证据：`sglang_rollout.py:699-705,757-766`、`ray_trainer.py:1227-1250`。
- **[O] old policy**：最终 filtered/rebalanced batch 上，actor eval + `torch.no_grad()` forward，发生于 advantage 和任何 actor update 之前。证据：`dp_actor.py:414-473`、`ray_trainer.py:1217-1226`。
- **[O] current policy**：update actor 时每 micro-batch重新 forward；第一次 micro-batch在该 outer step 尚未 optimizer update，后续 mini-batch可能已看到更新后的参数。证据：`dp_actor.py:496-610`。
- **含义**：Stage 3 的 pre-update Top-K 应直接取 old-log-prob forward 返回的 `old_topk_indices`；不能取 update-policy micro-batch 的变化中 Top-K。

### 7.2 Gumbel likelihood 与 FlipGrad

**[O] 专用 likelihood（仅 FlashAttention branch）**：

```text
component_log_prob = log_softmax(current_logits)[rollout_topk_ids]
surrogate_margin/raw_diff = rollout_perturbed_topk_scores - component_log_prob
standard_forward_component = -raw_diff - exp(-raw_diff)
flip_trigger = (advantage <= 0) & (raw_diff < 0)
flipped_proxy_component = raw_diff - exp(raw_diff)
output = standard.detach() + (selected_proxy - selected_proxy.detach())
latent-position likelihood = mean over K components
```

证据：`verl-0.4.x/verl/utils/torch_functional.py:133-195`。

- **[O]** hard response tokens走标准 token log-prob；soft latent positions走上述 Gumbel likelihood。证据：同文件 `:182-191`。
- **[O]** PPO ratio、KL、dual clip 使用 `exp(current_log_prob-old_log_prob)` 和 response mask；policy loss按 config aggregate。证据：`core_algos.py:707-769`。
- **[O]** FlipGrad forward 值保持不变，只改变 trigger component 的 gradient proxy；因此 logging-only observer 必须 detach，不可改动这段 graph。
- **[U]** target credit 的 `u_i`、surrogate alignment sign、zero-gradient handling 当前没有接口；必须 checkpoint-only 受控 `autograd.grad` 验证，不能由这段源码直接宣称可用。

### 7.3 必须阻止的静默退化

- **[O]** `flash_attn.ops.triton.cross_entropy` import 失败时 `FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE=False`；latent专用函数直接退回普通 label log-prob，忽略 Top-K/Gumbel/FlipGrad。证据：`torch_functional.py:32-37,133-195`。
- **结论**：FlashAttention 不只是性能依赖，而是当前实现 latent policy likelihood 语义的事实依赖；runtime probe 必须断言该 branch 生效并做小张量数值/梯度测试，不能只检查包版本。

## 8. 现有 metric 能否复用

### 8.1 可映射但需重新聚合

- policy loss / clip fraction / PPO KL：actor update 已产生 `actor/pg_loss`, `actor/pg_clipfrac`, `actor/ppo_kl`（`dp_actor.py:563-608`）。
- entropy：old-log-prob forward计算 full-vocabulary entropy，driver以 response mask聚合 `actor/entropy_loss`（`ray_trainer.py:1217-1225`；`dp_actor.py:283-305`）。需要记录其“pre-update actor full-vocab token entropy”时间/概率空间；不能叫 noisy-mixture entropy。
- reward / advantage / response length：现有 helper 有 mean/min/max，但没有 target 的 count/sum_sq，且 advantage 使用 response attention mask（`metric_utils.py:80-169`）。
- step time：outer `with _timer("step")` 覆盖 generation 到 checkpoint/validation，现有 `perf/time_per_step` 可作为事实来源，但应明确是否包含 eval/save并按契约拆出 logger compute/write（`ray_trainer.py:1038-1349`; `metric_utils.py:172-244`）。

### 8.2 不能直接复用现有 reduction

- **[O]** worker output metric 列表最终用 mean reduce；没有不同 mask 的 count，也没有 `sum_sq`。证据：`metric_utils.py:29-47`、`ray_trainer.py:1320-1322`。
- **结论**：target 权威表必须新建 sufficient-stat packet，由 worker 返回 `sum/sum_sq/count/...`，driver合并；不得把现有 reduce 后的均值写成严格全局指标。

### 8.3 当前缺失

- stable `group_id/trajectory_id`；
- `optimizer_step`；
- latent length/effective latent mask的契约化定义；
- noisy mixture `effective_k/top1`充分统计；
- zero-advantage eligible latent count；
- OCP winner ID/score；
- Support selected trajectories与位置对齐结果；
- checkpoint-only one-sided/credit统计；
- record/family availability、definition versions、worker count；
- append-only Parquet与resume去重状态。

## 9. Step、optimizer 与 checkpoint/resume

### 9.1 global step

- **[O]** `global_steps` 初始0；load checkpoint后恢复；训练从 `global_steps+1` 开始；只有完整 outer iteration结束才 `global_steps += 1`。证据：`ray_trainer.py:984-1004,1365-1373`。
- **[O]** dynamic-filter retry 的 `continue` 不增加 `global_steps`，但错误地推进 progress bar；不影响 step ID但使显示进度不可靠。证据：`ray_trainer.py:1143-1153`。
- **[O]** checkpoint在 actor update之后、metrics log之前，以当前 `global_steps` 命名。证据：`ray_trainer.py:1313-1349,1351-1369`。

### 9.2 optimizer step

- **[O]** actor 每个 PPO epoch × 每个 local mini-batch执行一次 `_optimizer_step()`；micro-batches在此前累积。证据：`dp_actor.py:496-520,522-610`。
- **[O]** 非有限 grad norm 时 zero grad并跳过 optimizer step，否则 `optimizer.step()`。证据：`dp_actor.py:379-393`。
- **[O]** actor scheduler在整个 worker `update_actor()` 返回前只 step 一次，而 optimizer 可能 step 多次。证据：`fsdp_workers.py:579-607`。
- **[O]** driver没有收到“成功 optimizer updates count”，也没有 checkpoint该计数。
- **设计结论**：需在 actor worker的 `_optimizer_step()` 返回 `(grad_norm, did_update)` 或等价结果并做全局一致性断言，driver累计 immutable `optimizer_step`；不能把 outer global step、scheduler step或配置估算值冒充 optimizer step。

### 9.3 checkpoint 内容与缺口

已保存：

- **[O]** 每 rank FSDP model/optimizer shard、scheduler state、Python/NumPy/Torch CPU/当前 CUDA RNG（`fsdp_checkpoint_manager.py:92-128,162-185`; `checkpoint_manager.py:108-132`）。
- **[O]** rank0 tokenizer/model config；可选 HF full model（`fsdp_checkpoint_manager.py:187-250`）。
- **[O]** driver dataloader state (`data.pt`) 和 path中的 global step（`ray_trainer.py:869-900`）。

未保存/不足：

- **[O]** 没有 optimizer_step、profile/config hash、schema version、upstream commit。
- **[O]** driver 的 Python/NumPy RNG没有独立 checkpoint；OCP fallback会用 driver NumPy RNG，group UUID使用 OS entropy。
- **[O/U]** SGLang server random seed未由训练 config显式传入；server默认 seed可随机生成。需 probe其实际 state和恢复能力。
- **[O]** latest tracker `latest_checkpointed_iteration.txt` 是普通覆盖写；不是 temp+fsync+rename原子协议（`ray_trainer.py:897-900`）。
- **[O]** resume依赖同 world-size命名的 rank shard；world-size变化不能直接假定可加载（`fsdp_checkpoint_manager.py:92-103`）。
- **[O]** dataloader state存在才恢复，否则警告后从头开始；没有与checkpoint/metrics part做一致性检查（`ray_trainer.py:944-951`）。
- **结论**：上游 checkpoint可作为模型训练状态基础，但必须由新层补充 manifest、optimizer step、config/schema/provenance、metrics writer state、原子提交和兼容性检查；严格 trajectory/RNG resume仍需实测。

## 10. 分布式拓扑

### 10.1 upstream native topology

```text
one CLI process
  -> local/existing Ray cluster
    -> one CPU TaskRunner (driver-side trainer/advantage/reward by default)
      -> one global Ray GPU resource pool
        -> colocated actor_rollout workers on N GPUs
          - FSDP/FSDP2 actor mesh: world_size=N
          - SGLang rollout mesh: DP=N/infer_tp, TP=infer_tp
          - actor weights handed to SGLang through FSDP-SGLang sharding manager
        -> optional ref worker group on same resource pool
        -> optional critic/RM groups
```

证据：

- Ray driver/resource pool：`main_ppo.py:29-45,95-108`。
- GRPO不启用 critic：`ray_trainer.py:417-435`。
- colocated worker group构建：`ray_trainer.py:786-858`。
- worker初始化 Gloo/NCCL、FSDP mesh、Ulysses mesh：`fsdp_workers.py:94-164`。
- FSDP/FSDP2 wrap：`fsdp_workers.py:280-345`。
- rollout DP/TP mesh与 SGLang sharding manager：`fsdp_workers.py:389-396,438-476`。

### 10.2 3×46 GB 推论边界

- **[O]** 发布脚本 TP=1；三卡下可形成 3-way rollout DP、3-way FSDP actor。`world_size % infer_tp == 0` 是硬约束（`fsdp_workers.py:389-396`）。
- **[I]** 1B profile最可能从 TP=1、FSDP2、micro-batch小值开始；7B high-smoke可能需要保留 param/optimizer offload与gradient checkpoint，且大幅缩小response/batch。具体容量属于 runtime agent，不在本只读审计中声称。
- **[O]** 高配置 `actor.use_kl_loss=True` 会额外创建 ref policy worker（`main_ppo.py:126-129`），显存/CPU负担显著；低配置 false则不创建。
- **[U]** Ray在3卡上的colocation、SGLang memory-saver、FSDP2+CPU offload、NCCL topology、46 GB实际峰值必须目标机测试。

## 11. 公共接口、私有 adapter 与 patch 决策

### 11.1 可直接复用的相对稳定接口

| 接口 | 决定 | 理由/证据 |
|---|---|---|
| Hydra config + `run_ppo(config)` | 可由 adapter复用 | `main_ppo.py:24-37` |
| `create_rl_dataset/create_rl_sampler` | 可复用 | `main_ppo.py:157-211` |
| `load_reward_manager/compute_reward` | 可复用 | `trainer/ppo/reward.py:60-130` |
| `compute_response_mask` | 可复用但定义version化 | `ray_trainer.py:198-213` |
| Latent GRPO advantage functions | 算法内核复用，需扩展观测返回 | `core_algos.py:113-359` |
| `compute_policy_loss` | 算法内核直接复用 | `core_algos.py:707-769` |
| SGLang latent sampler/weighted embedding | 必须复用 | `sampler.py:70-159`; `vocab_parallel_embedding.py:493-559` |
| FSDP/SGLang sharding manager | 必须复用 | `fsdp_workers.py:438-476` |
| FSDP checkpoint manager | 状态基础复用，外层补metadata/atomicity | `fsdp_checkpoint_manager.py:33-49,76-252` |

### 11.2 需要版本锁定的私有 adapter 接口

| 私有点 | 需要的观测/变更 | 风险 |
|---|---|---|
| `RayPPOTrainer.fit` | stable IDs、final rollout count、group raw facts、immutable step context、writer events | 单体长方法，无 hook |
| `compute_latent_grpo_*` | 返回同一内存 OCP winner、candidate、score和pre-zero facts | 改返回签名会影响 caller |
| `SGLangRollout._post_process_outputs` | 携带实际 `output_topk_prob_list` mixture weights | meta key/shape依赖定制 SGLang |
| `DataParallelPPOActor._forward_micro_batch` | component log-prob、surrogate margin、flip mask的局部统计 | 不能保留graph/全tensor |
| `DataParallelPPOActor._optimizer_step` | `did_update`与成功optimizer step计数 | 分布式rank必须一致 |
| actor worker `compute_log_prob/update_actor` | Support pre-update Top-K与sufficient stats | dispatch/reorder语义复杂 |
| validation path | question/generation raw facts + clean Top-K | 当前只返回aggregate dict |

### 11.3 是否必须修改作者仓库

**结论：很可能必须有最小 instrumentation patch，但应先尝试 subclass/adapter；禁止大规模fork。** 原因：

1. current component log-prob与FlipGrad mask只存在 actor内部，driver拿不到；
2. OCP winner未返回；
3. stable trajectory ID必须在 repeat后、filter/reorder前插入；
4. optimizer成功更新只能在 `_optimizer_step` 当场知道；
5. actual mixture weights在SGL response里已有但verl丢弃；
6. eval raw facts/top-K在 `_validate()` 内被聚合后丢失。

推荐顺序：

1. 新目录 adapter负责config、run context、schemas/storage、runtime probes；
2. 若可维护地 subclass trainer/worker并让 `TaskRunner`选择新类，则不改vendor算法文件；
3. 否则在 `patches/` 保存少量明确diff，仅增加 observer callbacks/返回detached sufficient stats/IDs；
4. 对每个 logging-only patch做关闭日志等价性测试。

## 12. 静态发现的高风险点

### 12.1 Blocker candidates

1. **FlashAttention import失败会静默退化算法**（上文 §7.3）。必须 fail-fast而不是继续训练。
2. **remove-padding事实依赖**：Top-K replay变量只在 `use_remove_padding` 分支完整构造；non-remove-padding分支返回 `full_topk_*` 名称但没有赋值。发布脚本均启用 remove-padding。证据：`dp_actor.py:119-378`、两个训练脚本 `:19`。
3. **fused-kernel不兼容风险**：fused branch没有定义后续使用的 `logits_rmpad`，但Top-K后处理无条件引用它。发布配置默认 fused kernels false。证据：`dp_actor.py:231-236,317-343`。
4. **`topk_logits`拼接笔误**：收集 `cur_topk_logits` 后实际 `torch.concat(topk_ids_lst)`，返回的 `topk_logits`其实是 IDs。证据：`dp_actor.py:441-464`。当前主训练未消费该字段，但任何 probe不能信任它。

### 12.2 Major candidates

1. **Qwen latent-end硬编码差异**：Qwen脚本配置522，但 sampler的一个 mask比较524。request的后处理会按配置ID将 end token变hard，可能部分抵消；仍必须做边界probe。证据：`Latent-GRPO-math500-qwen.sh:43-46`、`sampler.py:132-149`、`schedule_batch.py:746-761`。
2. **dynamic-bsz reorder不完整**：`compute_log_prob()`只对 log-probs做 reverse reorder，没有同样reorder entropy/top-K。发布脚本默认 dynamic_bsz false；新配置若开启必须修复/禁用。证据：`dp_actor.py:461-473`。
3. **old/current sampling参数一致性不完整**：rollout把实际 `temperature/top_p`放 meta，但 worker old-log-prob又用config覆盖，并只明确写 temperature。当前脚本两者相同，adapter仍应 assert。证据：`sglang_rollout.py:779-792`、`fsdp_workers.py:685-690`。
4. **OCP include-mode std实现可疑**：`torch.std(torch.tensor([id2score[idx]]))`多包一维，数值元素仍相同但构造方式应在 synthetic test确认；单元素用 mean=0/std=1。证据：`core_algos.py:284-301`。
5. **checkpoint latest tracker非原子、driver/SGL RNG不完整**（上文 §9.3）。

### 12.3 Notes

- validation generation强制 noise off，但仍由定制 SGLang返回 clean Top-K；当前 `_validate()`没有持久化这些tensor。证据：`sglang_rollout.py:662-670,699-705`、`ray_trainer.py:687-784`。
- 作者可选 JSONL generation dump只有 input/output/score/step，没有 question_id、generation_id、reference hash或clean top-K，不满足目标eval schema。证据：`ray_trainer.py:630-652`。
- 现有 `actor/grad_norm` 是训练内部指标；target contract明确禁止把 `train/gradient_norm`写入新schema。证据：`dp_actor.py:610-612`。

## 13. Runtime probe 问题清单

### 13.1 必须在载入训练前确认

1. Python/PyTorch/CUDA/driver/cuDNN/NCCL版本、GPU count/name/memory/compute capability、BF16、disk。
2. `flash_attn.ops.triton.cross_entropy.cross_entropy_loss`是否真实可import，并且 `FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE=True`。
3. 定制 `sglang` import是否指向仓库版本，而非环境中另一份发行版；`verl`同理。
4. FSDP2 fully_shard、Ray GPU visibility、NCCL 3-rank init、SGLang kernel/attention backend。
5. tokenizer的 `latent_end_token_id` 522/524是否确实对应目标token，Top10模型的K是否一致。

### 13.2 首个无更新tensor probe

1. `rollout_topk_ids/gumbels/original_probs/original_indices`的shape/dtype/device/K及padding sentinel。
2. `output_topk_prob_list`与 `softmax(rollout_topk_gumbels / gumbel_temperature)`逐元素误差。
3. `old_topk_indices`与 rollout trajectory在filter/balance/dispatch后对齐；dynamic-bsz禁用/启用各测。
4. hard/latent mask、response/loss/attention mask、prompt offset、roll/shift后 latent position。
5. EOS是否计长；达到max length且最后token是EOS的finish reason/overlong判定。
6. clean Top-K probs究竟是full-distribution概率（预期sum<=1）还是Top-K内归一化；从实际返回确认。

### 13.3 最小算法数值/梯度 probe

1. SGLang perturb score = clean log-prob + transformed scaled noise；one-sided noise范围及clip rate。
2. actor `raw_diff`与target surrogate margin逐项相等。
3. FlipGrad trigger真值表和finite difference/autograd符号；forward with/without flip完全相同。
4. old/current log-prob在第一个mini-batch前相等范围；后续mini-batch时间点变化。
5. OCP winner与独立reference实现相同，tie规则固定映射stable trajectory ID。
6. include/exclude overlong两模式：group stats、winner、actor-side zeroing的确切结果。

### 13.4 checkpoint/resume probe

1. 连续4 step vs 2+resume到4：数据顺序、rollout seed、OCP winner、model update可比性。
2. 每rank Python/NumPy/Torch/CUDA RNG及SGLang RNG恢复。
3. 一个outer step内实际成功optimizer step数，nonfinite skip是否跨rank一致。
4. 3卡save/load shard命名与same-world-size约束。
5. probe前后parameter/optimizer/scheduler/grad/RNG/global/optimizer/token counters hash。

## 14. 建议的观测切点（供 implementation plan 使用）

| Observation phase | 最小切点 | 可取得变量 |
|---|---|---|
| post-repeat/pre-filter | `ray_trainer.py:1064-1069` 后 | stable group/trajectory IDs |
| final-rollout/post-filter | `ray_trainer.py:1173-1214` 后 | final trajectory lengths、overlong raw facts、generated count |
| rollout sampler local | `sampler.py:91-128` | Gumbel diagnostic局部reduce、actual weights |
| pre-update old-log-prob | `ray_trainer.py:1217-1226` | old log-prob、entropy、pre-update clean Top-K、Support |
| post-advantage/pre-update | `ray_trainer.py:1288-1303` 后 | rewards、final advantage、zero-adv、group raw facts、OCP winner |
| current actor micro-batch | `dp_actor.py:551-608` | current likelihood、ratio、loss、component Delta/Flip sufficient stats |
| optimizer commit | `dp_actor.py:379-393,610` | did_update、optimizer step |
| post-update driver | `ray_trainer.py:1320-1366` | global train-step event、timing/learning rate |
| checkpoint eval | `ray_trainer.py:678-784` | per-question/generation raw facts、clean Top-K |
| checkpoint save/load | `ray_trainer.py:869-951` | state manifest/resume integration |

## 15. 复用/patch 最终决策表

| 项目 | 决策 | 状态 |
|---|---|---|
| SGLang latent rollout与embedding | 原样复用，instrument only | observed |
| Gumbel/one-sided/FlipGrad公式 | 原样复用，添加局部统计接口 | observed |
| Latent GRPO/OCP | 复用函数，扩展winner观测返回 | observed |
| Ray/FSDP/SGLang topology | 复用；三卡重新配置 | observed + runtime probe |
| generic vLLM rollout | 不用于Latent主路径 | inferred unsafe |
| generic HF rollout | 不用于Latent主路径 | inferred unsafe |
| existing metrics reducer | 不作为权威target统计 | observed incompatible |
| existing generation JSONL | 仅debug参考，不作为权威eval表 | observed incompatible |
| stable IDs | 新增 | missing |
| optimizer_step | 新增actor→driver commit接口 | missing |
| checkpoint metadata/atomic metrics resume | 新外层实现 | missing |
| Support | 利用既有old forward输出，禁止另加forward | feasible, probe required |
| checkpoint credit probe | 新的隔离probe路径，默认关闭 | unconfirmed |

## 16. 证据索引

| 主题 | 主要证据（均为仓库根 `Latent-GRPO/` 下相对路径，1-based） |
|---|---|
| 发布说明/模型/环境 | `README.md:22-34,102-143,145-197` |
| 两个CLI入口 | `Latent-GRPO-gsm8k-llama3.sh:1-72`; `Latent-GRPO-math500-qwen.sh:1-72` |
| Hydra/Ray main | `verl-0.4.x/verl/trainer/main_ppo.py:18-154` |
| 默认config | `verl-0.4.x/verl/trainer/config/ppo_trainer.yaml:1-291` |
| trainer主循环 | `verl-0.4.x/verl/trainer/ppo/ray_trainer.py:966-1373` |
| reward path | `verl-0.4.x/verl/trainer/ppo/reward.py:60-130`; `verl-0.4.x/verl/workers/reward_manager/naive.py:32-108` |
| advantage/OCP | `verl-0.4.x/verl/trainer/ppo/core_algos.py:113-359` |
| PPO loss | `verl-0.4.x/verl/trainer/ppo/core_algos.py:707-769` |
| actor latent replay/current update | `verl-0.4.x/verl/workers/actor/dp_actor.py:71-378,379-613` |
| FlipGrad likelihood | `verl-0.4.x/verl/utils/torch_functional.py:32-37,133-195` |
| rollout tensor plumbing | `verl-0.4.x/verl/workers/rollout/sglang_rollout/sglang_rollout.py:191-224,625-792` |
| SGLang noise/mixture | `sglang_latent_reasoning_pkg/python/sglang/srt/layers/sampler.py:70-159,240-273` |
| latent weighted embedding | `sglang_latent_reasoning_pkg/python/sglang/srt/models/llama.py:303-327`; `.../models/qwen2.py:285-309`; `.../layers/vocab_parallel_embedding.py:493-559` |
| FSDP/Ray topology | `verl-0.4.x/verl/workers/fsdp_workers.py:94-164,280-345,389-482,484-577` |
| checkpoint/RNG | `verl-0.4.x/verl/trainer/ppo/ray_trainer.py:869-951`; `verl-0.4.x/verl/utils/checkpoint/fsdp_checkpoint_manager.py:76-252`; `.../checkpoint_manager.py:108-171` |
| existing metrics | `verl-0.4.x/verl/trainer/ppo/metric_utils.py:29-47,50-169,172-244` |

## 17. 本审计实际执行的只读操作

仅执行了 `rg`、`find`、`sed`、`nl`、`wc`、`git rev-parse/show/status/diff/ls-files` 等读取命令；未 import CUDA/Ray/SGLang/verl 模块，未安装依赖，未启动训练，未修改作者仓库。

