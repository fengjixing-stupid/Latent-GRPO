# 作者仓库训练链路只读审计

审计日期：2026-08-02。范围：`./Latent-GRPO` commit `c0994fb781a2d180662bb522d8ff3e8638dcf56d`。本阶段未安装依赖、未 import CUDA/Ray/SGLang 训练模块、未启动训练、未修改作者 tracked 文件。

结论标记：**Observed** 为当前源码直接证明；**Inferred** 为多处静态证据的保守组合；**Runtime probe required** 为只能在目标 Linux/3-GPU 环境确认。

## 1. Executive summary

1. **Observed：原生入口是单一 Hydra/Ray driver，不是 torchrun。** 两个作者脚本都运行 `python3 -m verl.trainer.main_ppo`；它创建 remote `TaskRunner`、Ray GPU resource pool、FSDP actor/rollout worker 和 `RayPPOTrainer.fit()`。
2. **Observed：Latent rollout 必须复用作者定制 SGLang。** generic HF/vLLM 分支没有证明具备相同 latent token、noisy Top-K mixture、one-sided noise 与回传 Tensor 语义。
3. **Observed：`rollout_topk_gumbels` 实际是扰动后分数 `log p + transformed noise`，不是 raw Gumbel。** noisy mixture 是选中 K 个扰动分数按 Gumbel temperature 的 softmax，下一 decode step使用 K 个词表 embedding 的加权和。
4. **Observed：old policy log-prob 在最终 rollout/filter 后、advantage 前由 actor no-grad forward重算；current log-prob在 PPO update micro-batch内计算。** 现有路径可提供 pre-update clean Top-K，但缺 component 充分统计返回。
5. **Observed：Optimal Correct Path 在 positive first-step advantage 候选中选 mean old log-prob最高者，只把同组其他轨迹的首个 response token advantage置零。** 不是 reward-only，也不是将败者整条 advantage置零。
6. **Observed：FlipGrad 已嵌入 Gumbel surrogate。** `surrogate_margin = rollout_perturbed_score - current_component_log_prob`，trigger 为 `(advantage <= 0) & (margin < 0)`；straight-through保持前向值，只改变反向 proxy。
7. **Observed：当前代码不满足 stable trajectory ID、成功 optimizer-step clock、sufficient-stat count、availability、原子存储与完整 checkpoint sidecar。** 需要外部 runner/observer/storage 和很小的上游 instrumentation patch。
8. **Runtime probe required：** FlashAttention latent log-prob是否真的启用、remove-padding/top-K shape、fused/dynamic-batch分支、Qwen latent-end 522/硬编码524、EOS/overlong、三卡 Ray/FSDP/SGLang 和 checkpoint RNG 连续性。

完整逐行子报告见 `work_reports/agent_a_repo_audit.md`。

## 2. 仓库身份与状态

- HEAD：`c0994fb781a2d180662bb522d8ff3e8638dcf56d`
- branch：`main`
- commit time/subject：`2026-05-12 19:38:56 +0800`, `fixed`
- origin：`https://github.com/DJC-GO-SOLO/Latent-GRPO.git`
- tracked diff：无。
- worktree：dirty，仅有原有 untracked `.DS_Store`, `.cairn/`, `AGENTS.md`, `CLAUDE.md`, `cairn/`。本审计没有删除或覆盖。

## 3. 训练入口、配置解析与 profile

```text
author shell + environment + Hydra overrides
  -> python3 -m verl.trainer.main_ppo
  -> @hydra.main(config_path="config", config_name="ppo_trainer")
  -> run_ppo(config) -> ray.init(...)
  -> TaskRunner.remote().run(config)
  -> tokenizer/processor/RLHFDataset/StatefulDataLoader
  -> Ray resource pool + FSDP actor/rollout(/ref) workers
  -> RayPPOTrainer.init_workers() -> fit()
```

证据：

- 作者入口：`Latent-GRPO-gsm8k-llama3.sh:8-72`, `Latent-GRPO-math500-qwen.sh:8-72`。
- Hydra/Ray：`verl-0.4.x/verl/trainer/main_ppo.py:18-58`。
- worker/resource pool/reward/dataset/trainer：同文件 `:75-154`。
- 默认配置：`verl-0.4.x/verl/trainer/config/ppo_trainer.yaml:1-291`；shell 以 CLI override改写。

作者 1B/GSM8K profile：train batch 64、rollout n=8、response 128、SGLang TP=1、FSDP no offload、BF16、remove-padding、8 GPU。作者 7B/Math profile：train batch 32、n=8、response 4096、TP=1、gradient checkpointing、actor param/optimizer offload、8 GPU。二者都不是可直接声称适用于三卡的配置；上游检查 `train_batch_size*n` 与 GPU数整除，并按 DP size归一化 mini-batch（`ray_trainer.py:440-524`, `fsdp_workers.py:143-164`）。

直接对 `main_ppo` 执行 `torchrun --nproc_per_node=3` 会使三个进程各自进入 `ray.init()`/TaskRunner；仓库没有 rank-0 guard。设计采用 torchrun 控制外壳 + rank0唯一Ray coordinator，仍需目标机验证。

## 4. 真实 rollout → actor update 数据流

| 顺序 | 操作与关键字段 | 证据 |
|---:|---|---|
| 1 | StatefulDataLoader取 prompt，弹出 generation keys | `ray_trainer.py:1010-1034` |
| 2 | actor-rollout worker调定制SGLang生成 | `ray_trainer.py:1038-1047` |
| 3 | prompt赋随机 UUID `uid`，按 rollout n interleave repeat | `ray_trainer.py:1064-1069` |
| 4 | union response/top-K fields，构造 response mask | `ray_trainer.py:1069-1078` |
| 5 | rule/model reward；动态 group filter/retry/complete-group accumulation | `ray_trainer.py:1081-1212` |
| 6 | 最终batch上重算 old log-prob/entropy/pre-update clean Top-K | `ray_trainer.py:1214-1250`, `fsdp_workers.py:669-713` |
| 7 | 可选 ref policy log-prob | `ray_trainer.py:1253-1260` |
| 8 | score→reward；driver计算 Latent-GRPO advantage/OCP | `ray_trainer.py:1268-1302` |
| 9 | actor逐 PPO mini/micro-batch current forward、loss/backward/step | `ray_trainer.py:1312-1322`, `dp_actor.py:475-613` |
| 10 | validation/checkpoint/现有metrics/log，global step递增 | `ray_trainer.py:1339-1373` |

动态 filter 默认丢弃 reward std=0且group size>1的 group，继续生成直到累计够完整 groups；最终 batch在 `acc_batch[kept_indices]` 后形成并重新 balance/reorder（`ray_trainer.py:1109-1212`）。因此 `train/generated_token_count` 必须冻结于最终 batch，排除 earlier discarded retries；不能复用包含 prompt 的 `perf/total_num_tokens`。

stable ID 必须紧跟 repeat 后、任何 filter/balance/reorder前创建。现有每轮 `uuid4()` group uid与局部 batch index均不满足 trajectory ID契约。

## 5. Latent/noisy Top-K/one-sided 语义

定制 SGLang sampler（`sglang/srt/layers/sampler.py:70-159`）：

1. full-vocabulary logits → float32 log-softmax/probability；top-p mask，并强制至少保留 `max_topk`；
2. 采标准 Gumbel，clip到 `[-1.5, 3]`；启用 one-sided 时加 1.5，再乘 `noise_scale`；
3. 形成 `sampling_log_prob + transformed_noise`，取 noisy Top-K；
4. 对选中 K 个扰动分数除 `gumbel_temperature` 后 softmax，得到实际 mixture weights；
5. 保存 noisy IDs/perturbed scores，并另存 clean top-K probability/ID。

`rollout_topk_gumbels` 对应第3步的 perturbed scores。raw Gumbel diagnostics 不能从它反推，只能在独立 diagnostic 的真实采样点局部 reduce。

下一 token forward 经 `weighted_forward{,_tp}` 将 K 个 vocabulary embedding按 mixture weights加权（`models/llama.py:303-327`, `models/qwen2.py:285-309`, `layers/vocab_parallel_embedding.py:493-559`）。actor重算路径也以 `softmax(perturbed_scores/T)` 复建 mixture并 `inputs_embeds.detach()`（`dp_actor.py:119-229`）。

hard/latent sentinel：hard token的 Top-K 第2..K列是 `-100`；latent mask必须再联合 response区间、attention/loss mask和 runtime shape，不能仅凭 sentinel。仓库没有独立 `max_latent_length` 或现成 `valid_latent_position_mask`。

## 6. Reward、advantage、mask、EOS 与 overlong

- reward manager由 `main_ppo` 的 `load_reward_manager`创建；默认 naive manager将 scorer结果放在每条 response最后一个有效位置（`trainer/ppo/reward.py:60-130`, `workers/reward_manager/naive.py:32-108`）。GSM8K/Math发布 scorer产生0/1 outcome，但自定义 profile必须重新审计统计单位。
- response mask是 response slice的 attention mask；multi-turn才改用 loss mask（`ray_trainer.py:198-213,267-290`, `dp_actor.py:529-535`）。
- EOS位置计入 response length：mask从 EOS之后清零（`utils/torch_functional.py:333-353`）。stop-string/额外stop token与finish reason仍需 runtime probe。
- exclude-overlong advantage路径用 `response_mask.sum == max_response_length` 定义 truncated，不读取SGLang finish reason；其轨迹不参加group mean/std并得0 advantage（`core_algos.py:145-186`）。include模式在 advantage后，actor update仍把达到cap的sample advantage置0（`dp_actor.py:556-562`）。
- correct/non_correct必须从训练真实reward/scorer定义形成二分类；overlong是可重叠布尔，不是第三类。

## 7. Optimal Correct Path

实现：`core_algos.py:113-359`。

- 先按group outcome reward做GRPO advantage；exclude模式可排除max-length轨迹。
- 找每条轨迹第一个 response位置。
- 候选是该位置 advantage > 0 的轨迹。
- 评分是 response mask上的 mean old log-prob；取最大者。
- 仅将其他轨迹的 first-step advantage置0。

当前函数不返回 winner ID/score，只返回 modified scores/returns/group std；old log-prob缺失时还有随机 fallback。下一阶段必须从同一次内存选择直接返回 stable winner，不得事后从已置零 advantage反推或以reward-only替代。

## 8. old/current policy、surrogate 与 FlipGrad

- rollout log-prob来自 SGLang，但训练 old log-prob是 filter后 actor pre-update no-grad forward重算值；两者差值当前仅作诊断（`ray_trainer.py:1217-1250`）。
- current log-prob在每个 update micro-batch forward计算（`dp_actor.py:547-575`）；后续 mini-batch可能已在更晚参数状态，时间点必须在definition中固定。
- latent component全局 log-softmax后 gather rollout noisy IDs；`margin = perturbed_score - component_log_prob`（`utils/torch_functional.py:143-156`）。标准 surrogate component值是 `-margin-exp(-margin)`，K维平均后进入PPO ratio。
- advantage存在时，trigger `(advantage <= 0) & (margin < 0)`；straight-through使forward与标准路径相同，backward使用翻转margin proxy（同文件 `:158-191`）。上游没有独立 `use_flipgrad`开关；默认 noisy update forward会触发该逻辑。

Stage2只能暴露/测试 margin/trigger充分统计接口，不持久化每步 `onesided/*`。Stage4 checkpoint probe才聚合one-sided指标；credit默认关闭。

## 9. Step、optimizer、checkpoint 与 resume

### 9.1 Step

上游 `global_steps` 在resume后设置为checkpoint目录step，然后训练开始前加1，每个外层iteration末尾加1（`ray_trainer.py:984-1003,1368-1369`）。它可映射为外层 global step，但契约要求事件创建时冻结。

一个outer step可包含多个 PPO epoch/mini-batch，每个mini-batch一次 `_optimizer_step`（`dp_actor.py:496-613`）。非有限grad时跳过 `optimizer.step()`（`:379-393`），worker却仍无条件推进 scheduler（`fsdp_workers.py:602-604`）。不能由 global step/scheduler推断 optimizer step；必须返回跨rank一致的 `did_update`，按实际成功次数累计。是否修正 scheduler skip属于算法语义决策，不能标作 logging-only。

### 9.2 Checkpoint/resume

可复用基础：

- FSDP每rank model/optimizer shard；scheduler与worker Python/NumPy/torch/CUDA RNG extra state；tokenizer/config；
- dataloader state；
- `global_step_<N>`目录和latest tracker；
- resume加载actor/critic/dataloader并设置global step。

证据：`ray_trainer.py:869-951`, `utils/checkpoint/fsdp_checkpoint_manager.py:76-252`, `checkpoint_manager.py:108-171`。

缺口：optimizer_step、profile/config/schema/upstream hash、writer manifests、driver RNG、SGLang generator state；latest tracker是普通覆盖写；8卡sharded checkpoint不能假定可直接按3卡恢复。外层需 sidecar + 原子状态，same-world-size resume先验证；HF merge只能作为新run初始化而不是完整optimizer resume。

## 10. 分布式拓扑与三卡边界

```text
one Ray head
  -> one CPU TaskRunner / RayPPOTrainer coordinator
  -> one global GPU resource pool
     -> colocated actor_rollout workers
        -> FSDP/FSDP2 actor world=N
        -> SGLang rollout DP=N/infer_tp, TP=infer_tp
        -> sharding manager进行权重同步
     -> optional ref/critic/RM workers
```

证据：`main_ppo.py:29-45,95-108`, `ray_trainer.py:399-435,786-858`, `fsdp_workers.py:94-164,280-345,389-482`。

发布profile TP=1；三卡可形成3-way FSDP和rollout DP，且 `world_size % infer_tp == 0`。1B从TP=1/FSDP/小micro-batch开始；7B smoke保留作者同类 gradient checkpoint/offload并大幅缩小batch/length。具体46GB容量、Ray colocation、memory-saver、NCCL与高模型TP只能runtime验证，不在本报告声称成功。

## 11. 复用、adapter 与是否 patch

### 11.1 可直接复用

| 接口 | 决定 | 证据 |
|---|---|---|
| Hydra config/`run_ppo` | adapter复用 | `main_ppo.py:24-37` |
| dataset/sampler | 复用 | `main_ppo.py:157-211` |
| reward manager/compute_reward | 复用 | `trainer/ppo/reward.py:60-130` |
| response mask | 复用并version化 | `ray_trainer.py:198-213` |
| Latent GRPO/OCP | 算法复用、扩展观测返回 | `core_algos.py:113-359` |
| PPO loss | 复用 | `core_algos.py:707-769` |
| SGLang sampler/weighted embedding | 必须复用 | `sampler.py:70-159`, embedding路径 |
| FSDP/SGLang sharding | 必须复用 | `fsdp_workers.py:438-476` |
| FSDP checkpoint | 状态基础复用，补sidecar/atomicity | checkpoint manager |

### 11.2 必须 adapter/instrument 的私有切点

- `RayPPOTrainer.fit`：stable IDs、final rollout、group raw facts、immutable events；
- `compute_latent_grpo_*`：同一次 OCP winner/candidates；
- SGLang postprocess：actual mixture/diagnostic局部统计；
- actor micro-batch：component margin/flip/PG充分统计；
- actor optimizer：`did_update`；
- validation：question/generation raw facts + clean Top-K；
- checkpoint：sidecar、writer resume、probe调度。

**结论：必须修改作者仓库，但只做最小、显式 instrumentation patch。** 仅靠外部公开入口拿不到 stable ID、OCP winner、component stats、optimizer success与eval raw facts；复制整个 `fit()`或脆弱monkey patch风险更高。所有上游改动进入 `patches/`和`docs/upstream_changes.md`；logging-off做等价性测试。

## 12. 已发现风险与处理

### Blocker candidates

1. `utils/torch_functional.py:32-37,133-195`：FlashAttention cross-entropy import失败会回退普通 token log-prob；Latent profile必须fail-fast，不能静默退化。
2. `dp_actor.py:119-378`：Top-K replay输出只在 remove-padding分支完整构造；non-remove-padding返回未定义字段。三个profile必须保持remove-padding并probe。
3. `dp_actor.py:231-343`：fused-kernel分支未定义后续Top-K处理使用的 `logits_rmpad`；默认禁用直至修复/验证。
4. `dp_actor.py:441-464`：`topk_logits`错误拼接 `topk_ids_lst`。任何probe不能消费该字段；下一阶段以失败测试最小修复。

### Major candidates

1. Qwen脚本 latent-end=522，但 sampler `:132-145` 有硬编码524；改为配置值前做边界probe。
2. dynamic-bsz仅reverse reorder log-prob，未同步entropy/top-K（`dp_actor.py:461-473`）；默认禁用，直至全字段回序测试通过。
3. rollout/old-current sampling参数可能被config覆盖；新adapter assert temperature/top-p/K一致。
4. include OCP std构造可疑；用synthetic reference测试确认。
5. checkpoint latest tracker非原子且driver/SGL RNG不完整。

## 13. Runtime probe 清单

加载训练前：Python/torch/CUDA/driver/cuDNN/NCCL/GPU/BF16/disk；fork import路径；FlashAttention latent分支；FSDP2/Ray/NCCL/SGLang kernels；tokenizer latent-end/K。

首个无更新Tensor probe：

- rollout IDs/perturbed scores/clean fields shape/dtype/device/K/padding；
- actual weights与 `softmax(score/T)`；
- old-topK在filter/balance/dispatch后对齐，dynamic-bsz off/on；
- response/attention/loss/sentinel/roll-shift的latent mask；
- EOS/stop/max-length finish；
- clean top-K概率空间与sum规则。

最小算法/梯度probe：perturb score公式、one-sided范围/clip、margin、FlipGrad truth table/forward invariance/gradient sign、OCP winner/tie、include/exclude overlong。

checkpoint/resume probe：连续4步 vs 2+resume到4；每rank和SGLang RNG；成功optimizer次数；三卡same-world-size shards；probe前后状态hash。

## 14. 计划观测切点

| Phase | 切点 | 变量 |
|---|---|---|
| post-repeat/pre-filter | `ray_trainer.py:1064-1069`后 | stable group/trajectory IDs |
| final-rollout/post-filter | `:1173-1214`后 | generated count、length、overlong raw facts |
| rollout sampler local | `sampler.py:91-128` | diagnostic局部reduce、actual weights |
| pre-update old-log-prob | `ray_trainer.py:1217-1226` | old log-prob/entropy/pre-update Top-K/Support |
| post-advantage/pre-update | `:1288-1303`后 | reward/final adv/zero-adv/group/OCP |
| current actor micro-batch | `dp_actor.py:551-608` | current ratio/loss/component margin/flip stats |
| optimizer commit | `dp_actor.py:379-393,610` | did_update/optimizer step |
| post-update driver | `ray_trainer.py:1320-1366` | train-step/timing/LR event |
| checkpoint eval | `ray_trainer.py:678-784` | per-question/generation/clean Top-K |
| save/load | `ray_trainer.py:869-951` | sidecar/resume/probe |

本阶段最终判断：复用算法与分布式内核；新增外部runner/metrics/storage；对上游做可逆最小patch；三卡与全部CUDA/版本/shape事实保持未验证，等待下一阶段 runtime probe。
