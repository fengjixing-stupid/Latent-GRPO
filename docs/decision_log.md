# Latent-GRPO 审计阶段决策日志

基线：作者仓库 commit `c0994fb781a2d180662bb522d8ff3e8638dcf56d`。本文件记录当前设计决策；带“runtime gate”的条目只有在目标 Linux/GPU 上 probe 通过后才能升级为已验证事实。

## D-001：新 runner 放在作者仓库外

- **状态：** 采用。
- **决定：** `train_latent_grpo.py` 与 `latent_grpo_runner/` 放在项目根目录，通过显式 `upstream_repo_path=./Latent-GRPO` 接入 vendored verl/SGLang。
- **理由：** 规范要求职责隔离；作者仓库继续作为算法真相，避免复制一套漂移的训练实现。
- **影响：** import bootstrap、commit/hash 校验和 Hydra override 由 adapter 统一处理。

## D-002：必须对作者仓库做最小、显式 patch

- **状态：** 已实施最小补丁，Mac 仅完成静态/合成测试；目标机运行等价性待验证。
- **决定：** 不复制 `RayPPOTrainer.fit()`，不使用运行时 monkey patch；增加 observer 接口、stable trajectory ID、真实 optimizer-step 结果和有限统计返回，并修复阻塞观测的已证实缺陷。
- **证据：** 上游训练控制点集中在 `verl-0.4.x/verl/trainer/ppo/ray_trainer.py:1016-1369`；worker Tensor 只在 `verl-0.4.x/verl/workers/actor/dp_actor.py:71-378,476-614` 可精确观察。
- **风险：** logging-only 声明需要 observer-off 等价性测试；任何有算法影响的修复必须独立标识。
- **回滚：** `patches/latent_grpo_observer.patch` 可反向应用；`metrics_enabled=false` 关闭 observer。

## D-003：`ray_direct` 是默认启动面，`torchrun_control` 仅为显式兼容模式

- **状态：** 更正后采用，Mac 命令生成/模拟测试通过，runtime gate。
- **决定：** 目标机以一个直接 Python 进程启动上游 Ray driver，由 Ray 创建 3 个 GPU worker；默认 `launcher.mode=ray_direct`。`launcher.mode=torchrun_control` 仅保留为显式兼容模式，不得成为三卡主路径。Ray resource pool 候选使用 3 GPUs，FSDP DP=3，SGLang TP=1/SP=1。
- **证据：** 作者入口 `verl/trainer/main_ppo.py:24-37` 直接 `ray.init()`，`main_ppo.py:95-154` 创建 Ray GPU resource pool 和 trainer；原脚本也以单个 `python3 -m verl.trainer.main_ppo` 启动（两个作者 shell 的第 8 行）。直接让三个 torchrun rank各自进入上游会创建三套 Ray 作业。
- **拒绝方案：** 直接三 rank 各跑 Hydra/Ray；让 rank 1/2 长期空闲作为默认路径；把 vendored Ray trainer 重写成原生 torchrun trainer；要求 Docker/Slurm。

## D-004：权威 writer 位于 Ray coordinator

- **状态：** coordinator transport/merge 接口已实施并通过 Mac 合成测试；权威 Parquet sink 尚未接入，真实 metrics launch 为 `unavailable_with_reason`。
- **决定：** FSDP worker 只返回 sufficient statistics；Ray coordinator 绑定不可变 step context、聚合并写 Parquet。torchrun envelope 不写动态指标。
- **理由：** 现有 `fit()` 本身是 coordinator dataflow，能同时看到 group/reward/advantage/update/checkpoint；避免 worker 竞争文件和 worker mean 简单平均。
- **当前边界：** 自定义 DP collector 保留所有 rank 的小型 packet，coordinator 校验 optimizer outcome 共识并以 `sum/sum_sq/count/min` 合并 component packet；但尚未把 actor/OCP/eval events 映射成 schema-complete Stage 1/2 rows 并交给 `AppendOnlyPartWriter`。因此默认 `metrics_enabled=true` 会在进入目标 runtime 前阻断，不能用内存 buffer 或 JSONL 冒充权威 sink。

## D-005：Support 复用现有 old-log-prob forward，不增加 forward

- **状态：** 采用，shape/alignment runtime gate。
- **决定：** rollout 端用 batch 中 `rollout_topk_ids`；pre-update 端用同一步 `compute_log_prob` 已返回的 `old_topk_indices`，在 advantage/Optimal Correct winner 已知后、actor update 前于 coordinator 做严格对齐。
- **证据：** rollout fields 在 `verl/workers/rollout/sglang_rollout/sglang_rollout.py:699-775` 形成；pre-update forward 在 `ray_trainer.py:1217-1226`；worker 已返回 `old_topk_probs/old_topk_indices`（`verl/workers/fsdp_workers.py:669-713`）。
- **失败语义：** K、mask、trajectory order、shape 任一不一致，整 Support family 为 unavailable；禁止补 forward、截断或广播。

## D-006：`rollout_topk_gumbels` 按“扰动后分数”解释

- **状态：** 采用。
- **决定：** 该上游名字不能按 raw Gumbel noise 解读。它来自 `log_prob + transformed_noise` 的 top-K score；noisy mixture 权重是其按 Gumbel temperature 的 softmax；Stage 4 Delta 是它减当前 component log-prob。
- **证据：** 定制 SGLang `sglang/srt/layers/sampler.py:74-126` 先构造 `sampling_log_probs + gumbels`，再 top-k 并存入 `topk_gumbels`；actor `verl/utils/torch_functional.py:143-175` 计算 `raw_diff = rollout_topk_gumbels - topk_log_probs` 并据此 FlipGrad。
- **影响：** raw Gumbel diagnostic 不能从持久化 rollout score 反推，必须在独立 diagnostic 的采样点局部统计。

## D-007：Optimal Correct Path 使用作者正 first-step advantage + mean old log-prob

- **状态：** 采用；稳定 ID 需 patch。
- **决定：** 不以 reward-only 近似 winner；在 advantage 置零前保存 winner 的 stable trajectory ID 与 mean old log-prob，并通过内存接口给 group 表和 Support 共用。
- **证据：** `verl/trainer/ppo/core_algos.py:188-245` 与 `304-359` 实现正 first-step advantage 候选和最大 mean old-log-prob 选择。
- **风险：** 当前函数只返回 scores/id2std，没有返回 winner；无 old log-prob 时还会随机 fallback。新 adapter 必须保证训练路径有 old log-prob并显式返回 winner，不能事后反推。

## D-008：独立维护 `global_step` 与成功的 `optimizer_step`

- **状态：** worker outcome 与 coordinator consensus 接口已实施；checkpoint sidecar 累计/恢复仍被 durable sink 缺口阻塞。
- **决定：** `global_step` 按一次外层 rollout→reward→advantage→actor update 所属迭代；`optimizer_step` 按每个实际执行的 actor `optimizer.step()` 累加。非有限 grad 被跳过不计数。
- **证据：** 上游只维护 `global_steps`（`ray_trainer.py:984-1003,1368-1369`）；actor 在非有限 grad 时跳过更新（`dp_actor.py:379-393`），但 worker 仍无条件推进 scheduler（`fsdp_workers.py:602-604`）。
- **scheduler 决定：** 有至少一次成功参数更新时保持作者原有的每 outer actor update 前进一步；全部 attempts 因 non-finite grad 跳过时 scheduler 不前进；成功/跳过混合时前进一步。该修复即使 observer 关闭也生效，是明确的训练语义修复，不声明为 logging-only。Mac 已完成源码/合成边界测试，真实 FSDP non-finite 共识仍为 `target_machine_test_deferred`。
- **剩余影响：** checkpoint sidecar 必须保存两个计数；在 Parquet sink/sidecar 接入前，driver event 中的 `update_count` 不能被宣称为已恢复的累计 `optimizer_step`。

## D-019：metrics 配置权威控制 observer，但无 Parquet sink 时阻断真实启动

- **状态：** 采用，当前真实 metrics launch 为 `unavailable_with_reason=durable_parquet_sink_not_wired`。
- **决定：** 三个工程 profile 默认 `features.metrics_enabled=true`；launcher 权威覆盖 `LATENT_GRPO_OBSERVER_ENABLED`，启用时设为 `1`，显式 `--disable-metrics` 时设为 `0`，不接受外部环境静默反转。dry-run 显示开关和 sink 状态，resume 复用相同映射。
- **安全边界：** `BufferedObserver` 仅用于 synthetic tests。当前没有把 `AppendOnlyPartWriter`、schema 和 Stage 1/2 builder 注入 `main_ppo`，所以真实 metrics 启动在环境 probe、模型加载和 upstream subprocess 前返回 blocked。`--disable-metrics` 只允许明确的无指标 smoke，不能用于声称 Stage 1/2 验收通过。
- **后续解锁条件：** 同一个 driver-owned durable sink 必须消费 OCP/eval/actor-update events，构造 schema-complete rows，提交 append-only Parquet parts，并将累计 `optimizer_step` 写入 resume sidecar。

## D-009：上游已发现的 top-K 返回缺陷需要独立修复

- **状态：** 补丁实施中；只有独立回归审查通过后才记为 `synthetic_test_passed`。
- **决定：** `dp_actor.compute_log_prob()` 的 `topk_logits` 应拼接 `topk_logits_lst`，而当前拼接的是 `topk_ids_lst`。在使用该字段前写失败测试并最小修复。
- **证据：** `verl/workers/actor/dp_actor.py:441-473`。
- **算法影响：** 当前训练未消费 `topk_logits` 时对训练结果无影响；新指标若消费错误字段会产生错误 Delta，因此必须先修复或不使用它。

## D-010：latent-end token ID 禁止硬编码

- **状态：** 已实施配置传播和 fail-closed 补丁，Mac 静态/合成测试待独立审查，runtime gate。
- **决定：** 全链路从 profile/tokenizer 验证得到 `latent_end_token_id`；作者 LLaMA=524、Qwen=522 只能是显式 profile 候选，必须校验 token 字符串、tokenizer/model vocab 范围和 ID 映射。SGLang sampler 改读取启动配置；无法解析时不做静默 fallback。
- **证据：** 作者脚本分别设 524/522；README `:187-195` 要求 tokenizer 验证；定制 sampler `sglang/srt/layers/sampler.py:132-145` 仍硬编码 524。

## D-011：clean top-K 只转存已有 eval forward 结果

- **状态：** 采用。
- **决定：** checkpoint eval 噪声关闭时转存定制 SGLang 已返回的 original top-K IDs/probs；字段不在同一 forward 可得或无法对齐时写 unavailable，不额外 softmax/top-k/full forward。
- **证据：** `sglang_rollout.py:191-224,702-766` 已传回 original probs/indices；训练 old-log-prob 也产生 top-K，但不把训练表误作 checkpoint eval raw facts。

## D-012：requirements 分层，不复制作者完整环境快照

- **状态：** 采用；精确 pin 受 dependency audit/runtime gate 约束。
- **决定：** 根 `requirements.txt` 只引用 `requirements/runtime-core.txt`、`requirements/runtime-sglang.txt` 与 `requirements/metrics.txt`；tracking、math reward、可选 vLLM、测试和文档依赖分别进入 `requirements/tracking.txt`、`reward-math.txt`、`vllm-optional.txt`、`test.txt`、`docs.txt`。目标机验证后的 Python 组合才进入 `constraints/linux-cu124-py311.txt`。vendored verl/SGLang 用 editable path；系统 driver/CUDA toolkit/NCCL/compiler 不写入 Python requirements。
- **理由：** 作者根 requirements 含 200+ 直接/传递 pin 和 editable path，内部还存在 torch/torchvision/torchaudio、SGLang post1/post5、sgl-kernel 版本口径需核对；不得用 `pip freeze` 或印象重新生成。

## D-013：三卡用 FSDP DP=3、SGLang TP=1

- **状态：** 采用，runtime gate。
- **决定：** 低难度和 7B smoke 都先使用 FSDP 三路数据并行；不使用不能整除 3 的 TP=2。所有 prompt×rollout、mini/micro batch 满足上游可除性检查。
- **证据：** 上游配置校验在 `ray_trainer.py:440-457,511-524`；actor worker 会按 DP world size 归一化 mini-batch（`fsdp_workers.py:143-155`）。
- **影响：** 7B high 只做链路 smoke；OOM 时生成明确更小 profile，不能静默量化或换模型。

## D-014：当前机器不能替代目标机验证

- **状态：** 已确认并持续适用。
- **事实：** 当前开发环境是 Darwin arm64、Python 3.11.9，无 NVIDIA runtime；Linux/CUDA 训练栈不在 Mac 上安装或验证。
- **决定：** Mac 可完成实现、CPU/合成测试与 PyArrow 存储验证；所有 CUDA/NCCL/BF16/显存/扩展 ABI/单卡/三卡结论保持 `target_machine_test_deferred`，等待 Linux 目标机实测。

## D-015：状态声明采用证据分级

- **状态：** 采用。
- **决定：** 报告与 RTM 只使用当前任务规定的证据标签：`implemented`、`static_check_passed`、`synthetic_test_passed`、`mac_development_check_passed`、`target_machine_test_deferred`、`target_machine_probe_passed`、`single_gpu_tested`、`three_gpu_ray_tested`、`cuda_runtime_verified`、`requirements_lock_verified`、`memory_feasibility_verified`、`blocked`、`unavailable_with_reason`。Mac 上最高只声明前五种中有证据的状态。

## D-016：三卡候选批量采用可证明整除的 `P/n/M/micro`

- **状态：** 采用，显存与吞吐仍为 runtime gate。
- **决定：** low 用 `P=6,n=4,M=3,micro=2`；high-smoke 用 `P=3,n=4,M=3,micro=1`。两者 TP=1、SP=1；分别得到每 rank local batch 8/4、local mini 4/4、gradient accumulation 2/4。
- **理由：** 必须同时满足 `P*n % 3 == 0`、`M*n % 3 == 0`、local batch/mini/micro 整除；不能依赖上游整数地板除。详细推导见 `work_reports/agent_d_3gpu_plan.md`。

## D-017：不在 runtime 证据前修改 DeviceMesh 维名

- **状态：** 采用。
- **决定：** rollout 自建 `dp/tp/pp` mesh，sharding manager 使用单独 `dp/infer_tp` mesh；静态维名不同不等价于通信必然失败。先执行三卡 weight-sync + latent-rollout probe，只有复现 group/collective 不一致后才设计修复。

## D-018：`max_latent_length` 当前是实现缺口

- **状态：** 已确认。
- **决定：** profile 可记录 desired latent cap，但作者现有链路没有独立生效旋钮；在 adapter/patch 和训练重放/mask 测试完成前，不能声称该 cap 已执行。
