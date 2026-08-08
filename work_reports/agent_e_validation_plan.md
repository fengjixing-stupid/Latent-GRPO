# Agent E — 测试与验收方案（只读设计）

审计日期：2026-08-02  
状态：**planned**  
审计范围：`./spec` 七份规范、`./Latent-GRPO`、Agent A/B/C 报告与 `docs/implementation_plan.md`  
本阶段实际动作：只读检查与本文档写入；**未安装依赖、未 import GPU 训练栈、未运行本机 GPU、未启动训练**。

## 1. 验收结论与边界

本阶段只能给出可执行的验收设计，不能把任何项目写为 passed、verified、single-GPU tested 或 3-GPU tested。规范要求“代码存在 + 关键测试实际执行 + 单卡 smoke + 三卡最小训练 + validator + 全字段追踪 + no-mutation”同时满足后才能声明完成（`spec/05_VALIDATION_AND_DELIVERABLES.md:17-30`），并要求区分 synthetic、单卡、三卡和未测试（同文件 `:438-474`）。因此本文全部测试状态均为 **planned**。

CPU/synthetic distributed 测试只能证明聚合算法、rank ownership 和错误传播协议，不证明 CUDA、NCCL、Ray/FSDP/SGLang、三卡显存或真实 Latent-GRPO 训练可用。目标机器不可用时必须写“3 GPU 未实测”，不能把 mock/synthetic 证据升级为三卡证据（`spec/05_VALIDATION_AND_DELIVERABLES.md:281-305`）。

### 1.1 证据分类

| 代码 | 证据分类 | 可以证明 | 不可以证明 |
|---|---|---|---|
| `S` | static | 语法、文件/schema/禁止项、配置静态关系 | import 成功、运行时 Tensor 语义 |
| `U` | unit-synthetic | 纯函数公式、mask、ID、存储故障注入 | 上游真实调用点或 GPU 语义 |
| `D` | synthetic-distributed | sufficient stats 在多个模拟 worker 的合并、owner 去重 | NCCL/Ray/FSDP 三卡工作 |
| `I-CPU` | CPU integration | CLI/dry-run、进程控制、无 CUDA 的集成边界 | GPU kernel/显存/训练更新 |
| `R-1GPU` | target-runtime single GPU | 目标机单卡真实 rollout→update→checkpoint | 三卡初始化/聚合 |
| `R-3GPU` | target-runtime 3 GPU | 三卡真实 rank/Ray/FSDP/SGLang/训练与写盘 | 论文严格复现或长训练效果 |
| `P` | performance-runtime | 指定 commit/profile/hardware 上的采样开销 | 其他机器或其他 batch 的承诺 |
| `IR` | independent review | 独立检查者对 RTM、语义、证据与夸大声明的复核 | 替代实际测试 |

每份实际测试记录必须包含：源码 commit、上游 commit、config hash、schema/definition versions、命令、开始/结束时间、exit code、脱敏 platform snapshot、stdout/stderr artifact、输出目录和证据分类。只有 exit code 与所有断言同时满足才可把对应 RTM 行从 `planned` 提升；静态或 synthetic 通过不能提升为 GPU verified。

### 1.2 阶段门禁

```text
G0 静态/配置/依赖 probe 设计通过
  -> G1 unit + synthetic distributed + storage fault tests 全通过
  -> G2 dry-run/import/runtime tensor probe 通过
  -> G3 单卡真实 2-step smoke + validator + no-mutation + resume 通过
  -> G4 三卡 rank-init/NCCL/Ray single-driver 通过
  -> G5 3gpu-low 2-step + checkpoint load + validator + 性能证据通过
  -> G6 3gpu-high-smoke 1-step（仅链路验收）
  -> G7 target coverage missing=[] + 独立 reviewer blocker=0
  -> 才允许长训练或完成声明
```

规范明确禁止在 generated token count、stable ID、Support 对齐、probe no-mutation、resume/去重或 reviewer blocker 未闭合时进入长训练（`spec/04_AGENT_ORCHESTRATION.md:339-350`）；实施计划也规定单卡/validator/resume/probe 失败不得进入三卡，三卡 writer ownership 失败不得进入 7B（`docs/implementation_plan.md:263-271`）。

## 2. 测试 ID 矩阵

以下“精确命令”是下一阶段实现对应测试文件后的执行命令，不表示本阶段已经执行。所有命令从项目根目录运行；pytest node 名称也构成实现契约，不应通过改名或删测试规避失败。

### 2.1 静态、配置与 runtime probe

| ID | Purpose / level | Fixture | Exact command | Expected result | Evidence | Blocking gate |
|---|---|---|---|---|---|---|
| `T-STATIC-001` | Python 语法与测试可收集；static | 完整新代码树 | `python -m compileall -q train_latent_grpo.py latent_grpo_runner scripts tests && python -m pytest --collect-only -q` | exit 0；无 syntax/import collection error | S | G0 |
| `T-STATIC-002` | 不含用户绝对路径、secret、作者仓库复制；static | repo scan fixture | `python -m pytest tests/unit/test_static_safety.py -q` | 无 `/Users/...` 等私有硬编码、token/key 内容；输出目录逻辑不复制 `Latent-GRPO` | S | G0 |
| `T-STATIC-003` | 项目已采用的 lint/type gate；static | repo tool configuration | `python -m pytest tests/unit/test_tooling_contract.py -q` | 若仓库选择 ruff/flake8/mypy/pyright，测试调用并要求 exit0；未采用的工具明确 `not_adopted`，不为形式安装新工具 | S | G0 |
| `T-CONFIG-001` | 三个 profile 与全部 CLI 参数；unit | `smoke/3gpu-low/3gpu-high-smoke` golden configs | `python -m pytest tests/unit/test_config.py::test_required_profiles_and_cli_contract -q` | 三配置解析；未知字段/非法 enum/缺关键字段失败；CLI 覆盖确定且进入 config hash | U | G0 |
| `T-CONFIG-002` | 硬件/profile 与整除门禁；unit | 1/2/3 GPU fake snapshots、batch matrix | `python -m pytest tests/unit/test_config.py::test_hardware_and_batch_gates -q` | 三卡 profile 在 GPU 数/显存/整除不符时默认拒绝；override 只记录 mismatch，不改 profile 身份 | U | G0 |
| `T-CONFIG-003` | paper profile 不冒充；unit | paper/adapter profile fake configs | `python -m pytest tests/unit/test_config.py::test_reproduction_claim_gate -q` | 少于论文条件时拒绝或要求显式 override；三个交付 profile 均标 device-adapted/non-paper | U | G0 |
| `T-DRYRUN-001` | dry-run 不加载训练、不更新参数；CPU integration | monkeypatched Ray/CUDA/model loaders that fail if called | `python train_latent_grpo.py --config configs/smoke.yaml --dry-run --validate-config --output-root /tmp/latent-grpo-validation/dry-run` | exit 0；验证 upstream/config/model-data-cache path/schema/output；无 Ray init、CUDA init、model load、forward 或 optimizer step | I-CPU | G2 |
| `T-PROBE-001` | platform snapshot 完整与脱敏；unit | fake hostname/path/GPU/library responses | `python -m pytest tests/unit/test_environment.py::test_snapshot_schema_and_redaction -q` | 覆盖 OS/Python/torch/CUDA driver/runtime/cuDNN/NCCL/GPU/memory/CC/BF16/world/rank/disk/commit/workspace；hostname、executable path 与用户名哈希 | U | G0 |
| `T-PROBE-002` | dependency/runtime branch fail-fast；unit | missing FlashAttention, wrong verl/sglang origin, K mismatch, remove-padding/fused variants | `python -m pytest tests/unit/test_environment.py::test_latent_runtime_semantic_gates -q` | latent FlashAttention likelihood 分支未生效、自定义 fork 来源错误、K/latent-end/remove-padding 不符均稳定 reason 并拒绝训练，不静默退回普通 token log-prob | U | G2 |
| `T-PROBE-003` | 真实 tensor interface probe；single GPU runtime | 最小本地 latent model，1 prompt/2 tokens | `CUDA_VISIBLE_DEVICES=0 python scripts/inspect_environment.py --config configs/smoke.yaml --tensor-probe --output-root /tmp/latent-grpo-validation/tensor-probe` | 记录 logical name/producer/shape/dtype/device/requires_grad、K、response slice、next-token shift、mask sum、DP/TP ownership；不做 optimizer step | R-1GPU | G2 |
| `T-DIST-001` | synthetic sufficient-stats；distributed synthetic | 3 worker unequal counts `[1,2,7]`、TP duplicate packet | `python -m pytest tests/integration/test_synthetic_distributed.py -q` | 全局 sum/sum_sq/count 正确；worker means 的简单平均会与 golden 不同且被拒；TP replica 只计 DP owner | D | G1；**非三卡证据** |
| `T-DIST-002` | CPU 三 wrapper rank single-driver ownership；CPU integration | env `WORLD_SIZE=3` fake launcher | `python -m pytest tests/integration/test_launcher.py::test_three_wrapper_ranks_one_ray_driver -q` | 只有 wrapper rank 0 启动 Ray driver/writer；rank 1/2 不初始化 CUDA、不写权威文件；driver failure 向全体传播非零 | I-CPU | G2；**非三卡证据** |
| `T-DIST-003` | 目标机 NCCL/rank-only；3 GPU runtime | 3 CUDA devices | `CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 scripts/probe_distributed.py --backend nccl --output-root /tmp/latent-grpo-validation/rank-init` | 3 ranks 设备映射唯一；all-reduce/barrier 正确；仅 rank0 artifact；超时/异常非零并保留日志 | R-3GPU | G4 |

依据：runtime snapshot 必填字段与 mismatch 行为见 `spec/02_SYSTEM_AND_IMPLEMENTATION_CONTRACT.md:87-130`；dry-run 不加载训练见 `spec/05_VALIDATION_AND_DELIVERABLES.md:234-255`。上游高风险路径包括 latent likelihood import 失败静默退化（`Latent-GRPO/verl-0.4.x/verl/utils/torch_functional.py:32-37,133-195`）、pre-update top-K 只在 remove-padding 分支完整且 K 硬编码（`Latent-GRPO/verl-0.4.x/verl/workers/actor/dp_actor.py:119-378`）以及直接 torchrun 会建立三个 Ray driver 的风险（`work_reports/agent_a_repo_audit.md:92-96`）。

### 2.2 统计、mask、generated-token、stable ID 与 group

| ID | Purpose / level | Fixture | Exact command | Expected result | Evidence | Blocking gate |
|---|---|---|---|---|---|---|
| `T-STATS-001` | moments/rate/min/max；unit | unequal worker packets, finite/NaN values | `python -m pytest tests/unit/test_aggregators.py::test_global_sufficient_statistics -q` | population mean/std 由 global sum/sum_sq/count 得到；rate=numerator/count；variance 下界 0；min/max 全局归并 | U | G1 |
| `T-STATS-002` | empty mask/NaN/独立 count；unit | empty、all-NaN、different-mask arrays | `python -m pytest tests/unit/test_aggregators.py::test_empty_nan_and_independent_counts -q` | empty -> null,count=0,`empty_effective_mask`；NaN count 明确；不同 mask count 不共享 | U | G1 |
| `T-MASK-001` | response/latent/eligible masks；unit | prompt+latent+hard+EOS+post-EOS+padding+loss-masked positions | `python -m pytest tests/unit/test_masks.py::test_response_latent_and_eligible_masks -q` | prompt/padding/hard/loss-excluded 均排除；EOS 口径按 version fixture；zero numerator 与 eligible denominator 完全同 mask | U | G1 |
| `T-MASK-002` | actor 最终 advantage 与 overlong 清零边界；unit | include/exclude-overlong 两模式 | `python -m pytest tests/unit/test_stage2.py::test_final_actor_advantage_and_zero_rate -q` | 使用 actor-loss 最终 advantage；每 trajectory 只计一次，不随 PPO epoch/microbatch 重复；zero threshold 固定 | U | G1 |
| `T-STAGE1-001` | 10个Stage1字段的source/mask/phase/count；unit | instrumented actor update packets | `python -m pytest tests/unit/test_stage1.py::test_all_core_metrics_sources_counts_and_phase -q` | 10项齐全；policy loss/KL/clip/ratio/entropy复用真实update路径；ratio mean/std同mask；entropy source/probability-space/version明确；row为post_update且普通表无checkpoint_step | U | G1 |
| `T-STEP-001` | global/optimizer step与immutable event context；unit | success, nonfinite skip, multi-minibatch, async writer delay | `python -m pytest tests/unit/test_step_context.py -q` | optimizer_step只计成功`optimizer.step()`且各rank一致；global_step只计完整outer loop；writer延迟不改变已冻结step；scheduler语义与metadata一致 | U | G1 |
| `T-TOKEN-001` | generated token 固定口径；unit | normal、EOS、overlong、retained retry、discarded retry、actor-filtered trajectory | `python -m pytest tests/unit/test_generated_token_count.py -q` | 只加 final training rollout；进入 reward/advantage 的 overlong 保留；discarded retry/probe 排除；整数 worker sum；拒绝 mean×batch estimate | U+D | G1/G3 |
| `T-TOKEN-002` | probe/Support 不进入训练累计；integration | train event + Support + probe token events | `python -m pytest tests/integration/test_token_accounting.py -q` | response/latent/generated/cumulative 均只因训练 rollout 改变；probe/support count 独立 | I-CPU | G3 |
| `T-ID-001` | repeat 后、reorder 前创建 stable IDs；unit | 2 groups×4 trajectories，permutation/filter/select | `python -m pytest tests/unit/test_stable_ids.py::test_repeat_reorder_filter_select -q` | 同 group IDs 为稳定出现序且唯一；重排/过滤/选择后 identity 不变；local batch index 从不落盘 | U | G1 |
| `T-ID-002` | resume/重复输入规则稳定；integration | same prompt/seed/step and resumed state | `python -m pytest tests/integration/test_stable_ids_resume.py -q` | 对相同 run identity 与恢复状态得到相同 group/trajectory keys；动态 filter retry 不制造已丢弃 final rows | I-CPU | G3 |
| `T-GROUP-001` | 二分类、overlong overlap、count identity；unit | correct-overlong、noncorrect-overlong、normal rows | `python -m pytest tests/unit/test_group_metrics.py::test_classification_and_count_identity -q` | correct/non_correct 互斥且和=trajectory_count；overlong 与两类重叠，不作为第三类 | U | G1 |
| `T-OCP-001` | Optimal Correct Path；unit | positive first-step advantages、old log-probs、ties、no candidate | `python -m pytest tests/unit/test_optimal_correct_path.py -q` | 仅 positive first-step advantage 候选；按 masked mean old log-prob；tie 规则稳定；winner 为 stable trajectory ID；不得 reward-only 或后验反推 | U | G1 |
| `T-MIX-001` | actual noisy mixture；unit/runtime probe | known perturbed scores/T、captured SGLang weights | `python -m pytest tests/unit/test_stage2.py::test_noisy_mixture_metrics_from_actual_weights -q` | effective-K/top1 与 actual weights golden 一致；只计 final noisy latent positions；NaN/degenerate 可用性正确；不计算 clean mixture | U（actual equality 后需 R-1GPU） | G1/G3 |
| `T-GUMBEL-001` | 5个diagnostic仅独立模式；unit/runtime | fixed raw Gumbel/clip/one-sided values + invocation spy | `python -m pytest tests/unit/test_gumbel_diagnostics.py -q` | raw mean/std、lower/upper clip rate、zero rate和两count正确；正式训练disabled且不执行reduce；不保存/传输full-vocab Tensor | U（采样点需R-1GPU） | G1/G3 |
| `T-EVAL-001` | eval generation completeness/raw facts；integration | 2 questions×3 generations，含 wrong/parse fail/timeout/generation fail | `python -m pytest tests/integration/test_checkpoint_eval.py::test_all_generations_and_failures_are_rows -q` | generation_id 完整 0..2；每失败仍一行且 reason；reference hash 与 manifest 一致；无 checkpoint `eval_metrics` | I-CPU | G3 |
| `T-EVAL-002` | clean top-K 转存与列表约束；unit/integration | existing top-K IDs/probs + shape mismatch | `python -m pytest tests/integration/test_checkpoint_eval.py::test_clean_topk_passthrough_only -q` | 仅同 forward 已有数据；逐有效 latent position；IDs/probs/K 等长、概率有限非负；shape 错误 unavailable，不调用额外 softmax/top-k/forward | I-CPU | G3 |

依据：统计与 mask 必测项见 `spec/05_VALIDATION_AND_DELIVERABLES.md:109-126`；generated token 六类 fixture 与整数聚合见同文件 `:128-147`，固定语义见 `spec/target_variables.md:267-292`；stable ID 规则见 `spec/target_variables.md:679-713`；OCP 必须使用 positive first-step advantage 与 mean old log-prob（同文件 `:772-800`）。上游的 final rollout 边界在 `Latent-GRPO/verl-0.4.x/verl/trainer/ppo/ray_trainer.py:1109-1212`，actor 还会在 update 内清零 overlong advantage（`Latent-GRPO/verl-0.4.x/verl/workers/actor/dp_actor.py:551-562`），所以测试不得只覆盖 driver 的 pre-mutation Tensor。

### 2.3 Support、one-sided/FlipGrad 与 credit

| ID | Purpose / level | Fixture | Exact command | Expected result | Evidence | Blocking gate |
|---|---|---|---|---|---|---|
| `T-SUPPORT-001` | strict same trajectory/position/K alignment；unit | keyed rollout/pre-update Top-K rows, reorder, missing/extra/K mismatch | `python -m pytest tests/unit/test_support.py::test_strict_alignment_or_unavailable -q` | 只按 step/group/trajectory/latent_position 对齐；length/K/order 不能核对时 whole family unavailable；不截断/广播/补 forward | U | G1/G3 |
| `T-SUPPORT-002` | retention/top1 与 position weighting；unit | trajectories with 1 and 3 positions, set overlaps | `python -m pytest tests/unit/test_support.py::test_retention_top1_and_position_weighting -q` | retention=交集/K；top1=rollout top1 是否在 pre-update set（不是两侧 top1 相等）；按 4 positions 加权而非 trajectory mean 平均 | U | G1 |
| `T-SUPPORT-003` | trajectory selection/timepoint；unit/integration | OCP winner + non-correct ties/overlong | `python -m pytest tests/unit/test_support.py::test_selection_and_preupdate_timepoint -q` | correct 复用同一内存 OCP winner；non-correct 排除 overlong、masked mean old log-prob、tie 最小 stable ID；phase 为 pre-update，无 checkpoint_step | U | G1 |
| `T-SUPPORT-004` | no extra forward/no mutation/no token count；integration/runtime | instrumented forward counter and full state snapshot | `python -m pytest tests/integration/test_support_no_mutation.py -q` | Support 复用 old-log-prob forward；forward count不增加；参数/optimizer/scheduler/.grad/RNG/steps/token counters 完全不变 | I-CPU，最终 R-1GPU | G3 |
| `T-DELTA-001` | surrogate margin/trigger truth table；unit | hand tensors for scores/logp/advantage including zero | `python -m pytest tests/unit/test_probe.py::test_surrogate_margin_and_flipgrad_truth_table -q` | delta=perturbed score-component logp；trigger=(adv<=0)&(delta<0)；noise offset/perturbed score 不冒充 delta | U | G1 |
| `T-DELTA-002` | 7 metrics、p05、independent counts；unit | deltas `[-2,-.01,0,.01,3]`, masks, empty | `python -m pytest tests/unit/test_probe.py::test_onesided_metrics_counts_quantile_and_empty -q` | mean/std/p05/min/negative/near-zero 按同 delta mask；FlipGrad 用独立 count；分位算法/version 固定；empty unavailable | U | G1 |
| `T-FLIP-001` | FlipGrad forward value/gradient sign；unit + GPU runtime | tiny differentiable component tensors | `python -m pytest tests/runtime/test_flipgrad_semantics.py -m gpu -q` | with/without flip forward 值相同；trigger 与非 trigger 的 autograd/finite-difference 符号符合上游真实公式；latent branch 确实生效 | R-1GPU | G3 |
| `T-CREDIT-001` | default-off schema and zero autograd；unit | autograd spy that raises if invoked | `python -m pytest tests/unit/test_probe.py::test_credit_disabled_schema_without_autograd -q` | credit 字段存在且 null；family available=false/reason=`disabled_by_config`；record/onesided 仍可用；autograd call count=0 | U | G1 |
| `T-CREDIT-002` | `u_i/q_i` sign/normalization/concentration；unit | analytical tiny loss/component logp | `python -m pytest tests/unit/test_credit.py::test_credit_sign_q_top1_effective_k -q` | `u_i=-dL/dlogp_i`；q 非负且在非零梯度时和≈1；top1/effective-K golden；concentration count 独立 | U | G1（启用 credit 前） |
| `T-CREDIT-003` | Spearman ties/constant/zero gradient/alignment；unit | tied vectors、constant weights、zero grads、known signs | `python -m pytest tests/unit/test_credit.py::test_spearman_degeneracy_and_alignment -q` | tie 规则固定；constant -> null + `degenerate_constant_vector`；zero grad -> null + `zero_gradient`；不伪造 alignment；独立 counts | U | G1（启用 credit 前） |
| `T-CREDIT-004` | 单次 autograd、所有分组复用、`.grad` 不污染；integration/runtime | autograd call counter, grouped tiny probe | `python -m pytest tests/runtime/test_credit_probe_gpu.py -m gpu -q` | 每 probe 最多一次受控 `autograd.grad`；trajectory/position 分组只做 mask；parameter `.grad` bitwise/None-state 不变；不保留 graph/full gradient | R-1GPU | G3（credit 默认仍关闭；通过后方可实验启用） |
| `T-SEMANTICS-001` | logging-off与Stage1/2-on首个可比step等价；runtime | same initial checkpoint/config/seed/data | `python -m pytest tests/runtime/test_training_semantics.py::test_logging_on_off_first_update_equivalence -m gpu -q` | forward/backward/optimizer/sampling调用数相同；loss/reward/update在预注册容差内；无法bitwise时报告随机源，禁止直接称无影响 | R-1GPU | G3 |
| `T-SEMANTICS-002` | Support schedule与write failure不篡改训练；runtime | paired Support on/off non-trigger step + injected metrics disk failure | `python -m pytest tests/runtime/test_training_semantics.py::test_support_schedule_and_write_failure_semantics -m gpu -q` | 非Support step训练等价；日志故障按配置降级或失败但不改batch/optimizer/RNG、不标completed、不伪造commit | R-1GPU | G3 |

依据：Support 对齐和不可静默截断要求见 `spec/05_VALIDATION_AND_DELIVERABLES.md:159-173` 与 `spec/target_variables.md:928-1000`；pre-update Top-K 必须来自已有 old-policy forward，禁止额外 forward（`spec/target_variables.md:844-870`）。上游返回的 pre-update Top-K 目前不是 response-ready shape、K 还硬编码为 10（`Latent-GRPO/verl-0.4.x/verl/workers/actor/dp_actor.py:320-343`），dynamic batch 只恢复 log-prob 顺序（同文件 `:441-473`）。Delta/FlipGrad 源路径为 `Latent-GRPO/verl-0.4.x/verl/utils/torch_functional.py:143-195`；credit 默认关闭、只允许一次局部 autograd 且不得写正常 `.grad`（`spec/target_variables.md:1233-1282`）。

### 2.4 存储、validator、resume 与 probe no-mutation

| ID | Purpose / level | Fixture | Exact command | Expected result | Evidence | Blocking gate |
|---|---|---|---|---|---|---|
| `T-STORAGE-001` | JSON atomic replace；unit fault injection | fail before/after fsync and rename | `python -m pytest tests/unit/test_storage.py::test_json_atomic_replace_fault_matrix -q` | 任一时刻旧文件或完整新文件可读；无半 JSON；NaN 写 null+availability | U | G1 |
| `T-STORAGE-002` | Parquet commit protocol；unit | typed rows/list fields; injected fail at write/close/read-check/rename/manifest/checkpoint | `python -m pytest tests/unit/test_storage.py::test_parquet_commit_crash_matrix -q` | 只在 readable/schema/row count 校验与 rename 后有效；`.tmp` 无效；manifest/writer checkpoint 不越过未提交 part | U | G1 |
| `T-STORAGE-003` | schema mismatch/list/null；unit | wrong dtype/missing column/list K mismatch/unavailable value non-null | `python -m pytest tests/unit/test_storage.py::test_schema_list_and_availability_validation -q` | 全部损坏 fixture 拒绝；合法 top-K lists 与 null+reason roundtrip | U | G1 |
| `T-STORAGE-004` | manifest rebuild/quarantine；unit | committed parts with absent/corrupt manifest、unknown file、future row | `python -m pytest tests/unit/test_storage.py::test_manifest_rebuild_and_quarantine -q` | 从可读 committed parts 重建；unknown/future 不删除，隔离并报告；part sequence 单调不复用 | U | G1 |
| `T-STORAGE-005` | duplicate PK；unit | same-batch、across-part、resume duplicate, same payload duplicate | `python -m pytest tests/unit/test_storage.py::test_duplicate_primary_keys_are_fatal -q` | 所有重复默认停止写入，绝不覆盖或默默幂等；run status 非 completed | U | G1 |
| `T-VALIDATOR-001` | 完整 validator positive/negative；unit CLI | valid golden run + one corrupted fixture per rule | `python -m pytest tests/unit/test_output_validator.py -q` | valid fixture exit0并生成三 validation JSON；每种损坏非零，检查 schema/PK/step/count/rate/std/availability/class identity/generation/top-K/probe tokens/manifest | U | G1/G3 |
| `T-NEGATIVE-001` | 禁止字段与临时 Tensor 不落盘；static+unit | every schema and sample part | `python -m pytest tests/unit/test_forbidden_fields.py -q` | `eval_metrics`,`train/gradient_norm`,`train/throughput`,每-step onesided/credit、group incorrect/invalid、Support checkpoint_step/current_topk_ids 等全部不存在；full logits/hidden/grad/graph/top-K training cache 不持久化 | S+U | G1 |
| `T-RESUME-001` | storage-only resume append；integration | committed steps 1-2 + checkpoint/writer sidecar | `python -m pytest tests/integration/test_resume.py::test_resume_append_sidecar_and_keys -q` | new parts 继续编号；旧 part byte/hash不变；`is_resume_run=true`,`resume_from_step=2`；无 future/duplicate keys；config/schema/upstream mismatch 拒绝 | I-CPU | G3 |
| `T-RESUME-002` | 真实 2→4 对连续 4 对照；single GPU runtime | same initial model/data/seed; output roots `continuous4` and `resume2to4` | `python -m pytest tests/runtime/test_resume_gpu.py::test_two_then_resume_four_matches_continuous_four -m gpu -q` | 见 §4：step/optimizer/data order/IDs/RNG/part/PK/最终 state 对照通过；validator 两边 exit0 | R-1GPU | G3 |
| `T-PROBE-SAFE-001` | probe success no-mutation；single GPU runtime | checkpoint with non-None and None grads; fixed probe | `python -m pytest tests/runtime/test_probe_no_mutation.py::test_success_restores_full_training_state -m gpu -q` | §3 全部 before/after hash相同；probe flags 正确；token counters不变 | R-1GPU | G3 |
| `T-PROBE-SAFE-002` | probe exception no-mutation；single GPU runtime | exception injected after RNG consumption, forward, credit grad and before write | `python -m pytest tests/runtime/test_probe_no_mutation.py::test_failure_restores_state_and_writes_unavailable -m gpu -q` | `finally` 恢复全部状态；无伪造 metrics；family/probe failure reason 稳定；run training state可继续 | R-1GPU | G3 |

依据：atomic part 的严格提交顺序、crash 恢复和 run status 见 `spec/03_METRICS_STORAGE_CONTRACT.md:264-301`；resume 规则见同文件 `:441-456`；validator 全检查项见同文件 `:460-498`。作者 checkpoint 当前不含 optimizer_step/config/schema/upstream commit，latest tracker 非原子，driver/SGLang RNG 不完整（`work_reports/agent_a_repo_audit.md:275-291`），所以 sidecar 与 no-mutation 测试是阻塞门禁，不能仅验证 model `state_dict`。

### 2.5 真实 smoke、性能、覆盖与独立审查

| ID | Purpose / level | Fixture | Exact command | Expected result | Evidence | Blocking gate |
|---|---|---|---|---|---|---|
| `T-SMOKE-1GPU-001` | 单卡真实 2-step E2E；runtime | target Linux/CUDA, local 1B latent model/data cache | `CUDA_VISIBLE_DEVICES=0 python train_latent_grpo.py --config configs/smoke.yaml --max-steps 2 --output-root /tmp/latent-grpo-validation/smoke-1gpu` | rollout/reward/advantage/update/checkpoint/Stage1/2/eval，配置时一次 Support/one-sided；至少一次 successful optimizer update；无 silent unavailable；run status completed | R-1GPU | G3 |
| `T-SMOKE-1GPU-002` | 单卡输出 validator | preceding output | `python scripts/validate_outputs.py /tmp/latent-grpo-validation/smoke-1gpu` | exit0；三份 validation JSON；target coverage artifact存在 | R-1GPU | G3 |
| `T-RANK-3GPU-001` | 三卡 single Ray driver/resource ownership | target 3×≈46GB | `CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 train_latent_grpo.py --config configs/3gpu-low.yaml --max-steps 0 --dry-run --output-root /tmp/latent-grpo-validation/3gpu-rank` | 3 wrapper ranks协调但只一 Ray coordinator；Ray可见3 GPU；三 worker device mapping；唯一 writer；无训练更新 | R-3GPU | G4 |
| `T-SMOKE-3GPU-001` | 3gpu-low 2-step E2E | target 3×≈46GB, 1B local cache | `CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 train_latent_grpo.py --config configs/3gpu-low.yaml --max-steps 2 --output-root /tmp/latent-grpo-validation/3gpu-low` | 三 worker初始化；真实 latent rollout与至少一次 update；worker充分统计正确；rank0-only writer；checkpoint可加载；PK无重复；记录 peak memory/time | R-3GPU | G5 |
| `T-SMOKE-3GPU-002` | 三卡输出 validator | preceding output | `python scripts/validate_outputs.py /tmp/latent-grpo-validation/3gpu-low` | exit0；aggregation_worker_count/owner、parts、keys、token counts一致 | R-3GPU | G5 |
| `T-HIGH-3GPU-001` | 7B/high chain 1-step | target 3×≈46GB, local 7B latent model | `CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 train_latent_grpo.py --config configs/3gpu-high-smoke.yaml --max-steps 1 --output-root /tmp/latent-grpo-validation/3gpu-high-smoke` | 完整链或可复现 OOM 诊断；不得静默换模型、量化、CPU offload或原 profile 参数；无论成功都不称性能/论文结果 | R-3GPU | G6（仅 low 已通过后） |
| `T-PERF-001` | logging off vs Stage1/2；runtime paired sampling | same checkpoint/config/data, warmup≥2, measured steps≥5 | `python -m pytest tests/runtime/test_metrics_performance.py::test_logging_disabled_vs_stage12 -m gpu -q` | 记录每样本 step time、compute/write time、allocated/reserved、queue peak、part bytes/rows；报告 median/p95和样本数；不作未实测绝对承诺 | P | G5 |
| `T-PERF-002` | Support/probe/credit sampling | fixed checkpoint and configured trigger | `python -m pytest tests/runtime/test_metrics_performance.py::test_support_and_probe_modes -m gpu -q` | 分别比较 probe off、checkpoint触发但probe off、forward-only、Delta、credit（若启用）及上限；记录 extra time/cache/peak memory；credit不每step执行 | P | G5；credit启用另需 safety tests |
| `T-COVERAGE-001` | target variables canonical coverage | parser inventory from target spec + RTM/schema | `python -m pytest tests/unit/test_target_coverage.py -q` | 动态、静态、memory-only、availability/count、negative requirements 全覆盖；生成 `validation/target_variable_coverage.json`; `missing_fields=[]`；credit/diagnostic disabled 不算 missing | S+U | G7 |
| `T-REVIEW-001` | independent reviewer | frozen commit, RTM, outputs, all test evidence | `python -m pytest tests/unit/test_claim_evidence_consistency.py -q` | 自动拒绝“未实测三卡却称通过”、synthetic升级、缺命令/exit code/config hash 等声明 | U | G7 |
| `T-REVIEW-002` | 独立 agent review | full implementation + reports, reviewer不编辑实现 | `test -f work_reports/agent_f_review.md && ! rg -n '^status: (blocker|major)$|\| *(blocker|major) *\|' work_reports/agent_f_review.md` | reviewer 报告存在且 blocker/major 均有修复+复测证据；若格式不同由结构化 review validator 判定，不用文本扫描替代正式解析 | IR | G7 |

单卡与三卡 smoke 的必验内容来自 `spec/05_VALIDATION_AND_DELIVERABLES.md:257-305`。性能必须分别测 logging disabled、Stage1/2、Support、one-sided 和 credit（若启用），并记录时间/显存/队列/part规模（同文件 `:363-395`）。覆盖报告必须 `missing_fields=[]`（同文件 `:399-434`）；最终 reviewer 必须查漏项、Classic-GRPO误用、mask/count/时点、probe状态污染、resume损坏与复现夸大（`spec/04_AGENT_ORCHESTRATION.md:199-228`）。

## 3. Probe / Support no-mutation 的精确快照协议

只比较 Python 对象 identity 或 `state_dict()` 字典相等不充分。`T-PROBE-SAFE-*` 与 `T-SUPPORT-004` 在操作前后都要生成 canonical snapshot；快照只保存 hash/元数据，不能把完整参数、梯度或 RNG 长期写进 metrics 表。

| 状态 | Canonical snapshot | 断言 |
|---|---|---|
| model parameters/buffers | 按 fully-qualified name 排序；hash(name,dtype,shape,layout,detached contiguous CPU bytes)；FSDP 用受控 local/full state 方案并记录类型 | 每项 hash 与集合完全相等；training/eval mode也恢复 |
| optimizer | canonical recursive hash of param groups（排除不稳定 object id，保留参数名映射）+ tensor/scalar state | hash完全相等；state key集合不变；无隐式 step |
| scheduler | canonical serialized `state_dict` + current LR per param group | hash/LR完全相等 |
| `.grad` | 每参数记录 `None` marker；非 None 记录 dtype/shape/layout/bytes；sparse grad先 canonical coalesce | `None` 不能变 zero tensor；已有 grad bitwise相同；credit 不积累 parameter grad |
| Python RNG | `pickle.dumps(random.getstate())` 的 hash | 相同 |
| NumPy RNG | legacy global state + runner持有的每个 `Generator.bit_generator.state` canonical hash | 相同；不能只测 global singleton |
| Torch CPU RNG | `torch.get_rng_state()` bytes hash | 相同 |
| CUDA RNG | `torch.cuda.get_rng_state_all()` 每 visible device独立 hash | device数与每卡 hash相同；异常路径也恢复 |
| rollout/SGLang/sampler RNG | 上游显式 generator/server sampling state或独立 seed/context metadata | 能导出的状态相同；若接口缺失，probe安全测试失败/blocked，不能写成功 |
| counters | `global_step`,`optimizer_step`,`cumulative_rollout_tokens`,`cumulative_train_samples`,`cumulative_gpu_hours` | 全部相同；probe token不进入长度/token count |
| writer state | committed part set、manifest hash、last committed step | 只允许新增 probe 专属合法 part；训练表、writer checkpoint不推进 |

执行顺序固定：`snapshot_before -> enter independent RNG context -> probe/support -> synchronize CUDA -> collect bounded detached stats -> restore in finally -> synchronize -> snapshot_after -> compare -> only then commit probe rows`。如果 compare 失败，probe row 不得伪造数值；写 family unavailable 或令 probe 失败，并保留 no-mutation diff。规范要求比较参数、optimizer、scheduler、正常 `.grad`、Python/NumPy/CPU/CUDA RNG、两个 step 和 cumulative tokens（`spec/05_VALIDATION_AND_DELIVERABLES.md:334-359`），target contract 还要求 probe 使用独立 RNG、恢复训练 RNG、绝不 optimizer step（`spec/target_variables.md:1099-1128`）。

## 4. Resume 2→4 与连续 4-step 对照协议

`T-RESUME-002` 不以“能 load checkpoint”作为通过。测试准备同一个 immutable 初始模型/cache、同一个数据 manifest、seed、profile/config hash和上游 commit，运行两个隔离目录：

```bash
CUDA_VISIBLE_DEVICES=0 python train_latent_grpo.py --config configs/smoke.yaml --seed 1234 --max-steps 4 --output-root /tmp/latent-grpo-validation/continuous4
CUDA_VISIBLE_DEVICES=0 python train_latent_grpo.py --config configs/smoke.yaml --seed 1234 --max-steps 2 --output-root /tmp/latent-grpo-validation/resume2to4
CUDA_VISIBLE_DEVICES=0 python train_latent_grpo.py --config configs/smoke.yaml --seed 1234 --max-steps 4 --resume-from /tmp/latent-grpo-validation/resume2to4/smoke/seed_1234/checkpoints/step_00000002 --output-root /tmp/latent-grpo-validation/resume2to4
python scripts/validate_outputs.py /tmp/latent-grpo-validation/continuous4/smoke/seed_1234
python scripts/validate_outputs.py /tmp/latent-grpo-validation/resume2to4/smoke/seed_1234
```

通过条件：

1. resumed run 从 checkpoint 的 `global_step=2`、真实累计 `optimizer_step` 和 scheduler 恢复；最终恰到 4，而不是再跑 4 step。
2. step 1–2 已提交 parts 的文件 hash/manifest entry 不变；新 part sequence 单调递增；无 `.tmp` 被计入、无重复 part/PK、无“未来记录”混入。
3. resumed 动态行 `is_resume_run=true`,`resume_from_step=2`；checkpoint/eval/probe 的 `checkpoint_step` 与执行 `global_step` 不混淆。
4. DataLoader/sampler顺序、final rollout stable group/trajectory IDs、generation seeds、OCP winner ID、generated-token integer sums在连续与resume的可比 step 3–4一致。
5. Python、NumPy、Torch CPU、每卡 CUDA、Ray worker和SGLang generation RNG都恢复。任何不可导出的上游随机状态必须令“deterministic resume equivalence”保持 blocked，而不是降低成只比 loss。
6. 若后端承诺 deterministic，step 3–4 raw facts、最终 model/optimizer/scheduler hashes应精确一致；若已证实 GPU kernel非确定性，则先固定 deterministic mode，仍无法 bitwise时只允许使用预注册容差比较 loss/reward/update norm，并在报告列出随机源、绝对/相对误差、原始 hashes。不得事后扩大容差。
7. 两个输出 validator 均 exit 0；run status、last committed global/optimizer step和last checkpoint一致。

规范明确要求 2 step→checkpoint→resume 到4并与连续4对照（`spec/05_VALIDATION_AND_DELIVERABLES.md:309-330`）。上游每 rank checkpoint虽已有部分 RNG，但 driver/SGLang state和 optimizer_step 缺口仍在（`Latent-GRPO/verl-0.4.x/verl/utils/checkpoint/fsdp_checkpoint_manager.py:92-128,162-185`; `work_reports/agent_a_repo_audit.md:275-290`），所以这些不能从断言中删除。

## 5. Storage crash / schema / duplicate 的故障注入点

Parquet part 测试至少在以下阶段逐一抛异常并重启 writer：临时文件创建后、写一半、close 前、close 后、readback 前/后、rename 前/后、manifest temp 写入前/后/rename 后、writer checkpoint 前/后。期望状态严格遵循：`.tmp` 无效；已 rename 且可读 part 有效；manifest缺失可扫描重建；writer checkpoint只能落后或等于已提交 part，绝不能领先（`spec/03_METRICS_STORAGE_CONTRACT.md:264-280`）。

重复测试必须覆盖：同 RecordBatch 内重复、两个待 flush batch重复、跨 committed parts重复、resume后重复、相同PK相同payload、相同PK不同payload、eval/probe旧 checkpoint重跑。所有冲突默认停止并报告；同 payload也不能静默覆盖（`spec/03_METRICS_STORAGE_CONTRACT.md:227-260,441-456`）。

schema 测试必须覆盖：缺列、多列、错误 nullable、int溢出、float NaN JSON、list IDs/probs长度不等、K不等、负 count、rate越界、std负、availability=false但值非 null/reason为空、definition version/hash改变后resume。任何 schema/definition 不兼容必须拒绝，不能 cast/reshape/truncate/broadcast。

## 6. Negative requirements 与 no-extra-compute 证明

canonical 禁止字段清单必须直接从 `spec/target_variables.md:50-84` 建 inventory，不靠人工摘录；Support 特有禁止列来自同文件 `:1056-1064`。validator同时扫描 schema manifest、每个 Parquet physical schema、definition JSON和debug输出。完整 logits/hidden/embedding/gradient/graph、训练全量 top-K/delta/u/q/mask 等不得持久化，唯一例外只有 eval existing clean top-K raw facts（`spec/target_variables.md:1459-1485`）。

此外使用 instrumentation counters 验证：

- Stage1/2 logging on/off 的 model forward、backward、optimizer-step、rollout sampling调用数一致；
- Support 只消费已有 pre-update old-log-prob forward，不新增 forward；
- Stage2 不落盘 `onesided/*`、不注册 gradient hook；
- credit default-off 的 `autograd.grad/backward` 增量为0；开启 credit 每次 probe最多1次局部 autograd，所有分组复用；
- probe/eval generation调用不进入 training token/length counters；
- worker不写权威文件，rank0 writer queue有界，queue满时 core metrics 不 silent drop。

这些断言对应“不得额外每-step full forward/credit、不得长期图、不得隐式 optimizer 或 logger随机采样”的性能与语义要求（`spec/02_SYSTEM_AND_IMPLEMENTATION_CONTRACT.md:388-431`）。

## 7. 性能采样协议

性能不是单次 wall-clock 截图。`T-PERF-*` 使用同一目标机、同一 commit、同一预热 checkpoint、相同 prompt顺序/batch/seed，先预热至少2个不计样本 step，再记录至少5个可比 step；若 smoke配置无法提供5个样本，则增加专用短 benchmark run，但不得称正式训练。每个 mode 独立新输出目录，记录 GPU clocks/其他负载局限，不混用有 eval/checkpoint 的普通 step与无触发 step。

必须采集：step time、metrics compute/write、Support extra、probe extra、每卡 peak allocated/reserved、Support cache peak、probe peak、writer queue peak、part bytes、row count（`spec/05_VALIDATION_AND_DELIVERABLES.md:363-387`）。报告 raw samples、median、p95、相对 logging-off overhead、样本数；CUDA阶段前后同步，writer flush单独计时。Support/probe触发 step 与普通 step分别汇报。性能验收不设置未经实测的绝对承诺，但出现额外 full-vocabulary保存、credit每step、无界队列或异常放大就失败，而不是仅写 caveat（同文件 `:389-395`）。

## 8. Target coverage 与状态晋级

coverage generator 必须集合比较四类 inventory：

1. 所有动态/静态持久化字段与具体 availability/count 字段；
2. `run_config.json`、Support/Probe definition 字段；
3. memory-only logical intermediates，标记 `memory_only_not_persisted`；
4. negative requirements，绑定 `T-NEGATIVE-001`。

RTM 每个字段必须有 test ID，不允许 family 一行代替字面字段；`missing_fields=[]` 是硬条件。credit/gumbel默认关闭时也必须有 schema、availability和 reason，不能算缺失（`spec/05_VALIDATION_AND_DELIVERABLES.md:399-434`；`work_reports/agent_c_metrics_storage.md:260-273`）。

状态晋级规则：

- 本阶段及“只有测试代码未执行”：`planned`；
- 生产路径已存在但没通过相应测试：`implemented`；
- 通过 synthetic：只能在证据列写 `tested_with_synthetic_data`，不能自动把目标运行字段升级成目标机 verified；
- 运行时接口确实缺失且有稳定 reason：`unavailable_with_reason`；
- 只有对应 level 的真实命令、exit0、artifact 与 config hash齐全，才可 `verified`；三卡字段必须有 `R-3GPU`。

## 9. 最终 acceptance checklist（当前全部 planned）

- [ ] `planned` 作者仓库固定从 `./Latent-GRPO` 读取，commit/dirty status入证据。
- [ ] `planned` 非 Docker；VSCode Remote/Linux准确命令已进入 runbook。
- [ ] `planned` `train_latent_grpo.py` 可执行，全部 CLI 与三个 profile通过配置测试。
- [ ] `planned` compileall、pytest collection、unit、synthetic distributed、CPU integration实际 exit0。
- [ ] `planned` runtime snapshot完整、脱敏；FlashAttention latent branch、自定义 verl/SGLang来源和tensor shape实际 probe通过。
- [ ] `planned` 29 core、5 Gumbel diagnostics、raw facts、所有count/availability/versions在RTM无漏项。
- [ ] `planned` 禁止字段/全量Tensor均未出现在权威 schema/parts/debug artifacts。
- [ ] `planned` generated-token固定口径、stable trajectory ID、OCP、overlong overlap通过。
- [ ] `planned` Support strict alignment/timepoint/position weighting通过，无新增 forward/no mutation。
- [ ] `planned` one-sided/FlipGrad formula、counts、p05通过；Stage2不持久化 onesided。
- [ ] `planned` credit schema默认关闭且零 autograd；实验启用前符号/tie/constant/zero-gradient/一次autograd/.grad安全全通过。
- [ ] `planned` JSON/Parquet atomic、crash recovery、manifest rebuild、schema、duplicate、quarantine通过。
- [ ] `planned` 2→4 resume与连续4对照通过，包含所有 RNG、optimizer_step、IDs、parts/PK。
- [ ] `planned` probe成功与异常路径的参数/optimizer/scheduler/.grad/RNG/counter hashes全不变。
- [ ] `planned` 单卡2-step真实闭环与validator通过。
- [ ] `planned` 三卡rank-init/NCCL/Ray single-driver与3gpu-low 2-step真实闭环、checkpoint load、validator通过；否则明确“三卡未实测”。
- [ ] `planned` 3gpu-high-smoke只在low通过后执行，不冒充性能或论文复现；OOM不静默改profile。
- [ ] `planned` 性能按paired protocol采样并报告raw/median/p95/显存/队列/part规模。
- [ ] `planned` `target_variable_coverage.json` 的 `missing_fields=[]`。
- [ ] `planned` 独立 reviewer 完成，blocker/major修复并复测，最终报告只引用真实证据。

此清单无降低断言；它覆盖 `spec/05_VALIDATION_AND_DELIVERABLES.md:86-105,109-230,234-359,363-434,438-496` 的每项静态、unit、integration、resume、probe、performance、coverage、report与最终检查要求。

## 10. 下一阶段建议命令顺序

以下命令必须在实现产生相应文件后、并在用户授权的目标 Linux 环境分阶段执行。先不安装、不训练：

```bash
python -m compileall -q train_latent_grpo.py latent_grpo_runner scripts tests
python -m pytest --collect-only -q
python -m pytest tests/unit -q
python -m pytest tests/integration/test_synthetic_distributed.py tests/integration/test_launcher.py tests/integration/test_resume.py -q
python train_latent_grpo.py --config configs/smoke.yaml --dry-run --validate-config --output-root /tmp/latent-grpo-validation/dry-run
python scripts/inspect_environment.py --config configs/smoke.yaml --output-root /tmp/latent-grpo-validation/environment
```

目标机已有候选依赖时，再做只读/最小 runtime probe：

```bash
python -m pip check
CUDA_VISIBLE_DEVICES=0 python scripts/inspect_environment.py --config configs/smoke.yaml --tensor-probe --output-root /tmp/latent-grpo-validation/tensor-probe
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 scripts/probe_distributed.py --backend nccl --output-root /tmp/latent-grpo-validation/rank-init
```

只有 G0–G2 通过后执行单卡；只有 G3 通过后执行三卡：

```bash
CUDA_VISIBLE_DEVICES=0 python train_latent_grpo.py --config configs/smoke.yaml --max-steps 2 --output-root /tmp/latent-grpo-validation/smoke-1gpu
python scripts/validate_outputs.py /tmp/latent-grpo-validation/smoke-1gpu
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 train_latent_grpo.py --config configs/3gpu-low.yaml --max-steps 2 --output-root /tmp/latent-grpo-validation/3gpu-low
python scripts/validate_outputs.py /tmp/latent-grpo-validation/3gpu-low
```

最后运行 resume、no-mutation、performance、coverage 和 review tests。不得在这些门禁闭合前启动长训练；实际依赖安装命令应以目标机 probe 后批准的 requirements/constraints 为准，本报告不猜版本、不提供安装动作。

## 11. 本报告自审

- 覆盖了 spec/05 的 static、unit、dry-run、single/3GPU smoke、resume、probe safety、performance、target coverage、final report与独立 review。
- 明确区分 S/U/D/I-CPU/R-1GPU/R-3GPU/P/IR，synthetic distributed 明确不是三卡证据。
- probe 比较包含参数、optimizer、scheduler、`.grad`、Python/NumPy/CPU/CUDA及上游生成 RNG hashes，并覆盖异常路径。
- storage 覆盖 atomic crash matrix、schema、duplicate、manifest rebuild、future record/quarantine与validator nonzero。
- negative inventory直接绑定规范，不从现有实现猜测；所有测试状态严格为 `planned`。
- 本阶段未安装依赖、未执行本机 GPU、未运行训练，也未修改 `./Latent-GRPO/**`、`docs/**`、`spec/**` 或其他 `work_reports/**`。
