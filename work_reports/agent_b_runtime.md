# Agent B — Runtime 与分布式训练只读审计

## 0. 审计范围、方法与状态词

- 审计对象：外层项目根目录下的 `Latent-GRPO/` 作者仓库；未修改作者仓库、规范或其他报告。
- 允许的唯一写入：本文件。
- 禁止项执行情况：未安装依赖、未联网、未下载模型/数据、未启动训练或 GPU 进程。
- 证据版本：只读 `git -C Latent-GRPO rev-parse HEAD` 得到 `c0994fb781a2d180662bb522d8ff3e8638dcf56d`；提交时间 `2026-05-12T19:38:56+08:00`，主题 `fixed`。
- 工作树：`git status --porcelain=v1` 仅显示未跟踪的 `.DS_Store`、`.cairn/`、`AGENTS.md`、`CLAUDE.md`、`cairn/`；这些不是本次审计产生的改动。
- 状态词：
  - **confirmed**：由静态源码、配置或只读 Git 命令直接确认。
  - **inferred**：由已确认实现推导出的运行影响；尚未在目标机执行。
  - **runtime_probe_required**：必须在 Linux + 3×约 46 GB CUDA GPU 目标机确认。

本报告不声称目标硬件可运行；尤其没有进行单卡或三卡 smoke。

## 1. 执行摘要

1. **[confirmed] 作者训练入口不是 torchrun-native。** 两个正式脚本都以单个 `python3 -m verl.trainer.main_ppo` 启动 Hydra；`main_ppo` 随后创建本地 Ray，Ray 再创建 FSDP/SGLang workers。证据：`Latent-GRPO/Latent-GRPO-gsm8k-llama3.sh:8`、`Latent-GRPO/Latent-GRPO-math500-qwen.sh:8`、`Latent-GRPO/verl-0.4.x/verl/trainer/main_ppo.py:24-37`、`:44-58`。
2. **[inferred, blocker] 不能把作者命令直接包在 `torchrun --nproc_per_node=3` 下。** 三个 launcher rank 会各自执行 `ray.init()` 并各自提交完整训练；至少会发生重复训练、Ray 端口/资源争抢或资源池不足。`ray.init()` 没有 torchrun rank gate，且训练完整调用位于所有进程都会执行的 `main()` 中。证据：`Latent-GRPO/verl-0.4.x/verl/trainer/main_ppo.py:24-37`。
3. **[confirmed, blocker] 当前 SGLang FSDP rollout 有 DeviceMesh 维名不一致。** worker 建立的是 `mesh_dim_names=["dp", "infer_tp"]`，但 `SGLangRollout` 读取 `self._device_mesh_cpu["tp"]`；同一个 mesh 的 sharding manager 又读取 `"infer_tp"`。证据：`Latent-GRPO/verl-0.4.x/verl/workers/fsdp_workers.py:389-396`、`Latent-GRPO/verl-0.4.x/verl/workers/rollout/sglang_rollout/sglang_rollout.py:329-340`、`:358-364`、`:688-695`、`Latent-GRPO/verl-0.4.x/verl/workers/sharding_manager/fsdp_sglang.py:78-79`、`:136-169`。按 PyTorch DeviceMesh 语义，这应在 rollout 初始化/首次通信时失败；仍需目标环境 import/smoke 给出实际异常。
4. **[confirmed, blocker] 顶层依赖清单无法视为一致 lock。** 顶层固定 `tensordict==0.10.0`，而 vendored VERL 要求 `tensordict<=0.6.2`；本地 SGLang metadata 要求 `sgl-kernel==0.1.0`，顶层和运行时代码要求 `0.1.1`；顶层同时固定 `torch==2.6.0` 与 `torchaudio==2.8.0`。证据：`Latent-GRPO/requirements.txt:168-189`、`Latent-GRPO/verl-0.4.x/setup.py:26-57`、`Latent-GRPO/sglang_latent_reasoning_pkg/python/pyproject.toml:47-57`、`Latent-GRPO/verl-0.4.x/verl/workers/rollout/sglang_rollout/sglang_rollout.py:113-125`。
5. **[confirmed] 三卡 prime world size 本身可用于 FSDP + rollout TP=1，但作者 batch 不兼容。** 两个脚本均为 8 GPU；`real_train_batch_size=train_batch_size×rollout.n` 必须被 GPU 数整除。低难度为 `64×8=512`，高难度为 `32×8=256`，都不能被 3 整除。证据：两脚本 `:12`、`:20-21`、`:53`、`:65-66`，以及 `Latent-GRPO/verl-0.4.x/verl/trainer/ppo/ray_trainer.py:440-457`。
6. **[confirmed] 上游只有 `global_steps`，没有可靠的 `optimizer_step`。** 一次外层 step 可执行多个 PPO mini-batch optimizer steps；非有限梯度会跳过参数更新，但 scheduler 和 global step 仍推进。证据：`Latent-GRPO/verl-0.4.x/verl/workers/actor/dp_actor.py:379-393`、`:496-520`、`:595-613`、`Latent-GRPO/verl-0.4.x/verl/workers/fsdp_workers.py:579-605`、`Latent-GRPO/verl-0.4.x/verl/trainer/ppo/ray_trainer.py:1000-1004`、`:1312-1322`、`:1365-1370`。
7. **[confirmed] checkpoint 恢复 actor model/optimizer/scheduler、worker RNG 和 dataloader state，但不满足任务契约的完整 resume。** 缺少 `optimizer_step`、profile/config hash、schema version、upstream commit；SGLang engine 的采样 RNG 没有保存；checkpoint shard 文件嵌入 world size，因此 8 卡 checkpoint 不能直接由 3 卡 FSDP loader 读取。证据见第 7 节。

## 2. 真实 entrypoint → config parser → worker → trainer 调用链

### 2.1 作者支持的 shell 表面

仓库顶层只有两个训练 shell：

- 低难度：`Latent-GRPO/Latent-GRPO-gsm8k-llama3.sh`
- 高难度：`Latent-GRPO/Latent-GRPO-math500-qwen.sh`

两者都：

- 通过环境变量解析 `DATA_DIR`、`MODEL_PATH`、`OUTPUT_DIR`、`GPUS`；证据：两个脚本 `:3-6`。
- 通过 Hydra dotlist 将所有实验参数覆盖到 `ppo_trainer.yaml`；证据：两个脚本 `:8-72`。
- 默认 8 GPU、单节点；证据：两个脚本 `:65-66`。
- 使用 SGLang、rollout TP=1、BF16、latent + noisy top-K + one-sided Gumbel；证据：两个脚本 `:35-47`。

没有作者提供的顶层 `torchrun` RL 脚本。vendored VERL 中存在 SFT 的 torchrun examples，但它们不是 Latent-GRPO PPO 入口。

### 2.2 精确调用链

```text
Latent-GRPO-*.sh
  └─ python3 -m verl.trainer.main_ppo + Hydra overrides
     └─ @hydra.main(config_path="config", config_name="ppo_trainer") main(config)
        └─ run_ppo(config)
           ├─ ray.init(local cluster)
           ├─ TaskRunner.remote()
           └─ ray.get(TaskRunner.run(config))
              ├─ OmegaConf.resolve(config)
              ├─ copy_to_local(model path)
              ├─ hf_tokenizer / hf_processor
              ├─ select FSDP/FSDP2 or Megatron worker classes
              ├─ construct one Ray global resource pool
              ├─ load reward managers
              ├─ create RLHFDataset + sampler
              └─ RayPPOTrainer(...)
                 ├─ _validate_config()
                 ├─ init_workers()
                 │  ├─ Ray placement groups
                 │  ├─ colocated actor/rollout[/ref] worker class
                 │  ├─ ActorRolloutRefWorker.init_model()
                 │  │  ├─ HF model + FSDP/FSDP2 actor
                 │  │  ├─ SGLangRollout
                 │  │  └─ FSDPSGLangShardingManager
                 │  └─ optional ref/RM/critic init
                 └─ fit()
                    ├─ rollout generate_sequences RPC
                    ├─ reward on driver / optional RM worker
                    ├─ old log-prob RPC
                    ├─ driver advantage
                    ├─ actor update RPC
                    ├─ eval/checkpoint
                    └─ global_steps += 1
```

逐段证据：

- shell → Python module：两脚本 `:8`。
- Hydra parser → `run_ppo`：`Latent-GRPO/verl-0.4.x/verl/trainer/main_ppo.py:24-37`。
- Ray TaskRunner / config resolve / model tokenizer：同文件 `:44-65`。
- FSDP/Megatron worker class selection：同文件 `:75-93`。
- Ray resource pool和 role mapping：同文件 `:95-108`。
- reward/dataset/sampler/trainer：同文件 `:110-154`。
- dataset factory/sampler seed：同文件 `:157-211`。
- trainer config checks / dataloader：`Latent-GRPO/verl-0.4.x/verl/trainer/ppo/ray_trainer.py:437-563`、`:565-628`。
- worker init和角色共置：同文件 `:786-858`。
- actor/rollout worker构建：`Latent-GRPO/verl-0.4.x/verl/workers/fsdp_workers.py:94-164`、`:389-482`、`:484-577`。
- fit 数据流：`Latent-GRPO/verl-0.4.x/verl/trainer/ppo/ray_trainer.py:966-1079`、`:1214-1322`、`:1339-1373`。

### 2.3 配置树与解析边界

- **[confirmed]** Hydra 默认配置固定为 `Latent-GRPO/verl-0.4.x/verl/trainer/config/ppo_trainer.yaml`；入口不读取作者项目专属 YAML。证据：`main_ppo.py:24-26`。
- **[confirmed]** 核心树为 `data`、`actor_rollout_ref.{model,actor,ref,rollout}`、`critic`、`reward_model`、`algorithm`、`trainer`、`ray_init`。证据：`ppo_trainer.yaml:1-291`。
- **[confirmed]** 默认 rollout 是 vLLM/TP=2，但正式脚本覆盖为 SGLang/TP=1。证据：`ppo_trainer.yaml:100-145`，两脚本 `:35-36`。
- **[confirmed]** config validation 不是轻量 dry-run：TaskRunner 先解析/加载 tokenizer、创建 dataset，随后构造 trainer 时才 `_validate_config()`。证据：`main_ppo.py:57-65`、`:130-152`，`ray_trainer.py:437-438`。
- **[inferred]** 新用户入口必须在触发 HF/Parquet 读取和 Ray workers 前自行完成路径、硬件、整除性、latent token id、offline cache 与 profile validation；不能把上游 `_validate_config()` 当 `--validate-config` 实现。

## 3. 依赖与非 Docker Linux 边界

### 3.1 已确认依赖表面

- Python：README 建议 3.11.13；PyTorch 2.6.0；证据：`Latent-GRPO/README.md:102-118`。
- 两个 editable 包：本地定制 SGLang 和 vendored VERL；证据：`Latent-GRPO/requirements.txt:1-4`。
- 关键 runtime：Hydra 1.3.2、Ray 2.49.2、torch 2.6.0、transformers 4.51.1、flash-attn 2.7.3、flashinfer 0.2.3、sgl-kernel 0.1.1、NCCL wheel 2.21.5；证据：`requirements.txt:45-46`、`:59-60`、`:110`、`:156`、`:168`、`:184`、`:192`。
- SGLang metadata 固定 torch 2.6.0/torchvision 0.21.0/transformers 4.51.1；证据：`Latent-GRPO/sglang_latent_reasoning_pkg/python/pyproject.toml:19-57`。
- VERL setup 基础依赖含 Ray、tensordict、pyarrow，SGLang extra 指向 upstream `sglang==0.4.6.post5`；证据：`Latent-GRPO/verl-0.4.x/setup.py:26-57`。

### 3.2 静态冲突

| 结论 | 状态 | 证据 | 影响 |
|---|---|---|---|
| `tensordict==0.10.0` vs `<=0.6.2` | confirmed blocker | `requirements.txt:179`; `verl-0.4.x/setup.py:40,51,53` | pip resolver可能直接拒绝，或未受支持 API 组合运行失败 |
| local SGLang version `0.4.6.post1` vs VERL extra `post5` | confirmed | `sglang.../pyproject.toml:5-8`; `verl-0.4.x/setup.py:52-57` | 必须确保实际 import 是本地定制包，不能让 upstream wheel 覆盖 |
| `sgl-kernel==0.1.0` metadata vs 顶层/runtime `0.1.1` | confirmed | `sglang.../pyproject.toml:47-53`; `requirements.txt:168`; `sglang_rollout.py:120-125` | 安装元数据与运行时断言不一致 |
| `flashinfer==0.2.3` vs flashinfer backend 断言最低 `0.2.5` | confirmed, conditional | `requirements.txt:46`; `sglang_rollout.py:113-119`; eval 强制 backend 见 `eval/eval_high_tasks_sglang.py:480-495` | 若 backend 解析为 flashinfer，训练/评估初始化失败 |
| torch 2.6.0 vs torchaudio 2.8.0 | confirmed suspicious | `requirements.txt:184,187` | ABI/依赖解析需 lock probe；训练本身未见 torchaudio 必需 |

### 3.3 非 Docker 可行性

- **[confirmed]** README 提供 conda + pip editable 的非 Docker 路径；证据：`README.md:102-130`。
- **[inferred]** 可在 Linux 非 Docker 运行，但 `flash_attn`、CUDA extension、SGL kernel 与 NVIDIA driver/CUDA ABI 是主要环境风险；README 已说明 flash-attn 可能编译或符号失败。证据：`README.md:111-120`。
- **[runtime_probe_required]** 必须建立经过 resolver 验证的 constraints/lock，而不是直接宣称 `requirements.txt` 可安装。至少执行无安装的 resolver plan 或在隔离环境按 runbook 安装，并验证 `import torch, flash_attn, flashinfer, sglang, verl` 以及本地包 `__file__`/版本。

## 4. Ray / VERL / FSDP / SGLang / vLLM / DP / TP 拓扑

### 4.1 当前正式脚本的真实拓扑

- **[confirmed]** 单个 CPU Python driver 启动本地 Ray；训练 driver 实际在一个 `TaskRunner` Ray actor 中。证据：`main_ppo.py:29-46`。
- **[confirmed]** 一个全局 Ray pool 请求 `[n_gpus_per_node] * nnodes`；所有角色映射到该池。证据：`main_ppo.py:95-108`。
- **[confirmed]** Ray pool 每 GPU 一个 bundle，worker group world size 等于请求 GPU 数；Ray 向 worker 注入 `WORLD_SIZE/RANK/RAY_LOCAL_*`。证据：`Latent-GRPO/verl-0.4.x/verl/single_controller/ray/base.py:103-128`、`:309-363`。
- **[confirmed]** FSDP actor、SGLang rollout及可选 ref policy共置在同一 fused worker group，而不是独占多套 GPU。证据：`ray_trainer.py:795-842`。
- **[confirmed]** actor 默认 FSDP，正式脚本没有覆盖 actor strategy；low 的 ref 未启用，high 因 `use_kl_loss=True` 创建 ref policy且脚本令 ref 为 FSDP2。证据：`ppo_trainer.yaml:42-43`、两脚本 `:22,55-56`、`main_ppo.py:126-129`。
- **[confirmed]** GRPO 不创建 critic；证据：`ray_trainer.py:422-435`。
- **[confirmed]** rollout TP=1时三卡形状为 rollout DP=3、TP=1；FSDP world size=3。构造公式证据：`fsdp_workers.py:389-396`。
- **[confirmed]** SGLang rollout在每个 TP group首 rank启动一个 engine；TP=1意味着每个 GPU/DP rank一个完整 rollout engine/model replica。证据：`sglang_rollout.py:352-403`。
- **[confirmed]** 训练与 rollout混合引擎通过 sharding manager在进入 rollout时同步 FSDP权重、交换 RNG状态，退出时释放 SGLang memory并恢复 actor CUDA RNG。证据：`fsdp_sglang.py:92-169`。

### 4.2 vLLM 不是 Latent-GRPO 的等价替换

- **[confirmed]** vLLM路径返回 `responses/input_ids/rollout_log_probs/attention_mask/position_ids`，不返回 latent top-K ids/Gumbels。证据：`Latent-GRPO/verl-0.4.x/verl/workers/rollout/vllm_rollout/vllm_rollout.py:270-288`。
- **[confirmed]** actor old/current log-prob和update强制选择 `rollout_topk_ids`、`rollout_topk_gumbels`、`gumbel_temperature`。证据：`Latent-GRPO/verl-0.4.x/verl/workers/actor/dp_actor.py:417-427`、`:475-493`。
- **[confirmed]** 只有定制 SGLang rollout组装这些 tensor。证据：`sglang_rollout.py:699-720`。
- **[inferred]** 不得通过把 `rollout.name` 改回 vLLM解决三卡或依赖问题；这会丢失 Latent-GRPO所需数据并在 actor路径报缺 key，或退化为非目标算法。

### 4.3 三卡拓扑的边界

- **[confirmed]** FSDP world size=3和rollout TP=1在整数拓扑上成立；rollout TP必须整除world size。证据：`fsdp_workers.py:393-396`。
- **[inferred]** 若为了7B内存将rollout TP改为3，则只剩一个rollout replica；还要验证模型attention/KV heads能否三分、定制SGLang TP latent路径一致性和吞吐，不能静态承诺。
- **[inferred]** TP=1下7B BF16完整rollout model在每张46 GB卡上通常有机会容纳，但SGLang `mem_fraction_static=0.8`、FSDP shard、optimizer/offload、全词表Gumbel tensor会共同决定峰值；必须实测。高难度脚本证据：`Latent-GRPO/Latent-GRPO-math500-qwen.sh:23,29-35,52`。
- **[confirmed]** latent sampler先构造全词表 `log_softmax/softmax/sort/Gumbel`，再取top-K；并不是内存成本仅为K。证据：`Latent-GRPO/sglang_latent_reasoning_pkg/python/sglang/srt/layers/sampler.py:70-122`。
- **[runtime_probe_required]** 低/高配置分别测init、首次weight sync、rollout峰值、old-logprob forward、backward/optimizer peak；高配置OOM时只能显式缩短batch/response/latent/step，不能静默改算法、模型、量化或offload profile语义。

## 5. 三卡 batch 与 optimizer 拓扑风险

### 5.1 上游显式检查

- `real_train_batch_size = data.train_batch_size × rollout.n` 必须被最小并行batch（FSDP时为GPU总数）整除；证据：`ray_trainer.py:440-457`。
- actor worker将配置的 `ppo_mini_batch_size` 先乘 `rollout.n`，再对DP world size做整数地板除；证据：`fsdp_workers.py:143-155`。
- 上游只检查real train batch整除GPU数，没有检查 `ppo_mini_batch_size × n` 被world size整除；证据：`ray_trainer.py:455-524`。

### 5.2 作者参数在3卡上的结果

| profile来源 | train prompts | rollout n | trajectories | ppo mini prompts | `ppo_mini×n` | 三卡结果 |
|---|---:|---:|---:|---:|---:|---|
| low作者脚本 | 64 | 8 | 512 | 16 | 128 | 512不能被3整除，config validation失败；mini也不能被3整除 |
| high作者脚本 | 32 | 8 | 256 | 32 | 256 | 256不能被3整除，config validation失败；mini也不能被3整除 |

证据：两个脚本 `:12,20-21,53,65-66`。

- **[inferred, major]** 只把 `trainer.n_gpus_per_node=3` 改掉不够；必须联合选择 `train_batch_size`、`ppo_mini_batch_size`、`ppo_micro_batch_size_per_gpu`、`rollout.n`，至少满足：
  1. `train_batch_size × n` 被3整除；
  2. `ppo_mini_batch_size × n` 被3整除，避免worker地板除悄悄改变有效mini-batch；
  3. 每rank local trajectories能被归一化后的mini/micro合理切分；
  4. group完整性保持n条；动态过滤路径已有完整组断言，证据：`ray_trainer.py:1155-1199`。
- **[inferred]** 一个保留作者量级的候选仅用于设计起点：low可考虑train prompts=60、ppo mini prompts=15、n=8、micro=2（480/120=4个optimizer updates）；high可考虑30/30/8/micro=1（240/240=1个update）。这不是已验证配置，也不是论文复现；内存/吞吐必须probe后决定。

## 6. `global_step` 与 `optimizer_step`

### 6.1 上游 `global_steps`

- **[confirmed]** 新run设0，resume可覆盖；训练loop前先加到1。证据：`ray_trainer.py:984-1004`。
- **[confirmed]** 一个global step包含：rollout、reward、old log-prob、advantage、可选critic、actor update、eval/checkpoint、metric log；结束后 `global_steps += 1`。证据：`ray_trainer.py:1036-1079`、`:1217-1322`、`:1339-1370`。
- **[confirmed]** 动态group过滤不足时会继续消费/生成下一dataloader batch而不增加global step；证据：`ray_trainer.py:1109-1153`。
- **[confirmed]** checkpoint在step结束、global increment之前保存到 `global_step_<current>`；证据：`ray_trainer.py:1347-1369`。

因此可将上游 `global_steps`解释为“完成该次外层训练迭代所绑定的1-based step id”，但要注意首次validation使用step 0。

### 6.2 optimizer step并非一对一

- **[confirmed]** actor update按 `ppo_epochs × local mini-batches`调用 `_optimizer_step()`；每个mini内部由micro-batches累积梯度。证据：`dp_actor.py:496-520`、`:522-600`、`:610-613`。
- **[confirmed]** grad norm非有限时，参数update被跳过；证据：`dp_actor.py:379-393`。
- **[confirmed]** worker不返回“是否成功更新”，只返回grad norm metrics；scheduler无条件 `step()`。证据：`dp_actor.py:610-614`、`fsdp_workers.py:592-605`。
- **[confirmed]** driver无 `optimizer_step`字段/计数，且无论worker内部是否跳过，global step都会推进。证据：`ray_trainer.py:1312-1322`、`:1365-1370`。

**结论 [confirmed, contract gap]：**不能令 `optimizer_step == global_step`，也不能用scheduler state反推成功更新数。

**最小patch建议：**

1. 私有adapter点 `DataParallelPPOActor._optimizer_step()` 返回 `(grad_norm, did_step)`；文件：`verl-0.4.x/verl/workers/actor/dp_actor.py:379-393`。
2. `update_policy()`累计本次成功/跳过optimizer steps并通过worker metrics/meta返回；文件：同文件`:496-614`。
3. Ray driver在 `actor_output` 汇总后校验各FSDP rank计数一致，再增加持久 `optimizer_step`；文件：`verl-0.4.x/verl/trainer/ppo/ray_trainer.py:1312-1322`。
4. checkpoint metadata必须写入/恢复该计数，不能只加sidecar而不与模型checkpoint提交一致。

此patch改变可观测性和scheduler异常语义的记录，不应改变参数更新；仍需 logging-off等价测试。

## 7. Checkpoint、resume 与 RNG链

### 7.1 checkpoint内容

- **[confirmed]** driver目录结构为 `global_step_<N>/actor`、可选`critic`、`data.pt`及根目录tracker。证据：`ray_trainer.py:869-900`。
- **[confirmed]** FSDP actor每rank保存model shard、optimizer shard和extra shard；extra含scheduler和RNG。rank0另存model config、tokenizer/processor，可选full HF model。证据：`Latent-GRPO/verl-0.4.x/verl/utils/checkpoint/fsdp_checkpoint_manager.py:129-185`、`:187-250`。
- **[confirmed]** RNG extra包含该worker进程的Torch CPU、NumPy、Python random和当前device CUDA/NPU RNG。证据：`Latent-GRPO/verl-0.4.x/verl/utils/checkpoint/checkpoint_manager.py:108-132`。
- **[confirmed]** `data.pt`来自 `StatefulDataLoader.state_dict()`；random sampler有独立generator，默认seed为 `data.seed`或1。证据：`ray_trainer.py:891-895`、`main_ppo.py:190-209`。
- **[confirmed]** checkpoint配置默认强制含`model/optimizer/extra`。证据：`ppo_trainer.yaml:66-67`、`fsdp_checkpoint_manager.py:60-66`。

### 7.2 resume链

- **[confirmed]** 模式为disable/auto/resume_path；auto依赖 `latest_checkpointed_iteration.txt`，resume_path必须包含`global_step_`。证据：`ppo_trainer.yaml:273-276`、`ray_trainer.py:902-929`。
- **[confirmed]** global step通过目录名解析，不在extra state中；随后加载actor/critic shard和dataloader state。证据：`ray_trainer.py:930-951`。
- **[confirmed]** FSDP loader恢复model、optimizer、RNG、scheduler。证据：`fsdp_checkpoint_manager.py:76-127`。
- **[confirmed]** shard名含 `world_size_<N>_rank_<R>`，loader用当前world size拼路径。证据：`fsdp_checkpoint_manager.py:92-103`、`:176-185`。
- **[inferred, major]** 8卡作者checkpoint不能直接作为3卡训练resume checkpoint；当前3卡loader会寻找world_size_3文件。需要先合并为HF权重再作为新的初始化run，或提供经过验证的reshard converter；optimizer/scheduler trajectory无法靠HF merge保留。
- **[confirmed]** HDFS load明确未实现；证据：`ray_trainer.py:906-914`。
- **[confirmed, durability gap]** tracker用普通覆写，不是原子replace；证据：`ray_trainer.py:897-900`。worker shard先barrier后tracker写，能降低但不能消除driver崩溃/partial filesystem风险。

### 7.3 不满足新契约的元数据

上游checkpoint没有可确认的：

- `optimizer_step`；
- profile/config hash；
- metrics schema version；
- upstream commit；
- run id / resume lineage；
- 原子完成marker。

证据是实际extra仅含`lr_scheduler`和`rng`：`fsdp_checkpoint_manager.py:162-185`；global step仅在folder/tracker：`ray_trainer.py:869-900`。

### 7.4 RNG恢复缺口

1. **[confirmed]** SGLang sharding manager维护独立 `torch_random_states`和`gen_random_states`，在训练/生成间切换；证据：`fsdp_sglang.py:81-90`、`:113-133`。
2. **[confirmed]** `gen_random_states`只是内存属性，FSDP checkpoint extra不保存它；证据对照：`fsdp_sglang.py:81-90`与`fsdp_checkpoint_manager.py:172-175`。
3. **[confirmed]** 真正Gumbel采样发生在SGLang engine worker的CUDA上下文中；证据：`sglang.../layers/sampler.py:70-122`。
4. **[confirmed]** SGLang server seed若未配置会由Python random生成，engine worker初始化后设置seed；训练创建engine时没有传 `random_seed`。证据：`sglang.../server_args.py:83,230-231`、`sglang.../managers/tp_worker.py:138-145`、`sglang_rollout.py:374-401`。
5. **[confirmed]** resume顺序是先 `trainer.init_workers()`（构建新engine），再在`fit()`中load checkpoint；证据：`main_ppo.py:153-154`、`ray_trainer.py:984-987`。

**结论 [inferred, major]：**当前resume可恢复actor worker训练RNG和dataloader，但不能保证恢复SGLang engine latent/Gumbel rollout RNG，因新engine已用新seed初始化，且没有engine RNG序列化API接入checkpoint。连续4步与2+resume+2步的trajectory很可能不同。

其他driver RNG：

- group uid用`uuid.uuid4()`创建；证据：`ray_trainer.py:1064-1069`。它不是稳定trajectory id，也不由上述worker RNG checkpoint恢复。
- driver/TaskRunner自身Python/NumPy全局状态未见checkpoint；FSDP extra保存的是各训练worker状态。

**需要的patch/adapter：**给每个rollout DP replica配置确定性base seed并实现可读取/设置的SGLang engine RNG或按 `(run_seed, global_step, stable_trajectory_id)` 的请求级seed；把状态与checkpoint同提交。该点触及私有定制SGLang接口，不能仅在外层wrapper猜测已恢复。

## 8. 模型、数据与cache链

### 8.1 模型

- **[confirmed]** 作者要求Latent-SFT初始化模型，不能用普通base model；证据：`README.md:132-138`。
- **[confirmed]** 推荐低/高模型ID分别为 `DJCheng/LLaMA3.2-1B-Instruct-Latent-SFT-Top10`、`DJCheng/Qwen2.5-Math-7B-Latent-SFT-4k-Top10`；证据：`README.md:140-143`。
- **[confirmed]** TaskRunner先`copy_to_local(model.path)`；对非HDFS路径函数直接原样返回，随后HF tokenizer/model APIs自行处理本地路径或Hub ID。证据：`main_ppo.py:57-65`、`Latent-GRPO/verl-0.4.x/verl/utils/fs.py:190-251`。
- **[confirmed]** 每个actor/rollout/ref worker再次调用`copy_to_local`和HF加载；证据：`fsdp_workers.py:411-420`、`:456-464`、`:508-524`、`:550-567`。
- **[inferred, gap]** 对HF ID没有作者实现的“rank0下载→barrier→其余rank只读本地完整snapshot”逻辑。HF cache自身可能加锁，但不等于契约要求的显式cache准备、完整性与offline错误报告。
- **[confirmed]** latent end token id在训练脚本中硬编码（LLaMA 524、Qwen 522），README要求对tokenizer验证；证据：两脚本`:43-46`，`README.md:187-194`。

**adapter建议：**外层rank0/driver用配置化cache root解析完整snapshot，验证config/tokenizer/weight文件和`</think>`首token ID，再把绝对local snapshot path传给所有Ray workers；不得记录token。

### 8.2 数据

- **[confirmed]** 低/高脚本分别读取指定Parquet；证据：两脚本`:10-16`，数据说明`README.md:58-100`。
- **[confirmed]** `RLHFDataset`的默认远端cache是`~/.cache/verl/rlhf`，本地Parquet不复制；之后使用HF datasets加载Parquet并可并行过滤overlong prompts。证据：`Latent-GRPO/verl-0.4.x/verl/utils/dataset/rl_dataset.py:90-148`。
- **[confirmed]** 数据必须含prompt/reward_model/extra_info等schema；预处理输出证据：`Latent-GRPO/data_preprocess_code/gsm8k_aug.py:22-52`、`Latent-GRPO/data_preprocess_code/math500_aug.py:22-50`。
- **[confirmed]** dataloader `drop_last=True`，batch size为`gen_batch_size`或train batch；证据：`ray_trainer.py:585-592`。
- **[confirmed]** resume会重新下载/读取原始Parquet（若dataset未被序列化）并加载StatefulDataLoader state；证据：`rl_dataset.py:150-157`、`ray_trainer.py:944-951`。
- **[inferred]** 新入口需要显式`dataset_cache_dir`、offline validation、Parquet schema/row count/fingerprint；不能把模型大文件或dataset cache放进metrics output目录。

## 9. eval与checkpoint-eval边界

- **[confirmed]** 上游trainer在`val_before_train`和`test_freq`处执行reward-based validation，并把汇总metric记在当前global step；证据：`ray_trainer.py:989-997`、`:1339-1345`。
- **[confirmed]** 独立eval脚本不是trainer checkpoint hook；它们各GPU启动TP=1 SGLang完整模型replica，以multiprocessing分片数据。证据：`Latent-GRPO/eval/eval_low_tasks_sglang.py:320-345`、`:406-432`、`:451-480`，高任务对应`eval/eval_high_tasks_sglang.py:470-496`、`:548-569`、`:607-618`。
- **[confirmed]** 三张卡可作为三个独立eval replicas，不要求模型头被3整除，因为每个engine固定`tp_size=1`。证据：上述engine构造行。
- **[confirmed, bug]** `--add_noise_gumbel_softmax`没有`type`或boolean action；README示例传入字符串`False`。argparse会保留非空字符串，容易被下游当truthy。证据：`README.md:238-244`、`eval/eval_low_tasks_sglang.py:577-585`、`eval/eval_high_tasks_sglang.py:832-840`。
- **[inferred, reproducibility gap]** eval在engine创建后才调用父worker的`random.seed/torch.manual_seed`，但SGLang engine已生成随机server seed并可能运行在子进程；因此`base_seed`是否真正控制Gumbel/采样必须probe。证据：low `:320-364`，high `:470-510`，以及SGLang seed证据见7.4。
- **[inferred]** 新checkpoint probe/eval必须区分`checkpoint_step`与执行时`global_step`；上游当前只有logger step，无该双重context。

## 10. 可复用公共接口、私有adapter点与最小patch

### 10.1 可直接复用

| 接口 | 用途 | 证据 |
|---|---|---|
| `verl.trainer.main_ppo.run_ppo(config)` | 已解析OmegaConf的程序化入口 | `main_ppo.py:29-42` |
| Hydra `ppo_trainer.yaml` + dotlist | 形成上游完整config | `main_ppo.py:24-26`; `ppo_trainer.yaml:1-291` |
| `create_rl_dataset` / `create_rl_sampler` | 数据与可恢复sampler | `main_ppo.py:157-211` |
| `RayPPOTrainer`、`init_workers()`、`fit()` | 现有单controller数据流 | `ray_trainer.py:369-438,786-868,966-1373` |
| registered worker RPCs | rollout/logprob/update/save/load | `fsdp_workers.py:578-619,641-713,752-803` |
| `FSDPCheckpointManager` | model/optim/scheduler/worker RNG shard | `fsdp_checkpoint_manager.py:76-185` |
| `RLHFDataset` | 作者数据schema与prompt处理 | `rl_dataset.py:90-148,181-270` |

### 10.2 必须由adapter处理但可不改算法

1. **torchrun rank gate：**outer rank 0唯一调用`run_ppo`，rank 1/2不得启动Ray；非0 rank只参与CPU-side状态/错误协调并等待torch elastic结束。不要让outer ranks初始化CUDA/NCCL占卡。依据：`main_ppo.py:29-37`。
2. **配置转换与早期validation：**argparse/profile→Hydra compose/OmegaConf；在任何Hub/dataset/Ray动作前检查三卡、整除、路径、cache、latent token ID、profile标签。
3. **rank0 cache preparation：**模型/数据只准备一次，再把本地snapshot/Parquet path交给Ray workers。
4. **错误传播：**`ray.get`能把TaskRunner失败抛回outer rank0，但上游没有run status/final flush；wrapper需让torchrun整体非零退出并写失败状态。证据：`main_ppo.py:36-37,153-154`。
5. **3卡profile必须明确device-adapted，不是paper exact。** 作者脚本为8卡且batch需重配。

### 10.3 最小上游patch候选（按优先级）

1. **P0：修正SGLang DeviceMesh维名。** 统一`SGLangRollout`的`"tp"`与worker/sharding manager的`"infer_tp"`，覆盖所有引用：`sglang_rollout.py:330-331,362,693-694,1008-1009`。这是运行正确性patch。
2. **P0：真实optimizer_step。** 按第6.2节返回每次did_step和累计数；这是logging/state patch，需等价测试。
3. **P0：checkpoint metadata/RNG。** 保存optimizer_step、配置/commit/schema hash，并接入SGLang engine seed/RNG状态；这是resume正确性patch。
4. **P1：trainer callback/factory。** `TaskRunner`当前硬编码`RayPPOTrainer`并直接`fit()`（`main_ppo.py:138-154`）；为指标事件、step context和checkpoint sidecar提供显式callback/factory，优于脆弱monkey patch。
5. **P1：eval boolean parser。** 将`--add_noise_gumbel_softmax`改为正确的boolean optional action/显式解析；证据见第9节。

必须把任何上游修改放入任务规定的patch与变更说明；本阶段未实施。

## 11. 三卡用户入口推荐架构边界

推荐的最小控制流（设计建议，未实现）：

```text
torchrun rank 0/1/2 start train_latent_grpo.py
  ├─ all ranks: parse profile + non-CUDA environment probe
  ├─ rank 0: validate config/cache; prepare local model/data; write run snapshot
  ├─ rank 1/2: wait; never call upstream main/run_ppo and never own writer
  └─ rank 0: Hydra compose → patched run_ppo(config)
       └─ Ray owns the real 3-GPU FSDP/SGLang worker world
```

为何不推荐在三个torchrun rank上直接运行upstream：见第1、2节。为何不推荐重写torchrun-native trainer：现有Latent-GRPO语义跨定制SGLang、Ray RPC、FSDP hybrid sharding manager和driver advantage；绕开Ray等于重写高风险核心，而非最小adapter。

分布式指标的权威driver应位于`TaskRunner/RayPPOTrainer`单controller，而不是三个outer torchrun rank。worker只返回sufficient statistics；outer rank0最终写盘。现有`reduce_metrics`仅用于作者scalar logger，不能直接满足目标契约的sum/sum_sq/count语义；该部分由metrics owner定义，本报告不独立定义指标。

## 12. 必做 runtime probes

按阻断顺序：

1. **环境/ABI probe：**Linux、Python、torch/CUDA/driver/cuDNN/NCCL、3张GPU名称/显存/compute capability/BF16、磁盘；验证local SGLang/VERL实际import路径及flash-attn/flashinfer/sgl-kernel import。
2. **依赖resolver probe：**用隔离环境验证一致constraints；重点解决tensordict、SGLang post版本、sgl-kernel、flashinfer、torchaudio冲突。
3. **纯配置probe：**三profile在不加载模型/数据前通过整除、TP、micro/mini、max model len、latent token id和offline cache校验。
4. **torchrun/Ray ownership probe：**`torchrun --nproc_per_node=3`只能出现一个Ray head、一个TaskRunner、一套3-worker placement group；rank1/2无GPU context；rank0异常让全job非零退出。
5. **SGLang mesh probe：**修patch前复现维名异常，patch后三rank完成engine init、weight sync、一次最短latent rollout。
6. **低模型1-step分阶段峰值：**init、rollout、old-logprob、backward、optimizer、checkpoint；确认GPU映射和每step optimizer update数。
7. **高模型1-step OOM probe：**TP=1、offload配置下分别记录allocated/reserved/engine static memory；若尝试TP=3，验证模型结构与定制latent TP正确性。
8. **checkpoint save/load probe：**3卡保存后原world size恢复；验证model/optimizer/scheduler/dataloader/global/optimizer step；显式拒绝8→3直接resume。
9. **RNG连续性probe：**连续4步 vs 2步+resume+2步，对比dataloader样本、SGLang top-K/Gumbels、response、loss和参数；预期当前上游不能通过rollout RNG一致性。
10. **eval reproducibility probe：**相同base seed重复运行；确认engine真正接收固定seed；验证CLI `False`不被当True。
11. **cache/offline probe：**有完整cache时所有Ray workers不联网；缺文件时rank0给精确错误，其他rank不会并发下载或悬挂。
12. **故障传播probe：**任一Ray worker OOM/异常时其余worker停止、outer torchrun退出、checkpoint/metrics已提交部分可读、run status不是completed。

## 13. 主智能体可直接采用的决策清单

- 选择“**torchrun外壳 + rank0唯一Ray controller**”，不要三份Ray训练。
- 在任何GPU动作前做独立config/environment/cache validator。
- 三卡profile联合重配train/mini/micro/n，且明确不是论文8卡复现。
- 保持SGLang latent rollout；vLLM现有路径不是算法等价替换。
- 先修SGLang DeviceMesh维名，再谈三卡smoke。
- 新增真实optimizer_step，不能从global step或scheduler推断。
- checkpoint扩展metadata并解决SGLang engine RNG；现有resume只“部分可恢复”。
- 8卡sharded checkpoint不能直接3卡resume；HF merge仅能作为新run初始化。
- 模型/数据cache由rank0显式准备，传local path给Ray workers。
- 对高模型TP=1/3、全词表Gumbel开销和46 GB峰值只在runtime probe后定案。

