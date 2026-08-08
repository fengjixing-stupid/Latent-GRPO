# Agent D：三卡约 46 GB GPU 运行方案设计（只读阶段）

审计日期：2026-08-02  
作者仓库：`./Latent-GRPO`，审计 commit `c0994fb781a2d180662bb522d8ff3e8638dcf56d`  
目标表面：Linux + VSCode Remote + 非 Docker + 3×约 46 GB NVIDIA GPU  
本报告状态：静态设计；未安装依赖、未下载模型/数据、未初始化 CUDA/Ray/NCCL、未执行训练  
唯一写入：`work_reports/agent_d_3gpu_plan.md`

## 1. 结论先行

1. 推荐采用 **`torchrun` 外壳 + rank 0 唯一 Ray controller + Ray 内部 3-GPU FSDP/SGLang worker world**。三个外层 `torchrun` 进程不是三个训练 rank；rank 1/2 只是无 CUDA context 的监督进程。否则三个进程都会执行 `ray.init()`、创建 `TaskRunner` 并提交完整训练。上游没有 torchrun rank gate：`Latent-GRPO/verl-0.4.x/verl/trainer/main_ppo.py:24-37`。
2. 真实训练所有权必须留在上游：单一 `TaskRunner/RayPPOTrainer` 负责数据、reward、advantage 和控制流；Ray 创建三个共置 actor/rollout workers；worker 内建立 FSDP/NCCL；rollout 必须继续用作者定制 SGLang。证据：`main_ppo.py:95-154`、`ray_trainer.py:786-858,966-972`、`fsdp_workers.py:100-122,389-476`。
3. 两个候选 profile 都从 **TP=1、SP=1** 起步。此时 actor FSDP world size=3，rollout mesh 为 DP=3×infer-TP=1，每卡一个 SGLang engine。`world_size % infer_tp == 0` 是上游明示约束：`fsdp_workers.py:389-396`；SGLang 每个 TP group 只有首 rank 启 engine：`sglang_rollout.py:352-403`。
4. `3gpu-low` 的保守候选是 6 prompts × 4 rollouts、mini prompts=3、actor micro=2；`3gpu-high-smoke` 是 3 prompts × 4 rollouts、mini prompts=3、actor micro=1。两者所有已知整除式均闭合，见第 6 节。它们只是 probe 起点，不是显存成功承诺。
5. 高难 7B profile 明确使用作者高难路径中的 gradient checkpointing 和 actor parameter/optimizer offload，并额外将 reference parameter offload 作为该设备适配 profile 的显式属性；不量化、不换模型、不切 vLLM。显存、CPU RAM、PCIe 开销与 checkpoint 峰值都必须在目标机分阶段测量。
6. 上游没有独立 `max_latent_length` 运行时旋钮；只存在总 `max_new_tokens=response_length` 和 latent-end token 切换。因此配置可以先声明目标 latent cap，但在最小 adapter/patch 真正执行和验证前，不能声称该 cap 生效。证据：`sglang_rollout.py:408-425,640-665`、`schedule_batch.py:682-715,728-771`；全仓无 `max_latent_length` 实现。
7. 本方案明确 **不是论文复现**。作者发布脚本是 8 GPU：low 为 `64×8` trajectories，高难为 `32×8`，均不能被 3 整除；证据：`Latent-GRPO-gsm8k-llama3.sh:12,20-21,53,65-66`、`Latent-GRPO-math500-qwen.sh:12,20-21,53,65-66` 及上游校验 `ray_trainer.py:440-457`。

## 2. 上游运行事实与设计边界

### 2.1 真实启动和数据控制流

```text
作者 shell
  -> python -m verl.trainer.main_ppo
  -> Hydra ppo_trainer config
  -> run_ppo(): ray.init()
  -> TaskRunner Ray actor
  -> RayPPOTrainer.init_workers()
  -> Ray placement group / three GPU workers
  -> rollout -> reward -> old log-prob -> advantage -> actor update -> checkpoint
```

证据：

- Hydra 与 `run_ppo`：`Latent-GRPO/verl-0.4.x/verl/trainer/main_ppo.py:24-37`。
- resource pool 要求 `n_gpus_per_node × nnodes`：同文件 `:95-108`。
- `TaskRunner` 构造 dataset、trainer、workers 并调用 `fit()`：同文件 `:130-154`。
- 一个 GPU 一个 Ray bundle：`Latent-GRPO/verl-0.4.x/verl/single_controller/ray/base.py:103-128`。
- Ray 为 worker 注入自己的 `WORLD_SIZE/RANK/RAY_LOCAL_*`：同文件 `:309-363`。
- controller 上做轻量 advantage，worker RPC 做计算：`ray_trainer.py:966-972,1214-1322`。

### 2.2 不可替换的 Latent-GRPO 路径

- 作者两个训练脚本都显式 `rollout.name=sglang` 并启用 latent、Top-K、Gumbel 和 one-sided noise：两个 shell 各 `:35-47`。
- vLLM 路径不返回 actor 所需的 `rollout_topk_ids` / `rollout_topk_gumbels`；SGLang 路径才组装它们：`verl/workers/rollout/vllm_rollout/vllm_rollout.py:270-288`、`sglang_rollout.py:688-710`、`dp_actor.py:425-427,487-493`。
- 因此 OOM 或依赖问题不得通过 `rollout.name=vllm` 规避；这会改变或破坏 Latent-GRPO，而不是等价降级。

### 2.3 一个需更正/复核的静态判断

`fsdp_workers.py` 为 sharding manager 创建 `mesh_dim_names=["dp", "infer_tp"]`（`:389-396`），但当前调用 `SGLangRollout(...)` **没有传入该 mesh**（`:456-464`）；`SGLangRollout` 因此自行创建 `["dp", "tp", "pp"]` mesh（`sglang_rollout.py:300-330`），而 manager 使用自己的 `infer_tp` mesh（`fsdp_sglang.py:78-79,135-170`）。这不是单凭维名不同即可断言的初始化 blocker；两套 mesh 的 group/通信一致性仍须三卡 runtime probe。不得在未复现异常前先修改维名。

## 3. Launcher / topology 方案比较

| 方案 | 控制与 GPU 拓扑 | 优点 | 风险/缺点 | 决策 |
|---|---|---|---|---|
| A. `torchrun` 外壳，rank 0 唯一启动 Ray | outer world=3；仅 outer rank 0 运行 adapter/Ray；Ray world=3 真正占 GPU | 满足规范公开命令；保留作者 Ray+FSDP+SGLang；可利用 torch elastic 统一失败退出 | 需要严格 rank gate、状态文件、env 隔离；两个外层进程闲置；不能误初始化 outer NCCL | **推荐主入口** |
| B. 单进程 `python train_latent_grpo.py` 启 Ray | 一个普通 Python driver；Ray world=3 | 与作者启动语义最接近；故障面最小；适合定位 Ray/SGLang 问题 | 不满足规范指定的主要 `torchrun --nproc_per_node=3` 用户表面 | **保留为诊断/回退入口**，不得替代三卡 torchrun 验收 |
| C. 三个 torchrun rank 直接做 FSDP、移除 Ray | torchrun world=3 同时是训练 world | 理论上少一层 orchestrator | 需要重写 Ray RPC、driver advantage、hybrid sharding、SGLang weight sync、checkpoint；极易改变算法 | **拒绝** |

禁止方案：在三个 outer rank 上直接调用 `verl.trainer.main_ppo.run_ppo()`。`run_ppo()` 每次都会 `ray.init()` 并创建一个 `TaskRunner`（`main_ppo.py:29-37`），结果是三套训练而非一个三卡作业。

## 4. 推荐 torchrun envelope 协议

### 4.1 进程职责

```text
torchrun outer rank 0
  - 解析/验证 profile
  - 执行 runtime probe 与 cache 准备
  - 创建唯一输出 writer/control state
  - 清理 torchrun 分布式环境污染后启动一套 local Ray
  - 等待 TaskRunner；flush；写 completed/failed

torchrun outer rank 1/2
  - 只做无 CUDA 的最小参数一致性检查
  - 不 import 会初始化 CUDA 的 SGLang/kernel 模块
  - 不调用 torch.distributed.init_process_group
  - 不调用 ray.init，不下载，不写权威输出
  - 观察 run-scoped control state；与 rank 0 同结果退出

Ray controller / TaskRunner
  - 唯一训练控制器、数据/reward/advantage/step owner

Ray GPU worker 0/1/2
  - 唯一真实 FSDP/NCCL world
  - 每 worker 绑定一张唯一 GPU
  - 执行 actor、rollout、old/ref log-prob、update、checkpoint shard
```

### 4.2 避免“三套 Ray”的强制规则

1. 在 import upstream 或 CUDA 组件前读取 `LOCAL_RANK/RANK/WORLD_SIZE`；只有 outer rank 0 可进入 `run_ppo()`。
2. envelope 不建立 outer NCCL/Gloo process group；否则它会与 Ray worker world 形成第二套分布式状态，并可能持有 GPU/端口。
3. rank 0 在调用 Ray 前保存并从自身环境移除 outer 的 `RANK`、`WORLD_SIZE`、`LOCAL_RANK`、`LOCAL_WORLD_SIZE`、`MASTER_ADDR`、`MASTER_PORT` 等 torchrun rendezvous 变量；保留已验证的 `CUDA_VISIBLE_DEVICES=0,1,2`。Ray worker 的 rank/world/master 信息必须只由 Ray worker group 注入，证据：`ray/base.py:336-363`。
4. 使用 `TORCHELASTIC_RUN_ID + config hash + output run id` 派生独占 control namespace；状态文件仅由 outer rank 0 原子写入，防止读到旧 run 的完成标记。
5. control state 至少有 `initializing/cache_ready/ray_ready/running/flushing/completed/failed`、heartbeat、owner PID、error class、config hash。followers 对 `failed` 返回非零，对 `completed` 返回 0；状态陈旧或 owner 消失则返回非零。
6. rank 0 捕获 `ray.get(TaskRunner...)` 异常，先标记 failed、尽力 flush 已提交 metrics、写 traceback 指针，再 `ray.shutdown()` 并重新抛出；不可把失败 run 标 completed。上游 `ray.get` 是异常回传点：`main_ppo.py:36-37`。
7. 发现已有未授权 Ray address、同 output run 的活跃 owner、多个 TaskRunner、GPU bundle 数不等于 3 时立即停止；不得自动连接未知 Ray cluster。

## 5. FSDP / DP / TP / SP / Ray / SGLang ownership

| 层 | 3gpu-low | 3gpu-high-smoke | Owner 与证据 |
|---|---:|---:|---|
| outer torchrun world | 3 envelope processes | 3 envelope processes | 只做 launcher；不是训练 world |
| Ray controller | 1 | 1 | `run_ppo` 创建一个 `TaskRunner`：`main_ppo.py:29-46` |
| Ray GPU bundles/workers | 3 | 3 | pool `[3]`，每 bundle 1 GPU：`main_ppo.py:101-108`、`ray/base.py:103-125` |
| actor FSDP world | 3 | 3 | worker 内 init process group 和 device mesh：`fsdp_workers.py:100-120` |
| actor Ulysses SP | 1 | 1 | 默认/作者配置路径，配置定义：`ppo_trainer.yaml:63-65` |
| effective actor DP divisor | 3 (`world/SP`) | 3 (`world/SP`) | mini normalization：`fsdp_workers.py:143-155` |
| rollout TP | 1 | 1 | world 必须整除 TP：`fsdp_workers.py:393-396` |
| rollout DP replicas | 3 | 3 | `dp=world/infer_tp`：同上 |
| SGLang engines | 3×TP1 engine | 3×TP1 engine | 每个 TP group 首 rank启动 engine：`sglang_rollout.py:352-403` |
| critic | 无 | 无 | GRPO 不建 critic：`ray_trainer.py:422-435` |
| reference policy | 无（`use_kl_loss=false`） | 有，FSDP2，逻辑共置并显式 param offload | role 创建条件：`main_ppo.py:126-129`；共置 worker class：`ray_trainer.py:815-842` |
| 权威 metrics writer | TaskRunner/controller 单实例 | 同左 | worker 只返回充分统计；规范 `spec/02...:296-319` |

### 为什么默认不使用 TP=3 或 SP=3

- TP=3 会把 rollout DP 从 3 降为 1，三卡共同承载一个 engine。虽然 `3 % 3 == 0` 通过表层约束，但模型 attention/KV heads、作者 latent weighted embedding TP 路径、SGLang collectives 和吞吐均未实测；必须读取目标 snapshot 的真实 config 并做最小 latent rollout，不能静态承诺。
- SP=3 会令 actor effective DP divisor 从 3 变 1，并改变 remove-padding、位置恢复、batch 归一化与指标 mask 的通信路径。当前作者脚本没有启用；只有 TP=1/SP=1 基线通过后才可单独 probe。
- TP/SP 变化不是 OOM 时的静默 fallback；必须使用单独实验 profile 和独立等价性测试。

## 6. 候选 profile 与全部静态算术校验

### 6.1 统一符号与上游公式

- `W=3`：Ray/FSDP world size。
- `S=1`：Ulysses sequence parallel size。
- `D=W/S=3`：worker 中用于 mini normalization 的 divisor。
- `P=data.train_batch_size`：每个最终 outer step 的 prompt group 数；当前同步上游没有独立 rollout prompt batch，故新 `rollout_batch_size` 在该路径应与 `P` 相同。
- `n=actor_rollout_ref.rollout.n`：每 prompt trajectory 数。
- `T=P×n`：全局最终 trajectory 数。
- `M=actor.ppo_mini_batch_size`：配置层 prompt mini 数；worker 本地实际 mini 为 `Lmini=(M×n)/D`。
- `μ=actor.ppo_micro_batch_size_per_gpu`。

必须增加到新 config validator 的公式：

```text
T = P * n
T % W == 0                              # upstream 已检查
(M * n) % D == 0                        # upstream 未完整检查；避免 // 静默截断
L = T / D
Lmini = (M * n) / D
L % Lmini == 0                          # 完整 mini-batch
Lmini % μ == 0                          # upstream worker 检查
gradient_accumulation_steps = Lmini / μ
optimizer_attempts_per_global_step
  = ppo_epochs * L / Lmini
  = ppo_epochs * P / M
```

证据：上游只检查 `P×n` 被最小 batch 整除：`ray_trainer.py:440-457`；worker 先做 `M*=n` 再 `//=D`：`fsdp_workers.py:143-155`；actor 对本地 batch 按 mini/micro 切分并每 mini 尝试一次 optimizer step：`dp_actor.py:496-520,595-613`。因此必须显式禁止 `(M×n)%D != 0`，不能依赖整数地板除。

### 6.2 `3gpu-low` 保守候选（1B，非论文复现）

| 配置 | 候选值 | 说明 |
|---|---:|---|
| model | `DJCheng/LLaMA3.2-1B-Instruct-Latent-SFT-Top10` | 规范候选；必须先解析成本地完整 snapshot |
| P / rollout_batch_size | 6 / 6 prompts | 最终训练 group 数；dynamic filtering 可能额外生成被丢弃 batch |
| rollout `n` | 4 | 保留组内比较；低于作者 8 |
| PPO mini `M` | 3 prompts | P 可被 M 整分 |
| actor micro `μ` | 2 trajectories/GPU | 若 backward OOM，第一降级为 1 |
| rollout old-log-prob micro | 2 trajectories/GPU | 对应 upstream `rollout.log_prob_micro_batch_size_per_gpu`；不是 SGLang generation microbatch |
| PPO epochs | 1 | 上游默认：`ppo_trainer.yaml:63` |
| TP / SP | 1 / 1 | rollout DP=3，actor divisor=3 |
| dtype | BF16（仅硬件 probe 通过） | 不支持则停止该 profile，不静默伪装 |
| remove padding | true | 作者 low 使用；`Latent-GRPO-gsm8k-llama3.sh:19` |
| gradient checkpointing | false 起步 | 作者 low 为 false；OOM 后若启用，记录 profile deviation |
| actor param/optimizer offload | false / false | 作者 low；OOM 后只用新命名 offload profile |
| max prompt / response | 192 / 128 | 与作者 low 一致，先减少 batch/并发而不改长度语义 |
| max model len | 512 候选 | 满足 `>=192+128`；还须模型 config probe，约束见 `sglang_rollout.py:342-347` |
| max batched tokens | 512 候选 | SGLang 调度内存旋钮；吞吐/可启动性需 probe |
| desired max latent | 64（计划） | 上游当前不执行此 cap；实现并验证前标 `runtime_probe_required` |
| SGLang static memory fraction | 0.50 候选 | 作者 low 是 0.60；只作为起点，不承诺 engine 能加载 |
| support / checkpoint credit | 初始 off / off | rank-init 与训练闭环后分阶段开启；credit 始终默认 off |

静态验算：

```text
T = 6 * 4 = 24; 24 % 3 = 0
L = 24 / 3 = 8 trajectories per Ray/FSDP rank
M*n = 3*4 = 12; 12 % 3 = 0
Lmini = 12/3 = 4 trajectories per rank
L % Lmini = 8 % 4 = 0 -> 2 minis/rank/epoch
Lmini % μ = 4 % 2 = 0
gradient_accumulation_steps = 4/2 = 2
optimizer attempts/global step = 1 * 6/3 = 2
old-log-prob local batch 8 % micro 2 = 0
```

### 6.3 `3gpu-high-smoke` 保守候选（7B 链路验证，非论文复现）

| 配置 | 候选值 | 说明 |
|---|---:|---|
| model | `DJCheng/Qwen2.5-Math-7B-Latent-SFT-4k-Top10` | 规范候选；不是普通 Qwen base |
| P / rollout_batch_size | 3 / 3 prompts | 每 rank 约一个 prompt group 的起点 |
| rollout `n` | 4 | 低于作者 8；保留组内 reward variance 机会 |
| PPO mini `M` | 3 prompts | 一个 global mini |
| actor micro `μ` | 1 trajectory/GPU | 最小非零 micro |
| rollout old-log-prob micro | 1 trajectory/GPU | ref log-prob micro 也为 1 |
| PPO epochs | 1 | 只验证一条 update 链 |
| TP / SP | 1 / 1 | TP=3 仅单独 probe，不作为默认 fallback |
| dtype | BF16（仅 probe 通过） | 不降到未知混合 dtype |
| remove padding / grad checkpoint | true / true | 作者高难设置：`Latent-GRPO-math500-qwen.sh:19,29` |
| actor param/optimizer offload | true / true | 作者高难设置：同脚本 `:30-31` |
| reference policy | `use_kl_loss=true`, FSDP2, param offload=true | 保留作者高难 KL objective；ref offload 是本设备 profile 的显式偏差，必须测 CPU RAM/性能 |
| max prompt / response | 512 / 256 | 仅链路 smoke，显著低于作者 1024/4096 |
| max model len | 1024 候选 | `>=512+256`；必须读取真实 model context limit |
| max batched tokens | 1024 候选 | 初始限制调度规模；需 engine probe |
| desired max latent | 128（计划） | 当前上游没有独立执行点；未实现前不能声称受限 |
| SGLang static memory fraction | 0.50 候选 | 作者高难为 0.80；能否容纳 model+KV 必须实测 |
| max steps | 1，成功后 2 | 不用于性能/效果结论 |
| support / checkpoint probe / credit | 初始 off / 保存后单独 one-sided / off | 先闭环再做 no-mutation probe |

静态验算：

```text
T = 3 * 4 = 12; 12 % 3 = 0
L = 12 / 3 = 4 trajectories per Ray/FSDP rank
M*n = 3*4 = 12; 12 % 3 = 0
Lmini = 12/3 = 4 trajectories per rank
L % Lmini = 4 % 4 = 0 -> 1 mini/rank/epoch
Lmini % μ = 4 % 1 = 0
gradient_accumulation_steps = 4/1 = 4
optimizer attempts/global step = 1 * 3/3 = 1
old/ref-log-prob local batch 4 % micro 1 = 0
```

### 6.4 参数语义警告

1. target contract 的 `rollout_micro_batch_size_per_gpu` 不能误写为 SGLang 生成 microbatch。上游只明确暴露 `rollout.log_prob_micro_batch_size_per_gpu`（`ppo_trainer.yaml:122-125`）；SGLang生成并发由 engine 与 `max_num_batched_tokens/max_num_seqs` 控制。新 schema 应将二者分开命名/解释，无法一一映射时写明 availability，而不是猜一个数。
2. dynamic group filtering 会丢掉 zero-variance groups并继续生成，故实际 GPU rollout 计算量可能大于最终 `P×n`；逻辑见 `ray_trainer.py:1109-1211`。必须限制 `max_num_gen_batches` 并记录 retry/丢弃成本，但 target `train/generated_token_count` 仍只计最终训练 rollout trajectory。
3. `optimizer_attempts` 不是契约 `optimizer_step`。非有限 grad 会跳过 `optimizer.step()`（`dp_actor.py:379-393`）；必须 patch/adapter 回传 `did_step` 后才能累计成功更新数。
4. profile 的 latent cap 只有在 sampler/scheduler 明确执行、训练重放与 mask 一致、测试覆盖后才有效。仅在 YAML 写 `max_latent_length` 不算实现。

### 6.5 约 46 GB/卡的预算门（不是容量预测）

不得按“46 GB”字样直接承诺可用空间。对每卡实测总显存 `C_i`，先采用以下工程门：

```text
safety_headroom_i = max(2 GiB, 0.10 * C_i)
admitted_peak_i   = C_i - safety_headroom_i

若 C_i = 46 GiB（仅示例）：
safety_headroom >= 4.6 GiB
admitted_peak <= 41.4 GiB
```

`admitted_peak` 是 `nvidia-smi` 进程总占用与框架 `max_memory_reserved` 的联合停止阈值，不是给模型可随意分配的固定值。任何阶段超过 90% 实测卡容量、出现持续增长/碎片化、或一张卡比其余卡高出 10% 以上，都停止并诊断，不进入下一阶段。

候选 profile 的 `gpu_memory_utilization=0.50` 对 46 GiB 名义上对应约 23 GiB 的 SGLang static fraction，但其是否包含权重、KV cache及实现保留区必须实测；不能把剩余约18.4 GiB简单当成actor预算。hybrid engine在 rollout enter/exit时同步权重、释放SGLang memory并切换actor，实际峰值可能发生在交接而非稳态：`fsdp_sglang.py:92-133`。

每卡、每阶段必须记录以下分解；所有字节来自运行时，而不是模型参数量估算：

| 阶段 | 必须采样的显存项 | 预算判定 |
|---|---|---|
| Ray worker空载 | baseline process/context、其他进程占用 | baseline 已占用 > safety headroom 时先清理/停止 |
| actor/ref FSDP init | allocated/reserved、FSDP shard、optimizer init、ref init | 三卡均 `< admitted_peak`，卡间偏差≤10% |
| SGLang engine init | engine static pool、model weights、KV reservation | 单独记录，不以config fraction替代实测 |
| FSDP→SGLang weight sync | state_dict/gather/update瞬时峰值 | 必须低于admitted peak；这是model-load后首个高风险门 |
| rollout | prefill、decode、full-vocab Gumbel、KV峰值 | 按最长实际sequence和并发记录 |
| old/ref log-prob | actor/ref reload、remove-padding forward峰值 | 与rollout峰值分开，不做相加估算 |
| backward/optimizer | activation、grad、optimizer step峰值 | low/high分别验证micro和accumulation |
| checkpoint | GPU reserved + CPU RSS +磁盘临时量 | 必须确认rollout已release；同时满足host预算 |

high-smoke中7B BF16权重、FSDP shard、ref、optimizer、SGLang权重/KV、全词表Gumbel会在不同阶段交替/短时重叠；静态审计无法给出可信字节总和。因此 **high显存状态为 `blocked_pending_runtime_probe`**，而非“预计可放下”。

### 6.6 启动候选与回滚配置

| profile状态 | P | n | M | actor / old-log-prob micro | prompt / response | desired latent cap | 其他变化 | 用途 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| low 初始 | 6 | 4 | 3 | 2 / 2 | 192 / 128 | 64（待实现） | TP1/SP1，无offload | 1-step候选 |
| low 回滚 | 3 | 2 | 3 | 1 / 1 | 192 / 64 | 32（待实现） | max batched tokens 256；可启gradient checkpointing但必须记录偏差 | rollout/backward OOM后的最小三卡闭环 |
| high 初始 | 3 | 4 | 3 | 1 / 1（ref=1） | 512 / 256 | 128（待实现） | TP1/SP1，actor param+optim及ref param offload | 7B 1-step候选 |
| high 回滚 | 3 | 2 | 3 | 1 / 1（ref=1） | 384 / 128 | 64（待实现） | max batched tokens 512，其他算法开关不变 | rollout/backward OOM后的最小链路 |

两份回滚的算术相同：`T=3×2=6`，每rank `L=2`，`Lmini=(3×2)/3=2`，micro=1，accumulation=2，每global step一个optimizer attempt。若 high 在 actor/ref/SGLang **初始化或weight sync** 阶段OOM，回滚batch/length未必有效，应直接停止并标 blocked；不能继续靠缩batch反复试。

## 7. 分阶段 runtime probes（必须按顺序）

| 阶段 | 必须观测 | 通过标准 | 失败处置 |
|---|---|---|---|
| P0 环境/ABI | Python、torch local version、driver/runtime、cuDNN/NCCL、3 GPU型号/显存/CC/BF16、拓扑/P2P、系统 RAM、磁盘、`/dev/shm`、文件句柄 | 三卡唯一可见；BF16和目标 wheel/kernel真实可运行；依赖来源是 vendored verl/SGLang | 任何 ABI/import/kernel 失败则不进入 launcher probe |
| P1 outer rank ownership | 3 outer PID、唯一 owner、followers 无 CUDA context、无 outer process group、control state/heartbeat | 只有 rank0进入 cache/Ray；followers不占 GPU、不写输出 | 出现 >1 owner/Ray controller 立即停止 |
| P2 Ray/FSDP rank init | 一个 Ray head、一个 TaskRunner、3 bundles、Ray ranks 0..2、每 worker唯一GPU、NCCL all-reduce/barrier | world=3、device映射无重复、collective无 hang | 任一 rank/init失败，全作业非零，保留 status/traceback |
| P3 model/cache load | model/tokenizer文件完整性、fork import `__file__`、latent-end token ID、actor/ref/FSDP init、SGLang engine init、每阶段 allocated/reserved/host RAM | 所有 worker只读同一 local snapshot；无网络；模型与tokenizer一致 | 缺文件仅 owner下载；错误不允许 workers各自重试 |
| P4 weight sync + rollout | sharding manager enter/update/release；最短 latent response；top-K ID/score shape、K、sentinel、latent→hard transition；peak memory | 定制 SGLang输出可被actor重放；无 vLLM fallback；三 DP replica均返回 | shape、通信、latent token不一致直接停止 |
| P5 reward/old log-prob/advantage | reward domain、group完整性、old log-prob、Optimal Correct Path、final advantage、动态过滤次数 | group count一致；最终 batch恰为P groups×n；无无限 retry | 达到生成批次上限仍不足则失败，不用伪造 group |
| P6 backward/update | current log-prob、loss、micro/accum步数、finite grad、`did_step`、optimizer/scheduler计数、peak memory | 每 rank attempt数一致；成功 update后参数hash变化；非有限时 optimizer_step不增 | rank计数不一致或nonfinite立即停止 profile |
| P7 checkpoint save/load | rollout memory已释放、FSDP model/optim/extra shard、dataloader、global/optimizer step、config/schema/commit、CPU RAM/磁盘峰值 | 3-card same-world checkpoint可原地加载；完成marker原子；无部分checkpoint被当latest | reload失败不得进入2-step/resume |
| P8 one-sided/probe | checkpoint-only、RNG/parameter/optimizer/grad hash前后、probe token计数、peak memory | no-mutation与RNG restore通过；credit保持关闭 | 任一状态变化则 probe family失败并停止集成 |

上游 checkpoint 当前按 world size 命名 shard（`fsdp_checkpoint_manager.py:92-103,176-185`），所以 8-card shard 不能直接作为 3-card resume；只能将兼容 HF 权重作为新 run 初始化，除非另有经验证的 reshard 工具。上游 extra 目前只含 scheduler/RNG（`:162-175`），driver tracker 普通覆写（`ray_trainer.py:891-900`），不能直接宣称满足新 resume 契约。

## 8. OOM 降级顺序（按发生阶段，不猜显存成功）

### 8.1 rollout / KV / full-vocabulary sampling OOM

1. 记录每卡 allocated/reserved、SGLang static memory、请求数、prompt/response实际长度、Top-K与发生点；先结束当前 run。
2. 降低 `max_num_batched_tokens` / `max_num_seqs` 和调度并发；不改变 trajectory 集合语义。
3. low 将 `P:6→3` 且 `M=3`；所有整除式仍成立。high 已为 P=3。
4. 将 `n:4→2`；P=3/M=3 时 `P×n=6`、`M×n=6` 均可被3整除，local mini=2。n 不低于2，否则 GRPO group variance语义退化。
5. 缩短 response，再缩短 prompt fixture；若独立 latent cap 已实现，再缩短 latent cap。每次变化生成新 resolved config hash，不能仍标原 profile结果。
6. 降低 SGLang static fraction仅作为独立 probe；太低也可能无法装入模型，不能假定一定改善。

定制 sampler 在取 top-K 前先构造全词表 log-prob/softmax/Gumbel临时量，内存并非只与K有关：`sglang_latent_reasoning_pkg/python/sglang/srt/layers/sampler.py:70-122`。

### 8.2 old/current forward 或 backward OOM

1. actor/ref/log-prob micro全部降至1。
2. 打开 gradient checkpointing（high 已开；low 是显式 profile deviation）。
3. 缩短 response/prompt；保持 TP=1/SP=1 和模型不变。
4. low 若仍失败，创建明确的新 `3gpu-low-offload-smoke`，再启 actor param/optimizer offload；不能静默改原 profile。
5. high 已显式 offload；若仍失败，停止。TP=3 只能进入单独 `3gpu-high-tp3-probe`，先验证模型 head divisibility和 latent TP数值一致性。
6. 不量化、不切 base model、不改成 vLLM、不关闭 Latent-GRPO机制来“通过”。

### 8.3 model load / weight sync / checkpoint OOM

- batch/micro缩小通常不能解决模型常驻或 state-dict峰值；应先确认 rollout memory release、offload时机、sharded state而非 full HF state、CPU RAM和磁盘。
- checkpoint probe首轮只保存上游要求的 sharded model/optimizer/extra，不请求 `hf_model` full state。FSDP save会在CPU构建sharded state/optimizer并每rank写文件：`fsdp_checkpoint_manager.py:159-185`。
- 仍 OOM/host OOM则停止该 profile；不得删除旧 checkpoint或自动改量化。

## 9. Failure propagation、cache 与下载设计

### 9.1 failure propagation

- Ray worker致命异常沿 RPC → `ray.get` → outer owner传播；owner将 `run_status=failed`，flush已提交part并原子更新control state，然后重新抛出。
- outer followers看见failed后以非零退出；若owner进程死亡且heartbeat超时，同样非零。torch elastic应终止同一 worker group；必须用故障注入实测，不能只依据设计声称。
- metrics family不可用只写 availability/reason；算法所需 tensor、rank collective、checkpoint一致性失败则是fatal。
- writer只有controller一份；Ray worker绝不能直接写同一权威 Parquet part。

### 9.2 cache / download

1. profile只保存 Hugging Face model ID和配置化 cache root；output metrics目录不存模型。
2. outer rank0在 Ray init前获得 run-scoped/cache-scoped lock；先查本地显式路径，再查完整 snapshot及完成marker。
3. 缺失且允许联网时，只有owner下载到临时snapshot，校验 config/tokenizer/weights/latent token ID与文件指纹后原子发布；followers和未来Ray workers只收到绝对 local snapshot路径。
4. download失败写精确状态并停止，不能让三个 Ray worker继续各自下载。上游当前对普通HF ID只原样返回并由后续HF API解析：`main_ppo.py:57-65`、`verl/utils/fs.py:190-251`，因此此逻辑必须由新 adapter补足。
5. Ray启动后强制 local/offline读取；记录模型ID、snapshot revision/hash和cache fingerprint，不记录token。
6. 数据同理：owner验证Parquet schema、行数、hash；上游 dataloader `drop_last=True`，batch必须至少满足P并有足够行：`ray_trainer.py:565-592`。

## 10. 风险矩阵

| 风险 | 等级 | 触发/证据 | 控制与停止条件 |
|---|---|---|---|
| 三个outer rank各启Ray | blocker | `run_ppo`无rank gate：`main_ppo.py:29-37` | 唯一owner断言；检测到第二个controller立即停止 |
| torchrun env污染Ray rank/world | blocker | Ray自行注入rank env：`ray/base.py:336-363` | owner启动Ray前清理outer rendezvous env；probe env来源 |
| 依赖/ABI不一致 | blocker | tensordict、sgl-kernel、FlashInfer等静态冲突见 dependency audit | import+kernel+collective不通过，不进入模型加载 |
| TP/mesh通信不一致 | blocker | rollout与manager使用两套等形mesh，尚未三卡验证 | 最短weight sync/rollout失败则停；不先拍脑袋patch维名 |
| 7B actor+ref+engine峰值超46GB | blocker | 三类逻辑角色共置，高难含ref | 分阶段峰值；显式offload；仍OOM则标未通过 |
| CPU offload导致host OOM/极慢 | major | high actor optim/param及ref param offload | probe RAM、page fault、step time；超阈值停，不转swap长跑 |
| full-vocab Gumbel临时峰值 | major | sampler `:70-122` | 降并发/长度/batch；不得认为Top-K=10即低内存 |
| dynamic filtering放大rollout成本 | major | `ray_trainer.py:1109-1211` | 上限、retry计数、最终batch断言；不足即失败 |
| `M*n`被3地板除 | major | `fsdp_workers.py:143-155` | 新validator强制整除并验算local mini/micro |
| 独立 latent cap不存在 | major | 无 `max_latent_length`，只有total max_new_tokens | 未实现前标unsupported；不得虚报生效 |
| TP=3模型结构不兼容 | major | 仅有world%TP表层校验 | 读取真实model config/head数；单独TP3 probe |
| checkpoint world-size绑定 | major | shard名含world size：checkpoint manager `:92-103` | 8→3直接resume拒绝；仅同world smoke |
| SGLang生成RNG未随checkpoint恢复 | major | sharding manager内存state `fsdp_sglang.py:81-90,113-133`，extra未保存 | 连续2步 vs 1+resume对比失败则resume未验证 |
| owner失败但followers挂起 | major | envelope新增风险 | heartbeat/watchdog/故障注入；超时非零退出 |
| cache并发/半下载 | major | upstream无显式rank0预取 | lock+temp+完整marker+offline worker；失败不启动Ray |
| 高profile被误称论文复现 | major | 作者是8卡长上下文/大batch | profile/result处固定 `device_adapted_smoke_only` |

### 10.1 未确认项总表

以下全部状态均为 `runtime_probe_required`；任一阻断项未闭合时不得声称三卡可运行：

| ID | 未确认项 | 闭合方式 |
|---|---|---|
| GPU-U01 | 实际GPU型号、每卡精确总/空闲显存、compute capability、BF16、P2P/NVLink/PCIe | P0环境probe并保存脱敏snapshot |
| GPU-U02 | 目标driver、torch CUDA runtime、cuDNN、实际NCCL与kernel ABI | import、tiny kernel、三卡all-reduce |
| GPU-U03 | 46 GiB预算下actor/ref/SGLang各阶段及交接峰值 | 第6.5节逐阶段采样 |
| GPU-U04 | `gpu_memory_utilization=0.50`在此fork的真实内存含义和最低可启动值 | engine init + memory saver probe |
| DIST-U01 | torchrun env清理、followers无CUDA、唯一Ray controller与故障联动 | P1/P2及failure injection |
| DIST-U02 | rollout自建`tp` mesh与manager `infer_tp` mesh的三卡通信一致性 | weight sync + latent rollout，不预先patch |
| DIST-U03 | TP=3的真实模型attention/KV head整除、latent weighted embedding一致性 | 读取local model config + 独立TP3 probe；非默认 |
| DIST-U04 | SP>1的remove-padding/位置恢复/指标mask语义 | 独立SP probe；本方案默认SP1 |
| DATA-U01 | 两个目标数据集有足够fixture、schema正确、P=3/6 drop-last可取batch | 本地Parquet schema/row count probe |
| ALGO-U01 | tokenizer实际latent-end ID（LLaMA候选524、Qwen候选522） | 对local snapshot编码并与生成transition核对 |
| ALGO-U02 | 独立`max_latent_length`如何执行且不改变latent→hard语义 | 最小adapter/patch + unit/runtime test；当前blocked |
| ALGO-U03 | dynamic filtering在小P/n下能否及时得到非零方差完整groups | 固定短smoke并记录retry；超上限失败 |
| ALGO-U04 | FlashAttention latent likelihood是否真实启用、未退化普通token log-prob | import/source选择与数值对照probe |
| MEM-U01 | full-vocab Gumbel在目标vocab/长度/并发下的瞬时显存 | rollout分阶段peak probe |
| CKPT-U01 | high offload时CPU RAM、checkpoint RSS/临时磁盘和I/O时长 | P7 host+device采样 |
| CKPT-U02 | SGLang生成RNG/engine state的checkpoint-resume连续性 | 连续2步 vs 1+resume对比；当前上游预计有缺口但需实测 |
| CACHE-U01 | owner-only下载、完整snapshot marker、offline workers与失败恢复 | cache miss/hit/corrupt故障注入 |
| DEP-U01 | Ray、TensorDict、FlashInfer、sgl-kernel等最终兼容版本 | dependency audit列出的resolver/import/kernel probes |

## 11. 进入正式训练前的停止条件

以下任一项成立就不得进入长训练：

1. 目标环境、依赖resolver、CUDA kernel、三卡NCCL任何一项未验证。
2. 不是恰好3张目标GPU，或显存/BF16/profile硬件不符且未使用显式 mismatch smoke标签。
3. 不是恰好一个Ray controller/TaskRunner、一套3-worker placement group。
4. outer rank 1/2出现CUDA context、Ray实例或权威输出写入。
5. Ray rank/device映射重复、FSDP collective hang、TP/mesh weight sync失败。
6. 实际import不是作者 vendored `verl` 和定制 `sglang`。
7. rollout缺 latent Top-K/perturbed score，或 actor走普通token likelihood静默退化。
8. batch/mini/micro任一整除式不成立；dynamic filtering无法得到完整P个group。
9. optimizer `did_step`无法可靠计数、rank间计数不一致或出现nonfinite update。
10. 单卡最小闭环、schema validator、三卡rank初始化、generated-token语义、stable trajectory ID、Support对齐、probe no-mutation/RNG restore、resume/去重中的任何必需项未通过。规范总停止条件：`spec/04_AGENT_ORCHESTRATION.md:339-349`。
11. 同world checkpoint不能reload，或partial checkpoint会被误识别为latest。
12. high-smoke按第8节安全降级后仍OOM/host OOM；结果必须写“未在目标硬件通过”，不能换算法掩盖。
13. reviewer仍有blocker。

## 12. 下一阶段准备执行的精确命令（本阶段未执行）

### 12.1 只读环境与依赖 probe

```bash
uname -a
cat /etc/os-release
python3 --version
python3 -c 'import sys,platform; print(sys.executable); print(platform.platform()); print(platform.libc_ver())'
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version,compute_cap --format=csv
nvidia-smi topo -m
python3 -m pip check
python3 -c 'import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.cuda.nccl.version()); print(torch.cuda.device_count()); print([(torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory, torch.cuda.is_bf16_supported(i)) for i in range(torch.cuda.device_count())])'
python3 -c 'import verl,sglang,ray,transformers,tensordict,torchdata,pyarrow,flash_attn,flashinfer,sgl_kernel; print(verl.__file__); print(sglang.__file__); print("imports_ok")'
```

### 12.2 下一阶段新增 probe 脚本后

```bash
CUDA_VISIBLE_DEVICES=0,1,2 NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nproc_per_node=3 scripts/probe_distributed.py
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 scripts/probe_ray_topology.py --expected-gpus 3 --expected-controller-count 1
python3 scripts/probe_sglang_runtime.py --config configs/3gpu-low.yaml --max-new-tokens 2 --offline
```

### 12.3 config/cache dry-run（必须先实现；不训练）

```bash
python3 train_latent_grpo.py --config configs/3gpu-low.yaml --profile-name 3gpu-low --validate-config --dry-run
python3 train_latent_grpo.py --config configs/3gpu-high-smoke.yaml --profile-name 3gpu-high-smoke --validate-config --dry-run
python3 scripts/prepare_assets.py --config configs/3gpu-low.yaml
python3 scripts/prepare_assets.py --config configs/3gpu-high-smoke.yaml
```

`prepare_assets.py` 只应在用户允许网络/下载后执行；已有完整cache时必须离线命中，不重复下载。

### 12.4 三卡分阶段 smoke（依赖、cache、probe均通过后）

```bash
CUDA_VISIBLE_DEVICES=0,1,2 NCCL_DEBUG=INFO torchrun --standalone --nproc_per_node=3 train_latent_grpo.py --config configs/3gpu-low.yaml --profile-name 3gpu-low --max-steps 1 --disable-support --disable-checkpoint-probe
python3 scripts/validate_outputs.py --output-root outputs/3gpu-low/seed_42
CUDA_VISIBLE_DEVICES=0,1,2 NCCL_DEBUG=INFO torchrun --standalone --nproc_per_node=3 train_latent_grpo.py --config configs/3gpu-high-smoke.yaml --profile-name 3gpu-high-smoke --max-steps 1 --disable-support --disable-checkpoint-probe
python3 scripts/validate_outputs.py --output-root outputs/3gpu-high-smoke/seed_42
```

### 12.5 checkpoint / resume / failure propagation

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 train_latent_grpo.py --config configs/3gpu-low.yaml --profile-name 3gpu-low --max-steps 2 --disable-support --disable-checkpoint-probe
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 train_latent_grpo.py --config configs/3gpu-low.yaml --profile-name 3gpu-low --resume-from outputs/3gpu-low/seed_42/checkpoints/step_00000002 --max-steps 4 --disable-support --disable-checkpoint-probe
python3 scripts/validate_outputs.py --output-root outputs/3gpu-low/seed_42
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 scripts/probe_failure_propagation.py --failure-ray-rank 1 --phase backward
```

这些命令是下一阶段的预定执行面，不是本报告的运行证据。具体 output root/seed 必须由最终 YAML 与 runbook统一；执行后应将 resolved/redacted command写入 `run_config.json`。

## 13. 可直接合并到 implementation plan 的决策

- 主入口：`torchrun` envelope；outer rank0唯一启动上游 Ray，outer rank1/2无CUDA等待。
- 保留单进程Python→Ray作为诊断路径，但三卡验收仍执行torchrun表面。
- 真正训练world由Ray管理，world=3；actor FSDP=3；TP=1；SP=1；rollout DP=3。
- 继续使用作者定制SGLang；vLLM不是fallback。
- 先使用第6节候选做1-step分阶段probe；任何显存结论必须来自目标机峰值记录。
- high profile显式gradient checkpoint + actor param/optimizer offload + ref param offload；不量化。
- 新validator补足 `M*n`、local mini/micro和P/M整除；保存推导出的gradient accumulation与预期optimizer attempts。
- 不在未复现前修改SGLang DeviceMesh维名；先做group/weight-sync/latent-rollout probe。
- `max_latent_length` 当前是实现缺口，不能只写配置假装生效。
- cache由唯一owner准备并原子发布，Ray workers只读本地offline snapshot。
- checkpoint仅同world-size resume；8卡shard到3卡直接resume明确拒绝。
- 所有profile和报告固定标记：`device_adapted` / `smoke_only` / `not_paper_reproduction`。
