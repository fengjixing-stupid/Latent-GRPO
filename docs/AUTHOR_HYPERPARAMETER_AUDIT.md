# 作者超参数审计

Git 基线：`53438ec07b804ebd1b670d6fe118199798350505`。

优先级：对应实验的作者 shell > README > 论文 Appendix B。机器真值分别为 `configs/author/latent_grpo_gsm8k_llama3.yaml` 和 `configs/author/latent_grpo_math_qwen.yaml`。作者论文 Appendix B 明确 low/high 的 LR、train batch、PPO mini batch、rollout 数和 response length，并说明全部实验使用 8×A100；这些数值与 shell 一致：[arXiv Appendix B](https://arxiv.org/html/2604.27998#A2)。

## 低难度完整作者表

来源文件：`Latent-GRPO/Latent-GRPO-gsm8k-llama3.sh`。下表 key 均为该 shell 的 Hydra key（表内省略共同前缀时以 group 补全）。

| group | exact keys and values |
|---|---|
| algorithm | `adv_estimator=grpo`; `use_kl_in_reward=false`; `exclude_overlong_samples_from_advantage=false` |
| data | `train_files=GSM8k-Aug-oss-dup-all.parquet`; `val_files=GSM8k-Aug-test.parquet`; `train_batch_size=64`; `val_batch_size=128`; `max_prompt_length=192`; `max_response_length=128`; `filter_overlong_prompts=true` |
| actor | `optim.lr=1e-6`; `ppo_mini_batch_size=16`; `ppo_micro_batch_size_per_gpu=2`; `use_kl_loss=false`; `ppo_max_token_len_per_gpu=2048`; `kl_loss_coef=0.001`; `neg_adv_weight=1.0`; `kl_loss_type=low_var_kl`; `entropy_coeff=0`; `freeze_embedding=true` |
| model / FSDP | `use_remove_padding=true`; `enable_gradient_checkpointing=false`; `fsdp_config.param_offload=false`; `fsdp_config.optimizer_offload=false` |
| rollout | `log_prob_micro_batch_size_per_gpu=2`; `max_model_len=1024`; `max_num_batched_tokens=2048`; `tensor_model_parallel_size=1`; `name=sglang`; `dtype=bfloat16`; `top_p=0.95`; `top_k=30`; `max_topk=10`; `temperature=0.6`; `gumbel_softmax_temperature=1.0`; `enable_latent=true`; `latent_end_token_id=524`; `add_noise_gumbel_softmax=true`; `use_one_sided_gumbel_noise=true`; `noise_scale=1.0`; `gpu_memory_utilization=0.6`; `n=8` |
| validation / ref | `val_kwargs.do_sample=true`; `val_kwargs.temperature=0.6`; `val_kwargs.top_p=0.95`; `val_kwargs.top_k=30`; `ref.log_prob_micro_batch_size_per_gpu=2`; `ref.strategy=fsdp2` |
| dynamic filtering | `filter_groups.enable=true`; `filter_groups.max_num_gen_batches=50` |
| trainer | `critic_warmup=0`; `logger=[console,wandb]`; `project_name=latent-grpo`; `experiment_name=latent-grpo-gsm8k-llama3`; `val_before_train=true`; `n_gpus_per_node=8`; `nnodes=1`; `save_freq=40`; `test_freq=10`; `default_hdfs_dir=null`; `balance_batch=true`; `default_local_dir=./saved/latent-grpo-gsm8k-llama3`; `total_epochs=10` |

## 高难度完整作者表

来源文件：`Latent-GRPO/Latent-GRPO-math500-qwen.sh`。其完整字段如下。

| group | exact keys and values |
|---|---|
| algorithm | `adv_estimator=grpo`; `use_kl_in_reward=false`; `exclude_overlong_samples_from_advantage=true` |
| data | `train_files=DAPO-Math-17k-en-train.parquet`; `val_files=Math-500-test.parquet`; `train_batch_size=32`; `val_batch_size=500`; `max_prompt_length=1024`; `max_response_length=4096`; `filter_overlong_prompts=true` |
| actor | `optim.lr=1e-6`; `ppo_mini_batch_size=32`; `ppo_micro_batch_size_per_gpu=1`; `use_kl_loss=true`; `ppo_max_token_len_per_gpu=20480`; `kl_loss_coef=0.001`; `neg_adv_weight=1.0`; `kl_loss_type=low_var_kl`; `entropy_coeff=0`; `freeze_embedding=true` |
| model / FSDP | `use_remove_padding=true`; `enable_gradient_checkpointing=true`; `fsdp_config.param_offload=true`; `fsdp_config.optimizer_offload=true` |
| rollout | `log_prob_micro_batch_size_per_gpu=2`; `max_model_len=12000`; `max_num_batched_tokens=12000`; `tensor_model_parallel_size=1`; `name=sglang`; `dtype=bfloat16`; `top_p=0.95`; `top_k=30`; `max_topk=10`; `temperature=0.6`; `gumbel_softmax_temperature=1.0`; `enable_latent=true`; `latent_end_token_id=522`; `add_noise_gumbel_softmax=true`; `use_one_sided_gumbel_noise=true`; `noise_scale=1.0`; `gpu_memory_utilization=0.8`; `n=8` |
| validation / ref | `val_kwargs.do_sample=true`; `val_kwargs.temperature=0.6`; `val_kwargs.top_p=0.95`; `val_kwargs.top_k=30`; `ref.log_prob_micro_batch_size_per_gpu=2`; `ref.strategy=fsdp2` |
| dynamic filtering | `filter_groups.enable=true`; `filter_groups.max_num_gen_batches=50` |
| trainer | `critic_warmup=0`; `logger=[console,wandb]`; `project_name=latent-grpo`; `experiment_name=latent-grpo-math500-qwen`; `val_before_train=true`; `n_gpus_per_node=8`; `nnodes=1`; `save_freq=40`; `test_freq=10`; `default_hdfs_dir=null`; `balance_batch=true`; `default_local_dir=./saved/latent-grpo-math500-qwen`; `total_epochs=5` |

## 完整数值对照

| parameter / exact key | author low | author high | shell source | paper value | 3GPU low | deviation? |
|---|---:|---:|---|---:|---:|---|
| `algorithm.adv_estimator` | grpo | grpo | both shells | GRPO | grpo | no |
| `data.train_batch_size` | 64 | 32 | `data.train_batch_size` | 64 / 32 | 48 | yes, topology |
| `data.val_batch_size` | 128 | 500 | `data.val_batch_size` | AUTHOR_NOT_PUBLISHED | 128 | no |
| `data.max_prompt_length` | 192 | 1024 | `data.max_prompt_length` | AUTHOR_NOT_PUBLISHED | 192 | no |
| `data.max_response_length` | 128 | 4096 | `data.max_response_length` | 128 / 4096 | 128 | no |
| `data.filter_overlong_prompts` | true | true | `data.filter_overlong_prompts` | invalid sample masking described | true | no |
| `actor.optim.lr` | 1e-6 | 1e-6 | `actor_rollout_ref.actor.optim.lr` | 1e-6 | 1e-6 | no |
| `actor.ppo_mini_batch_size` | 16 | 32 | `actor_rollout_ref.actor.ppo_mini_batch_size` | 16 / 32 | 12 | yes, topology |
| `actor.ppo_micro_batch_size_per_gpu` | 2 | 1 | exact shell key | AUTHOR_NOT_PUBLISHED | 2 | no |
| `actor.use_kl_loss` | false | true | exact shell key | AUTHOR_NOT_PUBLISHED | false | no |
| `actor.ppo_max_token_len_per_gpu` | 2048 | 20480 | exact shell key | AUTHOR_NOT_PUBLISHED | 2048 | no |
| `actor.kl_loss_coef` | 0.001 | 0.001 | exact shell key | AUTHOR_NOT_PUBLISHED | 0.001 | no |
| `actor.neg_adv_weight` | 1.0 | 1.0 | exact shell key | AUTHOR_NOT_PUBLISHED | 1.0 | no |
| `actor.kl_loss_type` | low_var_kl | low_var_kl | exact shell key | AUTHOR_NOT_PUBLISHED | low_var_kl | no |
| `actor.entropy_coeff` | 0 | 0 | exact shell key | AUTHOR_NOT_PUBLISHED | 0 | no |
| `actor.freeze_embedding` | true | true | exact shell key | AUTHOR_NOT_PUBLISHED | true | no |
| `model.use_remove_padding` | true | true | exact shell key | AUTHOR_NOT_PUBLISHED | true | no |
| `model.enable_gradient_checkpointing` | false | true | exact shell key | AUTHOR_NOT_PUBLISHED | false | no |
| `actor.fsdp_config.param_offload` | false | true | exact shell key | AUTHOR_NOT_PUBLISHED | false | no |
| `actor.fsdp_config.optimizer_offload` | false | true | exact shell key | AUTHOR_NOT_PUBLISHED | false | no |
| `rollout.log_prob_micro_batch_size_per_gpu` | 2 | 2 | exact shell key | AUTHOR_NOT_PUBLISHED | 2 | no |
| `rollout.max_model_len` | 1024 | 12000 | exact shell key | AUTHOR_NOT_PUBLISHED | 1024 | no |
| `rollout.max_num_batched_tokens` | 2048 | 12000 | exact shell key | AUTHOR_NOT_PUBLISHED | 2048 | no |
| `rollout.tensor_model_parallel_size` | 1 | 1 | exact shell key | AUTHOR_NOT_PUBLISHED | 1 | no |
| `rollout.name` | sglang | sglang | exact shell key | AUTHOR_NOT_PUBLISHED | sglang | no |
| `rollout.dtype` | bfloat16 | bfloat16 | exact shell key | AUTHOR_NOT_PUBLISHED | bfloat16 | no |
| `rollout.top_p` | 0.95 | 0.95 | exact shell key | AUTHOR_NOT_PUBLISHED | 0.95 | no |
| `rollout.top_k` | 30 | 30 | exact shell key | K=10 latent mixture only | 30 | no |
| `rollout.max_topk` | 10 | 10 | exact shell key | top-K=10 | 10 | no |
| `rollout.temperature` | 0.6 | 0.6 | exact shell key | AUTHOR_NOT_PUBLISHED | 0.6 | no |
| `rollout.gumbel_softmax_temperature` | 1.0 | 1.0 | exact shell key | AUTHOR_NOT_PUBLISHED for GRPO | 1.0 | no |
| `rollout.enable_latent` | true | true | exact shell key | method requirement | true | no |
| `rollout.latent_end_token_id` | 524 | 522 | exact shell key | AUTHOR_NOT_PUBLISHED | 524 | no |
| `rollout.add_noise_gumbel_softmax` | true | true | exact shell key | method requirement | true | no |
| `rollout.use_one_sided_gumbel_noise` | true | true | exact shell key | one-sided method | true | no |
| `rollout.noise_scale` | 1.0 | 1.0 | exact shell key | AUTHOR_NOT_PUBLISHED | 1.0 | no |
| `rollout.gpu_memory_utilization` | 0.6 | 0.8 | exact shell key | AUTHOR_NOT_PUBLISHED | 0.6 | no |
| `rollout.n` | 8 | 8 | exact shell key | low=8; high not separately restated | 8 | no |
| `rollout.val_kwargs.do_sample` | true | true | exact shell key | AUTHOR_NOT_PUBLISHED | true | no |
| `rollout.val_kwargs.temperature` | 0.6 | 0.6 | exact shell key | AUTHOR_NOT_PUBLISHED | 0.6 | no |
| `rollout.val_kwargs.top_p` | 0.95 | 0.95 | exact shell key | AUTHOR_NOT_PUBLISHED | 0.95 | no |
| `rollout.val_kwargs.top_k` | 30 | 30 | exact shell key | AUTHOR_NOT_PUBLISHED | 30 | no |
| `ref.log_prob_micro_batch_size_per_gpu` | 2 | 2 | exact shell key | AUTHOR_NOT_PUBLISHED | 2 | no |
| `ref.strategy` | fsdp2 | fsdp2 | exact shell key | AUTHOR_NOT_PUBLISHED | fsdp2 | no |
| `algorithm.use_kl_in_reward` | false | false | exact shell key | AUTHOR_NOT_PUBLISHED | false | no |
| `algorithm.exclude_overlong_samples_from_advantage` | false | true | exact shell key | masking method; numeric mode not restated | false | no |
| `filter_groups.enable` | true | true | exact shell key | AUTHOR_NOT_PUBLISHED | true | no |
| `filter_groups.max_num_gen_batches` | 50 | 50 | exact shell key | AUTHOR_NOT_PUBLISHED | 50 | no |
| `trainer.critic_warmup` | 0 | 0 | exact shell key | AUTHOR_NOT_PUBLISHED | 0 | no |
| `trainer.val_before_train` | true | true | exact shell key | AUTHOR_NOT_PUBLISHED | true | no |
| `trainer.n_gpus_per_node` | 8 | 8 | exact shell key | 8×A100 | 3 | yes, topology |
| `trainer.nnodes` | 1 | 1 | exact shell key | AUTHOR_NOT_PUBLISHED | 1 | no |
| `trainer.save_freq` | 40 | 40 | exact shell key | AUTHOR_NOT_PUBLISHED | 40 | no |
| `trainer.test_freq` | 10 | 10 | exact shell key | AUTHOR_NOT_PUBLISHED | 10 | no |
| `trainer.balance_batch` | true | true | exact shell key | AUTHOR_NOT_PUBLISHED | true | no |
| `trainer.total_epochs` | 10 | 5 | exact shell key | AUTHOR_NOT_PUBLISHED | 10 | no |

`ppo_epochs=1`、sequence parallel size 1、ref param offload false、seed 17 和 latent marker string `</think>` 未由作者 shell 公开；它们分别来自 vendored upstream default、runner 既有显式默认或 target validation 需要，均标为 `AUTHOR_NOT_PUBLISHED`，没有伪装成作者值。

## 来源冲突

| parameter | author shell value | README value | paper value | chosen value | reason |
|---|---|---|---|---|---|
| high initialization model | shell 仅占位路径 | 模型列表为 `DJCheng/Qwen2.5-Math-7B-Latent-SFT-4k-Top10`；启动示例却写 `...Latent-GRPO-4k-Top10` | best Latent-SFT checkpoint | `...Latent-SFT-4k-Top10` | 本任务和 README 模型列表都要求训练从 Latent-SFT 初始化；README 启动示例疑似命名冲突 |
| low/high batch/LR/response | 64/32；1e-6；128/4096 | README 指向 shell，无第二组数值 | 与 shell一致 | shell value | 实验 shell 是第一来源 |
| GPU topology | 8 GPU | README shell 默认 8 IDs | 8×A100 | target=3 | 只能作为明确工程适配，不称 strict reproduction |

论文还公开 one-sided noise clipping `[-1.5, 3.0]`，但作者 shell没有对应 Hydra 字段；当前 vendored 实现是代码语义，不在本次 3GPU 打包中改写。
