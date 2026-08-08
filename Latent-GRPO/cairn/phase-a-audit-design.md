# Phase A：只读审计与实现设计

## 结论

- 作者训练入口是 `python3 -m verl.trainer.main_ppo`；Hydra/Ray 负责协调，FSDP actor 与定制 SGLang rollout 才是实际训练面。
- 新系统采用作者仓库外的 runner/metrics/storage；作者算法链仅增加可逆的最小 observer instrumentation patch。
- 仅靠公开入口无法可靠取得 stable trajectory ID、OCP winner、成功 optimizer step、component sufficient statistics 与逐题 eval raw facts，因此下一阶段必须修改少量作者文件，但不得复制整个训练循环或改写算法。
- 三卡方案以外层 `torchrun` 控制面、单一 Ray coordinator、FSDP DP=3、SGLang TP=1 起步；任何 46 GB 显存、CUDA/NCCL/ABI 与性能结论都必须由目标 Linux runtime probe 证明。
- 作者根 `requirements.txt` 是存在内部冲突的环境快照，不能直接安装或作为版本真相；Python requirements 与系统 driver/CUDA/NCCL/compiler 必须分层。
- 本阶段没有安装依赖、没有启动 GPU/训练、没有编写训练代码；所有 RTM 与测试状态保持 `planned`/`blocked`。

## 关键风险

- Qwen latent-end 配置 522 与 sampler 硬编码 524 不一致。
- actor 的 `topk_logits` 返回路径错误拼接了 top-K IDs。
- FlashAttention import 失败可能静默回退，破坏 latent component log-prob 语义。
- dynamic batch、remove-padding、fused kernel、EOS/overlong、三卡 DeviceMesh/collective 与 SGLang RNG resume 仍需 runtime probe。

## 权威设计资产

- `../../docs/repo_audit.md`
- `../../docs/dependency_audit.md`
- `../../docs/implementation_plan.md`
- `../../docs/decision_log.md`
- `../../docs/requirements_traceability_matrix.md`
- `../../work_reports/agent_a_repo_audit.md`
- `../../work_reports/agent_b_dependency_audit.md`
- `../../work_reports/agent_c_target_variable_mapping.md`
- `../../work_reports/agent_d_3gpu_plan.md`
- `../../work_reports/agent_e_validation_plan.md`

## 下一步门禁

等待用户授权后，先在目标 Linux/3-GPU 机器执行只读环境与依赖 probe；只有版本、CUDA/NCCL、tensor shape 和 launcher gate 闭合，才进入测试先行的实现阶段。不得在单卡 smoke、resume、validator 与 no-mutation 证据通过前启动三卡正式训练。
