# Kaggle 双 T4：P1 Runtime Compatibility Gate

## 当前结论

在 Git 基线 `8fab3a52a4af98cd753769b7e1c99457814d82f8` 上，不应直接启动 P1 真实训练 smoke。

原因不是训练数据，而是当前训练运行时与 T4/SM75 的兼容性：

1. 仓库的 target environment gate 当前要求 BF16；T4/SM75 不具备该 BF16 能力。
2. `dp_actor.py` 的 latent actor forward 路径当前显式使用 `torch.bfloat16`。
3. `fsdp_workers.py` 当前强制 Hugging Face actor 使用 `flash_attention_2` attention implementation。
4. 这三点都发生在真实训练数据之前，所以不应先要求用户提供数据，更不能通过跳过 gate 来伪造一次“通过”的 T4 smoke。

## 本 gate 做什么

运行：

```bash
python tools/probe_kaggle_p1_t4_compatibility.py
```

默认报告：

```text
/kaggle/working/latent-grpo-p1-t4-compatibility.json
```

报告只可能把当前 checkout 判为：

- `BLOCKED`：仍存在平台/运行时 blocker；不得启动训练，也不需要训练数据。
- `READY_FOR_DATA`：仅表示平台 blocker 已消除，下一步才向用户索取真实 train/val Parquet。

`READY_FOR_DATA` **不是训练 PASS**。

## 数据安全约定

本 compatibility gate：

- 不接受训练/验证数据参数；
- 不读取训练/验证数据；
- 不生成 synthetic 数据；
- 不抽样、裁剪、复制或改写数据；
- 不调用训练入口；
- 不初始化 Latent-GRPO optimizer/update。

只有 compatibility gate 变为 `READY_FOR_DATA` 后，才进入真实数据 preflight 和 1–2 global-step runtime smoke。

## 后续 T4 兼容工作必须单独评审

如果坚持使用双 T4 完成真实 actor update，需要把它作为独立的 **T4 runtime compatibility** 工程任务，而不是在 smoke 脚本里隐式绕过：

- 显式选择 T4 支持的训练精度；
- 解决 latent actor 当前 BF16 固定路径；
- 解决 packed/remove-padding latent forward 对 FlashAttention-2 的依赖；
- 验证替代 attention 路径不会改变 trajectory、advantage、Gumbel/FlipGrad 与 optimizer 语义；
- 重新通过 Mac 静态/unit gate 后，才可在 T4 上使用用户提供的真实 Parquet。
