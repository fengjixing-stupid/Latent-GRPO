# Kaggle dual-T4 runtime compatibility (semantics-preserving)

Base commit: `def499cd587c28c6be7811a76b4139d519fb6375`.

## What changes

- T4 runtime precision may use FP16; existing BF16 profiles remain BF16.
- `use_remove_padding=True` keeps the existing packed FlashAttention path.
- `use_remove_padding=False` now has a real latent path using the same rollout Top-K IDs, Gumbel scores, mixture formula, Gumbel likelihood, advantages, and FlipGrad rule.
- HF actor attention becomes SDPA only on the padded path.
- SGLang can receive `engine_kwargs.sglang.attention_backend=triton`.
- SGLang `sampling_backend` remains `flashinfer`; the custom latent sampler is not replaced.
- `logprobs_from_logits_topk_gumbel` drops only an obsolete FlashAttention availability gate; the existing PyTorch formula is retained.

## Explicitly unchanged

Latent trajectory state/termination, Top-K selection semantics, Gumbel noise and one-sided transform, latent-end switching, dynamic group filtering, advantages, OCP, PPO loss, FlipGrad straight-through formula, and P1 metrics definitions/sampling points.

## T4 dependency policy

Use the dedicated `tools/install_kaggle_t4_runtime.sh`; do not replace the general 3-GPU runtime requirements. The T4 script selects PyTorch 2.6 CUDA 11.8 and the CUDA-11.8 SGL kernel wheel so SM75 remains included. It does not read or create training data.

## Gate

After installing the stack, run:

```bash
python tools/probe_kaggle_p1_t4_compatibility.py
```

`READY_FOR_DATA` means only that the T4 code/runtime compatibility gate passed. It is not a training result. Real train/validation Parquet must still be supplied by the user before the runtime smoke.
