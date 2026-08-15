# Optional code and video dependencies

The active Latent-GRPO Low and High profiles are text-only mathematical
reasoning workloads.

## pyext

`pyext` is excluded from `requirements/reward-math.txt` because the available
legacy package uses `inspect.getargspec`, which is unavailable on Python 3.11.
It is relevant only to optional code/prime reward execution paths and is not
required by GSM8K or DAPO-Math text rewards.

## decord

`decord` is excluded from `requirements/runtime-sglang.txt` because the
available `decord==0.6.0` wheel is not compatible with the active Python 3.11
platform. It is relevant only to optional video or multimodal workloads and is
not required by the LLaMA/Qwen text-only Latent-GRPO profiles.

Neither package should be reintroduced into the default text runtime.
