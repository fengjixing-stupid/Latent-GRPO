# Requirements candidate report

Status date: 2026-08-03.  The files in `requirements/` are evidence-based input groups, not a Linux lock.  They were not generated with `pip freeze`; `constraints/linux-cu124-py311.txt` is a candidate and remains `target_machine_test_deferred` until the teammate returns target-machine reports.

## Runtime and development Python distributions

`mac_static_status=static_check_passed` means the declaration was checked against repository metadata/import evidence. It does not mean the Linux/CUDA package was installed on macOS.

| package | candidate_version | class | required_by | evidence | installation_method | mac_static_status | target_machine_status | known_risks |
|---|---|---|---|---|---|---|---|---|
| torch | 2.6.0 + official cu124 wheel | runtime-special | FSDP actor/ref, SGLang | author README; vendored SGLang `pyproject.toml` | separate official PyTorch CUDA-index step | static_check_passed | target_machine_test_deferred | driver/runtime/ABI and BF16 unverified |
| torchvision | 0.21.0 + cu124 | runtime-special | SGLang torch family | vendored SGLang `srt` extra | same PyTorch step | static_check_passed | target_machine_test_deferred | must match torch 2.6.0 |
| transformers | 4.51.1 | runtime | model/tokenizer and SGLang | exact vendored SGLang metadata; `fsdp_workers.py` | runtime-core | static_check_passed | target_machine_test_deferred | model custom-code compatibility |
| accelerate | unresolved | runtime | verl model/FSDP utilities | vendored verl `setup.py`; SGLang test metadata | runtime-core | static_check_passed | target_machine_test_deferred | resolve with peft/transformers |
| codetiming | unresolved | runtime | trainer/worker timers | `ray_trainer.py`, `fsdp_workers.py` imports | runtime-core | static_check_passed | target_machine_test_deferred | exact release unconfirmed |
| datasets | unresolved | runtime | RL dataset/data preparation | `rl_dataset.py`; both vendored metadata files | runtime-core | static_check_passed | target_machine_test_deferred | PyArrow/NumPy combination |
| dill | unresolved | runtime | verl serialization | vendored verl `setup.py` | runtime-core | static_check_passed | target_machine_test_deferred | Ray/cloudpickle interaction |
| hydra-core | unresolved | runtime | author entry/config composition | `main_ppo.py` import; verl metadata | runtime-core | static_check_passed | target_machine_test_deferred | exact OmegaConf pair unconfirmed |
| numpy | `<2.0.0` candidate | runtime | DataProto, pandas, Ray, SGLang | `verl/protocol.py`; CUDA reference scripts | runtime-core | static_check_passed | target_machine_test_deferred | author snapshot 2.3.3 conflicts with reference stack |
| pandas | unresolved | runtime | DataProto/data tables | `verl/protocol.py`; verl metadata | runtime-core | static_check_passed | target_machine_test_deferred | NumPy/PyArrow ABI |
| peft | unresolved | runtime | optional adapter/model path imported by verl | vendored verl `setup.py`; model utilities | runtime-core | static_check_passed | target_machine_test_deferred | Transformers compatibility |
| pybind11 | unresolved | runtime/build helper | vendored verl base metadata | vendored verl `setup.py` | runtime-core | static_check_passed | target_machine_test_deferred | compiler needed only for source builds |
| PyYAML | unresolved | runtime | outer profile parser | `latent_grpo_runner/config.py` import | runtime-core | static_check_passed | target_machine_test_deferred | pure Python declaration; parser tested on Mac |
| ray[default] | `>=2.41.0`, exact unresolved | runtime | one driver, GPU workers | `main_ppo.py`; vendored verl metadata | runtime-core | static_check_passed | target_machine_test_deferred | exact version/3-GPU placement/error propagation |
| safetensors | unresolved | runtime | checkpoint/model serialization | vendored model/checkpoint modules | runtime-core | static_check_passed | target_machine_test_deferred | filesystem/large-checkpoint behavior |
| tensordict | `<=0.6.2` | runtime | DataProto/rollout batch | vendored verl metadata | runtime-core | static_check_passed | target_machine_test_deferred | root author snapshot 0.10.0 conflicts |
| torchdata | unresolved | runtime | `StatefulDataLoader` resume | `ray_trainer.py` import | runtime-core | static_check_passed | target_machine_test_deferred | exact save/resume API version |
| packaging | `>=20.0` | runtime | version guards | vendored verl and SGLang metadata | runtime-core | static_check_passed | target_machine_test_deferred | no exact pin needed yet |
| pyarrow | `>=19.0.0` | runtime-metrics | training parquet and append-only metrics | vendored verl metadata; storage contract | metrics | static_check_passed | target_machine_test_deferred | real Mac roundtrip pending dev install; Linux ABI pending |
| aiohttp | unresolved | runtime-sglang | SGLang base/HTTP internals | vendored SGLang metadata | runtime-sglang | static_check_passed | target_machine_test_deferred | transitive HTTP versions |
| requests | unresolved | runtime-sglang | SGLang base | vendored SGLang metadata | runtime-sglang | static_check_passed | target_machine_test_deferred | resolver candidate |
| tqdm | unresolved | runtime-sglang | SGLang base progress | vendored SGLang metadata | runtime-sglang | static_check_passed | target_machine_test_deferred | low risk |
| setproctitle | unresolved | runtime-sglang | SGLang process management | vendored SGLang metadata | runtime-sglang | static_check_passed | target_machine_test_deferred | target wheel architecture |
| ipython | unresolved | runtime-sglang | SGLang declared base | vendored SGLang metadata | runtime-sglang | static_check_passed | target_machine_test_deferred | upstream declares it although training does not directly import it |
| compressed-tensors | unresolved | runtime-sglang | SGLang model loading | vendored SGLang `runtime_common` | runtime-sglang | static_check_passed | target_machine_test_deferred | model path dependent |
| decord | unresolved | runtime-sglang | SGLang runtime_common | vendored SGLang metadata | runtime-sglang | static_check_passed | target_machine_test_deferred | text-only path may not execute; wheel availability |
| fastapi | unresolved | runtime-sglang | SGLang server | vendored SGLang metadata | runtime-sglang | static_check_passed | target_machine_test_deferred | Pydantic compatibility |
| hf-transfer | unresolved | runtime-sglang | SGLang/HF transfer support | vendored SGLang metadata | runtime-sglang | static_check_passed | target_machine_test_deferred | optional at execution but declared upstream |
| huggingface-hub | unresolved | runtime-sglang | model/tokenizer assets | vendored SGLang metadata | runtime-sglang | static_check_passed | target_machine_test_deferred | offline/cache policy |
| interegular | unresolved | runtime-sglang | constrained decoding | vendored SGLang metadata | runtime-sglang | static_check_passed | target_machine_test_deferred | resolver candidate |
| llguidance | `>=0.7.11,<0.8.0` | runtime-sglang | grammar backend | vendored SGLang metadata | runtime-sglang | static_check_passed | target_machine_test_deferred | platform wheel |
| ninja | unresolved | runtime-sglang/build | kernel build helper | vendored SGLang metadata | runtime-sglang | static_check_passed | target_machine_test_deferred | system compiler still separate |
| orjson | unresolved | runtime-sglang | server serialization | vendored SGLang metadata | runtime-sglang | static_check_passed | target_machine_test_deferred | platform wheel |
| pillow | unresolved | runtime-sglang | declared runtime_common | vendored SGLang metadata | runtime-sglang | static_check_passed | target_machine_test_deferred | text-only path may not execute |
| prometheus-client | `>=0.20.0` | runtime-sglang | SGLang server metrics | vendored SGLang metadata | runtime-sglang | static_check_passed | target_machine_test_deferred | port/process behavior |
| psutil | unresolved | runtime-sglang/core | worker memory metrics, SGLang | `fsdp_workers.py`; SGLang metadata | runtime-sglang | static_check_passed | target_machine_test_deferred | target process permissions |
| pydantic | unresolved | runtime-sglang | SGLang server models | vendored SGLang metadata | runtime-sglang | static_check_passed | target_machine_test_deferred | FastAPI major-version pair |
| pynvml | unresolved | runtime-sglang | GPU telemetry | vendored SGLang metadata | runtime-sglang | static_check_passed | target_machine_test_deferred | NVIDIA driver required |
| python-multipart | unresolved | runtime-sglang | SGLang server | vendored SGLang metadata | runtime-sglang | static_check_passed | target_machine_test_deferred | resolver candidate |
| pyzmq | `>=25.1.2` | runtime-sglang | SGLang process IPC | vendored SGLang metadata | runtime-sglang | static_check_passed | target_machine_test_deferred | IPC under target topology |
| soundfile | 0.13.1 | runtime-sglang | declared runtime_common | vendored SGLang metadata | runtime-sglang | static_check_passed | target_machine_test_deferred | system libsndfile may be needed; text path unused |
| torchao | `>=0.7.0` | runtime-sglang | SGLang runtime_common | vendored SGLang metadata | runtime-sglang | static_check_passed | target_machine_test_deferred | torch 2.6 compatibility probe |
| uvicorn | unresolved | runtime-sglang | SGLang server | vendored SGLang metadata | runtime-sglang | static_check_passed | target_machine_test_deferred | runtime process probe |
| uvloop | unresolved | runtime-sglang | SGLang event loop | vendored SGLang metadata; author README 0.21.0 candidate | runtime-sglang | static_check_passed | target_machine_test_deferred | Linux-only execution |
| xgrammar | 0.1.17 | runtime-sglang | grammar backend | vendored SGLang metadata | runtime-sglang | static_check_passed | target_machine_test_deferred | binary wheel compatibility |
| outlines | `>=0.0.44,<=0.1.11` | runtime-sglang | constrained decoding | vendored SGLang metadata | runtime-sglang | static_check_passed | target_machine_test_deferred | Pydantic/Transformers interaction |
| partial-json-parser | unresolved | runtime-sglang | streaming structured output | vendored SGLang metadata (`partial_json_parser`) | runtime-sglang | static_check_passed | target_machine_test_deferred | distribution name normalization |
| einops | unresolved | runtime-sglang | tensor rearrangement | vendored SGLang metadata | runtime-sglang | static_check_passed | target_machine_test_deferred | low risk |
| cuda-python | unresolved | runtime-special | SGLang CUDA bindings | vendored SGLang `srt` extra | runtime-sglang after torch | static_check_passed | target_machine_test_deferred | Python package does not replace toolkit/driver |
| flashinfer-python | 0.2.3 candidate | runtime-special | SGLang attention backend | vendored SGLang metadata | guarded target install | static_check_passed | target_machine_test_deferred | wheel/GPU ABI and existing code version gate differ |
| modelscope | unresolved | runtime-sglang | declared alternate model source | vendored SGLang metadata | runtime-sglang | static_check_passed | target_machine_test_deferred | optional path, large dependency surface |
| sgl-kernel | 0.1.0 vs 0.1.1 unresolved | runtime-special | SGLang CUDA custom ops | vendored metadata vs author README | guarded target install after probe | static_check_passed | target_machine_test_deferred | exact ABI conflict must be resolved on target |
| flash-attn | 2.7.3 candidate | runtime-special | latent Gumbel log-prob/remove-padding | author README; `torch_functional.py`/actor path | wheel if compatible, otherwise `--no-build-isolation` source build | static_check_passed | target_machine_test_deferred | cp311/cu124/torch2.6/CXX11 ABI/GCC/nvcc |
| math-verify | unresolved | runtime-reward | math reward scorer | vendored verl math extra; author README | reward-math | static_check_passed | target_machine_test_deferred | package import name uses underscore |
| mathruler | unresolved | runtime-reward | geometry/math scorer | vendored verl geo extra | reward-math | static_check_passed | target_machine_test_deferred | only selected reward path |
| pylatexenc | unresolved | runtime-reward | answer parsing | vendored verl base metadata/scorer | reward-math | static_check_passed | target_machine_test_deferred | direct upstream base despite reward-specific use |
| pyext | unresolved | runtime-reward | prime scorer | vendored verl prime extra | reward-math | static_check_passed | target_machine_test_deferred | only selected reward path |
| pytest | unresolved | dev/test | unit/integration suite | vendored verl test extra; local tests | test | mac_development_check_passed | target_machine_test_deferred | not a runtime dependency |
| wandb | unresolved | optional-tracking | upstream tracking backend | `verl/utils/tracking.py`; vendored base metadata | tracking-optional only | static_check_passed | target_machine_test_deferred | excluded from authoritative metrics core |
| tensorboard | unresolved | optional-tracking | optional author logger | author README | tracking-optional only | static_check_passed | target_machine_test_deferred | not required for Parquet writer |

Vendored sources are installed only after the dependency groups:

```bash
python -m pip install --no-deps -e ./Latent-GRPO/sglang_latent_reasoning_pkg/python
python -m pip install --no-deps -e ./Latent-GRPO/verl-0.4.x
```

This avoids `verl[sglang]` pulling upstream SGLang `0.4.6.post5` over the author's `0.4.6.post1` fork and avoids the vendored metadata silently choosing unverified CUDA packages.

## Not Python requirements

The NVIDIA kernel driver, physical GPUs/topology, system CUDA toolkit and `nvcc`, system NCCL/cuDNN, GCC/G++, glibc/libstdc++, CMake, system Ninja/Rust/headers, `/dev/shm`, disk and network configuration cannot be placed in a requirements file. `cuda-python` and `nvidia-*-cu12` wheels do not replace those system components.

## Unresolved target lock gates

The teammate must verify the exact Python 3.11 patch, glibc/architecture, torch 2.6.0+cu124 wheel, driver/runtime, NCCL, Ray, NumPy/PyArrow/TensorDict/TorchData combination, sgl-kernel 0.1.0 versus 0.1.1, FlashAttention 2.7.3, FlashInfer 0.2.3, SGLang ABI and three-GPU P2P. Until the returned `requirements_validation.json` passes imports, `pip check`, kernel probes, single-GPU smoke and three-GPU smoke, the status remains `target_machine_test_deferred`, never `requirements_lock_verified`.
