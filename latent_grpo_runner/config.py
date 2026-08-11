"""Strict profile parsing and upstream Hydra override construction.

This module deliberately has no training-runtime imports.  In particular, it
must remain importable on a CUDA-less Mac during ``--dry-run``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when a runner profile is unsafe or internally inconsistent."""


SUPPORTED_PROFILES = frozenset(
    {
        "smoke",
        "3gpu-low",
        "3gpu-high-smoke",
        "3gpu-final-low",
        "3gpu-final-validation",
        "3gpu-final-high",
        "3gpu-final-high-validation",
        "kaggle-t4-monitor",
        "kaggle-t4-30-metric",
    }
)
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "profile_name",
        "profile_kind",
        "description",
        "launcher",
        "target_hardware",
        "batch",
        "model",
        "data",
        "rollout",
        "training",
        "paths",
        "features",
        "upstream_overrides",
    }
)
_NESTED_KEYS = {
    "launcher": frozenset({"mode"}),
    "target_hardware": frozenset({"required_gpus", "min_vram_gb"}),
    "batch": frozenset(
        {
            "prompt_batch",
            "rollout_n",
            "mini_prompt_batch",
            "actor_micro_batch_per_gpu",
            "rollout_log_prob_micro_batch_per_gpu",
            "ref_log_prob_micro_batch_per_gpu",
            "ppo_epochs",
        }
    ),
    "model": frozenset(
        {
            "path",
            "latent_end_token_id",
            "latent_end_token",
            "latent_end_source",
            "use_remove_padding",
            "enable_gradient_checkpointing",
            "actor_param_offload",
            "actor_optimizer_offload",
            "ref_param_offload",
            "use_kl_loss",
        }
    ),
    "data": frozenset({
        "train_files",
        "val_files",
        "max_prompt_length",
        "max_response_length",
        "filter_overlong_prompts",
        "filter_overlong_prompts_workers",
    }),
    "rollout": frozenset(
        {
            "name",
            "dtype",
            "attention_backend",
            "tensor_parallel_size",
            "sequence_parallel_size",
            "max_model_len",
            "max_num_batched_tokens",
            "gpu_memory_utilization",
            "max_topk",
            "temperature",
            "top_p",
            "top_k",
            "gumbel_softmax_temperature",
            "noise_scale",
            "enable_latent",
            "add_noise_gumbel_softmax",
            "use_one_sided_gumbel_noise",
        }
    ),
    "training": frozenset(
        {"max_steps", "seed", "filter_groups_max_num_gen_batches", "pre_backward_monitor_probe"}
    ),
    "paths": frozenset({"upstream_repo_path", "output_root", "cache_root"}),
    "features": frozenset({"metrics_enabled", "support_enabled", "checkpoint_probe_enabled", "credit_probe_enabled"}),
}

_ALLOWED_UPSTREAM_OVERRIDES = frozenset(
    {
        "data.val_batch_size",
        "actor_rollout_ref.actor.optim.lr",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu",
        "actor_rollout_ref.actor.kl_loss_coef",
        "actor_rollout_ref.actor.neg_adv_weight",
        "actor_rollout_ref.actor.kl_loss_type",
        "actor_rollout_ref.actor.entropy_coeff",
        "actor_rollout_ref.actor.freeze_embedding",
        "actor_rollout_ref.rollout.val_kwargs.do_sample",
        "actor_rollout_ref.rollout.val_kwargs.temperature",
        "actor_rollout_ref.rollout.val_kwargs.top_p",
        "actor_rollout_ref.rollout.val_kwargs.top_k",
        "actor_rollout_ref.ref.strategy",
        "algorithm.use_kl_in_reward",
        "algorithm.exclude_overlong_samples_from_advantage",
        "trainer.critic_warmup",
        "trainer.logger",
        "trainer.project_name",
        "trainer.experiment_name",
        "trainer.val_before_train",
        "trainer.save_freq",
        "trainer.test_freq",
        "trainer.balance_batch",
        "trainer.total_epochs",
    }
)
_TOPOLOGY_OWNED_UPSTREAM_OVERRIDES = frozenset(
    {
        "data.train_batch_size",
        "actor_rollout_ref.actor.ppo_mini_batch_size",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
        "actor_rollout_ref.rollout.n",
        "trainer.n_gpus_per_node",
        "trainer.nnodes",
        "trainer.total_training_steps",
    }
)


@dataclass(frozen=True)
class LauncherConfig:
    mode: str


@dataclass(frozen=True)
class HardwareConfig:
    required_gpus: int
    min_vram_gb: int


@dataclass(frozen=True)
class BatchConfig:
    prompt_batch: int
    rollout_n: int
    mini_prompt_batch: int
    actor_micro_batch_per_gpu: int
    rollout_log_prob_micro_batch_per_gpu: int
    ref_log_prob_micro_batch_per_gpu: int
    ppo_epochs: int


@dataclass(frozen=True)
class ModelConfig:
    path: str
    latent_end_token_id: int
    latent_end_token: str
    latent_end_source: str
    use_remove_padding: bool
    enable_gradient_checkpointing: bool
    actor_param_offload: bool
    actor_optimizer_offload: bool
    ref_param_offload: bool
    use_kl_loss: bool


@dataclass(frozen=True)
class DataConfig:
    train_files: str
    val_files: str
    max_prompt_length: int
    max_response_length: int
    filter_overlong_prompts: bool | None = None
    filter_overlong_prompts_workers: int | None = None


@dataclass(frozen=True)
class RolloutConfig:
    name: str
    dtype: str
    attention_backend: str | None
    tensor_parallel_size: int
    sequence_parallel_size: int
    max_model_len: int
    max_num_batched_tokens: int
    gpu_memory_utilization: float
    max_topk: int
    temperature: float
    top_p: float
    top_k: int
    gumbel_softmax_temperature: float
    noise_scale: float
    enable_latent: bool
    add_noise_gumbel_softmax: bool
    use_one_sided_gumbel_noise: bool


@dataclass(frozen=True)
class TrainingConfig:
    max_steps: int
    seed: int
    filter_groups_max_num_gen_batches: int
    pre_backward_monitor_probe: bool = False


@dataclass(frozen=True)
class FeatureConfig:
    metrics_enabled: bool
    support_enabled: bool
    checkpoint_probe_enabled: bool
    credit_probe_enabled: bool


@dataclass(frozen=True)
class ResolvedConfig:
    profile_name: str
    profile_kind: str
    description: str
    launcher: LauncherConfig
    hardware: HardwareConfig
    batch: BatchConfig
    model: ModelConfig
    data: DataConfig
    rollout: RolloutConfig
    training: TrainingConfig
    paths: Mapping[str, Path]
    features: FeatureConfig
    upstream_overrides: Mapping[str, Any]
    workspace_root: Path
    resume_from: Path | None = None

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self._hashable_mapping(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @property
    def resume_compatibility_hash(self) -> str:
        """Hash training semantics while excluding only the checkpoint locator.

        A resumed run must be able to validate against the sidecar written by
        the original run even though ``resume_from`` itself is necessarily a
        new runtime locator.  Output root and all training/metrics semantics
        remain hash-bound.
        """
        mapping = self._hashable_mapping()
        mapping["resume_from"] = None
        return hashlib.sha256(
            json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _hashable_mapping(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "profile_kind": self.profile_kind,
            "launcher": {"mode": self.launcher.mode},
            "hardware": {"required_gpus": self.hardware.required_gpus, "min_vram_gb": self.hardware.min_vram_gb},
            "batch": self.batch.__dict__,
            "model": self.model.__dict__,
            "data": {key: value for key, value in self.data.__dict__.items() if value is not None},
            "rollout": self.rollout.__dict__,
            "training": self.training.__dict__,
            "paths": {key: _safe_relative_path(value, self.workspace_root) for key, value in self.paths.items()},
            "features": self.features.__dict__,
            "upstream_overrides": dict(self.upstream_overrides),
            "resume_from": (
                None if self.resume_from is None else _safe_relative_path(self.resume_from, self.workspace_root)
            ),
        }

    def batch_arithmetic(self) -> tuple[int, int, int, int]:
        """Return (local batch, local mini, accumulation, optimizer attempts)."""
        divisor = self.hardware.required_gpus // self.rollout.sequence_parallel_size
        total = self.batch.prompt_batch * self.batch.rollout_n
        local_batch = total // divisor
        local_mini = (self.batch.mini_prompt_batch * self.batch.rollout_n) // divisor
        accumulation = local_mini // self.batch.actor_micro_batch_per_gpu
        attempts = self.batch.ppo_epochs * self.batch.prompt_batch // self.batch.mini_prompt_batch
        return local_batch, local_mini, accumulation, attempts

    def author_hydra_overrides(self) -> tuple[str, ...]:
        """Map runner fields only to Hydra keys observed in the vendored config."""
        values = {
            "algorithm.adv_estimator": "grpo",
            "data.train_files": self.data.train_files,
            "data.val_files": self.data.val_files,
            "data.train_batch_size": self.batch.prompt_batch,
            "data.max_prompt_length": self.data.max_prompt_length,
            "data.max_response_length": self.data.max_response_length,
            "actor_rollout_ref.model.path": self.model.path,
            "actor_rollout_ref.model.use_remove_padding": self.model.use_remove_padding,
            "actor_rollout_ref.model.enable_gradient_checkpointing": self.model.enable_gradient_checkpointing,
            "actor_rollout_ref.actor.ppo_mini_batch_size": self.batch.mini_prompt_batch,
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": self.batch.actor_micro_batch_per_gpu,
            "actor_rollout_ref.actor.ppo_epochs": self.batch.ppo_epochs,
            "actor_rollout_ref.actor.use_kl_loss": self.model.use_kl_loss,
            "actor_rollout_ref.actor.fsdp_config.param_offload": self.model.actor_param_offload,
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload": self.model.actor_optimizer_offload,
            "actor_rollout_ref.actor.ulysses_sequence_parallel_size": self.rollout.sequence_parallel_size,
            "actor_rollout_ref.rollout.name": self.rollout.name,
            "actor_rollout_ref.rollout.dtype": self.rollout.dtype,
            "actor_rollout_ref.rollout.tensor_model_parallel_size": self.rollout.tensor_parallel_size,
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": self.batch.rollout_log_prob_micro_batch_per_gpu,
            "actor_rollout_ref.rollout.max_model_len": self.rollout.max_model_len,
            "actor_rollout_ref.rollout.max_num_batched_tokens": self.rollout.max_num_batched_tokens,
            "actor_rollout_ref.rollout.gpu_memory_utilization": self.rollout.gpu_memory_utilization,
            "actor_rollout_ref.rollout.max_topk": self.rollout.max_topk,
            "actor_rollout_ref.rollout.temperature": self.rollout.temperature,
            "actor_rollout_ref.rollout.top_p": self.rollout.top_p,
            "actor_rollout_ref.rollout.top_k": self.rollout.top_k,
            "actor_rollout_ref.rollout.gumbel_softmax_temperature": self.rollout.gumbel_softmax_temperature,
            "actor_rollout_ref.rollout.noise_scale": self.rollout.noise_scale,
            "actor_rollout_ref.rollout.enable_latent": self.rollout.enable_latent,
            "actor_rollout_ref.rollout.latent_end_token_id": self.model.latent_end_token_id,
            "actor_rollout_ref.rollout.add_noise_gumbel_softmax": self.rollout.add_noise_gumbel_softmax,
            "actor_rollout_ref.rollout.use_one_sided_gumbel_noise": self.rollout.use_one_sided_gumbel_noise,
            "actor_rollout_ref.rollout.n": self.batch.rollout_n,
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": self.batch.ref_log_prob_micro_batch_per_gpu,
            "actor_rollout_ref.ref.fsdp_config.param_offload": self.model.ref_param_offload,
            "algorithm.filter_groups.enable": True,
            "algorithm.filter_groups.max_num_gen_batches": self.training.filter_groups_max_num_gen_batches,
            "trainer.n_gpus_per_node": self.hardware.required_gpus,
            "trainer.nnodes": 1,
            "trainer.default_local_dir": self.paths["output_root"],
        }
        if self.profile_kind != "formal_training":
            values["trainer.total_training_steps"] = self.training.max_steps
        values.update(self.upstream_overrides)
        if self.data.filter_overlong_prompts is not None:
            values["data.filter_overlong_prompts"] = self.data.filter_overlong_prompts
        if self.data.filter_overlong_prompts_workers is not None:
            values["data.filter_overlong_prompts_workers"] = self.data.filter_overlong_prompts_workers
        if self.rollout.attention_backend is not None:
            # Attention backend is an engineering/runtime knob. Deliberately do
            # not expose sampling_backend here: the author latent/response sampler
            # remains the canonical flashinfer path.
            values["actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend"] = (
                self.rollout.attention_backend
            )
        if self.training.pre_backward_monitor_probe:
            # Engineering-only real-intermediate probe.  The vendored actor is
            # required to stop before backward/optimizer.step; validation and
            # checkpointing are disabled so this can never masquerade as a train step.
            values["trainer.pre_backward_monitor_probe"] = True
            values["trainer.val_before_train"] = False
            values["trainer.test_freq"] = -1
            values["trainer.save_freq"] = -1
            values["trainer.logger"] = "[console]"
        elif self.profile_name == "kaggle-t4-30-metric":
            values["trainer.val_before_train"] = False
            values["trainer.test_freq"] = -1
            values["trainer.save_freq"] = 1
            values["trainer.logger"] = "[console]"
            values["actor_rollout_ref.actor.checkpoint.contents"] = "[model,extra]"
            values["trainer.resume_mode"] = "disable"
        if self.resume_from is not None:
            values["trainer.resume_mode"] = "resume_path"
            values["trainer.resume_from_path"] = self.resume_from
        return tuple(f"{key}={_hydra_value(value)}" for key, value in values.items())

    def with_launcher_mode(self, mode: str) -> "ResolvedConfig":
        if mode not in {"ray_direct", "torchrun_control"}:
            raise ConfigError(f"unsupported launcher mode: {mode}")
        return replace(self, launcher=LauncherConfig(mode=mode))

    def with_runtime_overrides(
        self,
        *,
        seed: int | None = None,
        output_root: Path | None = None,
        max_steps: int | None = None,
        resume_from: Path | None = None,
        model_path: str | None = None,
        train_files: str | None = None,
        val_files: str | None = None,
        support_enabled: bool | None = None,
        metrics_enabled: bool | None = None,
        checkpoint_probe_enabled: bool | None = None,
        credit_probe_enabled: bool | None = None,
    ) -> "ResolvedConfig":
        if self.profile_name == "kaggle-t4-30-metric" and resume_from is not None:
            raise ConfigError("kaggle-t4-30-metric lightweight checkpoint does not support checkpoint resume")
        if self.profile_kind == "formal_training" and max_steps is not None:
            raise ConfigError(
                "formal training is epoch-controlled and rejects --max-steps; "
                "use the corresponding *-validation.yaml profile for bounded execution"
            )
        if self.training.pre_backward_monitor_probe:
            if max_steps not in {None, 1}:
                raise ConfigError("kaggle-t4-monitor cannot override max_steps beyond 1")
            if resume_from is not None:
                raise ConfigError("kaggle-t4-monitor does not support checkpoint resume")
            if metrics_enabled is False:
                raise ConfigError("kaggle-t4-monitor requires metrics_enabled=true")
            if support_enabled is True or checkpoint_probe_enabled is True or credit_probe_enabled is True:
                raise ConfigError("kaggle-t4-monitor forbids extra support/checkpoint/credit probes")
        updated_training = replace(
            self.training,
            seed=self.training.seed if seed is None else seed,
            max_steps=self.training.max_steps if max_steps is None else max_steps,
        )
        updated_features = replace(
            self.features,
            metrics_enabled=self.features.metrics_enabled if metrics_enabled is None else metrics_enabled,
            support_enabled=self.features.support_enabled if support_enabled is None else support_enabled,
            checkpoint_probe_enabled=(
                self.features.checkpoint_probe_enabled if checkpoint_probe_enabled is None else checkpoint_probe_enabled
            ),
            credit_probe_enabled=(
                self.features.credit_probe_enabled if credit_probe_enabled is None else credit_probe_enabled
            ),
        )
        updated_model = replace(
            self.model,
            path=self.model.path if model_path is None else str(model_path),
        )
        updated_data = replace(
            self.data,
            train_files=self.data.train_files if train_files is None else str(train_files),
            val_files=self.data.val_files if val_files is None else str(val_files),
        )
        updated_paths = dict(self.paths)
        if output_root is not None:
            updated_paths["output_root"] = _resolve_workspace_path(self.workspace_root, str(output_root))
        resolved = replace(
            self,
            training=updated_training,
            model=updated_model,
            data=updated_data,
            features=updated_features,
            paths=updated_paths,
            resume_from=(None if resume_from is None else _resolve_workspace_path(self.workspace_root, str(resume_from))),
        )
        _validate_semantics(resolved)
        return resolved


def load_config(path: str | Path, *, workspace_root: str | Path | None = None) -> ResolvedConfig:
    """Load and strictly validate a runner YAML profile without importing training code."""
    config_path = Path(path).expanduser().resolve()
    root = Path(workspace_root).expanduser().resolve() if workspace_root else config_path.parent.parent.resolve()
    raw = _load_yaml(config_path)
    _validate_structure(raw)
    config = _build_config(raw, root)
    _validate_semantics(config)
    return config


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as error:  # pragma: no cover - exercised on minimal target installs
        raise ConfigError("PyYAML is required to parse runner profiles") from error
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"cannot read config: {path}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML: {path}") from error
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    return raw


def _validate_structure(raw: Mapping[str, Any]) -> None:
    _unknown_keys("config", raw, _TOP_LEVEL_KEYS)
    if _expect_int(raw, "schema_version") != 1:
        raise ConfigError("schema_version must be 1")
    for section, allowed in _NESTED_KEYS.items():
        value = raw.get(section)
        if not isinstance(value, dict):
            raise ConfigError(f"{section} must be a mapping")
        _unknown_keys(section, value, allowed)
    overrides = raw.get("upstream_overrides", {})
    if not isinstance(overrides, dict):
        raise ConfigError("upstream_overrides must be a mapping")


def _unknown_keys(label: str, values: Mapping[str, Any], allowed: frozenset[str]) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ConfigError(f"unknown field in {label}: {', '.join(unknown)}")


def _build_config(raw: Mapping[str, Any], root: Path) -> ResolvedConfig:
    try:
        paths = {key: _resolve_workspace_path(root, value) for key, value in raw["paths"].items()}
        return ResolvedConfig(
            profile_name=_expect_str(raw, "profile_name"),
            profile_kind=_expect_str(raw, "profile_kind"),
            description=_expect_str(raw, "description"),
            launcher=LauncherConfig(mode=_expect_str(raw["launcher"], "mode")),
            hardware=HardwareConfig(
                required_gpus=_expect_int(raw["target_hardware"], "required_gpus"),
                min_vram_gb=_expect_int(raw["target_hardware"], "min_vram_gb"),
            ),
            batch=BatchConfig(
                prompt_batch=_expect_int(raw["batch"], "prompt_batch"),
                rollout_n=_expect_int(raw["batch"], "rollout_n"),
                mini_prompt_batch=_expect_int(raw["batch"], "mini_prompt_batch"),
                actor_micro_batch_per_gpu=_expect_int(raw["batch"], "actor_micro_batch_per_gpu"),
                rollout_log_prob_micro_batch_per_gpu=_expect_int(raw["batch"], "rollout_log_prob_micro_batch_per_gpu"),
                ref_log_prob_micro_batch_per_gpu=_expect_int(raw["batch"], "ref_log_prob_micro_batch_per_gpu"),
                ppo_epochs=_expect_int(raw["batch"], "ppo_epochs"),
            ),
            model=ModelConfig(
                path=_expect_str(raw["model"], "path"),
                latent_end_token_id=_expect_int(raw["model"], "latent_end_token_id"),
                latent_end_token=_expect_str(raw["model"], "latent_end_token"),
                latent_end_source=_expect_str(raw["model"], "latent_end_source"),
                use_remove_padding=_expect_bool(raw["model"], "use_remove_padding"),
                enable_gradient_checkpointing=_expect_bool(raw["model"], "enable_gradient_checkpointing"),
                actor_param_offload=_expect_bool(raw["model"], "actor_param_offload"),
                actor_optimizer_offload=_expect_bool(raw["model"], "actor_optimizer_offload"),
                ref_param_offload=_expect_bool(raw["model"], "ref_param_offload"),
                use_kl_loss=_expect_bool(raw["model"], "use_kl_loss"),
            ),
            data=DataConfig(
                train_files=_expect_str(raw["data"], "train_files"),
                val_files=_expect_str(raw["data"], "val_files"),
                max_prompt_length=_expect_int(raw["data"], "max_prompt_length"),
                max_response_length=_expect_int(raw["data"], "max_response_length"),
                filter_overlong_prompts=(
                    _expect_bool(raw["data"], "filter_overlong_prompts")
                    if "filter_overlong_prompts" in raw["data"]
                    else None
                ),
                filter_overlong_prompts_workers=(
                    _expect_int(raw["data"], "filter_overlong_prompts_workers")
                    if "filter_overlong_prompts_workers" in raw["data"]
                    else None
                ),
            ),
            rollout=RolloutConfig(
                name=_expect_str(raw["rollout"], "name"),
                dtype=_expect_str(raw["rollout"], "dtype"),
                attention_backend=(
                    None
                    if raw["rollout"].get("attention_backend") is None
                    else _expect_str(raw["rollout"], "attention_backend")
                ),
                tensor_parallel_size=_expect_int(raw["rollout"], "tensor_parallel_size"),
                sequence_parallel_size=_expect_int(raw["rollout"], "sequence_parallel_size"),
                max_model_len=_expect_int(raw["rollout"], "max_model_len"),
                max_num_batched_tokens=_expect_int(raw["rollout"], "max_num_batched_tokens"),
                gpu_memory_utilization=_expect_float(raw["rollout"], "gpu_memory_utilization"),
                max_topk=_expect_int(raw["rollout"], "max_topk"),
                temperature=_expect_float(raw["rollout"], "temperature"),
                top_p=_expect_float(raw["rollout"], "top_p"),
                top_k=_expect_int(raw["rollout"], "top_k"),
                gumbel_softmax_temperature=_expect_float(raw["rollout"], "gumbel_softmax_temperature"),
                noise_scale=_expect_float(raw["rollout"], "noise_scale"),
                enable_latent=_expect_bool(raw["rollout"], "enable_latent"),
                add_noise_gumbel_softmax=_expect_bool(raw["rollout"], "add_noise_gumbel_softmax"),
                use_one_sided_gumbel_noise=_expect_bool(raw["rollout"], "use_one_sided_gumbel_noise"),
            ),
            training=TrainingConfig(
                max_steps=_expect_int(raw["training"], "max_steps"),
                seed=_expect_int(raw["training"], "seed"),
                filter_groups_max_num_gen_batches=_expect_int(raw["training"], "filter_groups_max_num_gen_batches"),
                pre_backward_monitor_probe=(
                    _expect_bool(raw["training"], "pre_backward_monitor_probe")
                    if "pre_backward_monitor_probe" in raw["training"]
                    else False
                ),
            ),
            paths=paths,
            features=FeatureConfig(
                metrics_enabled=_expect_bool(raw["features"], "metrics_enabled"),
                support_enabled=_expect_bool(raw["features"], "support_enabled"),
                checkpoint_probe_enabled=_expect_bool(raw["features"], "checkpoint_probe_enabled"),
                credit_probe_enabled=_expect_bool(raw["features"], "credit_probe_enabled"),
            ),
            upstream_overrides=_build_upstream_overrides(raw.get("upstream_overrides", {})),
            workspace_root=root,
        )
    except KeyError as error:
        raise ConfigError(f"missing required field: {error.args[0]}") from error
    except TypeError as error:
        raise ConfigError(f"invalid config field type: {error}") from error


def _validate_semantics(config: ResolvedConfig) -> None:
    if config.profile_name.startswith("paper-") or config.profile_kind == "paper":
        raise ConfigError("paper profiles are not accepted as engineering profiles")
    _validate_upstream_overrides(config.upstream_overrides)
    if config.profile_name not in SUPPORTED_PROFILES:
        raise ConfigError(f"unsupported profile: {config.profile_name}")
    if config.launcher.mode not in {"ray_direct", "torchrun_control"}:
        raise ConfigError("launcher.mode must be ray_direct or torchrun_control")
    if config.rollout.name != "sglang":
        raise ConfigError("Latent-GRPO profiles must use the observed sglang rollout")
    if config.rollout.attention_backend not in {None, "flashinfer", "triton", "torch_native"}:
        raise ConfigError("rollout.attention_backend must be flashinfer, triton, torch_native, or omitted")
    if config.rollout.tensor_parallel_size != 1 or config.rollout.sequence_parallel_size != 1:
        raise ConfigError("initial profiles require TP=1 and SP=1")
    if config.profile_name.startswith("3gpu-") and config.hardware.required_gpus != 3:
        raise ConfigError("3gpu profile requires exactly 3 GPUs")
    if config.profile_name == "smoke" and config.hardware.required_gpus != 1:
        raise ConfigError("smoke profile requires exactly 1 GPU")
    if config.profile_name == "kaggle-t4-monitor":
        if config.profile_kind != "monitor_probe":
            raise ConfigError("kaggle-t4-monitor must use profile_kind=monitor_probe")
        if config.hardware.required_gpus != 2:
            raise ConfigError("kaggle-t4-monitor requires exactly 2 GPUs")
        if not config.training.pre_backward_monitor_probe:
            raise ConfigError("kaggle-t4-monitor must stop before backward")
        if config.training.max_steps != 1:
            raise ConfigError("kaggle-t4-monitor must use max_steps=1")
        if config.rollout.dtype not in {"float16", "fp16"}:
            raise ConfigError("kaggle-t4-monitor requires FP16 runtime")
        if config.model.use_remove_padding:
            raise ConfigError("kaggle-t4-monitor requires padded SDPA actor path")
        if config.rollout.attention_backend != "triton":
            raise ConfigError("kaggle-t4-monitor requires SGLang Triton attention")
        if not (
            config.model.enable_gradient_checkpointing
            and config.model.actor_param_offload
            and config.model.actor_optimizer_offload
            and config.model.ref_param_offload
        ):
            raise ConfigError("kaggle-t4-monitor requires checkpointing and all configured offloads")
        if (
            config.batch.prompt_batch != 2
            or config.batch.rollout_n != 2
            or config.batch.mini_prompt_batch != 1
            or config.batch.actor_micro_batch_per_gpu != 1
            or config.batch.rollout_log_prob_micro_batch_per_gpu != 1
            or config.batch.ref_log_prob_micro_batch_per_gpu != 1
            or config.batch.ppo_epochs != 1
        ):
            raise ConfigError("kaggle-t4-monitor requires the fixed minimal monitor batch geometry")
        if config.data.max_prompt_length > 256 or config.data.max_response_length > 32:
            raise ConfigError("kaggle-t4-monitor prompt/response limits exceed the monitor budget")
        if config.data.filter_overlong_prompts is not True:
            raise ConfigError("kaggle-t4-monitor requires filter_overlong_prompts=true")
        if config.data.filter_overlong_prompts_workers != 1:
            raise ConfigError("kaggle-t4-monitor requires exactly one overlong-prompt filter worker")
        if config.rollout.gpu_memory_utilization > 0.30:
            raise ConfigError("kaggle-t4-monitor SGLang memory utilization must stay <= 0.30")
    elif config.profile_name == "kaggle-t4-30-metric":
        if config.profile_kind != "metric_validation":
            raise ConfigError("kaggle-t4-30-metric must use profile_kind=metric_validation")
        if config.hardware.required_gpus != 2:
            raise ConfigError("kaggle-t4-30-metric requires exactly 2 GPUs")
        if config.training.pre_backward_monitor_probe or config.training.max_steps != 1:
            raise ConfigError("kaggle-t4-30-metric requires one real actor update")
        if config.rollout.dtype not in {"float16", "fp16"}:
            raise ConfigError("kaggle-t4-30-metric requires FP16 runtime")
        if config.model.use_remove_padding:
            raise ConfigError("kaggle-t4-30-metric requires padded SDPA actor path")
        if config.rollout.attention_backend != "triton":
            raise ConfigError("kaggle-t4-30-metric requires SGLang Triton attention")
        if not (
            config.model.enable_gradient_checkpointing
            and config.model.actor_param_offload
            and config.model.actor_optimizer_offload
            and config.model.ref_param_offload
        ):
            raise ConfigError("kaggle-t4-30-metric requires checkpointing and all configured offloads")
        if (
            config.batch.prompt_batch != 2
            or config.batch.rollout_n != 4
            or config.batch.mini_prompt_batch != 1
            or config.batch.actor_micro_batch_per_gpu != 1
            or config.batch.rollout_log_prob_micro_batch_per_gpu != 1
            or config.batch.ref_log_prob_micro_batch_per_gpu != 1
            or config.batch.ppo_epochs != 1
        ):
            raise ConfigError("kaggle-t4-30-metric requires the fixed minimal validation batch geometry")
        if config.data.max_prompt_length > 256 or config.data.max_response_length > 32:
            raise ConfigError("kaggle-t4-30-metric prompt/response limits exceed the T4 budget")
        if config.data.filter_overlong_prompts is not True or config.data.filter_overlong_prompts_workers != 1:
            raise ConfigError("kaggle-t4-30-metric requires one overlong-prompt filter worker")
        if config.rollout.gpu_memory_utilization > 0.25:
            raise ConfigError("kaggle-t4-30-metric SGLang memory utilization must stay <= 0.25")
        if not (
            config.features.metrics_enabled
            and config.features.support_enabled
            and config.features.checkpoint_probe_enabled
            and config.features.credit_probe_enabled
        ):
            raise ConfigError("kaggle-t4-30-metric requires all metric feature flags")
    elif config.training.pre_backward_monitor_probe:
        raise ConfigError("pre_backward_monitor_probe is reserved for kaggle-t4-monitor")
    if not config.paths["upstream_repo_path"].is_dir():
        raise ConfigError("upstream_repo_path does not exist")
    positive_values = (
        config.hardware.required_gpus,
        config.hardware.min_vram_gb,
        config.batch.prompt_batch,
        config.batch.rollout_n,
        config.batch.mini_prompt_batch,
        config.batch.actor_micro_batch_per_gpu,
        config.batch.rollout_log_prob_micro_batch_per_gpu,
        config.batch.ref_log_prob_micro_batch_per_gpu,
        config.batch.ppo_epochs,
        config.training.max_steps,
        config.data.max_prompt_length,
        config.data.max_response_length,
    )
    if any(value <= 0 for value in positive_values):
        raise ConfigError("all capacity and batch fields must be positive")
    world = config.hardware.required_gpus
    total = config.batch.prompt_batch * config.batch.rollout_n
    mini_total = config.batch.mini_prompt_batch * config.batch.rollout_n
    if total % world:
        raise ConfigError("prompt_batch * rollout_n must divide across the GPU world")
    if mini_total % world:
        raise ConfigError("mini_prompt_batch * rollout_n must divide across the GPU world")
    if config.batch.prompt_batch % config.batch.mini_prompt_batch:
        raise ConfigError("prompt_batch must divide evenly into mini_prompt_batch")
    local_batch, local_mini, _, _ = config.batch_arithmetic()
    if local_batch % local_mini:
        raise ConfigError("per-rank batch must divide evenly into per-rank mini batch")
    if local_mini % config.batch.actor_micro_batch_per_gpu:
        raise ConfigError("per-rank mini batch must divide actor micro batch")
    if local_batch % config.batch.rollout_log_prob_micro_batch_per_gpu:
        raise ConfigError("per-rank batch must divide rollout log-prob micro batch")
    if local_batch % config.batch.ref_log_prob_micro_batch_per_gpu:
        raise ConfigError("per-rank batch must divide ref log-prob micro batch")
    if config.data.max_prompt_length + config.data.max_response_length > config.rollout.max_model_len:
        raise ConfigError("max_model_len must cover prompt and response lengths")
    if config.model.latent_end_token_id < 0 or not config.model.latent_end_token:
        raise ConfigError("latent-end token configuration is incomplete")
    if config.features.credit_probe_enabled and not config.features.checkpoint_probe_enabled:
        raise ConfigError("credit_probe_enabled requires checkpoint_probe_enabled=true")
    if not 0 < config.rollout.gpu_memory_utilization <= 1:
        raise ConfigError("gpu_memory_utilization must be in (0, 1]")
    if not 0 < config.rollout.temperature or not 0 < config.rollout.gumbel_softmax_temperature:
        raise ConfigError("rollout temperatures must be positive")
    if not 0 < config.rollout.top_p <= 1:
        raise ConfigError("top_p must be in (0, 1]")


def _expect_str(values: Mapping[str, Any], key: str) -> str:
    value = values[key]
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{key} must be a non-empty string")
    return value


def _expect_int(values: Mapping[str, Any], key: str) -> int:
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} must be an integer")
    return value


def _expect_float(values: Mapping[str, Any], key: str) -> float:
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be a number")
    return float(value)


def _expect_bool(values: Mapping[str, Any], key: str) -> bool:
    value = values[key]
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be a boolean")
    return value


def _build_upstream_overrides(values: Mapping[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ConfigError("upstream override keys must be non-empty strings")
        if type(value) not in {str, int, float, bool}:
            raise ConfigError(f"upstream override {key} must be a scalar")
        if isinstance(value, str) and not value:
            raise ConfigError(f"upstream override {key} must not be empty")
        overrides[key] = value
    return overrides


def _validate_upstream_overrides(values: Mapping[str, Any]) -> None:
    topology_owned = sorted(set(values) & _TOPOLOGY_OWNED_UPSTREAM_OVERRIDES)
    if topology_owned:
        raise ConfigError(
            "topology-owned upstream override must use typed config fields: "
            + ", ".join(topology_owned)
        )
    unsupported = sorted(set(values) - _ALLOWED_UPSTREAM_OVERRIDES)
    if unsupported:
        raise ConfigError("unsupported upstream override: " + ", ".join(unsupported))


def validate_latent_end_token(model: ModelConfig, tokenizer: Any, model_config: Any) -> dict[str, Any]:
    """Fail closed unless the configured ID is the marker's first tokenizer ID."""
    if tokenizer is None or model_config is None:
        raise ConfigError("latent-end validation requires both tokenizer and model config")
    tokenizer_vocab_size = getattr(tokenizer, "vocab_size", None)
    model_vocab_size = getattr(model_config, "vocab_size", None)
    if isinstance(tokenizer_vocab_size, bool) or not isinstance(tokenizer_vocab_size, int):
        raise ConfigError("tokenizer vocab_size is unavailable for latent-end validation")
    if isinstance(model_vocab_size, bool) or not isinstance(model_vocab_size, int):
        raise ConfigError("model vocab_size is unavailable for latent-end validation")
    if model.latent_end_token_id >= tokenizer_vocab_size or model.latent_end_token_id >= model_vocab_size:
        raise ConfigError("latent-end token ID is outside tokenizer/model vocabulary")
    try:
        marker_ids = tokenizer.encode(model.latent_end_token, add_special_tokens=False)
    except Exception as error:
        raise ConfigError("tokenizer could not encode configured latent-end marker") from error
    if (
        not isinstance(marker_ids, (list, tuple))
        or not marker_ids
        or any(type(token_id) is not int for token_id in marker_ids)
    ):
        raise ConfigError("configured latent-end marker did not produce valid tokenizer IDs")
    if marker_ids[0] != model.latent_end_token_id:
        raise ConfigError(
            "configured latent-end token ID does not match the first tokenizer ID "
            "of the latent-end marker"
        )
    return {
        "latent_end_token_id": model.latent_end_token_id,
        "latent_end_token": model.latent_end_token,
        "latent_end_source": model.latent_end_source,
        "latent_end_validation_status": "validated",
    }


def _resolve_workspace_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError("paths values must be non-empty strings")
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _safe_relative_path(value: Path, root: Path) -> str:
    try:
        return str(value.relative_to(root))
    except ValueError:
        return "external:" + hashlib.sha256(str(value).encode("utf-8")).hexdigest()
