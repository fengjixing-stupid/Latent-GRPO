"""Writer-facing Stage 1/2 field manifest; no synthetic metrics tables."""

from pathlib import Path
from typing import Optional

_FORBIDDEN_SUBSTRINGS = ("gradient", "full_logits", "full_vocab", "hidden_state", "embedding", "tensor", "component_log_prob",
                         "model_export", "hub_url", "artifact_ref")


def persistent_field_is_allowed(name: str) -> bool:
    return not any(value in name.lower() for value in _FORBIDDEN_SUBSTRINGS)


def _field(name: str, logical_type: str, stage: int, *, nullable: bool = True, family: Optional[str] = None) -> dict:
    if not persistent_field_is_allowed(name):
        raise ValueError(f"prohibited persistent field: {name}")
    return {"name": name, "logical_type": logical_type, "physical_type": logical_type, "nullable": nullable,
            "stage": stage, "record_type": "dynamic_metric", "unit": "count" if name.endswith("count") else "scalar",
            "definition_version": "metrics_schema_v1", "primary_key_member": False, "availability_family": family}


def _observed_metric(name: str, stage: int, family: str, logical_type: str = "float64") -> list[dict]:
    return [_field(name, logical_type, stage, family=family), _field(f"{name}__available", "bool", stage, nullable=False, family=family),
            _field(f"{name}__unavailable_reason", "string", stage, family=family)]


def _rtm_fields_by_table() -> dict[str, list[dict]]:
    """Use RTM's table-qualified inventory for deferred persistent schemas."""
    root = Path(__file__).resolve().parents[2]
    path = root / "docs" / "requirements_traceability_matrix.md"
    records: dict[str, list[dict]] = {}
    if not path.is_file():
        return records
    type_map = {"bool": "bool", "int64": "int64", "int32": "int32", "float64": "float64", "float32": "float32", "string": "string"}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 14:
            continue
        name, table, schema_type = (cells[0].strip("`"), cells[7].strip("`"), cells[8])
        if table in {"memory_only_not_persisted", "schema_manifest_or_memory_only"} or not persistent_field_is_allowed(name):
            continue
        logical_type = next((value for key, value in type_map.items() if key in schema_type), "string")
        records.setdefault(table, []).append(_field(name, logical_type, 3, family="deferred"))
    return records


def schema_manifest() -> dict:
    common = [
        _field("profile_name", "string", 1, nullable=False), _field("seed", "int64", 1, nullable=False),
        _field("metric_scope", "string", 1, nullable=False), _field("global_step", "int64", 1, nullable=False),
        _field("optimizer_step", "int64", 1, nullable=False), _field("observation_phase", "string", 1, nullable=False),
        _field("learning_rate", "float64", 1), _field("wall_clock_seconds", "float64", 1),
        _field("cumulative_train_samples", "int64", 1), _field("cumulative_rollout_tokens", "int64", 1),
        _field("cumulative_gpu_hours", "float64", 1), _field("is_resume_run", "bool", 1, nullable=False),
        _field("resume_from_step", "int64", 1), _field("aggregation_worker_count", "int64", 1),
        _field("record_version", "string", 1, nullable=False), _field("metrics_compute_time", "float64", 1),
        _field("metrics_write_time", "float64", 1), _field("record_available", "bool", 1, nullable=False),
        _field("record_unavailable_reason", "string", 1),
    ]
    step = list(common)
    for name, family in [
        ("train/policy_loss", "policy_loss"), ("train/entropy", "entropy"), ("train/kl", "kl"),
        ("train/clip_fraction", "clip_fraction"), ("train/importance_ratio_mean", "importance_ratio"),
        ("train/importance_ratio_std", "importance_ratio"), ("train/response_length", "response_length"),
        ("train/latent_length", "latent_length"), ("train/generated_token_count", "generated_token_count"),
        ("train/step_time", "step_time"),
    ]:
        step.extend(_observed_metric(name, 1, family, "int64" if name == "train/generated_token_count" else "float64"))
    step.extend([_field(name, "int64", 1, nullable=False, family=family) for name, family in [
        ("train/policy_loss_count", "policy_loss"), ("train/entropy_count", "entropy"), ("train/kl_count", "kl"),
        ("train/clip_fraction_count", "clip_fraction"), ("train/importance_ratio_count", "importance_ratio"),
        ("train/response_length_count", "response_length"), ("train/latent_length_count", "latent_length"),
        ("final_training_trajectory_count", "generated_token_count"),
    ]])
    step.extend([_field(name, "string", 1) for name in [
        "generated_token_count_definition_version", "generated_token_count_scope", "entropy_source",
        "entropy_probability_space", "entropy_mask_definition", "entropy_definition_version",
        "response_length_definition_version", "latent_length_definition_version", "length_counting_rule_version",
    ]])
    step.extend([
        _field("train_core_available", "bool", 1, nullable=False, family="train_core"),
        _field("train_core_unavailable_reason", "string", 1, family="train_core"),
    ])
    for name, family in [
        ("mixture/effective_k_noisy", "mixture"), ("mixture/top1_weight_noisy", "mixture"),
        ("mask/zero_advantage_rate", "mask"), ("signal/reward_mean", "signal"),
        ("signal/reward_std", "signal"), ("signal/advantage_std", "signal"),
    ]:
        step.extend(_observed_metric(name, 2, family))
    step.extend([
        _field("mixture/noisy_count", "int64", 2, nullable=False, family="mixture"),
        _field("mixture_available", "bool", 2, nullable=False, family="mixture"), _field("mixture_unavailable_reason", "string", 2, family="mixture"),
        _field("mask/eligible_latent_token_count", "int64", 2, nullable=False, family="mask"),
        _field("mask_available", "bool", 2, nullable=False, family="mask"), _field("mask_unavailable_reason", "string", 2, family="mask"),
        _field("signal/reward_count", "int64", 2, nullable=False, family="signal"), _field("signal/advantage_count", "int64", 2, nullable=False, family="signal"),
        _field("signal_available", "bool", 2, nullable=False, family="signal"), _field("signal_unavailable_reason", "string", 2, family="signal"),
        _field("stage2_available", "bool", 2, nullable=False, family="stage2"), _field("stage2_unavailable_reason", "string", 2, family="stage2"),
    ])
    group = [
        _field("profile_name", "string", 2, nullable=False), _field("seed", "int64", 2, nullable=False),
        _field("metric_scope", "string", 2, nullable=False), _field("global_step", "int64", 2, nullable=False),
        _field("optimizer_step", "int64", 2, nullable=False), _field("observation_phase", "string", 2, nullable=False),
        _field("learning_rate", "float64", 2), _field("wall_clock_seconds", "float64", 2),
        _field("cumulative_train_samples", "int64", 2), _field("cumulative_rollout_tokens", "int64", 2),
        _field("cumulative_gpu_hours", "float64", 2), _field("is_resume_run", "bool", 2, nullable=False),
        _field("resume_from_step", "int64", 2), _field("aggregation_worker_count", "int64", 2),
        _field("record_version", "string", 2, nullable=False), _field("metrics_compute_time", "float64", 2),
        _field("metrics_write_time", "float64", 2),
        _field("group_id", "string", 2, nullable=False), _field("prompt_id_or_hash", "string", 2, nullable=False),
    ]
    group.extend([_field(name, typ, 2, nullable=False if name.endswith("count") else True, family="group") for name, typ in [
        ("group/trajectory_count", "int64"), ("group/correct_trajectory_count", "int64"), ("group/non_correct_trajectory_count", "int64"),
        ("group/overlong_trajectory_count", "int64"), ("group/overlong_generated_token_count", "int64"),
        ("group/overlong_response_length_max", "int64"), ("group/zero_variance_reward", "bool"),
        ("optimal_correct_trajectory_id", "int64"), ("optimal_correct_mean_old_log_prob", "float64"),
    ]])
    group.extend([_field(name, "string", 2) for name in ["group_definition_version", "trajectory_classification_version", "overlong_definition_version"]])
    group.extend([_field("group_available", "bool", 2, nullable=False, family="group"), _field("group_unavailable_reason", "string", 2, family="group"),
                  _field("optimal_correct_path_available", "bool", 2, nullable=False, family="optimal_correct_path"),
                  _field("optimal_correct_path_unavailable_reason", "string", 2, family="optimal_correct_path"),
                  _field("record_available", "bool", 2, nullable=False), _field("record_unavailable_reason", "string", 2)])
    gumbel = [_field("profile_name", "string", 2, nullable=False), _field("seed", "int64", 2, nullable=False),
              _field("diagnostic_run_id", "string", 2, nullable=False), _field("diagnostic_batch_index", "int64", 2, nullable=False),
              _field("gumbel_diagnostics_enabled", "bool", 2, nullable=False),
              _field("gumbel_diagnostics_mode", "string", 2, nullable=False)]
    for name in ["gumbel/raw_mean", "gumbel/raw_std", "gumbel/lower_clip_rate", "gumbel/upper_clip_rate", "gumbel/zero_rate"]:
        gumbel.extend(_observed_metric(name, 2, "gumbel"))
    gumbel.extend([_field("gumbel/raw_count", "int64", 2, nullable=False, family="gumbel"), _field("gumbel/one_sided_count", "int64", 2, nullable=False, family="gumbel"),
                   _field("gumbel_compute_time_seconds", "float64", 2),
                   _field("record_available", "bool", 2, nullable=False), _field("record_unavailable_reason", "string", 2),
                   _field("gumbel_available", "bool", 2, nullable=False, family="gumbel"), _field("gumbel_unavailable_reason", "string", 2, family="gumbel")])
    deferred = {
        "status": "target_machine_test_deferred", "deferred_reason": "runtime_collection_not_executed_on_mac",
        "fields": [],
    }
    deferred_tables = {
        "eval_question_results": ["profile_name", "seed", "checkpoint_step", "question_id", "generation_id"],
        "eval_clean_topk": ["profile_name", "seed", "checkpoint_step", "question_id", "generation_id", "latent_position"],
        "support_metrics": ["profile_name", "seed", "global_step", "group_id", "trajectory_id", "trajectory_class"],
        "support_benchmark_metrics": ["profile_name", "seed", "global_step"],
        "probe_metrics": ["profile_name", "seed", "checkpoint_step", "probe_batch_id", "trajectory_group", "latent_position_group"],
        "probe_benchmark_metrics": ["profile_name", "seed", "checkpoint_step", "probe_batch_id"],
    }
    tables = {
        "train_step_metrics": {"primary_key": ["profile_name", "seed", "global_step"], "fields": step},
        "train_group_metrics": {"primary_key": ["profile_name", "seed", "global_step", "group_id"], "fields": group},
        "gumbel_diagnostics": {"primary_key": ["profile_name", "seed", "diagnostic_run_id", "diagnostic_batch_index"], "fields": gumbel},
        "eval_dataset_manifest.parquet": {
            "primary_key": ["eval_dataset_name", "eval_dataset_version", "question_id"],
            "fields": [_field(name, "string", 1, nullable=False) for name in (
                "eval_dataset_name", "eval_dataset_version", "question_id", "prompt_hash",
                "reference_answer", "reference_answer_hash",
            )],
        },
    }
    deferred_fields = _rtm_fields_by_table()
    tables.update({name: {**deferred, "primary_key": primary_key, "fields": deferred_fields.get(name, [])} for name, primary_key in deferred_tables.items()})
    return {"metrics_schema_version": "metrics_schema_v1", "tables": tables, "stages": {"stage1": {"status": "enabled"}, "stage2": {"status": "enabled"},
                     "stage3": {"status": "deferred", "reason": "runtime_alignment_probe_required"},
                     "stage4": {"status": "disabled", "reason": "checkpoint_probe_and_backward_deferred"}},
        "persistence_policy": {"forbidden": ["full_vocabulary_logits", "full_hidden_states", "full_gradients", "gradient_norm"], "only_detached_scalar_statistics": True}}
