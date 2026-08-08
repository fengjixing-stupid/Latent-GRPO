"""Torch runtime collectors for P1.

Torch is imported lazily so config parsing and CPU-only dry-runs do not acquire a
new training-runtime dependency.  Every public collector returns only detached
Python scalars/lists.
"""

from __future__ import annotations

from typing import Any


def sufficient_stats_from_tensor(
    values: Any,
    *,
    mask: Any | None = None,
    numerator_mask: Any | None = None,
    definition_version: str,
) -> dict[str, int | float | str | None]:
    import torch

    if not definition_version:
        raise ValueError("definition_version must be non-empty")
    detached_values = values.detach().float()
    detached_mask = (
        torch.ones_like(detached_values, dtype=torch.bool)
        if mask is None
        else mask.detach().bool()
    )
    if detached_mask.shape != detached_values.shape:
        detached_mask = torch.broadcast_to(detached_mask, detached_values.shape)
    detached_numerator = (
        torch.zeros_like(detached_values, dtype=torch.bool)
        if numerator_mask is None
        else numerator_mask.detach().bool()
    )
    if detached_numerator.shape != detached_values.shape:
        detached_numerator = torch.broadcast_to(detached_numerator, detached_values.shape)

    finite = torch.isfinite(detached_values)
    effective = detached_mask & finite
    selected = detached_values[effective]
    count = int(effective.sum().item())
    return {
        "sum": float(selected.double().sum().item()) if count else 0.0,
        "sum_sq": float(selected.double().square().sum().item()) if count else 0.0,
        "count": count,
        "nan_count": int((detached_mask & ~finite).sum().item()),
        "masked_count": int((~detached_mask).sum().item()),
        "min": float(selected.min().item()) if count else None,
        "max": float(selected.max().item()) if count else None,
        "numerator_count": int((detached_numerator & effective).sum().item()),
        "definition_version": definition_version,
    }


def collect_driver_p1_from_tensors(
    *,
    response_mask: Any,
    rollout_topk_ids: Any,
    rollout_topk_gumbels: Any,
    gumbel_temperature: Any,
    entropies: Any,
    token_level_rewards: Any,
    advantages: Any,
    exclude_overlong_samples_from_advantage: bool,
) -> dict[str, Any]:
    """Reduce the final, already-filtered training batch on the driver."""
    import torch

    response_mask = response_mask.detach().bool()
    if response_mask.dim() != 2:
        raise ValueError("response_mask must be rank-2")
    batch_size, response_width = response_mask.shape
    for name, tensor in (
        ("entropies", entropies),
        ("token_level_rewards", token_level_rewards),
        ("advantages", advantages),
    ):
        if tuple(tensor.shape) != (batch_size, response_width):
            raise ValueError(f"{name} must align with response_mask")
    if rollout_topk_ids.dim() != 3 or rollout_topk_gumbels.shape != rollout_topk_ids.shape:
        raise ValueError("rollout Top-K ids/gumbels must be aligned rank-3 tensors")
    if rollout_topk_ids.size(0) != batch_size or rollout_topk_ids.size(1) < response_width:
        raise ValueError("rollout Top-K tensors do not contain the response domain")
    if rollout_topk_ids.size(-1) < 2:
        raise ValueError("latent Top-K requires at least two component slots")

    response_topk_ids = rollout_topk_ids.detach()[:, -response_width:, :]
    response_gumbels = rollout_topk_gumbels.detach().float()[:, -response_width:, :]
    hard_token_mask = (response_topk_ids[..., 1:] == -100).all(dim=-1)
    latent_position_mask = response_mask & ~hard_token_mask

    response_lengths = response_mask.sum(dim=-1).to(torch.int64)
    latent_lengths = latent_position_mask.sum(dim=-1).to(torch.int64)

    final_advantages = advantages.detach().float().clone()
    if not exclude_overlong_samples_from_advantage:
        is_clipped = response_lengths == response_width
        final_advantages[is_clipped] = 0.0

    sequence_rewards = (
        token_level_rewards.detach().float() * response_mask.to(token_level_rewards.dtype)
    ).sum(dim=-1)

    temperatures = gumbel_temperature.detach().float().reshape(-1)
    if temperatures.numel() == 1:
        temperatures = temperatures.expand(batch_size)
    if temperatures.numel() != batch_size or bool((temperatures <= 0).any().item()):
        raise ValueError("gumbel_temperature must provide one positive value per trajectory")

    # Mirror the actor embedding path exactly: only all-hard positions mask
    # candidate slots 1..K. Latent positions retain the actual noisy K-vector.
    mixture_logits = response_gumbels.clone()
    mixture_logits[..., 1:] = mixture_logits[..., 1:].masked_fill(
        hard_token_mask.unsqueeze(-1), -torch.inf
    )
    mixture_weights = torch.softmax(
        mixture_logits / temperatures.reshape(batch_size, 1, 1), dim=-1
    )
    log_weights = torch.where(
        mixture_weights > 0,
        torch.log(mixture_weights),
        torch.zeros_like(mixture_weights),
    )
    effective_k = torch.exp(-(mixture_weights * log_weights).sum(dim=-1))
    top1_weight = mixture_weights.max(dim=-1).values

    statistics = {
        "train/entropy": sufficient_stats_from_tensor(
            entropies,
            mask=response_mask,
            definition_version="old_policy_entropy_response_mask_v1",
        ),
        "train/response_length": sufficient_stats_from_tensor(
            response_lengths.float(),
            definition_version="runtime_response_mask_length_v1",
        ),
        "train/latent_length": sufficient_stats_from_tensor(
            latent_lengths.float(),
            definition_version="rollout_topk_sentinel_latent_length_v1",
        ),
        "mixture/effective_k_noisy": sufficient_stats_from_tensor(
            effective_k,
            mask=latent_position_mask,
            definition_version="rollout_gumbel_softmax_actual_embedding_v1",
        ),
        "mixture/top1_weight_noisy": sufficient_stats_from_tensor(
            top1_weight,
            mask=latent_position_mask,
            definition_version="rollout_gumbel_softmax_actual_embedding_v1",
        ),
        "mask/zero_advantage_rate": sufficient_stats_from_tensor(
            final_advantages,
            mask=latent_position_mask,
            numerator_mask=(final_advantages == 0),
            definition_version="post_advantage_pre_update_latent_mask_v1",
        ),
        "signal/reward": sufficient_stats_from_tensor(
            sequence_rewards,
            definition_version="final_sequence_reward_v1",
        ),
        "signal/advantage": sufficient_stats_from_tensor(
            final_advantages,
            mask=response_mask,
            definition_version="final_actor_advantage_response_mask_v1",
        ),
    }
    return {
        "p1_driver_metrics_available": True,
        "p1_driver_sufficient_stats": statistics,
        "final_training_trajectory_lengths": [int(value) for value in response_lengths.tolist()],
        "final_training_trajectory_count": batch_size,
        "noisy_mixture_source": "rollout_topk_gumbels_softmax_same_as_actor_embedding",
    }


def collect_driver_p1_statistics(
    batch: Any,
    *,
    entropies: Any,
    exclude_overlong_samples_from_advantage: bool,
) -> dict[str, Any]:
    response_mask = batch.batch["response_mask"]
    response_width = response_mask.shape[-1]
    return collect_driver_p1_from_tensors(
        response_mask=response_mask,
        rollout_topk_ids=batch.batch["rollout_topk_ids"],
        rollout_topk_gumbels=batch.batch["rollout_topk_gumbels"],
        gumbel_temperature=batch.batch["gumbel_temperature"],
        entropies=entropies,
        token_level_rewards=batch.batch["token_level_rewards"],
        advantages=batch.batch["advantages"],
        exclude_overlong_samples_from_advantage=exclude_overlong_samples_from_advantage,
    )
