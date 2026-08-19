# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits, logprobs_from_logits_topk_gumbel, top_p_renorm_logprobs
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs, \
    ulysses_pad_and_slice_inputs_3d
from verl.workers.actor import BasePPOActor
index_first_axis = pad_input = rearrange = unpad_input = None
if is_cuda_available:
    try:
        from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
    except ImportError:
        # The T4 padded latent path intentionally does not require FlashAttention.
        pass
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
LATENT_GRPO_STAGE2_DEFINITION_VERSION = "stage2_surrogate_v1"
LATENT_GRPO_NEAR_ZERO_THRESHOLD = 1e-6


def _require_remove_padding_runtime(use_remove_padding):
    """Require FlashAttention padding helpers only for the packed actor path."""
    if use_remove_padding and any(
        helper is None for helper in (index_first_axis, pad_input, rearrange, unpad_input)
    ):
        raise RuntimeError(
            "use_remove_padding=True requires flash_attn bert_padding helpers; "
            "use the validated padded latent path on Turing/T4"
        )


def _latent_mixture_weights(rollout_topk_ids, rollout_topk_gumbels, gumbel_temperature):
    """Exact latent mixture used by both packed and padded actor paths."""
    hard_token_mask = (rollout_topk_ids[..., 1:] == -100).all(dim=-1)
    masked_gumbels = rollout_topk_gumbels.clone()
    masked_gumbels[..., 1:] = masked_gumbels[..., 1:].masked_fill(
        hard_token_mask.unsqueeze(-1), -torch.inf
    )
    weights = torch.softmax(masked_gumbels / gumbel_temperature, dim=-1).to(
        rollout_topk_gumbels.dtype
    )
    return hard_token_mask, weights


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1
        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()
        self.runtime_dtype = PrecisionType.to_dtype(
            self.config.get("runtime_dtype", "bfloat16")
        )

        # Optional, bounded observer state.  The default remains disabled and
        # does not change the actor's public return values.
        observer_setting = str(os.getenv("LATENT_GRPO_OBSERVER_ENABLED", "0")).strip().lower()
        self._latent_grpo_observer_enabled = observer_setting in {"1", "true", "yes", "on"}
        self._latent_grpo_observer_facts = {
            "optimizer_steps": [],
            "component_sufficient_stats": [],
            "p1_sufficient_stats": {},
            "checkpoint_probe": None,
        }
        self._last_optimizer_did_step = False
        self.last_update_did_step = False
        self.last_update_count = 0

    def _record_optimizer_step_outcome(self, did_step, grad_norm):
        """Record detached scalars only; never retain gradients or optimizer state."""
        if not self._latent_grpo_observer_enabled:
            return
        grad_norm_value = float(grad_norm.detach().float().item()) if did_step else None
        self._latent_grpo_observer_facts["optimizer_steps"].append(
            {
                "did_step": bool(did_step),
                "skip_reason": None if did_step else "nonfinite_grad_norm",
                "grad_norm": grad_norm_value,
            }
        )

    def _record_component_sufficient_stats(self, margins, valid_mask, flip_mask):
        """Immediately reduce component tensors to detached scalar sufficient stats."""
        if not self._latent_grpo_observer_enabled:
            return
        with torch.no_grad():
            detached_margins = margins.detach().float()
            detached_valid = valid_mask.detach().bool()
            detached_flip = flip_mask.detach().bool()
            finite_mask = torch.isfinite(detached_margins)
            effective_mask = detached_valid & finite_mask
            values = detached_margins[effective_mask]
            count = int(effective_mask.sum().item())
            threshold = LATENT_GRPO_NEAR_ZERO_THRESHOLD
            facts = {
                "sum": float(values.double().sum().item()) if count else 0.0,
                "sum_sq": float(values.double().square().sum().item()) if count else 0.0,
                "count": count,
                "nan_count": int((detached_valid & ~finite_mask).sum().item()),
                "masked_count": int((~detached_valid).sum().item()),
                "min": float(values.min().item()) if count else None,
                "negative_count": int((effective_mask & (detached_margins < 0)).sum().item()),
                "near_zero_count": int((effective_mask & (detached_margins.abs() <= threshold)).sum().item()),
                "flipgrad_trigger_count": int((effective_mask & detached_flip).sum().item()),
                "near_zero_threshold": threshold,
                "definition_version": LATENT_GRPO_STAGE2_DEFINITION_VERSION,
            }
        self._latent_grpo_observer_facts["component_sufficient_stats"].append(facts)

    def _record_component_stats_unavailable(self, reason):
        if self._latent_grpo_observer_enabled:
            self._latent_grpo_observer_facts["component_sufficient_stats"].append(
                {"available": False, "unavailable_reason": str(reason)}
            )

    def _record_p1_sufficient_stats(
        self, name, values, mask, *, numerator_mask=None, definition_version
    ):
        """Reduce one already-computed PPO tensor immediately to scalar stats."""
        if not self._latent_grpo_observer_enabled:
            return
        try:
            from latent_grpo_runner.metrics.p1 import merge_serialized_sufficient_stats
            from latent_grpo_runner.metrics.torch_collectors import sufficient_stats_from_tensor
        except ImportError as error:
            raise RuntimeError("P1 observer collectors are unavailable on the actor worker") from error
        local = sufficient_stats_from_tensor(
            values,
            mask=mask,
            numerator_mask=numerator_mask,
            definition_version=definition_version,
        )
        mapping = self._latent_grpo_observer_facts["p1_sufficient_stats"]
        previous = mapping.get(name)
        mapping[name] = (
            local
            if previous is None
            else merge_serialized_sufficient_stats([previous, local])
        )

    def consume_latent_grpo_observer_facts(self):
        # Observer-off training must not export worker observer packets.
        # last_update_count / last_update_did_step remain available to the
        # scheduler independently of this optional metrics payload.
        if not self._latent_grpo_observer_enabled:
            return {}

        facts = {
            "did_step": bool(self.last_update_did_step),
            "did_update": bool(self.last_update_did_step),
            "update_count": int(self.last_update_count),
            "optimizer_steps": list(self._latent_grpo_observer_facts["optimizer_steps"]),
            "component_sufficient_stats": list(
                self._latent_grpo_observer_facts["component_sufficient_stats"]
            ),
            "p1_sufficient_stats": dict(
                self._latent_grpo_observer_facts["p1_sufficient_stats"]
            ),
            "checkpoint_probe": self._latent_grpo_observer_facts["checkpoint_probe"],
            "gpu_memory": self._collect_cuda_memory_facts(),
        }
        self._latent_grpo_observer_facts = {
            "optimizer_steps": [],
            "component_sufficient_stats": [],
            "p1_sufficient_stats": {},
            "checkpoint_probe": None,
        }
        return facts

    def _collect_cuda_memory_facts(self):
        """Return detached allocator counters for this rank's selected CUDA device."""
        if not self._latent_grpo_observer_enabled or not is_cuda_available:
            return None
        cuda = get_torch_device()
        device_index = int(cuda.current_device())
        return {
            "device_index": device_index,
            "current_allocated_bytes": int(cuda.memory_allocated(device_index)),
            "current_reserved_bytes": int(cuda.memory_reserved(device_index)),
            "peak_allocated_bytes": int(cuda.max_memory_allocated(device_index)),
            "peak_reserved_bytes": int(cuda.max_memory_reserved(device_index)),
        }

    def _forward_micro_batch(self, micro_batch, temperature, top_p,  calculate_entropy=False, add_noise_dirichlet=False,
                             add_noise_gumbel_softmax=True, collect_component_stats=False,
                             component_response_mask=None, collect_checkpoint_probe=False) -> Tuple[
        torch.Tensor, torch.Tensor]:
        _require_remove_padding_runtime(self.use_remove_padding)

        def safe_lookup_embeddings(fsdp_wrapped_module, input_ids, target_device=None, target_dtype=None):
            """Look up embeddings safely when the module is wrapped by FSDP."""
            embed = fsdp_wrapped_module.get_input_embeddings()

            get_torch_device().empty_cache()
            ctx = FSDP.summon_full_params(
                fsdp_wrapped_module,
                recurse=False,
                writeback=False,
                with_grads=False,
            )
            with ctx:
                w = embed.weight
                _input_ids = input_ids.to(w.device).clone()
                mask = (_input_ids < 0)
                if mask.any():
                    _input_ids[mask] = 0
                
                embs = embed(_input_ids).detach()
                
                if mask.any():
                    embs[mask] = 0
            if target_dtype is not None and embs.dtype != target_dtype:
                embs = embs.to(dtype=target_dtype)
            if target_device is not None and embs.device != target_device:
                embs = embs.to(target_device)
            return embs

        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]],
                                                    dim=0)
        with torch.autocast(device_type=self.device_name, dtype=self.runtime_dtype):
            input_ids = micro_batch["input_ids"] # whole sentences B*n:I+O
            rollout_topk_ids = micro_batch["rollout_topk_ids"]
            rollout_topk_gumbels = micro_batch["rollout_topk_gumbels"]
            batch_size, seqlen = input_ids.shape
            k_num = rollout_topk_gumbels.size(-1)
            attention_mask = micro_batch["attention_mask"]
            response_mask = attention_mask[:, -response_length:].bool()
            position_ids = micro_batch["position_ids"]
            gumbel_temperature = micro_batch["gumbel_temperature"][0].item()
            entropy = None
            checkpoint_probe_tensors = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)
            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
                topk_ids_rmpad, _, *_ = unpad_input(rollout_topk_ids,
                                                    attention_mask)  # topk_ids_rmpad (total_nnz, K)
                topk_gumbels_rmpad, _, *_ = unpad_input(rollout_topk_gumbels,
                                                        attention_mask)  # topk_gumbels_rmpad (total_nnz, K)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."),
                                                          indices).transpose(0, 1).unsqueeze(
                        1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                          indices).transpose(0, 1)

                # for compute the log_prob
                hard_token_mask, gumbel_y = _latent_mixture_weights(
                    topk_ids_rmpad, topk_gumbels_rmpad, gumbel_temperature
                )
                
                has_soft_tokens = (~hard_token_mask).any()
                lookup_ids = topk_ids_rmpad if has_soft_tokens else topk_ids_rmpad[:, :1]
                lookup_embs = safe_lookup_embeddings(
                    self.actor_module,
                    lookup_ids,
                    target_device=topk_gumbels_rmpad.device,
                    target_dtype=topk_gumbels_rmpad.dtype,
                )
                all_first_embs = lookup_embs[:, 0, :]
                if has_soft_tokens:
                    # Soft tokens need all K candidates
                    mask_expanded = hard_token_mask.unsqueeze(-1) # (total_nnz, 1)
                    soft_embs = torch.sum(
                        gumbel_y.unsqueeze(-1).float() * lookup_embs.float(), dim=1
                    ).to(self.runtime_dtype)
                    # Output is hard_embs for hard tokens, soft_embs for soft tokens
                    topk_embs_final = torch.where(mask_expanded, all_first_embs, soft_embs)
                else:
                    # All are hard tokens
                    topk_embs_final = all_first_embs

                topk_embs = topk_embs_final
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                topk_gumbels_rmpad_rolled = torch.roll(topk_gumbels_rmpad, shifts=-1, dims=0)  # (total_nnz, k)
                topk_ids_rmpad_rolled = torch.roll(topk_ids_rmpad, shifts=-1, dims=0)  # (total_nnz, k)
                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    topk_ids_rmpad, _, _ = ulysses_pad_and_slice_inputs_3d(
                        rollout_topk_ids,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                    topk_gumbels_rmpad, _, _ = ulysses_pad_and_slice_inputs_3d(
                        rollout_topk_gumbels,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                # Model Forward
                output = self.actor_module(
                    # input_ids=input_ids_rmpad,
                    inputs_embeds=topk_embs.unsqueeze(0).detach(),#
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    # ================================================================
                    # Process advantages for Gumbel gating.
                    # ================================================================
                    advantages = micro_batch.get("advantages", None)
                    current_advantages = None
                    current_response_mask = None
                    if advantages is not None:
                        # advantages: (B, R) - Response only
                        # input_ids: (B, I+R) - Prompt + Response
                        # 1. Left Pad with 0s for Prompt portion
                        prompt_len = seqlen - advantages.size(1)  # I = (I+R) - R
                        if prompt_len > 0:
                            # F.pad: (left_pad, right_pad)
                            full_advantages = torch.nn.functional.pad(advantages, (prompt_len, 0), value=0.0)
                        else:
                            full_advantages = advantages
                        # full_advantages: (B, I+R)

                        # 2. Unpad to match input_ids_rmpad structure
                        adv_expanded = full_advantages.unsqueeze(-1)  # (B, T, 1)
                        adv_rmpad, _, *_ = unpad_input(adv_expanded, attention_mask)
                        # adv_rmpad: (total_nnz, 1)
                        
                        # 3. Roll to align with input_ids_rmpad_rolled (Next Token)
                        current_advantages = torch.roll(adv_rmpad, shifts=-1, dims=0)
                        # current_advantages: (total_nnz, 1)
                    if component_response_mask is not None:
                        if (
                            component_response_mask.dim() != 2
                            or component_response_mask.size(0) != batch_size
                            or component_response_mask.size(1) != response_length
                        ):
                            self._record_component_stats_unavailable("component_response_mask_shape_mismatch")
                        else:
                            prompt_mask_length = seqlen - response_length
                            full_response_mask = torch.nn.functional.pad(
                                component_response_mask, (prompt_mask_length, 0), value=0
                            )
                            mask_rmpad, _, *_ = unpad_input(
                                full_response_mask.unsqueeze(-1), attention_mask
                            )
                            current_response_mask = torch.roll(mask_rmpad, shifts=-1, dims=0)
                    # ================================================================

                    if add_noise_gumbel_softmax:
                        log_prob_result = logprobs_from_logits_topk_gumbel(
                            logits=logits_rmpad,
                            rollout_topk_ids=topk_ids_rmpad_rolled,
                            rollout_topk_gumbels=topk_gumbels_rmpad_rolled,
                            labels=input_ids_rmpad_rolled,
                            top_p=top_p,
                            temperature=temperature,
                            inplace_backward=inplace_backward,
                            advantages=current_advantages,
                            return_probe_tensors=collect_checkpoint_probe,
                        )
                        if collect_checkpoint_probe:
                            log_probs, packed_checkpoint_tensors = log_prob_result
                        else:
                            log_probs = log_prob_result
                    else:
                        full_logprobs = top_p_renorm_logprobs(logits_rmpad / temperature, top_p)
                        log_probs = full_logprobs.gather(-1, input_ids_rmpad_rolled.unsqueeze(-1)).squeeze(-1)

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                if collect_checkpoint_probe:
                    from latent_grpo_runner.metrics.probe import restore_packed_probe_tensors

                    checkpoint_probe_tensors = restore_packed_probe_tensors(
                        packed_checkpoint_tensors,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                        response_length=response_length,
                    )
                    # ``policy_loss`` depends on the packed tensor used to build
                    # ``log_probs``.  The restored tensor above is a downstream
                    # view/copy and therefore is not an autograd ancestor of the
                    # loss.  Keep the graph-connected packed target and enough
                    # metadata to restore its gradient to response layout.
                    checkpoint_probe_tensors["autograd_topk_log_probs"] = (
                        packed_checkpoint_tensors["topk_log_probs"]
                    )
                    checkpoint_probe_tensors["autograd_restore_spec"] = {
                        "indices": indices,
                        "batch": batch_size,
                        "seqlen": seqlen,
                        "response_length": response_length,
                    }
                safe_gather_ids = topk_ids_rmpad_rolled.clone()
                safe_gather_ids[safe_gather_ids == -100] = 0

                current_topk_logits_rmpad = logits_rmpad.gather(-1, safe_gather_ids)
                if collect_component_stats:
                    if not add_noise_gumbel_softmax:
                        self._record_component_stats_unavailable("gumbel_path_disabled")
                    elif current_advantages is None:
                        self._record_component_stats_unavailable("advantages_unavailable")
                    elif current_response_mask is None:
                        self._record_component_stats_unavailable("component_response_mask_unavailable")
                    elif not (
                        topk_ids_rmpad_rolled.shape == topk_gumbels_rmpad_rolled.shape
                        == current_topk_logits_rmpad.shape
                        and current_advantages.dim() == 2
                        and current_response_mask.dim() == 2
                        and current_advantages.shape == current_response_mask.shape
                        and current_advantages.size(0) == current_topk_logits_rmpad.size(0)
                        and current_advantages.size(1) == 1
                    ):
                        self._record_component_stats_unavailable("component_alignment_shape_mismatch")
                    else:
                        # Reuse this forward's selected logits.  Reduce immediately
                        # and do not retain full-vocabulary logits or component tensors.
                        with torch.no_grad():
                            component_log_probs = current_topk_logits_rmpad.detach().float() - torch.logsumexp(
                                logits_rmpad.detach().float(), dim=-1, keepdim=True
                            )
                            surrogate_margins = topk_gumbels_rmpad_rolled.detach().float() - component_log_probs
                            valid_latent_positions = current_response_mask.detach().bool().squeeze(-1) & ~(
                                topk_ids_rmpad_rolled[:, 1:] == -100
                            ).all(dim=-1)
                            valid_components = valid_latent_positions.unsqueeze(-1) & (
                                topk_ids_rmpad_rolled != -100
                            )
                            flip_mask = (
                                current_advantages.detach().float().expand_as(surrogate_margins) <= 0
                            ) & (surrogate_margins < 0)
                        self._record_component_sufficient_stats(
                            surrogate_margins, valid_components, flip_mask
                        )
                full_current_topk_logits = pad_input(
                    hidden_states=current_topk_logits_rmpad, 
                    indices=indices, 
                    batch=batch_size, 
                    seqlen=seqlen
                ) # (B, Seq, K)
                # logits_rmpad.div_(temperature)
                latent_probs = torch.softmax(logits_rmpad / temperature, dim=-1)
                topk_original_probs, topk_indices = torch.topk(
                    latent_probs, k=int(topk_ids_rmpad_rolled.size(-1)), dim=-1
                )
                full_topk_probs = pad_input(
                    hidden_states=topk_original_probs,
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                full_topk_indices = pad_input(
                    hidden_states=topk_indices,
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                full_topk_probs = full_topk_probs[:, :-1, :].contiguous()
                full_topk_indices = full_topk_indices[:, :-1, :].contiguous()



                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1: -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1: -1]  # (bsz, response_length)
                full_current_topk_logits = full_current_topk_logits.squeeze(-1)[:, -response_length - 1: -1]  # (bsz, response_length)
            else:  # padded latent path: same Top-K/Gumbel semantics, no packed FlashAttention
                if self.use_ulysses_sp:
                    raise RuntimeError("padded latent path requires ulysses_sequence_parallel_size=1")
                if self.use_fused_kernels:
                    raise RuntimeError("padded latent path requires use_fused_kernels=False")

                hard_token_mask, gumbel_y = _latent_mixture_weights(
                    rollout_topk_ids, rollout_topk_gumbels, gumbel_temperature
                )
                mask_expanded = hard_token_mask.unsqueeze(-1)
                topk_embs_all = safe_lookup_embeddings(
                    self.actor_module,
                    rollout_topk_ids,
                    target_device=rollout_topk_gumbels.device,
                    target_dtype=rollout_topk_gumbels.dtype,
                )
                all_first_embs = topk_embs_all[..., 0, :]
                soft_embs = torch.sum(
                    gumbel_y.unsqueeze(-1).float() * topk_embs_all.float(), dim=-2
                ).to(self.runtime_dtype)
                topk_embs_final = torch.where(mask_expanded, all_first_embs, soft_embs)

                output = self.actor_module(
                    inputs_embeds=topk_embs_final.detach(),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )
                logits_full = output.logits
                response_logits = logits_full[:, -response_length - 1: -1, :]
                next_topk_ids = rollout_topk_ids[:, -response_length:, :]
                next_topk_gumbels = rollout_topk_gumbels[:, -response_length:, :]
                advantages = micro_batch.get("advantages", None)

                if add_noise_gumbel_softmax:
                    log_prob_result = logprobs_from_logits_topk_gumbel(
                        logits=response_logits,
                        rollout_topk_ids=next_topk_ids,
                        rollout_topk_gumbels=next_topk_gumbels,
                        labels=micro_batch["responses"],
                        top_p=top_p,
                        temperature=temperature,
                        inplace_backward=not calculate_entropy,
                        advantages=advantages,
                        return_probe_tensors=collect_checkpoint_probe,
                        valid_mask=response_mask,
                    )
                    if collect_checkpoint_probe:
                        log_probs, checkpoint_probe_tensors = log_prob_result
                    else:
                        log_probs = log_prob_result
                else:
                    full_logprobs = top_p_renorm_logprobs(response_logits / temperature, top_p)
                    log_probs = full_logprobs.gather(
                        -1, micro_batch["responses"].unsqueeze(-1)
                    ).squeeze(-1)

                if calculate_entropy:
                    entropy = self.compute_entropy_from_logits(response_logits)

                safe_next_ids = next_topk_ids.clone()
                safe_next_ids[safe_next_ids == -100] = 0
                full_current_topk_logits = response_logits.gather(-1, safe_next_ids)

                if collect_component_stats:
                    if not add_noise_gumbel_softmax:
                        self._record_component_stats_unavailable("gumbel_path_disabled")
                    elif advantages is None:
                        self._record_component_stats_unavailable("advantages_unavailable")
                    elif component_response_mask is None:
                        self._record_component_stats_unavailable("component_response_mask_unavailable")
                    elif component_response_mask.shape != advantages.shape:
                        self._record_component_stats_unavailable("component_response_mask_shape_mismatch")
                    else:
                        with torch.no_grad():
                            component_log_probs = full_current_topk_logits.detach().float() - torch.logsumexp(
                                response_logits.detach().float(), dim=-1, keepdim=True
                            )
                            surrogate_margins = next_topk_gumbels.detach().float() - component_log_probs
                            valid_latent_positions = component_response_mask.detach().bool() & ~(
                                next_topk_ids[..., 1:] == -100
                            ).all(dim=-1)
                            valid_components = valid_latent_positions.unsqueeze(-1) & (
                                next_topk_ids != -100
                            )
                            flip_mask = (
                                advantages.detach().float().unsqueeze(-1).expand_as(surrogate_margins) <= 0
                            ) & (surrogate_margins < 0)
                        self._record_component_sufficient_stats(
                            surrogate_margins, valid_components, flip_mask
                        )

                latent_probs = torch.softmax(logits_full / temperature, dim=-1)
                full_topk_probs, full_topk_indices = torch.topk(
                    latent_probs, k=int(rollout_topk_ids.size(-1)), dim=-1
                )
                full_topk_probs = full_topk_probs[:, :-1, :].contiguous()
                full_topk_indices = full_topk_indices[:, :-1, :].contiguous()
            
            result = (entropy, log_probs, full_topk_probs, full_topk_indices, full_current_topk_logits)
            if collect_checkpoint_probe:
                return (*result, checkpoint_probe_tensors)
            return result
    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        # if grad_norm is not finite, skip the update
        did_step = False
        if not torch.isfinite(grad_norm):
            logger.warning("Rank %s grad_norm is not finite: %s", torch.distributed.get_rank(), grad_norm)
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
            did_step = True
        self._last_optimizer_did_step = did_step
        self.last_update_did_step = self.last_update_did_step or did_step
        self._record_optimizer_step_outcome(did_step, grad_norm)
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        top_p = data.meta_info["top_p"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        add_noise_dirichlet = data.meta_info['add_noise_dirichlet']
        add_noise_gumbel_softmax = data.meta_info['add_noise_gumbel_softmax']

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "rollout_topk_ids",
                       "rollout_topk_gumbels", "gumbel_temperature"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        topk_probs_lst = []
        topk_ids_lst = []
        entropy_lst = []
        topk_logits_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, topk_porbs, topk_indices, cur_topk_logits = self._forward_micro_batch(micro_batch, temperature=temperature, top_p=top_p,
                                                               calculate_entropy=calculate_entropy,
                                                               add_noise_dirichlet=add_noise_dirichlet,
                                                               add_noise_gumbel_softmax=add_noise_gumbel_softmax,)
            log_probs_lst.append(log_probs)
            topk_probs_lst.append(topk_porbs)
            topk_ids_lst.append(topk_indices)
            topk_logits_lst.append(cur_topk_logits)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        topk_probs = torch.concat(topk_probs_lst, dim=0)
        topk_ids = torch.concat(topk_ids_lst, dim=0)
        # This slot carries logits gathered by the same forward, not token IDs.
        topk_logits = torch.concat(topk_logits_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
        return log_probs, entropys, topk_probs, topk_ids, topk_logits

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()
        self.last_update_did_step = False
        self.last_update_count = 0
        update_count = 0
        if self._latent_grpo_observer_enabled and is_cuda_available:
            cuda = get_torch_device()
            cuda.reset_peak_memory_stats(cuda.current_device())
        pre_backward_monitor_probe = bool(
            data.meta_info.get("pre_backward_monitor_probe", False)
        )
        checkpoint_probe_requested = bool(data.meta_info.get("checkpoint_probe", False))
        credit_probe_enabled = str(
            os.getenv("LATENT_GRPO_CREDIT_PROBE_ENABLED", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        if pre_backward_monitor_probe and not self._latent_grpo_observer_enabled:
            raise RuntimeError("pre-backward monitor probe requires the durable observer")
        if checkpoint_probe_requested and (
            not self._latent_grpo_observer_enabled or not credit_probe_enabled
        ):
            raise RuntimeError("checkpoint credit probe requires the durable observer and credit feature")

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        top_p = data.meta_info["top_p"]
        multi_turn = data.meta_info.get("multi_turn", False)

        add_noise_dirichlet = data.meta_info['add_noise_dirichlet']
        add_noise_gumbel_softmax = data.meta_info['add_noise_gumbel_softmax']
        exclude_overlong_samples_from_advantage = data.meta_info.get("exclude_overlong_samples_from_advantage", False)
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages",
                       "rollout_topk_ids", "rollout_topk_gumbels", "gumbel_temperature","token_level_rewards"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        checkpoint_probe_collected = False
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True

                    # Freeze the final advantage before both the Gumbel gating
                    # path and observer use it. Multi-turn response_mask already
                    # contains the loss-mask domain selected above.
                    if not exclude_overlong_samples_from_advantage:
                        cur_response_length = data["attention_mask"][:, -response_length:].sum(dim=-1)
                        is_clipped = cur_response_length == response_length
                        if "advantages" in data:
                            data["advantages"][is_clipped] = 0
                    collect_checkpoint_probe = (
                        checkpoint_probe_requested and not checkpoint_probe_collected
                    )
                    forward_result = self._forward_micro_batch(
                        micro_batch=data,
                        temperature=temperature,
                        top_p=top_p,
                        calculate_entropy=calculate_entropy,
                        add_noise_dirichlet=add_noise_dirichlet,
                        add_noise_gumbel_softmax=add_noise_gumbel_softmax,
                        collect_component_stats=self._latent_grpo_observer_enabled,
                        component_response_mask=response_mask,
                        collect_checkpoint_probe=collect_checkpoint_probe,
                    )
                    if collect_checkpoint_probe:
                        entropy, log_prob, _, _, current_logits, checkpoint_tensors = forward_result
                    else:
                        entropy, log_prob, _, _, current_logits = forward_result
                    
                    neg_adv_weight = self.config.get("neg_adv_weight", 1.0)
                    policy_result = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        neg_adv_weight=neg_adv_weight,
                        loss_agg_mode=loss_agg_mode,
                        return_observer_tensors=self._latent_grpo_observer_enabled,
                    )
                    if self._latent_grpo_observer_enabled:
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, p1_tensors = policy_result
                        self._record_p1_sufficient_stats(
                            "train/policy_loss", p1_tensors["policy_loss_elements"], response_mask,
                            definition_version="ppo_policy_loss_elements_v1",
                        )
                        self._record_p1_sufficient_stats(
                            "train/kl", p1_tensors["kl_elements"], response_mask,
                            definition_version="ppo_negative_approx_kl_v1",
                        )
                        self._record_p1_sufficient_stats(
                            "train/clip_fraction", p1_tensors["clip_mask"], response_mask,
                            numerator_mask=p1_tensors["clip_mask"],
                            definition_version="ppo_primary_clip_fraction_v1",
                        )
                        self._record_p1_sufficient_stats(
                            "train/importance_ratio", p1_tensors["importance_ratio"], response_mask,
                            definition_version="ppo_importance_ratio_v1",
                        )
                    else:
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_result
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob,
                                         kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation

                    metric_row = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item() if hasattr(pg_clipfrac, 'item') else pg_clipfrac,
                        "actor/ppo_kl": ppo_kl.detach().item() if hasattr(ppo_kl, 'item') else ppo_kl,
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item() if hasattr(pg_clipfrac_lower, 'item') else pg_clipfrac_lower,
                    }
                    append_to_dict(metrics, metric_row)

                    if collect_checkpoint_probe:
                        from latent_grpo_runner.metrics.probe import collect_checkpoint_probe_packet

                        if checkpoint_tensors is None:
                            raise RuntimeError("checkpoint probe tensors were not returned by the Gumbel path")
                        next_topk_ids = data["rollout_topk_ids"][:, -response_length:, :]
                        next_topk_gumbels = data["rollout_topk_gumbels"][:, -response_length:, :]
                        hard_token_mask, mixture_weights = _latent_mixture_weights(
                            next_topk_ids,
                            next_topk_gumbels,
                            data["gumbel_temperature"][0].item(),
                        )
                        valid_positions = response_mask.bool() & ~hard_token_mask
                        valid_components = valid_positions.unsqueeze(-1) & (next_topk_ids != -100)
                        expanded_advantages = advantages.unsqueeze(-1).expand_as(
                            checkpoint_tensors["topk_log_probs"]
                        )
                        trajectory_masks, position_masks = _checkpoint_probe_group_masks(
                            valid_positions=valid_positions,
                            response_mask=response_mask,
                            token_level_rewards=data["token_level_rewards"],
                            component_count=next_topk_ids.size(-1),
                        )
                        self._latent_grpo_observer_facts["checkpoint_probe"] = (
                            collect_checkpoint_probe_packet(
                                policy_loss=loss,
                                topk_log_probs=checkpoint_tensors["topk_log_probs"],
                                deltas=checkpoint_tensors["raw_diff"],
                                mixture_weights=mixture_weights,
                                valid_component_mask=valid_components,
                                advantages=expanded_advantages,
                                trajectory_masks=trajectory_masks,
                                position_masks=position_masks,
                                model=self.actor_module,
                                optimizer=self.actor_optimizer,
                                flipgrad_trigger_mask=checkpoint_tensors[
                                    "flipgrad_trigger_mask"
                                ],
                                autograd_topk_log_probs=checkpoint_tensors.get(
                                    "autograd_topk_log_probs"
                                ),
                                autograd_restore_spec=checkpoint_tensors.get(
                                    "autograd_restore_spec"
                                ),
                                retain_graph=True,
                            )
                        )
                        checkpoint_probe_collected = True

                    if pre_backward_monitor_probe:
                        # The real actor forward, Gumbel/FlipGrad gating, PPO loss,
                        # and observer reductions have all run.  Stop here so the
                        # Kaggle engineering probe cannot allocate gradients or
                        # mutate model/optimizer state.
                        append_to_dict(metrics, {
                            "monitor_probe/pre_backward_reached": 1.0,
                            "monitor_probe/optimizer_step_skipped": 1.0,
                        })
                        self.actor_optimizer.zero_grad()
                        self.last_update_count = 0
                        self.last_update_did_step = False
                        return metrics

                    loss.backward()

                grad_norm = self._optimizer_step()
                did_update = self._last_optimizer_did_step
                update_count += int(did_update)
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        self.last_update_count = update_count
        self.last_update_did_step = update_count > 0
        return metrics


def _checkpoint_probe_group_masks(
    *, valid_positions, response_mask, token_level_rewards, component_count
):
    """Build all Stage-4 groups as masks over one shared probe result."""
    batch_size, response_length = valid_positions.shape
    shape = (batch_size, response_length, component_count)

    def expand_positions(mask):
        return mask.unsqueeze(-1).expand(shape)

    reward_sum = token_level_rewards.sum(dim=-1)
    correct = reward_sum > 0
    overlong = response_mask.bool().sum(dim=-1) >= response_length

    def expand_trajectories(mask):
        return mask.view(batch_size, 1, 1).expand(shape)

    ordinal = valid_positions.long().cumsum(dim=-1) - 1
    latent_count = valid_positions.long().sum(dim=-1, keepdim=True)
    first = valid_positions & (ordinal == 0)
    last = valid_positions & (ordinal == (latent_count - 1))
    middle = valid_positions & ~first & ~last
    return (
        {
            "all": expand_trajectories(torch.ones_like(correct, dtype=torch.bool)),
            "correct": expand_trajectories(correct),
            "non_correct": expand_trajectories(~correct),
            "overlong": expand_trajectories(overlong),
            "not_overlong": expand_trajectories(~overlong),
        },
        {
            "all": expand_positions(torch.ones_like(valid_positions, dtype=torch.bool)),
            "first_latent": expand_positions(first),
            "middle_latent": expand_positions(middle),
            "last_latent": expand_positions(last),
        },
    )
