# Kaggle Lightweight Checkpoint Design

## Goal

Prevent Ray host-memory OOM during the dual-T4 30-metric validation checkpoint without changing PPO, Stage 4 probe semantics, or the checkpoint behavior of non-Kaggle profiles.

## Design

The `kaggle-t4-30-metric` profile requests FSDP checkpoint contents `model` and `extra`, and disables resume because the resulting artifact intentionally has no optimizer state. The checkpoint remains a real scheduled checkpoint: actor model shards, scheduler/RNG extra state, tokenizer/config, observer sidecar, and checkpoint marker are persisted after the Stage 4 probe.

The FSDP checkpoint manager honors its existing `checkpoint.contents` interface. Default contents remain `model`, `optimizer`, and `extra`; save and load conditionally process selected components. State dictionaries are created, written, and released one component at a time so full checkpoints also avoid retaining model and optimizer snapshots simultaneously.

## Constraints

- Only `kaggle-t4-30-metric` omits optimizer state.
- Three-GPU and ordinary training profiles retain resumable full checkpoints.
- Stage 4 remains checkpoint-only and uses the existing real training graph.
- No extra forward, backward, optimizer step, or metric recomputation is introduced.
- The lightweight artifact is not accepted as a resume source.

## Verification

Unit contracts cover the Kaggle Hydra overrides, default profile behavior, conditional FSDP save/load, sequential state-dict lifetime, and the existing Stage 4 checkpoint gate. Full unit tests and notebook static validation must pass before publication.
