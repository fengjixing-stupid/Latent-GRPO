"""Small immutable event values shared by metric builders."""

from dataclasses import asdict, dataclass
from typing import Optional


_CHECKPOINT_PHASES = frozenset(("checkpoint_eval", "checkpoint_probe"))


@dataclass(frozen=True)
class StepContext:
    """The time and identity fields for one metrics event.

    It deliberately carries only detached scalar metadata, never tensors.
    """

    profile_name: str
    seed: int
    global_step: int
    optimizer_step: int
    observation_phase: str
    learning_rate: Optional[float] = None
    wall_clock_seconds: Optional[float] = None
    cumulative_train_samples: Optional[int] = None
    cumulative_rollout_tokens: Optional[int] = None
    cumulative_gpu_hours: Optional[float] = None
    checkpoint_step: Optional[int] = None
    is_resume_run: bool = False
    resume_from_step: Optional[int] = None

    def __post_init__(self) -> None:
        if self.global_step < 0 or self.optimizer_step < 0:
            raise ValueError("step values must be non-negative")
        if self.checkpoint_step is not None and self.observation_phase not in _CHECKPOINT_PHASES:
            raise ValueError("checkpoint_step is reserved for checkpoint-derived events")

    def to_record(self) -> dict:
        record = asdict(self)
        if self.checkpoint_step is None:
            record.pop("checkpoint_step")
        return record
