"""Mergeable sufficient statistics; no worker-average aggregation is allowed."""

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class SufficientStats:
    sum: float = 0.0
    sum_sq: float = 0.0
    count: int = 0
    nan_count: int = 0
    masked_count: int = 0
    min: Optional[float] = None
    max: Optional[float] = None
    numerator_count: int = 0

    @classmethod
    def from_values(
        cls,
        values: Iterable[float],
        mask: Optional[Iterable[bool]] = None,
        numerator_mask: Optional[Iterable[bool]] = None,
    ) -> "SufficientStats":
        values_list = list(values)
        mask_list = list(mask) if mask is not None else [True] * len(values_list)
        numerators = list(numerator_mask) if numerator_mask is not None else [False] * len(values_list)
        if len(values_list) != len(mask_list) or len(values_list) != len(numerators):
            raise ValueError("values, mask, and numerator_mask must have equal length")
        finite = []
        nan_count = 0
        masked_count = 0
        numerator_count = 0
        for value, included, numerator in zip(values_list, mask_list, numerators):
            if not included:
                masked_count += 1
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                nan_count += 1
                continue
            finite.append(numeric)
            if numerator:
                numerator_count += 1
        return cls(
            sum=sum(finite), sum_sq=sum(value * value for value in finite), count=len(finite),
            nan_count=nan_count, masked_count=masked_count,
            min=min(finite) if finite else None, max=max(finite) if finite else None,
            numerator_count=numerator_count,
        )

    def merge(self, other: "SufficientStats") -> "SufficientStats":
        minima = [value for value in (self.min, other.min) if value is not None]
        maxima = [value for value in (self.max, other.max) if value is not None]
        return SufficientStats(
            sum=self.sum + other.sum, sum_sq=self.sum_sq + other.sum_sq,
            count=self.count + other.count, nan_count=self.nan_count + other.nan_count,
            masked_count=self.masked_count + other.masked_count,
            min=min(minima) if minima else None, max=max(maxima) if maxima else None,
            numerator_count=self.numerator_count + other.numerator_count,
        )

    @classmethod
    def merge_all(cls, statistics: Sequence["SufficientStats"]) -> "SufficientStats":
        result = cls()
        for statistic in statistics:
            result = result.merge(statistic)
        return result

    @property
    def available(self) -> bool:
        return self.count > 0

    @property
    def unavailable_reason(self) -> Optional[str]:
        return None if self.available else "empty_effective_mask"

    def mean(self) -> float:
        return self.sum / self.count if self.count else math.nan

    def std(self) -> float:
        if not self.count:
            return math.nan
        variance = max(self.sum_sq / self.count - self.mean() ** 2, 0.0)
        return math.sqrt(variance)

    def rate(self) -> float:
        return self.numerator_count / self.count if self.count else math.nan

    def to_record(self, prefix: str) -> dict:
        return {
            f"{prefix}_sum": self.sum,
            f"{prefix}_sum_sq": self.sum_sq,
            f"{prefix}_count": self.count,
            f"{prefix}_nan_count": self.nan_count,
            f"{prefix}_masked_count": self.masked_count,
            f"{prefix}_min": self.min,
            f"{prefix}_max": self.max,
            f"{prefix}_numerator_count": self.numerator_count,
        }
