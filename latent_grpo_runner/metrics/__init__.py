"""Pure-Python metric contracts for the Mac-safe development surface."""

from .aggregators import SufficientStats
from .events import StepContext

__all__ = ["StepContext", "SufficientStats"]
