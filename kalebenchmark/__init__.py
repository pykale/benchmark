# =============================================================================
# Author: Haolin Wang, LWang0101@outlook.com
# =============================================================================

"""A thin benchmark wrapper that composes existing PyKale components."""

from .benchmark import Benchmark
from .results import Results
from .task import TaskCard

__all__ = ["Benchmark", "Results", "TaskCard"]
