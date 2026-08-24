# =============================================================================
# Author: Haolin Wang, LWang0101@outlook.com
# =============================================================================

"""Official benchmark tracks

A task card fixes only the stages that a leaderboard needs in order to compare
submissions. Every other stage stays open, so competing methods can replace it.
It uses the same vocabulary as :class:`~kalebenchmark.benchmark.Benchmark`: a
non-``None`` field is fixed, a ``None`` field is free.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TaskCard:
    """Constraints defining one official track.

    Args:
        name (str): Track identifier used by ``Benchmark.from_task``.
        dataset: Fixed dataset, or ``None`` to leave it open.
        splitter: Fixed splitter, or ``None`` to leave it open.
        prepdata: Fixed preparation, or ``None`` to leave it open.
        embed: Fixed representation, or ``None`` to leave it open.
        predict: Fixed predictor, or ``None`` to leave it open.
        evaluate: Fixed metric(s), or ``None`` to leave them open.
        interpret: Fixed interpreter(s), or ``None`` to leave them open.
        description (str): One-line summary of the track.
    """

    name: str
    dataset: Optional[Any] = None
    splitter: Optional[Any] = None
    prepdata: Optional[Any] = None
    embed: Optional[Any] = None
    predict: Optional[Any] = None
    evaluate: Optional[Any] = None
    interpret: Optional[Any] = None
    description: str = ""

    def components(self) -> Dict[str, Any]:
        """Return the seven configurable pipeline fields."""
        return {
            "dataset": self.dataset,
            "splitter": self.splitter,
            "prepdata": self.prepdata,
            "embed": self.embed,
            "predict": self.predict,
            "evaluate": self.evaluate,
            "interpret": self.interpret,
        }
