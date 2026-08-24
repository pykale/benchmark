# =============================================================================
# Author: Haolin Wang, LWang0101@outlook.com
# =============================================================================

"""Benchmark result value object"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Results:
    """What one benchmark run produced.

    Args:
        predictions: Predictions for the evaluated partition, keyed by predictor
            name when more than one predictor was run.
        evaluations (dict): Metric values, keyed the same way.
        interpretations (dict): Interpretation outputs, keyed the same way.
        config (dict): The components as the user selected them.
        task (TaskCard, optional): Set only for ``Benchmark.from_task`` runs.
        artifacts (dict): Fitted objects kept for inspection.
    """

    predictions: Any
    evaluations: Dict[str, Any]
    interpretations: Dict[str, Any]
    config: Dict[str, Any]
    task: Optional[Any] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    _submitter: Optional[Any] = field(default=None, repr=False)

    def submit(self) -> Any:
        """Submit the stored predictions; never run prediction again.

        Returns:
            The submitter's return value, or the payload itself when no
            submitter was given.

        Raises:
            RuntimeError: If the run was not created from a task card.
        """
        if self.task is None:
            raise RuntimeError(
                "This result is not associated with an official TaskCard. "
                "Use Benchmark.from_task(...) for leaderboard runs."
            )
        payload = {
            "task": self.task.name,
            "predictions": self.predictions,
            "evaluations": self.evaluations,
            "config": self.config,
        }
        return self._submitter(payload) if self._submitter else payload
