"""Feature attribution components used by the RealMat-BaG example.

Both interpreters delegate: one to :mod:`shap`, the other to
:func:`sklearn.inspection.permutation_importance`. Neither computes attributions itself.

This is the benchmark counterpart of :mod:`kale.interpret`, which currently provides captum-based
signal/image attribution and plotting helpers, but no model-agnostic feature attribution.
"""

from typing import Any


class ShapInterpreter:
    """Interpretation of a fitted estimator with SHAP.

    TODO(pykale): ``kale.interpret`` provides no SHAP support; upstream this once a general
    interpreter interface exists there.

    Args:
        max_samples (int, optional): Number of samples to explain. Defaults to 100.
    """

    name = "shap"

    def __init__(self, max_samples: int = 100) -> None:
        self.max_samples = max_samples

    def interpret(self, model: Any, representations: Any, **context: Any) -> Any:
        """Explain the predictions on the evaluated partition.

        Args:
            model: The fitted predictor.
            representations: Features of the evaluated partition.
            **context: The remaining run context, which is not used.

        Returns:
            shap.Explanation: The SHAP values.
        """
        import shap

        estimator = getattr(model, "model", model)
        data = representations[: self.max_samples]
        return shap.Explainer(estimator.predict, data)(data)


class PermutationImportance:
    """Feature importance from scikit-learn's permutation importance.

    Args:
        n_repeats (int, optional): Number of permutations per feature. Defaults to 10.
        random_state (int, optional): Seed. Defaults to 42.
        scoring (str, optional): Scikit-learn scoring name. Defaults to
            "neg_mean_absolute_error".
    """

    name = "permutation_importance"

    def __init__(self, n_repeats: int = 10, random_state: int = 42, scoring: str = "neg_mean_absolute_error") -> None:
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.scoring = scoring

    def interpret(self, model: Any, representations: Any, targets: Any, **context: Any) -> Any:
        """Score feature importance on the evaluated partition.

        Args:
            model: The fitted predictor.
            representations: Features of the evaluated partition.
            targets: Targets of the evaluated partition.
            **context: The remaining run context, which is not used.

        Returns:
            sklearn.utils.Bunch: The permutation importance result.
        """
        from sklearn.inspection import permutation_importance

        estimator = getattr(model, "model", model)
        return permutation_importance(
            estimator,
            representations,
            targets,
            scoring=self.scoring,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )
