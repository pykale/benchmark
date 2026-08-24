"""Minimal stage components for testing the benchmark wiring.

These fakes keep the fast tests independent of torch, pymatgen and the RealMat-BaG checkout, so
that the pipeline contract can be tested without the materials stack.
"""

import numpy as np


class Dataset:
    """Minimal dataset stage."""

    def load(self):
        return list(range(10))


class Splitter:
    """Minimal splitter stage."""

    def split(self, data):
        return {"train": data[:8], "test": data[8:]}


class PrepData:
    """Records the partition it was fitted on, then emits ``(x, y)``."""

    def __init__(self):
        self.fitted_on = None

    def fit(self, data):
        self.fitted_on = list(data)

    def transform(self, data):
        values = np.asarray(data, dtype=float)
        return values[:, None], values * 2


class Embed:
    """Records the training targets it saw, then passes the data through."""

    def __init__(self):
        self.training_targets = None

    def fit(self, data):
        self.training_targets = data[1].copy()

    def transform(self, data):
        return data


class Predictor:
    """Exact linear predictor, counting its ``predict`` calls."""

    def __init__(self):
        self.calls = 0
        self.offset = 0.0

    def fit(self, x, y):
        self.offset = float(np.mean(y - 2 * x[:, 0]))

    def predict(self, x):
        self.calls += 1
        return 2 * x[:, 0] + self.offset


class Metric:
    """Mean absolute error under a custom name."""

    name = "custom"

    def evaluate(self, targets, predictions):
        return float(np.mean(np.abs(targets - predictions)))


class Interpreter:
    """Reports the parts of the run context it asked for."""

    name = "custom_interpretation"

    def interpret(self, predictions, evaluations, model):
        return {"count": len(predictions), "metrics": sorted(evaluations), "model": type(model).__name__}
