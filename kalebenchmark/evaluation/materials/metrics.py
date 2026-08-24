"""RealMat-BaG metrics for the ``evaluate`` stage

Only metrics with no existing implementation appear here. ``mae``, ``mse`` and ``r2`` resolve
straight to :mod:`sklearn.metrics`, so this module defines none of them.

This is the benchmark counterpart of :mod:`kale.evaluate`, which provides no regression metric in
its released form, so the mean relative error comes from the RealMat-BaG reference instead.
"""

from typing import Iterable

import numpy as np

from kalebenchmark.utils.materials.realmat_bag import require_realmat_bag


def mean_relative_error(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Mean relative error, delegating to the RealMat-BaG implementation.

    TODO(pykale): use ``kale.evaluate.metrics.mean_relative_error`` once the materials work is
    released; the released package provides no regression metric.

    Args:
        y_true (array-like): Ground truth targets.
        y_pred (array-like): Predicted targets.

    Returns:
        float: The mean relative error.
    """
    require_realmat_bag()
    from realmat_bag.pipeline.models.classical_ml import mean_relative_error as _mean_relative_error

    return float(_mean_relative_error(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)))
