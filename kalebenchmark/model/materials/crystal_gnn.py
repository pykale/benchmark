# =============================================================================
# Author: Haolin Wang, LWang0101@outlook.com
# =============================================================================

"""Crystal graph networks for the ``predict`` stage

The benchmark's ``model`` stage covers what PyKale splits between :mod:`kale.predict` (the
network) and :mod:`kale.pipeline` (the trainer that fits it), because a benchmark selects the two
together.

The networks, the loss, the metrics, the optimisers, the batching and the training loop all live
in the RealMat-BaG reference implementation, in ``realmat_bag.pipeline.regressor``. This module
only defers the import until a run needs it, so that the package stays importable without the
checkout, and passes its arguments through unchanged.

Classical estimators need no module here at all. Their shortcuts resolve straight to
:class:`sklearn.svm.SVR`, :class:`sklearn.ensemble.RandomForestRegressor` and
:class:`sklearn.linear_model.LinearRegression`, which already provide ``fit``/``predict``.
"""

from typing import Any, Optional

import numpy as np

from kalebenchmark.utils.materials.realmat_bag import PathLike, require_realmat_bag


class CrystalGraphRegressor:
    """A crystal network trained by the RealMat-BaG regressor.

    Args:
        root (str or Path, optional): Checkout root. Defaults to ``REALMAT_BAG_ROOT`` or
            ``./bandgap-benchmark``.
        **kwargs: Passed to ``realmat_bag.pipeline.regressor.CrystalRegressor``, which documents
            them: ``model`` (an architecture name, a ``torch.nn.Module`` or a callable sized from
            one crystal), ``model_params``, ``optimizer``, ``max_epochs``, ``batch_size``,
            ``init_lr``, ``layer_freeze``, ``accelerator``, ``devices`` and ``num_workers``.

    Examples:
        >>> from kalebenchmark.model.materials.crystal_gnn import CrystalGraphRegressor
        >>> CrystalGraphRegressor(model="leftnet", max_epochs=50)  # doctest: +SKIP
    """

    def __init__(self, root: Optional[PathLike] = None, **kwargs: Any) -> None:
        self.root = root
        self.kwargs = kwargs
        self.regressor: Optional[Any] = None

    def fit(self, data: Any, targets: Optional[Any] = None) -> "CrystalGraphRegressor":
        """Train on a crystal graph dataset, whose items carry their own targets.

        Args:
            data: A crystal graph dataset.
            targets (optional): Ignored; accepted for protocol compatibility.

        Returns:
            CrystalGraphRegressor: This object.
        """
        # TODO(pykale): use kale.pipeline once the materials work is released.
        require_realmat_bag(self.root)
        from realmat_bag.pipeline.regressor import CrystalRegressor

        self.regressor = CrystalRegressor(**self.kwargs)
        self.regressor.fit(data)
        return self

    def predict(self, data: Any) -> np.ndarray:
        """Predict one value per crystal.

        Args:
            data: A crystal graph dataset.

        Returns:
            numpy.ndarray: One prediction per crystal.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.regressor is None:
            raise RuntimeError("CrystalGraphRegressor.predict called before fit.")
        return self.regressor.predict(data)
